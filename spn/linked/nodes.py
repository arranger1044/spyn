from spn import utils
from spn import LOG_ZERO
from spn import MARG_IND
from spn import IS_LOG_ZERO

import numpy

from math import log
from math import exp

from cltree.cltree import CLTree

import dataset

import numba

NODE_SYM = 'u'  # unknown type
SUM_NODE_SYM = '+'
PROD_NODE_SYM = '*'
INDICATOR_NODE_SYM = 'i'
DISCRETE_VAR_NODE_SYM = 'd'
CHOW_LIU_TREE_NODE_SYM = 'c'


class Node(object):

    """
    WRITEME
    """
    # class id counter
    id_counter = 0

    def __init__(self, var_scope=None):
        """
        WRITEME
        """
        # default val is 0.
        self.log_val = LOG_ZERO

        # setting id and incrementing
        self.id = Node.id_counter
        Node.id_counter += 1

        # derivative computation
        self.log_der = LOG_ZERO

        self.var_scope = var_scope

    def __repr__(self):
        return 'id: {id} scope: {scope}'.format(id=self.id,
                                                scope=self.var_scope)

    # this is probably useless, using it for test purposes
    def set_val(self, val):
        """
        WRITEME
        """
        if numpy.allclose(val, 0, 1e-10):
            self.log_val = LOG_ZERO
        else:
            self.log_val = log(val)

    def __hash__(self):
        """
        A node has a unique id
        """
        return hash(self.id)

    def __eq__(self, other):
        """
        WRITEME
        """
        return self.id == other.id

    def node_type_str(self):
        return NODE_SYM

    def node_short_str(self):
        return "{0} {1}\n".format(self.node_type_str(),
                                  self.id)

    @classmethod
    def reset_id_counter(cls):
        """
        WRITEME
        """
        Node.id_counter = 0


@numba.njit
def eval_sum_node(children_log_vals, log_weights):
    """
    numba version
    """

    max_log = LOG_ZERO

    n_children = children_log_vals.shape[0]

    # getting the max
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        w_sum = ch_log_val + log_weight
        if w_sum > max_log:
            max_log = w_sum

    # log_unnorm = LOG_ZERO
    # max_child_log = LOG_ZERO

    sum_val = 0.
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        log_weight = log_weights[i]
        # for node, log_weight in zip(children, log_weights):
        # if node.log_val is False:
        ww_sum = ch_log_val + log_weight
        sum_val += exp(ww_sum - max_log)

    # is this bad code?
    log_val = LOG_ZERO
    if sum_val > 0.:
        log_val = log(sum_val) + max_log

    return log_val
    # log_unnorm = log(sum_val) + max_log
    # self.log_val = log_unnorm - numpy.log(self.weights_sum)
    # return self.log_val


class SumNode(Node):

    """
    WRITEME
    """

    def __init__(self, var_scope=None):
        """
        WRITEME
        """
        Node.__init__(self, var_scope)
        self.children = []
        self.weights = []
        self.log_weights = []
        self.weights_sum = 0

    def add_child(self, child, weight):
        """
        WRITEME
        """
        self.children.append(child)
        self.weights.append(weight)
        self.log_weights.append(log(weight))
        self.weights_sum += weight

    def set_weights(self, weights):
        """
        WRITEME
        """

        self.weights = weights

        # normalizing self.weights
        w_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / w_sum

        # updating log weights
        for i, weight in enumerate(weights):
            self.log_weights[i] = log(weight) if weight > 0.0 else LOG_ZERO

        # and also the sum
        self.weights_sum = sum(weights)

    # @numba.jit
    def eval(self):
        """
        WRITEME
        """
        # resetting the log derivative
        self.log_der = LOG_ZERO

        max_log = LOG_ZERO

        # getting the max
        for node, log_weight in zip(self.children, self.log_weights):
            w_sum = node.log_val + log_weight
            if w_sum > max_log:
                max_log = w_sum

        # log_unnorm = LOG_ZERO
        # max_child_log = LOG_ZERO

        sum_val = 0.
        for node, log_weight in zip(self.children, self.log_weights):
            # if node.log_val is False:
            ww_sum = node.log_val + log_weight
            sum_val += exp(ww_sum - max_log)

        # is this bad code?
        if sum_val > 0.:
            self.log_val = log(sum_val) + max_log

        else:
            self.log_val = LOG_ZERO

        # # up to now numba

        # log_unnorm = log(sum_val) + max_log
        # self.log_val = log_unnorm - numpy.log(self.weights_sum)
        # return self.log_val

        # self.log_val = eval_sum_node(numpy.array([child.log_val
        #                                           for child in self.children]),
        #                              numpy.array(self.log_weights))

    def mpe_eval(self):
        """
        WRITEME
        """
        # resetting the log derivative
        self.log_der = LOG_ZERO

        # log_val is used as an accumulator, one less var
        self.log_val = LOG_ZERO

        # getting the max
        for node, log_weight in zip(self.children, self.log_weights):
            w_sum = node.log_val + log_weight
            if w_sum > self.log_val:
                self.log_val = w_sum

    def backprop(self):
        """
        WRITE
        """
        # if it is not zero we can pass
        if self.log_der > LOG_ZERO:
            # dS/dS_n = sum_{p}: dS/dS_p * dS_p/dS_n
            # per un nodo somma p
            #
            for child, log_weight in zip(self.children, self.log_weights):
                # print('child before', child.log_der)
                # if child.log_der == LOG_ZERO:
                # if IS_LOG_ZERO(child.log_der):
                if child.log_der <= LOG_ZERO:
                    child.log_der = self.log_der + log_weight
                else:
                    child.log_der = numpy.logaddexp(child.log_der,
                                                    self.log_der + log_weight)
                # print('child after', child.log_der)
        # update weight log der too ?

    def mpe_backprop(self):
        """
        WRITEME
        """
        if self.log_der > LOG_ZERO:
            # the child der is the max der among parents
            for child in self.children:
                child.log_der = max(child.log_der, self.log_der)

    def normalize(self):
        """
        WRITEME
        """
        # normalizing self.weights
        w_sum = sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / w_sum

        # computing log(self.weights)
        for i, weight in enumerate(self.weights):
            self.log_weights[i] = log(weight) if weight > 0.0 else LOG_ZERO

    def is_complete(self):

        _complete = True
        # all children scopes shall be equal
        children_scopes = [child.var_scope
                           for child in self.children]

        # adding this node scope
        children_scopes.append(self.var_scope)

        for scope1, scope2 in utils.pairwise(children_scopes):
            if scope1 != scope2:
                _complete = False
                break

        return _complete

    def n_children(self):
        """
        WRITEME
        """
        return len(self.children)

    def node_type_str(self):
        return SUM_NODE_SYM

    def node_short_str(self):
        children_str = " ".join(["{id}:{weight}".format(id=node.id,
                                                        weight=weight)
                                 for node, weight in zip(self.children,
                                                         self.weights)])
        return "{type} {id} [{children}]".format(type=self.node_type_str(),
                                                 id=self.id,
                                                 children=children_str)

    def __repr__(self):
        base = Node.__repr__(self)
        children_info = [(node.id, weight)
                         for node, weight in zip(self.children,
                                                 self.weights)]
        msg = ''
        for id, weight in children_info:
            msg += ' ({id} {weight})'.format(id=id,
                                             weight=weight)
        return 'Sum Node {line1}\n{line2}'.format(line1=base,
                                                  line2=msg)


@numba.njit
def eval_prod_node(children_log_vals):
    """
    WRITEME
    """

    n_children = children_log_vals.shape[0]

    # and the zero children counter
    # zero_children = 0

    # computing the log value
    log_val = 0.0
    for i in range(n_children):
        ch_log_val = children_log_vals[i]
        # if ch_log_val <= LOG_ZERO:
        #     zero_children += 1

        log_val += ch_log_val

    return log_val  # , zero_children


class ProductNode(Node):

    """
    WRITEME
    """

    def __init__(self, var_scope=None):
        """
        WRITEME
        """
        Node.__init__(self, var_scope)
        self.children = []
        # bit for zero children, see Darwiche
        self.zero_children = 0

    def add_child(self, child):
        """
        WRITEME
        """
        self.children.append(child)

    def eval(self):
        """
        WRITEME
        """
        # resetting the log derivative
        self.log_der = LOG_ZERO

        # and the zero children counter
        self.zero_children = 0

        # computing the log value
        self.log_val = 0.0
        for node in self.children:
            if node.log_val <= LOG_ZERO:
                self.zero_children += 1

            self.log_val += node.log_val

        #
        # numba
        # self.log_val = \
        #     eval_prod_node(numpy.array([child.log_val
        #                                 for child in self.children]))
        # return self.log_val

    def mpe_eval(self):
        """
        Just redirecting normal evaluation
        """
        self.eval()

    def backprop(self):
        """
        WRITEME
        """
        if self.log_der > LOG_ZERO:

            for child in self.children:
                log_der = LOG_ZERO
                # checking the bit
                if self.zero_children == 0:
                    log_der = self.log_val - child.log_val
                elif self.zero_children == 1 and child.log_val <= LOG_ZERO:
                    log_der = sum([node.log_val for node in self.children
                                   if node != child])
                    # log_der = 0.0
                    # for node in self.children:
                    #     if node != child:
                    #         log_der += node.log_val
                # adding this parent value
                log_der += self.log_der
                # if child.log_der <= LOG_ZERO:
                # if IS_LOG_ZERO(child.log_der):
                if child.log_der <= LOG_ZERO:
                    # first assignment
                    child.log_der = log_der
                else:
                    child.log_der = numpy.logaddexp(child.log_der,
                                                    log_der)

    def mpe_backprop(self):
        """
        WRITEME
        """
        if self.log_der > LOG_ZERO:
            for child in self.children:
                log_der = LOG_ZERO
                # checking the bit
                if self.zero_children == 0:
                    log_der = self.log_val - child.log_val
                elif self.zero_children == 1 and child.log_val <= LOG_ZERO:
                    log_der = sum([node.log_val for node in self.children
                                   if node != child])
                # adding this parent value
                log_der += self.log_der
                # updating child log der with the max instead of sum
                child.log_der = max(child.log_der, log_der)

    def backprop2(self):
        """
        WRITEME
        """
        # if more than one child has a zero value, cannot propagate
        if self.log_val <= LOG_ZERO:
            count = 0
            for child in self.children:
                if child.log_val <= LOG_ZERO:
                    count += 1
                    if count > 1:
                        return

        # only when needed
        if self.log_der > LOG_ZERO:
            for child in self.children:
                # print('b child val', child.log_val, child.log_der)
                if child.log_val <= LOG_ZERO:
                    # print('child log zero')
                    # shall loop on other children
                    # maybe this is memory consuming, but shall be faster
                    # going to numpy array shall be faster
                    log_der = sum([node.log_val for node in self.children
                                   if node.log_val > LOG_ZERO]) + \
                        self.log_der
                    if child.log_der <= LOG_ZERO:
                        # print('first log, add', log_der)
                        child.log_der = log_der
                    else:

                        child.log_der = numpy.logaddexp(child.log_der,
                                                        log_der)
                        # print('not first log, added', child.log_der)
                # if it is 0 there is no point updating children
                elif self.log_val > LOG_ZERO:
                    # print('par val not zero')
                    if child.log_der <= LOG_ZERO:
                        child.log_der = self.log_der + \
                            self.log_val - \
                            child.log_val
                        # print('child val not zero', child.log_der)
                    else:
                        child.log_der = numpy.logaddexp(child.log_der,
                                                        self.log_der +
                                                        self.log_val -
                                                        child.log_val)
                        # print('child log der not first', child.log_der)

    def is_decomposable(self):

        decomposable = True
        whole = set()
        for child in self.children:
            child_scope = child.var_scope
            for scope_var in child_scope:
                if scope_var in whole:
                    decomposable = False
                    break
                else:
                    whole.add(scope_var)
            else:
                continue
            break

        if whole != self.var_scope:
            decomposable = False
        return decomposable

    def n_children(self):
        """
        WRITEME
        """
        return len(self.children)

    def node_type_str(self):
        return PROD_NODE_SYM

    def node_short_str(self):
        children_str = " ".join(["{id}".format(id=node.id)
                                 for node in self.children])
        return "{type} {id} [{children}]".format(type=self.node_type_str(),
                                                 id=self.id,
                                                 children=children_str)

    def __repr__(self):
        base = Node.__repr__(self)
        children_info = [node.id
                         for node in self.children]
        msg = ''
        for id in children_info:
            msg += ' ({id})'.format(id=id)
        return 'Prod Node {line1}\n{line2}'.format(line1=base,
                                                   line2=msg)


class CategoricalIndicatorNode(Node):

    """
    WRITEME
    """

    def __init__(self, var, var_val):
        """
        WRITEME
        """
        Node.__init__(self, frozenset({var}))
        self.var = var
        self.var_val = var_val

    def eval(self, obs):
        """
        WRITEME
        """
        self.log_der = LOG_ZERO

        if obs == MARG_IND:
            self.log_val = 0.
        elif obs == self.var_val:
            self.log_val = 0.
        else:
            self.log_val = LOG_ZERO

    def mpe_eval(self, obs):
        """
        Just redirecting normal evaluation
        """
        self.eval(obs)

    def n_children(self):
        return 0

    def node_type_str(self):
        return INDICATOR_NODE_SYM

    def node_short_str(self):

        return "{type} {id} <{var}> {val}".format(type=self.node_type_str(),
                                                  id=self.id,
                                                  var=self.var,
                                                  val=self.var_val)

    def __repr__(self):
        base = Node.__repr__(self)

        return """Indicator Node {line1}\n
    var: {var} val: {val}""".format(line1=base,
                                    var=self.var,
                                    val=self._var_val)


class CLTreeNode(Node):

    """
    An input node representing a Chow-Liu Tree over a set of r.v.
    """

    def __init__(self,
                 vars,
                 var_values,
                 data,
                 factors=None,
                 alpha=0.1):
        """
        vars = the sequence of feature ids
        var_values = the sequence of feature values
        alpha = smoothing parameter
        data = the data slice (2d ndarray) upon which to grow a cltree
        factors = the already computed factors (this is when the model has already been conputed)
        """
        Node.__init__(self, frozenset(vars))

        self.vars = numpy.array(vars)

        self._alpha = alpha
        #
        # assuming all variables to be homogeneous
        # TODO: generalize this
        self._n_var_vals = var_values[0]
        self.var_values = numpy.array(var_values)

        #
        # assuming data is never None
        self._data = data
        self._cltree = CLTree(data,
                              features=self.vars,
                              n_feature_vals=self._n_var_vals,
                              feature_vals=self.var_values,
                              alpha=alpha,
                              sparse=True,
                              mem_free=True)

    def smooth_probs(self, alpha, data=None):
        """
        The only waya to smooth here is to rebuild the whole tree
        """
        self._alpha = alpha

        if data is not None:
            self._data = data
        # else:
        #     raise ValueError('Cannot smooth without data')

        self._cltree = CLTree(data=self._data,
                              features=self.vars,
                              n_feature_vals=self._n_var_vals,
                              feature_vals=self.var_values,
                              alpha=alpha,
                              # copy_mi=False,
                              sparse=True,
                              mem_free=True)

    def eval(self, obs):
        """
        Dispatching inference to the cltree
        """
        #
        # TODO: do something for the derivatives
        self.log_der = LOG_ZERO

        # self.log_val = self._cltree.eval(obs)
        self.log_val = self._cltree.eval_fact(obs)

    def mpe_eval(self, obs):
        """
        WRITEME
        """
        raise NotImplementedError('MPE inference not yet implemented')

    def n_children(self):
        return len(self.vars)

    def node_type_str(self):
        return CHOW_LIU_TREE_NODE_SYM

    def node_short_str(self):
        vars_str = ','.join([var for var in self.vars])
        return "{type} {id}" +\
            " <{vars}>" +\
            " {tree} {factors}".format(type=self.node_type_str(),
                                       id=self.id,
                                       vars=vars_str,
                                       tree=self._cltree.tree_repr(),
                                       factors=self._cltree.factors_repr())

    def __repr__(self):
        """
        WRITEME
        """
        base = Node.__repr__(self)

        return ("""CLTree Smoothed Node {line1}
            vars: {vars} vals: {vals} tree:{tree}""".
                format(line1=base,
                       vars=self.vars,
                       vals=self._n_var_vals,
                       tree=self._cltree.tree_repr()))


@numba.njit
def eval_numba(obs, vars):
    if obs == MARG_IND:
        return 0.
    else:
        return vars[obs]


class CategoricalSmoothedNode(Node):

    """
    WRITEME
    """

    def __init__(self, var, var_values, alpha=0.1,
                 freqs=None, data=None, instances=None):
        """
        WRITEME
        """

        Node.__init__(self, frozenset({var}))

        self.var = var
        self.var_val = var_values

        # building storing freqs
        if data is None:
            if freqs is None:
                self._var_freqs = [1 for i in range(var_values)]
            else:
                self._var_freqs = freqs[:]
        else:
            # better checking for numpy arrays shape
            assert data.shape[1] == 1
            (freqs_dict,), _features = dataset.data_2_freqs(data)
            self._var_freqs = freqs_dict['freqs']

        # computing the smoothed ll
        self._var_probs = CategoricalSmoothedNode.smooth_ll(self._var_freqs[:],
                                                            alpha)

        # storing instance ids (it is a list)
        self._instances = instances

    def smooth_ll(freqs, alpha):
        """
        WRITEME
        """

        vals = len(freqs)
        freqs_sum = sum(freqs)
        for i, freq in enumerate(freqs):
            log_freq = LOG_ZERO
            if (freq + alpha) > 0.:
                log_freq = log(freq + alpha)
            freqs[i] = (log_freq -
                        log(freqs_sum + vals * alpha))
        # return freqs
        return numpy.array(freqs)

    def smooth_freq_from_data(data, alpha):
        """
        WRITEME
        """
        # data here shall have only one feature
        assert data.shape[1] == 1
        (freqs_dict,), _features = dataset.data_2_freqs(data)

        return CategoricalSmoothedNode.smooth_ll(freqs_dict['freqs'], alpha)

    def smooth_probs(self, alpha, data=None):
        """
        WRITEME
        """
        if data is None:
            # var_values = len(self._var_freqs)
            smooth_probs = \
                CategoricalSmoothedNode.smooth_ll(self._var_freqs[:],
                                                  alpha)
        else:
            # slicing in two different times to preserve the 2 dims
            data_slice_var = data[:, [self.var]]
            # checking to be sure: it shall be a list btw
            if isinstance(self._instances, list):
                data_slice = data_slice_var[self._instances]
            else:
                data_slice = data_slice_var[list(self._instances)]
            # print('SLICE', data_slice_var, data_slice)
            smooth_probs = \
                CategoricalSmoothedNode.smooth_freq_from_data(data_slice,
                                                              alpha)

        self._var_probs = smooth_probs

    def eval(self, obs):
        """
        WRITEME
        """

        self.log_der = LOG_ZERO

        # if obs == MARG_IND:
        #     self.log_val = 0.
        # else:
        #     self.log_val = self._var_probs[obs]
        self.log_val = eval_numba(obs, self._var_probs)

    def mpe_eval(self, obs):
        """
        Just redirecting normal evaluation, it surely is the one associated
        to the observed value
        """
        self.eval(obs)

    def n_children(self):
        return 0

    def node_type_str(self):
        return DISCRETE_VAR_NODE_SYM

    def node_short_str(self):
        freqs_str = " ".join(self._var_freqs)
        return "{type} {id}" +\
            " <{var}>" +\
            " {freqs}".format(type=self.node_type_str(),
                              id=self.id,
                              vars=self.var,
                              freqs=freqs_str)

    def __repr__(self):
        base = Node.__repr__(self)

        return ("""Categorical Smoothed Node {line1}
            var: {var} val: {val} [{ff}] [{ll}]""".
                format(line1=base,
                       var=self.var,
                       val=len(self._var_freqs),
                       ff=[freq for freq in self._var_freqs],
                       ll=[ll for ll in self._var_probs]))

    def var_values(self):
        """
        WRITEME
        """
        return len(self._var_freqs)
