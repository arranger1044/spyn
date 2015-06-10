from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalIndicatorNode
from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import CLTreeNode

from math import exp

import numba


@numba.jit
def eval_numba(nodes):
    for node in nodes:
        node.eval()


class Layer(object):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        self._nodes = None
        self._n_nodes = None

        if nodes is None:
            self._nodes = []
            self._n_nodes = 0
        else:
            self._nodes = nodes
            self._n_nodes = len(nodes)

    def add_node(self, node):
        """
        WRITEME
        """
        self._nodes.append(node)
        self._n_nodes += 1

    def nodes(self):
        """
        WRITEME
        """
        for node in self._nodes:
            yield node

    def eval(self):
        """
        layer bottom-up evaluation
        """
        for node in self._nodes:
            node.eval()
        # eval_numba(self._nodes)

    def mpe_eval(self):
        """
        layer MPE bpttom-up evaluation
        """
        for node in self._nodes:
            node.mpe_eval()

    def backprop(self):
        """
        WRITEME
        """
        for node in self._nodes:
            node.backprop()

    def mpe_backprop(self):
        """
        WRITEME
        """
        for node in self._nodes:
            node.mpe_backprop()

    def set_log_derivative(self, log_der):
        """
        WRITEME
        """
        for node in self._nodes:
            node.log_der = log_der

    def node_values(self):
        """
        WRITEME
        """
        # depending on the freq of the op I could allocate
        # just once the list
        return [node.log_val for node in self._nodes]

    def get_nodes_by_id(self, node_pos):
        """
        this may be inefficient, atm used only in factory
        """
        node_list = [None for i in range(self._n_nodes)]
        for node in self._nodes:
            pos = node_pos[node.id]
            node_list[pos] = node
        return node_list

    def get_node(self, node_id):
        """
        WRITEME
        """
        return self._nodes[node_id]

    def n_nodes(self):
        """
        WRITEME
        """
        return self._n_nodes

    def n_edges(self):
        """
        WRITEME
        """
        edges = 0
        for node in self._nodes:
            # input layers have nodes with no children attr
            # try:
                # for child in node.children:
                #     edges += 1
            edges += node.n_children()
            # except:
            #     pass
        return edges

    def n_weights(self):
        """
        Only a sum layer has params
        """
        return 0

    def __repr__(self):
        """
        WRITEME
        """
        div = '\n**********************************************************\n'
        return '\n'.join([str(node) for node in self._nodes]) + div


class SumLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)

    def normalize(self):
        """
        WRITEME
        """
        for node in self._nodes:
            node.normalize()

    def add_edge(self, parent, child, weight):
        """
        WRITEME
        """
        parent.add_child(child, weight)

    # def update_weights(self, update_rule):
    #     """
    #     WRITEME
    #     """
    #     for node in self._nodes:
    #         weight_updates = [update_rule(weight,
    #                                       exp(child.log_val + node.log_der))
    #                           for child, weight
    #                           in zip(node.children, node.weights)]
    #         node.set_weights(weight_updates)

    def update_weights(self, update_rule, layer_id):
        """
        WRITEME
        """
        for node_id, node in enumerate(self._nodes):
            weight_updates = [update_rule(layer_id,
                                          node_id,
                                          weight_id,
                                          weight,
                                          exp(child.log_val + node.log_der))
                              for weight_id, (child, weight)
                              in enumerate(zip(node.children, node.weights))]
            node.set_weights(weight_updates)

    def is_complete(self):
        """
        WRITEME
        """
        return all([node.is_complete() for node in self.nodes()])

    def n_weights(self):
        """
        For a sum layer, its number of edges
        """
        return self.n_edges()

    def __repr__(self):
        return '[sum layer:]\n' + Layer.__repr__(self)


class ProductLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)

    def add_edge(self, parent, child):
        """
        WRITEME
        """
        parent.add_child(child)

    def is_decomposable(self):
        """
        WRITEME
        """
        return all([node.is_decomposable() for node in self.nodes()])

    def __repr__(self):
        return '[prod layer:]\n' + Layer.__repr__(self)


class CategoricalInputLayer(Layer):

    """
    WRITEME
    """

    def __init__(self, nodes=None, vars=None):
        """
        WRITEME
        """
        Layer.__init__(self, nodes)
        self._vars = vars
        self._feature_vals = None

    def eval(self, input):
        """
        WRITEME
        """
        for node in self._nodes:
            # get the observed value
            obs = input[node.var]
            # and eval the node
            node.eval(obs)

    def vars(self):
        """
        WRITEME
        """
        return self._vars

    def feature_vals(self):
        """
        WRITEME
        """
        return self._feature_vals

    def smooth_probs(self, alpha):
        """
        This shall be implemented in the class specializing
        """
        raise NotImplementedError('Smoothing not implemented for input layer')

    def __repr__(self):
        return '[input layer:]\n' + Layer.__repr__(self)


def compute_feature_vals(nodes):
    """
    From a set of input nodes, determine the feature ranges
    """
    feature_vals_dict = {}

    for node in nodes:
        if isinstance(node, CLTreeNode):

            #
            # updating nodes vars ranges (assuming no inconsistencies)
            for n_var, n_var_val in zip(node.vars, node.var_values):
                if n_var not in feature_vals_dict:
                    feature_vals_dict[n_var] = n_var_val

        elif isinstance(node, CategoricalIndicatorNode):

            if node.var not in feature_vals_dict:
                feature_vals_dict[node.var] = node.var_val + 1
            else:
                feature_vals_dict[node.var] = max(node.var_val + 1,
                                                  feature_vals_dict[node.var])

        elif isinstance(node, CategoricalSmoothedNode):

            if node.var not in feature_vals_dict:
                feature_vals_dict[node.var] = node.var_val

    feature_vals = [feature_vals_dict[var]
                    for var in sorted(feature_vals_dict.keys())]

    return feature_vals


class CategoricalCLInputLayer(CategoricalInputLayer):

    """
    This layer contains
    TODO: rewrite this hierarchy, it is a mess
    """

    def __init__(self, nodes=None):
        """
        WRITEME
        """
        if nodes is not None:
            Layer.__init__(self, nodes)
            #
            # updating node counts
            n_nodes = 0

            for node in nodes:
                n_nodes += len(node.vars) + 1 if isinstance(node,
                                                            CLTreeNode) else 1

            self._n_nodes = n_nodes
            self._feature_vals = compute_feature_vals(nodes)

        else:
            raise NotImplementedError('No nodes provided')

    def eval(self, input):
        """
        WRITEME
        """
        for node in self._nodes:
            #
            # I am not using polymorphism at all...
            if isinstance(node, CLTreeNode):
                node.eval(input)
            else:
                # the other node type is assumed to be CategoricalS
                # extract the observed var value
                obs = input[node.var]
                # and eval the node
                node.eval(obs)

    def smooth_probs(self, alpha):
        """
        WRITEME
        """
        for node in self._nodes:
            node.smooth_probs(alpha)


class CategoricalIndicatorLayer(CategoricalInputLayer):

    """
    WRITEME
    """

    def __init__(self, nodes=None, vars=None):
        """
        WRITEME
        """
        if nodes is None:
            # self._vars = vars
            nodes = [CategoricalIndicatorNode(var, i)
                     for var in range(len(vars))
                     for i in range(vars[var])]

            # self._feature_vals = [2 for i in range(len(nodes))]
        else:
            # assuming the nodes are complete and coherent
            vars_dict = {}
            for node in nodes:
                try:
                    vars_dict[node.var] += 1
                except:
                    vars_dict[node.var] = 1

            sorted_keys = sorted(vars_dict.items(), key=lambda t: t[0])
            vars = [vals for id, vals in sorted_keys]

            # self.feature_vals = compute_feature_vals(nodes)

        CategoricalInputLayer.__init__(self, nodes, vars)
        self._feature_vals = compute_feature_vals(nodes)


class CategoricalSmoothedLayer(CategoricalInputLayer):

    """
    WRITEME
    """

    def __init__(self, nodes=None, vars=None, node_dicts=None, alpha=0.1):
        """
        WRITEME
        """
        self.alpha = alpha

        if nodes is None:
            nodes = []
            # self._vars = vars

            for node_dict in node_dicts:
                var_id = node_dict['var']
                var_values = vars[var_id]
                var_freqs = node_dict['freqs'] \
                    if 'freqs' in node_dict else None
                nodes.append(CategoricalSmoothedNode(var_id,
                                                     var_values,
                                                     alpha,
                                                     var_freqs))
        else:
            # vars shall be computed by hand
            # TODO test it
            vars_dict = {}
            for node in nodes:
                vars_dict[node.var] = node.var_values()
            # getting sorted keys
            sorted_keys = sorted(vars_dict.items(), key=lambda t: t[0])

            vars = [vals for id, vals in sorted_keys]

        CategoricalInputLayer.__init__(self, nodes, vars)
        self._feature_vals = compute_feature_vals(nodes)

    def smooth_probs(self, alpha):
        """
        WRITEME
        """
        for node in self._nodes:
            node.smooth_probs(alpha)
