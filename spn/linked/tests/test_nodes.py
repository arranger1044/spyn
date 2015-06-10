from spn.linked.nodes import Node
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalIndicatorNode
from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import CLTreeNode

from spn.tests import compute_smoothed_ll

from spn import LOG_ZERO
from spn import MARG_IND
from spn import IS_LOG_ZERO

import numpy

from math import log

from spn.tests import assert_log_array_almost_equal
from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def test_node_set_val():
    node = Node()

    # asserting log(0) == LOG_ZERO
    node.set_val(0)
    assert node.log_val == LOG_ZERO

    # any other value shall get to its log value
    half_truth = 0.5
    node.set_val(half_truth)
    assert node.log_val == log(half_truth)

    # truth checking 1 -> 0
    truth = 1.
    node.set_val(truth)
    assert node.log_val == log(truth)

    print(node)


def test_sum_node_create_and_eval():
    # create child nodes
    child1 = Node()
    val1 = 1.
    child1.set_val(val1)

    child2 = Node()
    val2 = 1.
    child2.set_val(val2)

    # create sum node and adding children to it
    sum_node = SumNode()
    weight1 = 0.8
    weight2 = 0.2
    sum_node.add_child(child1, weight1)
    sum_node.add_child(child2, weight2)
    assert len(sum_node.children) == 2
    assert len(sum_node.weights) == 2
    assert len(sum_node.log_weights) == 2
    log_weights = [log(weight1), log(weight2)]
    assert log_weights == sum_node.log_weights

    print(sum_node)

    # evaluating
    sum_node.eval()
    print(sum_node.log_val)
    assert_almost_equal(sum_node.log_val,
                        log(val1 * weight1 + val2 * weight2),
                        places=15)

    # changing values 1,0
    val1 = 1.
    child1.set_val(val1)
    val2 = 0.
    child2.set_val(val2)

    # evaluating
    sum_node.eval()
    print(sum_node.log_val)
    assert_almost_equal(sum_node.log_val,
                        log(val1 * weight1 + val2 * weight2),
                        places=15)

    # changing values 0,0 -> LOG_ZERO
    val1 = 0.
    child1.set_val(val1)
    val2 = 0.
    child2.set_val(val2)

    # evaluating
    sum_node.eval()
    print(sum_node.log_val)
    assert_almost_equal(sum_node.log_val,
                        LOG_ZERO,
                        places=15)


def test_sum_node_backprop():
    # create child nodes
    child1 = Node()
    val1 = 1.
    child1.set_val(val1)

    child2 = Node()
    val2 = 1.
    child2.set_val(val2)

    # create sum node and adding children to it
    sum_node1 = SumNode()
    weight11 = 0.8
    weight12 = 0.2
    sum_node1.add_child(child1, weight11)
    sum_node1.add_child(child2, weight12)

    # adding a coparent
    sum_node2 = SumNode()
    weight21 = 0.6
    weight22 = 0.4
    sum_node2.add_child(child1, weight21)
    sum_node2.add_child(child2, weight22)

    # evaluating
    sum_node1.eval()
    sum_node2.eval()

    # setting the log derivatives to the parents
    sum_node_der1 = 1.0
    sum_node1.log_der = log(sum_node_der1)
    sum_node1.backprop()

    sum_node_der2 = 1.0
    sum_node2.log_der = log(sum_node_der2)
    sum_node2.backprop()

    # checking for correctness
    log_der1 = log(weight11 * sum_node_der1 +
                   weight21 * sum_node_der2)

    log_der2 = log(weight12 * sum_node_der1 +
                   weight22 * sum_node_der2)

    print('log ders 1:{lgd1} 2:{lgd2}'.format(lgd1=log_der1,
                                              lgd2=log_der2))
    assert_almost_equal(log_der1, child1.log_der, 15)
    assert_almost_equal(log_der2, child2.log_der, 15)

    # resetting
    child1.log_der = LOG_ZERO
    child2.log_der = LOG_ZERO

    # now changing the initial der values
    sum_node_der1 = 0.5
    sum_node1.log_der = log(sum_node_der1)
    sum_node1.backprop()

    sum_node_der2 = 0.0
    sum_node2.log_der = LOG_ZERO
    sum_node2.backprop()

    # checking for correctness
    log_der1 = log(weight11 * sum_node_der1 +
                   weight21 * sum_node_der2)

    log_der2 = log(weight12 * sum_node_der1 +
                   weight22 * sum_node_der2)

    print('log ders 1:{lgd1} 2:{lgd2}'.format(lgd1=log_der1,
                                              lgd2=log_der2))
    assert_almost_equal(log_der1, child1.log_der, 15)
    assert_almost_equal(log_der2, child2.log_der, 15)


def test_product_node_create_and_eval():
    # create child nodes
    child1 = Node()
    val1 = 1.
    child1.set_val(val1)

    child2 = Node()
    val2 = 1.
    child2.set_val(val2)

    # create product node and add children
    prod_node = ProductNode()
    prod_node.add_child(child1)
    prod_node.add_child(child2)
    assert len(prod_node.children) == 2

    print(prod_node)

    # evaluation
    prod_node.eval()
    print(prod_node.log_val)
    assert_almost_equal(prod_node.log_val,
                        log(val1 * val2),
                        places=15)

    # changing values 0,1 -> LOG_ZERO
    val1 = 0.
    child1.set_val(val1)
    val2 = 1.
    child2.set_val(val2)

    prod_node.eval()
    print(prod_node.log_val)
    assert_almost_equal(prod_node.log_val,
                        LOG_ZERO,
                        places=15)

    # changing values 0,1 -> LOG_ZERO
    val1 = 0.
    child1.set_val(val1)
    val2 = 0.
    child2.set_val(val2)

    prod_node.eval()
    print(prod_node.log_val)
    # now testing with macro since -1000 + -1000 != -1000
    assert IS_LOG_ZERO(prod_node.log_val) is True


def test_product_node_backprop():
    # create child nodes
    child1 = Node()
    val1 = 1.
    child1.set_val(val1)

    child2 = Node()
    val2 = 1.
    child2.set_val(val2)

    child3 = Node()
    val3 = 0.0
    child3.set_val(val3)

    # create a product node and add children
    prod_node1 = ProductNode()
    prod_node1.add_child(child1)
    prod_node1.add_child(child2)

    # create a second node on all children
    prod_node2 = ProductNode()
    prod_node2.add_child(child1)
    prod_node2.add_child(child2)
    prod_node2.add_child(child3)

    # eval
    prod_node1.eval()
    prod_node2.eval()

    # set der and backprop
    prod_node_der1 = 1.0
    prod_node1.log_der = log(prod_node_der1)
    prod_node1.backprop()

    prod_node_der2 = 1.0
    prod_node2.log_der = log(prod_node_der2)
    prod_node2.backprop()

    # check for correctness
    log_der1 = log(prod_node_der1 * val2 +
                   prod_node_der2 * val2 * val3)
    log_der2 = log(prod_node_der1 * val1 +
                   prod_node_der2 * val1 * val3)
    log_der3 = log(prod_node_der2 * val1 * val2)

    print('log ders 1:{lgd1} 2:{lgd2} 3:{lgd3}'.format(lgd1=log_der1,
                                                       lgd2=log_der2,
                                                       lgd3=log_der3))

    assert_almost_equal(log_der1, child1.log_der, 15)
    assert_almost_equal(log_der2, child2.log_der, 15)
    assert_almost_equal(log_der3, child3.log_der, 15)

    # setting different values for children
    val1 = 0.
    child1.set_val(val1)

    val2 = 0.
    child2.set_val(val2)

    val3 = 1.
    child3.set_val(val3)

    # eval
    prod_node1.eval()
    prod_node2.eval()

    child1.log_der = LOG_ZERO
    child2.log_der = LOG_ZERO
    child3.log_der = LOG_ZERO

    # set der and backprop
    prod_node_der1 = 0.5
    prod_node1.log_der = log(prod_node_der1)
    prod_node1.backprop()

    prod_node_der2 = 0.1
    prod_node2.log_der = log(prod_node_der2)
    prod_node2.backprop()

    # check for correctness
    try:
        log_der1 = log(prod_node_der1 * val2 +
                       prod_node_der2 * val2 * val3)
    except:
        log_der1 = LOG_ZERO
    try:
        log_der2 = log(prod_node_der1 * val1 +
                       prod_node_der2 * val1 * val3)
    except:
        log_der2 = LOG_ZERO
    try:
        log_der3 = log(prod_node_der2 * val1 * val2)
    except:
        log_der3 = LOG_ZERO

    print('log ders 1:{lgd1} 2:{lgd2} 3:{lgd3}'.format(lgd1=log_der1,
                                                       lgd2=log_der2,
                                                       lgd3=log_der3))
    print('log ders 1:{lgd1} 2:{lgd2} 3:{lgd3}'.format(lgd1=child1.log_der,
                                                       lgd2=child2.log_der,
                                                       lgd3=child3.log_der))

    if IS_LOG_ZERO(log_der1):
        assert IS_LOG_ZERO(child1.log_der)
    else:
        assert_almost_equal(log_der1, child1.log_der, 15)
    if IS_LOG_ZERO(log_der2):
        assert IS_LOG_ZERO(child2.log_der)
    else:
        assert_almost_equal(log_der2, child2.log_der, 15)
    if IS_LOG_ZERO(log_der3):
        assert IS_LOG_ZERO(child3.log_der)
    else:
        assert_almost_equal(log_der3, child3.log_der, 15)

    # setting different values for children
    val1 = 0.
    child1.set_val(val1)

    val2 = 0.2
    child2.set_val(val2)

    val3 = 1.
    child3.set_val(val3)

    # eval
    prod_node1.eval()
    prod_node2.eval()

    child1.log_der = LOG_ZERO
    child2.log_der = LOG_ZERO
    child3.log_der = LOG_ZERO

    # set der and backprop
    prod_node_der1 = 0.5
    prod_node1.log_der = log(prod_node_der1)
    prod_node1.backprop()

    prod_node_der2 = 0.1
    prod_node2.log_der = log(prod_node_der2)
    prod_node2.backprop()

    # check for correctness
    try:
        log_der1 = log(prod_node_der1 * val2 +
                       prod_node_der2 * val2 * val3)
    except:
        log_der1 = LOG_ZERO
    try:
        log_der2 = log(prod_node_der1 * val1 +
                       prod_node_der2 * val1 * val3)
    except:
        log_der2 = LOG_ZERO
    try:
        log_der3 = log(prod_node_der2 * val1 * val2)
    except:
        log_der3 = LOG_ZERO

    print('log ders 1:{lgd1} 2:{lgd2} 3:{lgd3}'.format(lgd1=log_der1,
                                                       lgd2=log_der2,
                                                       lgd3=log_der3))
    print('log ders 1:{lgd1} 2:{lgd2} 3:{lgd3}'.format(lgd1=child1.log_der,
                                                       lgd2=child2.log_der,
                                                       lgd3=child3.log_der))

    if IS_LOG_ZERO(log_der1):
        assert IS_LOG_ZERO(child1.log_der)
    else:
        assert_almost_equal(log_der1, child1.log_der, 15)
    if IS_LOG_ZERO(log_der2):
        assert IS_LOG_ZERO(child2.log_der)
    else:
        assert_almost_equal(log_der2, child2.log_der, 15)
    if IS_LOG_ZERO(log_der3):
        assert IS_LOG_ZERO(child3.log_der)
    else:
        assert_almost_equal(log_der3, child3.log_der, 15)


def test_sum_node_normalize():
    # create child nodes
    child1 = Node()
    val1 = 1.
    child1.set_val(val1)

    child2 = Node()
    val2 = 1.
    child2.set_val(val2)

    # create sum node and adding children to it
    sum_node = SumNode()
    weight1 = 1.
    weight2 = 0.2
    weights = [weight1, weight2]
    sum_node.add_child(child1, weight1)
    sum_node.add_child(child2, weight2)
    un_sum = sum(weights)

    # normalizing
    sum_node.normalize()
    assert len(sum_node.children) == 2
    assert len(sum_node.weights) == 2
    assert len(sum_node.log_weights) == 2

    # checking weight sum
    w_sum = sum(sum_node.weights)
    assert w_sum == 1.

    # and check the correct values
    normal_sum = [weight / un_sum for weight in weights]
    print(normal_sum)
    assert normal_sum == sum_node.weights

    # checking log_weights
    log_weights = [log(weight) for weight in normal_sum]
    print(log_weights)
    assert log_weights == sum_node.log_weights


def test_categorical_indicator_node_create_and_eval():

    # created a node on the first var and its first value
    ind = CategoricalIndicatorNode(0, 0)

    # seen x0 = 0 -> 1.
    ind.eval(0)
    assert ind.log_val == 0.

    # this indicator is not fired
    ind.eval(1)
    assert ind.log_val == LOG_ZERO

    # all indicators for that var are fired
    ind.eval(MARG_IND)
    assert ind.log_val == 0.

    # the var has only 2 values, but the node does not know!
    ind.eval(2)
    assert ind.log_val == LOG_ZERO

# list of var values (var = position in the list)
vars = [2, 2, 3, 4]
freqs = [[1, 2],
         [5, 5],
         [1, 0, 2],
         None]

# observed values for the 4 vars
obs = [0, MARG_IND, 1, 2]

# testing for each variable for alpha = 0
alphas = [0., 0.1, 1., 10.]


def test_categorical_smoothed_node_create_and_eval():

    for alpha in alphas:
        for i, var in enumerate(vars):
            var_freq = freqs[i]
            smo = CategoricalSmoothedNode(i, var, alpha, var_freq)
            smo.eval(obs[i])
            print('smo values')
            print(smo.log_val)
            ll = compute_smoothed_ll(obs[i], var_freq, var, alpha)
            print('log values')
            print(ll)
            assert_almost_equal(ll, smo.log_val, 15)


def test_categorical_smoothed_node_resmooth():
    for i, var in enumerate(vars):
        alpha = alphas[0]
        var_freq = freqs[i]
        smo = CategoricalSmoothedNode(i, var, alpha, var_freq)
        smo.eval(obs[i])
        print('smo values')
        print(smo.log_val)
        # checking the right value
        ll = compute_smoothed_ll(obs[i], var_freq, var, alpha)
        print('log values')
        print(ll)
        assert_almost_equal(ll, smo.log_val, 15)
        # now setting another alpha
        print('Changing smooth level')
        for alpha_new in alphas:
            smo.smooth_probs(alpha_new)
            smo.eval(obs[i])
            print('smo values')
            print(smo.log_val)
            ll = compute_smoothed_ll(obs[i], var_freq, var, alpha_new)
            print('log values')
            print(ll)
            assert_almost_equal(ll, smo.log_val, 15)


def test_sum_node_is_complete():
    # create a sum node with a scope
    scope = frozenset({0, 2, 7, 13})
    sum_node = SumNode(var_scope=scope)

    # creating children with same scope
    children = [ProductNode(var_scope=scope) for i in range(4)]
    for prod_node in children:
        sum_node.add_child(prod_node, 1.0)

    assert sum_node.is_complete()

    # now altering one child's scope with one less var
    children[0].var_scope = frozenset({0, 7, 13})

    assert sum_node.is_complete() is False

    # now adding one more
    children[0].var_scope = scope
    children[3].var_scope = frozenset({0, 2, 7, 13, 3})

    assert not sum_node.is_complete()

    # now checking with indicator input nodes
    var = 4
    sum_node = SumNode(var_scope=frozenset({var}))
    children = [CategoricalIndicatorNode(var=var, var_val=i)
                for i in range(4)]
    for input_node in children:
        sum_node.add_child(input_node, 1.0)

    assert sum_node.is_complete()


def test_product_node_is_decomposable():
    # create a prod node with a scope
    scope = frozenset({0, 2, 7, 13})

    # creating sub scopes
    sub_scope_1 = frozenset({0})
    sub_scope_2 = frozenset({0, 2})
    sub_scope_3 = frozenset({7})
    sub_scope_4 = frozenset({17})
    sub_scope_5 = frozenset({7, 13})

    # now with decomposable children
    child1 = SumNode(var_scope=sub_scope_2)
    child2 = SumNode(var_scope=sub_scope_5)
    child3 = SumNode(var_scope=sub_scope_2)
    child4 = SumNode(var_scope=sub_scope_1)

    prod_node = ProductNode(var_scope=scope)
    prod_node.add_child(child1)
    prod_node.add_child(child2)

    assert prod_node.is_decomposable()

    prod_node = ProductNode(var_scope=scope)
    prod_node.add_child(child4)
    prod_node.add_child(child1)
    prod_node.add_child(child2)

    assert not prod_node.is_decomposable()

    prod_node = ProductNode(var_scope=scope)
    prod_node.add_child(child4)
    prod_node.add_child(child2)

    assert not prod_node.is_decomposable()

    # now with input nodes
    child5 = CategoricalSmoothedNode(var=0, var_values=2)
    child6 = CategoricalSmoothedNode(var=2, var_values=2)
    child7 = CategoricalSmoothedNode(var=7, var_values=2)
    child8 = CategoricalSmoothedNode(var=13, var_values=2)
    child9 = CategoricalSmoothedNode(var=17, var_values=2)

    prod_node = ProductNode(var_scope=scope)
    prod_node.add_child(child5)
    prod_node.add_child(child6)
    prod_node.add_child(child7)
    prod_node.add_child(child8)

    assert prod_node.is_decomposable()

    prod_node = ProductNode(var_scope=scope)
    prod_node.add_child(child5)
    prod_node.add_child(child6)
    prod_node.add_child(child7)
    prod_node.add_child(child9)

    assert not prod_node.is_decomposable()

    prod_node = ProductNode(var_scope=scope)
    prod_node.add_child(child5)
    prod_node.add_child(child6)
    prod_node.add_child(child8)

    assert not prod_node.is_decomposable()


def test_categorical_smoothed_node_data_smooth():
    data_1 = numpy.array([[1],
                          [0],
                          [1],
                          [0],
                          [1]])

    data_2 = numpy.array([[1, 0],
                          [0, 1],
                          [1, 1],
                          [0, 1],
                          [1, 0]])

    alpha = 0

    freqs = CategoricalSmoothedNode.smooth_freq_from_data(data_1, alpha)
    print('freqs', freqs)

    exp_freqs = CategoricalSmoothedNode.smooth_ll([2 / 5, 3 / 5], alpha)
    print('exp freqs', exp_freqs)
    assert_array_almost_equal(exp_freqs, freqs)

    # now create a node
    input_node = CategoricalSmoothedNode(var=0,
                                         var_values=2,
                                         instances={0, 2, 4})
    input_node.smooth_probs(alpha, data=data_1)
    exp_probs = CategoricalSmoothedNode.smooth_ll([0, 1], alpha)
    print('exp probs', exp_probs)
    print('probs', input_node._var_probs)

    assert_log_array_almost_equal(exp_probs,
                                  input_node._var_probs)

    input_node.smooth_probs(alpha, data=data_2)
    assert_log_array_almost_equal(exp_probs,
                                  input_node._var_probs)

    # TODO: check that data_2 raises an exception


def test_cltree_node_init_and_eval():
    s_data = numpy.array([[1, 1, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 0],
                          [1, 0, 0, 0],
                          [1, 0, 1, 0],
                          [0, 1, 1, 0]])

    features = [0, 1, 2, 3]
    feature_vals = [2, 2, 2, 2]
    clt_node = CLTreeNode(data=s_data,
                          vars=features,
                          var_values=feature_vals,
                          alpha=0.0)

    assert_array_equal(numpy.array(features), clt_node.vars)
    assert_almost_equal(clt_node._alpha, 0.0)
    assert_array_almost_equal(s_data, clt_node._data)

    nico_cltree_tree = numpy.array([-1,  2,  0,  2])
    nico_cltree_lls = numpy.array([-2.01490302054,
                                   -1.20397280433,
                                   -1.20397280433,
                                   -1.79175946923,
                                   -1.60943791243,
                                   -1.60943791243])

    assert_array_equal(nico_cltree_tree, clt_node._cltree._tree)

    print('Created node')
    print(clt_node)

    #
    # evaluating
    for i, instance in enumerate(s_data):
        clt_node.eval(instance)
        assert_almost_equal(nico_cltree_lls[i], clt_node.log_val)
        print(clt_node.log_val, nico_cltree_lls[i])

    #
    # creating now with a subset from data
    features = [0, 2, 3]
    feature_vals = [2, 2, 2]
    clt_node = CLTreeNode(data=s_data[:, features],
                          vars=features,
                          var_values=feature_vals,
                          alpha=0.0)

    nico_cltree_subtree = numpy.array([-1,  0,  1])
    nico_cltree_sublls = numpy.array([-1.09861228867,
                                      -0.69314718056,
                                      -0.69314718056,
                                      -1.79175946923,
                                      -1.09861228867,
                                      -0.69314718056])

    assert_array_equal(numpy.array(features), clt_node.vars)
    assert_array_equal(nico_cltree_subtree,
                       clt_node._cltree._tree)
    for i, instance in enumerate(s_data):
        clt_node.eval(instance)
        assert_almost_equal(nico_cltree_sublls[i], clt_node.log_val)
        print(clt_node.log_val, nico_cltree_sublls[i])


def test_cltree_node_eval():
    #
    # testing for the correctness of data masking while evaluating
    # shall I move this test to CLTree?
    s_data = numpy.random.binomial(n=1, p=0.5, size=(1000, 100))
    print(s_data)

    random_features = numpy.random.choice(s_data.shape[1], 20)
    sub_s_data = s_data[:, random_features]

    clt_node = CLTreeNode(vars=random_features,
                          data=sub_s_data,
                          var_values=numpy.array([2 for i in
                                                  range(sub_s_data.shape[1])]))
    # evaluating on all s_data
    lls = []
    for instance in s_data:
        clt_node.eval(instance)
        lls.append(clt_node.log_val)

    #
    # now do one on the only sub
    clt_node = CLTreeNode(vars=[i for i in range(sub_s_data.shape[1])],
                          data=sub_s_data,
                          var_values=numpy.array([2 for i in
                                                  range(sub_s_data.shape[1])]))
    lls_s = []
    for instance in sub_s_data:
        clt_node.eval(instance)
        lls_s.append(clt_node.log_val)

    # print(lls)
    # print(lls_s)
    assert_array_almost_equal(numpy.array(lls), numpy.array(lls_s))
