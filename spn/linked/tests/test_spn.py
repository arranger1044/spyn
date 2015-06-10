from spn.linked.spn import Spn

from spn.linked.layers import SumLayer
from spn.linked.layers import ProductLayer
from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import CategoricalSmoothedLayer
from spn.linked.layers import CategoricalInputLayer

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import Node
from spn.linked.nodes import CategoricalIndicatorNode

from spn import MARG_IND
from spn import LOG_ZERO
from spn import IS_LOG_ZERO

from spn.tests import logify
from spn.tests import assert_log_array_almost_equal


import numpy

from math import log

from nose.tools import assert_almost_equal

# the SPN build is on 4 binary vars X_1, ..., X_4
vars = numpy.array([2, 2, 2, 2])

alpha = 0.

dicts = [{'var': 0, 'freqs': [1, 1]},
         {'var': 1, 'freqs': [1, 9]},
         {'var': 2, 'freqs': [3, 7]},
         {'var': 3, 'freqs': [6, 4]}]
# the input layer is made of 8 indicator vars for X_1, ..., X_4:
# lambda_{X_1}, lambda_{not X_1} ... lambda_{X_4}, lambda_{not X_4} =
# ind1, ind2, ..., ind7, ind8

# then there is a sum layer of 4 nodes
# sum1 -> 0.5 ind1, 0.5 ind2
# sum2 -> 0.1 ind3, 0.9 ind4
# sum3 -> 0.3 ind5, 0.7 ind6
# sum4 -> 0.6 ind7, 0.4 ind8

# on top a prod layer with 3 nodes:
# prod1 -> sum1, sum2
# prod2 -> sum2, sum3
# prod3 -> sum3, sum4


def build_spn_indicator_layer(the_vars):
    input_layer = CategoricalIndicatorLayer(vars=the_vars)
    return input_layer


def build_spn_smoothed_layer(the_vars, node_dicts, the_alpha):
    input_layer = CategoricalSmoothedLayer(vars=the_vars,
                                           node_dicts=node_dicts,
                                           alpha=the_alpha)
    # print('FREQS')
    # print([node._var_freqs for node in input_layer._nodes])
    # print('PROBS')
    # print([node._var_probs for node in input_layer._nodes])

    return input_layer


def build_spn_layers(input_layer):

    # this is ugly... TODO try to beutify this process
    ind1 = input_layer._nodes[0]
    ind2 = input_layer._nodes[1]
    ind3 = input_layer._nodes[2]
    ind4 = input_layer._nodes[3]
    ind5 = input_layer._nodes[4]
    ind6 = input_layer._nodes[5]
    ind7 = input_layer._nodes[6]
    ind8 = input_layer._nodes[7]

    # creating sum nodes
    sum_node1 = SumNode()
    sum_node2 = SumNode()
    sum_node3 = SumNode()
    sum_node4 = SumNode()

    # linking them with nodes
    sum_node1.add_child(ind1, 0.5)
    sum_node1.add_child(ind2, 0.5)
    sum_node2.add_child(ind3, 0.1)
    sum_node2.add_child(ind4, 0.9)
    sum_node3.add_child(ind5, 0.3)
    sum_node3.add_child(ind6, 0.7)
    sum_node4.add_child(ind7, 0.6)
    sum_node4.add_child(ind8, 0.4)

    # creating sumlayer
    sum_layer = SumLayer([sum_node1,
                          sum_node2,
                          sum_node3,
                          sum_node4])

    # creating product nodes
    prod_node1 = ProductNode()
    prod_node2 = ProductNode()
    prod_node3 = ProductNode()

    # linking them to sum nodes
    prod_node1.add_child(sum_node1)
    prod_node1.add_child(sum_node2)
    prod_node2.add_child(sum_node2)
    prod_node2.add_child(sum_node3)
    prod_node3.add_child(sum_node3)
    prod_node3.add_child(sum_node4)

    # creating a product layer
    prod_layer = ProductLayer([prod_node1,
                               prod_node2,
                               prod_node3])

    return sum_layer, prod_layer

# when a smoothed layer is the input layer
# then there is no sum layer


def build_spn_layers_II(input_layer):

    # this is ugly... TODO try to beutify this process
    ind1 = input_layer._nodes[0]
    ind2 = input_layer._nodes[1]
    ind3 = input_layer._nodes[2]
    ind4 = input_layer._nodes[3]

    # creating product nodes
    prod_node1 = ProductNode()
    prod_node2 = ProductNode()
    prod_node3 = ProductNode()

    # linking them to sum nodes
    prod_node1.add_child(ind1)
    prod_node1.add_child(ind2)
    prod_node2.add_child(ind2)
    prod_node2.add_child(ind3)
    prod_node3.add_child(ind3)
    prod_node3.add_child(ind4)

    # creating a product layer
    prod_layer = ProductLayer([prod_node1,
                               prod_node2,
                               prod_node3])

    return prod_layer

# creating the input matrix for the tests
I = numpy.array([[MARG_IND, MARG_IND, MARG_IND, MARG_IND],
                 [0, 1, MARG_IND, 0],
                 [0, 0, 0, MARG_IND],
                 [1, 0, 0, 0]]).T

# evaluating S(I) shall lead to:
root_vals = numpy.array([[1., 1., 1.],
                         [0.45, 0.9, 0.6],
                         [0.05, 0.03, 0.3],
                         [0.05, 0.03, 0.18]])

# whose logs are
logify(root_vals)


def test_spn_construction_by_add_and_evaluation():
    spn = Spn()

    # building the same levels
    input_layer = build_spn_indicator_layer(vars)
    sum_layer, prod_layer = build_spn_layers(input_layer)

    # adding all layers to the spn
    spn.set_input_layer(input_layer)
    spn.add_layer(sum_layer)
    spn.add_layer(prod_layer)

    res = spn.eval(I)
    print('First evaluation')
    print(res)
    assert_log_array_almost_equal(root_vals, res)


def test_spn_construction_by_add_and_evaluation_II():
    spn = Spn()

    # print('empty spn')
    # print(spn)

    input_layer = build_spn_smoothed_layer(vars, dicts, alpha)
    prod_layer = build_spn_layers_II(input_layer)

    # adding all layers to the spn
    spn.set_input_layer(input_layer)
    spn.add_layer(prod_layer)

    # print('created spn')
    # print(spn)

    res = spn.eval(I)
    print('First smoothed evaluation')
    print(res)
    assert_log_array_almost_equal(root_vals, res)


def test_spn_construction_by_init_and_evaluation():

    # building the same levels
    input_layer = build_spn_indicator_layer(vars)
    sum_layer, prod_layer = build_spn_layers(input_layer)

    spn = Spn(input_layer=input_layer, layers=[sum_layer, prod_layer])

    res = spn.eval(I)
    print('First evaluation')
    print(res)

    assert_log_array_almost_equal(root_vals, res)


def test_spn_backprop():
    # create initial layer
    node1 = Node()
    node2 = Node()
    node3 = Node()
    node4 = Node()
    node5 = Node()

    input_layer = CategoricalInputLayer([node1, node2,
                                         node3, node4,
                                         node5])

    # top layer made by 3 sum nodes
    sum1 = SumNode()
    sum2 = SumNode()
    sum3 = SumNode()

    # linking to input nodes
    weight11 = 0.3
    sum1.add_child(node1, weight11)
    weight12 = 0.3
    sum1.add_child(node2, weight12)
    weight13 = 0.4
    sum1.add_child(node3, weight13)

    weight22 = 0.15
    sum2.add_child(node2, weight22)
    weight23 = 0.15
    sum2.add_child(node3, weight23)
    weight24 = 0.7
    sum2.add_child(node4, weight24)

    weight33 = 0.4
    sum3.add_child(node3, weight33)
    weight34 = 0.25
    sum3.add_child(node4, weight34)
    weight35 = 0.35
    sum3.add_child(node5, weight35)

    sum_layer = SumLayer([sum1, sum2, sum3])

    # another layer with two product nodes
    prod1 = ProductNode()
    prod2 = ProductNode()

    prod1.add_child(sum1)
    prod1.add_child(sum2)
    prod2.add_child(sum2)
    prod2.add_child(sum3)

    prod_layer = ProductLayer([prod1, prod2])

    # root layer, double sum
    root1 = SumNode()
    root2 = SumNode()

    weightr11 = 0.5
    root1.add_child(prod1, weightr11)
    weightr12 = 0.5
    root1.add_child(prod2, weightr12)

    weightr21 = 0.9
    root2.add_child(prod1, weightr21)
    weightr22 = 0.1
    root2.add_child(prod2, weightr22)

    root_layer = SumLayer([root1, root2])
    # root_layer = SumLayer([root1])

    # create the spn
    spn = Spn(input_layer=input_layer,
              layers=[sum_layer, prod_layer, root_layer])

    # setting the input values
    val1 = 0.0
    node1.set_val(val1)
    val2 = 0.5
    node2.set_val(val2)
    val3 = 0.3
    node3.set_val(val3)
    val4 = 1.0
    node4.set_val(val4)
    val5 = 0.0
    node5.set_val(val5)

    # evaluating the spn
    res = spn.test_eval()
    print('spn eval\'d', res)

    # backprop
    spn.backprop()

    # computing derivatives by hand
    # topdown: root layer
    root_der = 1.0
    log_root_der = log(root_der)

    # print('root ders', root1.log_der, root2.log_der)
    print('root ders', root1.log_der)
    assert_almost_equal(log_root_der, root1.log_der)
    assert_almost_equal(log_root_der, root2.log_der)

    # product layer
    prod_der1 = (root_der * weightr11 +
                 root_der * weightr21)

    prod_der2 = (root_der * weightr12 +
                 root_der * weightr22)

    # prod_der1 = (root_der * weightr11)
    # prod_der2 = (root_der * weightr12)

    log_prod_der1 = log(prod_der1) if prod_der1 > 0.0 else LOG_ZERO
    log_prod_der2 = log(prod_der2) if prod_der2 > 0.0 else LOG_ZERO

    print('found  prod ders', prod1.log_der, prod2.log_der)
    print('expect prod ders', log_prod_der1, log_prod_der2)

    if IS_LOG_ZERO(log_prod_der1):
        assert IS_LOG_ZERO(prod1.log_der)
    else:
        assert_almost_equal(log_prod_der1, prod1.log_der)
    if IS_LOG_ZERO(log_prod_der2):
        assert IS_LOG_ZERO(prod2.log_der)
    else:
        assert_almost_equal(log_prod_der2, prod2.log_der)

    # sum layer
    sum_der1 = (
        prod_der1 * (weight22 * val2 +
                     weight23 * val3 +
                     weight24 * val4))

    log_sum_der1 = log(sum_der1) if sum_der1 > 0.0 else LOG_ZERO

    sum_der2 = (prod_der1 * (weight11 * val1 +
                             weight12 * val2 +
                             weight13 * val3) +
                prod_der2 * (weight33 * val3 +
                             weight34 * val4 +
                             weight35 * val5))

    log_sum_der2 = log(sum_der2) if sum_der2 > 0.0 else LOG_ZERO

    sum_der3 = (prod_der2 * (weight22 * val2 +
                             weight23 * val3 +
                             weight24 * val4))

    log_sum_der3 = log(sum_der3) if sum_der3 > 0.0 else LOG_ZERO

    print('expected sum ders', log_sum_der1,
          log_sum_der2,
          log_sum_der3)
    print('found    sum ders', sum1.log_der,
          sum2.log_der,
          sum3.log_der)

    if IS_LOG_ZERO(log_sum_der1):
        assert IS_LOG_ZERO(sum1.log_der)
    else:
        assert_almost_equal(log_sum_der1, sum1.log_der)
    if IS_LOG_ZERO(log_sum_der2):
        assert IS_LOG_ZERO(sum2.log_der)
    else:
        assert_almost_equal(log_sum_der2, sum2.log_der)
    if IS_LOG_ZERO(log_sum_der3):
        assert IS_LOG_ZERO(sum3.log_der)
    else:
        assert_almost_equal(log_sum_der3, sum3.log_der)

    # final level, the first one
    try:
        log_der1 = log(sum_der1 * weight11)
    except:
        log_der1 = LOG_ZERO

    try:
        log_der2 = log(sum_der1 * weight12 +
                       sum_der2 * weight22)
    except:
        log_der2 = LOG_ZERO

    try:
        log_der3 = log(sum_der1 * weight13 +
                       sum_der2 * weight23 +
                       sum_der3 * weight33)
    except:
        log_der3 = LOG_ZERO

    try:
        log_der4 = log(sum_der2 * weight24 +
                       sum_der3 * weight34)
    except:
        log_der4 = LOG_ZERO

    try:
        log_der5 = log(sum_der3 * weight35)
    except:
        log_der5 = LOG_ZERO

    # printing, just in case
    print('child log der', node1.log_der, node2.log_der,
          node3.log_der, node4.log_der, node5.log_der)
    print('exact log der', log_der1, log_der2, log_der3,
          log_der4, log_der5)

    if IS_LOG_ZERO(log_der1):
        assert IS_LOG_ZERO(node1.log_der)
    else:
        assert_almost_equal(log_der1, node1.log_der, 15)
    if IS_LOG_ZERO(log_der2):
        assert IS_LOG_ZERO(node2.log_der)
    else:
        assert_almost_equal(log_der2, node2.log_der, 15)
    if IS_LOG_ZERO(log_der3):
        assert IS_LOG_ZERO(node3.log_der)
    else:
        assert_almost_equal(log_der3, node3.log_der, 15)
    if IS_LOG_ZERO(log_der4):
        assert IS_LOG_ZERO(node4.log_der)
    else:
        assert_almost_equal(log_der4, node4.log_der, 15)
    if IS_LOG_ZERO(log_der5):
        assert IS_LOG_ZERO(node5.log_der)
    else:
        assert_almost_equal(log_der5, node5.log_der, 15)


def test_spn_mpe_eval_and_traversal():
    # create initial layer
    node1 = Node()
    node2 = Node()
    node3 = Node()
    node4 = Node()
    node5 = Node()

    input_layer = CategoricalInputLayer([node1, node2,
                                         node3, node4,
                                         node5])

    # top layer made by 3 sum nodes
    sum1 = SumNode()
    sum2 = SumNode()
    sum3 = SumNode()

    # linking to input nodes
    weight11 = 0.3
    sum1.add_child(node1, weight11)
    weight12 = 0.3
    sum1.add_child(node2, weight12)
    weight13 = 0.4
    sum1.add_child(node3, weight13)

    weight22 = 0.15
    sum2.add_child(node2, weight22)
    weight23 = 0.15
    sum2.add_child(node3, weight23)
    weight24 = 0.7
    sum2.add_child(node4, weight24)

    weight33 = 0.4
    sum3.add_child(node3, weight33)
    weight34 = 0.25
    sum3.add_child(node4, weight34)
    weight35 = 0.35
    sum3.add_child(node5, weight35)

    sum_layer = SumLayer([sum1, sum2, sum3])

    # another layer with two product nodes
    prod1 = ProductNode()
    prod2 = ProductNode()

    prod1.add_child(sum1)
    prod1.add_child(sum2)
    prod2.add_child(sum2)
    prod2.add_child(sum3)

    prod_layer = ProductLayer([prod1, prod2])

    # root layer, double sum
    root1 = SumNode()
    root2 = SumNode()

    weightr11 = 0.5
    root1.add_child(prod1, weightr11)
    weightr12 = 0.5
    root1.add_child(prod2, weightr12)

    weightr21 = 0.9
    root2.add_child(prod1, weightr21)
    weightr22 = 0.1
    root2.add_child(prod2, weightr22)

    root_layer = SumLayer([root1, root2])

    # create the spn
    spn = Spn(input_layer=input_layer,
              layers=[sum_layer, prod_layer, root_layer])

    print('===================')
    print(spn)
    print('===================')

    # setting the input values
    val1 = 0.0
    node1.set_val(val1)
    val2 = 0.5
    node2.set_val(val2)
    val3 = 0.3
    node3.set_val(val3)
    val4 = 1.0
    node4.set_val(val4)
    val5 = 0.0
    node5.set_val(val5)

    # evaluating the spn with MPE inference
    res = spn.test_mpe_eval()
    print('spn eval\'d', res)

    # testing it
    #
    # testing the max layer
    max1 = max(val1 * weight11,
               val2 * weight12,
               val3 * weight13)
    max2 = max(val2 * weight22,
               val3 * weight23,
               val4 * weight24)
    max3 = max(val3 * weight33,
               val4 * weight34,
               val5 * weight35)
    log_max1 = log(max1) if not numpy.isclose(max1, 0) else LOG_ZERO
    log_max2 = log(max2) if not numpy.isclose(max2, 0) else LOG_ZERO
    log_max3 = log(max3) if not numpy.isclose(max3, 0) else LOG_ZERO

    print('expected max vals {0}, {1}, {2}'.format(log_max1,
                                                   log_max2,
                                                   log_max3))
    print('found    max vals {0}, {1}, {2}'.format(sum1.log_val,
                                                   sum2.log_val,
                                                   sum3.log_val))
    if IS_LOG_ZERO(log_max1):
        assert IS_LOG_ZERO(sum1.log_val)
    else:
        assert_almost_equal(log_max1, sum1.log_val)
    if IS_LOG_ZERO(log_max2):
        assert IS_LOG_ZERO(sum2.log_val)
    else:
        assert_almost_equal(log_max2, sum2.log_val)
    if IS_LOG_ZERO(log_max3):
        assert IS_LOG_ZERO(sum3.log_val)
    else:
        assert_almost_equal(log_max3, sum3.log_val)

    # product layer is assumed to be fine, but let's check
    # it anyways
    prod_val1 = max1 * max2
    prod_val2 = max2 * max3
    prod_log_val1 = log_max1 + log_max2
    prod_log_val2 = log_max2 + log_max3

    print('exp prod vals {0}, {1}'.format(prod_log_val1,
                                          prod_log_val2))
    print('rea prod vals {0}, {1}'.format(prod1.log_val,
                                          prod2.log_val))
    if IS_LOG_ZERO(prod_log_val1):
        assert IS_LOG_ZERO(prod1.log_val)
    else:
        assert_almost_equal(prod_log_val1, prod1.log_val)

    if IS_LOG_ZERO(prod_log_val2):
        assert IS_LOG_ZERO(prod2.log_val)
    else:
        assert_almost_equal(prod_log_val2, prod2.log_val)

    # root layer, again a sum layer
    root_val1 = max(prod_val1 * weightr11,
                    prod_val2 * weightr12)
    root_val2 = max(prod_val1 * weightr21,
                    prod_val2 * weightr22)
    root_log_val1 = log(root_val1) if not numpy.isclose(
        root_val1, 0) else LOG_ZERO
    root_log_val2 = log(root_val2) if not numpy.isclose(
        root_val2, 0) else LOG_ZERO

    print('exp root vals {0}, {1}'.format(root_log_val1,
                                          root_log_val2))
    print('found ro vals {0}, {1}'.format(root1.log_val,
                                          root2.log_val))

    if IS_LOG_ZERO(root_log_val1):
        assert IS_LOG_ZERO(root1.log_val)
    else:
        assert_almost_equal(root_log_val1, root1.log_val)
    if IS_LOG_ZERO(root_log_val2):
        assert IS_LOG_ZERO(root2.log_val)
    else:
        assert_almost_equal(root_log_val2, root2.log_val)

    # now we are traversing top down the net
    print('mpe traversing')
    for i, j, k in spn.mpe_traversal():
        print(i, j, k)


def create_valid_toy_spn():
    # root layer
    whole_scope = frozenset({0, 1, 2, 3})
    root_node = SumNode(var_scope=whole_scope)
    root_layer = SumLayer([root_node])

    # prod layer
    prod_node_1 = ProductNode(var_scope=whole_scope)
    prod_node_2 = ProductNode(var_scope=whole_scope)
    prod_layer_1 = ProductLayer([prod_node_1, prod_node_2])

    root_node.add_child(prod_node_1, 0.5)
    root_node.add_child(prod_node_2, 0.5)

    # sum layer
    scope_1 = frozenset({0, 1})
    scope_2 = frozenset({2})
    scope_3 = frozenset({3})
    scope_4 = frozenset({2, 3})

    sum_node_1 = SumNode(var_scope=scope_1)
    sum_node_2 = SumNode(var_scope=scope_2)
    sum_node_3 = SumNode(var_scope=scope_3)
    sum_node_4 = SumNode(var_scope=scope_4)

    prod_node_1.add_child(sum_node_1)
    prod_node_1.add_child(sum_node_2)
    prod_node_1.add_child(sum_node_3)

    prod_node_2.add_child(sum_node_1)
    prod_node_2.add_child(sum_node_4)

    sum_layer_1 = SumLayer([sum_node_1, sum_node_2,
                            sum_node_3, sum_node_4])

    # another product layer
    prod_node_3 = ProductNode(var_scope=scope_1)
    prod_node_4 = ProductNode(var_scope=scope_1)

    prod_node_5 = ProductNode(var_scope=scope_4)
    prod_node_6 = ProductNode(var_scope=scope_4)

    sum_node_1.add_child(prod_node_3, 0.5)
    sum_node_1.add_child(prod_node_4, 0.5)

    sum_node_4.add_child(prod_node_5, 0.5)
    sum_node_4.add_child(prod_node_6, 0.5)

    prod_layer_2 = ProductLayer([prod_node_3, prod_node_4,
                                 prod_node_5, prod_node_6])

    # last sum one
    scope_5 = frozenset({0})
    scope_6 = frozenset({1})

    sum_node_5 = SumNode(var_scope=scope_5)
    sum_node_6 = SumNode(var_scope=scope_6)
    sum_node_7 = SumNode(var_scope=scope_5)
    sum_node_8 = SumNode(var_scope=scope_6)

    sum_node_9 = SumNode(var_scope=scope_2)
    sum_node_10 = SumNode(var_scope=scope_3)
    sum_node_11 = SumNode(var_scope=scope_2)
    sum_node_12 = SumNode(var_scope=scope_3)

    prod_node_3.add_child(sum_node_5)
    prod_node_3.add_child(sum_node_6)
    prod_node_4.add_child(sum_node_7)
    prod_node_4.add_child(sum_node_8)

    prod_node_5.add_child(sum_node_9)
    prod_node_5.add_child(sum_node_10)
    prod_node_6.add_child(sum_node_11)
    prod_node_6.add_child(sum_node_12)

    sum_layer_2 = SumLayer([sum_node_5, sum_node_6,
                            sum_node_7, sum_node_8,
                            sum_node_9, sum_node_10,
                            sum_node_11, sum_node_12])

    # input layer
    vars = [2, 3, 2, 2]
    input_layer = CategoricalIndicatorLayer(vars=vars)
    last_sum_nodes = [sum_node_2, sum_node_3,
                      sum_node_5, sum_node_6,
                      sum_node_7, sum_node_8,
                      sum_node_9, sum_node_10,
                      sum_node_11, sum_node_12]
    for sum_node in last_sum_nodes:
        (var_scope,) = sum_node.var_scope
        for input_node in input_layer.nodes():
            if input_node.var == var_scope:
                sum_node.add_child(input_node, 1.0)

    spn = Spn(input_layer=input_layer,
              layers=[sum_layer_2, prod_layer_2,
                      sum_layer_1, prod_layer_1,
                      root_layer])

    # print(spn)
    return spn


def test_spn_is_valid():
    # create an SPN
    spn = create_valid_toy_spn()

    assert spn.is_complete()
    assert spn.is_decomposable()
    assert spn.is_valid()

    # now changing completeness
    sum_node_2.add_child(input_layer._nodes[0], 1.0)
    assert not spn.is_complete()
    assert spn.is_decomposable()
    assert not spn.is_valid()

    # now even decomposability
    prod_node_6.var_scope = scope_1
    assert not spn.is_complete()
    assert not spn.is_decomposable()
    assert not spn.is_valid()

# two sum layers, 2 nodes first, 3 second (top down)
weights_ds = [[[0.2, 0.8]],
              [[0.15, 0.85],
               [0.5, 0.25, 0.25],
               [0.1, 0.9]]]


def test_spn_set_get_weights():
    # create a simple spn
    root_node = SumNode()
    root_layer = SumLayer([root_node])

    prod_node_1 = ProductNode()
    prod_node_2 = ProductNode()
    root_node.add_child(prod_node_1, 0.5)
    root_node.add_child(prod_node_2, 0.5)
    prod_layer = ProductLayer([prod_node_1,
                               prod_node_2])

    sum_node_1 = SumNode()
    sum_node_2 = SumNode()
    sum_node_3 = SumNode()
    prod_node_1.add_child(sum_node_1)
    prod_node_1.add_child(sum_node_2)
    prod_node_2.add_child(sum_node_2)
    prod_node_2.add_child(sum_node_3)
    sum_layer = SumLayer([sum_node_1, sum_node_2,
                          sum_node_3])

    ind_node_1 = CategoricalIndicatorNode(var=0, var_val=1)
    ind_node_2 = CategoricalIndicatorNode(var=0, var_val=1)
    ind_node_3 = CategoricalIndicatorNode(var=0, var_val=1)
    ind_node_4 = CategoricalIndicatorNode(var=0, var_val=1)
    ind_node_5 = CategoricalIndicatorNode(var=0, var_val=1)
    input_layer = CategoricalInputLayer(nodes=[ind_node_1,
                                               ind_node_2,
                                               ind_node_3,
                                               ind_node_4,
                                               ind_node_5])
    sum_node_1.add_child(ind_node_1, 0.2)
    sum_node_1.add_child(ind_node_2, 0.2)
    sum_node_2.add_child(ind_node_2, 0.2)
    sum_node_2.add_child(ind_node_3, 0.2)
    sum_node_2.add_child(ind_node_4, 0.2)
    sum_node_3.add_child(ind_node_4, 0.2)
    sum_node_3.add_child(ind_node_5, 0.2)

    spn = Spn(input_layer=input_layer,
              layers=[sum_layer, prod_layer, root_layer])

    print(spn)

    # storing these weights
    curr_weights = spn.get_weights()

    # setting the new weights
    spn.set_weights(weights_ds)

    # getting them again
    new_weights = spn.get_weights()

    # comparing them
    assert new_weights == weights_ds

    # now setting back the previous one
    spn.set_weights(curr_weights)

    # getting them back again
    old_weights = spn.get_weights()

    # and checking
    assert old_weights == curr_weights


def test_to_text():
    spn = create_valid_toy_spn()
    spn.to_text('test.spn')
