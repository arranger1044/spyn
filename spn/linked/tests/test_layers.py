from spn.linked.layers import Layer
from spn.linked.layers import ProductLayer
from spn.linked.layers import SumLayer
from spn.linked.layers import CategoricalInputLayer
from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import CategoricalSmoothedLayer
from spn.linked.layers import CategoricalCLInputLayer

from spn.linked.nodes import Node
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import CategoricalIndicatorNode
from spn.linked.nodes import CLTreeNode

from spn.linked.spn import Spn

from spn.tests import compute_smoothed_ll
from spn.tests import PRECISION

from spn import LOG_ZERO
from spn import MARG_IND
from spn import IS_LOG_ZERO

from math import log

import numpy

from nose.tools import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


def test_layer_create():
    # creating generic nodes
    node1 = Node()
    node2 = Node()
    node3 = Node()

    # and a generic Layer
    layer = Layer()
    layer.add_node(node1)
    layer.add_node(node2)
    layer.add_node(node3)

    layer2 = Layer([node1, node2, node3])
    assert layer._nodes == layer2._nodes


def test_sum_layer_create_and_eval():
    # creating generic nodes
    node1 = Node()
    node2 = Node()
    node3 = Node()

    # whose values are
    val1 = 1.
    val2 = 1.
    val3 = 0.
    node1.set_val(val1)
    node2.set_val(val2)
    node3.set_val(val3)

    # setting weights
    weight11 = 0.2
    weight12 = 0.3
    weight13 = 0.5

    weight21 = 0.3
    weight22 = 0.7

    weight32 = 0.4
    weight33 = 0.6

    # creating sum nodes
    sum1 = SumNode()
    sum2 = SumNode()
    sum3 = SumNode()

    # adding children
    sum1.add_child(node1, weight11)
    sum1.add_child(node2, weight12)
    sum1.add_child(node3, weight13)

    sum2.add_child(node1, weight21)
    sum2.add_child(node2, weight22)

    sum3.add_child(node2, weight32)
    sum3.add_child(node3, weight33)

    # adding to layer
    sum_layer = SumLayer([sum1, sum2, sum3])

    # evaluation
    sum_layer.eval()

    # computing 'log values by hand'
    layer_evals = sum_layer.node_values()
    print('Layer eval nodes')
    print(layer_evals)

    logval1 = log(weight11 * val1 +
                  weight12 * val2 +
                  weight13 * val3)
    logval2 = log(weight21 * val1 +
                  weight22 * val2)
    logval3 = log(weight32 * val2 +
                  weight33 * val3)
    logvals = [logval1, logval2, logval3]

    print('log vals')
    print(logvals)
    # checking for correctness
    for logval, eval in zip(logvals, layer_evals):
        assert_almost_equal(logval, eval, PRECISION)


def test_product_layer_create_and_eval():
    # creating generic nodes
    node1 = Node()
    node2 = Node()
    node3 = Node()

    # whose values are
    val1 = 0.8
    val2 = 1.
    val3 = 0.
    node1.set_val(val1)
    node2.set_val(val2)
    node3.set_val(val3)

    # creating product nodes
    prod1 = ProductNode()
    prod2 = ProductNode()
    prod3 = ProductNode()

    # adding children
    prod1.add_child(node1)
    prod1.add_child(node2)

    prod2.add_child(node1)
    prod2.add_child(node3)

    prod3.add_child(node2)
    prod3.add_child(node3)

    # adding product nodes to layer
    product_layer = ProductLayer([prod1, prod2, prod3])

    # evaluating
    product_layer.eval()

    # getting log vals
    layer_evals = product_layer.node_values()
    print('layer eval nodes')
    print(layer_evals)

    # computing our values
    prodval1 = val1 * val2
    logval1 = log(prodval1) if prodval1 > 0. else LOG_ZERO
    prodval2 = val1 * val3
    logval2 = log(prodval2) if prodval2 > 0. else LOG_ZERO
    prodval3 = val2 * val3
    logval3 = log(prodval3) if prodval3 > 0. else LOG_ZERO
    logvals = [logval1, logval2, logval3]
    print('log vals')
    print(logvals)

    for logval, eval in zip(logvals, layer_evals):
        if logval == LOG_ZERO:
            # for zero log check this way for correctness
            assert IS_LOG_ZERO(eval) is True
        else:
            assert_almost_equal(logval, eval, PRECISION)


vars = [2, 2, 3, 4]
freqs = [[1, 2],
         [5, 5],
         [1, 0, 2],
         None]
obs = [0, MARG_IND, 1, 2]


def test_categorical_input_layer():
    print('categorical input layer')
    # I could loop through alpha as well
    alpha = 0.1

    for var_id1 in range(len(vars)):
        for var_id2 in range(len(vars)):
            for var_val1 in range(vars[var_id1]):
                print('varid1, varid2, varval1',
                      var_id1, var_id2, var_val1)
                # var_id1 = 0
                # var_val1 = 0
                node1 = CategoricalIndicatorNode(var_id1,
                                                 var_val1)
                # var_id2 = 0
                var_vals2 = vars[var_id2]
                node2 = CategoricalSmoothedNode(
                    var_id2, var_vals2, alpha, freqs[var_id2])

                # creating the generic input layer
                input_layer = CategoricalInputLayer([node1,
                                                     node2])

                # evaluating according to an observation
                input_layer.eval(obs)

                layer_evals = input_layer.node_values()
                print('layer eval nodes')
                print(layer_evals)

                # computing evaluation by hand
                val1 = 1 if var_val1 == obs[var_id1] or obs[
                    var_id1] == MARG_IND else 0
                logval1 = log(val1) if val1 == 1 else LOG_ZERO

                logval2 = compute_smoothed_ll(
                    obs[var_id2], freqs[var_id2], vars[var_id2], alpha)
                logvals = [logval1, logval2]
                print('log vals')
                print(logvals)

                for logval, eval in zip(logvals, layer_evals):
                    if logval == LOG_ZERO:
                        # for zero log check this way for correctness
                        assert IS_LOG_ZERO(eval) is True
                    else:
                        assert_almost_equal(logval, eval, PRECISION)


def test_categorical_indicator_layer_eval():
    # create a layer from vars
    input_layer = CategoricalIndicatorLayer(vars=vars)
    # evaluating for obs
    input_layer.eval(obs)
    # getting values
    layer_evals = input_layer.node_values()
    print('layer eval nodes')
    print(layer_evals)
    # bulding the log vals by hand
    log_vals = []
    for var, obs_val in zip(vars, obs):
        var_log_vals = None
        if obs_val == MARG_IND:
            # all 1s
            var_log_vals = [0. for i in range(var)]
        else:
            # just one is 1, the rest are 0
            var_log_vals = [LOG_ZERO for i in range(var)]
            var_log_vals[obs_val] = 0.
        # concatenate vals
        log_vals.extend(var_log_vals)
    print('log vals')
    print(log_vals)
    assert log_vals == layer_evals


# a dictionary for vars
dicts = [{'var': 0, 'freqs': [6, 5]},
         {'var': 0},
         {'var': 1, 'freqs': [1, 1]},
         {'var': 2, 'freqs': [0, 1, 1]},
         {'var': 2, 'freqs': [10, 10, 1]},
         {'var': 3},
         {'var': 3, 'freqs': [6, 5, 1, 1]}]


def test_categorical_smoothed_layer_eval():
    alpha = 0.1

    # creating input layer
    input_layer = CategoricalSmoothedLayer(vars=vars,
                                           node_dicts=dicts,
                                           alpha=alpha)
    # evaluate it
    input_layer.eval(obs)
    # getting values
    layer_evals = input_layer.node_values()
    print('layer eval nodes')
    print(layer_evals)

    # crafting by hand
    logvals = []
    for node_dict in dicts:
        var_id = node_dict['var']
        freqs = node_dict['freqs'] if 'freqs' in node_dict else None

        logvals.append(compute_smoothed_ll(obs[var_id],
                                           freqs,
                                           vars[var_id],
                                           alpha))
    print('log vals')
    print(logvals)
    assert logvals == layer_evals

    # now changing alphas
    print('\nCHANGING ALPHAS\n')
    alphas = [0., 0.1, 1., 10.]
    for alpha_new in alphas:
        print('alpha', alpha_new)
        input_layer.smooth_probs(alpha_new)
        # evaluating again
        input_layer.eval(obs)
        # getting values
        layer_evals = input_layer.node_values()
        print('layer evals')
        print(layer_evals)
        logvals = []
        for node_dict in dicts:
            var_id = node_dict['var']
            freqs = node_dict['freqs'] if 'freqs' in node_dict else None

            logvals.append(compute_smoothed_ll(obs[var_id],
                                               freqs,
                                               vars[var_id],
                                               alpha_new))
        print('logvals')
        print(logvals)
        assert_array_almost_equal(logvals, layer_evals)


def test_categorical_indicator_layer_vars():
    # create indicator nodes first
    ind1 = CategoricalIndicatorNode(var=0, var_val=0)
    ind2 = CategoricalIndicatorNode(var=3, var_val=0)
    ind3 = CategoricalIndicatorNode(var=3, var_val=1)
    ind4 = CategoricalIndicatorNode(var=2, var_val=0)
    ind5 = CategoricalIndicatorNode(var=1, var_val=1)
    ind6 = CategoricalIndicatorNode(var=2, var_val=1)
    ind7 = CategoricalIndicatorNode(var=1, var_val=0)
    ind8 = CategoricalIndicatorNode(var=0, var_val=1)
    ind9 = CategoricalIndicatorNode(var=2, var_val=2)
    ind10 = CategoricalIndicatorNode(var=3, var_val=2)
    ind11 = CategoricalIndicatorNode(var=3, var_val=3)

    # building the layer from nodes
    layer = CategoricalIndicatorLayer(nodes=[ind1, ind2,
                                             ind3, ind4,
                                             ind5, ind6,
                                             ind7, ind8,
                                             ind9, ind10, ind11])

    # checking for the construction of the vars property
    layer_vars = layer.vars()

    assert vars == layer_vars


def test_categorical_smoothed_layer_vars():
    # creating single nodes in a list from dicts
    nodes = [CategoricalSmoothedNode(dict_i['var'], vars[dict_i['var']])
             for dict_i in dicts]
    # creating the layer
    layer = CategoricalSmoothedLayer(nodes)

    # evaluating for the construction of vars
    layer_vars = layer.vars()

    assert vars == layer_vars


def test_prod_layer_backprop():
    # input layer made of 5 generic nodes
    node1 = Node()
    node2 = Node()
    node3 = Node()
    node4 = Node()
    node5 = Node()

    input_layer = CategoricalInputLayer([node1, node2,
                                         node3, node4,
                                         node5])

    # top layer made by 3 prod nodes
    prod1 = ProductNode()
    prod2 = ProductNode()
    prod3 = ProductNode()

    # linking to input nodes
    prod1.add_child(node1)
    prod1.add_child(node2)
    prod1.add_child(node3)

    prod2.add_child(node2)
    prod2.add_child(node3)
    prod2.add_child(node4)

    prod3.add_child(node3)
    prod3.add_child(node4)
    prod3.add_child(node5)

    prod_layer = ProductLayer([prod1, prod2, prod3])

    # setting input values
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

    print('input', [node.log_val for node in input_layer.nodes()])
    # evaluating
    prod_layer.eval()
    print('eval\'d layer:', prod_layer.node_values())

    # set the parent derivatives
    prod_der1 = 1.0
    prod1.log_der = log(prod_der1)

    prod_der2 = 1.0
    prod2.log_der = log(prod_der2)

    prod_der3 = 0.0
    prod3.log_der = LOG_ZERO

    # back prop layer wise
    prod_layer.backprop()

    # check for correctness
    try:
        log_der1 = log(prod_der1 * val2 * val3)
    except:
        log_der1 = LOG_ZERO

    try:
        log_der2 = log(prod_der1 * val1 * val3 +
                       prod_der2 * val3 * val4)
    except:
        log_der2 = LOG_ZERO

    try:
        log_der3 = log(prod_der2 * val2 * val4 +
                       prod_der3 * val4 * val5 +
                       prod_der1 * val1 * val2)
    except:
        log_der3 = LOG_ZERO

    try:
        log_der4 = log(prod_der2 * val2 * val3 +
                       prod_der3 * val3 * val5)
    except:
        log_der4 = LOG_ZERO

    try:
        log_der5 = log(prod_der3 * val3 * val4)
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

    # resetting derivatives
    node1.log_der = LOG_ZERO
    node2.log_der = LOG_ZERO
    node3.log_der = LOG_ZERO
    node4.log_der = LOG_ZERO
    node5.log_der = LOG_ZERO

    # setting new values as inputs
    val1 = 0.0
    node1.set_val(val1)
    val2 = 0.0
    node2.set_val(val2)
    val3 = 0.3
    node3.set_val(val3)
    val4 = 1.0
    node4.set_val(val4)
    val5 = 1.0
    node5.set_val(val5)

    # evaluating again
    prod_layer.eval()
    print('eval\'d layer:', prod_layer.node_values())

    # set the parent derivatives
    prod_der1 = 1.0
    prod1.log_der = log(prod_der1)

    prod_der2 = 1.0
    prod2.log_der = log(prod_der2)

    prod_der3 = 0.0
    prod3.log_der = LOG_ZERO

    # back prop layer wise
    prod_layer.backprop()

    # check for correctness
    try:
        log_der1 = log(prod_der1 * val2 * val3)
    except:
        log_der1 = LOG_ZERO

    try:
        log_der2 = log(prod_der1 * val1 * val3 +
                       prod_der2 * val3 * val4)
    except:
        log_der2 = LOG_ZERO

    try:
        log_der3 = log(prod_der2 * val2 * val4 +
                       prod_der3 * val4 * val5 +
                       prod_der1 * val1 * val2)
    except:
        log_der3 = LOG_ZERO

    try:
        log_der4 = log(prod_der2 * val2 * val3 +
                       prod_der3 * val3 * val5)
    except:
        log_der4 = LOG_ZERO

    try:
        log_der5 = log(prod_der3 * val3 * val4)
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


def test_sum_layer_backprop():
        # input layer made of 5 generic nodes
    node1 = Node()
    node2 = Node()
    node3 = Node()
    node4 = Node()
    node5 = Node()

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

    # setting input values
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

    # evaluating
    sum_layer.eval()
    print('eval\'d layer:', sum_layer.node_values())

    # set the parent derivatives
    sum_der1 = 1.0
    sum1.log_der = log(sum_der1)

    sum_der2 = 1.0
    sum2.log_der = log(sum_der2)

    sum_der3 = 0.0
    sum3.log_der = LOG_ZERO

    # back prop layer wise
    sum_layer.backprop()

    # check for correctness
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

    # updating weights
    eta = 0.1
    sum_layer.update_weights(Spn.test_weight_update, 0)
    # checking for correctness
    weight_u11 = sum_der1 * val1 * eta + weight11
    weight_u12 = sum_der1 * val2 * eta + weight12
    weight_u13 = sum_der1 * val3 * eta + weight13

    weight_u22 = sum_der2 * val2 * eta + weight22
    weight_u23 = sum_der2 * val3 * eta + weight23
    weight_u24 = sum_der2 * val4 * eta + weight24

    weight_u33 = sum_der3 * val3 * eta + weight33
    weight_u34 = sum_der3 * val4 * eta + weight34
    weight_u35 = sum_der3 * val5 * eta + weight35

    # normalizing
    weight_sum1 = weight_u11 + weight_u12 + weight_u13
    weight_sum2 = weight_u22 + weight_u23 + weight_u24
    weight_sum3 = weight_u33 + weight_u34 + weight_u35

    weight_u11 = weight_u11 / weight_sum1
    weight_u12 = weight_u12 / weight_sum1
    weight_u13 = weight_u13 / weight_sum1

    weight_u22 = weight_u22 / weight_sum2
    weight_u23 = weight_u23 / weight_sum2
    weight_u24 = weight_u24 / weight_sum2

    weight_u33 = weight_u33 / weight_sum3
    weight_u34 = weight_u34 / weight_sum3
    weight_u35 = weight_u35 / weight_sum3

    print('expected weights', weight_u11, weight_u12, weight_u13,
          weight_u22, weight_u23, weight_u24,
          weight_u33, weight_u34, weight_u35)
    print('found weights', sum1.weights[0], sum1.weights[1], sum1.weights[2],
          sum2.weights[0], sum2.weights[1], sum2.weights[2],
          sum3.weights[0], sum3.weights[1], sum3.weights[2])
    assert_almost_equal(weight_u11, sum1.weights[0], 10)
    assert_almost_equal(weight_u12, sum1.weights[1], 10)
    assert_almost_equal(weight_u13, sum1.weights[2], 10)

    assert_almost_equal(weight_u22, sum2.weights[0], 10)
    assert_almost_equal(weight_u23, sum2.weights[1], 10)
    assert_almost_equal(weight_u24, sum2.weights[2], 10)

    assert_almost_equal(weight_u33, sum3.weights[0], 10)
    assert_almost_equal(weight_u34, sum3.weights[1], 10)
    assert_almost_equal(weight_u35, sum3.weights[2], 10)

    #
    # resetting derivatives
    #
    node1.log_der = LOG_ZERO
    node2.log_der = LOG_ZERO
    node3.log_der = LOG_ZERO
    node4.log_der = LOG_ZERO
    node5.log_der = LOG_ZERO

    # setting new values as inputs
    val1 = 0.0
    node1.set_val(val1)
    val2 = 0.0
    node2.set_val(val2)
    val3 = 0.3
    node3.set_val(val3)
    val4 = 1.0
    node4.set_val(val4)
    val5 = 1.0
    node5.set_val(val5)

    # evaluating again
    sum_layer.eval()
    print('eval\'d layer:', sum_layer.node_values())

    # set the parent derivatives
    sum_der1 = 1.0
    sum1.log_der = log(sum_der1)

    sum_der2 = 1.0
    sum2.log_der = log(sum_der2)

    sum_der3 = 0.0
    sum3.log_der = LOG_ZERO

    # back prop layer wise
    sum_layer.backprop()

    # check for correctness
    try:
        log_der1 = log(sum_der1 * weight_u11)
    except:
        log_der1 = LOG_ZERO

    try:
        log_der2 = log(sum_der1 * weight_u12 +
                       sum_der2 * weight_u22)
    except:
        log_der2 = LOG_ZERO

    try:
        log_der3 = log(sum_der1 * weight_u13 +
                       sum_der2 * weight_u23 +
                       sum_der3 * weight_u33)
    except:
        log_der3 = LOG_ZERO

    try:
        log_der4 = log(sum_der2 * weight_u24 +
                       sum_der3 * weight_u34)
    except:
        log_der4 = LOG_ZERO

    try:
        log_der5 = log(sum_der3 * weight_u35)
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


def test_sum_layer_is_complete():
    # creating two scopes and two sum nodes
    scope1 = frozenset({0, 2, 3})
    scope2 = frozenset({10})
    sum_node_1 = SumNode(var_scope=scope1)
    sum_node_2 = SumNode(var_scope=scope2)

    # adding product nodes as children to the first, indicator the second
    for i in range(4):
        sum_node_1.add_child(ProductNode(var_scope=scope1), 1.0)
        sum_node_2.add_child(CategoricalIndicatorNode(var=10, var_val=i), 1.0)

    # creating sum layer
    sum_layer = SumLayer(nodes=[sum_node_1, sum_node_2])

    assert sum_layer.is_complete()

    # now with errors in scope
    scope3 = frozenset({6})
    sum_node_1 = SumNode(var_scope=scope1)
    sum_node_2 = SumNode(var_scope=scope3)

    # adding product nodes as children to the first, indicator the second
    for i in range(4):
        sum_node_1.add_child(ProductNode(var_scope=scope1), 1.0)
        sum_node_2.add_child(CategoricalIndicatorNode(var=10, var_val=i), 1.0)

    # creating sum layer
    sum_layer = SumLayer(nodes=[sum_node_1, sum_node_2])

    assert not sum_layer.is_complete()

    sum_node_2.var_scope = scope2

    assert sum_layer.is_complete()

    sum_node_2.children[3].var_scope = scope3

    assert not sum_layer.is_complete()


def test_product_layer_is_decomposable():
    # creating scopes and nodes
    scope1 = frozenset({0, 2, 3})
    scope2 = frozenset({10, 9})
    prod_node_1 = ProductNode(var_scope=scope1)
    prod_node_2 = ProductNode(var_scope=scope2)

    # creating children manually (argh=)
    for var in scope1:
        prod_node_1.add_child(SumNode(var_scope=frozenset({var})))
    for var in scope2:
        prod_node_2.add_child(CategoricalSmoothedNode(var=var,
                                                      var_values=2))

    # creating layer
    prod_layer = ProductLayer(nodes=[prod_node_1, prod_node_2])

    assert prod_layer.is_decomposable()

    # making it not decomposable anymore
    scope3 = frozenset({2})
    prod_node_1.add_child(SumNode(var_scope=scope3))

    assert not prod_layer.is_decomposable()


def test_categorical_clt_input_layer_eval():
    #
    # just a little test to see how mixed nodes layers
    # dispatch eval
    s_data = numpy.array([[1, 1, 1, 0],
                          [0, 0, 1, 0],
                          [0, 0, 1, 0],
                          [1, 1, 0, 0],
                          [1, 0, 1, 0],
                          [0, 1, 1, 0]])
    features = [0, 2, 3]
    feature_vals = [2, 2, 2]
    clt_node = CLTreeNode(data=s_data[:, features],
                          vars=features,
                          var_values=feature_vals,
                          alpha=0.0)

    #
    # creating a categorical smoothed node
    cs_node = CategoricalSmoothedNode(var=1,
                                      var_values=2,
                                      alpha=0.,
                                      data=s_data[:, [1]])

    clti_layer = CategoricalCLInputLayer(nodes=[cs_node, clt_node])

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

    s_log_prob = numpy.log(0.5)

    #
    # evaluating the layer
    for i, instance in enumerate(s_data):
        clti_layer.eval(instance)
        for node in clti_layer.nodes():
            if isinstance(node, CLTreeNode):
                assert_almost_equal(nico_cltree_sublls[i], node.log_val)
            else:
                # in the other case is 0.5
                assert_almost_equal(s_log_prob, node.log_val)
