from spn.factory import SpnFactory

from spn.linked.spn import Spn as SpnLinked

from spn.linked.layers import SumLayer as SumLayerLinked
from spn.linked.layers import ProductLayer as ProductLayerLinked
from spn.linked.layers import CategoricalIndicatorLayer
from spn.linked.layers import CategoricalSmoothedLayer

from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CategoricalIndicatorNode
from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import CLTreeNode

from spn import MARG_IND

import numpy
from numpy.testing import assert_array_almost_equal

import dataset

try:
    from time import perf_counter
except:
    from time import time as perf_counter

from algo.dataslice import DataSlice

from collections import deque

import random

# creating the input matrix for the tests
I = numpy.array([[MARG_IND, MARG_IND, MARG_IND, MARG_IND],
                 [0, 1, MARG_IND, 0],
                 [0, 0, 0, MARG_IND],
                 [1, 0, 0, 0]])

II = numpy.array([[MARG_IND, MARG_IND, MARG_IND, MARG_IND, MARG_IND],
                  [0, 1, MARG_IND, 0, 0],
                  [0, 0, 0, MARG_IND, 0],
                  [1, 0, 0, 0, 1]])


def test_linked_to_theano_indicator():
    # creating single nodes
    root = SumNode()

    prod1 = ProductNode()
    prod2 = ProductNode()
    prod3 = ProductNode()

    sum1 = SumNode()
    sum2 = SumNode()
    sum3 = SumNode()
    sum4 = SumNode()

    ind1 = CategoricalIndicatorNode(var=0, var_val=0)
    ind2 = CategoricalIndicatorNode(var=0, var_val=1)
    ind3 = CategoricalIndicatorNode(var=1, var_val=0)
    ind4 = CategoricalIndicatorNode(var=1, var_val=1)
    ind5 = CategoricalIndicatorNode(var=2, var_val=0)
    ind6 = CategoricalIndicatorNode(var=2, var_val=1)
    ind7 = CategoricalIndicatorNode(var=2, var_val=2)
    ind8 = CategoricalIndicatorNode(var=3, var_val=0)
    ind9 = CategoricalIndicatorNode(var=3, var_val=1)
    ind10 = CategoricalIndicatorNode(var=3, var_val=2)
    ind11 = CategoricalIndicatorNode(var=3, var_val=3)

    prod4 = ProductNode()
    prod5 = ProductNode()
    prod6 = ProductNode()
    prod7 = ProductNode()

    # linking nodes
    root.add_child(prod1, 0.3)
    root. add_child(prod2, 0.3)
    root.add_child(prod3, 0.4)

    prod1.add_child(sum1)
    prod1.add_child(sum2)
    prod2.add_child(ind7)
    prod2.add_child(ind8)
    prod2.add_child(ind11)
    prod3.add_child(sum3)
    prod3.add_child(sum4)

    sum1.add_child(ind1, 0.3)
    sum1.add_child(ind2, 0.3)
    sum1.add_child(prod4, 0.4)

    sum2.add_child(ind2, 0.5)
    sum2.add_child(prod4, 0.2)
    sum2.add_child(prod5, 0.3)

    sum3.add_child(prod6, 0.5)
    sum3.add_child(prod7, 0.5)
    sum4.add_child(prod6, 0.5)
    sum4.add_child(prod7, 0.5)

    prod4.add_child(ind3)
    prod4.add_child(ind4)
    prod5.add_child(ind5)
    prod5.add_child(ind6)
    prod6.add_child(ind9)
    prod6.add_child(ind10)
    prod7.add_child(ind9)
    prod7.add_child(ind10)

    # building layers from nodes
    root_layer = SumLayerLinked([root])
    prod_layer = ProductLayerLinked([prod1, prod2, prod3])
    sum_layer = SumLayerLinked([sum1, sum2, sum3, sum4])
    aprod_layer = ProductLayerLinked([prod4, prod5, prod6, prod7])
    ind_layer = CategoricalIndicatorLayer(nodes=[ind1, ind2,
                                                 ind3, ind4,
                                                 ind5, ind6,
                                                 ind7, ind8,
                                                 ind9, ind10,
                                                 ind11])

    # creating the linked spn
    spn_linked = SpnLinked(input_layer=ind_layer,
                           layers=[aprod_layer,
                                   sum_layer,
                                   prod_layer,
                                   root_layer])

    print(spn_linked)

    # converting to theano repr
    spn_theano = SpnFactory.linked_to_theano(spn_linked)
    print(spn_theano)

    # time for some inference comparison
    for instance in I:
        print('linked')
        res_l = spn_linked.eval(instance)
        print(res_l)
        print('theano')
        res_t = spn_theano.eval(instance)
        print(res_t)
        assert_array_almost_equal(res_l, res_t)


def test_linked_to_theano_categorical():
    vars = [2, 2, 3, 4]
    freqs = [{'var': 0, 'freqs': [1, 2]},
             {'var': 1, 'freqs': [2, 2]},
             {'var': 0, 'freqs': [3, 2]},
             {'var': 1, 'freqs': [0, 3]},
             {'var': 2, 'freqs': [1, 0, 2]},
             {'var': 3, 'freqs': [1, 2, 1, 2]},
             {'var': 3, 'freqs': [3, 4, 0, 1]}]

    # create input layer first
    input_layer = CategoricalSmoothedLayer(vars=vars,
                                           node_dicts=freqs)
    # get nodes
    ind_nodes = [node for node in input_layer.nodes()]

    root_node = ProductNode()

    sum1 = SumNode()
    sum2 = SumNode()

    prod1 = ProductNode()
    prod2 = ProductNode()

    sum3 = SumNode()
    sum4 = SumNode()

    # linking
    root_node.add_child(sum1)
    root_node.add_child(sum2)
    root_node.add_child(ind_nodes[0])
    root_node.add_child(ind_nodes[1])

    sum1.add_child(ind_nodes[2], 0.4)
    sum1.add_child(ind_nodes[3], 0.6)
    sum2.add_child(ind_nodes[3], 0.2)
    sum2.add_child(prod1, 0.5)
    sum2.add_child(prod2, 0.3)

    prod1.add_child(ind_nodes[4])
    prod1.add_child(sum3)
    prod1.add_child(sum4)
    prod2.add_child(sum3)
    prod2.add_child(sum4)

    sum3.add_child(ind_nodes[5], 0.5)
    sum3.add_child(ind_nodes[6], 0.5)
    sum4.add_child(ind_nodes[5], 0.4)
    sum4.add_child(ind_nodes[6], 0.6)

    # creating layers
    root_layer = ProductLayerLinked([root_node])
    sum_layer = SumLayerLinked([sum1, sum2])
    prod_layer = ProductLayerLinked([prod1, prod2])
    sum_layer2 = SumLayerLinked([sum3, sum4])

    # create the linked spn
    spn_linked = SpnLinked(input_layer=input_layer,
                           layers=[sum_layer2, prod_layer,
                                   sum_layer, root_layer])

    print(spn_linked)

    # converting to theano repr
    spn_theano = SpnFactory.linked_to_theano(spn_linked)
    print(spn_theano)

    # time for some inference comparison
    for instance in I:
        print('linked')
        res_l = spn_linked.eval(instance)
        print(res_l)
        print('theano')
        res_t = spn_theano.eval(instance)
        print(res_t)
        assert_array_almost_equal(res_l, res_t)

vars = [2, 2, 3, 4]
freqs = [{'var': 0, 'freqs': [1, 2]},
         {'var': 0, 'freqs': [5, 5]},
         {'var': 1, 'freqs': [6, 10]},
         {'var': 2, 'freqs': [0, 0, 3]},
         {'var': 2, 'freqs': [0, 10, 10]},
         {'var': 2, 'freqs': [0, 4, 3]},
         {'var': 3, 'freqs': [1, 3, 1, 3]},
         {'var': 3, 'freqs': [4, 4, 0, 5]}]

naive_freqs = [{'var': 0, 'freqs': [1, 2]},
               {'var': 1, 'freqs': [6, 10]},
               {'var': 2, 'freqs': [0, 0, 3]},
               {'var': 3, 'freqs': [4, 4, 0, 5]}]


def test_linked_kernel_density_estimation():
    num_instances = 5
    spn = SpnFactory.linked_kernel_density_estimation(num_instances,
                                                      vars)
    print('Kernel density estimation')
    print(spn)
    print(spn.stats())


def atest_linked_nltcs_kernel_spn():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('ninst', n_instances, 'feats', features)
    print('Build kernel density estimation')
    spn = SpnFactory.linked_kernel_density_estimation(n_instances,
                                                      features)
    print(spn.stats())
    print('Evaluating on test')
    # evaluating one at a time since we are using a sparse representation
    lls = []
    for i in range(test.shape[0]):
        print('instance', i)
        lls.append(spn.eval(test[i, :]))
    print('Mean lls')
    # avg_ll = sum(lls) / float(len(lls))
    avg_ll = numpy.mean(lls)
    print(avg_ll)


def test_linked_nltcs_kernel_spn_perf():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('ninst', n_instances, 'feats', features)
    print('Build kernel density estimation')
    spn = SpnFactory.linked_kernel_density_estimation(n_instances,
                                                      features)
    print(spn.stats())
    print('Evaluating on test')
    # evaluating one at a time since we are using a sparse representation
    lls = []
    eval_start_t = perf_counter()
    for i in range(test.shape[0]):
        print('instance', i)
        lls.append(spn.eval(test[i, :]))
    print('Mean lls')
    # avg_ll = sum(lls) / float(len(lls))
    avg_ll = numpy.mean(lls)
    eval_end_t = perf_counter()
    print('AVG LL {0} in {1} secs'.format(avg_ll, eval_end_t - eval_start_t))


def test_linked_naive_factorization():
    spn = SpnFactory.linked_naive_factorization(vars)
    print('Naive factorization (indicators)')
    print(spn)

    spn = SpnFactory.linked_naive_factorization(vars,
                                                naive_freqs)
    print('Naive factorization (smoothing)')
    print(spn)


def aatest_linked_nltcs_naive_spn():
    # load the dataset
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('tmovie')

    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('Building naive factorization')
    spn = SpnFactory.linked_naive_factorization(features,
                                                freqs,
                                                alpha=0)
    print('Evaluating on test')
    lls = []
    for i in range(test.shape[0]):
        print('instance', i)
        lls.append(spn.eval(test[i, :]))
    print('Mean lls')
    avg_ll = numpy.mean(lls)
    print(avg_ll)


def atest_theano_kernel_density_estimation():
    num_instances = 5
    spn = SpnFactory.theano_kernel_density_estimation(num_instances,
                                                      vars)
    print('Kernel density estimation (indicators)')
    print(spn)
    spn = SpnFactory.theano_kernel_density_estimation(num_instances,
                                                      vars,
                                                      sparse=True)
    print('Sparse kernel density estimation')
    print(spn)

    spn = SpnFactory.theano_kernel_density_estimation(num_instances,
                                                      vars,
                                                      freqs)
    print('Kernel density estimation (smoothing)')
    print(spn)


def test_theano_kernel_density_estimation_categorical():
    num_instances = 5
    spn = SpnFactory.theano_kernel_density_estimation(num_instances,
                                                      vars,
                                                      node_dict=freqs,
                                                      alpha=0.1)
    print('Sparse kernel density estimation' +
          'with smoothed categorical input layer')
    print(spn)
    print(spn.stats())


def test_theano_naive_factorization():
    spn = SpnFactory.theano_naive_factorization(vars)
    print('Naive factorization (indicators)')
    print(spn)

    spn = SpnFactory.theano_naive_factorization(vars,
                                                naive_freqs)
    print('Naive factorization (smoothing)')
    print(spn)


def atest_theano_nltcs_naive_spn():
    # load the dataset
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('tmovie')
    n_test_instances = test.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('Building naive factorization')
    spn = SpnFactory.theano_naive_factorization(features,
                                                freqs,
                                                alpha=0,
                                                batch_size=n_test_instances)
    print('Evaluating on test')
    ll = spn.eval(test.T)
    avg_ll = ll.mean()
    print(avg_ll)
    # print('Building naive factorization')
    # spn = SpnFactory.naive_factorization(features,
    #                                      alpha=0,
    #                                      batch_size=n_test_instances)
    # print('Evaluating on test')
    # ll = spn.eval(test.T)
    # avg_ll = ll.mean()
    # print(avg_ll)


def atest_theano_nltcs_kernel_spn():
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')
    n_instances = train.shape[0]
    n_test_instances = test.shape[0]
    # estimating the frequencies for the features
    print('Estimating features')
    freqs, features = dataset.data_2_freqs(train)

    print('Build kernel density estimation')
    spn = SpnFactory.theano_kernel_density_estimation(
        n_instances,
        features,
        batch_size=n_test_instances,
        sparse=True)
    print('Evaluating on test')
    # evaluating one at a time since we are using a sparse representation
    lls = []
    for i in range(test.shape[0]):
        print('instance', i)
        lls.append(spn.eval(test[i, :]))
    print('Mean lls')
    # avg_ll = sum(lls) / float(len(lls))
    avg_ll = numpy.mean(lls)
    print(avg_ll)


def test_linked_random_spn_top_down():
    # number small parameters
    n_levels = 10
    vars = [2, 3, 2, 2, 4]
    n_max_children = 2
    n_scope_children = 3
    max_scope_split = 2
    merge_prob = 0.5

    # building it
    print('creating random spn')
    rand_gen = random.Random(789)

    #
    # doing this for more than once
    n_times = 10
    for _i in range(n_times):
        spn = SpnFactory.linked_random_spn_top_down(vars,
                                                    n_levels,
                                                    n_max_children,
                                                    n_scope_children,
                                                    max_scope_split,
                                                    merge_prob,
                                                    rand_gen=rand_gen)

        # printing for comparison
        print(spn)
        print(spn.stats())
        assert spn.is_valid()

        # translating to theano representation
        theano_spn = SpnFactory.linked_to_theano(spn)

        print(theano_spn)
        print(theano_spn.stats())

        #
        # looking for the same computations
        # time for some inference comparison
        for instance in II:
            print('linked')
            res_l = spn.eval(instance)
            print(res_l)
            print('theano')
            res_t = theano_spn.eval(instance)
            print(res_t)
            assert_array_almost_equal(res_l, res_t)


def test_layered_linked_spn():
    # creating single nodes
    # this code is replicated TODO: make a function
    root = SumNode()

    prod1 = ProductNode()
    prod2 = ProductNode()
    prod3 = ProductNode()

    sum1 = SumNode()
    sum2 = SumNode()
    sum3 = SumNode()
    sum4 = SumNode()

    ind1 = CategoricalIndicatorNode(var=0, var_val=0)
    ind2 = CategoricalIndicatorNode(var=0, var_val=1)
    ind3 = CategoricalIndicatorNode(var=1, var_val=0)
    ind4 = CategoricalIndicatorNode(var=1, var_val=1)
    ind5 = CategoricalIndicatorNode(var=2, var_val=0)
    ind6 = CategoricalIndicatorNode(var=2, var_val=1)
    ind7 = CategoricalIndicatorNode(var=2, var_val=2)
    ind8 = CategoricalIndicatorNode(var=3, var_val=0)
    ind9 = CategoricalIndicatorNode(var=3, var_val=1)
    ind10 = CategoricalIndicatorNode(var=3, var_val=2)
    ind11 = CategoricalIndicatorNode(var=3, var_val=3)

    prod4 = ProductNode()
    prod5 = ProductNode()
    prod6 = ProductNode()
    prod7 = ProductNode()

    # linking nodes
    root.add_child(prod1, 0.3)
    root. add_child(prod2, 0.3)
    root.add_child(prod3, 0.4)

    prod1.add_child(sum1)
    prod1.add_child(sum2)
    prod2.add_child(ind7)
    prod2.add_child(ind8)
    prod2.add_child(ind11)
    prod3.add_child(sum3)
    prod3.add_child(sum4)

    sum1.add_child(ind1, 0.3)
    sum1.add_child(ind2, 0.3)
    sum1.add_child(prod4, 0.4)

    sum2.add_child(ind2, 0.5)
    sum2.add_child(prod4, 0.2)
    sum2.add_child(prod5, 0.3)

    sum3.add_child(prod6, 0.5)
    sum3.add_child(prod7, 0.5)
    sum4.add_child(prod6, 0.5)
    sum4.add_child(prod7, 0.5)

    prod4.add_child(ind3)
    prod4.add_child(ind4)
    prod5.add_child(ind5)
    prod5.add_child(ind6)
    prod6.add_child(ind9)
    prod6.add_child(ind10)
    prod7.add_child(ind9)
    prod7.add_child(ind10)

    spn = SpnFactory.layered_linked_spn(root)

    print(spn)
    print(spn.stats())


def test_pruned_spn_from_slices():
    #
    # creating all the data slices
    # the slicing is a fake stub
    rows = 5
    cols = 5
    var = 1
    values = 2

    node_assoc = {}
    building_stack = deque()

    slice_1 = DataSlice.whole_slice(rows, cols)
    slice_1.type = SumNode
    node_1 = SumNode()
    node_1.id = slice_1.id
    node_assoc[node_1.id] = node_1
    building_stack.append(slice_1)

    slice_2 = DataSlice.whole_slice(rows, cols)
    slice_2.type = ProductNode
    node_2 = ProductNode()
    node_2.id = slice_2.id
    node_assoc[node_2.id] = node_2
    building_stack.append(slice_2)

    slice_3 = DataSlice.whole_slice(rows, cols)
    slice_3.type = SumNode
    node_3 = SumNode()
    node_3.id = slice_3.id
    node_assoc[node_3.id] = node_3
    building_stack.append(slice_3)

    # adding first level
    slice_1.add_child(slice_2, 0.8)
    slice_1.add_child(slice_3, 0.2)

    slice_4 = DataSlice.whole_slice(rows, cols)
    slice_4.type = ProductNode
    node_4 = ProductNode()
    node_4.id = slice_4.id
    node_assoc[node_4.id] = node_4
    building_stack.append(slice_4)

    leaf_5 = CategoricalSmoothedNode(var,
                                     values)
    slice_5 = DataSlice.whole_slice(rows, cols)
    leaf_5.id = slice_5.id
    node_assoc[leaf_5.id] = leaf_5
    # not adding the slice to the stack

    slice_2.add_child(slice_4)
    slice_2.add_child(slice_5)

    slice_6 = DataSlice.whole_slice(rows, cols)
    slice_6.type = SumNode
    node_6 = SumNode()
    node_6.id = slice_6.id
    node_assoc[node_6.id] = node_6
    building_stack.append(slice_6)

    slice_7 = DataSlice.whole_slice(rows, cols)
    slice_7.type = SumNode
    node_7 = SumNode()
    node_7.id = slice_7.id
    node_assoc[node_7.id] = node_7
    building_stack.append(slice_7)

    slice_3.add_child(slice_6, 0.4)
    slice_3.add_child(slice_7, 0.6)

    slice_8 = DataSlice.whole_slice(rows, cols)
    slice_8.type = ProductNode
    node_8 = ProductNode()
    node_8.id = slice_8.id
    node_assoc[node_8.id] = node_8
    building_stack.append(slice_8)

    leaf_15 = CategoricalSmoothedNode(var,
                                      values)
    slice_15 = DataSlice.whole_slice(rows, cols)
    leaf_15.id = slice_15.id
    node_assoc[leaf_15.id] = leaf_15

    slice_4.add_child(slice_8)
    slice_4.add_child(slice_15)

    leaf_13 = CategoricalSmoothedNode(var,
                                      values)
    slice_13 = DataSlice.whole_slice(rows, cols)
    leaf_13.id = slice_13.id
    node_assoc[leaf_13.id] = leaf_13

    leaf_14 = CategoricalSmoothedNode(var,
                                      values)
    slice_14 = DataSlice.whole_slice(rows, cols)
    leaf_14.id = slice_14.id
    node_assoc[leaf_14.id] = leaf_14

    slice_8.add_child(slice_13)
    slice_8.add_child(slice_14)

    slice_9 = DataSlice.whole_slice(rows, cols)
    slice_9.type = ProductNode
    node_9 = ProductNode()
    node_9.id = slice_9.id
    node_assoc[node_9.id] = node_9
    building_stack.append(slice_9)

    leaf_16 = CategoricalSmoothedNode(var,
                                      values)
    slice_16 = DataSlice.whole_slice(rows, cols)
    leaf_16.id = slice_16.id
    node_assoc[leaf_16.id] = leaf_16

    leaf_17 = CategoricalSmoothedNode(var,
                                      values)
    slice_17 = DataSlice.whole_slice(rows, cols)
    leaf_17.id = slice_17.id
    node_assoc[leaf_17.id] = leaf_17

    slice_9.add_child(slice_16)
    slice_9.add_child(slice_17)

    slice_10 = DataSlice.whole_slice(rows, cols)
    slice_10.type = ProductNode
    node_10 = ProductNode()
    node_10.id = slice_10.id
    node_assoc[node_10.id] = node_10
    building_stack.append(slice_10)

    leaf_18 = CategoricalSmoothedNode(var,
                                      values)
    slice_18 = DataSlice.whole_slice(rows, cols)
    leaf_18.id = slice_18.id
    node_assoc[leaf_18.id] = leaf_18

    leaf_19 = CategoricalSmoothedNode(var,
                                      values)
    slice_19 = DataSlice.whole_slice(rows, cols)
    leaf_19.id = slice_19.id
    node_assoc[leaf_19.id] = leaf_19

    slice_10.add_child(slice_18)
    slice_10.add_child(slice_19)

    slice_6.add_child(slice_9, 0.1)
    slice_6.add_child(slice_10, 0.9)

    slice_11 = DataSlice.whole_slice(rows, cols)
    slice_11.type = ProductNode
    node_11 = ProductNode()
    node_11.id = slice_11.id
    node_assoc[node_11.id] = node_11
    building_stack.append(slice_11)

    leaf_20 = CategoricalSmoothedNode(var,
                                      values)
    slice_20 = DataSlice.whole_slice(rows, cols)
    leaf_20.id = slice_20.id
    node_assoc[leaf_20.id] = leaf_20

    leaf_21 = CategoricalSmoothedNode(var,
                                      values)
    slice_21 = DataSlice.whole_slice(rows, cols)
    leaf_21.id = slice_21.id
    node_assoc[leaf_21.id] = leaf_21

    slice_11.add_child(slice_20)
    slice_11.add_child(slice_21)

    slice_12 = DataSlice.whole_slice(rows, cols)
    slice_12.type = ProductNode
    node_12 = ProductNode()
    node_12.id = slice_12.id
    node_assoc[node_12.id] = node_12
    building_stack.append(slice_12)

    leaf_22 = CategoricalSmoothedNode(var,
                                      values)
    slice_22 = DataSlice.whole_slice(rows, cols)
    leaf_22.id = slice_22.id
    node_assoc[leaf_22.id] = leaf_22

    leaf_23 = CategoricalSmoothedNode(var,
                                      values)
    slice_23 = DataSlice.whole_slice(rows, cols)
    leaf_23.id = slice_23.id
    node_assoc[leaf_23.id] = leaf_23

    slice_12.add_child(slice_22)
    slice_12.add_child(slice_23)

    slice_7.add_child(slice_11, 0.2)
    slice_7.add_child(slice_12, 0.7)

    root_node = SpnFactory.pruned_spn_from_slices(node_assoc,
                                                  building_stack)

    print('ROOT nODE', root_node)

    spn = SpnFactory.layered_linked_spn(root_node)

    print('SPN', spn)

    assert spn.n_layers() == 3

    for i, layer in enumerate(spn.top_down_layers()):
        if i == 0:
            assert layer.n_nodes() == 1
        elif i == 1:
            assert layer.n_nodes() == 5
        elif i == 2:
            assert layer.n_nodes() == 12


def test_layered_pruned_linked_spn():
        #
    # creating all the data slices
    # the slicing is a fake stub
    rows = 5
    cols = 5
    var = 1
    values = 2

    node_1 = SumNode()
    node_1.id = 1

    node_2 = ProductNode()
    node_2.id = 2

    node_3 = SumNode()
    node_3.id = 3

    # adding first level
    weight_12 = 0.4
    weight_13 = 0.6
    node_1.add_child(node_2, weight_12)
    node_1.add_child(node_3, weight_13)

    node_4 = ProductNode()
    node_4.id = 4

    leaf_5 = CategoricalSmoothedNode(var,
                                     values)
    leaf_5.id = 5

    # not adding the slice to the stack

    node_2.add_child(node_4)
    node_2.add_child(leaf_5)

    node_6 = SumNode()
    node_6.id = 6

    node_7 = SumNode()
    node_7.id = 7

    weight_36 = 0.1
    weight_37 = 0.9
    node_3.add_child(node_6, weight_36)
    node_3.add_child(node_7, weight_37)

    node_8 = ProductNode()
    node_8.id = 8

    leaf_15 = CategoricalSmoothedNode(var,
                                      values)
    leaf_15.id = 15

    node_4.add_child(node_8)
    node_4.add_child(leaf_15)

    leaf_13 = CategoricalSmoothedNode(var,
                                      values)
    leaf_13.id = 13

    leaf_14 = CategoricalSmoothedNode(var,
                                      values)
    leaf_14.id = 14

    node_8.add_child(leaf_13)
    node_8.add_child(leaf_14)

    node_9 = ProductNode()
    node_9.id = 9

    leaf_16 = CategoricalSmoothedNode(var,
                                      values)
    leaf_16.id = 16

    leaf_17 = CategoricalSmoothedNode(var,
                                      values)
    leaf_17.id = 17

    node_9.add_child(leaf_16)
    node_9.add_child(leaf_17)

    node_10 = ProductNode()
    node_10.id = 10

    leaf_18 = CategoricalSmoothedNode(var,
                                      values)
    leaf_18.id = 18

    leaf_19 = CategoricalSmoothedNode(var,
                                      values)
    leaf_19.id = 19

    node_10.add_child(leaf_18)
    node_10.add_child(leaf_19)

    weight_69 = 0.3
    weight_610 = 0.7
    node_6.add_child(node_9, weight_69)
    node_6.add_child(node_10, weight_610)

    node_11 = ProductNode()
    node_11.id = 11

    leaf_20 = CategoricalSmoothedNode(var,
                                      values)
    leaf_20.id = 20

    leaf_21 = CategoricalSmoothedNode(var,
                                      values)
    leaf_21.id = 21

    node_11.add_child(leaf_20)
    node_11.add_child(leaf_21)

    node_12 = ProductNode()
    node_12.id = 12

    leaf_22 = CategoricalSmoothedNode(var,
                                      values)
    leaf_22.id = 22

    leaf_23 = CategoricalSmoothedNode(var,
                                      values)
    leaf_23.id = 23

    node_12.add_child(leaf_22)
    node_12.add_child(leaf_23)

    weight_711 = 0.5
    weight_712 = 0.5
    node_7.add_child(node_11, weight_711)
    node_7.add_child(node_12, weight_712)

    root_node = SpnFactory.layered_pruned_linked_spn(node_1)

    print('ROOT nODE', root_node)

    spn = SpnFactory.layered_linked_spn(root_node)

    print('SPN', spn)

    assert spn.n_layers() == 3

    for i, layer in enumerate(spn.top_down_layers()):
        if i == 0:
            assert layer.n_nodes() == 1
        elif i == 1:
            assert layer.n_nodes() == 5
        elif i == 2:
            assert layer.n_nodes() == 12


def test_layered_pruned_linked_spn_cltree():
    #
    # creating all the data slices
    # the slicing is a fake stub
    rows = 5
    cols = 5
    var = 1
    values = 2

    vars = [2, 3]
    var_values = [2, 2]
    s_data = numpy.array([[0, 1], [1, 1], [1, 0], [0, 0]])

    node_1 = SumNode()
    node_1.id = 1

    node_2 = ProductNode()
    node_2.id = 2

    node_3 = SumNode()
    node_3.id = 3

    # adding first level
    weight_12 = 0.4
    weight_13 = 0.6
    node_1.add_child(node_2, weight_12)
    node_1.add_child(node_3, weight_13)

    node_4 = ProductNode()
    node_4.id = 4

    leaf_5 = CategoricalSmoothedNode(var,
                                     values)
    leaf_5.id = 5

    # not adding the slice to the stack

    node_2.add_child(node_4)
    node_2.add_child(leaf_5)

    node_6 = SumNode()
    node_6.id = 6

    node_7 = SumNode()
    node_7.id = 7

    weight_36 = 0.1
    weight_37 = 0.9
    node_3.add_child(node_6, weight_36)
    node_3.add_child(node_7, weight_37)

    node_8 = ProductNode()
    node_8.id = 8

    #
    # this is a cltree
    leaf_15 = CLTreeNode(vars=vars,
                         var_values=var_values,
                         data=s_data)
    leaf_15.id = 15

    node_4.add_child(node_8)
    node_4.add_child(leaf_15)

    leaf_13 = CategoricalSmoothedNode(var,
                                      values)
    leaf_13.id = 13

    leaf_14 = CLTreeNode(vars=vars,
                         var_values=var_values,
                         data=s_data)
    leaf_14.id = 14

    node_8.add_child(leaf_13)
    node_8.add_child(leaf_14)

    leaf_9 = CLTreeNode(vars=vars,
                        var_values=var_values,
                        data=s_data)
    leaf_9.id = 9

    node_10 = ProductNode()
    node_10.id = 10

    leaf_18 = CategoricalSmoothedNode(var,
                                      values)
    leaf_18.id = 18

    leaf_19 = CategoricalSmoothedNode(var,
                                      values)
    leaf_19.id = 19

    node_10.add_child(leaf_18)
    node_10.add_child(leaf_19)

    weight_69 = 0.3
    weight_610 = 0.7
    node_6.add_child(leaf_9, weight_69)
    node_6.add_child(node_10, weight_610)

    node_11 = ProductNode()
    node_11.id = 11

    leaf_20 = CategoricalSmoothedNode(var,
                                      values)
    leaf_20.id = 20

    leaf_21 = CategoricalSmoothedNode(var,
                                      values)
    leaf_21.id = 21

    node_11.add_child(leaf_20)
    node_11.add_child(leaf_21)

    node_12 = ProductNode()
    node_12.id = 12

    leaf_22 = CLTreeNode(vars=vars,
                         var_values=var_values,
                         data=s_data)
    leaf_22.id = 22

    leaf_23 = CategoricalSmoothedNode(var,
                                      values)
    leaf_23.id = 23

    node_12.add_child(leaf_22)
    node_12.add_child(leaf_23)

    weight_711 = 0.5
    weight_712 = 0.5
    node_7.add_child(node_11, weight_711)
    node_7.add_child(node_12, weight_712)

    print('Added nodes')

    root_node = SpnFactory.layered_pruned_linked_spn(node_1)

    print('ROOT nODE', root_node)

    spn = SpnFactory.layered_linked_spn(root_node)

    print('SPN', spn)

    assert spn.n_layers() == 3

    for i, layer in enumerate(spn.top_down_layers()):
        if i == 0:
            assert layer.n_nodes() == 1
        elif i == 1:
            assert layer.n_nodes() == 4
        elif i == 2:
            assert layer.n_nodes() == 10
