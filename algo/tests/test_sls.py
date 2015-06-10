from algo import sls

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import math

import random

import numpy
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from spn.linked.nodes import ProductNode
from spn.linked.nodes import SumNode

import dataset

import logging

synth_data = numpy.array([[1, 0, 1, 1],
                          [1, 0, 0, 0],
                          [0, 0, 1, 2],
                          [3, 0, 1, 1]])
synth_feature_vals = numpy.array([4, 2, 2, 3])


#
# creating a large synthetic matrix, testing for performance
rand_gen = numpy.random.RandomState(1337)
n_instances = 2000
n_features = 2000
large_matrix = rand_gen.binomial(1, 0.5, (n_instances, n_features))
n_samples = 1000
#
# random indexes too
rand_instance_ids = rand_gen.choice(n_instances, n_samples, replace=False)
rand_feature_ids = rand_gen.choice(n_features, n_samples, replace=False)


def test_estimate_counts_perf():

    binary_feature_vals = [2 for i in range(n_features)]
    #
    # testing
    numpy_start_t = perf_counter()
    print('KM', large_matrix)

    e_counts_numpy = sls.estimate_counts_numpy(large_matrix,
                                               rand_instance_ids,
                                               rand_feature_ids)
    numpy_end_t = perf_counter()
    print('numpy done in {0} secs'.format(numpy_end_t - numpy_start_t))

    binary_feature_vals = numpy.array(binary_feature_vals)
    print('KM', large_matrix[298, 8])
    #
    #
    curr_feature_vals = binary_feature_vals[rand_feature_ids]
    #
    # creating a data structure to hold the slices
    # (with numba I cannot use list comprehension?)
    # e_counts_numba = [[0 for val in range(feature_val)]
    #                   for feature_val in
    #                   curr_feature_vals]
    e_counts_numba = numpy.zeros((n_samples, 2), dtype='int32')
    #
    # first time is for compiling
    numba_start_t = perf_counter()

    sls.estimate_counts_numba(large_matrix,
                              rand_instance_ids,
                              rand_feature_ids,
                              binary_feature_vals,
                              e_counts_numba)
    numba_end_t = perf_counter()
    print('numba done in {0} secs'.format(numba_end_t - numba_start_t))
    # resetting the counter
    e_counts_numba = numpy.zeros((n_samples, 2), dtype='int32')
    numba_start_t = perf_counter()

    sls.estimate_counts_numba(large_matrix,
                              rand_instance_ids,
                              rand_feature_ids,
                              binary_feature_vals,
                              e_counts_numba)
    numba_end_t = perf_counter()
    print('numba done in {0} secs'.format(numba_end_t - numba_start_t))
    #
    # checking for correctness
    for e_numpy, e_numba in zip(e_counts_numpy,
                                e_counts_numba):
        assert e_numpy.tolist() == e_numba


def test_k_partitions_correctness():
    n_instances = 4
    k = 2
    partitions = numpy.zeros(n_instances, dtype='int32')
    m_partitions = numpy.zeros(n_instances, dtype='int32')

    for part in sls.k_partitions(partitions,
                                 m_partitions,
                                 n_instances,
                                 k):
        print(part)


def test_k_partitions_perf():
    n_instances = 30
    k = 2
    partitions = numpy.zeros(n_instances, dtype='int32')
    m_partitions = numpy.zeros(n_instances, dtype='int32')

    part_start_t = perf_counter()
    # for part in
    sls.k_partitions(partitions,
                     m_partitions,
                     n_instances,
                     k)  # :
    # print(part)
    # pass
    part_end_t = perf_counter()
    print('elapsed {0}'.format(part_end_t - part_start_t))

    partitions = numpy.zeros(n_instances, dtype='int32')
    m_partitions = numpy.zeros(n_instances, dtype='int32')

    part_start_t = perf_counter()
    # for part in
    sls.k_partitions(partitions,
                     m_partitions,
                     n_instances,
                     k)  # :
    # print(part)
    # pass
    part_end_t = perf_counter()
    print('elapsed {0}'.format(part_end_t - part_start_t))


data = numpy.array([[1, 0, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [0, 0, 1, 1, 1],
                    [1, 0, 1, 1, 0],
                    [0, 0, 1, 0, 0]])

#
# these tests are broken
#


def test_evaluate_dataset_ll_numba_corretness():

    n_features = 5
    feature_ids = numpy.array([1, 3, 4])
    feature_values = 2
    ll_frequencies = numpy.zeros((n_features, feature_values))
    for i in range(ll_frequencies.shape[0]):
        rand_prop = rand_gen.rand()
        ll_frequencies[i, 0] = numpy.log(rand_prop)
        ll_frequencies[i, 1] = numpy.log(1 - rand_prop)

    print('estimated fake ll', ll_frequencies)
    ll_1 = sls.evaluate_dataset_ll_numba(ll_frequencies,
                                         feature_ids,
                                         data)
    print('LL', ll_1)

    ll_2 = sls.evaluate_dataset_naive_ll(ll_frequencies,
                                         feature_ids,
                                         data)
    print('LL', ll_2)

    assert_almost_equal(ll_1, ll_2)


def test_evaluate_dataset_ll_numba_perf():
    #
    # synth log likelihood frequencies
    feature_vals = 2
    ll_frequencies = numpy.zeros((n_samples, feature_vals))
    for i in range(ll_frequencies.shape[0]):
        rand_prop = rand_gen.rand()
        ll_frequencies[i, 0] = numpy.log(rand_prop)
        ll_frequencies[i, 1] = numpy.log(1 - rand_prop)

    eval_start_t = perf_counter()
    ll = sls.evaluate_dataset_ll_numba(ll_frequencies, rand_feature_ids,
                                       large_matrix)
    eval_end_t = perf_counter()
    print('LL {0} evaluated in {1} secs'.format(ll, eval_end_t - eval_start_t))

    eval_start_t = perf_counter()
    ll = sls.evaluate_dataset_ll_numba(ll_frequencies, rand_feature_ids,
                                       large_matrix)
    eval_end_t = perf_counter()
    print('LL {0} evaluated in {1} secs'.format(ll, eval_end_t - eval_start_t))

    eval_start_t = perf_counter()
    ll = sls.evaluate_dataset_naive_ll(ll_frequencies, rand_feature_ids,
                                       large_matrix)
    eval_end_t = perf_counter()
    print('LL {0} evaluated in {1} secs'.format(ll, eval_end_t - eval_start_t))

    eval_start_t = perf_counter()
    ll = sls.evaluate_dataset_naive_ll(ll_frequencies, rand_feature_ids,
                                       large_matrix)
    eval_end_t = perf_counter()
    print('LL {0} evaluated in {1} secs'.format(ll, eval_end_t - eval_start_t))


def test_estimate_counts_numba():
    instance_ids = numpy.array([0, 1, 3])
    feature_ids = numpy.array([0, 1, 2])
    feature_vals = numpy.array([2, 2, 2, 2, 2])

    e_counts = numpy.array([[2, 3],
                            [1, 1],
                            [0, 0],
                            [3, 0],
                            [9, 11]])

    e_counts = sls.estimate_counts_numba(data,
                                         instance_ids,
                                         feature_ids,
                                         feature_vals,
                                         e_counts)

    print('Estimated counts\n:', e_counts)

    #
    # computing the right one by hand
    exp_e_counts = numpy.array([[0, 3],
                                [3, 0],
                                [1, 2],
                                [3, 0],
                                [9, 11]])
    assert_array_equal(e_counts, exp_e_counts)


def test_smooth_ll_parameters():

    instance_ids = numpy.array([0, 2, 3, 4, 1])
    feature_ids = numpy.array([0])
    feature_vals = numpy.array([2, 2, 2, 2, 2])

    #
    # counting first
    e_counts = numpy.array([[2, 3],
                            [1, 1],
                            [0, 0],
                            [3, 0],
                            [9, 11]])

    e_counts = sls.estimate_counts_numba(data,
                                         instance_ids,
                                         feature_ids,
                                         feature_vals,
                                         e_counts)

    print('Estimated counts\n:', e_counts)

    #
    # now estimating the parameters
    ll_freqs = numpy.array([[-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1]])
    alpha = 0.0

    ll_freqs = sls.smooth_ll_parameters(e_counts,
                                        ll_freqs,
                                        instance_ids,
                                        feature_ids,
                                        feature_vals,
                                        alpha)

    print(ll_freqs)

    exp_ll_freqs = numpy.array([[-1.00109861e+03,   0.0000],
                                [0.00000000, -1.00109861e+03],
                                [-1.09861229e+00, -4.05465108e-01],
                                [-9.0000e+00, -1.0000e-01],
                                [-9.0000e+00, -1.0000e-01]])

    # assert_array_almost_equal(exp_ll_freqs, ll_freqs, decimal=5)
    # now with another alpha
    alpha = 0.1
    ll_freqs = sls.smooth_ll_parameters(e_counts,
                                        ll_freqs,
                                        instance_ids,
                                        feature_ids,
                                        feature_vals,
                                        alpha)

    print(ll_freqs)

    exp_ll_freqs = numpy.array([[-3.46573590,   -0.0317486],
                                [-0.0317486, -3.46573590],
                                [-1.06784063, -0.4212134],
                                [-9.0000e+00, -1.0000e-01],
                                [-9.0000e+00, -1.0000e-01]])
    # assert_array_almost_equal(exp_ll_freqs, ll_freqs)


def test_estimate_data_slice_ll():
    #
    # create data slice
    instance_ids = numpy.array([0, 2, 3, 4, 1])
    feature_ids = numpy.array([0])
    feature_vals = numpy.array([2, 2, 2, 2, 2])
    data_slice = sls.DataSlice(instance_ids,
                               feature_ids)

    e_counts = numpy.array([[2, 3],
                            [1, 1],
                            [0, 0],
                            [3, 0],
                            [9, 11]])

    ll_freqs = numpy.array([[-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1]])

    alpha = 0.0

    lls = sls.estimate_data_slice_ll(data_slice,
                                     e_counts,
                                     ll_freqs,
                                     data,
                                     feature_vals,
                                     alpha,
                                     data)
    ll = numpy.sum(lls)
    print('Estimated LL:', lls)

    exp_ll = -2004.9176926
    print('Manual LL:', exp_ll)

    # assert_almost_equal(exp_ll, ll, decimal=5)

    alpha = 0.1

    lls = sls.estimate_data_slice_ll(data_slice,
                                     e_counts,
                                     ll_freqs,
                                     data,
                                     feature_vals,
                                     alpha,
                                     data)
    ll = numpy.sum(lls)
    print('Estimated LL:', lls)

    exp_ll = - 9.93815588
    print('Manual LL:', exp_ll)

    # assert_almost_equal(exp_ll, ll)


def test_estimate_row_split_ll():

    n_instances = data.shape[0]

    #
    # having two data slices
    instance_ids_1 = numpy.array([0, 1, 3])
    instance_ids_2 = numpy.array([2, 4])

    feature_ids = numpy.array([0, 1, 2])
    feature_vals = numpy.array([2, 2, 2, 2, 2])

    data_slice_1 = sls.DataSlice(instance_ids_1,
                                 feature_ids)
    data_slice_2 = sls.DataSlice(instance_ids_2,
                                 feature_ids)
    data_partition = [data_slice_1, data_slice_2]
    alpha = 0.1

    e_counts = numpy.array([[2, 3],
                            [1, 1],
                            [0, 0],
                            [3, 0],
                            [9, 11]])
    ll_freqs = numpy.array([[-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1]])

    ll_data_slice_1s = sls.estimate_data_slice_ll(data_slice_1,
                                                  e_counts,
                                                  ll_freqs,
                                                  data,
                                                  feature_vals,
                                                  alpha,
                                                  data)

    ll_data_slice_2s = sls.estimate_data_slice_ll(data_slice_2,
                                                  e_counts,
                                                  ll_freqs,
                                                  data,
                                                  feature_vals,
                                                  alpha,
                                                  data)

    ll_data_slice_2 = numpy.sum(ll_data_slice_2s) / n_instances
    ll_data_slice_1 = numpy.sum(ll_data_slice_1s) / n_instances
    print('LL on the single slices', ll_data_slice_1, ll_data_slice_2)

    lls = sls.estimate_row_split_ll(data_partition,
                                    e_counts,
                                    ll_freqs,
                                    data,
                                    feature_vals,
                                    alpha,
                                    data)
    ll = numpy.sum(lls)
    print('Estimated LL:', ll)
    assert_almost_equal(ll_data_slice_1, data_slice_1.ll)
    assert_almost_equal(ll_data_slice_2, data_slice_2.ll)

    #
    # expected one
    weight_1 = len(instance_ids_1) / data.shape[0]
    weight_2 = len(instance_ids_2) / data.shape[0]
    exp_lls = numpy.log(numpy.exp(numpy.log(weight_1) + ll_data_slice_1s) +
                        numpy.exp(numpy.log(weight_2) + ll_data_slice_2s))
    exp_ll = numpy.sum(exp_lls)
    print('Expected LL:', exp_ll)

    assert_almost_equal(exp_ll, ll)

    #
    # creating a fake parent slice
    parent = sls.DataSlice.whole_slice(n_instances, n_instances)
    parent.type = SumNode
    parent.add_child(data_slice_1, weight_1)
    parent.add_child(data_slice_2, weight_2)

    sls.estimate_row_slice_lls(parent, data)
    print('parent LLS', parent.lls)
    assert_array_almost_equal(exp_lls, parent.lls)


def test_estimate_ll():
    #
    # here I am assuming that the simplest function are correct...

    n_instances = data.shape[0]
    #
    # having two data slices
    instance_ids_1 = numpy.array([3, 2, 0, 4])
    instance_ids_2 = numpy.array([1])
    instance_ids_3 = numpy.array([0, 1, 2, 3, 4])

    feature_ids_1 = numpy.array([0, 3, 2, 4])
    feature_ids_2 = numpy.array([0])

    feature_vals = numpy.array([2, 2, 2, 2, 2])

    data_slice_1 = sls.DataSlice(instance_ids_1,
                                 feature_ids_1)
    data_slice_2 = sls.DataSlice(instance_ids_2,
                                 feature_ids_1)
    data_slice_3 = sls.DataSlice(instance_ids_3,
                                 feature_ids_2)

    row_partition = [data_slice_1, data_slice_2]
    data_partition = [[data_slice_3], row_partition]
    alpha = 0.1

    e_counts = numpy.array([[2, 3],
                            [1, 1],
                            [0, 0],
                            [3, 0],
                            [9, 11]])
    ll_freqs = numpy.array([[-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1],
                            [-9.0, -0.1]])

    #
    # computing the single slices' estimated lls
    ll_data_slice_1s = sls.estimate_data_slice_ll(data_slice_1,
                                                  e_counts,
                                                  ll_freqs,
                                                  data,
                                                  feature_vals,
                                                  alpha,
                                                  data)

    ll_data_slice_2s = sls.estimate_data_slice_ll(data_slice_2,
                                                  e_counts,
                                                  ll_freqs,
                                                  data,
                                                  feature_vals,
                                                  alpha,
                                                  data)

    ll_data_slice_3s = sls.estimate_data_slice_ll(data_slice_3,
                                                  e_counts,
                                                  ll_freqs,
                                                  data,
                                                  feature_vals,
                                                  alpha,
                                                  data)

    ll_data_slice_2 = numpy.sum(ll_data_slice_2s) / n_instances
    ll_data_slice_1 = numpy.sum(ll_data_slice_1s) / n_instances
    ll_data_slice_3 = numpy.sum(ll_data_slice_3s) / n_instances

    print('LL on the single slices',
          ll_data_slice_1s,
          ll_data_slice_2s,
          ll_data_slice_3s)

    ll_row_s = sls.estimate_ll([row_partition],
                               e_counts,
                               ll_freqs,
                               data,
                               feature_vals,
                               alpha,
                               data)

    ll_rows = sls.estimate_row_split_ll(row_partition,
                                        e_counts,
                                        ll_freqs,
                                        data,
                                        feature_vals,
                                        alpha,
                                        data)

    ll_row_s_tot = numpy.sum(ll_row_s) / n_instances
    ll_rows_i = numpy.sum(ll_rows) / n_instances
    # print('Estimated row LL:', ll_row_s, (numpy.sum(ll_rows) / n_instances))
    # assert_almost_equal(ll_row, (numpy.sum(ll_rows) / n_instances))
    print('Estimated row LL:', ll_row_s_tot, ll_rows_i)

    lls = sls.estimate_ll(data_partition,
                          e_counts,
                          ll_freqs,
                          data,
                          feature_vals,
                          alpha,
                          data)
    ll = numpy.sum(lls) / n_instances
    print('Estimated total LL:', lls)

    assert_almost_equal(ll_data_slice_1, data_slice_1.ll)
    assert_almost_equal(ll_data_slice_2, data_slice_2.ll)
    assert_almost_equal(ll_data_slice_3, data_slice_3.ll)

    exp_ll = ll_rows_i + ll_data_slice_3
    print('Expected total LL:', exp_ll)

    assert_almost_equal(exp_ll, ll)


def test_random_binary_row_split():
    #
    # creating a data slice
    n_features = n_instances = 5
    data_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2))
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = [2 for i in range(n_features)]

    ll = sls.estimate_data_slice_ll(data_slice,
                                    e_counts,
                                    ll_freqs,
                                    data,
                                    feature_vals,
                                    alpha,
                                    data)

    print('Whole data slice:', data_slice)
    print('ll:', ll)

    #
    # splitting horizontally
    # rand_gen = random.Random(1337)
    row_partition = sls.random_binary_row_split(data_slice,
                                                rand_gen)

    #
    # compacting and decompacting
    #
    ll_row_1 = sls.estimate_ll(row_partition,
                               e_counts,
                               ll_freqs,
                               data,
                               feature_vals,
                               alpha,
                               data)

    s_row_part, = row_partition
    ll_row_2 = sls.estimate_row_split_ll(s_row_part,
                                         e_counts,
                                         ll_freqs,
                                         data,
                                         feature_vals,
                                         alpha,
                                         data)

    # assert_almost_equal(numpy.sum(ll_row_2) / n_instances, ll_row_1)
    assert_almost_equal(ll_row_2, ll_row_1)

    for i, data_slice_r in enumerate(row_partition):
        print('ROW SLICE {0}'.format(i), data_slice_r)

    # data_slice_1 = row_partition[0]
    # data_slice_2 = row_partition[1]
    # ll_data_slice_1 = data_slice_1.ll
    # ll_data_slice_2 = data_slice_2.ll
    # w_data_slice_1 = data_slice_1.n_instances()
    # w_data_slice_2 = data_slice_2.n_instances()
    # tot_instances = data_slice.n_instances()
    # print('LLS', ll_data_slice_1, ll_data_slice_2)

    #
    # the original slice instance ids have been shuffled
    print('Original slice', data_slice)
    #
    # does it change its ll?
    n_ll = sls.estimate_data_slice_ll(data_slice,
                                      e_counts,
                                      ll_freqs,
                                      data,
                                      feature_vals,
                                      alpha,
                                      data)
    print('ll:', n_ll)
    assert_almost_equal(ll, n_ll)

    # exp_ll = math.log(w_data_slice_1 / tot_instances *
    #                   math.exp(ll_data_slice_1) +
    #                   w_data_slice_2 / tot_instances *
    #                   math.exp(ll_data_slice_2))
    # print('exp ll:', exp_ll)


def test_random_binary_col_then_row_split():
    #
    # creating a data slice
    n_features = n_instances = 5
    data_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2))
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = [2 for i in range(n_features)]

    ll = sls.estimate_data_slice_ll(data_slice,
                                    e_counts,
                                    ll_freqs,
                                    data,
                                    feature_vals,
                                    alpha,
                                    data)

    print('Whole data slice:', data_slice)
    print('ll:', ll)

    #
    # splitting vertically, then horizontally once
    # rand_gen = random.Random(1337)
    data_partitions = sls.random_binary_col_then_row_split(data_slice,
                                                           rand_gen,
                                                           False)

    for partition in data_partitions:
        if getattr(partition, '__iter__', False):
            for slice in partition:
                print('ROW SLICE', slice)
        else:
            print('COL SLICE', partition)

    #
    # checking for the same ll
    n_ll = sls.estimate_data_slice_ll(data_slice,
                                      e_counts,
                                      ll_freqs,
                                      data,
                                      feature_vals,
                                      alpha,
                                      data)

    print('Whole data slice:', data_slice)
    print('ll:', n_ll)

    assert_almost_equal(n_ll, ll)

#
# this is taken from
#


def stirling_2(n, k):
    n1 = n
    k1 = k
    if n <= 0:
        return 1

    elif k <= 0:
        return 0

    elif (n == 0 and k == 0):
        return -1

    elif n != 0 and n == k:
        return 1

    elif n < k:
        return 0

    else:
        temp1 = stirling_2(n1 - 1, k1)
        temp1 = k1 * temp1
        return (k1 * (stirling_2(n1 - 1, k1))) + stirling_2(n1 - 1, k1 - 1)


def test_partitioning_correctness():
    n_objects = 6
    k = 2
    s_n_k = stirling_2(n_objects, k)

    #
    # first with the more general routine
    #
    partitions = numpy.zeros(n_objects, dtype=int)
    m_partitions = numpy.zeros(n_objects, dtype=int)

    k_partitions = []

    sls.first_k_partition(partitions, m_partitions, n_objects, k)
    k_partitions.append(numpy.copy(partitions))

    while sls.next_k_partition(partitions, m_partitions, n_objects, k):
        k_partitions.append(numpy.copy(partitions))

    k_length = len(k_partitions)
    print('There are {0} partitions'.format(k_length))
    print(k_partitions)

    assert k_length == s_n_k

    #
    # now the binary case
    #
    partitions = numpy.zeros(n_objects, dtype=bool)
    m_partitions = numpy.zeros(n_objects, dtype=bool)

    bin_partitions = []

    sls.first_bin_partition(partitions, m_partitions, n_objects)
    bin_partitions.append(numpy.copy(partitions))

    while sls.next_bin_partition(partitions, m_partitions, n_objects):
        bin_partitions.append(numpy.copy(partitions))

    bin_length = len(bin_partitions)
    print('There are {0} partitions'.format(bin_length))
    print(bin_partitions)

    assert bin_length == s_n_k

    #
    # cheking for equality of generated partitions
    #
    for k_part, bin_part in zip(k_partitions, bin_partitions):
        assert_array_equal(k_part, bin_part.astype(int))


def test_binary_row_split_by_partition():
    #
    # creating a data slice
    n_features = n_instances = 5
    data_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2))
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = [2 for i in range(n_features)]

    ll = sls.estimate_data_slice_ll(data_slice,
                                    e_counts,
                                    ll_freqs,
                                    data,
                                    feature_vals,
                                    alpha,
                                    data)

    print('Whole data slice:', data_slice)
    print('ll:', ll)

    #
    # splitting horizontally
    partition = numpy.array([True, True, False, True, False])
    row_partition = sls.binary_row_split_by_partition(data_slice,
                                                      partition)

    for i, data_slice_r in enumerate(row_partition):
        print('ROW SLICE {0}'.format(i), data_slice_r)

    #
    # the original slice instance ids have been shuffled
    print('Original slice', data_slice)
    #
    # does it change its ll?
    n_ll = sls.estimate_data_slice_ll(data_slice,
                                      e_counts,
                                      ll_freqs,
                                      data,
                                      feature_vals,
                                      alpha,
                                      data)
    print('ll:', n_ll)
    assert_almost_equal(ll, n_ll)


def test_binary_col_then_row_split_by_partition():
    #
    # creating a data slice
    n_features = n_instances = 5
    data_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2))
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = [2 for i in range(n_features)]

    ll = sls.estimate_data_slice_ll(data_slice,
                                    e_counts,
                                    ll_freqs,
                                    data,
                                    feature_vals,
                                    alpha,
                                    data)

    print('Whole data slice:', data_slice)
    print('ll:', ll)

    #
    # splitting vertically, then horizontally once
    # rand_gen = random.Random(1337)
    col_partition = numpy.array([True, False, False, True, False])
    row_partition = numpy.array([False, False, True, True, True])
    instance_ids_1 = numpy.array([0, 1, 4, 2, 3])
    instance_ids_2 = numpy.array([3, 1, 2, 0, 4])
    #
    # splitting both half sizes horizontally as well
    data_partitions = \
        sls.binary_col_then_row_split_by_partition(data_slice,
                                                   col_partition,
                                                   row_partition,
                                                   instance_ids_1,
                                                   instance_ids_2,
                                                   rand_gen,
                                                   True)

    for partition in data_partitions:
        if getattr(partition, '__iter__', False):
            for slice in partition:
                print('ROW SLICE', slice)
        else:
            print('COL SLICE', partition)

    #
    # checking for the same ll
    n_ll = sls.estimate_data_slice_ll(data_slice,
                                      e_counts,
                                      ll_freqs,
                                      data,
                                      feature_vals,
                                      alpha,
                                      data)

    print('Whole data slice:', data_slice)
    print('ll:', n_ll)

    assert_almost_equal(n_ll, ll)


def test_best_neighbor_naive_nltcs():

    logger = logging.getLogger('algo.sls')
    logger.setLevel(logging.DEBUG)

    dataset_name = 'nltcs'
    print('Loading dataset', dataset_name)
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

    n_instances = train.shape[0]
    n_features = train.shape[1]

    whole_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    print('WHOLE slice', whole_slice)
    max_neigh = 1000

    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2), dtype=int)
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = numpy.array([2 for i in range(n_features)])

    # rand_gen = random.Random(1337)
    rand_gen = numpy.random.RandomState(1337)
    stats = sls.SlsStats()

    propagate = False

    best_partition, best_ll, best_path = sls.best_neighbor_naive(whole_slice,
                                                                 max_neigh,
                                                                 0.7,
                                                                 e_counts,
                                                                 ll_freqs,
                                                                 train,
                                                                 feature_vals,
                                                                 alpha,
                                                                 valid,
                                                                 False,
                                                                 propagate,
                                                                 stats,
                                                                 rand_gen)
    print('best partition', best_partition, best_ll, best_path)

    # now with propagation
    propagate = True

    best_partition, best_ll, best_path = sls.best_neighbor_naive(whole_slice,
                                                                 max_neigh,
                                                                 0.7,
                                                                 e_counts,
                                                                 ll_freqs,
                                                                 train,
                                                                 feature_vals,
                                                                 alpha,
                                                                 valid,
                                                                 False,
                                                                 propagate,
                                                                 stats,
                                                                 rand_gen)
    print('best partition', best_partition, best_ll, best_path)


def test_best_neighbor_enum_nltcs():

    logger = logging.getLogger('algo.sls')
    logger.setLevel(logging.DEBUG)

    dataset_name = 'nltcs'
    print('Loading dataset', dataset_name)
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

    n_instances = train.shape[0]
    n_features = train.shape[1]

    whole_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    print('WHOLE slice', whole_slice)
    max_neigh = 1000

    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2), dtype=int)
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = numpy.array([2 for i in range(n_features)])

    # rand_gen = random.Random(1337)
    rand_gen = numpy.random.RandomState(1337)
    stats = sls.SlsStats()

    #
    # no propagation
    propagation = False
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                0.7,
                                                                e_counts,
                                                                ll_freqs,
                                                                train,
                                                                feature_vals,
                                                                alpha,
                                                                valid,
                                                                False,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print('best partition', best_partition, best_ll)

    #
    # with propagation
    propagation = True
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                0.7,
                                                                e_counts,
                                                                ll_freqs,
                                                                train,
                                                                feature_vals,
                                                                alpha,
                                                                valid,
                                                                False,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print('best partition', best_partition, best_ll)


def test_best_neighbor_enum_corr():

    logger = logging.getLogger('algo.sls')
    logger.setLevel(logging.DEBUG)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    whole_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    print('WHOLE slice', whole_slice)
    max_neigh = 1000000

    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2), dtype=int)
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = numpy.array([2 for i in range(n_features)])

    propagation = True
    #
    # beta, only the first operator
    beta = 1.0

    #
    # so at most s_n_k_i
    s_n_k_i = stirling_2(n_instances, 2)
    s_n_k_f = stirling_2(n_features, 2)
    print('Stirling {5 2}', s_n_k_i, s_n_k_f)
    rand_gen = numpy.random.RandomState(1337)
    stats = sls.SlsStats()
    #
    # first with single split
    # num row ops = snk_i
    # num col ops = snk_f * snk_i
    # num row splits = snk_i + snk_f * snk_i
    # num col splits = snk_f
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                False,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_col_splits == s_n_k_f
    #
    # now double split
    # num row ops = snk_i
    # num col ops = snk_f * snk_i
    # num row splits = snk_i + snk_f * snk_i * 2
    # num col splits = snk_f
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                True,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i * 2
    assert stats.n_col_splits == s_n_k_f
    #
    # now with all products first
    beta = 0.0
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                False,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_col_splits == s_n_k_f
    #
    # all products but bisplit
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                True,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i * 2
    assert stats.n_col_splits == s_n_k_f
    #
    # now applying less candiates than the possible ones
    # if alpha_n = 10 -> alpha_m = 5
    # so I can generate only 5 + 5 col-row split ops in total
    # by generating two different col splots
    col_tot = 10
    col_times = 2
    max_neigh = 10
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                False,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == col_tot
    assert stats.n_row_splits == col_tot
    assert stats.n_col_splits == 2
    #
    # now bisplits
    # so the number of row splits is doubled
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                True,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == col_tot
    assert stats.n_row_splits == col_tot * 2
    assert stats.n_col_splits == 2
    #
    # now limited only rows, expecting just 10 splits an nothing more
    beta = 1.0
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_enum(whole_slice,
                                                                max_neigh,
                                                                beta,
                                                                e_counts,
                                                                ll_freqs,
                                                                data,
                                                                feature_vals,
                                                                alpha,
                                                                data,
                                                                True,
                                                                propagation,
                                                                stats,
                                                                rand_gen)
    print(stats)
    assert stats.n_row_op == max_neigh
    assert stats.n_col_then_row_op == 0
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == 0


def test_best_neighbor_naive_corr():

    logger = logging.getLogger('algo.sls')
    logger.setLevel(logging.DEBUG)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    whole_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    print('WHOLE slice', whole_slice)
    max_neigh = 1000

    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2), dtype=int)
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = numpy.array([2 for i in range(n_features)])

    propagation = True
    #
    # beta, only the first operator
    beta = 1.0

    rand_gen = numpy.random.RandomState(1337)
    stats = sls.SlsStats()
    #
    # all rows
    # expecting nothing less that max_neigh ops
    best_partition, best_ll, best_path = sls.best_neighbor_naive(whole_slice,
                                                                 max_neigh,
                                                                 beta,
                                                                 e_counts,
                                                                 ll_freqs,
                                                                 data,
                                                                 feature_vals,
                                                                 alpha,
                                                                 data,
                                                                 False,
                                                                 propagation,
                                                                 stats,
                                                                 rand_gen)
    print(stats)
    assert stats.n_row_op == max_neigh
    assert stats.n_col_then_row_op == 0
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == 0
    #
    # now all cols
    # expecting something similar
    beta = 0.0
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_naive(whole_slice,
                                                                 max_neigh,
                                                                 beta,
                                                                 e_counts,
                                                                 ll_freqs,
                                                                 data,
                                                                 feature_vals,
                                                                 alpha,
                                                                 data,
                                                                 False,
                                                                 propagation,
                                                                 stats,
                                                                 rand_gen)
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == max_neigh
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == max_neigh
    #
    # finally all cols with bisplit
    # expecting double row splits
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_naive(whole_slice,
                                                                 max_neigh,
                                                                 beta,
                                                                 e_counts,
                                                                 ll_freqs,
                                                                 data,
                                                                 feature_vals,
                                                                 alpha,
                                                                 data,
                                                                 True,
                                                                 propagation,
                                                                 stats,
                                                                 rand_gen)
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == max_neigh
    assert stats.n_row_splits == max_neigh * 2
    assert stats.n_col_splits == max_neigh
    #
    # just checking for an arbitrary proportion if they are unbiased
    #
    beta = 0.4
    stats = sls.SlsStats()
    best_partition, best_ll, best_path = sls.best_neighbor_naive(whole_slice,
                                                                 max_neigh,
                                                                 beta,
                                                                 e_counts,
                                                                 ll_freqs,
                                                                 data,
                                                                 feature_vals,
                                                                 alpha,
                                                                 data,
                                                                 False,
                                                                 propagation,
                                                                 stats,
                                                                 rand_gen)
    print(stats)
    assert (stats.n_row_op + stats.n_col_then_row_op) == max_neigh
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_then_row_op == stats.n_col_splits


def test_is_complex_partition():
    n_features = n_instances = 5
    data_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2))
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = [2 for i in range(n_features)]

    ll = sls.estimate_data_slice_ll(data_slice,
                                    e_counts,
                                    ll_freqs,
                                    data,
                                    feature_vals,
                                    alpha,
                                    data)

    print('Whole data slice:', data_slice)
    print('ll:', ll)

    #
    # splitting horizontally, simple partition
    partition = numpy.array([True, True, False, True, False])
    row_partition = sls.binary_row_split_by_partition(data_slice,
                                                      partition)

    is_complex = sls.is_complex_partition(row_partition)
    print(row_partition)
    assert is_complex is False

    # now cols then rows
    rand_gen = numpy.random.RandomState(1337)
    col_partition = numpy.array([True, False, False, True, False])
    row_partition = numpy.array([False, False, True, True, True])
    instance_ids_1 = numpy.array([0, 1, 4, 2, 3])
    instance_ids_2 = numpy.array([3, 1, 2, 0, 4])
    #
    # splitting both half sizes horizontally as well
    data_partitions = \
        sls.binary_col_then_row_split_by_partition(data_slice,
                                                   col_partition,
                                                   row_partition,
                                                   instance_ids_1,
                                                   instance_ids_2,
                                                   rand_gen,
                                                   False)
    print(data_partitions)
    is_complex = sls.is_complex_partition(data_partitions)
    assert is_complex is True
    #
    # now with bisplit
    data_partitions = \
        sls.binary_col_then_row_split_by_partition(data_slice,
                                                   col_partition,
                                                   row_partition,
                                                   instance_ids_1,
                                                   instance_ids_2,
                                                   rand_gen,
                                                   True)
    print(data_partitions)
    is_complex = sls.is_complex_partition(data_partitions)
    assert is_complex is True

    #
    # now with random splitters
    #
    row_partition = sls.random_binary_row_split(data_slice,
                                                rand_gen)
    is_complex = sls.is_complex_partition(row_partition)
    print(row_partition)
    assert is_complex is False

    data_partitions = sls.random_binary_col_then_row_split(data_slice,
                                                           rand_gen,
                                                           False)
    print(data_partitions)
    is_complex = sls.is_complex_partition(data_partitions)
    assert is_complex is True

    data_partitions = sls.random_binary_col_then_row_split(data_slice,
                                                           rand_gen,
                                                           True)
    print(data_partitions)
    is_complex = sls.is_complex_partition(data_partitions)
    assert is_complex is True


def test_simple_rii_spn_toy():
    #
    # using the 5x5 toy matrix
    feature_vals = [2 for i in range(5)]
    max_neigh = 100
    beta = 0.5
    alpha = 0.1
    min_inst = 1
    bisplit = False
    theta = 0.0
    propagate = True
    spn, stats = sls.simple_rii_spn(data,
                                    feature_vals,
                                    data,
                                    max_neigh,
                                    theta,
                                    beta,
                                    alpha,
                                    min_inst,
                                    bisplit,
                                    propagate,
                                    rand_gen)

    print(spn)


def test_propagate_ll():
    #
    # testing the propagation upwards of the lls
    #

    #
    # simple case: no parents
    n_instances = data.shape[0]
    n_features = data.shape[1]

    data_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    old_lls = numpy.log(numpy.array([0.1, 0.2, 0.4, 0.2, 0.3]))
    old_ll = numpy.sum(old_lls) / n_instances
    data_slice.lls = old_lls
    new_lls = numpy.log(numpy.array([0.2, 0.1, 0.6, 0.6, 0.7]))
    new_ll = numpy.sum(new_lls) / n_instances

    print('old ll {0} new lls {1}'.format(old_ll, new_ll))

    #
    # there are no parents so ti shall return the same new_lls array
    p_lls, p_lls_path = sls.propagate_ll(data_slice, new_lls)

    assert_array_almost_equal(new_lls, p_lls)
    assert [new_lls] == p_lls_path

    print('out lls {0}, exp lls {1}'.format(p_lls, new_lls))

    #
    # now adding a parent which is a product data slice
    # it does not matter now for data slices to correctly represent slices
    prod_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    prod_lls = numpy.log(numpy.array([0.1, 0.3, 0.2, 0.5, 0.1]))
    prod_slice.lls = prod_lls
    prod_slice.type = ProductNode  # type(ProductNode)
    prod_slice.add_child(data_slice)

    p_lls, p_lls_path = sls.propagate_ll(data_slice, new_lls)

    # manual check
    new_prod_lls = prod_lls - old_lls + new_lls
    assert_array_almost_equal(new_prod_lls, p_lls)
    print('PL path', p_lls_path)
    for exp_lls, f_lls in zip([new_lls, new_prod_lls], p_lls_path):
        assert_array_almost_equal(exp_lls, f_lls)

    print('out lls {0}, exp lls {1}'.format(p_lls, new_prod_lls))

    #
    # now adding another parent which is a sum node
    sum_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    sum_lls = numpy.log(numpy.array([0.2, 0.23, 0.33, 0.5, 0.2]))
    sum_slice.lls = sum_lls
    sum_slice.type = SumNode  # type(SumNode)
    w = 0.44
    sum_slice.add_child(prod_slice, w)

    assert_almost_equal(prod_slice.w, w)

    p_lls, p_lls_path = sls.propagate_ll(data_slice, new_lls)

    new_sum_lls = numpy.log(numpy.exp(sum_lls) -
                            numpy.exp(numpy.log(w) + prod_lls) +
                            numpy.exp(numpy.log(w) + new_prod_lls))

    print('out lls {0}, exp lls {1}'.format(
        p_lls, new_sum_lls))

    assert_array_almost_equal(new_sum_lls, p_lls)

    for exp_lls, f_lls in zip([new_lls, new_prod_lls, new_sum_lls],
                              p_lls_path):
        assert_array_almost_equal(exp_lls, f_lls)

    #
    # and now I am trying to update the best lls in the path
    sls.update_best_lls(data_slice,
                        p_lls_path)

    #
    # checking manually
    assert_array_almost_equal(new_lls, data_slice.lls)
    assert_array_almost_equal(new_prod_lls, prod_slice.lls)
    assert_array_almost_equal(new_sum_lls, sum_slice.lls)


def test_best_candidate_neighbors_naive_corr():

    logger = logging.getLogger('algo.sls')
    logger.setLevel(logging.DEBUG)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    whole_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    print('WHOLE slice', whole_slice)
    current_best_ll = -1000.0
    max_neigh = 1000
    marg = 0.0

    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2), dtype=int)
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = numpy.array([2 for i in range(n_features)])

    propagation = True
    #
    # beta, only the first operator
    beta = 1.0

    rand_gen = numpy.random.RandomState(1337)
    stats = sls.SlsStats()
    #
    # all rows
    # expecting nothing less that max_neigh ops
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_naive(whole_slice,
                                           current_best_ll,
                                           max_neigh, marg,
                                           beta,
                                           e_counts,
                                           ll_freqs,
                                           data,
                                           feature_vals,
                                           alpha,
                                           data,
                                           False,
                                           propagation,
                                           stats,
                                           rand_gen)
    assert len(best_partitions) == max_neigh
    assert len(best_lls) == max_neigh
    assert len(best_paths) == max_neigh
    print(stats)
    assert stats.n_row_op == max_neigh
    assert stats.n_col_then_row_op == 0
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == 0
    #
    # now all cols
    # expecting something similar
    beta = 0.0
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_naive(whole_slice,
                                           current_best_ll,
                                           max_neigh, marg,
                                           beta,
                                           e_counts,
                                           ll_freqs,
                                           data,
                                           feature_vals,
                                           alpha,
                                           data,
                                           False,
                                           propagation,
                                           stats,
                                           rand_gen)
    assert len(best_partitions) == max_neigh
    assert len(best_lls) == max_neigh
    assert len(best_paths) == max_neigh
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == max_neigh
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == max_neigh
    #
    # finally all cols with bisplit
    # expecting double row splits
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_naive(whole_slice,
                                           current_best_ll,
                                           max_neigh, marg,
                                           beta,
                                           e_counts,
                                           ll_freqs,
                                           data,
                                           feature_vals,
                                           alpha,
                                           data,
                                           True,
                                           propagation,
                                           stats,
                                           rand_gen)
    assert len(best_partitions) == max_neigh
    assert len(best_lls) == max_neigh
    assert len(best_paths) == max_neigh
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == max_neigh
    assert stats.n_row_splits == max_neigh * 2
    assert stats.n_col_splits == max_neigh
    #
    # just checking for an arbitrary proportion if they are unbiased
    #
    beta = 0.4
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_naive(whole_slice,
                                           current_best_ll,
                                           max_neigh, marg,
                                           beta,
                                           e_counts,
                                           ll_freqs,
                                           data,
                                           feature_vals,
                                           alpha,
                                           data,
                                           False,
                                           propagation,
                                           stats,
                                           rand_gen)
    assert len(best_partitions) == max_neigh
    assert len(best_lls) == max_neigh
    assert len(best_paths) == max_neigh
    print(stats)
    assert (stats.n_row_op + stats.n_col_then_row_op) == max_neigh
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_then_row_op == stats.n_col_splits

    #
    # now with the highest ll, so expecting no split
    current_best_ll = 0.0
    beta = 0.0

    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_naive(whole_slice,
                                           current_best_ll,
                                           max_neigh, marg,
                                           beta,
                                           e_counts,
                                           ll_freqs,
                                           data,
                                           feature_vals,
                                           alpha,
                                           data,
                                           True,
                                           propagation,
                                           stats,
                                           rand_gen)
    assert len(best_partitions) == 0
    assert len(best_lls) == 0
    assert len(best_paths) == 0
    print(stats)
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == max_neigh
    assert stats.n_row_splits == max_neigh * 2
    assert stats.n_col_splits == max_neigh
    #
    # just checking for an arbitrary proportion if they are unbiased
    #
    beta = 0.4
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_naive(whole_slice,
                                           current_best_ll,
                                           max_neigh, marg,
                                           beta,
                                           e_counts,
                                           ll_freqs,
                                           data,
                                           feature_vals,
                                           alpha,
                                           data,
                                           False,
                                           propagation,
                                           stats,
                                           rand_gen)
    assert len(best_partitions) == 0
    assert len(best_lls) == 0
    assert len(best_paths) == 0
    print(stats)
    assert (stats.n_row_op + stats.n_col_then_row_op) == max_neigh
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_then_row_op == stats.n_col_splits


def test_best_candidate_neighbors_enum_corr():

    logger = logging.getLogger('algo.sls')
    logger.setLevel(logging.DEBUG)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    whole_slice = sls.DataSlice.whole_slice(n_instances, n_features)
    print('WHOLE slice', whole_slice)
    max_neigh = 1000000
    marg = 0.0

    alpha = 0.1
    e_counts = numpy.zeros((n_features, 2), dtype=int)
    ll_freqs = numpy.zeros((n_features, 2))
    feature_vals = numpy.array([2 for i in range(n_features)])

    current_best_ll = -1000.0

    propagation = True
    #
    # beta, only the first operator
    beta = 1.0

    #
    # so at most s_n_k_i
    s_n_k_i = stirling_2(n_instances, 2)
    s_n_k_f = stirling_2(n_features, 2)
    print('Stirling {5 2}', s_n_k_i, s_n_k_f)
    rand_gen = numpy.random.RandomState(1337)
    stats = sls.SlsStats()
    #
    # first with single split
    # num row ops = snk_i
    # num col ops = snk_f * snk_i
    # num row splits = snk_i + snk_f * snk_i
    # num col splits = snk_f
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          False,
                                          propagation,
                                          stats,
                                          rand_gen)
    print('BP', len(best_partitions),  s_n_k_i + s_n_k_f * s_n_k_i)
    assert len(best_partitions) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_lls) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_paths) == s_n_k_i + s_n_k_f * s_n_k_i
    print(stats)
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_col_splits == s_n_k_f
    #
    # now double split
    # num row ops = snk_i
    # num col ops = snk_f * snk_i
    # num row splits = snk_i + snk_f * snk_i * 2
    # num col splits = snk_f
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_lls) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_paths) == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i * 2
    assert stats.n_col_splits == s_n_k_f
    #
    # now with all products first
    beta = 0.0
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          False,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_lls) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_paths) == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_col_splits == s_n_k_f
    #
    # all products but bisplit
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_lls) == s_n_k_i + s_n_k_f * s_n_k_i
    assert len(best_paths) == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i * 2
    assert stats.n_col_splits == s_n_k_f
    #
    # now applying less candiates than the possible ones
    # if alpha_n = 10 -> alpha_m = 5
    # so I can generate only 5 + 5 col-row split ops in total
    # by generating two different col splots
    col_tot = 10
    col_times = 2
    max_neigh = 10
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          False,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == col_tot
    assert len(best_paths) == col_tot
    assert len(best_lls) == col_tot

    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == col_tot
    assert stats.n_row_splits == col_tot
    assert stats.n_col_splits == 2
    #
    # now bisplits
    # so the number of row splits is doubled
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == col_tot
    assert len(best_paths) == col_tot
    assert len(best_lls) == col_tot
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == col_tot
    assert stats.n_row_splits == col_tot * 2
    assert stats.n_col_splits == 2
    #
    # now limited only rows, expecting just 10 splits an nothing more
    beta = 1.0
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == max_neigh
    assert len(best_paths) == max_neigh
    assert len(best_lls) == max_neigh
    assert stats.n_row_op == max_neigh
    assert stats.n_col_then_row_op == 0
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == 0

    #
    # now making an impossible addition
    #
    #
    current_best_ll = 0.0
    # now with all products first
    beta = 0.0
    max_neigh = 100000
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          False,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == 0
    assert len(best_lls) == 0
    assert len(best_paths) == 0
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i
    assert stats.n_col_splits == s_n_k_f
    #
    # all products but bisplit
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == 0
    assert len(best_lls) == 0
    assert len(best_paths) == 0
    assert stats.n_row_op == s_n_k_i
    assert stats.n_col_then_row_op == s_n_k_f * s_n_k_i
    assert stats.n_row_splits == s_n_k_i + s_n_k_f * s_n_k_i * 2
    assert stats.n_col_splits == s_n_k_f
    #
    # now applying less candiates than the possible ones
    # if alpha_n = 10 -> alpha_m = 5
    # so I can generate only 5 + 5 col-row split ops in total
    # by generating two different col splots
    col_tot = 10
    col_times = 2
    max_neigh = 10
    marg = 0.0
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          False,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == 0
    assert len(best_paths) == 0
    assert len(best_lls) == 0

    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == col_tot
    assert stats.n_row_splits == col_tot
    assert stats.n_col_splits == 2
    #
    # now bisplits
    # so the number of row splits is doubled
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == 0
    assert len(best_paths) == 0
    assert len(best_lls) == 0
    assert stats.n_row_op == 0
    assert stats.n_col_then_row_op == col_tot
    assert stats.n_row_splits == col_tot * 2
    assert stats.n_col_splits == 2
    #
    # now limited only rows, expecting just 10 splits an nothing more
    beta = 1.0
    stats = sls.SlsStats()
    best_partitions, best_lls, best_paths = \
        sls.best_candidate_neighbors_enum(whole_slice,
                                          current_best_ll,
                                          max_neigh, marg,
                                          beta,
                                          e_counts,
                                          ll_freqs,
                                          data,
                                          feature_vals,
                                          alpha,
                                          data,
                                          True,
                                          propagation,
                                          stats,
                                          rand_gen)
    print(stats)
    assert len(best_partitions) == 0
    assert len(best_paths) == 0
    assert len(best_lls) == 0
    assert stats.n_row_op == max_neigh
    assert stats.n_col_then_row_op == 0
    assert stats.n_row_splits == max_neigh
    assert stats.n_col_splits == 0


def test_rii_spn_toy():
    #
    # using the 5x5 toy matrix
    feature_vals = [2 for i in range(5)]
    max_neigh = 10000
    beta = 0.5
    alpha = 0.1
    min_inst = 1
    bisplit = False
    theta = 0.0  # 1.0
    propagate = True
    enum = False
    marg = 0.1

    spn, stats = sls.rii_spn(data,
                             feature_vals,
                             data,
                             max_neigh,
                             marg,
                             theta,
                             beta,
                             alpha,
                             min_inst,
                             bisplit,
                             propagate,
                             enum,
                             rand_gen)

    print(spn)
