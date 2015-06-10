from spn import LOG_ZERO

import numpy

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

# from cltree.cltree import minimum_spanning_tree
# from cltree.cltree import minimum_spanning_tree_numba

from cltree.cltree import CLTree
from cltree.cltree import instantiate_factors
from cltree.cltree import tree_2_factor_matrix
from cltree.cltree import marginalize

try:
    from time import perf_counter
except:
    from time import time as perf_counter

data = numpy.array([[1, 1, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 1, 0]])
n_features = 4
n_instances = 6
copy_mi = True


def test_compute_probs_sparse():

    #
    # creating the tree upon synthetic data
    clt = CLTree(data=data,
                 alpha=0.0,
                 sparse=True,
                 mem_free=False)

    counts_1 = data.sum(axis=0)
    counts_0 = n_instances - counts_1
    counts = numpy.column_stack([counts_0, counts_1])
    probs = counts / n_instances
    print('Computed marg freqs and probs', counts, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    print ('Computed marg logs for alpha=0', log_probs)
    print ('CLT marg logs for alpha=0', clt._log_marg_probs)

    assert_almost_equal(clt._log_marg_probs, log_probs)
    assert_almost_equal(clt._marg_freqs, counts_1)

    #
    # now with another value for alpha
    alpha = 1.0
    clt = CLTree(data=data,
                 alpha=alpha,
                 sparse=True,
                 mem_free=False)
    probs = (counts + 2 * alpha) / (n_instances + 4 * alpha)
    print('Computed probs with alpha=', alpha, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    print ('Computed marg logs for alpha=', alpha, log_probs)
    print ('CLT marg logs for alpha=', alpha, clt._log_marg_probs)

    assert_almost_equal(clt._log_marg_probs, log_probs)
    assert_almost_equal(clt._marg_freqs, counts_1)


def test_compute_probs_dense():

    #
    # creating the tree upon synthetic data
    clt = CLTree(data=data,
                 alpha=0.0,
                 sparse=False,
                 mem_free=False)

    counts_1 = data.sum(axis=0)
    counts_0 = n_instances - counts_1
    counts = numpy.column_stack([counts_0, counts_1])
    probs = counts / n_instances
    print('Computed marg freqs and probs', counts, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    print ('Computed marg logs for alpha=0', log_probs)
    print ('CLT marg logs for alpha=0', clt._log_marg_probs)

    assert_almost_equal(clt._log_marg_probs, log_probs)
    assert_almost_equal(clt._marg_freqs, counts_1)

    #
    # now with another value for alpha
    alpha = 1.0
    clt = CLTree(data=data,
                 alpha=alpha,
                 sparse=False,
                 mem_free=False)
    probs = (counts + 2 * alpha) / (n_instances + 4 * alpha)
    print('Computed probs with alpha=', alpha, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    print ('Computed marg logs for alpha=', alpha, log_probs)
    print ('CLT marg logs for alpha=', alpha, clt._log_marg_probs)

    assert_almost_equal(clt._log_marg_probs, log_probs)
    assert_almost_equal(clt._marg_freqs, counts_1)


def test_compute_joint_probs():
    #
    # creating the tree upon synthetic data
    clt = CLTree(data=data,
                 alpha=0.0,
                 sparse=True,
                 mem_free=False)
    joint_counts = numpy.zeros((n_features,
                                n_features,
                                2,
                                2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for instance in data:
                    joint_counts[i, j, instance[i], instance[j]] += 1

    print('Computed co freqs')
    print(joint_counts)
    print(clt._joint_freqs)

    assert_almost_equal(clt._joint_freqs, joint_counts)
    #
    # going to logs
    joint_probs = joint_counts / n_instances
    log_joint_probs = numpy.log(joint_probs)
    log_joint_probs[numpy.isinf(log_joint_probs)] = LOG_ZERO
    #
    # to have a complete match the diagonal entries are left to zero
    for i in range(n_features):
        log_joint_probs[i, i] = 0

    assert_almost_equal(log_joint_probs, clt._log_joint_probs)
    #
    # dense case
    clt = CLTree(data=data,
                 alpha=0.0,
                 sparse=False,
                 mem_free=False)
    assert_almost_equal(log_joint_probs, clt._log_joint_probs)

    #
    # changing alpha
    alpha = 1.0
    joint_probs = (joint_counts + alpha) / (n_instances + 4.0 * alpha)
    log_joint_probs = numpy.log(joint_probs)
    log_joint_probs[numpy.isinf(log_joint_probs)] = LOG_ZERO
    #
    # to have a complete match the diagonal entries are left to zero
    for i in range(n_features):
        log_joint_probs[i, i] = 0

    clt = CLTree(data=data,
                 alpha=alpha,
                 sparse=True,
                 mem_free=False)
    assert_almost_equal(log_joint_probs, clt._log_joint_probs)

    #
    # doing a dense version
    clt = CLTree(data=data,
                 alpha=alpha,
                 sparse=False,
                 mem_free=False)
    assert_almost_equal(log_joint_probs, clt._log_joint_probs)

#
# TODO: this test is deprecated since there is no point in computing cond
# probs anymore


def test_compute_cond_probs():
    #
    # creating the tree upon synthetic data
    clt = CLTree(data=data,
                 alpha=0.0,
                 sparse=False,
                 mem_free=False)

    # cond_probs = numpy.zeros((n_features,
    #                           n_features,
    #                           2,
    #                           2))
    # for i in range(n_features):
    #     for j in range(n_features):
    #         if i != j:
    #             for instance in data:
    #                 cond_probs[i, j, instance[i], instance[j]] += 1
    #             #
    #             # now normalizing
    #             sums = cond_probs[i, j].sum(axis=0)
    #             print('sums', sums)
    #             cond_probs[i, j] /= sums

    # cond_probs[numpy.isnan(cond_probs)] = 0
    # print ('Computed cond probs', cond_probs)
    # log_cond_probs = numpy.log(cond_probs)
    # log_cond_probs[numpy.isinf(log_cond_probs)] = LOG_ZERO

    # for i in range(n_features):
    #     log_cond_probs[i, i] = 0

    # print(log_cond_probs)
    # print('logs\n', clt._log_cond_probs)
    # assert_almost_equal(log_cond_probs, clt._log_cond_probs)

    joint_counts = numpy.zeros((n_features,
                                n_features,
                                2,
                                2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for instance in data:
                    joint_counts[i, j, instance[i], instance[j]] += 1

    print('Computed co freqs')
    print(joint_counts)
    print(clt._joint_freqs)

    assert_almost_equal(clt._joint_freqs, joint_counts)

    #
    # checking sparseness
    clt = CLTree(data=data,
                 alpha=0.0,
                 sparse=True,
                 mem_free=False)
    assert_almost_equal(clt._joint_freqs, joint_counts)

    #
    # going to logs
    joint_probs = joint_counts / n_instances
    log_joint_probs = numpy.log(joint_probs)
    log_joint_probs[numpy.isinf(log_joint_probs)] = LOG_ZERO
    #
    # to have a complete match the diagonal entries are left to zero
    for i in range(n_features):
        log_joint_probs[i, i] = 0

    counts_1 = data.sum(axis=0)
    counts_0 = n_instances - counts_1
    counts = numpy.column_stack([counts_0, counts_1])
    probs = counts / n_instances
    print('Computed marg freqs and probs', counts, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    log_cond_probs = numpy.zeros((n_features,
                                  n_features,
                                  2,
                                  2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for k in range(2):
                    for h in range(2):
                        log_cond_probs[i, j, k, h] = \
                            log_joint_probs[i, j, k, h] - log_probs[j, h]

    print('Computed log cond probs with alpha=0', log_cond_probs)
    assert_array_almost_equal(log_cond_probs, clt._log_cond_probs)

    #
    # testing factors
    print('\nTesting factors')
    for i in range(n_features):
        parent_id = clt._tree[i]
        for k in range(2):
            for h in range(2):
                if i != parent_id:
                    print(clt._log_cond_probs[i, parent_id, k, h],
                          clt._factors[i, k, h])
                    assert_array_almost_equal(clt._log_cond_probs[i, parent_id, k, h],
                                              clt._factors[i, k, h])

    alpha = 2.0
    clt = CLTree(data, alpha=alpha,
                 sparse=False,
                 mem_free=False)
    #
    # going to logs
    joint_probs = (joint_counts + alpha) / (n_instances + 4 * alpha)
    log_joint_probs = numpy.log(joint_probs)
    log_joint_probs[numpy.isinf(log_joint_probs)] = LOG_ZERO
    #
    # to have a complete match the diagonal entries are left to zero
    for i in range(n_features):
        log_joint_probs[i, i] = 0

    counts_1 = data.sum(axis=0)
    counts_0 = n_instances - counts_1
    counts = numpy.column_stack([counts_0, counts_1])
    probs = (counts + 2 * alpha) / (n_instances + 4 * alpha)
    print('Computed marg freqs and probs', counts, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    log_cond_probs = numpy.zeros((n_features,
                                  n_features,
                                  2,
                                  2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for k in range(2):
                    for h in range(2):
                        log_cond_probs[i, j, k, h] = \
                            log_joint_probs[i, j, k, h] - log_probs[j, h]

    print('Computed log cond probs with alpha=0', log_cond_probs)
    assert_array_almost_equal(log_cond_probs, clt._log_cond_probs)

    #
    # sparse version
    clt = CLTree(data, alpha=alpha,
                 sparse=True,
                 mem_free=False)
    assert_array_almost_equal(log_cond_probs, clt._log_cond_probs)


def test_compute_mi():
    counts_1 = data.sum(axis=0)
    counts_0 = n_instances - counts_1
    counts = numpy.column_stack([counts_0, counts_1])
    probs = counts / n_instances
    print('Computed marg freqs and probs', counts, probs)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    log_prods = numpy.zeros((n_features,
                             n_features,
                             2,
                             2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for k in range(2):
                    for h in range(2):
                        log_prods[i, j, k, h] = \
                            log_probs[i, k] + log_probs[j, h]

    joint_counts = numpy.zeros((n_features,
                                n_features,
                                2,
                                2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for instance in data:
                    joint_counts[i, j, instance[i], instance[j]] += 1

    print('Computed co freqs')
    print(joint_counts)

    #
    # going to logs
    joint_probs = joint_counts / n_instances
    log_joint_probs = numpy.log(joint_probs)
    log_joint_probs[numpy.isinf(log_joint_probs)] = LOG_ZERO

    for i in range(n_features):
        log_joint_probs[i, i] = 0

    mutual_info = numpy.exp(log_joint_probs) * (log_joint_probs - log_prods)
    mutual_info = mutual_info.sum(axis=2).sum(axis=2)

    print('Computed MI:', mutual_info, type(mutual_info))

    clt = CLTree(data, alpha=0.0, sparse=False, mem_free=False)
    print('CLTree', clt._mutual_info, type(clt._mutual_info))

    assert_almost_equal(mutual_info, clt._mutual_info)

    #
    # adding sparsity
    clt = CLTree(data, alpha=0.0, sparse=True, mem_free=False)
    assert_almost_equal(mutual_info, clt._mutual_info)

    #
    # now with alpha
    alpha = 0.5
    probs = (counts + 2 * alpha) / (n_instances + 4 * alpha)

    log_probs = numpy.log(probs)
    log_probs[numpy.isinf(log_probs)] = LOG_ZERO

    log_prods = numpy.zeros((n_features,
                             n_features,
                             2,
                             2))
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                for k in range(2):
                    for h in range(2):
                        log_prods[i, j, k, h] = \
                            log_probs[i, k] + log_probs[j, h]

    joint_probs = (joint_counts + alpha) / (n_instances + 4 * alpha)
    log_joint_probs = numpy.log(joint_probs)
    log_joint_probs[numpy.isinf(log_joint_probs)] = LOG_ZERO

    for i in range(n_features):
        log_joint_probs[i, i] = 0

    mutual_info = numpy.exp(log_joint_probs) * (log_joint_probs - log_prods)
    mutual_info = mutual_info.sum(axis=2).sum(axis=2)

    clt = CLTree(data, alpha=alpha, sparse=False, mem_free=False)
    assert_almost_equal(mutual_info, clt._mutual_info)

    clt = CLTree(data, alpha=alpha, sparse=True, mem_free=False)
    assert_almost_equal(mutual_info, clt._mutual_info)

#
# this test is completely useless now, commenting it
# def test_mst():
#     #
#     # creating as matrix the mi matrix from the cltree
#     clt = CLTree(data, alpha=0.0, copy_mi=copy_mi)
#     mst = minimum_spanning_tree(clt._mutual_info.copy())
#     print(mst)

#     visited_vertices = numpy.zeros(n_features, dtype=bool)
#     cltmi = clt._mutual_info.copy()
#     spanning_edges = numpy.zeros((n_features - 1, 2), dtype=int)
#     diag_indices = numpy.arange(n_features)
#     mst_1 = minimum_spanning_tree_numba(cltmi,
#                                         visited_vertices,
#                                         spanning_edges,
#                                         diag_indices)
#     print(mst_1)

#     #
#     # random big value
#     n_m_features = 1000
#     random_W = numpy.random.random((n_m_features, n_m_features))
#     visited_vertices = numpy.zeros(n_m_features, dtype=bool)
#     spanning_edges = numpy.zeros((n_m_features - 1, 2), dtype=int)
#     diag_indices = numpy.arange(n_m_features)

#     print(random_W)
#     mst_start_t = perf_counter()
#     mst_1 = minimum_spanning_tree(random_W.copy())
#     mst_end_t = perf_counter()
#     print('Classical done in', mst_end_t - mst_start_t, 'secs')

#     mst_start_t = perf_counter()
#     mst_2 = minimum_spanning_tree_numba(random_W.copy(),
#                                         visited_vertices,
#                                         spanning_edges,
#                                         diag_indices)
#     mst_end_t = perf_counter()
#     spanning_edges = numpy.zeros((n_m_features - 1, 2), dtype=int)
#     visited_vertices = numpy.zeros(n_m_features, dtype=bool)

#     mst_start_t = perf_counter()
#     mst_2 = minimum_spanning_tree_numba(random_W.copy(),
#                                         visited_vertices,
#                                         spanning_edges,
#                                         diag_indices)
#     mst_end_t = perf_counter()
#     print('Numba done in', mst_end_t - mst_start_t, 'secs')

#     assert_array_equal(mst_1, mst_2)


def test_eval_instance():
    #
    # comparing against values taken from Nico's code
    nico_cltree_tree = numpy.array([-1,  2,  0,  2])
    nico_cltree_tree[0] = 0
    nico_cltree_lls = numpy.array([-2.01490302054,
                                   -1.20397280433,
                                   -1.20397280433,
                                   -1.79175946923,
                                   -1.60943791243,
                                   -1.60943791243])

    nico_cltree_subtree = numpy.array([-1,  0,  1])
    nico_cltree_subtree[0] = 0
    nico_cltree_sublls = numpy.array([-1.09861228867,
                                      -0.69314718056,
                                      -0.69314718056,
                                      -1.79175946923,
                                      -1.09861228867,
                                      -0.69314718056])
    #
    # growing the tree on data
    clt = CLTree(data, alpha=0.0, sparse=True, mem_free=False)
    print(clt)

    # assert_array_equal(nico_cltree_tree,
    #                    clt._tree)
    for i, instance in enumerate(data):
        ll = clt.eval(instance)
        ll_f = clt.eval_fact(instance)
        # assert_almost_equal(nico_cltree_lls[i], ll)
        assert_almost_equal(ll, ll_f)
        print(ll, nico_cltree_lls[i])

    #
    # now by obscuring one column
    subdata = data[:, [0, 2, 3]]
    subclt = CLTree(subdata,
                    features=numpy.array([0, 2, 3]),
                    alpha=0.0, sparse=True, mem_free=False)
    print(subclt)

    # assert_array_equal(nico_cltree_subtree,
    #                    subclt._tree)
    for i, instance in enumerate(data):
        ll = subclt.eval(instance)
        ll_f = subclt.eval_fact(instance)
        assert_almost_equal(ll, ll_f)
        print(ll, nico_cltree_sublls[i])

from spn import MARG_IND


def test_instantiate_factors():
    # 0 -> 1
    # 1 -> 2
    # 1 -> 3
    # 3 -> 4
    tree = numpy.array([-1, 0, 1, 1, 3])
    #
    # all binary features
    feature_vals = [2, 2, 2, 2, 2]
    #
    # B = 0, D = 1 : 1 = 0, 3 = 1
    evidence = [MARG_IND, 0, MARG_IND, 1, MARG_IND]

    #
    # an array of arrays
    factors = [numpy.array([0.1, 0.9]),
               numpy.array([[0.2, 0.45], [0.8, 0.55]]),
               numpy.array([[0.5, 0.9], [0.5, 0.1]]),
               numpy.array([[0.2, 0.3], [0.8, 0.7]]),
               numpy.array([[0.7, 0.6], [0.3, 0.4]])]

    ev_factors = [factor for factor in factors]

    ev_factors = instantiate_factors(tree,
                                     feature_vals,
                                     evidence,
                                     factors,
                                     ev_factors)

    print(ev_factors)


def test_marginalize():

    n_features = 5

    tree = numpy.array([-1, 0, 1, 1, 3])
    # 0 -> 1
    # 1 -> 2
    # 1 -> 3
    # 3 -> 4
    tree = numpy.array([-1, 0, 1, 1, 3])
    #
    # all binary features
    feature_vals = [2, 2, 2, 2, 2]
    #
    # B = 0, D = 1 : 1 = 0, 3 = 1
    evidence = numpy.array([MARG_IND, 0, MARG_IND, 1, MARG_IND])

    factors = [numpy.array([0.1, 0.9]),
               numpy.array([[0.2, 0.45], [0.8, 0.55]]),
               numpy.array([[0.5, 0.9], [0.5, 0.1]]),
               numpy.array([[0.2, 0.3], [0.8, 0.7]]),
               numpy.array([[0.7, 0.6], [0.3, 0.4]])]
    print(factors)

    #
    # an array of arrays
    factors = [numpy.log(numpy.array([0.1, 0.9])),
               numpy.log(numpy.array([[0.2, 0.45], [0.8, 0.55]])),
               numpy.log(numpy.array([[0.5, 0.9], [0.5, 0.1]])),
               numpy.log(numpy.array([[0.2, 0.3], [0.8, 0.7]])),
               numpy.log(numpy.array([[0.7, 0.6], [0.3, 0.4]]))]

    factor_matrix = numpy.zeros((n_features, n_features), dtype=bool)
    ev_factors = [factor for factor in factors]

    ev_factors = instantiate_factors(tree,
                                     feature_vals,
                                     evidence,
                                     factors,
                                     ev_factors)

    print(ev_factors)
    features = numpy.arange(n_features)

    factor_matrix = tree_2_factor_matrix(tree,
                                         factor_matrix)
    print(factor_matrix)

    print(evidence)
    prob = marginalize(features,
                       feature_vals,
                       ev_factors,
                       factor_matrix,
                       tree,
                       evidence)
    print(prob)
