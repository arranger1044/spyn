import numpy

from cltree.probs import numba_cumsum
from cltree.probs import scope_union
from cltree.probs import compute_factor_stride
from cltree.probs import n_factor_features
from cltree.probs import compute_factor_product
from cltree.probs import factor_product

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


def test_numba_cumsum():
    array = numpy.array([1, 1, 2, 6, 0, 1])
    array_cum_sum = array.cumsum()
    numba_cumsum(array)
    assert_array_equal(array_cum_sum, array)


def test_scope_union():
    n = 10
    scope_1 = numpy.zeros(n, dtype=bool)
    scope_1[:n // 2] = True
    scope_2 = numpy.zeros(n, dtype=bool)
    scope_2[n // 2:] = True
    full_scope = numpy.ones(n, dtype=bool)
    assert_array_equal(full_scope, scope_union(scope_1, scope_2))

    print(scope_1, scope_2)

    scope_1[- 1] = True
    scope_2[0] = True
    assert_array_equal(full_scope, scope_union(scope_1, scope_2))


def test_compute_factor_stride():
    n_features = 6
    feature_vals = numpy.array([2, 2, 2, 3, 2, 3])
    factor_scope = numpy.array([1, 0, 1, 1, 0, 0], dtype=bool)
    prod_scope = numpy.array([1, 1, 1, 1, 0, 1], dtype=bool)
    print(prod_scope)
    n_prod_features = n_factor_features(prod_scope)
    factor_stride = numpy.zeros(n_features, dtype=int)

    factor_stride = compute_factor_stride(feature_vals,
                                          factor_scope,
                                          prod_scope,
                                          factor_stride)
    print(factor_stride)
    assert factor_stride.shape[0] == n_prod_features

    prod_stride = numpy.array([1, 0, 2, 4, 0])
    assert_array_equal(prod_stride, factor_stride)


def test_compute_factor_product():
    factor_1 = numpy.log(numpy.array([0.5, 0.5, 0.2, 0.8]))
    factor_2 = numpy.log(numpy.array([0.7, 0.3]))
    prod_factor = numpy.zeros(4)
    assignment = numpy.zeros(2)
    feature_vals = numpy.array([2, 2])
    factor_stride_1 = numpy.array([1, 2])
    factor_stride_2 = numpy.array([1, 0])

    prod_factor = compute_factor_product(factor_1,
                                         factor_2,
                                         prod_factor,
                                         assignment,
                                         feature_vals,
                                         factor_stride_1,
                                         factor_stride_2)

    print(prod_factor)
    prob_prod_factor = numpy.log(numpy.array([0.35, 0.15, 0.14, 0.24]))
    assert_array_almost_equal(prob_prod_factor, prod_factor)

    #
    # now on different features
    factor_1 = numpy.log(numpy.array([0.5, 0.5, 0.2, 0.8]))
    factor_2 = numpy.log(numpy.array([0.3, 0.7, 0.2, 0.8]))
    prod_factor = numpy.zeros(8)
    assignment = numpy.zeros(3)
    feature_vals = numpy.array([2, 2, 2])
    factor_stride_1 = numpy.array([1, 2, 0])
    factor_stride_2 = numpy.array([0, 1, 2])

    prod_factor = compute_factor_product(factor_1,
                                         factor_2,
                                         prod_factor,
                                         assignment,
                                         feature_vals,
                                         factor_stride_1,
                                         factor_stride_2)

    print(prod_factor)
    prob_prod_factor = numpy.log(numpy.array([0.5 * 0.3, 0.5 * 0.3,
                                              0.2 * 0.7, 0.8 * 0.7,
                                              0.5 * 0.2, 0.5 * 0.2,
                                              0.2 * 0.8, 0.8 * 0.8]))

    print(prob_prod_factor)
    assert_array_almost_equal(prob_prod_factor, prod_factor)


def test_factor_product():
    factor_1 = numpy.log(numpy.array([0.5, 0.5, 0.2, 0.8], dtype='float32'))
    factor_2 = numpy.log(numpy.array([0.3, 0.7, 0.2, 0.8], dtype='float32'))
    factor_scope_1 = numpy.array([1, 1, 0, 0], dtype=bool)
    factor_scope_2 = numpy.array([0, 1, 0, 1], dtype=bool)
    feature_vals = numpy.array([2, 2, 2, 2])

    prod_factor, prod_factor_scope = factor_product(factor_1,
                                                    factor_2,
                                                    factor_scope_1,
                                                    factor_scope_2,
                                                    feature_vals)
    print(prod_factor)
    print(prod_factor_scope)
    prob_prod_factor = numpy.log(numpy.array([0.5 * 0.3, 0.5 * 0.3,
                                              0.2 * 0.7, 0.8 * 0.7,
                                              0.5 * 0.2, 0.5 * 0.2,
                                              0.2 * 0.8, 0.8 * 0.8]))
    assert_array_almost_equal(prob_prod_factor, prod_factor)
