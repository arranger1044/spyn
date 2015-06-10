import numpy
import numba


@numba.jit
def scope_union(factor_scope_1, factor_scope_2):
    """
    A factor scope is a numpy boolean array
    """
    return factor_scope_1 + factor_scope_2


# @numba.njit
def factor_length(factor_scope, feature_vals):
    """
    WRITEME
    """
    return feature_vals[factor_scope].prod()


# @numba.jit
def n_factor_features(factor_scope):
    """
    WRITEME
    """
    return factor_scope.sum()


@numba.njit
def numba_cumsum(array):
    """
    """
    for i in range(1, array.shape[0]):
        array[i] += array[i - 1]


@numba.jit
def compute_factor_stride(feature_vals,
                          factor_scope,
                          prod_scope,
                          factor_stride):
    """
    WRITEME
    """
    n_features = feature_vals.shape[0]
    #
    # copying in stride the feature vals for the vars in scope
    for i in range(n_features):
        if factor_scope[i]:
            factor_stride[i] = feature_vals[i] - 1

    #
    # computing the strides
    numba_cumsum(factor_stride)
    #
    # masking with the prod scope
    factor_stride[~factor_scope] = 0
    return factor_stride[prod_scope]


@numba.njit
def compute_factor_product(factor_1,
                           factor_2,
                           prod_factor,
                           assignment,
                           feature_vals,
                           factor_stride_1,
                           factor_stride_2):
    """
    WRITEME
    """

    j = 0
    k = 0

    factor_length = prod_factor.shape[0]
    n_features = assignment.shape[0]

    for i in range(factor_length):
        #
        # operating in the log domain
        prod_factor[i] = factor_1[j] + factor_2[k]
        for l in range(n_features):
            assignment[l] += 1
            if assignment[l] == feature_vals[l]:
                assignment[l] = 0
                l_feature_val = feature_vals[l] - 1
                j -= (l_feature_val * factor_stride_1[l])
                k -= (l_feature_val * factor_stride_2[l])
            else:
                j += factor_stride_1[l]
                k += factor_stride_2[l]
                break

    return prod_factor


@numba.jit
def factor_product(factor_1,
                   factor_2,
                   factor_scope_1,
                   factor_scope_2,
                   feature_vals):
    """
    WRITEME
    """
    n_features = feature_vals.shape[0]
    #
    # getting the scope
    prod_factor_scope = scope_union(factor_scope_1, factor_scope_2)

    #
    # preallocating
    prod_factor_length = factor_length(prod_factor_scope, feature_vals)
    prod_factor = numpy.zeros(prod_factor_length, dtype=factor_1.dtype)
    n_prod_features = n_factor_features(prod_factor_scope)
    assignment = numpy.zeros(n_prod_features, dtype=factor_1.dtype)

    #
    # computing strides
    factor_stride_1 = numpy.zeros(n_features, dtype=numpy.uint16)
    factor_stride_2 = numpy.zeros(n_features, dtype=numpy.uint16)
    factor_stride_1 = compute_factor_stride(feature_vals,
                                            factor_scope_1,
                                            prod_factor_scope,
                                            factor_stride_1)
    factor_stride_2 = compute_factor_stride(feature_vals,
                                            factor_scope_2,
                                            prod_factor_scope,
                                            factor_stride_2)

    #
    # computing the actual product
    prod_feature_vals = feature_vals[prod_factor_scope]
    prod_factor = compute_factor_product(factor_1,
                                         factor_2,
                                         prod_factor,
                                         assignment,
                                         prod_feature_vals,
                                         factor_stride_1,
                                         factor_stride_2)
    return prod_factor, prod_factor_scope
