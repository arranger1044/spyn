from math import log
from spn import MARG_IND
from spn import LOG_ZERO
from spn import IS_LOG_ZERO

# a constant for float comparison
# it specifies the number of digits that are meaningful
PRECISION = 15

from nose.tools import assert_almost_equal

import numpy


def logify(ndarray):
    """
    routine to make the log of a numpy array
    and substituting -inf with LOG_ZERO
    """
    numpy.log(ndarray, out=ndarray)
    ndarray[numpy.isinf(ndarray)] = LOG_ZERO


def compute_smoothed_ll(obs, freqs, values, alpha):
    """
    WRITEME
    """
    ll = None
    if obs == MARG_IND:
        ll = 0.
    else:
        obs_freq = freqs[obs] + \
            alpha if freqs is not None else 1 + alpha
        log_freq = log(obs_freq) if obs_freq > 0. else LOG_ZERO
        sum_freq = sum(freqs) + alpha * \
            values if freqs is not None else values * (alpha + 1)
        log_tot = log(sum_freq)
        ll = log_freq - log_tot
    return ll


def flatten_list(list):
    """
    works only on matrices at most
    """
    return [item for sublist in list for item in sublist]


def assert_log_array_almost_equal(array_1, array_2):
    """
    WRITEME
    """
    # if it is a numpy array, then flatten it
    try:
        array_1 = array_1.flatten()
    except:
        try:
            array_1 = flatten_list(array_1)
        except:
            pass
    try:
        array_2 = array_2.flatten()
    except:
        try:
            array_2 = flatten_list(array_2)
        except:
            pass

    for elem_1, elem_2 in zip(array_1, array_2):
        if elem_1 == LOG_ZERO:
            # -2000 == -1000 since exp(-2000) == exp(-1000)
            assert IS_LOG_ZERO(elem_2) is True
        else:
            assert_almost_equal(elem_1, elem_2, PRECISION)
