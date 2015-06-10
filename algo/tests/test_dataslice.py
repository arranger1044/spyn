from algo.dataslice import DataSlice

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import numpy

import logging


def test_whole_slice():
    n_cols = 15
    n_rows = 10
    data_slice = DataSlice.whole_slice(n_rows, n_cols)

    assert data_slice.id == 0
    row_ids_t = data_slice.instance_ids == [i for i in range(n_rows)]
    print(data_slice.instance_ids, row_ids_t, numpy.all(row_ids_t))
    assert numpy.all(row_ids_t)
    assert (data_slice.feature_ids == [i for i in range(n_cols)]).all()
