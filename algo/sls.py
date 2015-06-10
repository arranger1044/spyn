import numpy

import numba

from scipy.misc import logsumexp

import sys

import itertools

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from spn import MARG_IND
from spn import LOG_ZERO

from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode

from spn.factory import SpnFactory

from collections import deque

from algo.dataslice import DataSlice

import math

NEG_INF = -sys.float_info.max

import gc
GC_COLLECT = 5000

FLOAT_TYPE = numpy.float32

#
# logging
#
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s -'
#                               ' %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(__name__)


# @numba.jit
class SlsStats(object):

    """
    A place to save the search statistics
    """
    # @numba.void

    def __init__(self):
        self.n_row_op = 0
        self.n_col_then_row_op = 0
        self.n_row_splits = 0
        self.n_col_splits = 0

    # @numba.void
    def inc_row_split(self):
        self.n_row_op += 1
        self.n_row_splits += 1

    def inc_col_split(self):
        self.n_col_splits += 1

    def inc_row_col_split(self, bisplit):
        self.n_col_then_row_op += 1
        self.n_row_splits += 1
        if bisplit:
            self.n_row_splits += 1

    # @numba.jit
    def inc_col_row_split(self, bisplit):
        self.n_col_then_row_op += 1
        self.n_col_splits += 1
        self.n_row_splits += 1
        if bisplit:
            self.n_row_splits += 1

    def add_stats(self, stats):
        self.n_row_op += stats.n_row_op
        self.n_col_then_row_op += stats.n_col_then_row_op
        self.n_row_splits += stats.n_row_splits
        self.n_col_splits += stats.n_col_splits

    # @numba.void
    def __repr__(self):
        return ('# row op calls:{0}'
                '\n# col op calls:{1}'
                '\n# row splits:{2}'
                '\n# col splits:{3}'.format(self.n_row_op,
                                            self.n_col_then_row_op,
                                            self.n_row_splits,
                                            self.n_col_splits))


#
# SCORE FUNCTION
#


def estimate_counts_numpy(data,
                          instance_ids,
                          feature_ids):
    """
    WRITEME
    """
    #
    # slicing the data array (probably memory consuming)
    curr_data_slice = data[instance_ids, :][:, feature_ids]

    estimated_counts = []
    for feature_slice in curr_data_slice.T:
        counts = numpy.bincount(feature_slice)
        #
        # checking just for the all 0 case:
        # this is not stable for not binary datasets TODO: fix it
        if counts.shape[0] < 2:
            counts = numpy.append(counts, [0], 0)
        estimated_counts.append(counts)

    return estimated_counts


@numba.jit
def estimate_counts_numba_part(data,
                               instance_ids,
                               feature_ids,
                               feature_vals,
                               estimated_counts):
    """
    this one assumes that estimated_counts is a numpy 2D array
    (features are homoegeneous)
    in this version estimated_counts has the same length as feature_ids
    """

    #
    # actual counting
    for i, feature_id in enumerate(feature_ids):
        for instance_id in instance_ids:
            estimated_counts[i, data[instance_id, feature_id]] += 1

    return estimated_counts


@numba.jit
def estimate_counts_numba(data,
                          instance_ids,
                          feature_ids,
                          feature_vals,
                          estimated_counts):
    """
    this one assumes that estimated_counts is a numpy 2D array
    (features are homoegeneous)
    """

    #
    # zeroing entries
    for feature_id in feature_ids:
        for val in range(feature_vals[feature_id]):
            estimated_counts[feature_id, val] = 0

    #
    # actual counting
    for feature_id in feature_ids:
        for instance_id in instance_ids:
            estimated_counts[feature_id, data[instance_id, feature_id]] += 1

    return estimated_counts


@numba.jit
def evaluate_instance_ll_numba(ll_frequencies,
                               feature_ids,
                               instance_values):
    """
    WRITEME
    """
    ll = 0
    for i, feature in enumerate(feature_ids):
        #
        # this control may be useless for this computation
        # since on those dataset there are no MARG_INDs
        value = instance_values[feature]
        if value != MARG_IND:
            ll += ll_frequencies[i, value]
    return ll


@numba.jit
def evaluate_dataset_ll_numba(ll_frequencies,
                              feature_ids,
                              dataset):
    """
    WRITEME
    """
    ll = 0.0
    for i in range(dataset.shape[0]):
        ll += evaluate_instance_ll_numba(ll_frequencies,
                                         feature_ids,
                                         dataset[i, :])
    return ll


@numba.jit
def evaluate_dataset_naive_ll(ll_frequencies,
                              feature_ids,
                              dataset):
    """
    This version is a little bit more efficient
    since does all together
    The name is misleading : )
    """
    # ll = 0.0
    n_instances = dataset.shape[0]
    lls = numpy.zeros(n_instances, dtype=FLOAT_TYPE)
    for i in range(n_instances):
        for j in feature_ids:
            # ll += ll_frequencies[j, dataset[i, j]]
            lls[i] += ll_frequencies[j, dataset[i, j]]
    return lls


@numba.jit
def smooth_ll_parameters(estimated_counts,
                         ll_frequencies,
                         instance_ids,
                         feature_ids,
                         feature_vals,
                         alpha):
    """
    WRITEME
    """
    tot_counts = len(instance_ids)
    for feature_id in feature_ids:
        feature_val = feature_vals[feature_id]
        smooth_tot_ll = math.log(tot_counts + feature_val * alpha)
        for val in range(feature_val):
            smooth_n = estimated_counts[feature_id, val] + alpha
            smooth_n_ll = math.log(smooth_n) if smooth_n > 0.0 else LOG_ZERO
            ll_frequencies[feature_id, val] = smooth_n_ll - smooth_tot_ll
    return ll_frequencies


@numba.jit
def estimate_data_slice_ll(data_slice,
                           estimated_counts,
                           ll_frequencies,
                           train_dataset,
                           feature_vals,
                           alpha,
                           valid_dataset):
    """
    WRITEME
    """
    #
    # all these memory allocations can be prevented by allocanting
    # just one big array before and filling it to zero each time
    # TODO: we could make a clojure to reduce paramters name
    #
    # counting
    estimated_counts = estimate_counts_numba(train_dataset,
                                             data_slice.instance_ids,
                                             data_slice.feature_ids,
                                             feature_vals,
                                             estimated_counts)
    # logger.debug('EST', estimated_counts)
    #
    # smoothing and computing ll parameters
    ll_frequencies = smooth_ll_parameters(estimated_counts,
                                          ll_frequencies,
                                          data_slice.instance_ids,
                                          data_slice.feature_ids,
                                          feature_vals,
                                          alpha)
    # logger.debug('LLFREQS', ll_frequencies)
    #
    # evaluating on the validation dataset
    lls_hat = evaluate_dataset_naive_ll(ll_frequencies,
                                        data_slice.feature_ids,
                                        valid_dataset)

    return lls_hat


@numba.jit
def estimate_row_split_ll(data_partition,
                          estimated_counts,
                          ll_frequencies,
                          train_dataset,
                          feature_vals,
                          alpha,
                          valid_dataset):
    """
    WRITEME
    """
    n_slices = len(data_partition)
    n_instances = valid_dataset.shape[0]
    ll_hats = numpy.zeros((n_instances, n_slices), dtype=FLOAT_TYPE)
    weights = numpy.zeros(n_slices, dtype=FLOAT_TYPE)

    for i, data_slice in enumerate(data_partition):
        weights[i] = data_slice.n_instances()
        ll_hats[:, i] = estimate_data_slice_ll(data_slice,
                                               estimated_counts,
                                               ll_frequencies,
                                               train_dataset,
                                               feature_vals,
                                               alpha,
                                               valid_dataset)
        #
        # storing the ll into the data slice
        data_slice.ll = numpy.sum(ll_hats[:, i]) / n_instances
        data_slice.lls = ll_hats[:, i]

    # damn numba, this is totally anti-pythonic
    tot_weights = numpy.sum(weights)
    for i in range(n_slices):
        weights[i] /= tot_weights
    # doing the log
    log_weights = numpy.log(weights, out=weights)
    # summing
    ll_hat = logsumexp(ll_hats + log_weights[numpy.newaxis, :], axis=1)

    # print('weights', weights)
    # w_ll_hats = numpy.log(weights) + ll_hats
    # ll_hat = logsumexp(w_ll_hats)

    return ll_hat


@numba.jit
def estimate_row_slice_lls(data_slice,
                           valid_dataset):
    """
    assuming data_slice is a sum slice (row partition)
    TODO: refactor estimate_row_split_ll and this to reduce the code
    """
    n_children = len(data_slice.children)
    n_instances = valid_dataset.shape[0]
    ll_hats = numpy.zeros((n_instances, n_children), dtype=FLOAT_TYPE)
    weights = numpy.zeros(n_children, dtype=FLOAT_TYPE)

    for i, child in enumerate(data_slice.children):
        weights[i] = data_slice.weights[i]
        ll_hats[:, i] = child.lls

    tot_weights = numpy.sum(weights)
    for i in range(n_children):
        weights[i] /= tot_weights

    log_weights = numpy.log(weights, out=weights)
    ll_hat = logsumexp(ll_hats + log_weights[numpy.newaxis, :], axis=1)

    data_slice.lls = ll_hat


@numba.jit
def estimate_ll(data_partitions,
                estimated_counts,
                ll_frequencies,
                train_dataset,
                feature_vals,
                alpha,
                valid_dataset):
    """
    In this way I can abstract from the operator result type
    data_partitions can be a seq of seqs (partitions of partitions)
    and single data_slices
    ie. [D_0, [D_1, D_2]] (but they can be also tuples)
    """
    # print('DATA', data_partitions, type(data_partitions))

    n_instances = valid_dataset.shape[0]
    n_slices = len(data_partitions)
    ll_hats = numpy.zeros((n_instances, n_slices), dtype=FLOAT_TYPE)
    for j, partition in enumerate(data_partitions):
        #
        # is this a row split?
        # if getattr(partition, '__iter__', False):
        if len(partition) > 1:
            # print('ROW SPLIT', partition)
            ll_hats[:, j] += estimate_row_split_ll(partition,
                                                   estimated_counts,
                                                   ll_frequencies,
                                                   train_dataset,
                                                   feature_vals,
                                                   alpha,
                                                   valid_dataset)
        else:
            # print('SINGLE PARTITION')
            #
            # just one element, partition is a data_slice
            single_partition, = partition
            ll_hat_p = estimate_data_slice_ll(single_partition,
                                              estimated_counts,
                                              ll_frequencies,
                                              train_dataset,
                                              feature_vals,
                                              alpha,
                                              valid_dataset)
            #
            # storing
            single_partition.ll = numpy.sum(ll_hat_p) / n_instances
            single_partition.lls = ll_hat_p
            ll_hats[:, j] += ll_hat_p

    # return numpy.sum(ll_hats) / n_instances

    return numpy.sum(ll_hats, axis=1)


@numba.jit
def propagate_ll(data_slice,
                 lls):
    """
    WRITEME
    """
    curr_slice = data_slice
    parent_slice = curr_slice.parent
    new_lls = lls
    lls_path = [lls]
    # logger.info('propagate')
    # print(data_slice)
    # print(parent_slice)
    # print(lls)

    #
    # curse you numba the right condition shall be:
    # while parent_slice is not None:
    while parent_slice:
        old_lls = curr_slice.lls
        # print('old', curr_slice.lls, parent_slice.lls)
        if parent_slice.type == ProductNode:
            # print('PROD', parent_slice, parent_slice.type)
            new_lls = parent_slice.lls - old_lls + new_lls
        elif parent_slice.type == SumNode:
            new_lls = numpy.log(numpy.exp(parent_slice.lls) + curr_slice.w *
                                (numpy.exp(new_lls) - numpy.exp(old_lls)))
            # print('EXP', new_lls, parent_slice.lls, old_lls, new_lls)
        curr_slice = parent_slice
        parent_slice = curr_slice.parent
        lls_path.append(new_lls)
    return new_lls, lls_path


@numba.jit
def update_best_lls(data_slice,
                    best_lls_path):
    """
    WRITEME
    """
    curr_slice = data_slice
    #
    # here I am assuming that the length of the path is correct
    for slice_lls in best_lls_path:
        curr_slice.lls = slice_lls
        # curr_slice.ll = numpy.sum(slice_lls) / valid_dataset.shape[0]
        curr_slice = curr_slice.parent


#
# NEIGHBORHOOD OPERATORS
#


@numba.jit
def random_binary_row_split(data_slice,
                            rand_gen):
    """
    WRITEME
    """
    #
    # is it necessary to make a copy?
    rand_gen.shuffle(data_slice.instance_ids)
    n_instances = len(data_slice.instance_ids)
    #
    # uniform split
    # split_k = rand_gen.randint(0, n_instances - 1)
    split_k = rand_gen.randint(1, n_instances)

    #
    # generate two new slices
    instance_ids_1 = numpy.copy(data_slice.instance_ids[:split_k])
    instance_ids_2 = numpy.copy(data_slice.instance_ids[split_k:])

    data_slice_1 = DataSlice(instance_ids_1,
                             numpy.copy(data_slice.feature_ids))
    data_slice_2 = DataSlice(instance_ids_2,
                             numpy.copy(data_slice.feature_ids))

    return ((data_slice_1, data_slice_2),)


@numba.jit
def random_binary_col_then_row_split(data_slice,
                                     rand_gen,
                                     bisplit):
    """
    WRITEME
    """
    rand_gen.shuffle(data_slice.feature_ids)
    n_features = len(data_slice.feature_ids)
    #
    # uniform split
    # split_k = rand_gen.randint(0, n_features - 1)
    split_k = rand_gen.randint(1, n_features)

    #
    # generate vertical slices
    feature_ids_1 = numpy.copy(data_slice.feature_ids[:split_k])
    feature_ids_2 = numpy.copy(data_slice.feature_ids[split_k:])

    data_slice_1 = DataSlice(numpy.copy(data_slice.instance_ids),
                             feature_ids_1)
    data_slice_2 = DataSlice(numpy.copy(data_slice.instance_ids),
                             feature_ids_2)

    #
    # checking whether or not to split both slices
    data_partitions = None

    if bisplit:
        row_partition_1, = random_binary_row_split(data_slice_1,
                                                   rand_gen)
        row_partition_2, = random_binary_row_split(data_slice_2,
                                                   rand_gen)
        data_partitions = (row_partition_1, row_partition_2)
    else:
        data_slice_to_split = None
        data_slice_to_preserve = None

        #
        # again, choosing which vertical to split horizontally is
        # a random choice
        # if rand_gen.random() > 0.5:
        if rand_gen.rand() > 0.5:
            data_slice_to_split = data_slice_1
            data_slice_to_preserve = data_slice_2
        else:
            data_slice_to_split = data_slice_2
            data_slice_to_preserve = data_slice_1

        row_partition, = random_binary_row_split(data_slice_to_split,
                                                 rand_gen)

        data_partitions = (row_partition, (data_slice_to_preserve,))

    # logger.info('partitions %r\n%r %r', data_partitions,
    #             feature_ids_1, feature_ids_2)
    return data_partitions


@numba.jit
def best_neighbor_naive(data_slice,
                        alpha_n,
                        beta,
                        estimated_counts,
                        ll_frequencies,
                        train_dataset,
                        feature_vals,
                        alpha,
                        valid_dataset,
                        bisplit,
                        propagate,
                        stats,
                        rand_gen):
    """
    This is the naive version, in which duplicate neighbors can be visited
    """
    best_ll = NEG_INF
    best_partition = None
    best_path = None

    opt_start_t = perf_counter()

    n_instances = data_slice.n_instances()
    n_features = data_slice.n_features()

    #
    # TODO here I could heavily reuse the code from
    # random neighbor but  I want to monitor only the time of splittin
    # nevertheless I HAVE to refactor this code

    #
    # if only one element, cannot partition it anymore
    if n_instances < 2 or n_features < 2:
        return best_partition, best_ll, best_path

    avg_split_t = 0.0

    #
    # generate at most alpha_n neighbors
    # I have no guarantee that two generated neighbors are different
    for i in range(alpha_n):
        #
        # which operator to use?
        # split = rand_gen.random()
        split = rand_gen.rand()

        partition = None
        if split < beta:
            split_start_t = perf_counter()
            partition = random_binary_row_split(data_slice,
                                                rand_gen)
            split_end_t = perf_counter()
            # logger.info('Split on rows %f', split_end_t - split_start_t)
            #
            # updating stats
            stats.inc_row_split()
        else:
            split_start_t = perf_counter()
            partition = random_binary_col_then_row_split(data_slice,
                                                         rand_gen,
                                                         bisplit)
            split_end_t = perf_counter()
            # logger.info('Split on cols %f', split_end_t - split_start_t)

            stats.inc_col_row_split(bisplit)

        avg_split_t += split_end_t - split_start_t
        sys.stdout.write('\ravg time {0} secs [{1}/{2}]'.
                         format(avg_split_t / (i + 1),
                                i + 1, alpha_n))
        sys.stdout.flush()
        #
        # how good is it?
        partition_lls = estimate_ll(partition,
                                    estimated_counts,
                                    ll_frequencies,
                                    train_dataset,
                                    feature_vals,
                                    alpha,
                                    valid_dataset)

        partition_ll = None
        lls_path = None
        if propagate:
            #
            # propagate it to obtain the global ll
            global_lls, lls_path = propagate_ll(data_slice,
                                                partition_lls)

            partition_ll = numpy.sum(global_lls) / valid_dataset.shape[0]
        else:
            partition_ll = numpy.sum(partition_lls) / valid_dataset.shape[0]

        # logger.info('Partition ll %f', partition_ll)

        #
        # any improvement?
        if partition_ll > best_ll:
            logger.info('\nFOUND NEW BEST LL --> %f', partition_ll)
            best_ll = partition_ll
            best_partition = partition
            best_path = lls_path

    #
    # found a split improving the original slice ll?
    # found_partition = None
    # if best_ll > data_slice.ll:
    #     logger.info('\nFINAL BEST LL %f', best_ll)
    #     found_partition = best_partition
    # else:
    #     logger.info('NO IMPROVEMENT')

    # return found_partition
    opt_end_t = perf_counter()
    logger.info('\nFINAL BEST LL %f [in %f secs] after %d checks',
                best_ll,
                (opt_end_t - opt_start_t),
                i)

    return best_partition, best_ll, best_path


@numba.jit
def random_neighbor(data_slice,
                    beta,
                    estimated_counts,
                    ll_frequencies,
                    train_dataset,
                    feature_vals,
                    alpha,
                    valid_dataset,
                    bisplit,
                    propagate,
                    stats,
                    rand_gen):
    """
    WRITEME
    """

    n_instances = data_slice.n_instances()
    n_features = data_slice.n_features()

    partition = None
    partition_ll = None
    path = None
    #
    # if only one element, cannot partition it anymore
    if n_instances < 2 or n_features < 2:
        return partition, partition_ll, path

    # avg_split_t = 0.0

    #
    # which operator to use?
    # split = rand_gen.random()
    split = rand_gen.rand()

    partition = None
    if split < beta:
        # split_start_t = perf_counter()
        partition = random_binary_row_split(data_slice,
                                            rand_gen)
        # split_end_t = perf_counter()
        # logger.info('Split on rows %f', split_end_t - split_start_t)
        #
        # updating stats
        stats.inc_row_split()
    else:
        # split_start_t = perf_counter()
        partition = random_binary_col_then_row_split(data_slice,
                                                     rand_gen,
                                                     bisplit)
        # split_end_t = perf_counter()
        # logger.info('Split on cols %f', split_end_t - split_start_t)

        stats.inc_col_row_split(bisplit)

    # avg_split_t += split_end_t - split_start_t
    # sys.stdout.write('\ravg time {0} secs'.
    #                  format(avg_split_t / (i + 1)))
    # sys.stdout.flush()
    #
    # how good is it?
    partition_lls = estimate_ll(partition,
                                estimated_counts,
                                ll_frequencies,
                                train_dataset,
                                feature_vals,
                                alpha,
                                valid_dataset)
    partition_ll = None
    if propagate:
        global_lls, path = propagate_ll(data_slice, partition_lls)
        partition_ll = numpy.sum(global_lls) / valid_dataset.shape[0]
    else:
        partition_ll = numpy.sum(partition_lls) / valid_dataset.shape[0]

    return partition, partition_ll, path


def k_partitions(partitions, m_partitions, n_objects, k):
    """
    WRITEME
    """
    #
    # resetting the values
    partitions.fill(0)
    m_partitions.fill(0)

    #
    # generating the first partition
    for i in range(n_objects - k + 1, n_objects):
        partitions[i] = i - (n_objects - k)
        m_partitions[i] = i - (n_objects - k)
    # partitions[n_objects - 1] = k
    # yield partitions

    #
    # now the next ones
    stop = False
    while not stop:
        more = False
        for i in range(n_objects - 1, 0, -1):
            if partitions[i] < k - 1 and partitions[i] <= m_partitions[i - 1]:
                partitions[i] += 1
                if m_partitions[i] < partitions[i]:
                    m_partitions[i] = partitions[i]
                for j in range(i + 1, n_objects - (k - m_partitions[i]) + 1):
                    partitions[j] = 0
                    m_partitions[j] = m_partitions[i]
                for j in range(n_objects - (k - m_partitions[i]) + 1,
                               n_objects):
                    partitions[j] = k - (n_objects - j)
                    m_partitions[j] = k - (n_objects - j)
                # yield partitions
                more = True
                break
        if not more:
            stop = True

#
# numba does not allow for python generators
# (this code is everything but pythonic lol)
# so I need to have to function to call on demand: the first one generates
# the first partition in lexycographical order, the second the remaining
# ones (when there are no more it return False)
#


@numba.jit
def first_k_partition(partitions, m_partitions, n_objects, k):
    """
    partitions and m_partitions are buffer arrays in which to write
    partitions will contain the partitioning
    """
    #
    # resetting the values
    partitions.fill(0)
    m_partitions.fill(0)

    #
    # generating the first partition
    for i in range(n_objects - k + 1, n_objects):
        partitions[i] = i - (n_objects - k)
        m_partitions[i] = i - (n_objects - k)


@numba.jit
def next_k_partition(partitions, m_partitions, n_objects, k):
    """
    Returns True whether there is a next partition to generate, False otherwise
    """
    for i in range(n_objects - 1, 0, -1):
        if partitions[i] < k - 1 and partitions[i] <= m_partitions[i - 1]:
            partitions[i] += 1
            if m_partitions[i] < partitions[i]:
                m_partitions[i] = partitions[i]
            for j in range(i + 1, n_objects - (k - m_partitions[i]) + 1):
                partitions[j] = 0
                m_partitions[j] = m_partitions[i]
            for j in range(n_objects - (k - m_partitions[i]) + 1,
                           n_objects):
                partitions[j] = k - (n_objects - j)
                m_partitions[j] = k - (n_objects - j)
            return True
    return False


@numba.jit
def first_bin_partition(partitions, m_partitions, n_objects):
    """
    With just two clusters I can use boolean arrays
    (more efficient)
    """
    #
    # resetting the values

    partitions.fill(False)
    m_partitions.fill(False)

    #
    # generating the first partition
    for i in range(n_objects - 2 + 1, n_objects):
        partitions[i] = True
        m_partitions[i] = True


@numba.jit
def next_bin_partition(partitions, m_partitions, n_objects):
    """
    Returns True whether there is a next partition to generate, False otherwise
    """
    for i in range(n_objects - 1, 0, -1):
        if partitions[i] < 2 - 1 and partitions[i] <= m_partitions[i - 1]:
            partitions[i] = True
            if m_partitions[i] < partitions[i]:
                m_partitions[i] = partitions[i]
            for j in range(i + 1, n_objects - (2 - m_partitions[i]) + 1):
                partitions[j] = False
                m_partitions[j] = m_partitions[i]
            for j in range(n_objects - (2 - m_partitions[i]) + 1,
                           n_objects):
                partitions[j] = True
                m_partitions[j] = True
            return True
    return False


@numba.jit
def binary_row_split_by_partition(data_slice,
                                  partition):
    """
    here I am assuming that partition is a numpy array of same length
    as data_slice.instance_ids
    """
    #
    # here I am assuming partition to be a boolean array
    instance_ids_1 = data_slice.instance_ids[partition]
    instance_ids_2 = data_slice.instance_ids[~partition]

    data_slice_1 = DataSlice(instance_ids_1,
                             numpy.copy(data_slice.feature_ids))
    data_slice_2 = DataSlice(instance_ids_2,
                             numpy.copy(data_slice.feature_ids))
    return ((data_slice_1, data_slice_2),)


@numba.jit
def binary_col_then_row_split_by_partition(data_slice,
                                           col_partition,
                                           row_partition,
                                           instance_ids_1,
                                           instance_ids_2,
                                           rand_gen,
                                           bisplit):
    """
    This one takes one or more row partitions in addition to the col one
    """
    #
    # splitting on cols, assuming boolean partitions
    feature_ids_1 = data_slice.feature_ids[col_partition]
    feature_ids_2 = data_slice.feature_ids[~col_partition]

    #
    # instance_ids_X are just permutations of a same set
    data_slice_1 = DataSlice(instance_ids_1, feature_ids_1)
    data_slice_2 = DataSlice(instance_ids_2, feature_ids_2)

    #
    # checking whether or not to split both slices
    data_partitions = None

    if bisplit:

        row_partition_1, = binary_row_split_by_partition(data_slice_1,
                                                         row_partition)
        row_partition_2, = binary_row_split_by_partition(data_slice_2,
                                                         row_partition)
        data_partitions = (row_partition_1, row_partition_2)
    else:

        data_slice_to_split = None
        data_slice_to_preserve = None

        #
        # again, choosing which vertical to split horizontally is
        # a random choice
        # if rand_gen.random() > 0.5:
        if rand_gen.rand() > 0.5:
            data_slice_to_split = data_slice_1
            data_slice_to_preserve = data_slice_2
        else:
            data_slice_to_split = data_slice_2
            data_slice_to_preserve = data_slice_1

        row_partition, = binary_row_split_by_partition(data_slice_to_split,
                                                       row_partition)

        data_partitions = (row_partition, (data_slice_to_preserve,))

    return data_partitions


@numba.jit
def best_neighbor_enum(data_slice,
                       alpha_n,
                       beta,
                       estimated_counts,
                       ll_frequencies,
                       train_dataset,
                       feature_vals,
                       alpha,
                       valid_dataset,
                       bisplit,
                       propagate,
                       stats,
                       rand_gen):
    """
    This is the enumerative version, in which no two same neighbors
    can be generated
    """
    best_ll = NEG_INF
    best_partition = None
    best_path = None

    n_instances = data_slice.n_instances()
    n_features = data_slice.n_features()

    #
    # if only one element, cannot partition it anymore
    if n_instances < 2 or n_features < 2:
        return best_partition, best_ll, best_path

    #
    # allocating the buffers for the first partitions, assuming boolean arrays
    row_partitions = numpy.zeros(n_instances, dtype=bool)
    m_row_partitions = numpy.zeros(n_instances, dtype=bool)
    col_partitions = numpy.zeros(n_features, dtype=bool)
    m_col_partitions = numpy.zeros(n_features, dtype=bool)

    #
    # shuffling the data_slice instances and rows
    rand_gen.shuffle(data_slice.instance_ids)
    rand_gen.shuffle(data_slice.feature_ids)

    #
    # first partitions boolean
    first_row_partition = True
    first_col_partition = True

    #
    #
    FIRST_COL_GEN = 0
    NEXT_COL_GEN = 1
    FIRST_COL_ROW_GEN = 2
    NEXT_COL_ROW_GEN = 3
    WAIT_GEN = 4
    col_state = FIRST_COL_GEN
    col_row_state = FIRST_COL_ROW_GEN
    #
    # generate at most alpha_n neighbors
    # if there are no more partition to generate, exit
    can_split_rows = True
    can_split_cols = True

    avg_split_t = 0.0
    opt_start_t = perf_counter()

    i = 0
    j = 0

    #
    # TODO: this is just a stub, it needs to be defined better
    alpha_m = max(n_instances, int(alpha_n * 0.001))

    split_rows = False
    split_cols = False

    row_col_partitions = numpy.zeros(n_instances, dtype=bool)
    m_row_col_partitions = numpy.zeros(n_instances, dtype=bool)
    instance_ids_1 = None
    instance_ids_2 = None

    while i < alpha_n and (can_split_cols or can_split_rows):
        #
        #

        #
        # which operator to use?
        if can_split_cols and can_split_rows:
            split = rand_gen.rand()
            # print('\nBETA', split, beta, can_split_rows, can_split_cols)
            if split < beta:
                split_rows = True
                split_cols = False
            else:
                split_rows = False
                split_cols = True
        elif can_split_rows:
            split_rows = True
            split_cols = False
        elif can_split_cols:
            split_rows = False
            split_cols = True

        partition = None

        # if split < beta and can_split_rows:
        # print('\nSPLIT', split_rows, split_cols)
        if split_rows:
            # print('\nSPLIT ROWS', i)

            split_start_t = perf_counter()
            if first_row_partition:
                # logger.debug('FIRST SPLIT')
                first_bin_partition(row_partitions,
                                    m_row_partitions,
                                    n_instances)
                first_row_partition = False
            else:

                #
                # generating a row partition
                can_split_rows = next_bin_partition(row_partitions,
                                                    m_row_partitions,
                                                    n_instances)
            if can_split_rows:
                # logger.debug('SPLIT')
                #
                # splitting it
                partition = binary_row_split_by_partition(data_slice,
                                                          row_partitions)

                split_end_t = perf_counter()
                # logger.debug('EVAL')
                #
                # evaluate the split
                partition_lls = estimate_ll(partition,
                                            estimated_counts,
                                            ll_frequencies,
                                            train_dataset,
                                            feature_vals,
                                            alpha,
                                            valid_dataset)

                partition_ll = None
                lls_path = None
                if propagate:
                    global_lls, lls_path = propagate_ll(data_slice,
                                                        partition_lls)
                    partition_ll = (numpy.sum(global_lls) /
                                    valid_dataset.shape[0])
                else:
                    partition_ll = (numpy.sum(partition_lls) /
                                    valid_dataset.shape[0])

                avg_split_t += split_end_t - split_start_t
                sys.stdout.write('\ravg time {0:.8f} secs [{1}/{2}] {3:.5f}'.
                                 format(avg_split_t / (i + 1),
                                        i + 1,
                                        alpha_n,
                                        partition_ll))
                sys.stdout.flush()

                #
                # any improvement? if yes, store it
                if partition_ll > best_ll:
                    logger.info('\nFOUND NEW BEST LL --> %f', partition_ll)
                    best_ll = partition_ll
                    best_partition = partition
                    best_path = lls_path

                i += 1
                stats.inc_row_split()
            # else:
            #     print('\nNO MORE ROWS', i)

        # elif can_split_cols:  # andsplit >= beta
        elif split_cols:
            # print('\nSPLIT COLS', i)

            split_start_t = perf_counter()

            if col_state == FIRST_COL_GEN:
                split_start_t = perf_counter()

                first_bin_partition(col_partitions,
                                    m_col_partitions,
                                    n_features)

                split_end_t = perf_counter()
                avg_split_t += split_end_t - split_start_t

                col_state = WAIT_GEN
                col_row_state = FIRST_COL_ROW_GEN
                stats.inc_col_split()

            elif col_state == NEXT_COL_GEN:
                split_start_t = perf_counter()
                can_split_cols = next_bin_partition(col_partitions,
                                                    m_col_partitions,
                                                    n_features)
                split_end_t = perf_counter()
                avg_split_t += split_end_t - split_start_t
                if can_split_cols:
                    col_state = WAIT_GEN
                    stats.inc_col_split()

            if can_split_cols:

                can_split_rows_col = True

                if col_row_state == FIRST_COL_ROW_GEN:

                    j = 0

                    split_start_t = perf_counter()
                    # row_col_partitions = numpy.zeros(n_instances, dtype=bool)
                    # m_row_col_partitions = numpy.zeros(n_instances,
                    # dtype=bool)

                    #
                    # copying the original data_slice instance ids
                    # to shuffle them
                    instance_ids_1 = numpy.copy(data_slice.instance_ids)
                    rand_gen.shuffle(instance_ids_1)

                    instance_ids_2 = numpy.copy(data_slice.instance_ids)
                    rand_gen.shuffle(instance_ids_2)

                    first_bin_partition(row_col_partitions,
                                        m_row_col_partitions,
                                        n_instances)

                    split_end_t = perf_counter()
                    avg_split_t += split_end_t - split_start_t

                    col_row_state = NEXT_COL_ROW_GEN

                elif col_row_state == NEXT_COL_ROW_GEN:
                    split_start_t = perf_counter()
                    can_split_rows_col = \
                        next_bin_partition(row_col_partitions,
                                           m_row_col_partitions,
                                           n_instances)
                    split_end_t = perf_counter()
                    avg_split_t += split_end_t - split_start_t

                    if not can_split_rows_col or j >= alpha_m:
                        col_row_state = FIRST_COL_ROW_GEN
                        col_state = NEXT_COL_GEN
                    else:
                        col_row_state = NEXT_COL_ROW_GEN

                if can_split_rows_col and j < alpha_m:
                    # logger.info('\nj %d', j)
                    split_start_t = perf_counter()
                    partition = \
                        binary_col_then_row_split_by_partition(
                            data_slice,
                            col_partitions,
                            row_col_partitions,
                            instance_ids_1,
                            instance_ids_2,
                            rand_gen,
                            bisplit)

                    split_end_t = perf_counter()

                    #
                    # evaluate the split
                    partition_lls = estimate_ll(partition,
                                                estimated_counts,
                                                ll_frequencies,
                                                train_dataset,
                                                feature_vals,
                                                alpha,
                                                valid_dataset)

                    partition_ll = None
                    lls_path = None
                    if propagate:
                        global_lls, lls_path = propagate_ll(data_slice,
                                                            partition_lls)
                        partition_ll = (numpy.sum(global_lls) /
                                        valid_dataset.shape[0])
                    else:
                        partition_ll = (numpy.sum(partition_lls) /
                                        valid_dataset.shape[0])

                    avg_split_t += split_end_t - split_start_t
                    sys.stdout.write('\ravg time {0:.8f} secs [{1}/{2}]'
                                     ' {3:.5f}'.
                                     format(avg_split_t / (i + 1),
                                            i + 1,
                                            alpha_n,
                                            partition_ll))

                    sys.stdout.flush()

                    #
                    # any improvement? if yes, store it
                    if partition_ll > best_ll:
                        logger.info(
                            '\nFOUND NEW BEST LL --> %f', partition_ll)
                        best_ll = partition_ll
                        best_partition = partition
                        best_path = lls_path

                    i += 1
                    j += 1
                    stats.inc_row_col_split(bisplit)

    opt_end_t = perf_counter()

    logger.info('\nFINAL BEST LL %f [in %f secs] after %d checks',
                best_ll,
                (opt_end_t - opt_start_t),
                i)

    return best_partition, best_ll, best_path


@numba.jit
def is_complex_partition(partition):
    # return any(len(slice) > 1 for slice in partition)
    return len(partition) > 1


@numba.jit
def split_slice_into_univariates(data_slice,
                                 slices_to_process,
                                 node_assoc,
                                 building_stack):
    """
    WRITEME
    """
    # current slice will correspond to a product node, storing it
    data_slice.type = ProductNode  # type(ProductNode)
    building_stack.append(data_slice)

    #
    # splitting for each feature
    for feature_id in data_slice.feature_ids:
        single_feature_slice = DataSlice(data_slice.instance_ids,
                                         numpy.array([feature_id]))
        slices_to_process.append(single_feature_slice)
        logger.info('\tAdding slice %d to process', single_feature_slice.id)
        # saving children refs
        data_slice.add_child(single_feature_slice)

    #
    # creating a product node
    prod_node = ProductNode(var_scope=frozenset(data_slice.feature_ids))
    prod_node.id = data_slice.id
    node_assoc[prod_node.id] = prod_node
    logger.info('>>> Added a prod node %d', prod_node.id)
    logger.info('Splitting into univariate %s', data_slice.feature_ids)


@numba.jit
def simple_rii_spn(train_dataset,
                   feature_vals,
                   valid_dataset,
                   alpha_n,
                   theta,
                   beta,
                   alpha,
                   min_instances_slice,
                   bisplit,
                   propagate,
                   rand_gen):
    """
    WRITEME
    """

    spn_start_t = perf_counter()

    #
    # damn numba that is choosy about keyworded parameters
    # if rand_gen is None:
    #     rand_gen = numpy.random.RandomState(1337)

    #
    # allocating buffers (assuming homogeneous features)
    n_instances = train_dataset.shape[0]
    n_features = train_dataset.shape[1]
    logger.info('Training set with %d instances & %d features',
                n_instances, n_features)
    logger.info('Validation set with %d instances & %d features',
                valid_dataset.shape[0], valid_dataset.shape[1])
    feature_val = feature_vals[0]
    estimated_counts = numpy.zeros((n_features, feature_val),
                                   dtype=numpy.uint32)
    ll_frequencies = numpy.zeros((n_features, feature_val),
                                 dtype=FLOAT_TYPE)

    #
    # a queue to process slices
    slices_to_process = deque()
    # a map to store nodes
    node_assoc = {}
    # a stack for the building blocks
    building_stack = deque()

    leaves = []
    #
    # creating the first slice and putting it to process
    whole_slice = DataSlice.whole_slice(n_instances, n_features)
    slices_to_process.append(whole_slice)
    # and estimating its ll as a naive factorization
    whole_slice_lls = estimate_data_slice_ll(whole_slice,
                                             estimated_counts,
                                             ll_frequencies,
                                             train_dataset,
                                             feature_vals,
                                             alpha,
                                             valid_dataset)

    whole_slice.ll = numpy.sum(whole_slice_lls) / valid_dataset.shape[0]
    whole_slice.lls = whole_slice_lls
    current_best_ll = numpy.sum(whole_slice_lls) / valid_dataset.shape[0]
    logger.info('Initial ll on dataset: %.6f', current_best_ll)

    #
    # saving statistics
    global_stats = SlsStats()

    #
    # trying to split as much as possible
    while slices_to_process:

        #
        # pop one to optimize
        current_slice = slices_to_process.popleft()
        current_n_instances = current_slice.n_instances()
        current_feature_ids = current_slice.feature_ids
        logger.info('> Processing slice %d', current_slice.id)
        logger.info('%r', current_slice)
        logger.info('remaining slices %r', slices_to_process)

        #
        # just one feature?
        if current_slice.n_features() < 2:
            #
            # create a leaf node
            feature_id = current_slice.feature_ids[0]
            feature_size = feature_vals[feature_id]
            # feature_data = train_dataset[current_slice.instance_ids,
            #                              feature_id]
            feature_data = \
                train_dataset[current_slice.instance_ids, :][:, [feature_id]]
            logger.info(feature_data)
            leaf_node = CategoricalSmoothedNode(var=feature_id,
                                                var_values=feature_size,
                                                alpha=alpha,
                                                data=feature_data)
            leaves.append(leaf_node)
            leaf_node.id = current_slice.id
            node_assoc[leaf_node.id] = leaf_node
            logger.info('>>> Adding a LEAF node (for feature %d)', feature_id)

        #
        # checking for min instances requirements
        elif current_n_instances <= min_instances_slice:

            #
            # since I am calling this subrouting twice I'm making a function
            # out of it, but it is less legible this way
            #
            split_slice_into_univariates(current_slice,
                                         slices_to_process,
                                         node_assoc,
                                         building_stack)
            #
            # we can split in peace
        else:

            best_partition = None
            best_ll = None

            #
            # looking for the best neighbor or trying to get a random one?
            split = rand_gen.rand()
            stats = SlsStats()

            if split > theta:
                #
                # maybe here a way to choose from the two alternatives is
                # needed
                logger.info('\t- Looking for best neighbor')
                neigh_start_t = perf_counter()
                best_partition, best_ll, best_path = \
                    best_neighbor_enum(current_slice,
                                       alpha_n,
                                       beta,
                                       estimated_counts,
                                       ll_frequencies,
                                       train_dataset,
                                       feature_vals,
                                       alpha,
                                       valid_dataset,
                                       bisplit,
                                       propagate,
                                       stats,
                                       rand_gen)

                neigh_end_t = perf_counter()
                #
                # updating global stats
                global_stats.add_stats(stats)
                logger.info('\tNeighborhood gen&search done in %.6f secs',
                            neigh_end_t - neigh_start_t)
            #
            # getting one completely at random
            else:
                logger.info('\tLooking for random neighbor')
                best_partition, best_ll, best_path = \
                    random_neighbor(current_slice,
                                    beta,
                                    estimated_counts,
                                    ll_frequencies,
                                    train_dataset,
                                    feature_vals,
                                    alpha,
                                    valid_dataset,
                                    bisplit,
                                    propagate,
                                    stats,
                                    rand_gen)
                global_stats.add_stats(stats)

            logger.info('\tBest partitioning: %r', best_partition)
            #
            # if some improvement has been done the partition is not None
            #
            # if best_partition is not None and best_ll > current_slice.ll:
            # if best_partition and best_ll > current_slice.ll:
            if best_partition and best_ll > current_best_ll:

                current_best_ll = best_ll

                logger.info('- Found a split improving the slice %.6f/%.6f',
                            best_ll, current_slice.ll)

                #
                # adding to the queue all the slices in the partition
                # remember, partition can be a nested sequence
                for slice in itertools.chain.from_iterable(best_partition):
                    logger.info('\tAdding slice %d to process', slice.id)
                    slices_to_process.append(slice)

                #
                # building the structure, for col+row partitions
                if is_complex_partition(best_partition):

                    # for simplicity here I am assuming the binary split
                    # a complex partition can be:
                    # ((a,), (b, c))
                    # ((a, b), (c,))
                    # ((a, b), (c, d)) # bisplit
                    # TODO: make it more general
                    row_partition = None
                    single_partition = None
                    if bisplit:

                        row_partition_1 = best_partition[0]
                        row_partition_2 = best_partition[1]

                        current_slice.type = ProductNode  # type(ProductNode)
                        building_stack.append(current_slice)
                        #
                        # creating a product node
                        prod_node = \
                            ProductNode(
                                var_scope=frozenset(current_feature_ids))
                        prod_node.id = current_slice.id
                        node_assoc[prod_node.id] = prod_node
                        #
                        # then two fake slices
                        sum_slice_1 = DataSlice([], [])
                        sum_slice_1.type = SumNode  # type(SumNode)
                        building_stack.append(sum_slice_1)

                        sum_slice_2 = DataSlice([], [])
                        sum_slice_2.type = SumNode  # type(SumNode)
                        building_stack.append(sum_slice_2)
                        #
                        # linking to the children
                        current_slice.add_child(sum_slice_1)
                        current_slice.add_child(sum_slice_2)

                        row_slice_feats_1 = None
                        for row_slice in row_partition_1:
                            sum_slice_1.add_child(row_slice,
                                                  row_slice.n_instances() /
                                                  current_n_instances)
                            #
                            # row slices have same scope
                            row_slice_feats_1 = row_slice.feature_ids

                        row_slice_feats_2 = None
                        for row_slice in row_partition_2:
                            sum_slice_2.add_child(row_slice,
                                                  row_slice.n_instances() /
                                                  current_n_instances)
                            #
                            # row slices have same scope
                            row_slice_feats_2 = row_slice.feature_ids

                        #
                        # creating the sum nodes
                        sum_node_1 = \
                            SumNode(var_scope=frozenset(row_slice_feats_1))
                        sum_node_1.id = sum_slice_1.id
                        node_assoc[sum_node_1.id] = sum_node_1

                        sum_node_2 = \
                            SumNode(var_scope=frozenset(row_slice_feats_2))
                        sum_node_2.id = sum_slice_2.id
                        node_assoc[sum_node_2.id] = sum_node_2

                        logger.info('>>> Added a prod node %d', prod_node.id)
                        logger.info('>>> Added a sum node %d', sum_node_1.id)
                        logger.info('>>> Added a sum node %d', sum_node_2.id)

                    else:
                        if len(best_partition[0]) > 1:
                            row_partition = best_partition[0]
                            single_partition = best_partition[1]
                        else:
                            row_partition = best_partition[1]
                            single_partition = best_partition[0]

                        current_slice.type = ProductNode  # type(ProductNode)
                        building_stack.append(current_slice)
                        #
                        # creating a product node
                        prod_node = \
                            ProductNode(
                                var_scope=frozenset(current_feature_ids))
                        prod_node.id = current_slice.id
                        node_assoc[prod_node.id] = prod_node
                        #
                        # then a fake slice (for the sum)
                        # for the building part
                        sum_slice = DataSlice([], [])
                        sum_slice.type = SumNode  # type(SumNode)
                        building_stack.append(sum_slice)
                        #
                        # linking to the children
                        single_prod_slice, = single_partition
                        current_slice.add_child(single_prod_slice)
                        current_slice.add_child(sum_slice)

                        row_slice_feats = None
                        for row_slice in row_partition:
                            sum_slice.add_child(row_slice,
                                                row_slice.n_instances() /
                                                current_n_instances)
                            #
                            # row slices have same scope
                            row_slice_feats = row_slice.feature_ids

                        #
                        # creating the sum node
                        sum_node = \
                            SumNode(var_scope=frozenset(row_slice_feats))
                        sum_node.id = sum_slice.id
                        node_assoc[sum_node.id] = sum_node

                        logger.info('>>> Added a prod node %d', prod_node.id)
                        logger.info('>>> Added a sum node %d', sum_node.id)
                #
                # a flat one means a row partition
                # ((a, b),)
                else:

                    #
                    current_slice.type = SumNode  # type(SumNode)
                    building_stack.append(current_slice)
                    unpacked_partition, = best_partition
                    for row_slice in unpacked_partition:

                        current_slice.add_child(row_slice,
                                                row_slice.n_instances() /
                                                current_n_instances)

                        # already appended everything
                        # slices_to_process.append(row_slice)

                    #
                    # creating a sum node
                    sum_node = \
                        SumNode(var_scope=frozenset(current_feature_ids))
                    sum_node.id = current_slice.id
                    node_assoc[sum_node.id] = sum_node

                    logger.info('>>> Added a sum node %d', sum_node.id)

            #
            # otherwise just split for each column, calling again
            # split_slice_into_univariates/4
            else:
                split_slice_into_univariates(current_slice,
                                             slices_to_process,
                                             node_assoc,
                                             building_stack)

    #
    # building and pruning the spn
    # Gens's style
    #
    logger.info('\n===> Building Treed SPN')
    # saving a reference now to the root (the first node)
    root_build_node = building_stack[0]
    root_node = node_assoc[root_build_node.id]
    print('root node: %r', root_node)

    # traversing the building stack
    # to link and prune nodes
    # for build_node in reversed(building_stack):
    root_node = SpnFactory.pruned_spn_from_slices(node_assoc,
                                                  building_stack,
                                                  logger)
    #
    # building layers
    #
    logger.info('\n===> Layering spn')
    spn = SpnFactory.layered_linked_spn(root_node)

    spn_end_t = perf_counter()
    logger.info('Spn learnt in {0} secs'.format(spn_end_t - spn_start_t))

    logger.info(spn.stats())

    #
    # testing on the validation set again
    #
    lls = []
    start_eval = perf_counter()
    for i in range(valid_dataset.shape[0]):
        # print('instance', i)
        lls.append(spn.eval(valid_dataset[i, :]))
    end_eval = perf_counter()

    valid_avg_ll = numpy.mean(lls)
    logger.info('\nValid Mean ll %.6f {in %.6f secs}\n',
                valid_avg_ll, end_eval - start_eval)

    return spn, global_stats


# @numba.jit
# def is_degenerate_partition(partition):
#     """
#     WRITEME
#     """
#     degenerate = False
#     if is_complex_partition(partition):
#         n_slices = len(partition)
#         for slice_partition in partition:
#             if len(slice_partition) > 1:
#                 pass
#     return degenerate


@numba.jit
def best_candidate_neighbors_naive(data_slice,
                                   current_best_ll,
                                   alpha_n,
                                   marg,
                                   beta,
                                   estimated_counts,
                                   ll_frequencies,
                                   train_dataset,
                                   feature_vals,
                                   alpha,
                                   valid_dataset,
                                   bisplit,
                                   propagate,
                                   stats,
                                   rand_gen):
    """
    This is the naive version, in which duplicate neighbors can be visited
    """
    best_partitions = []
    best_paths = []
    best_lls = []

    opt_start_t = perf_counter()

    n_instances = data_slice.n_instances()
    n_features = data_slice.n_features()

    #
    # TODO here I could heavily reuse the code from
    # random neighbor but  I want to monitor only the time of splittin
    # nevertheless I HAVE to refactor this code

    #
    # if only one element, cannot partition it anymore
    if n_instances < 2 or n_features < 2:
        return best_partitions, best_lls, best_paths

    avg_split_t = 0.0

    #
    # generate at most alpha_n neighbors
    # I have no guarantee that two generated neighbors are different
    for i in range(alpha_n):
        #
        # which operator to use?
        # split = rand_gen.random()
        split = rand_gen.rand()

        partition = None
        partition_lls = None

        split_start_t = perf_counter()

        if split < beta:
            # split_start_t = perf_counter()
            partition = random_binary_row_split(data_slice,
                                                rand_gen)
            # split_end_t = perf_counter()

            # logger.info('Split on rows %f', split_end_t - split_start_t)

            #
            # updating stats
            stats.inc_row_split()
        else:
            # split_start_t = perf_counter()

            partition = random_binary_col_then_row_split(data_slice,
                                                         rand_gen,
                                                         bisplit)

            # split_end_t = perf_counter()

            # logger.info('Split on cols %f', split_end_t - split_start_t)

            stats.inc_col_row_split(bisplit)

        partition_ll = None
        lls_path = None

        #
        # adding a check to the degenerate case:
        # if not is_degenerate_partition(partition):
        #
        # how good is it?
        partition_lls = estimate_ll(partition,
                                    estimated_counts,
                                    ll_frequencies,
                                    train_dataset,
                                    feature_vals,
                                    alpha,
                                    valid_dataset)
        if propagate:
            #
            # propagate it to obtain the global ll
            global_lls, lls_path = propagate_ll(data_slice,
                                                partition_lls)

            partition_ll = numpy.sum(global_lls) / valid_dataset.shape[0]
        else:
            partition_ll = numpy.sum(
                partition_lls) / valid_dataset.shape[0]

        split_end_t = perf_counter()
        avg_split_t += split_end_t - split_start_t

        sys.stdout.write('\ravg time {0} secs [{1}/{2}] {3:.5f}'.
                         format(avg_split_t / (i + 1),
                                i + 1, alpha_n))
        sys.stdout.flush()

        #
        # any improvement?
        if partition_ll > current_best_ll + marg:
            # logger.info('\nCandidate partition added --> %f',
            #             partition_ll)
            best_lls.append(partition_ll)
            best_partitions.append(partition)
            best_paths.append(lls_path)

        # if len(best_partitions) > 244:
        #     logger.info('FOCUSED %r', best_partitions[244])

        if (i + 1) % GC_COLLECT == 0:
            gc.collect()
    #
    # found a split improving the original slice ll?
    # found_partition = None
    # if best_ll > data_slice.ll:
    #     logger.info('\nFINAL BEST LL %f', best_ll)
    #     found_partition = best_partition
    # else:
    #     logger.info('NO IMPROVEMENT')

    # return found_partition
    opt_end_t = perf_counter()
    logger.info('\nCandidate list computed [in %f secs] after %d checks',
                (opt_end_t - opt_start_t), i)

    # logger.info('Partition ll %r', best_partitions[244])

    return best_partitions, best_lls, best_paths


@numba.jit
def best_candidate_neighbors_enum(data_slice,
                                  current_best_ll,
                                  alpha_n,
                                  marg,
                                  beta,
                                  estimated_counts,
                                  ll_frequencies,
                                  train_dataset,
                                  feature_vals,
                                  alpha,
                                  valid_dataset,
                                  bisplit,
                                  propagate,
                                  stats,
                                  rand_gen):
    """
    This is the enumerative version, in which no two same neighbors
    can be generated
    """
    best_lls = []
    best_partitions = []
    best_paths = []

    n_instances = data_slice.n_instances()
    n_features = data_slice.n_features()

    #
    # if only one element, cannot partition it anymore
    if n_instances < 2 or n_features < 2:
        return best_partitions, best_lls, best_paths

    #
    # allocating the buffers for the first partitions, assuming boolean arrays
    row_partitions = numpy.zeros(n_instances, dtype=bool)
    m_row_partitions = numpy.zeros(n_instances, dtype=bool)
    col_partitions = numpy.zeros(n_features, dtype=bool)
    m_col_partitions = numpy.zeros(n_features, dtype=bool)

    #
    # shuffling the data_slice instances and rows
    rand_gen.shuffle(data_slice.instance_ids)
    rand_gen.shuffle(data_slice.feature_ids)

    #
    # first partitions boolean
    first_row_partition = True
    first_col_partition = True

    #
    #
    FIRST_COL_GEN = 0
    NEXT_COL_GEN = 1
    FIRST_COL_ROW_GEN = 2
    NEXT_COL_ROW_GEN = 3
    WAIT_GEN = 4
    col_state = FIRST_COL_GEN
    col_row_state = FIRST_COL_ROW_GEN
    #
    # generate at most alpha_n neighbors
    # if there are no more partition to generate, exit
    can_split_rows = True
    can_split_cols = True

    avg_split_t = 0.0
    opt_start_t = perf_counter()

    i = 0
    j = 0

    #
    # TODO: this is just a stub, it needs to be defined better
    alpha_m = max(n_instances, int(alpha_n * 0.001))

    split_rows = False
    split_cols = False

    row_col_partitions = numpy.zeros(n_instances, dtype=bool)
    m_row_col_partitions = numpy.zeros(n_instances, dtype=bool)
    instance_ids_1 = None
    instance_ids_2 = None

    while i < alpha_n and (can_split_cols or can_split_rows):
        #
        #

        if (i + 1) % GC_COLLECT == 0:
            gc.collect()

        #
        # which operator to use?
        if can_split_cols and can_split_rows:
            split = rand_gen.rand()
            # print('\nBETA', split, beta, can_split_rows, can_split_cols)
            if split < beta:
                split_rows = True
                split_cols = False
            else:
                split_rows = False
                split_cols = True
        elif can_split_rows:
            split_rows = True
            split_cols = False
        elif can_split_cols:
            split_rows = False
            split_cols = True

        partition = None

        # if split < beta and can_split_rows:
        # print('\nSPLIT', split_rows, split_cols)
        if split_rows:
            # print('\nSPLIT ROWS', i)

            split_start_t = perf_counter()
            if first_row_partition:
                # logger.debug('FIRST SPLIT')
                first_bin_partition(row_partitions,
                                    m_row_partitions,
                                    n_instances)
                first_row_partition = False
            else:

                #
                # generating a row partition
                can_split_rows = next_bin_partition(row_partitions,
                                                    m_row_partitions,
                                                    n_instances)
            if can_split_rows:
                # logger.debug('SPLIT')
                #
                # splitting it
                partition = binary_row_split_by_partition(data_slice,
                                                          row_partitions)

                # split_end_t = perf_counter()

                # logger.debug('EVAL')
                #
                # evaluate the split
                partition_lls = estimate_ll(partition,
                                            estimated_counts,
                                            ll_frequencies,
                                            train_dataset,
                                            feature_vals,
                                            alpha,
                                            valid_dataset)

                partition_ll = None
                lls_path = None
                if propagate:
                    global_lls, lls_path = propagate_ll(data_slice,
                                                        partition_lls)
                    partition_ll = (numpy.sum(global_lls) /
                                    valid_dataset.shape[0])
                else:
                    partition_ll = (numpy.sum(partition_lls) /
                                    valid_dataset.shape[0])

                split_end_t = perf_counter()
                avg_split_t += split_end_t - split_start_t

                sys.stdout.write('\ravg time {0:.8f} secs [{1}/{2}] {3:.5f}'.
                                 format(avg_split_t / (i + 1),
                                        i + 1,
                                        alpha_n,
                                        partition_ll))
                sys.stdout.flush()

                #
                # any improvement? if yes, store it
                if partition_ll > current_best_ll + marg:
                    # logger.info('\nCandidate partition added --> %f',
                    #             partition_ll)
                    best_lls.append(partition_ll)
                    best_partitions.append(partition)
                    best_paths.append(lls_path)

                i += 1
                stats.inc_row_split()
            # else:
            #     print('\nNO MORE ROWS', i)

        # elif can_split_cols:  # andsplit >= beta
        elif split_cols:
            # print('\nSPLIT COLS', i)

            split_start_t = perf_counter()

            if col_state == FIRST_COL_GEN:
                split_start_t = perf_counter()

                first_bin_partition(col_partitions,
                                    m_col_partitions,
                                    n_features)

                split_end_t = perf_counter()
                avg_split_t += split_end_t - split_start_t

                col_state = WAIT_GEN
                col_row_state = FIRST_COL_ROW_GEN
                stats.inc_col_split()

            elif col_state == NEXT_COL_GEN:
                split_start_t = perf_counter()
                can_split_cols = next_bin_partition(col_partitions,
                                                    m_col_partitions,
                                                    n_features)

                split_end_t = perf_counter()
                avg_split_t += split_end_t - split_start_t

                if can_split_cols:
                    col_state = WAIT_GEN
                    stats.inc_col_split()

            if can_split_cols:

                can_split_rows_col = True

                if col_row_state == FIRST_COL_ROW_GEN:

                    j = 0

                    split_start_t = perf_counter()
                    # row_col_partitions = numpy.zeros(n_instances, dtype=bool)
                    # m_row_col_partitions = numpy.zeros(n_instances,
                    # dtype=bool)

                    #
                    # copying the original data_slice instance ids
                    # to shuffle them
                    instance_ids_1 = numpy.copy(data_slice.instance_ids)
                    rand_gen.shuffle(instance_ids_1)

                    instance_ids_2 = numpy.copy(data_slice.instance_ids)
                    rand_gen.shuffle(instance_ids_2)

                    first_bin_partition(row_col_partitions,
                                        m_row_col_partitions,
                                        n_instances)

                    split_end_t = perf_counter()
                    avg_split_t += split_end_t - split_start_t

                    col_row_state = NEXT_COL_ROW_GEN

                elif col_row_state == NEXT_COL_ROW_GEN:
                    split_start_t = perf_counter()
                    can_split_rows_col = \
                        next_bin_partition(row_col_partitions,
                                           m_row_col_partitions,
                                           n_instances)

                    split_end_t = perf_counter()
                    avg_split_t += split_end_t - split_start_t

                    if not can_split_rows_col or j >= alpha_m:
                        col_row_state = FIRST_COL_ROW_GEN
                        col_state = NEXT_COL_GEN
                    else:
                        col_row_state = NEXT_COL_ROW_GEN

                if can_split_rows_col and j < alpha_m:
                    # logger.info('\nj %d', j)
                    split_start_t = perf_counter()
                    partition = \
                        binary_col_then_row_split_by_partition(
                            data_slice,
                            col_partitions,
                            row_col_partitions,
                            instance_ids_1,
                            instance_ids_2,
                            rand_gen,
                            bisplit)

                    # split_end_t = perf_counter()

                    #
                    # evaluate the split
                    partition_lls = estimate_ll(partition,
                                                estimated_counts,
                                                ll_frequencies,
                                                train_dataset,
                                                feature_vals,
                                                alpha,
                                                valid_dataset)

                    partition_ll = None
                    lls_path = None
                    if propagate:
                        global_lls, lls_path = propagate_ll(data_slice,
                                                            partition_lls)
                        partition_ll = (numpy.sum(global_lls) /
                                        valid_dataset.shape[0])
                    else:
                        partition_ll = (numpy.sum(partition_lls) /
                                        valid_dataset.shape[0])

                    split_end_t = perf_counter()
                    avg_split_t += split_end_t - split_start_t

                    sys.stdout.write('\ravg time {0:.8f} secs [{1}/{2}]'
                                     ' {3:.5f}'.
                                     format(avg_split_t / (i + 1),
                                            i + 1,
                                            alpha_n,
                                            partition_ll))

                    sys.stdout.flush()

                    #
                    # any improvement? if yes, store it
                    if partition_ll > current_best_ll + marg:
                        # logger.info(
                        #     '\nCandidate partition added --> %f',
                        #     partition_ll)
                        best_lls.append(partition_ll)
                        best_partitions.append(partition)
                        best_paths.append(lls_path)

                    i += 1
                    j += 1
                    stats.inc_row_col_split(bisplit)

    opt_end_t = perf_counter()

    logger.info('\nFINAL BEST LL [in %f secs] after %d checks',
                (opt_end_t - opt_start_t),
                i)

    return best_partitions, best_lls, best_paths


def print_data_slice_tree(root_slice, n_instances):
    """
    utility
    """

    slices_to_process = deque()
    slices_to_process.append(root_slice)

    while slices_to_process:
        curr_slice = slices_to_process.popleft()
        ll = (numpy.sum(curr_slice.lls) / n_instances
              if curr_slice.lls is not None else -1000.0)
        logger.info('curr slice id: %d\n%r --- %.6f', curr_slice.id,
                    curr_slice.lls, ll)
        children_str = ""
        for child_slice in curr_slice.children:
            slices_to_process.append(child_slice)
            children_str += "({0}) ".format(child_slice.id)
        logger.info(children_str)


@numba.jit
def best_candidate_neighbors(current_slice,
                             current_best_ll,
                             enum,
                             max_neigh,
                             marg,
                             beta,
                             estimated_counts,
                             ll_frequencies,
                             train_dataset,
                             feature_vals,
                             alpha,
                             valid_dataset,
                             bisplit,
                             propagate,
                             stats,
                             rand_gen):
    """
    WRITEME
    """
    candidate_partitions = None
    candidate_lls = None
    candidate_paths = None

    if enum:
        logger.info('Generate best candidates by enumeration')
        candidate_partitions, candidate_lls, candidate_paths = \
            best_candidate_neighbors_enum(current_slice,
                                          current_best_ll,
                                          max_neigh,
                                          marg,
                                          beta,
                                          estimated_counts,
                                          ll_frequencies,
                                          train_dataset,
                                          feature_vals,
                                          alpha,
                                          valid_dataset,
                                          bisplit,
                                          propagate,
                                          stats,
                                          rand_gen)
    else:
        logger.info('Generate best candidates by random extractions')
        candidate_partitions, candidate_lls, candidate_paths = \
            best_candidate_neighbors_naive(current_slice,
                                           current_best_ll,
                                           max_neigh,
                                           marg,
                                           beta,
                                           estimated_counts,
                                           ll_frequencies,
                                           train_dataset,
                                           feature_vals,
                                           alpha,
                                           valid_dataset,
                                           bisplit,
                                           propagate,
                                           stats,
                                           rand_gen)

    return candidate_partitions, candidate_lls, candidate_paths


@numba.jit
def add_leaf_split(data_slice,
                   train_dataset,
                   feature_vals,
                   alpha,
                   node_assoc):
    """
    WRITEME
    """
    #
    # adding a leaf, assuming just one feature in the slice
    feature_id = data_slice.feature_ids[0]
    feature_size = feature_vals[feature_id]

    feature_data = train_dataset[data_slice.instance_ids, :][:, [feature_id]]
    # logger.info(feature_data)
    leaf_node = CategoricalSmoothedNode(var=feature_id,
                                        var_values=feature_size,
                                        alpha=alpha,
                                        data=feature_data)
    leaf_node.id = data_slice.id
    node_assoc[leaf_node.id] = leaf_node
    logger.info('>>> Adding a LEAF node (for feature %d)', feature_id)


@numba.jit
def split_slice_into_univariate_leaves(data_slice,
                                       train_dataset,
                                       feature_vals,
                                       alpha,
                                       node_assoc,
                                       building_stack):
    """
    WRITEME
    """
    # current slice will correspond to a product node, storing it
    data_slice.type = ProductNode  # type(ProductNode)
    building_stack.append(data_slice)

    #
    # splitting for each feature
    for feature_id in data_slice.feature_ids:
        single_feature_slice = DataSlice(data_slice.instance_ids,
                                         numpy.array([feature_id]))
        # slices_to_process.append(single_feature_slice)
        logger.info('\t\t\tAdding slice %d to process',
                    single_feature_slice.id)
        # saving children refs
        data_slice.add_child(single_feature_slice)

        # adding children
        add_leaf_split(single_feature_slice,
                       train_dataset,
                       feature_vals,
                       alpha,
                       node_assoc)

    #
    # creating a product node
    prod_node = ProductNode(var_scope=frozenset(data_slice.feature_ids))
    prod_node.id = data_slice.id
    node_assoc[prod_node.id] = prod_node
    logger.info('>>> Added a prod node %d', prod_node.id)
    logger.info('\t\tSplitted into univariate %s\n', data_slice.feature_ids)


# @numba.jit
def rii_spn(train_dataset,
            feature_vals,
            valid_dataset,
            alpha_n,
            marg,
            theta,
            beta,
            alpha,
            min_instances_slice,
            bisplit,
            propagate,
            enum,
            rand_gen):
    """
    WRITEME
    """
    spn_start_t = perf_counter()

    #
    # damn numba that is choosy about keyworded parameters
    # if rand_gen is None:
    #     rand_gen = numpy.random.RandomState(1337)

    #
    # allocating buffers (assuming homogeneous features)
    n_instances = train_dataset.shape[0]
    n_features = train_dataset.shape[1]
    logger.info('Training set with %d instances & %d features',
                n_instances, n_features)
    logger.info('Validation set with %d instances & %d features',
                valid_dataset.shape[0], valid_dataset.shape[1])
    feature_val = feature_vals[0]
    estimated_counts = numpy.zeros((n_features, feature_val),
                                   dtype=numpy.uint32)
    ll_frequencies = numpy.zeros((n_features, feature_val),
                                 dtype=FLOAT_TYPE)

    #
    # a set to process slices
    slices_to_process = set()
    # a map to store nodes
    node_assoc = {}
    # a stack for the building blocks
    building_stack = deque()

    #
    # creating the first slice and putting it to process
    whole_slice = DataSlice.whole_slice(n_instances, n_features)
    slices_to_process.add(whole_slice)
    # and estimating its ll as a naive factorization
    whole_slice_lls = estimate_data_slice_ll(whole_slice,
                                             estimated_counts,
                                             ll_frequencies,
                                             train_dataset,
                                             feature_vals,
                                             alpha,
                                             valid_dataset)

    whole_slice.ll = numpy.sum(whole_slice_lls) / valid_dataset.shape[0]
    whole_slice.lls = whole_slice_lls

    current_best_ll = numpy.sum(whole_slice_lls) / valid_dataset.shape[0]
    logger.info('Initial ll on dataset: %.6f', current_best_ll)

    current_tot_size = whole_slice.n_instances()

    # enum = True
    #
    # saving statistics
    global_stats = SlsStats()

    #
    # as long as there are splittable slices
    while slices_to_process:

        #
        # selecting the best one from it while looping through them
        candidate_splits = []
        candidate_lls = []
        candidate_paths = []
        neighborhood_assoc = {}
        n_tot_candidates = 0
        for current_slice in slices_to_process:

            #
            # getting a proportion
            max_neigh = int((current_slice.n_instances() / current_tot_size) *
                            alpha_n)
            stats = SlsStats()
            #
            # generate a list of best candidates
            best_neighbors, best_lls, best_paths = \
                best_candidate_neighbors(current_slice,
                                         current_best_ll,
                                         enum,
                                         max_neigh,
                                         marg,
                                         beta,
                                         estimated_counts,
                                         ll_frequencies,
                                         train_dataset,
                                         feature_vals,
                                         alpha,
                                         valid_dataset,
                                         bisplit,
                                         propagate,
                                         stats,
                                         rand_gen)
            #
            # keeping track
            candidate_splits.extend(best_neighbors)
            candidate_lls.extend(best_lls)
            candidate_paths.extend(best_paths)
            n_candidates = len(candidate_splits)
            #
            # also of the assoc neighbor split -> parent slice to remove
            # damn numba
            for i in range(n_candidates):
                neighborhood_assoc[n_tot_candidates + i] = current_slice
            n_tot_candidates += n_candidates

            #
            # aggregating stats
            global_stats.add_stats(stats)

            #
        # choosing the best of a random pick among the improving ones?
        split = rand_gen.rand()

        # next_split = None
        # next_lls = None
        # next_path = None
        index = None

        #
        # no improving splits, aka empty candidate list
        # checking this way for numba, bleargh
        if n_tot_candidates == 0:

            logger.info('\n+++++++++ No more candidate splits ++++++++++\n')
            #
            # so I am splitting all slices in the fringe and removing them
            for slice in slices_to_process:

                split_slice_into_univariate_leaves(slice,
                                                   train_dataset,
                                                   feature_vals,
                                                   alpha,
                                                   node_assoc,
                                                   building_stack)

            slices_to_process.clear()

        #
        # else there are candidates, so let's process the best
        else:
            if split > theta:
                #
                # getting the one with the highest score
                logger.info('GETTING BEST CANDIDATE')
                index = numpy.argmax(candidate_lls)
            else:
                #
                # getting one at random
                logger.info('GETTING RANDOM CANDIDATE')
                index = rand_gen.choice(len(candidate_splits))

            next_split = candidate_splits[index]
            next_ll = candidate_lls[index]
            next_path = candidate_paths[index]
            # next_lls = next_path[0]

            logger.info('---- %r ---- [%.6f]', next_split, next_ll)

            #
            # removing the parent slice from the slices to process
            logger.info('Retrieving the parent slice',
                        neighborhood_assoc)
            parent_slice = neighborhood_assoc[index]
            slices_to_process.remove(parent_slice)

            current_n_instances = parent_slice.n_instances()
            current_feature_ids = parent_slice.feature_ids
            logger.info('%r', parent_slice)
            logger.info('> Processing slice %d', parent_slice.id)

            #
            # propagate the ll to update the tree
            update_best_lls(parent_slice, next_path)
            current_best_ll = next_ll

            #
            # adding the nodes for the spn
            # discriminating on the partition type
            #
            if is_complex_partition(next_split):

                # for simplicity here I am assuming the binary split
                # a complex partition can be:
                # ((a,), (b, c))
                # ((a, b), (c,))
                # ((a, b), (c, d)) # bisplit
                # TODO: make it more general
                row_partition = None
                single_partition = None
                if bisplit:

                    logger.info('The best split was a bisplit')

                    row_partition_1 = next_split[0]
                    row_partition_2 = next_split[1]

                    parent_slice.type = ProductNode  # type(ProductNode)
                    building_stack.append(parent_slice)
                    #
                    # creating a product node
                    prod_node = \
                        ProductNode(
                            var_scope=frozenset(current_feature_ids))
                    prod_node.id = parent_slice.id
                    node_assoc[prod_node.id] = prod_node
                    #
                    # then two fake slices
                    sum_slice_1 = DataSlice([], [])
                    sum_slice_1.type = SumNode  # type(SumNode)
                    building_stack.append(sum_slice_1)

                    sum_slice_2 = DataSlice([], [])
                    sum_slice_2.type = SumNode  # type(SumNode)
                    building_stack.append(sum_slice_2)
                    #
                    # linking to the children
                    parent_slice.add_child(sum_slice_1)
                    parent_slice.add_child(sum_slice_2)

                    row_slice_feats_1 = None
                    for row_slice in row_partition_1:
                        sum_slice_1.add_child(row_slice,
                                              row_slice.n_instances() /
                                              current_n_instances)
                        #
                        # row slices have same scope
                        row_slice_feats_1 = row_slice.feature_ids

                    #
                    # estimating lls after linking
                    estimate_row_slice_lls(sum_slice_1, valid_dataset)

                    row_slice_feats_2 = None
                    for row_slice in row_partition_2:
                        sum_slice_2.add_child(row_slice,
                                              row_slice.n_instances() /
                                              current_n_instances)
                        #
                        # row slices have same scope
                        row_slice_feats_2 = row_slice.feature_ids

                    #
                    estimate_row_slice_lls(sum_slice_2, valid_dataset)

                    #
                    # creating the sum nodes
                    sum_node_1 = \
                        SumNode(var_scope=frozenset(row_slice_feats_1))
                    sum_node_1.id = sum_slice_1.id
                    node_assoc[sum_node_1.id] = sum_node_1

                    sum_node_2 = \
                        SumNode(var_scope=frozenset(row_slice_feats_2))
                    sum_node_2.id = sum_slice_2.id
                    node_assoc[sum_node_2.id] = sum_node_2

                    logger.info('>>> Added a prod node %d', prod_node.id)
                    logger.info('>>> Added a sum node %d', sum_node_1.id)
                    logger.info('>>> Added a sum node %d', sum_node_2.id)

                else:

                    logger.info('Best split was a col then row')
                    if len(next_split[0]) > 1:
                        row_partition = next_split[0]
                        single_partition = next_split[1]
                    else:
                        row_partition = next_split[1]
                        single_partition = next_split[0]

                    parent_slice.type = ProductNode  # type(ProductNode)
                    building_stack.append(parent_slice)
                    #
                    # creating a product node
                    prod_node = \
                        ProductNode(
                            var_scope=frozenset(current_feature_ids))
                    prod_node.id = parent_slice.id
                    node_assoc[prod_node.id] = prod_node
                    #
                    # then a fake slice (for the sum)
                    # for the building part
                    sum_slice = DataSlice([], [])
                    sum_slice.type = SumNode  # type(SumNode)
                    building_stack.append(sum_slice)
                    #
                    # linking to the children
                    single_prod_slice, = single_partition
                    parent_slice.add_child(single_prod_slice)
                    parent_slice.add_child(sum_slice)

                    row_slice_feats = None
                    for row_slice in row_partition:
                        sum_slice.add_child(row_slice,
                                            row_slice.n_instances() /
                                            current_n_instances)
                        #
                        # row slices have same scope
                        row_slice_feats = row_slice.feature_ids

                    #
                    # now that it's linked setting its lls
                    estimate_row_slice_lls(sum_slice, valid_dataset)
                    #
                    # creating the sum node
                    sum_node = \
                        SumNode(var_scope=frozenset(row_slice_feats))
                    sum_node.id = sum_slice.id
                    node_assoc[sum_node.id] = sum_node

                    logger.info('>>> Added a prod node %d', prod_node.id)
                    logger.info('>>> Added a sum node %d', sum_node.id)
            #
            # a flat one means a row partition
            # ((a, b),)
            else:

                logger.info('Best split is a simple row split')
                #
                parent_slice.type = SumNode  # type(SumNode)
                building_stack.append(parent_slice)
                unpacked_partition, = next_split
                for row_slice in unpacked_partition:

                    parent_slice.add_child(row_slice,
                                           row_slice.n_instances() /
                                           current_n_instances)

                #
                # creating a sum node
                sum_node = \
                    SumNode(var_scope=frozenset(current_feature_ids))
                sum_node.id = parent_slice.id
                node_assoc[sum_node.id] = sum_node

                logger.info('>>> Added a sum node %d', sum_node.id)

            logger.info('What to do with the new slices?')
            #
            # adding the new slices to the set to process
            for slice in itertools.chain.from_iterable(next_split):
                #
                # checking for the requisites
                logger.info('\tCheking how to process slice %d', slice.id)
                #

                if slice.n_features() < 2:
                    #
                    # adding a leaf
                    logger.info('\t\tAdding a leaf')
                    add_leaf_split(slice,
                                   train_dataset,
                                   feature_vals,
                                   alpha,
                                   node_assoc)

                elif slice.n_instances() <= min_instances_slice:

                    logger.info('\t\tSplitting into univariate')
                    split_slice_into_univariate_leaves(slice,
                                                       train_dataset,
                                                       feature_vals,
                                                       alpha,
                                                       node_assoc,
                                                       building_stack)
                else:

                    #
                    # just adding it to the fringe
                    slices_to_process.add(slice)
                    logger.info('\t\tAdding to the slices to process')

        #
        # printing the slice tree
        # logger.info('\n====> Printing data slices tree')
        # print_data_slice_tree(whole_slice, valid_dataset.shape[0])
        # logger.info('=========================================\n')

    #
    # building and pruning the spn
    # Gens's style
    #
    logger.info('\n===> Building Treed SPN')
    # saving a reference now to the root (the first node)
    root_build_node = building_stack[0]
    root_node = node_assoc[root_build_node.id]
    print('root node: %r', root_node)

    SpnFactory.pruned_spn_from_slices(node_assoc,
                                      building_stack)
    #
    # building layers
    #
    logger.info('\n===> Layering spn')
    spn = SpnFactory.layered_linked_spn(root_node)

    spn_end_t = perf_counter()
    logger.info('Spn learnt in {0} secs'.format(spn_end_t - spn_start_t))

    logger.info(spn.stats())

    #
    # testing on the validation set again
    #
    lls = []
    start_eval = perf_counter()
    for i in range(valid_dataset.shape[0]):
        # print('instance', i)
        lls.append(spn.eval(valid_dataset[i, :]))
    end_eval = perf_counter()

    valid_avg_ll = numpy.mean(lls)
    logger.info('Inferences %r', lls)
    logger.info('\nValid Mean ll %.6f {in %.6f secs}\n',
                valid_avg_ll, end_eval - start_eval)

    return spn, global_stats
