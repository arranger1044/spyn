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
from spn import RND_SEED

from spn.linked.nodes import CategoricalSmoothedNode
from spn.linked.nodes import SumNode
from spn.linked.nodes import ProductNode
from spn.linked.nodes import CLTreeNode

from spn.factory import SpnFactory

from collections import deque

import math

import logging

import sklearn.mixture

from algo.dataslice import DataSlice

import dataset

# import tests

NEG_INF = -sys.float_info.max


@numba.jit
def g_test(feature_id_1,
           feature_id_2,
           instance_ids,
           data,
           feature_vals,
           g_factor):
    """
    Applying a G-test on the two features (represented by ids) on the data
    """
    # print(feature_id_1, feature_id_2, instance_ids)

    #
    # swap to preserve order, is this needed?
    if feature_id_1 > feature_id_2:
        #
        # damn numba this cannot be done lol
        # feature_id_1, feature_id_2 = feature_id_2, feature_id_1
        tmp = feature_id_1
        feature_id_1 = feature_id_2
        feature_id_2 = tmp

    # print(feature_id_1, feature_id_2, instance_ids)

    n_instances = len(instance_ids)

    feature_size_1 = feature_vals[feature_id_1]
    feature_size_2 = feature_vals[feature_id_2]

    #
    # support vectors for counting the occurrences
    feature_tot_1 = numpy.zeros(feature_size_1, dtype=numpy.uint32)
    feature_tot_2 = numpy.zeros(feature_size_2, dtype=numpy.uint32)
    co_occ_matrix = numpy.zeros((feature_size_1, feature_size_2),
                                dtype=numpy.uint32)

    #
    # counting for the current instances
    for i in instance_ids:
        co_occ_matrix[data[i, feature_id_1], data[i, feature_id_2]] += 1

    # print('Co occurrences', co_occ_matrix)
    #
    # getting the sum for each feature
    for i in range(feature_size_1):
        for j in range(feature_size_2):
            count = co_occ_matrix[i, j]
            feature_tot_1[i] += count
            feature_tot_2[j] += count

    # print('Feature tots', feature_tot_1, feature_tot_2)

    #
    # counputing the number of zero total co-occurrences for the degree of
    # freedom
    feature_nonzero_1 = numpy.count_nonzero(feature_tot_1)
    feature_nonzero_2 = numpy.count_nonzero(feature_tot_2)

    dof = (feature_nonzero_1 - 1) * (feature_nonzero_2 - 1)

    g_val = 0.0
    for i, tot_1 in enumerate(feature_tot_1):
        for j, tot_2 in enumerate(feature_tot_2):
            count = co_occ_matrix[i, j]
            if count != 0:
                exp_count = tot_1 * tot_2 / n_instances
                g_val += count * math.log(count / exp_count)

    g_val *= 2

    # testing against p value
    dep_val = 2 * dof * g_factor + 0.001
    # logging.info('\t[G: %f dep-val: %f]', g_val, dep_val)
    # print("(", feature_id_1, feature_id_2, ") G:", g_val, "dep_val:", dep_val)
    return g_val < dep_val


@numba.jit
def greedy_feature_split(data,
                         data_slice,
                         feature_vals,
                         g_factor,
                         rand_gen):
    """
    WRITEME
    """
    n_features = data_slice.n_features()

    feature_ids_mask = numpy.ones(n_features, dtype=bool)

    #
    # extracting one feature at random
    rand_feature_id = rand_gen.randint(0, n_features)
    feature_ids_mask[rand_feature_id] = False

    dependent_features = numpy.zeros(n_features, dtype=bool)
    dependent_features[rand_feature_id] = True

    # greedy bfs searching
    features_to_process = deque()
    features_to_process.append(rand_feature_id)

    while features_to_process:
        # get one
        current_feature_id = features_to_process.popleft()
        feature_id_1 = data_slice.feature_ids[current_feature_id]
        # print('curr FT', current_feature_id)

        # features to remove later
        features_to_remove = numpy.zeros(n_features, dtype=bool)

        for other_feature_id in feature_ids_mask.nonzero()[0]:

            #
            # print('considering other features', other_feature_id)
            feature_id_2 = data_slice.feature_ids[other_feature_id]
            #
            # apply a G-test
            if not g_test(feature_id_1,
                          feature_id_2,
                          data_slice.instance_ids,
                          data,
                          feature_vals,
                          g_factor):
                #
                # print('found dependency!', (feature_id_1, feature_id_2))

                #
                # updating 'sets'
                features_to_remove[other_feature_id] = True
                dependent_features[other_feature_id] = True
                features_to_process.append(other_feature_id)

        # now removing from future considerations
        feature_ids_mask[features_to_remove] = False

    # translating remaining features
    first_component = data_slice.feature_ids[dependent_features]
    second_component = data_slice.feature_ids[~ dependent_features]

    return first_component, second_component


def retrieve_clustering(assignment, indexes=None):
    """
    from [2, 3, 8, 3, 1]
    to [{0}, {1, 3}, {2}, {3}]

    or

    from [2, 3, 8, 3, 1] and [21, 1, 4, 18, 11]
    to [{21}, {1, 18}, {4}, {11}]

    """

    clustering = []
    seen_clusters = dict()

    if indexes is None:
        indexes = [i for i in range(len(assignment))]

    for index, label in zip(indexes, assignment):
        if label not in seen_clusters:
            seen_clusters[label] = len(clustering)
            clustering.append([])
        clustering[seen_clusters[label]].append(index)

    return clustering


def cluster_rows(data,
                 data_slice,
                 n_clusters=2,
                 cluster_method='GMM',
                 n_iters=100,
                 n_restarts=3,
                 cluster_penalty=1.0,
                 rand_gen=None,
                 sklearn_args=None):
    """
    A wrapper to abstract from the implemented clustering method

    cluster_method = GMM | DPGMM | HOEM
    """

    clustering = None

    #
    # slicing the data
    sliced_data = data[data_slice.instance_ids, :][:, data_slice.feature_ids]

    if cluster_method == 'GMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'
        #
        # creating the cluster from sklearn
        gmm_c = sklearn.mixture.GMM(n_components=n_clusters,
                                    covariance_type=cov_type,
                                    random_state=rand_gen,
                                    n_iter=n_iters,
                                    n_init=n_restarts)

        #
        # fitting to training set
        fit_start_t = perf_counter()
        gmm_c.fit(sliced_data)
        fit_end_t = perf_counter()

        #
        # getting the cluster assignment
        pred_start_t = perf_counter()
        clustering = gmm_c.predict(sliced_data)
        pred_end_t = perf_counter()

    elif cluster_method == 'DPGMM':
        #
        # retrieving other properties
        cov_type = sklearn_args['covariance_type'] \
            if 'covariance_type' in sklearn_args else 'diag'
        verbose = sklearn_args['verbose']\
            if 'verbose' in sklearn_args else False

        dpgmm_c = sklearn.mixture.DPGMM(n_components=n_clusters,
                                        covariance_type=cov_type,
                                        random_state=rand_gen,
                                        n_iter=n_iters,
                                        alpha=cluster_penalty,
                                        verbose=verbose)

        #
        # fitting to training set
        fit_start_t = perf_counter()
        dpgmm_c.fit(sliced_data)
        fit_end_t = perf_counter()

        #
        # getting the cluster assignment
        pred_start_t = perf_counter()
        clustering = dpgmm_c.predict(sliced_data)
        pred_end_t = perf_counter()

    elif cluster_method == 'HOEM':
        raise NotImplementedError('Hard Online EM is not implemented yet')
    else:
        raise Exception('Clustering method not valid')

    logging.info('Clustering done in %f secs', (fit_end_t - fit_start_t))

    #
    # translating the cluster assignment to
    # a list of clusters (set of instances)

    return retrieve_clustering(clustering, data_slice.instance_ids)


def cache_data_slice(data_slice, cache):
    """
    WRITEME
    """
    #
    # getting ids
    instance_ids = data_slice.instance_ids
    feature_ids = data_slice.feature_ids
    #
    # ordering
    instance_ids.sort()
    feature_ids.sort()
    #
    # making unmutable
    instances_tuple = tuple(instance_ids)
    features_tuple = tuple(feature_ids)
    hashed_slice = (instances_tuple, features_tuple)
    #
    #
    cached_slice = None
    try:
        cached_slice = cache[hashed_slice]
    except:
        cache[hashed_slice] = data_slice

    return cached_slice


class LearnSPN(object):

    """
    Implementing Gens and Domingos
    """

    def __init__(self,
                 g_factor=1.0,
                 min_instances_slice=100,
                 min_features_slice=0,
                 alpha=0.1,
                 row_cluster_method='GMM',
                 cluster_penalty=2.0,
                 n_cluster_splits=2,
                 n_iters=100,
                 n_restarts=3,
                 sklearn_args={},
                 cltree_leaves=False,
                 rand_gen=None):
        """
        WRITEME
        """
        self._g_factor = g_factor
        self._min_instances_slice = min_instances_slice
        self._min_features_slice = min_features_slice
        self._alpha = alpha
        self._row_cluster_method = row_cluster_method
        self._cluster_penalty = cluster_penalty
        self._n_cluster_splits = n_cluster_splits
        self._n_iters = n_iters
        self._n_restarts = n_restarts
        self._sklearn_args = sklearn_args
        self._cltree_leaves = cltree_leaves
        self._rand_gen = rand_gen if rand_gen is not None \
            else numpy.random.RandomState(RND_SEED)

        logging.info('LearnSPN:\n\tg factor:%f\n\tmin inst:%d\n' +
                     '\tmin feat:%d\n' +
                     '\talpha:%f\n\tcluster pen:%f\n\tn clusters:%d\n' +
                     '\tcluster method=%s\n\tn iters: %d\n' +
                     '\tn restarts: %d\n\tcltree leaves:%s\n' +
                     '\tsklearn args: %s\n',
                     self._g_factor,
                     self._min_instances_slice,
                     self._min_features_slice,
                     self._alpha,
                     self._cluster_penalty,
                     self._n_cluster_splits,
                     self._row_cluster_method,
                     self._n_iters,
                     self._n_restarts,
                     self._cltree_leaves,
                     self._sklearn_args)

        #
        # resetting the data slice ids (just in case)
        DataSlice.reset_id_counter()

    def fit_mixture_bootstrap(self,
                              train,
                              n_mix_components,
                              bootstrap_samples_ids=None,
                              valid=None,
                              test=None,
                              feature_sizes=None,
                              perc=1.0,
                              replace=True,
                              evaluate=True):
        """
        WRITEME
        """
        n_train_instances = train.shape[0]
        n_features = train.shape[1]

        #
        # if not present, assuming all binary features
        if feature_sizes is None:
            feature_sizes = [2 for i in range(n_features)]

        train_mixture_lls = None
        valid_mixture_lls = None
        test_mixture_lls = None

        if evaluate:
            train_mixture_lls = numpy.zeros((n_train_instances,
                                             n_mix_components))

            if valid is not None:
                n_valid_instances = valid.shape[0]
                valid_mixture_lls = numpy.zeros((n_valid_instances,
                                                 n_mix_components))

            if test is not None:
                n_test_instances = test.shape[0]
                test_mixture_lls = numpy.zeros((n_test_instances,
                                                n_mix_components))

        mixture = []

        mix_start_t = perf_counter()
        #
        # generating the mixtures
        for m in range(n_mix_components):
            #
            # slicing the training set via bootstrap samples ids
            # (if present, otherwise sampling)
            train_mix = None
            if bootstrap_samples_ids is not None:
                train_mix = train[bootstrap_samples_ids[m, :], :]
                logging.debug('Bootstrap sample ids: %s',
                              bootstrap_samples_ids[m, :])
            #
            # TODO: this branch shall be deprecated and pruned
            # the bootstrap sample ids shall be generated always from the
            # calling function and not here
            else:
                raise RuntimeError('This is deprecated')
                # train_mix = dataset.sample_instances(train,
                #                                     perc = perc,
                #                                     replace = replace,
                # rndState = self._rand_gen)
            logging.info('Sampled dataset (%d X %d) for mixture: %d',
                         train_mix.shape[0], train_mix.shape[1],
                         m)

            #
            # learning an spn for it
            learn_start_t = perf_counter()
            spn_mix = self.fit_structure(train_mix, feature_sizes)
            learn_end_t = perf_counter()
            logging.info('> SPN learned in %f secs',
                         learn_end_t - learn_start_t)

            if evaluate:
                #
                # now doing inference
                train_ll = 0.0
                for i, train_instance in enumerate(train):
                    (pred_ll, ) = spn_mix.single_eval(train_instance)
                    train_mixture_lls[i, m] = pred_ll
                    train_ll += pred_ll

                logging.info('\ttrain avg ll: %f',
                             train_ll / train.shape[0])

                if valid is not None:
                    valid_ll = 0.0
                    for i, valid_instance in enumerate(valid):
                        (pred_ll, ) = spn_mix.single_eval(valid_instance)
                        valid_mixture_lls[i, m] = pred_ll
                        valid_ll += pred_ll

                    logging.info('\tvalid avg ll: %f',
                                 valid_ll / valid.shape[0])

                if test is not None:
                    test_ll = 0.0
                    for i, test_instance in enumerate(test):
                        (pred_ll, ) = spn_mix.single_eval(test_instance)
                        test_mixture_lls[i, m] = pred_ll
                        test_ll += pred_ll

                    logging.info('\ttest avg ll: %f',
                                 test_ll / test.shape[0])
            else:
                #
                # adding to the mixture
                mixture.append(spn_mix)

        mix_end_t = perf_counter()
        logging.info('-- mixtures computed in %f', mix_end_t - mix_start_t)
        #
        # with evaluate we return just the computed values
        if evaluate:
            return (train_mixture_lls,
                    valid_mixture_lls,
                    test_mixture_lls)
        else:
            return mixture

    def fit_structure(self,
                      data,
                      feature_sizes):
        """
        data is a numpy array
        """

        tot_n_instances = data.shape[0]
        tot_n_features = data.shape[1]

        logging.info('Learning SPN structure on a (%d X %d) dataset',
                     tot_n_instances, tot_n_features)
        learn_start_t = perf_counter()

        #
        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        building_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        # creating the first slice
        whole_slice = DataSlice.whole_slice(tot_n_instances,
                                            tot_n_features)
        slices_to_process.append(whole_slice)

        first_run = True

        #
        # iteratively process & split slices
        #
        while slices_to_process:

            # process a slice
            current_slice = slices_to_process.popleft()

            # pointers to the current data slice
            current_instances = current_slice.instance_ids
            current_features = current_slice.feature_ids
            current_id = current_slice.id

            n_instances = len(current_instances)
            n_features = len(current_features)

            logging.info('\n*** Processing slice %d (%d X %d)',
                         current_id,
                         n_instances, n_features)
            logging.debug('\tinstances:%s\n\tfeatures:%s',
                          current_instances,
                          current_features)

            #
            # is this a leaf node or we can split?
            if n_features == 1:
                logging.info('---> Adding a leaf (just one feature)')

                (feature_id, ) = current_features
                feature_size = feature_sizes[feature_id]

                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                # create the node
                leaf_node = \
                    CategoricalSmoothedNode(var=feature_id,
                                            var_values=feature_size,
                                            data=current_slice_data,
                                            instances=current_instances,
                                            alpha=self._alpha)
                # storing links
                # input_nodes.append(leaf_node)
                leaf_node.id = current_id
                node_id_assoc[current_id] = leaf_node

                logging.debug('\tCreated Smooth Node %s', leaf_node)

            elif (n_instances <= self._min_instances_slice and n_features > 1):
                #
                # splitting the slice on each feature
                logging.info('---> Few instances (%d), decompose all features',
                             n_instances)
                #
                # shall put a cltree or
                if self._cltree_leaves:
                    logging.info('into a Chow-Liu tree')
                    #
                    # slicing data
                    slice_data_rows = data[current_instances, :]
                    current_slice_data = slice_data_rows[:, current_features]

                    current_feature_sizes = [feature_sizes[i]
                                             for i in current_features]
                    #
                    # creating a Chow-Liu tree as leaf
                    leaf_node = \
                        CLTreeNode(vars=current_features,
                                   var_values=current_feature_sizes,
                                   data=current_slice_data,
                                   alpha=self._alpha)
                    #
                    # storing links
                    leaf_node.id = current_id
                    node_id_assoc[current_id] = leaf_node

                    logging.debug('\tCreated Chow-Liu Tree Node %s', leaf_node)

                else:
                    logging.info('into a naive factorization')
                    #
                    # putting them in queue
                    child_slices = [DataSlice(current_instances, [feature_id])
                                    for feature_id in current_features]
                    slices_to_process.extend(child_slices)

                    children_ids = [child.id for child in child_slices]

                    #
                    # storing the children links
                    for child_slice in child_slices:
                        current_slice.add_child(child_slice)
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)

                    #
                    # creating the product node
                    prod_node = ProductNode(
                        var_scope=frozenset(current_features))
                    prod_node.id = current_id

                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

            else:

                #
                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                split_on_features = False
                #
                # first run is a split on rows
                if first_run:
                    logging.info('-- FIRST RUN --')
                    first_run = False
                else:
                    #
                    # try clustering on cols
                    logging.debug('...trying to split on columns')
                    dependent_features, other_features = \
                        greedy_feature_split(data,
                                             current_slice,
                                             feature_sizes,
                                             self._g_factor,
                                             self._rand_gen)
                    if len(other_features) > 0:
                        split_on_features = True
                #
                # have dependent components been found?
                if split_on_features:
                    #
                    # splitting on columns
                    logging.info('---> Splitting on features')

                    #
                    # creating two new data slices and putting them on queue
                    first_slice = DataSlice(current_instances,
                                            dependent_features)
                    second_slice = DataSlice(current_instances,
                                             other_features)
                    slices_to_process.append(first_slice)
                    slices_to_process.append(second_slice)

                    children_ids = [first_slice.id, second_slice.id]

                    #
                    # storing link parent children
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)
                    current_slice.add_child(first_slice)
                    current_slice.add_child(second_slice)

                    #
                    # creating product node
                    prod_node = \
                        ProductNode(var_scope=frozenset(current_features))
                    prod_node.id = current_id
                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

                else:
                    #
                    # clustering on rows
                    logging.info('---> Splitting on rows')

                    #
                    # at most n_rows clusters, for sklearn
                    k_row_clusters = min(self._n_cluster_splits,
                                         n_instances - 1)

                    clustering = \
                        cluster_rows(data,
                                     current_slice,
                                     n_clusters=k_row_clusters,
                                     cluster_method=self._row_cluster_method,
                                     n_iters=self._n_iters,
                                     n_restarts=self._n_restarts,
                                     cluster_penalty=self._cluster_penalty,
                                     rand_gen=self._rand_gen,
                                     sklearn_args=self._sklearn_args)

                    # logging.debug('obtained clustering %s', clustering)
                    logging.info('clustered into %d parts (min %d)',
                                 len(clustering), k_row_clusters)
                    # splitting
                    cluster_slices = [DataSlice(cluster, current_features)
                                      for cluster in clustering]
                    cluster_slices_ids = [slice.id
                                          for slice in cluster_slices]
                    cluster_weights = [slice.n_instances() / n_instances
                                       for slice in cluster_slices]

                    #
                    # appending for processing
                    slices_to_process.extend(cluster_slices)

                    #
                    # storing links
                    # current_slice.children = cluster_slices_ids
                    # current_slice.weights = cluster_weights
                    current_slice.type = SumNode
                    building_stack.append(current_slice)
                    for child_slice, child_weight in zip(cluster_slices,
                                                         cluster_weights):
                        current_slice.add_child(child_slice, child_weight)

                    #
                    # building a sum node
                    sum_node = SumNode(var_scope=frozenset(current_features))
                    sum_node.id = current_id
                    node_id_assoc[current_id] = sum_node
                    logging.debug('\tCreated Sum Node %s (with children %s)',
                                  sum_node,
                                  cluster_slices_ids)

        learn_end_t = perf_counter()
        logging.info('Structure learned in %f secs',
                     (learn_end_t - learn_start_t))

        #
        # linking the spn graph (parent -> children)
        #
        logging.info('===> Building tree')

        link_start_t = perf_counter()
        root_build_node = building_stack[0]
        root_node = node_id_assoc[root_build_node.id]
        logging.debug('root node: %s', root_node)

        root_node = SpnFactory.pruned_spn_from_slices(node_id_assoc,
                                                      building_stack)
        link_end_t = perf_counter()
        logging.info('\tLinked the spn in %f secs (root_node %s)',
                     (link_end_t - link_start_t),
                     root_node)

        #
        # building layers
        #
        logging.info('===> Layering spn')
        layer_start_t = perf_counter()
        spn = SpnFactory.layered_linked_spn(root_node)
        layer_end_t = perf_counter()
        logging.info('\tLayered the spn in %f secs',
                     (layer_end_t - layer_start_t))

        logging.info('\nLearned SPN\n\n%s', spn.stats())

        return spn

    def fit_structure_bagging(self,
                              data,
                              feature_sizes,
                              n_components,
                              initial_bagging_only=True,
                              perc=1.0,
                              replace=True):
        """
        data is a numpy array
        """

        bagging = True if n_components > 1 else False

        if not bagging:
            initial_bagging_only = False

        tot_n_instances = data.shape[0]
        tot_n_features = data.shape[1]

        inst_compo_ratio = tot_n_instances / n_components

        logging.info('Learning SPN structure on a (%d X %d) dataset',
                     tot_n_instances, tot_n_features)
        learn_start_t = perf_counter()

        #
        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        building_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        # creating the first slice
        whole_slice = DataSlice.whole_slice(tot_n_instances,
                                            tot_n_features)
        slices_to_process.append(whole_slice)
        whole_slice.bagging = False

        first_run = True

        #
        # caching
        slice_cache = {}
        n_cached_objects = 0

        #
        # iteratively process & split slices
        #
        while slices_to_process:

            # process a slice
            current_slice = slices_to_process.popleft()

            cached = cache_data_slice(current_slice, slice_cache)
            if cached is not None:
                n_cached_objects += 1

            # pointers to the current data slice
            current_instances = current_slice.instance_ids
            current_features = current_slice.feature_ids
            current_id = current_slice.id

            n_instances = len(current_instances)
            n_features = len(current_features)

            logging.info('*** Processing slice %d (%d X %d)\n\t',
                         current_id,
                         n_instances, n_features)
            logging.debug('instances:%s\n\tfeatures:%s',
                          current_instances,
                          current_features)

            #
            # is this a leaf node or we can split?
            if n_features == 1:
                logging.info('---> Adding a leaf (just one feature)')

                (feature_id, ) = current_features
                feature_size = feature_sizes[feature_id]

                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                # create the node
                leaf_node = \
                    CategoricalSmoothedNode(var=feature_id,
                                            var_values=feature_size,
                                            data=current_slice_data,
                                            instances=current_instances,
                                            alpha=self._alpha)
                # storing links
                # input_nodes.append(leaf_node)
                leaf_node.id = current_id
                node_id_assoc[current_id] = leaf_node

                logging.debug('\tCreated Smooth Node %s', leaf_node)

            elif (n_instances <= self._min_instances_slice and n_features > 1):
                #
                # splitting the slice on each feature
                logging.info('---> Few instances (%d), decompose all features',
                             n_instances)
                #
                # shall put a cltree or
                if self._cltree_leaves:
                    logging.info('into a Chow-Liu tree')
                    #
                    # slicing data
                    slice_data_rows = data[current_instances, :]
                    current_slice_data = slice_data_rows[:, current_features]

                    current_feature_sizes = [feature_sizes[i]
                                             for i in current_features]
                    #
                    # creating a Chow-Liu tree as leaf
                    leaf_node = \
                        CLTreeNode(vars=current_features,
                                   var_values=current_feature_sizes,
                                   data=current_slice_data,
                                   alpha=self._alpha)
                    #
                    # storing links
                    leaf_node.id = current_id
                    node_id_assoc[current_id] = leaf_node

                    logging.debug('\tCreated Chow-Liu Tree Node %s', leaf_node)

                else:
                    logging.info('into a naive factorization')
                    #
                    # putting them in queue
                    child_slices = [DataSlice(current_instances, [feature_id])
                                    for feature_id in current_features]
                    slices_to_process.extend(child_slices)

                    children_ids = [child.id for child in child_slices]

                    #
                    # storing the children links
                    for child_slice in child_slices:
                        current_slice.add_child(child_slice)
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)

                    #
                    # creating the product node
                    prod_node = ProductNode(
                        var_scope=frozenset(current_features))
                    prod_node.id = current_id

                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

            else:

                #
                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                split_on_features = False

                #
                # first run is a split on rows
                if first_run:
                    logging.info('-- FIRST RUN --')
                    # first_run = False
                else:
                    #
                    # try clustering on cols
                    logging.debug('...trying to split on columns')
                    dependent_features, other_features = \
                        greedy_feature_split(data,
                                             current_slice,
                                             feature_sizes,
                                             self._g_factor,
                                             self._rand_gen)
                    if len(other_features) > 0:
                        split_on_features = True
                #
                # have dependent components been found?
                if split_on_features:
                    #
                    # splitting on columns
                    logging.info('---> Splitting on features')

                    #
                    # creating two new data slices and putting them on queue
                    first_slice = DataSlice(current_instances,
                                            dependent_features)
                    second_slice = DataSlice(current_instances,
                                             other_features)
                    slices_to_process.append(first_slice)
                    slices_to_process.append(second_slice)

                    children_ids = [first_slice.id, second_slice.id]

                    # first_slice.bagging = current_slice.bagging
                    # second_slice.bagging = current_slice.bagging

                    #
                    # storing link parent children
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)
                    current_slice.add_child(first_slice)
                    current_slice.add_child(second_slice)

                    #
                    # creating product node
                    prod_node = \
                        ProductNode(var_scope=frozenset(current_features))
                    prod_node.id = current_id
                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

                else:
                    #
                    # clustering on rows
                    logging.info('---> Splitting on rows')

                    #
                    # at most n_rows clusters, for sklearn
                    k_row_clusters = min(self._n_cluster_splits,
                                         n_instances - 1)

                    n_components = 1
                    if first_run:
                        first_run = False

                        #
                        # decreasing the number of components proportionally to
                        # the number of instances in the slice

                    n_components = max(
                        int(n_instances / inst_compo_ratio), 1)

                    # if n_instances < 50:
                    #     n_components = 10
                    # if not current_slice.bagging:
                    #     #
                    #     # random prob
                    #     if self._rand_gen.rand() > 0.5:
                    #         current_slice.bagging = True
                    #         n_components = 10

                    #
                    # creating many data slices
                    sampled_slices = []
                    for m in range(n_components):

                        logging.info('\t considering component: %d/%d',
                                     m + 1,
                                     n_components)
                        #
                        # sampling indices
                        sampled_instance_ids = None
                        if n_components > 1:
                            sampled_instance_ids = \
                                dataset.sample_indexes(current_instances,
                                                       perc=perc,
                                                       replace=replace,
                                                       rand_gen=self._rand_gen)
                        else:
                            sampled_instance_ids = current_instances

                        #
                        # creating new data slices (samp instances x current
                        # features)
                        sampled_data_slice = DataSlice(sampled_instance_ids,
                                                       current_features)
                        sampled_slices.append(sampled_data_slice)

                        #
                        # apply clustering on them
                        clustering = \
                            cluster_rows(data,
                                         sampled_data_slice,
                                         n_clusters=k_row_clusters,
                                         cluster_method=self._row_cluster_method,
                                         n_iters=self._n_iters,
                                         n_restarts=self._n_restarts,
                                         cluster_penalty=self._cluster_penalty,
                                         rand_gen=self._rand_gen,
                                         sklearn_args=self._sklearn_args)

                        # logging.debug('obtained clustering %s', clustering)

                        # splitting
                        cluster_slices = [DataSlice(cluster, current_features)
                                          for cluster in clustering]
                        cluster_slices_ids = [slice.id
                                              for slice in cluster_slices]
                        cluster_weights = [slice.n_instances() / n_instances
                                           for slice in cluster_slices]

                        #
                        # appending for processing
                        slices_to_process.extend(cluster_slices)

                        #
                        # storing links
                        sampled_data_slice.type = SumNode
                        sampled_id = sampled_data_slice.id
                        # building_stack.append(sampled_data_slice)
                        for child_slice, child_weight in zip(cluster_slices,
                                                             cluster_weights):
                            sampled_data_slice.add_child(child_slice,
                                                         child_weight)

                            # child_slice.bagging = current_slice.bagging

                        #
                        # building a sum node
                        sum_node = SumNode(
                            var_scope=frozenset(current_features))
                        sum_node.id = sampled_id
                        node_id_assoc[sampled_id] = sum_node
                        logging.debug('\tCreated Sum Node %s (with children %s)',
                                      sum_node,
                                      cluster_slices_ids)

                    #
                    # linking mixtures to original node
                    sampled_unif_weights = numpy.ones(n_components)
                    sampled_weights = (sampled_unif_weights /
                                       sampled_unif_weights.sum())

                    sampled_ids = [slice.id for slice in sampled_slices]

                    current_slice.type = SumNode
                    building_stack.append(current_slice)
                    building_stack.extend(sampled_slices)

                    for child_slice, child_weight in zip(sampled_slices,
                                                         sampled_weights):
                        current_slice.add_child(child_slice, child_weight)

                    #
                    # building a sum node
                    sum_node = SumNode(var_scope=frozenset(current_features))
                    sum_node.id = current_id
                    node_id_assoc[current_id] = sum_node
                    logging.debug('\tCreated Sum Node %s (with children %s)',
                                  sum_node,
                                  sampled_ids)

        learn_end_t = perf_counter()
        logging.info('Structure learned in %f secs',
                     (learn_end_t - learn_start_t))

        #
        # linking the spn graph (parent -> children)
        #
        logging.info('===> Building tree')

        link_start_t = perf_counter()
        root_build_node = building_stack[0]
        root_node = node_id_assoc[root_build_node.id]
        logging.debug('root node: %s', root_node)

        root_node = SpnFactory.pruned_spn_from_slices(node_id_assoc,
                                                      building_stack)
        link_end_t = perf_counter()
        logging.info('\tLinked the spn in %f secs (root_node %s)',
                     (link_end_t - link_start_t),
                     root_node)

        #
        # building layers
        #
        logging.info('===> Layering spn')
        layer_start_t = perf_counter()
        spn = SpnFactory.layered_linked_spn(root_node)
        layer_end_t = perf_counter()
        logging.info('\tLayered the spn in %f secs',
                     (layer_end_t - layer_start_t))

        logging.info('\nLearned SPN\n\n%s', spn.stats())

        logging.info('\ncached slices:%d\n', n_cached_objects)

        return spn


class RandomLearnSPN(object):

    """
    Implementing Gens and Domingos
    """

    def __init__(self,
                 min_instances_slice=100,
                 min_features_slice=0,
                 n_cluster_splits=2,
                 alpha=0.1,
                 cltree_leaves=False,
                 rand_gen=None):
        """
        WRITEME
        """
        self._min_instances_slice = min_instances_slice
        self._min_features_slice = min_features_slice
        self._alpha = alpha
        self._cltree_leaves = cltree_leaves
        self._n_cluster_splits = n_cluster_splits
        self._rand_gen = rand_gen if rand_gen is not None \
            else numpy.random.RandomState(RND_SEED)

        logging.info('RandLearnSPN:\n\tmin inst:%d\n' +
                     '\tmin feat:%d\n' +
                     '\talpha:%f\n\tn clusters:%d\n' +
                     '\tcluster method=%s\n\tn iters: %d\n' +
                     '\tn restarts: %d\n\tcltree leaves:%s\n',
                     self._min_instances_slice,
                     self._min_features_slice,
                     self._alpha,
                     self._cltree_leaves)

        #
        # resetting the data slice ids (just in case)
        DataSlice.reset_id_counter()

    def fit_structure(self,
                      data,
                      feature_sizes):
        """
        data is a numpy array
        """

        tot_n_instances = data.shape[0]
        tot_n_features = data.shape[1]

        logging.info('Learning Random SPN structure on a (%d X %d) dataset',
                     tot_n_instances, tot_n_features)
        learn_start_t = perf_counter()

        #
        # a queue containing the data slices to process
        slices_to_process = deque()

        # a stack for building nodes
        building_stack = deque()

        # a dict to keep track of id->nodes
        node_id_assoc = {}

        # creating the first slice
        whole_slice = DataSlice.whole_slice(tot_n_instances,
                                            tot_n_features)
        slices_to_process.append(whole_slice)

        first_run = True

        #
        # iteratively process & split slices
        #
        while slices_to_process:

            # process a slice
            current_slice = slices_to_process.popleft()

            # pointers to the current data slice
            current_instances = current_slice.instance_ids
            current_features = current_slice.feature_ids
            current_id = current_slice.id

            n_instances = len(current_instances)
            n_features = len(current_features)

            logging.info('\n*** Processing slice %d (%d X %d)',
                         current_id,
                         n_instances, n_features)
            logging.debug('\tinstances:%s\n\tfeatures:%s',
                          current_instances,
                          current_features)

            #
            # is this a leaf node or we can split?
            if n_features == 1:
                logging.info('---> Adding a leaf (just one feature)')

                (feature_id, ) = current_features
                feature_size = feature_sizes[feature_id]

                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                # create the node
                leaf_node = \
                    CategoricalSmoothedNode(var=feature_id,
                                            var_values=feature_size,
                                            data=current_slice_data,
                                            instances=current_instances,
                                            alpha=self._alpha)
                # storing links
                # input_nodes.append(leaf_node)
                leaf_node.id = current_id
                node_id_assoc[current_id] = leaf_node

                logging.debug('\tCreated Smooth Node %s', leaf_node)

            elif (n_instances <= self._min_instances_slice and n_features > 1):
                #
                # splitting the slice on each feature
                logging.info('---> Few instances (%d), decompose all features',
                             n_instances)
                #
                # shall put a cltree or
                if self._cltree_leaves:
                    logging.info('into a Chow-Liu tree')
                    #
                    # slicing data
                    slice_data_rows = data[current_instances, :]
                    current_slice_data = slice_data_rows[:, current_features]

                    current_feature_sizes = [feature_sizes[i]
                                             for i in current_features]
                    #
                    # creating a Chow-Liu tree as leaf
                    leaf_node = \
                        CLTreeNode(vars=current_features,
                                   var_values=current_feature_sizes,
                                   data=current_slice_data,
                                   alpha=self._alpha)
                    #
                    # storing links
                    leaf_node.id = current_id
                    node_id_assoc[current_id] = leaf_node

                    logging.debug('\tCreated Chow-Liu Tree Node %s', leaf_node)

                else:
                    logging.info('into a naive factorization')
                    #
                    # putting them in queue
                    child_slices = [DataSlice(current_instances, [feature_id])
                                    for feature_id in current_features]
                    slices_to_process.extend(child_slices)

                    children_ids = [child.id for child in child_slices]

                    #
                    # storing the children links
                    for child_slice in child_slices:
                        current_slice.add_child(child_slice)
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)

                    #
                    # creating the product node
                    prod_node = ProductNode(
                        var_scope=frozenset(current_features))
                    prod_node.id = current_id

                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

            else:

                #
                # slicing from the original dataset
                slice_data_rows = data[current_instances, :]
                current_slice_data = slice_data_rows[:, current_features]

                split_on_features = False

                #
                # first run is a split on rows
                if first_run:
                    logging.info('-- FIRST RUN --')
                    first_run = False
                else:
                    #
                    # try clustering on cols
                    logging.debug('...trying to split on columns')
                    # dependent_features, other_features = \
                    #     greedy_feature_split(data,
                    #                          current_slice,
                    #                          feature_sizes,
                    #                          self._g_factor,
                    #                          self._rand_gen)
                    # if len(other_features) > 0

                    # randomly choosing wheather to split on columns or not
                    if self._rand_gen.randint(2) > 0:
                        split_on_features = True
                #
                # have dependent components been found?
                if split_on_features:
                    #
                    # splitting on columns
                    logging.info('---> Splitting on features')

                    # shuffling features
                    self._rand_gen.shuffle(current_features)

                    # randomly selecting a cut point
                    feature_cut = self._rand_gen.randint(1, n_features)
                    dependent_features = current_features[:feature_cut]
                    other_features = current_features[feature_cut:]

                    #
                    # creating two new data slices and putting them on queue
                    first_slice = DataSlice(current_instances,
                                            dependent_features)
                    second_slice = DataSlice(current_instances,
                                             other_features)
                    slices_to_process.append(first_slice)
                    slices_to_process.append(second_slice)

                    children_ids = [first_slice.id, second_slice.id]

                    #
                    # storing link parent children
                    current_slice.type = ProductNode
                    building_stack.append(current_slice)
                    current_slice.add_child(first_slice)
                    current_slice.add_child(second_slice)

                    #
                    # creating product node
                    prod_node = \
                        ProductNode(var_scope=frozenset(current_features))
                    prod_node.id = current_id
                    node_id_assoc[current_id] = prod_node
                    logging.debug('\tCreated Prod Node %s (with children %s)',
                                  prod_node,
                                  children_ids)

                else:
                    #
                    # clustering on rows
                    logging.info('---> Splitting on rows')

                    #
                    # at most n_rows clusters, for sklearn
                    k_row_clusters = min(self._n_cluster_splits,
                                         n_instances - 1)

                    # shuffling instances
                    self._rand_gen.shuffle(current_instances)

                    # random clustering
                    clustering = []
                    remaining_instances = current_instances[:]
                    for i in range(k_row_clusters - 1):
                        n_rem_instances = len(remaining_instances)
                        instance_cut = self._rand_gen.randint(1,
                                                              n_rem_instances)
                        clustering.append(remaining_instances[:instance_cut])
                        remaining_instances = \
                            remaining_instances[instance_cut:]
                    clustering.append(remaining_instances)

                    # logging.debug('obtained clustering %s', clustering)
                    logging.info('clustered into %d parts (min %d)',
                                 len(clustering), k_row_clusters)
                    # splitting
                    cluster_slices = [DataSlice(cluster, current_features)
                                      for cluster in clustering]
                    cluster_slices_ids = [slice.id
                                          for slice in cluster_slices]
                    cluster_weights = [slice.n_instances() / n_instances
                                       for slice in cluster_slices]

                    #
                    # appending for processing
                    slices_to_process.extend(cluster_slices)

                    #
                    # storing links
                    # current_slice.children = cluster_slices_ids
                    # current_slice.weights = cluster_weights
                    current_slice.type = SumNode
                    building_stack.append(current_slice)
                    for child_slice, child_weight in zip(cluster_slices,
                                                         cluster_weights):
                        current_slice.add_child(child_slice, child_weight)

                    #
                    # building a sum node
                    sum_node = SumNode(var_scope=frozenset(current_features))
                    sum_node.id = current_id
                    node_id_assoc[current_id] = sum_node
                    logging.debug('\tCreated Sum Node %s (with children %s)',
                                  sum_node,
                                  cluster_slices_ids)

        learn_end_t = perf_counter()
        logging.info('Structure learned in %f secs',
                     (learn_end_t - learn_start_t))

        #
        # linking the spn graph (parent -> children)
        #
        logging.info('===> Building tree')

        link_start_t = perf_counter()
        root_build_node = building_stack[0]
        root_node = node_id_assoc[root_build_node.id]
        logging.debug('root node: %s', root_node)

        root_node = SpnFactory.pruned_spn_from_slices(node_id_assoc,
                                                      building_stack)
        link_end_t = perf_counter()
        logging.info('\tLinked the spn in %f secs (root_node %s)',
                     (link_end_t - link_start_t),
                     root_node)

        #
        # building layers
        #
        logging.info('===> Layering spn')
        layer_start_t = perf_counter()
        spn = SpnFactory.layered_linked_spn(root_node)
        layer_end_t = perf_counter()
        logging.info('\tLayered the spn in %f secs',
                     (layer_end_t - layer_start_t))

        logging.info('\nLearned SPN\n\n%s', spn.stats())

        # print(spn)

        return spn
