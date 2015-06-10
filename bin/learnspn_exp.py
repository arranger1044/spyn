import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset

import numpy
from numpy.testing import assert_almost_equal

import random

import datetime

import os

import logging

from algo.learnspn import LearnSPN

from spn import NEG_INF
from spn.utils import stats_format

#########################################
# creating the opt parser
parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str, nargs=1,
                    help='Specify a dataset name from data/ (es. nltcs)')

parser.add_argument('-k', '--n-row-clusters', type=int, nargs='?',
                    default=2,
                    help='Number of clusters to split rows into' +
                    ' (for DPGMM it is the max num of clusters)')

parser.add_argument('-c', '--cluster-method', type=str, nargs='?',
                    default='GMM',
                    help='Cluster method to apply on rows' +
                    ' ["GMM"|"DPGMM"|"HOEM"]')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/learnspn-b/',
                    help='Output dir path')


parser.add_argument('-g', '--g-factor', type=float, nargs='+',
                    default=[1.0],
                    help='The "p-value like" for G-Test on columns')

parser.add_argument('-i', '--n-iters', type=int, nargs='?',
                    default=100,
                    help='Number of iterates for the row clustering algo')

parser.add_argument('-r', '--n-restarts', type=int, nargs='?',
                    default=4,
                    help='Number of restarts for the row clustering algo' +
                    ' (only for GMM)')

parser.add_argument('-p', '--cluster-penalty', type=float, nargs='+',
                    default=[1.0],
                    help='Penalty for the cluster number' +
                    ' (i.e. alpha in DPGMM and rho in HOEM, not used in GMM)')

parser.add_argument('-s', '--sklearn-args', type=str, nargs='?',
                    help='Additional sklearn parameters in the for of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('-m', '--min-inst-slice', type=int, nargs='+',
                    default=[50],
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[0.1],
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('--clt-leaves', action='store_true',
                    help='Whether to use Chow-Liu trees as leaves')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')
#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)
# I shall print here all the stats

#
# gathering parameters
alphas = args.alpha
min_inst_slices = args.min_inst_slice
g_factors = args.g_factor
cluster_penalties = args.cluster_penalty

cltree_leaves = args.clt_leaves

sklearn_args = None
if args.sklearn_args is not None:
    sklearn_key_value_pairs = args.sklearn_args.translate(
        {ord('['): '', ord(']'): ''}).split(',')
    sklearn_args = {key.strip(): value.strip() for key, value in
                    [pair.strip().split('=')
                     for pair in sklearn_key_value_pairs]}
else:
    sklearn_args = {}
logging.info(sklearn_args)

# initing the random generators
seed = args.seed
MAX_RAND_SEED = 99999999  # sys.maxsize
rand_gen = random.Random(seed)
numpy_rand_gen = numpy.random.RandomState(seed)

#
# elaborating the dataset
#
logging.info('Loading datasets: %s', args.dataset)
(dataset_name,) = args.dataset
train, valid, test = dataset.load_train_val_test_csvs(dataset_name)
n_instances = train.shape[0]
n_test_instances = test.shape[0]
#
# estimating the frequencies for the features
logging.info('Estimating features on training set...')
freqs, features = dataset.data_2_freqs(train)


#
# Opening the file for test prediction
#
logging.info('Opening log file...')
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = args.output + dataset_name + '_' + date_string
out_log_path = out_path + '/exp.log'
test_lls_path = out_path + '/test.lls'

#
# creating dir if non-existant
if not os.path.exists(os.path.dirname(out_log_path)):
    os.makedirs(os.path.dirname(out_log_path))

best_valid_avg_ll = NEG_INF
best_state = {}
best_test_lls = None

preamble = ("""g-factor:\tclu-pen:\tmin-ins:\talpha:\tn_edges:""" +
            """\tn_levels:\tn_weights:\tn_leaves:""" +
            """\ttrain_ll\tvalid_ll:\ttest_ll\n""")

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.write(preamble)
    out_log.flush()
    #
    # looping over all parameters combinations
    for g_factor in g_factors:
        for cluster_penalty in cluster_penalties:
            for min_inst_slice in min_inst_slices:

                #
                # Creating the structure learner
                learner = LearnSPN(g_factor=g_factor,
                                   min_instances_slice=min_inst_slice,
                                   # alpha=alpha,
                                   row_cluster_method=args.cluster_method,
                                   cluster_penalty=cluster_penalty,
                                   n_cluster_splits=args.n_row_clusters,
                                   n_iters=args.n_iters,
                                   n_restarts=args.n_restarts,
                                   sklearn_args=sklearn_args,
                                   cltree_leaves=cltree_leaves,
                                   rand_gen=numpy_rand_gen)

                learn_start_t = perf_counter()

                #
                # build an spn on the training set
                spn = learner.fit_structure(data=train,
                                            feature_sizes=features)
                # spn = learner.fit_structure_bagging(data=train,
                #                                     feature_sizes=features,
                #                                     n_components=10)

                learn_end_t = perf_counter()
                print('Structure learned in', learn_end_t - learn_start_t,
                      'secs')

                #
                # print(spn)

                #
                # gathering statistics
                n_edges = spn.n_edges()
                n_levels = spn.n_layers()
                n_weights = spn.n_weights()
                n_leaves = spn.n_leaves()

                #
                # smoothing can be done after the spn has been built
                for alpha in alphas:
                    logging.info('Smoothing leaves with alpha = %f', alpha)
                    spn.smooth_leaves(alpha)

                    #
                    # Compute LL on training set
                    logging.info('Evaluating on training set')
                    train_ll = 0.0
                    for instance in train:
                        (pred_ll, ) = spn.eval(instance)
                        train_ll += pred_ll
                    train_avg_ll = train_ll / train.shape[0]

                    #
                    # Compute LL on validation set
                    logging.info('Evaluating on validation set')
                    valid_ll = 0.0
                    for instance in valid:
                        (pred_ll, ) = spn.eval(instance)
                        valid_ll += pred_ll
                    valid_avg_ll = valid_ll / valid.shape[0]

                    #
                    # Compute LL on test set
                    test_lls = numpy.zeros(test.shape[0])
                    logging.info('Evaluating on test set')
                    test_ll = 0.0
                    for i, instance in enumerate(test):
                        (pred_ll, ) = spn.eval(instance)
                        test_ll += pred_ll
                        test_lls[i] = pred_ll
                    test_avg_ll = test_ll / test.shape[0]

                    #
                    # updating best stats according to valid ll
                    if valid_avg_ll > best_valid_avg_ll:
                        best_valid_avg_ll = valid_avg_ll
                        best_state['alpha'] = alpha
                        best_state['min-inst-slice'] = min_inst_slice
                        best_state['g-factor'] = g_factor
                        best_state['cluster-penalty'] = cluster_penalty
                        best_state['train_ll'] = train_avg_ll
                        best_state['valid_ll'] = valid_avg_ll
                        best_state['test_ll'] = test_avg_ll
                        best_test_lls = test_lls

                    #
                    # writing to file a line for the grid
                    stats = stats_format([g_factor,
                                          cluster_penalty,
                                          min_inst_slice,
                                          alpha,
                                          n_edges, n_levels,
                                          n_weights, n_leaves,
                                          train_avg_ll,
                                          valid_avg_ll,
                                          test_avg_ll],
                                         '\t',
                                         digits=5)
                    out_log.write(stats + '\n')
                    out_log.flush()

    #
    # writing as last line the best params
    out_log.write("{0}".format(best_state))
    out_log.flush()

    #
    # saving the best test_lls
    assert_almost_equal(best_state['test_ll'], best_test_lls.mean())
    numpy.savetxt(test_lls_path, best_test_lls, delimiter='\n')


logging.info('Grid search ended.')
logging.info('Best params:\n\t%s', best_state)
