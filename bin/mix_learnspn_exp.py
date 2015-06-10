import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

import dataset

import numpy
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal

# import random

import datetime

import os

import logging

from scipy.misc import logsumexp

from algo.learnspn import LearnSPN
from algo import sgd

from spn import NEG_INF
from spn.utils import stats_format
from spn.utils import sample_all_worlds

import visualize

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
                    default='./exp/mix-learnspn-f/',
                    help='Output dir path')

parser.add_argument('-n', '--n-mixtures', type=int, nargs='?',
                    default=2,
                    help='Number of mixture components to test')

parser.add_argument('-g', '--g-factor', type=float, nargs='+',
                    default=[1.0],
                    help='The "p-value like" for G-Test on columns')

parser.add_argument('-i', '--n-iters', type=int, nargs='?',
                    default=100,
                    help='Number of iterates for the row clustering algo')

parser.add_argument('-r', '--n-restarts', type=int, nargs='?',
                    default=3,
                    help='Number of restarts for the row clustering algo' +
                    ' (only for GMM)')

parser.add_argument('-p', '--cluster-penalty', type=float, nargs='+',
                    default=[1.0],
                    help='Penalty for the cluster number' +
                    ' (i.e. alpha in DPGMM and rho in HOEM, not used in GMM)')

parser.add_argument('-s', '--sklearn-args', type=str, nargs='?',
                    help='Additional sklearn params in the form of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('-q', '--sgd-args', type=str, nargs='?',
                    help='Additional sgd parameters in the form of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('-m', '--min-inst-slice', type=int, nargs='+',
                    default=[50],
                    help='Min number of instances in a slice to split by cols')

parser.add_argument('-a', '--alpha', type=float, nargs='+',
                    default=[0.1],
                    help='Smoothing factor for leaf probability estimation')

parser.add_argument('--clt-leaves', action='store_true',
                    help='Whether to use Chow-Liu trees as leaves')

parser.add_argument('-b', '--bootstrap-ids', type=str, nargs='?',
                    default=None,
                    help='Path to boostrap ids file (numpy matrix '
                    'n_samples X (n_instances*perc) serialized')

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
n_mix = args.n_mixtures

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

sgd_args = None
if args.sgd_args is not None:
    sgd_key_value_pairs = args.sgd_args.translate(
        {ord('['): '', ord(']'): ''}).split(',')
    sgd_args = {key.strip(): value.strip() for key, value in
                [pair.strip().split('=')
                 for pair in sgd_key_value_pairs]}
else:
    sgd_args = {}

#
# extracting default parameters for sgd
sgd_n_epochs = int(sgd_args['n_epochs']) if 'n_epochs' in sgd_args else 10
sgd_reg = sgd_args['regularization'] if 'regularization' in sgd_args else None
sgd_reg_l1 = float(
    sgd_args['l1_reg_coef']) if 'l1_reg_coef' in sgd_args else 1.0
sgd_lr = float(
    sgd_args['learning_rate']) if 'learning_rate' in sgd_args else 0.1

logging.info(sgd_args)

# initing the random generators
seed = args.seed
MAX_RAND_SEED = 99999999  # sys.maxsize
# rand_gen = random.Random(seed)
numpy_rand_gen = numpy.random.RandomState(seed)

#
# these are constants ftm
perc = 1.0
replace = True


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

probability_dist_check = False

#
# Opening the file for test prediction
#
logging.info('Opening log file...')
date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_path = args.output + dataset_name + '_' + date_string
train_out_log_path = out_path + '/train.m.log'
valid_out_log_path = out_path + '/valid.m.log'
test_out_log_path = out_path + '/test.m.log'
out_log_path = out_path + '/curves.log'
out_png_path = out_path + '/curves.png'
out_comp_path = out_path + '/n.components.log'
out_comp_png_path = out_path + '/n.components.png'
out_comp_log_path = out_path + '/components.log'
test_lls_path = out_path + '/test.lls'
stats_log_path = out_path + '/mix.stats'
exp_log_path = out_path + '/exp.log'

#
# creating dir if non-existant
if not os.path.exists(os.path.dirname(train_out_log_path)):
    os.makedirs(os.path.dirname(train_out_log_path))
if not os.path.exists(os.path.dirname(valid_out_log_path)):
    os.makedirs(os.path.dirname(valid_out_log_path))
if not os.path.exists(os.path.dirname(test_out_log_path)):
    os.makedirs(os.path.dirname(test_out_log_path))


#
# getting bootstrap samples if necessary
logging.info('Generating bootstrap samples...')
bootstrap_ids = None
if args.bootstrap_ids is not None:
    #
    # reading file as array
    bootstrap_ids = numpy.loadtxt(args.bootstrap_ids, dtype=int,
                                  delimiter=',')
    logging.info('Reading them from file')
else:
    #
    # generating them
    bootstrap_ids = numpy.zeros((n_mix, n_instances), dtype=int)
    for m in range(n_mix):
        train_mix = dataset.sample_indexes(numpy.arange(n_instances),
                                           perc=perc,
                                           replace=replace,
                                           rand_gen=numpy_rand_gen)
        bootstrap_ids[m, :] = train_mix
        assert len(bootstrap_ids[m, :]) == n_instances

    #
    # resetting the generator to enhance reproducibility
    numpy_rand_gen = numpy.random.RandomState(seed)

    #
    # saving it, just in case
    bootstrap_ids_path = args.output + dataset_name + '.bootstrap.ids'
    logging.info('Saving boostrap ids to %s', bootstrap_ids_path)
    numpy.savetxt(bootstrap_ids_path, bootstrap_ids,
                  delimiter=',', fmt='%d')

    # #
    # # re-read it
    # bootstrap_ids_r = bootstrap_ids
    # bootstrap_ids = numpy.loadtxt(bootstrap_ids_path, dtype=int,
    #                               delimiter=',')
    # logging.info('Read it again')
    # assert_array_equal(bootstrap_ids_r, bootstrap_ids)


logging.info('Bootstrap ids %s', bootstrap_ids.shape)

best_valid_avg_ll = NEG_INF
best_state_mix = {}

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

            #
            # learning a mixture

            spns = \
                learner.fit_mixture_bootstrap(train,
                                              n_mix_components=n_mix,
                                              bootstrap_samples_ids=bootstrap_ids,
                                              feature_sizes=features,
                                              perc=perc,
                                              replace=replace,
                                              evaluate=False)

            all_worlds = None
            if probability_dist_check:
                all_worlds = sample_all_worlds(train.shape[1])

            #
            # gathering statistics
            n_edges = numpy.zeros(n_mix, dtype=int)
            n_levels = numpy.zeros(n_mix, dtype=int)
            n_weights = numpy.zeros(n_mix, dtype=int)
            n_leaves = numpy.zeros(n_mix, dtype=int)

            train_ll_history = numpy.zeros(n_mix)
            valid_ll_history = numpy.zeros(n_mix)
            test_ll_history = numpy.zeros(n_mix)

            train_ll_m_history = numpy.zeros(n_mix)
            valid_ll_m_history = numpy.zeros(n_mix)
            test_ll_m_history = numpy.zeros(n_mix)

            # train_ll_wm_history = numpy.zeros(n_mix)
            # valid_ll_wm_history = numpy.zeros(n_mix)
            # test_ll_wm_history = numpy.zeros(n_mix)

            train_ll_sgd_m_history = numpy.zeros(n_mix)
            valid_ll_sgd_m_history = numpy.zeros(n_mix)
            test_ll_sgd_m_history = numpy.zeros(n_mix)

            n_components_history = numpy.zeros(n_mix, dtype=int)

            # mixture_weights = numpy.zeros(n_mix)

            x_axis = [i for i in range(1, n_mix + 1)]
            #
            # smoothing can be done after the spn has been built
            for alpha in alphas:
                logging.debug('Smoothing leaves with alpha = %f', alpha)

                #
                # file for the components
                comp_log = open(out_comp_log_path, 'w')

                #
                # allocating matrices for storing lls
                train_mixture_lls = numpy.zeros((train.shape[0], n_mix))
                valid_mixture_lls = numpy.zeros((valid.shape[0], n_mix))
                test_mixture_lls = numpy.zeros((test.shape[0], n_mix))

                world_mixture_lls = None
                if probability_dist_check:
                    world_mixture_lls = numpy.zeros((all_worlds.shape[0],
                                                     n_mix))
                #
                # doing inference for each spn
                for m, spn in enumerate(spns):
                    logging.info('Considering component %d', m)

                    #
                    # gathering statistics
                    n_edges[m] = spn.n_edges()
                    n_levels[m] = spn.n_layers()
                    n_weights[m] = spn.n_weights()
                    n_leaves[m] = spn.n_leaves()

                    #
                    # smoothing to alpha
                    spn.smooth_leaves(alpha)

                    #
                    # Compute LL on training set
                    logging.info('Evaluating on training set')
                    train_ll = 0.0
                    for i, instance in enumerate(train):
                        (pred_ll, ) = spn.eval(instance)
                        train_mixture_lls[i, m] = pred_ll
                        train_ll += pred_ll
                    train_avg_ll = train_ll / train.shape[0]

                    #
                    # Compute LL on validation set
                    logging.info('Evaluating on validation set')
                    valid_ll = 0.0
                    for i, instance in enumerate(valid):
                        (pred_ll, ) = spn.eval(instance)
                        valid_mixture_lls[i, m] = pred_ll
                        valid_ll += pred_ll
                    valid_avg_ll = valid_ll / valid.shape[0]

                    #
                    # Compute LL on test set
                    logging.info('Evaluating on test set')
                    test_ll = 0.0
                    for i, instance in enumerate(test):
                        (pred_ll, ) = spn.eval(instance)
                        test_mixture_lls[i, m] = pred_ll
                        test_ll += pred_ll
                    test_avg_ll = test_ll / test.shape[0]

                    train_ll_history[m] = train_avg_ll
                    valid_ll_history[m] = valid_avg_ll
                    test_ll_history[m] = test_avg_ll

                    #
                    #
                    # asserting the correctness of computations so far
                    assert_almost_equal(train_mixture_lls[:, m].mean(),
                                        train_avg_ll)
                    assert_almost_equal(valid_mixture_lls[:, m].mean(),
                                        valid_avg_ll)
                    assert_almost_equal(test_mixture_lls[:, m].mean(),
                                        test_avg_ll)

                    #
                    # now computing the values for the mixture so far
                    weights = numpy.array([1 for i in range(m + 1)])
                    weights = weights / weights.sum()
                    print('weights', weights)

                    # #
                    # # weights proportional to the ll on the training set
                    # ll_weights = valid_ll_history[:m + 1]
                    # probs_weights = numpy.exp(ll_weights)
                    # probs_weights = probs_weights / probs_weights.sum()
                    # print('ll weights normalized', ll_weights, probs_weights)

                    log_weights = numpy.log(weights)

                    # log_probs_weights = numpy.log(probs_weights)

                    train_w_ll = logsumexp(train_mixture_lls[:, :m + 1] +
                                           log_weights,
                                           axis=1).mean()
                    valid_w_ll = logsumexp(valid_mixture_lls[:, :m + 1] +
                                           log_weights,
                                           axis=1).mean()
                    test_w_lls = logsumexp(test_mixture_lls[:, :m + 1] +
                                           log_weights,
                                           axis=1)
                    test_w_ll = test_w_lls.mean()

                    #
                    # saving the best result for now
                    if valid_w_ll > best_valid_avg_ll:
                        best_valid_avg_ll = valid_w_ll

                        best_state_mix['train_ll'] = train_w_ll
                        best_state_mix['valid_ll'] = valid_w_ll
                        best_state_mix['test_ll'] = test_w_ll
                        best_state_mix['n_mix'] = m + 1
                        best_state_mix['n_edges'] = n_edges[:m + 1].sum()
                        best_state_mix['n_levels'] = n_levels[:m + 1].max()
                        best_state_mix['n_weights'] = n_weights[:m + 1].sum()
                        best_state_mix['n_leaves'] = n_leaves[:m + 1].sum()

                        numpy.savetxt(test_lls_path,
                                      test_w_lls,
                                      delimiter='\n')

                    # train_wm_ll = logsumexp(train_mixture_lls[:, :m + 1] +
                    #                         log_probs_weights,
                    #                         axis=1).mean()
                    # valid_wm_ll = logsumexp(valid_mixture_lls[:, :m + 1] +
                    #                         log_probs_weights,
                    #                         axis=1).mean()
                    # test_wm_ll = logsumexp(test_mixture_lls[:, :m + 1] +
                    #                        log_probs_weights,
                    #                        axis=1).mean()

                    #
                    # and storing them into the arrays
                    train_ll_m_history[m] = train_w_ll
                    valid_ll_m_history[m] = valid_w_ll
                    test_ll_m_history[m] = test_w_ll

                    # train_ll_wm_history[m] = train_wm_ll
                    # valid_ll_wm_history[m] = valid_wm_ll
                    # test_ll_wm_history[m] = test_wm_ll

                    #
                    # now I can try the sgd, just in case
                    best_state_sgd = \
                        sgd.simple_sgd(train=train_mixture_lls[:, :m + 1],
                                       valid=valid_mixture_lls[:, :m + 1],
                                       test=test_mixture_lls[:, :m + 1],
                                       rand_gen=numpy_rand_gen,
                                       n_epochs=sgd_n_epochs,
                                       regularization=sgd_reg,
                                       reg_l1_coeff=sgd_reg_l1,
                                       learning_rate=sgd_lr)

                    # all other params are defaulted for now
                    train_ll_sgd_m_history[m] = best_state_sgd['best_train_ll']
                    valid_ll_sgd_m_history[m] = best_state_sgd['best_valid_ll']
                    test_ll_sgd_m_history[m] = best_state_sgd['best_test_ll']

                    best_params = best_state_sgd['best_params']
                    comp_log.write(numpy.array_str(best_params))
                    comp_log.write('\n')
                    comp_log.flush()

                    # print(best_params.nonzero()[0])
                    n_components_history[m] = numpy.count_nonzero(best_params)

                    if probability_dist_check:
                        print('Evaluating on all worlds')
                        #
                        #
                        # are these probability distributions?

                        world_ll = 0.0
                        for i, world in enumerate(all_worlds):
                            (pred_ll, ) = spn.eval(world)
                            world_mixture_lls[i, m] = pred_ll
                            world_ll += pred_ll

                        world_w_lls = logsumexp(world_mixture_lls[:, :m + 1] +
                                                log_weights,
                                                axis=1)
                        world_prob_lls = numpy.exp(world_w_lls)
                        world_prob_lls2 = (
                            weights * numpy.exp(world_mixture_lls[:, :m + 1])).sum(axis=1)
                        print('\n\n*************************************\n')
                        print('wpl', world_prob_lls.shape)
                        print('Sum over all worlds is', world_prob_lls.sum(),
                              world_prob_lls.max(),
                              world_prob_lls.min())
                        print('wxd', world_prob_lls2.shape)
                        print('SUm2', world_prob_lls2.sum())

                #
                # storing them somewhere
                numpy.savetxt(train_out_log_path, train_mixture_lls,
                              delimiter=',', fmt='%.8e')
                numpy.savetxt(valid_out_log_path, valid_mixture_lls,
                              delimiter=',', fmt='%.8e')
                numpy.savetxt(test_out_log_path, test_mixture_lls,
                              delimiter=',', fmt='%.8e')
                ll_m_history = numpy.vstack((train_ll_m_history,
                                             valid_ll_m_history,
                                             test_ll_m_history,
                                             train_ll_sgd_m_history,
                                             valid_ll_sgd_m_history,
                                             test_ll_sgd_m_history
                                             # ,
                                             # train_ll_wm_history,
                                             # valid_ll_wm_history,
                                             # test_ll_wm_history
                                             ))
                numpy.savetxt(out_log_path, ll_m_history,
                              delimiter=',', fmt='%.8e')
                numpy.savetxt(out_comp_path, n_components_history,
                              delimiter=',', fmt='%d')

                comp_log.close()

                numpy.savetxt(stats_log_path,
                              numpy.vstack((n_edges,
                                            n_levels,
                                            n_weights,
                                            n_leaves)),
                              fmt='%d')
                #
                # now visualizing
                visualize.visualize_curves([(x_axis, train_ll_m_history),
                                            (x_axis, valid_ll_m_history),
                                            (x_axis, test_ll_m_history),
                                            (x_axis, -train_ll_sgd_m_history),
                                            (x_axis, -valid_ll_sgd_m_history),
                                            (x_axis, -test_ll_sgd_m_history)
                                            # ,
                                            # (x_axis, train_ll_wm_history),
                                            # (x_axis, valid_ll_wm_history),
                                            # (x_axis, test_ll_wm_history)
                                            ],
                                           labels=['train', 'valid', 'test',
                                                   'train-sgd', 'valid-sgd',
                                                   'test-sgd'
                                                   # , 'train-wm',
                                                   # 'valid-wm', 'test-wm'
                                                   ],
                                           linestyles=['-', '-', '-',
                                                       ':', ':', ':'
                                                       # ,
                                                       # '-', '-', '-'
                                                       ],
                                           output=out_png_path)

                #
                # visualizing the number of components
                visualize.visualize_curves([(x_axis, n_components_history)],
                                           labels=['n components'],
                                           output=out_comp_png_path)

                # if valid_avg_ll > best_valid_avg_ll:
                #     best_valid_avg_ll = valid_avg_ll
                #     best_state['alpha'] = alpha
                #     best_state['min-inst-slice'] = min_inst_slice
                #     best_state['g-factor'] = g_factor
                #     best_state['cluster-penalty'] = cluster_penalty
                #     best_state['train_ll'] = train_avg_ll
                #     best_state['valid_ll'] = valid_avg_ll
                #     best_state['test_ll'] = test_avg_ll

                #
                # writing to file a line for the grid
                # stats = stats_format([g_factor,
                #                       cluster_penalty,
                #                       min_inst_slice,
                #                       alpha,
                #                       n_edges, n_levels,
                #                       n_weights, n_leaves,
                #                       train_avg_ll,
                #                       valid_avg_ll,
                #                       test_avg_ll],
                #                      '\t',
                #                      digits=5)
                # out_log.write(stats + '\n')
                # out_log.flush()

#
# writing as last line the best params
# out_log.write("{0}".format(best_state))
# out_log.flush()

logging.info('Grid search ended.')
logging.info('Best params:\n\t%s', best_state_mix)
with open(exp_log_path, 'w') as exp_log:
    exp_log.write("{0}".format(best_state_mix))
