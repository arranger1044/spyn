from algo import sgd

import numpy
from numpy.testing import assert_almost_equal

from scipy.misc import logsumexp

import theano

import logging

import dataset


def test_simple_avg_ll():
    #
    # synth mini dataset
    data = numpy.array([[0.2, 0.3, 0.1, 0.22],
                        [0.1, 0.5, 0.6, 0.3],
                        [0.8, 0.88, 0.1, 0.2],
                        [0.1, 0.4, 0.7, 0.1],
                        [0.5, 0.55, 0.54, 0.32],
                        [0.3, 0.4, 0.1, 0.9]])
    data_log = numpy.log(data)  # lol

    weights = numpy.array([0.2, 0.4, 0.3, 0.1])

    #
    # computing the ll by hand
    lse = logsumexp(data_log + numpy.log(weights), axis=1)

    print('numpy lse', lse)

    avg_ll = numpy.mean(lse)
    print('numpy avg ll', avg_ll)

    #
    # now with theano
    D = theano.shared(data_log, 'Data')
    W = theano.shared(weights, 'Weights')
    avg_ll_th = sgd.simple_avg_ll(D, W)
    avg_ll_func = theano.function([], avg_ll_th)

    avg_ll_2 = avg_ll_func()
    print('theano avg ll', avg_ll_2)

    assert_almost_equal(avg_ll, avg_ll_2)

    #
    # doing the same with the negative version
    neg_avg_ll_th = sgd.simple_avg_negative_ll(D, W)
    neg_avg_ll_func = theano.function([], neg_avg_ll_th)

    neg_avg_ll = neg_avg_ll_func()
    print('neg avg ll', neg_avg_ll)

    assert_almost_equal(neg_avg_ll, -avg_ll)


def test_training_function():
    #
    # synth mini dataset
    data = numpy.array([[0.2, 0.3, 0.1, 0.22],
                        [0.1, 0.5, 0.6, 0.3],
                        [0.8, 0.88, 0.1, 0.2],
                        [0.1, 0.4, 0.7, 0.1],
                        [0.5, 0.55, 0.54, 0.32],
                        [0.3, 0.4, 0.1, 0.9]])

    data_log = numpy.log(data)  # lol

    weights = numpy.array([0.2, 0.4, 0.3, 0.1])

    #
    # computing the ll by hand
    lse = logsumexp(data_log + numpy.log(weights), axis=1)
    avg_ll = numpy.mean(lse)
    print('numpy avg ll', avg_ll)

    learning_rate = 0.1
    tr_func = sgd.training_function(sgd.simple_avg_negative_ll,
                                    weights,
                                    learning_rate=learning_rate)

    cost, W_ = tr_func(data_log)
    print('theano cost', cost)

    #
    # asserting equality
    assert_almost_equal(-avg_ll, cost)

    print('new parameters', W_)


def test_simple_sgd():

    logging.basicConfig(level=logging.DEBUG)

    #
    # creating a synth dataset
    n_instances = 105
    n_features = 10

    rand_gen = numpy.random.RandomState(1337)
    train = rand_gen.random_sample((n_instances, n_features))
    train_log = numpy.log(train)

    batch_size = None
    learning_rate = 0.1
    sgd.simple_sgd(train_log,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   rand_gen=rand_gen)


def test_mixture_weight_init():
    train_m_file = 'nltcs_2015-01-29_18-39-06/train.m.log'
    valid_m_file = 'nltcs_2015-01-29_18-39-06/valid.m.log'
    test_m_file = 'nltcs_2015-01-29_18-39-06/test.m.log'

    logging.basicConfig(level=logging.DEBUG)

    train = dataset.csv_2_numpy(train_m_file,
                                path='',
                                type='float32')
    valid = dataset.csv_2_numpy(valid_m_file,
                                path='',
                                type='float32')
    test = dataset.csv_2_numpy(test_m_file,
                               path='',
                               type='float32')

    k_components = train.shape[1]
    unif_weights = numpy.array([1 for i in range(k_components)])
    unif_weights = unif_weights / unif_weights.sum()

    rand_weights = numpy.random.rand(k_components)
    rand_weights = rand_weights / rand_weights.sum()

    unif_mixture = logsumexp(train + numpy.log(unif_weights), axis=1).mean()
    rand_mixture = logsumexp(train + numpy.log(rand_weights), axis=1).mean()

    print('UNIF W LL', unif_mixture)
    print('RAND W LL', rand_mixture)


def test_simple_sgd_from_file():
    train_m_file = 'nltcs_2015-01-29_18-39-06/train.m.log'
    valid_m_file = 'nltcs_2015-01-29_18-39-06/valid.m.log'
    test_m_file = 'nltcs_2015-01-29_18-39-06/test.m.log'

    logging.basicConfig(level=logging.DEBUG)

    train = dataset.csv_2_numpy(train_m_file,
                                path='',
                                type='float32')
    valid = dataset.csv_2_numpy(valid_m_file,
                                path='',
                                type='float32')
    test = dataset.csv_2_numpy(test_m_file,
                               path='',
                               type='float32')

    batch_size = None
    learning_rate = 0.1
    uniform = False
    normalize = False
    uncons_init = False
    best_state = sgd.simple_sgd(train,
                                valid=valid,
                                test=test,
                                loss=sgd.simple_avg_negative_ll,
                                regularization='L1',
                                # loss=sgd.uncons_avg_negative_ll,
                                uniform_init=uniform,
                                normalize=normalize,
                                # uncons_init=uncons_init,
                                learning_rate=learning_rate,
                                batch_size=batch_size)

    print('TRAIN:', best_state['best_train_ll'])
    print('VALID:', best_state['best_valid_ll'])
    print('TEST:', best_state['best_test_ll'])
    print('PARAMS:', best_state['best_params'])
