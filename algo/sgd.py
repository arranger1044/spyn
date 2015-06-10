import numpy

import theano

from spn import LOG_ZERO
from spn import RND_SEED

import logging

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from scipy.misc import logsumexp


def simple_avg_ll(data_batch, weights):
    """
    data_batch in {n_instances X n_features} ALREADY IN LOG FORM
    weights = {n_features X 1}
    """

    # weights = theano.printing.Print('weights')(weights)

    log_weights = theano.tensor.switch(theano.tensor.eq(weights, 0.),
                                       LOG_ZERO,
                                       theano.tensor.log(weights))
    # log_weights = theano.printing.Print('PL')(log_weights)
    w_batch = log_weights.dimshuffle('x', 0) + data_batch
    sum_exp_batch = theano.tensor.sum(theano.tensor.exp(w_batch), axis=1)
    #
    # preventing logarithms to be zero, could be useless
    log_sum_exp = theano.tensor.switch(theano.tensor.eq(sum_exp_batch, 0.),
                                       LOG_ZERO,
                                       theano.tensor.log(sum_exp_batch))
    # log_sum_exp = theano.tensor.log(sum_exp_batch)
    # log_sum_exp = theano.printing.Print('lse')(log_sum_exp)

    ll = theano.tensor.mean(log_sum_exp)
    return ll


def uncons_avg_ll(data_batch, weights):
    """
    data_batch in {n_instances X n_features} ALREADY IN LOG FORM
    weights = {n_features X 1}

    weights can vary here
    """
    #
    # making a softmax
    exp_weights = theano.tensor.exp(weights)
    soft_weights = exp_weights / exp_weights.sum()
    # soft_weights = theano.tensor.nnet.softmax(weights)

    #
    # translated to probabilities
    w_batch = theano.tensor.log(soft_weights).dimshuffle('x', 0) + data_batch
    sum_exp_batch = theano.tensor.sum(theano.tensor.exp(w_batch), axis=1)
    #
    # preventing logarithms to be zero, could be useless
    log_sum_exp = theano.tensor.switch(theano.tensor.eq(sum_exp_batch, 0),
                                       LOG_ZERO,
                                       theano.tensor.log(sum_exp_batch))
    # log_sum_exp = theano.tensor.log(sum_exp_batch)

    ll = theano.tensor.mean(log_sum_exp)
    return ll


def uncons_avg_negative_ll(data_batch, weights):
    """
    WRITEME
    """
    return - uncons_avg_ll(data_batch, weights)


def simple_avg_negative_ll(data_batch, weights):
    """
    WRITEME
    """
    return - simple_avg_ll(data_batch, weights)


def simple_avg_negative_ll_l1(data_batch, weights):
    """
    Adding a L1 norm
    """
    L1 = theano.tensor.sum(abs(weights))
    return simple_avg_negative_ll(data_batch, weights) + L1


def training_function(loss,
                      W,
                      regularization=None,
                      normalize=True,
                      pos_projection=True,
                      learning_rate=0.1,
                      reg_l1_coeff=None):
    """
    loss = python function
    W = theano var
    regularization = None | L1
    """

    #
    # input
    batch_input = theano.tensor.matrix()

    #
    # defining the cost function
    cost = loss(batch_input, W)

    #
    # reg

    if regularization == 'L1':
        if reg_l1_coeff is None:
            delta_l1 = 1.0
        else:
            delta_l1 = reg_l1_coeff
        L1 = theano.tensor.sum(abs(W))
        cost += delta_l1 * L1

    #
    # deriving the gradient
    W_grad = theano.tensor.grad(cost, W)

    #
    # update rule
    W_upd = W - learning_rate * W_grad

    # W_upd = theano.printing.Print('WUP')(W_upd)

    #
    # FIXME: notmalization shall be done after projecting
    #

    if pos_projection:
        W_upd = theano.tensor.switch(W_upd < 0., 0., W_upd)
        # W_upd = theano.printing.Print('WPROPO')(W_upd)

    # normalizing
    if normalize:
        #
        # check for all zeros, this may happen with L1 reg
        W_upd = theano.tensor.switch(W_upd <= 0.,
                                     0.,
                                     W_upd / W_upd.sum())
    #
    # creating and returning the function
    return theano.function([batch_input], [cost, W_upd],
                           updates=[(W, W_upd)])


def evaluation_function(loss, params, normalize=True):
    """
    WRITEME
    """

    #
    # input
    batch_input = theano.tensor.matrix()

    #
    # normalize params
    if normalize:
        params = theano.tensor.switch(params <= 0.,
                                      0.,
                                      params / params.sum())

    return theano.function([batch_input], loss(batch_input, params))


def init_params(rand_gen, n_params, uniform=False, normalize=True):
    """
    WRITEME
    """

    W_rand = None
    if uniform:
        W_rand = numpy.ones(n_params)
    else:
        W_rand = rand_gen.rand(n_params)

    if normalize:
        W_rand_norm = W_rand / W_rand.sum()
    else:
        W_rand_norm = W_rand

    return W_rand_norm


def uncons_init_params(rand_gen, n_params, uniform=False, normalize=False):
    """
    WRITEME
    """

    W_rand = None

    low = -numpy.sqrt(6. / (n_params))
    high = numpy.sqrt(6. / (n_params))

    W_rand = rand_gen.uniform(low=low, high=high, size=n_params)

    if normalize:
        W_rand_norm = W_rand / W_rand.sum()
    else:
        W_rand_norm = W_rand

    return W_rand_norm


def simple_sgd(train,
               valid=None,
               test=None,
               loss=simple_avg_negative_ll,
               uniform_init=False,
               uncons_init=False,
               normalize=True,
               learning_rate=0.1,
               n_epochs=10,
               batch_size=None,
               regularization=None,
               cost_check_freq=None,
               rand_gen=None,
               reg_l1_coeff=None,
               epsilon=1e-7):
    """

    A simple version of sgd to learn the weights of a mixture of
    generative models
    """

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    #
    # TODO: create shared variables for all the datasets
    # for improving performance on GPUs
    n_train_instances = train.shape[0]
    n_features = train.shape[1]

    if batch_size is None:
        batch_size = n_train_instances
    #
    # computing number of batches from batch_size
    n_batches = n_train_instances // batch_size

    #
    # default is once each epoch
    if cost_check_freq is None:
        cost_check_freq = n_batches

    logging.info('Starting SGD:\n\tloss: %s\n\tl rate:%f\n\tn epochs:%d\n' +
                 '\tbatch size: %d\n\tregularize: %s\n' +
                 '\tcost freq: %d\n\teps : %f',
                 loss,
                 learning_rate, n_epochs, batch_size,
                 regularization, cost_check_freq, epsilon)

    #
    # initialize parameters
    W_val = None
    if uncons_init:
        W_val = uncons_init_params(rand_gen, n_features, normalize=normalize)
        print ('W val shape', W_val.shape)
        print (W_val)

    else:
        W_val = init_params(rand_gen, n_features, uniform_init, normalize)

        logging.info('Initial param values %s', W_val)
        lse = logsumexp(train + numpy.log(W_val), axis=1)
        avg_ll = numpy.mean(lse)

        print('avg ll', avg_ll)

    #
    # allocating the weights as a shared var to be updated
    W = theano.shared(W_val, 'W')

    #
    # creating theano functions
    train_func = training_function(loss,
                                   W,
                                   regularization=regularization,
                                   learning_rate=learning_rate,
                                   reg_l1_coeff=reg_l1_coeff)

    eval_func = evaluation_function(loss, W)
    #
    # printing the initial value on train
    init_train_cost = eval_func(train)
    init_valid_cost = -numpy.Inf
    init_test_cost = -numpy.Inf

    if valid is not None:
        init_valid_cost = eval_func(valid)
    if test is not None:
        init_test_cost = eval_func(test)

    logging.info('Initial values:\n\ttrain: %f\n\tvalid: %f\n\ttest %f',
                 init_train_cost,
                 init_valid_cost,
                 init_test_cost)

    best_state = {}
    best_state['best_valid_ll'] = init_valid_cost
    best_state['best_train_ll'] = init_train_cost
    best_state['best_test_ll'] = init_test_cost
    best_state['best_params'] = W_val

    for epoch_count in range(n_epochs):

        epoch_start_t = perf_counter()
        logging.info('Starting epoch %d/%d', epoch_count + 1, n_epochs)

        #
        # permuting instances
        order = rand_gen.permutation(train.shape[0])
        train = train[order, :]

        iter_count = 0

        epoch_cost = 0.0

        #
        # iterating over batches
        for index in range(n_batches):

            iter_count += 1

            logging.debug('Computing batch %d/%d', index + 1, n_batches)
            batch_start_t = perf_counter()

            #
            # extract mini batch
            train_batch = train[index * batch_size:(index + 1) * batch_size, :]

            #
            # apply training function
            cost, W_ = train_func(train_batch)
            batch_end_t = perf_counter()
            logging.debug('Batch done in %f secs. COST: %f',
                          batch_end_t - batch_start_t,
                          cost)

            epoch_cost += cost

            #
            # evaluating on the datasets
            train_cost = numpy.Inf
            if iter_count % cost_check_freq == 0:
                logging.debug('Evaluating on whole training set')
                train_start_t = perf_counter()
                train_cost = eval_func(train)
                train_end_t = perf_counter()
                logging.info('\t>>>>>> Eval on train done in %f secs. NLL: %f',
                             train_end_t - train_start_t,
                             train_cost)
                #
                # evaluate on validation
                valid_cost = numpy.Inf
                if valid is not None:
                    logging.debug('Evaluating on validation set')
                    valid_start_t = perf_counter()
                    valid_cost = eval_func(valid)
                    valid_end_t = perf_counter()
                    logging.info('\t>>>>>> Eval on valid done in %f secs. NLL: %f',
                                 valid_end_t - valid_start_t,
                                 valid_cost)

                #
                # evaluate on test set
                test_cost = numpy.Inf
                if test is not None:
                    logging.debug('Evaluating on test set')
                    test_start_t = perf_counter()
                    test_cost = eval_func(test)
                    test_end_t = perf_counter()
                    logging.info('\t>>>>>> Eval on test done in %f secs. NLL: %f',
                                 test_end_t - test_start_t,
                                 test_cost)

                #
                # early stopping
                if (valid_cost < best_state['best_valid_ll']):
                    best_state['best_valid_ll'] = valid_cost
                    best_state['best_train_ll'] = train_cost
                    best_state['best_test_ll'] = test_cost
                    best_state['best_params'] = W.get_value()

        epoch_end_t = perf_counter()
        logging.info('epoch done in %f secs! COST:%f (AVG:%f)',
                     epoch_end_t - epoch_start_t,
                     epoch_cost, epoch_cost / n_batches)
        logging.info('Param values %s', W.get_value())

    return best_state
