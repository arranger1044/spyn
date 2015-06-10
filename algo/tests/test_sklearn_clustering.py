import dataset

import numpy

from sklearn import mixture

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time


def test_GMM():

    #
    # random generator
    seed = 1337
    rand_gen = numpy.random.RandomState(seed)

    verbose = True

    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

    #
    # creating the classfier object
    n_components = 10
    # 'spherical', 'tied', 'diag', 'full'. Defaults to 'diag'.
    cov_type = 'diag'
    n_iters = 1000
    n_restarts = 10

    gmm_c = mixture.GMM(n_components=n_components,
                        covariance_type=cov_type,
                        random_state=rand_gen,
                        n_iter=n_iters,
                        n_init=n_restarts)

    #
    # fitting to training set
    fit_start_t = perf_counter()
    gmm_c.fit(train)
    fit_end_t = perf_counter()

    #
    # getting the cluster assignment
    pred_start_t = perf_counter()
    clustering = gmm_c.predict(train)
    pred_end_t = perf_counter()

    print('Clustering')
    print('for instances: ', clustering.shape[0])
    print(clustering)
    print('smallest cluster', numpy.min(clustering))
    print('biggest cluster', numpy.max(clustering))
    print('clustering done in', (fit_end_t - fit_start_t), 'secs')
    print('prediction done in', (pred_end_t - pred_start_t), 'secs')


def test_DPGMM():

    #
    # random generator
    seed = 1337
    rand_gen = numpy.random.RandomState(seed)

    verbose = True

    #
    # loading a very simple dataset
    dataset_name = 'nltcs'
    train, valid, test = dataset.load_train_val_test_csvs(dataset_name)

    #
    # this is the max number of clustering for a truncated DP
    n_components = 100

    cov_type = 'diag'
    n_iters = 1000

    # 'spherical', 'tied', 'diag', 'full'. Defaults to 'diag'.
    cov_type = 'diag'
    concentration = 1.0
    # a higher alpha means more clusters
    # as the expected number of clusters is alpha*log(N).
    dpgmm_c = mixture.DPGMM(n_components=n_components,
                            covariance_type=cov_type,
                            random_state=rand_gen,
                            n_iter=n_iters,
                            alpha=concentration,
                            verbose=verbose)

    #
    # fitting to training set
    fit_start_t = perf_counter()
    dpgmm_c.fit(train)
    fit_end_t = perf_counter()

    #
    # getting the cluster assignment
    pred_start_t = perf_counter()
    clustering = dpgmm_c.predict(train)
    pred_end_t = perf_counter()

    print('Clustering')
    print('for instances: ', clustering.shape[0])
    print(clustering)
    print('smallest cluster', numpy.min(clustering))
    print('biggest cluster', numpy.max(clustering))
    print('clustering done in', (fit_end_t - fit_start_t), 'secs')
    print('prediction done in', (pred_end_t - pred_start_t), 'secs')

    #
    # predicting probabilities
    pred_start_t = perf_counter()
    clustering_p = dpgmm_c.predict_proba(train)
    pred_end_t = perf_counter()
    print('prediction done in', (pred_end_t - pred_start_t), 'secs')
    print(clustering_p.shape[0], clustering_p.shape[1])
