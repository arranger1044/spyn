import dataset

import numpy


def test_sampling():
    # loading nltcs
    print('Loading datasets')
    train, valid, test = dataset.load_train_val_test_csvs('nltcs')

    # checking for their shape
    n_instances = train.shape[0]
    n_test_instances = test.shape[0]
    n_valid_instances = valid.shape[0]

    nltcs_train = 16181
    nltcs_valid = 2157
    nltcs_test = 3236

    print('Training set with {0} instances\n'.format(n_instances) +
          'Validation set with {0} instances\n'.format(n_valid_instances) +
          'Test set with {0} instances'.format(n_test_instances))

    assert n_instances == nltcs_train
    assert n_valid_instances == nltcs_valid
    assert n_test_instances == nltcs_test

    # random sampling
    perc = 0.1
    sample_train, sample_valid, sample_test = \
        dataset.sample_sets((train, valid, test), perc)

    n_s_instances = sample_train.shape[0]
    n_s_valid_instances = sample_valid.shape[0]
    n_s_test_instances = sample_test.shape[0]

    print('Sampled training set with {0} instances\n'
          .format(n_s_instances) +
          'Sampled validation set with {0} instances\n'
          .format(n_s_valid_instances) +
          'Sampled test set with {0} instances'
          .format(n_s_test_instances))

    assert n_s_instances == int(nltcs_train * perc)
    assert n_s_valid_instances == int(nltcs_valid * perc)
    assert n_s_test_instances == int(nltcs_test * perc)


def test_bootstrap_sampling():
    # loading nltcs
    synth_data = numpy.array([[1, 1, 1],
                              [1, 1, 0],
                              [1, 0, 0],
                              [0, 0, 0],
                              [0, 0, 1],
                              [0, 1, 0],
                              [0, 1, 1],
                              [1, 0, 1]])

    perc = 1.0
    replace = True
    sampled_data_1 = dataset.sample_instances(synth_data,
                                              perc=perc,
                                              replace=replace)

    sampled_data_2 = dataset.sample_instances(synth_data,
                                              perc=perc,
                                              replace=replace)

    sampled_data_3 = dataset.sample_instances(synth_data,
                                              perc=perc,
                                              replace=replace)

    print('first sample:\n', sampled_data_1)
    print('second sample:\n', sampled_data_2)
    print('third sample:\n', sampled_data_3)


def test_cluster_freqs():
    data = numpy.array([[0, 1, 1, 0],
                        [1, 2, 0, 0],
                        [1, 0, 1, 0],
                        [0, 0, 2, 0],
                        [0, 1, 3, 0],
                        [1, 1, 1, 0]])
    n_clusters = 3
    freqs, features = dataset.data_clust_freqs(data,
                                               n_clusters)
    print('frequencies, features',
          freqs, features)


def test_merge_datasets():
    #
    # a loop to convert them all
    for dataset_name in dataset.DATASET_NAMES:
        dataset.merge_datasets(dataset_name)
