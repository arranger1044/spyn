import numpy
import csv

DATA_PATH = "data/"
DATA_FULL_PATH = DATA_PATH + 'full/'

DATASET_NAMES = ['accidents',
                 'ad',
                 'baudio',
                 'bbc',
                 'bnetflix',
                 'book',
                 'c20ng',
                 'cr52',
                 'cwebkb',
                 'dna',
                 'jester',
                 'kdd',
                 'msnbc',
                 'msweb',
                 'nltcs',
                 'plants',
                 'pumsb_star',
                 'tmovie',
                 'tretail']
import os

from spn import RND_SEED


def csv_2_numpy(file, path=DATA_PATH, sep=',', type='int8'):
    """
    WRITEME
    """
    file_path = path + file
    reader = csv.reader(open(file_path, "r"), delimiter=sep)
    x = list(reader)
    dataset = numpy.array(x).astype(type)
    return dataset


def load_train_val_test_csvs(dataset,
                             path=DATA_PATH,
                             sep=',',
                             type='int32',
                             suffixes=['.ts.data',
                                       '.valid.data',
                                       '.test.data']):
    """
    WRITEME
    """
    csv_files = [dataset + ext for ext in suffixes]
    return [csv_2_numpy(file, path, sep, type) for file in csv_files]


def sample_indexes(indexes, perc, replace=False, rand_gen=None):
    """
    index sampling
    """
    n_indices = indexes.shape[0]
    sample_size = int(n_indices * perc)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RND_SEED)

    sampled_indices = rand_gen.choice(  # n_indices,
        indexes,
        size=sample_size,
        replace=replace)

    return sampled_indices


def sample_instances(dataset, perc, replace=False, rndState=None):
    """
    Little utility to sample instances (rows) from
    a dataset (2d numpy array)
    """
    n_instances = dataset.shape[0]
    sample_size = int(n_instances * perc)
    if rndState is None:
        row_indexes = numpy.random.choice(n_instances,
                                          sample_size,
                                          replace)
    else:
        row_indexes = rndState.choice(n_instances,
                                      sample_size,
                                      replace)
    # print(row_indexes)
    return dataset[row_indexes, :]


def sample_sets(datasets, perc, replace=False, rndState=None):
    """
    WRITEME
    """
    sampled_datasets = [sample_instances(dataset, perc, replace, rndState)
                        for dataset in datasets]
    return sampled_datasets


def data_2_freqs(dataset):
    """
    WRITEME
    """
    freqs = []
    features = []
    for j, col in enumerate(dataset.T):
        freq_dict = {'var': j}
        # transforming into a set to get the feature value
        # this is assuming not missing values features
        # feature_values = max(2, len(set(col)))
        feature_values = max(2, max(set(col)) + 1)
        features.append(feature_values)
        # create a list whose length is the number of feature values
        freq_list = [0 for i in range(feature_values)]
        # populate it with the seen values
        for val in col:
            freq_list[val] += 1
        # update the dictionary and the resulting list
        freq_dict['freqs'] = freq_list
        freqs.append(freq_dict)

    return freqs, features


def update_feature_count(old_freqs, new_freqs):
    if not old_freqs:
        return new_freqs
    else:
        for i, frew in enumerate(old_freqs):
            old_freqs[i] = max(old_freqs[i], new_freqs[i])
        return old_freqs


def data_clust_freqs(dataset,
                     n_clusters,
                     rand_state=None):
    """
    WRITEME
    """
    freqs = []
    features = []

    n_instances = dataset.shape[0]

    # assign clusters randomly to instances
    if rand_state is None:
        rand_state = numpy.random.RandomState(RND_SEED)

    # inst_2_clusters = numpy.random.randint(0, n_clusters, n_instances)

    # getting the indices for each cluster
    # this all stuff could be done with a single loop
    clusters = [[] for i in range(n_clusters)]
    # for instance in range(n_instances):
    #     rand_cluster = rand_state.randint(0, n_clusters)
    #     clusters[rand_cluster].append(instance)

    instance_ids = numpy.arange(n_instances)
    rand_state.shuffle(instance_ids)
    print(instance_ids)
    for i in range(n_instances):
        clusters[i % n_clusters].append(instance_ids[i])

    # now we can operate cluster-wise
    for cluster_ids in clusters:
        # collecting all the data for the cluster
        cluster_data = dataset[cluster_ids, :]
        # count the frequencies for the var values
        cluster_freqs, cluster_features = data_2_freqs(cluster_data)
        # updating stats
        features = update_feature_count(features, cluster_features)
        freqs.extend(cluster_freqs)

    return freqs, features


def merge_datasets(dataset_name,
                   shuffle=True,
                   path=DATA_PATH,
                   sep=',',
                   type='int32',
                   suffixes=['.ts.data',
                             '.valid.data',
                             '.test.data'],
                   savetxt=True,
                   out_path=DATA_FULL_PATH,
                   output_suffix='.all.data',
                   rand_gen=None):
    """
    Merging portions of a dataset
    Loading them from file and optionally writing them to file
    """
    dataset_parts = load_train_val_test_csvs(dataset_name,
                                             path,
                                             sep,
                                             type,
                                             suffixes)

    print('Loaded dataset parts for', dataset_name)

    #
    # checking features
    assert len(dataset_parts) > 0
    first_dataset = dataset_parts[0]
    n_features = first_dataset.shape[1]
    for dataset_p in dataset_parts:
        assert dataset_p.shape[1] == n_features

    print('\tFeatures are conform')

    #
    # storing instances
    n_instances = [dataset_p.shape[0]
                   for dataset_p in dataset_parts]

    #
    # merging
    merged_dataset = numpy.concatenate(dataset_parts)
    print('\tParts merged')

    #
    # shuffling
    if shuffle:
        if rand_gen is None:
            rand_gen = numpy.random.RandomState(RND_SEED)
        rand_gen.shuffle(merged_dataset)
        print('\tShuffled')

    #
    #
    tot_n_instances = sum(n_instances)
    assert merged_dataset.shape[0] == tot_n_instances

    #
    # writing out
    if savetxt:
        out_path = out_path + dataset_name + output_suffix
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        fmt = '%.8e'
        if 'int' in type:
            fmt = '%d'

        numpy.savetxt(out_path, merged_dataset, delimiter=sep, fmt=fmt)
        print('\tMerged Dataset saved to', out_path)

    return merged_dataset


def split_into_folds(dataset,
                     n_folds=10,
                     percentages=[0.81, 0.09, 0.1]):
    """
    Splitting a dataset into N folds (e.g. for cv)
    and optionally each fold into train-valid-test
    """
