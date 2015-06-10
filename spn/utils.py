try:
    from itertools import izip as zip
except:
    pass

import itertools

from itertools import tee

import numpy

import scipy
import scipy.stats

import os

import visualize


def pairwise(iterable):
    """
    s = <s0, s1, ...>
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def stats_format(stats_list, separator, digits=5):
    formatted = []
    float_format = '{0:.' + str(digits) + 'f}'
    for stat in stats_list:
        if isinstance(stat, int):
            formatted.append(str(stat))
        elif isinstance(stat, float):

            formatted.append(float_format.format(stat))
        else:
            formatted.append(stat)
    # concatenation
    return separator.join(formatted)


def sample_all_worlds(n_features, n_feature_vals=2):
    """
    features are assumed to be homogeneous
    """
    all_values = range(n_feature_vals)
    all_worlds = itertools.product(all_values, repeat=n_features)
    sampled_worlds = []
    for world in all_worlds:
        sampled_worlds.append(world)
    return numpy.array(sampled_worlds)


def statistically_significant(x, y,
                              p_threshold=0.05,
                              test='wilcoxon'):
    """
    WRITEME
    """
    #
    # paired samples shall have the same length
    assert x.shape[0] == y.shape[0]

    p_value = None
    if test == 'wilcoxon':
        W, p_value = scipy.stats.wilcoxon(x, y)
    elif test == 'ttest':
        T, p_value = scipy.stats.ttest_rel(x, y)

    return (p_value < p_threshold), p_value


DATASET_LIST = ['nltcs', 'msnbc', 'kdd',
                'plants', 'baudio', 'jester', 'bnetflix',
                'accidents', 'tretail', 'pumsb_star',
                'dna', 'kosarek', 'msweb',
                'book', 'tmovie', 'cwebkb',
                'cr52', 'c20ng', 'bbc', 'ad']

TEST_LL_FILE = '/test.lls'


def matching_dirs(name_list, prefix):
    matched = []
    for name in name_list:
        #
        # more precise controls can be made (i.e. regex)
        if prefix in name:
            matched.append(name)
    return matched


def directory_stat_test(exp_dir_1,
                        exp_dir_2,
                        dataset_names=DATASET_LIST,
                        test_filename=TEST_LL_FILE,
                        test='wilcoxon',
                        p_threshold=0.05):
    """
    WRITEME
    """

    print('\n\n')
    print('**********************************************************')
    print('Comparing two experiments by statistical test significance')
    print('**********************************************************\n')

    #
    # getting all the folders for each of the two input paths
    folders_1 = [elems[0] for elems in os.walk(exp_dir_1)]
    print('There are', len(folders_1), 'possible dirs for exp 1')
    folders_2 = [elems[0] for elems in os.walk(exp_dir_2)]
    print('There are', len(folders_2), 'possible dirs for exp 2')

    #
    # initing the results
    victories_1 = []
    victories_2 = []
    draws = []
    p_values = {}

    #
    # cycling through the names of the dataset we need to compare
    for dataset_name in dataset_names:

        print('Considering dataset', dataset_name)

        #
        # looking for a folder in both dirs
        dataset_exps_1 = matching_dirs(folders_1, dataset_name)

        dataset_exps_2 = matching_dirs(folders_2, dataset_name)
        #
        # checking for correctness
        if len(dataset_exps_1) > 1:
            print('There is more than one exp for dataset', dataset_name,
                  'in folder', exp_dir_1, 'exiting!')
            return

        if len(dataset_exps_2) > 1:
            print('There is more than one exp for dataset', dataset_name,
                  'in folder', exp_dir_2, 'exiting!')
            return

        # if there are no results in both, skip it
        if len(dataset_exps_1) > 0 and len(dataset_exps_2) > 0:
            #
            # trying to load the files with the exps
            test_lls_path_1 = dataset_exps_1[0] + test_filename
            test_lls_path_2 = dataset_exps_2[0] + test_filename

            if (os.path.exists(test_lls_path_1) and
                    os.path.exists(test_lls_path_2)):
                test_lls_1 = numpy.loadtxt(test_lls_path_1)
                test_lls_2 = numpy.loadtxt(test_lls_path_2)

                #
                # checking for shape
                assert test_lls_1.shape == test_lls_2.shape
                print('Testing for', test_lls_1.shape[0], 'instances')
                print('With p-value significance of', p_threshold)

                different, p_value = statistically_significant(test_lls_1,
                                                               test_lls_2,
                                                               p_threshold=p_threshold,
                                                               test=test)

                #
                # storing them
                test_ll_1 = test_lls_1.mean()
                test_ll_2 = test_lls_2.mean()

                p_values[dataset_name] = {'p-value': p_value,
                                          'avg-lls': (test_ll_1, test_ll_2)}

                if not different:
                    draws.append(dataset_name)
                else:
                    #
                    # which one is bigger?

                    if test_ll_1 > test_ll_2:
                        victories_1.append(dataset_name)
                    else:
                        victories_2.append(dataset_name)

    #
    # printing final stats
    print('---------------------------------------------')
    print('Final results')
    print('ALGO 1:\t #victories', len(victories_1), '{', victories_1, '}')
    print('ALGO 2:\t #victories', len(victories_2), '{', victories_2, '}')
    print('Draws:\t', len(draws), '{', draws, '}')

    #
    # printing p-values for latex
    for dataset_name in DATASET_LIST:
        p_value_str = None
        try:
            p_value = p_values[dataset_name]['p-value']
            p_value_str = '& ' + "{:.2e}".format(p_value)
        except:
            p_value_str = "& "
        print(p_value_str)

    return p_values

N_COMPONENTS = 50
CURVE_FILE = 'curves.log'
CURVE_INDEX = 2  # test lls


def directory_curves_vis(curve_dirs,
                         line_dirs,
                         output='./',
                         dataset_names=DATASET_LIST,
                         test_filename=TEST_LL_FILE,
                         curve_filename=CURVE_FILE):
    """
    WRITEME
    """

    print('\n\n')
    print('**********************************************************')
    print('Drawing curves for different experiments')
    print('**********************************************************\n')

    curve_contents = []
    for curve_dir in curve_dirs:
        curve_contents.append([elems[0] for elems in os.walk(curve_dir)])

    line_contents = []
    for line_dir in line_dirs:
        line_contents.append([elems[0] for elems in os.walk(line_dir)])

    x_axis = numpy.arange(1, N_COMPONENTS + 1)
    #
    # precomputing folder contents
    #
    # better to loop through experiments first
    for dataset_name in dataset_names:
        #
        # cheking if it is present for all of them

        #
        # curves first
        curve_exps = []
        not_found = False

        for i, curve_dir in enumerate(curve_dirs):
            #
            # composing path
            curves_log_path = None
            curve_dataset_exps = matching_dirs(curve_contents[i], dataset_name)
            if len(curve_dataset_exps) > 1:
                raise ValueError('More than one folder for dataset',
                                 dataset_name, curve_dir,
                                 curve_dataset_exps)
            if len(curve_dataset_exps) < 1:
                not_found = True
                print('Skipping dataset', dataset_name,
                      'Exp not present in', curve_dir)
                break
            else:
                curves_log_path = curve_dataset_exps[0] + '/' + curve_filename
                if os.path.exists(curves_log_path):
                    #
                    # loading it
                    curves_array = numpy.loadtxt(
                        curves_log_path, delimiter=',')
                    curve_exps.append((x_axis, curves_array[CURVE_INDEX]))
                else:
                    not_found = True
                    print('Curves file not found for dataset', dataset_name,
                          'and experiment', curve_dir)

        #
        # lol
        if not not_found:
            line_exps = []
            not_found = False
            for i, line_dir in enumerate(line_dirs):
                test_lls_file_path = None
                line_dataset_exps = matching_dirs(
                    line_contents[i], dataset_name)
                if len(line_dataset_exps) > 1:
                    raise ValueError('More than one folder for dataset',
                                     dataset_name, line_dir,
                                     line_dataset_exps)
                if len(line_dataset_exps) < 1:
                    not_found = True
                    print('Skipping dataset', dataset_name,
                          'Exp not present in', line_dir)
                    break
                else:
                    test_lls_file_path = line_dataset_exps[0] + \
                        '/' + test_filename
                    if os.path.exists(test_lls_file_path):
                        test_lls = numpy.loadtxt(test_lls_file_path)
                        test_ll = test_lls.mean()
                        print('Test LL', test_ll,
                              'for exp', curve_dir,
                              'for dataset', dataset_name)
                        line_exps.append(test_ll)
                    else:
                        not_found = True
                        print('Test.lls file not found for dataset',
                              dataset_name,
                              'and experiment', line_dir)

        if not not_found:
            #
            # if we have all the values we can plot them
            output_path = output + dataset_name + '.png'
            print('Printint to', output_path)
            visualize.visualize_curves(curve_exps,
                                       output=output_path,
                                       labels=['SPN-BB',
                                               'SPN-BTB',
                                               'LearnSPN',
                                               'ID-SPN',
                                               'MT'],
                                       lines=line_exps,
                                       loc=7)
