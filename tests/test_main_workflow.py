"""
tests running a 'typical' workflow
all thrown into one file
because the tests have to run in a certain order
and this seemed like the easiest least fragile way to do taht
"""

import os
import glob

import pytest
import yaml
import numpy as np
from sklearn.externals import joblib

import hvc

configs = './test_data/config.yaml/'


@pytest.fixture(scope='session')
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


@pytest.fixture(scope='session')
def tmp_config_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


#########################
#   utility functions   #
#########################
def rewrite_config(config_dir,
                   search_str,
                   replace_str,
                   replacement_str,
                   config_filename):
    """rewrites config files,
    e.g. to insert name of temporary directories
    """

    config_file_path = os.path.join(configs, config_filename)

    # replace value of interest
    with open(config_file_path) as config_file:
        config_as_list = config_file.readlines()
    for ind, val in enumerate(config_as_list):
        if search_str in val:
            config_as_list[ind] = config_as_list[ind].replace(
                replace_str,
                replacement_str
            )
            break

    # write to file in temporary configs dir
    tmp_config_path = os.path.join(str(config_dir), config_filename)
    with open(tmp_config_path, 'w') as tmp_config_file:
        tmp_config_file.writelines(config_as_list)
    # return location of that file
    return tmp_config_path


#########################
#     actual tests      #
#########################
def test_01_knn_extract(tmp_config_dir, tmp_output_dir):
    """
    """

    # have to put tmp_output_dir into yaml file
    config_filename = 'test_extract_knn.config.yml'
    tmp_config_path = rewrite_config(tmp_config_dir,
                                     search_str='output_dir',
                                     replace_str='replace with tmp_output_dir',
                                     replacement_str=str(tmp_output_dir),
                                     config_filename=config_filename)
    hvc.extract(tmp_config_path)


def test_02_knn_select(tmp_config_dir, tmp_output_dir):
    """test select with features for model specified by feature list indices"""

    config_filename = 'test_select_knn.config.yml'

    extract_output_dir = glob.glob(os.path.join(str(tmp_output_dir), '*extract*'))

    feature_file = glob.glob(os.path.join(extract_output_dir[0], 'summary*'))
    feature_file = feature_file[0]  # because glob returns list

    tmp_config_path = rewrite_config(tmp_config_dir,
                                     search_str='feature_file',
                                     replace_str='replace with feature_file',
                                     replacement_str=feature_file,
                                     config_filename=config_filename)
    hvc.select(tmp_config_path)


# def test_03_knn_extract(tmp_config_dir, tmp_output_dir):
#     """
#     """
#
#     # have to put tmp_output_dir into yaml file
#     config_filename = 'test_extract_multiple_feature_groups.config.yml'
#     tmp_config_path = rewrite_config(tmp_config_dir,
#                                      tmp_output_dir,
#                                      config_filename)
#     hvc.extract(tmp_config_path)

# def test_02_extract_tests_for_all_extract():
#     homedir = os.getcwd()
#     search_path = os.path.join('.',
#                                'test_data',
#                                'config.yaml',
#                                'test_extract_*.config.yml')
#     extract_config_files = glob.glob(search_path)
#     for extract_config_file in extract_config_files:
#         if os.getcwd() != homedir:
#             os.chdir(homedir)
#         hvc.extract(extract_config_file)
#         extract_config = hvc.parse_config(extract_config_file, 'extract')
#
#         for todo in extract_config['todo_list']:
#             # switch to test dir
#             os.chdir(todo['output_dir'])
#             extract_outputs = list(
#                 filter(os.path.isdir, glob.glob('*extract_output*')
#                        )
#             )
#             extract_outputs.sort(key=os.path.getmtime)
#
#             os.chdir(extract_outputs[-1])  # most recent
#             ftr_files = glob.glob('features_from*')
#             ftr_dicts = []
#             for ftr_file in ftr_files:
#                 ftr_dicts.append(joblib.load(ftr_file))
#
#             if any(['features' in ftr_dict for ftr_dict in ftr_dicts]):
#                 assert all(['features' in ftr_dict for ftr_dict in ftr_dicts])
#                 for ftr_dict in ftr_dicts:
#                     labels = ftr_dict['labels']
#                     if 'features' in ftr_dict:
#                         features = ftr_dict['features']
#                         assert features.shape[0] == len(labels)
#
#                 # make sure number of features i.e. columns is constant across feature matrices
#                 ftr_cols = [ftr_dict['features'].shape[1] for ftr_dict in ftr_dicts]
#                 assert np.unique(ftr_cols).shape[-1] == 1
#
#
#             if any(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts]):
#                 assert all(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts])
#
#             # make sure rows in summary dict features == sum of rows of each ftr file features
#             summary_file = glob.glob('summary_feature_file_*')
#             # (should only be one summary file)
#             assert len(summary_file) == 1
#             summary_dict = joblib.load(summary_file[0])
