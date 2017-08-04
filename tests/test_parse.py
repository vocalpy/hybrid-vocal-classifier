"""
test parse module
"""

import pytest
import yaml
import numpy as np
from sklearn.externals import joblib

import hvc.parse.extract
import hvc.parse.select

with open('./test_data/config.yaml/test_parse_extract.config.yml', 'r') as yaml_file:
    test_yaml_extract = yaml.load(yaml_file)

with open('./test_data/config.yaml/test_parse_select.config.yml', 'r') as yaml_file:
    test_yaml_select = yaml.load(yaml_file)

with open('../hvc/parse/features.yml') as features_yml:
    VALID_FEATURES = yaml.load(features_yml)['features']

with open('../hvc/parse/feature_groups.yml') as ftr_grps_yml:
    FTR_GROUPS = yaml.load(ftr_grps_yml)


class TestParseExtract:

    def test_features(self):
        # each valid feature group should be a subset of valid features
        for ftr_group in FTR_GROUPS.values():
            assert set(ftr_group) < set(VALID_FEATURES)

    def test_validate_yaml(self):
        test_yaml = test_yaml_extract['test_parse']
        # test whether valid yaml parses *without* throwing error
        hvc.parse.extract.validate_yaml(test_yaml['valid_with_default_spect_and_seg_params'])
        hvc.parse.extract.validate_yaml(test_yaml['valid_with_default_spect_params'])
        hvc.parse.extract.validate_yaml(test_yaml['valid_with_default_segment_params'])
        hvc.parse.extract.validate_yaml(test_yaml['valid_test_spect_params_with_ref'])

        with pytest.raises(KeyError):
            hvc.parse.extract.validate_yaml(test_yaml['invalid_no_todo'])

        with pytest.raises(KeyError):
            hvc.parse.extract.validate_yaml(test_yaml['invalid_missing_spect_params'])

        with pytest.raises(KeyError):
            hvc.parse.extract.validate_yaml(test_yaml['invalid_missing_segment_params'])

    def test_validate_feature_group_and_convert_to_list(self):

        # test one feature group as a string
        ftr_tuple = hvc.parse.extract._validate_feature_group_and_convert_to_list('knn')
        assert ftr_tuple[0] == FTR_GROUPS['knn']  # 1st item in tuple is feature list
        assert np.array_equal(ftr_tuple[1],
                              np.zeros((9,)))  # cuz there are 9 knn features
        assert ftr_tuple[2] == {'knn': 0}

        # test one feature group as a list
        ftr_tuple = hvc.parse.extract._validate_feature_group_and_convert_to_list(['knn'])
        assert ftr_tuple[0] == FTR_GROUPS['knn']
        assert np.array_equal(ftr_tuple[1],
                              np.zeros((9,)))  # cuz there are 9 knn features
        assert ftr_tuple[2] == {'knn': 0}

        # test two feature groups as a list
        ftr_tuple = hvc.parse.extract._validate_feature_group_and_convert_to_list(['knn',
                                                                                   'svm'])
        assert ftr_tuple[0] == FTR_GROUPS['knn'] + FTR_GROUPS['svm']
        assert np.array_equal(ftr_tuple[1],
                              np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        assert ftr_tuple[2] == {'knn': 0, 'svm': 1}

        # test feature list and feature group
        a_feature_list = ['duration group',
                          'preceding syllable duration',
                          'following syllable duration',
                          'preceding silent gap duration',
                          'following silent gap duration',
                          'mean smoothed rectified amplitude',
                          'mean spectral entropy',
                          'mean hi lo ratio',
                          'delta smoothed rectified amplitude']

        ftr_tuple = hvc.parse.extract._validate_feature_group_and_convert_to_list(feature_group='knn',
                                                                                  feature_list=a_feature_list)
        assert ftr_tuple[0] == a_feature_list + FTR_GROUPS['knn']
        assert np.array_equal(ftr_tuple[1],
                              np.asarray(([None] * len(a_feature_list) + [0] * 9)))
        assert ftr_tuple[2] == {'knn': 0}

    def test_validate_todo_list_dict(self):

        ftr_test_yml = test_yaml_extract['test_validate_todo_list_dict']

        str_grp = hvc.parse.extract._validate_todo_list_dict(ftr_test_yml[
                                                       'test_single_group_as_str'],
                                                   index=0)
        assert str_grp['feature_list'] == FTR_GROUPS['knn']

        single_grp_list = hvc.parse.extract._validate_todo_list_dict(ftr_test_yml[
                                                               'test_single_group_as_str'],
                                                           index=0)
        assert single_grp_list['feature_list'] == FTR_GROUPS['knn']

        two_grp_list = hvc.parse.extract._validate_todo_list_dict(ftr_test_yml[
                                                            'test_two_groups_as_list'],
                                                        index=0)
        assert two_grp_list['feature_list'] == FTR_GROUPS['knn'] + FTR_GROUPS['svm']

        actual_ftr_list = hvc.parse.extract._validate_todo_list_dict(ftr_test_yml[
                                                               'test_feature_list'],
                                                           index=0)
        assert actual_ftr_list['feature_list'] == FTR_GROUPS['knn']

    def test_validate_segment_params(self):

        test_yaml = test_yaml_extract['test_segment_params']

        # raises key error when key missing
        with pytest.raises(KeyError):
            hvc.parse.extract.validate_segment_params(test_yaml['segparams_missing_key'])

        # raises value error when 'threshold' value is not int
        with pytest.raises(ValueError):
            hvc.parse.extract.validate_segment_params(test_yaml['segparams_threshold_wrong_type'])

        # raises value error when 'min_syl_dur' value is not float
        with pytest.raises(ValueError):
            hvc.parse.extract.validate_segment_params(test_yaml['segparams_min_syl_dur_wrong_type'])

        # raises value error when 'min_silent_dur' value is not float
        with pytest.raises(ValueError):
            hvc.parse.extract.validate_segment_params(test_yaml['segparams_min_silent_dur_wrong_type'])


class TestParseSelect:

    def test_validate_model_dict(self):
        test_yaml = test_yaml_select['test_validate_model_dict']

        # model dict called with ftr gropu but without ftr_grp_ID_dict or ftr_grp_ID_arr
        model_dict = hvc.parse.select._validate_model_dict(test_yaml['valid_dict_with_feature_group'],
                                                           index=0)
        assert model_dict == {'feature_group': 'knn',
                              'hyperparameters': {'k': 4},
                              'model': 'knn'}
        assert 'feature_list_indices' not in model_dict

        # model dict called feature list indices entered as a list
        model_dict = hvc.parse.select._validate_model_dict(test_yaml['valid_dict_with_feature_list'],
                                                           index=0)
        assert model_dict == {'feature_list_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                              'hyperparameters': {'k': 4},
                              'model': 'knn'}

        ftr_file = joblib.load('.//test_data//feature_files//has_ftr_group_dict_and_arr'
                               '//summary_feature_file_created_170722_232106')
        model_dict = hvc.parse.select._validate_model_dict(test_yaml['valid_dict_with_feature_group'],
                                                           index=0,
                                                           ftr_list_grp_ID=ftr_file['feature_list_group_ID'],
                                                           ftr_grp_ID_dict=ftr_file['feature_group_ID_dict'])
        np.testing.assert_equal(model_dict,
                                {'feature_group': 'knn',
                                 'feature_list_indices': np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int),
                                 'hyperparameters': {'k': 4},
                                 'model': 'knn'})

        with pytest.raises(KeyError):
            hvc.parse.select._validate_model_dict(test_yaml[
                                                      'invalid_dict_with_feature_group_and_list'
                                                  ], index=0)



#class TestParsePredict()