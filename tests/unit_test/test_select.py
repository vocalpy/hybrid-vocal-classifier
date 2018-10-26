"""module to test high-level select function in hvc.select"""
import os
from glob import glob

import hvc
from hvc.select import determine_model_output_folder_name
from config import rewrite_config


class TestSelect:
    def _yaml_config_asserts(self, select_yaml_config_path,
                             tmp_output_dir, feature_file):
        select_config_rewritten = rewrite_config(select_yaml_config_path,
                                                 tmp_output_dir,
                                                 replace_dict={'feature_file':
                                                                   ('replace with feature_file',
                                                                    feature_file),
                                                               'output_dir':
                                                                   ('replace with tmp_output_dir',
                                                                    str(tmp_output_dir))})
        select_outputs_before = glob(os.path.join(str(tmp_output_dir),
                                                  'select_output*',
                                                  'summary_model_select_file*'))
        hvc.select(select_config_rewritten)
        # helper function with assertions shared by all
        # tests for hvc.select run with config.yml files
        select_outputs_after = glob(os.path.join(str(tmp_output_dir),
                                                 'select_output*',
                                                 'summary_model_select_file*'))
        select_output = [after for after in select_outputs_after
                         if after not in select_outputs_before]
        # should only be one summary output file
        assert len(select_output) == 1

        # now check for every model in config
        # if there is corresponding folder with model files etc
        select_config = hvc.parse_config(select_config_rewritten, 'select')
        select_output_dir = os.path.dirname(select_output[0])
        select_model_dirs = next(
            os.walk(
                select_output_dir)
        )[1]  # [1] to return just dir names
        select_model_folder_names = [determine_model_output_folder_name(model_dict)
                                     for model_dict in select_config['models']]
        for folder_name in select_model_folder_names:
            assert folder_name in select_model_dirs

        return True

    def test_select_knn_ftr_grp_yaml(self, tmp_output_dir, configs_path, test_data_dir):
        # test select with features for model specified by feature list indices
        knn_select_config = os.path.join(configs_path,
                                         'test_select_knn_ftr_grp.config.yml')
        feature_file = glob(os.path.join(test_data_dir,
                                         'feature_files',
                                         'test_extract_knn*'))[0]
        self._yaml_config_asserts(knn_select_config, tmp_output_dir, feature_file)

    def test_select_knn_ftr_list_inds_yaml(self, tmp_output_dir, configs_path, test_data_dir):
        # test select with features for model specified by a feature group
        knn_select_config = os.path.join(configs_path,
                                         'test_select_knn_ftr_list_inds.config.yml')
        feature_file = glob(os.path.join(test_data_dir,
                                         'feature_files',
                                         'test_extract_knn*'))[0]
        self._yaml_config_asserts(knn_select_config, tmp_output_dir, feature_file)

    def test_select_multiple_ftr_grp_yaml(self, tmp_output_dir, configs_path, test_data_dir):
        # test select with features for model specified by list of multiple feature groups
        select_config = os.path.join(configs_path,
                                     'test_select_multiple_ftr_grp.config.yml')
        feature_file = glob(os.path.join(test_data_dir,
                                         'feature_files',
                                         'test_extract_multiple_feature*'))[0]
        self._yaml_config_asserts(select_config, tmp_output_dir, feature_file)