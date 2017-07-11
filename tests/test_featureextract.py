"""
tests extract module
"""

import os
import glob

import numpy as np
from sklearn.externals import joblib

import hvc

class TestExtract:

    def tests_for_all_extract(self,config):
        for todo in config['todo_list']:
            # switch to test dir
            os.chdir(todo['output_dir'])
            ftr_files = glob.glob('features_from*')
            ftr_dicts = []
            for ftr_file in ftr_files:
                ftr_dicts.append(joblib.load(ftr_file))

            if any(['features' in ftr_dict for ftr_dict in ftr_dicts]):
                assert all(['features' in ftr_dict for ftr_dict in ftr_dicts])
                for ftr_dict in ftr_dicts:
                    labels = ftr_dict['labels']
                    if 'features' in ftr_dict:
                        features = ftr_dict['features']
                        assert features.shape[0] == labels.shape[-1]

                # make sure number of features i.e. columns is constant across feature matrices
                ftr_cols = [ftr_dict.shape[1] for ftr_dict in ftr_dicts]
                assert np.unique(ftr_cols).shape[-1] == 1


            if any(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts]):
                assert all(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts])

            # make sure rows in summary dict features == sum of rows of each ftr file features
            summary_file = glob.glob('summary_feature_file_*')
            # (should only be one summary file)
            assert len(summary_file) == 1
            summary_dict = joblib.load(summary_file[0])


    def test_extract(self):
        extract_config_files = glob.glob('./test_data/config.yaml/test_extract_*.config.yml')
        for extract_config_file in extract_config_files:
            hvc.extract(extract_config_file)
            extract_config = hvc.parse_config(extract_config_file, 'extract')
            self.tests_for_all_extract(extract_config)

