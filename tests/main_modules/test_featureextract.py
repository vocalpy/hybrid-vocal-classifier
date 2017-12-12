"""
tests featureextract module
"""

import os
import glob

import numpy as np
from sklearn.externals import joblib

import hvc

configs = os.path.join(
    os.path.dirname(__file__),
    os.path.normpath('test_data/config.yml/'))


class TestExtract:

    def tests_for_all_extract(self):
        search_path = os.path.join(configs,
                                   os.path.normpath(
                                       'test_data/config.yml/'
                                       'test_extract_*.config.yml'))
        extract_config_files = glob.glob(search_path)
        for extract_config_file in extract_config_files:
            if os.getcwd() != homedir:
                os.chdir(homedir)
            hvc.extract(extract_config_file)
            extract_config = hvc.parse_config(extract_config_file, 'extract')

            for todo in extract_config['todo_list']:
                # switch to test dir
                os.chdir(todo['output_dir'])
                extract_outputs = list(
                    filter(os.path.isdir, glob.glob('*extract_output*')
                           )
                )
                extract_outputs.sort(key=os.path.getmtime)

                os.chdir(extract_outputs[-1])  # most recent
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
                            assert features.shape[0] == len(labels)

                    # make sure number of features i.e. columns is constant across feature matrices
                    ftr_cols = [ftr_dict['features'].shape[1] for ftr_dict in ftr_dicts]
                    assert np.unique(ftr_cols).shape[-1] == 1


                if any(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts]):
                    assert all(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts])

                # make sure rows in summary dict features == sum of rows of each ftr file features
                summary_file = glob.glob('summary_feature_file_*')
                # (should only be one summary file)
                assert len(summary_file) == 1
                summary_dict = joblib.load(summary_file[0])
