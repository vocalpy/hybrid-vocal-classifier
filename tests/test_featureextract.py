"""
tests extract module
"""

import os
import glob

from sklearn.externals import joblib

import hvc

class TestExtract:

    def test_extract(self):
        hvc.extract('./tests/config.yaml/test_extract.config.yml')
        # switch to test dir
        os.chdir(TEST_DIR)
        ftr_files = glob.glob('features_from*')
        ftr_dicts = []
        for ftr_file in ftr_files:
            ftr_dicts.append(joblib.load(ftr_file))

        # for each dict, make sure length of labels == num rows in features

        # make sure number of features i.e. columns is constant across feature matrices

        # load summary dict

        # make sure rows in summary dict features == sum of rows of each ftr file features
