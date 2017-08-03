"""
tests extract module
"""

import os
import glob

import numpy as np
from sklearn.externals import joblib

import hvc

configs = './test_data/config.yaml/'


class TestSelect:

    def test_select_knn(self):
        knn_select_config = os.path.join(configs,
                                         'test_select_knn.config.yml')
        hvc.select(knn_select_config)

    def test_select_knn_ftr_grp(self):
        knn_select_config = os.path.join(configs,
                                         'test_select_knn.config.yml')
        hvc.select(knn_select_config)
