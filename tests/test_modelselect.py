"""
tests extract module
"""

import os

import hvc

configs = os.path.join(
    os.path.dirname(__file__),
    'test_data/config.yaml/')


class TestSelect:

    def test_select_knn(self):
        """test select with features for model specified by feature list indices"""
        knn_select_config = os.path.join(configs,
                                         'test_select_knn_ftr_list_inds.config.yml')
        hvc.select(knn_select_config)

    def test_select_knn_ftr_grp(self):
        """test select with features for model specified by a feature group"""
        knn_select_config = os.path.join(configs,
                                         'test_select_knn_ftr_list_inds.config.yml')
        hvc.select(knn_select_config)

    def test_select_multiple_ftr_grp(self):
        """test select with features for model specified by list of multiple feature groups"""
        knn_select_config = os.path.join(configs,
                                         'test_select_multiple_ftr_grp.config.yml')
        hvc.select(knn_select_config)