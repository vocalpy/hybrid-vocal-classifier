"""module to test high-level select function in hvc.select"""
import os

import hvc


class TestSelect:

    def test_select_knn(self, configs_dir):
        # test select with features for model specified by feature list indices
        knn_select_config = os.path.join(configs_dir,
                                         'test_select_knn_ftr_list_inds.config.yml')
        hvc.select(knn_select_config)

    def test_select_knn_ftr_grp(self, configs_dir):
        # test select with features for model specified by a feature group
        knn_select_config = os.path.join(configs_dir,
                                         'test_select_knn_ftr_list_inds.config.yml')
        hvc.select(knn_select_config)

    def test_select_multiple_ftr_grp(self, configs_dir):
        # test select with features for model specified by list of multiple feature groups
        knn_select_config = os.path.join(configs_dir,
                                         'test_select_multiple_ftr_grp.config.yml')
        hvc.select(knn_select_config)