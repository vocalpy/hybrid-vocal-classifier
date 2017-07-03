"""
tests features module
"""

# from standard library
import os
import glob

# from dependencies
import yaml
import numpy as np

import hvc

with open(os.path.join(dir_path, '../hvc/parse/feature_groups.yml')) as ftr_grp_yaml:
    valid_feature_groups_dict = yaml.load(ftr_grp_yaml)


class TestTachibana:

    def test_tachibana(self):
        """tests Tachibana features
        """
