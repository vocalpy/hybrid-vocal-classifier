"""
tests knn module
"""

# from standard library
import os
from glob import glob

# from dependencies
import yaml
import pytest
import numpy as np

import hvc.audiofileIO
from hvc.features import knn
from hvc.parse.ref_spect_params import refs_dict

ftr_grp_yaml_file = '../hvc/parse/feature_groups.yml'
this_file_with_path = __file__
this_file_just_path = os.path.split(this_file_with_path)[0]
ftr_grp_yaml_file = os.path.join(this_file_just_path,
                                 os.path.normpath(ftr_grp_yaml_file))
with open(ftr_grp_yaml_file) as fileobj:
    valid_feature_groups_dict = yaml.load(fileobj)


class TestKNN:
    """
    unit tests that specifically test functions
    for getting features for k-Nearest Neighbors
    """

    def test_duration(self):
        """test duration
        """
        songfiles_dir = os.path.join(this_file_just_path,
                                 os.path.normpath('test_data/cbins/gy6or6/032412/*.cbin'))
        songfiles_list = glob(songfiles_dir)
        first_song = songfiles_list[0]
        first_song_notmat = first_song + '.not.mat'
        notmat_dict = hvc.evfuncs.load_notmat(first_song_notmat)

        dur = knn.duration(notmat_dict['onsets']/1000,
                           notmat_dict['offsets']/1000,
                           np.ones(notmat_dict['onsets'].shape).astype(bool))
