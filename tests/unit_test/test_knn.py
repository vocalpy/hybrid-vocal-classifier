"""
tests knn module
"""

import os
from glob import glob

import numpy as np

import hvc.audiofileIO
from hvc.features import knn


class TestKNN:
    """
    unit tests that specifically test functions
    for getting features for k-Nearest Neighbors
    """

    def test_duration(self, test_data_dir):
        """test duration
        """
        songfiles_dir = os.path.join(test_data_dir,
                                 os.path.normpath('cbins/gy6or6/032412/*.cbin'))
        songfiles_list = glob(songfiles_dir)
        first_song = songfiles_list[0]
        first_song_notmat = first_song + '.not.mat'
        notmat_dict = hvc.evfuncs.load_notmat(first_song_notmat)

        dur = knn.duration(notmat_dict['onsets']/1000,
                           notmat_dict['offsets']/1000,
                           np.ones(notmat_dict['onsets'].shape).astype(bool))
