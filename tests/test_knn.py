"""
tests knn module
"""

# from standard library
import os
import glob

# from dependencies
import yaml
import pytest
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

    @pytest.fixture()
    def song(self):
        """make a song object

        Should get fancy later and have this return random songs
        for more thorough testing

        Returns
        -------
        song: song object
            used to text feature extraction functions
        """
        segment_params = {'threshold': 1500,
                          'min_syl_dur': 0.01,
                          'min_silent_dur': 0.006
                          }

        songfiles_dir = os.path.join(this_file_just_path,
                                 os.path.normpath('test_data/cbins/gy6or6/032412/*.cbin'))
        songfiles_list = glob.glob(songfiles_dir)
        song = hvc.audiofileIO.Song(songfiles_list[0], 'evtaf', segment_params)
        song.set_syls_to_use('iabcdefghjk')
        song.make_syl_spects(spect_params=refs_dict['evsonganaly'])
        return song

    def test_duration(self, song):
        """test duration
        """
        dur = knn.duration(song.offsets_s,
                           song.onsets_s,
                           song.syls_to_use)
