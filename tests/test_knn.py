"""
tests knn module
"""

# from standard library
import glob

# from dependencies
import yaml
import pytest
import hvc.audiofileIO

from hvc.features import knn
from hvc.parse.ref_spect_params import refs_dict

with open('../hvc/parse/feature_groups.yml') as ftr_grp_yaml:
    valid_feature_groups_dict = yaml.load(ftr_grp_yaml)


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
        songfiles_list = glob.glob('./test_data/cbins/gy6or6/032412/*.cbin')
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

