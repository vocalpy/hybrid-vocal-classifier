"""
tests tachibana module
"""

# from standard library
import glob

# from dependencies
import yaml
import numpy as np
from sklearn.externals import joblib
import pytest

import hvc.audiofileIO
from hvc.features.extract import single_syl_features_switch_case_dict
from hvc.features.extract import multiple_syl_features_switch_case_dict
from hvc.features import tachibana

with open('../hvc/parse/feature_groups.yml') as ftr_grp_yaml:
    valid_feature_groups_dict = yaml.load(ftr_grp_yaml)


class TestTachibana:
    """
    unit tests that specifically test functions
    in the Tachibana module
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
        songfiles_list = glob.glob('./test_data/cbins/032412/*.cbin')
        song = hvc.audiofileIO.Song(songfiles_list[0], 'evtaf', segment_params)
        song.set_syls_to_use('iabcdefghjk')
        song.make_syl_spects(spect_params={'ref': 'tachibana'})
        return song

    def test_mean_spect(self, song):
        """test mean spectrum
        """
        # if using `tachibana` reference to make syllable spectra
        # (with 32 kHz sampling rate)
        # length of vector returned by mean_spectrum should be 128.
        # Want to be sure to return exact same number of features
        # in case it matters for e.g. feature selection
        assert tachibana.mean_spectrum(song.syls[0]).shape[0] == 128

    def test_delta_mean_spect(self, song):
        """test delta spectrum
        """
        assert tachibana.mean_delta_spectrum(song.syls[0]).shape[0] == 128

    def test_mean_cepst(self, song):
        """test mean cepstrum
        """
        assert tachibana.mean_cepstrum(song.syls[0]).shape[0] == 128

    def test_delta_mean_cepstrum(self, song):
        """test delta cepstrum
        """
        assert tachibana.mean_delta_cepstrum(song.syls[0]).shape[0] == 128

    # def test_svm_features(self):
    #     """tests features from svm feature group
    #     for svm features in particular, need to ensure that values approximate
    #     the values from original feature extraction code written in Matlab,
    #     so compare features values extracted with Matlab script to values
    #     that hvc extracts
    #     Currently this is really gross with a lot of hard-coded constants.
    #     Haven't figured out how to prettify yet, not sure if I need to.
    #     """
    #
    #     svm_features = valid_feature_groups_dict['svm']
    #     segment_params = {'threshold': 1500,
    #                       'min_syl_dur': 0.01,
    #                       'min_silent_dur': 0.006
    #                       }
    #
    #     songfiles_list = glob.glob('./test_data/cbins*.cbin')
    #
    #     # note that I'm only testing first 10 songs!!!
    #     for songfile in songfiles_list[:10]:
    #         song = hvc.audiofileIO.Song(songfile, 'evtaf', segment_params)
    #         song.set_syls_to_use('iabcdefghjk')
    #         song.make_syl_spects(spect_params={'ref': 'tachibana'})
    #
    #         for syl in song.syls:
    #             for feature in svm_features:
    #                 ftr = single_syl_features_switch_case_dict[feature](syl)
    #                 if 'feature_vec' in locals():
    #                     feature_vec = np.append(feature_vec, ftr)
    #                 else:  # if feature_vec doesn't exist yet
    #                     feature_vec = ftr
    #             if 'curr_feature_arr' in locals():
    #                 curr_feature_arr = np.concatenate((curr_feature_arr,
    #                                                    feature_vec[np.newaxis, :]),
    #                                                   axis=0)
    #                 import pdb;
    #                 pdb.set_trace()
    #             else:
    #                 curr_feature_arr = feature_vec[np.newaxis, :]
    #             del feature_vec
    #
    #         # after looping through all syllables:
    #         if 'features_arr' in locals():
    #             features_arr = np.concatenate((features_arr,
    #                                            curr_feature_arr),
    #                                           axis=0)
    #         else:  # if 'features_arr' doesn't exist yet
    #             features_arr = curr_feature_arr
    #         del curr_feature_arr
    #
    #     ftrs_dict = joblib.load('./test_data/cbins/features_from_MakeAllFeatures_songs1through10')
    #     # features_mat is has 536 columns
    #     # because in my wrapper script around MakeAllFeatures.mat,
    #     # I added duration features on top of original 532-feature set
    #     # but only want the original 532 in this case
    #     ftrs = ftrs_dict['features_mat'][:, :532]
    #     # hmm, values are often off even at the first place
