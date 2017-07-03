"""
tests features module
"""

# from standard library
import os
import glob

# from dependencies
import yaml
import numpy as np
from scipy.io import loadmat

import hvc
from hvc.features.extract import single_syl_features_switch_case_dict
from hvc.features.extract import multiple_syl_features_switch_case_dict

with open('../hvc/parse/feature_groups.yml') as ftr_grp_yaml:
    valid_feature_groups_dict = yaml.load(ftr_grp_yaml)


class TestSVM:

    def test_svm_features(self):
        """tests features from svm feature group
        for svm features in particular, need to ensure that values approximate
        the values from original feature extraction code written in Matlab,
        so compare features values extracted with Matlab script to values
        that hvc extracts
        Currently this is really gross with a lot of hard-coded constants.
        Haven't figured out how to prettify yet, not sure if I need to.
        """

        svm_features = valid_feature_groups_dict['svm']
        segment_params = {'threshold': 1500,
                          'min_syl_dur': 0.01,
                          'min_silent_dur': 0.006
                          }

        os.chdir('./test_data/cbins')
        songfiles_list = glob.glob('*.cbin')

        for songfile in songfiles_list[:10]:
            song = hvc.audiofileIO.Song(songfile, 'evtaf', segment_params)
            song.set_syls_to_use('iabcdefghjk')
            song.make_syl_spects(spect_params={'ref': 'tachibana'})

            for syl in song.syls:
                for feature in svm_features:
                    ftr = single_syl_features_switch_case_dict[feature](syl)
                    if 'feature_vec' in locals():
                        feature_vec = np.append(feature_vec, ftr)
                    else:  # if feature_vec doesn't exist yet
                        feature_vec = ftr
                if 'curr_feature_arr' in locals():
                    curr_feature_arr = np.concatenate((curr_feature_arr,
                                                       feature_vec[np.newaxis, :]),
                                                      axis=0)
                    import pdb;
                    pdb.set_trace()
                else:
                    curr_feature_arr = feature_vec[np.newaxis, :]
                del feature_vec

            # after looping through all syllables:
            if 'features_arr' in locals():
                features_arr = np.concatenate((features_arr,
                                               curr_feature_arr),
                                              axis=0)
            else:  # if 'features_arr' doesn't exist yet
                features_arr = curr_feature_arr

        matlab_ftrs = loadmat('gy6or6_svm_ftr_file_from_03-24-12_generated_07-03-17_00-59.mat')
