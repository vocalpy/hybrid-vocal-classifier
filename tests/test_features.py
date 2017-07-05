"""
test features.extract module
"""

import pytest
import yaml
import numpy as np

import hvc.audiofileIO
import hvc.features

@pytest.fixture()
def has_window_error():
    filename = './test_data/cbins/window_error/gy6or6_baseline_220312_0901.106.cbin'
    index = 19
    return filename, index


class TestFromFile:

    def test_song_w_nan(self, has_window_error):
        """tests that features_arr[ind,:] == np.nan
        where ind is the row corresponding to
        a syllable from a song
        for which a spectrogram could not be generated, and
        so single-syllable features cannot be extracted from it
        """
        filename, index = has_window_error
        with open('../hvc/parse/feature_groups.yml') as ftr_grp_yaml:
            valid_feature_groups_dict = yaml.load(ftr_grp_yaml)
        svm_features = valid_feature_groups_dict['svm']
        segment_params = {
            'threshold': 1500,
            'min_syl_dur': 0.01,
            'min_silent_dur': 0.006
        }
        with pytest.warns(UserWarning):
            ftr_arr, labels, inds =  hvc.features.extract.from_file(filename=filename,
                                                                    file_format='evtaf',
                                                                    feature_list=svm_features,
                                                                    spect_params={'ref': 'koumura'},
                                                                    labels_to_use='iabcdefghjk',
                                                                    segment_params=segment_params)
        assert np.alltrue(np.isnan(ftr_arr[19, :]))
