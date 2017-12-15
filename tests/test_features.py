"""
test features.extract module
"""

import os

import pytest
import yaml
import numpy as np

import hvc.audiofileIO
import hvc.features
from hvc.parse.ref_spect_params import refs_dict

@pytest.fixture()
def has_window_error():
    filename = os.path.join(
        os.path.dirname(__file__),
        os.path.normpath('test_data/cbins/window_error/'
                         'gy6or6_baseline_220312_0901.106.cbin'))
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
        with open(os.path.join(os.path.dirname(__file__),
                               os.path.normpath('../hvc/parse/feature_groups.yml'))
                  ) as ftr_grp_yaml:
            valid_feature_groups_dict = yaml.load(ftr_grp_yaml)
        svm_features = valid_feature_groups_dict['svm']
        segment_params = {
            'threshold': 1500,
            'min_syl_dur': 0.01,
            'min_silent_dur': 0.006
        }
        with pytest.warns(UserWarning):
             extract_dict = hvc.features.extract.from_file(filename=filename,
                                                           file_format='evtaf',
                                                           calling_function='extract',
                                                           feature_list=svm_features,
                                                           spect_params=refs_dict['koumura'],
                                                           labels_to_use='iabcdefghjk',
                                                           segment_params=segment_params)
        ftr_arr = extract_dict['features_arr']
        assert np.alltrue(np.isnan(ftr_arr[19, :]))

    def test_cbin(self):
        """tests alll features on a single .cbin file"""

        cbin = os.path.join(os.path.dirname(__file__),
                            os.path.normpath(
                                'test_data/cbins/gy6or6/032412/'
                                'gy6or6_baseline_240312_0811.1165.cbin'))

        segment_params = {
            'threshold': 1500,
            'min_syl_dur': 0.01,
            'min_silent_dur': 0.006
        }

        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf',
                                    segment_params=segment_params)

        with open(os.path.join(
                os.path.dirname(__file__),
                os.path.normpath('../hvc/parse/feature_groups.yml'))) as ftr_grp_yaml:
            ftr_grps = yaml.load(ftr_grp_yaml)

        extract_dict = hvc.features.extract.from_file(cbin,
                                                      file_format='evtaf',
                                                      calling_function='extract',
                                                      spect_params=refs_dict['tachibana'],
                                                      feature_list=ftr_grps['knn'],
                                                      segment_params=segment_params,
                                                      labels_to_use='iabcdefghjk')
        knn_ftrs = extract_dict['features_arr']
        feature_inds = extract_dict['feature_inds']
        assert type(knn_ftrs) == np.ndarray
        assert knn_ftrs.shape[-1] == np.unique(feature_inds).shape[-1]
        assert knn_ftrs.shape[-1] == len(ftr_grps['knn'])

        extract_dict = hvc.features.extract.from_file(cbin,
                                                      file_format='evtaf',
                                                      calling_function='extract',
                                                      spect_params=refs_dict['tachibana'],
                                                      feature_list=ftr_grps['svm'],
                                                      segment_params=segment_params,
                                                      labels_to_use='iabcdefghjk')

        svm_ftrs = extract_dict['features_arr']
        feature_inds = extract_dict['feature_inds']
        assert type(svm_ftrs) == np.ndarray
        assert svm_ftrs.shape[-1] == feature_inds.shape[-1]
        assert np.unique(feature_inds).shape[-1] == len(ftr_grps['svm'])

        extract_dict = hvc.features.extract.from_file(cbin,
                                                      file_format='evtaf',
                                                      calling_function='extract',
                                                      spect_params=refs_dict['tachibana'],
                                                      feature_list=['flatwindow'],
                                                      segment_params=segment_params,
                                                      labels_to_use='iabcdefghjk')
        neuralnet_ftrs = extract_dict['neuralnet_inputs_dict']
        assert type(neuralnet_ftrs) == dict
