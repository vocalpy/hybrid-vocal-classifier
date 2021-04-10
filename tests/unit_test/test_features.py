"""
test features.extract module
"""
import os

import evfuncs
import numpy as np
import pytest
import yaml

import hvc.audiofileIO
import hvc.features
from hvc.utils import annotation
from hvc.parse.ref_spect_params import refs_dict


@pytest.fixture()
def has_window_error(test_data_dir):
    filename = os.path.join(
        test_data_dir,
        os.path.normpath("cbins/window_error/gy6or6_baseline_220312_0901.106.cbin"),
    )
    index = 19
    return filename, index


class TestFromFile:
    def test_song_w_nan(self, has_window_error, hvc_source_dir):
        """tests that features_arr[ind,:] == np.nan
        where ind is the row corresponding to
        a syllable from a song
        for which a spectrogram could not be generated, and
        so single-syllable features cannot be extracted from it
        """

        with open(
            os.path.join(hvc_source_dir, os.path.normpath("parse/feature_groups.yml"))
        ) as ftr_grp_yaml:
            valid_feature_groups_dict = yaml.load(ftr_grp_yaml, Loader=yaml.FullLoader)
        spect_params = refs_dict["koumura"]
        segment_params = {
            "threshold": 1500,
            "min_syl_dur": 0.01,
            "min_silent_dur": 0.006,
        }
        svm_features = valid_feature_groups_dict["svm"]
        fe = hvc.features.extract.FeatureExtractor(
            spect_params=spect_params,
            segment_params=segment_params,
            feature_list=svm_features,
        )

        filename, index = has_window_error
        annotation_dict = annotation.notmat_to_annot_dict(filename + ".not.mat")
        with pytest.warns(UserWarning):
            extract_dict = fe._from_file(
                filename=filename,
                file_format="evtaf",
                labels_to_use="iabcdefghjk",
                labels=annotation_dict["labels"],
                onsets_Hz=annotation_dict["onsets_Hz"],
                offsets_Hz=annotation_dict["offsets_Hz"],
            )
        ftr_arr = extract_dict["features_arr"]
        assert np.alltrue(np.isnan(ftr_arr[19, :]))

    def test_cbin(self, hvc_source_dir, test_data_dir):
        """tests all features on a single .cbin file"""

        spect_params = refs_dict["tachibana"]
        segment_params = {
            "threshold": 1500,
            "min_syl_dur": 0.01,
            "min_silent_dur": 0.006,
        }
        with open(
            os.path.join(hvc_source_dir, os.path.normpath("parse/feature_groups.yml"))
        ) as ftr_grp_yaml:
            ftr_grps = yaml.load(ftr_grp_yaml, Loader=yaml.FullLoader)

        cbin = os.path.join(
            test_data_dir,
            os.path.normpath(
                "cbins/gy6or6/032412/" "gy6or6_baseline_240312_0811.1165.cbin"
            ),
        )
        annotation_dict = annotation.notmat_to_annot_dict(cbin + ".not.mat")

        for feature_list in (
            ftr_grps["knn"],
            ftr_grps["svm"],
            ["flatwindow"],
        ):
            fe = hvc.features.extract.FeatureExtractor(
                spect_params=spect_params,
                segment_params=segment_params,
                feature_list=feature_list,
            )

            extract_dict = fe._from_file(
                cbin,
                file_format="evtaf",
                labels_to_use="iabcdefghjk",
                labels=annotation_dict["labels"],
                onsets_Hz=annotation_dict["onsets_Hz"],
                offsets_Hz=annotation_dict["offsets_Hz"],
            )

            if "features_arr" in extract_dict:
                ftrs = extract_dict["features_arr"]
                feature_inds = extract_dict["feature_inds"]
                # _from_file should return an ndarray
                assert type(ftrs) == np.ndarray
                # and the number of columns should equal tbe number of feature indices
                # that _from_file determined there were (not necessarily equal to the
                # number of features in the list; some features such as the spectrogram
                # averaged over columns occupy several columns
                assert ftrs.shape[-1] == feature_inds.shape[-1]
                # however the **unique** number of features in feature indices should be
                # equal to the number of items in the feature list
                assert np.unique(feature_inds).shape[-1] == len(feature_list)
            elif "neuralnet_inputs_dict" in extract_dict:
                neuralnet_ftrs = extract_dict["neuralnet_inputs_dict"]
                assert type(neuralnet_ftrs) == dict
            else:
                raise ValueError(
                    "neither features_arr or neuralnet_inputs_dict "
                    "were returned by FeatureExtractor"
                )
