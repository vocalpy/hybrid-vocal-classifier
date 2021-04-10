"""
tests knn module
"""
import os
from glob import glob

import evfuncs
import numpy as np
import pytest

import hvc.audiofileIO
from hvc.parse.ref_spect_params import refs_dict
from hvc.features import knn


@pytest.fixture
def notmat_dict(test_data_dir):
    songfiles_dir = os.path.join(
        test_data_dir, os.path.normpath("cbins/gy6or6/032412/*.cbin")
    )
    songfiles_list = glob(songfiles_dir)
    first_song = songfiles_list[0]
    notmat_dict = evfuncs.load_notmat(first_song)
    return notmat_dict


@pytest.fixture
def some_syllables(test_data_dir):
    songfiles_dir = os.path.join(
        test_data_dir, os.path.normpath("cbins/gy6or6/032412/*.cbin")
    )
    songfiles_list = glob(songfiles_dir)
    first_song = songfiles_list[0]
    annot_dict = hvc.utils.annotation.notmat_to_annot_dict(
        notmat=first_song + ".not.mat"
    )

    spect_params = refs_dict["evsonganaly"]
    spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
    raw_song, samp_freq = evfuncs.load_cbin(first_song)

    some_syllables = hvc.audiofileIO.make_syls(
        raw_audio=raw_song,
        samp_freq=samp_freq,
        spect_maker=spect_maker,
        onsets_Hz=annot_dict["onsets_Hz"],
        offsets_Hz=annot_dict["offsets_Hz"],
        labels=annot_dict["labels"],
    )
    return some_syllables


class TestDurationFeatures:
    def _assert_dur_features(self, notmat_dict, dur_type_func):
        """tests to perform on all duration-type features,
        i.e, duration, pre-syllable duration, following-syllable duration,
        silent-gap durations, etc."""
        ftr = dur_type_func(
            notmat_dict["onsets"] / 1000,
            notmat_dict["offsets"] / 1000,
            np.ones(notmat_dict["onsets"].shape).astype(bool),
        )

        assert type(ftr) == np.ndarray
        assert ftr.dtype == np.asarray([1.0]).dtype

    def test_duration(self, notmat_dict):
        self._assert_dur_features(notmat_dict, knn.duration)

    def test_pre_duration(self, notmat_dict):
        self._assert_dur_features(notmat_dict, knn.pre_duration)

    def test_foll_duration(self, notmat_dict):
        self._assert_dur_features(notmat_dict, knn.foll_duration)

    def test_pre_gapdur(self, notmat_dict):
        self._assert_dur_features(notmat_dict, knn.pre_gapdur)

    def test_foll_gapdur(self, notmat_dict):
        self._assert_dur_features(notmat_dict, knn.foll_gapdur)


class ScalarFeaturesTestClass:
    def _assert_scalar_features(self, a_list_of_syllables, scalar_type_func):
        """tests to perform on all scalar features extracted from spectrograms"""
        for a_syllable in a_list_of_syllables:
            out = scalar_type_func(a_syllable)
            assert np.isscalar(out)


class TestMeanAmpSmoothRect(ScalarFeaturesTestClass):
    def test_mn_amp_smooth_rect(self, some_syllables):
        self._assert_scalar_features(some_syllables, knn.mn_amp_smooth_rect)


class TestDeltaAmpSmoothRect(ScalarFeaturesTestClass):
    def test_delta_amp_smooth_rect(self, some_syllables):
        self._assert_scalar_features(some_syllables, knn.delta_amp_smooth_rect)


class TestMeanSpectralEntropy(ScalarFeaturesTestClass):
    def test_mean_spectral_entropy(self, some_syllables):
        self._assert_scalar_features(some_syllables, knn.mean_spect_entropy)


class TestDeltaEntropy(ScalarFeaturesTestClass):
    def test_delta_entropy(self, some_syllables):
        self._assert_scalar_features(some_syllables, knn.delta_entropy)


class TestMeanHiLoRatio(ScalarFeaturesTestClass):
    def test_mean_hi_lo_ratio(self, some_syllables):
        self._assert_scalar_features(some_syllables, knn.mean_hi_lo_ratio)
