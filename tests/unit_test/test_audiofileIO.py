"""
test audiofileIO module
"""
import os
from glob import glob

import evfuncs
import numpy as np
import pytest
from scipy.io import wavfile

import hvc.audiofileIO
import hvc.koumura
import hvc.parse.ref_spect_params
from hvc.utils import annotation


@pytest.fixture()
def has_window_error(test_data_dir):
    filename = os.path.join(
        test_data_dir,
        os.path.normpath("cbins/window_error" "/gy6or6_baseline_220312_0901.106.cbin"),
    )
    index = 19
    return filename, index


def test_segment_song(test_data_dir):
    cbins = glob(
        os.path.join(
            test_data_dir,
            os.path.normpath("./data_for_tests/cbins/gy6or6/032312/*.cbin"),
        )
    )
    for cbin in cbins:
        print("loading {}".format(cbin))
        raw_audio, samp_freq = evfuncs.load_cbin(cbin)
        notmat = evfuncs.load_notmat(cbin)
        segment_params = {
            "threshold": notmat["threshold"],
            "min_syl_dur": notmat["min_dur"] / 1000,
            "min_silent_dur": notmat["min_int"] / 1000,
        }
        segmenter = hvc.audiofileIO.Segmenter(**segment_params)
        segment_dict = segmenter.segment(
            raw_audio, samp_freq=samp_freq, method="evsonganaly"
        )
        if segment_dict["onsets_s"].shape == notmat["onsets"].shape:
            np.testing.assert_allclose(
                actual=segment_dict["onsets_s"],
                desired=notmat["onsets"] / 1000,
                rtol=1e-3,
            )
            print("segmentation passed assert_allclose")
        else:
            print("different number of segments in original due to user editing")


class TestAudiofileIO:
    def test_Spectrogram_init(self):
        """#test whether can init a spec object"""
        spect_maker = hvc.audiofileIO.Spectrogram(
            nperseg=128,
            noverlap=32,
            window="Hann",
            freq_cutoffs=[1000, 5000],
            filter_func="diff",
            spect_func="scipy",
        )

    def test_Spectrogram_make(self, has_window_error, test_data_dir):
        """test whether Spectrogram.make works"""
        # test whether make works with .cbin
        cbin = os.path.join(
            test_data_dir,
            os.path.normpath(
                "cbins/gy6or6/032412/" "gy6or6_baseline_240312_0811.1165.cbin"
            ),
        )
        dat, fs = evfuncs.load_cbin(cbin)

        spect_params = hvc.parse.ref_spect_params.refs_dict["evsonganaly"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        spect_params = hvc.parse.ref_spect_params.refs_dict["tachibana"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        # test whether make works with .wav from Koumura dataset
        wav = os.path.join(test_data_dir, os.path.normpath("koumura/Bird0/Wave/0.wav"))
        fs, dat = wavfile.read(wav)
        hvc.koumura.load_song_annot(wav)

        spect_params = hvc.parse.ref_spect_params.refs_dict["koumura"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        # test custom exceptions
        filename, index = has_window_error
        dat, fs = evfuncs.load_cbin(filename)
        notmat_dict = evfuncs.load_notmat(filename)
        onset = notmat_dict["onsets"][index]
        onset = np.round(onset / 1000 * fs).astype(int)
        offset = notmat_dict["offsets"][index]
        offset = np.round(offset / 1000 * fs).astype(int)
        raw_audio = dat[onset:offset]
        spect_params = hvc.parse.ref_spect_params.refs_dict["koumura"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        with pytest.raises(hvc.audiofileIO.WindowError):
            spect_maker.make(raw_audio, fs)

    def test_make_syls(self, test_data_dir):
        """test make_syls function"""

        segment_params = {
            "threshold": 1500,
            "min_syl_dur": 0.01,
            "min_silent_dur": 0.006,
        }

        # test that make_syl_spects works
        # with spect params given individually
        cbin = os.path.join(
            test_data_dir,
            os.path.normpath(
                "cbins/gy6or6/032412/" "gy6or6_baseline_240312_0811.1165.cbin"
            ),
        )
        raw_audio, samp_freq = evfuncs.load_cbin(cbin)
        spect_params = {"nperseg": 512, "noverlap": 480, "freq_cutoffs": [1000, 8000]}
        labels_to_use = "iabcdefghjk"
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        annot_dict = annotation.notmat_to_annot_dict(cbin + ".not.mat")
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        wav = os.path.join(test_data_dir, os.path.normpath("koumura/Bird0/Wave/0.wav"))
        samp_freq, raw_audio = wavfile.read(wav)
        annot_dict = hvc.koumura.load_song_annot(wav)
        labels_to_use = "0123456"
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        # test make_syl_spects works with 'ref' set to 'tachibana'
        raw_audio, samp_freq = evfuncs.load_cbin(cbin)
        spect_params = hvc.parse.ref_spect_params.refs_dict["tachibana"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        annot_dict = annotation.notmat_to_annot_dict(cbin + ".not.mat")
        labels_to_use = "iabcdefghjk"
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        wav = os.path.join(test_data_dir, os.path.normpath("koumura/Bird0/Wave/0.wav"))
        samp_freq, raw_audio = wavfile.read(wav)
        labels_to_use = "0123456"
        annot_dict = hvc.koumura.load_song_annot(wav)
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        # test make_syl_spects works with 'ref' set to 'koumura'
        raw_audio, samp_freq = evfuncs.load_cbin(cbin)
        spect_params = hvc.parse.ref_spect_params.refs_dict["koumura"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        annot_dict = annotation.notmat_to_annot_dict(cbin + ".not.mat")
        labels_to_use = "iabcdefghjk"
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        wav = os.path.join(test_data_dir, os.path.normpath("koumura/Bird0/Wave/0.wav"))
        samp_freq, raw_audio = wavfile.read(wav)
        labels_to_use = "0123456"
        annot_dict = hvc.koumura.load_song_annot(wav)
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        # test that make_syl_spects works the same way when
        # using evsonganaly
        raw_audio, samp_freq = evfuncs.load_cbin(cbin)
        spect_params = hvc.parse.ref_spect_params.refs_dict["evsonganaly"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        annot_dict = annotation.notmat_to_annot_dict(cbin + ".not.mat")
        labels_to_use = "iabcdefghjk"
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

        wav = os.path.join(test_data_dir, os.path.normpath("koumura/Bird0/Wave/0.wav"))
        samp_freq, raw_audio = wavfile.read(wav)
        annot_dict = hvc.koumura.load_song_annot(wav)
        labels_to_use = "0123456"
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annot_dict["labels"],
            annot_dict["onsets_Hz"],
            annot_dict["offsets_Hz"],
            labels_to_use=labels_to_use,
        )

    def test_window_error_set_to_nan(self, has_window_error):
        """check that, if an audio file raises a window error for Spectrogram.make
        for a certain syllable, then that syllable's spectrogram is set to np.nan
        """
        filename, index = has_window_error
        raw_audio, samp_freq = evfuncs.load_cbin(filename)
        spect_params = hvc.parse.ref_spect_params.refs_dict["koumura"]
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        annotation_dict = annotation.notmat_to_annot_dict(filename + ".not.mat")
        syls = hvc.audiofileIO.make_syls(
            raw_audio,
            samp_freq,
            spect_maker,
            annotation_dict["labels"],
            annotation_dict["onsets_Hz"],
            annotation_dict["offsets_Hz"],
        )
        assert syls[index].spect is np.nan
