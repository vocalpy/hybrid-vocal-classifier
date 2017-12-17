"""
test audiofileIO module
"""

import os
from glob import glob

import pytest
from scipy.io import wavfile
import numpy as np

import hvc.audiofileIO
import hvc.evfuncs
import hvc.koumura
import hvc.parse.ref_spect_params

@pytest.fixture()
def has_window_error():
    filename = os.path.join(os.path.dirname(__file__),
                            os.path.normpath('./test_data/cbins/window_error'
                            '/gy6or6_baseline_220312_0901.106.cbin'))
    index = 19
    return filename, index


def test_segment_song():
    cbins = glob(
        os.path.normpath('./test_data/cbins/gy6or6/032312/*.cbin'))
    for cbin in cbins:
        print('loading {}'.format(cbin))
        data, samp_freq = hvc.evfuncs.load_cbin(cbin)
        spect_params = hvc.parse.ref_spect_params.refs_dict['evsonganaly']
        amp = hvc.evfuncs.smooth_data(data, samp_freq, spect_params['freq_cutoffs'])
        notmat = hvc.evfuncs.load_notmat(cbin)
        segment_params = {'threshold': notmat['threshold'],
                          'min_syl_dur': notmat['min_dur'] / 1000,
                          'min_silent_dur': notmat['min_int'] / 1000}
        onsets, offsets = hvc.audiofileIO.segment_song(amp,
                                                       segment_params,
                                                       samp_freq=samp_freq)
        if onsets.shape == notmat['onsets'].shape:
            np.testing.assert_allclose(actual=onsets,
                                       desired=notmat['onsets'] / 1000,
                                       rtol=1e-3)
            print('segmentation passed assert_allclose')
        else:
            print('different number of segments in original due to user editing')


class TestAudiofileIO:

    def test_Spectrogram_init(self):
        """#test whether can init a spec object
        """
        spect_maker = hvc.audiofileIO.Spectrogram(nperseg=128,
                                                  noverlap=32,
                                                  window='Hann',
                                                  freq_cutoffs=[1000, 5000],
                                                  filter_func='diff',
                                                  spect_func='scipy')

    def test_Spectrogram_make(self, has_window_error):
        """ test whether Spectrogram.make works
        """
        # test whether make works with .cbin
        cbin = os.path.join(os.path.dirname(__file__),
                            os.path.normpath('test_data/cbins/gy6or6/032412/'
                            'gy6or6_baseline_240312_0811.1165.cbin'))
        dat, fs = hvc.evfuncs.load_cbin(cbin)

        spect_params = hvc.parse.ref_spect_params.refs_dict['evsonganaly']
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        spect_params = hvc.parse.ref_spect_params.refs_dict['tachibana']
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        # test whether make works with .wav from Koumura dataset
        wav = os.path.join(os.path.dirname(__file__),
                           os.path.normpath('test_data/koumura/Bird0/Wave/0.wav'))
        fs, dat = wavfile.read(wav)

        spect_params = hvc.parse.ref_spect_params.refs_dict['koumura']
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        # test custom exceptions
        filename, index = has_window_error
        dat, fs = hvc.evfuncs.load_cbin(filename)
        notmat_dict = hvc.evfuncs.load_notmat(filename)
        onset = notmat_dict['onsets'][index]
        onset = np.round(onset / 1000 * fs).astype(int)
        offset = notmat_dict['offsets'][index]
        offset = np.round(offset / 1000 * fs).astype(int)
        raw_audio = dat[onset:offset]
        spect_params = hvc.parse.ref_spect_params.refs_dict['koumura']
        spect_maker = hvc.audiofileIO.Spectrogram(**spect_params)
        with pytest.raises(hvc.audiofileIO.WindowError):
            spect_maker.make(raw_audio, fs)

    def test_Song_init(self):
        """test whether Song object inits properly
        """

        segment_params = {
            'threshold': 1500,
            'min_syl_dur': 0.01,
            'min_silent_dur': 0.006
        }

        cbin = os.path.join(os.path.dirname(__file__),
                            os.path.normpath('test_data/cbins/gy6or6/032412/'
                            'gy6or6_baseline_240312_0811.1165.cbin'))
        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf',
                                    segment_params=segment_params)

        wav = os.path.join(os.path.dirname(__file__),
                           os.path.normpath('test_data/koumura/Bird0/Wave/0.wav'))
        song = hvc.audiofileIO.Song(filename=wav,
                                    file_format='koumura')


    def test_Song_set_and_make_syls(self):
        """test that set_syls_to_use and make_syl_spects work
        """

        segment_params = {
            'threshold': 1500,
            'min_syl_dur': 0.01,
            'min_silent_dur': 0.006
        }

        # test that make_syl_spects works
        # with spect params given individually
        spect_params = {
            'nperseg': 512,
            'noverlap': 480,
            'freq_cutoffs': [1000, 8000]}
        cbin = os.path.join(os.path.dirname(__file__),
                            os.path.normpath('test_data/cbins/gy6or6/032412/'
                            'gy6or6_baseline_240312_0811.1165.cbin'))
        cbin_song = hvc.audiofileIO.Song(filename=cbin,
                                         file_format='evtaf',
                                         segment_params=segment_params)
        cbin_song.set_syls_to_use('iabcdefghjk')
        cbin_song.make_syl_spects(spect_params)

        wav = os.path.join(os.path.dirname(__file__),
                           os.path.normpath('test_data/koumura/Bird0/Wave/0.wav'))
        wav_song = hvc.audiofileIO.Song(filename=wav,
                                        file_format='koumura')
        wav_song.set_syls_to_use('0123456')
        wav_song.make_syl_spects(spect_params)

        # test make_syl_spects works with 'ref' set to 'tachibana'
        cbin_song = hvc.audiofileIO.Song(filename=cbin,
                                         file_format='evtaf',
                                         segment_params=segment_params)
        cbin_song.set_syls_to_use('iabcdefghjk')
        spect_params = hvc.parse.ref_spect_params.refs_dict['tachibana']
        cbin_song.make_syl_spects(spect_params=spect_params)

        wav = os.path.join(os.path.dirname(__file__),
                           os.path.normpath('test_data/koumura/Bird0/Wave/0.wav'))
        wav_song = hvc.audiofileIO.Song(filename=wav,
                                        file_format='koumura')
        wav_song.set_syls_to_use('0123456')
        spect_params = hvc.parse.ref_spect_params.refs_dict['tachibana']
        wav_song.make_syl_spects(spect_params=spect_params)

        # test make_syl_spects works with 'ref' set to 'koumura'
        cbin_song = hvc.audiofileIO.Song(filename=cbin,
                                         file_format='evtaf',
                                         segment_params=segment_params)
        cbin_song.set_syls_to_use('iabcdefghjk')
        spect_params = hvc.parse.ref_spect_params.refs_dict['koumura']
        cbin_song.make_syl_spects(spect_params=spect_params)

        wav = os.path.join(os.path.dirname(__file__),
                           os.path.normpath('test_data/koumura/Bird0/Wave/0.wav'))
        wav_song = hvc.audiofileIO.Song(filename=wav,
                                        file_format='koumura')
        wav_song.set_syls_to_use('0123456')
        spect_params = hvc.parse.ref_spect_params.refs_dict['koumura']
        wav_song.make_syl_spects(spect_params=spect_params)

        # test that make_syl_spects works the same way when
        #
        cbin_song = hvc.audiofileIO.Song(filename=cbin,
                                         file_format='evtaf',
                                         segment_params=segment_params)
        cbin_song.set_syls_to_use('iabcdefghjk')

    def check_window_error_set_to_nan(self, has_window_error):
        """check that, if an audio file raises a window error for Spectrogram.make
        for a certain syllable, then that syllable's spectrogram is set to np.nan
        """
        filename, index = has_window_error
        segment_params = {
            'threshold': 1500,
            'min_syl_dur': 0.01,
            'min_silent_dur': 0.006
        }
        cbin_song = hvc.audiofileIO.Song(filename=filename,
                                         file_format='evtaf',
                                         segment_params=segment_params)
        cbin_song.set_syls_to_use('iabcdefghjk')
        spect_params = hvc.parse.ref_spect_params.refs_dict['koumura']
        cbin_song.make_syl_spects(spect_params)
        assert cbin_song.syls[index] is np.nan
