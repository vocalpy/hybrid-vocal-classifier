"""
test audiofileIO module
"""

import pytest
from scipy.io import wavfile
import numpy as np

import hvc.audiofileIO
import hvc.evfuncs
import hvc.koumura

@pytest.fixture()
def has_window_error():
    filename = './test_data/cbins/window_error/gy6or6_baseline_220312_0901.106.cbin'
    index = 19
    return filename, index


class TestAudiofileIO:

    def test_Spectrogram_init(self):
        """#test whether can init a spec object
        """
        spec = hvc.audiofileIO.Spectrogram(nperseg=128,
                                           noverlap=32,
                                           window='Hann',
                                           freq_cutoffs=[1000, 5000],
                                           filter_func='diff',
                                           spect_func='scipy')

        #test whether init works with 'ref' parameter
        #instead of passing spect params
        spect_maker = hvc.audiofileIO.Spectrogram(ref='tachibana')

        spect_maker = hvc.audiofileIO.Spectrogram(ref='koumura')

        #test that specify 'ref' and specifying other params raises warning
        #(because other params specified will be ignored)
        with pytest.warns(UserWarning):
            spect_maker = hvc.audiofileIO.Spectrogram(nperseg=512,
                                                      ref='tachibana')
        with pytest.warns(UserWarning):
            spect_maker = hvc.audiofileIO.Spectrogram(nperseg=512,
                                                    ref='tachibana')

        with pytest.warns(UserWarning):
            spect_maker = hvc.audiofileIO.Spectrogram(spect_func='scipy',
                                                      ref='tachibana')

    def test_Spectrogram_make(self, has_window_error):
        """ test whether Spectrogram.make works
        """
        # test whether make works with .cbin
        cbin = './test_data/cbins/gy6or6/032412/gy6or6_baseline_240312_0811.1165.cbin'
        dat, fs = hvc.evfuncs.load_cbin(cbin)

        spect_maker = hvc.audiofileIO.Spectrogram(ref='tachibana')
        spect,freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        spect_maker = hvc.audiofileIO.Spectrogram(ref='koumura')
        spect,freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        # test whether make works with .wav from Koumura dataset
        wav = './test_data/koumura/Bird0/Wave/0.wav'
        fs, dat = wavfile.read(wav)

        spect_maker = hvc.audiofileIO.Spectrogram(ref='tachibana')
        spect, freq_bins, time_bins = spect_maker.make(dat, fs)
        assert spect.shape[0] == freq_bins.shape[0]
        assert spect.shape[1] == time_bins.shape[0]

        spect_maker = hvc.audiofileIO.Spectrogram(ref='koumura')
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
        spect_maker = hvc.audiofileIO.Spectrogram(ref='koumura')
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

        cbin = './test_data/cbins/gy6or6/032412/gy6or6_baseline_240312_0811.1165.cbin'
        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf',
                                    segment_params=segment_params)

        wav = './test_data/koumura/Bird0/Wave/0.wav'
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

        cbin = './test_data/cbins/gy6or6/032412/gy6or6_baseline_240312_0811.1165.cbin'
        cbin_song = hvc.audiofileIO.Song(filename=cbin,
                                         file_format='evtaf',
                                         segment_params=segment_params)
        cbin_song.set_syls_to_use('iabcdefghjk')

        wav = './test_data/koumura/Bird0/Wave/0.wav'
        wav_song = hvc.audiofileIO.Song(filename=wav,
                                        file_format='koumura')
        wav_song.set_syls_to_use('0123456')

        # test that make_syl_spects works with spect params given individually
        spect_params = {
            'nperseg': 512,
            'noverlap': 480,
            'freq_cutoffs': [1000, 8000]}
        cbin_song.make_syl_spects(spect_params)
        wav_song.make_syl_spects(spect_params)

        # test make_syl_spects works with 'ref' set to 'tachibana'
        cbin_song.make_syl_spects(spect_params={'ref': 'tachibana'})
        wav_song.make_syl_spects(spect_params={'ref': 'tachibana'})

        # test make_syl_spects works with 'ref' set to 'koumura'
        cbin_song.make_syl_spects(spect_params={'ref': 'koumura'})
        wav_song.make_syl_spects(spect_params={'ref': 'koumura'})

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
        cbin_song.make_syl_spects(spect_params={'ref': 'koumura'})
        assert cbin_song.syls[index] is np.nan
