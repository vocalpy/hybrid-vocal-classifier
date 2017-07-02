"""
test audiofileIO module
"""

import pytest
from scipy.io import wavfile

import hvc.audiofileIO
import hvc.evfuncs
import hvc.koumura

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

    def test_Spectrogram_make(self):
        """ test whether Spectrogram.make works
        """
        #test whether make works with .cbin
        cbin  = './test_data/cbins/gy6or6_baseline_240312_0811.1165.cbin'
        dat, fs = hvc.evfuncs.load_cbin(cbin)
        spect_maker = hvc.audiofileIO.Spectrogram(ref='tachibana')
        spect,freq_bins, time_bins = spect_maker.make(dat, fs)
        spect_maker = hvc.audiofileIO.Spectrogram(ref='koumura')
        spect,freq_bins, time_bins = spect_maker.make(dat, fs)

        #test whether make works with .wav from Koumura dataset
        wav = './test_data/koumura/Bird0/Wave/0.wav'
        fs, dat = wavfile.read(wav)
        spect_maker = hvc.audiofileIO.Spectrogram(ref='tachibana')
        spect,freq_bins, time_bins = spect_maker.make(dat, fs)
        spect_maker = hvc.audiofileIO.Spectrogram(ref='koumura')
        spect,freq_bins, time_bins = spect_maker.make(dat, fs)

    def test_Song_init(self):
        """test whether Song object inits properly
        """
        cbin  = './test_data/cbins/gy6or6_baseline_240312_0811.1165.cbin'
        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf')

        wav = './test_data/koumura/Bird0/Wave/0.wav'
        song = hvc.audiofileIO.Song(filename=wav,
                                    file_format='koumura')

    def test_Song_set_and_make_syls(self):
        """
        """
        cbin  = './test_data/cbins/gy6or6_baseline_240312_0811.1165.cbin'
        cbin_song = hvc.audiofileIO.Song(filename=cbin,
                                         file_format='evtaf')
        cbin_song.set_syls_to_use('iabcdef')

        wav = './test_data/koumura/Bird0/Wave/0.wav'
        wav_song = hvc.audiofileIO.Song(filename=wav,
                                        file_format='koumura')
        wav_song.set_syls_to_use('0123456')

        spect_params = {
            'nperseg': 512,
            'noverlap': 480,
            'freq_cutoffs': [1000, 8000]}
        cbin_song.make_syl_spects(spect_params)
        wav_song.make_syl_spects(spect_params)

        cbin_song.make_syl_spects(spect_params={'ref': 'tachibana'})
        wav_song.make_syl_spects(spect_params={'ref': 'tachibana'})

        cbin_song.make_syl_spects(spect_params={'ref': 'koumura'})
        wav_song.make_syl_spects(spect_params={'ref': 'koumura'})
