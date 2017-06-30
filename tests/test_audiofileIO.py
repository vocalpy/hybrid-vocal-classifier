"""
test audiofileIO module
"""

import pytest
import numpy as np

import hvc.audiofileIO

class TestAudiofileIO:
    
    def test_Spectrogram_init(self):
        #test whether can init a spec object
        spec = hvc.audiofileIO.Spectrogram(nperseg=128,
                                           noverlap=32,
                                           window='Hann',
                                           freq_cutoffs=[1000, 5000],
                                           filter_func='diff',
                                           spect_func='scipy')

        #test whether init works with 'ref' parameter
        #instead of passing spect params
        spec = hvc.audiofileIO.Spectrogram(ref='tachibana')

        spec = hvc.audiofileIO.Spectrogram(ref='koumura')

        #test that specify 'ref' and specifying other params raises warning
        #(because other params specified will be ignored)
        with pytest.warns(UserWarning):
            spec = hvc.audiofileIO.Spectrogram(nperseg=512,
                                               ref='tachibana')
        with pytest.warns(UserWarning):
            spec = hvc.audiofileIO.Spectrogram(nperseg=512,
                                               ref='tachibana')

        with pytest.warns(UserWarning):
            spec = hvc.audiofileIO.Spectrogram(spect_func='scipy',
                                               ref='tachibana')

    def test_Song_init(self):

        cbin  = './test_data/cbins/gy6or6_baseline_240312_0811.1165.cbin'
        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf')

        wav = './test_data/koumura/Bird0/Wave/0.wav'
        song = hvc.audiofileIO.Song(filename=wav,
                                    file_format='koumura')

    def test_Song_set_and_make_syls(self):

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

        cbin_song.make_syl_spects(spect_params={'ref':'tachibana'})
        wav_song.make_syl_spects(spect_params={'ref':'tachibana'})
