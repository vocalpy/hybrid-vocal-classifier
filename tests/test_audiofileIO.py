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
                                           filter_func = 'diff',
                                           spec_func = 'scipy')

        #test whether init works with 'ref' parameter
        #instead of passing spect params
        spec = hvc.audiofileIO.Spectrogram(ref='tachibana')

        spec = hvc.audiofileIO.Spectrogram(ref='koumura')

        #test whether lack of samp_freq raises error
        with pytest.raises(ValueError):
            spec = hvc.audiofileIO.Spectrogram()

        #test that ref with other params raises warning
        with pytest.warns(UserWarning):
            spec = hvc.audiofileIO.Spectrogram(nperseg=512,
                                               ref='tachibana')

    def test_Song(self):
        cbin  = './test_data/cbins/gy6or6_baseline_240312_0811.1165.cbin'
        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf')
