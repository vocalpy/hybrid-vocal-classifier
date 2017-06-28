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
                                           spect_func = 'scipy')

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


    def test_Song(self):
        cbin  = './test_data/cbins/gy6or6_baseline_240312_0811.1165.cbin'
        song = hvc.audiofileIO.Song(filename=cbin,
                                    file_format='evtaf')
