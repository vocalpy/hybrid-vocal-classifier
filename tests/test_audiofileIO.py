"""
test audiofileIO module
"""

import pytest
import numpy as np
import hvc.audiofileIO

class TestAudiofileIO:
    
    def test_Spectrogram_init(self):
        #test whether can init a spec object
        spec = hvc.audiofileIO.Spectrogram(samp_freq=32000,
                                           nperseg=128,
                                           noverlap=32,
                                           window='Hann',
                                           freq_cutoffs=[1000, 5000],
                                           filter_func = 'diff',
                                           spec_func = 'scipy')

        #test whether lack of samp_freq raises error
        with pytest.raises(ValueError):
            spec = hvc.audiofileIO.Spectrogram()

        #test that def with other params raises warning
        with pytest.warns(UserWarning):
            spec = hvc.audiofileIO.Spectrogram(samp_freq=32000,
                                               nperseg=512,
                                               ref='tachibana')