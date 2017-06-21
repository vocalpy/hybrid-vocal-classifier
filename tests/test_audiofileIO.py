import numpy as np
import hvc.audiofileIO

class TestAudiofileIO:
    
    def test_Spectrogram(self):
        spec = hvc.audiofileIO.Spectrogram(samp_freq=32000,
                                           nperseg=128,
                                           noverlap=32,
                                           window='Hann',
                                           freq_cutoffs=[1000, 5000],
                                           filter_func = 'diff',
                                           spec_func = 'scipy')
