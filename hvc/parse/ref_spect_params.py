import numpy as np
import scipy.signal

refs = {
    'tachibana': {
        'nperseg': 256,
        'noverlap': 192,
        'window': np.hanning(256),  # Hann window
        'freq_cutoffs': [10, 15990],  # basically no bandpass, as in Tachibana
        'filter_func': 'diff',
        'spect_func': 'mpl',
        'log_transform_spect': False,  # see tachibana feature docs
        'thresh': None
    },

    'koumura': {
        'nperseg': 512,
        'noverlap': 480,
        'window': scipy.signal.slepian(512, 4 / 512),  # slepian, called dpss in Koumura
        'freq_cutoffs': [1000, 8000],
        'filter_func': None,
        'spect_func': 'scipy',
        'log_transform_spect': True,
        'thresh': None
    }
}
