import numpy as np
import scipy.signal

refs_dict = {
    'tachibana': {
        'nperseg': 256,
        'noverlap': 192,
        'window': 'Hann',  # Hann window
        'freq_cutoffs': [10, 15990],  # basically no bandpass, as in Tachibana
        'filter_func': 'diff',
        'spect_func': 'mpl',
        'log_transform_spect': False,  # see tachibana feature docs
        'thresh': None
    },

    'koumura': {
        'nperseg': 512,
        'noverlap': 480,
        'window': 'dpss',
        'freq_cutoffs': [1000, 8000],
        'filter_func': None,
        'spect_func': 'scipy',
        'log_transform_spect': True,
        'thresh': None
    },

    'evsonganaly': {
        'nperseg': 512,
        'noverlap': 409,
        'window': 'Hann',
        'freq_cutoffs': [500, 10000],
        'filter_func': None,
        'spect_func': 'mpl',
        'log_transform_spect': False,
        'thresh': None
    }
}
