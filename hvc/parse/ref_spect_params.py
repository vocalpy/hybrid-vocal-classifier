refs_dict = {
    # tachibana has remove_dc set to False because some features
    # require that zero-frequency component, e.g. the cepstrum.
    # Might be a more elegant way to do this, e.g. just add a row
    # of zeros to the spectrum before the inverse FFT and not have
    # all this overhead. But this is the current solution.
    'tachibana': {
        'nperseg': 256,
        'noverlap': 192,
        'window': 'Hann',  # Hann window
        'freq_cutoffs': [0, 16000],  # basically no bandpass, as in Tachibana
        'filter_func': 'diff',
        'spect_func': 'mpl',
        'log_transform_spect': False,  # see tachibana feature docs
        'thresh': None,
        'remove_dc': False
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
        'filter_func': 'bandpass_filtfilt',
        'spect_func': 'mpl',
        'log_transform_spect': False,
        'thresh': None
    }
}
