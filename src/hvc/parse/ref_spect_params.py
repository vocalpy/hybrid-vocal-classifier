refs_dict = {
    # tachibana has remove_dc set to False because some features
    # require that zero-frequency component, e.g. the cepstrum.
    # Might be a more elegant way to do this, e.g. just add a row
    # of zeros to the spectrum before the inverse FFT and not have
    # all this overhead. But this is the current solution.
    # The same goes for freq_cutoffs, which is set to None here but
    # for some features in features.tachibana there is a subset of
    # frequencies used, and that subset is the default for those
    # frequencies. E.g., pitch looks between 0.5 and 6kHz. Also the
    # spectrogram is log transformed, to convert to dBs, but
    # not for all features so here it is set to False.
    'tachibana': {
        'nperseg': 256,
        'noverlap': 192,
        'window': 'Hann',
        'freq_cutoffs': None, 
        'filter_func': 'diff',
        'spect_func': 'mpl',
        'log_transform_spect': False,
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
