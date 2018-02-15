"""
"feature" extraction for neuralnet models.
Each function returns inputs for a specific model.
"""


def flatwindow(song, spect_params, spect_width=0.3):
    """returns input for flatwindow neuralnet model.
    input is stack of spectrograms, all of the same width and height
    width is specified by spect_width parameter
    height is determined by spect_params, i.e. frequency band and
    number of frequency bins in that band

    Parameters
    ----------
    song : song object
        of class audiofileIO.Song
    spect_params : dict
        as declared in config file and returned by parser
    spect_width : float
        width of spectrogram in ms
        default is 0.3, i.e. 300 ms

    Returns
    -------
    spects : 3-d array, (m x n x p)
        spectrograms from m syllables,
        all with n rows and p columns
    """

    return song.make_syl_spects(spect_params=spect_params,
                                syl_spect_width=spect_width,
                                set_syl_spects=False,
                                return_spects=True)
