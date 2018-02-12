import numpy as np


class SpectScaler:
    """class that scales spectrograms that all have the
    same number of frequency bins. Any input spectrogram
    will be scaled by subtracting off the mean of each
    frequency bin from the 'fit' set of spectrograms, and
    then dividing by the standard deviation of each
    frequency bin from the 'fit' set.
    """

    def __init__(self):
        pass

    def fit(self, spects):
        """fit a SpectScaler.
        takes a 3d array of spectrograms, aligns them all
        horizontally, and then rotates to the right 90° so
        that the columns are frequency bins. Then finds the
        mean and standard deviation of each frequency bin,
        which are used by `transform` method to
        scale other spects

        Parameters
        ----------
        spects : 3-d numpy array
            with dimensions (samples, frequency bins, time bins)
        """

        if spects.ndim != 3:
            raise ValueError('spects should be a 3-d array')

        # concatenate all spects then rotate so
        # Hz bins are columns, i.e., 'features'
        one_long_spect_rotated = np.rot90(np.hstack(spects[:, :, :]))
        self.columnMeans = np.mean(one_long_spect_rotated)
        self.columnStds = np.std(one_long_spect_rotated)

    def _transform(self, spect):
        """
        """

        return (spect - self.columnMeans) / self.columnStds

    def transform(self, spects):
        """transform spect
        """

        if any([not hasattr(self, attr) for attr in ['columnMeans',
                                                     'columnStds']]):
            raise AttributeError('SpectScaler properties are set to None,'
                                 'must call fit method first to set the'
                                 'value of these properties before calling'
                                 'transform')

        if spects.ndim != 3:
            raise ValueError('spects should be a 3-d array')

        z_norm_spects = np.empty(spects.shape)
        for i in range(spects.shape[0]):
            z_norm_spects[i, :, :] = self._transform(spects[i, :, :])

        return z_norm_spects
