"""utility functions for features"""

import joblib


def load_feature_file(filename):
    """loads a feature file.
    Wrapper around joblib.load.

    Parameters
    ----------
    filename : str
        name of feature file

    Returns
    -------
    feature_file : dict
        containing features, labels, and metadata related to feature extraction
    """

    return joblib.load(filename)
