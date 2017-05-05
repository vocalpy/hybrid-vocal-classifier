import collections
import warnings

import numpy as np

from . import tachibana

feature_switch_case_dict = {
    'mean spectrum' : tachibana.mean_spectrum,
    'mean delta spectrum' : tachibana.mean_delta_spectrum,
    'mean cepstrum' : tachibana.mean_cepstrum,
    'mean delta cepstrum' : tachibana.mean_delta_cepstrum,
    'duration' : tachibana.duration,
    'mean spectral centroid' : tachibana.mean_spectral_centroid,
    'mean spectral spread' : tachibana.mean_spectral_spread,
    'mean spectral skewness' : tachibana.mean_spectral_skewness,
    'mean spectral kurtosis' : tachibana.mean_spectral_kurtosis,
    'mean spectral flatness' : tachibana.mean_spectral_flatness,
    'mean spectral slope' : tachibana.mean_spectral_slope,
    'mean pitch' : tachibana.mean_pitch,
    'mean pitch goodness' : tachibana.mean_pitch_goodness,
    'mean delta spectral centroid' : tachibana.mean_delta_spectral_centroid,
    'mean delta spectral spread' : tachibana.mean_delta_spectral_spread,
    'mean delta spectral skewness' : tachibana.mean_delta_spectral_skewness,
    'mean delta spectral kurtosis' : tachibana.mean_delta_spectral_kurtosis,
    'mean delta spectral flatness' : tachibana.mean_delta_spectral_flatness,
    'mean delta spectral slope' : tachibana.mean_delta_spectral_slope,
    'mean delta pitch' : tachibana.mean_delta_pitch,
    'mean delta pitch goodness' : tachibana.mean_delta_pitch_goodness,
    'zero crossings' : tachibana.zero_crossings,
    'mean amplitude' : tachibana.mean_amplitude,
    'mean delta amplitude' : tachibana.mean_delta_amplitude
    }

def _flatten(list_to_flatten):
    """
    flattens list of extracted features.
    Acts as a generator.

    Parameters
    ----------
    list_to_flatten

    Returns
    -------
    flattened list. Note this is a generator function.
    """

    for element in list_to_flatten:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(element)
        else:
            yield element

def _extract_features(feature_list,syllable):
    """
    helper function
    
    Parameters
    ----------
    feature_list
    syllable

    Returns
    -------
    feature_arr : nd-array
        list of extracted features, flattened and converted to numpy array
    """
    extracted_features = []
    for feature in feature_list:
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                extracted_features.append(feature_switch_case_dict[feature](syllable))
            except Warning as e:
                print('error found:', e)
                import pdb;pdb.set_trace()
    return np.asarray(_flatten(extracted_features))

def extract_features_from_syllable(feature_list,syllable,feature_groups=None):
    """
    function called by main feature extraction function, extract, that
    does the actual work of looping through the feature list and calling
    the functions that extract the features from the syllable
    
    Parameters
    ----------
    feature_list : list of strings, or list of list of strings
        from extract config
    syllable : syllable object
    feature_groups : list of strings or ints
        default is None
        if feature_list is a list of lists and feature_groups is
        not None then the function will return features_dict
        where each key is a feature group and the value associated
        with that key is the 1d array with all features for
        that feature group
        
    Returns
    -------
    either features_dict or feature_arr
    """
    if all(isinstance(element, list) for element in feature_list):
        # if feature_list is a list of lists
        features_dict = {}
        for ftr_grp,ftr_list in zip(feature_groups,feature_list):
            features_dict[ftr_grp] = _extract_features(ftr_list,syllable)
        return features_dict
    else:
        return _extract_features(feature_list,syllable)