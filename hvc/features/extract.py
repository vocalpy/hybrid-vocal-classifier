import collections
import warnings

import numpy as np

from . import tachibana, knn

spectral_features_switch_case_dict = {
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

temporal_features_switch_case_dict = {
    'duration group' : knn.duration,
    'preceding syllable duration' : knn.pre_duration,
    'following syllable duration' : knn.foll_duration,
    'preceding silent gap duration' : knn.pre_gapdur,
    'following silent gap duration' : knn.foll_gapdur
 }

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

    for feature in feature_list:
        if 'extracted_features' in locals():
            extracted_features = np.append(extracted_features,
                                           spectral_features_switch_case_dict[feature](syllable))
        else:
            extracted_features = spectral_features_switch_case_dict[feature](syllable)
    return extracted_features

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

def extract_features_from_file(filename,file_format,feature_list):
    """
    
    Parameters
    ----------
    filename : string
    
    file_format : string
    
    feature_list : list of strings

    Returns
    -------
    features_arr : numpy array
    
    labels : list of chars
    """

    for feature in feature_list:
        if feature in spectral_features_switch_case_dict:
            if 'syls' not in locals():
                if file_format == 'evtaf':
                    syls, labels = evfuncs.get_syls(filename,
                                                    extract_config['spect_params'],
                                                    todo['labelset'])

                elif file_format == 'koumura':
                    syls, labels = koumura.get_syls(filename,
                                                    extract_config['spect_params'],
                                                    todo['labelset'])

            for syl in syls:
                if is_ftr_list_of_lists:
                    ftrs = extract_features_from_syllable(todo['feature_list'],
                                                                           syl,
                                                                           feature_groups)
                else:
                    ftrs = features.extract.extract_features_from_syllable(todo['feature_list'],
                                                                           syl)
        elif feature in duration_features_switch_case_dict:
            ftrs = ftrs.extract.extract_duration_features

        return features_arr,labels