import collections
import warnings

import numpy as np

from . import tachibana, knn
from hvc import audiofileIO

single_syl_features_switch_case_dict = {
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

multiple_syl_features_switch_case_dict = {
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

def from_file(filename, file_format, feature_list, spect_params, labels_to_use):
    """
    
    Parameters
    ----------
    filename : string
    
    file_format : string
    
    feature_list : list of strings

    spect_params : 

    labels_to_use :

    Returns
    -------
    features_arr : numpy array
    
    labels : list of chars
    """

    song = audiofileIO.song(filename,file_format)

    for current_feature in feature_list:
        if current_feature in single_syl_features_switch_case_dict:
            if not hasattr(song, 'syls'):
                song.set_syls_to_use(labels_to_use)
                song.make_syl_spects(spect_params)
            if 'curr_feature_arr' in locals():
                del curr_feature_arr
            for syl in song.syls:
                ftr = single_syl_features_switch_case_dict[current_feature](syl)
                if 'curr_feature_arr' in locals():
                    if np.isscalar(ftr):
                        curr_feature_arr = np.append(curr_feature_arr, ftr)
                    else:
                        # note have to add dimension with newaxis because np.concat requires
                        # same number of dimensions, but extract_features returns 1d.
                        # Decided to keep it explicit that we go to 2d here.
                        curr_feature_arr = np.concatenate((curr_feature_arr,
                                                           ftr[np.newaxis,:]),
                                                          axis=0)
                else:
                    if np.isscalar(ftr):
                        curr_feature_arr = ftr
                    else:
                        curr_feature_arr = ftr[np.newaxis,:] # make 2-d for concatenate
            if 'features_arr' in locals():
                if np.isscalar(ftr):
                    features_arr = np.concatenate((features_arr,
                                                   curr_feature_arr[np.newaxis, :].T),
                                                  axis=1)
                else:
                    features_arr = np.concatenate((features_arr,
                                                   curr_feature_arr),
                                                  axis=1)
            else:
                features_arr = curr_feature_arr

        elif current_feature in multiple_syl_features_switch_case_dict:
            curr_feature_arr = multiple_syl_features_switch_case_dict(current_feature, song)
            curr_feature_arr = curr_feature_arr[np.asarray(use_syl_or_not)]
            if 'features_arr' in locals():
                features_arr = np.concatenate((features_arr,
                                               curr_feature_arr[np.newaxis,:]),
                                              axis=1)
            else:
                features_arr = curr_feature_arr

    labels = [label for label in song.labels if label in labels_to_use]
    return features_arr,labels