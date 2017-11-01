"""
functions that convert predicted labels to file types used by different programs,
e.g., evsonganaly (GUI for labeling song), Sound Analysis Pro, etc.
"""
import os

import numpy as np
import scipy.io


def to_notmat(songfile_name, pred_labels, clf_file, samp_freq, segment_params,
              onsets_s, offsets_s, alternate_path=None):
    """converts predicted labels into a .not.mat file
    that can be read by evsonganaly (MATLAB GUI for labeling song)

    Parameters
    ----------
    songfile_name : str
        filename
    pred_labels : ndarray
        output from model / classifier
    clf_file : str
        name of file from which model / classifier was loaded
    samp_freq : int
        sampling frequency of audio file
    segment_params : dict
        parameters used to find segments, i.e. song syllables.
        Will be same segment_params used for training model that made predictions,
        taken from feature_file.
    onsets_s : ndarray
        onsets of syllables in seconds.
        Found by applying segment_params applied to audio file.
    offsets_s : ndarray
        offsets of syllables in seconds.
    alternate_path : str
        Alternate path to which notmat files should be saved
        if .not.mat files with same name already exist in directory
        containing audio files.
        Default is None.
        Labelpredict assigns the output_dir from the
        predict.config.yml file as an alternate.

    Returns
    -------
    None
    """

    SHOULD_BE_DOUBLE = ['Fs',
                        'min_dur',
                        'min_int',
                        'offsets',
                        'onsets',
                        'sm_win',
                        'threshold']

    # notmat files have onsets/offsets in units of ms
    # need to convert
    onsets = onsets_s * 1e3
    offsets = offsets_s * 1e3

    # chr() to convert back to character from uint32
    if pred_labels.dtype == 'int32':
        pred_labels = [chr(val) for val in pred_labels]
    elif pred_labels.dtype == '<U1':
        pred_labels = pred_labels.tolist()
    # convert into one long string, what evsonganaly expects
    pred_labels = ''.join(pred_labels)
    notmat_dict = {'Fs': samp_freq,
                   'min_dur': segment_params['min_syl_dur'],
                   'min_int': segment_params['min_silent_dur'],
                   'offsets': offsets,
                   'onsets' : onsets,
                   'sm_win': 2,
                   'threshold': segment_params['threshold']
                   }
    notmat_dict['labels'] = pred_labels
    notmat_dict['classifier_file'] = clf_file
    # evsonganaly/Matlab expects all vars as double
    for key, val in notmat_dict.items():
        if key in SHOULD_BE_DOUBLE:
            try:
                notmat_dict[key] = val.astype('d')
            except AttributeError:  # gross hack
                notmat_dict[key] = np.asarray([val]).astype('d')[0]

    notmat_name = songfile_name + '.not.mat'
    if os.path.exists(notmat_name):
        if alternate_path:
            alternate_notmat_name = os.path.join(alternate_path,
                                                 os.path.basename(songfile_name)
                                                 + '.not.mat')
            if os.path.exists(alternate_notmat_name):
                raise FileExistsError('Tried to save {} in alternate path {},'
                                      'but file already exists'.format(alternate_notmat_name,
                                                                       alternate_path))
            else:
                scipy.io.savemat(alternate_notmat_name, notmat_dict)
        else:
            raise FileExistsError('{} already exists but no alternate path provided'
                                  .format(notmat_name))
    else:
        scipy.io.savemat(notmat_name, notmat_dict)


def to_sap():
    pass

def to_koumura():
    pass