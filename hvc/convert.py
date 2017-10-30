"""
functions that convert predicted labels to file types used by different programs,
e.g., evsonganaly (GUI for labeling song), Sound Analysis Pro, etc.
"""
import numpy as np
import scipy.io

def to_notmat(songfile_name, pred_labels, clf_file, samp_freq, segment_params,
              onsets_s, offsets_s):
    """converts predicted labels into a .not.mat file
    that can be read by evsonganaly.m (MATLAB GUI for labeling song)

    Parameters
    ----------
    songfile_name: str
        filename
    pred_labels: ndarray
        output from model / classifier
    clf_file: str
        name of file from which model / classifier was loaded

    Returns
    -------
    None.
    Saves .not.mat file with additional information
        predicted_labels
        classifier_file
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
    scipy.io.savemat(songfile_name + '.not.mat', notmat_dict)


def to_sap():
    pass

def to_koumura():
    pass