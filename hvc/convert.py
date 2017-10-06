"""
functions that convert predicted labels to file types used by different programs,
e.g., evsonganaly (GUI for labeling song), Sound Analysis Pro, etc.
"""

def to_notmat(notmat, pred_labels, clf_file):
    """converts predicted labels into a .not.mat file
    that can be read by evsonganaly.m (MATLAB GUI for labeling song)

    Parameters
    ----------
    notmat: str
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

    # chr() to convert back to character from uint32
    pred_labels = [chr(val) for val in pred_labels]
    # convert into one long string, what evsonganaly expects
    pred_labels = ''.join(pred_labels)
    notmat_dict = scio.loadmat(notmat)
    notmat_dict['predicted_labels'] = pred_labels
    notmat_dict['classifier_file'] = clf_file
    print('saving ' + notmat)
    # evsonganaly/Matlab expects all vars as double
    for key, val in notmat_dict.items():
        if key in SHOULD_BE_DOUBLE:
            notmat_dict[key] = val.astype('d')
    scio.savemat(notmat, notmat_dict)


def to_sap:
    pass

def to_koumura:
    pass