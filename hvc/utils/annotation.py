import os

import numpy as np
import scipy.io

from .. import evfuncs


def notmat_to_annotat_dict(notmat):
    notmat_dict = evfuncs.load_notmat(notmat)
    # in .not.mat files saved by evsonganaly,
    # onsets and offsets are in units of ms, have to convert to s
    onsets_s = notmat_dict['onsets'] / 1000
    offsets_s = notmat_dict['offsets'] / 1000

    # convert to Hz using sampling frequency
    audio_filename = notmat.replace('.not.mat','')
    if audio_filename.endswith('.cbin'):
        rec_filename = audio_filename.replace('.cbin','.rec')
    elif audio_filename.endswith('.wav'):
        rec_filename = audio_filename.replace('.wav', '.rec')
    else:
        raise ValueError("Can't find .rec file for {}."
                         .format(notmat))
    rec_dict = evfuncs.readrecf(rec_filename)
    sample_freq = rec_dict['sample_freq']
    # subtract one because of Python's zero indexing (first sample is sample zero)
    onsets_Hz = np.round(onsets_s * sample_freq).astype(int) - 1
    offsets_Hz = np.round(offsets_s * sample_freq).astype(int)

    annotation_dict = {
        'filename': audio_filename,
        'labels': np.asarray(list(notmat_dict['labels'])),
        'onsets_s': onsets_s,
        'offsets_s': offsets_s,
        'onsets_Hz': onsets_Hz,
        'offsets_Hz': offsets_Hz,
    }

    return annotation_dict


def make_notmat(filename,
                labels,
                onsets_Hz,
                offsets_Hz,
                samp_freq,
                threshold,
                min_syl_dur,
                min_silent_dur,
                clf_file,
                alternate_path=None):
    """make a .not.mat file
    that can be read by evsonganaly (MATLAB GUI for labeling song)

    Parameters
    ----------
    filename : str
        name of audio file associated with .not.mat,
        will be used as base of name for .not.mat file
        e.g., if filename is
        'bl26lb16\041912\bl26lb16_190412_0721.20144.cbin'
        then the .not.mat file will be
        'bl26lb16\041912\bl26lb16_190412_0721.20144.cbin.not.mat'
    labels : ndarray
        of type str.
        array of labels given to segments, i.e. syllables, found in filename
    onsets_Hz : ndarray
        onsets of syllables in sample number.
    offsets_Hz : ndarray
        offsets of syllables in sample number.
    samp_freq : int
        sampling frequency of audio file
    threshold : int
        value above which amplitude is considered part of a segment. default is 5000.
    min_syl_dur : float
        minimum duration of a segment. default is 0.02, i.e. 20 ms.
    min_silent_dur : float
        minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.
    clf_file : str
        name of file from which model / classifier was loaded
        so that user of .not.mat file knows which model/classifier was used to predict labels
    alternate_path : str
        Alternate path to which .not.mat files should be saved
        if .not.mat files with same name already exist in directory
        containing audio files.
        Default is None.
        Labelpredict assigns the output_dir from the
        predict.config.yml file as an alternate.

    Returns
    -------
    None
    """
    # chr() to convert back to character from uint32
    if labels.dtype == 'int32':
        labels = [chr(val) for val in labels]
    elif labels.dtype == '<U1':
        labels = labels.tolist()
    # convert into one long string, what evsonganaly expects
    labels = ''.join(labels)
    # notmat files have onsets/offsets in units of ms
    # need to convert back from s
    onsets_s = onsets_Hz / samp_freq
    offsets_s = offsets_Hz / samp_freq
    onsets = (onsets_s * 1e3).astype(float)
    offsets = (offsets_s * 1e3).astype(float)

    # same goes for min_int and min_dur
    # also wrap everything in float so Matlab loads it as double
    # because evsonganaly expects doubles
    notmat_dict = {'Fs': float(samp_freq),
                   'min_dur': float(min_syl_dur * 1e3),
                   'min_int': float(min_silent_dur * 1e3),
                   'offsets': offsets,
                   'onsets': onsets,
                   'sm_win': float(2),  # evsonganaly.m doesn't actually let user change this value
                   'threshold': float(threshold)
                   }
    notmat_dict['labels'] = labels
    notmat_dict['classifier_file'] = clf_file

    notmat_name = filename + '.not.mat'
    if os.path.exists(notmat_name):
        if alternate_path:
            alternate_notmat_name = os.path.join(alternate_path,
                                                 os.path.basename(filename)
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


def load_annotation_csv(annotation_file):
    return annotation_csv


def csv_to_list(annotation_csv):
    annotation_list = []
    annotation_dict = {

    }
    annotation_list.append(annotation_dict)
    for row in annotation_csv[1:]:
        if annotation_csv[0] == annotation_dict['filename']:
            pass
        else:
            annotation_dict = {}
            annotation_list.append(annotation_dict)

    return annotation_list
