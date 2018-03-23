import numpy as np

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
    # subtract one because of Python's zero indexing (first sanmple is sample zero)
    onsets_Hz = np.round(onsets_s * sample_freq).astype(int) - 1
    offsets_Hz = np.round(offsets_s * sample_freq).astype(int)

    annotation_dict = {
        'filename': audio_filename,
        'labels': notmat_dict['labels'],
        'onsets_s': onsets_s,
        'offsets_s': offsets_s,
        'onsets_Hz': onsets_Hz,
        'offsets_Hz': offsets_Hz,
    }
    
    return annotation_dict


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
