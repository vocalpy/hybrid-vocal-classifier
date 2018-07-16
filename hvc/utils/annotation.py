import os
import csv

import numpy as np
import scipy.io

from .. import evfuncs

# fields that must be present for each syllable that is annotated.
# these field names are used below by annot_list_to_csv and csv_to_annot_list
# but defined at top-level of the module, since these fields determine
# what annotations the library can and cannot interpret.
# The idea is to use the bare minimum of fields required.
SYL_ANNOT_COLUMN_NAMES = ['filename',
                          'onset_Hz',
                          'offset_Hz',
                          'onset_s',
                          'offset_s',
                          'label']
set_SYL_ANNOT_COLUMN_NAMES = set(SYL_ANNOT_COLUMN_NAMES)

# below maps each column in csv to a key in an annot_dict
# used when appending to lists that correspond to each key
SYL_ANNOT_TO_SONG_ANNOT_MAPPING = {'onset_Hz':'onsets_Hz',
                                   'offset_Hz': 'offsets_Hz',
                                   'onset_s': 'onsets_s',
                                   'offset_s': 'offsets_s',
                                   'label': 'labels'}

SONG_ANNOT_TYPE_MAPPING = {'onsets_Hz': int,
                           'offsets_Hz': int,
                           'onsets_s': float,
                           'offsets_s': float,
                           'labels': str}


def notmat_to_annotat_dict(notmat):
    """open .not.mat file and return as annotation dict,
    the data structure that hybrid-vocal-classifier uses
    internally to represent annotation for one audio file

    Parameters
    ----------
    notmat : str
        filename of a .not.mat annotation file,
        created by the evsonganaly GUI for MATLAB

    Returns
    -------
    annotation_dict : dict
        with keys 'filename', 'labels', 'onsets_Hz', 'offsets_Hz', 'onsets_s', 'offsets_s'
    """
    if not notmat.endswith('.not.mat'):
        raise ValueError("notmat filename should end with  '.not.mat',"
                         "but '{}' does not".format(notmat))
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


def annot_list_to_csv(annot_list, filename):
    """writes annotations from files to a comma-separated value (csv) file.

    Parameters
    ----------
    annot_list : list
        list of annot_dicts, where each annot_dict has the following keys:
            filename
            onsets_Hz
            offsets_Hz
            onsets_s
            offsets_s
            labels
    filename : str
        name to write

    Returns
    -------
    None
    """
    with open(filename, 'w', newline='') as csvfile:

        # SYL_ANNOT_COLUMN_NAMES is defined above, at the level of the module,
        # to ensure consistency across all functions in this module
        # that make use of it
        writer = csv.DictWriter(csvfile, fieldnames=SYL_ANNOT_COLUMN_NAMES)

        writer.writeheader()
        for annot_dict in annot_list:
            song_filename = annot_dict['filename']
            annot_dict_zipped = zip(annot_dict['onsets_Hz'],
                                    annot_dict['offsets_Hz'],
                                    annot_dict['onsets_s'],
                                    annot_dict['offsets_s'],
                                    annot_dict['labels'],
                                    )
            for onset_Hz, offset_Hz, onset_s, offset_s, label in annot_dict_zipped:
                syl_annot_dict = {'filename': song_filename,
                                  'onset_Hz': onset_Hz,
                                  'offset_Hz': offset_Hz,
                                  'onset_s': onset_s,
                                  'offset_s': offset_s,
                                  'label': label} 
                writer.writerow(syl_annot_dict)


def notmat_list_to_csv(notmat_list, csv_fname):
    """takes a list of .not.mat filenames and saves the
    annotation from all files in one comma-separated values (csv)
    file, where each row represents one syllable from one of the
    .not.mat files.0

    Parameters
    ----------
    notmat_list : list
        list of str, where eachs tr is a .not.mat file

    csv_fname : str
        name for csv file that is created

    Returns
    -------
    None
    """
    if not all([notmat.endswith('.not.mat')
                for notmat in notmat_list]
               ):
        raise ValueError("all filenames in .not.mat must end with '.not.mat'")

    annot_list = []
    for notmat in notmat_list:
        annot_list.append(notmat_to_annotat_dict(notmat))
    annot_list_to_csv(annot_list, csv_fname)


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


def csv_to_annot_list(csv_fname):
    """loads a comma-separated values (csv) file containing
    annotations for song files and returns an annot_list

    Parameters
    ----------
    csv_fname : str
        filename for comma-separated values file

    Returns
    -------
    annot_list : list
        list of dicts
    """
    annot_list = []

    with open(csv_fname, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)

        header = next(reader)
        set_header = set(header)
        if set_header != set_SYL_ANNOT_COLUMN_NAMES:
            not_in_FIELDNAMES = set_header.difference(set_SYL_ANNOT_COLUMN_NAMES)
            if not_in_FIELDNAMES:
                raise ValueError('The following column names in {} are not recognized: {}'
                                 .format(csv_fname, not_in_FIELDNAMES))
            not_in_header = set_FIELDNAMES.difference(set_header)
            if not_in_header:
                raise ValueError(
                    'The following column names in {} are required but missing: {}'
                    .format(csv_fname, not_in_header))

        column_name_index_mapping = {column_name: header.index(column_name)
                                     for column_name in SYL_ANNOT_COLUMN_NAMES}

        row = next(reader)
        curr_filename = row[column_name_index_mapping['filename']]
        annot_dict = {'filename': curr_filename,
                      'onsets_Hz': [],
                      'offsets_Hz': [],
                      'onsets_s': [],
                      'offsets_s': [],
                      'labels': []}
        for col_name in (set_SYL_ANNOT_COLUMN_NAMES - {'filename'}):
            annot_dict[SYL_ANNOT_TO_SONG_ANNOT_MAPPING[col_name]].append(
                row[column_name_index_mapping[col_name]])

        for row in reader:
            row_filename = row[column_name_index_mapping['filename']]
            if row_filename == curr_filename:
                for col_name in (set_SYL_ANNOT_COLUMN_NAMES - {'filename'}):
                    annot_dict[SYL_ANNOT_TO_SONG_ANNOT_MAPPING[col_name]].append(
                        row[column_name_index_mapping[col_name]])
            else:
                for key, type_to_convert in SONG_ANNOT_TYPE_MAPPING.items():
                    list_from_key = annot_dict[key]
                    if type_to_convert == int:
                        list_from_key = [int(el) for el in list_from_key]
                    elif type_to_convert == float:
                        list_from_key = [float(el) for el in list_from_key]
                    elif type_to_convert == str:
                        pass
                    else:
                        raise TypeError('Unexpected type {} specified in '
                                        'hvc.utils.annotation'
                                        .format(type_to_convert))
                    annot_dict[key] = list_from_key
                # convert all lists to ndarray
                for col_name in (set_SYL_ANNOT_COLUMN_NAMES - {'filename'}):
                    annot_dict[SYL_ANNOT_TO_SONG_ANNOT_MAPPING[col_name]] = \
                        np.asarray(annot_dict[SYL_ANNOT_TO_SONG_ANNOT_MAPPING[col_name]])
                # now append annot_dict to annot_list
                annot_list.append(annot_dict)
                # and start a new annot_dict
                curr_filename = row_filename
                annot_dict = {'filename': curr_filename,
                              'onsets_Hz': [],
                              'offsets_Hz': [],
                              'onsets_s': [],
                              'offsets_s': [],
                              'labels': []}
                for col_name in (set_SYL_ANNOT_COLUMN_NAMES - {'filename'}):
                    annot_dict[SYL_ANNOT_TO_SONG_ANNOT_MAPPING[col_name]].append(
                        row[column_name_index_mapping[col_name]])
        # line below appends annot_dict corresponding to last file
        # since there won't be another file after it to trigger the 'else' logic above
        annot_list.append(annot_dict)

    return annot_list
