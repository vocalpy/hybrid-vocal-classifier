import os
from glob import glob
import csv

import numpy as np
import pytest

from hvc.utils import annotation

ANNOT_DICT_FIELDNAMES = {'filename': str,
                         'onsets_Hz': np.ndarray,
                         'offsets_Hz': np.ndarray,
                         'onsets_s': np.ndarray,
                         'offsets_s': np.ndarray,
                         'labels': np.ndarray}

SYL_DICT_FIELDNAMES = ['filename', 
                       'onset_Hz', 
                       'offset_Hz', 
                       'onset_s', 
                       'offset_s', 
                       'label']

@pytest.fixture(scope='session')
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


def test_notmat_to_annot_dict():
    notmat = os.path.join(os.path.dirname(__file__),
                          os.path.normpath(
                              'test_data/cbins/gy6or6/032412/'
                              'gy6or6_baseline_240312_0811.1165.cbin.not.mat'))
    annot_dict = annotation.notmat_to_annot_dict(notmat)
    for fieldname, fieldtype in ANNOT_DICT_FIELDNAMES.items():
        assert fieldname in annot_dict
        assert type(annot_dict[fieldname]) == fieldtype


def test_annot_list_to_csv(tmp_output_dir):
    """compares csv created by annot_list_to_csv
    with correctly generated csv saved in hvc/tests/test_data
    """
    cbin_dir = os.path.join(os.path.dirname(__file__),
                            os.path.normpath(
                                'test_data/cbins/gy6or6/032312/'))
    notmat_list = glob(os.path.join(cbin_dir, '*.not.mat'))
    annot_list = []
    for notmat in notmat_list:
        annot_list.append(annotation.notmat_to_annot_dict(notmat))
    csv_filename = os.path.join(str(tmp_output_dir),
                                'test.csv')
    # below, set basename to True so we can easily run tests on any system without
    # worrying about where audio files are relative to root of directory tree
    annotation.annot_list_to_csv(annot_list,
                                 csv_filename,
                                 basename=True)
    assert os.path.isfile(csv_filename)
    test_rows = []
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            test_rows.append(row)
    
    csv_to_compare_with = os.path.join(os.path.dirname(__file__),
                                       os.path.normpath(
                                           'test_data/csv/gy6or6_032312.csv'))
    compare_rows = []
    with open(csv_to_compare_with, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            compare_rows.append(row)
    for test_row, compare_row in zip(test_rows, compare_rows):
        assert test_row == compare_row


def test_notmat_list_to_csv(tmp_output_dir):
    # since notmat_list_to_csv is basically a wrapper around
    # notmat_to_annot_dict and annot_list_to_csv,
    # and those are tested above,
    # here just need to make sure this function doesn't fail
    cbin_dir = os.path.join(os.path.dirname(__file__),
                            os.path.normpath(
                                'test_data/cbins/gy6or6/032312/'))
    notmat_list = glob(os.path.join(cbin_dir, '*.not.mat'))
    csv_filename = os.path.join(str(tmp_output_dir),
                                'test.csv')
    annotation.notmat_list_to_csv(notmat_list, csv_filename)
    # make sure file was created
    assert os.path.isfile(csv_filename)

    # to be extra sure, make sure all filenames from 
    # .not.mat list are in csv 
    filenames_from_csv = []
    with open(csv_filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=SYL_DICT_FIELDNAMES)
        header = next(reader)
        for row in reader:
            filenames_from_csv.append(row['filename'])
    for fname_from_csv in filenames_from_csv:
        assert(fname_from_csv + '.not.mat' in notmat_list)


def test_make_notmat():
    pass


def test_load_annotation_csv():
    pass


def test_csv_to_annot_list():
    csv_fname = os.path.join(os.path.dirname(__file__),
                             os.path.normpath(
                                 'test_data/csv/gy6or6_032312.csv'))
    annot_list = annotation.csv_to_annot_list(csv_fname)
    cbin_dir = os.path.join(os.path.dirname(__file__),
                            os.path.normpath(
                                'test_data/cbins/gy6or6/032312/'))
    notmat_list = glob(os.path.join(cbin_dir, '*.not.mat'))
    annot_list_to_compare_with = []
    for notmat in notmat_list:
        annot_list_to_compare_with.append(annotation.notmat_to_annot_dict(notmat))
    for from_csv, from_notmat in zip(annot_list, annot_list_to_compare_with):
        assert from_csv == from_notmat
