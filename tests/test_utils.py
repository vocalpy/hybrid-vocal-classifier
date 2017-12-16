"""
test utils module
"""

import os
from glob import glob

import pytest

import hvc.utils

this_file_with_path = __file__
this_file_just_path = os.path.split(this_file_with_path)[0]

@pytest.fixture(scope='session')
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


def test_fetch(tmp_output_dir):
    hvc.utils.fetch(dataset_str='sober.repo1.gy6or6.032612',  # one of the smaller .gz, ~31 MB
                    destination_path=str(tmp_output_dir))
    compare_dir = os.path.join(this_file_just_path,
                               os.path.normpath('test_data/cbins/gy6or6/032612'))
    os.chdir(compare_dir)
    test_data_032612 = glob('gy6or6*')
    test_data_fetched = os.listdir(os.path.join(str(tmp_output_dir),
                                                '032612'))
    for file in test_data_032612:
        assert file in test_data_fetched
