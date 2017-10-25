"""
test utils module
"""

import os
from glob import glob

import pytest

import hvc.utils


@pytest.fixture(scope='session')
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


def test_fetch(tmp_output_dir):
    hvc.utils.fetch(dataset_name='sober.repo1.gy6or6.032612',  # one of the smaller .gz, ~31 MB
                    destination_path=str(tmp_output_dir))
    os.chdir('.\\test_data\\cbins\\gy6or6\\032612')
    test_data_032612 = glob('gy6or6*')
    test_data_fetched = os.listdir(os.path.join(str(tmp_output_dir),
                                                '032612'))
    for file in test_data_032612:
        assert file in test_data_fetched
