import sys
import os
from os.path import normpath, join, dirname

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

this_file_dirname = dirname(__file__)
TEST_DATA_DIR = join(this_file_dirname, 'test_data')
HVC_SOURCE_DIR = join(this_file_dirname, normpath('../hvc/'))


@pytest.fixture
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture
def hvc_source_dir():
    return HVC_SOURCE_DIR


@pytest.fixture(scope='session')
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


@pytest.fixture
def configs_path(test_data_dir):
    return join(test_data_dir, 'config.yml')
