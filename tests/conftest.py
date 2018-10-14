from os.path import normpath, join, dirname

import pytest

this_file_dirname = dirname(__file__)
TEST_DATA_DIR = join(this_file_dirname, 'test_data')
HVC_SOURCE_DIR = join(this_file_dirname, normpath('../hvc/'))


@pytest.fixture
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture
def hvc_source_dir():
    return HVC_SOURCE_DIR
