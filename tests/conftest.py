from os.path import normpath, abspath, join

import pytest

this_file = abspath(__file__)
TEST_DATA_DIR =join(this_file, normpath(abspath('test_data/')))
HVC_SOURCE_DIR = join(this_file, normpath(abspath('../hvc/')))

@pytest.fixture
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture
def hvc_source_dir():
    return HVC_SOURCE_DIR
