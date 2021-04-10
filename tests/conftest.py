from pathlib import Path
import sys

import pytest

TESTS_ROOT = Path(__file__).parent
sys.path.append(str(TESTS_ROOT / "scripts"))

TEST_DATA_DIR = TESTS_ROOT / "data_for_tests"
HVC_SOURCE_DIR = TESTS_ROOT / ".." / "src" / "hvc"


@pytest.fixture
def test_data_dir():
    return TEST_DATA_DIR


@pytest.fixture
def hvc_source_dir():
    return HVC_SOURCE_DIR


@pytest.fixture(scope="session")
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp("tmp_output_dir")
    return fn


@pytest.fixture
def configs_path(test_data_dir):
    return str(test_data_dir / "config.yml")
