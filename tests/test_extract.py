"""this file tests **just** the high-level extract function in hvc.extract.
More specifically it tests running the function **without** config.yml scripts,
instead using pure Python"""

import os
from glob import glob
import copy

import pytest
from sklearn.externals import joblib

import hvc


configs = os.path.join(
    os.path.dirname(__file__),
    os.path.normpath('test_data/config.yml/'))


@pytest.fixture(scope='session')
def tmp_output_dir(tmpdir_factory):
    fn = tmpdir_factory.mktemp('tmp_output_dir')
    return fn


class TestExtract:
    def test_data_dirs_cbins(self, tmp_output_dir):
        """test that calling extract doesn't fail when we
        pass a data_dirs list that contain cbin audio files"""
        data_dirs = [
            'test_data/cbins/gy6or6/032312',
            'test_data/cbins/gy6or6/032412']
        file_format = 'cbin'
        labels_to_use = 'iabcdefghjk'
        feature_group = 'knn'
        return_features = True
        ftrs = hvc.extract(data_dirs=data_dirs,
                           file_format=file_format,
                           labels_to_use=labels_to_use,
                           feature_group=feature_group,
                           output_dir=str(tmp_output_dir),
                           return_features=return_features)
        assert type(ftrs) == dict
        assert tuple(ftrs.keys()) == ('labels', 'features')
