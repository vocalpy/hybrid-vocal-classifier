"""module to test high-level extract function in hvc.extract"""
import os
from glob import glob

import hvc
from hvc.utils import annotation
from config import rewrite_config


class TestExtract:
    def test_data_dirs_cbins(self, test_data_dir, tmp_output_dir):
        # test that calling extract doesn't fail when we
        # pass a data_dirs list that contain cbin audio files
        data_dirs = ["cbins/gy6or6/032312", "cbins/gy6or6/032412"]
        data_dirs = [
            os.path.join(test_data_dir, os.path.normpath(data_dir))
            for data_dir in data_dirs
        ]

        file_format = "cbin"
        labels_to_use = "iabcdefghjk"
        feature_group = "knn"
        return_features = True
        ftrs = hvc.extract(
            data_dirs=data_dirs,
            file_format=file_format,
            labels_to_use=labels_to_use,
            feature_group=feature_group,
            output_dir=str(tmp_output_dir),
            return_features=return_features,
        )
        assert type(ftrs) == dict
        assert sorted(ftrs.keys()) == ["features", "labels"]

    def test_annotation_file_cbins(self, test_data_dir, tmp_output_dir):
        # test that calling extract doesn't fail when we
        # pass a data_dirs list that contain cbin audio files
        cbin_dirs = ["cbins/gy6or6/032312", "cbins/gy6or6/032412"]
        cbin_dirs = [
            os.path.join(test_data_dir, os.path.normpath(cbin_dir))
            for cbin_dir in cbin_dirs
        ]

        notmat_list = []
        for cbin_dir in cbin_dirs:
            notmat_list.extend(glob(os.path.join(cbin_dir, "*.not.mat")))
        # below, sorted() so it's the same order on different platforms
        notmat_list = sorted(notmat_list)
        csv_filename = os.path.join(str(tmp_output_dir), "test.csv")
        annotation.notmat_list_to_csv(notmat_list, csv_filename)

        file_format = "cbin"
        labels_to_use = "iabcdefghjk"
        feature_group = "knn"
        return_features = True
        ftrs = hvc.extract(
            file_format=file_format,
            annotation_file=csv_filename,
            labels_to_use=labels_to_use,
            feature_group=feature_group,
            output_dir=str(tmp_output_dir),
            return_features=return_features,
        )
        assert type(ftrs) == dict
        assert sorted(ftrs.keys()) == ["features", "labels"]

    def _yaml_config_asserts(self, extract_yaml_config_file, tmp_output_dir):
        replace_dict = {
            "output_dir": ("replace with tmp_output_dir", str(tmp_output_dir))
        }
        # have to put tmp_output_dir into yaml file
        extract_config_rewritten = rewrite_config(
            extract_yaml_config_file, tmp_output_dir, replace_dict
        )

        # helper function that is called by tests below
        hvc.extract(extract_config_rewritten)
        extract_config = hvc.parse_config(extract_config_rewritten, "extract")

        for todo in extract_config["todo_list"]:
            os.chdir(todo["output_dir"])
            extract_outputs = list(filter(os.path.isdir, glob("*extract_output*")))
            extract_outputs.sort(key=os.path.getmtime)

            os.chdir(extract_outputs[-1])  # most recent
            ftr_files = glob("features_from*")
            ftr_dicts = []
            for ftr_file in ftr_files:
                ftr_dicts.append(joblib.load(ftr_file))

            if any(["features" in ftr_dict for ftr_dict in ftr_dicts]):
                assert all(["features" in ftr_dict for ftr_dict in ftr_dicts])
                for ftr_dict in ftr_dicts:
                    labels = ftr_dict["labels"]
                    if "features" in ftr_dict:
                        features = ftr_dict["features"]
                        assert features.shape[0] == len(labels)

                # make sure number of features i.e. columns is constant across feature matrices
                ftr_cols = [ftr_dict["features"].shape[1] for ftr_dict in ftr_dicts]
                assert np.unique(ftr_cols).shape[-1] == 1

            if any(["neuralnets_input_dict" in ftr_dict for ftr_dict in ftr_dicts]):
                assert all(
                    ["neuralnets_input_dict" in ftr_dict for ftr_dict in ftr_dicts]
                )

    def test_extract_knn_yaml(self, tmp_output_dir, configs_path):
        extract_yaml_config_file = os.path.join(
            configs_path, "test_extract_knn.config.yml"
        )
        self._yaml_config_asserts(
            extract_yaml_config_file=extract_yaml_config_file,
            tmp_output_dir=tmp_output_dir,
        )

    def test_extract_svm_yaml(self, tmp_output_dir, configs_path):
        extract_yaml_config_file = os.path.join(
            configs_path, "test_extract_svm.config.yml"
        )
        self._yaml_config_asserts(
            extract_yaml_config_file=extract_yaml_config_file,
            tmp_output_dir=tmp_output_dir,
        )

    def test_extract_flatwindow_yaml(self, tmp_output_dir, configs_path):
        extract_yaml_config_file = os.path.join(
            configs_path, "test_extract_flatwindow.config.yml"
        )
        self._yaml_config_asserts(
            extract_yaml_config_file=extract_yaml_config_file,
            tmp_output_dir=tmp_output_dir,
        )

    def test_extract_multiple_feature_groups_yaml(self, tmp_output_dir, configs_path):
        extract_yaml_config_file = os.path.join(
            configs_path, "test_extract_multiple_feature_groups.config.yml"
        )
        self._yaml_config_asserts(
            extract_yaml_config_file=extract_yaml_config_file,
            tmp_output_dir=tmp_output_dir,
        )
