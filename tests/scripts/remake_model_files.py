import os
from pathlib import Path
import shutil

import joblib

import hvc

from config import rewrite_config


HERE = Path(__file__).parent
DATA_FOR_TESTS = HERE / ".." / "data_for_tests"
TEST_CONFIGS = DATA_FOR_TESTS.joinpath("config.yml").resolve()
FEATURE_FILES_DST = DATA_FOR_TESTS.joinpath("feature_files").resolve()
MODEL_FILES_DST = DATA_FOR_TESTS.joinpath("model_files").resolve()

config_feature_file_pairs = {
    "knn": ("test_select_knn_ftr_grp.config.yml", "knn.features"),
    "svm": ("test_select_svm.config.yml", "svm.features"),
    "flatwindow": ("test_select_flatwindow.config.yml", "flatwindow.features"),
}


def main():
    for model_name, (
        select_config,
        feature_filename,
    ) in config_feature_file_pairs.items():
        print("running {} to create model files".format(select_config))
        # have to put tmp_output_dir into yaml file
        select_config = TEST_CONFIGS / select_config
        feature_file = sorted(FEATURE_FILES_DST.glob(feature_filename))
        if len(feature_file) != 1:
            raise ValueError(
                "found more than one feature file with search {}:\n{}".format(
                    feature_filename, feature_file
                )
            )
        else:
            # call `resolve` to get full path to model file, so pytest fixtures find it from inside tmp directories
            feature_file = feature_file[0].resolve()

        replace_dict = {
            "feature_file": ("replace with feature_file", str(feature_file)),
            "output_dir": ("replace with tmp_output_dir", str(MODEL_FILES_DST)),
        }

        select_config_rewritten = rewrite_config(
            select_config, str(MODEL_FILES_DST), replace_dict
        )
        select_output_before = [
            select_output_dir
            for select_output_dir in sorted(MODEL_FILES_DST.glob("*select*output*"))
            if select_output_dir.is_dir()
        ]

        hvc.select(select_config_rewritten)

        select_output_after = [
            select_output_dir
            for select_output_dir in sorted(MODEL_FILES_DST.glob("*select*output*"))
            if select_output_dir.is_dir()
        ]

        select_output_dir = [
            after for after in select_output_after if after not in select_output_before
        ]

        if len(select_output_dir) != 1:
            raise ValueError(
                "incorrect number of outputs when looking for select "
                "ouput dirs:\n{}".format(select_output_dir)
            )
        else:
            select_output_dir = select_output_dir[0]

        # arbitrarily grab the last .model and associated .meta file
        model_file = sorted(select_output_dir.glob("*/*.model"))[-1]
        # call `resolve` to get full path to model file, so pytest fixtures find it from inside tmp directories
        model_file_dst = MODEL_FILES_DST.joinpath(model_name + ".model").resolve()
        shutil.move(src=model_file, dst=model_file_dst)
        meta_file = sorted(select_output_dir.glob("*/*.meta"))[-1]
        meta_file_dst = MODEL_FILES_DST.joinpath(model_name + ".meta")
        shutil.move(src=str(meta_file), dst=str(meta_file_dst))

        # need to change 'model_filename' in .meta file
        meta_file = joblib.load(meta_file_dst)
        meta_file["model_filename"] = os.path.abspath(model_file_dst)
        joblib.dump(meta_file, meta_file_dst)

        # clean up -- delete all the other model files, directory, and config
        shutil.rmtree(select_output_dir)
        os.remove(select_config_rewritten)


if __name__ == "__main__":
    main()
