import os
from pathlib import Path
import shutil

import hvc

HERE = Path(__file__).parent
DATA_FOR_TESTS = HERE / ".." / "data_for_tests"
TEST_CONFIGS = DATA_FOR_TESTS / "config.yml"
FEATURE_FILES_DST = DATA_FOR_TESTS / "feature_files"

from config import rewrite_config


def main():
    feature_files_to_create = [
        "knn",
        "svm",
        "multiple_feature_groups",
        "flatwindow",
    ]
    for feature_to_create in feature_files_to_create:
        extract_config = TEST_CONFIGS.joinpath(
            f"test_extract_{feature_to_create}.config.yml"
        )
        print("running {} to create feature file".format(extract_config))
        replace_dict = {
            "output_dir": ("replace with tmp_output_dir", str(FEATURE_FILES_DST))
        }
        # have to put tmp_output_dir into yaml file
        extract_config_rewritten = rewrite_config(
            extract_config, FEATURE_FILES_DST, replace_dict
        )
        hvc.extract(extract_config_rewritten)
        extract_output_dir = sorted(FEATURE_FILES_DST.glob("*extract*output*"))
        if len(extract_output_dir) != 1:
            raise ValueError(
                "incorrect number of outputs when looking for extract "
                "ouput dirs:\n{}".format(extract_output_dir)
            )
        else:
            extract_output_dir = extract_output_dir[0]

        features_created = sorted(extract_output_dir.glob("features_created*"))
        if len(features_created) != 1:
            raise ValueError(
                "incorrect number of outputs when looking for extract "
                "feature files:\n{}".format(features_created)
            )
        else:
            # call `resolve` to get full path to model file, so pytest fixtures find it from inside tmp directories
            features_created = features_created[0].resolve()
        movename = feature_to_create + "." + "features"
        shutil.move(src=features_created, dst=FEATURE_FILES_DST.joinpath(movename))
        os.rmdir(extract_output_dir)
        os.remove(extract_config_rewritten)


if __name__ == "__main__":
    main()
