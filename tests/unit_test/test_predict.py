"""module to test high-level extract function in hvc.predict"""
import os
from glob import glob

import joblib

import hvc
from config import rewrite_config


class TestPredict:
    def _generic_predict_asserts(self, predict):
        # assertions shared by all unit tests for hvc.predict
        assert type(predict) == dict
        for key in [
            "labels",
            "pred_labels",
            "songfile_IDs",
            "onsets_Hz",
            "offsets_Hz",
        ]:
            assert key in predict
        assert ("features" in predict) or ("neuralnet_inputs") in predict

    def test_predict_knn_data_dirs_notmat(self, tmp_output_dir, test_data_dir):
        # tests predict with knn model, using data dirs, and
        # converting output to notmat files
        data_dirs = ["cbins/gy6or6/032312", "cbins/gy6or6/032412"]
        data_dirs = [
            os.path.join(test_data_dir, os.path.normpath(data_dir))
            for data_dir in data_dirs
        ]
        file_format = "cbin"
        model_meta_file = os.path.join(test_data_dir, "model_files", "knn.meta")
        output_dir = tmp_output_dir
        # explicitly set segment to None because we want to test
        # that default behavior works that happens when
        # we supply argument for data_dirs parameter, **and**
        # segment is set to None (as it should be by default)
        segment = None
        predict_proba = False
        convert_to = "notmat"  # to check that this works
        return_predictions = True
        predict = hvc.predict(
            data_dirs=data_dirs,
            file_format=file_format,
            model_meta_file=model_meta_file,
            segment=segment,
            output_dir=str(tmp_output_dir),
            predict_proba=predict_proba,
            convert_to=convert_to,
            return_predictions=return_predictions,
        )
        assert type(predict) == dict
        for key in [
            "labels",
            "pred_labels",
            "songfile_IDs",
            "onsets_Hz",
            "offsets_Hz",
            "features",
        ]:
            assert key in predict
        # check there are cbin files in output_dir!

    def test_predict_svm_data_dirs(self, tmp_output_dir, test_data_dir):
        # tests predict with svm model, using data dirs
        data_dirs = ["cbins/gy6or6/032312", "cbins/gy6or6/032412"]
        data_dirs = [
            os.path.join(test_data_dir, os.path.normpath(data_dir))
            for data_dir in data_dirs
        ]
        file_format = "cbin"
        model_meta_file = os.path.join(test_data_dir, "model_files", "svm.meta")
        output_dir = tmp_output_dir
        # explicitly set segment to None because we want to test
        # that default behavior works that happens when
        # we supply argument for data_dirs parameter, **and**
        # segment is set to None (as it should be by default)
        segment = None
        predict_proba = False
        return_predictions = True
        predict = hvc.predict(
            data_dirs=data_dirs,
            file_format=file_format,
            model_meta_file=model_meta_file,
            segment=segment,
            output_dir=str(tmp_output_dir),
            predict_proba=predict_proba,
            return_predictions=return_predictions,
        )
        self._generic_predict_asserts(predict)

    def test_predict_flatwindow_data_dirs(self, tmp_output_dir, test_data_dir):
        # tests predict with svm model, using data dirs
        data_dirs = ["cbins/gy6or6/032312", "cbins/gy6or6/032412"]
        data_dirs = [
            os.path.join(test_data_dir, os.path.normpath(data_dir))
            for data_dir in data_dirs
        ]
        file_format = "cbin"
        model_meta_file = os.path.join(test_data_dir, "model_files", "flatwindow.meta")
        output_dir = tmp_output_dir
        # explicitly set segment to None because we want to test
        # that default behavior works that happens when
        # we supply argument for data_dirs parameter, **and**
        # segment is set to None (as it should be by default)
        segment = None
        predict_proba = False
        return_predictions = True
        predict = hvc.predict(
            data_dirs=data_dirs,
            file_format=file_format,
            model_meta_file=model_meta_file,
            segment=segment,
            output_dir=str(tmp_output_dir),
            predict_proba=predict_proba,
            return_predictions=return_predictions,
        )
        self._generic_predict_asserts(predict)

    def _yaml_config_run(
        self, predict_yaml_config_path, tmp_output_dir, model_meta_file
    ):
        predict_config_rewritten = rewrite_config(
            predict_yaml_config_path,
            tmp_output_dir,
            replace_dict={
                "model_meta_file": ("replace with model_meta_file", model_meta_file),
                "output_dir": ("replace with tmp_output_dir", str(tmp_output_dir)),
            },
        )
        predict_outputs_before = glob(
            os.path.join(str(tmp_output_dir), "predict_output*", "features_created*")
        )
        hvc.predict(predict_config_rewritten)
        # helper function with assertions shared by all
        # tests for hvc.select run with config.yml files
        predict_outputs_after = glob(
            os.path.join(str(tmp_output_dir), "predict_output*", "features_created*")
        )
        predict_output = [
            after
            for after in predict_outputs_after
            if after not in predict_outputs_before
        ]
        # should only be one summary output file
        if len(predict_output) != 1:
            raise ValueError(
                "found wrong number of predict outputs after "
                "running .yaml config {}.\n"
                "This was the output found: {}".format(
                    predict_config_rewritten, predict_output
                )
            )
        else:
            predict_output = predict_output[0]
        predict = joblib.load(predict_output)
        self._generic_predict_asserts(predict)

    def test_predict_knn_yaml(self, tmp_output_dir, configs_path, test_data_dir):
        # test select with knn classifier and features specified by feature list indices
        knn_predict_config = os.path.join(configs_path, "test_predict_knn.config.yml")
        model_meta_file = os.path.join(test_data_dir, "model_files", "knn.meta")
        self._yaml_config_run(knn_predict_config, tmp_output_dir, model_meta_file)

    def test_predict_svm_yaml(self, tmp_output_dir, configs_path, test_data_dir):
        # test select with knn classifier and features specified by feature list indices
        svm_predict_config = os.path.join(configs_path, "test_predict_svm.config.yml")
        model_meta_file = os.path.join(test_data_dir, "model_files", "svm.meta")
        self._yaml_config_run(svm_predict_config, tmp_output_dir, model_meta_file)

    def test_predict_flatwindow_yaml(self, tmp_output_dir, configs_path, test_data_dir):
        # test select with knn classifier and features specified by feature list indices
        flatwindow_predict_config = os.path.join(
            configs_path, "test_predict_flatwindow.config.yml"
        )
        model_meta_file = os.path.join(test_data_dir, "model_files", "flatwindow.meta")
        self._yaml_config_run(
            flatwindow_predict_config, tmp_output_dir, model_meta_file
        )
