"""
tests running a 'typical' workflow
all thrown into one file
because the tests have to run in a certain order
and this seemed like the easiest, least fragile way to do that
"""

import os
from glob import glob

import joblib

import hvc
from hvc.select import determine_model_output_folder_name
from config import rewrite_config


def check_extract_output(output_dir):
    """"""

    # ftr_files = glob(os.path.join(output_dir, 'features_from*'))
    # ftr_dicts = []
    # for ftr_file in ftr_files:
    #     ftr_dicts.append(joblib.load(ftr_file))
    #
    # # if features were extracted (not spectrograms)
    # if any(['features' in ftr_dict for ftr_dict in ftr_dicts]):
    #     # then all ftr_dicts should have `features` key
    #     assert all(['features' in ftr_dict for ftr_dict in ftr_dicts])
    #     # and the number of rows in features should equal number of labels
    #     for ftr_dict in ftr_dicts:
    #         labels = ftr_dict['labels']
    #         features = ftr_dict['features']
    #         assert features.shape[0] == len(labels)
    #
    #     # make sure number of features i.e. columns is constant across feature matrices
    #     ftr_cols = [ftr_dict['features'].shape[1] for ftr_dict in ftr_dicts]
    #     assert np.unique(ftr_cols).shape[-1] == 1
    #
    # # if features are spectrograms for neural net
    # if any(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts]):
    #     # then all feature dicts should have spectrograms
    #     assert all(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts])
    #     neuralnet_keys = [ftr_dict['neuralnets_input_dict'].keys()
    #                       for ftr_dict in ftr_dicts]
    #     # make sure keys are all the same for neuralnets_input_dict from every ftr_dict
    #     for ind, keyset in enumerate(neuralnet_keys):
    #         other_keysets = neuralnet_keys[:ind] + neuralnet_keys[(ind+1):]
    #         assert keyset.difference(other_keysets) == set()
    #     # if they are all the same, then save that set of keys
    #     # to compare with summary feature dict below
    #     neuralnet_keys = neuralnet_keys[0]
    #
    #     for ftr_dict in ftr_dicts:
    #         labels = ftr_dict['labels']
    #         for key, val in ftr_dict['neuralnet_inputs_dict']:
    #             assert val.shape[0] == len(labels)
    # if all(['neuralnets_input_dict' in ftr_dict for ftr_dict in ftr_dicts]):
    #     assert summary_dict['neuralnet_inputs_dict'].keys() == neuralnet_keys
    #     for key, val in summary_dict['neuralnet_inputs_dict']:
    #         sum_ftr_rows = summary_dict['neuralnets_input_dict'][key].shape[0]
    #         total_ftr_dict_rows = sum(
    #             [ftr_dict['neuralnet_inputs_dict'][key].shape[0]
    #              for ftr_dict in ftr_dicts])
    #         assert sum_ftr_rows == total_ftr_dict_rows

    return True  # because called with assert


def check_select_output(config_path, output_dir):
    """"""

    select_output = glob(os.path.join(str(output_dir), "summary_model_select_file*"))
    # should only be one summary output file
    assert len(select_output) == 1

    # now check for every model in config
    # if there is corresponding folder with model files etc
    select_config = hvc.parse_config(config_path, "select")
    select_model_dirs = next(os.walk(output_dir))[1]  # [1] to return just dir names
    select_model_folder_names = [
        determine_model_output_folder_name(model_dict)
        for model_dict in select_config["models"]
    ]
    for folder_name in select_model_folder_names:
        assert folder_name in select_model_dirs

    return True


def run_main_workflow(tmp_output_dir, script_tuple_dict, configs_path):
    """tests main workflow for hybrid-vocal-classifier
    by iterating through test_main_workflow_dict,
    running the scripts named in each tuple in the dict
    """

    extract_config_filename = os.path.join(configs_path, script_tuple_dict["extract"])
    replace_dict = {"output_dir": ("replace with tmp_output_dir", str(tmp_output_dir))}
    # have to put tmp_output_dir into yaml file
    extract_config_rewritten = rewrite_config(
        extract_config_filename, tmp_output_dir, replace_dict
    )
    hvc.extract(extract_config_rewritten)
    extract_outputs = list(
        filter(os.path.isdir, glob(os.path.join(str(tmp_output_dir), "*extract*")))
    )
    extract_outputs.sort(key=os.path.getmtime)
    extract_output_dir = extract_outputs[-1]  # [-1] is newest dir, after sort
    assert check_extract_output(extract_output_dir)

    feature_file = glob(os.path.join(extract_output_dir, "features_created*"))
    feature_file = feature_file[0]  # because glob returns list

    os.remove(extract_config_rewritten)

    select_and_predict_tuples = script_tuple_dict["select and predict"]
    for select_and_predict_tuple in select_and_predict_tuples:
        (select_config_filename, predict_config_filename) = select_and_predict_tuple
        select_config_filename = os.path.join(configs_path, select_config_filename)

        select_config_rewritten = rewrite_config(
            select_config_filename,
            tmp_output_dir,
            replace_dict={
                "feature_file": ("replace with feature_file", feature_file),
                "output_dir": ("replace with tmp_output_dir", str(tmp_output_dir)),
            },
        )
        hvc.select(select_config_rewritten)
        select_outputs = list(
            filter(os.path.isdir, glob(os.path.join(str(tmp_output_dir), "*select*")))
        )
        select_outputs.sort(key=os.path.getmtime)
        select_output_dir = select_outputs[-1]  # [-1] is newest dir, after sort
        assert check_select_output(select_config_rewritten, select_output_dir)
        os.remove(select_config_rewritten)

        select_outputs.sort(key=os.path.getmtime)
        select_output_dir = select_outputs[-1]  # [-1] is newest dir, after sort
        model_meta_files = glob(os.path.join(select_output_dir, "*", "*meta*"))
        replace_dict = {
            "model_meta_file": ("replace with model_meta_file", model_meta_files[-1]),
            "output_dir": ("replace with tmp_output_dir", str(tmp_output_dir)),
        }
        predict_config_filename_with_path = os.path.join(
            configs_path, predict_config_filename
        )

        predict_config_rewritten = rewrite_config(
            predict_config_filename_with_path, tmp_output_dir, replace_dict
        )
        hvc.predict(predict_config_rewritten)
        os.remove(predict_config_rewritten)
        predict_outputs = list(
            filter(os.path.isdir, glob(os.path.join(str(tmp_output_dir), "*predict*")))
        )
        predict_outputs.sort(key=os.path.getmtime)
        predict_output_dir = predict_outputs[-1]  # [-1] is newest dir, after sort
        feature_files = glob(os.path.join(predict_output_dir, "feature*"))
        for ftr_filename in feature_files:
            ftr_file = joblib.load(ftr_filename)
            assert "pred_labels" in ftr_file
            if "predict_proba_True" in extract_config_filename:
                assert "pred_probs" in ftr_file
                assert (
                    ftr_file["pred_labels"].shape[0] == ftr_file["pred_probs"].shape[0]
                )


def test_knn(tmp_output_dir, configs_path):
    knn_script_tuple_dict = {
        "extract": "test_extract_knn.config.yml",
        "select and predict": (
            ("test_select_knn_ftr_list_inds.config.yml", "test_predict_knn.config.yml"),
            ("test_select_knn_ftr_grp.config.yml", "test_predict_knn.config.yml"),
            (
                "test_select_knn_ftr_grp_predict_proba_True.config.yml",
                "test_predict_knn_predict_proba_True.config.yml",
            ),
        ),
    }
    run_main_workflow(tmp_output_dir, knn_script_tuple_dict, configs_path)


def test_svm(tmp_output_dir, configs_path):
    svm_script_tuple_dict = {
        "extract": "test_extract_svm.config.yml",
        "select and predict": (
            ("test_select_svm.config.yml", "test_predict_svm.config.yml"),
            (
                "test_select_svm_predict_proba_True.config.yml",
                "test_predict_svm_predict_proba_True.config.yml",
            ),
        ),
    }
    run_main_workflow(tmp_output_dir, svm_script_tuple_dict, configs_path)


def test_flatwindow(tmp_output_dir, configs_path):
    flatwindow_script_tuple_dict = {
        "extract": "test_extract_flatwindow.config.yml",
        "select and predict": (
            ("test_select_flatwindow.config.yml", "test_predict_flatwindow.config.yml"),
        ),
    }
    run_main_workflow(tmp_output_dir, flatwindow_script_tuple_dict, configs_path)
