"""
predict labels for birdsong syllables,
using already-trained models specified in config file
"""

import os
import sys
import glob

# from dependencies
import yaml
import numpy as np
from sklearn.externals import joblib

# from hvc
import hvc.featureextract
from .parseconfig import parse_config
from .utils import timestamp
import hvc.convert

path = os.path.abspath(__file__)  # get the path of this file
dir_path = os.path.dirname(path)  # but then just take the dir

with open(os.path.join(dir_path, 'parse', 'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml)
valid_models = validate_dict['valid_models']


def predict(config_file):
    """main function that does prediction
    Does not return anything, just runs through directories specified in config_file
    and classifies syllables using model.

    Parameters
    ----------
    config_file : string
        filename of YAML file that configures label prediction   
    """

    predict_config = parse_config(config_file, 'predict')
    print('parsed predict config')

    home_dir = os.getcwd()

    for todo in predict_config['todo_list']:
        # get absolute path before changing directories
        # in case user specified output as a relative dir
        output_dir = os.path.abspath(todo['output_dir'])
        output_dir = os.path.join(output_dir, 'predict_output_' + timestamp())
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        model_meta_file = joblib.load(todo['model_meta_file'])

        extract_params = {
            'bird_ID': todo['bird_ID'],
            'feature_list': model_meta_file['feature_list'],
            'output_dir': output_dir,
            'home_dir': home_dir,
            'data_dirs': todo['data_dirs'],
            'labelset': 'all',
            'file_format': todo['file_format']
        }

        feature_file_for_model = model_meta_file['feature_file']
        feature_file = joblib.load(feature_file_for_model)
        extract_params['segment_params'] = feature_file['segment_params']
        extract_params['spect_params'] = feature_file['spect_params']

        hvc.featureextract._extract(extract_params,
                                    calling_function='predict',
                                    make_summary_file=False)

        os.chdir(output_dir)
        ftr_files = glob.glob('features_from*')
        model_filename = model_meta_file['model_filename']
        model_name = model_meta_file['model_name']
        if model_name in valid_models['sklearn']:
            clf = joblib.load(model_filename)
            scaler = model_meta_file['scaler']
        elif model_name in valid_models['keras']:
            if 'keras.models' not in locals():
                import keras.models
            clf = keras.models.load_model(model_filename)
            spect_scaler = model_meta_file['spect_scaler']

        for ftr_file in ftr_files:
            print("predicting labels for features in file: {}"
                  .format(ftr_file))
            ftr_file_dict = joblib.load(ftr_file)
            if model_name in valid_models['sklearn']:
                features = ftr_file_dict['features']
                features_scaled = scaler.transform(features)
                pred_labels = clf.predict(features_scaled)
            elif model_name in valid_models['keras']:
                neuralnet_inputs_dict = ftr_file_dict['neuralnet_inputs']
                inputs_key = model_meta_file['feature_list'][0]
                neuralnet_inputs = neuralnet_inputs_dict[inputs_key]
                neuralnet_inputs_scaled = spect_scaler.transform(neuralnet_inputs)
                neuralnet_inputs_scaled = neuralnet_inputs_scaled[:, :, :, np.newaxis]
                pred_labels = clf.predict(neuralnet_inputs_scaled)
                label_binarizer = model_meta_file['label_binarizer']
                pred_labels = label_binarizer.inverse_transform(pred_labels)

            ftr_file_dict['pred_labels'] = pred_labels

            if 'predict_proba' in todo:
                if todo['predict_proba'] == True:
                    pred_probs = clf.predict_proba(features_scaled)
                    ftr_file_dict['pred_probs'] = pred_probs
            joblib.dump(ftr_file_dict, ftr_file)

            if 'convert' in todo:
                songfiles = ftr_file_dict['songfiles']
                songfile_IDs = ftr_file_dict['songfile_IDs']
                if todo['convert'] == 'notmat':
                    print('converting to .not.mat files')
                    for curr_song_id, songfile_name in enumerate(songfiles):
                        these = np.asarray(songfile_IDs) == curr_song_id
                        pred_labels = ftr_file_dict['pred_labels'][these]
                        onsets_s = ftr_file_dict['onsets_s'][these]
                        offsets_s = ftr_file_dict['offsets_s'][these]
                        segment_params = ftr_file_dict['segment_params']

                        hvc.convert.to_notmat(songfile_name,
                                              pred_labels,
                                              model_name,
                                              32000,
                                              segment_params,
                                              onsets_s,
                                              offsets_s,
                                              alternate_path=output_dir)

    os.chdir(home_dir)
