"""
predict labels for birdsong syllables,
using already-trained models specified in config file
"""

import os
import glob

# from dependencies
import numpy as np
import scipy.io as scio # to load matlab files
from sklearn.externals import joblib

# from hvc
import hvc.featureextract
from .parseconfig import parse_config
from .utils import timestamp

# used by convert_predicted_labels_to_notmat
SHOULD_BE_DOUBLE = ['Fs',
                    'min_dur',
                    'min_int',
                    'offsets',
                    'onsets',
                    'sm_win',
                    'threshold']


def convert_predicted_labels_to_notmat(notmat, pred_labels, clf_file):
    """converts predicted labels into a .not.mat file
    that can be read by evsonganaly.m (MATLAB GUI for labeling song)

    Parameters
    ----------
    notmat: str
        filename
    pred_labels: ndarray
        output from model / classifier
    clf_file: str
        name of file from which model / classifier was loaded

    Returns
    -------
    None.
    Saves .not.mat file with additional information
        predicted_labels
        classifier_file
    """

    # chr() to convert back to character from uint32
    pred_labels = [chr(val) for val in pred_labels]
    # convert into one long string, what evsonganaly expects
    pred_labels = ''.join(pred_labels)
    notmat_dict = scio.loadmat(notmat)
    notmat_dict['predicted_labels'] = pred_labels
    notmat_dict['classifier_file'] = clf_file
    print('saving ' + notmat)
    # evsonganaly/Matlab expects all vars as double
    for key, val in notmat_dict.items():
        if key in SHOULD_BE_DOUBLE:
            notmat_dict[key] = val.astype('d')
    scio.savemat(notmat, notmat_dict)


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

        output_dir = 'predict_output_' + timestamp()
        output_dir_with_path = os.path.join(todo['output_dir'], output_dir)
        if not os.path.isdir(output_dir_with_path):
            os.mkdir(output_dir_with_path)

        model_file = joblib.load(todo['model_file'])

        extract_params = {
            'bird_ID': todo['bird_ID'],
            'feature_list': model_file['model_feature_list'],
            'output_dir': output_dir_with_path,
            'home_dir': home_dir,
            'data_dirs': todo['data_dirs'],
            'labelset': 'all',
            'file_format': todo['file_format']
        }

        feature_file_for_model = model_file['feature_file']
        feature_file = joblib.load(feature_file_for_model)
        extract_params['segment_params'] = feature_file['segment_params']
        extract_params['spect_params'] = feature_file['spect_params']

        hvc.featureextract._extract(extract_params, make_summary_file=False)

        os.chdir(output_dir_with_path)
        ftr_files = glob.glob('features_from*')
        model = model_file['model']
        if model in ['knn', 'svm']:
            try:
                clf = model_file['clf']
            except:
                raise KeyError('model in {} is {} but '
                               'no corresponding \'clf\''
                               '(classifier) found in model file.'
                               .format(todo['model_file'],
                                       model))
        elif model in ['flatwindow']:
            if 'keras.models' not in locals():
                import keras.models
            clf = keras.models.load_model()
        scaler = model_file['scaler']

        for ftr_file in ftr_files:
            print("predicting labels for features in file: {}"
                  .format(ftr_file))
            ftr_file_dict = joblib.load(ftr_file)
            features = ftr_file_dict['features']

            features_scaled = scaler.transform(features)
            pred_labels = clf.predict(features_scaled)
            ftr_file_dict['pred_labels'] = pred_labels
            joblib.dump(ftr_file_dict, ftr_file)
