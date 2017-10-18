"""
predict labels for birdsong syllables,
using already-trained models specified in config file
"""

import os
import sys
import glob

# from dependencies
from sklearn.externals import joblib

# from hvc
import hvc.featureextract
from .parseconfig import parse_config
from .utils import timestamp


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
            if 'keras.models' not in sys.modules:
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
