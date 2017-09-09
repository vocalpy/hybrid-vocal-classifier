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
# from sklearn import neighbors, SVC

# from hvc
import hvc.featureextract
from .parseconfig import parse_config
from .utils import timestamp

# used in loop below, see there for explanation
SHOULD_BE_DOUBLE = ['Fs',
                    'min_dur',
                    'min_int',
                    'offsets',
                    'onsets',
                    'sm_win',
                    'threshold']

model_str_rep_map = {
    "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>": 'knn',
    "svm thingy": 'svm',
    "keras thingy": 'flatwindow'
}


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

        model_feature_file = model_file['feature_file']
        model_feature_file = joblib.load(model_feature_file)
        extract_params['segment_params'] = model_feature_file['segment_params']
        extract_params['spect_params'] = model_feature_file['spect_params']

        hvc.featureextract._extract(extract_params, make_summary_file=False)

        os.chdir(output_dir_with_path)
        ftr_files = glob.glob('features_from*')
        clf = model_file['clf']
        scaler = model_file['scaler']

        #clf_type = model_str_rep_map[str(clf)]
        # if clf_type == 'knn':
        #     if 'neighbors.KNeighborsClassifier' not in locals():
        #         import neighbors.KNeighborsClassifier
        # elif clf_type == 'svm':
        #     if SVC not in locals():
        #         from sklearn.svm import SVC
        # elif clf_type == flatwindow

        for ftr_file in ftr_files:
            ftr_file_dict = joblib.load(ftr_file)
            features = ftr_file_dict['features']
            # check classifier type without importing every model
            if str(type(clf)) == \
                    "<class 'sklearn.neighbors.classification.KNeighborsClassifier'>":
                pass
            elif type(clf) == SVC:
                pass
            features_scaled = scaler.transform(features)
            pred_labels = clf.predict(features_scaled)

            #chr() to convert back to character from uint32
            pred_labels = [chr(val) for val in pred_labels]
            # convert into one long string, what evsonganalty expects
            pred_labels = ''.join(pred_labels)
            notmat_dict = scio.loadmat(notmat)
            notmat_dict['predicted_labels'] = pred_labels
            notmat_dict['classifier_type'] = clf_type
            notmat_dict['classifier_file'] = clf_file
            print('saving ' + notmat)
            # evsonganaly/Matlab expects all vars as double
            for key, val in notmat_dict.items():
                if key in SHOULD_BE_DOUBLE:
                    notmat_dict[key] = val.astype('d')
            scio.savemat(notmat,notmat_dict)
