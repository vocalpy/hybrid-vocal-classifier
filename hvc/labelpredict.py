"""
predict labels for birdsong syllables,
using already-trained models specified in config file
"""

# from dependencies
import numpy as np
import scipy.io as scio # to load matlab files
from sklearn.externals import joblib

# from hvc
import hvc.featureextract._extract
from .parseconfig import parse_config
from .utils import load_from_mat

# used in loop below, see there for explanation
SHOULD_BE_DOUBLE = ['Fs',
                    'min_dur',
                    'min_int',
                    'offsets',
                    'onsets',
                    'sm_win',
                    'threshold']


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
        model_file = joblib.load(todo['model_file'])

        #need to get full directory path
        clf = model_file['model']
        scaler = model_file['scaler']

        extract_params = {
            'bird_ID': todo['bird_ID'],
            'feature_list': model_file['model_feature_list'],
            'output_dir': output_dir_with_path,
            'home_dir': home_dir,
            'data_dirs': todo['data_dirs'],
            'labelset': todo['labelset'],
            'file_format': todo['file_format']
        }

        # segment_params defined for todo_list item takes precedence over any default
        # defined for `extract` config
        if 'segment_params' in todo:
            extract_params['segment_params'] = todo['segment_params']
        else:
            extract_params['segment_params'] = extract_config['segment_params']

        if 'spect_params' in todo:
            extract_params['spect_params'] = todo['spect_params']
        else:
            extract_params['spect_params'] = extract_config['spect_params']

        hvc.featureextract._extract(extract_params)

        for ftr_file, notmat in zip(ftr_files,notmats):
            if type(clf) == neighbors.classification.KNeighborsClassifier:
                samples = load_from_mat(ftr_file,'knn','classify')
            elif type(clf) == SVC:
                samples = load_from_mat(ftr_file,'svm','classify')
            samples_scaled = scaler.transform(samples)
            pred_labels = clf.predict(samples_scaled)
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
