"""
predict labels for birdsong syllables,
using already-trained models specified in config file
"""

# #from standard library
# import glob
# import sys
# import os
# import shelve
#
# import numpy as np
# import scipy.io as scio # to load matlab files
# import yaml
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn import neighbors
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib

#from hvc
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

    for todo in predict_config['todo_list']:
        model_file = joblib.open(todo['model_file'])

        #need to get full directory path
        clf = model_file['model']
        scaler = model_file['scaler']
        feature_list = model_file['model_feature_list']

        #loop through dirs
        for data_dir in todo['data_dirs']:
            os.chdir(data_dir)
            if todo['file_format'] == 'evtaf':
                songfiles = glob.glob('*.not.mat')
            elif todo['file_format'] == 'koumura':
                songfiles = glob.glob('*.wav')


            for ftr_file,notmat in zip(ftr_files,notmats):
                if type(clf)==neighbors.classification.KNeighborsClassifier:
                    samples = load_from_mat(ftr_file,'knn','classify')
                elif type(clf)==SVC:
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
