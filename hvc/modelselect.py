"""
model selection:
trains models that classify birdsong syllables,
using algorithms and other parameters specified in config file
"""

#from standard library
import sys
import glob
import glob
from datetime import datetime

# from dependencies
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

#from hvc
from .parseconfig import parse_config
from .utils import filter_samples

def select(config_file):
    """
    main function that runs model selection
    
    Parameters
    ----------
    config_file  : string
        filename of YAML file that configures feature extraction    
    """
    select_config = parse_config(config_file,'select')
    print('Parsed select config.')

    # # if no feature_files provided, extract feature_files automatically
    # if 'model_selection' in locals() and 'feature_files' not in model_selection:
    #     auto_extract_features = True
    #
    # if auto_extract_features:
    #     import hvc.extract
    #     hvc.extract('')

    todo_list = extract_config['todo_list']
    for ind, todo in enumerate(todo_list):

        timestamp = datetime.now().strftime('%y%m%d_%H%M')

        print('Completing item {} of {} in to-do list'.format(ind+1,len(todo_list)))

        #import models objects from sklearn + keras if not imported already
        if 'svm' in todo['models'].keys():
            if 'SVC' not in locals():
                from sklearn.svm import SVC
                svm_scaler = StandardScaler()
            svm_clf = SVC(C=best_params['C'], gamma=best_params['gamma'], decision_function_shape='ovr')

        if 'knn' in todo['models'].keys():
            if 'neighbors' not in locals():
                from sklearn import neighbors
                knn_scaler = StandardScaler()

        num_samples_list = todo['num_samples']
        num_replicates = todo['num_replicates']
        features_file = joblib.load(todo['features_file'])

        for num_samples in num_samples_list:
            for replicate in num_replicates:
                sample_inds = random.samples(num_samples)
                for model, feature_inds in todo['models'].items():
                    if 'svm' in models:
                        pass
                    if 'knn' in models:
                        pass