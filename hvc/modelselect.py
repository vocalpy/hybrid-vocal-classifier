"""
model selection:
trains models that classify birdsong syllables,
using algorithms and other parameters specified in config file
"""

#from standard library
import sys
import os
import glob
from datetime import datetime

# from dependencies
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

#from hvc
from .parseconfig import parse_config
from .utils import filter_samples, grab_n_samples_by_song, get_acc_by_label

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

    todo_list = select_config['todo_list']

    for ind, todo in enumerate(todo_list):

        print('Completing item {} of {} in to-do list'.format(ind+1,len(todo_list)))

        timestamp = datetime.now().strftime('%y%m%d_%H%M')
        output_dir = todo['output_dir'] + 'select_output_' + timestamp
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        if 'models' in todo:
            model_list = todo['models']
        else:
            model_list = select_config['global']['models']

        for model_dict in model_list:
            # import models objects from sklearn + keras if not imported already
            if model_dict['model'] == 'svm':
                if 'SVC' not in locals():
                    from sklearn.svm import SVC

            elif model_dict['model'] == 'knn':
                if 'neighbors' not in locals():
                    from sklearn import neighbors

        if 'num_test_samples' in todo:
            num_test_samples = todo['num_test_samples']
        else:
            num_test_samples = select_config['global']['num_test_samples']

        if 'num_train_samples' in todo:
            num_train_samples_list = todo['num_train_samples']
        else:
            num_train_samples_list = select_config['global']['num_train_samples']

        if 'num_replicates' in todo:
            num_replicates = todo['num_replicates']
        else:
            num_replicates = select_config['global']['num_replicates']

        feature_file = joblib.load(todo['feature_file'])
        labels = np.asarray(feature_file['labels'])
        test_IDs, train_song_ID_list = grab_n_samples_by_song(feature_file['song_IDs'],
                                                              feature_file['labels'],
                                                              num_test_samples,
                                                              return_popped_songlist=True)
        labels_test = labels[test_IDs]

        score_arr = np.zeros((len(num_train_samples_list),
                           len(range(num_replicates)),
                           len(model_list)))
        avg_acc_arr = np.zeros((len(num_train_samples_list),
                                len(range(num_replicates)),
                                len(model_list)))
        pred_labels_arr = np.empty((len(num_train_samples_list),
                                    len(range(num_replicates)),
                                    len(model_list)),
                                   dtype='O')

        for num_samples_ind, num_train_samples in enumerate(num_train_samples_list):
            for replicate in range(num_replicates):
                print('Training models with {0} samples, replicate #{1}'
                      .format(num_train_samples,replicate))
                train_IDs = grab_n_samples_by_song(feature_file['song_IDs'],
                                                   feature_file['labels'],
                                                   num_train_samples,
                                                   song_ID_list=train_song_ID_list)
                labels_train = labels[train_IDs]
                for model_ind, model_dict in enumerate(model_list):
                    if model_dict['model'] == 'svm':
                        print('training svm')
                        clf = SVC(C=model_dict['hyperparameters']['C'],
                                      gamma=model_dict['hyperparameters']['gamma'],
                                      decision_function_shape='ovr')
                    elif model_dict['model'] == 'knn':
                        print('training knn')
                        clf = neighbors.KNeighborsClassifier(model_dict['hyperparameters']['k'],
                                                             'distance')

                    if model_dict['model'] is not 'neuralnet':
                        #use 'advanced indexing' to get only sample rows and only feature models
                        features_train = feature_file['features'][train_IDs[:,np.newaxis],
                                                                  model_dict['feature_indices']]
                        scaler = StandardScaler()
                        features_train = scaler.fit_transform(features_train)

                        features_test = feature_file['features'][test_IDs[:,np.newaxis],
                                                                 model_dict['feature_indices']]
                        features_test = scaler.transform(features_test)

                        print('fitting model')
                        clf.fit(features_train, labels_train)
                        score = clf.score(features_test, labels_test)
                        print('score on test set: {}'.format(score))
                        score_arr[num_samples_ind, replicate, model_ind] = score
                        pred_labels = clf.predict(features_test)
                        pred_labels_arr[num_samples_ind, replicate, model_ind] = pred_labels
                        acc_by_label, avg_acc = get_acc_by_label(labels_test,
                                                                 pred_labels,
                                                                 feature_file['labelset'])
                        print('average accuracy on test set: {}'.format(avg_acc))

                        model_output_dir = os.path.join(output_dir,model_dict['model'])
                        if not os.path.isdir(model_output_dir):
                            os.mkdir(model_output_dir)
                        model_fname_str = '{0}_{1}samples_replicate{2}'.format(model_dict['model'],
                                                                               num_train_samples,
                                                                               replicate)
                        model_filename = os.path.join(model_output_dir,model_fname_str)
                        joblib.dump(clf,model_filename)