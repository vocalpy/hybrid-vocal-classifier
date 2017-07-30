"""
model selection:
trains models that classify birdsong syllables,
using algorithms and other parameters specified in config file
"""

# from standard library
import os
from datetime import datetime

# from dependencies
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

# from hvc
from .parseconfig import parse_config
from .utils import grab_n_samples_by_song, get_acc_by_label, timestamp


def select(config_file):
    """main function that runs model selection.
    Saves model files and summary file, doesn't return anything.
    
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

        a_timestamp = timestamp()
        output_dir = todo['output_dir'] + 'select_output_' + a_timestamp
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_filename = os.path.join(output_dir, 'select_output_' + a_timestamp)

        if 'models' in todo:
            model_list = todo['models']
        else:
            model_list = select_config['models']

        for model_dict in model_list:
            # import models objects from sklearn + keras if not imported already
            if model_dict['model'] == 'svm':
                if 'SVC' not in locals():
                    from sklearn.svm import SVC

            elif model_dict['model'] == 'knn':
                if 'neighbors' not in locals():
                    from sklearn import neighbors

            elif model_dict['model'] == 'flatwindow':
                if 'flatwindow' not in locals():
                    from hvc.neuralnet.models import flatwindow
                    from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

        if 'num_test_samples' in todo:
            num_test_samples = todo['num_test_samples']
        else:
            num_test_samples = select_config['num_test_samples']

        if 'num_train_samples' in todo:
            num_train_samples_list = todo['num_train_samples']
        else:
            num_train_samples_list = select_config['num_train_samples']

        if 'num_replicates' in todo:
            num_replicates = todo['num_replicates']
        else:
            num_replicates = select_config['num_replicates']

        feature_file = joblib.load(todo['feature_file'])
        labels = np.asarray(feature_file['labels'])
        # call grab_n_samples this first time to get indices for test/validation set
        # and a list of song IDs from which we will draw the training set indices below
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
        train_IDs_arr = np.empty((len(num_train_samples_list),
                                    len(range(num_replicates))),
                                 dtype='O')

        for num_samples_ind, num_train_samples in enumerate(num_train_samples_list):
            for replicate in range(num_replicates):
                print('Training models with {0} samples, replicate #{1}'
                      .format(num_train_samples, replicate))
                # here we call grab_n_samples again with the train_song_ID_list
                # from above. currently each fold is a random grab without
                # anything like k-folds.
                # For testing on large datasets this is okay but in situations
                # where we're data-limited it's less than ideal, the whole point
                # is to not have to hand-label a large data set
                train_IDs = grab_n_samples_by_song(feature_file['song_IDs'],
                                                   feature_file['labels'],
                                                   num_train_samples,
                                                   song_ID_list=train_song_ID_list)
                train_IDs_arr[num_samples_ind, replicate] = train_IDs
                labels_train = labels[train_IDs]
                for model_ind, model_dict in enumerate(model_list):

                    # if model_dict specifies using a certain feature group
                    if 'feature_group' in model_dict:
                        # determine if we already figured out which features belong to that feature group.
                        # Can only do that if model_dict defined for todo_list, not if model_dict defined
                        # at top level of select config file
                        if 'feature_list_indices' in model_dict:
                            feature_inds = np.in1d(feature_file['features_arr_column_IDs'],
                                                   model_dict['feature_list_indices'])
                        else:  # have to figure out feature list indices
                            ftr_grp_ID_dict = feature_file['feature_group_ID_dict']
                            ftr_list_grp_ID = feature_file['feature_list_group_ID']
                            # figure out what they are by finding ID # corresponding to feature
                            # group or groups in ID_dict, and then finding all the indices in the
                            # feature list that have that group ID #, using ftr_list_grp_ID, a list
                            # the same length as feature list where each element indicates whether
                            # the element with the same index in the feature list belongs to a
                            # feature group and if so which one, by ID #
                            if type(model_dict['feature_group']) == str:
                                ftr_grp_ID = ftr_grp_ID_dict[model_dict['feature_group']]
                                # now find all the indices of features associated with the
                                # feature group for that model
                                ftr_list_inds = [ind for ind, val in
                                                 enumerate(ftr_list_grp_ID)
                                                 if val == ftr_grp_ID]

                            # if user specified more than one feature group
                            elif type(model_dict['feature_group']) == list:
                                ftr_list_inds = []
                                for ftr_grp in model_dict['feature_group']:
                                    ftr_grp_ID = ftr_grp_ID_dict[ftr_grp]
                                    # now find all the indices of features associated with the
                                    # feature group for that model
                                    ftr_list_inds.extend([ind for ind, val in
                                                          enumerate(ftr_list_grp_ID)
                                                          if val == ftr_grp_ID])
                            # finally use ftr_list_inds to get the actual columns we need from the
                            # features array. Need to this because multiple columns might belong to
                            # the same feature, e.g. if the feature is a spectrum
                            feature_inds = np.in1d(feature_file['features_arr_column_IDs'],
                                                   ftr_list_inds)
                            # put feature list indices in model dict so we have it later when
                            # saving summary file
                            model_dict['feature_list_indices'] = ftr_list_inds

                    elif 'feature_list_indices' in model_dict and\
                            'feature_group' not in model_dict:
                        # if no feature group in model dict, use feature list indices
                        # Note that for neuralnet models, there will be neither
                        if model_dict['feature_list_indices'] == 'all':
                            feature_inds = np.ones((
                                feature_file['features_arr_column_IDs'].shape[-1],)).astype(bool)
                        else:
                            # use 'feature_list_indices' from model_dict to get the actual columns
                            # we need from the features array. Again, need to this because multiple
                            # columns might belong to the same feature,
                            # e.g. if the feature is a spectrum
                            feature_inds = np.in1d(feature_file['features_arr_column_IDs'],
                                                   model_dict['feature_list_indices'])

                    # if-elif that switches based on model type,
                    # start with sklearn models
                    if model_dict['model'] in ['svm', 'knn']:
                        if model_dict['model'] == 'svm':
                            print('training svm. ', end='')
                            clf = SVC(C=model_dict['hyperparameters']['C'],
                                      gamma=model_dict['hyperparameters']['gamma'],
                                      decision_function_shape='ovr')

                        elif model_dict['model'] == 'knn':
                            print('training knn. ', end='')
                            clf = neighbors.KNeighborsClassifier(model_dict['hyperparameters']['k'],
                                                                 'distance')

                        #use 'advanced indexing' to get only sample rows and only feature models
                        features_train = feature_file['features'][train_IDs[:, np.newaxis],
                                                                  feature_inds]
                        scaler = StandardScaler()
                        features_train = scaler.fit_transform(features_train)

                        features_test = feature_file['features'][test_IDs[:,np.newaxis],
                                                                 feature_inds]
                        features_test = scaler.transform(features_test)

                        print('fitting model. ', end='')
                        clf.fit(features_train, labels_train)
                        score = clf.score(features_test, labels_test)
                        print('score on test set: {:05.4f} '.format(score), end='')
                        score_arr[num_samples_ind, replicate, model_ind] = score
                        pred_labels = clf.predict(features_test)
                        pred_labels_arr[num_samples_ind, replicate, model_ind] = pred_labels
                        acc_by_label, avg_acc = get_acc_by_label(labels_test,
                                                                 pred_labels,
                                                                 feature_file['labelset'])
                        print(', average accuracy on test set: {:05.4f}'.format(avg_acc))
                        avg_acc_arr[num_samples_ind, replicate, model_ind] = avg_acc

                        # # save info associated with model such as indices of training samples
                        # model_output_dir = os.path.join(output_dir,model_dict['model'])
                        # if not os.path.isdir(model_output_dir):
                        #     os.mkdir(model_output_dir)
                        # model_fname_str = '{0}_{1}samples_replicate{2}'.format(model_dict['model'],
                        #                                                        num_train_samples,
                        #                                                        replicate)
                        # model_filename = os.path.join(model_output_dir, model_fname_str)
                        # if model_dict['feature_list_indices'] == 'all':
                        #     model_feature_list = feature_file['feature_list']
                        # else:
                        #     model_feature_list = [feature_file['feature_list'][ind]
                        #                           for ind in model_dict['feature_list_indices']]
                        # model_output_dict = {
                        #     'model_feature_list': model_feature_list,
                        #     'model': clf,
                        #     'config_file': config_file,
                        #     'feature_file': todo['feature_file'],
                        #     'test_IDs': test_IDs,
                        #     'train_IDs': train_IDs,
                        #     'scaler': scaler
                        # }
                        # joblib.dump(model_output_dict,
                        #             model_filename)

                    # if-elif that switches based on model type, end sklearn, start keras models
                    elif model_dict['model'] == 'flatwindow':
                        spects = feature_file['neuralnet_inputs']['flatwindow']

                        if 'convert_labels_categorical' not in locals():
                            from hvc.neuralnet.utils import convert_labels_categorical

                        if 'SpectScaler' not in locals():
                            from hvc.neuralnet.utils import SpectScaler

                        if 'labels_test_onehot' not in locals():
                            labels_test_onehot, labels_test_zero_to_n, classes_zero_to_n = \
                                convert_labels_categorical(feature_file['labelset'],
                                                           labels_test,
                                                           return_zero_to_n=True)

                        if 'test_spects' not in locals():
                            # get spects for test set,
                            # also add axis so correct input_shape for keras.conv_2d
                            test_spects = spects[test_IDs, :, :]

                        labels_train_onehot = \
                            convert_labels_categorical(feature_file['labelset'],
                                                       labels_train)

                        # get spects for train set,
                        # also add axis so correct input_shape for keras.conv_2d
                        train_spects = spects[train_IDs, :, :]

                        # scale all spects by mean and std of training set
                        spect_scaler = SpectScaler()
                        # concatenate all spects then rotate so
                        # Hz bins are columns, i.e., 'features'
                        spect_scaler.fit(train_spects)
                        train_spects_scaled = spect_scaler.transform(train_spects)
                        test_spects_scaled = spect_scaler.transform(test_spects)

                        # have to add 'channels' axis for keras 2-d convolution
                        # even though these are spectrograms, don't have channels
                        # like an image would.
                        # Decided to leave it explicit here instead of hiding in a function
                        train_spects_scaled = train_spects_scaled[:, :, :, np.newaxis]
                        test_spects_scaled = test_spects_scaled[:, :, :, np.newaxis]

                        num_samples, num_freqbins, num_timebins, num_channels = train_spects_scaled.shape
                        num_label_classes = len(feature_file['labelset'])
                        input_shape = (num_freqbins, num_timebins, num_channels)
                        flatwin = flatwindow(input_shape=input_shape,
                                             num_label_classes=num_label_classes)

                        model_output_dir = os.path.join(output_dir, model_dict['model'])
                        if not os.path.isdir(model_output_dir):
                            os.mkdir(model_output_dir)
                        model_fname_str = '{0}_{1}samples_replicate{2}'.format(model_dict['model'],
                                                                               num_train_samples,
                                                                               replicate)
                        model_filename = os.path.join(model_output_dir, model_fname_str)
                        csv_str = ''.join(['flatwindow_training_',
                                           '{}_samples_'.format(num_train_samples),
                                           'replicate_{}'.format(replicate),
                                           '.log'])
                        csv_filename = os.path.join(model_output_dir, csv_str)
                        csv_logger = CSVLogger(csv_filename,
                                               separator=',',
                                               append=True)
                        weights_str = ''.join(['weights_',
                                               '{}_samples_'.format(num_train_samples),
                                               'replicate_{}'.format(replicate),
                                               '.best.hdf5'])
                        weights_filename = os.path.join(model_output_dir, weights_str)
                        checkpoint = ModelCheckpoint(weights_filename,
                                                     monitor='val_acc',
                                                     verbose=1,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     mode='max')
                        earlystop = EarlyStopping(monitor='val_acc',
                                                  min_delta=0,
                                                  patience=20,
                                                  verbose=1,
                                                  mode='auto')
                        callbacks_list = [csv_logger, checkpoint, earlystop]

                        flatwin.fit(train_spects_scaled,
                                    labels_train_onehot,
                                    validation_data=(test_spects_scaled,
                                                     labels_test_onehot),
                                    batch_size=model_dict['hyperparameters']['batch size'],
                                    epochs=model_dict['hyperparameters']['epochs'],
                                    callbacks=callbacks_list,
                                    verbose=1)

                        pred_labels = flatwin.predict_classes(test_spects_scaled,
                                                              batch_size=32,
                                                              verbose=1)

                        acc_by_label, avg_acc = get_acc_by_label(labels_test_zero_to_n,
                                                                 pred_labels,
                                                                 classes_zero_to_n)

                # save info associated with model such as indices of training samples.
                # Note this is done outside the if/elif list for switching between
                # models.
                model_output_dir = os.path.join(output_dir, model_dict['model'])
                if not os.path.isdir(model_output_dir):
                    os.mkdir(model_output_dir)
                model_fname_str = '{0}_{1}samples_replicate{2}'.format(model_dict['model'],
                                                                       num_train_samples,
                                                                       replicate)
                model_filename = os.path.join(model_output_dir, model_fname_str)
                model_output_dict = {
                    'config_file': config_file,
                    'feature_file': todo['feature_file'],
                    'test_IDs': test_IDs,
                    'train_IDs': train_IDs,
                }

                if 'clf' in locals():
                    model_output_dict['clf'] = clf

                if 'scaler' in locals():
                    model_output_dict['scaler'] = scaler
                    # have to delete scaler
                    # so it's not still in memory next loop
                    # (e.g. because a different model that doesn't use scaler
                    # is tested in next loop)
                    del scaler
                elif 'spect_scaler' in locals():
                    model_output_dict['spect_scaler'] = spect_scaler
                    del spect_scaler

                if model_dict['model'] in ['svm', 'knn']:
                    # to be able to extract features for predictions
                    # on unlabeled data set, need list of features
                    if model_dict['feature_list_indices'] == 'all':
                        model_feature_list = feature_file['feature_list']
                    else:
                        model_feature_list = [feature_file['feature_list'][ind]
                                              for ind in model_dict['feature_list_indices']]
                    model_output_dict['model_feature_list'] = model_feature_list
                joblib.dump(model_output_dict,
                            model_filename)

        # after looping through all samples + replicates
        output_dict = {
            'config_file': config_file,
            'feature_file': todo['feature_file'],
            'num_train_samples_list': num_train_samples_list,
            'num_replicates': num_replicates,
            'model_dict': model_dict,
            'test_IDs': test_IDs,
            'train_IDs_arr': train_IDs_arr,
            'score_arr': score_arr,
            'avg_acc_arr': avg_acc_arr,
            'pred_labels_arr': pred_labels_arr,
        }
        joblib.dump(output_dict, output_filename)
