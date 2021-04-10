"""
model selection:
trains models that classify birdsong syllables,
using algorithms and other parameters specified in config file
"""

# from standard library
import os
import copy

# from dependencies
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import yaml

# from hvc
from .parseconfig import parse_config
from .utils import grab_n_samples_by_song, get_acc_by_label, timestamp
from hvc.parse.select import _validate_models as validate_models


path = os.path.abspath(__file__)  # get the path of this file
dir_path = os.path.dirname(path)  # but then just take the dir

with open(os.path.join(dir_path,
                       'parse',
                       'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml, Loader=yaml.FullLoader)

model_types = validate_dict['valid_models']


def determine_model_output_folder_name(model_dict):
    """generates name for folder that holds model files
    by appending hyperparameter names and values
    in alphabetical order of hyperparameter name
    to name of model/algorithm
    """

    model_dict_copy = copy.copy(model_dict)
    return model_dict_copy.pop(
        'model_name'
    ) + ''.join('_{!s}{!r}'.format(key, val)
                for (key, val) in sorted(
        model_dict['hyperparameters'].items())
                )


def select(config_file=None,
           feature_file_path=None,
           feature_list_indices=None,
           feature_group=None,
           neuralnet_input=None,
           model_name=None,
           hyperparameters=None,
           models=None,
           train_samples_range=None,
           num_replicates=None,
           num_test_samples=None,
           output_dir=None,
           ):
    """high-level function for machine-learning model selection.
    Accepts either a config file or a set of parameters and
    uses them to train the specified classifiers on features extracted
    from syllables and / or spectrograms.

    Parameters
    ----------
    config_file : str
        filename of YAML file that configures feature extraction
    feature_file_path : str
        path to a feature file created with hvc.extract
    feature_list_indices : list
        list of integers. Columns from features matrix in feature file that will
        be used to train classifier. Can alternatively specify `feature_group`,
        see Other Parameters below.
    neuralnet_input : str
        input for neural network model. Currently just 'flatwindow'.
    model_name : str
        one of {'knn', 'svm', 'flatwindow'}, i.e., k-Nearest Neighbors,
        support vector machine, or convolutional neural net. Type of
        machine learning model / classifier to train.
    hyperparameters : dict
        hyperparameters for model, where each keys is a name of a hyperparameter
        and the corresponding value is the value for that hyperparameter.
    train_samples_range : range
        A range of training samples to train models with.
         E.g. `range(100,601,100)` (which gives [100, 200, 400, 500, 600]).
    num_replicates : int
        Number of times to "replicate" results for each number of
        training samples. The samples for each replicate are drawn
        randomly from the entire training set.
    num_test_samples : int
        Number of samples to use to test accuracy of the trained classifier.
    output_dir : str
        directory in which to save output

    Other Parameters
    ----------------
    feature_group : str
        one of {'knn', 'svm'}. If `extract` was run with one of these feature
        groups specified, then it can be referred to by name instead of
        supplying an argument feature_list_indices.
    models : list
        of dicts, where each dict has the following keys:
        {'model_name', 'hyperparameters', 'feature_list_indices' or 'feature_group'}
        The values should be as defined above. When a list of models is supplied,
        each model will be fit according to the

    Returns
    -------
    None

    Saves output in location `output_dir` specified by user.
    """
    ################## argument parsing #################################
    if config_file and feature_file_path:
        raise ValueError('Please specify either config_file or feature_file_path, '
                         'not clear which to use when both are specified')

    if config_file and (train_samples_range or num_replicates or model_name
                        or hyperparameters or models or feature_list_indices
                        or feature_group or neuralnet_input or output_dir):
        raise ValueError('Cannot specify config_file and other parameters '
                         'when calling hvc.select, '
                         'please specify either config_file or all other '
                         'parameters ')

    ################## actual function starts #################################
    if config_file:
        select_config = parse_config(config_file, 'select')
        print('Parsed select config.')

        todo_list = select_config['todo_list']

        for ind, todo in enumerate(todo_list):

            print('Completing item {} of {} in to-do list'.format(ind+1, len(todo_list)))

            output_dir = os.path.abspath(todo['output_dir'])
            output_dir = os.path.join(output_dir,
                                      'select_output_' + timestamp())
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            if 'models' in todo:
                models = todo['models']
            else:
                models = select_config['models']

            if 'num_test_samples' in todo:
                num_test_samples = todo['num_test_samples']
            else:
                num_test_samples = select_config['num_test_samples']

            if 'num_train_samples' in todo:
                train_samples_range = todo['num_train_samples']
            else:
                train_samples_range = select_config['num_train_samples']

            if 'num_replicates' in todo:
                num_replicates = todo['num_replicates']
            else:
                num_replicates = select_config['num_replicates']

            feature_file_path = todo['feature_file']
            feature_file = joblib.load(feature_file_path)

            select_config = {'feature_file': feature_file,
                             'feature_file_path': feature_file_path,
                             'train_samples_range': train_samples_range,
                             'num_replicates': num_replicates,
                             'num_test_samples': num_test_samples,
                             'models': models,
                             'output_dir': output_dir,
                             'config_file': config_file}
            _select(**select_config)

    else:  # if a config_file was not specified
        if not os.path.isfile(feature_file_path):
            raise FileNotFoundError("Can not find feature file: {}"
                                    .format(feature_file_path))
        else:
            feature_file = joblib.load(feature_file_path)

        output_dir = os.path.abspath(output_dir)
        if not os.path.isdir(output_dir):
            raise NotADirectoryError('could not find output_dir: {}'
                                     .format(output_dir))
        else:
            output_dir = os.path.join(output_dir,
                                      'select_output_' + timestamp())
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

        if type(train_samples_range) != range:
            raise TypeError('train_samples_range should be a range, not '
                            '{}'.format(type(train_samples_range)))

        if (type(num_replicates) is not int) or (num_replicates < 1):
            raise ValueError('num_replicates must be an int, greater than 0')

        if (type(num_test_samples) is not int) or (num_test_samples < 1):
            raise ValueError('num_test_samples must be an int, greater than 0')

        if (feature_list_indices is None and feature_group is None
           and neuralnet_input is None):
            raise ValueError("Must specify either `feature_list_indices`, "
                             "`feature_group`, or `neuralnet_input`.")

        if feature_list_indices and feature_group:
            raise ValueError("Cannot call `select` with arguments for both "
                             "feature_list_indices and feature_group, unclear "
                             "which models to fit.")

        if feature_list_indices and neuralnet_input:
            raise ValueError("Cannot call `select` with arguments for both "
                             "feature_list_indices and neuralnet_input, unclear "
                             "which models to fit.")

        if feature_group and neuralnet_input:
            raise ValueError("Cannot call `select` with arguments for both "
                             "feature_group and neuralnet_input, unclear "
                             "which models to fit.")

        if model_name and hyperparameters and models:
            raise ValueError("Cannot call `select` with arguments for both "
                             "models and model_name and hyperparameters, unclear "
                             "which models to fit.")

        if models is None:
            models = {'model_name': model_name,
                      'hyperparameters': hyperparameters}
            if feature_group is not None:
                models['feature_group'] = feature_group
            elif feature_list_indices is not None:
                models['feature_list_indices'] = feature_list_indices
            elif neuralnet_input is not None:
                models['neuralnet_input'] = neuralnet_input
            # make dict into a single-item list
            # because _select helper function expects to iterate
            # through a list
            models = [models]

        # pass these "just in case" parser needs them for validation
        if (('feature_list_group_ID' in feature_file) and
                ('feature_group_ID_dict' in feature_file)):
                ftr_list_group_ID = feature_file['feature_list_group_ID']
                ftr_grp_ID_dict = feature_file['feature_group_ID_dict']
        else:
            ftr_list_group_ID = None
            ftr_grp_ID_dict = None
        models = validate_models(models=models,
                                 ftr_list_group_ID=ftr_list_group_ID,
                                 ftr_grp_ID_dict=ftr_grp_ID_dict)

        select_config = {'feature_file': feature_file,
                         'feature_file_path': feature_file_path,
                         'train_samples_range': train_samples_range,
                         'num_replicates': num_replicates,
                         'num_test_samples': num_test_samples,
                         'models': models,
                         'output_dir': output_dir}

        _select(**select_config)


def _select(feature_file,
            feature_file_path,
            train_samples_range,
            num_replicates,
            num_test_samples,
            models,
            output_dir,
            config_file=None):
    """helper function to do model selection, used with either config_file or
    another set of arguments passed to hvc.select"""
    labels = np.asarray(feature_file['labels'])
    # call grab_n_samples this first time to get indices for test/validation set
    # and a list of song IDs from which we will draw the training set indices below
    test_IDs, train_song_ID_list = grab_n_samples_by_song(feature_file['songfile_IDs'],
                                                          feature_file['labels'],
                                                          num_test_samples,
                                                          return_popped_songlist=True)
    test_labels = labels[test_IDs]

    score_arr = np.zeros((len(train_samples_range),
                       len(range(num_replicates)),
                       len(models)))
    avg_acc_arr = np.zeros((len(train_samples_range),
                            len(range(num_replicates)),
                            len(models)))
    pred_labels_arr = np.empty((len(train_samples_range),
                                len(range(num_replicates)),
                                len(models)),
                               dtype='O')
    train_IDs_arr = np.empty((len(train_samples_range),
                                len(range(num_replicates))),
                             dtype='O')

    for num_samples_ind, num_train_samples in enumerate(train_samples_range):
        for replicate in range(num_replicates):
            print('Training models with {0} samples, replicate #{1}'
                  .format(num_train_samples, replicate))
            # here we call grab_n_samples again with the train_song_ID_list
            # from above. currently each fold is a random grab without
            # anything like k-folds.
            # For testing on large datasets this is okay but in situations
            # where we're data-limited it's less than ideal, the whole point
            # is to not have to hand-label a large data set
            train_IDs = grab_n_samples_by_song(feature_file['songfile_IDs'],
                                               feature_file['labels'],
                                               num_train_samples,
                                               song_ID_list=train_song_ID_list)
            train_IDs_arr[num_samples_ind, replicate] = train_IDs
            train_labels = labels[train_IDs]
            for model_ind, model_dict in enumerate(models):

                # lazy-imports to avoid loading all of
                # scikit-learn and tensorflow if possible
                if model_dict['model_name'] == 'svm':
                    if 'SVC' not in locals():
                        from sklearn.svm import SVC

                elif model_dict['model_name'] == 'knn':
                    if 'neighbors' not in locals():
                        from sklearn import neighbors

                elif model_dict['model_name'] == 'flatwindow':
                    if 'flatwindow' not in locals():
                        from hvc.neuralnet.models.flatwindow import flatwindow
                        from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

                # save info associated with model such as indices of training samples.
                # Note this is done outside the if/elif list for switching between
                # models.
                model_output_dir = os.path.join(output_dir,
                                                determine_model_output_folder_name(
                                                    model_dict))
                if not os.path.isdir(model_output_dir):
                    os.makedirs(model_output_dir)

                model_fname_str = \
                    '{0}_{1}samples_replicate{2}.model'.format(model_dict['model_name'],
                                                               num_train_samples,
                                                               replicate)
                model_filename = os.path.join(model_output_dir, model_fname_str)

                # if-elif that switches based on model type,
                # start with sklearn models
                if model_dict['model_name'] in model_types['sklearn']:

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
                        if type(model_dict['feature_list_indices']) is str:
                            if model_dict['feature_list_indices'] == 'all':
                                feature_inds = np.ones((
                                    feature_file['features_arr_column_IDs'].shape[-1],)).astype(bool)
                            else:
                                raise ValueError('received invalid string for feature_list_indices: {}'
                                                 .format(model_dict['feature_list_indices']))
                        else:
                            # use 'feature_list_indices' from model_dict to get the actual columns
                            # we need from the features array. Again, need to this because multiple
                            # columns might belong to the same feature,
                            # e.g. if the feature is a spectrum
                            feature_inds = np.in1d(feature_file['features_arr_column_IDs'],
                                                   model_dict['feature_list_indices'])

                    if model_dict['model_name'] == 'svm':
                        print('training svm. ', end='')
                        clf = SVC(C=model_dict['hyperparameters']['C'],
                                  gamma=model_dict['hyperparameters']['gamma'],
                                  decision_function_shape='ovr',
                                  probability=model_dict['predict_proba'])

                    elif model_dict['model_name'] == 'knn':
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
                    clf.fit(features_train, train_labels)
                    score = clf.score(features_test, test_labels)
                    print('score on test set: {:05.4f} '.format(score), end='')
                    score_arr[num_samples_ind, replicate, model_ind] = score
                    pred_labels = clf.predict(features_test)
                    pred_labels_arr[num_samples_ind, replicate, model_ind] = pred_labels
                    acc_by_label, avg_acc = get_acc_by_label(test_labels,
                                                             pred_labels,
                                                             feature_file['labels_to_use'])
                    print(', average accuracy on test set: {:05.4f}'.format(avg_acc))
                    avg_acc_arr[num_samples_ind, replicate, model_ind] = avg_acc
                    joblib.dump(clf, model_filename)

                # this is the middle of the if-elif that switches based on model type
                # end sklearn, start keras models
                elif model_dict['model_name'] in model_types['keras']:
                    if 'neuralnet_input' in model_dict:
                        neuralnet_input = model_dict['neuralnet_input']
                        spects = feature_file['neuralnet_inputs'][neuralnet_input]
                    else:
                        # if not specified, assume that input should be the one that
                        # corresponds to the neural net model being trained
                        neuralnet_input = model_dict['model_name']
                        try:
                            spects = feature_file['neuralnet_inputs'][neuralnet_input]
                        except KeyError:
                            raise KeyError('no input specified for model {}, and '
                                           'input type for that model was not found in '
                                           'feature file'
                                           .format(model_dict['model_name']))

                    if 'SpectScaler' not in locals():
                        from hvc.neuralnet.utils import SpectScaler

                    if 'test_labels_onehot' not in locals():
                        from sklearn.preprocessing import LabelBinarizer
                        label_binarizer = LabelBinarizer()
                        test_labels_onehot = label_binarizer.fit_transform(test_labels)

                    if 'test_spects' not in locals():
                        # get spects for test set,
                        # also add axis so correct input_shape for keras.conv_2d
                        test_spects = spects[test_IDs, :, :]

                    train_labels_onehot = label_binarizer.transform(train_labels)

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

                    num_samples, num_freqbins, num_timebins, num_channels = \
                        train_spects_scaled.shape
                    num_label_classes = len(feature_file['labels_to_use'])
                    input_shape = (num_freqbins, num_timebins, num_channels)
                    flatwin = flatwindow(input_shape=input_shape,
                                         num_label_classes=num_label_classes)

                    csv_str = ''.join(['flatwindow_training_',
                                       '{}_samples_'.format(num_train_samples),
                                       'replicate_{}'.format(replicate),
                                       '.log'])
                    csv_filename = os.path.join(model_output_dir, csv_str)
                    csv_logger = CSVLogger(csv_filename,
                                           separator=',',
                                           append=True)

                    checkpoint = ModelCheckpoint(model_filename,
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='max')
                    earlystop = EarlyStopping(monitor='val_accuracy',
                                              min_delta=0,
                                              patience=20,
                                              verbose=1,
                                              mode='auto')
                    callbacks_list = [csv_logger, checkpoint, earlystop]

                    flatwin.fit(train_spects_scaled,
                                train_labels_onehot,
                                validation_data=(test_spects_scaled,
                                                 test_labels_onehot),
                                batch_size=model_dict['hyperparameters']['batch_size'],
                                epochs=model_dict['hyperparameters']['epochs'],
                                callbacks=callbacks_list,
                                verbose=1)

                    pred_labels = flatwin.predict(test_spects_scaled,
                                                  batch_size=32,
                                                  verbose=1)
                    pred_labels = label_binarizer.inverse_transform(pred_labels)

                    score = accuracy_score(test_labels, pred_labels)
                    print('score on test set: {:05.4f} '.format(score), end='')
                    score_arr[num_samples_ind, replicate, model_ind] = score

                    pred_labels_arr[num_samples_ind, replicate, model_ind] = pred_labels

                    acc_by_label, avg_acc = get_acc_by_label(test_labels,
                                                             pred_labels,
                                                             feature_file['labels_to_use'])
                    print(', average accuracy on test set: {:05.4f}'.format(avg_acc))
                    avg_acc_arr[num_samples_ind, replicate, model_ind] = avg_acc

            model_meta_fname_str = \
                '{0}_{1}samples_replicate{2}.meta'.format(model_dict['model_name'],
                                                          num_train_samples,
                                                          replicate)
            model_meta_filename = os.path.join(model_output_dir,
                                               model_meta_fname_str)
            model_meta_output_dict = {
                'model_filename': model_filename,
                'config_file': config_file,
                'feature_file': feature_file_path,
                'test_IDs': test_IDs,
                'train_IDs': train_IDs,
                'model_name': model_dict['model_name'],
                'pred_labels': pred_labels,
                'test_labels': test_labels
            }

            if 'scaler' in locals():
                model_meta_output_dict['scaler'] = scaler
                # have to delete scaler
                # so it's not still in memory next loop
                # (e.g. because a different model that doesn't use scaler
                # is tested in next loop)
                del scaler
            elif 'spect_scaler' in locals():
                # neural net models uses scaler on spectrogram
                # instead of vanilla sklearn scalar
                model_meta_output_dict['spect_scaler'] = spect_scaler
                del spect_scaler

            if 'label_binarizer' in locals():
                model_meta_output_dict['label_binarizer'] = label_binarizer

            if model_dict['model_name'] in model_types['sklearn']:
                # to be able to extract features for predictions
                # on unlabeled data set, need list of features
                if ((type(model_dict['feature_list_indices']) is str) and
                        (model_dict['feature_list_indices'] == 'all')):
                    model_feature_list = feature_file['feature_list']
                else:
                    model_feature_list = [feature_file['feature_list'][ind]
                                          for ind in model_dict['feature_list_indices']]
                model_meta_output_dict['feature_list'] = model_feature_list
            elif model_dict['model_name'] in model_types['keras']:
                model_meta_output_dict['feature_list'] = [neuralnet_input]
            joblib.dump(model_meta_output_dict,
                        model_meta_filename)

    # after looping through all samples + replicates
    output_filename = os.path.join(output_dir,
                                   'summary_model_select_file_created_'
                                   + timestamp())
    select_summary_dict = {
        'config_file': config_file,
        'feature_file': feature_file_path,
        'train_samples_range': train_samples_range,
        'num_replicates': num_replicates,
        'model_dict': model_dict,
        'test_IDs': test_IDs,
        'train_IDs_arr': train_IDs_arr,
        'score_arr': score_arr,
        'avg_acc_arr': avg_acc_arr,
        'pred_labels_arr': pred_labels_arr,
    }
    joblib.dump(select_summary_dict, output_filename)
