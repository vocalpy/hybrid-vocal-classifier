"""
predict labels for birdsong syllables,
using already-trained models specified in config file
"""

import os
from glob import glob

import yaml
import numpy as np
import joblib

import hvc.utils
import hvc.utils.annotation as annotation
import hvc.parseconfig

path = os.path.abspath(__file__)  # get the path of this file
dir_path = os.path.dirname(path)  # but then just take the dir

with open(os.path.join(dir_path, 'parse', 'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml, Loader=yaml.FullLoader)
valid_models = validate_dict['valid_models']
valid_convert_types = validate_dict['valid_convert_types']


def predict(config_file=None,
            data_dirs=None,
            annotation_file=None,
            file_format=None,
            model_meta_file=None,
            output_dir=None,
            segment=None,
            predict_proba=False,
            convert_to=None,
            return_predictions=True):
    """high-level function for prediction of syllable labels.
    Accepts either a config file or a set of parameters and
    uses them to predict labels for syllable segments in audio files,
    based on features extracted from those segments.

    Parameters
    ----------
    config_file : string
        filename of YAML file that configures label prediction
    data_dirs : list
        of str, directories that contain audio files from which features should be extracted.
        hvc.extract attempts to create an annotation.csv file based on the audio file types in
        the directories.
    annotation_file : str
        filename of an annotation.csv file
    file_format : str
        format of audio files. One of the following: {'cbin','wav'}
    model_meta_file : str
        filename of .meta file for classifier to use.
    output_dir : str
        Name of parent directory in which to create output. If parent directory does not exist, it is created.
        Default is current working directory.
    segment : bool
        if True, segment song. If annotation file is passed as an argument, then segments from that file
        are used. If data_dirs is passed as an argument, and segment is False, then the FeatureExtractor
        will look for annotation files, and will raise an error if none are found. Default when data_dirs
        is passed as an argument is True, i.e. it is assumed the user has not already segmented the song
        and wants to do this in an automated way, then apply classifiers to the segments.
    predict_proba : bool
        If True, estimate probabilities that labels are correct. Default is False.
    convert_to: str
        If True, convert predictions to annotation files. Default is False.
    return_predictions : bool
        If True, return feature file with predicted labels. Default is True.

    Returns
    -------
    predictions : dict
        feature file returned as a Python dictionary, with the additional
        (key, value) pair of 'pred_labels', a Numpy array containing the
        labels predicted by the classifier.
        Only returned if return_predictions = True.
    """
    if config_file and (file_format or model_meta_file or output_dir or segment
                        or convert_to):
        raise ValueError('Cannot specify config_file and other parameters '
                         'when calling hvc.predict, '
                         'please specify either config_file or all other '
                         'parameters ')

    if config_file and data_dirs:
        raise ValueError('Please specify either config_file or data_dirs, '
                         'not clear which to use when both are specified')

    if config_file and annotation_file:
        raise ValueError('Please specify either config_file or annotation_file, '
                         'not clear which to use when both are specified')

    home_dir = os.getcwd()

    if config_file:
        predict_config = hvc.parseconfig.parse_config(config_file, 'predict')
        print('parsed predict config')

        for todo in predict_config['todo_list']:
            # get absolute path before changing directories
            # in case user specified output as a relative dir
            output_dir = os.path.abspath(todo['output_dir'])
            output_dir = os.path.join(output_dir, 'predict_output_'
                                      + hvc.utils.timestamp())
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            extract_params = {
                'output_dir': output_dir,
                'data_dirs': todo['data_dirs'],
                'labels_to_use': 'all',
                'file_format': todo['file_format']
            }

            model_meta_file = joblib.load(todo['model_meta_file'])
            feature_file_for_model = model_meta_file['feature_file']
            print('loading feature file')
            feature_file = joblib.load(feature_file_for_model)

            feature_extractor = feature_file['feature_extractor']
            print('extracting features')
            feature_extractor.extract(**extract_params,
                                      segment=True,
                                      make_output_subdir=False)

            os.chdir(output_dir)
            ftr_files = glob('features_created*')
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
                    if np.any(np.isnan(features)):  # if any rows have nan values for features
                        features_has_nan = True
                        # Initialize predictions vector, to later assign nan values
                        # to predictions for those rows
                        pred_labels_nan = np.full((features.shape[0],), 'nan')  # has to be same dtype as predictions
                        # Need to remove rows with nans before normalization + classification
                        features_not_nan_rows = np.where(~np.isnan(features).any(axis=1))[0]
                        features = features[features_not_nan_rows, :]
                    else:
                        features_has_nan = False
                    features_scaled = scaler.transform(features)
                    pred_labels = clf.predict(features_scaled)
                    if features_has_nan:
                        # index pred_labels into pred_labels_nan
                        # so that all nan rows will have 'nan' as prediction
                        pred_labels_nan[features_not_nan_rows] = pred_labels
                        # now make pred_labels point to ndarray with 'nan' predictions included
                        pred_labels = pred_labels_nan

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
                    if todo['predict_proba']:
                        pred_probs = clf.predict_proba(features_scaled)
                        ftr_file_dict['pred_probs'] = pred_probs
                joblib.dump(ftr_file_dict, ftr_file)

                if 'convert' in todo:
                    songfiles = ftr_file_dict['songfiles']
                    songfile_IDs = ftr_file_dict['songfile_IDs']
                    if todo['convert'] == 'notmat':
                        all_sampfreqs = ftr_file_dict['all_sampfreqs']
                        print('converting to .not.mat files')
                        for curr_song_id, songfile_name in enumerate(songfiles):
                            these = np.asarray(songfile_IDs) == curr_song_id
                            segment_params = ftr_file_dict['segment_params']
                            annotation.make_notmat(filename=songfile_name,
                                                   labels=ftr_file_dict['pred_labels'][these],
                                                   onsets_Hz=ftr_file_dict['onsets_Hz'][these],
                                                   offsets_Hz=ftr_file_dict['offsets_Hz'][these],
                                                   samp_freq=all_sampfreqs[curr_song_id],
                                                   threshold=segment_params['threshold'],
                                                   min_syl_dur=segment_params['min_syl_dur'],
                                                   min_silent_dur=segment_params['min_silent_dur'],
                                                   clf_file=model_filename,
                                                   alternate_path=output_dir)

            os.chdir(home_dir)

    elif data_dirs or annotation_file:
        if data_dirs and annotation_file:
            raise ValueError('hvc.predict received values for both data_dirs and '
                             'annotation_file arguments, unclear which to use. '
                             'Please only specify one or the other.')

        if model_meta_file is None:
            raise ValueError('model_meta_file required when as an argument when hvc.predict '
                             'is called with data_dirs or annotation_file.')

        if convert_to is not None:
            if convert_to not in valid_convert_types:
                raise ValueError('file format to convert predictions to, {}, is not a '
                                 'valid format'.format(convert_to))

        if segment is None:
            # default to True
            if data_dirs and (annotation_file is None):
                segment = True
            else:
                segment = False

        model_meta_file = joblib.load(model_meta_file)
        model_filename = model_meta_file['model_filename']
        model_name = model_meta_file['model_name']
        if predict_proba:
            if model_name not in valid_models['sklearn']:
                raise ValueError('predict_proba argument set to True, but model in {} is {}, '
                                 'which is not a valid scikit-learn model and does not have '
                                 'a predict probability function built in'.format(model_filename,
                                                                                  model))

        if output_dir is None:
            output_dir = os.getcwd()
        output_dir = os.path.abspath(output_dir)
        output_dir = os.path.join(output_dir, 'predict_output_'
                                  + hvc.utils.timestamp())
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        feature_file_for_model = model_meta_file['feature_file']
        print('loading feature file')
        feature_file = joblib.load(feature_file_for_model)

        extract_params = {
            'output_dir': output_dir,
            'labels_to_use': 'all',
            'file_format': file_format,
            'segment': segment
        }

        if annotation_file:
            extract_params['annotation_file'] = annotation_file
        elif data_dirs:
            extract_params['data_dirs'] = data_dirs

        feature_extractor = feature_file['feature_extractor']
        print('extracting features')
        feature_extractor.extract(**extract_params,
                                  make_output_subdir=False)

        os.chdir(output_dir)
        ftr_files = glob('features_created*')
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
                if np.any(np.isnan(features)):  # if any rows have nan values for features
                    features_has_nan = True
                    # Initialize predictions vector, to later assign nan values
                    # to predictions for those rows
                    pred_labels_nan = np.full((features.shape[0],), 'nan')  # has to be same dtype as predictions
                    # Need to remove rows with nans before normalization + classification
                    features_not_nan_rows = np.where(~np.isnan(features).any(axis=1))[0]
                    features = features[features_not_nan_rows, :]
                else:
                    features_has_nan = False
                features_scaled = scaler.transform(features)
                pred_labels = clf.predict(features_scaled)
                if features_has_nan:
                    # index pred_labels into pred_labels_nan
                    # so that all nan rows will have 'nan' as prediction
                    pred_labels_nan[features_not_nan_rows] = pred_labels
                    # now make pred_labels point to ndarray with 'nan' predictions included
                    pred_labels = pred_labels_nan
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

            if predict_proba:
                pred_probs = clf.predict_proba(features_scaled)
                ftr_file_dict['pred_probs'] = pred_probs
            joblib.dump(ftr_file_dict, ftr_file)

            if convert_to:
                songfiles = ftr_file_dict['songfiles']
                songfile_IDs = ftr_file_dict['songfile_IDs']
                if convert_to == 'notmat':
                    all_sampfreqs = ftr_file_dict['all_sampfreqs']
                    print('converting to .not.mat files')
                    for curr_song_id, songfile_name in enumerate(songfiles):
                        these = np.asarray(songfile_IDs) == curr_song_id
                        segment_params = ftr_file_dict['segment_params']
                        annotation.make_notmat(filename=songfile_name,
                                               labels=ftr_file_dict['pred_labels'][these],
                                               onsets_Hz=ftr_file_dict['onsets_Hz'][these],
                                               offsets_Hz=ftr_file_dict['offsets_Hz'][these],
                                               samp_freq=all_sampfreqs[curr_song_id],
                                               threshold=segment_params['threshold'],
                                               min_syl_dur=segment_params['min_syl_dur'],
                                               min_silent_dur=segment_params['min_silent_dur'],
                                               clf_file=model_filename,
                                               alternate_path=output_dir)

        if return_predictions:
            predict_dict = {}
            for ftr_file in ftr_files:
                ftrs = joblib.load(ftr_file)
                if predict_dict == {}:
                    predict_dict['labels'] = ftrs['labels']
                    predict_dict['pred_labels'] = ftrs['pred_labels']
                    predict_dict['songfile_IDs'] = ftrs['songfile_IDs']
                    predict_dict['onsets_Hz'] = ftrs['onsets_Hz']
                    predict_dict['offsets_Hz'] = ftrs['offsets_Hz']
                    predict_dict['songfiles'] = ftrs['songfiles']
                    predict_dict['feature_list'] = ftrs['feature_list']
                    predict_dict['labels_to_use'] = ftrs['labels_to_use']
                    if 'features' in ftrs:
                        predict_dict['features'] = ftrs['features']
                        predict_dict['features_arr_column_IDs'] = ftrs['features_arr_column_IDs']
                    if 'feature_group_ID_dict' in ftrs:
                        predict_dict['feature_group_ID_dict'] = ftrs['feature_group_ID_dict']
                        predict_dict['feature_list_group_ID'] = ftrs['feature_list_group_ID']
                    if 'pred_probs' in ftrs:
                        predict_dict['pred_probs'] = ftrs['pred_probs']
                    if 'neuralnet_inputs' in ftrs:
                        predict_dict['neuralnet_inputs'] = ftrs['neuralnet_inputs']
                else:  # if we already loaded one feature file and predict_dict is not empty
                    # then concatenate
                    predict_dict['labels'] = np.concatenate(predict_dict['labels'], ftrs['labels'])
                    predict_dict['pred_labels'] = np.concatenate(predict_dict['pred_labels'], ftrs['pred_labels'])
                    predict_dict['songfile_IDs'] = np.concatenate(predict_dict['songfile_IDs'], ftrs['songfile_IDs'])
                    predict_dict['onsets_Hz'] = np.concatenate(predict_dict['onsets_Hz'], ftrs['onsets_Hz'])
                    predict_dict['offsets_Hz'] = np.concatenate(predict_dict['offsets_Hz'], ftrs['offsets_Hz'])
                    if 'features' in predict_dict:
                        predict_dict['features'] = np.concatenate(predict_dict['features'], ftrs['features'])
                    if 'neuralnet_inputs' in predict_dict:
                        for key, val in ftrs['neuralnet_input']:
                            predict_dict['neuralnet_inputs'][key] = \
                                np.concatenate((predict_dict['neuralnet_inputs'][key],
                                                ftrs['neuralnet_inputs'][key]))
                    if 'pred_probs' in predict_dict:
                        predict_dict['pred_probs'] = np.concatenate(predict_dict['pred_probs'], ftrs['pred_probs'])
            os.chdir(home_dir)
            return predict_dict
        else:
            os.chdir(home_dir)
