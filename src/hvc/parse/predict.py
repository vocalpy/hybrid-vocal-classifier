"""
YAML parser for predict config files
"""

#from standard library
import os
import copy
import csv

# from dependencies
import yaml
import joblib

from .utils import check_for_missing_keys, flatten

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

with open(os.path.join(dir_path, 'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml, Loader=yaml.FullLoader)

REQUIRED_TODO_LIST_KEYS = set(validate_dict['required_predict_todo_list_keys'])
REQUIRED_TODO_LIST_KEYS_FLATTENED = set(flatten(
    validate_dict['required_predict_todo_list_keys']))
OPTIONAL_TODO_LIST_KEYS = set(validate_dict['optional_predict_todo_list_keys'])
VALID_MODELS = validate_dict['valid_models']
VALID_CONVERT_TYPES = validate_dict['valid_convert_types']
MUST_TRAIN_WITH_PROB_TRUE = validate_dict['must_train_with_prob_true']


def _validate_todo_list_dict(todo_list_dict, index, config_path):
    """
    validates to-do lists

    Parameters
    ----------
    todo_list_dict : dict
        from "to-do" list
    index : int
        index of element (i.e., dictionary) in list of dictionaries
    config_path : str
        absolute path to YAML config file from which dict was taken.
        Used to validate directory names.

    Returns
    -------
    todo_list_dict : dict
        after validation, may have new keys added if necessary
    """

    # if required_todo_list_keys is not a subset of todo_list_dict,
    # i.e., if not all required keys are in todo_list_dict
    missing_keys = check_for_missing_keys(todo_list_dict, REQUIRED_TODO_LIST_KEYS)
    if missing_keys:
        raise KeyError('The following required keys '
                       'were not found in todo_list item #{}: {}'
                       .format(index, missing_keys))
    else:
        additional_keys = set(todo_list_dict.keys()) - REQUIRED_TODO_LIST_KEYS_FLATTENED
        for extra_key in additional_keys:
            if extra_key not in OPTIONAL_TODO_LIST_KEYS:
                raise KeyError('key {} in todo_list item #{} is not recognized'
                               .format(extra_key,index))

    validated_todo_list_dict = copy.deepcopy(todo_list_dict)
    for key, val in todo_list_dict.items():
        # valid todo_list_dict keys in alphabetical order

        if key == 'annotation_file':
            with open(val, newline='') as f:
                reader = csv.reader(f, delimiter=',')
                first_row = next(reader)
                if first_row != 'filename,index,onset,offset,label'.split(','):
                    raise ValueError('annotation_file did not have correct header')

        elif key == 'bird_ID':
            if type(val) != str:
                raise ValueError('Value {} for key \'bird_ID\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))

        elif key == 'convert':
            if type(val) != str:
                raise TypeError('Specifier for `convert` in to-do list should be '
                                'a string, but parsed as a {}'.format(type(val)))
            elif val not in VALID_CONVERT_TYPES:
                raise ValueError('{} is not a valid format that predict output '
                                 'can be converted to'.format(val))

        elif key == 'data_dirs':
            if type(val) != list:
                raise TypeError('data_dirs should be a list')
            else:
                validated_data_dirs = []
                for item in val:
                    if not os.path.isdir(item):
                        # if item is not absolute path to dir
                        # try adding item to absolute path to config_file
                        # i.e. assume it is written relative to config file
                        item = os.path.join(
                            os.path.dirname(config_path),
                            os.path.normpath(item))
                        if not os.path.isdir(item):
                            raise ValueError('directory {} in {} is not a valid directory.'
                                             .format(item, key))
                    validated_data_dirs.append(item)
                validated_todo_list_dict['data_dirs'] = validated_data_dirs

        elif key == 'file_format':
            if type(val) != str:
                raise ValueError('Value {} for key \'file_format\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))
            else:
                if val not in validate_dict['valid_file_formats']:
                    raise ValueError('{} is not a known audio file format'.format(val))

        elif key == 'model_meta_file':
            if type(val) != str:
                raise ValueError('Value {} for key \'feature_file\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))
            if not os.path.isfile(os.path.normpath(val)):
                # if val is not absolute path to meta_file
                # try adding item to absolute path to config_file
                # i.e. assume path to file is written relative to config file
                val = os.path.join(
                    os.path.dirname(config_path),
                    os.path.normpath(val))
                if not os.path.isfile(val):
                    raise FileNotFoundError('{} is not found as a file'.format(val))

            # check that model file can be opened
            model_meta_file = joblib.load(val)
            model_filename = model_meta_file['model_filename']
            model_name = model_meta_file['model_name']
            if model_name in VALID_MODELS['sklearn']:
                try:
                    joblib.load(model_filename)
                except OSError:
                    raise OSError('Unable to open model file: {}'.format(model_filename))
            elif model_name in VALID_MODELS['keras']:
                try:
                    import keras.models
                    keras.models.load_model(model_filename)
                except OSError:
                    raise OSError('Unable to open model file: {}'.format(model_filename))

        elif key == 'output_dir':
            if type(val) != str:
                raise ValueError('output_dirs should be a string but it parsed as a {}'
                                 .format(type(val)))

        elif key == 'predict_proba':
            if type(val) != bool:
                raise ValueError('predict_proba should be a Boolean but it parsed as {}'
                                 .format(type(val)))

        else:  # if key is not found in list
            raise KeyError('key {} found in todo_list_dict but not validated'.
                            format(key))
    return validated_todo_list_dict


def validate_yaml(config_path, predict_config_yaml):
    """
    validates config from YAML file

    Parameters
    ----------
    config_path : str
        absolute path to YAML config file. Used to validate directory names
        in YAML files, which are assumed to be written relative to the
        location of the file itself.
    predict_config_yaml : dict
        dict should be config from YAML file as loaded with pyyaml.

    Returns
    -------
    predict_config_dict : dict
        after validation of all keys
    """

    validated_predict_config = copy.deepcopy(predict_config_yaml)
    for key, val in predict_config_yaml.items():

        if key == 'todo_list':
            if type(val) != list:
                raise TypeError('todo_list did not parse as a list, instead it parsed as {}. '
                                'Please check config file formatting.'.format(type(val)))
            else:
                for index, item in enumerate(val):
                    if type(item) != dict:
                        raise TypeError('item {} in todo_list did not parse as a dictionary, '
                                        'instead it parsed as a {}. Please check config file'
                                        ' formatting'.format(index, type(item)))
                    else:
                        val[index] = _validate_todo_list_dict(item, index, config_path)

            # make sure that if predict_proba is True, that the model
            # was trained with predict_proba set to True.
            # Need to do this *after* already validating all model_meta_file keys
            for item in val:  # where each item is a todo_list_dict
                if 'predict_proba' in item:
                    if item['predict_proba']:  # if it is True, then
                        # make sure model was trained with predict_proba set to True
                        model_meta_file = item['model_meta_file']
                        if not os.path.isfile(os.path.normpath(model_meta_file)):
                            # if val is not absolute path to meta_file
                            # try adding item to absolute path to config_file
                            # i.e. assume path to file is written relative to config file
                            model_meta_file = os.path.join(
                                os.path.dirname(config_path),
                                os.path.normpath(model_meta_file))
                        model_meta_file = joblib.load(model_meta_file)
                        model_name = model_meta_file['model_name']
                        if model_name in MUST_TRAIN_WITH_PROB_TRUE:
                            # if model not in MUST_TRAIN_WITH_PROB_TRUE
                            # then we get probabilities for free with the model
                            # as implemented, e.g. kNeighborsClassifier
                            # from scikit-learn, and any neural net that
                            # has a softmax layer as the output
                            model_filename = model_meta_file['model_filename']
                            model = joblib.load(model_filename)
                            if not model.probability:
                                raise AttributeError('predict_proba in config file is set to True, '
                                                     'but model was not trained with predict_proba '
                                                     'set to True.\n'
                                                     'config file is: {}\n'
                                                     'model meta file is: {}\n'
                                                     'model file is: {}'
                                                     .format(config_path,
                                                             item['model_meta_file'],
                                                             model_filename))

            validated_predict_config['todo_list'] = val

        else:  # if key is not found in list
            raise KeyError('key {} in \'predict\' is an invalid key'.
                           format(key))

    return validated_predict_config
