"""
YAML parser for select config files
"""

#from standard library
import os
import copy

#from dependencies
import numpy as np
from sklearn.externals import joblib

VALID_MODELS = set(['knn','svm','neuralnet'])

def _validate_model_list(model_list):
    """
    validates 'model' list that can appear in 'global' or in 'todo_list'
    
    Parameters
    ----------
    model_list : list of dictionaries
        each dictionary specifies:
            'model' : string
                a machine-learning model/algorithm
                currently valid: 'knn', 'svm', 'neuralnet'
            'hyperparameters' : dictionary
                parameters for "training" the model
            'feature_indices' : list of integers
                features to use from an already generated feature array

    Returns
    -------
    validated_model_list : list
        after validation
    """

    if type(model_list) != list:
        raise ValueError('\'models\' should be a list not a {}'
                         .format(type(model_list)))
    if not all([type(el) is dict for el in model_list]):
        raise ValueError('all items in \'models\' should be dictionaries')
    model_set = set([model_dict['model'] for model_dict in model_list])
    if not model_set.issubset(VALID_MODELS):
        invalid_models = list(model_set - VALID_MODELS)
        raise ValueError('{} in \'models\' are not valid model types'.format(invalid_models))

    validated_model_list = copy.deepcopy(model_list)

    for ind, model_dict in enumerate(model_list):
        validated_model_dict = copy.deepcopy(model_dict)
        for model_key, model_val in model_dict.items():
            if model_key == 'feature_indices':
                if type(model_val) != list and type(model_val) != str:
                    raise ValueError('\'feature_indices\' should be a list or string but parsed as a {}'
                                     .format(type(model_val)))
                if type(model_val) == str:
                    try:
                        model_val = [int(num) for num in model_val.split(',')]
                    except ValueError:
                        raise ValueError('feature_indices parsed as a string '
                                         'but could not convert following to list of ints: {}'
                                         .format(model_val))
                if not all([type(item_val) is int for item_val in model_val]):
                    raise ValueError('all indices in \'feature_indices\' should be integers')
                validated_model_dict[model_key] = model_val
            elif model_key == 'model':
                if model_val == 'knn':
                    if set(model_dict['hyperparameters'].keys()) != set(['k']):
                        raise KeyError('invalid keys in \'knn\' hyperparameters')
                    if type(model_dict['hyperparameters']['k']) != int:
                        raise ValueError('value for \'k\' should be an integer')
                elif model_val == 'svm':
                    if set(model_dict['hyperparameters'].keys()) != set(['C','gamma']):
                        raise KeyError('invalid keys in \'svm\' hyperparameters')
                    C = model_dict['hyperparameters']['C']
                    if  type(C) != float and type(C) != int:
                        raise ValueError('C value for svm should be float or int')
                    gamma = model_dict['hyperparameters']['gamma']
                    if  type(gamma) != float and type(gamma) != int:
                        raise ValueError('gamma value for svm should be float or int')
        validated_model_list[ind] = validated_model_dict
    return validated_model_list

VALID_NUM_SAMPLES_KEYS = set(['start','stop','step'])
REQUIRED_TODO_KEYS = set(['feature_file','output_dir'])
OPTIONAL_TODO_KEYS = set(['num_test_samples','num_train_samples','num_replicates','models'])

def _validate_todo_list_dict(todo_list_dict, index):
    """
    validates to-do lists

    Parameters
    ----------
    todo_list_dict : dictionary from "to-do" list
    index : index of element (i.e., dictionary) in list of dictionaries

    Returns
    -------
    todo_list_dict : dictionary after validation, may have new keys added if necessary
    """

    # if required_todo_list_keys is not a subset of todo_list_dict,
    # i.e., if not all required keys are in todo_list_dict
    todo_keys_set = set(todo_list_dict.keys())
    if not todo_keys_set >= REQUIRED_TODO_KEYS:
        raise KeyError('Not all required keys in todo_list item #{}.'
                       'Missing keys: {}'.format(index,
                                                 todo_keys_set - REQUIRED_TODO_KEYS))
    else:
        additional_keys = todo_keys_set - REQUIRED_TODO_KEYS
        if additional_keys:
            if not additional_keys <= OPTIONAL_TODO_KEYS:
                raise KeyError('keys in todo_list item #{} is not recognized: {}'
                               .format(additional_keys - OPTIONAL_TODO_KEYS))

    validated_todo_list_dict = copy.deepcopy(todo_list_dict)
    for key, val in todo_list_dict.items():
        # valid todo_list_dict keys in alphabetical order
        if key == 'feature_file':
            if type(val) != str:
                raise ValueError('Value {} for key \'feature_file\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))
            if not os.path.isfile(val):
                raise OSError('{} is not found as a file'.format(val))
            try:
                joblib.load(val)
            except:
                raise IOError('Unable to open {}'.format(val))

        elif key == 'models':
            validated_todo_list_dict['models'] = _validate_model_list(val)

        elif key == 'num_replicates':
            if type(val) != int:
                raise ValueError('{} should be an int but parsed as {}'
                                 .format(key, type(val)))

        elif key == 'num_test_samples':
            if type(val) != int:
                raise ValueError('{} in \'global\' should be an integer'.format(key))

        elif key == 'num_train_samples':
            if type(val) != dict:
                raise ValueError('{} should be a dict but parsed as {}'
                                 .format(key, type(val)))
            else:
                samples_keys = set(['start', 'stop', 'step'])
                if set(glob_val.keys()) != samples_keys:
                    raise KeyError('incorrect keys in {}'.format(glob_key))
                else:
                    num_samples = range(glob_val['start'],
                                        glob_val['stop'],
                                        glob_val['step'])
                    validated_select_config['num_train_samples'] = num_samples

        elif key == 'output_dir':
            if type(val) != str:
                raise ValueError('output_dirs should be a string but it parsed as a {}'
                                 .format(type(val)))

        else:  # if key is not found in list
            raise KeyError('key {} in todo_list_dict is an invalid key'.
                           format(key))
    return validated_todo_list_dict

VALID_SELECT_KEYS = set(['global','todo_list'])
VALID_GLOBAL_KEYS = set(['num_replicates',
                         'num_test_samples',
                         'num_train_samples',
                         'models'])

def validate_yaml(select_config_yaml):
    """
    validates config from YAML file

    Parameters
    ----------
    select_config_yaml : dictionary, config as loaded with YAML module

    Returns
    -------
    select_config_dict : dictionary, after validation of all keys
    """

    for global_key in VALID_GLOBAL_KEYS:
        if not all([global_key in todo for todo in select_config_yaml['todo_list']]):
            if global_key not in select_config_yaml['global']:
                raise KeyError('\'{0}\' not defined for every item in todo_list, '
                               'but no global {0} is defined. You must either '
                               'define \'{0}\' in the \'global\' dictionary '
                               '(that any \'{0}\' in a todo_list item will take '
                               'precedence over) or you must define \'{0}\' for'
                               ' every item in the todo_list.'.format(global_key))

    validated_select_config = copy.deepcopy(select_config_yaml)
    for key, val in select_config_yaml.items():
        if key == 'global':
            if type(val) != dict:
                raise ValueError('value for key \'global\' in config file did '
                                 'not parse as a dictionary of. Check '
                                 'file formatting.')
            global_keys = set(val.keys())
            if not global_keys <= VALID_GLOBAL_KEYS:
                raise KeyError('unrecognized keys in global_params dictionary: {}'
                               .format(list(global_keys - VALID_GLOBAL_KEYS)))
            else:
                for global_key, global_val in val.items():
                    if global_key == 'models':
                        validated_select_config['global']['models'] = _validate_model_list(global_val)

                    elif global_key == 'num_replicates':
                        if type(global_val) != int:
                            raise ValueError('{} in \'global\' should be an integer'.format(global_key))

                    elif global_key == 'num_test_samples':
                        if type(global_val) != int:
                            raise ValueError('{} in \'global\' should be an integer'.format(global_key))

                    elif global_key == 'num_train_samples':
                        if type(global_val) != dict:
                            raise ValueError('\'num_train_samples\' did not parse as dict. Please check formatting')
                        samples_key_set = set(global_val.keys())
                        if samples_key_set != VALID_NUM_SAMPLES_KEYS:
                            raise KeyError('\'num_samples\' contains invalid keys {}, '
                                           'should only contain the following keys: '
                                           '{}'.format(samples_key_set - VALID_NUM_SAMPLES_KEYS,
                                                       VALID_NUM_SAMPLES_KEYS))
                        for samples_key in samples_key_set:
                            if type(global_val[samples_key]) != int:
                                raise ValueError('value for \'{}\' in \'num_samples\' should be type int, not {}'
                                                 .format(samples_key,
                                                         global_key,
                                                         type(global_val[samples_key])))
                            if samples_key == 'stop':
                                if global_val['stop'] < global_val['start']:
                                    raise ValueError('stop value is {} but should be greater than start value, {}'
                                                     .format(global_val['stop'],
                                                             global_val['start']))
                        num_samples_vals = range(global_val['start'],
                                                 global_val['stop'],
                                                 global_val['step'])
                        validated_select_config['global']['num_train_samples'] = num_samples_vals

                    else:
                        raise KeyError('invalid key {} found in \'global\''.format(global_key))

        elif key == 'todo_list':
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
                        val[index] = _validate_todo_list_dict(item, index)
            validated_select_config['todo_list'] = val  # re-assign because feature list is added

        else:  # if key is not found in list
            raise KeyError('key {} in \'select\' is an invalid key'.
                           format(key))

    return validated_select_config