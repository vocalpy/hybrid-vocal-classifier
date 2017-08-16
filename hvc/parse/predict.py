"""
YAML parser for predict config files
"""

#from standard library
import os
import copy

# from dependencies
import yaml
from sklearn.externals import joblib

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

with open(os.path.join(dir_path, 'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml)

REQUIRED_TODO_LIST_KEYS = set(['model_file','file_format','data_dirs'])
OPTIONAL_TODO_LIST_KEYS = set(['bird_ID'])

def _validate_todo_list_dict(todo_list_dict,index):
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
    if not set(todo_list_dict.keys()) >= REQUIRED_TODO_LIST_KEYS:
        raise KeyError('not all required keys in todo_list item #{}'.format(index))
    else:
        additional_keys = set(todo_list_dict.keys()) - REQUIRED_TODO_LIST_KEYS
        for extra_key in additional_keys:
            if extra_key not in OPTIONAL_TODO_LIST_KEYS:
                raise KeyError('key {} in todo_list item #{} is not recognized'
                               .format(extra_key,index))

    validated_todo_list_dict = copy.deepcopy(todo_list_dict)
    for key, val in todo_list_dict.items():
        # valid todo_list_dict keys in alphabetical order

        if key == 'bird_ID':
            if type(val) != str:
                raise ValueError('Value {} for key \'bird_ID\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))

        elif key == 'data_dirs':
            if type(val) != list:
                raise ValueError('data_dirs should be a list')
            else:
                for item in val:
                    if not os.path.isdir(item):
                        raise ValueError('directory {} in {} is not a valid directory.'
                                         .format(item,key))

        elif key == 'file_format':
            if type(val) != str:
                raise ValueError('Value {} for key \'file_format\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))
            else:
                if val not in validate_dict['valid_file_formats']:
                    raise ValueError('{} is not a known audio file format'.format(val))

        elif key == 'model_file':
            if type(val) != str:
                raise ValueError('Value {} for key \'feature_file\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))
            if not os.path.isfile(val):
                raise OSError('{} is not found as a file'.format(val))
            try:
                joblib.load(val)
            except:
                raise IOError('Unable to open {}'.format(val))

        else:  # if key is not found in list
            raise KeyError('key {} in todo_list_dict is an invalid key'.
                            format(key))
    return validated_todo_list_dict

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

    validated_select_config = copy.deepcopy(select_config_yaml)
    for key, val in select_config_yaml.items():

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
                        val[index] = _validate_todo_list_dict(item, index)
            validated_select_config['todo_list'] = val

        else:  # if key is not found in list
            raise KeyError('key {} in \'select\' is an invalid key'.
                           format(key))

    return validated_select_config