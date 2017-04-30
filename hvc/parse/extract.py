#from standard library
import os

#from dependencies
import yaml

with open('./validation.yml') as val_yaml:
    validate_dict = yaml.load(val_yaml)

with open('./feature_groups.yml') as ftr_grp_yaml:
    feature_groups_dict = yaml.load(ftr_grp_yaml)

def _convert_feature_group_to_list(feature_group):
    """
    
    Parameters
    ----------
    feature_group

    Returns
    -------
    feature_list
    """
    return feature_groups_dict[feature_group]

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
    for key, val in todo_list_dict.items():
        # valid todo_list_dict keys in alphabetical order
        if key == 'bird_ID':
            if type(val) != str:
                raise ValueError('Value {} for key \'bird_ID\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))

        if key=='data_dirs':
            if val != list:
                raise ValueError('data_dirs should be a list')
            else for item in val:
                if not os.path.isdir(item):
                    raise ValueError('directory {} in {} is not a valid directory'.)

        elif key == 'feature_group':
            if type(val) != str and type(val) != list:
                raise TypeError('feature_group parsed as {} but it should be'
                                ' either a string or a list. Please check config'
                                ' file formatting.'.format(type(val)))
            elif type(val) == str:
                if val not in feature_groups_dict:
                    raise ValueError('{} not found in valid feature groups'.format(val))
                else:
                    todo_list_dict['feature_list'] = feature_groups_dict[val]
            elif type(val) == list:
                # if more than one feature group, than return a list of lists
                feature_list_lists = []
                for ftr_grp in val:
                    if ftr_grp not in feature_groups_dict:
                        raise ValueError('{} not found in valid feature groups'.format(val))
                    else:
                        feature_list_lists.append(feature_groups_dict[ftr_grp])
                todo_list_dict['feature_list'] = feature_list_lists

        elif key== 'feature_list':
            if type(val) != list:
                raise ValueError('feature_list should be a list but parsed as a {}'.format(type(val)))
            else:
                for feature in val:
                    if feature not in validate_dict['valid_features']:
                        raise ValueError('feature {} not found in valid feature list'.format(feature))

        elif key=='file_format':
            if type(val) != str:
                raise ValueError('Value {} for key \'file_format\' is type {} but it'
                                 ' should be a string'.format(val, type(val)))
            else:
                if val not in validate_dict['valid_file_formats']:
                    raise ValueError('{} is not a known audio file format'.format(val))

        elif key=='labelset':
            if type(val) != str:
                raise ValueError('Labelset should be a string, e.g., \'iabcde\'.')
            else:
                label_list = list(val)
                label_list = [ord(label) for label in label_list]
                dict_to_validate['label_list'] = label_list

        else: # if key is not found in list
            raise KeyError('key {} in todo_list_dict is an invalid key'.
                            format(key))
    return todo_list_dict

def _validate_extract_config(extract_config_yaml):
    """
    validates config from extract YAML file
    
    Parameters
    ----------
    extract_config_yaml : dictionary, config as loaded with YAML module

    Returns
    -------
    extract_config_dict : dictionary, after validation of all keys
    """
    for key, val in extract_config_yaml.items():
        if key == 'spect_params':
            if type(val) != dict:
                raise ValueError('value for key \'spect_params\' in config file did '
                                 'not parse as a dictionary of parameters. Check '
                                 'file formatting.')
            spect_param_keys = set(['samp_freq',
                                    'window_size',
                                    'window_step',
                                    'freq_cutoffs'])
            if set(val.keys()) != spect_param_keys:
                raise KeyError('unrecognized keys in spect_params dictionary')
            else:
                for sp_key, sp_val in val.items():
                    if sp_key=='samp_freq' or sp_key=='window_size' or sp_key=='window_step':
                        if sp_val != type(int):
                            raise ValueError('{} in spect_params should be an integer'.format(sp_key))
                    elif sp_key=='freq_cutoffs':
                        if len(sp_val) != 2:
                            raise ValueError('freq_cutoffs should be a 2 item list')
                        for freq_cutoff in sp_val:
                            if type(freq_cutoff) != int:
                                raise ValueError('freq_cutoff {} should be an int'.format(sp_val))
        elif key=='todo_list':
            if type(val) != list:
                raise TypeError('todo_list did not parse as a list, instead it parsed as {}.'
                                ' Please check config file formatting.'.format(type(val)))
            else:
                for index, item in enumerate(val):
                    if type(item) != dict:
                        raise TypeError('item {} in todo_list did not parse as a dictionary, '
                                        'instead it parsed as a {}. Please check config file'
                                        ' formatting'.format(index, type(item)))
                    else:
                        val[index] = _validate_todo_list_dict(item,index)
            extract_config_yaml['todo_list'] = val # re-assign because feature list is added
    return extract_config_yaml

def parse_extract_config(extract_config_file):
    """
    
    Parameters
    ----------
    extract_config_file

    Returns
    -------

    """
    with open(config_file) as yaml_to_parse:
        extract_config_yaml = yaml.load(yaml_to_parse)
        extract_config = _validate_extract_config(extract_config_yaml)
    return extract_config