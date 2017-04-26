#from standard library
import os

#from dependencies
import yaml

def _convert_feature_group_to_list(feature_group):
    """
    
    """
    #do stuff
    return feature_list

def _validate(extract_dict):
    for key, val in dict_to_validate.items():
        #valid keys, listed in alphabetical order
        if key = 'feature_group':
            pass

        elif key == 'spect_params':
        if type(val) != dict:
            raise ValueError('value for key \'spect_params\' in config file did '
                             'not parse as a dictionary of parameters. Check '
                             'file formatting.')
        spect_param_keys = set(['samp_freq',
                                'window_size',
                                'window_step',
                                'freq_cutoffs'])
        if set(val.keys()) != spect_param_keys:
            raise KeyError('unrecognized keys in spect_param dictionary')

def parse_extract_config(extract_config_file):
    """
    """
    with open(config_file) as yaml_to_parse:
        extract_config_yaml = yaml.load(yaml_to_parse)
        extract_config = _validate(extract_config_yaml)
    return extract_config