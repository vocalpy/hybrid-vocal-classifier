#from standard library
import os

#from dependencies
import yaml

#from hvc
from . import parse

parser_dict = {
    'extract' : parse.extract.validate_yaml,
    'select' : parse.select.validate_yaml,
}

def _parse_helper(config_type,config_file,config_yaml):
    """
    helper function to avoid repeating code
    
    Parameters
    ----------
    config_type : string
        as defined in parse_config
    config_file : string
        filename of YAML file
    config_yaml : dictionary
        parsed YAML file

    Returns
    -------
    validated dictionary
    """

    if config_type not in ['extract', 'select', 'predict']:
        raise ValueError('{} in {} is not a valid config_type. '
                         'Valid types are: \'extract\', \'select\', or \'predict\'.'
                          .format(config_type,config_file))

    if config_type not in config_yaml:
        raise KeyError('\'{}\' not defined in config file {}'.format(config_type,config_file))
    else:
        return parser_dict[config_type](config_yaml[config_type])

def parse_config(config_file,config_type=None):
    """
    
    Parameters
    ----------
    config_file : string
        filename of YAML file
    config_type : string
        valid strings are 'extract','select', and 'predict'
        if one of those strings is supplied, the matching key is found in the
        config file and only that configuration is parsed and returned.
        The extract, select, and predict modules make use of this functionality. 
        Default is None, in which case entire config file is parsed and returned

    Returns
    -------
        config : dict
            validated dictionary from parsed YAML file
    """

    with open(config_file) as yaml_to_parse:
        config_yaml = yaml.load(yaml_to_parse)

    if config_type is not None:
        return _parse_helper(config_type,config_file,config_yaml)

    elif config_type is None:
        config_dict = {}
        config_types = list(config_yaml.keys())
        for config_type in config_types:
            config_dict[config_type] = _parse_helper(config_type,config_file,config_yaml)
        return config_dict