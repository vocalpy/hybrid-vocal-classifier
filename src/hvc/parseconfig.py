#from standard library
import os

#from dependencies
import yaml

#from hvc
from . import parse

parser_dict = {
    'extract': parse.extract.validate_yaml,
    'select': parse.select.validate_yaml,
    'predict': parse.predict.validate_yaml
}


def _parse_helper(config_type, config_path, config_yaml):
    """
    helper function to avoid repeating code
    
    Parameters
    ----------
    config_type : string
        as defined in parse_config
    config_path : string
        absolute path to config_file
    config_yaml : dictionary
        parsed YAML file

    Returns
    -------
    validated dictionary
    """

    if config_type not in ['extract', 'select', 'predict']:
        raise ValueError('{} in {} is not a valid config_type. '
                         'Valid types are: \'extract\', \'select\', or \'predict\'.'
                         .format(config_type, config_path))

    if config_type not in config_yaml:
        raise KeyError('\'{}\' not defined in config file {}'.format(config_type, config_path))
    else:
        return parser_dict[config_type](config_path, config_yaml[config_type])


def parse_config(config_file, config_type=None):
    """Parse configurations in YAML file.
    Each configuration type must be defined as a dictionary with a
    
    Parameters
    ----------
    config_file : str
        filename of YAML file
    config_type : str
        {'extract','select','predict'}
        if one of those strings is supplied, the matching key is found in the
        config file and only that configuration is parsed and returned.
        The extract, select, and predict modules make use of this functionality. 
        Default is None, in which case entire config file is parsed and returned.
        Raises KeyError if no dictionary is defined with name that is a valid
        config_type.

    Returns
    -------
        config : dict
            validated dictionary from parsed YAML file
    """

    config_path = os.path.abspath(config_file)

    with open(config_path) as yaml_to_parse:
        config_yaml = yaml.load(yaml_to_parse, Loader=yaml.FullLoader)

    if config_type is not None:
        return _parse_helper(config_type, config_path, config_yaml)

    elif config_type is None:
        config_dict = {}
        config_types = list(config_yaml.keys())
        if not set(config_types.keys()).issubset(parser_dict.keys()):
            invalid_keys = set(config_types.keys()) - set(parser_dict.keys())
            raise KeyError('Invalid config keys in file \'{0}\': {1}'
                           .format(config_path, invalid_keys))
        for config_type in config_types:
            config_dict[config_type] = _parse_helper(config_type, config_path, config_yaml)
        return config_dict
