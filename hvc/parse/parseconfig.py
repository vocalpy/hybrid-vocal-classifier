#from standard library
import os

#from dependencies
import yaml

def _validate(dict_to_validate, config_file):
    """
    function called by parse to validate values of keys
    in config dictionaries
    """

    for key, val in dict_to_validate.items():
        #valid keys, listed in alphabetical order
        if key=='bird_ID':
            if type(val) != str:
                raise ValueError('Value {} for key \'bird_ID\' is type {} but it'
                                 ' should be a string'.format(val,type(val)))
            else:
                pass

        elif key=='jobs':
            if type(val) != list:
                raise ValueError('Jobs should be a list but instead it was a {}'.
                                 format(type(val)))
            else:
                pass

        elif key=='labelset':
            if type(val) != str:
                raise ValueError('Labelset should be a string, e.g., \'iabcde\'.')
            else:
                label_list = list(val)
                label_list = [ord(label) for label in label_list]
                dict_to_validate['label_list'] = label_list

        elif key=='models':
            if type(val) !=list:
                raise ValueError('models should be a list of model names.')
            else:
                valid_model_names = ['knn','svm','neural_net']
                for name in val:
                    if name not in valid_model_names:
                        raise ValueError('{} in models list is not a valid model name'.format(name))

        elif key=='num_replicates':
            if type(val) != int:
                raise ValueError('num_replicates should be an int')
            else:
                num_replicates = range(val)
                dict_to_validate['num_replicates'] = num_replicates

        elif key=='num_samples':
            if type(val) != dict:
                raise ValueError('num_samples should be an int')
            else:
                num_samples = range(val['start'],
                                    val['stop'],
                                    val['step'])
                dict_to_validate['num_samples'] = num_samples

        elif key=='num_songs':
            if type(val) != int:
                raise ValueError('num_replicates should be an int')
            else:
                return

        elif key=='spect_params':
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

        elif key=='syl_spect_width':
            if type(val) != int:
                raise ValueError('syl_spect_width should be an int')
            else:
                pass

        elif key=='test' or key=='train':
            if type(val) != list:
                raise ValueError('Value for key \'train\' should be a list of '
                                 'directories but instead it is of type {}'.format(
                                 type(val)))
            else:
                for dir_name in val:
                    if not os.path.isdir(dir_name):
                        raise NotADirectoryError('{} in directory list for \'{}\' '
                                                 'key is not a valid directory.'
                                                 .format(dir_name,key))

        else: # if key is not found in list
            raise KeyError('key {} in config file {} is an invalid key'.
                            format(key,config_file))
    return dict_to_validate

def parse(config_file):
    """
    parses YAML configuration files.
    This includes error checking.
    """

    with open(config_file) as yaml_to_parse:
        config_dict = yaml.load(yaml_to_parse)

    # make sure 'global_config' is defined, then validate
    if 'global_config' not in config_dict:
        raise KeyError('global_config was not defined in config file')
    else:
        config_dict['global_config'] = _validate(config_dict['global_config'], config_file)

    # make sure either model_selection or prediction is defined
    if 'model_selection' not in config_dict and 'prediction' not in config_dict:
        raise KeyError('Neither `model_selection` or `prediction` was defined in config file')

    if 'model_selection' in config_dict:
        config_dict['model_selection'] = _validate(config_dict['model_selection'], config_file)

    if 'prediction' in config_dict:
        config_dict['prediction'] = _validate(config_dict['prediction'], config_file)

    return config_dict