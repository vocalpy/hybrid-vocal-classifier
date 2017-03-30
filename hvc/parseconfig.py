import yaml

def _validate(key,val):
    """
    function called by parseconfig to validate values of keys
    in config dictionaries
    """

    #valid keys are listed in alphabetical order
    if key==
    

    elif key=='knn':
    

    elif key=='labelset':
    

    elif key=='neural_net':
    

    elif key=='num_replicates':

    elif key=='num_songs':

    elif key=='spect_params':
        if type(val) != dict:
            raise ValueError('value for key \'spect_params\' in config file did'
                             'not parse as a dictionary of parameters. Check '
                             'file formatting.')
        spect_param_keys = set(['samp_freq',
                                'window_size',
                                'window_step',
                                'freq_cutoffs'])
        if set(val.keys()) != spect_param_keys:
            raise KeyError('unrecognized keys in spect_param dictionary')
    

    elif key=='svm':
    

    elif key=='test':
    
    elif key=='train':
    


    else:
        raise KeyError('key {] in config_file {} is an invalid key'.
                        format(key,config_file))

def parseconfig(config_file):
    """
    parses YAML configuration files.
    This includes error checking.
    """

    with open(config_file) as yaml_to_parse:
        config_dict = yaml.load(yaml_to_parse)

    # make sure both 'global_config' and 'jobs' are defined
    try:
        global_config = config_dict['global_config']
    except KeyError:
        raise KeyError('global_config was not defined in config file')
    try:
        jobs = config_dict['jobs']
    except:
        raise KeyError('jobs was not defined in config file')

    for key, val in global_config.items()
        _validate(key,val)

    for job in jobs:
        for key, val in job.items()
            _validate(key,val)

    return config_dict
