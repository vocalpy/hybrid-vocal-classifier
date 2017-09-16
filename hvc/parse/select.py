"""
YAML parser for select config files
"""

# from standard library
import os
import copy

# from dependencies
import yaml
import numpy as np
from sklearn.externals import joblib

path = os.path.abspath(__file__)  # get the path of this file
dir_path = os.path.dirname(path)  # but then just take the dir

with open(os.path.join(dir_path, 'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml)

VALID_MODEL_KEYS = validate_dict['valid_model_keys']
VALID_HYPERPARAMS = validate_dict['hyperparameters']

# valid_models is a dict, iterate to concatenate all model names
# keys are model subtypes, i.e. sklearn and keras
# values are lists of valid model names
VALID_MODELS = []
for model_subtype in validate_dict['valid_models'].values():
    # iterate over list to append
    for model_name in model_subtype:
        VALID_MODELS.append(model_name)
VALID_MODELS = set(VALID_MODELS)

with open(os.path.join(dir_path, 'feature_groups.yml')) as ftr_grp_yaml:
    valid_feature_groups_dict = yaml.load(ftr_grp_yaml)
VALID_FEATURE_GROUPS = set(valid_feature_groups_dict.keys())


def _validate_model_dict(model_dict,
                         index,
                         ftr_grp_ID_dict=None,
                         ftr_list_group_ID=None):
    """validates model dictionaries from 'models' list

    Parameters
    ----------
    model_dict : dict
    index : int
    ftr_grp_ID_dict : dict
        from feature file. validate_yaml checks whether it is
        defined in feature_file and if so passes as an argument.
        Default is None.
    ftr_list_group_ID : numpy 1-d vector
        from feature file. validate_yaml checks whether it is
        defined in feature_file and if so passes as an argument.
        Default is None.

    Returns
    -------
    validated_model_dict : dict
    """

    # no need to validate value of 'model' key
    # because that was done in _validate_models function
    # before calling _validate_model_dict.
    # so now check all the other keys + values
    model_dict_keys = set(model_dict.keys()) - {'model'}
    # get list of valid keys by using model name as key for VALID_MODEL_KEYS dict
    valid_keys_for_this_model = set(VALID_MODEL_KEYS[model_dict['model']])
    if not model_dict_keys.issubset(valid_keys_for_this_model):
        invalid_keys = model_dict_keys - valid_keys_for_this_model
        raise KeyError('Found invalid keys in item {0} from '
                       'list of model dictionaries: {1}'
                       .format(index, invalid_keys))

    validated_model_dict = copy.deepcopy(model_dict)

    # if this is a model that uses features extracted from syllables
    # (i.e. scalar / vector, not spectrograms)
    # need to validate feature groups, etc.
    # For neural net models, don't need to do this
    if 'feature_group' and 'feature_list_indices' in \
            validate_dict['valid_model_keys'][model_dict['model']]:
        # throw an error if both feature_list_indices and
        # feature_group are defined as keys for model dict
        if 'feature_list_indices' in model_dict \
                and 'feature_group' in model_dict:
            raise KeyError('both feature_list_indices and'
                           'feature_group defined in model_dict, not'
                           'clear which to use.')

        # throw an error if both feature_list_indices and
        # feature_group are defined as keys for model dict
        if 'feature_list_indices' not in model_dict \
                and 'feature_group' not in model_dict:
            raise KeyError('Neither feature_list_indices or '
                           'feature_group are defined in model_dict,'
                           'at least one of them must be defined.')

        if 'feature_list_indices' in model_dict:
            ftr_list_inds = model_dict['feature_list_indices']
            if type(ftr_list_inds) != list and type(ftr_list_inds) != str:
                raise ValueError('\'feature_list_indices\' should be a list or string but parsed as a {}'
                                 .format(type(ftr_list_inds)))
            if type(ftr_list_inds) == str:
                if ftr_list_inds == 'all':
                    pass  # just keep as 'all'
                else:
                    try:
                        ftr_list_inds = [int(num) for num in model_val.split(',')]
                    except ValueError:
                        raise ValueError('feature_list_indices parsed as a string '
                                         'but could not convert following to list of ints: {}'
                                         .format(ftr_list_inds))

            if ftr_list_inds != 'all':
                if not all([type(item_val) is int for item_val in ftr_list_inds]):
                    raise ValueError('all indices in \'feature_list_indices\''
                                     ' should be integers')
            validated_model_dict['feature_list_indices'] = ftr_list_inds

        if 'feature_group' in model_dict:
            ftr_grp = model_dict['feature_group']
            # feature group should work as a string or as a list of strings
            if type(ftr_grp) != str and type(ftr_grp) != list:
                raise ValueError('value for feature_group should'
                                 'be string but parsed as {}'
                                 .format(type(ftr_grp)))

            if type(ftr_grp) == str:
                if ftr_grp not in VALID_FEATURE_GROUPS:
                    raise ValueError('{} is not a valid feature group.'
                                     .format(ftr_grp))
                # get appropriate ID # out of ftr_grp_ID_dict for this model
                # (if called from todo_list so we hve ftr_list_group_ID/dict)
                if ftr_list_group_ID is not None and ftr_grp_ID_dict is not None:
                    ftr_grp_ID = ftr_grp_ID_dict[model_dict['model']]
                    # now find all the indices of features associated with the
                    # feature group for that model
                    ftr_inds = np.where(
                        np.in1d(ftr_list_group_ID, ftr_grp_ID))[0]  # returns tuple
                else:
                    ftr_inds = None

                if ftr_inds is not None:
                    validated_model_dict['feature_list_indices'] = ftr_inds

            elif type(ftr_grp) == list:
                if not all([type(item) == str for item in ftr_grp]):
                    raise ValueError('Not all items in feature_group list are strings.')
                ftr_inds = []
                if not all([model_name in VALID_FEATURE_GROUPS
                            for model_name in ftr_grp]):
                    raise ValueError('{} is not a valid feature group.'
                                     .format(ftr_grp))
                # if called from todo_list so we hve ftr_list_group_ID/dict
                if ftr_list_group_ID is not None and ftr_grp_ID_dict is not None:
                    for model_name in ftr_grp:
                        # get appropriate ID
                        # out of ftr_grp_ID_dict for this model
                        ftr_grp_ID = ftr_grp_ID_dict[model_name]
                        # now find all the indices of features associated with the
                        # feature group for that model
                        ftr_inds_this_model = [ind for ind, val in
                                               enumerate(ftr_list_group_ID)
                                               if val == ftr_grp_ID]  # returns tuple
                        ftr_inds.append(ftr_inds_this_model)
                    ftr_inds = np.concatenate(ftr_inds)
                else:
                    ftr_inds = None
                if ftr_inds is not None:
                    validated_model_dict['feature_list_indices'] = ftr_inds

        hyperparams = model_dict['hyperparameters']
        required_hyperparams = set(VALID_HYPERPARAMS[model_dict['model']].keys())
        model_dict_hyperparams = set(hyperparams.keys())
        # if not all keys, i.e. model dict hyperparams is a subset of required
        if model_dict_hyperparams < required_hyperparams:
            missing_keys = required_hyperparams - model_dict_hyperparams
            raise KeyError('missing hyperparameters from model dict for {0}: {1}'
                           .format(model_dict['model'], missing_keys))
        # OTOH if extra keys , i.e. required is actually a subset of model dict hyperparams
        if model_dict_hyperparams > required_hyperparams:
            extra_keys = model_dict_hyperparams - required_hyperparams
            raise ValueError('invalid hyperparameters for model for {0}: {1}'
                             .format(model_dict['model'], extra_keys))

        # for validation,
        # replace `required hyperparams` set defined above
        # with `required hyperparams` dict
        # that has param names as keys and valid types as values
        # if more than one valid type, then it's a tuple
        required_hyperparams = VALID_HYPERPARAMS[model_dict['model']]
        for hyperparam_name, hyperparam_val in hyperparams.items():
            valid_type = required_hyperparams[hyperparam_name]
            if type(valid_type) != tuple:
                # wrap single type in tuple to be able to check if actual
                # value type is 'in' valid type(s)
                valid_type = (valid_type,)
            # have to use str representation of object, because that's all you
            # can load from YAML file, instead of comparing directly with type
            # itself
            if type(hyperparam_val).__name__ not in valid_type:
                raise ValueError('Type for hyperparameter {0} for a {1}'
                                 ' model should be {2} but parsed as {3}.'
                                 .format(hyperparam_name,
                                         model_dict['model'],
                                         valid_type,
                                         type(hyperparam_val)))

    return validated_model_dict


def _validate_models(models,
                     ftr_grp_ID_dict=None,
                     ftr_list_group_ID=None):
    """
    validates 'models' list that can appear in 'select' dictionary
    or in 'todo_list'
    
    Parameters
    ----------
    models : list of dictionaries
        each dictionary specifies:
            'model' : string
                a machine-learning model/algorithm
                currently valid: 'knn', 'svm', 'neuralnet'
            'hyperparameters' : dictionary
                parameters for "training" the model
            'feature_list_indices' : list of integers
                features to use from an already generated feature array
    ftr_grp_ID_dict : dict
        from feature file. validate_yaml checks whether it is
        defined in feature_file and if so passes as an argument
    ftr_list_group_ID : numpy 1-d vector
        from feature file. validate_yaml checks whether it is
        defined in feature_file and if so passes as an argument

    Returns
    -------
    validated_models : list
        after validation
    """

    # make sure value for models key is a list
    if type(models) != list:
        raise ValueError('\'models\' should be a list not a {}'
                         .format(type(models)))

    # make sure all items in models list are dictionaries
    if not all([type(el) is dict for el in models]):
        raise ValueError('all items in \'models\' should be dictionaries')

    # check if model key declared for all dicts in models list
    doesnt_have_model_key = ['model' not in model_dict for model_dict in models]
    if any(doesnt_have_model_key):
        no_model_inds = [ind for ind, boolval in enumerate(doesnt_have_model_key)
                         if boolval]
        raise KeyError('No model declared for following items in'
                       'list of model dictionaries: '
                       .format(no_model_inds))

    # check if all declared models are valid
    model_set = set([model_dict['model'] for model_dict in models])
    if not model_set.issubset(VALID_MODELS):
        invalid_models = list(model_set - VALID_MODELS)
        raise ValueError('{} in \'models\' are not valid model types'.format(invalid_models))

    # then valid everything else in every dict
    validated_models = copy.deepcopy(models)

    for index, model_dict in enumerate(models):
        if 'ftr_grp_ID_dict' in locals() and 'ftr_list_group_ID' in locals():
            validated_models[index] = _validate_model_dict(model_dict,
                                                           index,
                                                           ftr_grp_ID_dict,
                                                           ftr_list_group_ID)
        else:
            validated_models[index] = _validate_model_dict(model_dict,
                                                           index)
    return validated_models


VALID_NUM_SAMPLES_KEYS = {'start', 'stop', 'step'}
REQUIRED_TODO_KEYS = {'feature_file', 'output_dir'}
OPTIONAL_TODO_KEYS = {'num_test_samples', 'num_train_samples', 'num_replicates', 'models'}


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

    feature_file = todo_list_dict['feature_file']
    if type(feature_file) != str:
        raise ValueError('Value {} for key \'feature_file\' is type {} but it'
                         ' should be a string'.format(feature_file,
                                                      type(feature_file)))
    if not os.path.isfile(feature_file):
        raise OSError('{} is not found as a file'.format(feature_file))
    try:
        ftr_file = joblib.load(feature_file)
        feature_file_keys = ftr_file.keys()
        if 'feature_group_ID_dict' in feature_file_keys:
            ftr_list_group_ID = ftr_file['feature_list_group_ID']
            ftr_grp_ID_dict = ftr_file['feature_group_ID_dict']
        del ftr_file
    except:
        raise IOError('Unable to open {}'.format(feature_file))

    validated_todo_list_dict = copy.deepcopy(todo_list_dict)
    for key, val in todo_list_dict.items():
        # valid todo_list_dict keys in alphabetical order

        if key == 'models':
            if 'ftr_list_group_ID' in locals() and 'ftr_grp_ID_dict' in locals():
                validated_todo_list_dict['models'] = _validate_models(val,
                                                                      ftr_grp_ID_dict,
                                                                      ftr_list_group_ID)
            else:
                validated_todo_list_dict['models'] = _validate_models(val)

        elif key == 'num_replicates':
            if type(val) != int:
                raise ValueError('{} should be an int but parsed as {}'
                                 .format(key, type(val)))

        elif key == 'num_test_samples':
            if type(val) != int:
                raise ValueError('{} should be an integer'.format(key))

        elif key == 'num_train_samples':
            if type(val) != dict:
                raise ValueError('{} should be a dict but parsed as {}'
                                 .format(key, type(val)))
            else:
                samples_keys = {'start', 'stop', 'step'}
                if set(val.keys()) != samples_keys:
                    raise KeyError('incorrect keys in {}'.format(key))
                else:
                    num_samples = range(val['start'],
                                        val['stop'],
                                        val['step'])
                    validated_select_config['num_train_samples'] = num_samples

        elif key == 'output_dir':
            if type(val) != str:
                raise ValueError('output_dirs should be a string but it parsed as a {}'
                                 .format(type(val)))

    return validated_todo_list_dict

VALID_SELECT_KEYS = {'todo_list',
                     'num_replicates',
                     'num_test_samples',
                     'num_train_samples',
                     'models'}


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

    if type(select_config_yaml) is not dict:
        raise ValueError('select_config_yaml passed to parse.extract was '
                         'not recognized as a dict, instead was a {}.'
                         'Must pass a dict containing config loaded from YAML'
                         'file or a str that is a YAML filename.'
                         .format(type(select_config_yaml)))

    if 'todo_list' not in select_config_yaml:
        raise KeyError('Did not find \`todo_list\` defined in \`select\` config file.')

    select_keys = set(select_config_yaml.keys())
    if not select_keys <= VALID_SELECT_KEYS:
        raise KeyError('unrecognized keys in select dictionary: {}'
                       .format(list(select_keys - VALID_SELECT_KEYS)))

    for select_key in VALID_SELECT_KEYS:
        if not all([select_key in todo for todo in select_config_yaml['todo_list']]):
            if select_key not in select_config_yaml:
                raise KeyError('\'{0}\' not defined for every item in todo_list, '
                               'but no default {0} is defined. You must either '
                               'define \'{0}\' in the \'select\' dictionary '
                               '(that any \'{0}\' in a todo_list item will take '
                               'precedence over) or you must define \'{0}\' for'
                               ' every item in the todo_list.'.format(select_key))

    validated_select_config = copy.deepcopy(select_config_yaml)
    for key, val in select_config_yaml.items():

        if key == 'models':
            validated_select_config['models'] = _validate_models(val)

        elif key == 'num_replicates':
            if type(val) != int:
                raise ValueError('{} in \'select\' should be an integer'.format(key))

        elif key == 'num_test_samples':
            if type(val) != int:
                raise ValueError('{} in \'select\' should be an integer'.format(key))

        elif key == 'num_train_samples':
            if type(val) != dict:
                raise ValueError('\'num_train_samples\' did not parse as dict. Please check formatting')
            samples_key_set = set(val.keys())
            if samples_key_set != VALID_NUM_SAMPLES_KEYS:
                raise KeyError('\'num_samples\' contains invalid keys {}, '
                               'should only contain the following keys: '
                               '{}'.format(samples_key_set - VALID_NUM_SAMPLES_KEYS,
                                           VALID_NUM_SAMPLES_KEYS))
            for samples_key in samples_key_set:
                if type(val[samples_key]) != int:
                    raise ValueError('value for \'{}\' in \'num_samples\' should be type int, not {}'
                                     .format(samples_key,
                                             key,
                                             type(val[samples_key])))
                if samples_key == 'stop':
                    if val['stop'] < val['start']:
                        raise ValueError('stop value is {} but should be greater than start value, {}'
                                         .format(val['stop'],
                                                 val['start']))
            num_samples_vals = range(val['start'],
                                     val['stop'],
                                     val['step'])
            validated_select_config['num_train_samples'] = num_samples_vals

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
