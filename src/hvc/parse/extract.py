"""
YAML parser for extract config files
"""

# from standard library
import os
import csv
import copy
import warnings

# from dependencies
import yaml

from .ref_spect_params import refs_dict
from .utils import check_for_missing_keys, flatten

path = os.path.abspath(__file__)  # get the path of this file
dir_path = os.path.dirname(path)  # but then just take the dir

with open(os.path.join(dir_path, 'features.yml')) as features_yml:
    VALID_FEATURES = yaml.load(features_yml, Loader=yaml.FullLoader)['features']

with open(os.path.join(dir_path, 'validation.yml')) as val_yaml:
    validate_dict = yaml.load(val_yaml, Loader=yaml.FullLoader)

# feature groups in separate file from feature list because
# want to validate feature groups against feature list
# and some features are not in feature group
with open(os.path.join(dir_path, 'feature_groups.yml')) as ftr_grp_yaml:
    valid_feature_groups_dict = yaml.load(ftr_grp_yaml, Loader=yaml.FullLoader)

REQUIRED_TODO_LIST_KEYS = set(validate_dict['required_extract_todo_list_keys'])
REQUIRED_TODO_LIST_KEYS_FLATTENED = set(flatten(
    validate_dict['required_extract_todo_list_keys']))
OPTIONAL_TODO_LIST_KEYS = set(validate_dict['optional_extract_todo_list_keys'])
################################################################
# validation functions for individual configuration parameters #
################################################################

valid_spect_param_keys = {'nperseg',
                          'noverlap',
                          'freq_cutoffs',
                          'window',
                          'filter_func',
                          'spect_func',
                          'ref',
                          'log_transform_spect'
                          }


def validate_spect_params(spect_params):
    """validates spect_params

    Parameters
    ----------
    spect_params : dict
        with keys as specified in extract YAML spec
        also are the arguments to Spectrogram.__init__
            nperseg : int
                numper of samples per segment for FFT, e.g. 512
            noverlap : int
                number of overlapping samples in each segment
            freq_cutoffs : two-element list of integers
                limits of frequency band to keep, e.g. [1000,8000]
                Spectrogram.make keeps the band:
                    freq_cutoffs[0] >= spectrogram > freq_cutoffs[1]
            window : str
                window to apply to segments
                valid strings are 'Hann', 'dpss', None
                Hann -- Uses np.Hanning with parameter M (window width) set to value of nperseg
                dpss -- Discrete prolate spheroidal sequence AKA Slepian.
                    Uses scipy.signal.slepian with M parameter equal to nperseg and
                    width parameter equal to 4/nperseg, as in [2]_.
            filter_func : str
                filter to apply to raw audio. valid strings are 'diff' or None
                'diff' -- differential filter, literally np.diff applied to signal as in [1]_.
                'filt_song' -- filter used by evsonganaly.m with .cbin files recorded by evTAF
                    bandpass filter applied by filtfilt function
                None -- no filter, this is the default
            spect_func : str
                which function to use for spectrogram.
                valid strings are 'scipy' or 'mpl'.
                'scipy' uses scipy.signal.spectrogram,
                'mpl' uses matplotlib.matlab.specgram.
                Default is 'scipy'.
            ref : str
                {'tachibana','koumura'}
                Use spectrogram parameters from a reference.
                'tachibana' uses spectrogram parameters from [1]_,
                'koumura' uses spectrogram parameters from [2]_.
            log_transform_spect : bool
                if True, applies np.log10 to spectrogram to increase range. Default is True.

    Returns
    -------
    spect_params
    """

    if type(spect_params) != dict:
        raise TypeError('value for key \'spect_params\' in config file did '
                         'not parse as a dictionary of parameters, '
                         'it parsed as {}. Check file formatting.'
                         .format(spect_params))

    if not set(spect_params.keys()) <= valid_spect_param_keys:
        invalid_keys = set(spect_params.keys()) - valid_spect_param_keys
        raise KeyError('unrecognized keys in spect_params dictionary: {}'
                       .format(invalid_keys))

    if 'ref' in spect_params:
        if spect_params['ref'] not in refs_dict:
            raise ValueError('Value {} for \'ref\' not recognized.'
                             'Valid values are: {}.'
                             .format(spect_params['ref'],
                                     list(refs_dict.keys())
                                     )
                             )
        if len(spect_params.keys()) > 1:
            warnings.warn('spect_params contains \'ref\' parameter '
                          'but also contains other parameters. Defaults '
                          'for \'ref\' will override other parameters.')
            return {'ref': spect_params['ref']}
        else:
            return refs_dict[spect_params['ref']]

    if 'nperseg' not in spect_params.keys() and 'noverlap' not in spect_params.keys():
        raise KeyError('keys nperseg and noverlap are required in'
                       'spect_params but were not found.')
    for sp_key, sp_val in spect_params.items():
        if sp_key == 'nperseg' or sp_key == 'noverlap':
            if type(sp_val) != int:
                raise ValueError('{} in spect_params should be an integer'.format(sp_key))
        elif sp_key == 'freq_cutoffs':
            if len(sp_val) != 2:
                raise ValueError('freq_cutoffs should be a 2 item list')
            for freq_cutoff in sp_val:
                if type(freq_cutoff) != int:
                    raise ValueError('freq_cutoff {} should be an int'.format(sp_val))
        elif sp_key == 'window':
            if sp_val not in {'Hann', 'dspp', None}:
                raise ValueError('{} is invalid value for window in spect params.'
                                 'Valid values are: {\'Hann\', \'dspp\', None}'
                                 .format(sp_val))
        elif sp_key == 'filter_func':
            if sp_val not in {'diff', 'bandpass_filtfilt', 'butter_bandpass', None}:
                raise ValueError('{} is invalid value for filter_func in spect params.'
                                 'Valid values are: {\'diff\', \'bandpass_filtfilt\','
                                 '\'butter_andpass\', None}'
                                 .format(sp_val))
        elif sp_key == 'log_transform_spect':
            if type(sp_val) != bool:
                raise TypeError('log_transform_spect parsed as type {}, '
                                'but should be bool.'
                                .format(type(sp_val)))
    return spect_params

valid_segment_param_keys = {'threshold',
                            'min_syl_dur',
                            'min_silent_dur'}


def validate_segment_params(segment_params):
    """validates segmenting parameters

    Parameters
    ----------
    segment_params : dict
        with following keys:
            threshold : int
                amplitudes crossing above this are considered segments
            min_syl_dur : float
                minimum syllable duration, in seconds
            min_silent_dur : float
                minimum duration of silent gap between syllables, in seconds

    Returns
    -------
    nothing if parameters are valid
    else raises error
    """

    if type(segment_params) != dict:
        raise TypeError('segment_params did not parse as a dictionary, '
                        'instead it parsed as {}.'
                        ' Please check config file formatting.'.format(type(val)))

    elif set(segment_params.keys()) != valid_segment_param_keys:
        if set(segment_params.keys()) < valid_segment_param_keys:
            missing_keys = valid_segment_param_keys - set(segment_params.keys())
            raise KeyError('segment_params is missing keys: {}'
                           .format(missing_keys))
        elif valid_segment_param_keys < set(segment_params.keys()):
            extra_keys = set(segment_params.keys()) - segment_param_keys
            raise KeyError('segment_params has extra keys:'
                           .format(extra_keys))
        else:
            invalid_keys = set(segment_params.keys()) - valid_segment_param_keys
            raise KeyError('segment_params has invalid keys:'
                           .format(invalid_keys))
    else:
        for key, val in segment_params.items():
            if key == 'threshold':
                if type(val) != int:
                    raise ValueError('threshold should be int but parsed as {}'
                                     .format(type(val)))
            elif key == 'min_syl_dur':
                if type(val) != float:
                    raise ValueError('min_syl_dur should be float but parsed as {}'
                                     .format(type(val)))
            elif key == 'min_silent_dur':
                if type(val) != float:
                    raise ValueError('min_silent_dur should be float but parsed as {}'
                                     .format(type(val)))


def _validate_feature_list(feature_list):
    """helper function to validate feature_list
    """

    if type(feature_list) != list:
        raise ValueError('feature_list should be a list but parsed as a {}'.format(type(val)))
    else:
        for feature in feature_list:
            if feature not in VALID_FEATURES:
                raise ValueError('feature {} not found in valid features'.format(feature))


def _validate_feature_group_and_convert_to_list(feature_group,
                                                feature_list=None):
    """validates feature_group value from todo_list dicts, then converts
    to feature_list.
    Since todo_list dicts can include both feature_group and feature_list,
    this function will accept feature_list from the dict and then append
    the feature_group features to those already in the feature_list.

    Parameters
    ----------
    feature_group : str or list
        currently valid feature groups: {'svm','knn'}
        if list, must be a list of strings
    feature_list : list
        list of features, default is None.
        If not None, features from feature groups will be appended to this list.

    Returns
    -------
    feature_list : list
        list of features to extract for each feature group.
        if feature_list was passed to function along with feature_group, then
        features from feature group(s) are appended to end of the feature_list
        passed to the function.
    feature_group_ID_dict : dict
        dict where key is a feature group name and value is a corresponding ID, an int.
        Same length as feature_list.
        Used by hvc.modelselection.select to determine which columns in feature
        array belong to which feature group.
        If feature_list was passed to the function, it does not affect this dict,
        since the features in that list are not considered part of a feature group
    feature_list_group_ID_arr : list
        list of ints of same length as feature_list.
        If feature_list was passed to function, its features will have value None
    """

    if type(feature_group) != str and type(feature_group) != list:
        raise TypeError('feature_group parsed as {} but it should be'
                        ' either a string or a list. Please check config'
                        ' file formatting.'.format(type(feature_group)))

    # if user entered list with just one element
    if type(feature_group) == list and len(feature_group) == 1:
        # just take that one element, assuming it is str,
        # i.e. name of feature group
        feature_group = feature_group[0]

    if type(feature_group) == str:
        if feature_group not in valid_feature_groups_dict:
            raise ValueError('{} not found in valid feature groups'.format(feature_group))
        else:
            ftr_grp_list = valid_feature_groups_dict[feature_group]
            _validate_feature_list(ftr_grp_list)  # sanity check
            ftr_grp_ID_dict = {feature_group: 0}
            feature_list_group_ID = [0] * len(ftr_grp_list)

    elif type(feature_group) == list:
        # if a list of feature groups
        # make feature list that is concatenated feature groups
        # and also add 'feature_group_id' vector for indexing to config
        ftr_grp_list = []
        feature_list_group_ID = []
        ftr_grp_ID_dict = {}
        for grp_ind, ftr_grp in enumerate(feature_group):
            if ftr_grp not in valid_feature_groups_dict:
                raise ValueError('{} not found in valid feature groups'.format(ftr_grp))
            else:
                ftr_grp_list.extend(valid_feature_groups_dict[ftr_grp])
                feature_list_group_ID.extend([grp_ind] * len(valid_feature_groups_dict[ftr_grp]))
                ftr_grp_ID_dict[ftr_grp] = grp_ind
        _validate_feature_list(ftr_grp_list)

    if feature_list is not None:
        not_ftr_grp_features = [None] * len(feature_list)
        feature_list_group_ID = not_ftr_grp_features + feature_list_group_ID
        return (feature_list + ftr_grp_list,
                feature_list_group_ID,
                ftr_grp_ID_dict)
    else:
        feature_list_group_ID = feature_list_group_ID
        return (ftr_grp_list,
                feature_list_group_ID,
                ftr_grp_ID_dict)


def _validate_todo_list_dict(todo_list_dict, index, config_path):
    """
    validates to-do lists

    Parameters
    ----------
    todo_list_dict : dictionary
        from "to-do" list
    index : int
        index of element (i.e., dictionary) in list of dictionaries
    config_path : str
        absolute path to YAML config file from which dict was taken.
        Used to validate directory names.

    Returns
    -------
    validated_todo_list_dict : dictionary
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

    if 'feature_group' not in todo_list_dict and 'feature_list' not in todo_list_dict:
            raise ValueError('todo_list item #{} does not include feature_group or feature_list'
                             .format(index))

    # first make copy of todo_list_dict that can be chanegd
    validated_todo_list_dict = copy.deepcopy(todo_list_dict)

    if 'feature_list' in validated_todo_list_dict and \
                    'feature_group' not in validated_todo_list_dict:
        # if just feature_list, just validate it, don't have to
        # do anything else:
        _validate_feature_list(validated_todo_list_dict['feature_list'])

    elif 'feature_group' in validated_todo_list_dict and \
                    'feature_list' not in validated_todo_list_dict:
        # if just feature group, convert to feature list then validate
        ftr_grp_valid = \
            _validate_feature_group_and_convert_to_list(
                validated_todo_list_dict['feature_group'])

        validated_todo_list_dict['feature_list'] = ftr_grp_valid[0]
        validated_todo_list_dict['feature_list_group_ID'] = ftr_grp_valid[1]
        validated_todo_list_dict['feature_group_ID_dict'] = ftr_grp_valid[2]

    elif 'feature_list' in validated_todo_list_dict and \
                    'feature_group' in validated_todo_list_dict:
        ftr_grp_valid = _validate_feature_group_and_convert_to_list(
            validated_todo_list_dict['feature_group'],
            validated_todo_list_dict['feature_list'])

        validated_todo_list_dict['feature_list'] = ftr_grp_valid[0]
        validated_todo_list_dict['feature_list_group_ID'] = ftr_grp_valid[1]
        validated_todo_list_dict['feature_group_ID_dict'] = ftr_grp_valid[2]

    # okay now that we took care of that we can loop through everything else
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

        elif key == 'data_dirs':
            if type(val) != list:
                raise ValueError('data_dirs should be a list')
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

        elif key == 'labels_to_use':
            if type(val) != str:
                raise ValueError('labels_to_use should be a string, e.g., \'iabcde\'.')
            else:
                validated_todo_list_dict[key] = list(val) # convert from string to list of chars
                validated_todo_list_dict['labels_to_use_int'] = [ord(label) for label in list(val)]

        elif key == 'output_dir':
            if type(val) != str:
                raise ValueError('output_dirs should be a string but it parsed as a {}'
                                 .format(type(val)))
            # add 'save_features=True' since this is implied when user
            # specifies a directory for output
            if 'save_features' not in todo_list_dict:
                validated_todo_list_dict['save_features'] = True

        elif key == 'save_features':
            if ('output_dir' in todo_list_dict and
               todo_list_dict['save_features'] is False):
                raise ValueError('output_dir was specified but '
                                 'save_features was set to False')

        elif key == 'segment_params':
            validate_segment_params(val)

        elif key == 'spect_params':
            validate_spect_params(val)

    return validated_todo_list_dict

##########################################
# main function that validates yaml file #
##########################################


def validate_yaml(config_path, extract_config_yaml):
    """
    validates config from extract YAML file

    Parameters
    ----------
    config_path : str
        absolute path to YAML config file. Used to validate directory names
        in YAML files, which are assumed to be written relative to the
        location of the file itself.
    extract_config_yaml : dict
        dict should be config from YAML file as loaded with pyyaml.

    Returns
    -------
    extract_config_dict : dictionary, after validation of all keys
    """

    if type(extract_config_yaml) is not dict:
        raise ValueError('extract_config_yaml passed to parse.extract was '
                         'not recognized as a dict, instead was a {}.'
                         'Must pass a dict containing config loaded from YAML'
                         'file or a str that is a YAML filename.'
                         .format(type(extract_config_yaml)))

    if 'todo_list' not in extract_config_yaml:
        raise KeyError('extract config does not have required key \'todo_list\'')

    if 'spect_params' not in extract_config_yaml:
        has_spect_params = ['spect_params' in todo_dict
                            for todo_dict in extract_config_yaml['todo_list']]
        if not all(has_spect_params):
            raise KeyError('no default `spect_params` specified, but'
                           'not every todo_list in extract config has spect_params')

    if 'segment_params' not in extract_config_yaml:
        has_segment_params = ['segment_params' in todo_dict
                              for todo_dict in extract_config_yaml['todo_list']]
        if not all(has_segment_params):
            raise KeyError('no default `segment_params` specified, but'
                           'not every todo_list in extract config has segment_params')

    validated = copy.deepcopy(extract_config_yaml)
    for key, val in extract_config_yaml.items():
        if key == 'spect_params':
            validated['spect_params'] = validate_spect_params(val)
        elif key == 'segment_params':
            validate_segment_params(val)
        elif key == 'todo_list':
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
                        val[index] = _validate_todo_list_dict(item, index, config_path)
            validated['todo_list'] = val # re-assign because feature list is added

        else: # if key is not found in list
            raise KeyError('key {} in extract is an invalid key'.
                            format(key))

    return validated
