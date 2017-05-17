"""
feature extraction
"""

#from standard library
import sys
import os
import glob
from datetime import datetime

#from dependencies
import numpy as np
from sklearn.externals import joblib

#from hvc
from .parseconfig import parse_config
from . import features

def extract(config_file):
    """
    main function that runs feature extraction.
    Does not return anything, just runs through directories specified in config_file
    and extracts features.
    
    Parameters
    ----------
    config_file : string
        filename of YAML file that configures feature extraction    
    """
    extract_config = parse_config(config_file,'extract')
    print('Parsed extract config.')

    todo_list = extract_config['todo_list']
    for ind, todo in enumerate(todo_list):

        timestamp = datetime.now().strftime('%y%m%d_%H%M')

        print('Completing item {} of {} in to-do list'.format(ind+1,len(todo_list)))
        file_format = todo['file_format']
        if file_format == 'evtaf':
            if 'evfuncs' not in sys.modules:
                from . import evfuncs
        elif file_format == 'koumura':
            if 'koumura' not in sys.modules:
                from . import koumura

        feature_list = todo['feature_list']

        output_dir = todo['output_dir'] + 'extract_output_' + timestamp
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        data_dirs = todo['data_dirs']
        for data_dir in data_dirs:
            print('Changing to data directory: {}'.format(data_dir))
            os.chdir(data_dir)

            if 'features_from_all_files' in locals():
                # from last time through loop
                # (need to re-initialize for each directory)
                del features_from_all_files

            if file_format == 'evtaf':
                songfiles = glob.glob('*.not.mat')
            elif file_format == 'koumura':
                songfiles = glob.glob('*.wav')
            num_songfiles = len(songfiles)
            all_labels = []
            song_IDs = []
            song_ID_counter = 0
            for file_num, songfile in enumerate(songfiles):
                print('Processing audio file {} of {}.'.format(file_num+1,num_songfiles))
                if file_format == 'evtaf':
                    songfile = songfile[:-8] # remove .not.mat extension from filename to get name of associated .cbin file
                ftrs_from_curr_file, labels, ftr_inds = features.extract.from_file(songfile,
                                                                             todo['file_format'],
                                                                             todo['feature_list'],
                                                                             extract_config['spect_params'],
                                                                             todo['labelset'])
                all_labels.extend(labels)
                song_IDs.extend([song_ID_counter] * len(labels))
                song_ID_counter += 1

                if 'features_from_all_files' in locals():
                    features_from_all_files = np.concatenate((features_from_all_files,
                                                              ftrs_from_curr_file),
                                                             axis=0)
                else:
                    features_from_all_files = ftrs_from_curr_file

            # get dir name without the rest of path so it doesn't have separators in the name
            # because those can't be in filename
            just_dir_name = os.getcwd().split(os.path.sep)[-1]
            output_filename = os.path.join(output_dir,
                                           'features_from_' + just_dir_name + '_created_' + timestamp)
            output_dict = {
                'labels' : all_labels,
                'feature_list': todo['feature_list'],
                'spect_params' : extract_config['spect_params'],
                'labelset' : todo['labelset'],
                'file_format' : todo['file_format'],
                'bird_ID' : todo['bird_ID'],
                'song_IDs' : song_IDs,
                'features' : features_from_all_files
            }
            if 'feature_group_ID' in todo:
                output_dict['feature_group_ID'] = todo['feature_group_ID']
                output_dict['feature_group_ID_dict'] = todo['feature_group_ID_dict']

            joblib.dump(output_dict,
                        output_filename,
                        compress=3)

        ##########################################################
        # after looping through all data_dirs for this todo_item #
        ##########################################################
        print('making summary file')
        os.chdir(output_dir)
        ftr_output_files = glob.glob('*features_from_*')
        if len(ftr_output_files) > 1:
            #make a 'summary' data file
            list_of_output_dicts = []
            summary_output_dict = {}
            for output_file in ftr_output_files:
                output_dict = joblib.load(output_file)

                if 'features' not in summary_output_dict:
                    summary_output_dict['features'] = output_dict['features']
                else:
                    summary_output_dict['features'] = np.concatenate((summary_output_dict['features'],
                                                                         output_dict['features']))
                if 'labels' not in summary_output_dict:
                    summary_output_dict['labels'] = output_dict['labels']
                else:
                    summary_output_dict['labels'] = summary_output_dict['labels'] + output_dict['labels']

                if 'spect_params' not in summary_output_dict:
                    summary_output_dict['spect_params'] = output_dict['spect_params']
                else:
                    if output_dict['spect_params'] != summary_output_dict['spect_params']:
                        raise ValueError('mismatch between spect_params in {} '
                                         'and other feature files'.format(output_file))

                if 'labelset' not in summary_output_dict:
                    summary_output_dict['labelset'] = output_dict['labelset']
                else:
                    if output_dict['labelset'] != summary_output_dict['labelset']:
                        raise ValueError('mismatch between labelset in {} '
                                         'and other feature files'.format(output_file))

                if 'file_format' not in summary_output_dict:
                    summary_output_dict['file_format'] = output_dict['file_format']
                else:
                    if output_dict['file_format'] != summary_output_dict['file_format']:
                        raise ValueError('mismatch between file_format in {} '
                                         'and other feature files'.format(output_file))

                if 'bird_ID' not in summary_output_dict:
                    summary_output_dict['bird_ID'] = output_dict['bird_ID']
                else:
                    if output_dict['bird_ID'] != summary_output_dict['bird_ID']:
                        raise ValueError('mismatch between bird_ID in {} '
                                         'and other feature files'.format(output_file))

                if 'song_IDs' not in summary_output_dict:
                    summary_output_dict['song_IDs'] = output_dict['song_IDs']
                else:
                    curr_last_ID = summary_output_dict['song_IDs'][-1]
                    tmp_song_IDs = [el + curr_last_ID + 1 for el in output_dict['song_IDs']]
                    summary_output_dict['song_IDs'].extend(tmp_song_IDs)

                if 'feature_list' not in summary_output_dict:
                    summary_output_dict['feature_list'] = output_dict['feature_list']
                else:
                    if output_dict['feature_list'] != summary_output_dict['feature_list']:
                        raise ValueError('mismatch between feature_list in {} '
                                         'and other feature files'.format(output_file))

                if 'feature_group_ID' not in summary_output_dict:
                    summary_output_dict['feature_group_ID'] = output_dict['feature_group_ID']
                else:
                    if any(output_dict['feature_group_ID'] != summary_output_dict['feature_group_ID']):
                        raise ValueError('mismatch between feature_group_ID in {} '
                                         'and other feature files'.format(output_file))

                if 'feature_group_ID_dict' not in summary_output_dict:
                    summary_output_dict['feature_group_ID_dict'] = output_dict['feature_group_ID_dict']
                else:
                    if output_dict['feature_group_ID_dict'] != summary_output_dict['feature_group_ID_dict']:
                        raise ValueError('mismatch between feature_group_ID_dict in {} '
                                         'and other feature files'.format(output_file))

            joblib.dump(summary_output_dict,
                        'summary_feature_file_created_' + timestamp)
        else: # if only one feature_file
            os.rename(ftr_output_files[0],
                      'summary_feature_file_created_' + timestamp)

