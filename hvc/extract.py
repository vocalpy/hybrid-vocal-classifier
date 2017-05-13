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
from . import parse
from . import features

def run(config_file):
    """
    main function that runs feature extraction.
    Does not return anything, just runs through directories specified in config_file
    and extracts features.
    
    Parameters
    ----------
    config_file : string
        filename of YAML file that configures feature extraction    
    """
    extract_config = parse.extract.parse_extract_config(config_file)
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
                'bird_ID' : todo['bird_ID']
            }
            if 'feature_group_id' in todo:
                ftrs_dict = {}
                for grp_ind, ftr_grp in enumerate(todo['feature_group']):
                    ftrs_from_group = np.where(todo['feature_group_id'] == grp_ind)
                    group_ftr_inds = np.in1d(ftr_inds,ftrs_from_group)
                    ftrs_dict[ftr_grp] = features_from_all_files[:,group_ftr_inds]
                output_dict['features'] = ftrs_dict
            else:
                output_dict['features'] = features_from_all_files

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
            for output_file in ftr_output_files:
                list_of_output_dicts.append(joblib.load(output_file))

            summary_output_dict = {}
            for output_dict in list_of_output_dicts:
                if 'features' not in summary_output_dict:
                    summary_output_dict['features'] = output_dict['features']
                else:
                    if type(summary_output_dict['features']) == np.ndarray:
                        summary_output_dict['features'] = np.concatenate((summary_output_dict['features'],
                                                                         output_dict['features']))
                    elif type(summary_output_dict['features']) == dict:
                        for key in output_dict['features'].keys():
                            summary_output_dict['features'][key] = np.concatenate((summary_output_dict['features'][key],
                                                                                   output_dict['features'][key]))

                if 'labels' not in summary_output_dict:
                    summary_output_dict['labels'] = output_dict['labels']
                else:
                    summary_output_dict['labels'] = summary_output_dict['labels'] + output_dict['labels']

                if 'spect_params' not in summary_output_dict:
                    summary_output_dict['spect_params'] = output_dict['spect_params']

                if 'labelset' not in summary_output_dict:
                    summary_output_dict['labelset'] = output_dict['labelset']

                if 'file_format' not in summary_output_dict:
                    summary_output_dict['file_format'] = output_dict['file_format']

                if 'bird_ID' not in summary_output_dict:
                    summary_output_dict['spect_params'] = output_dict['spect_params']

                if 'feature_list' not in summary_output_dict:
                    summary_output_dict['feature_list'] = output_dict['feature_list']
            joblib.dump(summary_output_dict,
                        'summary_feature_file_created_' + timestamp)
        else: # if only one feature_file
            os.rename(ftr_output_files[0],
                      'summary_feature_file_created_' + timestamp)