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
        if all(isinstance(element, list) for element in feature_list):
            is_ftr_list_of_lists = True
            if 'feature_group' in todo:
                feature_groups = todo['feature_group']
            else:
                feature_groups = range(1,len(feature_list)+1)
        else:
            is_ftr_list_of_lists = False # for sanity check when debugging

        output_dir = todo['output_dir'] + 'extract_output_' + timestamp
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        data_dirs = todo['data_dirs']
        for data_dir in data_dirs:
            print('Changing to data directory: {}'.format(data_dir))
            os.chdir(data_dir)

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
                features_from_curr_file, labels = features.extract.from_file(songfile,
                                                                             todo['file_format'],
                                                                             todo['feature_list'],
                                                                             extract_config['spect_params'],
                                                                             todo['labelset'])
                all_labels.extend(labels)
                if 'features_from_all_files' in locals():
                    # note have to add dimension with newaxis because np.concat requires
                    # same number of dimensions, but extract_features returns 1d.
                    # Decided to keep it explicit that we go to 2d here.
                    features_from_all_files = np.concatenate((features_from_all_files,
                                                              features_from_curr_file))
                else:
                    features_from_all_files = features_from_curr_file



            #         syls, labels = evfuncs.get_syls(cbin,
            #                                         extract_config['spect_params'],
            #                                         todo['labelset'])
            #         for syl in syls:
            #             if is_ftr_list_of_lists:
            #                 ftrs = features.extract.extract_features_from_syllable(todo['feature_list'],
            #                                                                               syl,
            #                                                                               feature_groups)
            #             else:
            #                 ftrs = features.extract.extract_features_from_syllable(todo['feature_list'],
            #                                                                               syl)
            #
            #
            # elif file_format == 'koumura':
            #     annot_xml = glob.glob('Annotation.xml')
            #     if annot_xml==[]:
            #         raise ValueError('no Annotation.xml file found in directory {}'.format(dir))
            #     elif len(annot_xml) > 1:
            #         raise ValueError('more than one Annotation.xml file found in directory {}'.format(dir))
            #     else:
            #         seq_list = hvc.koumura.parse_xml(annot_xml[0])
            #     for seq in seq_list:
            #         for syl in seq.syls:
            #             ftrs = features.extract.extract_features_from_syllable(todo['feature_list'],
            #                                                                    syl)
            #         if 'features_from_each_file' in locals():
            #             np.append(features_from_each_file, ftrs)
            #         else:
            #             features_from_each_file = ftrs

            # get dir name without the rest of path so it doesn't have separators in the name
            # because those can't be in filename
            just_dir_name = os.getcwd().split(os.path.sep)[-1]
            output_filename = os.path.join(output_dir,
                                           'features_from_' + just_dir_name + '_created_' + timestamp)
            output_dict = {
                'features' : features_from_all_files,
                'labels' : all_labels,
                'feature_list': todo['feature_list'],
                'spect_params' : extract_config['spect_params'],
                'labelset' : todo['labelset'],
                'file_format' : todo['file_format'],
                'bird_ID' : todo['bird_ID']
            }
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
                    summary_output_dict['features'] = np.concatenate((summary_output_dict['features'],
                                                                     output_dict['features']))

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