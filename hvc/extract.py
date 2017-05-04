"""
feature extraction
"""

#from standard library
import sys
import os
import glob

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
    for todo in todo_list:
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

        data_dirs = todo['data_dirs']
        for data_dir in data_dirs:
            os.chdir(data_dir)

            if file_format == 'evtaf':
                notmats = glob.glob('*.not.mat')
                features_from_each_file = []
                for notmat in notmats:
                    cbin = notmat[:-8] # remove .not.mat extension from filename to get name of associated .cbin file
                    syls, labels = evfuncs.get_syls(cbin,
                                                    extract_config['spect_params'],
                                                    todo['labelset'])
                    for syl in syls:
                        if is_ftr_list_of_lists:
                            ftrs = features.extract.extract_features_from_syllable(todo['feature_list'],
                                                                                          syl,
                                                                                          feature_groups)
                        else:
                            ftrs = features.extract.extract_features_from_syllable(todo['feature_list'],
                                                                                          syl)
                        features_from_each_file.append(ftrs)
                import pdb;pdb.set_trace()
            elif file_format == 'koumura':
                annot_xml = glob.glob('Annotation.xml')
                if annot_xml==[]:
                    raise ValueError('no Annotation.xml file found in directory {}'.format(dir))
                elif len(annot_xml) > 1:
                    raise ValueError('more than one Annotation.xml file found in directory {}'.format(dir))
                else:
                    seq_list = hvc.koumura.parse_xml(annot_xml[0])

        # save yaml file for select.py with output_dir in it!!!
        output_dir = todo['output_dir']
