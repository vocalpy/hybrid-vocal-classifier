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

        data_dirs = todo['data_dirs']
        output_dir = todo['output_dir']
        labelset = todo['labelset']

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
                    import pdb;pdb.set_trace()
                    for syl in syls:
                        features = features.extract.extract_feature_from_syllable(syl)
                    features_from_each_file.append(features)

            elif file_format == 'koumura':
                annot_xml = glob.glob('Annotation.xml')
                if annot_xml==[]:
                    raise ValueError('no Annotation.xml file found in directory {}'.format(dir))
                elif len(annot_xml) > 1:
                    raise ValueError('more than one Annotation.xml file found in directory {}'.format(dir))
                else:
                    seq_list = hvc.koumura.parse_xml(annot_xml[0])

    # save yaml file for select.py with output_dir in it!!!