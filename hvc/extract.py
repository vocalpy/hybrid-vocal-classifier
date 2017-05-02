"""
feature extraction
"""

#from standard library
import sys
import glob

#from hvc
import hvc.parse.extract
import hvc.features

# get command line arguments
args = sys.argv
config_file = args[1]
extract_config = hvc.parse.extract.parse_extract_config(config_file)

format = extract['format']
if format=='evtaf':
    import hvc.evfuncs
elif format=='koumura':
    import hvc.koumura

todo_list = extract['todo_list']
for todo in todo_list:
    dirs = todo['dirs']
    output_dir = todo['output_dir']
    labelset = todo['labelset']

    for dir in dirs:
        os.chdir(dir)

        if format == 'evtaf':
            notmats = glob.glob('*.not.mat')
            features_from_each_file = []
            for notmat in notmats:
                cbin = notmat[:-8] # remove .not.mat extension from filename to get name of associated .cbin file
                syls, labels = hvc.evfuncs.get_syls(cbin,spect_params,labelset)
                features = extract_features(format, labelset)
                features_from_each_file.append(features)

        elif format == 'koumura':
            annot_xml = glob.glob('Annotation.xml')
            if annot_xml==[]:
                raise ValueError('no Annotation.xml file found in directory {}'.format(dir))
            elif len(annot_xml) > 1:
                raise ValueError('more than one Annotation.xml file found in directory {}'.format(dir))
            else:
                seq_list = hvc.koumura.parse_xml(annot_xml[0])

# save yaml file for select.py with output_dir in it!!!