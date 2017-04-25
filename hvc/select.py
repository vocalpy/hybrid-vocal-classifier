"""
model selection:
trains models that classify birdsong syllables,
using algorithms and other parameters specified in config file
"""

#from standard library
import sys
import glob

#from hvc
from parseconfig import parse
from utils import load_from_mat

# get command line arguments
args = sys.argv
config_file = args[1]
config = parse(config_file)

global_config = config['global_config']
if 'select' not in config:
    raise KeyError('select not defined in config file \'{}\''.format(config_file))
else:
    select = config['select']
    #imports here

# # if no feature_files provided, extract feature_files automatically
# if 'model_selection' in locals() and 'feature_files' not in model_selection:
#     auto_extract_features = True
#
# if auto_extract_features:
#     import hvc.extract
#     hvc.extract('')

if 'select' in locals():
    #from scikit-learn
    from sklearn.preprocessing import StandardScaler

    #from hvc
    from utils import filter_samples

    models = model_selection['models']
    if 'svm' in models:
        from sklearn.svm import SVC
        svm_scaler = StandardScaler()

    if 'knn' in models:
        from sklearn import neighbors
        knn_scaler = StandardScaler()

    num_samples = model_selection['num_samples']
    num_replicates = model_selection['num_replicates']

    jobs = model_selection['jobs']
    for job in jobs:
            os.chdir(dir)
            if 'svm' in models:
                svm_fname = svm_fname[0]
                svm_samples, svm_labels, svm_song_IDs = load_from_mat(svm_fname, 'svm')
                svm_samples, svm_labels, svm_song_IDs = filter_samples(svm_samples, svm_labels, labelset, svm_song_IDs)

            if 'knn' in models:
                knn_fname = glob.glob('*concat_knn_ftr*')
                if len(knn_fname) > 1:
                    raise ValueError('found more than one file of concatenated svm features')
                knn_fname = knn_fname[0]
                knn_samples, knn_labels, knn_song_IDs = load_ftr_files(knn_fname, 'knn')
                knn_samples, knn_labels, knn_song_IDs = filter_samples(knn_samples, knn_labels, labelset, knn_song_IDs)

        if 'svm' in models and 'knn' in models:
            # shape and elements for svm and knn labels should be the same
            # since samples are taken from exact same song files
            if not np.array_equal(svm_labels, knn_labels):
                raise ValueError('labels from svm feature files and knn feature files do not match')
                # (obviously features themselves won't be the same, hence compare labels not samples)