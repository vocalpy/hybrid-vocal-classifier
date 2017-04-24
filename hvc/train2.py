"""
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
if 'model_selection' in config:
    model_selection = config['model_selection']
    #imports here
if 'prediction' in config:
    prediction = config['prediction']
    #imports here

if 'model_selection' in locals():
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
        train_dirs = job['train']['dirs']
        for dir in train_dirs:
            os.chdir(dir)
            if 'svm' in models:
                svm_fname = glob.glob('*concat_svm_ftr*')
                if len(svm_fname) > 1:
                    raise ValueError('found more than one file of concatenated svm features')
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

if 'prediction' in locals():
    jobs = prediction['jobs']
    for job in jobs:
        model_file = jobs['model_file']