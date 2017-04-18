"""
trains models that classify birdsong syllables,
using algorithms and other parameters specified in config file
"""

#from standard library
import sys
import shelve
import os
import glob
import warnings
import pickle
import random
import copy

#from dependencies
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import neighbors

#from hvc
from utils import filter_samples, grid_search, find_best_k
from utils.matlab import load_ftr_files
from generate_summary import generate_summary_results_files

# get command line arguments
args = sys.argv
config_file = args[1]
config_dict = parseconfig(config_file)

#constants
HOME_DIR = os.getcwd()
#DATA_DIR = os.path.join(args[1],'train')
#if not os.path.isdir(DATA_DIR):
#    os.mkdir(DATA_DIR)
#os.chdir(DATA_DIR)
#RESULTS_DIR = os.path.join(DATA_DIR,'svmrbf_knn_results')
#if not os.path.isdir(RESULTS_DIR):
#    os.mkdir(RESULTS_DIR)
#RESULTS_SHELVE_BASE_FNAME = os.path.join(RESULTS_DIR,'svmrbf_knn_results_')

jobs = config_dict['jobs']

# scalers from scikit used in main loop
svm_scaler = StandardScaler()
knn_scaler = StandardScaler()

# constants used in main loop
NUM_SAMPLES_TO_TRAIN_WITH = range(100,1600,100)
REPLICATES = range(1,11)

#load train/test data, label names
LABELS = args[2]
labelset = list(LABELS)
labelset = [ord(label) for label in labelset]

svm_fname = glob.glob('*concat_svm_ftr*')
if len(svm_fname) > 1:
    raise ValueError('found more than one file of concatenated svm features')
svm_fname = svm_fname[0]
svm_samples,svm_labels,svm_song_IDs = load_ftr_files(svm_fname,'svm')
svm_samples,svm_labels,svm_song_IDs = filter_samples(svm_samples,svm_labels,labelset,svm_song_IDs)

knn_fname = glob.glob('*concat_knn_ftr*')
if len(knn_fname) > 1:
    raise ValueError('found more than one file of concatenated svm features')
knn_fname = knn_fname[0]
knn_samples, knn_labels, knn_song_IDs = load_ftr_files(knn_fname,'knn')
knn_samples, knn_labels, knn_song_IDs = filter_samples(knn_samples,knn_labels,labelset,knn_song_IDs)

# shape and elements for svm and knn labels should be the same
# since samples are taken from exact same song files
if not np.array_equal(svm_labels,knn_labels):
    raise ValueError('labels from svm feature files and knn feature files do not match')
# (obviously features themselves won't be the same, hence compare labels not samples)

# get max number of train samples and use that for number of samples in test set
num_samples_in_test_set = np.max(NUM_SAMPLES_TO_TRAIN_WITH)
# make sure we're not using a tiny test set
total_num_samples = svm_song_IDs.shape[0]
two_times_max = num_samples_in_test_set * 2
if total_num_samples < two_times_max:
    msg1 = 'Total number of samples is less than two times the number of samples tested. '
    msg2 = 'Total number of samples is {} and two times number tested is {}. '.format(total_num_samples,two_times_max)
    msg3 = 'Be aware that this could affect the estimate of accuracy.'
    msg = msg1 + msg2 + msg3
    warnings.warn(msg)

num_songs = np.max(svm_song_IDs)
song_ID_list = list(range(1,num_songs+1))

#create test set, pop those IDs off the song_ID_list so test set does not contain samples in train set
test_sample_IDs,song_ID_list = grab_n_samples_by_song(svm_song_IDs,
                                                      song_ID_list,
                                                      svm_labels,
                                                      num_samples_in_test_set,
                                                      return_popped_songlist=True)
svm_test_samples = svm_samples[test_sample_IDs]
knn_test_samples = knn_samples[test_sample_IDs]
test_labels = svm_labels[test_sample_IDs]      

for row_ind, num_samples_to_train_with in enumerate(NUM_SAMPLES_TO_TRAIN_WITH):
    print('\nestimating accuracy for training set with {} samples'.format(num_samples_to_train_with))
    for replicate in REPLICATES:
        print('replicate {}.'.format(replicate))

        ### test support vector machine with radial basis function ###
        ### using Tachibana features plus adjacent syllable features ###           
        print(" Training SVM w/RBF using Tachibana features plus features of adjacent syllables.")
        train_sample_IDs = shuffle_then_grab_n_samples_by_song_ID(svm_song_IDs,
                                                                    song_ID_list,
                                                                    svm_labels,
                                                                    num_samples_to_train_with,
                                                                    return_popped_songlist=False)
        svm_train_samples = svm_samples[train_sample_IDs]
        train_labels = svm_labels[train_sample_IDs]
        svm_train_samples_scaled = svm_scaler.fit_transform(svm_train_samples)
        #have to scale test samples each time by factors used to scale each distinct training set
        svm_test_samples_scaled = svm_scaler.transform(svm_test_samples)

        
        best_params, best_grid_score = grid_search(svm_train_samples_scaled,train_labels)
        svm_clf = SVC(C=best_params['C'],gamma=best_params['gamma'],decision_function_shape='ovr')
        svm_clf.fit(svm_train_samples_scaled,train_labels)
        svm_train_pred_labels = svm_clf.predict(svm_train_samples_scaled)
        svm_train_score = svm_clf.score(svm_train_samples_scaled,train_labels)
        svm_test_pred_labels = svm_clf.predict(svm_test_samples_scaled)
        svm_test_score = svm_clf.score(svm_test_samples_scaled,test_labels)
        svm_decision_func = svm_clf.decision_function(svm_test_samples_scaled)
        print(" svm score on train set: ",svm_train_score)
        print(" svm score on test set: ",svm_test_score)

        ### test k-Nearest neighbors ###
        ### using common songbird acoustic analysis parameters as features ###
        ### including some of those parameters from adjacent syllables in each sample ###
        print("'Training' k-NN model using acoustic params + adjacent syllable features.")

        knn_train_samples = knn_samples[train_sample_IDs]  
        knn_train_samples_scaled = knn_scaler.fit_transform(knn_train_samples)
        #have to scale test samples each time by factors used to scale each distinct training set
        knn_test_samples_scaled = knn_scaler.transform(knn_test_samples)
        
        print(" finding best k")
        k = find_best_k(knn_train_samples_scaled,train_labels,knn_test_samples_scaled,test_labels)[1]
        # ^ [1] because I don't want mn_scores, just k
        print(" best k was: " + str(k))
        knn_clf = neighbors.KNeighborsClassifier(k,'distance')
        knn_clf.fit(knn_train_samples_scaled,train_labels)    
        knn_train_pred_labels = knn_clf.predict(knn_train_samples_scaled)
        knn_train_score = knn_clf.score(knn_train_samples_scaled,train_labels)
        knn_test_pred_labels = knn_clf.predict(knn_test_samples_scaled)
        knn_test_score = knn_clf.score(knn_test_samples_scaled,test_labels)
        print(" knn score on train set: ",knn_train_score)
        print(" knn score on test set: ",knn_test_score)

        ### save results from this replicate ###
        rest_of_fname = "_" + str(num_samples_to_train_with) + '_samples_replicate ' + str(replicate)
        results_shelve_fname = RESULTS_SHELVE_BASE_FNAME + rest_of_fname + '.db'                        
        with shelve.open(results_shelve_fname) as shlv:
            shlv['train_sample_IDs'] = train_sample_IDs
            shlv['test_sample_IDs'] = test_sample_IDs

            shlv['best_params'] = best_params
            shlv['best_grid_score'] = best_grid_score
            shlv['svm_train_pred_labels'] = svm_train_pred_labels
            shlv['svm_train_score'] = svm_train_score
            shlv['svm_test_pred_labels'] = svm_test_pred_labels
            shlv['svm_test_score'] = svm_test_score
            shlv['svm_decision_func'] = svm_decision_func
            shlv['svm_clf'] = svm_clf
            shlv['svm_scaler'] = svm_scaler
                                    
            shlv['knn_train_pred_labels'] = knn_train_pred_labels
            shlv['knn_train_score'] = knn_train_score
            shlv['knn_test_pred_labels'] = knn_test_pred_labels
            shlv['knn_test_score'] = knn_test_score
            shlv['knn_clf'] = knn_clf
            shlv['knn_scaler'] = knn_scaler
        #svm_fname = RESULTS_SVM_DIR + "svm_" + rest_of_fname + ".pickle"        
        #with open(svm_fname, 'wb') as f:
        #    pickle.dump(svm_clf,f,pickle.HIGHEST_PROTOCOL)
        #knn_fname = RESULTS_KNN_DIR + "knn_" + rest_of_fname + ".pickle"
        #with open(knn_fname, 'wb') as f:
        #    pickle.dump(knn_clf,f,pickle.HIGHEST_PROTOCOL)

print("\nGenerating summary file.")
generate_summary_results_files(NUM_SAMPLES_TO_TRAIN_WITH,REPLICATES,labelset,svm_labels,RESULTS_SHELVE_BASE_FNAME)

os.chdir(HOME_DIR)
