import pdb

#from standard library
import os
import json
import shelve

# from Anaconda distrib
import numpy as np
from sklearn.metrics import confusion_matrix as confuse_mat
from sklearn.metrics import recall_score

# from functions files written for these experiments
from svm_rbf_test_utility_functions import load_from_mat
from knn_test_functions import load_knn_data

####utility functions for this script
def filter_labels(labels,labelset):
    """
    filter_labels(labels,labelset)
    returns labels with any elements removed that are not in labelset
    """
    labels_to_keep = np.in1d(labels,labelset) #returns boolean vector, True where label is in labelset
    labels = labels[labels_to_keep]
    return labels
    
def get_acc_by_label(labels,pred_labels,labelset):
    """
    get_acc_by_label(labels,pred_labels,labelset)

    arguments:
    labels -- vector of labels from a test data set
    pred_labels -- vector of predicted labels returned by algorithm given samples from test data set
    labelset -- set of unique labels from test data set, i.e., numpy.unique(labels)

    returns:
    acc_by_label -- vector of accuracies
    avg_acc -- average accuracy across labels, i.e., numpy.mean(acc_by_label)
    """
    acc_by_label = np.zeros((len(labelset)))
    for ind,label in enumerate(labelset):
        label_ids = np.in1d(labels,label) #find all occurences of label in test data
        if sum(label_ids) == 0: # if there were no instances of label in labels
            continue
        pred_for_that_label = pred_labels[label_ids]
        matches = pred_for_that_label==label
        #sum(matches) is equal to number of true positives
        #len(matches) is equal to number of true positives and false negatives
        acc = sum(matches) / len(matches)
        acc_by_label[ind] = acc
    avg_acc = np.mean(acc_by_label)
    return acc_by_label,avg_acc

def generate_summary_results_files(NUM_SAMPLES_TO_TRAIN_WITH,REPLICATES,labelset,
                                   svm_labels,RESULTS_SHELVE_BASE_FNAME):
    """
    Creates summary files in results_dir.

    input arguments:
    NUM_SAMPLES_TO_TRAIN_WITH -- sample size of training set, e.g., range(100,1600,100)
    REPLICATES -- number of estimates of accuracy for each sized training set, e.g., range(1,11)
    labelset -- labels given to song syllables, converted from char to int, e.g., [97,99,100,101]
    svm_labels -- labels associated with samples in svm feature file. It is assumed the same
    labels are used for the knn models (the testing script ensures that this is the case).
    RESULTS_SHELVE_BASE_FNAME -- base for filenames given to shelve files containing data
    from each replicate.
    """
   
    # to initialize arrays
    rows = len(REPLICATES)
    cols = len(NUM_SAMPLES_TO_TRAIN_WITH)
    
    #Rand accuracy
    svm_train_rnd_acc = np.zeros((rows,cols))
    svm_test_rnd_acc = np.zeros((rows,cols))
    knn_train_rnd_acc = np.zeros((rows,cols))
    knn_test_rnd_acc = np.zeros((rows,cols))

    #average of per-label accuracy ( true positive / (true positive + false negative))
    svm_train_avg_acc = np.zeros((rows,cols))
    svm_test_avg_acc = np.zeros((rows,cols))
    knn_train_avg_acc = np.zeros((rows,cols))
    knn_test_avg_acc = np.zeros((rows,cols))

    #confusion matrices
    num_labels = len(labelset)
    svm_train_acc_by_label = np.zeros((rows,cols,num_labels))
    svm_train_cm_arr = np.empty((rows,cols),dtype='O')
    svm_test_acc_by_label = np.zeros((rows,cols,num_labels))
    svm_test_cm_arr = np.empty((rows,cols),dtype='O')
    knn_train_acc_by_label = np.zeros((rows,cols,num_labels))
    knn_train_cm_arr = np.empty((rows,cols),dtype='O')
    knn_test_acc_by_label = np.zeros((rows,cols,num_labels))
    knn_test_cm_arr = np.empty((rows,cols),dtype='O')

    #annotations for use with graph; appear as labels when user hovers over points on plot
    #this way user can hover over point they see as "best" classifier (e.g. highest accuracy)
    #to find out which results file contains that saved classfier
    annotes = np.empty((rows,cols),dtype='O')
    
    ### loop that opens shelve database file for each replicate and puts values into summary data matrices
    for col_ind, num_samples in enumerate(NUM_SAMPLES_TO_TRAIN_WITH):
        for row_ind,replicate in enumerate(REPLICATES):
            annote_str = str(num_samples) + ' samples, replicate ' + str(replicate)
            annotes[row_ind,col_ind] = annote_str
            shelve_fname = RESULTS_SHELVE_BASE_FNAME + "_" + str(num_samples) + '_samples_replicate ' + str(replicate) + '.db'
            with shelve.open(shelve_fname,'r') as shv:
                # get number of samples, duration of samples 
                train_sample_IDs = shv['train_sample_IDs']
                train_labels = svm_labels[train_sample_IDs]
                test_sample_IDs = shv['test_sample_IDs']
                test_labels = svm_labels[test_sample_IDs]

                # put Rand accuracies in summary data matrices
                # below, [0] at end of line because liblinear Python API returns 3-element tuple, 1st element is acc.
                svm_train_rnd_acc[row_ind,col_ind] = shv['svm_train_score'] * 100
                svm_test_rnd_acc[row_ind,col_ind] = shv['svm_test_score'] * 100
                knn_train_rnd_acc[row_ind,col_ind] = shv['knn_train_score'] * 100
                knn_test_rnd_acc[row_ind,col_ind] = shv['knn_test_score'] * 100

                # put average per-label accuracies in summary data matrices
                # and make confusion matrices
                svm_train_pred_labels = shv['svm_train_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(train_labels,svm_train_pred_labels,labelset)
                svm_train_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                svm_train_avg_acc[row_ind,col_ind] = avg_acc * 100
                svm_train_confuse_mat = confuse_mat(train_labels,svm_train_pred_labels,labels=labelset)
                svm_train_cm_arr[row_ind,col_ind]  = svm_train_confuse_mat
                
                svm_test_pred_labels = shv['svm_test_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(test_labels,svm_test_pred_labels,labelset)
                svm_test_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                svm_test_avg_acc[row_ind,col_ind] = avg_acc * 100
                svm_test_confuse_mat = confuse_mat(test_labels,svm_test_pred_labels,labels=labelset)
                svm_test_cm_arr[row_ind,col_ind] = svm_test_confuse_mat

                knn_train_pred_labels = shv['knn_train_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(train_labels,knn_train_pred_labels,labelset)
                knn_train_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                knn_train_avg_acc[row_ind,col_ind] = avg_acc * 100
                knn_train_confuse_mat = confuse_mat(train_labels,knn_train_pred_labels,labels=labelset)
                knn_train_cm_arr[row_ind,col_ind] = knn_train_confuse_mat
                                                
                knn_test_pred_labels = shv['knn_test_pred_labels']
                acc_by_label,avg_acc = get_acc_by_label(test_labels,knn_test_pred_labels,labelset)
                knn_test_acc_by_label[row_ind,col_ind] = acc_by_label * 100
                knn_test_avg_acc[row_ind,col_ind] = avg_acc * 100
                knn_test_confuse_mat = confuse_mat(test_labels,knn_test_pred_labels,labels=labelset)
                knn_test_cm_arr[row_ind,col_ind] = knn_test_confuse_mat
            shv.close()

    # now put all the summary data matrices in a summary data shelve database
    with shelve.open('svmrbf_knn_results_summary.db') as shv:
        shv['labelset'] = labelset
        shv['NUM_SAMPLES_TO_TRAIN_WITH'] = NUM_SAMPLES_TO_TRAIN_WITH
        shv['REPLICATES'] = REPLICATES
        shv['annotes'] = annotes
        
        shv['svm_train_rnd_acc'] = svm_train_rnd_acc
        shv['svm_train_rnd_acc_mn'] = np.mean(svm_train_rnd_acc,axis=0)
        shv['svm_train_rnd_acc_std'] = np.std(svm_train_rnd_acc,axis=0)

        shv['svm_test_rnd_acc'] = svm_test_rnd_acc
        shv['svm_test_rnd_acc_mn'] = np.mean(svm_test_rnd_acc,axis=0)
        shv['svm_test_rnd_acc_std'] = np.std(svm_test_rnd_acc,axis=0)

        shv['knn_train_rnd_acc'] = knn_train_rnd_acc
        shv['knn_train_rnd_acc_mn'] = np.mean(knn_train_rnd_acc,axis=0)
        shv['knn_train_rnd_acc_std'] = np.std(knn_train_rnd_acc,axis=0)

        shv['knn_test_rnd_acc'] = knn_test_rnd_acc
        shv['knn_test_rnd_acc_mn'] = np.mean(knn_test_rnd_acc,axis=0)
        shv['knn_test_rnd_acc_std'] = np.std(knn_test_rnd_acc,axis=0)

        shv['svm_train_acc_by_label'] = svm_train_acc_by_label
        shv['svm_train_avg_acc'] = svm_train_avg_acc
        shv['svm_train_avg_acc_mn'] = np.mean(svm_train_avg_acc,axis=0)
        shv['svm_train_avg_acc_std'] = np.std(svm_train_avg_acc,axis=0)
        shv['svm_train_cm_arr'] = svm_train_cm_arr

        shv['svm_test_acc_by_label'] = svm_test_acc_by_label
        shv['svm_test_avg_acc'] = svm_test_avg_acc
        shv['svm_test_avg_acc_mn'] = np.mean(svm_test_avg_acc,axis=0)
        shv['svm_test_avg_acc_std'] = np.std(svm_test_avg_acc,axis=0)
        shv['svm_test_cm_arr'] = svm_test_cm_arr

        shv['knn_train_acc_by_label'] = knn_train_acc_by_label
        shv['knn_train_avg_acc'] = knn_train_avg_acc
        shv['knn_train_avg_acc_mn'] = np.mean(knn_train_avg_acc,axis=0)
        shv['knn_train_avg_acc_std'] = np.std(knn_train_avg_acc,axis=0)
        shv['knn_train_cm_arr'] = knn_train_cm_arr
        
        shv['knn_test_acc_by_label'] = knn_test_acc_by_label
        shv['knn_test_avg_acc'] = knn_test_avg_acc
        shv['knn_test_avg_acc_mn'] = np.mean(knn_test_avg_acc,axis=0)
        shv['knn_test_avg_acc_std'] = np.std(knn_test_avg_acc,axis=0)
        shv['knn_test_cm_arr'] = knn_test_cm_arr
    shv.close()
