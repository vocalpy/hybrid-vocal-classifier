import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import neighbors

def load_from_mat(fname,ftr_file_type,purpose='train'):
    """
    Loads feature files created in matlab

    Arguments
    ---------
    fname : string
        filename of .mat file
    ftr_file_type : {'knn','svm'}
        string that tells load_from_mat how to parse .mat feature file
    return_notmat_fnames : logical
        if 'True' then returns list of .not.mat files (to loop over in a script)
    purpose : {'train','classify'}
        Features files to classify contain only samples since they by definition
        haven't been labeled.

    Returns
    -------
    samples : numpy ndarray
        m-by-n numpy array, m rows of samples each with n features
    labels : numpy ndarray
        vector of length m, where each element is a label corresponding
        to row m of 'samples' array
    song_IDs_vec : numpy ndarray
        vector of length m, where each element denotes which song that
        sample m(sub i) belongs to in the samples array. E.g., if elements 15-37
        have the value "3", then all those samples belong to the third song.
        Used to relate number of songs to number of samples.  
    """   
    if ftr_file_type=='knn':
        FEATURES_TO_USE = np.r_[0:6,7:10,11] # using np._r to build index array 
        #load 'cell array' from .mat file. ftr_file is a dictionary of numpy 
        #record arrays
        #the 'feature_cell' record array has two columns: col 1 = actual vals,
        #col 0 is just ftr names
        ftr_file = loadmat(fname,chars_as_strings=True)
        # concatenate features horizontally (each row is a sample)
        samples = np.hstack(ftr_file['feature_cell'][1,:])
        samples = samples[:,FEATURES_TO_USE] #discard unused features

    elif ftr_file_type=='svm':
        ftr_file = loadmat(fname)
        samples = ftr_file['features_mat']

    if purpose=='classify':
        return samples

    elif purpose=='train':
        # flatten because matlab vectors are imported as 2d numpy arrays with one row
        labels = ftr_file['label_vec'].flatten()
        labels = labels.view(np.uint32) # convert from unicode to long 
        song_IDs_vec = ftr_file['song_IDs_vec'].flatten()
        return samples, labels, song_IDs_vec

def filter_samples(samples,labels,labels_to_filter,song_ID_vec=None,remove=False):
    """
    filter_samples(samples,labels,labels_to_keep,song_ID_vec=None,remove=False)
        input parameters:
            samples -- m-by-n numpy array with m rows of samples, each having n
                       columns of features
            labels -- 1d numpy array of m elements where each is a label that
                      corresponds to a row in samples. Expects an integer value
            song_ID_vec -- vector of length m, where each element denotes which song
                       that sample m(sub i) belongs to in the samples array.
                       "None" by default. If provided, is filtered just like labels
                       and samples.
            labels_to_filter -- like labels, 1d numpy array of integers like
                                lables, but filter_samples finds all indices of
                                a label that occurs in 'labels *and* in 'labels_to_filter',
                                and keeps only those, filtering out every other label
                              and filtering out the corresponding rows from samples
                              and the corresponding elements from song_ID_vec.
                              ***unless*** 'remove'=True, in which case it *removes*
                              _only_ the labels in labels_to_filter, and leaves
                              everything else. 
           remove -- set remove=True when you want to remove only the labels in
                     labels to filter instead of keeping only those labels
        returns: filtered_samples,filtered_labels,filtered_song_IDs
    """
    if remove is True:
        indices = np.in1d(labels,labels_to_filter,invert=True)
    elif remove is False:
        indices = np.in1d(labels,labels_to_filter)
    filtered_labels = labels[indices]
    filtered_samples = samples[indices,:]
    if song_ID_vec is None:
        return filtered_samples, filtered_labels
    else:
        filtered_song_IDs = song_ID_vec[indices]
        return filtered_samples, filtered_labels,filtered_song_IDs

def grid_search(X,y):
    """carries out a grid search of C and gamma parameters for an RBF kernel to
    use with a support vector classifier.

    Arguments:
        X -- numpy m-by-n array containing m samples each with n features
        y -- numpy array of length m containing m labels corresponding to the
             m samples in X

    Returns:
        best_params -- values for C and gamma that gave the highest accuracy with
                       cross-validation
        best_score -- highest accuracy with cross-validation
    """
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(y, n_iter=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)
    print("The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_))
    return grid.best_params_, grid.best_score_

def uniqify_filename_list(filename_list, idfun=None):
   # based on code by Peter Bengtsson
   # https://www.peterbe.com/plog/uniqifiers-benchmark
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for filename in filename_list:
       marker = idfun(filename)
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

def find_best_k(train_samples,train_labels,test_samples,test_labels):
    """find_best_k(train_samples,train_labels,holdout_samples,holdout_labels)
    Estimates accuracy of k-neearest neighbors algorithm using different values
    of k on the samples in data_fname. As currently written, this function loops
    from k=1 to k=10. For each value of k, it generates 10 replicates by using
    a random two-thirds of the data as a training set and then using the other
    third as the validation set.

    Note that the algorithm uses the distances weighted by 
    their inverse to determine the nearest neighbor, since I found empirically
    that the weighted distances always give slightly better accuracy.
    
    Arguments:
        train_samples -- m-by-n numpy array with m rows of samples, each having n features
        train_labels -- numpy vector of length m, where each element is a label corresponding
                  to a row in 'samples'
        holdout_samples, holdout_labels -- same as train_samples and train_labels except this
                                           is the set kept separate and used to find best hyper-
                                           -parameters
        
    Returns:
        mn_scores -- vector of mean scores for each value of k
        best_k -- value of k corresponding to max value in 'scores'

    """

    # test loop
    num_nabes_list = range(1,11,1)
    scores = np.empty((10,))
    for ind, num_nabes in enumerate(num_nabes_list):
        clf = neighbors.KNeighborsClassifier(num_nabes,'distance')
        clf.fit(train_samples,train_labels)
        scores[ind] = clf.score(test_samples,test_labels)
    k = num_nabes_list[scores.argmax()] #argmax returns index of max val
    print("best k was {} with accuracy of {}".format(k,np.max(scores)))
    return scores, k
