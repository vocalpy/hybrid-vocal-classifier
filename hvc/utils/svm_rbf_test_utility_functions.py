import pdb
import numpy as np
import scipy.io as scio # to load matlab files
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

def load_from_mat(fname,return_notmat_fnames=False):
    """Loads data for testing svm from matlab .mat files

    Argument:
        fname -- filename of .mat file

    Returns:
        samples -- m-by-n numpy array, m rows of samples each with n features
        labels -- vector of length m, where each element is a label corresponding
                  to row m of 'samples' array
        song_IDs_vec -- vector of length m, where each element denotes which song
                       that sample m(sub i) belongs to in the samples array.
                       E.g., if elements 15-37 have the value "3", then all those
                       samples belong to the third song. Used to relate number of
                       songs to number of samples.  
    """
    mat_file = scio.loadmat(fname)
    #loads 'cell array' from .mat file. ftr_file is a dictionary of numpy record arrays
    #the 'feature_cell' record array has two columns: col 1 = actual vals, col 0 is just ftr names
    samples = mat_file['features_mat']
    labels = mat_file['label_vec'].flatten() # flatten because matlab vectors are imported as 2d numpy arrays with one row
    song_IDs_vec = mat_file['song_IDs_vec'].flatten()
    if return_notmat_fnames is True:
        notmat_fnames = mat_file['notmat_fnames']
        return samples, song_IDs_vec, notmat_fnames
    else:
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
    if remove:
        indices = np.in1d(labels,labels_to_filter,invert=True)
    else:
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

def train_test_song_split(samples,labels,song_IDs,train_size,test_size = 0.2):
    """train_test_song_split:
    Splits samples from songs into training and test sets

    Input arguments:
		samples -- m-by-n numpy array, m rows of samples each with n features
        labels -- vector of length m, where each element is a label corresponding
                  to row m of 'samples' array
        song_IDs_vec -- vector of length m, where each element denotes which song
                       that sample m(sub i) belongs to in the samples array.
                       E.g., if elements 15-37 have the value "3", then all those
                       samples belong to the third song. Used to relate number of
                       songs to number of samples.  
    	train_size -- either a float between 0.0 and 1.0, representing the percent
                     of the songs to put into the train set, or an integer rep-
                     -resenting the number of songs to put into the train set.
                     Returned in 'train_samples' and 'train_labels'
       	test_size -- same as train_size, but for set returned in test_samples and
       				 test_labels. Default is 0.2, i.e., one-fifth of set for
       				 5-fold cross-validation

    Returns:
        train_samples, train_labels, test_samples, test_labels
    """
    uniq_song_IDs = np.unique(song_IDs)
    n_songs = max(uniq_song_IDs)
    if np.asarray(train_size).dtype.kind == 'f':
        if train_size >= 1.:
            raise ValueError(
                'train_size=%f should be smaller '
                'than 1.0 or be an integer' % test_size)
        else:
            n_train = int(round(test_size * n_songs))
    elif np.asarray(train_size).dtype.kind == 'i':
        if train_size >= n_songs:
            raise ValueError(
                'train_size=%d should be smaller '
                'than the number of songs %d' % (test_size, n_songs))
        else:
            n_train = int(train_size)
    
    if np.asarray(test_size).dtype.kind == 'f':
        if test_size >= 1.:
            raise ValueError(
                'test_size=%f should be smaller '
                'than 1.0 or be an integer' % test_size)
        else:
            n_test = int(round(test_size * n_songs))
    elif np.asarray(test_size).dtype.kind == 'i':
        if test_size >= n_songs:
            raise ValueError(
                'test_size=%d should be smaller '
                'than the number of songs %d' % (test_size, n_songs))
        else:
            n_test = int(test_size)
    
    if n_train + n_test > n_songs:
        raise ValueError(
            'Number of training songs, %d, plus number of test songs, %d,'
            'is greater than the total number of songs, %d' % (n_train, n_test, n_songs))

    # loop until there are at least two examples of each label in the training set
    # this is necessary for StratifiedShuffleSplit to work in the Grid Search function
    while 1:
        np.random.shuffle(uniq_song_IDs) # shuffles array in place, no need to assign to a different variable
        train_song_IDs = uniq_song_IDs[0:n_train]
        test_song_IDs = uniq_song_IDs[n_train:n_train + n_test]
        train_song_sample_IDs = np.where(np.in1d(song_IDs,train_song_IDs))[0] #[0] because where returns tuple
        train_samples = samples[train_song_sample_IDs,:]
        train_labels = labels[train_song_sample_IDs]
        test_song_sample_IDs = np.where(np.in1d(song_IDs,test_song_IDs))[0] #[0] because where returns tuple
        test_samples = samples[test_song_sample_IDs,:]
        test_labels = labels[test_song_sample_IDs]
        inds = np.unique(train_labels,return_inverse=True)[1] # indices of unique labels, don't need unique labels to bin and count
        if np.min(np.bincount(inds))<2:
            continue
        else:
            return train_samples, train_labels, test_samples, test_labels,train_song_sample_IDs,test_song_sample_IDs

def train_test_syllable_split(samples,labels,num_train_samples,test_size = 0.2):
    """train_test_syllable_split:
    Splits samples from songs into training and test sets. Different from train_test_song_split
    in that this function returns the same number of training samples for each class


    Input arguments:
		samples -- m-by-n numpy array, m rows of samples each with n features
        labels -- vector of length m, where each element is a label corresponding
                  to row m of 'samples' array

        song_IDs_vec -- vector of length m, where each element denotes which song
                       that sample m(sub i) belongs to in the samples array.
                       E.g., if elements 15-37 have the value "3", then all those
                       samples belong to the third song. Used to relate number of

                       songs to number of samples.  
    	train_size -- either a float between 0.0 and 1.0, representing the percent
                     of the songs to put into the train set, or an integer rep-
                     -resenting the number of songs to put into the train set.
                     Returned in 'train_samples' and 'train_labels'

       	test_size -- same as train_size, but for set returned in test_samples and
       				 test_labels. Default is 0.2, i.e., one-fifth of set for
       				 5-fold cross-validation


    Returns:
        train_samples, train_labels, test_samples, test_labels
    """
    uniq_song_IDs = np.unique(song_IDs)
    n_songs = max(uniq_song_IDs)
    if np.asarray(train_size).dtype.kind == 'f':
        if train_size >= 1.:
            raise ValueError(
                'train_size=%f should be smaller '
                'than 1.0 or be an integer' % test_size)
        else:
            n_train = int(round(test_size * n_songs))
    elif np.asarray(train_size).dtype.kind == 'i':
        if train_size >= n_songs:
            raise ValueError(
                'train_size=%d should be smaller '
                'than the number of songs %d' % (test_size, n_songs))
        else:
            n_train = int(train_size)
    
    if np.asarray(test_size).dtype.kind == 'f':
        if test_size >= 1.:
            raise ValueError(
                'test_size=%f should be smaller '
                'than 1.0 or be an integer' % test_size)
        else:
            n_test = int(round(test_size * n_songs))
    elif np.asarray(test_size).dtype.kind == 'i':
        if test_size >= n_songs:
            raise ValueError(
                'test_size=%d should be smaller '
                'than the number of songs %d' % (test_size, n_songs))

        else:
            n_test = int(test_size)
    
    if n_train + n_test > n_songs:
        raise ValueError(
            'Number of training songs, %d, plus number of test songs, %d,'
            'is greater than the total number of songs, %d' % (n_train, n_test, n_songs))

    # loop until there are at least two examples of each label in the training set
    # this is necessary for StratifiedShuffleSplit to work in the Grid Search function
    while 1:
        np.random.shuffle(uniq_song_IDs) # shuffles array in place, no need to assign to a different variable
        train_song_IDs = uniq_song_IDs[0:n_train]
        test_song_IDs = uniq_song_IDs[n_train:n_train + n_test]
        train_song_sample_IDs = np.where(np.in1d(song_IDs,train_song_IDs))[0] #[0] because where returns tuple
        train_samples = samples[train_song_sample_IDs,:]
        train_labels = labels[train_song_sample_IDs]
        test_song_sample_IDs = np.where(np.in1d(song_IDs,test_song_IDs))[0] #[0] because where returns tuple
        test_samples = samples[test_song_sample_IDs,:]
        test_labels = labels[test_song_sample_IDs]
        inds = np.unique(train_labels,return_inverse=True)[1] # indices of unique labels, don't need unique labels to bin and count
        if np.min(np.bincount(inds))<2:
            continue
        else:
            return train_samples, train_labels, test_samples, test_labels,train_song_sample_IDs,test_song_sample_IDs
