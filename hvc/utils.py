#from standard library
import copy
import random
from urllib.error import HTTPError
from datetime import datetime

#from dependencies
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import neighbors

#from hvc
from .randomdotorg import RandomDotOrg


def timestamp():
    """timestampe for dir + file names
    Use to make sure each dir/file has unique identifier
    (so we don't load the wrong data with joblib because
    both data files have the same name)
    """
    return datetime.now().strftime('%y%m%d_%H%M%S')


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

    if ftr_file_type == 'knn':
        FEATURES_TO_USE = np.r_[0:6,7:10,11] # using np._r to build index array 
        #load 'cell array' from .mat file. ftr_file is a dictionary of numpy 
        #record arrays
        #the 'feature_cell' record array has two columns: col 1 = actual vals,
        #col 0 is just ftr names
        ftr_file = loadmat(fname,chars_as_strings=True)
        # concatenate features horizontally (each row is a sample)
        samples = np.hstack(ftr_file['feature_cell'][1,:])
        samples = samples[:,FEATURES_TO_USE] #discard unused features

    elif ftr_file_type == 'svm':
        ftr_file = loadmat(fname)
        samples = ftr_file['features_mat']

    if purpose == 'classify':
        return samples

    elif purpose == 'train':
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

def filter_labels(labels,labelset):
    """
    filter_labels(labels,labelset)
    returns labels with any elements removed that are not in labelset
    """
    labels_to_keep = np.in1d(labels,labelset) #returns boolean vector, True where label is in labelset
    labels = labels[labels_to_keep]
    return labels


def grid_search(X, y):
    """carries out a grid search of C and gamma parameters for an RBF kernel to
    use with a support vector classifier.

    Arguments
    ---------
    X : ndarray
        m-by-n array containing m samples each with n features
    y : ndarray
        numpy array of length m containing m labels corresponding to the
        m samples in X

    Returns
    -------
    best_params :
        values for C and gamma that gave the highest accuracy with
        cross-validation
    best_score :
        highest accuracy with cross-validation
    """

    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
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
    print("best k was {} with accuracy of {}".format(k, np.max(scores)))
    return scores, k


def grab_n_samples_by_song(song_IDs,
                           labels,
                           num_samples,
                           song_ID_list = None,
                           seed=None,
                           return_popped_songlist=False,
                           use_random_dot_org=False):
    """Creates list of sample IDs for training or test set.
    Grabs samples by song ID from shuffled list of IDs. Keeps drawing IDs until
    we have more than num_samples, then truncate list at that number of samples.
    This way we approximate natural distribution of syllables from a random draw
    of songs while at the same time using a constant # of samples.

    In practice the way this function gets used by modelselect.select is that
    it's called one time to  get indices for a test set, along with a
    popped_songlist. Then it's called again repeatedly with the popped_songlist
    to get a random draw of indices/samples for training sets.

    Parameters
    ----------
    song_IDs : list of ints
        song ID for each sample in sample set. E.g., if
        sample_song_IDs[10:20]==31, that means all those
         samples in the set came from song #31
    labels : list of chars
        label for each sample in sample set. Used to verify that randomly
        drawn subset contains at least 2 examples of each label/class. This is necessary
        e.g. for using sklearn.StratifiedShuffleSplit
        (and just to not have totally imbalanced training sets).
    num_samples : int
        number of samples to return
    song_ID_list : list of ints
        ID numbers for songs from which to draw randomly.
        If None, all IDs from song_IDs are used, i.e., np.unique(song_IDs)
        Default is None.
    seed : int
        seed for random number generator.
        Default is None.
    return_popped_songlist : bool
        if True, return song_ID_list with IDs of songs assigned popped off.
        This is used when creating the test set so that the training set does not contain
        any songs in the test set.
    use_random_dot_org: bool
        if True, use random.org API to get truly random numbers for shuffling song IDs.
        default is False. When True, can slow down script if website hangs.

    Returns
    -------
    sample_IDs :

    song_ID_list_copy_to_pop :
    """

    song_IDs_arr = np.asarray(song_IDs)
    if song_ID_list is None:
        song_ID_list = np.unique(song_IDs).tolist()
    #make copy of list in case we need it back in loop below after popping off items
    song_ID_list_copy_to_pop = copy.deepcopy(song_ID_list)
    interwebz_random = RandomDotOrg() # access site that gives truly random #s from radio noise
    if use_random_dot_org:
        try:
            interwebz_random.shuffle(song_ID_list_copy_to_pop) # initial shuffle, happens in place
        except (HTTPError, URLError):  # i.e., if random service not available
            print('random.org service not working. Defaulting to Python random module')
            random.seed(seed)
            random.shuffle(song_ID_list_copy_to_pop)
    else:
        if seed:
            random.seed(seed)
        random.shuffle(song_ID_list_copy_to_pop)

    #outer while loop to make sure there's more than one sample for each class
    while 1: 
        sample_IDs = []
        while 1:
            curr_song_ID = song_ID_list_copy_to_pop.pop()
            curr_sample_IDs = np.argwhere(song_IDs_arr==curr_song_ID).ravel().tolist()
            sample_IDs.extend(curr_sample_IDs)
            if len(sample_IDs) > num_samples:
                sample_IDs = np.asarray(sample_IDs[:num_samples])
                break
        #if training set only contains one example of any syllable, get new set
        #(can't do c.v. stratified shuffle split with only one sample of a class
        #The sk-learn stratified shuffle split method is used by grid search for
        # the SVC.)
        temp_labels = np.asarray(labels)[sample_IDs]
        #in line below, just keep [1] cuz just want the counts
        uniq_label_counts = np.unique(temp_labels,return_counts=True)[1]
        if np.min(uniq_label_counts) > 1:
            break # but if not, break out of loop and keep current sample set
        else:
            print('training set only contains one example of any syllable')
            #get original list
            song_ID_list_copy_to_pop = copy.deepcopy(song_ID_list)
            # shuffle again so this time we hopefully get a set
            # where there's >1 sample of each syllable class
            if use_random_dot_org:
                try:
                    interwebz_random.shuffle(song_ID_list_copy_to_pop)  # initial shuffle, happens in place
                except (HTTPError, URLError):  # i.e., if random service not available
                    print('random.org service not working. Defaulting to Python random module')
                    random.seed(seed)
                    random.shuffle(song_ID_list_copy_to_pop)
            else:
                random.seed(seed)
                random.shuffle(song_ID_list_copy_to_pop)

    if return_popped_songlist:
        return sample_IDs, song_ID_list_copy_to_pop
    else:
        return sample_IDs


def get_acc_by_label(labels, pred_labels, labelset):
    """accuracy averaged across classes

    Parameters:
    labels : 1d numpy array
        vector of labels from a test data set
    pred_labels : 1d numpy array
        vector of predicted labels returned by algorithm given samples from test data set
    labelset : string
        set of unique labels from test data set, i.e., numpy.unique(labels)

    Returns:
    acc_by_label : 1d numpy array
        accuracy for each label / class
    avg_acc : scalar
        average accuracy across labels, i.e., numpy.mean(acc_by_label)
    """

    acc_by_label = np.zeros((len(labelset)))
    for ind,label in enumerate(labelset):
        label_ids = np.in1d(labels,label) #find all occurrences of label in test data
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