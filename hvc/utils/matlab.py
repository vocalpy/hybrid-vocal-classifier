import numpy as np
import scipy.io as scio # to load matlab files

def load_ftr_files(fname,ftr_file_type,purpose='train'):
    """
    Loads feature files created in matlab

    Argument:
        fname -- filename of .mat file
        ftr_file_type -- should be either 'knn' or 'svm'
        return_notmat_fnames -- logical, if 'True' then returns list of .cbin
        purpose -- 'train' or 'classify'. Features files to classify contain only
        samples since they by definition haven't been labeled.

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
    if ftr_file_type=='knn':
        FEATURES_TO_USE = np.r_[0:6,7:10,11] # using np._r to build index array 
        #load 'cell array' from .mat file. ftr_file is a dictionary of numpy record arrays
        #the 'feature_cell' record array has two columns: col 1 = actual vals, col 0 is just ftr names
        ftr_file = scio.loadmat(fname,chars_as_strings=True)
        # concatenate features horizontally (each row is a sample)
        samples = np.hstack(ftr_file['feature_cell'][1,:])
        samples = samples[:,FEATURES_TO_USE] #discard unused features

    elif ftr_file_type=='svm':
        ftr_file = scio.loadmat(fname)
        #loads 'cell array' from .mat file. ftr_file is a dictionary of numpy record arrays
        #the 'feature_cell' record array has two columns: col 1 = actual vals, col 0 is just ftr names
        samples = ftr_file['features_mat']

    if purpose=='classify':
        return samples

    elif purpose=='train':
        # flatten because matlab vectors are imported as 2d numpy arrays with one row
        labels = ftr_file['label_vec'].flatten()
        labels = labels.view(np.uint32) # convert from unicode to long 
        song_IDs_vec = ftr_file['song_IDs_vec'].flatten()
        return samples, labels, song_IDs_vec

