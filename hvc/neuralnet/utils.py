import numpy as np
from keras.utils.np_utils import to_categorical

def convert_labels_categorical(labelset,labels):
    """convert array of labels to matrix of one-hot vectors
    where each vector is the training label for a neural net.
    Can then be supplied as output to conditional crossentropy
    layer. Uses keras.utilts.np_utils.to_categorical

    Parameters
    ----------
    labelset : str
    labels : vector

    Returns
    -------
    labels_categorical : m x n 2d numpy array
    """

    # reshape labels so they match output for neural net
    num_syl_classes = np.size(labelset)
    # make a dictionary that maps labels to classes 0 to n-1 where n is number of
    # classes of syllables.
    # Need this map instead of e.g. converting from char to int because
    # keras to_categorical function requires
    # input where classes are labeled from 0 to n-1
    classes_zero_to_n = range(num_syl_classes)
    label_map = dict(zip(labelset, classes_zero_to_n))
    labels_zero_to_n = np.asarray([label_map[label] for label in labels])
    # so we can then convert to array of binary / one-hot vectors for training
    return to_categorical(labels_zero_to_n, num_syl_classes)

def reshape_spects(spects):
    """reshape spectrogram data so it works as input to neural net
    """

    spects = np.stack(train_syl_spects[:], axis=0)
    return np.expand_dims(train_syl_spects, axis=1)
