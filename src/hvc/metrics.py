import numpy as np
import scipy.spatial.distance
import sklearn.metrics
import joblib


def confusion_matrix(y_true=None, y_pred=None, model_meta_file=None, normalize=False):
    """generates a confusion matrix, given true and predicted labels
    wrapper around sklearn.metrics.confusion_matrix

    Parameters:
    -----------
    y_true : ndarray
        actual labels from data

    y_pred : ndarray
        labels predicted by a classifier

    model_meta_file : str
        filename of a .meta file generated by hvc.select that will
        contain test_labels and pred_labels, i.e. the labels from
        a test set and the predicted labels for that test set.
        A confusion matrix is generated using test_labels as y_true
        and pred_labels as y_pred.

    Returns
    -------
    cm : ndarray
        confusion matrix
    """

    if (y_true is not None and y_pred is not None) and model_meta_file is not None:
        raise ValueError(
            "arguments must be either y_true and y_pred, " "or meta_file, but not both"
        )

    if model_meta_file:
        meta = joblib.load(model_meta_file)
        y_true = meta["test_labels"]
        y_pred = meta["pred_labels"]

    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    return cm


def lev_np(source, target):
    """
    Levenshtein distance measured using numpy
    from:
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/
    Levenshtein_distance#Python

    Used under Creative Commons Attribution-ShareAlike License.

    Parameters:
    -----------
    source : string
    target : string

    Returns:
    --------
    Levenshtein distance : integer
    """
    if len(source) < len(target):
        return lev_np(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:], np.add(previous_row[:-1], target != s)
        )

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def average_accuracy(true_labels, pred_labels, labelset):
    """
    computes accuracy averaged across classes

    Parameters
    ----------
    true_labels : list of strings
        ground truth, correct labels used to calculate error

    pred_labels : list of strings
        vector of predicted labels returned by algorithm given samples from test data set

    labelset : list of chars
        set of unique labels from data set, i.e., numpy.unique(true_labels)

    Returns
    -------
    acc_by_label : nd_array
        1-d vector of accuracies
    avg_acc : scalar
        average accuracy across labels, i.e., numpy.mean(acc_by_label)
    """

    acc_by_label = np.zeros((len(labelset)))
    for ind, label in enumerate(labelset):
        label_ids = np.in1d(
            true_labels, label
        )  # find all occurences of label in test data
        if sum(label_ids) == 0:  # if there were no instances of label in labels
            continue
        pred_for_that_label = pred_labels[label_ids]
        matches = pred_for_that_label == label
        # sum(matches) is equal to number of true positives
        # len(matches) is equal to number of true positives and false negatives
        acc = sum(matches) / len(matches)
        acc_by_label[ind] = acc
    avg_acc = np.mean(acc_by_label)
    return acc_by_label, avg_acc


def frame_error(y_true, y_pred):
    """
    computes error rate for every frame
    equivalent to "note and timing error rate" in Koumura Okanoya 2016

    Parameters
    ----------
    y_true : 1-dimensional numpy array
        ground truth
    y_pred : 1-dimensional numpy array
        prediction, output of some model

    Returns
    -------
    frame_error : scalar
        1 - (correctly classified frames / total number of frames)
    """

    if y_true.ndim > 1:
        raise ValueError(
            "frame_error only defined for 1-dimensional inputs"
            " but y_true.ndim is {}".format(y_true.ndim)
        )

    if y_pred.ndim > 1:
        raise ValueError(
            "frame_error only defined for 1-dimensional inputs"
            " but y_pred.ndim is {}".format(y_pred.ndim)
        )

    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError(
            "y_true and y_pred should have the same length."
            "y_true.shape is {} and y_pred.shape is {}".format(
                y_true.shape, y_pred.shape
            )
        )

    return 1 - sum(y_true == y_pred) / y_true.shape[-1]


def hamming_dist(y_true, y_pred):
    """Hamming distance. Number of substitutions required to convert y_pred to y_true (or vice versa).
    Just a wrapper around scipy.spatial.distance.hamming.

    Parameters
    ----------
    y_true : 1-dimensional numpy array
        ground truth
    y_pred : 1-dimensional numpy array
        prediction, output of some model

    Returns
    -------
    hamming : scalar
    """

    # redundant code copied and pasted from frame error rate
    # better than overhead of having some error-checking function / making these classes?
    if y_true.ndim > 1:
        raise ValueError(
            "hamming_dist only defined for 1-dimensional inputs"
            " but y_true.ndim is {}".format(y_true.ndim)
        )

    if y_pred.ndim > 1:
        raise ValueError(
            "hamming_dist only defined for 1-dimensional inputs"
            " but y_pred.ndim is {}".format(y_pred.ndim)
        )

    if y_true.shape[-1] != y_pred.shape[-1]:
        raise ValueError(
            "y_true and y_pred should have the same length."
            "y_true.shape is {} and y_pred.shape is {}".format(
                y_true.shape, y_pred.shape
            )
        )

    return scipy.spatial.distance.hamming(y_true, y_pred)
