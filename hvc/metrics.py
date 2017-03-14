import numpy as np

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
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def average_accuracy(true_labels,pred_labels,labelset):
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
    for ind,label in enumerate(labelset):
        label_ids = np.in1d(true_labels,label) #find all occurences of label in test data
        if sum(label_ids) == 0: # if there were no instances of label in labels
            continue
        import pdb;pdb.set_trace()
        pred_for_that_label = pred_labels[label_ids]
        matches = pred_for_that_label==label
        #sum(matches) is equal to number of true positives
        #len(matches) is equal to number of true positives and false negatives
        acc = sum(matches) / len(matches)
        acc_by_label[ind] = acc
    avg_acc = np.mean(acc_by_label)
    return acc_by_label,avg_acc

def make_frame_vecs(seqs):
    frame_vecs = []
    for seq in seqs:
        frame_vec = np.ones((seq.length,),dtype=int) * -1
        for syl in seq.syls:
            label = ord(syl.label)
            frame_vec[syl.position:syl.position+syl.length] = label    
        frame_vecs.append(frame_vec)
    return frame_vecs

def frame_error(true_seqs,pred_seqs):
    """
    computes error rate for every frame
    
    Parameters
    ----------
    true_seqs : list of Sequence objects
        ground truth
    
    pred_seqs : list of Sequence objects
        predicted sequences returned by model

    Returns
    -------
    frame_error : scalar
        correctly classified frames / total number of frames
    """
    
    if len(true_seqs) != len(pred_seqs):
        raise ValueError('Number of true and predicted Sequences'
                         ' does not match')

    true_seqs_frame_vecs = make_frame_vecs(true_seqs)
    pred_seqs_frame_vecs = make_frame_vecs(pred_seqs)
    
    correct_frames = []
    total_frames = []
    
    for true_vec,pred_vec in zip(true_seqs_frame_vecs,pred_seqs_frame_vecs):
        correct_frames.append(np.sum(np.equal(true_vec,pred_vec)))
        total_frames.append(true_vec.shape[0])

    correct_frames = sum(correct_frames)
    total_frames = sum(total_frames)
    return 1 - (float(correct_frames) / float(total_frames))
