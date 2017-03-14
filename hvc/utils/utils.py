import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import wavfile
from scipy.signal import slepian # AKA DPSS, window used for FFT
from scipy.signal import spectrogram

from hvc.utils import sequences

class song_spect:
    """
    spectrogram object, returned by make_spect.
    Properties:
        spect -- 2-d m by n numpy array, spectrogram as computed by make_song_spect.
                 Each of the m rows is a frequency bin, and each of the n columns is a time bin.
        time_bins -- 1d vector, values are times represented by each bin in s
        freq_bins -- 1d vector, values are power spectral density in each frequency bin
        sampfreq -- sampling frequency in Hz as determined by scipy.io.wavfile function
    """
    def __init__(self,spect,freq_bins,time_bins,sampfreq):
        self.spect = spect
        self.freqBins = freq_bins
        self.timeBins = time_bins
        self.sampFreq = sampfreq

def make_spect(waveform,sampfreq,size=512,step=32,freq_cutoffs=[1000,8000]):
    """
    Computes spectogram of raw song waveform using FFT.
    Defaults to FFT parameters from Koumura Okanoya 2016.
    **Note that spectrogram is log transformed (base 10), and that
    both spectrogram and freq_bins are "flipped" (reflected across horizontal
    axis) so that when plotted the lower frequencies of the spectrogram are 
    at 0 on the y axis.

    Inputs:
        wav_file -- filename of .wav file corresponding to song
        size -- of FFT window, default is 512 samples
        step -- number of samples between the start of each window, default is 32
            i.e. if size == step then there will be no overlap of windows
        freq_range -- range of frequencies to return. Two-element list; frequencies
                      less than the first element or greater than the second are discarded.
    Returns:
        spect -- spectrogram, log transformed
        time_bins -- vector assigning time values to each column in spect
            e.g. [0,8,16] <-- 8 ms time bins
        freq_bins -- vector assigning frequency values to each row in spect
            e.g. [0,100,200] <-- 100 Hz frequency bins
    """
    win_dpss = slepian(size, 4/size)
    fft_overlap = size - step
    freq_bins, time_bins, spect = spectrogram(waveform,
                           sampfreq,
                           window=win_dpss,
                           nperseg=win_dpss.shape[0],
                           noverlap=fft_overlap)
    #below, I set freq_bins to >= freq_cutoffs 
    #so that Koumura default of [1000,8000] returns 112 freq. bins
    f_inds = np.nonzero((freq_bins >= freq_cutoffs[0]) & 
                        (freq_bins < freq_cutoffs[1]))[0] #returns tuple
    freq_bins = freq_bins[f_inds]
    spect = spect[f_inds,:]
    spect = np.log10(spect) # log transform to increase range

    #flip spect and freq_bins so lowest frequency is at 0 on y axis when plotted
    spect = np.flipud(spect)
    freq_bins = np.flipud(freq_bins)
    spect_obj = song_spect(spect,freq_bins,time_bins,sampfreq)
    return spect_obj
    
def compute_amp(spect):
    """
    compute amplitude of spectrogram
    Assumes the values for frequencies are power spectral density (PSD).
    Sums PSD for each time bin, i.e. in each column.
    Inputs:
        spect -- output from spect_from_song
    Returns:
        amp -- amplitude
    """

    return np.sum(spect,axis=0)

def segment_song(amp,time_bins,threshold=5000,min_syl_dur=0.02,min_silent_dur=0.002):
    """
    Divides songs into segments based on threshold crossings of amplitude.
    Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
    Inputs:
        amp -- amplitude of power spectral density. Returned by compute_amp.
        time_bins -- time in s, must be same length as log amp. Returned by make_song_spect.
        threshold -- value above which amplitude is considered part of a segment. default is 5000.
        min_syl_dur -- minimum duration of a syllable. default is 0.02, i.e. 20 ms.
        min_silent_dur -- minimum duration of silent gap between syllables. default is 0.002, i.e. 2 ms.
    Returns:
        onsets, offsets -- arrays of onsets and offsets of segments.
        So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
        To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """
    above_th = amp > threshold
    h = [1, -1] 
    above_th_convoluted = np.convolve(h,above_th) # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    onsets = time_bins[np.nonzero(above_th_convoluted > 0)]
    # and -1 whenever above_th changes from 1 to 0
    offsets = time_bins[np.nonzero(above_th_convoluted < 0)]
    
    #get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1] # duration of silent gaps
    keep_these = np.nonzero(silent_gap_durs > min_silent_dur)
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]
    
    #eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    keep_these = np.nonzero(syl_durs > min_syl_dur)
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]    
    
    return onsets,offsets
    
def parse_xml(xml_file,concat_seqs_into_songs=False):
    """
    parses Annotation.xml files.
    Inputs:
        xml_file -- string, filename of .xml file, e.g. 'Annotation.xml'
        concat_seqs_into_songs -- if True, concatenate sequences into songs,
                                  where each wav file is a song. Default is False.
    Returns:
        seq_list -- a list of Sequence objects. See sequences.py for properties.
    """

    tree = ET.ElementTree(file=xml_file)
    seq_list = []
    for seq in tree.iter(tag='Sequence'):
        wav_file = seq.find('WaveFileName').text
        position = int(seq.find('Position').text)
        length = int(seq.find('Length').text)
        syl_list = []
        for syl in seq.iter(tag='Note'):
            syl_position = int(syl.find('Position').text)
            syl_length = int(syl.find('Length').text)
            label = syl.find('Label').text
            
            syl_obj = sequences.Syllable(position = syl_position,
                                             length = syl_length,
                                             label = label)
            syl_list.append(syl_obj)
        seq_obj = sequences.Sequence(wav_file = wav_file,
                                         position = position,
                                         length = length,
                                         syl_list = syl_list)
        seq_list.append(seq_obj)
        
    if concat_seqs_into_songs:
        song_list = []
        curr_wavFile = seq_list[0].wavFile
        new_seq_obj = seq_list[0]

        for seq in seq_list[1:]:
            if seq.wavFile == curr_wavFile:
                new_seq_obj.length += seq.length
                new_seq_obj.numSyls += seq.numSyls
                new_syls = []
                for syl in seq.syls:
                    syl.position += seq.position
                new_seq_obj.syls += seq.syls

            else:
                song_list.append(new_seq_obj)
                new_seq_obj = seq
                curr_wavFile = seq.wavFile
        
        return song_list
                        
    else:    
        return seq_list

class resequencer():
    """
    Computes most likely sequence of labels given observation probabilities
    at each time step in sequence and a second-order transition probability
    matrix taken from training data.
    
    Uses a Viterbi-like dynamic programming algorithm. (Viterbi-like because
    the observation probabilities are not strictly speaking the emission
    probabilities from hidden states but instead are outputs from some machine
    learning model, e.g., the softmax layer of a DCNN that assigns a probability
    to each label at each time step.)
    
    This is a Python implementation of the algorithm from Koumura Okanoya 2016.
    See "compLabelSequence" in: 
        https://github.com/cycentum/birdsong-recognition/blob/master/
        birdsong-recognition/src/computation/ViterbiSequencer.java
        
        
    Parameters
    ----------
    sequences : list of strings
        Each string represents a sequence of syllables
    observation_prob : ndarray
        n x m x p matrix, n sequences of m estimated probabilities for p classes
    
    transition_prob : ndarray
        second-order transition matrix, n x m x p matrix where the value at
        [n,m,p] is the probability of transitioning to labels[p] at time step
        t given that labels[m] was observed at t-1 and labels[n] was observed
        at t-2
    
    labels : list of chars
        Contains all unique labels used to label songs being resequenced

    Returns
    -------
    resequenced : list of strings
        Each string represents the sequence of syllables after resequencing. So
        e.g. resequenced[0] is sequences[0] after running through the algorithm.
    """

    def __init__(self,transition_probs,labels):
        self.transition_probs = transition_probs
        self.labels = labels
        self.num_labels = len(labels)
        self.destination_label_ind = range(0,self.num_labels)

        # num_states calculation: +1 for 'e' state at beginning of initial states
        # * number of labels (now without 'e') and + 1 for the final 'tail' state
        self.num_states = (self.num_labels + 1) * self.num_labels + 1
        # create dict of lists used to determine 'destination' state
        # given source state (key) and emitted label (index into each list)
        # i..e if source state is 5, destination_state{5} will return a list as
        # long as the number of labels, indexing into that list with e.g. index 4
        # will return some state number that then becomes the destination state
        self.destination_states = {}  # so len(destination_states.keys()) == num_states
        dst_state_ctr = 0
        for label_one in range(0,self.num_labels+1):  # +1 for 'e' state
            for label_two in range(0,self.num_labels):  # now without e
                dest_label_one = label_two
                dest_state_list = []
                for dest_label_two in range(0,self.num_labels):
                    dest_state_list.append(
                        dest_label_one * self.num_labels + dest_label_two)
                self.destination_states[dst_state_ctr] = dest_state_list
                dst_state_ctr += 1
        # + 1 for the final tail states
        dest_label_one = self.num_labels
        dest_state_list = []
        for dest_label_two in range(0,self.num_labels):
            dest_state_list.append(
                dest_label_one * self.num_labels + dest_label_two)
        self.destination_states[dst_state_ctr] = dest_state_list
        # number of tail states = num_states because any state can transition to
        # a tail state and the tail state is non-emitting
        self.tail_states = list(range(0,self.num_states))

        self.head_state = self.num_states - 1  # last state in list is head state
        #prob. of tranisitioning from head state 'e' to any state
        #'e1)','e2',...'e(N-1)' where N is number of labels is equal for all
        #initial states.
        self.initial_transition_prob = 1.0 / self.num_labels  # 1.0 because 1 / anything = 0

    def resequence(self,observation_probs):
        num_time_steps = observation_probs.shape[0] - 1
        source_states = []
        for time_step in range(num_time_steps):
            source_states.append(np.zeros((self.num_states,),dtype=int))
        
        # initial inductive step of Viterbi
        current_score = np.ones((self.num_states,)) * -np.inf

        #use dest_labl_id to index into observation_prob array
        for dest_labl_id in self.destination_label_ind:
            # need destination state to assign it a score in the
            # next_score array
            dest_state = self.destination_states[self.head_state][dest_labl_id]
            obsv_prob = observation_probs[0,dest_labl_id]  # row 0 = 1st time step
            current_score[dest_state] = \
                np.log(self.initial_transition_prob) + np.log(obsv_prob)

        # main loop for Viterbi
        for time_step in range(num_time_steps):
            next_score = np.ones((self.num_states,)) * -np.inf
            for source_state in range(self.num_states):
                for dest_label_ind in self.destination_label_ind:
                    # need destination state to assign it a score in the
                    # next_score array
                    dest_state = \
                        self.destination_states[source_state][dest_label_ind]

                    label_one = source_state // self.num_labels  # floor division
                    if label_one == self.num_labels or source_state == self.head_state:
                        trans_prob = self.initial_transition_prob
                    else:
                        label_two = source_state % self.num_labels
                        trans_prob = self.transition_probs[label_one,
                                                           label_two,
                                                           dest_label_ind]
                    ob_prob = observation_probs[time_step+1][dest_label_ind]
                    tmp_next_score = current_score[source_state] + \
                                     np.log(trans_prob) + \
                                     np.log(ob_prob)
                    if tmp_next_score >= next_score[dest_state]:
                        next_score[dest_state] = tmp_next_score
                        source_states[time_step][dest_state] = source_state    
            tmp = current_score
            current_score = next_score
            next_score = tmp
        
        # retrieve best state sequence in reverse using scores directly
        current_state = -1
        
        # initial step to get best state
        for state in self.tail_states:
            if current_state == -1 or current_score[state] > current_score[current_state]:
                current_state = state
        resequenced = []
        # loop through len-2 because we already figured out last element at -1
        for time_step in range((len(observation_probs)-2),-1,-1):
            previous_state = source_states[time_step][current_state]
            source_label = -1

            possible_dest_states = self.destination_states[previous_state]
            for d in range(len(possible_dest_states)):
                if possible_dest_states[d] == current_state:
                    source_label_ind = self.destination_label_ind[d]
                    source_label = self.labels[source_label_ind]
                    break
            resequenced.append(source_label)
            current_state = previous_state
        
        previous_state = self.head_state
        source_label = -1
        possible_dest_states = self.destination_states[previous_state]
        for d in range(len(possible_dest_states)):
            if possible_dest_states[d] == current_state:
                source_label_ind = self.destination_label_ind[d]
                source_label = self.labels[source_label_ind]
                break
        resequenced.append(source_label)
        resequenced.reverse()
        return resequenced

def get_trans_mat(seqs,smoothing_constant=1e-4):
    """
    calculate second-order transition matrix given sequences of syllable labels
    
    Parameters
    ----------
    seqs : list of Sequence objects
    
    smoothing_constant : float
        default is 1e-4. Added to all probabilities so that none are zero.
        Mathematically convenient for computing Viterbi algorithm with
        exponential.
    
    Returns
    -------
    labels : 1-d array of ints
        set of unique labels across all Sequences.
    
    trans_mat : 3-d array
        Shape is n * n * n where n is the number of labels.
        trans_mat[i,j,k] is the probability of transitioning to labels[k]
        at time step t, given that label at time step t-1 was labels[k]
        and the label at time step t-2 was labels[i].
    """
    
    all_syls = [syl.label for seq in seqs for syl in seq.syls]
    labels = np.unique(all_syls)

    all_label_seqs = []
    for seq in seqs:
        all_label_seqs.append([syl.label for syl in seq.syls])

    num_labels = labels.shape[0]
    counts = np.zeros((num_labels,num_labels,num_labels))
    for label_seq in all_label_seqs:
        for ind in range(2,len(label_seq)):
            k = np.where(labels==label_seq[ind])
            j = np.where(labels==label_seq[ind-1])
            i = np.where(labels==label_seq[ind-2])
            counts[i,j,k] += 1
    trans_mat = np.zeros(counts.shape)
    for i in range(num_labels):
        for j in range(num_labels):
            num_ij_occurences = np.sum(counts[i,j,:])
            if num_ij_occurences > 0:
                for k in range(num_labels):
                    trans_mat[i,j,k] = counts[i,j,k] / num_ij_occurences

    if smoothing_constant:
        for i in range(num_labels):
            for j in range(num_labels):
                trans_mat[i,j,:] += smoothing_constant
                trans_mat[i,j,:] /= np.sum(trans_mat[i,j,:])

    return trans_mat
