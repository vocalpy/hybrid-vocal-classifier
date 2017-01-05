### Note this is Python 2.7, because Theano ***
from __future__ import division

import pdb
import os
import glob

import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger

import hvc # used as a proper noun

#constats for spectrogram
SAMP_FREQ = 32000 # Hz
WINDOW_SIZE= 512
WINDOW_STEP= 32
FREQ_CUTOFFS=[1000,8000]

#constants for training
NUM_TRAIN_SAMPLES = 128

WAVE_DIR = 'C:\DATA\koumura birds\Bird0\Wave'
ANNOTATION_FILE = 'C:\DATA\koumura birds\Bird0\Annotation.xml'

song_list = hvc.utils.parse_xml(ANNOTATION_FILE,concat_seqs_into_songs=True)
all_syls = [syl.label for song in song_list for syl in song.syls]
uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)
if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
    raise ValueError("One class of syllable occurs oimprders of magnitude less frequently than the others")

bigrams = []
for song in song_list:
    song_syls = [syl.label for syl in song.syls]
    ind = 0
    while ind < len(song_syls) - 1:
        bigrams.append(song_syls[ind:ind+2])
        ind += 1
bigrams = np.asarray(bigrams)
ncols = bigrams.shape[1]
dtype = bigrams.dtype.descr * ncols
struct = bigrams.view(dtype)
uniq_bigrams, counts = np.unique(struct,return_counts=True)
uniq_bigrams = uniq_bigrams.view(bigrams.dtype).reshape(-1, ncols)
sort_ids = np.argsort(counts)
bigrams_sorted = uniq_bigrams[sort_ids] # from least to most common
counts_sorted = np.sort(counts)
start_ind = bigrams_sorted.shape[0] - 1
#bigram_pair_scores = []

#find the two most frequent bigrams that share a common label
#while start_ind > 0:
bigram1 = bigrams_sorted[start_ind]
next_ind = start_ind -1
while next_ind > -1:
    bigram2 = bigrams_sorted[next_ind]
    if np.intersect1d(bigram1,bigram2).shape[0] > 0:
#            bigram_pair_scores.append([start_ind,
#                                       next_ind,
#                                       counts_sorted[start_ind] + counts_sorted[next_ind]])    
        break
    next_ind -= 1
#    if np.sum(counts_sorted[:start_ind+1]) < np.asarray(bigram_pair_scores)[-1,2]:
#        break
#    start_ind -= 1


#given that there's only one sampling frequency, use it to figure out the number of time bins in the
#fixed length spectrogram into which the sequences will be padded
timebin_size_in_s = WINDOW_STEP / SAMP_FREQ # for default, 32 / 32000 = 0.001 s, i.e. 1 ms

#print('Shuffling sequences')
## shuffle sequences before splitting into training and test sets
## get ids in variable instead of shuffling array in place
## so we can keep a record of which sequences were in training / test set.
#RANDOM_SEED = 42  # for reproducibility and to make a Douglas Adams reference
#while 1:
#    np.random.seed(RANDOM_SEED) 
#    rand_seq_ids = np.random.permutation(len(seq_list))
#    shuffled_seq_list = [seq_list[seq_id] for seq_id in rand_seq_ids]
#    all_train_syls = [int(syl.label) for seq in seq_list for syl in seq.syls]
#    uniq_train_syls, syl_counts = np.unique(all_train_syls,return_counts=True)
#    if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
#        raise ValueError("One class of syllable occurs orders of magnitude less frequently than others in training set")  
#    if not np.setdiff1d(uniq_train_syls,uniq_syls):
#        #i.e., if all syllable classes are in the training set
#        break  # because we don't need to shuffle the training set again

## add sequences to training set until just under cut-off duration
##train_set_duration_in_samples = TRAIN_SET_DURATION_IN_SECONDS * SAMP_FREQ
#train_seqs = []
##curr_train_set_dur = 0
##for seq in shuffled_seq_list:
##    if curr_train_set_dur + seq.length > train_set_duration_in_samples:
##        pass
##    else:
##        curr_train_set_dur += seq.length
#while len(train_seqs) < NUM_TRAIN_SAMPLES:
#    train_seqs.append(shuffled_seq_list.pop())

# need to pack vectors with labels for each time bin in the padded spectrogram.
# These label vectors are used by gradient descent to get error of network output. 
#keras requires all labels be positive integers to convert to boolean array for conditional cross entropy
#so assign label for "silent gap" between syllables a label that is max. label number + 1
#i.e. len(uniq_train_syls)
silent_gap_label = len(uniq_syls)
num_syl_classes = len(uniq_syls)+1

os.chdir(WAVE_DIR)
bigram_spects = []
bigram_label_vecs = []
for song in song_list:
    print('extracting bigrams from song {}'.format(song.wavFile))
    [sampfreq, wav] = wavfile.read(song.wavFile)
    if sampfreq != SAMP_FREQ:
        raise ValueError(
            'Sampling frequency for {}, {}, does not match expected sampling frequency of {}'.format(seq.wavFile,
                                                                                                     sampfreq,
                                                                                                     SAMP_FREQ))
    song_wav = wav[song.position:(song.position+song.length)]
    spect_obj = hvc.utils.make_spect(song_wav,sampfreq,
                                        size=WINDOW_SIZE,
                                        step=WINDOW_STEP,
                                        freq_cutoffs=FREQ_CUTOFFS)
    spect = spect_obj.spect
     
    #generate vector of training labels
    time_bins = spect_obj.timeBins
    label_vec = np.ones((time_bins.shape[0],1),dtype=int) * silent_gap_label
    labels = [int(syl.label) for syl in song.syls]
    onsets = [syl.position / SAMP_FREQ for syl in song.syls]
    offsets = [(syl.position + syl.length) / SAMP_FREQ for syl in song.syls]
    onset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - onset)) for onset in onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - offset)) for offset in offsets]
    for onset, offset, label in zip(onset_IDs_in_time_bins, offset_IDs_in_time_bins,labels):
        label_vec[onset:offset+1] = label

    #now find bigrams in labels and extract those parts of spect
    ind = 0
    while ind < len(labels) - 1:
        curr_song_bigram = labels[ind:ind+2]
        if np.array_equal(bigram1,np.asarray(curr_song_bigram)):
            if ind==0:
                #end just before onset of syllable after bigram
                tmp_spect = spect[:,0:onset_IDs_in_time_bins[2]-1]
                tmp_label_vec = label_vec[0:onset_IDs_in_time_bins[2]-1]
            elif ind >0 and ind < len(labels) - 3:
                try:
                    tmp_spect = spect[:,offset_IDs_in_time_bins[ind-1]+1:onset_IDs_in_time_bins[ind+2]-1]
                    tmp_label_vec = label_vec[offset_IDs_in_time_bins[ind-1]+1:onset_IDs_in_time_bins[ind+2]-1]
                except IndexError:
                    pdb.set_trace()
            elif ind == len(labels) - 2:
                tmp_spect = spect[:,offset_IDs_in_time_bins[ind-1]+1:]
                tmp_label_vec = label_vec[offset_IDs_in_time_bins[ind-1]+1:]
            bigram_spects.append(tmp_spect)
            bigram_label_vecs.append(tmp_label_vec)
        ind += 1
                                            
# need to zero pad spectrogram so they are all the same length
# First figure out max length
spect_lengths = [spect.shape[1] for spect in bigram_spects]
num_timebins_for_max_spect = np.max(spect_lengths)

# Also need to know number of rows, i.e. freqbins.
# Will be the same for all spects since we used the same FFT params for all.
# freqBins size is also input shape to LSTM net
# (since at each time point the input is one column of spectrogram)
freqBins_size = bigram_spects[0].shape[0]

bigram_spects_padded = []
bigram_labels_padded = []
counter = 0
for spect,label_vec in zip(bigram_spects,bigram_label_vecs):
    counter += 1
    print("Padding spect + label vec {}.".format(counter))
    curr_padded_spect = np.zeros((freqBins_size,num_timebins_for_max_spect))
    last_col_id = spect.shape[1]
    curr_padded_spect[:,:last_col_id] = spect
    bigram_spects_padded.append(curr_padded_spect)
        
    curr_padded_label_vec = np.ones((num_timebins_for_max_spect,1),dtype=int) * silent_gap_label
    curr_padded_label_vec[:last_col_id] = label_vec
    curr_padded_label_vec = to_categorical(curr_padded_label_vec,num_syl_classes)
    bigram_labels_padded.append(curr_padded_label_vec)

#scale all spects by mean and std of training set
spect_scaler = StandardScaler()
# concatenate all spects then transpose so Hz bins are 'features'
spect_scaler.fit(np.hstack(bigram_spects_padded[:]).T)
# now scale each individual training spect
for ind, spect in enumerate(bigram_spects_padded):
    bigram_spects_padded[ind] = np.transpose(spect_scaler.transform(spect.T))

input_shape = (num_timebins_for_max_spect,freqBins_size)
LSTM_model = hvc.models.naive_LSTM(input_shape=input_shape,num_syllable_classes=num_syl_classes) # +1 for silent gap label

#reshape training data for model
bigram_spects_padded = np.dstack(bigram_spects_padded[:])
x,y,n = bigram_spects_padded.shape
bigram_spects_padded = bigram_spects_padded.reshape(n,y,x)

bigram_labels_padded = np.dstack(bigram_labels_padded[:])
x,y,n = bigram_labels_padded.shape
bigram_labels_padded = bigram_labels_padded.reshape(n,x,y)

print('Shuffling bigrams')
# shuffle and split into training and test sets
RANDOM_SEED = 42 
np.random.seed(RANDOM_SEED) 
shuffle_ids = np.random.permutation(n)  # n is from bigram_spects_padded.shape
bigram_spects_padded = bigram_spects_padded[shuffle_ids,:,:]
bigram_labels_padded = bigram_labels_padded[shuffle_ids,:,:]

train_spects = bigram_spects_padded[:NUM_TRAIN_SAMPLES,:,:]
train_labels = bigram_labels_padded[:NUM_TRAIN_SAMPLES,:,:]

#set up to save after each epoch
csv_logger = CSVLogger('C://DATA//koumura birds/Bird0//LSTM_bigrams_Bird0_run1.log',
                       separator=',',
                       append=True)
callbacks_list = [csv_logger]


LSTM_model.fit(train_spects,
               train_labels,
               batch_size=32,
               nb_epoch=10,
               callbacks=callbacks_list,
               verbose=1,
               )
