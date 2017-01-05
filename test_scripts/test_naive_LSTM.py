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
#TRAIN_SET_DURATION_IN_SECONDS = 120 # seconds, i.e. two minutes
MAX_SPECT_LENGTH_IN_SECONDS = 8
TRAIN_SET_DURATION_IN_SECONDS = 120 # seconds, i.e. two minutes

WAVE_DIR = 'C:\DATA\koumura birds\Bird0\Wave'
ANNOTATION_FILE = 'C:\DATA\koumura birds\Bird0\Annotation.xml'

seq_list = hvc.utils.parse_xml(ANNOTATION_FILE,concat_seqs_into_songs=True)
all_syls = [syl.label for seq in seq_list for syl in seq.syls]
uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)
if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
    raise ValueError("One class of syllable occurs orders of magnitude less frequently than the others")


#given that there's only one sampling frequency, use it to figure out the number of time bins in the
#fixed length spectrogram into which the sequences will be padded
timebin_size_in_s = WINDOW_STEP / SAMP_FREQ # for default, 32 / 32000 = 0.001 s, i.e. 1 ms

print('Shuffling sequences')
# shuffle sequences before splitting into training and test sets
# get ids in variable instead of shuffling array in place
# so we can keep a record of which sequences were in training / test set.
RANDOM_SEED = 42  # for reproducibility and to make a Douglas Adams reference
while 1:
    np.random.seed(RANDOM_SEED) 
    rand_seq_ids = np.random.permutation(len(seq_list))
    shuffled_seq_list = [seq_list[seq_id] for seq_id in rand_seq_ids]
    all_train_syls = [int(syl.label) for seq in seq_list for syl in seq.syls]
    uniq_train_syls, syl_counts = np.unique(all_train_syls,return_counts=True)
    if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
        raise ValueError("One class of syllable occurs orders of magnitude less frequently than others in training set")  
    if not np.setdiff1d(uniq_train_syls,uniq_syls):
        #i.e., if all syllable classes are in the training set
        break  # because we don't need to shuffle the training set again

# add sequences to training set until just under cut-off duration
train_set_duration_in_samples = TRAIN_SET_DURATION_IN_SECONDS * SAMP_FREQ
train_seqs = []
curr_train_set_dur = 0
for seq in shuffled_seq_list:
    if curr_train_set_dur + seq.length > train_set_duration_in_samples:
        pass
    else:
        train_seqs.append(seq)
        curr_train_set_dur += seq.length

os.chdir(WAVE_DIR)
for seq_numbr, seq in enumerate(train_seqs):
    print("Generating spectrogram for training sequence {}".format(seq_numbr))
    [sampfreq, wav] = wavfile.read(seq.wavFile)
    if sampfreq != SAMP_FREQ:
        raise ValueError(
            'Sampling frequency for {}, {}, does not match expected sampling frequency of {}'.format(seq.wavFile,
                                                                                                     sampfreq,
                                                                                                     SAMP_FREQ))
    seq_wav = wav[seq.position:(seq.position+seq.length)]
    seq.seqSpect = hvc.utils.make_spect(seq_wav,sampfreq,
                                        size=WINDOW_SIZE,
                                        step=WINDOW_STEP,
                                        freq_cutoffs=FREQ_CUTOFFS)
                                            
# need to zero pad spectrogram so they are all the same length
# First figure out max length
spect_lengths = [seq.seqSpect.timeBins.shape[0] for seq in train_seqs]
num_timebins_for_max_spect = np.max(spect_lengths)

# Also need to know number of rows, i.e. freqbins.
# Will be the same for all spects since we used the same FFT params for all.
# freqBins size is also input shape to LSTM net
# (since at each time point the input is one column of spectrogram)
freqBins_size = len(train_seqs[0].seqSpect.freqBins)

# also need to pack vectors with labels for each time bin in the padded spectrogram.
# These label vectors are used by gradient descent to get error of network output. 
#keras requires all labels be positive integers to convert to boolean array for conditional cross entropy
#so assign label for "silent gap" between syllables a label that is max. label number + 1
#i.e. len(uniq_train_syls)
silent_gap_label = len(uniq_syls)
num_syl_classes = len(uniq_syls)+1

train_label_vec = np.ones((num_timebins_for_max_spect,1),dtype=int) * silent_gap_label
train_spects_padded = []
train_labels_padded = []
for seq in train_seqs:
    print("Padding spect for: {}".format(seq))
    curr_seq_padded_spect = np.zeros((freqBins_size,num_timebins_for_max_spect))
    last_col_id = seq.seqSpect.timeBins.shape[0]
    curr_seq_padded_spect[:,:last_col_id] = seq.seqSpect.spect
    train_spects_padded.append(curr_seq_padded_spect)
        
    #generate vector of training labels
    curr_seq_padded_label_vec = np.ones((num_timebins_for_max_spect,1),dtype=int) * silent_gap_label
    labels = [int(syl.label) for syl in seq.syls]
    onsets = [syl.position / seq.seqSpect.sampFreq for syl in seq.syls]
    offsets = [(syl.position + syl.length) / seq.seqSpect.sampFreq for syl in seq.syls]
    time_bins = seq.seqSpect.timeBins
    onset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - onset)) for onset in onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - offset)) for offset in offsets]
    for onset, offset, label in zip(onset_IDs_in_time_bins, offset_IDs_in_time_bins,labels):
        curr_seq_padded_label_vec[onset:offset+1] = label
    curr_seq_padded_label_vec = to_categorical(curr_seq_padded_label_vec,num_syl_classes)
    train_labels_padded.append(curr_seq_padded_label_vec)

#scale all spects by mean and std of training set
spect_scaler = StandardScaler()
# concatenate all spects then transpose so Hz bins are 'features'
spect_scaler.fit(np.hstack(train_spects_padded[:]).T)
# now scale each individual training spect
for ind, spect in enumerate(train_spects_padded):
    train_spects_padded[ind] = np.transpose(spect_scaler.transform(spect.T))

input_shape = (num_timebins_for_max_spect,freqBins_size)
LSTM_model = hvc.models.naive_LSTM(input_shape=input_shape,num_syllable_classes=num_syl_classes) # +1 for silent gap label

#reshape training data for model
train_spects_padded = np.dstack(train_spects_padded[:])
x,y,n = train_spects_padded.shape
train_spects_padded = train_spects_padded.reshape(n,y,x)

train_labels_padded = np.dstack(train_labels_padded[:])
x,y,n = train_labels_padded.shape
train_labels_padded = train_labels_padded.reshape(n,x,y)

#set up to save after each epoch
csv_logger = CSVLogger('C://DATA//koumura birds/Bird0//LSTM_whole_spect_Bird0_run1.log',
                       separator=',',
                       append=True)
callbacks_list = [csv_logger]

LSTM_model.fit(train_spects_padded,
               train_labels_padded,
               batch_size=4,
               nb_epoch=10,
               callbacks=callbacks_list,
               verbose=1,
               )
