#!/usr/bin/env python
from __future__ import division

import pdb
import os
import glob
from collections import OrderedDict

import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger
import caffe

import hvc # used as a proper noun

#constats for spectrogram
SAMP_FREQ = 32000 # Hz
WINDOW_SIZE= 512
WINDOW_STEP= 32
FREQ_CUTOFFS=[1000,8000]

#constants for training
MAX_SPECT_LENGTH_IN_SECONDS = 8
TRAIN_SET_DURATION_IN_SECONDS = 120 # seconds, i.e. two minutes
WAVE_DIR = 'C:\DATA\koumura birds\Bird0\Wave'
ANNOTATION_FILE = 'C:\DATA\koumura birds\Bird0\Annotation.xml'

song_list = hvc.utils.parse_xml(ANNOTATION_FILE,concat_seqs_into_songs=True)
all_syls = [syl.label for song in song_list for syl in song.syls]
uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)
if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
    raise ValueError("One class of syllable occurs orders of magnitude less"
                     " frequently than the others")

#given that there's only one sampling frequency, use it to figure out the number
# of time bins in the fixed length spectrogram into which the sequences will be
# packed
# for default, 32 / 32000 = 0.001 s, i.e. 1 ms
timebin_size_in_s = WINDOW_STEP / SAMP_FREQ 
# 8 s / (0.001s/bin) = 8k bins
num_timebins_for_max_spect = int(MAX_SPECT_LENGTH_IN_SECONDS /
                                 timebin_size_in_s) 

print('Shuffling songs')
# shuffle sequences before splitting into training and test sets
# get ids in variable instead of shuffling array in place
# so we can keep a record of which sequences were in training / test set.
while 1:
    np.random.seed(42) # for reproducibility, and for Douglas Adams
    rand_song_ids = np.random.permutation(len(song_list))
    shuffled_song_list = [song_list[song_id] for song_id in rand_song_ids]
    all_train_syls = [int(syl.label) for song in song_list for syl in song.syls]
    uniq_train_syls, syl_counts = np.unique(all_train_syls,return_counts=True)
    if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
        raise ValueError("One class of syllable occurs orders of magnitude less"
                         " frequently than others in training set")  
    if not np.setdiff1d(uniq_train_syls,uniq_syls):
        #i.e., if all syllable classes are in the training set
        break # because we don't need to shuffle the training set again
### instead of loading all spectrograms
### have constant for EXPECTED sampling frequency
### then figure out length of each sequence by converting with expected sampling
### frequency and only get spects for the ones you will use after shuffling and 
### figuring out how many it will take to fill the packed_spects required for
### the training data length
train_set_duration_in_samples = TRAIN_SET_DURATION_IN_SECONDS * SAMP_FREQ
train_songs = []
curr_train_set_dur = 0

os.chdir(WAVE_DIR)
for song_numbr, song in enumerate(shuffled_song_list):
    if curr_train_set_dur + song.length > train_set_duration_in_samples:
        pass
    else:
        print("Generating spectrogram for training sequence {}".format(song_numbr))
        [sampfreq, wav] = wavfile.read(song.wavFile)
        if sampfreq != SAMP_FREQ:
            raise ValueError(
                'Sampling frequency for {}, {}, does not match expected'
                ' sampling frequency of {}'.format(song.wavFile,
                                                   sampfreq,
                                                   SAMP_FREQ))        

        song.position -= margin_start
        song.length += margin_end
        for syl in song.syls:
            syl.position += margin_start
        song_wav = wav[int(song.position):int(song.position+song.length)]
        song.seqSpect = hvc.utils.make_spect(song_wav,sampfreq,
                                            size=WINDOW_SIZE,
                                            step=WINDOW_STEP,
                                            freq_cutoffs=FREQ_CUTOFFS)
        curr_train_set_dur += song.length
        train_songs.append(song)   

# need to make spectrograms all the same length
# First figure out length of max spectrogram
spect_lengths = [song.seqSpect.spect.shape[1] for song in train_songs]
num_timebins_for_max_spect = float(np.max(spect_lengths))

freqBins_size = len(train_songs[0].seqSpect.freqBins)
# also need to pack vectors with labels for each time bin in the packed spectrogram.
# These label vectors are used by gradient descent to get error of network output. 
#keras requires all labels be positive integers to convert to boolean array for conditional cross entropy
#so assign label for "silent gap" between syllables a label that is max. label number + 1
#i.e. len(uniq_train_syls)
silent_gap_label = len(uniq_syls)
num_syl_classes = len(uniq_syls)+1

train_spects_padded = []
train_labels_padded = []
for song in train_songs:
  
    print("Padding spect for: {}".format(song))
    curr_song_padded_spect = np.zeros((freqBins_size,length_for_all_spects))
    last_col_id = song.seqSpect.timeBins.shape[0]
    curr_song_padded_spect[:,:last_col_id] = song.seqSpect.spect
    train_spects_padded.append(curr_song_padded_spect)
        
    #generate vector of training labels
    curr_song_padded_label_vec = np.ones((label_vector_length,1),dtype=int) * silent_gap_label
    labels = [int(syl.label) for syl in song.syls]
    onsets = [syl.position / song.seqSpect.sampFreq for syl in song.syls]
    offsets = [(syl.position + syl.length) / song.seqSpect.sampFreq for syl in song.syls]
    time_bins = song.seqSpect.timeBins
    onset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - onset)) for onset in onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - offset)) for offset in offsets]
    for onset, offset, label in zip(onset_IDs_in_time_bins, offset_IDs_in_time_bins,labels):
        curr_song_padded_label_vec[onset:offset+1] = label
    curr_song_padded_label_vec = to_categorical(curr_song_padded_label_vec,num_syl_classes)
    train_labels_padded.append(curr_song_padded_label_vec)

spect_scaler = StandardScaler()
spect_scaler.fit(np.hstack(train_spects_padded[:]).T) # concatenate all spects then transpose so Hz bins are 'features'
for ind, spect in enumerate(train_spects_padded): # now scale each individual training spect
    train_spects_padded[ind] = np.transpose(spect_scaler.transform(spect.T))

silent_gap_label_onehot = to_categorical([silent_gap_label])[0]
input_shape = (int(scaling_factor),1,freqBins_size,int(length_for_all_spects),) # 1 because spects only have one channel (not e.g. RGB)
dcnn = hvc.models.DCNN2(input_shape=input_shape,
                        num_syllable_classes=num_syl_classes,
                        layers_dict=LAYERS_DICT,
                        silent_gap_label=silent_gap_label_onehot) # +1 for silent gap label
