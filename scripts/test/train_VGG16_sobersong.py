### Note this is Python 2.7, because Theano ###
from __future__ import division

import pdb
import os
import glob

import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

import hvc.utils.utils
import hvc.neuralnet.models
from hvc.utils import sequences
from hvc.audio.ev_funcs import load_cbin,load_notmat

#constants for spectrogram
SAMP_FREQ = 32000 # Hz
WINDOW_SIZE= 512
WINDOW_STEP= 32
FREQ_CUTOFFS=[1000,8000]
MAX_SILENT_GAP = 0.08 # s to keep before or after a syllable

# constants used by script
DATA_DIR = os.path.normpath('C:/DATA/gy6or6/032212')
NUM_SONGS_TO_USE = 20

os.chdir(DATA_DIR)
cbins = glob.glob('*.cbin')

all_syls = ''
for cbin in cbins:
    notmat = load_notmat(cbin)
    all_syls += notmat['labels'];

uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)
if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
    raise ValueError("One class of syllable occurs orders of magnitude less"
                     " frequently than the others")

#given that there's only one sampling frequency, use it to figure out the number of time bins in the
#fixed length spectrogram into which the sequences will be padded
timebin_size_in_s = WINDOW_STEP / SAMP_FREQ # for default, 32 / 32000 = 0.001 s, i.e. 1 ms

# need to pack vectors with labels for each time bin in the padded spectrogram.
# These label vectors are used by gradient descent to get error of network output. 
#keras requires all labels be positive integers to convert to boolean array for conditional cross entropy
#so assign label for "silent gap" between syllables a label that is max. label number + 1
#i.e. len(uniq_train_syls)
silent_gap_label = len(uniq_syls)
num_syl_classes = len(uniq_syls)+1

all_syl_labels = []
all_syl_spects = []
background_noise_to_pad_spectrograms = []
for cbin in cbins[:NUM_SONGS_TO_USE]:
    print('extracting syllables from song {}'.format(cbin))
    dat, fs = load_cbin(cbin)
    if fs != SAMP_FREQ:
        raise ValueError(
            'Sampling frequency for {}, {}, does not match expected sampling '
            'frequency of {}'.format(cbin,
                                     fs,
                                     SAMP_FREQ))
    dat,fs = load_cbin(cbin)
    spect_obj = hvc.utils.utils.make_spect(dat,fs,
                                           size=WINDOW_SIZE,
                                           step=WINDOW_STEP,
                                           freq_cutoffs=FREQ_CUTOFFS)
    spect = spect_obj.spect     
    time_bins = spect_obj.timeBins

    notmat = load_notmat(cbin)
    labels = notmat['labels']
    onsets = notmat['onsets'] / 1000.0
    offsets = notmat['offsets'] / 1000.0
    onset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - onset))
                                for onset in onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - offset))
                                for offset in offsets]
    #extract each syllable, but include the "silence" around it
    for ind,label in enumerate(labels):
        # include "silent" gaps around syllables
        if ind==0:
            # but we don't want a silent gap bigger than MAX_SILENT_GAP
            tmp_1st_onset = np.argmin(np.abs(
                                time_bins - (onsets[0] - MAX_SILENT_GAP)))
            tmp_spect = spect[:,tmp_1st_onset:onset_IDs_in_time_bins[1]-1]
            # 'background' is just the silent gaps without syllable
            # to append to background_noise_to_pad_spectrograms
            background = np.hstack(
                            (spect[:,tmp_1st_onset:onset_IDs_in_time_bins[0]-1],
                                    spect[:,offset_IDs_in_time_bins[0]+1:
                                          onset_IDs_in_time_bins[1]-1]))
        elif ind >0 and ind < len(labels) - 1:
            tmp_spect = spect[:,offset_IDs_in_time_bins[ind-1]+1:
                                onset_IDs_in_time_bins[ind+1]-1]
            background = np.hstack(
                        (spect[:,offset_IDs_in_time_bins[ind-1]+1:
                              onset_IDs_in_time_bins[ind]-1],
                        spect[:,offset_IDs_in_time_bins[ind]+1:
                              onset_IDs_in_time_bins[ind+1]-1]))
        elif ind == len(labels) - 1:
            tmp_last_onset = np.argmin(np.abs(
                                time_bins - (offsets[-1] + MAX_SILENT_GAP)))
            tmp_spect = spect[:,offset_IDs_in_time_bins[ind-1]+1:tmp_last_onset]
            background = np.hstack(
                        (spect[:,offset_IDs_in_time_bins[ind-1]+1:
                              onset_IDs_in_time_bins[ind]-1],
                        spect[:,offset_IDs_in_time_bins[ind]+1:
                              tmp_last_onset]))
        all_syl_labels.append(label)
        all_syl_spects.append(tmp_spect)

import pdb;pdb.set_trace()

# need to zero pad spectrogram so they are all the same length
# First figure out max length
spect_lengths = [spect.shape[1] for spect in all_syl_spects]
num_timebins_for_max_spect = np.max(spect_lengths)

# Also need to know number of rows, i.e. freqbins.
# Will be the same for all spects since we used the same FFT params for all.
freqBins_size = all_syl_spects[0].shape[0]

counter = 0
for ind, spect in enumerate(all_syl_spects):
    print("Padding spect + label vec {}.".format(counter))
    curr_padded_spect = np.zeros((freqBins_size,num_timebins_for_max_spect))
    last_col_id = spect.shape[1]
    width_diff = num_timebins_for_max_spect - last_col_id
    # take half of difference between spects and start there
    # so one half will be on one side of spect and other will be on other
    # i.e., center the spectrogram
    start_ind = round(width_diff / 2)
    curr_padded_spect[:,:last_col_id] = spect
    all_syl_spects[ind] = curr_padded_spect
        
#scale all spects by mean and std of training set
spect_scaler = StandardScaler()
# concatenate all spects then transpose so Hz bins are 'features'
spect_scaler.fit(np.hstack(all_syl_spects_padded[:]).T)
# now scale each individual training spect
for ind, spect in enumerate(all_syl_spects_padded):
    all_syl_spects_padded[ind] = np.transpose(spect_scaler.transform(spect.T))

#reshape training data for model
all_syl_spects_padded = np.dstack(all_syl_spects_padded[:])
x,y,n = all_syl_spects_padded.shape
all_syl_spects_padded = all_syl_spects_padded.reshape(n,1,x,y)

all_syl_labels = to_categorical(all_syl_labels,num_syl_classes)

input_shape = (1,freqBins_size,num_timebins_for_max_spect)
vgg16 = hvc.neuralnet.models.VGG_16(input_shape=input_shape,
                                   num_syllable_classes=num_syl_classes) 

print('Shuffling syllables.')
# shuffle and split into training and test sets
RANDOM_SEED = 42 
np.random.seed(RANDOM_SEED) 
shuffle_ids = np.random.permutation(n)  # n is from bigram_spects_padded.shape
all_syl_spects_padded = all_syl_spects_padded[shuffle_ids,:,:,:]
all_syl_labels = all_syl_labels[shuffle_ids,:]

#constants for training
NUM_TRAIN_SAMPLES = 512
train_spects = all_syl_spects_padded[:NUM_TRAIN_SAMPLES,:,:,:]
train_labels = all_syl_labels[:NUM_TRAIN_SAMPLES,:]

test_spects = all_syl_spects_padded[-1:-1:-NUM_TRAIN_SAMPLES,:,:,:]
test_labels = all_syl_labels[-1:-1:-NUM_TRAIN_SAMPLES,:]

print('Training model.')
vgg16.fit(train_spects,
          train_labels,
          validation_split=0.33,
          batch_size=32,
          nb_epoch=100,
          verbose=1,
          callbacks=callbacks_list,
               )

import pdb;pdb.set_trace()
