### Note this is Python 2.7, because Theano ***
from __future__ import division

import pdb
import os
import glob

import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger

import hvc # used as a proper noun

#constats for spectrogram
SAMP_FREQ = 32000 # Hz
WINDOW_SIZE= 512
WINDOW_STEP= 32
FREQ_CUTOFFS=[1000,8000]

WAVE_DIR = 'C://DATA//koumura birds//Bird0//Wave'
ANNOTATION_FILE = 'C://DATA//koumura birds//Bird0//Annotation.xml'

song_list = hvc.utils.parse_xml(ANNOTATION_FILE,concat_seqs_into_songs=True)
all_syls = [syl.label for song in song_list for syl in song.syls]
uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)

#given that there's only one sampling frequency, use it to figure out the number of time bins in the
#fixed length spectrogram into which the sequences will be padded
timebin_size_in_s = WINDOW_STEP / SAMP_FREQ # for default, 32 / 32000 = 0.001 s, i.e. 1 ms

# need to pack vectors with labels for each time bin in the padded spectrogram.
# These label vectors are used by gradient descent to get error of network output. 
#keras requires all labels be positive integers to convert to boolean array for conditional cross entropy
#so assign label for "silent gap" between syllables a label that is max. label number + 1
#i.e. len(uniq_train_syls)
#silent_gap_label = len(uniq_syls)
#num_syl_classes = len(uniq_syls)+1

os.chdir(WAVE_DIR)
all_syl_labels = []
all_syl_spects = []
all_syl_label_vecs = []
for song in song_list[:20]:
    print('extracting syllables from song {}'.format(song.wavFile))
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
#    label_vec = np.ones((time_bins.shape[0],1),dtype=int) * silent_gap_label
    labels = [int(syl.label) for syl in song.syls]
    onsets = [syl.position / SAMP_FREQ for syl in song.syls]
    offsets = [(syl.position + syl.length) / SAMP_FREQ for syl in song.syls]
    onset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - onset)) for onset in onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - offset)) for offset in offsets]
#    for onset, offset, label in zip(onset_IDs_in_time_bins, offset_IDs_in_time_bins,labels):
#        label_vec[onset:offset+1] = label

    for ind,label in enumerate(labels):
        if ind==0:
            #end just before onset of syllable after bigram
            tmp_spect = spect[:,0:onset_IDs_in_time_bins[1]-1]
#            tmp_label_vec = label_vec[0:onset_IDs_in_time_bins[1]-1]
        elif ind >0 and ind < len(labels) - 1:
            tmp_spect = spect[:,offset_IDs_in_time_bins[ind-1]+1:onset_IDs_in_time_bins[ind+1]-1]
#            tmp_label_vec = label_vec[offset_IDs_in_time_bins[ind-1]+1:onset_IDs_in_time_bins[ind+1]-1]
        elif ind == len(labels) - 1:
            tmp_spect = spect[:,offset_IDs_in_time_bins[ind-1]+1:]
#            tmp_label_vec = label_vec[offset_IDs_in_time_bins[ind-1]+1:]
        all_syl_labels.append(label)
        all_syl_spects.append(tmp_spect)
#        all_syl_label_vecs.append(tmp_label_vec)

                                            
# need to zero pad spectrogram so they are all the same length
# First figure out max length
spect_lengths = [spect.shape[1] for spect in all_syl_spects]
num_timebins_for_max_spect = np.max(spect_lengths)

# Also need to know number of rows, i.e. freqbins.
# Will be the same for all spects since we used the same FFT params for all.
# freqBins size is also input shape to LSTM net
# (since at each time point the input is one column of spectrogram)
freqBins_size = all_syl_spects[0].shape[0]

all_syl_spects_padded = []
#all_syl_labels_padded = []
counter = 0
#for spect,label_vec in zip(all_syl_spects,all_syl_label_vecs):
for spect in all_syl_spects:
    counter += 1
    print("Padding spectrogram {}.".format(counter))
    curr_padded_spect = np.zeros((freqBins_size,num_timebins_for_max_spect))
    last_col_id = spect.shape[1]
    curr_padded_spect[:,:last_col_id] = spect
    all_syl_spects_padded.append(curr_padded_spect)
        
#    curr_padded_label_vec = np.ones((num_timebins_for_max_spect,1),dtype=int) * silent_gap_label
#    curr_padded_label_vec[:last_col_id] = label_vec
#    curr_padded_label_vec = to_categorical(curr_padded_label_vec,num_syl_classes)
#    all_syl_labels_padded.append(curr_padded_label_vec)

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

#all_syl_labels_padded = np.dstack(all_syl_labels_padded[:])
#x,y,n = all_syl_labels_padded.shape
#all_syl_labels_padded = all_syl_labels_padded.reshape(n,x,y)
all_syl_labels = np.asarray(all_syl_labels)
for old_syl_label,new_syl_label in zip(uniq_syls,range(0,uniq_syls.shape[0])):
    if new_syl_label != old_syl_label:
        inds = np.where(all_syl_labels == old_syl_label)
        all_syl_labels[inds] = new_syl_label

num_syl_classes = np.unique(all_syl_labels).shape[0]
all_syl_labels = to_categorical(all_syl_labels,num_syl_classes)

input_shape = (1,freqBins_size,num_timebins_for_max_spect)
DCNN_flatwindow = hvc.models.DCNN_flatwindow(input_shape=input_shape,num_syllable_classes=num_syl_classes) 

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

#saving
#set up to save after each epoch
filepath = 'C://DATA//koumura birds/Bird0//run2//dcnn-flatwindow-{epoch:02d}-{acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose = 1, save_best_only=False)
csv_logger = CSVLogger('C://DATA//koumura birds/Bird0//DCNN_flatwindow_Bird0_run2.log',
                       separator=',',
                       append=True)
callbacks_list = [checkpoint, csv_logger]
print('Training model.')
DCNN_flatwindow.fit(train_spects,
          train_labels,
          batch_size=32,
          nb_epoch=100,
          verbose=1,
          callbacks=callbacks_list,
               )
