### Note this is Python 2.7, because Theano ***
from __future__ import division

import pdb
import os
import glob

import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

import hvc # used as a proper noun

#constats for spectrogram
SAMP_FREQ = 32000 # Hz
WINDOW_SIZE= 512
WINDOW_STEP= 32
FREQ_CUTOFFS=[1000,8000]

#constants for trianing
MAX_SPECT_LENGTH_IN_SECONDS = 8
TRAIN_SET_DURATION_IN_SECONDS = 120 # seconds, i.e. two minutes

WAVE_DIR = 'C:\DATA\koumura birds\Bird0\Wave'
ANNOTATION_FILE = 'C:\DATA\koumura birds\Bird0\Annotation.xml'

seq_list = hvc.utils.parse_xml(ANNOTATION_FILE)
all_syls = [syl.label for seq in seq_list for syl in seq.syls]
uniq_syls, syl_counts = np.unique(all_syls,return_counts=True)
if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
    raise ValueError("One class of syllable occurs orders of magnitude less frequently than the others")

### Koumura approach: take sequences from songs and then "pack" sequences into a fixed length spectrogram.
### This seems odd given that the 'sequences' were cut off at an arbitrary maximum number of syllables
### that in turn made them smaller than the length of songs. Given that the max. length he uses for the
### spectrogram is 8 seconds, he could have just put each full song in a spectrogram. Unless I'm
### misunderstanding his code.

#given that there's only one sampling frequency, use it to figure out the number of time bins in the
#fixed length spectrogram into which the sequences will be packed
timebin_size_in_s = WINDOW_STEP / SAMP_FREQ # for default, 32 / 32000 = 0.001 s, i.e. 1 ms
num_timebins_for_max_spect = int(MAX_SPECT_LENGTH_IN_SECONDS / timebin_size_in_s) # for 8 s / (0.001s/bin) = 8000 bins

print('Shuffling sequences')
# shuffle sequences before splitting into training and test sets
# get ids in variable instead of shuffling array in place
# so we can keep a record of which sequences were in training / test set.
while 1:
    np.random.seed(42) # for reproducibility and to make a Douglas Adams reference
    rand_seq_ids = np.random.permutation(len(seq_list))
    shuffled_seq_list = [seq_list[seq_id] for seq_id in rand_seq_ids]
    all_train_syls = [int(syl.label) for seq in seq_list for syl in seq.syls]
    uniq_train_syls, syl_counts = np.unique(all_train_syls,return_counts=True)
    if np.min(syl_counts) < 10 ** np.floor(np.log10(np.mean(syl_counts))):
        raise ValueError("One class of syllable occurs orders of magnitude less frequently than others in training set")  
    if not np.setdiff1d(uniq_train_syls,uniq_syls):
        #i.e., if all syllable classes are in the training set
        break # because we don't need to shuffle the training set again

### instead of loading all spectrograms
### have constant for EXPECTED sampling frequency
### then figure out length of each sequence by converting with expected sampling frequency
### and only get spects for the ones you will use after shuffling and figuring out how many
### it will take to fill the packed_spects required for the training data length
train_set_duration_in_samples = TRAIN_SET_DURATION_IN_SECONDS * SAMP_FREQ
train_seqs = []
curr_train_set_dur = 0
for seq in shuffled_seq_list:
    if curr_train_set_dur + seq.length > train_set_duration_in_samples:
        pass
    else:
        curr_train_set_dur += seq.length
        train_seqs.append(seq)   

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
                                            
freqBins_size = len(train_seqs[0].seqSpect.freqBins)
packed_spect = np.zeros((freqBins_size,num_timebins_for_max_spect))
# also need to pack vectors with labels for each time bin in the packed spectrogram.
# These label vectors are used by gradient descent to get error of network output. 
#keras requires all labels be positive integers to convert to boolean array for conditional cross entropy
#so assign label for "silent gap" between syllables a label that is max. label number + 1
#i.e. len(uniq_train_syls)
silent_gap_label = len(uniq_syls)
num_syl_classes = len(uniq_syls)+1

train_label_vec = np.ones((num_timebins_for_max_spect,1),dtype=int) * silent_gap_label
packed_spect_col_id = 0
train_spects_packed = []
train_labels_packed = []
for seq in train_seqs:
    print("Packing: {}".format(seq))
    spect_width = seq.seqSpect.spect.shape[1]
    new_col_id = packed_spect_col_id + spect_width
    
    if  new_col_id > num_timebins_for_max_spect:
        train_spects_packed.append(packed_spect)
        packed_spect = np.zeros((freqBins_size,num_timebins_for_max_spect))
        train_label_arr = to_categorical(train_label_vec,num_syl_classes)
        train_labels_packed.append(train_label_arr)
        train_label_vec = np.ones((num_timebins_for_max_spect,1),dtype=int) * silent_gap_label
        packed_spect_col_id = 0
        new_col_id = packed_spect_col_id + spect_width

    packed_spect[:,packed_spect_col_id:new_col_id] = seq.seqSpect.spect
    
    #generate vector of training labels
    curr_seq_label_vec = np.ones((spect_width,1)) * silent_gap_label
    labels = [int(syl.label) for syl in seq.syls]
    onsets = [syl.position / seq.seqSpect.sampFreq for syl in seq.syls]
    offsets = [(syl.position + syl.length) / seq.seqSpect.sampFreq for syl in seq.syls]
    time_bins = seq.seqSpect.timeBins
    onset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - onset)) for onset in onsets]
    offset_IDs_in_time_bins = [np.argmin(np.abs(time_bins - offset)) for offset in offsets]
    for ind in xrange(len(onsets)):
        curr_onset = packed_spect_col_id + onset_IDs_in_time_bins[ind]
        curr_offset = packed_spect_col_id + offset_IDs_in_time_bins[ind] + 1 # +1 to include bin considered offset
        curr_seq_label_vec[onset_IDs_in_time_bins[ind]:offset_IDs_in_time_bins[ind]+1] = labels[ind]
    train_label_vec[packed_spect_col_id:new_col_id] = curr_seq_label_vec
    
    packed_spect_col_id = new_col_id


spect_scaler = StandardScaler()
spect_scaler.fit(np.hstack(train_spects_packed[:]).T) # concatenate all spects then transpose so Hz bins are 'features'
for ind, spect in enumerate(train_spects_packed): # now scale each individual training spect
    train_spects_packed[ind] = np.transpose(spect_scaler.transform(spect.T))


input_shape = (1,freqBins_size,num_timebins_for_max_spect,) # 1 because spects only have one channel (not e.g. RGB)
dcnn = hvc.models.DCNN(input_shape=input_shape,num_syllable_classes=num_syl_classes) # +1 for silent gap label

#reshape training data for model
train_spects_packed = np.dstack(train_spects_packed[:])
x,y,n = train_spects_packed.shape
train_spects_packed = train_spects_packed.reshape(n,1,x,y)
train_labels_packed = np.dstack(train_labels_packed[:])
x,y,n = train_labels_packed.shape
train_labels_packed = train_labels_packed.reshape(n,y,x)  #to match output of DCNN model

dcnn.fit(train_spects_packed,
         train_labels_packed,
         nb_epoch=200,
         batch_size=2)

#save:
# train set seqs (?)
# uniq_syls
# silent_gab_label
# trained model
