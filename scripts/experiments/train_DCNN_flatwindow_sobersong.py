### Note this is Python 2.7, because Theano ###
from __future__ import division

import os
import sys
import glob
import datetime
import shelve

import numpy as np
from scipy.io import wavfile
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

import hvc.utils.utils
import hvc.neuralnet.models
from hvc.evfuncs import load_cbin,load_notmat
from hvc.audio import extract_syls

config_file = sys.argv[1]
with open(config_file) as yaml_to_parse:
    config_dict = yaml.load(yaml_to_parse)
labelset = list(config_dict['labelset'])
spect_params = config_dict['spect_params']
syl_spect_width = config_dict['syl_spect_width']

# constants used by script
train_dir = config_dict['train']['dirs'][0]
output_dir = config_dict['output_dir'] + 'hvc_neuralnet_results'
output_dir = os.path.normpath(output_dir)

#given that there's only one sampling frequency, use it to figure out the number
#of time bins in the fixed length spectrogram into which the sequences will be
# padded. for default, 32 / 32000 = 0.001 s, i.e. 1 ms
timebin_size_in_s = spect_params['window_step'] / spect_params['samp_freq']

os.chdir(train_dir)
cbins = glob.glob('*.cbin')
all_syl_labels = []
all_syl_spects = []
background_noise = []
for cbin_ind,cbin in enumerate(cbins):
    print('extracting syllables from song {} of {}\r'.format(cbin_ind,
                                                             len(cbins)))
    syls,labels = extract_syls(cbin,spect_params,labelset)
    import pdb;pdb.set_trace()

#scale all spects by mean and std of training set
spect_scaler = StandardScaler()
# concatenate all spects then rotate  so Hz bins are 'features'
spect_scaler.fit(np.rot90(np.hstack(all_syl_spects[:])))
# now scale each individual training spect
all_syl_spects_scaled = []
for spect in all_syl_spects:
    all_syl_spects_scaled.append(
        np.rot90(
            spect_scaler.transform(np.rot90(spect))
            ,3)
            )

#reshape training data for model
all_syl_spects = np.stack(all_syl_spects_scaled[:],axis=0)
all_syl_spects = np.expand_dims(all_syl_spects,axis=1)

num_syl_classes = np.size(labelset)
# make a dictionary that maps labels to classes 0 to n-1 where n is number of
# classes of syllables.
# Need this map instead of e.g. converting from char to int because
# keras to_categorical function requires
# input where classes are labeled from 0 to n-1
classes_zero_to_n = range(num_syl_classes)
label_map = dict(zip(labelset,classes_zero_to_n))
all_syl_labels_zero_to_n = np.asarray([label_map[syl]
                                        for syl in all_syl_labels])
#so we can then convert to array of binary / one-hot vectors for training
all_syl_labels_binary = to_categorical(all_syl_labels_zero_to_n,num_syl_classes)

num_syl_spects = all_syl_spects.shape[0]
half_spects = num_syl_spects // 2

train_spects = all_syl_spects[:half_spects,:,:,:]
train_labels = all_syl_labels_binary[:half_spects,:]

validat_spects = all_syl_spects[half_spects:,:,:,:]
validat_labels = all_syl_labels_binary[half_spects:,:]

#print('Shuffling syllables.')
## shuffle and split into training and test sets
#RANDOM_SEED = 42 
#np.random.seed(RANDOM_SEED) 
#shuffle_ids = np.random.permutation(all_syl_spects.shape[0])
#all_syl_spects_shuffled = all_syl_spects[shuffle_ids,:,:,:]
#all_syl_labels_shuffled = all_syl_labels_binary[shuffle_ids,:]

#constants for training
NUM_TRAIN_SAMPLES = 400
train_spects_subset = train_spects[:NUM_TRAIN_SAMPLES,:,:,:]
train_labels_subset = train_labels[:NUM_TRAIN_SAMPLES,:]


uniq_syls, syl_counts = np.unique(all_syl_labels[:NUM_TRAIN_SAMPLES],
                                  return_counts=True)
print('Training set:')
for syl,count in zip(uniq_syls,syl_counts):
    print('\tSyllable {} -- {} samples.'.format(syl,count)) 

# Also need to know number of rows, i.e. freqbins.
# Will be the same for all spects since we used the same FFT params for all.
# freqBins size is also input shape to LSTM net
# (since at each time point the input is one column of spectrogram)
num_channels,num_freqbins, num_timebins = all_syl_spects[0].shape
input_shape = (num_channels,num_freqbins,num_timebins)
flatwindow = hvc.neuralnet.models.DCNN_flatwindow(input_shape=input_shape,
                                   num_syllable_classes=num_syl_classes) 

now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
num_samples = "_" + str(NUM_TRAIN_SAMPLES) + "_samples"
filename = BIRD_ID + '_' + 'DCNN_flatwindow_training_' + now_str + \
           num_samples + '.log'
csv_logger = CSVLogger(filename,
                       separator=',',
                       append=True)
weights_filename = BIRD_ID + '_' + "weights " + now_str + num_samples + \
                   ".best.hdf5"
checkpoint = ModelCheckpoint(weights_filename,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')
earlystop = EarlyStopping(monitor='val_acc',
                          min_delta=0,
                          patience=20,
                          verbose=1,
                          mode='auto')
callbacks_list = [csv_logger,checkpoint,earlystop]

BATCH_SIZE = 32
NB_EPOCH = 200

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
os.chdir(OUTPUT_DIR)
print('Training model.')
flatwindow.fit(train_spects_subset,
          train_labels_subset,
          validation_data=(validat_spects,validat_labels),
          batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH,
          callbacks=callbacks_list,
          verbose=1)

shelve_fname = BIRD_ID + '_' + now_str + num_samples + "_training_set_data"
with shelve.open(shelve_fname) as shv:
    shv['config_file'] = config_file
    shv['config'] = config_dict
    shv['data_dir'] = DATA_DIR
#    shv['num_songs_to_use'] = NUM_SONGS_TO_USE
    shv['cbins'] = cbins
#    shv['shuffle_ids'] = shuffle_ids
    shv['half_of_spects'] = half_spects
    shv['num_train_samples'] = NUM_TRAIN_SAMPLES
#    shv['validation_split'] = VALIDAT_SPLIT
    shv['batch_size'] = BATCH_SIZE
    shv['nb_epoch'] = NB_EPOCH
    shv['train_labels'] = train_labels_subset
    shv['validation_labels'] = validat_labels
    shv['label_map'] = label_map

scaler_fname = BIRD_ID + '_' + now_str + num_samples + "_scaler"
with open(scaler_fname,'wb') as scaler_file:
    joblib.dump(spect_scaler,scaler_file)

train_spects_fname = BIRD_ID + '_' + now_str + num_samples + "_train_spects"
with open(train_spects_fname,'wb') as tr_spect_file:
    joblib.dump(train_spects_subset,tr_spect_file)

validat_spects_fname = BIRD_ID + '_' + now_str + num_samples + "_validat_spects"
with open(validat_spects_fname,'wb') as val_spect_file:
    joblib.dump(validat_spects,val_spect_file)
