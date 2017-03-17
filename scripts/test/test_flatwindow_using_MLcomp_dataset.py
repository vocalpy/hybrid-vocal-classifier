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
from hvc.evfuncs import load_cbin, load_notmat
from hvc.audio import extract_syls

SHELVE_BASE_FNAME = 'linsvm_svmrbf_knn_results_'
# constants used in main loop
NUM_SONGS_TO_TEST = list(range(3, 16, 3)) + [21, 27, 33, 39]
# i.e., [3,6,9,...15,21,27,33,39].
REPLICATES = range(1, 6)

avg_acc_arr = np.zeros((len(REPLICATES),len(NUM_SONGS_TO_TEST)))

config_file = sys.argv[1]
with open(config_file) as yaml_to_parse:
    config_dict = yaml.load(yaml_to_parse)
global_config = config_dict['global_config']
jobs = config_dict['jobs']

for job in jobs:
    #get params out of dict for this job
    bird_ID = job['bird_ID']
    labelset = list(job['labelset'])
    spect_params = job['spect_params']
    syl_spect_width = job['syl_spect_width']
    train_dir = job['train']['dirs'][0]
    test_dirs = job['test']['dirs']
    output_dir = job['output_dir'] + 'hvc_neuralnet_results'
    output_dir = os.path.normpath(output_dir)

    #load all the training spects and labels
    train_syl_labels = []
    train_syl_spects = []
    os.chdir(train_dir)
    cbins = glob.glob('*.cbin')
    notmats = glob.glob('*.not.mat')
    cbins = [cbin for cbin in cbins if cbin + '.not.mat' in notmats]
    for cbin_ind, cbin in enumerate(cbins):
        print('extracting syllables from song {} of {}\r'.format(cbin_ind,len(cbins)))
        syls, labels = hvc.audio.extract_syls(cbin, spect_params, labelset)
        train_syl_labels.extend(labels)
        train_syl_spects.extend(syls)

    #now load all test labels
    test_labels = []
    test_syl_spects = []
    for test_dir in test_dirs:
        os.chdir(test_dir)
        cbins = glob.glob('*.cbin')
        notmats = glob.glob('*.not.mat')
        cbins = [cbin for cbin in cbins if cbin + '.not.mat' in notmats]
        for cbin_ind, cbin in enumerate(cbins):
            print('extracting syllables from song {} of {}\r'.format(cbin_ind,len(cbins)))
            syls, labels = hvc.audio.extract_syls(cbin, spect_params, labelset)
            test_labels.extend(labels)
            test_syl_spects.extend(syls)

    #reshape labels so they match output for neural net
    num_syl_classes = np.size(labelset)
    # make a dictionary that maps labels to classes 0 to n-1 where n is number of
    # classes of syllables.
    # Need this map instead of e.g. converting from char to int because
    # keras to_categorical function requires
    # input where classes are labeled from 0 to n-1
    classes_zero_to_n = range(num_syl_classes)
    label_map = dict(zip(labelset, classes_zero_to_n))
    train_syl_labels_zero_to_n = np.asarray([label_map[syl] for syl in train_syl_labels])
    # so we can then convert to array of binary / one-hot vectors for training
    train_syl_labels_binary = to_categorical(train_syl_labels_zero_to_n, num_syl_classes)
    test_syl_labels_zero_to_n = np.asarray([label_map[syl] for syl in test_syl_labels])
    # so we can then convert to array of binary / one-hot vectors for training
    test_syl_labels_binary = to_categorical(test_syl_labels_zero_to_n, num_syl_classes)

    # reshape train and test data so it works as input to neural net
    train_syl_spects = np.stack(train_syl_spects[:], axis=0)
    train_syl_spects = np.expand_dims(train_syl_spects, axis=1)

    test_syl_spects = np.stack(test_syl_spects[:], axis=0)
    test_syl_spects = np.expand_dims(test_syl_spects, axis=1)

    os.chdir('C:\\DATA\\ML-comparison-birdsong\\experiment_code\\linsvm_svmrbf_knn_results_080616')

    for y_ind, num_songs in enumerate(NUM_SONGS_TO_TEST):
        for x_ind, replicate in enumerate(REPLICATES):
            source_shelve_fname = SHELVE_BASE_FNAME + bird_ID + ", " + str(num_songs) + \
                           ' songs, replicate ' + str(replicate) + '.db'
            print('training based on samples from {}'.format(source_shelve_fname))
            with shelve.open(source_shelve_fname,'r') as shv:
                holdout_IDs = shv['holdout_sample_IDs']
                train_IDs = shv['train_sample_IDs']

            train_spects_subset = train_syl_spects[train_IDs,:,:,:]
            train_labels_subset = train_syl_labels_binary[train_IDs,:]

            validat_spects = train_syl_spects[holdout_IDs,:,:,:]
            validat_labels = train_syl_labels_binary[holdout_IDs,:]

            #scale all spects by mean and std of training set
            spect_scaler = StandardScaler()
            #concatenate all spects then rotate so Hz bins are 'features'
            spect_scaler.fit(np.rot90(np.hstack(train_spects_subset[:,0,:,:])))

            # scale all the spectrograms. In case it actually does matter.
            for spect_ind in range(train_spects_subset.shape[0]):
                train_spects_subset[spect_ind,0,:,:] = np.rot90(spect_scaler.transform(
                                           np.rot90(train_spects_subset[spect_ind,0,:,:])),3)
            for spect_ind in range(validat_spects.shape[0]):
                validat_spects[spect_ind,0,:,:] = np.rot90(spect_scaler.transform(
                                           np.rot90(validat_spects[spect_ind,0,:,:])),3)

            test_spects_scaled = np.zeros((test_syl_spects.shape))
            for spect_ind in range(test_syl_spects.shape[0]):
                test_spects_scaled[spect_ind, 0, :, :] = np.rot90(spect_scaler.transform(
                    np.rot90(test_syl_spects[spect_ind, 0, :, :])), 3)

            # Also need to know number of rows, i.e. freqbins.
            # Will be the same for all spects since we used the same FFT params for all.
            # freqBins size is also input shape to LSTM net
            # (since at each time point the input is one column of spectrogram)
            num_channels,num_freqbins, num_timebins = train_spects[0].shape
            input_shape = (num_channels,num_freqbins,num_timebins)
            flatwindow = hvc.neuralnet.models.DCNN_flatwindow(input_shape=input_shape,
                                               num_syllable_classes=num_syl_classes)

            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = bird_ID + '_' + 'DCNN_flatwindow_training_' + now_str + \
                       '.log'
            csv_logger = CSVLogger(filename,
                                   separator=',',
                                   append=True)
            weights_filename = bird_ID + '_' + "weights " + now_str + \
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

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            os.chdir(output_dir)
            print('Training model.')
            flatwindow.fit(train_spects,
                           train_labels,
                           validation_data=(validat_spects,validat_labels),
                           batch_size=BATCH_SIZE,
                           nb_epoch=NB_EPOCH,
                           callbacks=callbacks_list,
                           verbose=1)

            pred_labels = flatwindow.predict_classes(test_spects_scaled, batch_size=32, verbose=1)
            acc_by_label, avg_acc = hvc.metrics.average_accuracy(test_syl_labels_zero_to_n, pred_labels,
                                                                 classes_zero_to_n)
            print('average accuracy on test set was {}'.format(avg_acc))
            avg_acc_arr[x_ind,y_ind] = avg_acc
            output_shelve_fname = bird_ID + '_' + now_str + "_NN_training_parameters"
            with shelve.open(output_shelve_fname) as shv:
                shv['num_songs'] = num_songs
                shv['replicate'] = replicate
                shv['config_file'] = config_file
                shv['job_dict'] = job
                shv['train_dir'] = train_dir
                shv['train_sample_IDs'] = train_IDs
                shv['holdout_sample_IDs'] = holdout_IDs
                shv['batch_size'] = BATCH_SIZE
                shv['nb_epoch'] = NB_EPOCH
                shv['train_labels'] = train_labels
                shv['validation_labels'] = validat_labels
                shv['label_map'] = label_map
                shv['source_shelve'] = shelve_file
                shv['avg_acc'] = avg_acc
                shv['acc_by_label'] = acc_by_label

joblib.dump(avg_acc_arr)