#from standard library
import glob
import sys
import os
import shelve

#from third-party
import numpy as np
import scipy.io as scio # to load matlab files
import numpy as np
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger

#from hvc
import hvc.utils.utils
import hvc.neuralnet.models
from hvc.utils import sequences
from hvc.audio.evfuncs import load_cbin,load_notmat

# get command line arguments
args = sys.argv
if len(args) != 2: # (first element, args[0], is the name of this script)
    raise ValueError('Script requires one command line arguments, TRAIN_DIR')

TRAIN_DIR = args[1]
os.chdir(TRAIN_DIR)

try:
    classify_dict = scio.loadmat('.\\classify\\to_classify.mat')
except FileNotFoundError:
    print("Did not find required files in the directory supplied as command-line
          " argument.\nPlease double check directory name.")
    
classify_dirs = classify_dict['classify_dirs']
clf_file = classify_dict['clf_file'][0] #[0] because string stored in np array
extension_id = clf_file.find('.dat')
# need to get rid of '.dat' extension before calling shelve with filename
clf_file = clf_file[:extension_id]
clf_file = '.\\train\\svmrbf_knn_results\\' + clf_file
clf_type = classify_dict['clf_type']

#need to get full directory path
with shelve.open(clf_file, 'r') as shlv:
    if clf_type=='knn':
        clf = shlv['knn_clf']
        scaler = shlv['knn_scaler']
    elif clf_type=='svm':
        clf = shlv['svm_clf']
        scaler = shlv['svm_scaler']

# used in loop below, see there for explanation
SHOULD_BE_DOUBLE = ['Fs',
                    'min_dur',
                    'min_int',
                    'offsets',
                    'onsets',
                    'sm_win',
                    'threshold']
      
#loop through dirs
for classify_dir in classify_dirs:
    os.chdir(classify_dir)
    notmats = glob.glob('*.not.mat')
    if type(clf)==neighbors.classification.KNeighborsClassifier:
        ftr_files = glob.glob('*knn_ftr.to_classify*')

    elif type(clf)==SVC:
        ftr_files = glob.glob('*svm_ftr.to_classify*')

    for ftr_file,notmat in zip(ftr_files,notmats):
        if type(clf)==neighbors.classification.KNeighborsClassifier:
            samples = load_from_mat(ftr_file,'knn','classify')
        elif type(clf)==SVC:
            samples = load_from_mat(ftr_file,'svm','classify')
        samples_scaled = scaler.transform(samples)
        pred_labels = clf.predict(samples_scaled)
        #chr() to convert back to character from uint32
        pred_labels = [chr(val) for val in pred_labels]
        # convert into one long string, what evsonganalty expects
        pred_labels = ''.join(pred_labels)
        notmat_dict = scio.loadmat(notmat)
        notmat_dict['predicted_labels'] = pred_labels
        notmat_dict['classifier_type'] = clf_type
        notmat_dict['classifier_file'] = clf_file
        print('saving ' + notmat)
        # evsonganaly/Matlab expects all vars as double
        for key, val in notmat_dict.items():
            if key in SHOULD_BE_DOUBLE:
                notmat_dict[key] = val.astype('d') 
        scio.savemat(notmat,notmat_dict)
