import sys
import os
import glob

import numpy as np
from scipy.io import loadmat

with open(sys.argv[1],'r') as argv1_file:
    argv1_txt = argv1_file.readlines()
if len(argv1_txt) > 1:
    raise ValueError("Directory names file longer than expected, \n"
                     "should only be one line.\n"
                     "Check formatting of string in cell.")
argv1_txt = argv1_txt[0].split(",")
dir_names = []
for dir_name in argv1_txt:
    if dir_name is not '':
        putative_dir_name = os.path.normpath(dir_name)
        if not os.path.isdir(putative_dir_name):
            raise ValueError(
                "{} is not recognized as a directory".format(putative_dir_name))
        else:
            dir_names.append(os.path.normpath(putative_dir_name))

labelset = list(sys.argv[2])

# same for Tachibana and for Todd? forgot
##spect_params

# shouldn't be a constant, should be an argument to the function, default True
##quantify_deltas

# set up output, but this can just be a file, right?

#determine segmenting parameters
all_segment_params = []
for dir_name in dir_names:
    os.chdir(dir_name)
    not_mats = glob.glob('*.not.mat')
    num_not_mats = len(not_mats)
    segment_params = np.zeros(num_not_mats,
                              dtype=[('min_syl_dur','i4'),
                                     ('min_silent_int','i4'),
                                     ('threshold','i4'),
                                     ('smooth_win','i4'),
                                     ])

    for ind, not_mat in enumerate(not_mats):
        not_mat_dict = loadmat(not_mat)
        segment_params[ind] = (not_mat_dict['min_dur'],
                               not_mat_dict['min_int'],
                               not_mat_dict['threshold'],
                               not_mat_dict['sm_win'],)
    all_segment_params.append(segment_params)

all_segment_params = np.concatenate(all_segment_params)
uniq_segment_params, counts = np.unique(all_segment_params,return_counts=True)
if uniq_segment_params.shape[-1] > 1:
    print('Found more than one set of segmenting parameters.')
    for ind, param_set, count in zip(range(len(uniq_segment_params)),
                                uniq_segment_params,
                                counts):
        print('{0}: {1} instances of {2}'.format(ind,
                                                 count,
                                                 param_set))
    while 1:
        s = input('Enter the index of the set of segment parameters you want'
                  ' to use: ')
        try:
            s = int(s)
        except ValueError:
            print("Input not recognized as a valid index. Enter an integer.")
            continue
        try:
            uniq_segment_params = uniq_segment_params[s]
            break
        except IndexError:
            print('That index is out of range. Enter a smaller number.')
            continue

for dir_name in dir_names:
    os.chdir(dir_name)
    not_mats = glob.glob('*.not.mat')
    if not_mats == []:
        print('Did not find .not.mat files in {}'.format(dir_name))
        continue
    for not_mat in not_mats:
        cbin = not_mat[:-8] + '.cbin'
        
        
