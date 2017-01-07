import sys
from os.path import isdir,normpath

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
        putative_dir_name = normpath(dir_name)
        if not isdir(putative_dir_name):
            raise ValueError(
                "{} is not recognized as a directory".format(putative_dir_name))
        else:
            dir_names.append(normpath(putative_dir_name))

labelset = list(sys.argv[2])

# same for Tachibana and for Todd? forgot
spect_params

# shouldn't be a constant, should be an argument to the function, default True
quantify_deltas

# set up output, but this can just be a file, right?

#determine segmenting parameters

