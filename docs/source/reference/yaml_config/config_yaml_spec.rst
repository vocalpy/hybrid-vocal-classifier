==========================
spec for yaml config files
==========================

This document specifies the structure of HVC config files written in yaml.
It is a painfully dry document that exists to guide the project code,
not to teach someone how to write HVC config files. For a gentle
introduction to writing config files, please see the
:doc:`writing_config_files`.

Essentially, each config file specifies a list of `jobs`. Each `job` in
a list will typically correspond to data files from one bird.

Config files consist of three sections:

 1. `global_config`: parameters that apply to all `jobs`
 2. `model_selection`: list of `jobs` for selecting machine learning models
 3. `prediction`: list of `jobs` that apply models to unclassified data

## global_config
As the name implies, parameters in the `global_config` section apply to all jobs.
The `global_config` is a dictionary of dictionaries.

Example:
``` yaml
global_config:
    spect_params :
        samp_freq : 32000 # Hz
        window_size : 512
        window_step : 32
        freq_cutoffs : [1000,8000]

    neural_net :
        syl_spect_width : 300
```

`model_selection`
-----------------

`model_selection` is a list of `jobs`. Each `job` is a dictionary.
 Hence `model_selection` is a list of dictionaries.

Each `job`, i.e. each item in the list, is marked with an empty dash.
Below each empty dash appear the keys and values that
make up the dictionary.

A `job` in the 'model_selection` section **must** include the following
keys:
 - `bird_ID` : string, alphanumeric, identifies bird
 - `train` : dictionary with parameters for training dataset
 - `test` : dictionary with parameters for testing dataset
   - both `train` and `test` contain a list `dirs`. Each item in `dirs`
     is a string, and that string **must** be a path to a directory of
     audio files (expected to contain song from the bird `bird_ID`).
 - `output_dir` : string, directory where output will be saved. HVC
 creates a new subfolder in the given directory.
 - `labelset` : string, labels used for syllabes. Only syllables with
 the labels in `labelset` will be included in the training and testing
  datasets.

**If a parameter is defined in `global_config` and then defined again in
a `job`, the value defined in the `job` takes precedence over the
 `global_config` value, but only for that job.**

Example:
``` yaml
model_selection: # list of dictionaries, dash without key next to is a list item so each dictionary is an item in the list
    - # i.e. this is dictionary 1
        bird_ID : gr41rd51

        train :
            dirs:
                - C:\DATA\gr41rd51\pre_surgery_baseline\06-21-12
        test :
            dirs:
                - C:\DATA\gr41rd51\pre_surgery_baseline\06-19-12
                - C:\DATA\gr41rd51\pre_surgery_baseline\06-20-12
                - C:\DATA\gr41rd51\pre_surgery_baseline\06-22-12

        output_dir: C:\DATA\gr41rd51\

        labelset : iabcdefgjkm

        spect_params : # not required, but will take precedence over spect_params in global_config
            samp_freq : 32000 # Hz
            window_size : 512
            window_step : 32
            freq_cutoffs : [1000,10000]

```

`prediction`
------------

Like `model_selection`, the `prediction` section is a list of `job`
dictionaries.

A `job` in the 'prediction` section **must** include the following keys:
 - `bird_ID` : string, alphanumeric, identifies bird
 - `model_file` : string, a file name. Either a scikit-learn model that
 has been `pickle`d or `dump`ed by joblib, or an hdf5 model output by
 Keras.

``` yaml
prediction:
    -
      bird_ID : gr41rd51
      model_file : gr41rd51_svm.pkl
```

parameters
----------

The parameters listed below can appear in either `global_config` or a `job`.
 - spect_params :
    - samp_freq : integer
    - window_size : integer
    - window_step : integer
    - freq_cutoffs : list
 - num_train_songs :
    - start : integer
    - stop : integer
    - step : integer
 - num_train_samples :
    - start : integer
    - stop : integer
    - step : integer
 - models :
    - knn
    - linsvm
    - svm
    - neural_net