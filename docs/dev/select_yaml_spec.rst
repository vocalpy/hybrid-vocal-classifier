===================================================
spec for YAML files to configure model selection
===================================================

This document specifies the structure of HVC config files written in
YAML. It is a painfully dry document that exists to guide the project
code, not to teach someone how to write the files. For a gentle
introduction to writing the files, please see
:doc:`writing_select_yaml.rst`.

structure
---------
Every `select.config.yml` file should be written in YAML as a dictionary with (key, value) pairs.
In other words, any YAML file that contains a configuration for model selection should define
a dictionary named `select` with keys as outlined below.

required key: todo_list
-----------------------
Every `select.config.yml` file has exactly one **required** key at the top level:
   `todo_list`: list of dicts
      list where each element is a dict.
      each dict sets parameters for a 'job', typically
      data associated with one set of vocalizations.

optional keys
-------------
`extract.config.yml` files *may* optionally define other keys at the same level as `todo_list`.
Those keys are:
   `num_replicates`: int
      number of replicates, i.e. number of folds for cross-validation

   `num_test_samples`: int
      number of samples from feature file to put in testing set

   `num_train_samples`: int
      number of samples from feature file to put in training set

   `models`: list
      list of dictionaries that define models to be tested on features

When defined at the same level as `todo_list` they are considered `default`.
If an element in `todo_list` defines different values for any of these keys,
the value assigned in that element takes precedence over the `default` value.

specification for dictionaries in todo_list
-------------------------------------------
required keys
~~~~~~~~~~~~~

Every dict in a `todo_list` has the following **required** keys:
  * feature_file : str
      for example, `bl26lb16`

  * output_dir: str
       path to directory in which to save output
       if it doesn't exist, HVC will create it
       for example, `C:\DATA\bl26lb16\`

**Finally, each dict in a `todo_list` must define *either*
`feature_list` *or* a `feature_group`**
   * feature_list : list
        named features. See the list of named features here:
        :doc:`named_features`

If `feature_group` is a list then it
   * feature_group : str or list
        named group of features, list if more than one group
        {'knn','svm'}




example `extract_config.yml`
----------------------------

```YAML
    spect_params:
      nperseg: 512
      noverlap: 480
      freq_cutoffs: [1000,8000]
    segment_params:
      threshold: 5000 # arbitrary units of amplitude
      min_syl_dur: 0.02 # ms
      min_silent_dur: 0.002 # ms

    todo_list:
      -
        bird_ID : gy6or6
        file_format: evtaf
        feature_group:
          - svm
          - knn
        data_dirs:
          - ./test_data/cbins
          - C:\Data\gy6gy6\010317
        output_dir: C:\Data\gy6gy6\
        labelset: iabcdef
      - #2
        bird_ID : bl26lb16
        file_format: evtaf
        feature_group:
          - svm
          - knn
        data_dirs:
          - C:\DATA\bl26lb16\041912
          - C:\DATA\bl26lb16\042012
        output_dir: C:\DATA\bl26lb16\
        labelset: iabcdef
```

.. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
automatic classification of birdsong elements using a linear support vector
 machine." PloS one 9.3 (2014): e92584.

.. [2] Koumura, Takuya, and Kazuo Okanoya. "Automatic recognition of element
classes and boundaries in the birdsong with variable sequences."
PloS one 11.7 (2016): e0159188.