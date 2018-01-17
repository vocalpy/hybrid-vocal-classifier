================================================
spec for YAML files to configure model selection
================================================

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
`select.config.yml` files *may* optionally define other keys at the same level as `todo_list`.
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
    for example:
    `C:\Data\gy6or6\extract_output_170711_0104\summary_feature_file_created_170711_0104`

  * output_dir: str
    path to directory in which to save output
    if it doesn't exist, HVC will create it
    for example, `C:\DATA\bl26lb16\`

optional keys
~~~~~~~~~~~~~
As stated above, these can all be defined at the top level of the file. If they are also defined
for any dict in a `todo_list`, then that definition will override the top-level definition.
  * models: list of dicts
     dictionary of models, as defined below.
     Required if not defined at top level of file.
  * `num_replicates`: int
      number of replicates, i.e. number of folds for cross-validation

  * `num_test_samples`: int
      number of samples from feature file to put in testing set

  * `num_train_samples`: int
      number of samples from feature file to put in training set

specification for models list of dicts
--------------------------------------
Every dict in a `models` list has the following **required** keys:
  * model_name: str
      name of model, e.g. 'svm'
  * hyperparameters: dict
      with hyperparameters defined for each model

Every dict in a `models` list must also specify the features with which to train the model.
One of the following is valid, as specified in `validation.yml`.
   * feature_list_indices: list of ints
      corresponding to elements in list of feature names in feature_file
      e.g., [0,1,2,5,7]
   * feature_group: str
      name of a feature group: {'knn','svm'}
   * neuralnet_input: str
      name of input for am artificial neural net: {'flatwindow'}

example `select_config.yml`
---------------------------

These are some of the `select.config.yml` files used for testing, found in
`hybrid-vocal-classifier//tests//test_data//config.yaml//`:

.. literalinclude:: ..//..//tests//test_data//config.yaml//test_select_knn_ftr_grp.config.yml

.. literalinclude:: ..//..//tests//test_data//config.yaml//test_select_svm.config.yml

.. literalinclude:: ..//..//tests//test_data//config.yaml//test_select_flatwindow.config.yml

