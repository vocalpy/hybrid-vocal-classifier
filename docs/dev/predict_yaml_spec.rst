=================================================
spec for YAML files to configure label prediction
=================================================

This document specifies the structure of HVC config files written in
YAML. It is a painfully dry document that exists to guide the project
code, not to teach someone how to write the files. For a gentle
introduction to writing the files, please see
:doc:`writing_predict_yaml.md`.

structure
---------
Every `predict.config.yml` file should be written in YAML as a dictionary with (key, value) pairs
In other words, any YAML file that contains a configuration for feature extraction
should define a dictionary named 'predict` with keys as outlined below.

required key: todo_list
-----------------------
Every `predict.config.yml` file has exactly one **required** key at the top level:
   `todo_list`: list of dicts
      list where each element is a dict.
      each dict sets parameters for a 'job', typically
      data associated with one set of vocalizations.

specification for dictionaries in todo_list
-------------------------------------------
required keys
~~~~~~~~~~~~~

Every dict in a `todo_list` has the following **required** keys:
  * bird_ID : str
    for example, `bl26lb16`

  * file_format: str
    {'evtaf','koumura'}

  * data_dirs: list of str
    directories containing data
    each str must be a valid directory that can be found on the path
    for example
    ```
        - C:\DATA\bl26lb16\pre_surgery_baseline\041912
        - C:\DATA\bl26lb16\pre_surgery_baseline\042012
    ```

  * model_file: str
    filename of machine learning model / neural network that will be used to predict labels for syllables
    example: `somedir//select_output_170814_005430//knn//knn_100samples_replicate0`

  * output_dir: str
    directory in which to save output
    if it doesn't exist, HVC will create it
    for example, `C:\DATA\bl26lb16\`

  * predict_proba : bool
    If True, calculate probabilities for predicted labels.

example `predict_config.yml`
----------------------------

These are some of the `predict.config.yml` files used for testing, found in
`hybrid-vocal-classifier//tests//test_data//config.yaml//`:

