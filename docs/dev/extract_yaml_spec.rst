===================================================
spec for YAML files to configure feature extraction
===================================================

This document specifies the structure of HVC config files written in
YAML. It is a painfully dry document that exists to guide the project
code, not to teach someone how to write the files. For a gentle
introduction to writing the files, please see
:doc:`writing_extract_yaml.md`.

structure
---------
Every `extract.config.yml` file should be written in YAML as a dictionary with (key, value) pairs
In other words, any YAML file that contains a configuration for feature extraction
should define a dictionary named 'extract` with keys as outlined below.

required key: todo_list
-----------------------
Every `extract.config.yml` file has exactly one **required** key at the top level:
   `todo_list`: list of dicts
      list where each element is a dict.
      each dict sets parameters for a 'job', typically
      data associated with one set of vocalizations.

optional keys
-------------
`extract.config.yml` files *may* optionally define two other keys at the same level as `todo_list`.
Those keys are `spect_params` and `segment_params`. As might be expected, `spect_params` is a dict
that contains parameters for making spectrograms. The `segment_params` dict contains parameters for
segmenting song. Specifications for these dictionaries are given below.

When defined at the same level as `todo_list` they are considered `default`.
If an element in `todo_list` defines different values for any of these keys,
the value assigned in that element takes precedence over the `default` value.

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

  * output_dir: str
    directory in which to save output
    if it doesn't exist, HVC will create it
    for example, `C:\DATA\bl26lb16\`

  * labelset: str
    string of labels corresponding to labeled segments
    from which features should be extracted.
    Segments with labels not in this str will be ignored.
    Converted to a list but not necessary to enter as a list.
    For example, `iabcdef`

**Finally, each dict in a `todo_list` must define *either*
`feature_list` *or* a `feature_group`**
   * feature_list : list
        named features. See the list of named features here:
        :doc:`named_features`

   * feature_group : str or list
        named group of features, list if more than one group
        {'knn','svm'}

   * Note that a `todo_list` can define *both* a `feature_list`
and a `feature_group`. In this case features from the `feature_group`
are added to the `feature_list`.

Additional variables are added to the feature files that are output by
`featureextract.extract` to keep track of which features belong to which
feature group.

specification for spect_params and segment_params dictionaries
--------------------------------------------------------------

   * spect_params: dict
      parameters to calculate spectrogram
      keys correspond to parameters/arguments passed to Spectrogram class for __init__.
      **must** have *either* a 'ref' key *or* the `nperseg` and `noverlap` keys
      as defined below:
         ref : str
            {'tachibana','koumura'}
            Use spectrogram parameters from a reference.
            'tachibana' uses spectrogram parameters from [1]_,
            'koumura' uses spectrogram parameters from [2]_.

         nperseg : int
            numper of samples per segment for FFT, e.g. 512
         noverlap : int
            number of overlapping samples in each segment

      the following keys are all **optional** for spect_params:
        freq_cutoffs : two-element list of integers
            limits of frequency band to keep, e.g. [1000,8000]
            Spectrogram.make keeps the band:
                freq_cutoffs[0] >= spectrogram > freq_cutoffs[1]
        window : str
            window to apply to segments
            valid strings are 'Hann', 'dpss', None
            Hann -- Uses np.Hanning with parameter M (window width) set to value of nperseg
            dpss -- Discrete prolate spheroidal sequence AKA Slepian.
                Uses scipy.signal.slepian with M parameter equal to nperseg and
                width parameter equal to 4/nperseg, as in [2]_.
        filter_func : str
            filter to apply to raw audio. valid strings are 'diff' or None
            'diff' -- differential filter, literally np.diff applied to signal as in [1]_.
            None -- no filter, this is the default
        spect_func : str
            which function to use for spectrogram.
            valid strings are 'scipy' or 'mpl'.
            'scipy' uses scipy.signal.spectrogram,
            'mpl' uses matplotlib.matlab.specgram.
            Default is 'scipy'.
        log_transform_spect : bool
            if True, applies np.log10 to spectrogram to increase range. Default is True.

   segment_params: dict
      parameters for dividing audio into segments, defined below
      with the following keys
         threshold : int
            value above which amplitude is considered part of a segment. default is 5000.
         min_syl_dur : float
            minimum duration of a segment. default is 0.02, i.e. 20 ms.
         min_silent_dur : float
            minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.


example `extract.config.yml` files
----------------------------------
These are some of the `extract.config.yml` files used for testing, found in
`hybrid-vocal-classifier//tests//test_data//config.yaml//`:

.. literalinclude:: ..//..//tests//test_data//config.yaml//test_extract_knn.config.yml

.. literalinclude:: ..//..//tests//test_data//config.yaml//test_extract_svm.config.yml

.. literalinclude:: ..//..//tests//test_data//config.yaml//test_extract_flatwindow.config.yml

references
----------
.. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
automatic classification of birdsong elements using a linear support vector
 machine." PloS one 9.3 (2014): e92584.

.. [2] Koumura, Takuya, and Kazuo Okanoya. "Automatic recognition of element
classes and boundaries in the birdsong with variable sequences."
PloS one 11.7 (2016): e0159188.