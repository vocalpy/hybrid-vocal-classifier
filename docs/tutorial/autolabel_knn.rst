.. _autolabel-knn:


autolabel with k-Nearest Neighbors
==================================

Here’s a walkthrough for applying the k-Nearest Neighbors algorithm to
birdsong.

There’s three main *modules* in ``hvc`` that you will use in the
autolabel workflow: ``extract`` to extract features, ``select`` to
select a model, and ``predict`` to predict labels for unlabeled data.
The steps below walk you through doing that.

A convenient way to walk through this tutorial would be in iPython, so
you might first start iPython from the commmand line, like this:

::

    (my-hvc-environment) $ ipython

iPython is not installed automatically with ``hvc`` so you’ll need to
install it. If you’re using the ``conda`` package manager, this is as
easy as:

::

    (my-hvc-environment) $ conda install ipython

| You can also use Jupyter notebooks from the tutorial here:
| https://github.com/NickleDave/hybrid-vocal-classifier-tutorial

First you ``import`` the library so you can work with it.

.. code:: ipython3

    import hvc  # in Python we have to import a library before we can use it

0. Label a small set of songs to provide **training data** for the models, typically ~20-40 songs.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here you would label your own song, using your software of choice
(evsonganaly, Sound Analysis Pro, Praat) but for this example you can
download some data that is already hand labeled from a repository.

.. code:: ipython3

    hvc.utils.fetch('gy6or6.032312')
    hvc.utils.fetch('gy6or6.032612')

1. Pick a machine learning algorithm/\ **model** and the **features** used to train the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case we use the k-Nearest Neighbors (k-NN) algorithm. This
algorithm is quick to apply to data but at least one empirical study
shows that it `does not give the best accuracy on Bengalese finch
song <http://conference.scipy.org/proceedings/scipy2016/david_nicholson.html>`__.
You’ll use the features built into the library that have been tested
with k-NN. These features are based in part on those developed by the
Troyer lab (http://www.utsa.edu/troyerlab/software.html).

You specify the models and features in a configuration file (“config”
for short). More information about all the parameters in the config file
can be found on the page :ref:``writing-extract-config``. For now you
can just copy the text below and save it in some file. The config is
written in YAML, a language for writing data structures (such as
different types of variables in a programming language).

.. code:: yaml

    extract:
      spect_params:
        ref: evsonganaly
      segment_params:
        threshold: 1500 # arbitrary units of amplitude
        min_syl_dur: 0.01 # ms
        min_silent_dur: 0.006 # ms

      todo_list:
        -
          bird_ID : gy6or6
          file_format: evtaf
          feature_group:
            - knn
          data_dirs:
            - .\gy6or6\032612

          output_dir: .\gy6or6\

          labelset: iabcdefghjk

2. Extract features for that model from song files that will be used to train the model.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You call the ``extract`` module and pass it the name of the ``yaml``
config file as an argument. In the example below, the config file was
saved as ``'gy6or6_autolabel_example.knn.extract.config.yml'``.

.. code:: ipython3

    # 1. pick a model and 2. extract features for that model
    # Model and features are defined in extract.config.yml file.
    hvc.extract('gy6or6_autolabel_example.knn.extract.config.yml')

3. Pick the **hyperparameters** used by the algorithm as it trains the model on the data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we use a convenience function to get an estimate of what value for
our **hyperparameters** will give us the best accuracy when we train our
machine learning models. The k-Nearest Neighbors algorithm has one main
hyperparameter, the number of neighbors :math:`k` in feature space that
we look at to determine the label for a new syllable we are trying to
classify.

.. code:: ipython3

    # 3. pick hyperparameters for model
    # Load summary feature file to use with helper functions for
    # finding best hyperparameters.
    from glob import glob
    summary_file = glob('./extract_output*/summary*')
    summary_data = hvc.load_feature_file(summary_file)
    # In this case, we picked a k-nearest neighbors model
    # and we want to find what value of k will give us the highest accuracy
    cv_scores, best_k = hvc.utils.find_best_k(summary_data['features'],
                                              summary_data['labels'],
                                              k_range=range(1, 11))

4. Train, i.e., fit the **model** to the data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5. Select the **best** model based on some measure of accuracy.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Again we use a config file. In the config file, we specify the name of
the feature file saved by ``hvc.extract``. Again you can just copy and
paste the text below.

**The key things to modify here are the hyperparameter :math:`k` and the
name of the feature file. You will choose the value for :math:`k` based
on your results from running ``hvc.utils.find_best_k``. You will get the
name of the feature file from the directory created when you ran
``hvc.extract``. The name of the directory will be something like
``extract_output_bird_ID_date``. Make sure that on the line that says
``feature_file:``, you paste the name of the feature file after the
colon. The name will have a format like ``summary_file_bird_ID_date``.**

.. code:: yaml

    select:
      
      num_replicates: 10
      num_train_samples:
        start : 50
        stop : 250
        step : 50
      num_test_samples: 500

      models:
        -
          model_name: knn
          feature_list_indices: [0,1,2,3,4,5,6,7,8]
          hyperparameters:
            k : 4

      todo_list:
        - #1
          feature_file: .\gy6or6\extract_output_171031_214453\summary_feature_file_created_171031_214642
          output_dir: .\gy6or6\

Now you can use ``hvc.select`` to select the best model. ``hvc.select``
takes the name of the config file as an argument, which in this example
is ``gy6or6_autolabel.example.select.knn.config.yml``.

.. code:: ipython3

    # 4. Fit the **model** to the data and 5. Select the **best** model
    hvc.select('gy6or6_autolabel.example.select.knn.config.yml')

6. Using the fit model, **predict** labels for unlabeled data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here you also use a config file.

\*\* The key things to modify here is the ``model_meta_file`` parameter.
``hvc.select`` will also have created a directory, and for each model it
fit, it will have saved two files, a ``.model`` file and a ``.meta``
file. The ``.meta`` file contains all the metadata that ``hvc`` needs to
be able to use the ``.model`` file. You choose whichever ``.meta`` file
gave you the best results according to the metric you’re using, e.g. the
default of average accuracy across syllable classes. You also need to
specify the directories with unlabeled data, under the ``data_dirs``
section.*\*

.. code:: yaml

    predict:
      todo_list:
        -
          bird_ID : gy6or6
          file_format: evtaf
          data_dirs:
            - C:\Users\Seymour Snyder\Documents\example_song\032612
          model_meta_file: .\gy6or6\select_output_171031_215004\knn_k4\knn_200samples_replicate9.meta
          output_dir: .\gy6or6
          predict_proba: True
          convert: notmat

1. In a text editor, open
2. On the line that says ``model_meta_file:``, after the colon, paste
   the name of a meta file from the ``select`` output. The name will
   have a format like ``summary_file_bird_ID_date``.
3. Below the line that says ``data_dirs:``, after the dash, add the path
   to the other folder of data that you downloaded.

Lastly you use the ``hvc.predict`` module to predict labels for new
syllables. ``hvc.predict`` also takes a config file name as an argument.
In this example the file name is
``gy6or6_autolabel.example.knn.predict.config.yml``.

.. code:: ipython3

    # 6. **Predict** labels for unlabeled data using the fit model.
    hvc.predict('gy6or6_autolabel.example.predict.knn.config.yml')


.. parsed-literal::

    parsed predict config
    Changing to data directory: C:/Data/gy6or6_all_files/032612
    Processing audio file 1 of 39.
    Processing audio file 2 of 39.
    ...
    Processing audio file 39 of 39.
    predicting labels for features in file: features_from_032612_created_171206_013759
    converting to .not.mat files
    

Congratulations! You have auto-labeled an entire day’s worth of data in
just a few minutes!
