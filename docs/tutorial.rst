========
tutorial
========

30 seconds to HVC
-----------------
.. code-block:: python

    import hvc

    # feature extraction
    hvc.extract('extract_config.yml')

    # model selection
    hvc.select('select_config.yml')

    # predict classes for new data
    hvc.predict('predict_config.yml')

2 minutes to HVC
----------------
Here we present two main workflows for using HVC:
1. **autolabel**: for scientists that want to automate labeling of vocalizations
2. **autocompare**: for researchers that want to compare different machine learning algorithms.

autolabel
~~~~~~~~~
Here's the steps in the workflow for autolabeling vocalizations. Bold terms are outlined on the pages that explain each
step.
1. Pick a **model**
2. Extract **features** for that model from song files that will be used to **train** the model.
3. Pick the **hyperparameters**
4. Fit the **model** to the data
5. Select the **best** model
6. **Predict** labels for unlabeled data using the fit model.

autocompare
~~~~~~~~~~~


In depth tutorials
------------------

.. toctree::
   :maxdepth: 2

   tutorial/writing_extract_yaml