========
tutorial
========

30 seconds to HVC
-----------------
.. code-block:: python

    import hvc

    # extract features to train machine learning models
    hvc.extract('extract_config.yml')

    # select models that best fit the data
    hvc.select('select_config.yml')

    # predict classes / labels for vocalizations using the fit models
    hvc.predict('predict_config.yml')

2 minutes to HVC
----------------
| Here we present two main workflows for using HVC:  
| 1. **autolabel**: for scientists that want to automate labeling of vocalizations  
| 2. **autocompare**: for researchers that want to compare different machine learning algorithms.  

autolabel
~~~~~~~~~
| Here's the steps in the workflow for autolabeling vocalizations.  
| Bold terms are defined on the page describing this workflow in detail, linked below this list.  
| 1. Pick a **model**  
| 2. Extract **features** for that model from song files that will be used to **train** the model.  
| 3. Pick the **hyperparameters**  
| 4. Fit the **model** to the data  
| 5. Select the **best** model  
| 6. **Predict** labels for unlabeled data using the fit model.
| To read about this workflow in detail, see :ref:`autolabel-workflow`

autocompare
~~~~~~~~~~~


In depth tutorials
------------------

.. toctree::
   :maxdepth: 2

   tutorial/writing_extract_yaml