========
tutorial
========

2 minutes to HVC
----------------
| Here we present two main workflows for using HVC:  
| 1. **autolabel**: for songbird scientists that want to automate labeling of vocalizations  
| 2. **autocompare**: for researchers that want to compare different machine learning algorithms.  

autolabel
~~~~~~~~~
| Here's the steps in the workflow for autolabeling vocalizations.  
| Bold terms are defined on the page describing this workflow in detail, linked below this list.  
| 0. Label a small set of songs to provide **training data** for the models, typically ~20 songs.
| 1. Pick a machine learning algorithm/**model** and the **features** used to train the model.  
| 2. Extract features for that model from song files that will be used to train the model.  
| 3. Pick the **hyperparameters** used by the algorithm as it trains the model on the data. 
| 4. Train, i.e., fit the **model** to the data  
| 5. Select the **best** model based on some measure of accuracy.  
| 6. Using the fit model, **Predict** labels for unlabeled data.
| To read about this workflow in detail, see :ref:`autolabel-workflow`
Below is a simple script that loads sample data from a repository and then walks through the rest of the steps. 
The script and the config files it uses are also described in more detail on the :ref:`autolabel-workflow` page. 

.. literalinclude:: ./tutorial/autolabel.py

autocompare
~~~~~~~~~~~
| Here's the steps in the workflow for comparing different machine learning models.  
| Bold terms are defined on the page describing this workflow in detail, linked below this list.  
| 0. Label a small set of songs to provide **training data** for the models, typically ~20 songs.
| 1. Pick the **models** you will compare.  
| 2. Extract **features** for those models from song files that will be used to **train** the model.  
| 3. Pick the **hyperparameters** for the different models. 
| 4. Fit the **models** to the data  
| 5. Select the **best** model
| To read about this workflow in detail, see :ref:`autocompare-workflow`

In depth tutorials
------------------

.. toctree::
   :maxdepth: 2

   tutorial/writing_extract_yaml