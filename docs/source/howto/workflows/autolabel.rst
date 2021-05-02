.. _autolabel:

=========
autolabel
=========

This page presents a high-level overview of the "autolabel" workflow. The description is aimed 
at scientists studying songbirds that want to automate annotation of their data, without having
to know all the nitty-gritty details of machine learning.

Here's the bare minimum background information you need to know:  
To automate labeling of birdsong syllables, `hvc` implements previously proposed machine 
learning algorithms. All the algorithms in `hvc` are **supervised learning** algorithms.
That means you need to provide them some hand-labeled training data. Using this data, the
algorithms "learn" to classify syllables. More precisely, the algorithms produce a function
that maps input features, such as acoustic parameters like pitch or duration, to a syllable class.
(This approach is distinct from clustering methods that can assign the identity of a syllable 
to some class with little or no input from the user, and without labeling training data. 
Unfortunately such algorithms are by and large still not as accurate as supervised learning
algorithms.)

That should hopefully give you just enough background follow the outline below
of the steps in the workflow for autolabeling vocalizations.

0. Label **training data**
-------------------------
Label a small set of songs to provide **training data** for the models, typically 20-40 songs.

.. literalinclude:: autolabel.py
   :lines: 1-7

1. Pick a **model**  
-------------------
Pick a machine learning algorithm/**model** and the **features** used to train the model.  
The models and features are specified in a configuration file ("config") written in YAML,
a very simple language meant to represent data in way that's easy for humans to read.

.. literalinclude:: gy6or6_autolabel_example.knn.extract.config.yml
   :lines: 13-14

Currently the highest accuracy is obtained with the support vector machine with a radial
basis function (SVM-RBF) or a convolutional neural net model ("flatwindow").
These results were obtained by the testing the models on the song of Bengalese finches.
At this time the only other model implemented is a k-Nearest Neighbors model (k-NN).
If you do not have a graphic processor unit (GPU), it will probably be easiest to use the SVM-RBF.
If you do have a GPU, you probably want to use the flatwindow model.

2. Extract **features** for that model from song files
------------------------------------------------------
Extract features for that model from the training data, i.e,. the song files
that will be used to train the model.  

.. literalinclude:: autolabel.py
   :lines: 9-11

Each machine learning model is fit to a set of **features**. These are either acoustic parameters
extracted from the song or spectrograms of the song itself. More information about what features
to use can be found on the pages about each model.

3. Pick the **hyperparameters**  
-------------------------------
Pick the **hyperparameters** used by the algorithm as it trains the model on the data. 
Hyperparameters can be thought of as the "knobs" on the algorithm that controls how it
learns. More on the hyperparameters for each algorithm can be found in the :ref:`walkthroughs`
for each algorithm on the main :ref:`tutorial` page.

.. literalinclude:: autolabel.py
   :lines: 13-22

4. Fit the **model** to the data  
--------------------------------
Train, i.e., fit the **model** to the data  

5. Select the **best** model 
----------------------------
Select the **best** model based on some measure of accuracy.  

.. literalinclude:: autolabel.py
   :lines: 24-25

6. **Predict** labels for unlabeled data using the fit model.
-------------------------------------------------------------
Using the fit model, **predict** labels for unlabeled data.

.. literalinclude:: autolabel.py
   :lines: 27-28
