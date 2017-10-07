.. _autolabel-workflow:

==================
autolabel workflow
==================

The autolabel workflow is for scientists that want to automate labeling of vocalizations.
A list of the steps is below.

| 1. Pick a **model**  
| 2. Extract **features** for that model from song files that will be used to **train** the model.  
| 3. Pick the **hyperparameters**  
| 4. Fit the **model** to the data  
| 5. Select the **best** model  
| 6. **Predict** labels for unlabeled data using the fit model.

1. Pick a **model**  
-------------------
Currently the highest accuracy is obtained with the support vector machine with a radial
 basis function (SVM-RBF) or a convolutional neural net model ("flatwindow").
These results were obtained by the testing the models on the song of Bengalese finches.
At this time the only other model implemented is a k-Nearest Neighbors model (k-NN).
If you do not have a graphic processor unit (GPU), it will probably be easiest to use the SVM-RBF.
If you do have a GPU, you probably want to use the flatwindow model.


2. Extract **features** for that model from song files
------------------------------------------------------
Each machine learning model is fit to a set of **features**. These are either acoustic parameters
extracted from the song or spectrograms of the song itself. More information about what features
to be use can be found on the pages about each model.

3. Pick the **hyperparameters**  
-------------------------------


4. Fit the **model** to the data  
--------------------------------


5. Select the **best** model 
----------------------------


6. **Predict** labels for unlabeled data using the fit model.
-------------------------------------------------------------

