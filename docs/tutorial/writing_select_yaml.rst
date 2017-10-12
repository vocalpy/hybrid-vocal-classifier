===================================================
how to write yaml files used by the `select` module
===================================================

As described in the introduction, a crucial step in using hybrid vocal classifier is
selecting which models to use. This can be done in an automated way using the `select`
module. Like the `extract` and `predict` modules, the `select` module works by parsing
configuration files. Below the steps are outlined in writing 
the configuration files in yaml format.

what the select module gets out of the config file: models and data
-------------------------------------------------------------------
There are two required elements in a select config file, that
 correspond to the two main things that the `select` module needs to know:
  1. `models`: what models to test. A Python list of dictionaries, as described below.
  2. `todo_list`: where the data is to train and test those models. Another Python list of dictionaries,
     also described below.

The parser that parses the `select` config file is written so that you don't have to repeat yourself.
You can put one `models` list at the top of the file, and then for each dataset in the `todo_list`,
the `select` module will train and test all the models that are specified in that top-level `models` list. Like so:

..include

However you can also define a `models` dictionary for each `todo_list`, in case you need to test
different models for different datasets, and want to run them all from one script.

..include

the `models` list
-----------------
To be parsed correctly, the `models` list needs to have the right structure.
In yaml terminology, this is a list.
Once parsed into Python, it becomes a list of dictionaries.
For that reason the structure is described in terms of the keys and values
required for each dictionary.
Each dictionary in the list represents one model that the `select` module will test.
There are a couple of required keys for each model dictionary.

required key 1: hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These models are found using machine learning algorithms.
A model can be thought of as a function with parameters, like the beta terms of a linear
regression.
To find these parameters, the algorithm must train on the data, and this training also has
parameters, for example the number of neighbors used by the K-nearest neighbor algorithm.
These parameters of the algorithm are known as **hyperparameters** to distinguish them from
the parameters found by the algorithm.


the `todo list`
--------------
