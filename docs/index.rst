.. hybrid-vocal-classifier documentation master file, created by
   sphinx-quickstart on Sat Jun 24 22:04:45 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

hybrid-vocal-classifier (HVC)
=============================

Voice to text for songbirds
---------------------------

.. image:: /images/gr41rd41_song.png

hybrid-vocal-classifier (HVC for short) makes it easy to
segment and classify vocalizations with machine learning algorithms,
and to compare the performance of different algorithms.

The main application is for scientists studying birdsong.
(You can read more about that on the :doc:`more_about` page.)

Running HVC requires almost no coding.
The user writes configuration files in YAML, a simple language that
is meant to be easy for humans to read and write.
Most users will only have to copy the example .yml files and then
change a couple of parameters.

Here's a code sample that gives a high-level view of how you run HVC:

.. code-block:: python

   import hvc

   # extract features from audio to train machine learning models
   hvc.extract('extract_config.yml')
   # train models and select model with best accuracy
   hvc.select('select_config.yml')
   # use trained model to predict labels for unlabeled data
   hvc.predict('predict_config.yml')

Advantages of hybrid-vocal-classifier:
--------------------------------------

+ frees up hundreds of hours spent hand labeling data
+ completely open source, free
+ makes it easy to compare multiple machine learning algorithms
+ almost no coding required, configurable with text files
+ built on top of Python packages road-tested by the greater data-science community:
   `numpy <http://www.numpy.org/>`_ , `scipy <https://www.scipy.org/scipylib/index.html>`_ ,
   `matplotlib <https://matplotlib.org/>`_ , `scikit-learn <http://scikit-learn.org/stable/>`_ ,
   `keras <https://keras.io/>`_

Documentation
-------------

.. toctree::
   :maxdepth: 2

   tutorial
   reference
   development

Installation
------------

see :ref:`install`

Contribute
----------

- Issue Tracker: https://github.com/NickleDave/hybrid-vocal-classifier/issues
- Source Code: https://github.com/NickleDave/hybrid-vocal-classifier/

Support
-------

| If you are having issues, please let us know.
| We have a mailing list located at: `hvc-users@google-groups.com`

License
-------

BSD license.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
