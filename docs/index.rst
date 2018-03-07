.. hybrid-vocal-classifier documentation master file, created by
   sphinx-quickstart on Sat Jun 24 22:04:45 2017.

hybrid-vocal-classifier (`hvc`)
===============================

Voice to text for songbirds
---------------------------

.. image:: /images/gr41rd41_song.png

hybrid-vocal-classifier (`hvc` for short) makes it easy to
segment and classify vocalizations with machine learning algorithms,
and to compare the performance of different algorithms.

The main application is for scientists studying birdsong.
(You can read more about that on the :doc:`more_about` page.)

Running `hvc` requires almost no coding.
The user writes configuration files in YAML, a simple language that
is meant to be easy for humans to read and write.
Most users will only have to copy the example .yml files and then
change a couple of parameters.

Here's a code sample that gives a high-level view of how you run `hvc`:

.. code-block:: python

   import hvc

   # extract features from audio to train machine learning models
   hvc.extract('extract_config.yml')
   # train models and select model with best accuracy
   hvc.select('select_config.yml')
   # use trained model to predict labels for unlabeled data
   hvc.predict('predict_config.yml')

Advantages of hybrid-vocal-classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+ frees up hundreds of hours spent hand labeling data
+ completely open source, free
+ makes it easy to compare multiple machine learning algorithms
+ almost no coding required, configurable with text files
+ built on top of Python packages road-tested by the greater data-science community:
   `numpy <http://www.numpy.org/>`_ , `scipy <https://www.scipy.org/scipylib/index.html>`_ ,
   `matplotlib <https://matplotlib.org/>`_ , `scikit-learn <http://scikit-learn.org/stable/>`_ ,
   `keras <https://keras.io/>`_

Documentation
~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   tutorial
   reference
   development

Installation
~~~~~~~~~~~~

see :ref:`install`

Contribute
~~~~~~~~~~

- Issue Tracker: https://github.com/NickleDave/hybrid-vocal-classifier/issues
- Source Code: https://github.com/NickleDave/hybrid-vocal-classifier/

Support
~~~~~~~

| If you are having issues, please let us know.
| Please post bugs on the Issue Tracker:
| https://github.com/NickleDave/hybrid-vocal-classifier/issues
| And please ask questions in the users' group:
| https://groups.google.com/forum/?hl=en#!forum/hvc-users/join

License
~~~~~~~

BSD license.

Citations, repositories, and related work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| For use of the library, please cite its DOI:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1154239.svg
   :target: https://doi.org/10.5281/zenodo.844932

| To cite the algorithms used, please see the listing in :ref:`citations`.
| A list of repositories of birdsong is here: :ref:`repos` 
| A list of related works is here: :ref:`related` 
| To suggest or contribute algorithms or repositories:
|  Please feel free to start an issue on the Github repository
|  https://github.com/NickleDave/hybrid-vocal-classifier/issues
|  or comment in the users' group:
|  https://groups.google.com/forum/?hl=en#!forum/hvc-users/join

Code of Conduct
~~~~~~~~~~~~~~~
We welcome contributions to the codebase and the documentation,
and are happy to help first-time contributors through the process.
Project maintainers and contributors are expected to uphold
the code of conduct described here: :ref:`code-of-conduct`
