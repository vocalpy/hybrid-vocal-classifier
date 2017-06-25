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

In depth tutorials
------------------

.. toctree::
   :maxdepth: 2

   tutorial/writing_extract_yaml