.. _install:

============
Installation
============

| Step by step instructions are below.
| If you need help, please join our email list at `hvc-users@google-groups.com` and ask a new question.

**Note that the ``$`` and ``>`` prompts below are just to indicate that you're in the command line,
you don't have to type them. If the command is the same on Mac/Linux/Windows then only the ``$``
prompt is shown.**

Easy install for general users
------------------------------
using the Anaconda distribution and the `conda` package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following should work on Linux, Mac, and Windows.

| 1. Install the Anaconda distribution for your operating system: https://www.anaconda.com/download/
|
| 2. Add the conda forge channel which contains some of the libraries necessary for *hybrid-vocal-classifier* to run.
|
|  ``$ conda config --add channels conda-forge``
|  For more about conda forge, see: https://conda-forge.org
|  If you have previously added this channel, you do not need to do so again.
|  You can check whether it's added by running the following command:
|  ``$ conda config --show``
|  If the channel is added, you should see something like the following lines in the config output:
|
.. code-block:: console

   channels:
   - conda-forge

| 3. Create a conda environment.
|
|  ``$ conda create -n hvc python=3.5``
|  **Currently the environment must use Python 3.5.**
|  The environment allows you have to install the libraries necessary for *hybrid-vocal-classifier* to run.
|  For a more in-depth explanation see https://conda.io/docs/user-guide/concepts.html#conda-environments.
| 
| 4. Activate the environment
| 
|  Mac/Linux:
|  ``$ source activate hvc``
|  Windows:
|  ``> activate hvc``
|   After you activate the environment, its name will appear in parentheses before the terminal prompt.
|   ``(hvc)$``
|
| 5. Install *hybrid-vocal-classifier* into the environment
|
|  ``(hvc)$ conda install -c nickledave hybrid-vocal-classifier``
|
| 6. Test whether the install worked.
|
| ``(hvc)$ python``
| ``>>> import hvc``

If the above line executes without any ``module not found`` error,
you have successfully installed *hybrid-vocal-classifier*.

For developers
--------------
using the Anaconda distribution and the `conda` package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| 1. Install the Anaconda distribution for your operating system: https://www.anaconda.com/download/
| 
| 2. ``git clone`` the repository
|  https://github.com/NickleDave/hybrid-vocal-classifier.git
| 
| 3. Open bash (Linux/Mac) or command prompt (Windows) and navigate to the *hybrid-vocal-classifier* directory.
|  (you don't have to type the prompt ``$``/ shown below, it's just there to indicate
|   that this is being done in bash / command prompt)
|  ``$ cd ~/Documents/hybrid-vocal-classifier``
|  (or wherever you downloaded / cloned the repository)
| 
| 4. Create a conda environment.
| 
|  ``$ conda env create -f environment.yml``
|  The environment allows you have to install the libraries necessary for *hybrid-vocal-classifier* to run.
|  For a more in-depth explanation see https://conda.io/docs/user-guide/concepts.html#conda-environments .
| 
| 5. Activate the environment
| 
|  Linux/Mac:
|  ``$ source activate hvc-env``
|  Windows:
|  ``> activate hvc-env``

You should now be able to start IPython or a Jupyter notebook and ``import hvc`` to work with it.
