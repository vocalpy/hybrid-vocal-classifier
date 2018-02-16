.. _install:

============
Installation
============

| Step by step instructions are below.
| If you need help, please join our email list at `hvc-users@google-groups.com` and ask a new question.

Note that the ``$`` and ``>`` prompts below are just to indicate that you're in the command line,
you don't have to type them. If the command is the same on Mac/Linux/Windows then only the ``$``
prompt is shown.

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

Install of bleeding-edge / development version
----------------------------------------------
using the Anaconda distribution and the `conda` package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| 1. Install the Anaconda distribution for your operating system: https://www.anaconda.com/download/
| 
| 2. Create an environment for the bleeding-edge version
| ``$ conda create --name hvc-bleeding-edge python=3.5 numpy scipy sklearn matplotlib pyyaml keras tensorflow``
| (say yes to everything)
|
| 3. ``git clone`` the repository
| ``$ activate hvc-bleeding-edge``
| ``(hvc-bleeding-edge) $ git clone https://github.com/NickleDave/hybrid-vocal-classifier.git``
|
| 4. use pip to install the code as editable in the conda environment, using the "-e" flag:
| ``(hvc-bleeding-edge) C:/Roman> pip install -e hybrid-vocal-classifier``
|
| and then whenever you want to get the most up-to-date version you can execute
| ``(hvc-bleeding-edge) $ git pull``
| and as long as you haven't made any changes to the code base, git should just pull new changes in from the remote and merge them with the old version
|
| You probably also want to install Jupyter and iPython in the bleeding-edge environment.
| If you try to run them without installing, you will run the versions in the conda root environment, but they won't know that hvc et al. are installed.
| ``(hvc-bleeding-edge) $ conda install ipython jupyter``
|
| You should now be able to start iPython or a Jupyter notebook and ``import hvc`` to work with it.
