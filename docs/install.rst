.. _install:

============
Installation
============

| Step by step instructions are below.
| If you need help, please join our email list at `hvc-users@google-groups.com` and ask a new question.

using the Anaconda distribution and the `conda` package manager
---------------------------------------------------------------
The following should work on Linux, Mac, and Windows.

| 1. Install the Anaconda distribution for your operating system: https://www.anaconda.com/download/
| 
| 2. Download the *hybrid-vocal-classifier* files
| (or ``git clone`` the repository if you are familiar with git)
| 
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


