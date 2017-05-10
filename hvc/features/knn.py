"""
extracts features used with k-Nearest Neighbors algorithm
"""

import numpy as np

_duration = lambda onsets, offsets: offsets-onsets
_gapdurs = lambda onsets, offsets: onsets[1:] - offsets[:-1]

def duration(onsets,offsets,syls_to_use):
    """
    
    Parameters
    ----------
    onsets
    offsets
    syls_to_use
    
    Returns
    -------

    """
    return _duration(onsets,offsets)[syls_to_use]

def pre_duration(onsets,offsets,syls_to_use):
    """
    
    Parameters
    ----------
    onsets
    offsets

    Returns
    -------

    """
    pre = np.zeros((onsets.shape[-1],))
    pre[1:] = _duration(onsets,offsets)[:-1]
    return pre[syls_to_use]

def foll_duration(onsets,offsets,syls_to_use):
    """
    
    Parameters
    ----------
    onsets
    offsets

    Returns
    -------

    """

    foll = np.zeros((onsets.shape[-1],))
    foll[:-1] = _duration(onsets,offsets)[1:]
    return foll[syls_to_use]

def pre_gapdur(onsets,offsets,syls_to_use):
    """
    
    Parameters
    ----------
    onsets
    offsets

    Returns
    -------

    """

    pre = np.zeros((onsets.shape[-1],))
    pre[1:] = _gapdurs(onsets,offsets)
    return pre[syls_to_use]

def foll_gapdur(onsets,offsets,syls_to_use):
    """
    
    Parameters
    ----------
    onsets
    offsets
    labelset

    Returns
    -------

    """

    foll = np.zeros((onsets.shape[-1],))
    foll[:-1] = _gapdurs(onsets,offsets)
    return foll[syls_to_use]

def mn_amp_smooth_rect():
    """
    
    Returns
    -------

    """

    return

def mn_amp_rms():
    """
    
    Returns
    -------

    """


    return

def mean_spect_entropy():
    """
    
    Returns
    -------

    """

    return

def mean_hi_lo_ratio():
    """
    
    Returns
    -------

    """

    return

def delta_amp_smooth_rect():
    """
    
    Returns
    -------

    """

    return

def delta_entropy():
    """
    
    Returns
    -------

    """

    return

def delta_hi_lo_ratio():
    """
    
    Returns
    -------

    """

    return