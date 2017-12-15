"""
extracts features used with k-Nearest Neighbors algorithm
"""

import numpy as np

import hvc.evfuncs

# helper function that calculates syllable durations
_duration = lambda onsets, offsets: offsets-onsets

# helper function that calculates duration of silent gaps between syllables
_gapdurs = lambda onsets, offsets: onsets[1:] - offsets[:-1]


def duration(onsets, offsets, syls_to_use):
    """
    durations of syllables, using onsets and offsets from segmentation
    
    Parameters
    ----------
    onsets : 1d numpy array
        syllable onset times as determined by a segmentation algorithm
    offsets : 1d numpy array
        syllable offset times as determined by a segmentation algorithm
    syls_to_use : 1d numpy Boolean array
        property of audiofileIO.song object, as set by set_syls_to_use(labels_to_use) 
    
    Returns
    -------
    _duration(onsets,offsets)[syls_to_use]
    """

    return _duration(onsets, offsets)[syls_to_use]


def pre_duration(onsets, offsets, syls_to_use):
    """
    duration of preceding syllable
        
    Parameters
    ----------
    onsets : 1d numpy array
        syllable onset times as determined by a segmentation algorithm
    offsets : 1d numpy array
        syllable offset times as determined by a segmentation algorithm
    syls_to_use : 1d numpy Boolean array
        property of audiofileIO.song object, as set by set_syls_to_use(labels_to_use) 
    
    Returns
    -------
    pre[syls_to_use] : 1d numpy array
        where foll[1:] = _duration(onsets,offsets)[1:] and pre[0]=0
        (because no syllable precedes the first syllable)
    """

    pre = np.zeros((onsets.shape[-1],))
    pre[1:] = _duration(onsets, offsets)[:-1]
    return pre[syls_to_use]


def foll_duration(onsets, offsets, syls_to_use):
    """
    duration of following syllable

    Parameters
    ----------
    onsets : 1d numpy array
        syllable onset times as determined by a segmentation algorithm
    offsets : 1d numpy array
        syllable offset times as determined by a segmentation algorithm
    syls_to_use : 1d numpy Boolean array
        property of audiofileIO.song object, as set by set_syls_to_use(labels_to_use) 

    Returns
    -------
    foll[syls_to_use] : 1d numpy array
        where foll[:-1] = _duration(onsets,offsets)[1:] and foll[-1]=0
        (because no syllable follows the last syllable)
    """

    foll = np.zeros((onsets.shape[-1],))
    foll[:-1] = _duration(onsets, offsets)[1:]
    return foll[syls_to_use]


def pre_gapdur(onsets, offsets, syls_to_use):
    """
    duration of silent gap between syllable and preceding syllable

    Parameters
    ----------
    onsets : 1d numpy array
        syllable onset times as determined by a segmentation algorithm
    offsets : 1d numpy array
        syllable offset times as determined by a segmentation algorithm
    syls_to_use : 1d numpy Boolean array
        property of audiofileIO.song object, as set by set_syls_to_use(labels_to_use) 

    Returns
    -------
    pre[syls_to_use] : 1d numpy array
        where pre[1:] = _gapdurs(onsets,offsets) and pre[0]=0
        (because no syllable precedes the first syllable)
    """

    pre = np.zeros((onsets.shape[-1],))
    pre[1:] = _gapdurs(onsets, offsets)
    return pre[syls_to_use]


def foll_gapdur(onsets, offsets, syls_to_use):
    """
    duration of silent gap between syllable and following syllable

    Parameters
    ----------
    onsets : 1d numpy array
        syllable onset times as determined by a segmentation algorithm
    offsets : 1d numpy array
        syllable offset times as determined by a segmentation algorithm
    syls_to_use : 1d numpy Boolean array
        property of audiofileIO.song object, as set by set_syls_to_use(labels_to_use) 

    Returns
    -------
    foll[syls_to_use] : 1d numpy array
        where foll[:-1] = _gapdurs(onsets,offsets)[1:] and foll[-1]=0
        (because no syllable follows the last syllable)
    """

    foll = np.zeros((onsets.shape[-1],))
    foll[:-1] = _gapdurs(onsets, offsets)
    return foll[syls_to_use]


def _smooth_rect_amp(syllable):
    """
    helper function to calculate smoothed rectified amplitude
    
    Parameters
    ----------
    syllable

    Returns
    -------
    smoothed : 1-d numpy array
        raw audio waveform amplitude,
        after bandpass filtering, squaring, and  
        and smoothing with evfuncs.smooth_data
    """

    return hvc.evfuncs.smooth_data(syllable.sylAudio,
                                   syllable.sampFreq,
                                   syllable.freqCutoffs)


def mn_amp_smooth_rect(syllable):
    """
    mean of smoothed rectified amplitude
    **from raw audio waveform**, not spectrogram
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean_smoothed_rectified : scalar
        np.mean(_smooth_rect_amp(syllable))
    """

    return np.mean(_smooth_rect_amp(syllable))


def mn_amp_rms(syllable):
    """
    
    Parameters
    ----------
    syllable : syllable object

    Returns
    -------
    root_mean_squared : scalar
        square root of value returned by mn_amp_smooth_rect
    """

    return np.sqrt(mn_amp_smooth_rect(syllable))


def _spect_entropy(syllable):
    """
    helper function that calculates spectral entropy for syllable spectrogram
    
    Parameters
    ----------
    syllable : syllable object

    Returns
    -------
    spectral_entropy : 1-d numpy array
        spectral entropy for each time bin in syllable spectrogram
        array will have length = number of columns in syllable.spect
    """
    psd = np.power(np.abs(syllable.spect), 2)
    psd_pdf = psd / np.sum(psd, axis=0)
    return -np.sum(psd_pdf * np.log(psd_pdf), axis=0)


def mean_spect_entropy(syllable):
    """
    mean of spectral entropy across syllable
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean(_spect_entropy(syllable))
    """

    return np.mean(_spect_entropy(syllable))


def _hi_lo_ratio(syllable, middle=5000):
    """
    helper function to calculate hi/lo ratio
    hi/lo ratio is ratio of sum of power in "high" frequencies
    and sum of power in "low" frequencies,
    where "hi" frequencies are those above "middle"
    and "low" frequencies are below "middle"
    
    Parameters
    ----------
    syllable : syllable object
    middle : int
        defaults to 5000

    Returns
    -------
    hi_lo_ratio : 1-d array
        hi/lo ratio for each time bin in syllable spectrogram
        array will have length = number of columns in syllable.spect
    """

    psd = np.power(np.abs(syllable.spect), 2)
    hi_ids = syllable.freqBins > middle
    lo_ids = syllable.freqBins < middle
    return np.log10(np.sum(psd[hi_ids, :], axis=0) /
                    np.sum(psd[lo_ids, :], axis=0))


def mean_hi_lo_ratio(syllable):
    """
    mean of hi/lo ratio across syllable
    
    Parameters
    ----------
    syllable

    Returns
    -------
    np.mean(_hi_lo_ratio(syllable))
    """

    return np.mean(_hi_lo_ratio(syllable))


def _delta_inds(syllable, delta_times):
    """
    helper function that converts times from percent of duration
    to seconds, then finds indices of time bins in sylllable
    spectrogram closest to those times
    
    Parameters
    ----------
    syllable : syllable object
    delta_times : list
        two-element list, early time and later time
        given in percent total duration of syllable

    Returns
    -------
    inds : list
        two-element list of indices
    
    Return values are used with _delta lambda function
    """
    dur = syllable.sylAudio.shape[-1] / syllable.sampFreq
    t_early = dur * delta_times[0]
    t_late = dur * delta_times[1]
    return [np.argmin(np.abs(syllable.timeBins - t_early)),
            np.argmin(np.abs(syllable.timeBins - t_late))]

_delta = lambda vec, inds: vec[inds[0]] - vec[inds[1]]


def delta_amp_smooth_rect(syllable, delta_times=[0.2, 0.8]):
    """
    change in smoothed rectified amplitude between two time points
    
    Parameters
    ----------
    syllable : syllable object
    delta_times : list
        two-element list, early time and later time
        given in percent total duration of syllable, default is [0.2,0.8]

    Returns
    -------
    delta_amp_smooth_rect : scalar
        _delta(_smooth_rect_amp(syllable))
    """
    inds = _delta_inds(syllable, delta_times)
    amp = _smooth_rect_amp(syllable)
    return _delta(amp,inds)


def delta_entropy(syllable, delta_times=[0.2, 0.8]):
    """
    change in entropy between two time points

    Parameters
    ----------
    syllable : syllable object
    delta_times : list
        two-element list, early time and later time
        given in percent total duration of syllable, default is [0.2,0.8]

    Returns
    -------
    delta_entropy : scalar
        _delta(_spect_entropy(syllable))
    """

    inds = _delta_inds(syllable, delta_times)
    entropy = _spect_entropy(syllable)
    return _delta(entropy,inds)


def delta_hi_lo_ratio(syllable, delta_times=[0.2, 0.8]):
    """
    change in hi/lo ratio between two time points

    Parameters
    ----------
    syllable
    delta_times : list
        two-element list, early time and later time
        given in percent total duration of syllable, default is [0.2,0.8]

    Returns
    -------
    delta_hi_lo_ratio : scalar
        _delta(_hi_lo_ratio(syllable))
    """
    inds = _delta_inds(syllable, delta_times)
    hi_lo = _hi_lo_ratio(syllable)
    return _delta(hi_lo, inds)
