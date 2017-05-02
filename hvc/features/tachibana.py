"""
Features used in Tachibana et al. 2014 [1]_.

Based on Matlab code written by R.O. Tachibana (rtachi@gmail.com) in Sept.
2013. These features were previously shown to be effective for classifying
Bengalese finch song syllables [1]_.

Many based on the CUIDADO feature set described in Peeters 2004 [2]_.

.. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. 
   "Semi-automatic classification of birdsong elements using a linear support vector machine."
   PloS one 9.3 (2014): e92584.

.. [2] Peeters, Geoffroy. "A large set of audio features for sound description (
   and classification) in the CUIDADO project." (2004).
"""

import numpy as np
from matplotlib.mlab import specgram

def duration(syl_spect):
    """
    computes duration as number of samples divided by sampling frequency
    
    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    duration: scalar
    """

    return syl_spect.sylAudio.shape[0] / syl_spect.sampFreq

def _spectrum(power):
    """
    helper function to calculate spectrum
    
    Parameters
    ----------
    power

    Returns
    -------
    spectrum
    """

    return 20 * np.log10(np.abs(power[1:, :]))

def mean_spectrum(syl_spect):
    """
    mean spectrum, as calculated in [1]_
    
    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    mean of power spectrum across time
    """

    spect = _spectrum(syl_spect.power)
    return np.mean(spect, axis=1)

def _cepstrum_for_mean(power,nfft):
    """
    helper function to compute cepstrum
    As computed in Tachibana et al. 2014 to get mean cepstrum
    
    Parameters
    ----------
    power

    Returns
    -------
    cepstrum
    
    """
    power2 = np.vstack((power, np.flipud(power[1:-1, :])))
    real_cepstrum = np.real(np.fft.fft(np.log10(np.abs(power2)), axis=0))
    return real_cepstrum[1:nfft // 2 + 1, :]


def mean_cepstrum(syl_spect):
    """
    mean cepstrum, as calculated in [1]_
    
    Parameters
    ----------
    power : numpy array, power spectrum for each time obtained by generating spectrogram of raw signal
    nfft : integer, number of samples used for each Fast Fourier Transform. Default (used by [1]) is 256
    
    Returns
    -------
    mean_cepstrum : 1d numpy array, mean of cepstrum (i.e. take mean across columns of spectrogram)
    """

    cepst = _cepstrum_for_mean(syl_spect.power,syl_spect.nfft)
    return np.mean(cepst, axis=1)

def _five_point_delta(x):
    """
    helper function to compute five-point delta as in Tachibana et al. 2014
    
    Parameters
    ----------
    x

    Returns
    -------

    """
    if len(x.shape) > 1:
        return -2 * x[:, :-4] - 1 * x[:, 1:-3] + 1 * x[:, 3:-1] + 2 * x[:, 4:]
    else: # if 1d vector, e.g. spectral slope
        return -2 * x[:-4] - 1 * x[1:-3] + 1 * x[3:-1] + 2 * x[4:]

def mean_delta_spectrum(syl_spect):
    """
    mean of 5-order delta of spectrum
    
    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    mean_deltra_spectrum 
    """

    if syl_spect.syl_audio.shape[-1] < (5 * syl_spect.nfft - 4 * syl_spect.overlap):
    # if number of time bins will be less than 5
        spect_width = syl_spect.nfft / 2
        # return a "delta spectrum" of 0
        delta_spectrum = np.zeros((spect_width,1))
    else:
        spect = _spectrum(syl_spect.power)
        delta_spectrum = _five_point_delta(spect)
    return np.mean(np.abs(delta_spectrum), axis=1)

def mean_delta_cepstrum(syl_spect):
    """
    mean of 5-order delta of spectrum
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    mean delta spectrum
    """

    if syl_spect.syl_audio.shape[-1] < (5 * syl_spect.nfft - 4 * syl_spect.overlap):
    # if number of time bins will be less than 5
        spect_width = syl_spect.nfft / 2
        # return a "delta cepstrum" of 0
        delta_cepstrum = np.zeros((spect_width, 1))
    else:
        cepst = _cepstrum_for_mean(syl_spect.power,syl_spect.nfft)
        delta_cepstrum = _five_point_delta(cepst)
    return np.mean(np.abs(delta_cepstrum), axis=1)

def _convert_spect_to_probability(power,freqs):
    """
    Helper function to compute features as computed in Tachibana et al. 2014.
    Converts spectrogram to "probability distribution" by taking sum of power for each column
    and then dividing power of each bin in that column by sum for that column
    
    Arguments
    ---------
    power : 2d numpy array, spectrogram where each element is power for that frequency and time bin
    freqs : 1d numpy array, frequency bins
    
    Returns
    -------
    prob : 2d numpy array, same size as power; spectrogram converted to probability
    freqs_mat : 2d numpy array, tiled frequency bins with same number of columns as power
    num_rows : int, number of rows in power
    num_cols : int, number of columns in power
    """
    num_rows, num_cols = power.shape
    freqs_mat = np.tile(freqs[:, np.newaxis], num_cols)
    # amplitude spectrum
    amplitude_spectrum = np.abs(power)
    # probability
    prob = amplitude_spectrum / np.matlib.repmat(np.sum(amplitude_spectrum, 0), num_rows, 1)
    return prob, freqs_mat, num_rows, num_cols

def spectral_centroid(prob,freqs_mat):
    """
    spectral centroid, mean of normalized amplitude spectrum
     
    Parameters
    ----------
    prob : 2d array, returned by _convert_spect_to_probability. Spectrogram converted to normalized amplitude spectra
    freqs_mat : 2d array, returned by _convert_spect_to_probability. Frequency bins tiled so columns = time bins
    
    Returns
    -------
    spectral centroid : 1d array with number of elements equal to width of prob.
                        Each element is spectral centroid for that time bin of prob.
                        As calculated in Tachibana et al. 2014
    """

    # 1st moment: centroid (mean of distribution)
    return np.sum(freqs_mat * prob, 0)  # mean

def _variance(freqs_mat,spect_centroid,num_rows,prob):
    """
    Helper function to compute variance.
    
    Parameters
    ----------
    freqs_mat
    spect_centroid
    num_rows
    prob

    Returns
    -------

    """

    return np.sum((np.power(freqs_mat - np.matlib.repmat(spect_centroid, num_rows, 1), 2)) * prob, 0)

def mean_spectral_centroid(syl_spect):
    """
    Mean of spectral centroid across syllable,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    mean_spectral_centroid : scalar, mean of spectral centroid across syllable
    """
    prob, freqs_mat = _convert_spect_to_probability(syl_spect.power)[:2]
    spect_centroid = _spectral_centroid(prob,freqs_mat)
    return np.mean(spect_centroid)

def mean_delta_spectral_centroid(syl_spect):
    """
    mean of 5-point delta of spectral centroid
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    mean_delta_spectral_centroid : scalar
    """

    prob, freqs_mat = _convert_spect_to_probability(syl_spect.power)[:2]
    spect_centroid = _spectral_centroid(prob,freqs_mat)
    delta_spect_centroid = _five_point_delta(spect_centroid)
    return np.mean(delta_spect_centroid)

def spectral_spread(power):
    """
    spectral spread, variance of normalized amplitude spectrum
    
    Parameters
    ----------
    power : 2d numpy array, spectrogram where each element is power for that frequency and time bin

    Returns
    -------
    spectral spread
    """
    prob, freqs_mat, num_rows = _convert_spect_to_probability(power)[:3]
    spect_centroid = _spectral_centroid(prob,freqs_mat)
    variance = _variance(mat,spect_centroid,num_rows,prob)
    return np.power(variance, 1 / 2)

def mean_spectral_spread(syl_spect):
    """
    mean of spectral spread across syllable,
    as computed in Tachibana et al. 2014

    Parameters
    ----------
    power

    Returns
    -------
    mean_spectral_spread : scalar, mean of spectral spread across syllable
    """
    return np.mean(spectral_spread(syl_spect.power))

def mean_delta_spectral_spread(syl_spect):
    """
    mean of 5-point delta of spectral spread
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_spread : scalar
    """

    return np.mean(_five_point_delta(spectral_spread(syl_spect.power)))

def spectral_skewness(power):
    """
    spectral skewness, measure of asymmetry of normalized amplitude spectrum around mean
    
    Parameters
    ----------
    power

    Returns
    -------
    spectral skewness
    """

    prob, freqs_mat, num_rows = _convert_spect_to_probability(power)[:3]
    spect_centroid = _spectral_centroid(prob,freqs_mat)
    variance = _variance(freqs_mat,spect_centroid,num_rows,prob)
    skewness = np.sum((np.power(mat - np.matlib.repmat(spect_centroid, num_rows, 1), 3)) * prob, 0)
    return skewness / np.power(variance, 3 / 2)

def mean_spectral_skewness(syl_spect):
    """
    mean of spectral skewness across syllable,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    power

    Returns
    -------
    mean spectral skewness : scalar, mean of spectral skewness across syllable
    """

    return np.mean(spectral_skewness(syl_spect.power))

def mean_delta_spectral_skewness(syl_spect):
    """
    mean of 5-point delta of spectral skewness
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_skewness : scalar
    """

    return np.mean(_five_point_delta(spectral_skewness(syl_spect.power)))

def spectral_kurtosis(power):
    """
    spectral kurtosis, measure of flatness of normalized amplitude spectrum
    
    Parameters
    ----------
    power

    Returns
    -------
    spectral kurtosis
    """

    prob, freqs_mat, num_rows = _convert_spect_to_probability(power)[:3]
    spect_centroid = _spectral_centroid(prob,freqs_mat)
    variance = _variance(freqs_mat,spect_centroid,num_rows,prob)
    kurtosis = np.sum((np.power(freqs_mat - np.matlib.repmat(spect_centroid, num_rows, 1), 4)) * prob, 0)
    return kurtosis / np.power(variance, 2)

def mean_spectral_kurtosis(syl_spect):
    """
    mean of spectral kurtosis across syllable,
    as computed in Tachibana et al. 2014

    Parameters
    ----------
    syl_spect

    Returns
    -------
    mean_spectral_kurtosis
    """

    return np.mean(spectral_kurtosis(syl_spect.power))

def mean_delta_spectral_kurtosis(syl_spect):
    """
    mean of 5-point delta of spectral kurtosis
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    mean_delta_spectral_kurtosis
    """

    return np.mean(_five_point_delta(spectral_kurtosis(syl_spect.power)))

def spectral_slope(power,freq_bins):
    """
    spectral slope, slope from linear regression of normalized amplitude spectrum
    
    Parameters
    ----------
    power : 2d array,
    freqs : 1d array, frequency bins as returned by spectrogram

    Returns
    -------
    spectral_slope : 1d array
    """

    amplitude_spectrum = np.abs(power)
    num_rows, num_cols = amplitude_spectrum.shape
    spect_slope = np.zeros((1, num_cols))
    mat2 = np.stack((freq_bins, np.ones((num_rows, 1))),axis=-1)
    for n in range(num_cols):
        beta = np.linalg.solve(np.dot(mat2.T,mat2),
                               np.dot(mat2.T,amplitude_spectrum[:, n]))
        spect_slope[n] = beta[0]
    return spect_slope

def mean_spectral_slope(syl_spect):
    """
    mean of spectral slope across syllable
    
    Parameters
    ----------
    power
    freqs

    Returns
    -------
    mean_spectral_slope : scalar
    """

    return np.mean(spectral_slope(syl_spect.power, syl_spect.freqBins))

def mean_delta_spectral_slope(syl_spect):
    """
    mean of 5-point delta of spectral slope
    
    Parameters
    ----------
    power
    freqs

    Returns
    -------
    mean_delta_spectral_slope : scalar
    """

    return np.mean(_five_point_delta(spectral_slope(syl_spect.power, syl_spect.freqBins)))

def _cepstrum_for_pitch(power,freqs,nfft,samp_freq,min_freq,max_freq):
    """
    cepstrum as computed in Tachibana et al. 2014
    for the purposes of calculating pitch and pitch goodness

    Parameters
    ----------
    power
    freqs
    nfft
    samp_freq
    min_freq
    max_freq
    
    Returns
    -------
    mv
    mid
    min_quef
    """

    max_quef = np.round(samp_freq / min_freq) * 2
    min_quef = np.round(samp_freq / max_freq) * 2

    exs = np.vstack((amplitude_spectrum,
                     np.matlib.repmat(amplitude_spectrum[-1, :], nfft, 1),
                     np.flipud(amplitude_spectrum[1:-1,:])))
    cepstrum = np.real(np.fft.fft(np.log10(exs)))
    max_val = np.max(cepstrum[min_quef:max_quef, :], axis=0)
    max_id = np.argmax(cepstrum[min_quef:max_quef, :], axis=0)
    return max_val, max_id, min_quef

def pitch(syl_spect,min_freq=500,max_freq=6000):
    """
    pitch, as calculated in Tachibana et al. 2014.
    Peak of the cepstrum.
    
    Parameters
    ----------
    syl_spect
    min_freq
    max_freq
    
    Returns
    -------
    pitch : scalar
    """

    max_val, max_id, min_quef = _cepstrum_for_pitch(syl_spect.power,
                                                    syl_spect.freqBins,
                                                    syl_spect.nfft,
                                                    syl_spect.samp_freq,
                                                    min_freq,
                                                    max_freq)
    return syl_spect.samp_freq / (max_id + min_quef - 1)

def mean_pitch(syl_spect):
    """
    mean pitch as measured across syllable

    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    mean_pitch
    """
    return np.mean(pitch(syl_spect))

def mean_delta_pitch(syl_spect):
    """
    mean of 5-point delta of pitch
    
    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    mean_delta_pitch
    """
    return mean(_five_point_delta(pitch(syl_spect)))

def pitch_goodness(syl_spect,min_freq=500,max_freq=6000):
    """
    pitch goodness, as calculated in Tachibana et al. 2014
    
    Parameters
    ----------
    syl_spect
    min_freq
    max_freq
        
    Returns
    -------
    pitch goodness
    """

    return _cepstrum_for_pitch(syl_spect.power,
                               syl_spect.freqBins,
                               syl_spect.nfft,
                               syl_spect.samp_freq,
                               min_freq,
                               max_freq)[0]


def mean_pitch_goodness(syl_spect):
    """
    mean of pitch goodness across syllable

    Parameters
    ----------
    syl_spect
        
    Returns
    -------
    mean_pitch_goodness
    """
    return np.mean(pitch_goodness(syl_spect))

def mean_delta_pitch_goodness(syl_spect):
    """

    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    mean_delta_pitch_goodness
    """
    return mean(_five_point_delta(pitch_goodness(syl_spect)))

def amplitude(syl_spect):
    """
    amplitude in decibels,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    amplitude
    """

    amplitude_spectrum = np.abs(syl_spect.power)
    return 20 * np.log10(np.sum(amplitude_spectrum) / syl_spect.nfft)

def mean_amplitude(syl_spect):
    """
    mean of amplitude across syllable
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    mean_amplitude : scalar
    """

    return np.mean(amplitude(syl_spect))

def mean_delta_amplitude(syl_spect):
    """
    mean of 5-point delta of amplitude
    
    Parameters
    ----------
    syl_spect
    
    Returns
    -------
    mean_delta_amplitude
    """

    return np.mean(_five_point_delta(amplitude(syl_spect)))

def zero_crossings(syl_spect):
    """
    zero crossings, in units of Hertz
    
    Parameters
    ----------
    syl_spect

    Returns
    -------
    zero_crossings : scalar
    """

    zero_crosses = np.length(np.find(np.diff(sign(syl_spect.sylAudio)) != 0)) / 2
    dur = duration(syl_spect.sylAudio, syl_spect.sampFreq)
    return zero_crosses / dur