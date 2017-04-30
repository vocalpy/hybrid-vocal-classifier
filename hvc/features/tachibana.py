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

def duration(syl,samp_freq):
    """
    duration
    """

    return syl.shape[0] / samp_freq

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

def mean_spectrum(power):
    """
    mean spectrum, as calculated in [1]_
    
    Parameters
    ----------
    power : numpy array, power spectrum for each time obtained by generating spectrogram of raw signal
    
    Returns
    -------
    mean of power spectrum across time
    """

    spect = _spectrum(power)
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


def mean_cepstrum(power,nfft=256):
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

    cepst = _cepstrum_for_mean(power,nfft)
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

def mean_delta_spectrum(power,nfft=256,overlap=192,spmax=128):
    """
    mean of 5-order delta of spectrum
    
    Parameters
    ----------
    power
    nfft
    overlap
    spmax

    Returns
    -------
    mean_deltra_spectrum : 
    """

    # 5-order delta
    if syl.shape[-1] < (5 * nfft - 4 * overlap):
        delta_spectrum = np.zeros(spmax, 1)
    else:
        spect = _spectrum(power)
        delta_spectrum = _five_point_delta(spect)
    return np.mean(np.abs(delta_spectrum), axis=1)

def mean_delta_cepstrum(power,spmax=128):
    """
    mean of 5-order delta of spectrum
    
    Parameters
    ----------
    power
    spmax

    Returns
    -------
    mean delta spectrum
    
    """
    # 5-order delta
    if syl.shape[-1] < (5 * nfft - 4 * overlap):
        delta_cepstrum = np.zeros(spmax, 1)
    else:
        cepst = _cepstrum_for_mean(power, nfft)
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
    num_rows :
    num_cols :
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

def mean_spectral_centroid(power):
    """
    Mean of spectral centroid across syllable,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    power : 2d numpy array, spectrogram where each element is power for that frequency and time bin

    Returns
    -------
    mean_spectral_centroid : scalar, mean of spectral centroid across syllable
    """
    prob, mat = _convert_spect_to_probability(power)[:2]
    spect_centroid = _spectral_centroid(prob,mat)
    return np.mean(spect_centroid)

def mean_delta_spectral_centroid(power):
    """
    mean of 5-point delta of spectral centroid
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_centroid : scalar
    """

    prob, mat = _convert_spect_to_probability(power)[:2]
    spect_centroid = _spectral_centroid(prob,mat)
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
    prob, mat, num_rows = _convert_spect_to_probability(power)[:3]
    spect_centroid = _spectral_centroid(prob,mat)
    variance = _variance(mat,spect_centroid,num_rows,prob)
    return np.power(variance, 1 / 2)

def mean_spectral_spread(power):
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
    return np.mean(spectral_spread(power))

def mean_delta_spectral_spread(power):
    """
    mean of 5-point delta of spectral spread
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_spread : scalar
    """

    return np.mean(_five_point_delta(spectral_spread(power)))

def spectral_skewness(power):
    """
    spectral skewness, measure of asymmetry of normalized amplitude spectrum around mean
    
    Parameters
    ----------
    power

    Returns
    -------

    """

    prob, mat, num_rows = _convert_spect_to_probability(power)[:3]
    spect_centroid = _spectral_centroid(prob,mat)
    variance = _variance(mat,spect_centroid,num_rows,prob)
    skewness = np.sum((np.power(mat - np.matlib.repmat(spect_centroid, num_rows, 1), 3)) * prob, 0)
    return skewness / np.power(variance, 3 / 2)

def mean_spectral_skewness(power):
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

    return np.mean(spectral_skewness(power))

def mean_delta_spectral_skewness(power):
    """
    mean of 5-point delta of spectral skewness
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_skewness : scalar
    """

    return np.mean(_five_point_delta(spectral_skewness(power)))

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

def mean_spectral_kurtosis(power):
    """
    mean of spectral kurtosis across syllable,
    as computed in Tachibana et al. 2014

    Parameters
    ----------
    power

    Returns
    -------
    mean_spectral_kurtosis
    """

    return np.mean(spectral_kurtosis(power))

def mean_delta_spectral_kurtosis(power):
    """
    mean of 5-point delta of spectral kurtosis
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_kurtosis
    """

    return np.mean(_five_point_delta(spectral_kurtosis(power)))

def spectral_slope(power,freqs):
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
    mat2 = np.stack((freqs, np.ones((num_rows, 1))),axis=-1)
    for n in range(num_cols):
        beta = np.linalg.solve(np.dot(mat2.T,mat2),
                               np.dot(mat2.T,amplitude_spectrum[:, n]))
        spect_slope[n] = beta[0]
    return spect_slope

def mean_spectral_slope(power, freqs):
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

    return np.mean(spectral_slope(power, freqs))

def mean_delta_spectral_slope(power,freqs):
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

    return np.mean(_five_point_delta(spectral_slope(power, freqs)))

def _cepstrum_for_pitch(power,freqs,nfft,samp_freq,min_freq=500,max_freq=6000):
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

def pitch(power,freqs,nfft,samp_freq,min_freq,max_freq):
    """
    pitch, as calculated in Tachibana et al. 2014.
    Peak of the cepstrum.
    
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
    pitch : scalar
    """

    max_val, max_id, min_quef = _cepstrum_for_pitch(power,freqs,nfft,samp_freq)
    return samp_freq / (max_id + min_quef - 1)

def mean_pitch(power,freqs,nfft,samp_freq):
    """
    
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

    """
    return np.mean(pitch(power,freqs,nfft,samp_freq,min_freq,max_freq))

def mean_delta_pitch(power, freqs, nfft, samp_freq, min_freq, max_freq):
    """
    mean of 5-point delta of pitch
    
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

    """
    return mean(_five_point_delta(pitch(power, freqs, nfft, samp_freq, min_freq, max_freq)))

def pitch_goodness(power,freqs,nfft,samp_freq, min_freq, max_freq):
    """
    pitch goodness, as calculated in Tachibana et al. 2014
    
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

    """

    return _cepstrum_for_pitch(power, freqs, nfft, samp_freq, min_freq, max_freq)[0]


def mean_pitch_goodness(power, freqs, nfft, samp_freq, min_freq, max_freq):
    """
    mean of pitch goodness across syllable

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

    """
    return np.mean(pitch_goodness(power, freqs, nfft, samp_freq, min_freq, max_freq))

def mean_delta_pitch_goodness(power, freqs, nfft, samp_freq):
    """

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

    """
    return mean(_five_point_delta(pitch_goodness(power, freqs, nfft, samp_freq, min_freq, max_freq)))

def amplitude(power,nfft):
    """
    amplitude in decibels,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    power
    nfft

    Returns
    -------
    amplitude
    """

    amplitude_spectrum = np.abs(power)
    return 20 * np.log10(np.sum(amplitude_spectrum) / nfft)

def mean_amplitude(power, nfft):
    """
    mean of amplitude across syllable
    
    Parameters
    ----------
    power
    nfft

    Returns
    -------
    mean_amplitude : scalar
    """

    return np.mean(amplitude(power, nfft))

def mean_delta_amplitude(power, nfft):
    """
    mean of 5-point delta of amplitude
    
    Parameters
    ----------
    power
    nfft

    Returns
    -------
    mean_delta_amplitude
    """

    return np.mean(_five_point_delta(amplitude(power, nfft)))

def zero_crossings(syl, samp_freq):
    """
    zero crossings, in units of Hertz
    
    Parameters
    ----------
    syl

    Returns
    -------
    zero_crossings : scalar
    """

    zero_crosses = np.length(np.find(np.diff(sign(syl)) != 0)) / 2
    dur = duration(syl, samp_freq)
    return zero_crosses / dur

def extract_svm_features(syls,fs,nfft=256,spmax=128,overlap=192,minf=500,
                         maxf=6000):
    """


    Parameters
    ----------
    syls : Python list of numpy vector
        each vector is the raw audio waveform of a segmented syllable
    fs : integer
        sampling frequency
    nfft : integer
        number of samples for each Fast Fourier Transform (FFT) in spectrogram.
        Default is 256.
    spmax : integer
        Default is 128.
    overlap : integer
        number of overlapping samples in each FFT. Default is 192.
    minf : integer
        minimum frequency in FFT
    maxf : integer
        maximum frequency in FFT

    Returns
    -------
    feature_arr : numpy array
            with dimensions of n (number of syllables) x 532 acoustic features

    References
    ----------
    .. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
    automatic classification of birdsong elements using a linear support vector
     machine." PloS one 9.3 (2014): e92584.

    """
    # for below, need to have a 'filter' option in extract files
    # and then have named_spec_params where one would be 'Tachibana', another 'Koumura', another 'Sober', etc.

    # spectrogram and cepstrum
    syl_diff = np.diff(syl) # Tachibana applied a differential filter
    # note that the matlab specgram function returns the STFT by default
    # whereas the default for the matplotlib.mlab version of specgra
    # returns the PSD. So to get the behavior of matplotlib.mlab.specgram
    # to match, mode must be set to 'complex'
    power,freqs = specgram(syl_diff,NFFT=nfft,Fs=fs,window=np.hanning(nfft),
                   noverlap=overlap,
                   mode='complex')[0:2]  # don't keep returned time vector
