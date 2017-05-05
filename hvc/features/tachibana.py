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

def duration(syllable):
    """
    computes duration as number of samples divided by sampling frequency
    
    Parameters
    ----------
    syllable
    
    Returns
    -------
    duration: scalar
    """

    return syllable.sylAudio.shape[0] / syllable.sampFreq

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

def mean_spectrum(syllable):
    """
    mean spectrum, as calculated in [1]_
    
    Parameters
    ----------
    syllable
    
    Returns
    -------
    mean of power spectrum across time
    """

    spect = _spectrum(syllable.power)
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


def mean_cepstrum(syllable):
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

    cepst = _cepstrum_for_mean(syllable.power,syllable.nfft)
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
        if x.shape[0] < 5:
            # if length of 1d vector is less than five, can't take five-point
            # delta--would return empty array that then raises warning when
            # you take mean. So instead return array of zeros.
            # Original MATLAB code solved this by replacing NaNs
            # in feature array with zeros, so this code produces the same
            # behavior / features.
            return np.zeros((x.shape[0],))
        else:
            return -2 * x[:-4] - 1 * x[1:-3] + 1 * x[3:-1] + 2 * x[4:]

def mean_delta_spectrum(syllable):
    """
    mean of 5-order delta of spectrum
    
    Parameters
    ----------
    syllable
    
    Returns
    -------
    mean_deltra_spectrum 
    """

    if syllable.power.shape[-1] < 5:
        #can't take five point delta if less than five time bins
        # so return zeros instead, as original code did.
        # Return vector of length = number of frequency bins - 1
        # Originally done to remove "DC component", first coefficient
        # of FFT.
        return np.zeros((syllable.power.shape[0],))
    else:
        spect = _spectrum(syllable.power)
        delta_spectrum = _five_point_delta(spect)
        return np.mean(np.abs(delta_spectrum), axis=1)

def mean_delta_cepstrum(syllable):
    """
    mean of 5-order delta of spectrum
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean delta spectrum
    """

    if syllable.power.shape[-1] < 5:
        #can't take five point delta if less than five time bins
        # so return zeros instead, as original code did.
        # Return vector of length = number of frequency bins - 1
        # Originally done to remove "DC component", first coefficient
        # of FFT.
        return np.zeros((syllable.power.shape[0],))
    else:
        cepst = _cepstrum_for_mean(syllable.power,syllable.nfft)
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

def mean_spectral_centroid(syllable):
    """
    Mean of spectral centroid across syllable,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean_spectral_centroid : scalar, mean of spectral centroid across syllable
    """
    prob, freqs_mat = _convert_spect_to_probability(syllable.power,syllable.freqBins)[:2]
    spect_centroid = spectral_centroid(prob,freqs_mat)
    return np.mean(spect_centroid)

def mean_delta_spectral_centroid(syllable):
    """
    mean of 5-point delta of spectral centroid
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean_delta_spectral_centroid : scalar
    """

    prob, freqs_mat = _convert_spect_to_probability(syllable.power,syllable.freqBins)[:2]
    spect_centroid = spectral_centroid(prob,freqs_mat)
    delta_spect_centroid = _five_point_delta(spect_centroid)
    return np.mean(delta_spect_centroid)

def spectral_spread(power,freqBins):
    """
    spectral spread, variance of normalized amplitude spectrum
    
    Parameters
    ----------
    power : 2d numpy array, spectrogram where each element is power for that frequency and time bin

    Returns
    -------
    spectral spread
    """
    prob, freqs_mat, num_rows = _convert_spect_to_probability(power,freqBins)[:3]
    spect_centroid = spectral_centroid(prob,freqs_mat)
    variance = _variance(freqs_mat,spect_centroid,num_rows,prob)
    return np.power(variance, 1 / 2)

def mean_spectral_spread(syllable):
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
    return np.mean(spectral_spread(syllable.power,syllable.freqBins))

def mean_delta_spectral_spread(syllable):
    """
    mean of 5-point delta of spectral spread
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_spread : scalar
    """

    return np.mean(_five_point_delta(spectral_spread(syllable.power, syllable.freqBins)))

def spectral_skewness(power, freqBins):
    """
    spectral skewness, measure of asymmetry of normalized amplitude spectrum around mean
    
    Parameters
    ----------
    power

    Returns
    -------
    spectral skewness
    """

    prob, freqs_mat, num_rows = _convert_spect_to_probability(power, freqBins)[:3]
    spect_centroid = spectral_centroid(prob,freqs_mat)
    variance = _variance(freqs_mat,spect_centroid,num_rows,prob)
    skewness = np.sum((np.power(freqs_mat - np.matlib.repmat(spect_centroid, num_rows, 1), 3)) * prob, 0)
    return skewness / np.power(variance, 3 / 2)

def mean_spectral_skewness(syllable):
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

    return np.mean(spectral_skewness(syllable.power, syllable.freqBins))

def mean_delta_spectral_skewness(syllable):
    """
    mean of 5-point delta of spectral skewness
    
    Parameters
    ----------
    power

    Returns
    -------
    mean_delta_spectral_skewness : scalar
    """

    return np.mean(_five_point_delta(spectral_skewness(syllable.power, syllable.freqBins)))

def spectral_kurtosis(power, freqBins):
    """
    spectral kurtosis, measure of flatness of normalized amplitude spectrum
    
    Parameters
    ----------
    power

    Returns
    -------
    spectral kurtosis
    """

    prob, freqs_mat, num_rows = _convert_spect_to_probability(power,freqBins)[:3]
    spect_centroid = spectral_centroid(prob,freqs_mat)
    variance = _variance(freqs_mat,spect_centroid,num_rows,prob)
    kurtosis = np.sum((np.power(freqs_mat - np.matlib.repmat(spect_centroid, num_rows, 1), 4)) * prob, 0)
    return kurtosis / np.power(variance, 2)

def mean_spectral_kurtosis(syllable):
    """
    mean of spectral kurtosis across syllable,
    as computed in Tachibana et al. 2014

    Parameters
    ----------
    syllable

    Returns
    -------
    mean_spectral_kurtosis
    """

    return np.mean(spectral_kurtosis(syllable.power, syllable.freqBins))

def mean_delta_spectral_kurtosis(syllable):
    """
    mean of 5-point delta of spectral kurtosis
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean_delta_spectral_kurtosis
    """

    return np.mean(_five_point_delta(spectral_kurtosis(syllable.power, syllable.freqBins)))

def spectral_flatness(power):
    """
    spectral flatness
    
    Parameters
    ----------
    power

    Returns
    -------
    spectral flatness
    """
    amplitude_spectrum = np.abs(power)
    return np.exp(np.mean(np.log(amplitude_spectrum), 0)) / np.mean(amplitude_spectrum, 0)

def mean_spectral_flatness(syllable):
    """
    mean of spectral flatness across syllable
    
    Parameters
    ----------
    syllable

    Returns
    -------

    """
    return np.mean(spectral_flatness(syllable.power))

def mean_delta_spectral_flatness(syllable):
    """
    mean delta spectral flatness
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean delta spectral flatness
    """
    return np.mean(_five_point_delta(spectral_flatness(syllable.power)))

def spectral_slope(power,freq_bins):
    """
    spectral slope, slope from linear regression of normalized amplitude spectrum
    
    Parameters
    ----------
    power : 2d array,
    freqBins : 1d array, frequency bins as returned by spectrogram

    Returns
    -------
    spectral_slope : 1d array
    """

    amplitude_spectrum = np.abs(power)
    num_rows, num_cols = amplitude_spectrum.shape
    spect_slope = np.zeros((num_cols,))
    mat2 = np.stack((freq_bins, np.ones((num_rows,))),axis=-1)
    for n in range(num_cols):
        beta = np.linalg.solve(np.dot(mat2.T,mat2),
                               np.dot(mat2.T,amplitude_spectrum[:, n]))
        spect_slope[n] = beta[0]
    return spect_slope

def mean_spectral_slope(syllable):
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

    return np.mean(spectral_slope(syllable.power, syllable.freqBins))

def mean_delta_spectral_slope(syllable):
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

    return np.mean(_five_point_delta(spectral_slope(syllable.power, syllable.freqBins)))

def _cepstrum_for_pitch(power,nfft,samp_freq,min_freq,max_freq):
    """
    cepstrum as computed in Tachibana et al. 2014
    for the purposes of calculating pitch and pitch goodness

    Parameters
    ----------
    power
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
    amplitude_spectrum = np.abs(power)

    max_quef = np.round(samp_freq / min_freq).astype(int) * 2
    min_quef = np.round(samp_freq / max_freq).astype(int) * 2

    exs = np.vstack((amplitude_spectrum,
                     np.matlib.repmat(amplitude_spectrum[-1, :], nfft, 1),
                     np.flipud(amplitude_spectrum[1:-1,:])))
    cepstrum = np.real(np.fft.fft(np.log10(exs)))
    max_val = np.max(cepstrum[min_quef:max_quef, :], axis=0)
    max_id = np.argmax(cepstrum[min_quef:max_quef, :], axis=0)
    return max_val, max_id, min_quef

def pitch(syllable,min_freq=500,max_freq=6000):
    """
    pitch, as calculated in Tachibana et al. 2014.
    Peak of the cepstrum.
    
    Parameters
    ----------
    syllable
    min_freq
    max_freq
    
    Returns
    -------
    pitch : scalar
    """

    max_val, max_id, min_quef = _cepstrum_for_pitch(syllable.power,
                                                    syllable.nfft,
                                                    syllable.sampFreq,
                                                    min_freq,
                                                    max_freq)
    return syllable.sampFreq / (max_id + min_quef - 1)

def mean_pitch(syllable):
    """
    mean pitch as measured across syllable

    Parameters
    ----------
    syllable
    
    Returns
    -------
    mean_pitch
    """
    return np.mean(pitch(syllable))

def mean_delta_pitch(syllable):
    """
    mean of 5-point delta of pitch
    
    Parameters
    ----------
    syllable
    
    Returns
    -------
    mean_delta_pitch
    """
    return np.mean(_five_point_delta(pitch(syllable)))

def pitch_goodness(syllable,min_freq=500,max_freq=6000):
    """
    pitch goodness, as calculated in Tachibana et al. 2014
    
    Parameters
    ----------
    syllable
    min_freq
    max_freq
        
    Returns
    -------
    pitch goodness
    """

    return _cepstrum_for_pitch(syllable.power,
                               syllable.nfft,
                               syllable.sampFreq,
                               min_freq,
                               max_freq)[0]


def mean_pitch_goodness(syllable):
    """
    mean of pitch goodness across syllable

    Parameters
    ----------
    syllable
        
    Returns
    -------
    mean_pitch_goodness
    """
    return np.mean(pitch_goodness(syllable))

def mean_delta_pitch_goodness(syllable):
    """

    Parameters
    ----------
    syllable
    
    Returns
    -------
    mean_delta_pitch_goodness
    """
    return np.mean(_five_point_delta(pitch_goodness(syllable)))

def amplitude(syllable):
    """
    amplitude in decibels,
    as computed in Tachibana et al. 2014
    
    Parameters
    ----------
    syllable

    Returns
    -------
    amplitude
    """

    amplitude_spectrum = np.abs(syllable.power)
    return 20 * np.log10(np.sum(amplitude_spectrum,0) / syllable.nfft)

def mean_amplitude(syllable):
    """
    mean of amplitude across syllable
    
    Parameters
    ----------
    syllable

    Returns
    -------
    mean_amplitude : scalar
    """

    return np.mean(amplitude(syllable))

def mean_delta_amplitude(syllable):
    """
    mean of 5-point delta of amplitude
    
    Parameters
    ----------
    syllable
    
    Returns
    -------
    mean_delta_amplitude
    """

    return np.mean(_five_point_delta(amplitude(syllable)))

def zero_crossings(syllable):
    """
    zero crossings, in units of Hertz
    
    Parameters
    ----------
    syllable

    Returns
    -------
    zero_crossings : scalar
    """

    #to find zero crossings:
    #   convert raw signal to sign, either +1 or -1
    #   then take difference at each point: [signal[1]-signal[0],signal[2]-signal[1],...]
    #   then find all the points where the difference is not zero
    #       because sign changed from -1 to 1 or from 1 to -1
    #   Not sure why it was divided in two but I notice by looking at examples that
    #   they are often right next to each other.
    #   One zero crossing should only result in one non-zero index though
    zero_crosses = np.where(np.diff(np.sign(syllable.sylAudio)) != 0)[0].shape[-1] / 2
    dur = duration(syllable)
    return zero_crosses / dur