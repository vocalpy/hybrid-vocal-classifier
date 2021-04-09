import warnings

import numpy as np
import scipy.signal
from matplotlib.mlab import specgram

import hvc.evfuncs
from .parse.ref_spect_params import refs_dict


class WindowError(Exception):
    pass


class SegmentParametersMismatchError(Exception):
    pass


def butter_bandpass(freq_cutoffs, samp_freq, order=8):
    """returns filter coefficients for Butterworth bandpass filter

    Parameters
    ----------
    freq_cutoffs: list
        low and high frequencies of pass band, e.g. [500, 10000]
    samp_freq: int
        sampling frequency
    order: int
        of filter, default is 8

    Returns
    -------
    b, a: ndarray, ndarray

    adopted from the SciPy cookbook:
    http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """

    nyquist = 0.5 * samp_freq
    freq_cutoffs = np.asarray(freq_cutoffs) / nyquist
    b, a = scipy.signal.butter(order, freq_cutoffs, btype='bandpass')
    return b, a


def butter_bandpass_filter(data, samp_freq, freq_cutoffs, order=8):
    """applies Butterworth bandpass filter to data

    Parameters
    ----------
    data: ndarray
        1-d array of raw audio data
    samp_freq: int
        sampling frequency
    freq_cutoffs: list
        low and high frequencies of pass band, e.g. [500, 10000]
    order: int
        of filter, default is 8

    Returns
    -------
    data: ndarray
        data after filtering

    adopted from the SciPy cookbook:
    http://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """

    b, a = butter_bandpass(freq_cutoffs, samp_freq, order=order)
    return scipy.signal.lfilter(b, a, data)


class Spectrogram:
    """class for making spectrograms.
    Abstracts out function calls so user just has to put spectrogram parameters
    in YAML config file.
    """

    def __init__(self,
                 nperseg=None,
                 noverlap=None,
                 freq_cutoffs=(500, 10000),
                 window=None,
                 filter_func=None,
                 spect_func=None,
                 log_transform_spect=True,
                 thresh=-4.0,
                 remove_dc=True):
        """Spectrogram.__init__ function

        Parameters
        ----------
        nperseg : int
            numper of samples per segment for FFT, e.g. 512
        noverlap : int
            number of overlapping samples in each segment

        nperseg and noverlap are required for __init__

        Other Parameters
        ----------------
        freq_cutoffs : two-element list of integers
            limits of frequency band to keep, e.g. [1000,8000]
            Spectrogram.make keeps the band:
                freq_cutoffs[0] >= spectrogram > freq_cutoffs[1]
            Default is [500, 10000].
        window : str
            window to apply to segments
            valid strings are 'Hann', 'dpss', None
            Hann -- Uses np.Hanning with parameter M (window width) set to value of nperseg
            dpss -- Discrete prolate spheroidal sequence AKA Slepian.
                Uses scipy.signal.slepian with M parameter equal to nperseg and
                width parameter equal to 4/nperseg, as in [2]_.
            Default is None.
        filter_func : str
            filter to apply to raw audio. valid strings are 'diff' or None
            'diff' -- differential filter, literally np.diff applied to signal as in [1]_.
            Default is None.
            Note this is different from filters applied to isolate frequency band.
        spect_func : str
            which function to use for spectrogram.
            valid strings are 'scipy' or 'mpl'.
            'scipy' uses scipy.signal.spectrogram,
            'mpl' uses matplotlib.matlab.specgram.
            Default is 'scipy'.
        log_transform_spect : bool
            if True, applies np.log10 to spectrogram to increase range.
            Default is True.
        thresh : float
            threshold for spectrogram.
            All values below thresh are set to thresh;
            increases contrast when visualizing spectrogram with a colormap.
            Default is -4 (assumes log_transform_spect==True)
        remove_dc : bool
            if True, remove the zero-frequency component of the spectrogram,
            i.e. the DC offset, which in a sound recording should be zero.
            Default is True. Calculation of some features (e.g. cepstrum)
            requires the DC component however.

        References
        ----------
        .. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
        automatic classification of birdsong elements using a linear support vector
         machine." PloS one 9.3 (2014): e92584.

        .. [2] Koumura, Takuya, and Kazuo Okanoya. "Automatic recognition of element
        classes and boundaries in the birdsong with variable sequences."
        PloS one 11.7 (2016): e0159188.
        """

        if nperseg is None:
            raise ValueError('nperseg requires a value for Spectrogram.__init__')
        if noverlap is None:
            raise ValueError('noverlap requires a value for Spectrogram.__init__')
        if spect_func is None:
            # switch to default
            # can't have in args list because need to check above for
            # conflict with default spectrogram functions for each ref
            spect_func = 'scipy'
        if type(nperseg) != int:
            raise TypeError('type of nperseg must be int, but is {}'.
                             format(type(nperseg)))
        else:
            self.nperseg = nperseg

        if type(noverlap) != int:
            raise TypeError('type of noverlap must be int, but is {}'.
                             format(type(noverlap)))
        else:
            self.noverlap = noverlap

        if window is None:
            self.window = None
        else:
            if type(window) != str:
                raise TypeError('type of window must be str, but is {}'.
                                 format(type(window)))
            else:
                if window not in ['Hann', 'dpss']:
                    raise ValueError('{} is not a valid specification for window'.
                                     format(window))
                else:
                    if window == 'Hann':
                        self.window = np.hanning(self.nperseg)
                    elif window == 'dpss':
                        self.window = scipy.signal.windows.dpss(self.nperseg, 4 / self.nperseg)

        if freq_cutoffs is None:
            self.freqCutoffs = None
        else:
            if freq_cutoffs == (500, 10000):
                # if default, convert to list
                # don't want to have a mutable list as the default
                # because mutable defaults can give rise to nasty bugs
                freq_cutoffs = list(freq_cutoffs)
    
            if type(freq_cutoffs) != list:
                raise TypeError('type of freq_cutoffs must be list, but is {}'.
                                 format(type(freq_cutoffs)))
            elif len(freq_cutoffs) != 2:
                raise ValueError('freq_cutoffs list should have length 2, but length is {}'.
                                 format(len(freq_cutoffs)))
            elif not all([type(val) == int for val in freq_cutoffs]):
                raise ValueError('all values in freq_cutoffs list must be ints')
            else:
                self.freqCutoffs = freq_cutoffs

        if freq_cutoffs is not None and filter_func is None:
            self.filterFunc = 'butter_bandpass'  # default

        if filter_func is not None and type(filter_func) != str:
            raise TypeError('type of filter_func must be str, but is {}'.
                             format(type(filter_func)))
        elif filter_func not in ['diff','bandpass_filtfilt','butter_bandpass',None]:
            raise ValueError('string \'{}\' is not valid for filter_func. '
                             'Valid values are: \'diff\' or None.'.
                             format(filter_func))
        else:
            self.filterFunc = filter_func

        if type(spect_func) != str:
            raise TypeError('type of spect_func must be str, but is {}'.
                             format(type(spect_func)))
        elif spect_func not in ['scipy', 'mpl']:
            raise ValueError('string \'{}\' is not valid for filter_func. '
                             'Valid values are: \'scipy\' or \'mpl\'.'.
                             format(spect_func))
        else:
            self.spectFunc = spect_func

        if type(log_transform_spect) is not bool:
            raise ValueError('Value for log_transform_spect is {}, but'
                             ' it must be bool.'
                             .format(type(log_transform_spect)))
        else:
            self.logTransformSpect = log_transform_spect

        if type(thresh) is not float and thresh is not None:
            try:
                thresh = float(thresh)
                self.tresh = thresh
            except:
                raise ValueError('Value for thresh is {}, but'
                                 ' it must be float.'
                                 .format(type(thresh)))
        else:
            self.thresh = thresh
        
        if type(remove_dc) is not bool:
            raise TypeError('Value for remove_dc should be boolean, not {}'
                            .format(type(remove_dc)))
        else:
            self.remove_dc = remove_dc

    def make(self,
             raw_audio,
             samp_freq):
        """makes spectrogram using assigned properties

        Parameters
        ----------
        raw_audio : 1-d numpy array
            raw audio waveform
        samp_freq : integer scalar
            sampling frequency in Hz

        Returns
        -------
        spect : 2-d numpy array
        freq_bins : 1-d numpy array
        time_bins : 1-d numpy array
        """

        if self.filterFunc == 'diff':
            raw_audio = np.diff(raw_audio)  # differential filter_func, as applied in Tachibana Okanoya 2014
        elif self.filterFunc == 'bandpass_filtfilt':
            raw_audio = hvc.evfuncs.bandpass_filtfilt(raw_audio,
                                                      samp_freq,
                                                      self.freqCutoffs)
        elif self.filterFunc == 'butter_bandpass':
            raw_audio = butter_bandpass_filter(raw_audio,
                                               samp_freq,
                                               self.freqCutoffs)

        try:  # try to make spectrogram
            if self.spectFunc == 'scipy':
                if self.window is not None:
                        freq_bins, time_bins, spect = scipy.signal.spectrogram(raw_audio,
                                                                               samp_freq,
                                                                               window=self.window,
                                                                               nperseg=self.nperseg,
                                                                               noverlap=self.noverlap)
                else:
                    freq_bins, time_bins, spect = scipy.signal.spectrogram(raw_audio,
                                                                           samp_freq,
                                                                           nperseg=self.nperseg,
                                                                           noverlap=self.noverlap)

            elif self.spectFunc == 'mpl':
                # note that the matlab specgram function returns the STFT by default
                # whereas the default for the matplotlib.mlab version of specgram
                # returns the PSD. So to get the behavior of matplotlib.mlab.specgram
                # to match, mode must be set to 'complex'

                # I think I determined empirically at one point (by staring at single
                # cases) that mlab.specgram gave me values that were closer to Matlab's
                # specgram function than scipy.signal.spectrogram
                # Matlab's specgram is what Tachibana used in his original feature
                # extraction code. So I'm maintaining the option to use it here.

                # 'mpl' is set to return complex frequency spectrum,
                # not power spectral density,
                # because some tachibana features (based on CUIDADO feature set)
                # need to use the freq. spectrum before taking np.abs or np.log10
                if self.window is not None:
                    spect, freq_bins, time_bins = specgram(raw_audio,
                                                           NFFT=self.nperseg,
                                                           Fs=samp_freq,
                                                           window=self.window,
                                                           noverlap=self.noverlap,
                                                           mode='complex')
                else:
                    spect, freq_bins, time_bins = specgram(raw_audio,
                                                           NFFT=self.nperseg,
                                                           Fs=samp_freq,
                                                           noverlap=self.noverlap,
                                                           mode='complex')
        except ValueError as err:  # if `try` to make spectrogram raised error
            if str(err) == 'window is longer than input signal':
                raise WindowError()
            else:  # unrecognized error
                raise

        if self.remove_dc:
            # remove zero-frequency component
            freq_bins = freq_bins[1:]
            spect = spect[1:,:]
        
        # we take the absolute magnitude
        # because we almost always want just that for our purposes
        spect = np.abs(spect)

        if self.logTransformSpect:
            spect = np.log10(spect)  # log transform to increase range

        if self.thresh is not None:
            spect[spect < self.thresh] = self.thresh

        # below, I set freq_bins to >= freq_cutoffs
        # so that Koumura default of [1000,8000] returns 112 freq. bins
        if self.freqCutoffs is not None:
            f_inds = np.nonzero((freq_bins >= self.freqCutoffs[0]) &
                                (freq_bins <= self.freqCutoffs[1]))[0]  # returns tuple
            freq_bins = freq_bins[f_inds]
            spect = spect[f_inds, :]

        return spect, freq_bins, time_bins


class Segmenter:
    def __init__(self,
                 threshold=5000,
                 min_syl_dur=0.02,
                 min_silent_dur=.002):
        """__init__ for Segmenter

        Parameters
        ----------
        segment_params : dict
        with the following keys
            threshold : int
                value above which amplitude is considered part of a segment. default is 5000.
            min_syl_dur : float
                minimum duration of a segment. default is 0.02, i.e. 20 ms.
            min_silent_dur : float
                minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.
        """
        self.threshold = threshold
        self.min_syl_dur = min_syl_dur
        self.min_silent_dur = min_silent_dur

    def compute_amp(spect):
        """
        compute amplitude of spectrogram
        Assumes the values for frequencies are power spectral density (PSD).
        Sums PSD for each time bin, i.e. in each column.
        Inputs:
            spect -- output from spect_from_song
        Returns:
            amp -- amplitude
        """

        return np.sum(spect, axis=0)

    def segment(self,
                array_to_segment,
                method,
                time_bins=None,
                samp_freq=None):
        """Divides songs into segments based on threshold crossings of amplitude.
        Returns onsets and offsets of segments, corresponding to syllables in a song.

        Parameters
        ----------
        array_to_segment : ndarray
            Either amplitude of power spectral density, returned by compute_amp,
            or smoothed amplitude of filtered audio, returned by evfuncs.smooth_data
        time_bins : 1-d numpy array
            time in s, must be same length as log amp. Returned by Spectrogram.make.
        samp_freq : int
            sampling frequency
        method : str
            {'evsonganaly','psd'}
            Method to use.
            evsonganaly -- gives same result as segmentation by evsonganaly.m
            (a Matlab GUI for labeling song developed in the Brainard lab)
            Uses smoothed filtered amplitude of audio, as returned by hvc.evfuncs.smooth_data
            psd -- uses power spectral density of spectrogram, as returned by _compute_amp 

        Returns
        -------
        segment_dict : dict
            with following key, value pairs
                onsets_Hz : ndarray
                    onset times given in sample number (Hz)
                offsets_Hz : ndarray
                    offset times given in sample number (Hz)
                onsets_s : ndarray
                    onset times given in sample number (Hz)
                offsets_s : 1-d numpy array
                    arrays of onsets and offsets of segments.

        So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
        To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
        """
        if time_bins is None and samp_freq is None:
            raise ValueError('Values needed for either time_bins or samp_freq parameters '
                             'needed to segment song.')
        if time_bins is not None and samp_freq is not None:
            raise ValueError('Can only use one of time_bins or samp_freq to segment song, '
                             'but values were passed for both parameters')

        if method == 'evsonganaly':
            if time_bins is not None:
                raise ValueError("cannot use time_bins with method 'evsonganaly'")
            if samp_freq is None:
                raise ValueError("must provide samp_freq with method 'evsonganaly'")
            if array_to_segment.ndim != 1:
                raise ValueError("If method is 'evsonganaly', then array_to_segment "
                                 "must be one-dimensional (i.e., raw audio signal)")

        if method == 'psd':
            if samp_freq is not None:
                raise ValueError("cannot use samp_freq with method 'psd'")
            if time_bins is None:
                raise ValueError("must provide time_bins with method 'psd'")
            if array_to_segment.ndim != 2:
                raise ValueError("If method is 'psd', then array_to_segment "
                                 "must be two-dimensional (i.e., a spectrogram)")
            if array_to_segment.shape[-1] != time_bins.shape[-1]:
                raise ValueError('if using time_bins, '
                                 'array_to_segment and time_bins must have same length')

        if method=='evsonganaly':
            amp = hvc.evfuncs.smooth_data(array_to_segment,
                                          samp_freq,
                                          refs_dict['evsonganaly']['freq_cutoffs'])
        elif method=='psd':
            amp = self.compute_amp(array_to_segment)

        above_th = amp > self.threshold
        h = [1, -1]
        # convolving with h causes:
        # +1 whenever above_th changes from 0 to 1
        # and -1 whenever above_th changes from 1 to 0
        above_th_convoluted = np.convolve(h, above_th)

        if time_bins is not None:
            # if amp was taken from time_bins using compute_amp
            # note that np.where calls np.nonzero which returns a tuple
            # but numpy "knows" to use this tuple to index into time_bins
            onsets_s = time_bins[np.where(above_th_convoluted > 0)]
            offsets_s = time_bins[np.where(above_th_convoluted < 0)]
        elif samp_freq is not None:
            # if amp was taken from smoothed audio using smooth_data
            # here, need to get the array out of the tuple returned by np.where
            # **also note we avoid converting from samples to s
            # until *after* we find segments** 
            onsets_Hz = np.where(above_th_convoluted > 0)[0]
            offsets_Hz = np.where(above_th_convoluted < 0)[0]
            onsets_s = onsets_Hz / samp_freq
            offsets_s = offsets_Hz / samp_freq

        if onsets_s.shape[0] < 1 or offsets_s.shape[0] < 1:
            return None, None  # because no onsets or offsets in this file

        # get rid of silent intervals that are shorter than min_silent_dur
        silent_gap_durs = onsets_s[1:] - offsets_s[:-1]  # duration of silent gaps
        keep_these = np.nonzero(silent_gap_durs > self.min_silent_dur)
        onsets_s = np.concatenate(
            (onsets_s[0, np.newaxis], onsets_s[1:][keep_these]))
        offsets_s = np.concatenate(
            (offsets_s[:-1][keep_these], offsets_s[-1, np.newaxis]))
        if 'onsets_Hz' in locals():
            onsets_Hz = np.concatenate(
                (onsets_Hz[0, np.newaxis], onsets_Hz[1:][keep_these]))
            offsets_Hz = np.concatenate(
                (offsets_Hz[:-1][keep_these], offsets_Hz[-1, np.newaxis]))

        # eliminate syllables with duration shorter than min_syl_dur
        syl_durs = offsets_s - onsets_s
        keep_these = np.nonzero(syl_durs > self.min_syl_dur)
        onsets_s = onsets_s[keep_these]
        offsets_s = offsets_s[keep_these]
        if 'onsets_Hz' in locals():
            onsets_Hz = onsets_Hz[keep_these]
            offsets_Hz = offsets_Hz[keep_these]

        segment_dict = {'onsets_s': onsets_s,
                        'offsets_s': offsets_s}
        if 'onsets_Hz' in locals():
            segment_dict['onsets_Hz'] = onsets_Hz
            segment_dict['offsets_Hz'] = offsets_Hz

        return segment_dict


class Syllable:
    """
    syllable object, returned by make_syl_spect.
    Properties
    ----------
    syl_audio : 1-d numpy array
        raw waveform from audio file
    sampfreq : integer
        sampling frequency in Hz as determined by scipy.io.wavfile function
    spect : 2-d m by n numpy array
        spectrogram as computed by Spectrogram.make(). Each of the m rows is a frequency bin,
        and each of the n columns is a time bin. Value in each bin is power at that frequency and time.
    nfft : integer
        number of samples used for each FFT
    overlap : integer
        number of samples that each consecutive FFT window overlapped
    time_bins : 1d vector
        values are times represented by each bin in s
    freq_bins : 1d vector
        values are power spectral density in each frequency bin
    index: int
        index of this syllable in song.syls.labels
    label: int
        label of this syllable from song.syls.labels
    """
    def __init__(self,
                 syl_audio,
                 samp_freq,
                 spect,
                 nfft,
                 overlap,
                 freq_cutoffs,
                 freq_bins,
                 time_bins,
                 index,
                 label):
        self.sylAudio = syl_audio
        self.sampFreq = samp_freq
        self.spect = spect
        self.nfft = nfft
        self.overlap = overlap
        self.freqCutoffs = freq_cutoffs
        self.freqBins = freq_bins
        self.timeBins = time_bins
        self.index = index
        self.label = label


def make_syls(raw_audio,
              samp_freq,
              spect_maker,
              labels,
              onsets_Hz,
              offsets_Hz,
              labels_to_use='all',
              syl_spect_width=-1,
              return_as_stack=False):
    """Make spectrograms from syllables.
    This method isolates making spectrograms from selecting syllables
    to use so that spectrograms can be loaded 'lazily', e.g., if only
    duration features are being extracted that don't require spectrograms.

    Parameters
    ----------
    raw_audio : ndarray
    samp_freq : int
    labels : str, list, or ndarray
    onsetz_Hz : ndarray
    offsets_Hz : ndarray
    labels_to_use : str or nmmpy ndarray
        if ndarray, must be of type bool and same length as labels, and
        will be used to index into labels
    syl_spect_width : float
        Optional parameter to set constant duration for each spectrogram of a
        syllable, in seconds. E.g., 0.05 for an average 50 millisecond syllable. 
        Used for creating inputs to neural network where each input
        must be of a fixed size.
        Default value is -1; in this case, the width of the spectrogram will
        be the duration of the syllable as determined by the segmentation
        algorithm, i.e. the onset and offset that are stored in an annotation file.
        If a different value is given, then the duration of each spectrogram
        will be that value. Note that if any individual syllable has a duration
        greater than syl_spect_duration, the function raises an error.
    """
    if syl_spect_width > 0:
        if syl_spect_width > 1:
            warnings.warn('syl_spect_width set greater than 1; note that '
                          'this parameter is in units of seconds, so using '
                          'a value greater than one will make it hard to '
                          'center the syllable/segment of interest within'
                          'the spectrogram, and additionally consume a lot '
                          'of memory.')
        syl_spect_width_Hz = int(syl_spect_width * samp_freq)
        if syl_spect_width_Hz > raw_audio.shape[-1]:
            raise ValueError('syl_spect_width, converted to samples, '
                             'is longer than song file.')

    if type(labels) not in [str, list, np.ndarray]:
        raise TypeError('labels must be of type str, list, or numpy ndarray, '
                        'not {}'.type(labels))

    if type(labels) is str:
        labels = list(labels)

    if type(labels) is list:
        labels = np.asarray(labels)

    if type(labels_to_use) is str:
        if labels_to_use == 'all':
            use_these_labels_bool = np.ones((labels.shape)).astype(bool)
        else:
            use_these_labels_bool = np.asarray([label in labels_to_use
                                                for label in labels])
    elif type(labels_to_use) is np.ndarray and labels_to_use.dtype == bool:
        if labels_to_use.ndim > 2:
            raise ValueError('if labels_to_use is array, should not have '
                             'more than two dimensions')
        else:
            labels_to_use = np.squeeze(labels_to_use)
            if labels_to_use.shape[-1] != len(labels):
                raise ValueError('if labels_to_use is an array, must have '
                                 'same length as labels')
    elif type(labels_to_use) is np.ndarray and labels_to_use.dtype != bool:
        raise TypeError('if labels_to_use is an array, must be of type bool')
    else:
        raise TypeError('labels_to_use should be a string or a boolean numpy '
                        'array, not type {}'.format(type(labels_to_use)))

    all_syls = []

    for ind, (label, onset, offset) in enumerate(zip(labels, onsets_Hz, offsets_Hz)):
        if 'syl_spect_width_Hz' in locals():
            syl_duration_in_samples = offset - onset
            if syl_duration_in_samples > syl_spect_width_Hz:
                raise ValueError('syllable duration of syllable {} with label {} '
                                 'width specified for all syllable spectrograms.'
                                 .format(ind, label))

        if 'syl_spect_width_Hz' in locals():
            width_diff = syl_spect_width_Hz - syl_duration_in_samples
            # take half of difference between syllable duration and spect width
            # so one half of 'empty' area will be on one side of spect
            # and the other half will be on other side
            # i.e., center the spectrogram
            left_width = int(round(width_diff / 2))
            right_width = width_diff - left_width
            if left_width > onset:  # if duration before onset is less than left_width
                # (could happen with first onset)
                syl_audio = raw_audio[0:syl_spect_width_Hz]
            elif offset + right_width > raw_audio.shape[-1]:
                # if right width greater than length of file
                syl_audio = raw_audio[-syl_spect_width_Hz:]
            else:
                syl_audio = raw_audio[onset - left_width:offset + right_width]
        else:
            syl_audio = raw_audio[onset:offset]

        try:
            spect, freq_bins, time_bins = spect_maker.make(syl_audio,
                                                           samp_freq)
        except WindowError as err:
            warnings.warn('Segment {0} with label {1} '
                          'not long enough for window function'
                          ' set with current spect_params.\n'
                          'spect will be set to nan.'
                          .format(ind, label))
            spect, freq_bins, time_bins = (np.nan,
                                           np.nan,
                                           np.nan)

        curr_syl = Syllable(syl_audio=syl_audio,
                            samp_freq=samp_freq,
                            spect=spect,
                            nfft=spect_maker.nperseg,
                            overlap=spect_maker.noverlap,
                            freq_cutoffs=spect_maker.freqCutoffs,
                            freq_bins=freq_bins,
                            time_bins=time_bins,
                            index=ind,
                            label=label)

        all_syls.append(curr_syl)

    if return_as_stack:
        # stack with dimensions (samples, height, width)
        return np.stack([syl.spect for syl in all_syls], axis=0)
    else:
        return all_syls
