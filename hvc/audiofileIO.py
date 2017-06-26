import warnings

import numpy as np
from scipy.io import wavfile
import scipy.signal
from scipy.signal import slepian # AKA DPSS, window used for FFT
from matplotlib.mlab import specgram

from . import evfuncs, koumura

class Spectrogram:
    """class for making spectrograms.
    Abstracts out function calls so user just has to put spectrogram parameters
    in YAML config file.
    """

    def __init__(self,
                 nperseg=None,
                 noverlap=None,
                 freq_cutoffs=None,
                 window=None,
                 filter_func=None,
                 spec_func='scipy',
                 ref=None):
        """Spectrogram.__init__ function
        
        Parameters
        ----------
        nperseg : integer
            numper of samples per segment for FFT, e.g. 512
        noverlap : integer
            number of overlapping samples in each segment
        freq_cutoffs : two-element list of integers
            limits of frequency band to keep, e.g. [1000,8000]
        window : string
            window to apply to segments
            valid strings are 'Hann', 'dpss', None
            Hann -- Uses np.Hanning with paramater M (window width) set to value of nperseg
            dpss -- Discrete prolate spheroidal sequences AKA Slepian.
                Uses scipy.signal.slepian with M parameter equal to nperseg and
                width parameter equal to 4/nperseg, as in [2]_.
        filter_func : string
            filter to apply to raw audio. valid strings are 'diff' or None
            'diff' -- differential filter, literally np.diff applied to signal as in [1]_.
            None -- no filter, this is the default
        spec_func : string
            which function to use for spectrogram.
            valid strings are 'scipy' or 'mpl'.
            'scipy' uses scipy.signal.spectrogram,
            'mpl' uses matplotlib.matlab.specgram.
            Default is 'scipy'.
        ref : string
            {'tachibana','koumura'}
            Use spectrogram parameters from a reference.
            'tachibana' uses spectrogram parameters from [1]_,
            'koumura' uses spectrogram parameters from [2]_.

        References
        ----------
        .. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
        automatic classification of birdsong elements using a linear support vector
         machine." PloS one 9.3 (2014): e92584.

        .. [2] Koumura, Takuya, and Kazuo Okanoya. "Automatic recognition of element
        classes and boundaries in the birdsong with variable sequences."
        PloS one 11.7 (2016): e0159188.
        """

        # check for 'reference' parameter first since it takes precedence
        if ref is not None:
            if ref not in ('tachibana','koumura'):
                raise ValueError('{} is not a valid value for reference argument'.
                                 format(ref))
            # throw error if called with 'ref' and with other params
            if any(param is not None
                   for param in [nperseg,
                               noverlap,
                               freq_cutoffs,
                               filter_func]):
                warnings.warn('Spectrogram class received ref '
                              'parameter but also received other parameters, '
                              'will over-write those with defaults for reference.')
            else:
                if ref == 'tachibana':
                    self.nperseg = 256
                    self.noverlap = 192
                    self.window = 'Hann'
                    self.freq_cutoffs = [500,6000]
                    self.filter_func = 'diff'
                    self.spec_func = 'mpl'
                elif ref == 'koumura':
                    self.nperseg = 512
                    self.noverlap = 480
                    self.window = 'dpss'
                    self.freq_cutoffs = [1000,8000]
                    self.filter_func = None
                    self.spec_func = 'scipy'

        else:
            if any(param is None
                   for param in [nperseg,
                                 noverlap,
                                 freq_cutoffs]):
                raise ValueError('not all parameters set for Spectrogram init')
            else:
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

                if window is not None and type(window) != str:
                    raise TypeError('type of window must be str, but is {}'.
                                     format(type(window)))
                else:
                    if window not in ['Hann','dpss',None]:
                        raise ValueError('{} is not a valid specification for window'.
                                         format(window))
                    else:
                        if window == 'Hann':
                            self.window = np.hanning(self.nperseg)
                        elif window == 'dpss':
                            self.window = slepian(self.nperseg, 4 / self.nperseg)
                        elif window == None:
                            self.window = None

                if type(freq_cutoffs) != list:
                    raise TypeError('type of freq_cutoffs must be list, but is {}'.
                                     format(type(freq_cutoffs)))
                elif len(freq_cutoffs) != 2:
                    raise ValueError('freq_cutoffs list should have length 2, but length is {}'.
                                     format(len(freq_cutoffs)))
                elif not all([type(val) == int for val in freq_cutoffs]):
                    raise ValueError('all values in freq_cutoffs list must be ints')
                else:
                    self.freq_cutoffs = freq_cutoffs

                if filter_func is not None and type(filter_func) != str:
                    raise TypeError('type of filter_func must be str, but is {}'.
                                     format(type(filter_func)))
                elif filter_func not in ['diff',None]:
                    raise ValueError('string \'{}\' is not valid for filter_func. '
                                     'Valid values are: \'diff\' or None.'.
                                     format(filter_func))
                else:
                    self.filter_func = filter_func

                if type(spec_func) != str:
                    raise TypeError('type of spec_func must be str, but is {}'.
                                     format(type(spec_func)))
                elif spec_func not in ['scipy','mpl']:
                    raise ValueError('string \'{}\' is not valid for filter_func. '
                                     'Valid values are: \'scipy\' or \'mpl\'.'.
                                     format(filter_func))
                else:
                    self.spec_func = spec_func

    def make(self,rawsong,samp_freq):
        """makes spectrogram using assigned properties
        
        Parameters
        ----------
        rawsong : 1-d numpy array
            raw audio waveform
        samp_freq : integer scalar
            sampling frequency in Hz

        Returns
        -------
        spect : 2-d numpy array
        freq_bins : 1-d numpy array
        time_bins : 1-d numpy array
        """

        if self.filter_func == 'diff':
            rawsong = np.diff(rawsong)  # differential filter_func, as applied in Tachibana Okanoya 2014

        if self.spec_func == 'scipy':
            if self.window:
                freq_bins, time_bins, spect = scipy.signal.spectrogram(rawsong,
                                                                       samp_freq,
                                                                       window=self.window,
                                                                       nperseg=self.nperseg,
                                                                       noverlap=self.noverlap)
            else:
                freq_bins, time_bins, spect = scipy.signal.spectrogram(rawsong,
                                                                       samp_freq,
                                                                       nperseg=self.nperseg,
                                                                       noverlap=self.noverlap)

        elif self.spec_func == 'mpl':
            # note that the matlab specgram function returns the STFT by default
            # whereas the default for the matplotlib.mlab version of specgram
            # returns the PSD. So to get the behavior of matplotlib.mlab.specgram
            # to match, mode must be set to 'complex'
            if self.window:
                spect, freq_bins, time_bins = specgram(rawsong,
                                                       NFFT=self.nperseg,
                                                       Fs=samp_freq,
                                                       window=self.window,
                                                       noverlap=self.noverlap,
                                                       mode='complex')
            else:
                spect, freq_bins, time_bins = specgram(rawsong,
                                                       NFFT=self.nperseg,
                                                       Fs=samp_freq,
                                                       noverlap=self.noverlap,
                                                       mode='complex')

        #below, I set freq_bins to >= freq_cutoffs
        #so that Koumura default of [1000,8000] returns 112 freq. bins
        f_inds = np.nonzero((freq_bins >= self.freq_cutoffs[0]) &
                            (freq_bins < self.freq_cutoffs[1]))[0] #returns tuple
        freq_bins = freq_bins[f_inds]
        spect = spect[f_inds,:]
        spect = np.log10(spect) # log transform to increase range

        #flip spect and freq_bins so lowest frequency is at 0 on y axis when plotted
        spect = np.flipud(spect)
        freq_bins = np.flipud(freq_bins)
        return spect, freq_bins, time_bins

class song_spect:
    """spectrogram object, returned by make_song_spect.
    Properties:
        spect -- 2-d m by n numpy array, spectrogram as computed by make_song_spect.
                 Each of the m rows is a frequency bin, and each of the n columns is a time bin.
        time_bins -- 1d vector, values are times represented by each bin in s
        freq_bins -- 1d vector, values are power spectral density in each frequency bin
        sampfreq -- sampling frequency in Hz as determined by scipy.io.wavfile function
    """
    def __init__(self,spect,freq_bins,time_bins,sampfreq):
        self.spect = spect
        self.freqBins = freq_bins
        self.timeBins = time_bins
        self.sampFreq = sampfreq

def make_song_spect(waveform,sampfreq,size=512,step=32,freq_cutoffs=[1000,8000]):
    """
    Computes spectogram of raw song waveform using FFT.
    Defaults to FFT parameters from Koumura Okanoya 2016.
    **Note that spectrogram is log transformed (base 10), and that
    both spectrogram and freq_bins are "flipped" (reflected across horizontal
    axis) so that when plotted the lower frequencies of the spectrogram are 
    at 0 on the y axis.

    Parameters
    ----------
    waveform : 1-d numpy array
        raw audio waveform as recorded in file
    sampfreq : integer
        sampling frequency in Hz, e.g. 32000
    size: integer
        size of FFT window, default is 512 samples
    step: integer
        number of samples between the start of each window, default is 32
        i.e. if size == step then there will be no overlap of windows
    freq_range: 2-element list
        range of frequencies to return. Frequencies
        less than the first element or greater than the second are discarded.

    Returns
    -------
    spect -- spectrogram, log transformed
    time_bins -- vector assigning time values to each column in spect
        e.g. [0,8,16] <-- 8 ms time bins
    freq_bins -- vector assigning frequency values to each row in spect
        e.g. [0,100,200] <-- 100 Hz frequency bins
    """
    win_dpss = slepian(size, 4/size)
    fft_overlap = size - step
    freq_bins, time_bins, spect = scipy.signal.spectrogram(waveform,
                                                           sampfreq,
                                                           window=win_dpss,
                                                           nperseg=win_dpss.shape[0],
                                                           noverlap=fft_overlap)
    #below, I set freq_bins to >= freq_cutoffs 
    #so that Koumura default of [1000,8000] returns 112 freq. bins
    f_inds = np.nonzero((freq_bins >= freq_cutoffs[0]) & 
                        (freq_bins < freq_cutoffs[1]))[0] #returns tuple
    freq_bins = freq_bins[f_inds]
    spect = spect[f_inds,:]
    spect = np.log10(spect) # log transform to increase range

    #flip spect and freq_bins so lowest frequency is at 0 on y axis when plotted
    spect = np.flipud(spect)
    freq_bins = np.flipud(freq_bins)
    spect_obj = song_spect(spect,freq_bins,time_bins,sampfreq)
    return spect_obj
    
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

    return np.sum(spect,axis=0)

def segment_song(amp,
                 time_bins,
                 segment_params=None):
    """Divides songs into segments based on threshold crossings of amplitude.
    Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
    Parameters
    ----------
    amp : 1-d numpy array
        amplitude of power spectral density. Returned by compute_amp.
    time_bins : 1-d numpy array
        time in s, must be same length as log amp. Returned by make_song_spect.
    segment_params : dict
        with the following keys
            threshold : int
                value above which amplitude is considered part of a segment. default is 5000.
            min_syl_dur : float
                minimum duration of a segment. default is 0.02, i.e. 20 ms.
            min_silent_dur : float
                minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.

    Returns
    -------
    onsets : 1-d numpy array
    offsets : 1-d numpy array
        arrays of onsets and offsets of segments.
        
    So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
    To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """

    if segment_params is None:
        segment_params = {'threshold' : 5000,
                          'min_syl_dur' : 0.2,
                          'min_silent_dur' : 0.02}
    above_th = amp > segment_params['threshold']
    h = [1, -1] 
    above_th_convoluted = np.convolve(h,above_th) # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    onsets = time_bins[np.nonzero(above_th_convoluted > 0)]
    # and -1 whenever above_th changes from 1 to 0
    offsets = time_bins[np.nonzero(above_th_convoluted < 0)]
    
    #get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1] # duration of silent gaps
    keep_these = np.nonzero(silent_gap_durs > segment_params['min_silent_dur'])
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]
    
    #eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    keep_these = np.nonzero(syl_durs > segment_params['min_syl_dur'])
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]

    return onsets, offsets

class syllable:
    """
    syllable object, returned by make_syl_spect.
    Properties
    ----------
    syl_audio : 1-d numpy array
        raw waveform from audio file
    sampfreq : integer
        sampling frequency in Hz as determined by scipy.io.wavfile function
    power : 2-d m by n numpy array
        spectrogram as computed by make_song_spect. Each of the m rows is a frequency bin, 
        and each of the n columns is a time bin. Value in each bin is power at that frequency and time.
    nfft : integer
        number of samples used for each FFT
    overlap : integer
        number of samples that each consecutive FFT window overlapped
    time_bins : 1d vector
        values are times represented by each bin in s
    freq_bins : 1d vector
        values are power spectral density in each frequency bin
    """
    def __init__(self,
                 syl_audio,
                 samp_freq,
                 power,
                 nfft,
                 overlap,
                 freq_cutoffs,
                 freq_bins,
                 time_bins):
        self.sylAudio = syl_audio
        self.sampFreq = samp_freq
        self.power = power
        self.nfft = nfft
        self.overlap = overlap
        self.freqCutoffs = freq_cutoffs
        self.freqBins = freq_bins
        self.timeBins = time_bins

def _make_syl_spect(syl_audio,
                    samp_freq,
                    nperseg,
                    overlap,
                    freq_cutoffs):
    """internal function that makes spectrograms for syllables

    Parameters
    ----------
    syl_audio : 1d numpy array
        raw audio waveform of a segmented syllable
    samp_freq : integer
        sampling frequency
    nfft : integer
        number of samples for each Fast Fourier Transform (FFT)
        in spectrogram.
    overlap : integer
        number of overlapping samples in each FFT.
    freq_cutoffs: list
        two-element list of integers, minimum and maximum frequency in FFT

    Returns
    -------
    syl_spect : object with properties as defined in the syl class
    """

    spect = Spectrogram(samp_freq,
                        nperseg,
                        noverlap,
                        freq_cutoffs)
    power,freq_bins,time_bins = spect.make(syl_audio)

    return syllable(syl_audio,
                    samp_freq,
                    power,
                    nperseg,
                    noverlap,
                    freq_cutoffs,
                    freq_bins,
                    time_bins)

class song:
    """song object
    used for feature extraction
    """

    def __init__(self,
                 filename,
                 file_format,
                 spect_params=None,
                 segment_params=None):
        """__init__ function for song object

        Parameters
        ----------
        filename : string
            name of file
        file_format : string
            {'evtaf','koumura'}
        spect_params : dictionary
            keys should be parameters for Spectrogram.__init__,
            see the docstring for those keys.
        segment_params : dictionary
            amplitude at which to threshold, default is None.
            if file not found that contains onsets and offsets,
            this value is used as threshold above which
            amplitude crossing are considered syllables.
            If file is found and this value is supplied,
            it will be ignored.
        """
        self.filename = filename
        self.fileFormat = file_format
        self.spectParams = spect_params
        if segment_params:
            self.segmentParams = segment_params
        if file_format == 'evtaf':
            dat, samp_freq = evfuncs.load_cbin(filename)
            try:
                song_dict = evfuncs.load_notmat(filename)
                self.labels = song_dict['labels']
            except FileNotFoundError:
                # if notmat not found, segment and get onsets and offsets
                song_dict = {}
                spect, time_bins, freq_bins = make_song_spect(dat,
                                                              samp_freq,
                                                              spect_params)
                amp = compute_amp(spect)
                onsets, offsets = segment_song(amp,
                                               time_bins,
                                               segment_params)
                song_dict['onsets'] = onsets
                song_dict['offsets'] = offsets
                self.labels = '-' * len(onsets)

            self.onsets_s = song_dict['onsets'] / 1000
            self.offsets_s = song_dict['offsets'] / 1000
            self.onsets_Hz = np.round(self.onsets_s * samp_freq).astype(int)
            self.offsets_Hz = np.round(self.offsets_s * samp_freq).astype(int)

        elif file_format == 'koumura':
            samp_freq, dat = wavfile.read(filename)
            song_dict = koumura.load_song_annot(filename)
            self.onsets_Hz = song_dict['onsets']
            self.offsets_Hz = song_dict['offsets']
            self.onsets_s = self.onsets_Hz / samp_freq
            self.offsets_s = song_dict['offsets'] / samp_freq
            self.labels = song_dict['labels']

    def set_syls_to_use(self,labels_to_use='all'):
        """        
        Parameters
        ----------
        labels_to_use : list or string
            List or string of all labels for which associated spectrogram should be made.
            When called by extract, this function takes a list created by the
            extract config parser. But a user can call the function with a string.
            E.g., if labels_to_use = 'iab' then syllables labeled 'i','a',or 'b'
            will be extracted and returned, but a syllable labeled 'x' would be
            ignored. If labels_to_use=='all' then all spectrograms are returned with
            empty strings for the labels. Default is 'all'.
        
        sets syls_to_use to a numpy boolean that can be used to index e.g. labels, onsets
        This method must be called before get_syls
        """

        if labels_to_use != 'all':
            if type(labels_to_use) != list and type(labels_to_use) != str:
                raise ValueError('labels_to_use argument should be a list or string')
            if type(labels_to_use) == str:
                labels_to_use = list(labels_to_use)

        if labels_to_use == 'all':
            self.syls_to_use = np.ones((self.onsets.shape),dtype=bool)
        else:
            self.syls_to_use = np.in1d(list(self.labels),
                                       labels_to_use)

    def make_syl_spects(self, spect_params, syl_spect_width=-1):
        """Make spectrograms from syllables.
        This method isolates making spectrograms from selecting syllables
        to use so that spectrograms can be loaded 'lazily', e.g., if only
        duration features are being extracted that don't require spectrograms.

        Parameters
        ----------
        spect_params: dict
            with the following keys
                    sampfreq : integer
                        sampling frequency in Hz, e.g. 32000
                    size: integer
                        size of FFT window, default is 512 samples
                    step: integer
                        number of samples between the start of each window, default is 32
                        i.e. if size == step then there will be no overlap of windows
                    freq_range: 2-element list
                        range of frequencies to return. Frequencies
                        less than the first element or greater than the second are discarded.

            Note that 'samp_freq' is the **expected** sampling frequency and the
            function throws an error if the actual sampling frequency of an audio file does
            not match the expected one.

        syl_spect_duration : int
            Optional parameter to set constant duration for each spectrogram of a
            syllable, in seconds. E.g., 0.05 for an average 50 millisecond syllable. 
            Used for creating inputs to neural network where each input
            must be of a fixed size.
            Default value is -1; in this case, the width of the spectrogram will
            be the duration of the spectrogram as determined by the segmentation
            algorithm in evsonganaly.m, i.e. the onset and offset that are stored
            in the .cbin.not.mat file.
            If a different value is given, then the duration of each spectrogram
            will be that value. Note that if any individual syllable has a duration
            greater than syl_spect_duration, the function raises an error.
        """

        if not hasattr(self,'syls_to_use'):
            raise ValueError('Must set syls_to_use by calling set_syls_to_use method '
                             'before calling get_syls.')

        if self.fileFormat == 'evtaf':
            dat, samp_freq = evfuncs.load_cbin(self.filename)
        elif self.fileFormat == 'koumura':
            samp_freq, dat = wavfile.read(self.filename)

        if samp_freq != spect_params['samp_freq']:
            raise ValueError(
                'Sampling frequency for {}, {}, does not match expected sampling '
                'frequency of {}'.format(filename,
                                         samp_freq,
                                         spect_params['samp_freq']))

        if syl_spect_width > 0:
            syl_spect_width_Hz = np.round(syl_spect_width * samp_freq)

        all_syls = []

        spect = Spectrogram(nperseg=spect_params['nperseg'],
                            noverlap=spect_params['noverlap'],
                            freq_cutoffs=spect_params['freq_cutoffs'])

        for ind, (label, onset, offset) in enumerate(zip(self.labels, self.onsets_Hz, self.offsets_Hz)):
            if 'syl_spect_width_Hz' in locals():
                syl_duration_in_samples = offset - onset
                if syl_duration_in_samples < syl_spect_width_Hz:
                    raise ValueError('syllable duration of syllable {} with label {}'
                                     'in file {} is greater than '
                                     'width specified for all syllable spectrograms.'
                                     .format(ind, label, cbin))

            if self.syls_to_use[ind]:
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
                        left_width = 0
                        right_width = width_diff - offset
                    elif offset + right_width > dat.shape[-1]:
                        # if right width greater than length of file
                        right_width = dat.shape[-1] - offset
                        left_width = width_diff - right_width
                    syl_audio = dat[:, onset - left_width:
                    offset + right_width]
                else:
                    syl_audio = dat[onset:offset]

                power, freq_bins, time_bins = spect.make(syl_audio,
                                                         samp_freq)

                curr_syl = syllable(syl_audio,
                                    spect_params['samp_freq'],
                                    power,
                                    spect_params['nperseg'],
                                    spect_params['noverlap'],
                                    spect_params['freq_cutoffs'],
                                    freq_bins,
                                    time_bins)

                all_syls.append(curr_syl)

        self.syls = all_syls