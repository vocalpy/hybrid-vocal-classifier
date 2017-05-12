import numpy as np
from scipy.io import wavfile
from scipy.signal import slepian # AKA DPSS, window used for FFT
from scipy.signal import spectrogram
from matplotlib.mlab import specgram

from . import evfuncs, koumura

class song_spect:
    """
    spectrogram object, returned by make_song_spect.
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
    wav_file : string, filename of .wav file corresponding to song
    size: integer, size of FFT window, default is 512 samples
    step: integer, number of samples between the start of each window, default is 32
        i.e. if size == step then there will be no overlap of windows
    freq_range: 2-element list, range of frequencies to return. Frequencies
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
    freq_bins, time_bins, spect = spectrogram(waveform,
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

def segment_song(amp,time_bins,threshold=5000,min_syl_dur=0.02,min_silent_dur=0.002):
    """
    Divides songs into segments based on threshold crossings of amplitude.
    Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
    Inputs:
        amp -- amplitude of power spectral density. Returned by compute_amp.
        time_bins -- time in s, must be same length as log amp. Returned by make_song_spect.
        threshold -- value above which amplitude is considered part of a segment. default is 5000.
        min_syl_dur -- minimum duration of a syllable. default is 0.02, i.e. 20 ms.
        min_silent_dur -- minimum duration of silent gap between syllables. default is 0.002, i.e. 2 ms.
    Returns:
        onsets, offsets -- arrays of onsets and offsets of segments.
        So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
        To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """
    above_th = amp > threshold
    h = [1, -1] 
    above_th_convoluted = np.convolve(h,above_th) # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    onsets = time_bins[np.nonzero(above_th_convoluted > 0)]
    # and -1 whenever above_th changes from 1 to 0
    offsets = time_bins[np.nonzero(above_th_convoluted < 0)]
    
    #get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1] # duration of silent gaps
    keep_these = np.nonzero(silent_gap_durs > min_silent_dur)
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]
    
    #eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    keep_these = np.nonzero(syl_durs > min_syl_dur)
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]    
    
    return onsets,offsets

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

def _make_syl_spect(syl_audio,samp_freq,nfft=256,overlap=192,freq_cutoffs=[500,6000]):
    """
    internal function that makes spectrograms for syllables
    Defaults are as in [1]_.

    Parameters
    ----------
    syl_audio : 1d numpy array
        raw audio waveform of a segmented syllable
    samp_freq : integer
        sampling frequency
    nfft : integer
        number of samples for each Fast Fourier Transform (FFT)
        in spectrogram. Default is 256.
    overlap : integer
        number of overlapping samples in each FFT. Default is 192.
    freq_cutoffs: list
        two-element list of integers, minimum and maximum frequency in FFT

    Returns
    -------
    syl_spect : object with properties as defined in the syl class

    References
    ----------
    .. [1] Tachibana, Ryosuke O., Naoya Oosugi, and Kazuo Okanoya. "Semi-
    automatic classification of birdsong elements using a linear support vector
     machine." PloS one 9.3 (2014): e92584.

    """
    # for below, need to have a 'filter' option in extract files
    # and then have named_spec_params where one would be 'Tachibana', another 'Koumura', another 'Sober', etc.

    # spectrogram and cepstrum
    syl_diff = np.diff(syl_audio) # Tachibana applied a differential filter
    # note that the matlab specgram function returns the STFT by default
    # whereas the default for the matplotlib.mlab version of specgra
    # returns the PSD. So to get the behavior of matplotlib.mlab.specgram
    # to match, mode must be set to 'complex'
    power,freq_bins,time_bins = specgram(syl_diff,
                                         NFFT=nfft,
                                         Fs=samp_freq,
                                         window=np.hanning(nfft),
                                         noverlap=overlap,
                                         mode='complex')

    f_inds = np.nonzero((freq_bins >= freq_cutoffs[0]) &
                        (freq_bins < freq_cutoffs[1]))[0] #returns tuple
    freq_bins = freq_bins[f_inds]
    power = power[f_inds,:]
    power = np.log10(power) # log transform to increase range

    #flip spect and freq_bins so lowest frequency is at 0 on y axis when plotted
    power = np.flipud(power)
    freq_bins = np.flipud(freq_bins)
    return syllable(syl_audio,
                    samp_freq,
                    power,
                    nfft,
                    overlap,
                    freq_cutoffs,
                    freq_bins,
                    time_bins)

class song:
    """

    """

    def __init__(self,songfile,file_format):
        self.file = songfile
        self.file_format = file_format
        if file_format == 'evtaf':
            song_dict = evfuncs.load_notmat(songfile)
            dat, samp_freq = evfuncs.load_cbin(songfile)
            self.onsets_s = song_dict['onsets'] / 1000
            self.offsets_s = song_dict['offsets'] / 1000
            self.onsets_Hz = np.round(self.onsets_s * samp_freq).astype(int)
            self.offsets_Hz = np.round(self.offsets_s * samp_freq).astype(int)

            self.labels = song_dict['labels']
        elif file_format == 'koumura':
            samp_freq, dat = wavfile.read(songfile)
            song_dict = koumura.load_song_annot(songfile)
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
        """
        Make spectrograms from syllables.
        This method isolates making spectrograms from selecting syllables
        to use so that spectrograms can be loaded 'lazily', e.g., if only
        duration features are being extracted that don't require spectrograms.

        Parameters
        ----------
        self : 
                    
        spect_params: dictionary
            with keys 'nperseg','noverlap','freq_cutoffs', and 'samp_freq'.
            Note that 'samp_freq' is the **expected** sampling frequency and the
            function throws an error if the actual sampling frequency of cbin does
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

        if self.file_format == 'evtaf':
            dat, samp_freq = evfuncs.load_cbin(self.file)
        elif self.file_format == 'koumura':
            samp_freq, dat = wavfile.read(self.file)

        if samp_freq != spect_params['samp_freq']:
            raise ValueError(
                'Sampling frequency for {}, {}, does not match expected sampling '
                'frequency of {}'.format(filename,
                                         samp_freq,
                                         spect_params['samp_freq']))

        if syl_spect_width > 0:
            syl_spect_width_Hz = np.round(syl_spect_width * samp_freq)

        all_syls = []

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
                curr_syl = _make_syl_spect(syl_audio,
                                           samp_freq,
                                           nfft=spect_params['nperseg'],
                                           overlap=spect_params['noverlap'],
                                           freq_cutoffs =spect_params['freq_cutoffs'])
                all_syls.append(curr_syl)

        self.syls = all_syls