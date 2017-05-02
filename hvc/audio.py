import numpy as np
from scipy.io import wavfile
from scipy.signal import slepian # AKA DPSS, window used for FFT
from scipy.signal import spectrogram

from hvc.evfuncs import load_cbin,load_notmat

class song_spect:
    """
    spectrogram object, returned by make_spect.
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

class syl_spect:
    """
    syllable spectrogram object, returned by make_syl_spect.
    Properties:
        spect -- 2-d m by n numpy array, spectrogram as computed by make_song_spect.
                 Each of the m rows is a frequency bin, and each of the n columns is a time bin.
        time_bins -- 1d vector, values are times represented by each bin in s
        freq_bins -- 1d vector, values are power spectral density in each frequency bin
        sampfreq -- sampling frequency in Hz as determined by scipy.io.wavfile function
    """
    def __init__(self,
                 syl_audio,
                 samp_freq,
                 power,
                 nfft,
                 overlap,
                 freq_bins,
                 time_bins):
        self.sylAudio = syl_audio
        self.sampFreq = samp_freq
        self.power = power
        self.nfft = nfft
        self.overlap = overlap
        self.freqBins = freq_bins
        self.timeBins = time_bins

def make_syl_spect(syl_audio,samp_freq,nfft=256,overlap=192,minf=500,maxf=6000):
    """
    makes spectrograms as in [1]_.

    Parameters
    ----------
    syl_audio : 1d numpy array, raw audio waveform of a segmented syllable
    samp_freq : integer, sampling frequency
    nfft : integer, number of samples for each Fast Fourier Transform (FFT)
           in spectrogram. Default is 256.
    overlap : integer, number of overlapping samples in each FFT. Default is 192.
    minf : integer, minimum frequency in FFT
    maxf : integer, maximum frequency in FFT

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
    syl_diff = np.diff(syl) # Tachibana applied a differential filter
    # note that the matlab specgram function returns the STFT by default
    # whereas the default for the matplotlib.mlab version of specgra
    # returns the PSD. So to get the behavior of matplotlib.mlab.specgram
    # to match, mode must be set to 'complex'
    power,freq_bins,time_bins = specgram(syl_diff,NFFT=nfft,Fs=fs,window=np.hanning(nfft),
                                         noverlap=overlap,mode='complex')
    return syl_spect(syl_audio,
                     samp_freq,
                     power,
                     nfft,
                     overlap,
                     freq_bins,
                     time_bins)