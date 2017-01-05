import numpy as np
from matplotlib.mlab import specgram

# constants
NFFT = 256
SPMAX = 128
OVERLAP = 192
MINF = 500
MAXF = 6000


#def specgram(wav,NFFT=256,Fs=32000,window=np.hanning,noverlap=192)
#    """specgram(wav,NFFT=256,Fs=32000,window=np.hanning,noverlap=192)
#    Internal function that (hopefully) returns same values as Matlab specgram function.
#    Parameters used by Tachibana are the defaults.
#    Cannibalized from matplotlib.mlab.spectral_helper_function
#    """

#    # does Matlab specgram detrend?
#    if detrend_func is None:
#        detrend_func = detrend_none
#    # not clear on diffs b/t windowsk
#    if window is None:
#        window = window_hanning

#    # if NFFT is set to None use the whole signal
#    if NFFT is None:
#        NFFT = 256

#    if mode is None or mode == 'default':
#        mode = 'psd'
#    elif mode not in ['psd', 'complex', 'magnitude', 'angle', 'phase']:
#        raise ValueError("Unknown value for mode %s, must be one of: "
#                         "'default', 'psd', 'complex', "
#                         "'magnitude', 'angle', 'phase'" % mode)

#    if not same_data and mode != 'psd':
#        raise ValueError("x and y must be equal if mode is not 'psd'")

#    # Make sure we're dealing with a numpy array. If y and x were the same
#    # object to start with, keep them that way
#    x = np.asarray(x)
#    if not same_data:
#        y = np.asarray(y)

#    if sides is None or sides == 'default':
#        if np.iscomplexobj(x):
#            sides = 'twosided'
#        else:
#            sides = 'onesided'
#    elif sides not in ['onesided', 'twosided']:
#        raise ValueError("Unknown value for sides %s, must be one of: "
#                         "'default', 'onesided', or 'twosided'" % sides)

#    # zero pad x and y up to NFFT if they are shorter than NFFT
#    if len(x) < NFFT:
#        n = len(x)
#        x = np.resize(x, (NFFT,))
#        x[n:] = 0

#    if not same_data and len(y) < NFFT:
#        n = len(y)
#        y = np.resize(y, (NFFT,))
#        y[n:] = 0

#    if pad_to is None:
#        pad_to = NFFT

#    if mode != 'psd':
#        scale_by_freq = False
#    elif scale_by_freq is None:
#        scale_by_freq = True

#    # For real x, ignore the negative frequencies unless told otherwise
#    if sides == 'twosided':
#        numFreqs = pad_to
#        if pad_to % 2:
#            freqcenter = (pad_to - 1)//2 + 1
#        else:
#            freqcenter = pad_to//2
#        scaling_factor = 1.
#    elif sides == 'onesided':
#        if pad_to % 2:
#            numFreqs = (pad_to + 1)//2
#        else:
#            numFreqs = pad_to//2 + 1
#        scaling_factor = 2.

#    result = stride_windows(x, NFFT, noverlap, axis=0)
#    result = detrend(result, detrend_func, axis=0)
#    result, windowVals = apply_window(result, window, axis=0,
#                                      return_window=True)
#    result = np.fft.fft(result, n=pad_to, axis=0)[:numFreqs, :]
#    freqs = np.fft.fftfreq(pad_to, 1/Fs)[:numFreqs]

#    if not same_data:
#        # if same_data is False, mode must be 'psd'
#        resultY = stride_windows(y, NFFT, noverlap)
#        resultY = apply_window(resultY, window, axis=0)
#        resultY = detrend(resultY, detrend_func, axis=0)
#        resultY = np.fft.fft(resultY, n=pad_to, axis=0)[:numFreqs, :]
#        result = np.conjugate(result) * resultY
#    elif mode == 'psd':
#        result = np.conjugate(result) * result
#    elif mode == 'magnitude':
#        result = np.absolute(result)
#    elif mode == 'angle' or mode == 'phase':
#        # we unwrap the phase later to handle the onesided vs. twosided case
#        result = np.angle(result)
#    elif mode == 'complex':
#        pass

#    if mode == 'psd':

#        # Also include scaling factors for one-sided densities and dividing by
#        # the sampling frequency, if desired. Scale everything, except the DC
#        # component and the NFFT/2 component:

#        # if we have a even number of frequencies, don't scale NFFT/2
#        if not NFFT % 2:
#            slc = slice(1, -1, None)
#        # if we have an odd number, just don't scale DC
#        else:
#            slc = slice(1, None, None)

#        result[slc] *= scaling_factor

#        # MATLAB divides by the sampling frequency so that density function
#        # has units of dB/Hz and can be integrated by the plotted frequency
#        # values. Perform the same scaling here.
#        if scale_by_freq:
#            result /= Fs
#            # Scale the spectrum by the norm of the window to compensate for
#            # windowing loss; see Bendat & Piersol Sec 11.5.2.
#            result /= (np.abs(windowVals)**2).sum()
#        else:
#            # In this case, preserve power in the segment, not amplitude
#            result /= np.abs(windowVals).sum()**2

#    t = np.arange(NFFT/2, len(x) - NFFT/2 + 1, NFFT - noverlap)/Fs

#    if sides == 'twosided':
#        # center the frequency range at zero
#        freqs = np.concatenate((freqs[freqcenter:], freqs[:freqcenter]))
#        result = np.concatenate((result[freqcenter:, :],
#                                 result[:freqcenter, :]), 0)
#    elif not pad_to % 2:
#        # get the last value correctly, it is negative otherwise
#        freqs[-1] *= -1

#    # we unwrap the phase here to handle the onesided vs. twosided case
#    if mode == 'phase':
#        result = np.unwrap(result, axis=0)

#    return result, freqs, t
    
def extract_Tachibana_features(wav,fs)
    """ extract_Tachibana_features
    Extracts acoustic features from waveform representing a songbird syllable.
    Based on Matlab code written by R.O. Tachibana (rtachi@gmail.com) in Sept 2013

    arguments:
        wav -- waveform of segmented syllable
        fs -- sampling frequency

    returns:
        features -- 532 dimensional vector of acoustic features from input waveform
    """

    #make sure wav is a numpy array
    wav = np.asarray(wav)
    
    #extract duration of syllable
    dur = wav.shape/fs

    # spectrogram and cepstrum -------------------
    wav_diff = np.diff(wav) # Tachibana applied a differential filter
    # note that the matlab specgram function returns the STFT by default
    # whereas the default for the matplotlib.mlab version of specgram returns the PSD.
    # So to get the behavior of matplotlib.mlab.specgram to match, mode must be set to 'complex'
    B,F = specgram(wav_diff,nfft,fs,hanning(nfft),overlap,mode='complex')[0:2] # don't need time vector
    B2 = np.vstack((B,np.flipud(B[1:-1,:]))
    S = 20*np.log10(np.abs(B[1:,:]))
    cc = np.real(np.fft.fft(np.log10(np.abs(B2)), axis=0))
    # ^ by this step, everything after the decimal point is already difft from what matlab returns
    C = cc(2:nfft/2+1,:);
    # 5-order delta
    if length(wav)<(5*nfft-4*overlap),
        SD = zeros(spmax,1);
        CD = zeros(spmax,1);
    else
        SD = -2*S(:,1:end-4)-1*S(:,2:end-3)+1*S(:,4:end-1)+2*S(:,5:end); 
        CD = -2*C(:,1:end-4)-1*C(:,2:end-3)+1*C(:,4:end-1)+2*C(:,5:end);
    end
    # mean 
    mS = mean(S,2)';
    mDS = mean(abs(SD),2)';
    mC = mean(C,2)';
    mDC = mean(abs(CD),2)';

#     make other acoustical features -------------------
#       SpecCentroid
#       SpecSpread
#       SpecSkewness
#       SpecKurtosis
#       SpecFlatness
#       SpecSlope
#       Pitch
#       PitchGoodness
#       Amp 
   
    maxq = round(fs/minf)*2;
    minq = round(fs/maxf)*2;
    [nr,nc] = size(B);
    x = repmat(F,1,nc);
    # amplitude spectrum
    s = abs(B);
    # probability
    p = s./repmat(sum(s,1),nr,1);
    # 1st moment: centroid (mean of distribution)
    m1 = sum(x.*p,1);
    # 2nd moment: variance (ƒÐ^2)
    m2 = sum(((x-repmat(m1,nr,1)).^2).*p, 1);
    # 3rd moment
    m3 = sum(((x-repmat(m1,nr,1)).^3).*p, 1);
    # 4th moment
    m4 = sum(((x-repmat(m1,nr,1)).^4).*p, 1);
    # distribution parameters
    SpecCentroid = m1; % mean
    SpecSpread = m2.^(1/2); % standard deviation
    SpecSkewness = m3./(m2.^(3/2));
    SpecKurtosis = m4./(m2.^2);
    # entropy
    SpecFlatness = exp(mean(log(s),1))./mean(s,1);
    # slope
    SpecSlope = zeros(1,nc);
    X = [F ones(nr,1)];
    for n=1:nc,
        beta = (X'*X)\X'*s(:,n);
        SpecSlope(n) = beta(1);
    end
    # cepstral pitch and pitch goodness
    exs = [s; repmat(s(end,:),nfft,1); flipud(s(2:end-1,:))];
    C = real(fft(log10(exs)));
    [mv,mid] = max(C(minq:maxq,:));
    Pitch = fs./(mid+minq-1);
    PitchGoodness = mv;
    % amplitude in dB
    Amp = 20*log10(sum(s)/nfft);

    # 5-order delta
    A = [SpecCentroid' SpecSpread' SpecSkewness' SpecKurtosis' SpecFlatness' SpecSlope' Pitch' PitchGoodness' Amp'];
    d5A = -2*A(1:end-4,:)-1*A(2:end-3,:)+1*A(4:end-1,:)+2*A(5:end,:); 

    % zerocross (in Hz)
    zc = length(find(diff(sign(wav))~=0))/2;
    ZeroCross = zc/dur;

    # mean
    mA = [mean(A,1) ZeroCross mean(abs(d5A),1)];
    mA(isnan(mA)) = 0;

    # output
    features = [mS mDS mC mDC dur mA];
    return features

