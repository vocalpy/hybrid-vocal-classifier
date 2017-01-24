import numpy as np
from matplotlib.mlab import specgram

def extract_Tachibana_features(wav,fs,nfft=256,spmax=128,overlap=192,minf=500,
                               maxf=6000)
    """
    Extracts acoustic features from waveform representing a songbird syllable.
    Based on Matlab code written by R.O. Tachibana (rtachi@gmail.com) in Sept.
    2013. Note all defaults for the spectrogram are taken from the Tachibana
    code and paper.

    arguments
    ---------
    wav : numpy vector
        waveform of segmented syllable
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

    returns
    -------
    features : numpy vector
            532 dimensional vector of acoustic features from input waveform
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
    C = cc[2:nfft/2+1,:]
    # 5-order delta
    if length(wav) < (5*nfft-4*overlap):
        SD = np.zeros(spmax,1);
        CD = np.zeros(spmax,1);
    else
        SD = -2*S(:,1:end-4)-1*S(:,2:end-3)+1*S(:,4:end-1)+2*S(:,5:end); 
        CD = -2*C(:,1:end-4)-1*C(:,2:end-3)+1*C(:,4:end-1)+2*C(:,5:end);
    end
    # mean 
    mS = np.mean(S,2)'
    mDS = np.mean(np.abs(SD),2)'
    mC = np.mean(C,2)'
    mDC = np.mean(np.abs(CD),2)'

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

