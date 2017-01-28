import numpy as np
from matplotlib.mlab import specgram

def extract_svm_features(syls,fs,nfft=256,spmax=128,overlap=192,minf=500,
                         maxf=6000):
    """
    Extracts acoustic features from waveform representing a songbird syllable.
    Based on Matlab code written by R.O. Tachibana (rtachi@gmail.com) in Sept.
    2013. These features were previously shown to be effective for classifying
    Bengalese finch song syllables [1]_. Note all defaults for the spectrogram
    are taken from the Tachibana code and paper.

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
    NUM_FEATURES = 532
    num_syls = len(syls)
    feature_arr = np.empty((num_syls,NUM_FEATURES))
    for syl in syls:
        #make sure syl is a numpy array
        syl = np.asarray(syl)
        
        #extract duration of syllable
        dur = syl.shape/fs

        # spectrogram and cepstrum -------------------
        syl_diff = np.diff(syl) # Tachibana applied a differential filter
        # note that the matlab specgram function returns the STFT by default
        # whereas the default for the matplotlib.mlab version of specgram returns the PSD.
        # So to get the behavior of matplotlib.mlab.specgram to match, mode must be set to 'complex'
        B,F = specgram(syl_diff,nfft,fs,np.hanning(nfft),overlap,
                       mode='complex')[0:2] # don't need time vector
        B2 = np.vstack((B,np.flipud(B[1:-1,:])))
        S = 20*np.log10(np.abs(B[1:,:]))
        import pdb;pdb.set_trace()
        cc = np.real(np.fft.fft(np.log10(np.abs(B2)), axis=0))
        # ^ by this step, everything after the decimal point is already difft from what matlab returns
#        C = cc[2:nfft/2+1,:]
#        # 5-order delta
#        if length(syl) < (5*nfft-4*overlap):
#            SD = np.zeros(spmax,1)
#            CD = np.zeros(spmax,1)
#        else:
#            SD = -2*S[:,1:-4]-1*S[:,2:-3]+1*S[:,4:-1]+2*S[:,5:]
#            CD = -2*C[:,1:-4]-1*C[:,2:-3]+1*C[:,4:-1]+2*C[:,5:]

#        # mean 
#        mS = np.mean(S,2).T
#        mDS = np.mean(np.abs(SD),2).T
#        mC = np.mean(C,2).T
#        mDC = np.mean(np.abs(CD),2).T

#        maxq = np.round(fs/minf)*2
#        minq = np.round(fs/maxf)*2
#        nr,nc = B.shape
#        x = repmat(F,1,nc);
#        # amplitude spectrum
#        s = np.abs(B)
#        # probability
#        p = s / repmat(np.sum(s,1),nr,1)
#        # 1st moment: centroid (mean of distribution)
#        m1 = np.sum(x * p,1)
#        # 2nd moment: variance (ƒÐ^2)
#        m2 = np.sum((np.power(x-repmat(m1,nr,1),2)) * p, 1)
#        # 3rd moment
#        m3 = np.sum((np.power(x-repmat(m1,nr,1),3)) * p, 1)
#        # 4th moment
#        m4 = np.sum((np.power(x-repmat(m1,nr,1),4)) * p, 1)
#        # distribution parameters
#        SpecCentroid = m1  # mean
#        SpecSpread = np.power(m2,1/2)  # standard deviation
#        SpecSkewness = m3 / np.power(m2,3/2)
#        SpecKurtosis = m4 / np.power(m2,2)
#        # entropy
#        SpecFlatness = np.exp(np.mean(np.log(s),1)) / np.mean(s,1)
#        # slope
#        SpecSlope = np.zeros((1,nc))
#        X = np.horzcat((F,np.ones((nr,1))
#        for n in range(nc):
#            beta = (X.T*X)\X.T*s[:,n]
#            SpecSlope[n] = beta[1]

#        # cepstral pitch and pitch goodness
#        exs = np.vertcat((s,repmat(s(end,:),nfft,1),flipud(s(2:end-1,:)))
#        C = real(fft(log10(exs)));
#        [mv,mid] = np.max(C(minq:maxq,:));
#        Pitch = fs./(mid+minq-1);
#        PitchGoodness = mv;
#        # amplitude in dB
#        Amp = 20*log10(sum(s)/nfft);

#        # 5-order delta
#        A = np.horzcat((SpecCentroid.T,SpecSpread.T,SpecSkewness.T,SpecKurtosis.T,
#                        SpecFlatness.T,SpecSlope.T,Pitch.T,PitchGoodness.T,Amp.T)
#        d5A = -2*A[1:-4,:]-1*A[2:-3,:]+1*A[4:-1,:]+2*A[5:,:]

#        # zerocross (in Hz)
#        zc = length(find(diff(sign(syl))~=0))/2
#        ZeroCross = zc/dur

#        # mean
#        mA = np.horzcat((np.mean(A,1),ZeroCross,np.mean(np.abs(d5A),1))
#        mA[np.isnan(mA)] = 0

#        # output
#        features = np.horzcat((mS,mDS,mC,mDC,dur,mA))
    return features

