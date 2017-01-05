function [S1,F1,T1,P1] = spect_from_rawsyl(Y,FS,spect_params)
% spect_from_rawsyl: a blatant ripoff of spect_from_waveform
% (made some minor changes --
% changed name so it doesn't shadow the original function)
%
% [S1,F1,T1,P1] =spect_from_waveform(Y,FS,plot,spect_params)
% 
% SPECT_FROM_WAVEFORM  Make spectrogram of smoothed waveform.  
% 
% ARGS:
% 
%     Y: raw waveform
%     FS: sampling rate in Hz.
%       Get Y and FS by calling the function EVSOUNDIN.     
%     SPECT_PARAMS: [spect_overlap spect_win_dur], i.e.
%                   [percent_overlap window_size_in_milliseconds]
%                   default is [0 32]
% RETURNS:
% S1: spectrogram of the signal
% F1: frequencies at which spectrogram was computed
% T1: times at which the spectrogram was computed
% P1: power spectral density of each segment.
% Note that only frequencies between 500 hZ and 10 kHz are returned because
% this is where most of the power for Bengalese finch song occurs

if nargin==2,spect_params=[0 32];end

spect_overlap = spect_params(1);  %percentage of overlap of specgram window
spect_win_dur = spect_params(2);

%first calculate nfft and noverlap
nfft=round(FS*spect_win_dur/1000);
% ^ number of frequency points to use is
% sampling rate times window duration in seconds 
% i.e., window length in number of samples

nfft=2^(round(log2(nfft))); % round nfft to nearest power of 2

spect_win = hanning(nfft);

noverlap = round(spect_overlap*length(spect_win)); %number of overlapping points

[S1,F1,T1,P1] = spectrogram(Y,spect_win,noverlap,nfft,FS);

low_cutoff=500;
high_cutoff=10000;
id1=find(F1>=low_cutoff & F1<=high_cutoff);
F1=F1(id1);
P1=P1(id1,:);
S1=S1(id1,:);
mean(diff(F1));

