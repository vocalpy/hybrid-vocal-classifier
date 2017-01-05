function [smooth,spec,t,f]=evsmooth(rawsong,Fs,SPTH,nfft,olap,sm_win,F_low,F_High)
% [smooth,spec,t,f]=evsmooth(rawsong,Fs,SPTH,nfft,olap,sm_win,F_low,F_High);
% returns the smoothed waveform/envelope + the spectrum
%

if (~exist('F_low'))
	F_low  = 500.0;
end
if (~exist('F_high'))
	F_high = 10000.0;
end
if (~exist('nfft'))
	nfft = 512;
end
if (~exist('olap'))
	olap = 0.8;
end
if (~exist('sm_win'))
	sm_win = 2.0;%ms
end
if(~exist('SPTH'))
    SPTH=0.01;
end

filter_type = 'hanningfir';

%filtsong=bandpass(rawsong,Fs,F_low,F_high,filter_type);
filtsong=bandpass_filtfilt(rawsong,Fs,F_low,F_high,filter_type);

if (nargout > 1)
	spect_win = nfft;
	noverlap = floor(olap*nfft);
	[spec,f,t] = specgram(filtsong,nfft,Fs,spect_win,noverlap);
	if (exist('SPTH'))
		p = find(abs(spec)<=SPTH);
		spec(p) = SPTH;
	end
end

squared_song = filtsong.^2;
       
%smooth the rectified song
len = round(Fs*sm_win/1000);
h   = ones(1,len)/len;
smooth = conv(h, squared_song);
offset = round((length(smooth)-length(filtsong))/2);
smooth=smooth(1+offset:length(filtsong)+offset);


return;
