function [filtsong]=bandpass_filtfilt(rawsong,Fs,F_low,F_high,varargin)

%this was rather slow, but gives nice filtereing, now using butterworth
%bandpass filters song using fir1 and filtfilt
%

if nargin == 5
  filter_type = varargin{1};
end
%get filter order
%nframes = length(rawsong);
	
%if ( nframes > 3*512 )
%   nfilt = 512;
%elseif ( nframes > 3*64 )
%   nfilt = 64;
%else

%if ( nframes > 3*16 )
%   nfilt = 16;
%else
%   disp ('filtsong is too short, modify parameters in bandpass.m')
%end 
 
v=version;
if str2num(v(1)) < 5
  disp(['warning! bandpass: filtering is lousy with mlab version < 5.xx'])
  disp(['current version = ',v]);
end     

%filter song with fir1/filtfilt
%	song_filt = fir1(nfilt,[F_low*2/Fs, F_high*2/Fs]);
%	filtsong = filtfilt(song_filt, 1, rawsong);

%check to make sure input F_high is OK w/ given sample rate
if F_high >= .5*Fs-500
   F_high = .5*Fs-1000;
   disp(['warning! (bandpass): F_high was >= .5*Fs-500']);
   disp(['F_high has been set to ',num2str(F_high)]);
end

if exist('filter_type','var') & strcmpi(filter_type,'hanningfir')
    
    if length(rawsong)<387
        nfir = 64;
%        disp('shortening hanning window to 128 in bandpass_filtfilt.m')
    elseif length(rawsong)<771
        nfir = 128;
%        disp('shortening hanning window to 128 in bandpass_filtfilt.m')
    elseif length(rawsong)<1539
        nfir = 256;
%        disp('shortening hanning window to 256 in bandpass_filtfilt.m')
    else
        nfir = 512;
    end
    ndelay = fix(nfir/2);
    bfir = fir1(nfir,[F_low*2/Fs, F_high*2/Fs]);
    %  filtsong = filter(bfir,1,rawsong);

% IMPORTANT - NEEDS TO BE ON FILTFILT SETTING, OR WILL GET A DELAY
filtsong = filtfilt(bfir,1,rawsong);  
else
  % Now using an 8 pole butterworth bandpass filter as default.
  [b,a]=butter(8,[F_low*2/Fs, F_high*2/Fs]);
  filtsong=filtfilt(b, a, rawsong);
end
  
if length(rawsong) ~= length(filtsong) 
  disp(['warning! bandpass: input and output file lengths do not match!']);
end
