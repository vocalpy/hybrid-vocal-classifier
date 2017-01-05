function features = makeAllFeatures(wav,fs)
% Usage: features = makeAllFeatures(wav)
%   wav         ... short waveform of one segmented syllable 
%   features    ... 532-dimensional acoustical features of input waveform
%
% R.O. Tachibana (rtachi@gmail.com)
%  Sept. 28, 2013

% parameters
nfft = 256;
spmax = 128;
overlap = 192;
minf = 500;
maxf = 6000;

% duration
dur = length(wav)/fs;

% spectrogram and cepstrum -------------------
[B,F,~] = specgram(diff(wav),nfft,fs,hanning(nfft),overlap); % with differential filter
S = 20*log10(abs(B(2:end,:))); % spectrum. Why multiply 20 * log10?

B2 = [B; flipud(B(2:end-1,:))]; % used for cc. I guess concat w/flipud to make the right length?
cc = real(fft(log10(abs(B2)))); % take inverse FFT (take fft again) of log of spectrum
C = cc(2:nfft/2+1,:); % cepstrum

% 5-order delta
if length(wav)<(5*nfft-4*overlap),
    SD = zeros(spmax,1);
    CD = zeros(spmax,1);
else
    SD = -2*S(:,1:end-4)-1*S(:,2:end-3)+1*S(:,4:end-1)+2*S(:,5:end); 
    CD = -2*C(:,1:end-4)-1*C(:,2:end-3)+1*C(:,4:end-1)+2*C(:,5:end);
end

% mean 
mS = mean(S,2)'; % mean spectrum
mDS = mean(abs(SD),2)'; % mean delta spectrum
mC = mean(C,2)'; % mean cesptrum
mDC = mean(abs(CD),2)';

% make other acoustical features -------------------
%   SpecCentroid
%   SpecSpread
%   SpecSkewness
%   SpecKurtosis
%   SpecFlatness
%   SpecSlope
%   Pitch
%   PitchGoodness
%   Amp 
%   
maxq = round(fs/minf)*2;
minq = round(fs/maxf)*2;
[nr,nc] = size(B);
x = repmat(F,1,nc); % F = freqbins returned by specgram
% amplitude spectrum
s = abs(B);
% probability
p = s./repmat(sum(s,1),nr,1);
% 1st moment: centroid (mean of distribution)
m1 = sum(x.*p,1);
% 2nd moment: variance (ƒÐ^2)
m2 = sum(((x-repmat(m1,nr,1)).^2).*p, 1);
% 3rd moment
m3 = sum(((x-repmat(m1,nr,1)).^3).*p, 1);
% 4th moment
m4 = sum(((x-repmat(m1,nr,1)).^4).*p, 1);
% distribution parameters
SpecCentroid = m1; % mean
SpecSpread = m2.^(1/2); % standard deviation
SpecSkewness = m3./(m2.^(3/2));
SpecKurtosis = m4./(m2.^2);
% entropy
SpecFlatness = exp(mean(log(s),1))./mean(s,1);
% slope
SpecSlope = zeros(1,nc);
X = [F ones(nr,1)];
for n=1:nc,
    beta = (X'*X)\X'*s(:,n);
    SpecSlope(n) = beta(1);
end
% cepstral pitch and pitch goodness
exs = [s; repmat(s(end,:),nfft,1); flipud(s(2:end-1,:))];
C = real(fft(log10(exs)));
[mv,mid] = max(C(minq:maxq,:));
Pitch = fs./(mid+minq-1);
PitchGoodness = mv;
% amplitude in dB
Amp = 20*log10(sum(s)/nfft);

% 5-order delta
A = [SpecCentroid' SpecSpread' SpecSkewness' SpecKurtosis' SpecFlatness' SpecSlope' Pitch' PitchGoodness' Amp'];
d5A = -2*A(1:end-4,:)-1*A(2:end-3,:)+1*A(4:end-1,:)+2*A(5:end,:); 

% zerocross (in Hz)
zc = length(find(diff(sign(wav))~=0))/2;
ZeroCross = zc/dur;

% mean
mA = [mean(A,1) ZeroCross mean(abs(d5A),1)];
mA(isnan(mA)) = 0;

% output
features = [mS mDS mC mDC dur mA];
