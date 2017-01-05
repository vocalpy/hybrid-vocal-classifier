function make_HDFnotmat(min_dur,min_int,threshold,sm_win)
%make_HDFnotmat
% generates HDF.not.mat files from unlabeled song.
% HDF.not.mat files are at first just temporary files used to extract
% features from unlabeled syllables so those syllables can be classified by
% the classification scripts. Unlabeled song files are segmented into
% syllables using the same segmenting parameters used with labeled song.
%
% After the computer classifies the syllables, the dummy labels in
% HDF.not.mat files are changed to the label selected by the computer. The
% user can view these results in the song-labeling GUI by checking the
% appropriate checkbox.
%
% syntax:
% make_temp_notmat(min_dur,min_int,threshold,sm_win)
%
% input arguments: segmenting parameters
%   min_dur -- minimum syllable duration in ms, default is 20
%   min_int -- minimum inter-syllable interval in ms, default is 2
%   threshold -- that amplitude crosses, default is 1k
%   sm_win -- size of smoothing window, default is 2

if ~exist('min_dur','var');min_dur=20;end
if ~exist('min_int','var');min_int=2;end
if ~exist('threshold','var');threshold=1000;end
if ~exist('sm_win','var');sm_win=2;end

if isunix
    !ls **.cbin > batchfile
else
    !dir /B **.cbin > batchfile
end

ct=0;n_files=0;fid=fopen('batchfile');while 1;fname=fgetl(fid);if (~ischar(fname));break;end;n_files=n_files+1;end;fclose(fid);

fid=fopen('batchfile');
while 1
    fname=fgetl(fid);
    if (~ischar(fname));break;end
    
    HDFnotmat_fname=[fname '.HDF.not.mat'];
    
    [dat,Fs]=evsoundin('',fname,'obs0');

    DOFILT=1;
    sm=SmoothData_local(dat,Fs,DOFILT);
    sm(1)=0.0;sm(end)=0.0;
    
    [onsets, offsets]=SegmentNotes_local(sm, Fs, min_int, min_dur, threshold);
    labels = char(ones([1,length(onsets)])*fix('Q'));
    
    onsets=onsets*1000;  % put in msec to match .not.mat output of evsonganaly.m
    offsets= offsets*1000;
    
    save(HDFnotmat_fname,'Fs','fname','labels','min_dur','min_int','threshold','sm_win','onsets','offsets')
    ct=ct+1;disp(['Processed ' num2str(ct) ' of ' num2str(n_files) ' files'])
end

%  This is copied from SmoothData.m, part of the evsonganaly.m package
% modified by removing spectrogram component
%
function smooth=SmoothData_local(rawsong,Fs,DOFILT,nfft,olap,sm_win,F_low,F_High);
% [smooth,spec,t,f]=evsmooth(rawsong,Fs,DOFILT,nfft,olap,sm_win,F_low,F_High,DOFILT);
% returns the smoothed waveform/envelope + the spectrum

if ~exist('F_low','var');F_low  = 500.0;end
if ~exist('F_high','var');F_high = 10000.0;end
if ~exist('nfft','var');nfft = 512;end
if ~exist('olap','var');olap = 0.5;end
if ~exist('sm_win','var');sm_win = 2.0;end % msec
if ~exist('DOFILT','var');DOFILT=1;end

filter_type = 'hanningfir';

if (DOFILT==1)
    %filtsong=bandpass(rawsong,Fs,F_low,F_high,filter_type);
    %disp('SmoothData.m called - now using filtfilt')
    filtsong=bandpass_filtfilt(rawsong,Fs,F_low,F_high,filter_type);
else
    filtsong=rawsong;
end

squared_song = filtsong.^2;

%smooth the rectified song
len = round(Fs*sm_win/1000);
h   = ones(1,len)/len;
smooth = conv(h, squared_song);
offset = round((length(smooth)-length(filtsong))/2);
smooth=smooth(1+offset:length(filtsong)+offset);


%  This is copied from SegmentNotes.m, part of the evsonganaly.m package
%
function [onsets, offsets]=SegmentNotes_local(smooth, Fs, min_int, min_dur, threshold)
% [ons,offs]=evsegment(smooth,Fs,min_int,min_dur,threshold);
% segment takes smoothed filtered song and returns vectors of note
% onsets and offsets values are in seconds

h=[1 -1];

%threshold input
%notetimes=abs(diff(smooth))>threshold;
notetimes=double(smooth>threshold);

%extract index values for note onsets and offsets
trans=conv(h,notetimes);
t_onsets  = find(trans>0);
t_offsets = find(trans<0);

onsets = t_onsets;offsets=t_offsets;
if ((length(onsets)<1)|(length(offsets)<1))
    onsets=[];offsets=[];
    return;
end

if (length(t_onsets) ~= length(t_offsets))
    disp('number of note onsets and offsets do not match')
else
    %eliminate short intervals
    temp_int=(onsets(2:length(onsets))-offsets(1:length(offsets)-1))*1000/Fs;
    real_ints=temp_int>min_int;
    onsets=[onsets(1); nonzeros(onsets(2:length(onsets)).*real_ints)];
    offsets=[nonzeros(offsets(1:length(offsets)-1).*real_ints); offsets(length(offsets))];
    
    %eliminate short notes
    temp_dur=(offsets-onsets)*1000/Fs;
    real_durs=temp_dur>min_dur;
    onsets=[nonzeros((onsets).*real_durs)];
    offsets=[nonzeros((offsets).*real_durs)];
    
    %convert to ms: peculiarities here are to prevent rounding problem
    % if t_ons is simply replaced with onsets, everything gets rounded
    onsets = onsets/Fs; % all in seconds
    offsets = offsets/Fs; %all in seconds
end

