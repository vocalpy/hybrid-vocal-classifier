function make_spect_files(notmats,purpose,varargin)
%make_spect_files(duration,overlap):
%   loops through directory of .cbin files, generating .spect.mat files.
%   These files contain spectrograms for each syllable detected in an
%   associated .cbin song file. The .spect.mat files are then used by the
%   extract_features function to measure acoustic parameters for each
%   spectral slice of a syllable, as well as the mean and standard
%   deviation of those parameters.
%
%   Input arguments:
%       Spectrograms are created by the spect_from_waveform function. Hence
%       the optional input arguments are the parameters used to create the
%       spectrograms.
%           win_duration: time in ms. Length of each bin used to generate a
%           spectrum from the raw voltage waveform of the syllable
%           Overlap: percentage. Amount time bins should overlap
%   Example:
%       make_spect_files(32,0) % 32 ms time bins with no overlap
%
%   Each .cbin file should already have an associated .not.mat file. This
%   does *not* mean the syllables in all the song files should already have
%   been labeled by a user. To generate .not.mat files without labeling by
%   hand, use the make_notmat function.

p = inputParser;
expectedPurposes = {'train','classify'};
addRequired(p,'purpose',...
    @(x) any(validatestring(x,expectedPurposes)));
p.addOptional('win_duration',32,@(x) (mod(x,1)==0) & (x>0));
p.addOptional('overlap',0,@(x) (x>=0 & x<=1));
p.addOptional('use_tmp',false,@(x) (x==true || x==false));
p.parse(purpose,varargin{:})

win_duration = p.Results.win_duration;
overlap = p.Results.overlap;


%TODO determine if segmenting parameters have already been decided, if not
%then alert user and suggest default parameters

%TODO save spect parameters in .spect file!

% Loop throu all .not.mat files in current directory
for i=1:size(notmats,1)
    notmat_fn = notmats(i,:); % get name of .not.mat filename
    if strcmp(purpose,'train')
        id = strfind(notmat_fn,'.not.mat')-1;
        cbin_fn=notmat_fn(1:id);                % name of corresponding .cbin file
    elseif strcmp(purpose,'classify')
        id = strfind(notmat_fn,'.HDF.not.mat')-1;
        cbin_fn=notmat_fn(1:id);
    end
    
    load(notmat_fn);
    
    % Load raw song waveforms
    [rawsong, Fs]=evread_labv_file(cbin_fn,0);
    
    % convert onsets and offsets from ms to s
    onsets=onsets/1000;
    offsets=offsets/1000;
    t=[1:length(rawsong)]/Fs; % Time
    
    rawsyl_cell = cell(length(labels),1);
    spect_cell = cell(length(labels),1);
    freqbins_cell = cell(length(labels),1);
    timebins_cell = cell(length(labels),1);
    psd_cell = cell(length(labels),1);
    
    % Loop through each syllable
    for syl=1:length(labels)
        syl_onset = onsets(syl);
        syl_offset = offsets(syl);

        %if syl is too short for spectrogram
        if (syl_offset-syl_onset) < (win_duration/1000)
            syl_offset = syl_onset + (win_duration/1000);    % extend end of syl
            disp({['Syllable ' labels(syl) 'is less then ' num2str(win_duration)],...
                ['Extending offset.']})
        end
        
        onset_id=find(abs((t-syl_onset))==min(abs(t-syl_onset)));
        offset_id=find(abs((t-syl_offset))==min(abs(t-syl_offset)));
        
        raw_syl=rawsong(onset_id:offset_id); % waveform of the syllable
        
        % Compute spectrogram for whole syllable
        try
            [spect,freqbins,timebins,psd] = ...
                spect_from_rawsyl(raw_syl,Fs,[overlap win_duration]);
                rawsyl_cell{syl} = raw_syl;
                spect_cell{syl} = spect;
                psd_cell{syl} = psd;
                freqbins_cell{syl} = freqbins;
                timebins_cell{syl} = timebins;
        catch exception
            if (strcmp(exception.identifier,'signal:welchparse:invalidSegmentLength'))
                warn_str = sprintf(['syllable ' num2str(syl) ' duration in ' cbin_fn ... 
                    ' not long enough to make spectrogram.\n'...
                    'Will be entered as NaN.']);
                warning(warn_str)
                rawsyl_cell{syl} = NaN;
                spect_cell{syl} = NaN;
                psd_cell{syl} = NaN;
                freqbins_cell{syl} = NaN;
                timebins_cell{syl} = NaN;
            end
        end
            
        
    end
    
    if strcmp(purpose,'train')
        save_fn = [cbin_fn '.spect.to_train.mat'];
    elseif strcmp(purpose,'classify')
        save_fn = [cbin_fn '.spect.to_classify.mat'];
    end
    
    disp(['saving ' save_fn])
    save(save_fn,...
        'Fs',...
        'rawsyl_cell',...
        'spect_cell',...
        'freqbins_cell',...
        'timebins_cell',...
        'psd_cell',...
        'win_duration',...
        'overlap')
end % of main "while 1" loop;
