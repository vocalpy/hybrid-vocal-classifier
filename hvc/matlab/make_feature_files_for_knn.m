function make_feature_files_for_knn(notmats,purpose,varargin)
% make_feature_files_for_knn
%   extracts features from spectrograms in .spect.mat files
%   and saves in knn_ftr files
% 
%   input arguments:
%       quantify_deltas -- if quantify_deltas = 1, then take the difference of
%           the values of the extracted features at the times t_early and 
%           t_late. Default is quantify_deltas = 0 (i.e., don't quantify)
%       use_tmp -- use tmp.not.mat files; for this function, that means
%       'use the .spect.mat files associated with 
%       t_early -- "early" time in percent * 0.01. Default is 0.2 (i.e., 20%)
%       t_late -- "late" time in percent * 0.01. Default is 0.8

p = inputParser;
expectedPurposes = {'train','classify'};
defaultQuantifyDeltas = true;
default_t_early = 0.2;
default_t_late = 0.8;
default_use_tmp = false;
addRequired(p,'purpose',...
    @(x) any(validatestring(x,expectedPurposes)));
addOptional(p,'quantify_deltas',defaultQuantifyDeltas,...
    @(x) islogical(x));
addOptional(p,'t_early',default_t_early,...
    @(x) (0<x)&&(x<1));
addOptional(p,'t_late',default_t_late,...
    @(x) (0<x)&&(x<1));
p.parse(purpose,varargin{:})
purpose = p.Results.purpose;
quantify_deltas = p.Results.quantify_deltas;
if quantify_deltas
    t_early = p.Results.t_early;
    t_late = p.Results.t_late;
end

% Loop through files
for i=1:size(notmats,1)
    notmat_fn=notmats(i,:); % get name of .not.mat filename
    %want index of char *before* '.not.mat'
    if strcmp(purpose,'train')
        id = strfind(notmat_fn,'.not.mat') - 1;
        spect_fn = [notmat_fn(1:id) '.spect.to_train.mat'];
    elseif strcmp(purpose,'classify')
        id = strfind(notmat_fn,'.HDF.not.mat') - 1;
        spect_fn = [notmat_fn(1:id) '.spect.to_classify.mat'];
    end

    load(notmat_fn,'onsets','offsets');
    load(spect_fn)
    
    %calculate durations for all syls from song from offsets and onsets vectors
    syl_durations = offsets-onsets;
    pre_durations = [0;syl_durations(1:end-1)];
    foll_durations = [syl_durations(2:end);0];
    gapdurs = onsets(2:end) - offsets(1:end-1);
    pre_gapdurs = [0;gapdurs];
    foll_gapdurs = [gapdurs;0];

    num_syls = length(onsets);
    
    %for amp, entropy, hi-lo ratio: measure across entire spect, put
    %resulting vector into cell array, and then take mean across each
    %vector to get one value per syllable; i.e. mean amp, entropy, and
    %hi-lo ratio
    amp_smooth_rect_cell = cell(num_syls,1);
    amp_rms_cell = cell(num_syls,1);
    %wiener_entropy_cell = cell(num_syls,1);
    spectral_entropy_cell = cell(num_syls,1);
    hi_lo_ratio_cell = cell(num_syls,1);

    if quantify_deltas
        delta_amp_smooth_rect_array = [];
        delta_entropy_array = [];
        delta_hi_lo_ratio_array = [];
    end
    
    %extract features from each spectral slice for each syllable
    for syl_id=1:length(onsets)
        raw_syl = rawsyl_cell{syl_id};
        if isnan(raw_syl) % raw_syl is NaN if duration not long enough to make spectrogram
            %need to make sure all feature vals are also NaN
            syl_durations(syl_id) = NaN;
            pre_durations(syl_id) = NaN;
            foll_durations(syl_id) = NaN;
            gapdurs(syl_id) = NaN;
            pre_gapdurs(syl_id) = NaN;
            foll_gapdurs(syl_id) = NaN;
            amp_smooth_rect_cell{syl_id} = NaN;
            amp_rms_cell{syl_id} = NaN;
            spectral_entropy_cell{syl_id} = NaN;
            hi_lo_ratio_cell{syl_id} = NaN;
            
            if quantify_deltas
                delta_amp_smooth_rect_array = [delta_amp_smooth_rect_array;NaN];
                delta_entropy_array = [delta_entropy_array;NaN];
                delta_hi_lo_ratio_array = [delta_hi_lo_ratio_array;NaN];
            end
            
            continue % and don't actually attempt to calculate feature values!
        end
        
        [amp_sm_rect,amp_rms] = compute_amps(raw_syl,Fs,win_duration,overlap);
        
        syl_spect = spect_cell{syl_id};
        syl_psd = psd_cell{syl_id};
        syl_freqbins = freqbins_cell{syl_id};
        syl_timebins = timebins_cell{syl_id};
        spectral_entropy = [];
        hi_lo_ratio = [];
        % wiener_entropy = [];
        
        for slice_id = 1:size(syl_spect,2)
            spect_slice = syl_spect(:,slice_id);
            % make spectrum into power spectral density
            psd = (abs(spect_slice)).^2; % changed
            % normalize psd --> probability density function (pdf)
            psd_pdf = psd / sum(psd);
            spectral_entropy = [spectral_entropy -sum(psd_pdf .* log(psd_pdf))];
            
            scalar_hi_lo_ratio = log10(sum(psd(syl_freqbins<5000)) ./ sum(psd(syl_freqbins>5000)));
            hi_lo_ratio = [hi_lo_ratio scalar_hi_lo_ratio];
        end

        if quantify_deltas
            t_early_in_s = t_early * (syl_durations(syl_id)/1000);
            t_early_id = find(abs((syl_timebins - t_early_in_s)) == min(abs(syl_timebins - t_early_in_s)));
            %if "early" time to quantify is halfway between two time bins,
            %choose the earliest of the two
            if length(t_early_id)>1
                t_early_id = min(t_early_id);
            end
            
            t_late_in_s = t_late * (syl_durations(syl_id)/1000);
            t_late_id = find(abs((syl_timebins - t_late_in_s)) == min(abs(syl_timebins - t_late_in_s)));
            %likewise, if "late" time to quantify is halfway between two time
            %bins, choose the latest
            if length(t_late_id)>1
                t_late_id = min(t_late_id);
            end
            
            delta_amp_smooth_rect =  amp_sm_rect(t_late_id) - amp_sm_rect(t_early_id);
            delta_amp_smooth_rect_array = [delta_amp_smooth_rect_array;delta_amp_smooth_rect];
            delta_entropy = spectral_entropy(t_late_id) - spectral_entropy(t_early_id);
            delta_entropy_array = [delta_entropy_array;delta_entropy];
            delta_hi_lo_ratio = hi_lo_ratio(t_late_id) - hi_lo_ratio(t_early_id);
            delta_hi_lo_ratio_array = [delta_hi_lo_ratio_array;delta_hi_lo_ratio];
        end
        
        amp_smooth_rect_cell{syl_id} = amp_sm_rect;
        amp_rms_cell{syl_id} = amp_rms;
        spectral_entropy_cell{syl_id} = spectral_entropy;
        hi_lo_ratio_cell{syl_id} = hi_lo_ratio;

       
    end % of loop to loop through syllables

    mn_amp_smooth_rect = cellfun(@mean,amp_smooth_rect_cell);
    mn_amp_rms = cellfun(@mean,amp_rms_cell);
    mn_spect_entropy = cellfun(@mean,spectral_entropy_cell);
    mn_hi_lo_ratio = cellfun(@mean,hi_lo_ratio_cell);
    
    NUM_FEATURES = 9; 
    NUM_FEATURES_PLUS_DELTAS = 12;
    if quantify_deltas
        feature_cell = cell(2,NUM_FEATURES_PLUS_DELTAS);
    else
        feature_cell = cell(2,NUM_FEATURES);
    end

    feature_cell{1,1} = 'syl_durations';feature_cell{2,1} = syl_durations;
    feature_cell{1,2} = 'pre_durations';feature_cell{2,2} = pre_durations;
    feature_cell{1,3} = 'foll_durations';feature_cell{2,3} = foll_durations;
    feature_cell{1,4} = 'pre_gapdurs';feature_cell{2,4} = pre_gapdurs;
    feature_cell{1,5} = 'foll_gapdurs';feature_cell{2,5} = foll_gapdurs;
    feature_cell{1,6} = 'mn_amp_smooth_rect';feature_cell{2,6} = mn_amp_smooth_rect;
    feature_cell{1,7} = 'mn_amp_rms';feature_cell{2,7} = mn_amp_rms;
    feature_cell{1,8} = 'mn_spect_entropy';feature_cell{2,8} = mn_spect_entropy;
    feature_cell{1,9} = 'mn_hi_lo_ratio';feature_cell{2,9} = mn_hi_lo_ratio;

    if quantify_deltas
        feature_cell{1,10} = 'delta_amp_smooth_rect';feature_cell{2,10} = delta_amp_smooth_rect_array;
        feature_cell{1,11} = 'delta_entropy';feature_cell{2,11} = delta_entropy_array;
        feature_cell{1,12} = 'delta_hi_lo_ratio';feature_cell{2,12} = delta_hi_lo_ratio_array;
    end
    
    if strcmp(purpose,'train')
        extens_id = strfind(notmat_fn,'.not.mat')-1; % where to put new file extension
        ftr_fn = [notmat_fn(1:extens_id) '.knn_ftr.to_train.mat'];
    elseif strcmp(purpose,'classify')
        extens_id = strfind(notmat_fn,'.HDF.not.mat')-1; % where to put new file extension
        ftr_fn = [notmat_fn(1:extens_id) '.knn_ftr.to_classify.mat'];
    end
    disp(['Saving: ' ftr_fn])
    save(ftr_fn,'feature_cell')
    
end
