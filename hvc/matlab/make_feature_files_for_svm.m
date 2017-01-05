function make_feature_files_for_svm(notmats,purpose)
%make_feature_file_for_svm
%
%loops across all .not.mat files in a directory and extracts features from
%syllables using code from Tachibana et al. 2014.

p = inputParser;
expectedPurposes = {'train','classify'};
addRequired(p,'purpose',...
    @(x) any(validatestring(x,expectedPurposes)));
p.parse(purpose)
purpose = p.Results.purpose;

ct=1;

for i=1:size(notmats,1)
    features_mat = {}; %"declare" cell array so interpreter doesn't gripe and crash
    notmat_fn=notmats(i,:); % get name of .not.mat filename
    
    %want index of char *before* '.not.mat'
    if strcmp(purpose,'train')
        id = strfind(notmat_fn,'.not.mat') - 1;
        spect_fn = [notmat_fn(1:id) '.spect.to_train.mat'];
    elseif strcmp(purpose,'classify')
        id = strfind(notmat_fn,'.HDF.not.mat') - 1;
        spect_fn = [notmat_fn(1:id) '.spect.to_classify.mat'];
    end
  
    load(notmat_fn);
    disp(['loading: ' notmat_fn])
    onsets=onsets/1000; % convert back to s
    offsets=offsets/1000; % convert back to s
    
    %additional features--testing whether these improve SVM accuracy
    syl_durations = offsets-onsets;
    pre_durations = [0;syl_durations(1:end-1)];
    foll_durations = [syl_durations(2:end);0];
    gapdurs = onsets(2:end) - offsets(1:end-1);
    pre_gapdurs = [0;gapdurs];
    foll_gapdurs = [gapdurs;0];

    load(spect_fn)
    %extract features from each spectral slice for each syllable
    for syl_id=1:length(onsets)
        raw_syl = rawsyl_cell{syl_id};
        % get features from syl using Tachibana's function
        features_vec = makeAllFeatures(raw_syl,Fs);
        % add duration features to those returned by Tachibana func
        dur_features = [pre_durations(syl_id) foll_durations(syl_id) pre_gapdurs(syl_id) foll_gapdurs(syl_id)];
        %appending to a cell gives you a growing row "vector" (of cells)
        %Why this is important is explained in line 52 below
        features_mat{ct} = [features_vec dur_features];
        label_vec(ct) = double(labels(syl_id));
        ct = ct + 1;
    end
    
    %Why it's important that features_mat is a row vector:
    %If you run cell2mat on it, you get one really long row
    %when what you actually want is each cell to be a row, with each element in
    %each cell being a column. At least that's what I want here.
    features_mat = cell2mat(features_mat'); % <--Hence the transpose
    
    if strcmp(purpose,'train')
        extens_id = strfind(notmat_fn,'.not.mat')-1; % where to put new file extension
        ftr_fn = [notmat_fn(1:extens_id) '.svm_ftr.to_train.mat'];
    elseif strcmp(purpose,'classify')
        extens_id = strfind(notmat_fn,'.HDF.not.mat')-1; % where to put new file extension
        ftr_fn = [notmat_fn(1:extens_id) '.svm_ftr.to_classify.mat'];
    end
    disp(['Saving: ' ftr_fn])
    save(ftr_fn,'features_mat')
end
