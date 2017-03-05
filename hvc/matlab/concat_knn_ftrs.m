function concat_knn_ftrs(output_dir)
%concat_knn_ftrs
%loads features for k-Nearest Neighbors from all ftr.mat files in a directory,
%concatenates them into one cell array, saves that array

if isunix
    !ls *cbin.not.mat > batchfile
else
    !dir /B *cbin.not.mat > batchfile
end
% elseif strcmp(purpose,'classify')
%     if isunix;
%         !ls *.HDF.not.mat > batchfile
%     else
%         !dir /B *HDF.not.mat > batchfile
%     end
% end

fid=fopen('batchfile','r');

CAT_labels = [];
CAT_syl_durations = [];
CAT_pre_durations = [];
CAT_foll_durations = [];
CAT_pre_gapdurs = [];
CAT_foll_gapdurs = [];
CAT_mn_amp_smooth_rect = [];
CAT_mn_amp_rms = [];
CAT_mn_spect_entropy = [];
CAT_mn_hi_lo_ratio = [];
CAT_delta_amp_smooth_rect = [];
CAT_delta_entropy = [];
CAT_delta_hi_lo_ratio = [];

counter = 1; % used for song ID vector
song_IDs_vec = [];
notmat_fnames = {};
%song ID vector makes it possible to determine where songs start/end
%so that even after concatenating all syllables into matrix
%(where row=syl and col=features)
%and removing unwanted samples (e.g., weird
%syllables, calls that shouldn't go into training set)
%can still identify what syllables belong to what song

% Loop throu all .not.mat files in current directory
while 1
    notmat_fn=fgetl(fid);
    % Break while loop when last filename is reached
    if (~ischar(notmat_fn));break;end

    %want index of char *before* '.not.mat'
    id = strfind(notmat_fn,'.not.mat') - 1;   
    ftr_fn = [notmat_fn(1:id) '.knn_ftr.to_train.mat'];

    load(notmat_fn,'labels')
    load(ftr_fn)
    
    CAT_labels = [CAT_labels labels];
    CAT_syl_durations = [CAT_syl_durations;feature_cell{2,1}];
    CAT_pre_durations = [CAT_pre_durations;feature_cell{2,2}];
    CAT_foll_durations = [CAT_foll_durations;feature_cell{2,3}];
    CAT_pre_gapdurs = [CAT_pre_gapdurs;feature_cell{2,4}];
    CAT_foll_gapdurs = [CAT_foll_gapdurs;feature_cell{2,5}];
    CAT_mn_amp_smooth_rect = [CAT_mn_amp_smooth_rect;feature_cell{2,6}];
    CAT_mn_amp_rms = [CAT_mn_amp_rms;feature_cell{2,7}];
    CAT_mn_spect_entropy = [CAT_mn_spect_entropy;feature_cell{2,8}];
    CAT_mn_hi_lo_ratio = [CAT_mn_hi_lo_ratio;feature_cell{2,9}];
    
    if size(feature_cell,2) > 9 % i.e. includes 'delta values'
        CAT_delta_amp_smooth_rect = [CAT_delta_amp_smooth_rect;feature_cell{2,10}];
        CAT_delta_entropy = [CAT_delta_entropy;feature_cell{2,11}];
        CAT_delta_hi_lo_ratio = [CAT_delta_hi_lo_ratio;feature_cell{2,12}];
    end
    
    song_ID_tmp = ones(numel(labels),1) * counter;
    song_IDs_vec = [song_IDs_vec;song_ID_tmp];
    counter = counter + 1;

    notmat_fname_tmp = cell(1, numel(labels));
    notmat_fname_tmp(:) = {notmat_fn};
    notmat_fnames = {notmat_fnames{:},notmat_fname_tmp{:}};
end % of while 1 loop
notmat_fnames = notmat_fnames';

NUM_FEATURES = 9;
NUM_FEATURES_PLUS_DELTAS = 12;

if size(feature_cell,2) > 9 % i.e. includes 'delta values'
    feature_cell = cell(2,NUM_FEATURES_PLUS_DELTAS);
else
    feature_cell = cell(2,NUM_FEATURES);
end

feature_cell{1,1} = 'syl_durations';feature_cell{2,1} = CAT_syl_durations;
feature_cell{1,2} = 'pre_durations';feature_cell{2,2} = CAT_pre_durations;
feature_cell{1,3} = 'foll_durations';feature_cell{2,3} = CAT_foll_durations;
feature_cell{1,4} = 'pre_gapdurs';feature_cell{2,4} = CAT_pre_gapdurs;
feature_cell{1,5} = 'foll_gapdurs';feature_cell{2,5} = CAT_foll_gapdurs;
feature_cell{1,6} = 'mn_amp_smooth_rect';feature_cell{2,6} = CAT_mn_amp_smooth_rect;
feature_cell{1,7} = 'mn_amp_rms';feature_cell{2,7} = CAT_mn_amp_rms;
feature_cell{1,8} = 'mn_spect_entropy';feature_cell{2,8} = CAT_mn_spect_entropy;
feature_cell{1,9} = 'mn_hi_lo_ratio';feature_cell{2,9} = CAT_mn_hi_lo_ratio;

if size(feature_cell,2) > 9 % i.e. includes 'delta values'
    feature_cell{1,10} = 'delta_amp_smooth_rect';feature_cell{2,10} = CAT_delta_amp_smooth_rect;
    feature_cell{1,11} = 'delta_entropy';feature_cell{2,11} = CAT_delta_entropy;
    feature_cell{1,12} = 'delta_hi_lo_ratio';feature_cell{2,12} = CAT_delta_hi_lo_ratio;
end

label_vec = CAT_labels;

% get filename of first cbin
dir = ls('*.cbin');
a_cbin = dir(1,:);
pat = '[a-z]{1,2}\d{1,3}[a-z]{1,2}\d{1,3}';
birdname = char(regexp(a_cbin,pat,'match')); % use regexp to extract birdname

curr_dir = pwd;
save_fname = [birdname '_knn_ftr_file_from_' curr_dir '_generated_' datestr(now,'mm-dd-yy_HH-MM')];
disp(['saving: ' save_fname]);
save_fname = [output_dir '\' save_fname];
save(save_fname,'feature_cell','label_vec','song_IDs_vec','notmat_fnames','dstr')