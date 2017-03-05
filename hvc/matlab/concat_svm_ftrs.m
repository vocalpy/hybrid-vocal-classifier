function concat_svm_ftrs(output_dir)
%concat_svm_ftrs
%
%loops across all .not.mat files in a directory and extracts features from
%syllables using code from Tachibana et al. 2014.

% % Create batchfile with all cbin.not.mat files
if isunix
    !ls *cbin.not.mat > batchfile
else
    !dir /B *cbin.not.mat > batchfile
end

%initialization
song_counter = 1; % used for song ID vector
song_IDs_vec = [];
%song ID vector makes it possible to determine where songs start/end even
%after concatenating and removing unwanted samples (e.g., weird syllables,
%calls that shouldn't go into training set)
%need to identify where songs start/stop for bootstrap analyses
notmat_fnames = {};
CAT_features_mat = []; % CAT: "to concatenate"
CAT_labels = [];

fid=fopen('batchfile','r');
% Loop throu all .not.mat files in current directory
while 1
    notmat_fn=fgetl(fid);
    % Break while loop when last filename is reached
    if (~ischar(notmat_fn));break;end

    %want index of char *before* '.not.mat'
    id = strfind(notmat_fn,'.not.mat') - 1;   
    ftr_fn = [notmat_fn(1:id) '.svm_ftr.to_train.mat'];
    
    load(notmat_fn,'labels')
    load(ftr_fn,'features_mat')
    
    CAT_labels = [CAT_labels labels];
        
    song_ID_tmp = ones(numel(labels),1) * song_counter;
    song_IDs_vec = [song_IDs_vec;song_ID_tmp];
    song_counter = song_counter + 1;
    
    %make cell array of .not.mat filename repeated n times
    %where n is the number of syllables in the .not.mat file
    %i.e., the number of samples.
    %This is used when generating .not.mat files from labels predicted by
    %classifier after training.
    notmat_fname_tmp = cell(1, numel(labels));
    notmat_fname_tmp(:) = {notmat_fn};
    notmat_fnames = {notmat_fnames{:},notmat_fname_tmp{:}};
    
    CAT_features_mat = [CAT_features_mat;features_mat];
    
end
notmat_fnames = notmat_fnames';
features_mat = CAT_features_mat; clear CAT_features_mat;% rename
label_vec = CAT_labels; clear CAT_labels;

% get filename of first cbin
dir = ls('*.cbin');
a_cbin = dir(1,:);
pat = '[a-z]{1,2}\d{1,3}[a-z]{1,2}\d{1,3}';
bird_name = char(regexp(a_cbin,pat,'match')); % use regexp to extract birdname

curr_dir = pwd;
save_fname = [bird_name '_svm_ftr_file_from_' curr_dir '_generated_' datestr(now,'mm-dd-yy_HH-MM')];
save_fname = [output_dir '\' save_fname];
save(save_fname,'features_mat','label_vec','song_IDs_vec','notmat_fnames','dstr')