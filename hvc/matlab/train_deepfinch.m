function train_deepfinch(varargin)
%train_deepfinch
%This function extracts features from song files used to train classifiers.
%The features are concatenated into files that are then used by the Python
%scripts to compare two different algorithms for classification, k Nearest
%Neighbors (k-NN) and Support Vector Machine (SVM).
%
%The function prompts the user to select the directories containing hand-
%labeled song, then enters those directories and creates the feature files.
%Note that it will create a folder with the output in the working directory
%when the function is called (i.e. whatever directory you're in when you
%call it).
%
%Input arguments (must be entered in this order):
%   spect_params -- parameter used to create spectrograms of syllables. The
%       parameters are the window duration and window overlap; they should
%       be entered as a two-element vector. The default is [8 0.5], i.e.,
%       an 8 ms window with 50% overlap of windows
%   quantify_deltas -- logical flag, default is true. 
%
%Example:
%>> train_deepfinch
%(will run using the defaults)
%
%>> train_deepfinch([16 0],false,[10 5 1500 2])
%(will run with non-defaults, because you are a true rebel)
p = inputParser;
defaultSpectParams = [8 0.5];
defaultQuantifyDeltas = true;
addOptional(p,'spect_params',defaultSpectParams,...
    @(x) numel(x)==2);
addOptional(p,'quantify_deltas',defaultQuantifyDeltas,...
    @(x) islogical(x));
parse(p,varargin{:});

spect_params = p.Results.spect_params;
quantify_deltas = p.Results.quantify_deltas;

duration = spect_params(1); % milliseconds, window for FFT
overlap = spect_params(2); % i.e., 50%, half of window overlaps for FFT

while 1
    prompt = 'Select folders that contain labeled song to generate feature files';
    dirs = uipickfiles('Prompt',prompt);
   
    is_a_dir = zeros(numel(dirs),1);
    for i=1:numel(dirs)
        is_a_dir(i) = isdir(dirs{i});
    end
    
    if sum(is_a_dir) == numel(dirs)
        ONLY_DIRS_SELECTED_DIRS = true;
    end
    
    if ONLY_DIRS_SELECTED_DIRS == true
        break
    end
    
end

pause(2) % pause gives uipickfiles window time to close
% without pause it stays open and maintains focus, so you can't Ctrl-C
% to quit the main loop of this script

home_dir = pwd;
nowstr = datestr(now,'mm-dd-yy_HH-MM');
foldername = ['deepfinch_train_results_' nowstr];
mkdir(foldername)
cd(foldername)
mkdir('train')
cd('train')
output_dir = pwd;

disp('Determining syllable segmenting parameters.')
allnotmats = {};
min_durs = [];
min_ints = [];
thresholds = [];
sm_wins = [];
for i=1:numel(dirs)
    cd(dirs{i})
    notmats = ls('*.not.mat');
    if isempty(notmats);cd(home_dir);continue;end
    for j=1:size(notmats,1)
        load(notmats(j,:),'min_dur','min_int','threshold','sm_win')
        allnotmats{end+1} = notmats(j,:);
        min_durs = [min_durs;min_dur];
        min_ints = [min_ints;min_int];
        thresholds = [thresholds;threshold];
        sm_wins = [sm_wins;sm_win];
    end
end
all_segment_params = [min_durs,min_ints,thresholds,sm_wins];
[uniq_seg_params,~,inds] = unique(all_segment_params,'rows');
if size(uniq_seg_params,1) > 1
    uniq_inds = unique(inds);
    counts = histc(inds,uniq_inds);
    [~,max_ind] = max(counts);
    disp('More than one unique set of segmenting parameters was detected:')
    for k = 1:size(uniq_seg_params,1)
        disp([num2str(k) ': [' num2str(uniq_seg_params(k,:)) '], appears ' num2str(counts(k)) ' times.'])
    end
    prompt = sprintf(['Enter the number of the segmenting parameter set you want to use.\n',...
        'Note that these will be used to segment unlabeled song for syllable classifcation.\n',...
        'Hit enter to use the most common set: ' num2str(max_ind)]);
    reply = input(prompt,'s');
    if isempty(reply)
        reply = max_ind;
    end
    uniq_seg_params = uniq_seg_params(reply,:);
    min_dur = uniq_seg_params(1);
    min_int = uniq_seg_params(2);
    threshold = uniq_seg_params(3);
    sm_win = uniq_seg_params(4);
else
    min_dur = uniq_seg_params(1);
    min_int = uniq_seg_params(2);
    threshold = uniq_seg_params(3);
    sm_win = uniq_seg_params(4);
end
segstr = sprintf(['min_dur = ' num2str(min_dur) ...
    '\nmin_int = ' num2str(min_int) ...
    '\nthreshold = ' num2str(threshold) ...
    '\nsmoothing window = ' num2str(sm_win)]);
disp('Using the following segmenting parameters:')
disp(segstr)
    
for i=1:numel(dirs)
    disp(['Changing to directory ' dirs{i}])
    cd(dirs{i})
    notmats = ls('*.not.mat');
    if isempty(notmats)
        disp(['Did not find .not.mat files in ' dirs{i}])
        continue
    end
    
    ftr_files = ls('*.ftr.to_train.mat');
    num_ftr_files = size(ftr_files,1);
    num_notmats = size(notmats,1);
%     if num_ftr_files == num_notmats
%         choice = questdlg('This directory already has feature files for all songs', ...
%         'Select an option:', ...
%         'Ignore this directory',...
%         'Keep feature files, re-use for training (default)',...
%         'Re-make feature files',...
%         'Cancel',...
%         'Keep feature files, re-use for training (default)');
%         switch choice
%             case 'Ignore this directory'
%                 continue
%             case 'Keep feature files, re-use for training (default)'
%         end
%     end
    if (num_ftr_files > 0) && (num_notmats > num_ftr_files)
        title = 'Found .not.mat files without .ftr.to_train.mat files';
        question = ['Found .not.mat files without .ftr.to_train.mat files.'...
            ' What would you like to do?'];
        opt1 = 'Re-make all ftr files';
        opt2 = 'Make only for .not.mats w/out ftr files';
        button = questdlg(question,title,opt1,opt2,opt2);
        %if user selected 'opt1' can just use current notmats list
        %but if they selected 2, need to remove notmats for which there are
        %already ftr files
        if strcmp(button,opt2)
            new_notmats = {};
            ftr_files_cell = cellstr(ftr_files);
            for j=1:num_notmats
                extens_id = strfind(notmats(j,:),'.cbin.not.mat');
                curr_notmat_stub = notmats(j,extens_id-1); % 'stub' = w/out extension
                has_ftr_file = strncmp(curr_notmat_stub,...
                                        ftr_files_cell,...
                                        length(curr_notmat_stub));
                has_ftr_file = sum(has_ftr_file);
                if ~has_ftr_file
                    new_notmats{end+1} = notmats(j,:);
                end
            end
            new_notmats = char(new_notmats); %convert from cell to char
            notmats = new_notmats; % replace old notmats variable with new
        end
    end

    make_spect_files(notmats,'train',duration,overlap)
    
    % feature files for k-NN
    disp('Making feature files to train k-NN classifiers')
    make_feature_files_for_knn(notmats,'train',quantify_deltas)
    concat_knn_ftrs(output_dir)
    
    disp('Making feature files to train SVM classifiers')
    % feature files for SVM
    make_feature_files_for_svm(notmats,'train')
    concat_svm_ftrs(output_dir)
    
end

% go to output dir and concatenate features files across days
cd(output_dir) 
%concatenate all feature files
nowstr = datestr(now,'mm-dd-yy_HH-MM');
% for knn
knn_ftr_files = ls('*knn_ftr*');
% use regexp to extract birdname
pattern = '[a-z]{1,2}\d{1,3}[a-z]{1,2}\d{1,3}';
birdname = char(regexp(knn_ftr_files(1,:),pattern,'match')); 
all_ftr_cells = cell(2,12);
all_labels_vec = [];
all_song_IDs_vec = [];
all_dstrs = []; %
for i = 1:size(knn_ftr_files,1)
    load(knn_ftr_files(i,:));
    all_ftr_cells(1,:) = feature_cell(1,:);
    for j = 1:size(feature_cell,2)
        all_ftr_cells{2,j} = [all_ftr_cells{2,j}; feature_cell{2,j}];
    end
    all_labels_vec = [all_labels_vec;label_vec'];
    all_song_IDs_vec = [all_song_IDs_vec;song_IDs_vec];
    all_dstrs = [all_dstrs;dstr];
end

feature_cell = all_ftr_cells;
label_vec = all_labels_vec;
song_IDs_vec = all_song_IDs_vec;
save([birdname '_concat_knn_ftr_cell_generated_' nowstr '.mat'],'feature_cell','label_vec','song_IDs_vec','all_dstrs')

% for svm
svm_ftr_files = ls('*svm_ftr*');
all_ftrs_mat = [];
all_labels_vec = [];
all_song_IDs_vec = [];
all_dstrs = []; %
for i = 1:size(svm_ftr_files,1)
    load(svm_ftr_files(i,:));
    all_ftrs_mat = [all_ftrs_mat;features_mat];
    all_labels_vec = [all_labels_vec;label_vec'];
    all_song_IDs_vec = [all_song_IDs_vec;song_IDs_vec];
    all_dstrs = [all_dstrs;dstr];
end
features_mat = all_ftrs_mat;
label_vec = all_labels_vec;
song_IDs_vec = all_song_IDs_vec;
save([birdname '_concat_svm_ftr_cell_generated_' nowstr '.mat'],'features_mat','label_vec','song_IDs_vec','all_dstrs')

%lastly save segmenting params used in train_deepfinch script
save([birdname '_training_params.mat'],...
    'spect_params',...
    'quantify_deltas',...
    'duration',...
    'overlap',...
    'min_dur',...
    'min_int',...
    'threshold',...
    'sm_win')
    