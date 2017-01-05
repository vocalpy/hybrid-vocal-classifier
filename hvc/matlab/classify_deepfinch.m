function classify_deepfinch(purpose,varargin)
%classify_deepfinch
%Run this function once you have determined which machine learning
%algorithm you want the computer to use to classify syllables.
%
%It will ask you for the directory output by the train_deepfinch
%script, the classifier file you want to use, and the directories
%containing unlabeled song syllables you'd like classified.

output_dir = uigetdir('','Select the directory created by the train_deepfinch script');
cd([output_dir '\train']);
train_param_file = ls('*training_params*');
if isempty(train_param_file)
    error('Training parameter file not found in selected output directory')
else
    load(train_param_file,...
        'spect_params',...
        'quantify_deltas',...
        'duration',...
        'overlap',...
        'min_dur',...
        'min_int',...
        'threshold',...
        'sm_win')
end

clf_type = questdlg('Which classifier do you want to use?', ...
	'Pick one:', ...
	'support vector machine(SVM)',...
    'k-Nearest Neighbors(k-NN)',...
    'support vector machine(SVM)');
if strfind(clf_type,'k-NN');
    clf_type = 'knn';
elseif strfind(clf_type,'SVM');
    clf_type = 'svm';
end

while 1
    [clf_file,clf_path,index] = uigetfile('*.db.dat','Select replicate file to use');
    if index == 0;disp('No file selected.');pause(1);continue;end
    break
end

while 1
    prompt = 'Select folders that contain song for computer to automatically label';
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
    else
        disp('Please select only directories.')
    end
end
pause(2); % gives uipickfiles window time to close

cd(output_dir)
mkdir('classify')
output_dir = [output_dir '\classify'];

for i=1:numel(dirs)  
    disp(['Changing to directory ' dirs{i}])
    cd(dirs{i})  

    make_HDFnotmat(min_dur,min_int,threshold,sm_win)
    HDFnotmats = ls('*.HDF.not.mat');
    make_spect_files(HDFnotmats,'classify',duration,overlap)
    if strcmp(clf_type,'knn')
        disp('Making feature files to train k-NN classifiers')
        % feature files for k-NN
        
        make_feature_files_for_knn(HDFnotmats,'classify',quantify_deltas)
    elseif strcmp(clf_type,'svm')
        disp('Making feature files to train SVM classifiers')
        % feature files for SVM
        make_feature_files_for_svm(HDFnotmats,'classify')
    end 
end

cd(output_dir);
classify_dirs = char(dirs);
save('to_classify.mat','classify_dirs','clf_file','clf_type')