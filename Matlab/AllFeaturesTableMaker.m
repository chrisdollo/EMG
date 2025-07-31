numberOfParticipants = 20;
numGestureRepPerPar = 28;
uniqueGestures = 7;
numSensors = 24;

sample_num = 1500


% Create the cell content
k = cell(sample_num, numSensors);
finalCellArray = repmat({k}, numberOfParticipants * numGestureRepPerPar, uniqueGestures);  % 84×7 cell array

% Save it to .mat
fullname = fullfile(pwd, 'gestureTable_clean_deep_learning_for_43_subject.mat');
save(fullname, 'finalCellArray');