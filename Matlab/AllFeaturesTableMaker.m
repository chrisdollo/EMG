% numberOfParticipants = 43;
numberOfParticipants = 3;
numGestureRepPerPar = 28;
uniqueGestures = 7;
numSensors = 24;
% numFeatures = 18;

sample_num = 1500


% Create the cell content
k = cell(sample_num, numSensors);
finalCellArray = repmat({k}, numberOfParticipants * numGestureRepPerPar, uniqueGestures);  % 84×7 cell array

% Save it to .mat
fullname = fullfile(pwd, 'gestureTable_clean_deep_learning.mat');
save(fullname, 'finalCellArray');