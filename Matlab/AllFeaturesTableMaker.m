% numberOfParticipants = 43;
numberOfParticipants = 3;
numGestureRepPerPar = 28;
uniqueGestures = 7;
numSensors = 24;
numFeatures = 18;

% Create the cell content
k = cell(numSensors, numFeatures);
finalCellArray = repmat({k}, numberOfParticipants * numGestureRepPerPar, uniqueGestures);  % 84×7 cell array

% Save it to .mat
fullname = fullfile(pwd, 'gestureTable_clean.mat');
save(fullname, 'finalCellArray');