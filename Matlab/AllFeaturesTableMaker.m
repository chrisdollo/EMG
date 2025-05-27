numberOfParticipants = 43;
numGestureRepPerPar = 28;
uniqueGestures = 7;
numSensors = 24;
numFeatures = 18;

k = cell(numSensors,numFeatures);
finalTable = cell2table(repmat({k}, numberOfParticipants*numGestureRepPerPar, uniqueGestures));


folder = pwd;
fullname = fullfile(folder,' gestureTable.mat');
save(fullname,"finalTable");
