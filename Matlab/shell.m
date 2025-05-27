featuresNum = 18;
channelsNum = 24;
repetitionsNum = 28;
subjectNum = 43;
typesOfGesturesNum = 7;

tableRowNum = repetitionsNum * subjectNum;
tableColumnNum = typesOfGesturesNum;


% stores all repetitions for the same gesture
handGesture = cell(featuresNum,channelsNum);
[handGesture{:}] = deal(0);


% table that stores all hand gestures
table = cell(tableRowNum, tableColumnNum);
[table{:}] = deal(handGesture);

save("emptyTable.mat", "table")

