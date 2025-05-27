daysOfExperiment = 2;
firstSubject = 3;
lastSubject = 54;
numberOfSubject = lastSubject - firstSubject;
numOfGestures = 7;


table = cell(numberOfSubject,1);
typeOfRepetitions = ["repeat_short","repeat_long"];
numTypesOfExperiment = length(typeOfRepetitions);

% for all subjects in the test
for subjectNumber = firstSubject:lastSubject

    % expect those suvbjects that are being excluded
    if subjectNumber ~= 21 && subjectNumber ~= 28 && subjectNumber ~= 32 && subjectNumber ~= 37 && subjectNumber ~= 40 && subjectNumber ~= 41 && subjectNumber ~= 44 && subjectNumber ~= 52

        subjectsSpecificSetGestures = cell(0,numOfGestures); %

        % for all experiment day
        for day = 1:daysOfExperiment

            % for short and long repetitions
            for type = 1:numTypesOfExperiment
                
        
                    filePath = "E:\Chris\putEMG Project\putEMG\sub_" + string(subjectNumber) + "\sub_" + string(subjectNumber) + "_" + typeOfRepetitions(type) + "_r" + string(day) + ".mat";
                    subjectGestures = getRepeatValues(filePath);

                    subjectsSpecificSetGestures = [subjectsSpecificSetGestures; subjectGestures];
            end
        end

        fileName = 'subject'+ string(subjectNumber) + '_gestures'

        save(fileName, 'subjectsSpecificSetGestures');        
    end
end
