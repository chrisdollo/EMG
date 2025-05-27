daysOfExperiment = 2;
firstSubject = 3;
lastSubject = 7;
numberOfSubject = lastSubject - firstSubject;
numOfGestures = 7;


table = cell(numberOfSubject,1);
typeOfRepetitions = ["repeat_short","repeat_long"];
numTypesOfExperiment = length(typeOfRepetitions);

for subjectNumber = firstSubject:lastSubject

    if subjectNumber ~= 21 && subjectNumber ~= 28 && subjectNumber ~= 32 && subjectNumber ~= 37 && subjectNumber ~= 40 && subjectNumber ~= 41 && subjectNumber ~= 44 && subjectNumber ~= 52

        subjectsSpecificSetGestures = cell(0,numOfGestures); %
    
        for day = 1:daysOfExperiment
            for type = 1:numTypesOfExperiment
        
                    filePath = "E:\Chris\putEMG\sub_" + string(subjectNumber) + "/sub_" + string(subjectNumber) + "_" + typeOfRepetitions(type) + "_r" + string(day) + ".mat";
                    subjectGestures = getRepeatValues(filePath);
        
                    subjectsSpecificSetGestures = [subjectsSpecificSetGestures; subjectGestures];
            end
        end

    end
    
    if subjectNumber ~= 21 && subjectNumber ~= 28 && subjectNumber ~= 32 && subjectNumber ~= 37 && subjectNumber ~= 40 && subjectNumber ~= 41 && subjectNumber ~= 44 && subjectNumber ~= 52
        table{subjectNumber} = subjectsSpecificSetGestures
    end

end

sample=table(1:3,:);
save('subject3_to_7_data.mat', 'table', '-v7.3');
