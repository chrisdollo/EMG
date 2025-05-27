function result = getRepeatShortValues(x)


    filePath = x;
    if exist(filePath, 'file')
        Data = load(filePath);
    else
        error('File does not exist: %s', filePath);
    end

    fields = fieldnames(Data);
    
    
    action_blocks = Data.(fields{1});                                      % stores the every action blocks 
    num_action_blocks = length(action_blocks(:,1));                        % stores the number of action blocks in the file
    
    
    gestures_repetitions = [];                                             % stores the

    % For every action block
    for a = 1:num_action_blocks
        
        block = action_blocks{1};                                          % stores one action block
    
        length_of_rs = length(block(:,1));                                 % store the length of the registered signal (the amount of data points)    



        % in order to get the parts of the signals where we have active
        % gestures we will find thederivative of one of the registered signals

        timestamp = block(1:length_of_rs ,28);                             % store the timestamp
        sensor1 = block(1:length_of_rs ,2);                                % store the readings of the first sensor
        dydx = gradient(sensor1(:)) ./ gradient(timestamp(:));             % stores the derivatives if the reading of the forst sensor
    
        % this fuction allows us to get the start and end times of every
        % active gesture periods based on the derivative
        start_end_of_hg = start_and_end_times(dydx, length_of_rs);
        
        


        % split every hand gestures in an action block

        count =  1;
        num_of_gestures = (length(start_end_of_hg)/2);                     % stores the number of gestures recorded in the action block
    
        % store the spread(times accross each hand gesture)
        hand_gestures_EMG = cell(num_of_gestures, 1);                      % stores the spread of the repeated hand gestures
    
        % for each hand gesture
        for i = 1:num_of_gestures
    
            hand_gesture_EMG = [];                                         % stores the spread of a gesture
    
            % populate the spread of a gesture
            for k = start_end_of_hg(count): start_end_of_hg(count+1)
                hand_gesture_EMG(end+1) = k;
            end
    
            % add the spread of this gesture to the spread of the others
            hand_gestures_EMG{i} = hand_gesture_EMG;
    
            % update the count
            count = count + 2;
        end



        % stores the data for the different repetition of each hand gesture
        repetition = cell(num_of_gestures,1);                              % stores the values of the signals for each gestures in the action block
                                                                           % it is a table that contains n number of rows and one column
    
        for i = 1:num_of_gestures
            sensors = [];                                                  % stores the reading for the hand gesure 

            % collect the readings for each sensors and
            % stores the reading for the hand gesture
            for j = 2:25
                sensor = block(hand_gestures_EMG{i}(1):hand_gestures_EMG{i}(end),j); 
                sensors = [sensors, sensor];                                        
            end


            % adds the gesture recorded to the list of repetiotion
            repetition{i} = sensors;
        end
        
        gestures_repetitions = [gestures_repetitions, repetition];
    end

    result = gestures_repetitions;
    




    function y = start_and_end_times(derivative, length_of_signal)
        prev = 0;
        start_end_of_hg = [];
        for i = 1:length_of_signal
             if isnan(derivative(i)) & prev == 1
                 start_end_of_hg(end+1) = i;
                 prev = 0;
             elseif (~ isnan(derivative(i))) & prev == 0
                 start_end_of_hg(end+1) = i;
                 prev = 1;
             end
        end
        y = start_end_of_hg;
    end

end


