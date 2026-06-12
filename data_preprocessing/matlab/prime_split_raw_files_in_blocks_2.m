function result = prime_split_raw_files_in_blocks_2(sensorMatrix)
    % -----------------------------------------------------------
    % Takes in a sensor matrix (N x 25) with inactive periods zeroed out,
    % and splits into separate gesture blocks.
    %
    % Input: sensorMatrix (N x 25) - output of getSensorReadings
    % Output: gestures - cell array, each cell = one action block (Mx24)
    % -----------------------------------------------------------
    

    % Load the table
    % T = "/Users/chrisdollo/Documents/Research/emg_gestures-03-repeats_short-2018-05-11-11-15-21-403/emg_gestures-03-repeats_short-2018-05-11-11-15-21-403.csv";
    % sensorMatrix = prime_get_sensor_readings_1(T);

    V = sensorMatrix;

    labels = V(:,25);
    valid = labels >= 0;

    edges = diff([0; valid; 0]);
    startIdx = find(edges == 1);
    endIdx   = find(edges == -1) - 1;

    action_blocks = cell(length(startIdx), 1);
    
    for i = 1:length(startIdx)
        action_blocks{i} = V(startIdx(i):endIdx(i), :);
    end

    result = action_blocks;
end
