function [resultTbl, blockRuns] = prime_organize_action_blocks_in_gesture_3(action_blocks)

    % T = "/Users/chrisdollo/Documents/Research/emg_gestures-03-repeats_short-2018-05-11-11-15-21-403/emg_gestures-03-repeats_short-2018-05-11-11-15-21-403.csv";
    % T = "/Users/chrisdollo/Documents/Research/emg_gestures-03-repeats_short-2018-06-14-12-43-13-875/emg_gestures-03-repeats_short-2018-06-14-12-43-13-875.csv";
    % T = "/Users/chrisdollo/Documents/Research/emg_gestures-03-repeats_long-2018-06-14-12-32-53-659/emg_gestures-03-repeats_long-2018-06-14-12-32-53-659.csv";
    % sensorMatrix  = prime_get_sensor_readings_1(T);
    % action_blocks = prime_split_raw_files_in_blocks_2(sensorMatrix);

    numBlocks = length(action_blocks);

    % Only these gestures exist in your dataset
    gestureSet = [1 2 3 6 7 8 9];
    numGestures = numel(gestureSet);

    % Map gestureID -> column index in gestureSet
    idToCol = containers.Map(gestureSet, 1:numGestures);

    % Temporary growing lists per gesture-column
    perGestureLists = cell(1, numGestures);
    for c = 1:numGestures
        perGestureLists{c} = {};
    end

    blockRuns = cell(numBlocks, 1);

    for b = 1:numBlocks
        V = action_blocks{b};
        labels = V(:,25);

        % only keep 1..9 gesture regions (skips 0 idle and -1 ignore)
        valid = (labels >= 1 & labels <= 9);

        edges = diff([0; valid; 0]);
        startIdx = find(edges == 1);
        endIdx   = find(edges == -1) - 1;

        runs = struct('gestureID', {}, 'startIdx', {}, 'endIdx', {}, 'data', {});

        for r = 1:length(startIdx)
            s = startIdx(r);
            e = endIdx(r);

            gid = mode(labels(s:e));      % robust if any tiny label noise
            if ~isKey(idToCol, gid)
                continue;                 % skip gestures not in [1 2 3 6 7 8 9]
            end

            runData = V(s:e, 1:24);       % EMG only

            runs(end+1).gestureID = gid; %#ok<AGROW>
            runs(end).startIdx = s;
            runs(end).endIdx   = e;
            runs(end).data     = runData;

            c = idToCol(gid);
            perGestureLists{c}{end+1} = runData;
        end

        blockRuns{b} = runs;
    end

    % Convert into rectangular cell matrix (maxReps x numGestures)
    counts  = cellfun(@numel, perGestureLists);
    maxReps = max(counts);

    gesturesMat = cell(maxReps, numGestures);
    for c = 1:numGestures
        reps = perGestureLists{c};
        for r = 1:numel(reps)
            gesturesMat{r, c} = reps{r};
        end
    end

    % Make table with consistent headers: G1, G2, G3, G6, G7, G8, G9
    varNames = "G" + string(gestureSet);
    resultTbl = cell2table(gesturesMat, "VariableNames", varNames);
end
