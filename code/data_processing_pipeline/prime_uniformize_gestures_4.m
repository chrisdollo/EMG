function uniformTbl = prime_uniformize_gestures_4(combinedTbl, targetLength)
% prime_uniformize_gestures_4
%
% Uniformizes all gesture repetitions to have the same length using
% interpolation. This allows for consistent input dimensions for ML models.
%
% Usage:
%   uniformTbl = prime_uniformize_gestures_4(combinedTbl);
%   uniformTbl = prime_uniformize_gestures_4(combinedTbl, 200);
%
% Inputs:
%   combinedTbl   - Table (N x 7) from prime_organize_action_blocks_in_gesture_3
%                   Each cell contains a gesture repetition matrix (M x 24)
%   targetLength  - (optional) Target number of samples. If not provided,
%                   uses the median length across all gestures.
%
% Output:
%   uniformTbl    - Table (N x 7) where each cell contains resampled data
%                   of size (targetLength x 24)

    if nargin < 2 || isempty(targetLength)
        % Auto-calculate target length based on median
        targetLength = calculate_median_length(combinedTbl);
        fprintf('Auto-calculated target length: %d samples\n', targetLength);
    end

    % Display statistics before uniformization
    display_length_statistics(combinedTbl);

    % Create a copy of the table to avoid modifying original
    uniformTbl = combinedTbl;

    [numRows, numCols] = size(combinedTbl);

    % Process each cell in the table
    for row = 1:numRows
        for col = 1:numCols
            gestureData = combinedTbl{row, col};

            % Skip empty cells
            if isempty(gestureData)
                continue;
            end

            gestureData = combinedTbl{row, col};
            if iscell(gestureData)
                fprintf("Cell at (%d,%d) is a cell: size=%s, numel=%d\n", ...
                    row, col, mat2str(size(gestureData)), numel(gestureData));
            else
                fprintf("Cell at (%d,%d) is %s: size=%s\n", ...
                    row, col, class(gestureData), mat2str(size(gestureData)));
            end

            % Resample the gesture data
            uniformTbl{row, col} = resample_gesture(gestureData, targetLength);
        end
    end

    fprintf('Uniformization complete. All gestures now have %d samples.\n', targetLength);
end


function resampledData = resample_gesture(gestureData, targetLength)

    % --- unwrap nested cells until we get numeric ---
    while iscell(gestureData)
        if isempty(gestureData)
            resampledData = [];
            return;
        end
        if numel(gestureData) == 1
            gestureData = gestureData{1};
        else
            % If it's multiple blocks in a cell, concatenate them vertically
            % (common if you stored action-blocks per repetition)
            try
                gestureData = vertcat(gestureData{:});
            catch
                error("gestureData is a cell with multiple elements that can't be concatenated.");
            end
        end
    end

    % Now enforce numeric
    if ~isnumeric(gestureData)
        error("Expected numeric gesture matrix after unwrapping, got: %s", class(gestureData));
    end

    gestureData = double(gestureData);

    [origLength, numChannels] = size(gestureData);

    if origLength == targetLength
        resampledData = gestureData;
        return;
    end

    origTime = 1:origLength;
    targetTime = linspace(1, origLength, targetLength);

    resampledData = zeros(targetLength, numChannels);
    for ch = 1:numChannels
        resampledData(:, ch) = interp1(origTime, gestureData(:, ch), targetTime, 'pchip');
    end
end



function medianLen = calculate_median_length(tbl)
% Calculates the median length across all non-empty gesture repetitions

    lengths = [];
    [numRows, numCols] = size(tbl);

    for row = 1:numRows
        for col = 1:numCols
            gestureData = tbl{row, col};
            if ~isempty(gestureData)
                lengths(end+1) = size(gestureData, 1); %#ok<AGROW>
            end
        end
    end

    medianLen = round(median(lengths));
end


function display_length_statistics(tbl)
% Displays statistics about gesture lengths before uniformization

    [numRows, numCols] = size(tbl);
    gestureNames = tbl.Properties.VariableNames;

    fprintf('\n=== Gesture Length Statistics ===\n');

    for col = 1:numCols
        lengths = [];
        for row = 1:numRows
            gestureData = tbl{row, col};
            if ~isempty(gestureData)
                lengths(end+1) = size(gestureData, 1); %#ok<AGROW>
            end
        end

        if ~isempty(lengths)
            fprintf('%s: Min=%d, Max=%d, Mean=%.1f, Median=%d (n=%d)\n', ...
                    gestureNames{col}, min(lengths), max(lengths), ...
                    mean(lengths), median(lengths), length(lengths));
        end
    end
    fprintf('=================================\n\n');
end
