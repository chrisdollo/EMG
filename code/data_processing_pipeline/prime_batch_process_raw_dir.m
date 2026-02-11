function combinedTbl = prime_batch_process_raw_dir(rawDir)
% prime_batch_process_raw_dir
%
% Runs your pipeline on every .csv in rawDir and combines results into
% one table of size (N x 7) with columns: G1,G2,G3,G6,G7,G8,G9.
% All gesture repetitions are uniformized to have the same length.
%
% Usage:
%   rawDir = "/path/to/Data";
%   combinedTbl = prime_batch_process_raw_dir(rawDir);

% Output:
%   combinedTbl: table (N x 7) where each cell is a uniformized gesture
%                repetition matrix (targetLength x 24).

    if nargin < 1 || strlength(string(rawDir)) == 0
        error('prime_batch_process_raw_dir:MissingInput', 'Provide rawDir (folder containing the .csv files).');
    end

    rawDir = string(rawDir);
    if ~isfolder(rawDir)
        error('prime_batch_process_raw_dir:NotAFolder', 'Folder not found: %s', rawDir);
    end

    files = dir(fullfile(rawDir, '*.csv'));


    if isempty(files)
        error('prime_batch_process_raw_dir:NoCSV', 'No .csv files found in: %s', rawDir);
    end

    combinedTbl = table();

    for i = 1:numel(files)
        fp = fullfile(files(i).folder, files(i).name);

        % 1) Load + preprocess raw file -> sensor matrix
        sensorMatrix  = prime_get_sensor_readings_1(fp);

        % 2) Split into action blocks
        action_blocks = prime_split_raw_files_in_blocks_2(sensorMatrix);

        % 3) Organize gestures across action blocks -> (Mi x 7) table
        fileTbl = prime_organize_action_blocks_in_gesture_3(action_blocks);

        % Ensure consistent headers (G1,G2,G3,G6,G7,G8,G9)
        expected = ["G1","G2","G3","G6","G7","G8","G9"];
        if ~isequal(string(fileTbl.Properties.VariableNames), expected)
            fileTbl = fileTbl(:, expected);
        end

        % Stack into one big (N x 7) table
        combinedTbl = [combinedTbl; fileTbl]; %#ok<AGROW>
    end
end
