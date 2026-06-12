function combinedTbl = prime_process_file_list(filePaths)
% prime_process_file_list
%
% Like prime_batch_process_raw_dir but accepts a cell array of specific
% file paths instead of a whole directory. Used by put_emg_driver to
% process train and eval splits independently.
%
% Input:
%   filePaths  - cell array of full CSV file paths to process
%
% Output:
%   combinedTbl - table (N x 7) with columns G1,G2,G3,G6,G7,G8,G9
%                 Each cell is a variable-length gesture rep (M x 24)

    combinedTbl = table();
    expected    = ["G1","G2","G3","G6","G7","G8","G9"];

    for i = 1:numel(filePaths)
        fp = filePaths{i};

        sensorMatrix  = prime_get_sensor_readings_1(fp);
        action_blocks = prime_split_raw_files_in_blocks_2(sensorMatrix);
        fileTbl       = prime_organize_action_blocks_in_gesture_3(action_blocks);

        if ~isequal(string(fileTbl.Properties.VariableNames), expected)
            fileTbl = fileTbl(:, expected);
        end

        combinedTbl = [combinedTbl; fileTbl]; %#ok<AGROW>
    end
end
