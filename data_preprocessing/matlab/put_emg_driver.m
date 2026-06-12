% for all subjects

% Root folder that contains subject subfolders like subject_03, subject_13, ...
rootDir = '/Users/chrisdollo/Documents/Research/putEMG prime/csv_data';

% Where to save per-subject outputs
outDir = '/Users/chrisdollo/Documents/Research/putEMG prime/data/unprocessed gestures';
trainDir = fullfile(outDir, 'train');
evalDir  = fullfile(outDir, 'eval');

if ~exist(trainDir, 'dir'), mkdir(trainDir); end
if ~exist(evalDir,  'dir'), mkdir(evalDir);  end


% List subfolders
d = dir(rootDir);
isSub = [d.isdir] & ~ismember({d.name}, {'.','..'});
subDirs = d(isSub);



for i = 1:numel(subDirs)
    subjectFolderName = string(subDirs(i).name);           % e.g., "subject_03"
    subjectDir = fullfile(rootDir, subjectFolderName);

    % Only process folders matching the expected naming pattern
    % Extract subject number from "subject_03" -> "03"
    tok = regexp(subjectFolderName, "subject_(\d+)$", "tokens", "once");
    if isempty(tok)
        fprintf("Skipping (name not recognized): %s\n", subjectFolderName);
        continue;
    end
    subjNum = tok{1};  % e.g., "03" or "13"

    fprintf("Processing subject %s in %s\n", subjNum, subjectDir);

    % --- Group CSVs by recording type, sort by date ----------------------
    % Filename format: emg_gestures-03-repeats_long-2018-05-11-...csv
    csvFiles = dir(fullfile(subjectDir, '*.csv'));
    byType = struct();

    for j = 1:numel(csvFiles)
        fname = csvFiles(j).name;
        tok2 = regexp(fname, 'emg_gestures-\d+-([\w]+)-(\d{4}-\d{2}-\d{2})', 'tokens', 'once');
        if isempty(tok2), continue; end
        recType = tok2{1};   % 'repeats_long', 'repeats_short', 'sequential'
        dateStr = tok2{2};   % '2018-05-11'
        fp      = fullfile(csvFiles(j).folder, fname);

        if ~isfield(byType, recType)
            byType.(recType) = {};
        end
        byType.(recType){end+1} = struct('date', dateStr, 'path', fp);
    end

    % For each type: sort by ISO date (lexicographic = chronological),
    % first occurrence -> train, second -> eval
    trainFiles = {};
    evalFiles  = {};

    for t = fieldnames(byType)'
        recType = t{1};
        entries = byType.(recType);

        % Sort by date string
        dates = cellfun(@(e) e.date, entries, 'UniformOutput', false);
        [~, order] = sort(dates);
        entries = entries(order);

        if numel(entries) >= 1
            trainFiles{end+1} = entries{1}.path;  %#ok<AGROW>
        end
        if numel(entries) >= 2
            evalFiles{end+1} = entries{2}.path;   %#ok<AGROW>
        end
    end

    fprintf("  Train: %d file(s)  |  Eval: %d file(s)\n", numel(trainFiles), numel(evalFiles));

    % --- Process and save train ------------------------------------------
    trainTbl  = prime_process_file_list(trainFiles);
    combinedCell = table2cell(trainTbl);
    save(fullfile(trainDir, sprintf("emg_gestures_%s_NU_mat", subjNum)), "combinedCell", "-v7");

    % --- Process and save eval -------------------------------------------
    evalTbl   = prime_process_file_list(evalFiles);
    combinedCell = table2cell(evalTbl);
    save(fullfile(evalDir, sprintf("emg_gestures_%s_NU_mat", subjNum)), "combinedCell", "-v7");

    fprintf("  Saved train + eval for subject %s\n\n", subjNum);
end
