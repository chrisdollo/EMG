% for all subjects

% Root folder that contains subject subfolders like emg_gestures-04, emg_gestures-13, ...
rootDir = "E:\Chris\EMG\Data\cvs_data_per_subject";

% Where to save per-subject outputs
outDir = "E:\Chris\EMG\Data\non_uniform_data_mat";
if ~exist(outDir, "dir")
    mkdir(outDir);
end


% List subfolders
d = dir(rootDir);
isSub = [d.isdir] & ~ismember({d.name}, {'.','..'});
subDirs = d(isSub);



for i = 1:numel(subDirs)
    subjectFolderName = string(subDirs(i).name);           % e.g., "emg_gestures-04"
    subjectDir = fullfile(rootDir, subjectFolderName);

    % Only process folders matching the expected naming pattern
    % Extract subject number from "emg_gestures-04" -> "04"
    tok = regexp(subjectFolderName, "subject(\d+)$", "tokens", "once");
    if isempty(tok)
        fprintf("Skipping (name not recognized): %s\n", subjectFolderName);
        continue;
    end
    subjNum = tok{1};  % e.g., "04" or "13"

    fprintf("Processing subject %s in %s\n", subjNum, subjectDir);

    % Run your existing pipeline on that subject directory
    combinedTbl = prime_batch_process_raw_dir(subjectDir);

    % Save per-subject
    combinedCell = table2cell(combinedTbl);

    outFile = fullfile(outDir, sprintf("emg_gestures_%s_combined_non_uniform.mat", subjNum));
    save(outFile, "combinedCell", "-v7");

    fprintf("Saved: %s\n", outFile);
end