function result = prime_get_sensor_readings_1(x)
    % -----------------------------------------------------------
    % Takes in raw data files and returns a table containing the reading of
    % the 24 sensors with the inactive periods zeroed out.
    % 
    % Input: cvs file path
    % Output: Matrix (1746944, 24)
    % -----------------------------------------------------------

    % T = readtable('/Users/chrisdollo/Documents/coding_projects/putEMG_research_project/dataForEMG/Data/emg_gestures-03-repeats_long-2018-05-11-11-05-00-695 (1).csv');
    T = readtable(x);
    k = size(T);
    headers = T.Properties.VariableNames;
    Names = headers(2:25);
        
    traj_1 = T{:,26};
    new_T = T{:, 2:25}; 
    
    
    % We nullify all the sensor reading for when the subject was allowed to relax
    mask = (traj_1 == -1);
    new_T(mask,:)= 0;
    new_T = [new_T, traj_1];


    result = new_T;
end

