import pandas as pd
from digital_processing import bp_filter, notch_filter, plot_signal
from feature_extraction import features_estimation
import numpy as np

# Load data
signal_path = 'data/emg_signal.xlsx'
emg_signal = [0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2,0,1,1,2,1,3,1,4,5,13,6,7,2]
emg_signal = np.array(emg_signal)
channel_name = 'Right Masseter'
sampling_frequency = 2e3
frame = 10
step = 5


# # Plot raw sEMG signal
# plot_signal(emg_signal, sampling_frequency, channel_name)

# Biomedical Signal Processing
emg_signal = emg_signal.reshape((emg_signal.size,))
filtered_signal = notch_filter(emg_signal, sampling_frequency,
                               True)
filtered_signal = bp_filter(filtered_signal, 10, 500,
                            sampling_frequency, True)

# EMG Feature Extraction
emg_features, features_names = features_estimation(filtered_signal, channel_name,
                                                   sampling_frequency, frame, step, False)


print(emg_features)
