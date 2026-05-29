import os
import glob
import scipy.io
import numpy as np
from scipy import signal
from scipy.io import savemat


# ── Default constants ─────────────────────────────────────────────────────────

APPLY_BANDPASS   = True
APPLY_ZSCORE     = True

ORIGINAL_FS      = 5120
TARGET_LENGTH    = 1500
N_CHANNELS       = 24
GESTURE_NAMES    = ["G1", "G2", "G3", "G6", "G7", "G8", "G9"]
MAT_KEY          = "combinedCell"

BANDPASS_LOWCUT  = 20
BANDPASS_HIGHCUT = 700
BANDPASS_ORDER   = 4

NOTCH_FREQS      = [30, 50, 60, 90, 150]   # Hz — mains + equipment interference
NOTCH_Q          = 30                        # bandwidth = freq / Q


# ── Signal processing ─────────────────────────────────────────────────────────

# Apply before resampling — filter at original 5120 Hz
def bandpass_filter(gesture_data, lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT,
                    fs=ORIGINAL_FS, order=BANDPASS_ORDER):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, gesture_data, axis=0)


def notch_filter(gesture_data, freqs=NOTCH_FREQS, Q=NOTCH_Q, fs=ORIGINAL_FS):
    data = gesture_data.copy()
    for freq in freqs:
        b, a = signal.iirnotch(freq, Q, fs)
        data = signal.filtfilt(b, a, data, axis=0)
    return data


def resample_gesture(gesture_data, target_length=TARGET_LENGTH):
    if gesture_data.shape[0] == target_length:
        return gesture_data
    resampled = np.zeros((target_length, gesture_data.shape[1]))
    for ch in range(gesture_data.shape[1]):
        resampled[:, ch] = signal.resample(gesture_data[:, ch], target_length)
    return resampled


# Apply after resampling
def zscore_normalize(gesture_data):
    mean = gesture_data.mean(axis=0, keepdims=True)
    std  = gesture_data.std(axis=0,  keepdims=True)
    return (gesture_data - mean) / (std + 1e-8)


# ── Core pipeline ─────────────────────────────────────────────────────────────

# Order: bandpass → notch → resample → z-score (filtering must precede resampling)
def preprocess_gesture(gesture_data, apply_bandpass=APPLY_BANDPASS,
                       apply_zscore=APPLY_ZSCORE, target_length=TARGET_LENGTH):
    data = gesture_data.copy().astype(np.float64)

    if apply_bandpass:
        data = bandpass_filter(data)
        data = notch_filter(data)

    data = resample_gesture(data, target_length)

    if apply_zscore:
        data = zscore_normalize(data)

    return data


def process_subject_file(input_path, output_path=None,
                         apply_bandpass=APPLY_BANDPASS, apply_zscore=APPLY_ZSCORE,
                         target_length=TARGET_LENGTH, mat_key=MAT_KEY):
    print(f"Loading: {os.path.basename(input_path)}")
    mat_data = scipy.io.loadmat(input_path)
    gesture_table = mat_data[mat_key]
    n_rows, n_cols = gesture_table.shape
    print(f"  Shape: {n_rows} repetitions × {n_cols} gestures")

    processed_table = np.empty_like(gesture_table, dtype=object)
    count = 0

    for row in range(n_rows):
        for col in range(n_cols):
            cell = gesture_table[row, col]
            if cell is None or not isinstance(cell, np.ndarray) or cell.size == 0:
                processed_table[row, col] = None
                continue
            processed_table[row, col] = preprocess_gesture(
                cell, apply_bandpass=apply_bandpass,
                apply_zscore=apply_zscore, target_length=target_length
            )
            count += 1

    print(f"  Processed {count} repetitions → shape ({target_length} × {N_CHANNELS}) each")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        savemat(output_path, {mat_key: processed_table})
        print(f"  Saved → {output_path}")

    return processed_table


def batch_process_subjects(input_dir, output_dir,
                           apply_bandpass=APPLY_BANDPASS, apply_zscore=APPLY_ZSCORE,
                           target_length=TARGET_LENGTH, mat_key=MAT_KEY):
    mat_files = sorted([f for f in os.listdir(input_dir)
                        if f.endswith('.mat') and not os.path.isdir(os.path.join(input_dir, f))])

    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return

    print(f"Found {len(mat_files)} subject file(s)")
    print(f"Bandpass 20–700 Hz + Notch {NOTCH_FREQS} Hz: {'ON' if apply_bandpass else 'OFF'} | "
          f"Z-score: {'ON' if apply_zscore else 'OFF'} | "
          f"Target length: {target_length}\n")

    os.makedirs(output_dir, exist_ok=True)
    succeeded, failed = [], []

    for i, filename in enumerate(mat_files, 1):
        sub_num = (filename.split("_"))[2]
        print(f"[{i}/{len(mat_files)}] {filename}")
        input_path  = os.path.join(input_dir,  filename)
        output_path = os.path.join(output_dir, f"emg_gestures_{sub_num}_U.mat")
        try:
            process_subject_file(
                input_path, output_path,
                apply_bandpass=apply_bandpass, apply_zscore=apply_zscore,
                target_length=target_length, mat_key=mat_key
            )
            succeeded.append(filename)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(filename)
        print()

    print(f"Done. {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed files: {failed}")