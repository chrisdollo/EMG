"""
preprocessing.py
----------------
EMG signal preprocessing pipeline for the putEMG dataset.

Converts raw per-subject .mat files (output from the MATLAB pipeline) into
processed, model-ready .mat files.

Processing order (per gesture repetition):
    1. Bandpass filter 20–500 Hz at original 5120 Hz  (optional)
    2. Resample to TARGET_LENGTH samples
    3. Per-channel z-score normalization               (optional)

Usage
-----
    from preprocessing import batch_process_subjects, combine_processed_files

    batch_process_subjects(
        input_dir  = "path/to/raw/",
        output_dir = "path/to/processed/",
    )
    combine_processed_files(
        input_dir   = "path/to/processed/",
        output_path = "path/to/model_ready/combined.mat",
    )
"""

import os
import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
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
BANDPASS_HIGHCUT = 500
BANDPASS_ORDER   = 4


# ── Signal processing ─────────────────────────────────────────────────────────

def bandpass_filter(gesture_data, lowcut=BANDPASS_LOWCUT, highcut=BANDPASS_HIGHCUT,
                    fs=ORIGINAL_FS, order=BANDPASS_ORDER):
    """
    Apply a zero-phase Butterworth bandpass filter to each channel.
    Should be applied at the original sampling rate, BEFORE resampling.

    Parameters
    ----------
    gesture_data : ndarray, shape (samples, 24)
    lowcut       : float  — lower cutoff frequency in Hz
    highcut      : float  — upper cutoff frequency in Hz
    fs           : float  — sampling rate in Hz
    order        : int    — filter order

    Returns
    -------
    filtered : ndarray, shape (samples, 24)
    """
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, gesture_data, axis=0)


def resample_gesture(gesture_data, target_length=TARGET_LENGTH):
    """
    Resample a gesture repetition to a fixed number of samples.

    Parameters
    ----------
    gesture_data  : ndarray, shape (samples, 24)
    target_length : int — desired number of samples

    Returns
    -------
    resampled : ndarray, shape (target_length, 24)
    """
    if gesture_data.shape[0] == target_length:
        return gesture_data
    resampled = np.zeros((target_length, gesture_data.shape[1]))
    for ch in range(gesture_data.shape[1]):
        resampled[:, ch] = signal.resample(gesture_data[:, ch], target_length)
    return resampled


def zscore_normalize(gesture_data):
    """
    Apply per-channel z-score normalization (zero mean, unit variance over time).
    Should be applied AFTER resampling.

    Parameters
    ----------
    gesture_data : ndarray, shape (samples, 24)

    Returns
    -------
    normalized : ndarray, shape (samples, 24)
    """
    mean = gesture_data.mean(axis=0, keepdims=True)
    std  = gesture_data.std(axis=0,  keepdims=True)
    return (gesture_data - mean) / (std + 1e-8)


# ── Diagnostics ───────────────────────────────────────────────────────────────

def plot_signal_modification(original, modified, channel=0, fs_orig=ORIGINAL_FS,
                              fs_new=None, title="Signal Before vs After"):
    """
    Plot a single channel before and after a preprocessing step.

    Parameters
    ----------
    original  : ndarray, shape (samples_orig, 24)
    modified  : ndarray, shape (samples_new, 24)
    channel   : int — which channel (0–23) to plot
    fs_orig   : float — sampling rate of the original signal
    fs_new    : float — sampling rate of the modified signal (defaults to fs_orig)
    title     : str
    """
    if fs_new is None:
        fs_new = fs_orig

    t_orig = np.arange(original.shape[0]) / fs_orig
    t_new  = np.arange(modified.shape[0]) / fs_new

    plt.figure(figsize=(12, 4))
    plt.plot(t_orig, original[:, channel], color='steelblue', linewidth=1.2, label='Original')
    plt.plot(t_new,  modified[:, channel], color='tomato',    linewidth=1.0, label='Modified', alpha=0.85)
    plt.xlabel("Time (s)")
    plt.ylabel("EMG Amplitude")
    plt.title(f"{title} — Channel {channel + 1}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def display_length_statistics(gesture_table, gesture_names=GESTURE_NAMES):
    """
    Print min/max/mean/median sample counts for each gesture column.

    Parameters
    ----------
    gesture_table : ndarray, shape (N_reps, 7) — cell array of gesture data
    """
    print("\n=== Gesture Length Statistics ===")
    for col in range(gesture_table.shape[1]):
        lengths = [
            gesture_table[row, col].shape[0]
            for row in range(gesture_table.shape[0])
            if gesture_table[row, col] is not None
            and isinstance(gesture_table[row, col], np.ndarray)
            and gesture_table[row, col].size > 0
        ]
        if lengths:
            name = gesture_names[col] if col < len(gesture_names) else f"Col{col}"
            print(f"  {name}: min={min(lengths):6d}  max={max(lengths):6d}  "
                  f"mean={np.mean(lengths):8.1f}  median={np.median(lengths):6.0f}  (n={len(lengths)})")
    print("=================================")


# ── Core pipeline ─────────────────────────────────────────────────────────────

def preprocess_gesture(gesture_data, apply_bandpass=APPLY_BANDPASS,
                       apply_zscore=APPLY_ZSCORE, target_length=TARGET_LENGTH):
    """
    Apply the full preprocessing chain to a single gesture repetition.

    Order: bandpass filter → resample → z-score normalize

    Parameters
    ----------
    gesture_data    : ndarray, shape (samples, 24) — raw gesture at original fs
    apply_bandpass  : bool
    apply_zscore    : bool
    target_length   : int

    Returns
    -------
    processed : ndarray, shape (target_length, 24)
    """
    data = gesture_data.copy().astype(np.float64)

    if apply_bandpass:
        data = bandpass_filter(data)

    data = resample_gesture(data, target_length)

    if apply_zscore:
        data = zscore_normalize(data)

    return data


def process_subject_file(input_path, output_path=None,
                         apply_bandpass=APPLY_BANDPASS, apply_zscore=APPLY_ZSCORE,
                         target_length=TARGET_LENGTH, mat_key=MAT_KEY):
    """
    Load a per-subject raw .mat file, preprocess every gesture repetition,
    and optionally save the result.

    Parameters
    ----------
    input_path     : str — path to raw subject .mat file
    output_path    : str or None — where to save processed file
    apply_bandpass : bool
    apply_zscore   : bool
    target_length  : int
    mat_key        : str — cell array key in the .mat file

    Returns
    -------
    processed_table : ndarray, shape (N_reps, 7)
    """
    print(f"Loading: {os.path.basename(input_path)}")
    mat_data = scipy.io.loadmat(input_path)
    gesture_table = mat_data[mat_key]
    n_rows, n_cols = gesture_table.shape
    print(f"  Shape: {n_rows} repetitions × {n_cols} gestures")

    display_length_statistics(gesture_table)

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
    """
    Process all per-subject .mat files in input_dir and save results to output_dir.

    Parameters
    ----------
    input_dir      : str — directory with raw per-subject .mat files
    output_dir     : str — directory to write processed files
    apply_bandpass : bool
    apply_zscore   : bool
    target_length  : int
    mat_key        : str
    """
    mat_files = sorted([f for f in os.listdir(input_dir)
                        if f.endswith('.mat') and not os.path.isdir(os.path.join(input_dir, f))])

    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return

    print(f"Found {len(mat_files)} subject file(s)")
    print(f"Bandpass: {'ON' if apply_bandpass else 'OFF'} | "
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