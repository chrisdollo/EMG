"""
feature_extraction.py
---------------------
Extracts hand-crafted EMG features from a per-subject putEMG .mat file using a
sliding window and saves the result as a MATLAB .mat file.

Input .mat format
-----------------
  Key          : "combinedCell"
  Shape        : (N_repetitions, 7 gestures)
  Each cell    : ndarray (M_samples, 24 channels) — raw EMG at the original
                 sampling rate (default 5120 Hz) OR at any resampled rate.

Output .mat keys
----------------
  X                   : (n_windows, n_channels * n_features_per_channel)
                        = (n_windows, 24 * 8 = 192)
                        Column order: [feat0_ch0, feat1_ch0, ..., feat7_ch0,
                                       feat0_ch1, ..., feat7_ch23]
  y                   : (n_windows, 1)  gesture class label (0–6)
  feature_names       : cell array (1, 8)   names of the 8 features, in column order
  n_channels          : scalar              24
  n_features_per_ch   : scalar              8
  window_size         : scalar
  window_shift        : scalar
  sampling_rate       : scalar
  gesture_names       : cell array (1, 7)  ["G1","G2","G3","G6","G7","G8","G9"]

Usage
-----
  from feature_extraction import extract_features

  extract_features(
      input_filepath  = "path/to/emg_gestures_03_combined_non_uniform.mat",
      output_filepath = "path/to/features_subject_03.mat",
      window_size     = 250,
      window_shift    = 50,
      sampling_rate   = 5120.0,
  )

  # Load back in Python
  import scipy.io
  data = scipy.io.loadmat("path/to/features_subject_03.mat")
  X, y = data["X"], data["y"].squeeze()   # X: (N, 192)  y: (N,)
"""

import os
import re
import glob
import numpy as np
import scipy.io
from scipy.signal import welch


# ── Constants ─────────────────────────────────────────────────────────────────

GESTURE_NAMES   = ["G1", "G2", "G3", "G6", "G7", "G8", "G9"]
N_CHANNELS      = 24
MAT_KEY         = "combinedCell"

# Feature names (fixed order — do not reorder without updating _extract_window)
FEATURE_NAMES = [
    "MAV",   # Mean Absolute Value
    "RMS",   # Root Mean Square
    "WL",    # Waveform Length
    "ZC",    # Zero Crossings
    "SSC",   # Slope Sign Changes
    "VAR",   # Variance
    "MNF",   # Mean (spectral) Frequency
    "MDF",   # Median (spectral) Frequency
]
N_FEATURES_PER_CH = len(FEATURE_NAMES)   # 8


# ── Feature helpers ───────────────────────────────────────────────────────────

def _zero_crossings(x: np.ndarray, threshold: float = 1e-4) -> float:
    """
    Count the number of times the signal crosses zero.
    A crossing is only counted when the absolute difference between successive
    samples exceeds `threshold` (reduces false counts from noise near zero).
    """
    signs  = np.sign(x)
    diffs  = np.abs(np.diff(x))
    cross  = (signs[:-1] != signs[1:]) & (diffs >= threshold)
    return float(cross.sum())


def _slope_sign_changes(x: np.ndarray, threshold: float = 1e-4) -> float:
    """
    Count the number of times the slope (first difference) changes sign.
    Threshold applied to the product of successive differences to suppress noise.
    """
    d    = np.diff(x)
    ssc  = np.sum((d[:-1] * d[1:]) < -threshold)
    return float(ssc)


def _spectral_features(x: np.ndarray, fs: float):
    """
    Compute Mean Frequency (MNF) and Median Frequency (MDF) via Welch's PSD.

    Returns
    -------
    mnf : float — power-weighted mean frequency (Hz)
    mdf : float — frequency at which cumulative power reaches 50% (Hz)
    """
    nperseg = min(len(x), 128)          # guard against very short windows
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    total_power = psd.sum()
    if total_power < 1e-12:             # essentially flat signal
        return 0.0, 0.0

    mnf = float(np.sum(freqs * psd) / total_power)

    cumulative = np.cumsum(psd)
    half_power = total_power / 2.0
    idx        = np.searchsorted(cumulative, half_power)
    idx        = min(idx, len(freqs) - 1)
    mdf        = float(freqs[idx])

    return mnf, mdf


# ── Window feature extraction ─────────────────────────────────────────────────

def _extract_window(window: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract 8 features from every channel of a single window.

    Parameters
    ----------
    window : ndarray, shape (window_size, n_channels)
    fs     : float — sampling rate in Hz (used for spectral features)

    Returns
    -------
    features : ndarray, shape (n_channels * 8,)
               Column order: [feat0_ch0 .. feat7_ch0, feat0_ch1 .. feat7_ch23]
    """
    n_ch     = window.shape[1]
    features = np.empty(n_ch * N_FEATURES_PER_CH)

    for ch in range(n_ch):
        x   = window[:, ch].astype(np.float64)
        mav = float(np.mean(np.abs(x)))
        rms = float(np.sqrt(np.mean(x ** 2)))
        wl  = float(np.sum(np.abs(np.diff(x))))
        zc  = _zero_crossings(x)
        ssc = _slope_sign_changes(x)
        var = float(np.var(x))
        mnf, mdf = _spectral_features(x, fs)

        base = ch * N_FEATURES_PER_CH
        features[base:base + N_FEATURES_PER_CH] = [mav, rms, wl, zc, ssc, var, mnf, mdf]

    return features


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    input_filepath:  str,
    output_filepath: str,
    window_size:     int   = 250,
    window_shift:    int   = 50,
    sampling_rate:   float = 5120.0,
) -> None:
    """
    Load one per-subject .mat file, slide a window over every gesture
    repetition, extract 8 EMG features per channel, and save the result.

    Parameters
    ----------
    input_filepath  : str
        Path to the per-subject .mat file produced by the MATLAB pipeline.
        Must contain the key "combinedCell" with shape (N_reps, 7 gestures).
        Each cell is an ndarray of shape (M_samples, 24).

    output_filepath : str
        Where to write the output .mat file.
        Parent directories are created automatically.

    window_size     : int, default 250
        Number of samples per window.

    window_shift    : int, default 50
        Number of samples to advance the window on each step (stride).

    sampling_rate   : float, default 5120.0
        Sampling rate of the input signal in Hz.
        Used only for spectral features (MNF, MDF).
        If you are passing already-resampled data (e.g. 1500-sample segments
        from the preprocessing pipeline), set this to the effective rate.

    Output
    ------
    Saves a .mat file containing:
        X                 : (n_windows, 192)  float64 feature matrix
        y                 : (n_windows, 1)    double  gesture labels (0–6)
        feature_names     : (1, 8) cell       str     per-feature names
        n_channels        : scalar            24
        n_features_per_ch : scalar            8
        window_size       : scalar
        window_shift      : scalar
        sampling_rate     : scalar
        gesture_names     : (1, 7) cell       str     class names
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if not os.path.isfile(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")

    mat_data      = scipy.io.loadmat(input_filepath)
    combined_cell = mat_data[MAT_KEY]           # (N_reps, 7)
    n_reps, n_gestures = combined_cell.shape

    print(f"Loaded  : {os.path.basename(input_filepath)}")
    print(f"Shape   : {n_reps} repetitions × {n_gestures} gestures")
    print(f"Window  : size={window_size}, shift={window_shift}, fs={sampling_rate} Hz")
    print(f"Features: {N_FEATURES_PER_CH} per channel × {N_CHANNELS} channels = "
          f"{N_FEATURES_PER_CH * N_CHANNELS} total\n")

    # ── Slide & extract ───────────────────────────────────────────────────────
    all_features: list[np.ndarray] = []
    all_labels:   list[int]        = []

    for gesture_idx in range(n_gestures):
        for rep_idx in range(n_reps):
            cell = combined_cell[rep_idx, gesture_idx]

            if cell is None or not isinstance(cell, np.ndarray) or cell.size == 0:
                continue

            signal = np.array(cell, dtype=np.float64)   # (M_samples, 24)

            # Guard: skip if signal is shorter than one window
            if signal.shape[0] < window_size:
                continue

            n_samples = signal.shape[0]
            start     = 0

            while start + window_size <= n_samples:
                window   = signal[start : start + window_size, :]   # (win, 24)
                feat_vec = _extract_window(window, fs=sampling_rate) # (192,)
                all_features.append(feat_vec)
                all_labels.append(gesture_idx)
                start += window_shift

        print(f"  Gesture {GESTURE_NAMES[gesture_idx]} (class {gesture_idx}) — "
              f"accumulated {len(all_labels)} windows so far")

    # ── Assemble arrays ───────────────────────────────────────────────────────
    X = np.array(all_features, dtype=np.float64)    # (n_windows, 192)
    y = np.array(all_labels,   dtype=np.int64)       # (n_windows,)

    print(f"\nFeature matrix : X={X.shape}, y={y.shape}")
    print(f"Class distribution: { {GESTURE_NAMES[i]: int((y == i).sum()) for i in range(n_gestures)} }")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # String lists are stored as (1, N) object arrays so MATLAB reads them as
    # cell arrays of char vectors rather than a character matrix.
    feature_names_cell = np.empty((1, len(FEATURE_NAMES)), dtype=object)
    feature_names_cell[0, :] = FEATURE_NAMES

    gesture_names_cell = np.empty((1, len(GESTURE_NAMES)), dtype=object)
    gesture_names_cell[0, :] = GESTURE_NAMES

    scipy.io.savemat(
        output_filepath,
        {
            "X":                 X,
            "y":                 y.reshape(-1, 1).astype(np.float64),
            "feature_names":     feature_names_cell,
            "n_channels":        float(N_CHANNELS),
            "n_features_per_ch": float(N_FEATURES_PER_CH),
            "window_size":       float(window_size),
            "window_shift":      float(window_shift),
            "sampling_rate":     float(sampling_rate),
            "gesture_names":     gesture_names_cell,
        },
    )

    print(f"\nSaved   : {output_filepath}")


# ── Batch API ─────────────────────────────────────────────────────────────────

def batch_extract_features(
    input_dir:     str,
    output_dir:    str,
    window_size:   int   = 250,
    window_shift:  int   = 50,
    sampling_rate: float = 5120.0,
) -> None:
    """
    Run extract_features on every .mat file directly inside input_dir
    (no recursion into subdirectories).

    Output files are always re-created: any existing file at the target path
    is deleted before extraction starts.

    Parameters
    ----------
    input_dir     : str   — folder containing per-subject .mat files.
    output_dir    : str   — folder where output feature files are written
                            (created automatically if it does not exist).
    window_size   : int   — passed through to extract_features.
    window_shift  : int   — passed through to extract_features.
    sampling_rate : float — passed through to extract_features.

    Output naming
    -------------
    Input  : emg_gestures_03_combined_non_uniform.mat
    Output : features_subject_03.mat

    Subject ID is parsed with the regex ``emg_gestures_(\\w+?)_``.
    If the pattern does not match, falls back to ``features_<stem>.mat``.
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(input_dir, "*.mat")))
    if not mat_files:
        print(f"No .mat files found in: {input_dir}")
        return

    print(f"Found {len(mat_files)} .mat file(s) in:\n  {input_dir}\n")

    processed = 0
    for input_filepath in mat_files:
        filename = os.path.basename(input_filepath)

        match = re.search(r"emg_gestures_(\w+?)_", filename)
        if match:
            subject_id      = match.group(1)
            output_filename = f"features_subject_{subject_id}.mat"
        else:
            stem            = os.path.splitext(filename)[0]
            output_filename = f"features_{stem}.mat"
            print(f"  [warn] Could not parse subject ID from '{filename}'. "
                  f"Using '{output_filename}'.")

        output_filepath = os.path.join(output_dir, output_filename)

        if os.path.isfile(output_filepath):
            os.remove(output_filepath)

        print(f"─── {filename}  →  {output_filename}")
        extract_features(
            input_filepath  = input_filepath,
            output_filepath = output_filepath,
            window_size     = window_size,
            window_shift    = window_shift,
            sampling_rate   = sampling_rate,
        )
        processed += 1
        print()

    print(f"Batch complete : {processed}/{len(mat_files)} file(s) processed.")
    print(f"Outputs written: {output_dir}")
