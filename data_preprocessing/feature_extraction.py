"""
feature_extraction.py
---------------------
Extracts hand-crafted EMG features from a per-subject putEMG .mat file using a
sliding window and saves the result.

Three output modes are supported via the `mode` parameter:

  "flat_window"  (default)
      Each window is an independent sample. One row per window.
      X shape : (n_windows, 192)   y shape: (n_windows, 1)
      Output  : .mat file
      Use for : SVM, XGBoost, MLP — real-time or offline with majority vote.

  "flat_rep"
      All windows from one gesture rep are concatenated into a single flat vector.
      One row per rep.
      X shape : (n_reps, 4992)     y shape: (n_reps, 1)
      Output  : .mat file
      Use for : SVM, XGBoost, MLP — offline only.

  "sequence"
      All windows from one gesture rep are stacked in temporal order.
      One 2-D array per rep.
      X shape : (n_reps, 26, 192)  y shape: (n_reps,)
      Output  : .npz file
      Use for : LSTM, GRU, Transformer, TCN — offline only.

Input .mat format
-----------------
  Key          : "combinedCell"
  Shape        : (N_repetitions, 7 gestures)
  Each cell    : ndarray (M_samples, 24 channels)

Usage
-----
  from feature_extraction import extract_features, batch_extract_features

  # Single file
  extract_features(
      input_filepath  = "path/to/emg_gestures_03_combined_non_uniform.mat",
      output_filepath = "path/to/features_subject_03_flat_window.mat",
      mode            = "flat_window",   # or "flat_rep" / "sequence"
      window_size     = 250,
      window_shift    = 50,
      sampling_rate   = 5120.0,
  )

  # Whole folder
  batch_extract_features(
      input_dir     = "path/to/raw_mat_files/",
      output_dir    = "path/to/feature_files/",
      mode          = "flat_window",
      window_size   = 250,
      window_shift  = 50,
      sampling_rate = 5120.0,
  )
"""

import os
import re
import glob
import numpy as np
import scipy.io
from scipy.signal import welch


# ── Constants ─────────────────────────────────────────────────────────────────

GESTURE_NAMES     = ["G1", "G2", "G3", "G6", "G7", "G8", "G9"]
N_CHANNELS        = 24
MAT_KEY           = "combinedCell"
VALID_MODES       = ("flat_window", "flat_rep", "sequence")

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
N_FEATURES_TOTAL  = N_FEATURES_PER_CH * N_CHANNELS  # 192


# ── Feature helpers ───────────────────────────────────────────────────────────

def _zero_crossings(x: np.ndarray, threshold: float = 1e-4) -> float:
    signs  = np.sign(x)
    diffs  = np.abs(np.diff(x))
    cross  = (signs[:-1] != signs[1:]) & (diffs >= threshold)
    return float(cross.sum())


def _slope_sign_changes(x: np.ndarray, threshold: float = 1e-4) -> float:
    d   = np.diff(x)
    ssc = np.sum((d[:-1] * d[1:]) < -threshold)
    return float(ssc)


def _spectral_features(x: np.ndarray, fs: float):
    nperseg = min(len(x), 128)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)

    total_power = psd.sum()
    if total_power < 1e-12:
        return 0.0, 0.0

    mnf = float(np.sum(freqs * psd) / total_power)

    cumulative = np.cumsum(psd)
    idx        = np.searchsorted(cumulative, total_power / 2.0)
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
    fs     : float — sampling rate in Hz

    Returns
    -------
    features : ndarray, shape (n_channels * 8,)  = (192,)
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


# ── Sliding window over one rep ───────────────────────────────────────────────

def _slide_rep(
    signal:       np.ndarray,
    window_size:  int,
    window_shift: int,
    fs:           float,
) -> list[np.ndarray]:
    """
    Slide a window over one gesture rep and extract features from each window.

    Parameters
    ----------
    signal       : ndarray, shape (M_samples, 24)
    window_size  : int
    window_shift : int
    fs           : float

    Returns
    -------
    windows : list of ndarray, each shape (192,)
              One entry per window. Empty list if signal is shorter than window_size.
    """
    n_samples = signal.shape[0]
    windows   = []
    start     = 0

    while start + window_size <= n_samples:
        window = signal[start : start + window_size, :]
        windows.append(_extract_window(window, fs=fs))
        start += window_shift

    return windows


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    input_filepath:  str,
    output_filepath: str,
    mode:            str   = "flat_window",
    window_size:     int   = 250,
    window_shift:    int   = 50,
    sampling_rate:   float = 5120.0,
) -> None:
    """
    Load one per-subject .mat file, slide a window over every gesture repetition,
    extract 8 EMG features per channel, and save the result.

    Parameters
    ----------
    input_filepath  : str
        Path to per-subject .mat file with key "combinedCell" shape (N_reps, 7).
        Each cell: ndarray (M_samples, 24).

    output_filepath : str
        Destination path. Use .mat for flat_window / flat_rep, .npz for sequence.
        Parent directories are created automatically.

    mode            : str, default "flat_window"
        "flat_window" — one row per window,    X: (n_windows, 192)
        "flat_rep"    — one row per rep,        X: (n_reps, 4992)
        "sequence"    — one matrix per rep,     X: (n_reps, n_windows_per_rep, 192)

    window_size     : int, default 250
    window_shift    : int, default 50
    sampling_rate   : float, default 5120.0
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {VALID_MODES}.")

    if not os.path.isfile(input_filepath):
        raise FileNotFoundError(f"Input file not found: {input_filepath}")

    mat_data      = scipy.io.loadmat(input_filepath)
    combined_cell = mat_data[MAT_KEY]
    n_reps, n_gestures = combined_cell.shape

    print(f"Loaded  : {os.path.basename(input_filepath)}")
    print(f"Shape   : {n_reps} repetitions × {n_gestures} gestures")
    print(f"Mode    : {mode}")
    print(f"Window  : size={window_size}, shift={window_shift}, fs={sampling_rate} Hz")
    print(f"Features: {N_FEATURES_PER_CH} per channel × {N_CHANNELS} channels = "
          f"{N_FEATURES_TOTAL} total\n")

    # ── Collect windows per rep ───────────────────────────────────────────────
    # all_reps[gesture_idx][rep_idx] = list of (192,) arrays
    all_reps   = []
    all_labels = []

    # Track the window count per rep so we can guard flat_rep/sequence assembly
    window_counts = []

    for gesture_idx in range(n_gestures):
        for rep_idx in range(n_reps):
            cell = combined_cell[rep_idx, gesture_idx]

            if cell is None or not isinstance(cell, np.ndarray) or cell.size == 0:
                continue

            signal = np.array(cell, dtype=np.float64)

            if signal.shape[0] < window_size:
                print(f"  [skip] gesture {GESTURE_NAMES[gesture_idx]} rep {rep_idx}: "
                      f"only {signal.shape[0]} samples, need {window_size}.")
                continue

            windows = _slide_rep(signal, window_size, window_shift, fs=sampling_rate)

            if len(windows) == 0:
                continue

            all_reps.append(windows)
            all_labels.append(gesture_idx)
            window_counts.append(len(windows))

        print(f"  Gesture {GESTURE_NAMES[gesture_idx]} (class {gesture_idx}) — "
              f"accumulated {len(all_labels)} reps so far")

    print()

    # ── Assemble arrays by mode ───────────────────────────────────────────────

    if mode == "flat_window":
        X_list, y_list = [], []
        for windows, label in zip(all_reps, all_labels):
            for feat_vec in windows:
                X_list.append(feat_vec)
                y_list.append(label)

        X = np.array(X_list, dtype=np.float64)     # (n_windows, 192)
        y = np.array(y_list, dtype=np.int64)        # (n_windows,)

    elif mode == "flat_rep":
        # Guard: all reps must have the same number of windows
        unique_counts = set(window_counts)
        if len(unique_counts) > 1:
            count_summary = {c: window_counts.count(c) for c in sorted(unique_counts)}
            raise ValueError(
                f"flat_rep mode requires every rep to produce the same number of "
                f"windows, but found varying counts: {count_summary}. "
                f"Check that all gesture segments have the same length, or switch "
                f"to mode='flat_window' or mode='sequence'."
            )

        expected_len = window_counts[0] * N_FEATURES_TOTAL
        X_list = []
        for windows in all_reps:
            flat = np.concatenate(windows)   # (n_windows * 192,)
            assert flat.shape[0] == expected_len
            X_list.append(flat)

        X = np.array(X_list, dtype=np.float64)     # (n_reps, 4992)
        y = np.array(all_labels, dtype=np.int64)   # (n_reps,)

    else:  # mode == "sequence"
        # Guard: all reps must have the same number of windows
        unique_counts = set(window_counts)
        if len(unique_counts) > 1:
            count_summary = {c: window_counts.count(c) for c in sorted(unique_counts)}
            raise ValueError(
                f"sequence mode requires every rep to produce the same number of "
                f"windows, but found varying counts: {count_summary}. "
                f"Check that all gesture segments have the same length, or switch "
                f"to mode='flat_window'."
            )

        X_list = [np.array(windows, dtype=np.float64) for windows in all_reps]
        X = np.stack(X_list)                        # (n_reps, n_windows, 192)
        y = np.array(all_labels, dtype=np.int64)    # (n_reps,)

    n_windows_per_rep = window_counts[0] if window_counts else 0
    print(f"Windows per rep : {n_windows_per_rep}")
    print(f"Feature matrix  : X={X.shape}, y={y.shape}")
    print(f"Class distribution: { {GESTURE_NAMES[i]: int((y == i).sum()) for i in range(n_gestures)} }")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if mode == "sequence":
        np.savez(
            output_filepath,
            X             = X,
            y             = y,
            feature_names = np.array(FEATURE_NAMES),
            gesture_names = np.array(GESTURE_NAMES),
            n_channels          = N_CHANNELS,
            n_features_per_ch   = N_FEATURES_PER_CH,
            n_windows_per_rep   = n_windows_per_rep,
            window_size         = window_size,
            window_shift        = window_shift,
            sampling_rate       = sampling_rate,
        )
    else:
        feature_names_cell = np.empty((1, len(FEATURE_NAMES)), dtype=object)
        feature_names_cell[0, :] = FEATURE_NAMES

        gesture_names_cell = np.empty((1, len(GESTURE_NAMES)), dtype=object)
        gesture_names_cell[0, :] = GESTURE_NAMES

        scipy.io.savemat(
            output_filepath,
            {
                "X":                   X,
                "y":                   y.reshape(-1, 1).astype(np.float64),
                "feature_names":       feature_names_cell,
                "gesture_names":       gesture_names_cell,
                "n_channels":          float(N_CHANNELS),
                "n_features_per_ch":   float(N_FEATURES_PER_CH),
                "n_windows_per_rep":   float(n_windows_per_rep),
                "window_size":         float(window_size),
                "window_shift":        float(window_shift),
                "sampling_rate":       float(sampling_rate),
            },
        )

    print(f"\nSaved   : {output_filepath}")


# ── Batch API ─────────────────────────────────────────────────────────────────

def batch_extract_features(
    input_dir:     str,
    output_dir:    str,
    mode:          str   = "flat_window",
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
    input_dir     : str
    output_dir    : str   — created automatically if it does not exist.
    mode          : str   — "flat_window", "flat_rep", or "sequence".
    window_size   : int
    window_shift  : int
    sampling_rate : float

    Output naming
    -------------
    Input  : emg_gestures_03_combined_non_uniform.mat
    Output : features_subject_03_flat_window.mat   (or .npz for sequence mode)

    Subject ID is parsed with the regex ``emg_gestures_(\\w+?)_``.
    Falls back to ``features_<stem>_<mode>.<ext>`` if the pattern does not match.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {VALID_MODES}.")

    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    ext       = ".npz" if mode == "sequence" else ".mat"
    mat_files = sorted(glob.glob(os.path.join(input_dir, "*.mat")))

    if not mat_files:
        print(f"No .mat files found in: {input_dir}")
        return

    print(f"Found {len(mat_files)} .mat file(s) in:\n  {input_dir}")
    print(f"Mode  : {mode}\n")

    processed = 0
    for input_filepath in mat_files:
        filename = os.path.basename(input_filepath)

        match = re.search(r"emg_gestures_(\w+?)_", filename)
        if match:
            subject_id      = match.group(1)
            output_filename = f"features_subject_{subject_id}_{mode}{ext}"
        else:
            stem            = os.path.splitext(filename)[0]
            output_filename = f"features_{stem}_{mode}{ext}"
            print(f"  [warn] Could not parse subject ID from '{filename}'. "
                  f"Using '{output_filename}'.")

        output_filepath = os.path.join(output_dir, output_filename)

        if os.path.isfile(output_filepath):
            os.remove(output_filepath)

        print(f"─── {filename}  →  {output_filename}")
        extract_features(
            input_filepath  = input_filepath,
            output_filepath = output_filepath,
            mode            = mode,
            window_size     = window_size,
            window_shift    = window_shift,
            sampling_rate   = sampling_rate,
        )
        processed += 1
        print()

    print(f"Batch complete : {processed}/{len(mat_files)} file(s) processed.")
    print(f"Outputs written: {output_dir}")
