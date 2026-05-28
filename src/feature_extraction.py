"""
src/feature_extraction.py
-----------------------------
Extracts 50 libemg features from loaded putEMG subjects and saves per-subject .npz files.

50 feature groups × 24 channels = 3192-dimensional vector per window.
(Some features are multi-valued, e.g. AR coefficients, CC, DFTR — 133 scalars/channel total.)

Modes
-----
  "flat_window"  — one row per sliding window  X: (n_windows_total, 1200)
  "flat_rep"     — windows concatenated per rep X: (n_reps, n_wins * 1200)
  "sequence"     — windows stacked per rep      X: (n_reps, n_wins, 1200)

Subjects whose output file already exists are skipped automatically.

Usage
-----
    from src.emg_loader import load_all_subjects
    from src.feature_extraction import batch_extract

    subjects = load_all_subjects('/Volumes/KRIS/data/UG_per_subject')
    batch_extract(subjects, output_dir='/path/to/features', mode='sequence')
"""

import os
import re
import numpy as np
import libemg

_fe           = libemg.feature_extractor.FeatureExtractor()
FEATURE_LIST  = _fe.get_feature_list()   # all 50 features
N_FEATURES    = len(FEATURE_LIST)        # 50
N_CHANNELS    = 24
VALID_MODES   = ('flat_window', 'flat_rep', 'sequence')


def _subject_id(filename):
    m = re.search(r'emg_gestures_(\w+?)_', filename)
    return m.group(1) if m else os.path.splitext(filename)[0]


def _slide_windows(signal, window_size, window_shift):
    """signal: (24, 1500)  →  (n_windows, 24, window_size)"""
    starts = range(0, signal.shape[1] - window_size + 1, window_shift)
    return np.stack([signal[:, s:s + window_size] for s in starts])


def _extract_subject(X_subj, y_subj, fe, mode, window_size, window_shift):
    """
    X_subj : (N_reps, 1, 24, 1500)
    y_subj : (N_reps,)
    Returns X_out, y_out, n_windows_per_rep
    """
    X_list, y_list, window_counts = [], [], []

    for i in range(X_subj.shape[0]):
        signal = X_subj[i, 0]                                           # (24, 1500)
        wins   = _slide_windows(signal, window_size, window_shift)      # (n_wins, 24, window_size)
        feats  = fe.extract_features(FEATURE_LIST, wins, array=True)    # (n_wins, 1200)
        X_list.append(feats)
        y_list.append(y_subj[i])
        window_counts.append(len(wins))

    if mode == 'flat_window':
        X = np.vstack(X_list).astype(np.float32)
        y = np.repeat(np.array(y_list, dtype=np.int64), window_counts)
    elif mode == 'flat_rep':
        X = np.array([f.flatten() for f in X_list], dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
    else:  # sequence
        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

    return X, y, window_counts[0]


def batch_extract(subjects, output_dir, mode='sequence', window_size=250, window_shift=50):
    """
    Extract libemg features for every subject and save as .npz files.
    Skips subjects whose output file already exists.

    Parameters
    ----------
    subjects     : list of (name, X, Y) — from load_all_subjects()
                   X shape: (N_reps, 1, 24, 1500)
    output_dir   : str
    mode         : 'flat_window' | 'flat_rep' | 'sequence'
    window_size  : int, default 250
    window_shift : int, default 50

    Output files
    ------------
    features_subject_<id>_<mode>.npz  — one per subject, in output_dir
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}")

    os.makedirs(output_dir, exist_ok=True)
    fe = libemg.feature_extractor.FeatureExtractor()

    print(f"libemg feature extraction")
    print(f"  {N_FEATURES} feature groups × {N_CHANNELS} channels = 3192 dims/window")
    print(f"  mode={mode}  window_size={window_size}  window_shift={window_shift}\n")

    for name, X, y in subjects:
        sid      = _subject_id(name)
        out_path = os.path.join(output_dir, f'features_subject_{sid}_{mode}.npz')

        if os.path.exists(out_path):
            print(f'  [SKIP] {name}')
            continue

        print(f'  {name} ...', end=' ', flush=True)
        X_out, y_out, n_wins = _extract_subject(X, y, fe, mode, window_size, window_shift)

        np.savez(out_path,
            X                 = X_out,
            y                 = y_out,
            feature_list      = np.array(FEATURE_LIST),
            n_windows_per_rep = n_wins,
            window_size       = window_size,
            window_shift      = window_shift,
        )
        print(f'X={X_out.shape}  y={y_out.shape}  →  {os.path.basename(out_path)}')

    print(f'\nDone. Files saved to: {output_dir}')
