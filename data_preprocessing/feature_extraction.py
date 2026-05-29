import os
import re
import glob
import numpy as np
import scipy.io
import libemg

GESTURE_NAMES = ["G1", "G2", "G3", "G6", "G7", "G8", "G9"]
N_CHANNELS    = 24
MAT_KEY       = "combinedCell"

_fe          = libemg.feature_extractor.FeatureExtractor()
FEATURE_LIST = _fe.get_feature_list()  # all 50 libemg features


def _subject_id(filename):
    m = re.search(r'emg_gestures_(\w+?)_', filename)
    return m.group(1) if m else os.path.splitext(filename)[0]


def _slide_windows(ch_first, window_size, window_shift):
    # ch_first: (24, 1500) → (n_wins, 24, window_size)
    starts = range(0, ch_first.shape[1] - window_size + 1, window_shift)
    return np.stack([ch_first[:, s:s + window_size] for s in starts])


def _rep_to_flat(signal, fe, window_size, window_shift):
    # signal from mat: (1500, 24) — transpose to channel-first for libemg
    ch_first = signal.T                                              # (24, 1500)
    wins     = _slide_windows(ch_first, window_size, window_shift)  # (n_wins, 24, window_size)
    feats    = fe.extract_features(FEATURE_LIST, wins, array=True)  # (n_wins, n_feat_total)
    return feats.flatten().astype(np.float32), len(wins)            # flat vector, window count


def batch_extract_features(input_dir, output_dir, window_size=250, window_shift=50):
    os.makedirs(output_dir, exist_ok=True)
    fe        = libemg.feature_extractor.FeatureExtractor()
    mat_files = sorted(glob.glob(os.path.join(input_dir, '*.mat')))

    if not mat_files:
        print(f'No .mat files found in: {input_dir}')
        return

    print(f'Found {len(mat_files)} file(s) — libemg {len(FEATURE_LIST)} features × {N_CHANNELS} ch, flat_rep\n')

    for input_path in mat_files:
        filename = os.path.basename(input_path)
        sid      = _subject_id(filename)
        out_path = os.path.join(output_dir, f'features_subject_{sid}_flat_rep.npz')

        if os.path.exists(out_path):
            print(f'  [SKIP] {filename}')
            continue

        mat_data      = scipy.io.loadmat(input_path)
        combined_cell = mat_data[MAT_KEY]
        n_reps, n_gest = combined_cell.shape

        X_list, y_list, n_wins = [], [], None

        for gest_idx in range(n_gest):
            for rep_idx in range(n_reps):
                cell = combined_cell[rep_idx, gest_idx]
                if cell is None or not isinstance(cell, np.ndarray) or cell.size == 0:
                    continue
                sig = np.array(cell, dtype=np.float64)
                if sig.shape[0] < window_size:
                    continue
                flat, nw = _rep_to_flat(sig, fe, window_size, window_shift)
                X_list.append(flat)
                y_list.append(gest_idx)
                if n_wins is None:
                    n_wins = nw

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        np.savez(out_path,
            X                 = X,
            y                 = y,
            feature_list      = np.array(FEATURE_LIST),
            n_windows_per_rep = n_wins,
            window_size       = window_size,
            window_shift      = window_shift,
        )
        print(f'  {filename} → X={X.shape}  y={y.shape}')

    print(f'\nDone → {output_dir}')
