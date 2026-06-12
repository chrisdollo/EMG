import os
import glob
import random
import numpy as np
from scipy.signal import welch

import torch
from torch.utils.data import Dataset, DataLoader


# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR         = '/Volumes/KRIS/data/UG_per_subject'
BATCH_SIZE       = 16
TEST_SUBJECT_IDX = 0
DEV_SUBJECT_IDX  = 1

# putEMG gesture IDs → 0-indexed class labels for CrossEntropyLoss
GESTURE_ID_TO_CLASS = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4, 8: 5, 9: 6}

def _remap_labels(y):
    return np.array([GESTURE_ID_TO_CLASS[int(g)] for g in y], dtype=np.int64)


# ── Dataset ───────────────────────────────────────────────────────────────────
class BCIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_subjects(dir_path):
    # Loads .npz files produced by build_dataset(); each file has X:(N,24,1500) y:(N,)
    npz_files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {dir_path}")

    print(f"Loading {len(npz_files)} subject file(s) from: {dir_path}\n")
    subjects = []
    for file_path in npz_files:
        name = os.path.basename(file_path)
        data = np.load(file_path)
        X = data['X'].astype(np.float32)[:, np.newaxis, :, :]  # (N, 1, 24, 1500)
        y = _remap_labels(data['y'])                            # (N,) → 0-indexed
        print(f"  → {name}  ({X.shape[0]} reps)")
        subjects.append((name, X, y))

    print(f"\nTotal subjects loaded: {len(subjects)}")
    return subjects


def load_subjects_combined(train_dir, eval_dir):
    """Load train+eval .npz files and concatenate per subject.

    Matches files by subject number extracted from filename.
    Returns list of (name, X, y) with X:(N_total, 1, 24, 1500), labels 0-indexed.
    """
    import re as _re

    def _sid(path):
        m = _re.search(r'emg_gestures_(\d+)', os.path.basename(path))
        return m.group(1) if m else None

    train_files = {_sid(p): p for p in sorted(glob.glob(os.path.join(train_dir, '*.npz'))) if _sid(p)}
    eval_files  = {_sid(p): p for p in sorted(glob.glob(os.path.join(eval_dir,  '*.npz'))) if _sid(p)}

    all_ids = sorted(set(train_files) | set(eval_files))
    print(f"Combining train+eval for {len(all_ids)} subject(s)\n")

    subjects = []
    for sid in all_ids:
        parts_X, parts_y = [], []
        for src in (train_files, eval_files):
            if sid in src:
                d = np.load(src[sid])
                parts_X.append(d['X'].astype(np.float32)[:, np.newaxis, :, :])
                parts_y.append(_remap_labels(d['y']))
        X = np.concatenate(parts_X, axis=0)
        y = np.concatenate(parts_y, axis=0)
        name = f'emg_gestures_{sid}_U.npz'
        print(f"  → {name}  ({X.shape[0]} reps)")
        subjects.append((name, X, y))

    print(f"\nTotal subjects loaded: {len(subjects)}")
    return subjects


def load_feature_subjects_combined(train_dir, eval_dir, mode='flat_rep'):
    """Load train+eval feature .npz files, concatenate, and reshape to sequence format.

    Returns list of (name, X, y) with X:(N, n_wins, feat_per_win), labels 0-indexed.
    """
    import re as _re

    def _sid(path):
        m = _re.search(r'features_(\d+)', os.path.basename(path))
        return m.group(1) if m else None

    pattern = f'*_{mode}.npz'
    train_files = {_sid(p): p for p in sorted(glob.glob(os.path.join(train_dir, pattern))) if _sid(p)}
    eval_files  = {_sid(p): p for p in sorted(glob.glob(os.path.join(eval_dir,  pattern))) if _sid(p)}

    all_ids = sorted(set(train_files) | set(eval_files))
    if not all_ids:
        raise FileNotFoundError(
            f"No '{pattern}' files found in {train_dir} or {eval_dir}.\n"
            "Run driver.ipynb Stage 2 first."
        )
    print(f"Combining train+eval feature files for {len(all_ids)} subject(s)\n")

    subjects = []
    for sid in all_ids:
        parts_X, parts_y = [], []
        n_wins = None
        for src in (train_files, eval_files):
            if sid in src:
                d = np.load(src[sid])
                parts_X.append(d['X'].astype(np.float32))
                parts_y.append(_remap_labels(d['y']))
                if n_wins is None:
                    n_wins = int(d['n_windows_per_rep'])
        X_flat = np.concatenate(parts_X, axis=0)           # (N, n_wins * feat_per_win)
        y      = np.concatenate(parts_y, axis=0)
        X      = X_flat.reshape(len(X_flat), n_wins, -1)   # (N, n_wins, feat_per_win)
        name   = f'features_{sid}_{mode}.npz'
        print(f"  → {name}  X={X.shape}")
        subjects.append((name, X, y))

    print(f"\nTotal subjects loaded: {len(subjects)}")
    return subjects


# ── LOSO splits ───────────────────────────────────────────────────────────────

# make_loso_loaders
# Build train / dev / test DataLoaders for one LOSO fold.
# train_loader, dev_loader, test_loader
def make_loso_loaders(subjects, test_subject_idx=-2, dev_subject_idx=-1, batch_size=BATCH_SIZE):
   
    N = len(subjects)
    if dev_subject_idx  < 0: dev_subject_idx  = N + dev_subject_idx
    if test_subject_idx < 0: test_subject_idx = N + test_subject_idx

    if dev_subject_idx == test_subject_idx:
        dev_subject_idx = (dev_subject_idx + 1) % N

    assert 0 <= test_subject_idx < N, "test_subject_idx out of range"
    assert 0 <= dev_subject_idx  < N, "dev_subject_idx out of range"

    test_name, X_test, y_test = subjects[test_subject_idx]
    dev_name,  X_dev,  y_dev  = subjects[dev_subject_idx]

    train_subjects = [
        (name, X, Y) for i, (name, X, Y) in enumerate(subjects)
        if i != test_subject_idx and i != dev_subject_idx
    ]

    X_train = np.concatenate([X for _, X, _ in train_subjects], axis=0)
    y_train = np.concatenate([Y for _, _, Y in train_subjects], axis=0)

    print(f"  Test  : {test_name}  — {X_test.shape[0]} samples")
    print(f"  Dev   : {dev_name}   — {X_dev.shape[0]} samples")
    print(f"  Train : {len(train_subjects)} subjects, {X_train.shape[0]} samples total\n")

    train_loader = DataLoader(BCIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(BCIDataset(X_dev,   y_dev),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(BCIDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader

# loso_folds
# Build DataLoaders for a single LOSO fold.
def loso_folds(subjects, test_subject_idx=-2, dev_subject_idx=-1):
 
    N = len(subjects)
    if dev_subject_idx  < 0: dev_subject_idx  = N + dev_subject_idx
    if test_subject_idx < 0: test_subject_idx = N + test_subject_idx

    if dev_subject_idx == test_subject_idx:
        dev_subject_idx = (dev_subject_idx + 1) % N

    assert 0 <= test_subject_idx < N, "test_subject_idx out of range"
    assert 0 <= dev_subject_idx  < N, "dev_subject_idx out of range"

    print(f"─ LOSO fold — testS ID: {test_subject_idx}, valS ID: {dev_subject_idx} ──────────────────────")
    train_loader, dev_loader, test_loader = make_loso_loaders(
        subjects, test_subject_idx, dev_subject_idx
    )
    return subjects[test_subject_idx][0], train_loader, dev_loader, test_loader


# generate_loso_configs
# Generate n_folds random (test_idx, dev_idx) pairs for LOSO evaluation.
# returns list of (fold_idx, test_idx, dev_idx) tuples
def generate_loso_configs(subjects, n_folds, seed=42):
   
    N   = len(subjects)
    rng = random.Random(seed)

    n_folds      = min(n_folds, N)
    test_indices = list(range(N))
    rng.shuffle(test_indices)
    test_indices = test_indices[:n_folds]

    configs = []
    for fold_idx, test_idx in enumerate(test_indices):
        remaining = [i for i in range(N) if i != test_idx]
        dev_idx   = rng.choice(remaining)
        configs.append((fold_idx, test_idx, dev_idx))

    print(f"Generated {n_folds} LOSO fold(s) (seed={seed}):")
    for fold_idx, test_idx, dev_idx in configs:
        print(f"  Fold {fold_idx + 1}: test={subjects[test_idx][0]}  dev={subjects[dev_idx][0]}")
    print()

    return configs


# ── Within-subject CV splits ──────────────────────────────────────────────────

def make_within_subject_loaders(X, y, n_folds=3, test_fold_idx=0, batch_size=BATCH_SIZE, seed=42):
    """
    Build train and test DataLoaders for one fold of within-subject k-fold CV.

    Stratified split: each class contributes equally to every fold so class
    balance is preserved regardless of fold index.

    Parameters
    ----------
    X             : ndarray (N, 1, 24, 1500)
    y             : ndarray (N,)
    n_folds       : int   — number of folds, default 3 (matches putEMG paper)
    test_fold_idx : int   — which fold is held out as test (0 to n_folds-1)
    batch_size    : int
    seed          : int   — for reproducible per-class shuffling

    Returns
    -------
    train_loader, test_loader
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)

    fold_indices = [[] for _ in range(n_folds)]
    for cls in classes:
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        for f, split in enumerate(np.array_split(cls_idx, n_folds)):
            fold_indices[f].extend(split.tolist())

    test_idx  = np.array(fold_indices[test_fold_idx])
    train_idx = np.concatenate([
        np.array(fold_indices[f]) for f in range(n_folds) if f != test_fold_idx
    ])

    X_test,  y_test  = X[test_idx],  y[test_idx]
    X_train, y_train = X[train_idx], y[train_idx]

    print(f"    Fold {test_fold_idx+1}/{n_folds}: train={len(train_idx)}  test={len(test_idx)}")

    train_loader = DataLoader(BCIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(BCIDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ── Feature extraction ────────────────────────────────────────────────────────
# 8 features per channel: MAV, RMS, WL, ZC, SSC, VAR, MNF, MDF
# Feature vector layout: [8 feats ch0 | 8 feats ch1 | ... | 8 feats ch23] → 192-dim

FEATURE_NAMES     = ["MAV", "RMS", "WL", "ZC", "SSC", "VAR", "MNF", "MDF"]
N_FEATURES_PER_CH = len(FEATURE_NAMES)   # 8
N_FEATURES_TOTAL  = N_FEATURES_PER_CH * 24  # 192







def _extract_window(window: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract 8 features from every channel of a single window.

    Parameters
    ----------
    window : ndarray (window_size, n_channels)
    fs     : float

    Returns
    -------
    features : ndarray (n_channels * 8,)  — i.e. (192,)
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


def _slide_rep(
    signal:       np.ndarray,
    window_size:  int,
    window_shift: int,
    fs:           float,
) -> list:
    """
    Slide a window over one gesture rep and extract features from each position.

    Parameters
    ----------
    signal       : ndarray (M_samples, 24)
    window_size  : int
    window_shift : int
    fs           : float

    Returns
    -------
    list of ndarray, each (192,).  Empty if signal shorter than window_size.
    """
    n_samples = signal.shape[0]
    windows, start = [], 0
    while start + window_size <= n_samples:
        windows.append(_extract_window(signal[start:start + window_size, :], fs=fs))
        start += window_shift
    return windows


def extract_features_from_subjects(
    subjects,
    window_size:  int   = 250,
    window_shift: int   = 50,
    sampling_rate: float = 5120.0,
):
    """
    Extract hand-crafted EMG features from a loaded subjects list.

    Takes the output of load_all_subjects() directly — no extra file I/O needed.
    Each rep in subjects has shape (N, 1, 24, 1500); this function un-reshapes it
    back to (1500, 24) before sliding the window.

    Parameters
    ----------
    subjects      : list of (name, X, Y)  — from load_all_subjects()
                    X shape: (N_reps, 1, 24, 1500)
    window_size   : int,   default 250  (~49 ms at 5120 Hz)
    window_shift  : int,   default 50   (~10 ms at 5120 Hz)
    sampling_rate : float, default 5120.0

    Returns
    -------
    X : ndarray (N_windows, 192)  float32
    y : ndarray (N_windows,)      int64
    """
    X_list, y_list = [], []

    for _name, X_s, Y_s in subjects:
        for i in range(X_s.shape[0]):
            signal  = X_s[i, 0, :, :].T          # (24, 1500) → (1500, 24)
            windows = _slide_rep(signal, window_size, window_shift, sampling_rate)
            if windows:
                X_list.extend(windows)            # each (192,)
                y_list.extend([Y_s[i]] * len(windows))

    X = np.vstack(X_list).astype(np.float32)     # (N_windows, 192)
    y = np.array(y_list, dtype=np.int64)          # (N_windows,)

    n_subjects = len(subjects)
    n_reps     = sum(X_s.shape[0] for _, X_s, _ in subjects)
    print(f"Feature extraction complete:")
    print(f"  Subjects : {n_subjects}")
    print(f"  Reps     : {n_reps}")
    print(f"  Windows  : {len(X_list)}  (~{len(X_list)//n_reps} per rep)")
    print(f"  X shape  : {X.shape}   y shape: {y.shape}")
    return X, y


# ── Feature file I/O ─────────────────────────────────────────────────────────

# VALID_MODES = ("flat_window", "flat_rep", "sequence")


# def load_feature_subjects(dir_path, mode):
#     """
#     Load pre-extracted per-subject feature files into a subjects list.

#     Files are produced by feature_extraction.batch_extract_features() and must
#     be saved as .npz with keys 'X' and 'y'.

#     Expected file pattern: features_subject_*_<mode>.npz

#     Parameters
#     ----------
#     dir_path : str   — folder containing the .npz files
#     mode     : str   — 'flat_window', 'flat_rep', or 'sequence'

#     Returns
#     -------
#     subjects : list of (name, X, y)
#         flat_window  X shape : (N_windows, 192)     y : (N_windows,)
#         flat_rep     X shape : (N_reps,    4992)    y : (N_reps,)
#         sequence     X shape : (N_reps,    26, 192) y : (N_reps,)
#         Sorted by filename.
#     """
#     if mode not in VALID_MODES:
#         raise ValueError(f"mode must be one of {VALID_MODES}, got '{mode}'")

#     pattern   = os.path.join(dir_path, f"*_{mode}.npz")
#     npz_files = sorted(glob.glob(pattern))

#     if not npz_files:
#         raise FileNotFoundError(
#             f"No '*_{mode}.npz' files found in: {dir_path}\n"
#             f"Run batch_extract_features(mode='{mode}') first."
#         )

#     print(f"Loading {len(npz_files)} '{mode}' feature file(s) from: {dir_path}\n")
#     subjects = []
#     for fpath in npz_files:
#         name = os.path.basename(fpath)
#         data = np.load(fpath)
#         X    = data['X'].astype(np.float32)
#         y    = data['y'].astype(np.int64)
#         print(f"  → {name}  X={X.shape}")
#         subjects.append((name, X, y))

#     print(f"\nTotal subjects loaded: {len(subjects)}")
#     return subjects


# def make_loso_train_val_test(subjects, test_subject_idx, val_frac=0.10, batch_size=BATCH_SIZE, seed=42):
#     """
#     LOSO split with within-pool validation.

#     Test  = one held-out subject (never seen during training).
#     Train = 90% of remaining subjects' data (stratified by class, seeded).
#     Val   = 10% of remaining subjects' data — used only for early stopping.

#     No subject is wasted as a dedicated dev holdout; all N-1 subjects
#     contribute to the training pool.
#     """
#     N = len(subjects)
#     assert 0 <= test_subject_idx < N

#     test_name, X_test, y_test = subjects[test_subject_idx]

#     pool_X = np.concatenate([X for i, (_, X, _) in enumerate(subjects) if i != test_subject_idx])
#     pool_y = np.concatenate([y for i, (_, _, y) in enumerate(subjects) if i != test_subject_idx])

#     rng = np.random.default_rng(seed)
#     train_idx, val_idx = [], []
#     for cls in np.unique(pool_y):
#         idx = np.where(pool_y == cls)[0].copy()
#         rng.shuffle(idx)
#         n_val = max(1, round(len(idx) * val_frac))
#         val_idx.extend(idx[:n_val])
#         train_idx.extend(idx[n_val:])

#     X_train, y_train = pool_X[train_idx], pool_y[train_idx]
#     X_val,   y_val   = pool_X[val_idx],   pool_y[val_idx]

#     print(f"  Test  : {test_name}  ({X_test.shape[0]} reps)")
#     print(f"  Train : {X_train.shape[0]} reps  |  Val: {X_val.shape[0]} reps  ({N-1} subjects, 90/10 split)")

#     train_loader = DataLoader(BCIDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
#     val_loader   = DataLoader(BCIDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
#     test_loader  = DataLoader(BCIDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader, test_loader


# def make_loso_splits(subjects, test_subject_idx, dev_subject_idx):
    """
    Build train / dev / test numpy splits for one LOSO fold.

    Numpy analogue of make_loso_loaders — returns raw arrays so sklearn models
    (SVM, XGBoost) can use them directly. PyTorch notebooks wrap the arrays in
    TensorDataset themselves.

    Parameters
    ----------
    subjects          : list of (name, X, y) — from load_feature_subjects()
    test_subject_idx  : int
    dev_subject_idx   : int

    Returns
    -------
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test)   — all ndarray
    """
    N = len(subjects)
    assert 0 <= test_subject_idx < N, "test_subject_idx out of range"
    assert 0 <= dev_subject_idx  < N, "dev_subject_idx out of range"
    assert test_subject_idx != dev_subject_idx, "test and dev must differ"

    test_name, X_test, y_test = subjects[test_subject_idx]
    dev_name,  X_dev,  y_dev  = subjects[dev_subject_idx]

    train_Xs, train_ys = [], []
    for i, (_, X, y) in enumerate(subjects):
        if i != test_subject_idx and i != dev_subject_idx:
            train_Xs.append(X)
            train_ys.append(y)

    X_train = np.concatenate(train_Xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)

    print(f"  Test  : {test_name}  — {X_test.shape[0]} samples")
    print(f"  Dev   : {dev_name}   — {X_dev.shape[0]} samples")
    print(f"  Train : {len(train_Xs)} subjects, {X_train.shape[0]} samples total\n")

    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test)