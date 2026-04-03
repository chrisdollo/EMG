"""
handler.py — Format 3: Sequence Per Rep
----------------------------------------
Loads per-subject sequence feature files (.npz), concatenates them across
subjects, splits into train / val / test sets, and exposes PyTorch DataLoaders.

Expected input file format
--------------------------
  Produced by feature_extraction.batch_extract_features(mode="sequence").
  File pattern : features_subject_<ID>_sequence.npz
  Keys:
    X                 : (n_reps, n_windows, 192)  float64
    y                 : (n_reps,)                 int64
    n_windows_per_rep : scalar                    (e.g. 26)
    window_size       : scalar
    window_shift      : scalar
    sampling_rate     : scalar

Module-level exports (ready to import in model.ipynb)
------------------------------------------------------
  X_train, X_val, X_test   : ndarray  (n, 26, 192)  float32  (normalized)
  y_train, y_val, y_test   : ndarray  (n,)           int64
  train_loader, val_loader, test_loader : DataLoader
  SEQ_LEN, INPUT_DIM       : int  (26, 192)
  NORM_MEAN, NORM_STD      : ndarray  (1, 1, 192) — apply to new data at inference
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR     = '/Users/chrisdollo/Documents/Research/putEMG prime/data/feature_gestures/format_sequence'
BATCH_SIZE   = 32
TEST_SIZE    = 0.2
DEV_SIZE     = 0.1
RANDOM_STATE = 42

GESTURE_NAMES = ['G1', 'G2', 'G3', 'G6', 'G7', 'G8', 'G9']
N_CLASSES     = 7


# ── Data loading ───────────────────────────────────────────────────────────────
def load_sequences_from_dir(dir_path):
    """
    Load and concatenate all sequence .npz files in dir_path.

    Parameters
    ----------
    dir_path : str — folder containing features_subject_*_sequence.npz files.

    Returns
    -------
    X : ndarray, shape (n_reps_total, n_windows, 192)  float32
    y : ndarray, shape (n_reps_total,)                 int64
    """
    npz_files = sorted(glob.glob(os.path.join(dir_path, '*_sequence.npz')))

    if not npz_files:
        raise FileNotFoundError(
            f"No '*_sequence.npz' files found in: {dir_path}\n"
            f"Run feature_extraction.batch_extract_features(mode='sequence') first."
        )

    print(f"Loading {len(npz_files)} subject file(s) from: {dir_path}\n")

    X_list, y_list = [], []
    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        X_sub = data['X'].astype(np.float32)   # (n_reps, n_windows, 192)
        y_sub = data['y'].astype(np.int64)     # (n_reps,)
        X_list.append(X_sub)
        y_list.append(y_sub)
        print(f"  {os.path.basename(fpath)}: {X_sub.shape[0]} reps, "
              f"sequence shape {X_sub.shape[1:]}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f"\nCombined : X={X.shape}, y={y.shape}")
    print(f"Classes  : { {GESTURE_NAMES[i]: int((y == i).sum()) for i in range(N_CLASSES)} }\n")

    return X, y


def _make_loader(X, y, shuffle=False):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


# ── Load, split, and build DataLoaders ────────────────────────────────────────
X, y = load_sequences_from_dir(DATA_DIR)

# Derive sequence dimensions from the loaded data
SEQ_LEN   = X.shape[1]   # 26
INPUT_DIM = X.shape[2]   # 192

# 80/20 train+val / test  — split at rep level, no window leakage possible here
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# 90/10 train / val  from the training portion only
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=DEV_SIZE, stratify=y_train, random_state=RANDOM_STATE
)

# Normalize features using training statistics only (per feature, over reps × timesteps)
# Shape: (1, 1, 192) — broadcasts over (n_reps, 26, 192)
X_mean = X_train.mean(axis=(0, 1), keepdims=True)
X_std  = X_train.std(axis=(0, 1),  keepdims=True) + 1e-8

X_train = (X_train - X_mean) / X_std
X_val   = (X_val   - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

print(f"Train : {len(X_train):>6,} reps")
print(f"Val   : {len(X_val):>6,} reps")
print(f"Test  : {len(X_test):>6,} reps")
print(f"SEQ_LEN={SEQ_LEN}, INPUT_DIM={INPUT_DIM}")
print(f"Feature range after norm — min: {X_train.min():.3f}, max: {X_train.max():.3f}, "
      f"mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")

train_loader = _make_loader(X_train, y_train, shuffle=True)
val_loader   = _make_loader(X_val,   y_val,   shuffle=False)
test_loader  = _make_loader(X_test,  y_test,  shuffle=False)

# Expose normalization stats for inference (apply same transform to new data)
NORM_MEAN = X_mean   # shape (1, 1, 192)
NORM_STD  = X_std    # shape (1, 1, 192)
