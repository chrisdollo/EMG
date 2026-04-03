import os
import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR         = '/Users/chrisdollo/Documents/Research/putEMG prime/data/UG_per_subject'
BATCH_SIZE       = 16
TEST_SUBJECT_IDX = 0   # index into the sorted subject list to hold out as test
DEV_SUBJECT_IDX  = 1   # index into the sorted subject list to use as dev (val) set


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
def load_data(file_path):
    """
    Load a single per-subject .mat file with key "combinedCell".
    Returns X (N, 1500, 24) float32 and Y (N,) int64.
    """
    mat_file   = scipy.io.loadmat(file_path)
    cell_array = mat_file["combinedCell"]

    X, Y = [], []
    num_classes     = cell_array.shape[1]
    num_training_ex = cell_array.shape[0]

    for gesture_type in range(num_classes):
        for row_idx in range(num_training_ex):
            cell = cell_array[row_idx, gesture_type]
            cell = np.array(cell)
            X.append(cell.astype(np.float32))
            Y.append(gesture_type)

    return np.array(X), np.array(Y)


def _reshape(X, Y):
    """Reshape from (N, 1500, 24) to (N, 1, 24, 1500) and squeeze Y."""
    X = X[:, np.newaxis, :, :]
    X = np.transpose(X, (0, 1, 3, 2))
    Y = Y.squeeze().astype(np.int64)
    return X, Y


def load_all_subjects(dir_path):
    """
    Load all per-subject .mat files from dir_path.
    Returns a list of (subject_name, X, Y) tuples in sorted order.
    X shape per subject: (N, 1, 24, 1500)  Y shape: (N,)
    """
    mat_files = sorted(glob.glob(os.path.join(dir_path, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {dir_path}")

    print(f"Loading {len(mat_files)} subject file(s) from: {dir_path}\n")
    subjects = []
    for file_path in mat_files:
        name = os.path.basename(file_path)
        X, Y = load_data(file_path)
        X, Y = _reshape(X, Y)
        print(f"  → {name}  ({X.shape[0]} samples)")
        subjects.append((name, X, Y))

    print(f"\nTotal subjects loaded: {len(subjects)}")
    return subjects


# ── LOSO split ────────────────────────────────────────────────────────────────
def make_loso_loaders(subjects, test_subject_idx, dev_subject_idx=-1):
    """
    Build train/dev/test DataLoaders for one LOSO fold.

    Parameters
    ----------
    subjects         : list of (name, X, Y) from load_all_subjects()
    test_subject_idx : index of the subject to hold out as the test set
    dev_subject_idx  : index of the subject to use as the dev (validation) set.
                       Supports negative indexing (-1 = last subject).
                       If it collides with test_subject_idx, shifts by +1 automatically.

    Returns
    -------
    train_loader, dev_loader, test_loader
    """
    N = len(subjects)
    if dev_subject_idx < 0:
        dev_subject_idx = N + dev_subject_idx  # resolve negative index

    # If dev and test would be the same subject, shift dev by 1
    if dev_subject_idx == test_subject_idx:
        dev_subject_idx = (dev_subject_idx + 1) % N

    assert 0 <= test_subject_idx < N, "test_subject_idx out of range"
    assert 0 <= dev_subject_idx  < N, "dev_subject_idx out of range"

    test_name, X_test, y_test = subjects[test_subject_idx]
    dev_name,  X_dev,  y_dev  = subjects[dev_subject_idx]

    # create the array of train_subject makes sure none of the training subject are used for test or validation
    train_subjects = [
        (name, X, Y) for i, (name, X, Y) in enumerate(subjects)
        if i != test_subject_idx and i != dev_subject_idx
    ]

    X_train = np.concatenate([X for _, X, _ in train_subjects], axis=0)
    y_train = np.concatenate([Y for _, _, Y in train_subjects], axis=0)

    print(f"  Test  : {test_name}  — {X_test.shape[0]} samples")
    print(f"  Dev   : {dev_name}   — {X_dev.shape[0]} samples")
    print(f"  Train : {len(train_subjects)} subjects, {X_train.shape[0]} samples total\n")

    train_loader = DataLoader(BCIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(BCIDataset(X_dev,   y_dev),   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(BCIDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, dev_loader, test_loader


def loso_folds(subjects, n_folds=1, dev_subject_idx=-1):
    """
    Generator that yields LOSO folds, rotating the test subject.

    Dev subject is fixed (default: last subject = index -1) so the same
    subject is always held out for validation across all folds. If the
    rotating test subject collides with dev, make_loso_loaders shifts dev
    automatically.

    Parameters
    ----------
    subjects        : list of (name, X, Y) from load_all_subjects()
    n_folds         : how many folds to run. Default: all subjects.
                      Set to a small number (e.g. 3) to run a quick subset.
    dev_subject_idx : fixed index for the dev subject across all folds.
                      Supports negative indexing.

    Yields
    ------
    fold_idx, test_subject_name, train_loader, dev_loader, test_loader

    Example
    -------
    # Quick run — 3 folds
    for fold, test_name, train_loader, dev_loader, test_loader in loso_folds(subjects, n_folds=3):
        # train model, record accuracy ...

    # Full LOSO — all subjects
    for fold, test_name, train_loader, dev_loader, test_loader in loso_folds(subjects):
        # train model, record accuracy ...
    """
    N       = len(subjects)

    print(f"Running {n_folds}/{N} LOSO fold(s)\n")
    for fold in range(n_folds):
        test_idx = fold  # rotate test subject: fold 0 → subject 0, fold 1 → subject 1 ...
        print(f"── Fold {fold + 1}/{n_folds} ──────────────────────")
        train_loader, dev_loader, test_loader = make_loso_loaders(
            subjects, test_idx, dev_subject_idx
        )
        yield fold, subjects[test_idx][0], train_loader, dev_loader, test_loader


# ── Training & evaluation functions ──────────────────────────────────────────
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total


def evaluateFinal(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    actual    = torch.tensor([])
    predicted = torch.tensor([])
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct   += (pred == y).sum().item()
            total     += y.size(0)
            actual    = torch.cat((actual,    y.cpu()),    dim=0)
            predicted = torch.cat((predicted, pred.cpu()), dim=0)

        print(np.unique(predicted.numpy(), return_counts=True))
        cm = confusion_matrix(actual.numpy(), predicted.numpy())
        ConfusionMatrixDisplay(cm).plot()

    print(f"The accuracy for the test is {(correct / total) * 100:.2f}%")
    return correct / total


# ── Load all subjects & build default single-fold LOSO loaders ───────────────
subjects = load_all_subjects(DATA_DIR)

train_loader, dev_loader, test_loader = make_loso_loaders(
    subjects,
    test_subject_idx=TEST_SUBJECT_IDX,
    dev_subject_idx=DEV_SUBJECT_IDX,   # -1 = last subject; auto-shifts if it collides with test
)