import scipy.io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
FILE_PATH    = '/Users/chrisdollo/Documents/Research/putEMG prime/Data/X/model_ready_5/model_ready_5_sub.mat'
BATCH_SIZE   = 16
TEST_SIZE    = 0.2   # held-out test set
DEV_SIZE     = 0.1   # dev split from training set (used for early stopping)
RANDOM_STATE = 42


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
    Load the combined model-ready .mat file.
    Returns X (N, 1500, 24) float32 and Y (N,) int64.
    """
    mat_file   = scipy.io.loadmat(file_path)
    cell_array = mat_file["combinedCell"]

    X, Y = [], []
    num_classes     = cell_array.shape[1]
    num_training_ex = cell_array.shape[0]

    print("num_classes",     num_classes)
    print("num_training_ex", num_training_ex)

    for gesture_type in range(num_classes):
        for row_idx in range(num_training_ex):
            cell = cell_array[row_idx, gesture_type]
            cell = np.array(cell)
            X.append(cell.astype(np.float32))
            Y.append(gesture_type)

    return np.array(X), np.array(Y)


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


def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_loader:
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


# ── Load & split data ─────────────────────────────────────────────────────────
X, Y = load_data(FILE_PATH)

# Reshape to (N, 1, channels, samples) — model input format
X = X[:, np.newaxis, :, :]
X = np.transpose(X, (0, 1, 3, 2))
Y = Y.squeeze().astype(np.int64)

print(f"X: {X.shape}, Y: {Y.shape}")

# 80/20 train/test split — test set held out until final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE
)

# 90/10 train/dev split from training set — used only for early stopping
X_train, X_dev, y_train, y_dev = train_test_split(
    X_train, y_train, test_size=DEV_SIZE, stratify=y_train, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)} | Dev: {len(X_dev)} | Test: {len(X_test)}")

# ── DataLoaders ───────────────────────────────────────────────────────────────
train_loader = DataLoader(BCIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(BCIDataset(X_dev,   y_dev),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(BCIDataset(X_test,  y_test),  batch_size=BATCH_SIZE, shuffle=False)

# Full dataset loader — used by the retrain section of model.ipynb
full_loader  = DataLoader(BCIDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)
