import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.emg_loader import make_loso_train_val_test


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


def evaluateFinal(model, test_loader, device, plot=True):
    # Use for post-hoc analysis; plot=True renders a confusion matrix
    model.eval()
    correct, total = 0, 0
    actual    = torch.tensor([], dtype=torch.int64)
    predicted = torch.tensor([], dtype=torch.int64)
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct   += (pred == y).sum().item()
            total     += y.size(0)
            actual    = torch.cat((actual,    y.cpu()),    dim=0)
            predicted = torch.cat((predicted, pred.cpu()), dim=0)

    acc = correct / total
    if plot:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        print(np.unique(predicted.numpy(), return_counts=True))
        cm = confusion_matrix(actual.numpy(), predicted.numpy())
        ConfusionMatrixDisplay(cm).plot()
    print(f'Test accuracy: {acc * 100:.2f}%')
    return acc


def subject_id(filename):
    # 'emg_gestures_03_U.mat' → '03'
    return filename.replace('emg_gestures_', '').replace('_U.mat', '')


def weight_path(subject_name, weights_dir, model_type):
    return os.path.join(weights_dir, f'{model_type}_{subject_id(subject_name)}.pt')


def update_log(weights_dir, log_path, n_total, model_type):
    # Rebuild results log from all valid checkpoint files in weights_dir
    checkpoints = []
    for fname in sorted(os.listdir(weights_dir)):
        if not fname.endswith('.pt'):
            continue
        ckpt = torch.load(os.path.join(weights_dir, fname), map_location='cpu', weights_only=False)
        if 'subject' not in ckpt:
            continue
        checkpoints.append(ckpt)

    if not checkpoints:
        return

    accs   = [c['test_acc'] * 100 for c in checkpoints]
    n_done = len(checkpoints)

    lines = [
        f'putEMG — {model_type} LOSO Results',
        '=' * 68,
        f"{'Subject':<28} {'Test Acc':>9}  {'Val Acc':>9}  {'Epoch':>6}  {'Date'}",
        '-' * 68,
    ]
    for c in checkpoints:
        lines.append(
            f"{c['subject']:<28} {c['test_acc']*100:>8.2f}%  "
            f"{c['val_acc']*100:>8.2f}%  {c['best_epoch']:>6}  {c['date']}"
        )
    lines += [
        '=' * 68,
        f"Mean: {np.mean(accs):.2f}%  ±  {np.std(accs):.2f}%  "
        f"({n_done} / {n_total} folds complete)",
    ]

    with open(log_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Log updated → {log_path}  ({n_done}/{n_total} folds)")


def run_loso(
    subjects,
    model_cls,
    model_type,
    weights_dir,
    log_path,
    device,
    val_frac   = 0.10,
    batch_size = 16,
    dropout    = 0.1,
    lr         = 1e-3,
    max_epochs = 20,
    patience   = 5,
    min_delta  = 0.002,
):
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    n_total   = len(subjects)
    criterion = nn.CrossEntropyLoss()

    for test_idx, (test_name, _, _) in enumerate(subjects):
        wpath = weight_path(test_name, weights_dir, model_type)

        if os.path.exists(wpath):
            print(f"[SKIP] {test_name}  — checkpoint found")
            continue

        print(f"\n{'='*60}")
        print(f"  Fold {test_idx+1}/{n_total}  —  test: {test_name}")
        print(f"{'='*60}")

        train_loader, val_loader, test_loader = make_loso_train_val_test(
            subjects, test_idx, val_frac=val_frac, batch_size=batch_size
        )

        model     = model_cls(dropout_rate=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Halve LR on val plateau — prevents overshooting near convergence
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

        best_val_acc = float('-inf')
        best_state   = None
        best_epoch   = 0
        bad_epochs   = 0

        for epoch in range(max_epochs):
            tr_loss = train(model, train_loader, criterion, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            curr_lr = optimizer.param_groups[0]['lr']

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Clone to CPU so GPU/MPS memory isn't held while training continues
                best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch   = epoch + 1

            scheduler.step(val_acc)
            bad_epochs = 0 if val_acc >= (best_val_acc - min_delta) else bad_epochs + 1

            print(f"  Epoch {epoch+1:3d}: loss={tr_loss:.4f}  val={val_acc*100:.2f}%  "
                  f"best={best_val_acc*100:.2f}%  lr={curr_lr:.2e}")

            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch+1}.")
                break

        model.load_state_dict(best_state)
        test_acc = evaluate(model, test_loader, device)
        print(f"\n  Test accuracy: {test_acc*100:.2f}%")

        torch.save({
            'subject':    test_name,
            'test_acc':   test_acc,
            'val_acc':    best_val_acc,
            'best_epoch': best_epoch,
            'dropout':    dropout,
            'state_dict': best_state,
            'date':       datetime.date.today().isoformat(),
        }, wpath)

        update_log(weights_dir, log_path, n_total, model_type)

    print(f"\n{'='*60}")
    print("  All folds complete.")
    print(f"{'='*60}")
    update_log(weights_dir, log_path, n_total, model_type)
