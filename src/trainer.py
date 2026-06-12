# import os
# import re
# import datetime
# import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim



def train_batch(model, train_loader, criterion, optimizer, device):
    # train_batch: run one training epoch over train_loader, return mean loss
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()        # clear last step's gradients
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()              # backprop the loss
        optimizer.step()             # update the weights
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, loader, device):
    # evaluate: return classification accuracy over loader (no gradients)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)        # predicted class per sample
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total


def evaluateFinal(model, test_loader, device, plot=True):
    # evaluateFinal: post-hoc accuracy; plot=True also draws a confusion matrix
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


def train(model, train_loader, val_loader, device, *,
          max_epochs, patience, min_delta, lr):
    # train: training loop with early stopping on validation accuracy.
    # Snapshots the best-epoch weights and stops when val accuracy hasn't improved
    # by more than min_delta for `patience` epochs.
    # Returns (best_state, best_val, best_epoch).
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val   = float('-inf')   # best val accuracy seen so far
    best_state = None            # CPU copy of weights at best_val
    best_epoch = 0
    bad_epochs = 0               # epochs since the last improvement

    for epoch in range(max_epochs):
        tr_loss = train_batch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val + min_delta:   # improved: snapshot weights, reset counter
            best_val   = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            bad_epochs = 0
        else:                                # no improvement: step toward early stop
            bad_epochs += 1

        print(f"  epoch {epoch+1:3d}: loss={tr_loss:.4f}  "
              f"val_acc={val_acc*100:.2f}%  best={best_val*100:.2f}%")

        if bad_epochs >= patience:           # patience exhausted: stop early
            print(f"  early stop at epoch {epoch+1}")
            break

    return best_state, best_val, best_epoch