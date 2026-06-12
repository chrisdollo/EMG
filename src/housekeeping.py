import os
import re
import glob
import datetime
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# putEMG gesture ids → 0-indexed class labels for CrossEntropyLoss
_GESTURE_TO_CLASS = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4, 8: 5, 9: 6}


def _list_subject_files(folder):
    # _list_subject_files: map subject_id -> filepath in folder, matching both the
    # raw signal files (emg_gestures_XX_*.npz) and feature files (features_XX_*.npz)
    files = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.npz'))):
        m = re.search(r'(?:emg_gestures|features)_(\d+)', os.path.basename(path))
        if m:
            files[m.group(1)] = path
    return files


def _load_combined_subject(train_dir, eval_dir, sid, channels=None):
    # _load_combined_subject: load + concat one subject's train and eval reps,
    # returning model-ready X (N, 1, n_ch, 1500) and 0-indexed labels y (N,).
    # channels (optional): keep only these channel indices (for the 8-channel model)
    parts_X, parts_y = [], []
    for folder in (train_dir, eval_dir):
        files = _list_subject_files(folder)
        if sid in files:
            data = np.load(files[sid])
            parts_X.append(data['X'])      # stored as (N, 24, 1500)
            parts_y.append(data['y'])
    X = np.concatenate(parts_X, axis=0).astype(np.float32)   # (N, 24, 1500)
    y = np.concatenate(parts_y, axis=0)
    X = X[:, np.newaxis, :, :]                               # add channel dim -> (N, 1, 24, 1500)
    if channels is not None:
        X = X[:, :, channels, :]                            # keep selected channels
    y = np.array([_GESTURE_TO_CLASS[int(g)] for g in y], dtype=np.int64)   # gesture id -> 0..6
    return X, y


def _load_combined_feature_subject(train_dir, eval_dir, sid, channels=None, feature_idx=None):
    # _load_combined_feature_subject: load + concat one subject's train and eval
    # feature reps, keeping each rep's windows separate for window-level training.
    # Returns X (N_reps, n_wins, feat) and 0-indexed labels y (N_reps,).
    # channels / feature_idx (optional): keep only these channel / per-channel feature
    # indices — the channel-major layout makes this a clean slice (8-channel model).
    parts_X, parts_y, n_wins, n_fpc = [], [], None, None
    for folder in (train_dir, eval_dir):
        files = _list_subject_files(folder)
        if sid in files:
            data = np.load(files[sid], allow_pickle=True)
            parts_X.append(data['X'])                    # (N, n_wins * feat)
            parts_y.append(data['y'])
            n_wins = int(data['n_windows_per_rep'])
            n_fpc  = int(data['n_features_per_ch'])       # 41
    X = np.concatenate(parts_X, axis=0).astype(np.float32)
    y = np.concatenate(parts_y, axis=0)
    feat = X.shape[1] // n_wins                           # features per window (984)
    X = X.reshape(X.shape[0], n_wins, feat)              # (N_reps, n_wins, feat)
    if channels is not None or feature_idx is not None:
        n_ch = feat // n_fpc                              # 24
        X = X.reshape(X.shape[0], n_wins, n_ch, n_fpc)    # split channel-major block
        if channels is not None:
            X = X[:, :, channels, :]                     # keep selected channels
        if feature_idx is not None:
            X = X[:, :, :, feature_idx]                  # keep selected per-channel features
        X = X.reshape(X.shape[0], n_wins, -1)            # flatten back to (N, n_wins, feat')
    y = np.array([_GESTURE_TO_CLASS[int(g)] for g in y], dtype=np.int64)
    return X, y


def _reps_to_windows(X_reps, y_reps):
    # _reps_to_windows: flatten (R, W, feat) reps into (R*W, feat) window samples,
    # repeating each rep's label across its W windows (for window-level training)
    R, W, feat = X_reps.shape
    return X_reps.reshape(R * W, feat), np.repeat(y_reps, W)


def _majority_vote(window_preds, n_wins):
    # _majority_vote: collapse per-window predictions to one label per rep by taking
    # the most common class among each rep's n_wins windows. window_preds: (R*W,)
    rows = np.asarray(window_preds).reshape(-1, n_wins)
    return np.array([np.bincount(row).argmax() for row in rows])


def _mlp_window_preds(model, X_windows, device, batch_size=256):
    # _mlp_window_preds: run a trained model over window feature vectors and return
    # the predicted class per window (used before majority-voting to rep level)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_windows), batch_size):
            xb = torch.from_numpy(X_windows[i:i + batch_size]).to(device)
            preds.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def _pool_except(cache, exclude_sid):
    # _pool_except: concatenate every cached subject's (X, y) except the held-out one
    # — used to build the LOSO training pool. Works for DL (4D) and feature (3D) X.
    Xs = [X for s, (X, _) in cache.items() if s != exclude_sid]
    ys = [y for s, (_, y) in cache.items() if s != exclude_sid]
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def _get_device():
    # _get_device: pick the best available torch device (mps > cuda > cpu)
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _stratified_folds(y, n_folds, seed):
    # _stratified_folds: split indices into n_folds index arrays, balanced per class
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0].copy()
        rng.shuffle(idx)                                  # shuffle within class for randomness
        for f, part in enumerate(np.array_split(idx, n_folds)):
            folds[f].extend(part.tolist())                # deal each class evenly across folds
    return [np.array(f) for f in folds]


def _stratified_val_split(y_pool, val_frac, seed):
    # _stratified_val_split: split pool indices (relative) into train/val, balanced per class
    rng = np.random.default_rng(seed)
    train_rel, val_rel = [], []
    for cls in np.unique(y_pool):
        idx = np.where(y_pool == cls)[0].copy()
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * val_frac))        # at least 1 val sample per class
        val_rel.extend(idx[:n_val])
        train_rel.extend(idx[n_val:])
    return np.array(train_rel), np.array(val_rel)


def _loader(X, y, batch_size, shuffle):
    # _loader: wrap numpy arrays in a torch DataLoader
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# run_type -> short approach tag used in result filenames
_APPROACH_TAG = {'deep_learning': 'DL', 'feature_based': 'feature'}


def _write_results(model_weight_folder, result_folder, run_type, n_total, model_name, *,
                   experiment, protocol, channels=24):
    # _write_results: scan a single model's checkpoint folder and write one result file:
    #   results_{experiment}_{protocol}_{approach}_{model_name}.txt
    # Each .pt in model_weight_folder must have scalar mean_acc / mean_val / mean_epoch.
    # Skips final.pt (the all-subjects deployment model, not a LOSO fold).
    approach = _APPROACH_TAG.get(run_type, run_type)
    rows = {}
    for path in sorted(glob.glob(os.path.join(model_weight_folder, '*.pt'))):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if ck.get('subject') == 'final':
            continue
        rows[ck['subject']] = {
            'test':   ck['mean_acc'],
            'val':    ck.get('mean_val'),
            'epochs': ck.get('mean_epoch'),
        }
    if not rows:
        return

    date_str = datetime.date.today().isoformat()
    os.makedirs(result_folder, exist_ok=True)

    test_accs  = {sid: r['test']   for sid, r in rows.items()}
    val_accs   = {sid: r['val']    for sid, r in rows.items()}
    epoch_accs = {sid: r['epochs'] for sid, r in rows.items()}

    test_vals = list(test_accs.values())
    val_vals  = [v for v in val_accs.values() if v is not None]

    lines = [
        f'phase        = {experiment}',
        f'protocol     = {protocol}',
        f'approach     = {approach}',
        f'model        = {model_name}',
        f'channels     = {channels}',
        f'date         = {date_str}',
        '',
        f'{"subject":<8}  {"val_acc":>8}  {"test_acc":>9}  {"epochs":>7}',
        '-' * 42,
    ]
    for sid in sorted(test_accs):
        v = val_accs.get(sid)
        e = epoch_accs.get(sid)
        val_str   = f'{v*100:>7.2f}%' if v is not None else '     N/A'
        epoch_str = f'{e:>6.1f}'      if e is not None else '    N/A'
        lines.append(f'{sid:<8}  {val_str}  {test_accs[sid]*100:>8.2f}%  {epoch_str}')

    lines += [
        '',
        f'mean_val  = {np.mean(val_vals)*100:.2f}%  ±  {np.std(val_vals)*100:.2f}%' if val_vals else 'mean_val  = N/A',
        f'mean_test = {np.mean(test_vals)*100:.2f}%  ±  {np.std(test_vals)*100:.2f}%',
        f'best      = {max(test_accs, key=test_accs.get)} ({max(test_accs.values())*100:.2f}%)',
        f'worst     = {min(test_accs, key=test_accs.get)} ({min(test_accs.values())*100:.2f}%)',
        f'n         = {len(test_accs)}/{n_total}',
    ]

    out = os.path.join(result_folder,
                       f'results_{experiment}_{protocol}_{approach}_{model_name}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  results → {out}  ({len(rows)}/{n_total} subjects)')
