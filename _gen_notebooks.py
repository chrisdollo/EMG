"""Generate the three remaining putEMG notebooks."""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.0"},
}


def nb(cells):
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": METADATA, "cells": cells}


def md(id_, src):
    return {"cell_type": "markdown", "id": id_, "metadata": {}, "source": src.splitlines(keepends=True)}


def code(id_, src):
    return {
        "cell_type": "code", "execution_count": None,
        "id": id_, "metadata": {}, "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 1: experiments/efficient/feature_based/model.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb1_cells = [

md("f001", """\
# putEMG — Efficient Feature-Based Model (8-Channel LOSO)

Retrains the LOSO cross-subject evaluation from `experiments/cross_subject/feature_based/`
but using only the **8 representative channels** selected by feature-based agglomerative
clustering (see `experiments/clustering/clustering_features.ipynb`).

**Goal:** match or closely approach the 24-channel feature-based accuracy with one-third
the electrode count — demonstrating that the discriminative channel selection works for
hand-crafted features, not just deep learning.

| Input | 24-channel baseline | This notebook |
|-------|-------------------|--------------|
| Shape | `(N, 26, 3192)` | `(N, 26, ≈1064)` |
| Source | `load_feature_subjects` | slice raw → extract 8-ch features |

**Prerequisites:**
1. `data_preprocessing/driver.ipynb` — per-subject `.mat` files
2. `experiments/clustering/clustering_features.ipynb` — `feat_representative_channels.npy`

Set `MODEL_TYPE` in the config cell to switch between `FeatureLSTM`, `FeatureGRU`, `FeatureTransformer`.
Completed folds are skipped automatically — safe to stop and resume.\
"""),

code("f002", """\
import os
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau\
"""),

code("f003", """\
# ── Shared utilities (src/) ────────────────────────────────────────────────────
sys.path.append(os.path.abspath('../../../'))

import src.feature_based_models as fbm
from src.emg_loader import load_all_subjects, load_feature_subjects, make_loso_train_val_test
from src.feature_extraction import batch_extract\
"""),

code("f004", """\
# ── Load channel selection from feature clustering notebook ────────────────────
CLUSTERING_DIR = os.path.abspath('../../clustering')
channels_path  = os.path.join(CLUSTERING_DIR, 'feat_representative_channels.npy')

if not os.path.exists(channels_path):
    raise FileNotFoundError(
        f'Channel selection file not found: {channels_path}\\n'
        'Run experiments/clustering/clustering_features.ipynb first.'
    )

CHANNELS = np.load(channels_path)   # (8,) — 0-indexed channel indices
N_CH     = len(CHANNELS)
print(f'Using {N_CH} channels (0-indexed): {list(CHANNELS)}')
print(f'(1-indexed): {[c + 1 for c in CHANNELS]}')\
"""),

code("f005", """\
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
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
            correct += (model(X).argmax(dim=1) == y).sum().item()
            total   += y.size(0)
    return correct / total\
"""),

code("f006", """\
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')\
"""),

code("f007", """\
_notebook_dir = os.path.abspath(os.getcwd())

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = '/Volumes/KRIS/data/UG_per_subject'
# Separate feature cache for 8-channel subjects — keeps full-channel cache intact
FEATURE_DIR_8CH = '/Volumes/KRIS/data/features_8ch_sequence'
MODE            = 'sequence'

MODEL_TYPE  = 'FeatureLSTM'   # 'FeatureGRU' | 'FeatureTransformer'
WEIGHTS_DIR = os.path.join(_notebook_dir, 'weights', MODEL_TYPE)
RESULTS_DIR = os.path.join(_notebook_dir, 'results')
LOG_PATH    = os.path.join(RESULTS_DIR, f'results_log_{MODEL_TYPE}.txt')

BATCH_SIZE = 16
VAL_FRAC   = 0.10
MAX_EPOCHS = 20
PATIENCE   = 5
MIN_DELTA  = 0.002
LR         = 1e-3
DROPOUT    = 0.3

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR_8CH, exist_ok=True)
print(f'Weights → {WEIGHTS_DIR}')
print(f'Log     → {LOG_PATH}')\
"""),

md("f008", """\
---
## Feature Extraction (8 Channels)

Slices raw subjects to the 8 representative channels, then extracts libemg features.
Output is cached in `FEATURE_DIR_8CH` — subjects with existing files are skipped.
The `INPUT_SIZE` for the models is derived dynamically from the loaded data shape.\
"""),

code("f009", """\
# Slice raw subjects to 8 representative channels before extraction
# X per subject: (N_reps, 1, 24, 1500) → (N_reps, 1, 8, 1500)
raw_subjects_full = load_all_subjects(DATA_DIR)
subjects_8ch_raw  = [(name, X[:, :, CHANNELS, :], y) for name, X, y in raw_subjects_full]

expected = len(subjects_8ch_raw)
present  = len([f for f in os.listdir(FEATURE_DIR_8CH) if f.endswith(f'_{MODE}.npz')])
print(f'Subjects          : {expected}')
print(f'Cached (8-ch)     : {present}  ({FEATURE_DIR_8CH})')

if present < expected:
    print(f'\\nExtracting 8-channel features for {expected - present} subject(s)...')
    batch_extract(subjects_8ch_raw, output_dir=FEATURE_DIR_8CH, mode=MODE)
else:
    print('All 8-channel feature files present — skipping extraction.')

subjects   = load_feature_subjects(FEATURE_DIR_8CH, mode=MODE)
INPUT_SIZE = subjects[0][1].shape[2]   # (N_reps, 26, input_size) — computed from data
print(f'\\nInput size (8-channel features): {INPUT_SIZE}')
print(f'Subject 0 X shape: {subjects[0][1].shape}')\
"""),

md("f010", """\
---
## LOSO Training Loop

Identical protocol to `experiments/cross_subject/feature_based/` — only the
feature dimensionality changes from ~3192 (24 channels) to ~1064 (8 channels).

- **Test** — 1 held-out subject, never seen during training
- **Train / Val** — all other 43 subjects, split 90/10 (stratified by class, seeded)
- Checkpointing: fold skipped if `weights/<MODEL_TYPE>/<MODEL_TYPE>_<id>.pt` already exists\
"""),

code("f011", """\
MODEL_MAP = {
    'FeatureLSTM':        fbm.FeatureLSTM,
    'FeatureGRU':         fbm.FeatureGRU,
    'FeatureTransformer': fbm.FeatureTransformer,
}


def subject_id(filename):
    # 'features_subject_03_sequence.npz' -> '03'
    return filename.split('_')[2]


def weight_path(subject_name):
    return os.path.join(WEIGHTS_DIR, f'{MODEL_TYPE}_{subject_id(subject_name)}.pt')


def update_log():
    checkpoints = []
    for fname in sorted(os.listdir(WEIGHTS_DIR)):
        if not fname.endswith('.pt'):
            continue
        ckpt = torch.load(os.path.join(WEIGHTS_DIR, fname), map_location='cpu', weights_only=False)
        if 'subject' not in ckpt:
            continue
        checkpoints.append(ckpt)

    if not checkpoints:
        return

    accs   = [c['test_acc'] * 100 for c in checkpoints]
    n_done = len(checkpoints)

    lines = [
        f'putEMG — {MODEL_TYPE} LOSO Results  (8-channel efficient feature-based model)',
        '=' * 72,
        f"{'Subject':<32} {'Test Acc':>9}  {'Val Acc':>9}  {'Epoch':>6}  {'Date'}",
        '-' * 72,
    ]
    for c in checkpoints:
        lines.append(
            f"{c['subject']:<32} {c['test_acc']*100:>8.2f}%  "
            f"{c['val_acc']*100:>8.2f}%  {c['best_epoch']:>6}  {c['date']}"
        )
    lines += [
        '=' * 72,
        f"Mean: {np.mean(accs):.2f}%  \\u00b1  {np.std(accs):.2f}%  "
        f"({n_done} / {len(subjects)} folds complete)",
    ]

    with open(LOG_PATH, 'w') as f:
        f.write('\\n'.join(lines) + '\\n')

    print(f'Log updated \\u2192 {LOG_PATH}  ({n_done}/{len(subjects)} folds)')\
"""),

code("f012", """\
for test_idx, (test_name, _, _) in enumerate(subjects):
    wpath = weight_path(test_name)

    if os.path.exists(wpath):
        print(f'[SKIP] {test_name}  — checkpoint found')
        continue

    print(f"\\n{'='*60}")
    print(f'  Fold {test_idx+1}/{len(subjects)}  —  test: {test_name}')
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = make_loso_train_val_test(
        subjects, test_idx, val_frac=VAL_FRAC, batch_size=BATCH_SIZE
    )

    # Pass INPUT_SIZE so the model matches the 8-channel feature dimensionality
    model     = MODEL_MAP[MODEL_TYPE](input_size=INPUT_SIZE, dropout_rate=DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = float('-inf')
    best_state   = None
    best_epoch   = 0
    bad_epochs   = 0

    for epoch in range(MAX_EPOCHS):
        tr_loss = train(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        curr_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch + 1

        scheduler.step(val_acc)
        bad_epochs = 0 if val_acc >= (best_val_acc - MIN_DELTA) else bad_epochs + 1

        print(f'  Epoch {epoch+1:3d}: loss={tr_loss:.4f}  val={val_acc*100:.2f}%  '
              f'best={best_val_acc*100:.2f}%  lr={curr_lr:.2e}')

        if bad_epochs >= PATIENCE:
            print(f'  Early stopping at epoch {epoch+1}.')
            break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    print(f'\\n  Test accuracy: {test_acc*100:.2f}%')

    torch.save({
        'subject':    test_name,
        'test_acc':   test_acc,
        'val_acc':    best_val_acc,
        'best_epoch': best_epoch,
        'channels':   list(CHANNELS),
        'n_channels': N_CH,
        'input_size': INPUT_SIZE,
        'dropout':    DROPOUT,
        'state_dict': best_state,
        'date':       datetime.date.today().isoformat(),
    }, wpath)

    update_log()

print(f"\\n{'='*60}")
print('  All folds complete.')
print(f"{'='*60}")
update_log()\
"""),

md("f013", """\
---
## Results\
"""),

code("f014", """\
if os.path.exists(LOG_PATH):
    with open(LOG_PATH) as f:
        print(f.read())
else:
    print('No results yet — run the training loop first.')\
"""),

code("f015", """\
checkpoints = []
for fname in sorted(os.listdir(WEIGHTS_DIR)):
    if not fname.endswith('.pt'):
        continue
    ckpt = torch.load(os.path.join(WEIGHTS_DIR, fname), map_location='cpu', weights_only=False)
    if 'subject' in ckpt:
        checkpoints.append(ckpt)

if checkpoints:
    sids = [subject_id(c['subject']) for c in checkpoints]
    accs = [c['test_acc'] * 100 for c in checkpoints]
    mean = np.mean(accs)

    fig, ax = plt.subplots(figsize=(max(10, len(sids) * 0.45), 4))
    ax.bar(sids, accs, color='steelblue')
    ax.axhline(mean, color='tomato', linestyle='--', linewidth=1.5, label=f'Mean {mean:.1f}%')
    ax.set_ylim(0, 105)
    ax.set_xlabel('Test Subject')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'{MODEL_TYPE} LOSO — 8-Channel Efficient Feature Model  ({len(checkpoints)}/{len(subjects)} folds)')
    ax.legend()
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.show()\
"""),

code("f016", """\
# ── Compare 8-channel vs 24-channel feature-based baseline ────────────────────
baseline_weights_dir = os.path.abspath(
    f'../../cross_subject/feature_based/weights/{MODEL_TYPE}'
)

if os.path.isdir(baseline_weights_dir) and checkpoints:
    baseline_ckpts = []
    for fname in sorted(os.listdir(baseline_weights_dir)):
        if not fname.endswith('.pt'):
            continue
        ckpt = torch.load(os.path.join(baseline_weights_dir, fname),
                          map_location='cpu', weights_only=False)
        if 'subject' in ckpt:
            baseline_ckpts.append(ckpt)

    if baseline_ckpts:
        eff_map  = {subject_id(c['subject']): c['test_acc'] * 100 for c in checkpoints}
        base_map = {subject_id(c['subject']): c['test_acc'] * 100 for c in baseline_ckpts}
        common   = sorted(set(eff_map) & set(base_map))

        eff_accs  = [eff_map[s]  for s in common]
        base_accs = [base_map[s] for s in common]
        drops     = [b - e for b, e in zip(base_accs, eff_accs)]

        print(f'\\n{MODEL_TYPE} — 24-channel vs 8-channel feature model  ({len(common)} subjects)')
        print(f'  24-channel mean : {np.mean(base_accs):.2f}%')
        print(f'   8-channel mean : {np.mean(eff_accs):.2f}%')
        print(f'  Accuracy drop   : {np.mean(drops):+.2f} pp  (mean per subject)')

        x = np.arange(len(common))
        w = 0.38
        fig, ax = plt.subplots(figsize=(max(12, len(common) * 0.5), 4))
        ax.bar(x - w/2, base_accs, w, label='24-channel baseline', color='steelblue')
        ax.bar(x + w/2, eff_accs,  w, label='8-channel efficient',  color='darkorange')
        ax.set_xticks(x)
        ax.set_xticklabels(common, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Test Subject')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{MODEL_TYPE}: 24-channel vs 8-channel Feature-Based LOSO')
        ax.legend()
        plt.tight_layout()
        plt.show()
else:
    print('Run training first, or ensure cross_subject/feature_based results exist for comparison.')\
"""),

]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 2: experiments/clustering/anatomical_validation.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb2_cells = [

md("a001", """\
# EMG Channel Clustering — Anatomical Validation

Validates that the 8 representative channels selected by agglomerative clustering
provide good **anatomical coverage** of the forearm.

**Rationale:** A good channel reduction should:
1. Cover all major forearm muscle groups (no anatomical blind spots)
2. Select channels from distinct angular positions around the forearm
3. Produce clusters where members are physically close (nearby channels are redundant)

**Electrode layout (putEMG bracelet):**
- 24 channels in 2 rows × 12 columns around the forearm circumference
- Row 1 (Ch 1–12, proximal) and Row 2 (Ch 13–24, distal)
- Each column spans 30° of arc — 12 angular positions total

**Prerequisites:** Run `clustering.ipynb` and `clustering_features.ipynb` first.\
"""),

code("a002", """\
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Path to shared src/ — clustering notebooks are at depth 2 from root
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
SRC       = os.path.join(REPO_ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

print('Imports OK.')\
"""),

code("a003", """\
CLUSTERING_DIR = os.path.abspath(os.getcwd())
N_CHANNELS     = 24
N_CLUSTERS     = 8
N_PER_ROW      = 12   # electrodes per ring

# Load DL and feature-based clustering results
dl_channels_path   = os.path.join(CLUSTERING_DIR, 'dl_representative_channels.npy')
dl_labels_path     = os.path.join(CLUSTERING_DIR, 'dl_cluster_labels.npy')
feat_channels_path = os.path.join(CLUSTERING_DIR, 'feat_representative_channels.npy')
feat_labels_path   = os.path.join(CLUSTERING_DIR, 'feat_cluster_labels.npy')

missing = [p for p in [dl_channels_path, dl_labels_path, feat_channels_path, feat_labels_path]
           if not os.path.exists(p)]

if missing:
    print('Missing files — run clustering.ipynb and clustering_features.ipynb first:')
    for p in missing:
        print(f'  {p}')
    raise SystemExit(0)

dl_reps     = np.load(dl_channels_path).astype(int)
dl_labels   = np.load(dl_labels_path).astype(int)
feat_reps   = np.load(feat_channels_path).astype(int)
feat_labels = np.load(feat_labels_path).astype(int)

print(f'DL reps   (0-idx): {list(dl_reps)}  →  1-indexed: {[c+1 for c in dl_reps]}')
print(f'Feat reps (0-idx): {list(feat_reps)}  →  1-indexed: {[c+1 for c in feat_reps]}')\
"""),

md("a004", """\
---
## Electrode Layout

The putEMG bracelet places 24 electrodes in two concentric rings around the forearm.
Each ring has 12 electrodes spaced 30° apart. The angular positions are approximate
and subject to bracelet placement, but the relative ordering is preserved.\
"""),

code("a005", """\
# putEMG bracelet geometry:
# Row 0 (proximal, Ch 1-12):  angular positions 0, 30, 60, ..., 330 degrees
# Row 1 (distal,   Ch 13-24): same angular positions as row 0
ANGLES_DEG = np.array([i * 30 for i in range(N_PER_ROW)] * 2)  # (24,)
ROWS       = np.array([0] * N_PER_ROW + [1] * N_PER_ROW)       # 0=proximal, 1=distal

# Approximate muscle group regions visible at mid-forearm level.
# Starting from the radial/dorsal aspect (0°), going clockwise (viewed from distal).
# These are estimates; exact positions vary with bracelet placement and subject anatomy.
MUSCLE_REGIONS = [
    (  0,  60, 'Brachioradialis\\n& ECRL',         '#d4e6f1'),
    ( 60, 150, 'Extensor Digitorum\\nCommunis',     '#d5f5e3'),
    (150, 210, 'ECU & FCU\\nboundary',              '#fdebd0'),
    (210, 300, 'Flexor Digitorum\\n(FDS/FDP)',      '#f9ebea'),
    (300, 360, 'FCR & Palmaris\\nLongus',           '#e8daef'),
]

print('Channel positions:')
for ch in range(N_CHANNELS):
    row_str = 'proximal' if ROWS[ch] == 0 else 'distal  '
    print(f'  Ch{ch+1:>2}  {row_str}  {ANGLES_DEG[ch]:>3}°')\
"""),

code("a006", """\
def plot_electrode_ring(cluster_labels, rep_channels, title, save_path=None):
    # Polar plot of the 24-electrode forearm bracelet, coloured by cluster.
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    colors  = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))

    # Background muscle-region shading
    for start_deg, end_deg, name, color in MUSCLE_REGIONS:
        # polar 0° is at 3 o'clock, angles are counter-clockwise;
        # we want 0° at 12 o'clock (north) going clockwise → transform = 90 - angle
        th_start = np.radians(90 - end_deg)
        th_end   = np.radians(90 - start_deg)
        # fill_between expects sorted thetas
        thetas = np.linspace(min(th_start, th_end), max(th_start, th_end), 50)
        ax.fill_between(thetas, 0.5, 1.25, color=color, alpha=0.35, zorder=0)
        mid_deg = (start_deg + end_deg) / 2
        mid_rad = np.radians(90 - mid_deg)
        ax.text(mid_rad, 1.22, name, ha='center', va='center',
                fontsize=8, color='dimgrey', zorder=1)

    rep_set = set(map(int, rep_channels))

    for ch_idx in range(N_CHANNELS):
        angle_rad = np.radians(90 - ANGLES_DEG[ch_idx])
        r         = 1.0 if ROWS[ch_idx] == 0 else 0.72
        cluster   = cluster_labels[ch_idx]
        color     = colors[cluster]
        is_rep    = ch_idx in rep_set

        ax.scatter(angle_rad, r,
                   s=220 if is_rep else 70,
                   c=[color],
                   marker='*' if is_rep else 'o',
                   edgecolors='black',
                   linewidths=1.5 if is_rep else 0.6,
                   zorder=5 if is_rep else 3)

        label_r  = r + (0.14 if is_rep else 0.10)
        row_mark = '' if ROWS[ch_idx] == 0 else "'"
        ax.text(angle_rad, label_r, f'Ch{ch_idx+1}{row_mark}',
                ha='center', va='center',
                fontsize=7.5, fontweight='bold' if is_rep else 'normal', zorder=6)

    # Legend
    cluster_handles = [mpatches.Patch(color=colors[c], label=f'Cluster {c}')
                       for c in range(N_CLUSTERS)]
    rep_handle = plt.Line2D([0], [0], marker='*', color='w',
                             markerfacecolor='grey', markeredgecolor='black',
                             markersize=13, label='Representative (★)')
    ax.legend(handles=cluster_handles + [rep_handle],
              loc='upper right', bbox_to_anchor=(1.42, 1.12), fontsize=8.5)

    ax.set_ylim(0, 1.35)
    ax.set_yticks([])
    degree_ticks = list(range(0, 360, 30))
    ax.set_xticks([np.radians(90 - d) for d in degree_ticks])
    ax.set_xticklabels([f'{d}°' for d in degree_ticks], fontsize=8)
    ax.set_title(title, pad=22, fontsize=13)
    ax.grid(alpha=0.25)

    # Ring labels at center
    ax.text(0, 1.02, 'Proximal', ha='center', va='center',
            fontsize=8, color='steelblue', style='italic')
    ax.text(0, 0.72, 'Distal', ha='center', va='center',
            fontsize=8, color='darkorange', style='italic')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


plot_electrode_ring(dl_labels, dl_reps,
                    f'DL Clustering — {N_CLUSTERS} Clusters (Raw Signal)',
                    save_path='anatomical_dl.png')\
"""),

code("a007", """\
plot_electrode_ring(feat_labels, feat_reps,
                    f'Feature Clustering — {N_CLUSTERS} Clusters (Hand-Crafted Features)',
                    save_path='anatomical_feat.png')\
"""),

md("a008", """\
---
## Angular Coverage Analysis

A good 8-channel selection should have no large angular gaps — otherwise an entire
muscle group is unsampled. The **max angular gap** between consecutive representatives
(in degrees) is the key metric: values below 90° mean all four forearm quadrants
are represented.\
"""),

code("a009", """\
def angular_coverage(rep_channels):
    # Returns angular coverage metrics for a set of representative channels.
    angles       = np.sort(np.unique(ANGLES_DEG[list(rep_channels)]))
    if len(angles) == 0:
        return {}
    # Circular gaps: include wrap-around from last to first + 360
    gaps         = np.diff(np.concatenate([angles, [angles[0] + 360]]))
    uniformity   = 1.0 - (gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    return {
        'n_distinct_angles': int(len(angles)),
        'max_gap_deg':       float(gaps.max()),
        'mean_gap_deg':      float(gaps.mean()),
        'uniformity_score':  float(uniformity),  # 1.0 = perfectly uniform
    }


print('Angular coverage (0° = no gap, 90° = one quadrant missing)\\n')

dl_cov   = angular_coverage(dl_reps)
feat_cov = angular_coverage(feat_reps)

header = f"{'Metric':<28}  {'DL':>10}  {'Feature':>10}"
print(header)
print('-' * len(header))
for k in dl_cov:
    dl_v   = dl_cov[k]
    feat_v = feat_cov[k]
    if isinstance(dl_v, float):
        print(f'  {k:<26}  {dl_v:>9.1f}  {feat_v:>9.1f}')
    else:
        print(f'  {k:<26}  {dl_v:>9}  {feat_v:>9}')

print()
for name, cov in [('DL', dl_cov), ('Feature', feat_cov)]:
    gap = cov['max_gap_deg']
    flag = 'OK — all quadrants covered' if gap < 90 else f'WARNING — {gap:.0f}° gap'
    print(f'  {name} max gap: {gap:.0f}°  ({flag})')\
"""),

md("a010", """\
---
## Cluster Composition Analysis

Shows which physical channels belong to each cluster and which one was selected
as the representative (medoid). Good clusters have high intra-cluster correlation
(all members carry similar information) and representative channels that span
the full circumference of the forearm.\
"""),

code("a011", """\
def cluster_summary(cluster_labels, rep_channels, label=''):
    print(f'Cluster composition — {label}')
    print(f"  {'Cluster':>7}  {'Members (1-indexed)':>24}  {'Rep':>5}  {'Angle':>6}  {'Row'}")
    print('  ' + '-' * 58)

    rep_set = set(map(int, rep_channels))
    for c in range(N_CLUSTERS):
        members = np.where(cluster_labels == c)[0]
        # The representative is whichever member is in rep_channels
        reps_in_cluster = [m for m in members if m in rep_set]
        rep_str   = f'Ch{reps_in_cluster[0]+1}' if reps_in_cluster else '?'
        rep_angle = ANGLES_DEG[reps_in_cluster[0]] if reps_in_cluster else -1
        rep_row   = 'prox' if (reps_in_cluster and ROWS[reps_in_cluster[0]] == 0) else 'dist'
        members_1 = sorted(m + 1 for m in members)
        print(f'  {c:>7}  {str(members_1):>24}  {rep_str:>5}  {rep_angle:>5}°  {rep_row}')
    print()


cluster_summary(dl_labels,   dl_reps,   label='DL (raw signal)')
cluster_summary(feat_labels, feat_reps, label='Feature-based')\
"""),

md("a012", """\
---
## DL vs Feature-Based Selection Comparison

Compares which channels were selected by each approach. A high Jaccard similarity
means both approaches agree on the informative channels, providing cross-validation
evidence that those channels are genuinely non-redundant.\
"""),

code("a013", """\
dl_set   = set(map(int, dl_reps))
feat_set = set(map(int, feat_reps))

overlap   = dl_set & feat_set
dl_only   = dl_set - feat_set
feat_only = feat_set - dl_set
jaccard   = len(overlap) / len(dl_set | feat_set)

print('=== DL vs Feature-Based Channel Selection ===\\n')
print(f'  DL selected    (1-idx): {sorted(c+1 for c in dl_set)}')
print(f'  Feat selected  (1-idx): {sorted(c+1 for c in feat_set)}')
print(f'\\n  Overlap ({len(overlap)} channels)     : {sorted(c+1 for c in overlap)}')
print(f'  DL only  ({len(dl_only)} channels)     : {sorted(c+1 for c in dl_only)}')
print(f'  Feat only ({len(feat_only)} channels)   : {sorted(c+1 for c in feat_only)}')
print(f'\\n  Jaccard similarity: {jaccard:.2f}  (1.0 = identical, 0.0 = no overlap)')\
"""),

code("a014", """\
# Side-by-side angular-position scatter for DL vs Feature selection
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for ax, reps, labels, title in zip(
    axes,
    [dl_reps, feat_reps],
    [dl_labels, feat_labels],
    ['DL Selection (raw signal)', 'Feature-Based Selection'],
):
    colors = plt.cm.tab10(np.linspace(0, 1, N_CLUSTERS))
    rep_set = set(map(int, reps))

    for ch in range(N_CHANNELS):
        cl    = labels[ch]
        is_r  = ch in rep_set
        ax.scatter(ch, ANGLES_DEG[ch],
                   s=160 if is_r else 50,
                   c=[colors[cl]],
                   marker='*' if is_r else ('s' if ROWS[ch] == 1 else 'o'),
                   edgecolors='black', linewidths=0.8 if is_r else 0.3,
                   zorder=4 if is_r else 2, label=f'Cluster {cl}')

    ax.set_xlabel('Channel index (0-based)', fontsize=10)
    ax.set_ylabel('Angular position (degrees)', fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_yticks(range(0, 360, 30))
    ax.set_ylim(-15, 375)
    ax.axhspan(  0,  90, alpha=0.07, color='blue',   label='Dorsal quadrant')
    ax.axhspan( 90, 180, alpha=0.07, color='green',  label='Ulnar quadrant')
    ax.axhspan(180, 270, alpha=0.07, color='red',    label='Palmar quadrant')
    ax.axhspan(270, 360, alpha=0.07, color='purple', label='Radial quadrant')
    ax.grid(alpha=0.25)

plt.suptitle('Angular Coverage: DL vs Feature-Based Channel Selection', fontsize=13)
plt.tight_layout()
plt.savefig('anatomical_comparison.png', dpi=150)
plt.show()\
"""),

code("a015", """\
# ── Summary ───────────────────────────────────────────────────────────────────
print('=== Anatomical Validation Summary ===\\n')

for name, reps in [('DL', dl_reps), ('Feature-based', feat_reps)]:
    print(f'{name} representatives (1-indexed):')
    for ch in sorted(reps):
        angle  = ANGLES_DEG[ch]
        row    = 'proximal' if ROWS[ch] == 0 else 'distal  '
        region = next(
            (r.replace('\\n', ' ') for s, e, r, _ in MUSCLE_REGIONS if s <= angle < e),
            'Brachioradialis/FCR',
        )
        print(f'  Ch{ch+1:>2}  {angle:>3}°  {row}  ≈ {region}')
    cov = angular_coverage(reps)
    print(f'  → max gap {cov["max_gap_deg"]:.0f}°  |  uniformity {cov["uniformity_score"]:.2f}\\n')\
"""),

]

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 3: experiments/analysis/per_class_accuracy.ipynb
# ─────────────────────────────────────────────────────────────────────────────

nb3_cells = [

md("p001", """\
# putEMG — Per-Subject / Per-Class Accuracy Analysis

Deep-dives into **which gestures are hard** and **which subjects are outliers** in the
cross-subject LOSO experiments.

Sections:
1. **Summary from logs** — lightweight, no inference required
2. **Model comparison** — per-subject accuracy across EMG_TCN / EEGNet / ShallowConvNet
3. **Per-class inference** — loads saved checkpoints and collects per-class predictions
4. **Confusion matrices** — per model (aggregated across subjects)
5. **Per-class accuracy heatmap** — subjects × gesture classes
6. **Outlier analysis** — best/worst subjects per gesture

**Prerequisites:** Run `experiments/cross_subject/deep_learning/model.ipynb` for all three models first.\
"""),

code("p002", """\
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import torch
import torch.nn as nn\
"""),

code("p003", """\
# ── Shared utilities ──────────────────────────────────────────────────────────
sys.path.append(os.path.abspath('../../'))

import src.deep_learning_models as dlm
from src.emg_loader import load_all_subjects, BCIDataset
from torch.utils.data import DataLoader\
"""),

code("p004", """\
_notebook_dir = os.path.abspath(os.getcwd())

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = '/Volumes/KRIS/data/UG_per_subject'
CS_DL_DIR      = os.path.abspath('../cross_subject/deep_learning')
WS_DL_DIR      = os.path.abspath('../within_subject/deep_learning')
RESULTS_DIR    = os.path.join(_notebook_dir, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Gesture names as used in the putEMG dataset (class indices 0-6)
GESTURE_NAMES = ['G1 Open', 'G2 Fist', 'G3 Pinch', 'G6 Ext', 'G7 Flex', 'G8 RadDev', 'G9 UlnDev']
NUM_CLASSES   = len(GESTURE_NAMES)

MODELS_CS = ['EMG_TCN', 'EEGNet', 'ShallowConvNet']   # cross-subject
MODEL_MAP  = {
    'EMG_TCN':        dlm.EMG_TCN,
    'EEGNet':         dlm.EEGNet,
    'ShallowConvNet': dlm.ShallowConvNet,
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')
print(f'Cross-subject results dir: {CS_DL_DIR}')\
"""),

md("p005", """\
---
## 1. Per-Subject Accuracy Summary (from log files)\
"""),

code("p006", """\
def load_checkpoints(weights_dir):
    # Load all .pt files in weights_dir and return a list of checkpoint dicts.
    ckpts = []
    if not os.path.isdir(weights_dir):
        return ckpts
    for fname in sorted(os.listdir(weights_dir)):
        if not fname.endswith('.pt'):
            continue
        ckpt = torch.load(os.path.join(weights_dir, fname),
                          map_location='cpu', weights_only=False)
        if 'test_acc' in ckpt:
            ckpts.append(ckpt)
    return ckpts


def subject_id_from_mat(mat_name):
    return mat_name.replace('emg_gestures_', '').replace('_U.mat', '')


# Gather results for all three models
model_results = {}   # model_type → list of (subject_id, test_acc)
for mt in MODELS_CS:
    wdir  = os.path.join(CS_DL_DIR, 'weights', mt)
    ckpts = load_checkpoints(wdir)
    model_results[mt] = [(subject_id_from_mat(c['subject']), c['test_acc'] * 100)
                         for c in ckpts]
    n = len(ckpts)
    accs = [c['test_acc'] * 100 for c in ckpts]
    print(f'  {mt:<18} {n:>2} folds  mean={np.mean(accs):.2f}%  std={np.std(accs):.2f}%'
          if accs else f'  {mt:<18} 0 folds  (no checkpoints found)')\
"""),

code("p007", """\
# Per-subject bar chart — all models side by side
all_sids = sorted({sid for sids_accs in model_results.values()
                   for sid, _ in sids_accs})
x        = np.arange(len(all_sids))
n_models = len(MODELS_CS)
w        = 0.25
colors   = ['steelblue', 'darkorange', 'seagreen']

fig, ax = plt.subplots(figsize=(max(14, len(all_sids) * 0.6), 5))

for i, (mt, col) in enumerate(zip(MODELS_CS, colors)):
    acc_map = dict(model_results[mt])
    accs    = [acc_map.get(sid, np.nan) for sid in all_sids]
    offset  = (i - n_models / 2 + 0.5) * w
    bars    = ax.bar(x + offset, accs, w, label=f'{mt} ({np.nanmean(accs):.1f}%)',
                     color=col, alpha=0.85, edgecolor='none')

ax.axhline(50, color='grey', linestyle=':', linewidth=1, label='Chance (50%)')
ax.set_xticks(x)
ax.set_xticklabels(all_sids, rotation=45, ha='right', fontsize=8)
ax.set_ylim(0, 105)
ax.set_xlabel('Test Subject')
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Cross-Subject LOSO — Per-Subject Accuracy by Model')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_subject_comparison.png'), dpi=150)
plt.show()\
"""),

code("p008", """\
# Rank subjects: easiest (highest mean accuracy) → hardest (lowest mean)
sid_means = {}
for sid in all_sids:
    vals = [acc for mt in MODELS_CS
            for s, acc in model_results[mt] if s == sid]
    if vals:
        sid_means[sid] = np.mean(vals)

ranked = sorted(sid_means.items(), key=lambda kv: kv[1], reverse=True)
print(f'{"Rank":>4}  {"Subject":>12}  {"Mean Acc":>9}')
print('-' * 30)
for rank, (sid, acc) in enumerate(ranked, 1):
    flag = '  ← easiest' if rank == 1 else ('  ← hardest' if rank == len(ranked) else '')
    print(f'{rank:>4}  {sid:>12}  {acc:>8.2f}%{flag}')\
"""),

md("p009", """\
---
## 2. Per-Class Inference

Loads every saved model checkpoint, runs it on its corresponding test subject, and
collects per-sample predictions. This allows computing confusion matrices and per-class
accuracy — at the cost of running inference on 44 subjects × 3 models.

> **Runtime:** ~10–30 seconds per fold (CPU/MPS inference only, no training).\
"""),

code("p010", """\
def collect_predictions(model_type, subjects_data, cs_dl_dir, device):
    # For every checkpoint of model_type, load the model and run inference
    # on its corresponding test subject.
    # Returns: all_true (N,), all_pred (N,), per_subj list of (sid, true, pred)
    weights_dir = os.path.join(cs_dl_dir, 'weights', model_type)
    if not os.path.isdir(weights_dir):
        print(f'  No weights directory for {model_type} — skipping.')
        return np.array([]), np.array([]), []

    # Build a fast lookup from subject_id → (X, y) for the test set
    subj_lookup = {subject_id_from_mat(name): (X, y)
                   for name, X, y in subjects_data}

    ModelClass = MODEL_MAP[model_type]
    all_true, all_pred, per_subj = [], [], []

    for fname in sorted(os.listdir(weights_dir)):
        if not fname.endswith('.pt'):
            continue
        ckpt = torch.load(os.path.join(weights_dir, fname),
                          map_location='cpu', weights_only=False)
        if 'subject' not in ckpt or 'state_dict' not in ckpt:
            continue

        sid = subject_id_from_mat(ckpt['subject'])
        if sid not in subj_lookup:
            print(f'  [WARN] {sid} not found in loaded subjects — skipping.')
            continue

        X_test, y_test = subj_lookup[sid]

        # Instantiate model and load weights
        model = ModelClass(num_classes=NUM_CLASSES, num_channels=24,
                           dropout_rate=ckpt.get('dropout', 0.1))
        model.load_state_dict(ckpt['state_dict'])
        model.eval().to(device)

        loader = DataLoader(BCIDataset(X_test, y_test), batch_size=32, shuffle=False)

        true_batch, pred_batch = [], []
        with torch.no_grad():
            for X_b, y_b in loader:
                preds = model(X_b.to(device)).argmax(dim=1).cpu()
                pred_batch.append(preds.numpy())
                true_batch.append(y_b.numpy())

        true_np = np.concatenate(true_batch)
        pred_np = np.concatenate(pred_batch)
        all_true.append(true_np)
        all_pred.append(pred_np)
        per_subj.append((sid, true_np, pred_np))
        print(f'  {model_type}  {sid}  acc={np.mean(true_np == pred_np)*100:.1f}%')

    return (np.concatenate(all_true) if all_true else np.array([]),
            np.concatenate(all_pred) if all_pred else np.array([]),
            per_subj)\
"""),

code("p011", """\
print('Loading raw subject data...')
subjects_data = load_all_subjects(DATA_DIR)
print(f'Loaded {len(subjects_data)} subjects.\\n')

predictions = {}   # model_type → (all_true, all_pred, per_subj)

for mt in MODELS_CS:
    print(f'--- {mt} ---')
    predictions[mt] = collect_predictions(mt, subjects_data, CS_DL_DIR, device)
    print()\
"""),

md("p012", """\
---
## 3. Confusion Matrices (Aggregated)\
"""),

code("p013", """\
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, len(MODELS_CS), figsize=(6 * len(MODELS_CS), 5))
if len(MODELS_CS) == 1:
    axes = [axes]

for ax, mt in zip(axes, MODELS_CS):
    all_true, all_pred, _ = predictions[mt]
    if len(all_true) == 0:
        ax.set_title(f'{mt}\\n(no data)')
        continue

    cm   = confusion_matrix(all_true, all_pred, normalize='true')
    acc  = np.mean(all_true == all_pred) * 100
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES,
                ax=ax, cbar=False, linewidths=0.3)
    ax.set_title(f'{mt}  (mean {acc:.1f}%)', fontsize=11)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.suptitle('Normalised Confusion Matrices — Cross-Subject LOSO (all subjects pooled)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.show()\
"""),

md("p014", """\
---
## 4. Per-Class Accuracy Breakdown\
"""),

code("p015", """\
# Per-class accuracy for each model
print(f"{'Gesture':<14}", end='')
for mt in MODELS_CS:
    print(f'  {mt:>16}', end='')
print()
print('-' * (14 + 18 * len(MODELS_CS)))

per_class_accs = {}   # model_type → (N_classes,) array

for mt in MODELS_CS:
    all_true, all_pred, _ = predictions[mt]
    if len(all_true) == 0:
        per_class_accs[mt] = np.full(NUM_CLASSES, np.nan)
        continue
    accs = np.array([
        np.mean(all_pred[all_true == c] == c) * 100
        if np.any(all_true == c) else np.nan
        for c in range(NUM_CLASSES)
    ])
    per_class_accs[mt] = accs

for g in range(NUM_CLASSES):
    print(f'{GESTURE_NAMES[g]:<14}', end='')
    for mt in MODELS_CS:
        v = per_class_accs[mt][g]
        print(f'  {v:>15.2f}%', end='')
    print()

print()
print(f"{'Overall':<14}", end='')
for mt in MODELS_CS:
    all_true, all_pred, _ = predictions[mt]
    v = np.mean(all_true == all_pred) * 100 if len(all_true) else np.nan
    print(f'  {v:>15.2f}%', end='')
print()\
"""),

code("p016", """\
# Per-class bar chart
x        = np.arange(NUM_CLASSES)
w        = 0.25
colors   = ['steelblue', 'darkorange', 'seagreen']

fig, ax = plt.subplots(figsize=(11, 5))
for i, (mt, col) in enumerate(zip(MODELS_CS, colors)):
    accs   = per_class_accs.get(mt, np.full(NUM_CLASSES, np.nan))
    offset = (i - len(MODELS_CS) / 2 + 0.5) * w
    ax.bar(x + offset, accs, w, label=mt, color=col, alpha=0.85, edgecolor='none')

ax.axhline(100 / NUM_CLASSES, color='grey', linestyle=':', linewidth=1,
           label=f'Chance ({100/NUM_CLASSES:.1f}%)')
ax.set_xticks(x)
ax.set_xticklabels(GESTURE_NAMES, rotation=25, ha='right', fontsize=9)
ax.set_ylim(0, 105)
ax.set_ylabel('Per-Class Accuracy (%)')
ax.set_title('Cross-Subject LOSO — Per-Class Accuracy by Model')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'per_class_accuracy.png'), dpi=150)
plt.show()\
"""),

md("p017", """\
---
## 5. Per-Subject Per-Class Heatmap\
"""),

code("p018", """\
# For the best-performing model, build a subjects × gesture heatmap
best_model = max(MODELS_CS, key=lambda mt: (
    np.mean(predictions[mt][0] == predictions[mt][1]) if len(predictions[mt][0]) else -1
))

_, _, per_subj = predictions[best_model]

if per_subj:
    sids_ordered = [sid for sid, _, _ in per_subj]
    heatmap_data = np.full((len(sids_ordered), NUM_CLASSES), np.nan)

    for row_idx, (sid, true_np, pred_np) in enumerate(per_subj):
        for c in range(NUM_CLASSES):
            mask = true_np == c
            if mask.any():
                heatmap_data[row_idx, c] = np.mean(pred_np[mask] == c) * 100

    fig, ax = plt.subplots(figsize=(10, max(6, len(sids_ordered) * 0.35)))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn',
                   vmin=0, vmax=100, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Per-class accuracy (%)', shrink=0.6)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(GESTURE_NAMES, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(sids_ordered)))
    ax.set_yticklabels(sids_ordered, fontsize=7)
    ax.set_title(f'{best_model} — Per-Subject Per-Class Accuracy  (Cross-Subject LOSO)',
                 fontsize=12)
    ax.set_xlabel('Gesture Class')
    ax.set_ylabel('Test Subject')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_subject_per_class_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
else:
    print(f'No predictions available for {best_model}.')\
"""),

md("p019", """\
---
## 6. Outlier Analysis — Best & Worst Subjects per Gesture\
"""),

code("p020", """\
_, _, per_subj = predictions[best_model]

if per_subj:
    # Per-subject per-class accuracy dict: sid → array (NUM_CLASSES,)
    subj_class_acc = {}
    for sid, true_np, pred_np in per_subj:
        subj_class_acc[sid] = np.array([
            np.mean(pred_np[true_np == c] == c) * 100
            if np.any(true_np == c) else np.nan
            for c in range(NUM_CLASSES)
        ])

    print(f'Gesture-level outliers  ({best_model})\\n')
    print(f"{'Gesture':<14}  {'Best subject':>16}  {'Best%':>6}  {'Worst subject':>16}  {'Worst%':>6}")
    print('-' * 68)
    for c in range(NUM_CLASSES):
        vals  = {sid: subj_class_acc[sid][c] for sid in subj_class_acc
                 if not np.isnan(subj_class_acc[sid][c])}
        if not vals:
            continue
        best_sid  = max(vals, key=vals.get)
        worst_sid = min(vals, key=vals.get)
        print(f'{GESTURE_NAMES[c]:<14}  {best_sid:>16}  {vals[best_sid]:>5.1f}%  '
              f'{worst_sid:>16}  {vals[worst_sid]:>5.1f}%')

    # Overall outlier subjects
    subj_means = {sid: np.nanmean(arr) for sid, arr in subj_class_acc.items()}
    easiest    = max(subj_means, key=subj_means.get)
    hardest    = min(subj_means, key=subj_means.get)
    print(f'\\nOverall easiest subject: {easiest}  ({subj_means[easiest]:.1f}%)')
    print(f'Overall hardest subject: {hardest}  ({subj_means[hardest]:.1f}%)')\
"""),

]

# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks
# ─────────────────────────────────────────────────────────────────────────────

notebooks = [
    (os.path.join(ROOT, 'experiments', 'efficient', 'feature_based', 'model.ipynb'), nb1_cells),
    (os.path.join(ROOT, 'experiments', 'clustering', 'anatomical_validation.ipynb'), nb2_cells),
    (os.path.join(ROOT, 'experiments', 'analysis',   'per_class_accuracy.ipynb'),    nb3_cells),
]

for path, cells in notebooks:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f'Wrote: {os.path.relpath(path, ROOT)}')

print('Done.')