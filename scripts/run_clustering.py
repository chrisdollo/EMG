"""
Phase 3 — Feature-Space Channel Clustering

Runs the MI-based channel selection and saves:
  clustering & analysis/feat_representative_channels.npy   (8,)
  clustering & analysis/feat_channel_mi_scores.npy         (24,)

Usage:
  python scripts/run_clustering.py
"""

import os
import sys
import warnings
import numpy as np

# Resolve project root (scripts/ is one level below root)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import libemg
from sklearn.feature_selection import mutual_info_classif
from emg_loader import load_subjects_combined

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
DATA_TRAIN = os.path.join(ROOT, 'data', 'processed gestures', 'train')
DATA_EVAL  = os.path.join(ROOT, 'data', 'processed gestures', 'eval')
OUT_DIR    = os.path.join(ROOT, 'clustering & analysis')
WINDOW_SIZE  = 250
WINDOW_SHIFT = 50
N_CHANNELS   = 24
N_SELECT     = 8
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature extractor ─────────────────────────────────────────────────────────
fe           = libemg.feature_extractor.FeatureExtractor()
FEATURE_LIST = fe.get_feature_list()

_dummy_1ch  = np.random.randn(2, 1,  250).astype(np.float32)
F_1CH = fe.extract_features(FEATURE_LIST, _dummy_1ch, array=True).shape[1]
print(f'1-channel libemg output: {F_1CH} scalars/window')

# ── Load all subjects ─────────────────────────────────────────────────────────
subjects = load_subjects_combined(DATA_TRAIN, DATA_EVAL)

X_all = np.concatenate([X for _, X, _ in subjects], axis=0)  # (N_total, 1, 24, 1500)
y_all = np.concatenate([y for _, _, y in subjects], axis=0)

N_REPS = len(y_all)
N_WINS = len(range(0, X_all.shape[-1] - WINDOW_SIZE + 1, WINDOW_SHIFT))

print(f'Total reps: {N_REPS}   Windows/rep: {N_WINS}')

# ── Per-channel MI ────────────────────────────────────────────────────────────
channel_mi    = np.zeros(N_CHANNELS)
channel_feats = np.zeros((N_CHANNELS, N_REPS, F_1CH), dtype=np.float32)

print(f'\nProcessing {N_CHANNELS} channels × {N_REPS} reps × {N_WINS} windows...')

for ch in range(N_CHANNELS):
    ring_name = ['Elbow', 'Middle', 'Wrist'][ch // 8]

    all_windows = []
    for rep_idx in range(N_REPS):
        sig    = X_all[rep_idx, 0, ch, :]
        starts = range(0, len(sig) - WINDOW_SIZE + 1, WINDOW_SHIFT)
        wins   = np.stack([sig[s:s + WINDOW_SIZE] for s in starts])
        all_windows.append(wins)

    all_wins_ch = np.concatenate(all_windows, axis=0)[:, np.newaxis, :]

    feats_flat = fe.extract_features(FEATURE_LIST,
                                     all_wins_ch.astype(np.float32),
                                     array=True)
    feats_flat = np.nan_to_num(feats_flat, nan=0.0, posinf=0.0, neginf=0.0)
    feats_rep  = feats_flat.reshape(N_REPS, N_WINS, F_1CH).mean(axis=1)
    channel_feats[ch] = feats_rep

    mi = mutual_info_classif(feats_rep, y_all, random_state=RANDOM_STATE)
    channel_mi[ch] = mi.mean()

    print(f'  ch {ch:2d}  {ring_name:6s}  {ch % 8 * 45:3d}°   MI = {channel_mi[ch]:.4f}')

# ── Select top-8 ──────────────────────────────────────────────────────────────
ranked   = np.argsort(channel_mi)[::-1]
selected = np.sort(ranked[:N_SELECT])
dropped  = np.sort(ranked[N_SELECT:])

print(f'\nSelected channels (0-indexed): {selected.tolist()}')
print(f'Dropped  channels:             {dropped.tolist()}')
print(f'\n{"Ch":>4}  {"Ring":8}  {"Angle":6}  {"MI":>8}')
print('-' * 36)
for rank, ch in enumerate(ranked[:N_SELECT]):
    ring_label = ['Elbow', 'Middle', 'Wrist'][ch // 8]
    print(f'#{rank+1:2d}  ch{ch:<3d}  {ring_label:8s}  {ch % 8 * 45:3d}°    {channel_mi[ch]:.4f}')

# ── Ring coverage ─────────────────────────────────────────────────────────────
ring_counts = [int((selected < 8).sum()),
               int(((selected >= 8) & (selected < 16)).sum()),
               int((selected >= 16).sum())]
ring_names  = ['Elbow (ch 0-7)', 'Middle (ch 8-15)', 'Wrist (ch 16-23)']
print('\nChannels per ring:')
for name, count in zip(ring_names, ring_counts):
    print(f'  {name:22s}: {count}/8  {"█" * count}')
if min(ring_counts) == 0:
    print('WARNING: one ring has zero selected channels.')
else:
    print('All three rings represented.')

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, 'feat_representative_channels.npy'), selected)
np.save(os.path.join(OUT_DIR, 'feat_channel_mi_scores.npy'),       channel_mi)
print(f'\nSaved → {OUT_DIR}/feat_representative_channels.npy')
print(f'Saved → {OUT_DIR}/feat_channel_mi_scores.npy')
