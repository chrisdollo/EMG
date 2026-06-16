"""
Phase 3 — Window-Level Correlation Clustering

For each of the 41 per-channel features, computes a 24×24 Pearson correlation
matrix across all windows from all subjects, then averages over features to get
one consensus redundancy matrix.  Agglomerative clustering (average linkage,
k=8) groups redundant channels; within each cluster the highest-MI channel is
chosen as the representative.

Saves:
  clustering & analysis/feat_representative_channels.npy  (8,)  0-indexed
  clustering & analysis/feat_cluster_labels.npy           (24,) cluster id per channel
  clustering & analysis/feat_channel_mi.npy               (24,) mean MI per channel
  clustering & analysis/feat_channel_corr.npy             (24,24) consensus corr matrix

  clustering & analysis/silhouette_sweep.png
  clustering & analysis/correlation_heatmap.png
  clustering & analysis/dendrogram.png

Usage:
  python scripts/run_clustering.py
"""

import os
import sys
import json
import datetime
import warnings
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from housekeeping import _list_subject_files, _load_combined_feature_subject

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
FEAT_TRAIN   = os.path.join(ROOT, 'data', 'features', 'train')
FEAT_EVAL    = os.path.join(ROOT, 'data', 'features', 'eval')
OUT_DIR      = os.path.join(ROOT, 'clustering & analysis')
N_CH         = 24
N_FPC        = 41
N_SELECT     = 8
K_RANGE      = range(4, 13)
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)

_RING = ['Elbow', 'Middle', 'Wrist']

def _ch_label(ch):
    return f'{_RING[ch // 8][0]}{ch % 8 * 45}°'   # e.g. E0°, M90°, W315°


# ── Load all subjects ─────────────────────────────────────────────────────────
train_sids = set(_list_subject_files(FEAT_TRAIN))
eval_sids  = set(_list_subject_files(FEAT_EVAL))
all_sids   = sorted(train_sids | eval_sids)
print(f'Subjects found: {len(all_sids)}')

X_list, y_list = [], []
for sid in all_sids:
    X, y = _load_combined_feature_subject(FEAT_TRAIN, FEAT_EVAL, sid)
    X_list.append(X)
    y_list.append(y)

X_all = np.concatenate(X_list, axis=0)   # (N_reps_total, N_wins, 984)
y_all = np.concatenate(y_list, axis=0)   # (N_reps_total,)
del X_list, y_list

N_reps, N_wins, N_feat = X_all.shape

# Flatten reps × windows → (N_windows_total, 984); float32 to keep memory down
X_wins = X_all.reshape(N_reps * N_wins, N_feat).copy()
del X_all
y_wins = np.repeat(y_all, N_wins)

np.nan_to_num(X_wins, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

# Channel-major reshape: view into X_wins, no extra copy
X_chan = X_wins.reshape(len(X_wins), N_CH, N_FPC)   # (N_windows, 24, 41)

print(f'Total windows : {len(X_wins):,}  (reps={N_reps}, wins/rep={N_wins})')


# ── Window-level correlation matrix ──────────────────────────────────────────
# For each feature f: shape (N_windows, 24).T → np.corrcoef → (24, 24).
# Average over all 41 features → consensus redundancy matrix.
print(f'\nComputing {N_FPC}-feature correlation average...')
corr_sum = np.zeros((N_CH, N_CH), dtype=np.float64)   # accumulator in float64
for f in range(N_FPC):
    vals = X_chan[:, :, f].T              # (24, N_windows) float32
    corr_sum += np.corrcoef(vals)         # np.corrcoef promotes internally → (24, 24) float64

corr_mean = corr_sum / N_FPC         # (24, 24)  diagonal ≈ 1.0
print(f'  mean off-diagonal correlation: {(corr_mean.sum() - np.trace(corr_mean)) / (N_CH*(N_CH-1)):.4f}')


# ── Per-channel MI with gesture labels ───────────────────────────────────────
print(f'\nMI per channel (mean over {N_FPC} features):')
channel_mi = np.zeros(N_CH)
for ch in range(N_CH):
    mi = mutual_info_classif(X_chan[:, ch, :], y_wins, random_state=RANDOM_STATE)
    channel_mi[ch] = mi.mean()
    print(f'  ch {ch:2d}  {_RING[ch//8]:6s}  {ch%8*45:3d}°   MI={channel_mi[ch]:.4f}')


# ── Silhouette sweep ──────────────────────────────────────────────────────────
dist_mat = np.clip(1.0 - corr_mean, 0.0, None)   # (24, 24) dissimilarity

print(f'\nSilhouette sweep k={K_RANGE.start}..{K_RANGE.stop - 1}:')
sil_scores = {}
for k in K_RANGE:
    agg   = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    lbls  = agg.fit_predict(dist_mat)
    score = silhouette_score(dist_mat, lbls, metric='precomputed')
    sil_scores[k] = score
    marker = ' ←' if k == N_SELECT else ''
    print(f'  k={k:2d}  silhouette={score:.4f}{marker}')

best_k = max(sil_scores, key=sil_scores.get)
print(f'\nBest k by silhouette: {best_k}  (score={sil_scores[best_k]:.4f})')
print(f'Using k={N_SELECT} (target electrode count)')


# ── Final clustering k=8 ──────────────────────────────────────────────────────
agg8          = AgglomerativeClustering(n_clusters=N_SELECT, metric='precomputed', linkage='average')
cluster_labels = agg8.fit_predict(dist_mat)   # (24,)


# ── Select representative: highest MI per cluster ─────────────────────────────
print(f'\nCluster composition and representative selection:')
representative_channels = np.empty(N_SELECT, dtype=int)
for cid in range(N_SELECT):
    members  = np.where(cluster_labels == cid)[0]
    best_idx = members[np.argmax(channel_mi[members])]
    representative_channels[cid] = best_idx
    member_str = ', '.join([f'ch{m}({_ch_label(m)})' for m in sorted(members)])
    print(f'  cluster {cid}: [{member_str}]  → ch{best_idx} ({_RING[best_idx//8]} {best_idx%8*45}°)  MI={channel_mi[best_idx]:.4f}')

representative_channels = np.sort(representative_channels)

print(f'\nSelected channels (0-indexed): {representative_channels.tolist()}')

ring_counts = [
    int((representative_channels < 8).sum()),
    int(((representative_channels >= 8) & (representative_channels < 16)).sum()),
    int((representative_channels >= 16).sum()),
]
for name, cnt in zip(['Elbow (ch 0-7)', 'Middle (ch 8-15)', 'Wrist (ch 16-23)'], ring_counts):
    print(f'  {name:22s}: {cnt}/8  {"█" * cnt}')
if min(ring_counts) == 0:
    print('WARNING: one ring has zero selected channels')
else:
    print('All three rings represented.')


# ── Visualisations ────────────────────────────────────────────────────────────

# 1 — Silhouette sweep
fig, ax = plt.subplots(figsize=(7, 4))
ks = list(sil_scores.keys())
ax.plot(ks, [sil_scores[k] for k in ks], 'o-', color='steelblue')
ax.axvline(N_SELECT, color='crimson', linestyle='--', label=f'k={N_SELECT} (target)')
ax.axvline(best_k,   color='green',   linestyle=':',  label=f'k={best_k} (best silhouette)', alpha=0.8)
ax.set_xlabel('k')
ax.set_ylabel('Silhouette score')
ax.set_title('Silhouette sweep — feature-space channel clustering')
ax.set_xticks(ks)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'silhouette_sweep.png'), dpi=150)
plt.close()

# 2 — Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr_mean, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title('24×24 inter-channel correlation (window-level, averaged over 41 features)')
ax.set_xlabel('Channel index')
ax.set_ylabel('Channel index')
ax.set_xticks(range(0, N_CH, 4))
ax.set_yticks(range(0, N_CH, 4))
for sep in [8, 16]:
    ax.axhline(sep - 0.5, color='black', linewidth=1.5)
    ax.axvline(sep - 0.5, color='black', linewidth=1.5)
# Mark selected channels
for ch in representative_channels:
    ax.plot(ch, ch, 'y*', markersize=14, zorder=5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'correlation_heatmap.png'), dpi=150)
plt.close()

# 3 — Dendrogram
# squareform requires perfect symmetry; force it to avoid float accumulation drift
dist_sym = (dist_mat + dist_mat.T) / 2
np.fill_diagonal(dist_sym, 0.0)
dist_condensed = squareform(dist_sym)
Z = scipy_linkage(dist_condensed, method='average')

ch_tick_labels = [_ch_label(ch) for ch in range(N_CH)]
cut_height = (Z[-(N_SELECT), 2] + Z[-(N_SELECT - 1), 2]) / 2   # midpoint between k=8 and k=7 merge

fig, ax = plt.subplots(figsize=(13, 5))
dendrogram(Z, labels=ch_tick_labels, ax=ax, leaf_rotation=90,
           color_threshold=cut_height)
ax.axhline(cut_height, color='crimson', linestyle='--', linewidth=1.2,
           label=f'k={N_SELECT} cut')
ax.set_title('Channel dendrogram — average linkage, correlation distance')
ax.set_ylabel('Distance (1 − correlation)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'dendrogram.png'), dpi=150)
plt.close()


# ── Save ──────────────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, 'feat_representative_channels.npy'), representative_channels)
np.save(os.path.join(OUT_DIR, 'feat_cluster_labels.npy'),          cluster_labels)
np.save(os.path.join(OUT_DIR, 'feat_channel_mi.npy'),              channel_mi)
np.save(os.path.join(OUT_DIR, 'feat_channel_corr.npy'),            corr_mean)

# Human-readable JSON — load with json.load() in Phase 5 or notebooks
clusters_info = []
for cid in range(N_SELECT):
    members = sorted(np.where(cluster_labels == cid)[0].tolist())
    rep     = next(int(r) for r in representative_channels if cluster_labels[r] == cid)
    clusters_info.append({
        'cluster_id':        cid,
        'members':           members,
        'member_labels':     [_ch_label(m) for m in members],
        'representative':    rep,
        'rep_label':         _ch_label(rep),
        'rep_ring':          _RING[rep // 8],
        'rep_angle_deg':     int(rep % 8 * 45),
        'rep_mi':            round(float(channel_mi[rep]), 6),
    })

results = {
    'date':                  datetime.date.today().isoformat(),
    'n_subjects':            len(all_sids),
    'n_windows':             int(len(X_wins)),
    'method':                'window-level Pearson correlation → agglomerative (average linkage) → highest-MI representative',
    'n_clusters':            N_SELECT,
    'selected_channels':     representative_channels.tolist(),
    'selected_labels':       [_ch_label(c) for c in representative_channels.tolist()],
    'cluster_labels_per_ch': cluster_labels.tolist(),
    'silhouette_at_k8':      round(sil_scores[N_SELECT], 6),
    'best_silhouette_k':     best_k,
    'silhouette_scores':     {str(k): round(v, 6) for k, v in sil_scores.items()},
    'clusters':              clusters_info,
    'ring_coverage':         {
        'Elbow':  int(ring_counts[0]),
        'Middle': int(ring_counts[1]),
        'Wrist':  int(ring_counts[2]),
    },
}

json_path = os.path.join(OUT_DIR, 'clustering_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved to {OUT_DIR}/')
print(f'  feat_representative_channels.npy  {representative_channels}')
print(f'  feat_cluster_labels.npy           {cluster_labels.tolist()}')
print(f'  feat_channel_mi.npy')
print(f'  feat_channel_corr.npy')
print(f'  clustering_results.json')
print(f'  silhouette_sweep.png')
print(f'  correlation_heatmap.png')
print(f'  dendrogram.png')
