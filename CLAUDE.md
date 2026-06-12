# putEMG Prime — Project Guide for Claude

EMG gesture recognition research project built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/).
44 subjects, 24 sEMG channels (3 rings × 8 electrodes, 45° spacing), 5120 Hz, 7 hand gestures, ~280 reps per subject.

---

## Project Layout

```
putEMG prime/
├── CLAUDE.md
├── baseline.ipynb                # All 4 baseline experiments (Colab-ready)
├── clustering.ipynb              # Phase 3: feature-space 24→8 channel selection
├── efficient.ipynb               # Phase 5: 8-channel models
│
├── src/
│   ├── deep_learning_models.py   # EEGNet, ShallowConvNet, EMG_TCN
│   ├── feature_based_models.py   # FeatureMLP
│   ├── feature_extraction.py     # libemg 41-feature extractor (batch_extract_features)
│   ├── housekeeping.py           # loaders, fold splits, majority vote, _write_results
│   ├── preprocessing.py          # bandpass/notch/resample/z-score
│   ├── runners.py                # run_within, run_cross
│   └── trainer.py                # train, evaluate
│
├── scripts/
│   └── run_baseline.py           # runs all 4 baseline experiments (FORCE_RERUN=True)
│
├── data/
│   ├── processed gestures/{train,eval}/   emg_gestures_SS_U.npz  (N×24×1500)
│   └── features/{train,eval}/             features_SS_flat_rep.npz
│
├── data_preprocessing/
│   ├── matlab/                   # CSV → combinedCell .mat (put_emg_driver.m)
│   └── driver.ipynb              # preprocessing entry point (Google Colab)
│
├── weights/baseline/
│   ├── within_dl/                # per-subject .pt (all 3 DL models inside each)
│   ├── within_feat/              # per-subject .pt (SVM + FeatureMLP)
│   ├── cross_dl/                 # per-LOSO-fold .pt (all 3 DL models inside each)
│   └── cross_feat/               # per-LOSO-fold .pt (SVM_W + FeatureMLP)
│
├── results/baseline/
│   └── results_baseline_{within,cross}_{DL,feature}_{model}.txt
│
├── clustering & analysis/        # Phase 3/4 outputs
│   ├── feat_representative_channels.npy  (8,) 0-indexed  [after Phase 3]
│   └── feat_cluster_labels.npy           (24,) cluster assignments
│
└── docs/
    ├── plan.txt
    └── project_summary.txt
```

---

## Import Pattern

All notebooks are at the project root:

```python
import sys
sys.path.insert(0, 'src')
```

On Colab, `baseline.ipynb` also mounts Drive and `os.chdir`s to the project root so all paths stay relative.

---

## Data

- Preprocessed signal files: `data/processed gestures/{train,eval}/emg_gestures_SS_U.npz`
- Feature files: `data/features/{train,eval}/features_SS_flat_rep.npz`

---

## Feature Dimensions

41 single-scalar libemg features × 24 channels = **984 features per window**.
Sliding window: size=250, shift=50 → **26 windows per rep**.
Stored flat: `(N_reps, 26 × 984)` = `(N_reps, 25584)`.
Loader reshapes to `(N_reps, 26, 984)` — channel-major layout, so channel c is columns `[c*41 : (c+1)*41]`.

The 9 excluded libemg groups (not 1-scalar-per-channel): AR, CC, DFTR, WENG, WV, WWL, WENT, RMSPHASOR, WLPHASOR.

---

## Model Protocols

### Within-subject (`run_within`)

| Type | Models | SVM kernel | Input to model |
|------|--------|-----------|----------------|
| `deep_learning` | EEGNet, ShallowConvNet, EMG_TCN | — | `(N, 1, 24, 1500)` raw signal |
| `feature_based` | SVM, FeatureMLP | **RBF**, C=50 | `(N×26, 984)` window-level → majority vote |

`WITHIN_FEATURE_MODELS = ['SVM', 'FeatureMLP']`

### Cross-subject (`run_cross`)

| Type | Models | SVM kernel | Input to model |
|------|--------|-----------|----------------|
| `deep_learning` | EEGNet, ShallowConvNet, EMG_TCN | — | `(N, 1, 24, 1500)` raw signal |
| `feature_based` | SVM_W, FeatureMLP | **LinearSVC**, C=50 | `(N×26, 984)` window-level → majority vote |

`CROSS_FEATURE_MODELS = ['SVM_W', 'FeatureMLP']`

LinearSVC is used instead of RBF for cross-subject because the 43-subject pool yields ~330k windows — RBF SVC is O(n²) and intractable at this scale. LinearSVC scales linearly. Window-level + majority vote protocol is identical to within-subject.

---

## Result File Format

One file per model: `results/baseline/results_baseline_{within,cross}_{DL,feature}_{model}.txt`

```
phase        = baseline
protocol     = within
approach     = DL
model        = EMG_TCN
channels     = 24
date         = 2026-06-12

subject    val_acc   test_acc   epochs
------------------------------------------
03          96.45%     98.27%      12.0
...

mean_val  = 96.1%  ±  2.3%
mean_test = 98.27% ±  2.83%
best      = 03 (99.5%)
worst     = 06 (88.2%)
n         = 44/44
```

SVM always shows `N/A` for val_acc and epochs (no training loop).
For within-subject (3-fold CV), val_acc and epochs are means across the 3 folds.

---

## Results (as of 2026-06-12) — RUNNING

All baselines are being re-run from scratch. Numbers below are from prior runs.

### Within-Subject (3-fold CV, 44 subjects)

| Model | Mean Acc | Std | Notes |
|-------|----------|-----|-------|
| **EMG_TCN** | **98.27%** | ±2.83% | best DL |
| ShallowConvNet | 95.22% | ±4.05% | |
| EEGNet | 93.17% | ±5.29% | |
| **SVM** (RBF, window-level) | **~96%** | ~±4% | best feature-based; putEMG paper protocol |
| FeatureMLP | ~82% | — | per-fold instability on some subjects |
| putEMG paper | ~90% | — | benchmark |

### Cross-Subject LOSO (44/44 folds)

| Model | Mean Acc | Std | Best | Worst |
|-------|----------|-----|------|-------|
| **EMG_TCN** | **85.01%** | ±12.20% | Subj 14 (99.63%) | Subj 06 (50.00%) |
| ShallowConvNet | 83.51% | ±10.34% | Subj 33 (98.91%) | Subj 26 (57.76%) |
| EEGNet | 83.28% | ±11.10% | Subj 33 (98.91%) | Subj 07 (59.42%) |
| SVM_W (LinearSVC) | pending | — | — | — |
| FeatureMLP | pending | — | — | — |

---

## Pending Work

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Within-subject DL | **RUNNING** |
| 1b | Within-subject feature-based | **RUNNING** |
| 2a | Cross-subject LOSO DL | **RUNNING** |
| 2b | Cross-subject LOSO feature-based (SVM_W LinearSVC + FeatureMLP) | **PENDING** |
| 3 | Feature-space channel clustering (24→8) | Not started |
| 4 | Anatomical validation of clusters | Not started |
| 5a | Reduced-channel DL (8 channels) | Not started |
| 5b | Reduced-channel feature-based (8 channels) | Not started |
| 6 | Per-subject per-class accuracy analysis | Not started |
| 7 | Real-time data collection | Future |

### Phase 3 & 5 Dependency Chain
1. `clustering.ipynb` → `clustering & analysis/feat_representative_channels.npy`
2. `efficient.ipynb` → loads channel indices, slices `X[:, :, channels, :]` for DL or `X[:, :, channels, :]` for features (41×8=328/window)

---

## Checkpointing

All training skips already-completed subjects/folds automatically — safe to interrupt and resume.
Checkpoints: `weights/baseline/{within_dl,within_feat,cross_dl,cross_feat}/{sid}.pt`
Results auto-written after each subject completes.
