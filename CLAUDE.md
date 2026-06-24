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

## Results (as of 2026-06-16) — PHASES 1–3 AND 5 COMPLETE

### Within-Subject (3-fold CV, 44 subjects)

| Model | 24-ch baseline | 8-ch efficient | Δ | Notes |
|-------|---------------|---------------|---|-------|
| **EMG_TCN** | **96.44%** ±2.91% | **95.00%** ±4.43% | −1.44% | best DL |
| ShallowConvNet | 92.19% ±4.46% | 89.00% ±6.37% | −3.19% | |
| EEGNet | 85.80% ±8.28% | 81.39% ±7.39% | −4.41% | |
| **SVM** (RBF) | **96.31%** ±4.01% | **93.93%** ±5.29% | −2.38% | best feature-based |
| FeatureMLP | 71.82% ±11.62% | 64.32% ±11.40% | −7.50% | |
| putEMG paper | ~90% | — | — | benchmark |

### Cross-Subject LOSO (44/44 folds)

| Model | 24-ch baseline | 8-ch efficient | Δ | Notes |
|-------|---------------|---------------|---|-------|
| **EMG_TCN** | **84.80%** ±11.56% | **83.37%** ±11.83% | −1.43% | best DL |
| ShallowConvNet | 83.92% ±10.13% | 80.42% ±10.73% | −3.50% | |
| EEGNet | 83.41% ±11.07% | 80.02% ±11.06% | −3.39% | |
| **SGD_SVM** | **69.07%** ±14.79% | **62.59%** ±10.83% | −6.48% | best feature-based |
| FeatureMLP | 67.89% ±13.47% | 53.34% ±10.67% | −14.55% | most sensitive to channel reduction |

EMG_TCN retains 98.3–98.5% of its accuracy at one-third the electrode count.

### Channel Selection (Phase 3)
Selected channels: `[0, 1, 5, 7, 14, 16, 19, 22]` → E0°, E45°, E225°, E315°, M270°, W0°, W135°, W270°
Method: window-level Pearson correlation → agglomerative clustering (k=8, average linkage) → highest-MI representative per cluster.
See `clustering & analysis/clustering_results.json` and `docs/project_summary.txt` for full methodology.

---

## Pending Work

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Within-subject DL | **COMPLETE** |
| 1b | Within-subject feature-based | **COMPLETE** |
| 2a | Cross-subject LOSO DL | **COMPLETE** |
| 2b | Cross-subject LOSO feature-based (SGD_SVM + FeatureMLP) | **COMPLETE** |
| 3 | Feature-space channel clustering (24→8) | **COMPLETE** |
| 4 | Anatomical validation of clusters | Not started |
| 5a | Reduced-channel DL (8 channels, within + LOSO) | **COMPLETE** |
| 5b | Reduced-channel feature-based (8 channels, within + LOSO) | **COMPLETE** |
| 6 | Per-subject per-class accuracy analysis | Not started |
| 7 | Real-time data collection | Future |

---

## Checkpointing

All training skips already-completed subjects/folds automatically — safe to interrupt and resume.
Checkpoints: `weights/baseline/{within_dl,within_feat,cross_dl,cross_feat}/{sid}.pt`
Results auto-written after each subject completes.
