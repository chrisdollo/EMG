# putEMG Prime — Project Guide for Claude

EMG gesture recognition research project built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/).
44 subjects, 24 sEMG channels (3 rings × 8 electrodes, 45° spacing), 5120 Hz, 7 hand gestures, ~280 reps per subject.

---

## Project Layout

```
putEMG prime/
├── src/                          # Shared Python modules (imported by all notebooks)
│   ├── emg_loader.py             # Data loading, LOSO & within-subject splits
│   ├── feature_extraction.py     # libemg 50-feature extractor (batch_extract)
│   ├── deep_learning_models.py   # EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
│   └── feature_based_models.py   # FeatureLSTM, FeatureGRU, FeatureTransformer
│
├── data_preprocessing/
│   ├── matlab/                   # CSV → combinedCell .mat (put_emg_driver.m)
│   ├── preprocessing.py          # Bandpass 20–700 Hz + notch 30/50/60/90/150 Hz, resample to 1500, z-score
│   ├── feature_extraction.py     # Legacy 8-feature extractor (used by driver.ipynb only)
│   └── driver.ipynb              # Entry point for preprocessing
│
├── experiments/
│   ├── within_subject/
│   │   ├── deep_learning/        # 3-fold CV per subject — COMPLETE (38 subjects)
│   │   └── feature_based/        # 3-fold CV per subject — NOT YET RUN
│   │
│   ├── cross_subject/            # LOSO (Leave-One-Subject-Out)
│   │   ├── deep_learning/        # COMPLETE — all 44 folds, all 3 models
│   │   └── feature_based/        # NOT YET RUN
│   │
│   ├── clustering/               # 24→8 channel reduction
│   │   ├── clustering_features.ipynb  # Feature-based channel clustering
│   │   ├── anatomical_validation.ipynb  # Anatomy coverage analysis
│   │   └── archive/              # Raw-signal clustering (superseded)
│   │
│   └── efficient/
│       └── deep_learning/        # Reduced 8-channel model — NOT YET RUN
│
└── docs/
    ├── plan.txt                  # Full 7-phase research plan
    └── claude_report.txt         # Earlier progress report
```

---

## Import Pattern

All notebooks add `src/` to sys.path using a relative path from their location:

```python
_SRC = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..', 'src'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
```

Notebooks at depth 3 from root (e.g. `experiments/within_subject/deep_learning/`) use 3 levels of `..`.
Clustering notebooks at depth 2 use 2 levels of `..`.

---

## Data

- Raw .mat files: `/Volumes/KRIS/data/UG_per_subject/emg_gestures_SS_U.mat` (44 files)
- Feature cache: `/Volumes/KRIS/data/features_sequence/` — shared between within- and cross-subject feature notebooks; auto-extracted on first run

---

## Results

### Within-Subject (3-fold CV) — deep learning — COMPLETE

38 subjects evaluated (38–44 subjects depending on model, some skipped due to data issues).

| Model | Mean Acc | Std |
|-------|----------|-----|
| **EMG_TCN** | **97.47%** | ±2.78% |
| ShallowConvNet | 93.50% | ±4.09% |
| EEGNet | 91.05% | ±5.56% |
| putEMG paper (SVM+RMS) | ~90% | — |

All three models beat the published ~90% benchmark. EMG_TCN is the clear winner.

### Cross-Subject LOSO — deep learning — COMPLETE

44/44 folds complete for all 3 models.

| Model | Mean Acc | Std | Best | Worst |
|-------|----------|-----|------|-------|
| **EMG_TCN** | **80.76%** | ±12.38% | Subj 14 & 48 (99.29%) | Subj 06 (53.93%) |
| EEGNet | 77.23% | ±12.05% | Subj 27 (98.21%) | Subj 06 (48.57%) |
| ShallowConvNet | 73.38% | ±11.94% | Subj 27 (95.00%) | Subj 17 (37.50%) |

Key observations:
- ~12 pp std is high — some subjects near-perfect, others near-chance
- Val accuracy (93% for EMG_TCN) is much higher than test (81%) — gap reflects inter-subject variance the model hasn't seen
- Subject 06 is hardest across all models; Subject 27 is easiest
- Cross-subject is fundamentally harder than within-subject — subject-adaptive fine-tuning is the natural next step

### Feature-Based & Reduced-Channel — NOT YET RUN

---

## Pending Work (from docs/plan.txt)

| Phase | Description | Status |
|-------|-------------|--------|
| 1b | Within-subject feature-based (FeatureLSTM/GRU/Transformer) | Notebook ready, not run |
| 2b | Cross-subject feature-based LOSO | Notebook ready, not run |
| 3 | Feature-based channel clustering (24→8) | Notebook ready, not yet run |
| 4 | Anatomical validation of clusters | Notebook ready, not yet run |
| 5a | Reduced-channel DL model (8 channels) | Notebook ready, not yet run |
| 5b | Reduced-channel feature-based model | Notebook ready, not yet run |
| 6 | Per-subject per-class accuracy analysis | Not started |
| 7 | Real-time data collection | Future |

### Phase 3 & 5 Dependency Chain
1. Run `experiments/clustering/clustering_features.ipynb` → produces `feat_representative_channels.npy`
2. Run `experiments/clustering/anatomical_validation.ipynb` → validates anatomical coverage
3. Run `experiments/efficient/deep_learning/model.ipynb` → loads `feat_representative_channels.npy`, trains 8-channel LOSO
4. Run `experiments/efficient/feature_based/model.ipynb` → same, for feature-based models
5. Final cells auto-compare 8-channel vs 24-channel baseline

---

## Model Input Shapes

| Approach | Input shape | Source |
|----------|-------------|--------|
| Deep learning | `(batch, 1, 24, 1500)` | `load_all_subjects()` |
| Feature-based | `(batch, 26, 3192)` | `load_feature_subjects(mode='sequence')` |
| Efficient (8-ch DL) | `(batch, 1, 8, 1500)` | `load_all_subjects()` + `X[:, :, channels, :]` |

Channel selection for efficient model:
```python
channels = np.load('experiments/clustering/feat_representative_channels.npy')  # (8,) 0-indexed
subjects = [(name, X[:, :, channels, :], y) for name, X, y in load_all_subjects(DATA_DIR)]
```

Feature dimensions: 50 feature groups × 24 channels = 3192. Sliding window: size=250, shift=50 → 26 windows/rep.

---

## Key Loader Functions (src/emg_loader.py)

```python
load_all_subjects(dir_path)
    → list of (name, X, y)   X: (N, 1, 24, 1500)

load_feature_subjects(dir_path, mode='sequence')
    → list of (name, X, y)   X: (N, 26, 3192)

make_loso_train_val_test(subjects, test_subject_idx, val_frac=0.10, batch_size=16)
    → train_loader, val_loader, test_loader

make_within_subject_loaders(X, y, n_folds=3, test_fold_idx=0, batch_size=16, seed=42)
    → train_loader, test_loader
```

Both loaders are shape-agnostic on axis 0 — they work identically for DL and feature inputs.

---

## Checkpointing

All training notebooks skip already-completed folds automatically — safe to interrupt and resume.
Weights saved to `experiments/<scope>/<approach>/weights/<MODEL_TYPE>/`.
Results logged to `experiments/<scope>/<approach>/results/`.
