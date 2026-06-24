# EMG Gesture Recognition — putEMG Prime

sEMG hand gesture recognition on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/) with two parallel classification pipelines evaluated across within-subject and cross-subject protocols, followed by principled channel reduction from 24 to 8 electrodes.

---

## Dataset

The **putEMG** dataset ([Kaczmarek et al., 2019](https://doi.org/10.3390/s19163548)) is a public sEMG benchmark from Poznan University of Technology.

| Property      | Value                                                   |
|---------------|---------------------------------------------------------|
| Subjects      | 44 able-bodied participants                             |
| Electrodes    | 24 sEMG channels (3 rings × 8 electrodes, 45° spacing) |
| Rings         | Elbow (ch 0–7), Middle (ch 8–15), Wrist (ch 16–23)     |
| Sampling rate | 5120 Hz                                                 |
| Gestures      | 7 active: G1, G2, G3, G6, G7, G8, G9                   |
| Repetitions   | ~280 per subject across 2 recording sessions            |

---

## How to Run

All experiments are driven by runner scripts in `scripts/`. Each script exposes named functions that can be called individually or sequenced. Checkpointing is built in — safe to interrupt and resume.

### 1. Preprocessing

```bash
# MATLAB: CSV → per-subject .mat
run('data_preprocessing/matlab/put_emg_driver.m')

# Python: bandpass / notch / resample / z-score → .npz
# Run data_preprocessing/driver.ipynb (Colab or local)

# Feature extraction: 41 libemg features × 24 channels → flat .npz
python -c "import sys; sys.path.insert(0, 'src'); from feature_extraction import batch_extract_features; batch_extract_features()"
# (or run src/feature_extraction.py directly)
```

### 2. Baselines (`scripts/run_baseline.py`)

```python
from scripts.run_baseline import *   # or edit and run directly

runners.run_within(type='deep_learning',  ...)  # EEGNet, ShallowConvNet, EMG_TCN
runners.run_within(type='feature_based',  ...)  # SVM (RBF), FeatureMLP
runners.run_cross( type='deep_learning',  ...)  # LOSO — EEGNet, ShallowConvNet, EMG_TCN
runners.run_cross( type='feature_based',  ...)  # LOSO — SGD_SVM, FeatureMLP
```

```bash
python scripts/run_baseline.py
```

Weights → `weights/baseline/{within_dl,within_feat,cross_dl,cross_feat}/{model}/{sid}.pt`
Results → `results/baseline/results_baseline_{within,cross}_{DL,feature}_{model}.txt`

### 3. Channel Clustering — Phase 3 (`scripts/run_clustering.py`)

Computes the 24×24 inter-channel correlation matrix across all ~311k classification
windows, applies agglomerative clustering (k=8), and selects the highest-MI
representative per cluster.

```bash
python scripts/run_clustering.py
```

Outputs → `clustering & analysis/` (`.npy` arrays + `clustering_results.json` + plots)

### 4. Efficient 8-Channel Models — Phase 5 (`scripts/run_efficient.py`)

Loads the 8 selected channels from `clustering_results.json` and retrains all
baseline models restricted to those channels only.

```python
from scripts.run_efficient import *

run_within_dl()    # within-subject: EEGNet, ShallowConvNet, EMG_TCN
run_within_feat()  # within-subject: SVM (RBF), FeatureMLP
run_cross_dl()     # LOSO: EEGNet, ShallowConvNet, EMG_TCN
run_cross_feat()   # LOSO: SGD_SVM, FeatureMLP
```

```bash
python scripts/run_efficient.py   # runs all four sequentially
```

Weights → `weights/efficient/{within_dl,within_feat,cross_dl,cross_feat}/{model}/{sid}.pt`
Results → `results/efficient/results_efficient_{within,cross}_{DL,feature}_{model}.txt`

---

## Results

### Channel Selection (Phase 3)

Window-level Pearson correlation clustering (k=8, average linkage) selects **8 electrodes** covering all three forearm rings:

| Selected channels (0-indexed) | Labels |
|-------------------------------|--------|
| 0, 1, 5, 7, 14, 16, 19, 22   | E0°, E45°, E225°, E315°, M270°, W0°, W135°, W270° |

Ring coverage: **Elbow 4 · Middle 1 · Wrist 3**. The middle ring is almost entirely redundant with the elbow and wrist rings (mean inter-channel correlation = 0.84). The largest cluster groups 6 anterior forearm channels (E45°–E180°, M90°–M135°) over the same extensor muscle group.

---

### Within-Subject (3-fold CV, 44 subjects)

| Model | 24-ch baseline | 8-ch efficient | Δ |
|-------|:--------------:|:--------------:|--:|
| **EMG_TCN** | **96.44%** ±2.91% | **95.00%** ±4.43% | −1.44% |
| SVM (RBF) | 96.31% ±4.01% | 93.93% ±5.29% | −2.38% |
| ShallowConvNet | 92.19% ±4.46% | 89.00% ±6.37% | −3.19% |
| EEGNet | 85.80% ±8.28% | 81.39% ±7.39% | −4.41% |
| FeatureMLP | 71.82% ±11.62% | 64.32% ±11.40% | −7.50% |
| putEMG paper | ~90% | — | — |

### Cross-Subject LOSO (44/44 folds)

| Model | 24-ch baseline | 8-ch efficient | Δ |
|-------|:--------------:|:--------------:|--:|
| **EMG_TCN** | **84.80%** ±11.56% | **83.37%** ±11.83% | −1.43% |
| ShallowConvNet | 83.92% ±10.13% | 80.42% ±10.73% | −3.50% |
| EEGNet | 83.41% ±11.07% | 80.02% ±11.06% | −3.39% |
| SGD_SVM | 69.07% ±14.79% | 62.59% ±10.83% | −6.48% |
| FeatureMLP | 67.89% ±13.47% | 53.34% ±10.67% | −14.55% |

**Key findings:**
- EMG_TCN retains **98.5% of within-subject** and **98.3% of cross-subject** accuracy at one-third the electrode count
- The DL drop is ≤1.5 pp for EMG_TCN — the channel reduction is essentially free for the best model
- The middle ring is structurally redundant; dropping 7 of 8 middle-ring channels costs almost nothing
- FeatureMLP is most sensitive to channel reduction (−14.6% cross-subject), tied to the drop from 984 to 328 features/window

---

## Pipeline

```
CSV recordings
  │
  ▼  MATLAB  data_preprocessing/matlab/put_emg_driver.m
     CSV → gesture segmentation → 24-ch → combinedCell .mat
  │
  ▼  Python  data_preprocessing/driver.ipynb + src/preprocessing.py
     Bandpass 20–700 Hz · Notch 30/50/60/90/150 Hz · Resample 1500 samples · Z-score
     → data/processed gestures/{train,eval}/emg_gestures_SS_U.npz  (N × 24 × 1500)
  │
  ├──[DL path]──────────────────────────────────────────────────────────────────────
  │  Input: (batch, 1, 24, 1500)                         scripts/run_baseline.py
  │  Models: EEGNet · ShallowConvNet · EMG_TCN           runners.run_within / run_cross
  │
  └──[Feature path]─────────────────────────────────────────────────────────────────
     src/feature_extraction.py
     Sliding window 250/50 → 26 windows/rep
     41 libemg features × 24 channels = 984-dim/window (channel-major)
     → data/features/{train,eval}/features_SS_flat_rep.npz  (N × 25584)
         │
         └── Models: SVM (RBF) · SGD_SVM · FeatureMLP    runners.run_within / run_cross

  Phase 3 — Channel clustering (scripts/run_clustering.py)
     Window-level Pearson correlation → agglomerative k=8 → MI-ranked representative
     → clustering & analysis/clustering_results.json  (8 selected channel indices)

  Phase 5 — Efficient models (scripts/run_efficient.py)
     Slice X[:, :, channels, :] → (batch, 1, 8, 1500) for DL
     Slice feature columns     → (N, 26, 328)         for features
     run_within_dl · run_within_feat · run_cross_dl · run_cross_feat
```

---

## Project Structure

```
putEMG prime/
├── src/
│   ├── runners.py              # run_within(type, ...), run_cross(type, ...)
│   ├── deep_learning_models.py # EEGNet, ShallowConvNet, EMG_TCN
│   ├── feature_based_models.py # FeatureMLP
│   ├── feature_extraction.py   # batch_extract_features (41 libemg features)
│   ├── housekeeping.py         # loaders, fold splits, majority vote, results writer
│   ├── preprocessing.py        # bandpass / notch / resample / z-score
│   └── trainer.py              # train, evaluate
│
├── scripts/
│   ├── run_baseline.py         # Phase 1 & 2: calls runners.run_within / run_cross
│   ├── run_clustering.py       # Phase 3: window-level correlation clustering
│   └── run_efficient.py        # Phase 5: run_within_dl/feat, run_cross_dl/feat
│
├── data/
│   ├── processed gestures/     # {train,eval}/emg_gestures_SS_U.npz  (N × 24 × 1500)
│   └── features/               # {train,eval}/features_SS_flat_rep.npz  (N × 25584)
│
├── weights/
│   ├── baseline/               # {within_dl,within_feat,cross_dl,cross_feat}/{model}/{sid}.pt
│   └── efficient/              # same layout, 8-channel models
│
├── results/
│   ├── baseline/               # results_baseline_{within,cross}_{DL,feature}_{model}.txt
│   └── efficient/              # results_efficient_{within,cross}_{DL,feature}_{model}.txt
│
├── clustering & analysis/
│   ├── clustering_results.json            # selected channels + cluster breakdown
│   ├── feat_representative_channels.npy   # (8,) 0-indexed
│   ├── feat_cluster_labels.npy            # (24,) cluster assignments
│   ├── dendrogram.png
│   ├── correlation_heatmap.png
│   └── silhouette_sweep.png
│
├── data_preprocessing/
│   ├── matlab/                 # put_emg_driver.m: CSV → .mat
│   └── driver.ipynb            # Python preprocessing (Colab)
│
└── docs/
    ├── plan.txt                # full 7-phase research plan with status
    ├── project_summary.txt     # methodology, results, references
    ├── paper.tex               # ICML-format paper
    └── paper.bib               # BibTeX references
```

---

## Requirements

**MATLAB:** R2019b or later

**Python:**
```bash
pip install scipy numpy matplotlib scikit-learn torch libemg seaborn
```

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.** *Journal of Neural Engineering.* https://doi.org/10.1088/1741-2552/aace8c
- Schirrmeister, R. T., et al. (2017). **Deep Learning with CNNs for EEG Decoding.** *Human Brain Mapping.* https://doi.org/10.1002/hbm.23730
- Qu, Y., et al. (2021). **Reduce sEMG Channels for Gesture Recognition by mRMR.** *Journal of Healthcare Engineering.* https://doi.org/10.1155/2021/9929684
- Scheme, E., Englehart, K. (2020). **Automated Channel Selection in HD-sEMG.** *Sensors 20*(19), 5679. https://doi.org/10.3390/s20195679
- libemg: https://github.com/libemg/libemg
- putEMG dataset: https://biolab.put.poznan.pl/putemg-dataset/
