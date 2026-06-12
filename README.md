# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning approach** — CNN/TCN models on raw time-series
- **Feature-based approach** — 50 libemg features (3192-dim) per window, SVM classifier

Each pipeline is evaluated under two protocols: **within-subject** (3-fold CV per subject) and **cross-subject** (LOSO).

---

## About the Dataset

The **putEMG** dataset ([Kaczmarek et al., 2019](https://www.mdpi.com/1424-8220/19/16/3548)) is a public sEMG benchmark from Poznan University of Technology.

| Property       | Value                                                        |
|----------------|--------------------------------------------------------------|
| Subjects       | 44 able-bodied participants                                  |
| Electrodes     | 24 sEMG channels (3 rings × 8 electrodes, 45° spacing)      |
| Sampling rate  | 5120 Hz                                                      |
| Gestures       | 7 active: G1, G2, G3, G6, G7, G8, G9                        |
| Repetitions    | ~40 per gesture per subject (2 sessions)                     |
| File format    | CSV per recording session                                    |

---

## Project Structure

```
putEMG prime/
├── README.md
├── docs/                                       # plan, notes, reports
│
├── src/                                        # shared Python modules
│   ├── emg_loader.py                           # data loading, LOSO & within-subject splits
│   ├── feature_extraction.py                   # libemg 50-feature extractor (3192-dim/window)
│   ├── deep_learning_models.py                 # EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
│   └── feature_based_models.py                 # FeatureLSTM, FeatureGRU, FeatureTransformer
│
├── data_preprocessing/
│   ├── matlab/                                 # CSV → per-subject combinedCell .mat
│   ├── preprocessing.py                        # bandpass 20–700 Hz, notch, resample, z-score
│   ├── feature_extraction.py                   # libemg flat_per_rep extractor (used by driver.ipynb)
│   └── driver.ipynb                            # entry point for preprocessing (Google Colab)
│
├── baselines/
│   ├── within_subject/
│   │   ├── deep_learning/    { model.ipynb · weights/ · results/ · README.md }
│   │   └── feature_based/    { model.ipynb · weights/ · results/ · README.md }
│   └── cross_subject/                          # LOSO
│       ├── deep_learning/    { model.ipynb · weights/ · results/ · README.md }
│       └── feature_based/    { model.ipynb · weights/ · results/ · README.md }
│
├── clustering & analysis/                      # 24 → 8 channel reduction + analysis
│   ├── clustering_features.ipynb               # feature-space channel clustering
│   ├── anatomical_validation.ipynb             # validates forearm coverage
│   ├── per_class_accuracy.ipynb                # per-subject per-class breakdown
│   └── archive/                                # earlier raw-signal clustering (superseded)
│
└── efficient/                                  # reduced-channel models (8 channels)
    ├── cross_subject/    { model.ipynb · weights/ · results/ }
    └── within_subject/   { model.ipynb · weights/ · results/ }
```

---

## Pipeline Overview

```
Raw CSVs  (data/csv_data_per_subject/)
   │
   ▼
[MATLAB]  data_preprocessing/matlab/put_emg_driver.m
   CSV → extract 24 channels → detect gesture blocks → combinedCell (N_reps × 7)
   Output: data/UG_per_subject/
   │
   ▼
[Python]  data_preprocessing/driver.ipynb  →  preprocessing.py
   Bandpass 20–700 Hz (Butterworth order 4, zero-phase, at 5120 Hz)
   → Notch filters: 30, 50, 60, 90, 150 Hz (IIR notch Q=30, zero-phase)
   → Resample: each rep → 1500 samples
   → Normalize: per-channel z-score
   │
   ├─── Deep Learning ─────────────────────────────────────────────────────────
   │    src/emg_loader.py  →  load_all_subjects() + make_loso_train_val_test()
   │    Input: (batch, 1, 24, 1500)
   │    baselines/{within,cross}_subject/deep_learning/model.ipynb
   │
   └─── Feature-Based ─────────────────────────────────────────────────────────
        data_preprocessing/feature_extraction.py  →  batch_extract_features()
        Sliding window (size=250, shift=50) → 50 libemg feature groups × 24 channels = 3192-dim/window
        baselines/{within,cross}_subject/feature_based/model.ipynb
```

---

## Results

### Within-Subject (3-fold CV) — COMPLETE

| Model                  | Mean Acc   | Std     | Notes                          |
|------------------------|------------|---------|--------------------------------|
| **EMG_TCN**            | **98.27%** | ±2.83%  | Best deep learning model       |
| ShallowConvNet         | 95.22%     | ±4.05%  |                                |
| EEGNet                 | 93.17%     | ±5.29%  |                                |
| **SVM_W** (feature)    | **95.95%** | ±4.12%  | Window-level, C=50, maj. vote  |
| SVM (feature)          | 89.42%     | ±6.21%  | Mean-pooled, C=50              |
| putEMG paper (SVM+RMS) | ~90%       | —       | Published benchmark            |

All models exceed the published ~90% within-subject benchmark. EMG_TCN leads at 98.27%; SVM_W (95.95%) is competitive with ShallowConvNet (95.22%) using only 50 libemg features.

### Cross-Subject LOSO — COMPLETE

| Model                  | Mean Acc   | Std      | Best Fold              | Worst Fold           |
|------------------------|------------|----------|------------------------|----------------------|
| **EMG_TCN**            | **85.01%** | ±12.20%  | Subj 14 (99.63%)       | Subj 06 (50.00%)     |
| ShallowConvNet         | 83.51%     | ±10.34%  | Subj 33 (98.91%)       | Subj 26 (57.76%)     |
| EEGNet                 | 83.28%     | ±11.10%  | Subj 33 (98.91%)       | Subj 07 (59.42%)     |
| **SVM** (feature)      | **78.83%** | ±12.23%  | Subj 33 (97.09%)       | Subj 06 (51.00%)     |
| FeatureMLP             | 27.80%     | ±12.20%  | Near random — fails    |                      |

Key observations:
- SVM LOSO (78.83%) is within 6 pp of EMG_TCN (85.01%) using only 50 libemg features — strong result
- ~10–12 pp std across all models: some subjects near-perfect (>99%), others near-chance (<55%)
- Subject 06 is hardest across all models and approaches; Subject 33 is easiest
- FeatureMLP fails cross-subject: no inductive bias for distribution shift; RBF kernel handles it implicitly

---

## How to Run

### 1. MATLAB Preprocessing
```matlab
% Edit put_emg_driver.m to set your data paths, then run:
run('data_preprocessing/matlab/put_emg_driver.m')
```

### 2. Python Signal Preprocessing
Open and run `data_preprocessing/driver.ipynb`.

### 3. Deep Learning (within-subject or cross-subject)
Open the relevant notebook:
- `baselines/within_subject/deep_learning/model.ipynb`
- `baselines/cross_subject/deep_learning/model.ipynb`

Set `MODEL_TYPE = 'EMG_TCN'` (or `'EEGNet'` / `'ShallowConvNet'`) and run.

### 4. Feature-Based Models
Open:
- `baselines/within_subject/feature_based/model.ipynb`
- `baselines/cross_subject/feature_based/model.ipynb`

Features are extracted once on first run and cached in `/Volumes/KRIS/data/features/`.

---

## Requirements

### MATLAB
MATLAB R2019b or later (`containers.Map`, `table`, `interp1`)

### Python
```bash
pip install scipy numpy matplotlib scikit-learn torch libemg seaborn
```

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors, 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet.** *Journal of Neural Engineering.* https://arxiv.org/abs/1611.08024
- Schirrmeister, R. T., et al. (2017). **Deep learning with CNNs for EEG.** *Human Brain Mapping.* https://doi.org/10.1002/hbm.23730
- putEMG dataset: https://biolab.put.poznan.pl/putemg-dataset/
- libemg: https://github.com/libemg/libemg
