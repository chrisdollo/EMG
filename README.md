# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning approach** — CNN/TCN models on raw time-series
- **Feature-based approach** — 50 libemg features (3192-dim) fed into sequence models (LSTM/GRU/Transformer)

Each pipeline is evaluated under two protocols: **within-subject** (3-fold CV per subject) and **cross-subject** (LOSO).

---

## About the Dataset

The **putEMG** dataset ([Kaczmarek et al., 2019](https://www.mdpi.com/1424-8220/19/16/3548)) is a public sEMG benchmark from Poznan University of Technology.

| Property       | Value                                                        |
|----------------|--------------------------------------------------------------|
| Subjects       | 44 able-bodied participants                                  |
| Electrodes     | 24 sEMG channels (sparse matrix on forearm)                  |
| Sampling rate  | 5120 Hz                                                      |
| Gestures       | 7 active: G1, G2, G3, G6, G7, G8, G9                        |
| Repetitions    | ~40 per gesture per subject (2 sessions)                     |
| File format    | CSV per recording session                                    |

---

## Project Structure

```
putEMG prime/
├── README.md
├── docs/                                       # plan, notes, slides
│
├── src/                                        # shared Python modules
│   ├── emg_loader.py                           # data loading, LOSO & within-subject splits, legacy 8-feat extractor
│   ├── feature_extraction.py                   # libemg 50-feature extractor (3192-dim/window)
│   ├── deep_learning_models.py                 # EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
│   └── feature_based_models.py                 # FeatureLSTM, FeatureGRU, FeatureTransformer
│
├── data_preprocessing/
│   ├── matlab/                                 # CSV → per-subject combinedCell .mat
│   ├── preprocessing.py                        # bandpass, resample, z-score
│   ├── feature_extraction.py                   # legacy 8-feature extractor (used by driver.ipynb)
│   └── driver.ipynb                            # entry point for preprocessing
│
└── experiments/
    ├── within_subject/
    │   ├── deep_learning/    { model.ipynb · weights/ · results/ · README.md }
    │   └── feature_based/    { model.ipynb · weights/ · results/ }
    │
    ├── cross_subject/                          # LOSO
    │   ├── deep_learning/    { model.ipynb · weights/ · results/ · README.md }
    │   └── feature_based/    { model.ipynb · weights/ · results/ · README.md }
    │
    ├── clustering/                             # 24 → 8 channel reduction (feature-based)
    │   ├── clustering_features.ipynb           # selects 8 representative channels
    │   ├── anatomical_validation.ipynb         # validates forearm coverage
    │   ├── archive/                            # earlier raw-signal clustering
    │   └── results/
    │
    └── efficient/                              # reduced-channel models (8 channels)
        ├── deep_learning/    { model.ipynb · weights/ · results/ }
        └── feature_based/    { model.ipynb · weights/ · results/ }
```

---

## Pipeline Overview

```
Raw CSVs  (data/cvs_data_per_subject/)
   │
   ▼
[MATLAB]  data_preprocessing/matlab/put_emg_driver.m
   CSV → extract 24 channels → detect gesture blocks → combinedCell (N_reps × 7)
   Output: data/UG_per_subject/
   │
   ▼
[Python]  data_preprocessing/driver.ipynb  →  preprocessing.py
   Bandpass 20–500 Hz → resample to 1500 samples → z-score normalize per channel
   │
   ├─── Deep Learning ─────────────────────────────────────────────────────────
   │    src/emg_loader.py  →  load_all_subjects() + make_loso_train_val_test()
   │    Input: (batch, 1, 24, 1500)
   │    experiments/{within,cross}_subject/deep_learning/model.ipynb
   │
   └─── Feature-Based ─────────────────────────────────────────────────────────
        src/feature_extraction.py  →  batch_extract()
        Sliding window (size=250, shift=50) → 50 feature groups × 24 channels = 3192-dim/window
        mode="sequence" → (26, 3192) per rep
        experiments/{within,cross}_subject/feature_based/model.ipynb
```

---

## Results

### Cross-Subject (LOSO) — 44/44 folds complete (deep learning)

| Model          | Mean Acc   | Std      | Notes                          |
|----------------|------------|----------|--------------------------------|
| **EMG_TCN**    | **80.76%** | ±12.38%  | Best cross-subject model       |
| EEGNet         | 77.23%     | ±12.05%  |                                |
| ShallowConvNet | 73.38%     | ±11.94%  |                                |
| Feature-based  | —          | —        | Training not yet run           |

### Within-Subject (3-fold CV) — 38 subjects (deep learning)

| Model          | Mean Acc   | Std     |
|----------------|------------|---------|
| EMG_TCN        | 97.47%     | ±2.78%  |
| ShallowConvNet | 93.50%     | ±4.09%  |
| EEGNet         | 91.05%     | ±5.56%  |
| putEMG paper (SVM+RMS) | ~90% | —   |

> All three deep learning models exceed the published ~90% within-subject benchmark.
> Cross-subject accuracy is significantly lower (~73–81%), with high variance (±12 pp),
> indicating that subject-adaptive methods will likely be needed for deployment.

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
- `experiments/within_subject/deep_learning/model.ipynb`
- `experiments/cross_subject/deep_learning/model.ipynb`

For cross-subject, set `MODEL_TYPE = 'EMG_TCN'` (or `'EEGNet'` / `'ShallowConvNet'`) and run.

### 4. Feature-Based Models
Open:
- `experiments/within_subject/feature_based/model.ipynb`
- `experiments/cross_subject/feature_based/model.ipynb`

Features are extracted once on first run and cached in `/Volumes/KRIS/data/features_sequence/`
(shared between both notebooks).

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
