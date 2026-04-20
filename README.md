# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning approach** — CNN/TCN models on raw time-series (best: EMG_TCN, 93.57%)
- **Feature approach** — hand-crafted features (8 × 24 channels = 192-dim) fed into SVM/XGBoost/MLP or sequence models, across three classification formats

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
├── data/
│   ├── cvs_data_per_subject/          # Raw CSVs — one folder per subject
│   ├── NUG_per_subject/               # Per-subject non-uniform gesture .mat files (MATLAB output)
│   └── UG_per_subject/                # Per-subject uniform gesture .mat files
│
├── data_preprocessing/
│   ├── gesture_splitting_pipeline/    # MATLAB: CSV → per-subject combinedCell .mat
│   │   ├── put_emg_driver.m           # Entry point — set paths and run
│   │   ├── prime_get_sensor_readings_1.m
│   │   ├── prime_split_raw_files_in_blocks_2.m
│   │   ├── prime_organize_action_blocks_in_gesture_3.m
│   │   └── prime_uniformize_gestures_4.m
│   ├── preprocessing.py               # Python: bandpass, resample, z-score normalize, combine
│   ├── feature_extraction.py          # Sliding-window feature extraction (3 output modes)
│   └── driver.ipynb                   # Driver notebook for preprocessing + feature extraction
│
├── baseline_models/
│   ├── deep_learning_approach/        # Format 4 — raw signal end-to-end CNN/TCN
│   │   ├── model/
│   │   │   ├── model.ipynb            # Training & evaluation
│   │   │   ├── models.py              # EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
│   │   │   ├── emg_datahandler.py     # Data loading, splitting, train/eval utilities
│   │   │   └── weights/
│   │   │       └── EMG_TCN_best.pt    # Best checkpoint — 93.57% test accuracy
│   │   └── README.md
│   │
│   ├── format1_flat_rep/              # Format 1 — all windows concatenated per rep → (4992,)
│   │   └── README.md
│   │
│   ├── format2_flat_window/           # Format 2 — one sample per window → (192,), real-time capable
│   │   ├── model/
│   │   │   └── model.ipynb            # SVM / XGBoost / MLP training
│   │   └── README.md
│   │
│   └── format3_sequence/              # Format 3 — temporal sequence per rep → (26, 192)
│       ├── model/
│       │   ├── model.ipynb            # LSTM / GRU / Transformer training
│       │   └── handler.py             # Data loading utilities for sequence format
│       └── README.md
│
├── clustering/
│   └── clustering.ipynb               # K-Means channel clustering (PCA → k=8, work in progress)
│
├── feature_approach_plan.txt          # Specification of all 4 formats
├── claudeAnalysis.txt                 # Full project analysis and improvement roadmap
└── claudeReport.txt                   # High-level project overview
```

---

## Pipeline Overview

```
Raw CSVs  (data/cvs_data_per_subject/)
   │
   ▼
[MATLAB]  data_preprocessing/gesture_splitting_pipeline/put_emg_driver.m
   CSV → extract 24 channels → detect gesture blocks → combinedCell (N_reps × 7)
   Output: data/NUG_per_subject/   and   data/UG_per_subject/
   │
   ▼
[Python]  data_preprocessing/driver.ipynb  →  preprocessing.py
   Bandpass 20–500 Hz → resample to 1500 samples → z-score normalize per channel
   │
   ├─── FORMAT 4: Deep Learning (raw signal) ───────────────────────────────────
   │
   ▼
baseline_models/deep_learning_approach/model/model.ipynb
   Train EEGNet / ShallowConvNet / DeepConvNet / CNN_LSTM / EMG_TCN
   Input: (batch, 1, 24, 1500) — Best: EMG_TCN at 93.57%
   │
   ├─── FORMATS 1–3: Feature Approach ──────────────────────────────────────────
   │
   ▼
data_preprocessing/feature_extraction.py
   Sliding window (size=250, shift=50) → 8 features × 24 channels = 192-dim per window
   │
   ├── mode="flat_rep"      → (4992,) per rep  → baseline_models/format1_flat_rep/
   ├── mode="flat_window"   → (192,)  per window → baseline_models/format2_flat_window/   [real-time]
   └── mode="sequence"      → (26, 192) per rep  → baseline_models/format3_sequence/
```

---

## Format Summary

| Format | Sample shape     | Dataset size (per subject) | Real-time? | Models                         |
|--------|------------------|-----------------------------|------------|--------------------------------|
| 1      | `(4992,)`        | 280 reps                    | No         | SVM, XGBoost, MLP              |
| 2      | `(192,)`         | 7280 windows                | **Yes**    | SVM, XGBoost, MLP, 2D CNN      |
| 3      | `(26, 192)`      | 280 reps                    | No         | LSTM, GRU, Transformer, TCN    |
| 4      | `(1, 24, 1500)`  | 280 reps                    | No         | EEGNet, TCN, CNN-LSTM          |

---

## Results (5 subjects, deep learning)

| Model                  | Test Accuracy |
|------------------------|---------------|
| **EMG_TCN**            | **93.57%**    |
| EEGNet                 | 86.43%        |
| ShallowConvNet         | 81.07%        |
| DeepConvNet            | 59.64%        |
| CNN_LSTM               | 51.43%        |
| putEMG paper (SVM+RMS) | ~90%          |

Feature approach (Formats 1–3): training in progress — no results yet.

---

## How to Run

### 1. MATLAB Preprocessing
```matlab
% Edit put_emg_driver.m to set your data paths, then run:
run('data_preprocessing/gesture_splitting_pipeline/put_emg_driver.m')
```

### 2. Python Signal Preprocessing
Open and run `data_preprocessing/driver.ipynb`.

### 3. Feature Extraction
```python
from feature_extraction import batch_extract_features

batch_extract_features(mode="flat_window")   # Format 2 — one sample per window
batch_extract_features(mode="flat_rep")      # Format 1 — full rep as one flat vector
batch_extract_features(mode="sequence")      # Format 3 — temporal sequence per rep
```

### 4. Deep Learning (Format 4)
Open and run `baseline_models/deep_learning_approach/model/model.ipynb`.  
See `baseline_models/deep_learning_approach/README.md` for details.

### 5. Feature-Based Models
- **Format 1:** `baseline_models/format1_flat_rep/` — see README.md
- **Format 2:** `baseline_models/format2_flat_window/model/model.ipynb`
- **Format 3:** `baseline_models/format3_sequence/model/model.ipynb`

---

## Requirements

### MATLAB
MATLAB R2019b or later (`containers.Map`, `table`, `interp1`)

### Python
```bash
pip install scipy numpy matplotlib scikit-learn torch xgboost
```

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors, 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet.** *Journal of Neural Engineering.* https://arxiv.org/abs/1611.08024
- Schirrmeister, R. T., et al. (2017). **Deep learning with CNNs for EEG.** *Human Brain Mapping.* https://doi.org/10.1002/hbm.23730
- putEMG dataset: https://biolab.put.poznan.pl/putemg-dataset/
