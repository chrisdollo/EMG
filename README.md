# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning approach** — CNN/TCN models on raw time-series (best: EMG_TCN, 93.57%)
- **Feature-based approach** — hand-crafted features (8 × 24 channels = 192-dim) fed into LSTM/GRU/Transformer sequence models

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
├── public/
│   ├── deep_learning_models.py        # Shared model definitions (EEGNet, EMG_TCN, etc.)
│   └── emg_loader.py                  # Shared data loading utilities (LOSO, within-subject)
│
├── data_preprocessing/
│   ├── gesture_splitting_pipeline/    # MATLAB: CSV → per-subject combinedCell .mat
│   │   ├── put_emg_driver.m           # Entry point — set paths and run
│   │   ├── prime_get_sensor_readings_1.m
│   │   ├── prime_split_raw_files_in_blocks_2.m
│   │   ├── prime_organize_action_blocks_in_gesture_3.m
│   │   └── prime_uniformize_gestures_4.m
│   ├── preprocessing.py               # Bandpass, resample, z-score normalize
│   ├── feature_extraction.py          # Sliding-window feature extraction
│   └── driver.ipynb                   # Driver notebook for preprocessing + feature extraction
│
├── generalizable/
│   ├── baseline_models/
│   │   ├── deep_learning_approach/    # Raw signal end-to-end CNN/TCN (LOSO)
│   │   │   ├── model/
│   │   │   │   ├── model.ipynb        # Training & evaluation
│   │   │   │   └── weights/
│   │   │   └── README.md
│   │   │
│   │   └── feature_based_approach/    # Temporal feature sequences → LSTM/GRU/Transformer
│   │       ├── model/
│   │       │   └── model.ipynb        # Training & evaluation
│   │       └── README.md
│   │
│   ├── efficient_model/               # Efficient model (channel-reduced)
│   │   └── deep_learning_approach/
│   │       └── model/
│   │           ├── deep_learning_models.py
│   │           ├── model.ipynb
│   │           └── weights/
│   │
│   └── clustering/                    # Channel reduction (24 → 8 channels)
│       ├── clustering.ipynb
│       ├── clustering_features.ipynb
│       └── clustering_raw.ipynb
│
└── within_subject/                    # Within-subject 3-fold CV (matches putEMG ~90% benchmark)
    └── deep_learning_approach/
        └── model/model.ipynb
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
   ├─── Deep Learning ───────────────────────────────────────────────────────────
   │    generalizable/baseline_models/deep_learning_approach/model/model.ipynb
   │    Input: (batch, 1, 24, 1500) — Best: EMG_TCN at 93.57%
   │
   └─── Feature-Based ───────────────────────────────────────────────────────────
        data_preprocessing/feature_extraction.py
        Sliding window (size=250, shift=50) → 8 features × 24 channels = 192-dim per window
        mode="sequence" → (26, 192) per rep
        generalizable/baseline_models/feature_based_approach/model/model.ipynb
        Models: LSTM, GRU, Transformer
```

---

## Results

| Model                  | Setting                  | Test Accuracy        |
|------------------------|--------------------------|----------------------|
| EMG_TCN                | pooled 5-subj (no holdout) | 93.57% *(optimistic)* |
| **EMG_TCN**            | **LOSO (15/44 folds)**   | **77.07% ± 13.28%**  |
| EEGNet                 | LOSO (1 fold, subj 12)   | 86.07%               |
| putEMG paper (SVM+RMS) | within-subject           | ~90%                 |

> **Note:** The pooled 93.57% is optimistic — test subjects were seen during training. The LOSO result (77.07%) is the honest cross-subject estimate. High variance (±13.28%) across subjects suggests subject-adaptive fine-tuning may be needed.

Feature-based approach (LSTM/GRU/Transformer): training not yet run.

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
batch_extract_features(mode="sequence")   # (26, 192) per rep → feature_based_approach
```

### 4. Deep Learning
Open and run `generalizable/baseline_models/deep_learning_approach/model/model.ipynb`.

### 5. Feature-Based Models
Open and run `generalizable/baseline_models/feature_based_approach/model/model.ipynb`.

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
