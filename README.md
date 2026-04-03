# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning approach** — CNN/TCN models on raw time-series (best: EMG_TCN, 93.57%)
- **Feature approach** — hand-crafted features + SVM/XGBoost/MLP, with planned channel reduction via clustering

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

Currently trained and evaluated on **5 subjects** (03–07).

---

## Project Structure

```
putEMG prime/
├── data/
│   ├── raw_data_5/                        # Raw CSVs — 5 subjects (03–07)
│   ├── X/
│   │   ├── gesture_per_subject_data_5/    # Per-subject .mat (MATLAB output)
│   │   ├── preprocessed_file_5/           # Bandpassed + resampled per-subject .mat
│   │   ├── model_ready_5/                 # Combined dataset for deep learning
│   │   └── feature_files/                 # Sliding-window feature .mat files
│   └── uniform_mat_data_per_subject/      # Full 44-subject uniformized .mat files
│
├── data_preprocessing/
│   ├── gesture_splitting_pipeline/        # MATLAB: CSV → per-subject combinedCell .mat
│   ├── preprocessing.py                   # Python: bandpass, resample, normalize, combine
│   └── preprocessing_driver.ipynb         # Driver notebook for preprocessing.py
│
├── deep_learning_approach/
│   ├── model/
│   │   ├── model.ipynb                    # Training & evaluation
│   │   ├── models.py                      # Model class definitions
│   │   ├── emg_datahandler.py             # Data loading & train/eval utilities
│   │   └── weights/                       # Saved checkpoints (.pt)
│   └── README.md
│
├── feature_approach/
│   ├── model/
│   │   ├── feature_extraction.py          # Sliding-window feature extraction
│   │   ├── driver.ipynb                   # Extraction usage examples
│   │   └── model.ipynb                    # SVM / XGBoost / MLP training
│   └── README.md
│
├── clustering/
│   └── clustering.ipynb                   # Channel clustering (work in progress)
│
├── claudeReport.txt                       # High-level project overview and plan
├── claudeAnalysis.txt                     # AI analysis of model performance
└── README.md
```

---

## Pipeline Overview

```
Raw CSVs  (data/raw_data_5/)
   │
   ▼
[MATLAB]  data_preprocessing/gesture_splitting_pipeline/
   CSV → extract 24 channels → detect gesture blocks → combinedCell (N_reps × 7)
   Output: data/X/gesture_per_subject_data_5/
   │
   ▼
[Python]  data_preprocessing/preprocessing_driver.ipynb
   Bandpass 20–500 Hz → resample to 1500 samples → z-score normalize → combine subjects
   Output: data/X/preprocessed_file_5/  and  data/X/model_ready_5/
   │
   ├─── APPROACH A: Deep Learning ─────────────────────────────────────────────
   │
   ▼
deep_learning_approach/model/model.ipynb
   Train EEGNet / ShallowConvNet / DeepConvNet / CNN_LSTM / EMG_TCN
   Input: (batch, 1, 24, 1500) — Best: EMG_TCN at 93.57%
   │
   ├─── APPROACH B: Feature Extraction ────────────────────────────────────────
   │
   ▼
feature_approach/model/feature_extraction.py  (or driver.ipynb)
   Sliding window (size=250, shift=50) → 8 features × 24 channels = 192-dim vector
   Output: data/X/feature_files/
   │
   ▼
feature_approach/model/model.ipynb
   Train SVM / XGBoost / MLP on 192-dim features
   │
   ▼  (next step)
clustering/clustering.ipynb
   Cluster 24 channels → select 8 → retrain with 64-dim input → compare vs baseline
```

---

## How to Run

### 1. MATLAB Preprocessing
```matlab
% Edit put_emg_driver.m to set your data paths, then run:
run('data_preprocessing/gesture_splitting_pipeline/put_emg_driver.m')
```

### 2. Python Signal Preprocessing
Open and run `data_preprocessing/preprocessing_driver.ipynb`.

### 3. Deep Learning
Open and run `deep_learning_approach/model/model.ipynb`.  
See `deep_learning_approach/README.md` for details.

### 4. Feature Approach
```python
# Extract features
from feature_extraction import batch_extract_features
batch_extract_features(input_dir="data/X/gesture_per_subject_data_5/",
                       output_dir="data/X/feature_files/")
```
Or run `feature_approach/model/driver.ipynb`, then `feature_approach/model/model.ipynb`.  
See `feature_approach/README.md` for details.

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
