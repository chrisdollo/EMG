# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning approach** — CNN/TCN models on raw time-series
- **Feature-based approach** — 50 libemg features (3192-dim) or 8 hand-crafted features (192-dim) fed into sequence models

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
│   ├── deep_learning_models.py        # Shared model definitions (EEGNet, EMG_TCN, ShallowConvNet, …)
│   ├── emg_loader.py                  # Shared data loading utilities (LOSO, within-subject splits)
│   └── feature_extraction.py          # libemg feature extraction (50 features → 3192-dim/window)
│
├── data_preprocessing/
│   ├── gesture_splitting_pipeline/    # MATLAB: CSV → per-subject combinedCell .mat
│   │   ├── put_emg_driver.m           # Entry point — set paths and run
│   │   ├── prime_get_sensor_readings_1.m
│   │   ├── prime_split_raw_files_in_blocks_2.m
│   │   ├── prime_organize_action_blocks_in_gesture_3.m
│   │   └── prime_uniformize_gestures_4.m
│   ├── preprocessing.py               # Bandpass, resample, z-score normalize
│   ├── feature_extraction.py          # Legacy 8-feature sliding-window extraction
│   └── driver.ipynb                   # Driver notebook for preprocessing + feature extraction
│
├── generalizable/
│   ├── baseline_models/
│   │   ├── deep_learning_approach/    # Raw signal end-to-end CNN/TCN (LOSO) — COMPLETE
│   │   │   ├── model.ipynb
│   │   │   ├── summary.txt
│   │   │   ├── weights/EMG_TCN/       # 44 per-subject checkpoints
│   │   │   ├── weights/EEGNet/
│   │   │   ├── weights/ShallowConvNet/
│   │   │   └── README.md
│   │   │
│   │   └── feature_based_approach/    # Temporal feature sequences → sequence models
│   │       ├── model.ipynb
│   │       └── README.md
│   │
│   ├── efficient_model/               # Channel-reduced model (24 → 8 channels)
│   └── clustering/                    # Channel reduction via agglomerative clustering
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
   Output: data/UG_per_subject/
   │
   ▼
[Python]  data_preprocessing/driver.ipynb  →  preprocessing.py
   Bandpass 20–500 Hz → resample to 1500 samples → z-score normalize per channel
   │
   ├─── Deep Learning ─────────────────────────────────────────────────────────
   │    public/emg_loader.py  →  load_all_subjects() + make_loso_train_val_test()
   │    Input: (batch, 1, 24, 1500)
   │    generalizable/baseline_models/deep_learning_approach/model.ipynb
   │
   └─── Feature-Based ─────────────────────────────────────────────────────────
        public/feature_extraction.py  →  batch_extract()
        Sliding window (size=250, shift=50) → 50 feature groups × 24 channels = 3192-dim/window
        mode="sequence" → (26, 3192) per rep
        generalizable/baseline_models/feature_based_approach/model.ipynb
```

---

## Results

### Cross-Subject (LOSO) — 44/44 folds complete

| Model          | Mean Acc   | Std      | Notes                          |
|----------------|------------|----------|--------------------------------|
| **EMG_TCN**    | **80.76%** | ±12.38%  | Best cross-subject model       |
| EEGNet         | 77.23%     | ±12.05%  |                                |
| ShallowConvNet | 73.38%     | ±11.94%  |                                |
| Feature-based  | —          | —        | Training not yet run           |

### Within-Subject (3-fold CV) — 38 subjects

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
run('data_preprocessing/gesture_splitting_pipeline/put_emg_driver.m')
```

### 2. Python Signal Preprocessing
Open and run `data_preprocessing/driver.ipynb`.

### 3. Feature Extraction (feature-based approach)
```python
from public.emg_loader import load_all_subjects
from public.feature_extraction import batch_extract

subjects = load_all_subjects('/Volumes/KRIS/data/UG_per_subject')
batch_extract(subjects, output_dir='/path/to/features', mode='sequence')
# Saves features_subject_SS_sequence.npz per subject; skips existing files
```

### 4. Deep Learning (LOSO)
Open `generalizable/baseline_models/deep_learning_approach/model.ipynb`.  
Set `MODEL_TYPE = 'EMG_TCN'` (or `'EEGNet'` / `'ShallowConvNet'`) and run.

### 5. Feature-Based Models
Open `generalizable/baseline_models/feature_based_approach/model.ipynb`.

---

## Requirements

### MATLAB
MATLAB R2019b or later (`containers.Map`, `table`, `interp1`)

### Python
```bash
pip install scipy numpy matplotlib scikit-learn torch libemg
```

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors, 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet.** *Journal of Neural Engineering.* https://arxiv.org/abs/1611.08024
- Schirrmeister, R. T., et al. (2017). **Deep learning with CNNs for EEG.** *Human Brain Mapping.* https://doi.org/10.1002/hbm.23730
- putEMG dataset: https://biolab.put.poznan.pl/putemg-dataset/
