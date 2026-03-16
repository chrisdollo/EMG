# EMG Gesture Recognition — putEMG Dataset

A deep learning pipeline for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/).

---

## About the Dataset

The **putEMG** dataset ([Kaczmarek et al., 2019](https://www.mdpi.com/1424-8220/19/16/3548)) is a public sEMG benchmark collected at Poznan University of Technology. Key characteristics:

| Property           | Value                                         |
|--------------------|-----------------------------------------------|
| Subjects           | 44 able-bodied participants                   |
| Electrodes         | 24 sEMG channels (sparse matrix configuration on forearm) |
| Sampling rate      | 5120 Hz                                       |
| Gestures           | 7 active gestures: G1, G2, G3, G6, G7, G8, G9 |
| Repetitions        | Up to 40 per gesture per subject (2 sessions) |
| Gesture duration   | 1 s (short protocol) and 3 s (long protocol)  |
| File format        | CSV per recording session                     |

The 24 electrodes are arranged in a sparse grid around the forearm. Each CSV file contains columns for all 24 channels plus a trajectory label column encoding the current gesture ID (1–9), rest (0), or ignore (−1).

Currently trained and evaluated on **5 subjects** (subjects 03–07).

---

## Project Structure

```
putEMG prime/
├── Data/
│   └── X/
│       ├── gesture_per_subject_data_5/     # Per-subject raw .mat files (from MATLAB)
│       ├── preprocessed_file_5/            # Per-subject preprocessed .mat files
│       └── model_ready_5/
│           └── model_ready_5_sub.mat       # Combined 5-subject model-ready dataset
│
├── code/
│   ├── data_processing_pipeline/           # Stage 1 — MATLAB pipeline
│   │   ├── prime_get_sensor_readings_1.m
│   │   ├── prime_split_raw_files_in_blocks_2.m
│   │   ├── prime_organize_action_blocks_in_gesture_3.m
│   │   ├── prime_uniformize_gestures_4.m
│   │   ├── prime_batch_process_raw_dir.m
│   │   └── put_emg_driver.m
│   │
│   ├── 2_data_preprocessing_pipeline.ipynb # Stage 2 — Python signal processing
│   ├── 3_model_evaluation.ipynb            # Stage 3 — Multi-model training & evaluation
│   ├── 4_model_retrain_full.ipynb          # Stage 4 — Retrain best model on full dataset
│   │
│   ├── models.py                           # All model class definitions (importable)
│   ├── data_utils.py                       # Data loading, splits, loaders, train/eval functions
│   │
│   ├── weights/                            # Saved model checkpoints (.pt files)
│   └── archive/                            # Older/superseded notebooks
│
└── README.md
```

---

## Pipeline Overview

```
Raw CSVs
   │
   ▼
[Stage 1 — MATLAB]  data_processing_pipeline/
   │  Extract channels → split into blocks → organize by gesture → save per-subject .mat
   │
   ▼
[Stage 2 — Python]  2_data_preprocessing_pipeline.ipynb
   │  Bandpass filter (20–500 Hz) → resample to 1500 samples → z-score normalize → combine subjects
   │
   ▼
[Stage 3 — Python]  3_model_evaluation.ipynb
   │  Train & evaluate 5 architectures → save best model weights to weights/
   │
   ▼
[Stage 4 — Python]  4_model_retrain_full.ipynb
      Load best checkpoint → retrain on full dataset → save final weights to weights/
```

---

## Stage 1 — MATLAB Preprocessing

The four MATLAB functions form a sequential pipeline, chained by `prime_batch_process_raw_dir.m` and `put_emg_driver.m`.

#### Step 1: `prime_get_sensor_readings_1.m`
Reads a raw putEMG CSV and returns a cleaned sensor matrix.
- Extracts the 24 EMG channel columns (columns 2–25)
- Reads the trajectory label column (column 26)
- Zeroes out rows where the label is `−1` (relax periods)
- Appends the label as column 25

**Output:** `(N × 25)` — 24 EMG channels + label

#### Step 2: `prime_split_raw_files_in_blocks_2.m`
Splits the sensor matrix into action blocks by detecting transitions in the label column.
- Finds contiguous segments where `label >= 0`

**Output:** Cell array of action blocks, each `(Mi × 25)`

#### Step 3: `prime_organize_action_blocks_in_gesture_3.m`
Scans each action block for gesture executions and organizes them into a structured table.
- Detects sub-segments labeled 1–9 within each block
- Identifies gesture ID using mode of labels (robust to noise)
- Keeps only gestures in `{1, 2, 3, 6, 7, 8, 9}` (skips 4, 5)
- Returns a rectangular cell table with columns `G1, G2, G3, G6, G7, G8, G9`

**Output:** MATLAB table `(M × 7)` — each cell is one gesture repetition `(samples × 24)`

#### Step 4: `prime_batch_process_raw_dir.m` + `put_emg_driver.m`
Orchestrates the pipeline across all subjects.
- `prime_batch_process_raw_dir.m`: runs Steps 1–3 on every CSV in a folder
- `put_emg_driver.m`: iterates over subject subfolders, saves per-subject `.mat` files

Each `.mat` contains `combinedCell` — shape `(N × 7)`, each cell `(M_raw × 24)`.

---

## Stage 2 — Python Preprocessing (`2_data_preprocessing_pipeline.ipynb`)

Takes per-subject `.mat` files and produces a uniform, model-ready dataset.

#### Processing chain (per gesture repetition)

```
Raw (M_raw × 24) at 5120 Hz
    │
    ├─ Bandpass filter 20–500 Hz  (zero-phase Butterworth, order 4)
    │
    ├─ Resample to 1500 samples   (scipy.signal.resample, per channel)
    │
    └─ Per-channel z-score normalization  (zero mean, unit variance over time)
```

Both the bandpass filter and z-score normalization are toggleable via `APPLY_BANDPASS` and `APPLY_ZSCORE` flags in the config cell.

#### Combining subjects
Processed per-subject `.mat` files are stacked into a single combined file saved as `model_ready_5_sub.mat`.

**Final dataset:** `(200 × 7)` cell table — 200 repetitions × 7 gestures = **1,400 samples**, each `(1500 × 24)`.

---

## Stage 3 — Model Training & Evaluation (`3_model_evaluation.ipynb`)

Trains and evaluates 5 architectures on the same data splits. Data loading, splitting, and all training/evaluation functions are imported from `data_utils.py`. Model definitions are imported from `models.py`.

#### Data splits

| Split | Size  | Purpose |
|-------|-------|---------|
| Train | 1,008 | Model training |
| Dev   | 112   | Early stopping signal |
| Test  | 280   | Final held-out evaluation |

Splits use stratified sampling (`random_state=42`). The test set is never seen during training.

#### Models

| # | Model | Description |
|---|-------|-------------|
| 1 | **EEGNet** | Depthwise separable CNN (Lawhern et al., 2018) |
| 2 | **ShallowConvNet** | Temporal + spatial conv, square/log nonlinearity (Schirrmeister et al., 2017) |
| 3 | **DeepConvNet** | 4 stacked conv blocks, increasing filter depth (Schirrmeister et al., 2017) |
| 4 | **CNN_LSTM** | Spatial CNN → temporal pooling → 2-layer LSTM |
| 5 | **EMG_TCN** | Spatial mixing + 4 dilated temporal conv residual blocks |

All models accept input `(batch, 1, 24, 1500)` and output `(batch, 7)` logits.

#### Training setup

- **Optimizer:** Adam, initial LR = `1e-3`
- **LR scheduling:** `ReduceLROnPlateau` — halves LR after 5 epochs of no dev accuracy improvement, floor `1e-6`
- **Early stopping:** patience = 15 epochs, min delta = 0.002
- **Epoch limit:** 50
- **Best-state checkpointing:** best dev accuracy weights restored before test evaluation

#### Results (5 subjects, 1,400 samples)

| Model | Test Accuracy |
|-------|--------------|
| EMG_TCN | **93.57%** |
| EEGNet | 86.43% |
| ShallowConvNet | 81.07% |
| DeepConvNet | 59.64% |
| CNN_LSTM | 51.43% |
| Random baseline | 14.3% |
| putEMG paper (SVM + RMS) | ~90% |

Best model weights are saved to `weights/<ModelName>_best.pt`.

---

## Stage 4 — Full-Dataset Retraining (`4_model_retrain_full.ipynb`)

Loads a checkpoint from `weights/` and retrains that model on **all available data** (no held-out test set) to produce final deployment weights.

- Set `CHECKPOINT_PATH` to any `.pt` file in `weights/`
- Model architecture and dropout are read automatically from the checkpoint
- Same training setup as Stage 3 (ReduceLROnPlateau + early stopping), monitored on training loss
- Final weights saved as `weights/<ModelName>_final.pt`

---

## Shared Modules

### `models.py`
Contains all five model class definitions. Import in any notebook:
```python
from models import EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
```

### `data_utils.py`
Handles data loading, splitting, DataLoaders, and training/evaluation functions. Executes data loading at import time and exports ready-to-use variables:
```python
from data_utils import train, evaluate, evaluateFinal
from data_utils import X_train, X_dev, X_test, y_train, y_dev, y_test
from data_utils import train_loader, dev_loader, test_loader, full_loader
```

---

## Requirements

### MATLAB
- MATLAB R2019b or later (`containers.Map`, `table`, `interp1`)

### Python
```
scipy
numpy
matplotlib
scikit-learn
torch
```

```bash
pip install scipy numpy matplotlib scikit-learn torch
```

---

## How to Run

### 1. MATLAB Preprocessing
```matlab
% Edit put_emg_driver.m to set your data paths, then run:
run('code/data_processing_pipeline/put_emg_driver.m')
```

### 2. Python Signal Processing
Open and run `code/2_data_preprocessing_pipeline.ipynb`.
Update `INPUT_DIR`, `OUTPUT_DIR`, and `COMBINED_OUT` paths in the config cell.

### 3. Model Training & Evaluation
Open and run `code/3_model_evaluation.ipynb`.
Adjust `MAX_EPOCHS`, `DROPOUT`, `LR`, and other hyperparameters in the config cell.
Best model weights are saved automatically to `code/weights/`.

### 4. Full-Dataset Retraining (optional)
Open `code/4_model_retrain_full.ipynb`.
Set `CHECKPOINT_PATH` to the desired `.pt` file, then run all cells.

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors, 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.** *Journal of Neural Engineering.* https://arxiv.org/abs/1611.08024
- Schirrmeister, R. T., et al. (2017). **Deep learning with convolutional neural networks for EEG decoding and visualization.** *Human Brain Mapping.* https://doi.org/10.1002/hbm.23730
- putEMG dataset: https://biolab.put.poznan.pl/putemg-dataset/
