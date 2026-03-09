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

The 24 electrodes are arranged in a sparse grid around the forearm, enabling electrode-placement-independent classification. Each CSV file contains columns for all 24 channels plus a trajectory label column encoding the current gesture ID (1–9), rest (0), or ignore (−1).

---

## Project Structure

```
putEMG prime/
├── Data/
│   ├── raw/                          # Raw CSV files from putEMG
│   ├── data_5_subjects/              # Per-subject processed .mat files
│   │   └── total/
│   │       └── data_for_5_subject.mat   # Combined 5-subject dataset
│   └── combined_per_per_subject/     # Intermediate combined outputs
│
├── code/
│   ├── data_processing_pipeline/     # MATLAB preprocessing pipeline
│   │   ├── prime_get_sensor_readings_1.m
│   │   ├── prime_split_raw_files_in_blocks_2.m
│   │   ├── prime_organize_action_blocks_in_gesture_3.m
│   │   ├── prime_uniformize_gestures_4.m
│   │   ├── prime_batch_process_raw_dir.m
│   │   └── put_emg_driver.m
│   │
│   ├── data_preprocessing_pipeline.ipynb   # Python resampling + combining
│   └── model_latest.ipynb                  # EEGNet model + training
│
└── README.md
```

---

## Pipeline

The pipeline runs in two stages: MATLAB preprocessing, then Python model training.

### Stage 1 — MATLAB Preprocessing

The four MATLAB functions form a sequential pipeline. They are chained together and batched by `prime_batch_process_raw_dir.m` and `put_emg_driver.m`.

```
Raw CSV  →  [Step 1]  →  [Step 2]  →  [Step 3]  →  .mat file
```

#### Step 1: `prime_get_sensor_readings_1.m`

Reads a raw putEMG CSV and returns a cleaned sensor matrix.

- Extracts the 24 EMG channel columns (columns 2–25)
- Reads the trajectory label column (column 26)
- Zeroes out all rows where the label is `−1` (subject was allowed to relax between gestures)
- Appends the label column as column 25

**Input:** CSV file path
**Output:** Matrix of shape `(N × 25)` — 24 EMG channels + label

---

#### Step 2: `prime_split_raw_files_in_blocks_2.m`

Splits the sensor matrix into separate action blocks by detecting transitions in the label column.

- Finds contiguous segments where `label >= 0` (valid recording periods)
- Each segment becomes one "action block"

**Input:** `(N × 25)` sensor matrix
**Output:** Cell array of action blocks, each of shape `(Mi × 25)`

---

#### Step 3: `prime_organize_action_blocks_in_gesture_3.m`

Scans each action block for individual gesture executions and organizes them into a structured table.

- Within each block, detects sub-segments labeled 1–9
- Identifies gesture ID using mode of labels in the segment (robust to label noise)
- Only keeps gestures in the set `{1, 2, 3, 6, 7, 8, 9}` (skips 4, 5)
- Accumulates repetitions per gesture across all blocks
- Returns a rectangular cell table with columns `G1, G2, G3, G6, G7, G8, G9`

**Input:** Cell array of action blocks
**Output:** MATLAB table of shape `(M × 7)` — each cell is one gesture repetition `(samples × 24)`

---

#### Step 4 (implicit): `prime_batch_process_raw_dir.m` + `put_emg_driver.m`

Orchestrates the pipeline across all subjects and sessions.

- `prime_batch_process_raw_dir.m`: runs Steps 1–3 on every CSV in a folder, concatenates tables row-wise
- `put_emg_driver.m`: iterates over all subject subfolders, calls the batch processor, and saves per-subject `.mat` files

Each `.mat` file contains a `combinedCell` variable — a cell array of shape `(N × 7)` where each cell holds one gesture repetition of shape `(M_raw × 24)`. Lengths vary per repetition at this stage.

---

### Stage 2 — Python Preprocessing (`data_preprocessing_pipeline.ipynb`)

Takes the per-subject `.mat` files and prepares a uniform dataset for the model.

#### Uniformization

Each gesture repetition has a different raw length (min ~5,124, max ~15,373 samples, median ~10,248). All repetitions are resampled to exactly **1500 samples** per channel using `scipy.signal.resample`.

```python
# Resamples each (M × 24) cell to (1500 × 24)
resampled = signal.resample(gesture_data[:, ch], target_length=1500)
```

#### Combining Subjects

Per-subject `.mat` files are stacked vertically into a single combined cell array and saved as `data_for_5_subject.mat`.

**Final dataset shape:** `(200 × 7)` cell table — 200 repetitions per gesture, 7 gestures (1,400 total samples), each of shape `(1500 × 24)`.

---

### Stage 3 — Model Training (`model_latest.ipynb`)

#### Data Loading

The combined `.mat` file is loaded and flattened into arrays:

```
X: (1400, 1, 24, 1500)   — (samples, 1, channels, time)
Y: (1400,)               — integer class labels 0–6
```

An 80/20 stratified train/test split is applied (1120 train / 280 test).

#### Model Architecture — EEGNet

The model is an **EEGNet** ([Lawhern et al., 2018](https://arxiv.org/abs/1611.08024)), a compact CNN originally designed for EEG classification, adapted here for EMG.

```
Input: (batch, 1, 24, 1500)
    │
    ├─ Temporal Conv2D (1 → F1=8 filters, kernel 1×64) + BatchNorm
    │
    ├─ Depthwise Conv2D (spatial across 24 channels, groups=F1) + BN + ELU
    │   └─ AvgPool 1×4  →  time: 1500 → 375
    │
    ├─ Separable Conv2D (F1*D=16 filters, kernel 1×16) + BN + ELU
    │   └─ AvgPool 1×8  →  time: ~47
    │
    └─ Flatten → Linear → 7 classes
```

| Hyperparameter | Value  |
|----------------|--------|
| F1 (temporal filters) | 8 |
| D (depth multiplier)  | 2 |
| F2 (separable filters)| 16 |
| Dropout rate   | 0.1    |
| Learning rate  | 1e-3   |
| Optimizer      | Adam   |
| Loss           | CrossEntropyLoss |
| Batch size     | 16     |

#### Training

1. **Hyperparameter search** via 5-fold StratifiedKFold cross-validation on the training set. Dropout rates `[0.1, 0.2, 0.3]` and learning rates are tested. Best combination: `dropout=0.1, lr=1e-3`.

2. **Final model** trained from scratch on the full training set (1120 samples) for 12 epochs using the best hyperparameters.

3. **Evaluation** on the held-out test set (280 samples), with a confusion matrix.

#### Results

| Split     | Accuracy |
|-----------|----------|
| CV (train)| ~64.9%   |
| Test set  | ~57–58%  |

---

## Requirements

### MATLAB
- MATLAB R2019b or later (uses `containers.Map`, `table`, `interp1`)

### Python
```
scipy
numpy
matplotlib
scikit-learn
torch
```

Install Python dependencies:
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

This produces one `.mat` file per subject in your output directory.

### 2. Python Uniformization + Combining

Open and run `code/data_preprocessing_pipeline.ipynb`.
Update `input_dir` and `output_file_path` to match your local paths.

### 3. Model Training

Open and run `code/model_latest.ipynb`.
Update `file_path` in the data loading cell to point to your combined `.mat` file.

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors, 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.** *Journal of Neural Engineering.* https://arxiv.org/abs/1611.08024
- putEMG dataset download: https://biolab.put.poznan.pl/putemg-dataset/
