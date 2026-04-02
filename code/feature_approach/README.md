# Feature Approach

Hand-crafted EMG feature extraction and classical/hybrid ML pipeline for the putEMG dataset.
Classical alternative to the deep learning models in `../deep_learning_approach/`.

## Workflow

```
Raw per-subject .mat files (folder)
        │
        ▼
batch_extract_features()          ← feature_extraction.py
        │
        ▼
features_subject_XX.mat (folder)
        │
        ▼
model.ipynb  →  SVM · XGBoost · MLP  →  accuracy + confusion matrix
```

## Files

| File | Description |
|------|-------------|
| `feature_extraction.py` | Sliding-window feature extraction — single-file and batch APIs |
| `driver.ipynb` | Quick-start notebook — shows single-file and batch extraction usage |
| `model.ipynb` | Trains and evaluates SVM, XGBoost, and MLP on extracted features |

---

## feature_extraction.py

### Single-file extraction

```python
from feature_extraction import extract_features

extract_features(
    input_filepath  = "path/to/emg_gestures_03_combined_non_uniform.mat",
    output_filepath = "path/to/features_subject_03.mat",
    window_size     = 250,    # samples per window
    window_shift    = 50,     # stride between windows
    sampling_rate   = 5120.0, # Hz — set to effective rate if using resampled data
)
```

### Batch extraction (whole folder)

```python
from feature_extraction import batch_extract_features

batch_extract_features(
    input_dir     = "path/to/raw_mat_files/",
    output_dir    = "path/to/feature_files/",
    window_size   = 250,
    window_shift  = 50,
    sampling_rate = 5120.0,
)
```

Processes every `.mat` file directly in `input_dir` (no subdirectory recursion).
Output filename is derived from the subject ID in the input filename:
`emg_gestures_03_combined_non_uniform.mat` → `features_subject_03.mat`
Existing output files are always deleted and re-created.

### Input format

Per-subject `.mat` file from the MATLAB pipeline (or the Python preprocessing pipeline).

| Property | Value |
|----------|-------|
| MAT key | `combinedCell` |
| Shape | `(N_reps, 7)` — one cell per gesture repetition |
| Each cell | `(M_samples, 24)` — 24 EMG channels |
| Sampling rate (raw) | 5120 Hz |
| Sampling rate (preprocessed) | effective rate based on resample target |

### Output `.mat` keys

| Key | Shape | Description |
|-----|-------|-------------|
| `X` | `(n_windows, 192)` | Feature matrix — 8 features × 24 channels |
| `y` | `(n_windows, 1)` | Gesture class label (0 = G1 … 6 = G9) |
| `feature_names` | `(1, 8)` cell | Names of the 8 features |
| `gesture_names` | `(1, 7)` cell | `["G1","G2","G3","G6","G7","G8","G9"]` |
| `n_channels` | scalar | 24 |
| `n_features_per_ch` | scalar | 8 |
| `window_size` | scalar | as passed |
| `window_shift` | scalar | as passed |
| `sampling_rate` | scalar | as passed |

Column order of `X`: all 8 features for channel 0, then all 8 for channel 1, ..., channel 23.

### 8 Features (per channel)

| # | Name | Description |
|---|------|-------------|
| 1 | MAV | Mean Absolute Value |
| 2 | RMS | Root Mean Square |
| 3 | WL | Waveform Length — sum of successive absolute differences |
| 4 | ZC | Zero Crossings — noise-thresholded sign changes |
| 5 | SSC | Slope Sign Changes — sign changes in the first difference |
| 6 | VAR | Variance |
| 7 | MNF | Mean Frequency — power-weighted mean of the PSD |
| 8 | MDF | Median Frequency — frequency where cumulative PSD reaches 50% |

### Load the output in Python

```python
import scipy.io
import numpy as np

data = scipy.io.loadmat("features_subject_03.mat")
X = data["X"]           # (n_windows, 192)  float64
y = data["y"].squeeze() # (n_windows,)       float64 → cast to int for sklearn
```

### Load the output in MATLAB

```matlab
data = load('features_subject_03.mat');
X = data.X;   % n_windows × 192
y = data.y;   % n_windows × 1
```

---

## model.ipynb

Trains three classifiers on feature files produced by `batch_extract_features` and compares them.

### Data loading

```python
FEATURES_DIR = 'path/to/your/features/folder'
```

`load_features_from_dir()` loads all `features_subject_*.mat` files in the folder and
concatenates their `X` and `y` arrays into a single dataset.

### Train / val / test split

Same strategy as the deep learning notebook:
- 80% train+val / 20% held-out test
- Training portion further split 90/10 into train/val (used for early stopping)

### Models

| # | Model | Training | Notes |
|---|-------|----------|-------|
| 1 | **SVM (RBF)** | Full-batch (sklearn) | StandardScaler included via Pipeline; may be slow on large datasets |
| 2 | **XGBoost** | Full-batch (xgboost) | Early stopping after 20 rounds, monitored on val set |
| 3 | **MLP** | Mini-batches (PyTorch) | 192→256→128→64→7; Adam + ReduceLROnPlateau + early stopping |

### MLP architecture

```
Input (batch, 192)
  → Linear(192,256) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(256,128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(128, 64) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64, 7)
Output (batch, 7) — logits
```

### Output

Each model produces a test accuracy print and a confusion matrix plot.
A summary table and bar chart comparing all three models are shown at the end.

---

## Notes

- Windows that would extend past the end of a segment are dropped (no zero-padding).
- Gesture segments shorter than `window_size` are skipped entirely.
- At 5120 Hz: window_size=250 → ~48.8 ms windows, shift=50 → ~9.8 ms stride.
- At resampled 1500 samples (from ~2 s raw): window_size=250 → ~333 ms windows.
- SVM with RBF kernel can be very slow on large window counts. Swap to `LinearSVC` if needed.

---

## Changelog

### 2026-04-01
- **`feature_extraction.py`**: added `batch_extract_features(input_dir, output_dir, ...)` — processes all `.mat` files in a folder, parses subject ID via regex, always re-creates output files. Added `import re` and `import glob` at module level.
- **`driver.ipynb`**: added second cell demonstrating `batch_extract_features`.
- **`model.ipynb`**: new notebook — loads feature files from a folder, splits data (80/20 + 90/10), trains SVM / XGBoost / MLP, and shows accuracy + confusion matrix for each model with a final comparison bar chart.
- **`README.md`**: updated to document batch API, `model.ipynb` workflow, and MLP architecture.
