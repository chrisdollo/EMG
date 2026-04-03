# Format 2 — Flat Per Window

**Classification approach:** Hand-crafted features, one independent sample per sliding window.  
**Input to classifier:** `(192,)` — 8 features × 24 channels, one vector per 49 ms window.  
**Label granularity:** One label per window (inherited from its parent gesture rep).  
**Real-time capable:** Yes.

---

## What This Approach Does

The preprocessed EMG signal for each gesture rep (1500 samples × 24 channels) is divided
into overlapping windows using a sliding window. Each window is processed independently:
8 hand-crafted features are computed for every one of the 24 channels, producing a flat
192-dimensional feature vector. Every window is treated as its own data point and assigned
the label of the gesture rep it came from.

```
Gesture rep  (1500 samples × 24 channels)
        │
        ▼  sliding window: size=250, shift=50
Window 1 … Window 26   → each (250 × 24)
        │
        ▼  _extract_window() — 8 features per channel
Feature vector per window: (192,)   [MAV, RMS, WL, ZC, SSC, VAR, MNF, MDF] × 24 channels
        │
        ▼
Dataset: (7280, 192)   [280 reps × 26 windows per rep, one label per window]
```

**Windows per rep:** `floor((1500 − 250) / 50) + 1 = 26`  
**Dataset size:** 280 reps × 26 windows = 7280 samples per subject

---

## Features

8 features are computed per channel per window:

| # | Name | Description |
|---|------|-------------|
| 1 | MAV  | Mean Absolute Value |
| 2 | RMS  | Root Mean Square |
| 3 | WL   | Waveform Length — sum of successive absolute differences |
| 4 | ZC   | Zero Crossings — noise-thresholded sign changes |
| 5 | SSC  | Slope Sign Changes — sign changes in first difference |
| 6 | VAR  | Variance |
| 7 | MNF  | Mean Frequency — power-weighted mean of the PSD |
| 8 | MDF  | Median Frequency — frequency where cumulative PSD reaches 50% |

Column order of the (192,) vector: all 8 features for channel 0, then channel 1, …, channel 23.

---

## Suitable Models

| Model | Notes |
|-------|-------|
| **SVM (RBF / LinearSVC)** | Primary target; strong on 192-dim tabular input |
| **XGBoost / LightGBM** | Handles high-dimensional input well; fast to tune |
| **MLP** | 192→256→128→64→7; Adam + early stopping |
| **2D CNN** | Reshape each sample to (24, 8) before passing in |

Current models implemented in `model/model.ipynb`: SVM (RBF), XGBoost, MLP.

---

## Train / Test Splitting

**Important:** splitting must be done at the rep level, not at the window level.
Windows from the same gesture rep must all land in the same split. Splitting at
the window level would allow windows from the same rep to appear in both train
and test, causing data leakage and inflated accuracy.

---

## Inference

**Offline:**
1. Capture the full gesture signal (1500 samples × 24 channels).
2. Apply the sliding window → 26 windows.
3. Extract features for each window → 26 × (192,) vectors.
4. Pass each vector through the classifier → 26 predictions.
5. Take the majority vote (or soft vote — average class probabilities) → final label.

**Real-time:**
1. As EMG streams in, extract a new window every 50 samples (~10 ms).
2. Compute features → (192,) vector → single classifier call → immediate prediction.
3. Optional: apply a rolling majority vote over the last N windows to smooth output.

---

## Workflow

```
data/preprocessed/emg_gestures_SS_combined_non_uniform.mat
        │
        ▼
data_preprocessing/feature_extraction.py
  batch_extract_features(mode="flat_window")
        │
        ▼
data/features/features_subject_SS_flat_window.mat
  Keys: X (n_windows, 192), y (n_windows, 1)
        │
        ▼
model/model.ipynb  →  SVM · XGBoost · MLP  →  test accuracy + confusion matrix
```

---

## Files

| File | Description |
|------|-------------|
| `model/model.ipynb` | Trains and evaluates SVM, XGBoost, MLP on flat-window features |
| `../../data_preprocessing/feature_extraction.py` | Feature extraction — use `mode="flat_window"` |

---

## Notes

- Windows that extend past the end of a segment are dropped (no zero-padding).
- Segments shorter than `window_size=250` are skipped entirely.
- At 5120 Hz: window_size=250 → ~48.8 ms, shift=50 → ~9.8 ms stride.
- This produces the largest dataset of the three feature formats (7280 vs 280 samples
  per subject), which generally benefits generalization for traditional ML models.
- SVM with RBF kernel can be slow on 7280+ samples. Use `LinearSVC` if needed.
