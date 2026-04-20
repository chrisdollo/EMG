# Format 1 — Flat Per Rep

**Classification approach:** Hand-crafted features, all windows concatenated into one vector per gesture rep.  
**Input to classifier:** `(4992,)` — 26 windows × 192 features, concatenated flat.  
**Label granularity:** One label per gesture rep.  
**Real-time capable:** No — requires the full gesture to be captured first.

---

## What This Approach Does

The preprocessed EMG signal for each gesture rep (1500 samples × 24 channels) is divided
into 26 overlapping windows. Features are extracted from each window identically to
Format 2, but instead of treating each window as an independent sample, all 26 feature
vectors are concatenated end-to-end into a single 4992-dimensional vector. The entire
gesture rep becomes one data point. Temporal order is preserved in the concatenation —
the first 192 values correspond to window 1, the last 192 to window 26.

```
Gesture rep  (1500 samples × 24 channels)
        │
        ▼  sliding window: size=250, shift=50
Window 1 … Window 26   → each (250 × 24)
        │
        ▼  _extract_window() — 8 features per channel
26 × (192,) feature vectors
        │
        ▼  concatenate in temporal order
One flat vector: (4992,)   [window1_feats | window2_feats | … | window26_feats]
        │
        ▼
Dataset: (280, 4992)   [280 reps, one label per rep]
```

**Windows per rep:** `floor((1500 − 250) / 50) + 1 = 26`  
**Dataset size:** 280 reps per subject (40 reps × 7 gesture classes)

---

## Features

Same 8 features per channel as all other feature formats:

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

The 4992-dim vector is ordered: [8 feats × 24 ch for window 1 | … | 8 feats × 24 ch for window 26].

---

## Suitable Models

| Model | Notes |
|-------|-------|
| **SVM (RBF / LinearSVC)** | Strong baseline; 4992-dim is high — use LinearSVC for speed |
| **XGBoost / LightGBM** | Handles high-dimensional input; built-in feature importance |
| **MLP** | Larger first layer needed to handle 4992-dim input |
| **Random Forest** | Robust to high dimensionality; interpretable feature importances |

---

## Train / Test Splitting

Split at the rep level (each rep is already one sample). Standard 80/20 train/test
split is straightforward here. No leakage risk since windows are never surfaced
individually to the model.

---

## Inference

1. Capture the full gesture signal (1500 samples × 24 channels).
2. Apply the sliding window → 26 windows.
3. Extract features for each window → 26 × (192,) vectors.
4. Concatenate in order → one (4992,) vector.
5. Single classifier call → single predicted label.

Latency: must wait for the entire gesture to complete before classifying.

---

## Workflow

```
data/preprocessed/emg_gestures_SS_combined_non_uniform.mat
        │
        ▼
data_preprocessing/feature_extraction.py
  batch_extract_features(mode="flat_rep")
        │
        ▼
data/features/features_subject_SS_flat_rep.mat
  Keys: X (n_reps, 4992), y (n_reps, 1)
        │
        ▼
model/model.ipynb  →  SVM · XGBoost · MLP  →  test accuracy + confusion matrix
```

---

## Files

| File | Description |
|------|-------------|
| `model/model.ipynb` | (to be created) Trains and evaluates classifiers on flat-rep features |
| `../../data_preprocessing/feature_extraction.py` | Feature extraction — use `mode="flat_rep"` |

---

## Notes

- The 4992-dim input is large relative to the 280-sample dataset per subject. Strong
  regularization (C parameter for SVM, max_depth/min_child_weight for XGBoost) is
  important to avoid overfitting.
- Adding more subjects will help significantly — 280 samples for a 4992-dim space is tight.
- This is the simplest offline baseline: no temporal modeling, no sequence structure,
  just the full gesture summarized as one flat vector.
