# Format 3 — Sequence Per Rep

**Classification approach:** Hand-crafted features, temporal sequence of windows per gesture rep.  
**Input to classifier:** `(26, 192)` — 26 timesteps, each a 192-dim feature vector.  
**Label granularity:** One label per gesture rep.  
**Real-time capable:** No — requires the full gesture to be captured first.

---

## What This Approach Does

The preprocessed EMG signal for each gesture rep (1500 samples × 24 channels) is divided
into 26 overlapping windows. Features are extracted from each window identically to
Formats 1 and 2, but instead of flattening, the 26 feature vectors are stacked in
temporal order to form a sequence matrix. Each gesture rep becomes one (26, 192) array —
26 timesteps, each described by the 192 features extracted from that window. The model
receives the full temporal evolution of the gesture and classifies it as a whole.

```
Gesture rep  (1500 samples × 24 channels)
        │
        ▼  sliding window: size=250, shift=50
Window 1 … Window 26   → each (250 × 24)
        │
        ▼  _extract_window() — 8 features per channel
26 × (192,) feature vectors
        │
        ▼  stack in temporal order (do NOT flatten)
Sequence matrix: (26, 192)   [timestep × features]
        │
        ▼
Dataset: (280, 26, 192)   [280 reps, one label per rep]
PyTorch batched input: (batch, 26, 192)   — matches (batch, seq_len, features)
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

At each timestep, the 192-dim vector is ordered: all 8 features for channel 0,
then channel 1, …, channel 23.

---

## Why Sequences?

Gestures are not instantaneous events — they have distinct phases: onset, peak
activation, and offset. A fist closing looks different at t=0 ms vs t=300 ms vs
t=600 ms. Format 2 (flat per window) ignores this by classifying each 49 ms slice
independently. Format 3 gives the model visibility into how features evolve across
the full ~1.3 s gesture duration, which is information that sequence-aware models
can exploit.

---

## Suitable Models

| Model | Notes |
|-------|-------|
| **LSTM** | Reads the 26-step sequence left-to-right; final hidden state → classifier head |
| **BiLSTM** | Reads forward and backward; captures onset and offset simultaneously |
| **GRU** | Lighter alternative to LSTM; often comparable accuracy with fewer parameters |
| **Transformer (encoder)** | Self-attention over all 26 timesteps; use CLS token or mean pooling for classification |
| **CNN-LSTM** | 1D CNN extracts local temporal patterns per timestep, LSTM models the sequence |
| **TCN** | Temporal Convolutional Network — dilated causal convolutions over the sequence |

Input shape for PyTorch sequence models: `(batch, 26, 192)` — matches the standard
`(batch, seq_len, input_size)` convention for `nn.LSTM`, `nn.GRU`, and Transformer encoders.

If you want spatial structure across channels preserved (e.g. for a CNN-LSTM that treats
channels as a spatial dimension), reshape the last dim: `(batch, 26, 24, 8)`.

---

## Train / Test Splitting

Split at the rep level (each rep is one sample). Standard 80/20 train/test split.
No leakage risk since windows are never surfaced individually.

With only 280 samples per subject, sequence models may need data from multiple subjects
to generalize well. Adding more subjects (currently 5, target 15+) is especially
important for this format.

---

## Inference

1. Capture the full gesture signal (1500 samples × 24 channels).
2. Apply the sliding window → 26 windows.
3. Extract features for each window → 26 × (192,) vectors.
4. Stack in temporal order → (26, 192) sequence.
5. Single forward pass through the sequence model → single predicted label.

Latency: must wait for the entire gesture to complete before classifying.

---

## Workflow

```
data/preprocessed/emg_gestures_SS_combined_non_uniform.mat
        │
        ▼
data_preprocessing/feature_extraction.py
  batch_extract_features(mode="sequence")
        │
        ▼
data/features/features_subject_SS_sequence.npz
  Keys: X (n_reps, 26, 192), y (n_reps,)
        │
        ▼
model/model.ipynb  →  LSTM · GRU · Transformer  →  test accuracy + confusion matrix
```

---

## Files

| File | Description |
|------|-------------|
| `model/model.ipynb` | (to be created) Trains and evaluates sequence models |
| `../../data_preprocessing/feature_extraction.py` | Feature extraction — use `mode="sequence"` |

---

## Notes

- Output is saved as `.npz` (NumPy) rather than `.mat` because 3D arrays are more
  naturally handled by NumPy; MATLAB can load `.npz` via `h5py` if needed.
- The 26 timesteps cover ~1.27 s of signal (26 windows × 49 ms per window, with 10 ms
  strides). This is the natural duration of one gesture repetition at the current
  window/stride settings.
- Fewer training samples than Format 2 (280 vs 7280 per subject) means sequence models
  here are most useful as a research comparison rather than a production baseline.
