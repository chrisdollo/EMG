# Feature-Based Approach — Cross-Subject (LOSO)

Sequence models trained on hand-crafted libemg features, evaluated with full
Leave-One-Subject-Out (LOSO) cross-validation across all 44 subjects.

Each gesture rep is converted to a **(26, 3192)** temporal sequence:
50 feature groups × 24 channels = 3192-dim per window, 26 windows per rep.

---

## Status

Training not yet run.

---

## Models Evaluated

| Model | Description |
|-------|-------------|
| **FeatureLSTM** | Bidirectional LSTM over the 26-step feature sequence |
| **FeatureGRU** | Bidirectional GRU — lighter alternative to LSTM |
| **FeatureTransformer** | Self-attention encoder with sinusoidal positional encoding |

All models: `NUM_CLASSES=7`, `INPUT_SIZE=3192`, `SEQ_LEN=26`  
Code: `src/feature_based_models.py`

---

## Folder Structure

```
experiments/cross_subject/feature_based/
├── model.ipynb                         # Training & LOSO evaluation notebook
├── README.md
├── weights/
│   ├── FeatureLSTM/                    # 44 checkpoints — FeatureLSTM_SS.pt
│   ├── FeatureGRU/                     # 44 checkpoints — FeatureGRU_SS.pt
│   └── FeatureTransformer/             # 44 checkpoints — FeatureTransformer_SS.pt
└── results/
    ├── results_log_FeatureLSTM.txt
    ├── results_log_FeatureGRU.txt
    └── results_log_FeatureTransformer.txt
```

---

## Prerequisites

Run `data_preprocessing/driver.ipynb` first to generate per-subject `.mat` files.  
Features are extracted automatically by this notebook on first run and cached in:
```
/Volumes/KRIS/data/features_sequence/
```
This cache is shared with `experiments/within_subject/feature_based/`.

---

## How to Run

Open `model.ipynb`. In the config cell, set `MODEL_TYPE` to the desired model:

```python
MODEL_TYPE = 'FeatureLSTM'    # or 'FeatureGRU' or 'FeatureTransformer'
```

Weights are saved to `weights/<MODEL_TYPE>/<MODEL_TYPE>_SS.pt` per fold.  
Results are logged to `results/results_log_<MODEL_TYPE>.txt` after each fold.  
Completed folds are skipped automatically — safe to stop and resume at any time.

---

## LOSO Split

| Split | Description |
|-------|-------------|
| **Test**  | 1 held-out subject — never seen during training |
| **Train** | 90% of remaining 43 subjects' data (stratified by class) |
| **Val**   | 10% of remaining 43 subjects' data — early stopping only |

---

## Feature Extraction

Sliding window over each 1500-sample rep:

```
Rep: (1500 samples × 24 channels)
     │
     ▼  sliding window: size=250, shift=50
     26 windows × 24 channels
     │
     ▼  libemg: 50 feature groups per channel
     26 × 3192  (seq_len × features)
```

**Window count per rep:** `floor((1500 − 250) / 50) + 1 = 26`  
**Feature vector size:** 50 groups × 24 channels = 3192

---

## Training Configuration

| Parameter     | Value                                        |
|---------------|----------------------------------------------|
| Optimizer     | Adam (lr=1e-3)                               |
| LR scheduler  | ReduceLROnPlateau (factor=0.5, patience=5)   |
| Early stop    | patience=5, min_delta=0.002                  |
| Max epochs    | 20                                           |
| Dropout       | 0.3                                          |
| Batch size    | 16                                           |
| Checkpoint    | Best val-epoch weights restored before test  |

---

## Key Differences from `experiments/cross_subject/deep_learning/`

| | Feature-based | Deep learning |
|---|---|---|
| Input | (26, 3192) feature sequence | (1, 24, 1500) raw signal |
| Models | LSTM / GRU / Transformer | EMG_TCN / EEGNet / ShallowConvNet |
| Feature engineering | libemg 50-feature extractor | End-to-end learned |
| Best cross-subject result | — (not yet run) | EMG_TCN 80.76% ±12.38% |

---

## Shared Code

- `src/feature_based_models.py` — FeatureLSTM, FeatureGRU, FeatureTransformer
- `src/emg_loader.py` — `load_feature_subjects()`, `make_loso_train_val_test()`
- `src/feature_extraction.py` — `batch_extract()` (libemg 50-feature extractor)

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). putEMG — A Surface EMG Hand
  Gesture Recognition Dataset. *Sensors, 19*(16), 3548.
  https://doi.org/10.3390/s19163548
