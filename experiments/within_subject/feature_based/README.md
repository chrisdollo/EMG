# Feature-Based Approach — Within-Subject (3-fold CV)

Sequence models trained on hand-crafted libemg features, evaluated with
within-subject 3-fold stratified cross-validation. Directly comparable to
the putEMG paper's ~90% SVM/RMS benchmark.

Each gesture rep is converted to a **(26, 3192)** temporal sequence:
50 feature groups × 24 channels = 3192-dim per window, 26 windows per rep.

---

## Protocol

For each of the 44 subjects individually:
- Split their ~280 repetitions into **3 stratified folds**
- **Train** on 2 folds (~187 reps) | **Test** on 1 fold (~93 reps)
- Rotate 3 times → 3 fold accuracies per subject → per-subject mean
- **Final result**: grand mean ± std across all subjects

This is a **personalized model** scenario — each subject's model is trained only on
their own data.

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
experiments/within_subject/feature_based/
├── model.ipynb                # Training & evaluation notebook
├── README.md                  ← this file
├── weights/
│   ├── FeatureLSTM/
│   │   ├── subject_03.pt
│   │   └── ...                ← one file per subject
│   ├── FeatureGRU/
│   │   └── ...
│   └── FeatureTransformer/
│       └── ...
└── results/
    └── report.txt             ← grand summary + per-subject table
```

Each checkpoint stores: `model_name`, `subject_id`, `fold_accs`, `mean_acc`,
`best_fold`, `state_dict`, `dropout`, `n_folds`.

---

## Prerequisites

Run `data_preprocessing/driver.ipynb` first to generate per-subject `.mat` files.  
Features are extracted automatically by this notebook on first run and cached in:
```
/Volumes/KRIS/data/features_sequence/
```
This cache is shared with `experiments/cross_subject/feature_based/`.

---

## How to Run

Open and run `model.ipynb`.

- Set `N_SUBJECTS = 5` in the config cell for a quick trial before committing to all 44
- Already-trained subject-model pairs are skipped automatically on re-run
- Weights are saved immediately after each subject-model pair completes

---

## Feature Extraction

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

## Key Differences from `experiments/within_subject/deep_learning/`

| | Feature-based | Deep learning |
|---|---|---|
| Input | (26, 3192) feature sequence | (1, 24, 1500) raw signal |
| Models | LSTM / GRU / Transformer | EMG_TCN / EEGNet / ShallowConvNet |
| Feature engineering | libemg 50-feature extractor | End-to-end learned |
| Best within-subject result | — (not yet run) | EMG_TCN 97.47% ±2.78% |

---

## Shared Code

- `src/feature_based_models.py` — FeatureLSTM, FeatureGRU, FeatureTransformer
- `src/emg_loader.py` — `load_feature_subjects()`, `make_within_subject_loaders()`
- `src/feature_extraction.py` — `batch_extract()` (libemg 50-feature extractor)

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). putEMG — A Surface EMG Hand
  Gesture Recognition Dataset. *Sensors, 19*(16), 3548.
  https://doi.org/10.3390/s19163548
