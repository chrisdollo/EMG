# Deep Learning Approach — Cross-Subject (LOSO)

End-to-end CNN/TCN gesture recognition on raw sEMG signals, evaluated with full
Leave-One-Subject-Out (LOSO) cross-validation across all 44 subjects.

Input shape per repetition: **(1, 24, 1500)** — 1 channel axis, 24 sEMG electrodes, 1500 samples.

---

## Status — COMPLETE

All 44 folds complete for all 3 models.

| Model          | Mean Acc  | Std      | Best Fold           | Worst Fold          |
|----------------|-----------|----------|---------------------|---------------------|
| **EMG_TCN**    | **80.76%**| ±12.38%  | Subj 14 & 48 (99.29%) | Subj 06 (53.93%) |
| EEGNet         | 77.23%    | ±12.05%  | Subj 27 (98.21%)    | Subj 06 (48.57%)    |
| ShallowConvNet | 73.38%    | ±11.94%  | Subj 27 (95.00%)    | Subj 17 (37.50%)    |

---

## Folder Structure

```
deep_learning_approach/
├── model.ipynb                    # Training & LOSO evaluation notebook
├── summary.txt                    # Plain-text summary of what was done and all results
├── results_log_EMG_TCN.txt        # Per-subject LOSO results — EMG_TCN (auto-updated)
├── results_log_EEGNet.txt         # Per-subject LOSO results — EEGNet (auto-updated)
├── results_log_ShallowConvNet.txt # Per-subject LOSO results — ShallowConvNet (auto-updated)
├── weights/
│   ├── EMG_TCN/                   # 44 checkpoints — EMG_TCN_SS.pt
│   ├── EEGNet/                    # 44 checkpoints — EEGNet_SS.pt
│   └── ShallowConvNet/            # 44 checkpoints — ShallowConvNet_SS.pt
└── README.md
```

---

## Prerequisites

Run `data_preprocessing/` first. The notebook loads preprocessed `.mat` files from:
```
/Volumes/KRIS/data/UG_per_subject/emg_gestures_SS_U.mat
```
Shared data loader: `public/emg_loader.py`  
Shared model definitions: `public/deep_learning_models.py`

---

## How to Run

Open `model.ipynb`. In the config cell, set `MODEL_TYPE` to the desired model:

```python
MODEL_TYPE = 'EMG_TCN'        # or 'EEGNet' or 'ShallowConvNet'
```

Weights are saved to `weights/<MODEL_TYPE>/<MODEL_TYPE>_SS.pt` per fold.  
Results are logged to `results_log_<MODEL_TYPE>.txt` after each fold.  
Completed folds are skipped automatically — safe to stop and resume at any time.

---

## LOSO Split

| Split | Description |
|-------|-------------|
| **Test**  | 1 held-out subject — never seen during training |
| **Train** | 90% of remaining 43 subjects' data (stratified by class) |
| **Val**   | 10% of remaining 43 subjects' data — early stopping only |

---

## Model Architectures

All models: `NUM_CLASSES=7`, `NUM_CHANNELS=24`, `INPUT_SAMPLES=1500`  
Code: `public/deep_learning_models.py`

| Model          | Architecture                                           |
|----------------|--------------------------------------------------------|
| EMG_TCN        | Spatial mixing (C×1) + 4 dilated TCN residual blocks   |
| EEGNet         | Depthwise + separable convolutions (standard BCI baseline) |
| ShallowConvNet | Temporal conv → square nonlinearity → avg pool → log   |

---

## Training Configuration

| Parameter     | Value                                        |
|---------------|----------------------------------------------|
| Optimizer     | Adam (lr=1e-3)                               |
| LR scheduler  | ReduceLROnPlateau (factor=0.5, patience=5)   |
| Early stop    | patience=5, min_delta=0.002                  |
| Max epochs    | 20                                           |
| Dropout       | 0.1                                          |
| Batch size    | 16                                           |
| Checkpoint    | Best val-epoch weights restored before test  |

---

## Per-Subject Results

Full per-subject tables are in the `results_log_<MODEL_TYPE>.txt` files.
See `summary.txt` for a combined overview and key observations.
