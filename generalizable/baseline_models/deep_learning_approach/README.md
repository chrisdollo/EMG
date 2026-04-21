# Deep Learning Approach — Cross-Subject (LOSO)

End-to-end CNN/TCN gesture recognition on raw sEMG signals, evaluated with full
Leave-One-Subject-Out (LOSO) cross-validation across all 44 subjects.

Input shape per repetition: **(1, 24, 1500)** — 1 channel axis, 24 sEMG electrodes, 1500 samples.

---

## Folder Structure

```
deep_learning_approach/
├── model.ipynb          # Training & LOSO evaluation notebook
├── results_log.txt      # Per-subject LOSO results (auto-updated during training)
├── weights/             # Per-subject best checkpoints (EMG_TCN_SS.pt)
├── EEGNet_best.pt       # Legacy: EEGNet checkpoint from 1-fold LOSO (subj 12, 86.07%)
├── EMG_TCN_best.pt      # Legacy: early pooled 5-subject checkpoint (93.57%) — see note
└── README.md
```

> **Note on EMG_TCN_best.pt (93.57%):** This checkpoint is from an early experiment
> that used a pooled random split on subjects 03–07 only. The test subjects were
> **not** held out from training, so the 93.57% figure is optimistic. It is kept for
> reference but should not be compared to the LOSO results below.

---

## Prerequisites

Run `data_preprocessing/` first. The notebook loads preprocessed `.mat` files from:
```
/Volumes/KRIS/data/UG_per_subject/emg_gestures_SS_U.mat
```
Shared data loader: `public/emg_loader.py` — `make_loso_loaders()`, `generate_loso_configs()`  
Shared model definitions: `public/deep_learning_models.py`

---

## How to Run

Open and run `model.ipynb`.  
Set `N_FOLDS` to the number of LOSO folds to run (max 44).  
Best weights per fold are saved automatically to `weights/EMG_TCN_SS.pt`.  
Results are appended to `results_log.txt` after each fold.

---

## LOSO Results — EMG_TCN (44 / 44 folds complete)

| Metric            | Value             |
|-------------------|-------------------|
| **Mean accuracy** | **80.76%**        |
| Std deviation     | ±12.38%           |
| Best fold         | Subj 14 & 48 — 99.29% |
| Worst fold        | Subj 06 — 53.93%  |
| Val accuracy range| 90.70% – 95.85%   |
| Folds completed   | 44 / 44           |

### Per-Subject Results

| Subject | Test Acc | Val Acc | Epoch | Date       |
|---------|----------|---------|-------|------------|
| 03      | 98.21%   | 94.85%  | 20    | 2026-04-20 |
| 04      | 71.07%   | 92.69%  | 9     | 2026-04-20 |
| 05      | 67.14%   | 94.68%  | 17    | 2026-04-20 |
| 06      | 53.93%   | 94.02%  | 11    | 2026-04-20 |
| 07      | 63.57%   | 91.03%  | 5     | 2026-04-20 |
| 08      | 73.21%   | 94.10%  | 19    | 2026-04-20 |
| 09      | 82.50%   | 94.10%  | 20    | 2026-04-20 |
| 10      | 88.93%   | 92.11%  | 6     | 2026-04-20 |
| 11      | 73.57%   | 94.02%  | 19    | 2026-04-20 |
| 12      | 92.50%   | 94.52%  | 20    | 2026-04-20 |
| 13      | 79.29%   | 95.43%  | 16    | 2026-04-20 |
| 14      | 99.29%   | 95.18%  | 20    | 2026-04-20 |
| 15      | 85.00%   | 93.60%  | 13    | 2026-04-20 |
| 16      | 62.50%   | 94.10%  | 18    | 2026-04-20 |
| 17      | 65.36%   | 93.94%  | 20    | 2026-04-20 |
| 18      | 71.07%   | 91.03%  | 5     | 2026-04-20 |
| 19      | 70.36%   | 93.44%  | 8     | 2026-04-20 |
| 20      | 76.79%   | 94.19%  | 18    | 2026-04-20 |
| 22      | 78.21%   | 92.03%  | 8     | 2026-04-20 |
| 23      | 86.07%   | 94.02%  | 17    | 2026-04-21 |
| 24      | 95.00%   | 90.70%  | 4     | 2026-04-21 |
| 25      | 66.07%   | 95.18%  | 20    | 2026-04-21 |
| 26      | 56.07%   | 95.18%  | 18    | 2026-04-21 |
| 27      | 97.50%   | 93.69%  | 14    | 2026-04-21 |
| 29      | 93.57%   | 94.52%  | 20    | 2026-04-21 |
| 30      | 76.79%   | 92.86%  | 13    | 2026-04-21 |
| 31      | 68.93%   | 93.85%  | 16    | 2026-04-21 |
| 33      | 93.21%   | 92.69%  | 12    | 2026-04-21 |
| 34      | 72.86%   | 92.94%  | 9     | 2026-04-21 |
| 35      | 96.79%   | 93.36%  | 16    | 2026-04-21 |
| 36      | 88.21%   | 94.27%  | 18    | 2026-04-21 |
| 38      | 86.43%   | 92.94%  | 19    | 2026-04-21 |
| 39      | 75.71%   | 93.77%  | 13    | 2026-04-21 |
| 42      | 81.43%   | 93.19%  | 13    | 2026-04-21 |
| 43      | 84.64%   | 95.85%  | 16    | 2026-04-21 |
| 45      | 84.29%   | 92.52%  | 11    | 2026-04-21 |
| 46      | 89.29%   | 93.11%  | 11    | 2026-04-21 |
| 47      | 61.07%   | 93.27%  | 12    | 2026-04-21 |
| 48      | 99.29%   | 93.94%  | 16    | 2026-04-21 |
| 49      | 93.93%   | 94.44%  | 18    | 2026-04-21 |
| 50      | 94.29%   | 94.27%  | 19    | 2026-04-21 |
| 51      | 77.14%   | 93.69%  | 18    | 2026-04-21 |
| 53      | 91.79%   | 93.85%  | 20    | 2026-04-21 |
| 54      | 90.71%   | 94.44%  | 19    | 2026-04-21 |

---

## Model Architectures

All models: `NUM_CLASSES=7`, `NUM_CHANNELS=24`, `INPUT_SAMPLES=1500`

| Model         | Architecture                                      |
|---------------|---------------------------------------------------|
| EMG_TCN       | Dilated temporal convolutions + residual blocks   |
| EEGNet        | Depthwise + separable convolutions (BCI standard) |
| ShallowConvNet| Temporal conv → band-power pooling                |

---

## Training Configuration

| Parameter     | Value                                        |
|---------------|----------------------------------------------|
| Optimizer     | Adam                                         |
| LR scheduler  | ReduceLROnPlateau (factor=0.5, patience=5)   |
| Early stop    | patience=5, min_delta=0.002                  |
| Max epochs    | 20                                           |
| Dropout       | 0.1                                          |
| Checkpoint    | Best val-epoch weights restored before test  |
