# Within-Subject Deep Learning Evaluation

Replicates the **putEMG paper's evaluation protocol** (Kaczmarek et al., 2019) using
the same CNN/TCN architectures from the cross-subject pipeline.

---

## Protocol

The original paper used **within-subject 3-fold cross-validation**:

> "Two trials were concatenated to generate a training set and the remaining trial was
> used as a testing set." — repeated across 88 subject-session subsets.

This notebook implements the same spirit:

- For each of the 44 subjects individually:
  - Split their ~280 repetitions into **3 stratified folds**
  - **Train** on 2 folds (~187 reps) | **Test** on 1 fold (~93 reps)
  - Rotate 3 times → 3 fold accuracies per subject → per-subject mean
- **Final result**: grand mean ± std across all 44 subjects
- **Winner**: architecture with the highest grand mean is selected for further use

This is a **personalized model** scenario — each subject's model is trained only on
their own data. It is the easiest evaluation setting and directly comparable to the
paper's ~90% SVM/RMS benchmark.

---

## Models Evaluated

| # | Model | Description |
|---|-------|-------------|
| 1 | **EEGNet** | Depthwise separable CNN, original baseline |
| 2 | **ShallowConvNet** | Temporal + spatial conv, square/log nonlinearity |
| 3 | **EMG_TCN** | Spatial mixing + 4 dilated temporal conv blocks |

---

## Key Differences from `experiments/cross_subject/`

| | `experiments/within_subject/` | `experiments/cross_subject/` |
|---|---|---|
| Train data | Same subject | Other subjects |
| Test data | Same subject (held-out fold) | Held-out subject |
| Evaluation | Within-subject 3-fold CV | LOSO cross-subject |
| Comparability | putEMG paper (~90%) | Harder; no published benchmark |
| Use case | Calibrated personal device | Universal classifier |

---

## Files

```
experiments/within_subject/deep_learning/
├── model.ipynb            ← training & evaluation notebook
├── README.md              ← this file
├── weights/
│   ├── EEGNet/
│   │   ├── subject_03.pt
│   │   ├── subject_04.pt
│   │   └── ...            ← one file per subject (38 total)
│   ├── ShallowConvNet/
│   │   └── ...
│   └── EMG_TCN/
│       └── ...
└── results/
    └── report.txt         ← grand summary + per-subject table
```

Each checkpoint (`subject_{id}.pt`) stores:
- `model_name`, `subject_id`, `subject_file`
- `fold_accs` — accuracy for each of the 3 folds
- `mean_acc` — mean across folds
- `best_fold` — index of the best-performing fold
- `state_dict` — weights from the best fold
- `dropout`, `n_folds`

Model definitions and data utilities are shared from `src/`:
- `src/deep_learning_models.py` — EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
- `src/emg_loader.py` — `make_within_subject_loaders()`, `load_all_subjects()`

---

## How to Run

Open and run `model.ipynb`.

- Set `N_SUBJECTS = 5` in the config cell for a quick trial run before committing to all 44
- Already-trained subject-model pairs are **skipped automatically** on re-run — training can be
  interrupted and resumed at any point without losing progress
- Weights are saved immediately after each subject-model pair completes (not at the end)

**Estimated runtime** (CPU): ~3–5 min per subject × 44 subjects × 3 models ≈ several hours.

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). putEMG — A Surface EMG Hand
  Gesture Recognition Dataset. *Sensors, 19*(16), 3548.
  https://doi.org/10.3390/s19163548
