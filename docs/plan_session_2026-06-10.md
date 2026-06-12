# putEMG Prime — Session Plan (2026-06-10)

This document is a dated copy of the overall project plan, extended with the work
planned and executed in this session. Refer to `docs/plan.txt` for the original
milestone-level plan.

---

## Status snapshot at session start

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | Within-subject DL (EMG_TCN, ShallowConvNet, EEGNet) | COMPLETE |
| 1b | Within-subject feature-based (SVM_W, SVM, FeatureMLP) | COMPLETE |
| 2a | Cross-subject LOSO DL | COMPLETE |
| 2b | Cross-subject LOSO SVM (mean-pooled) | COMPLETE |
| 2c | Cross-subject LOSO SVM_W (window-level) | **NOT RUN** — added this session |
| 3   | Feature-based MI channel clustering (24→8) | Not yet run |
| 4   | Anatomical validation | Skipped (out of scope this session) |
| 5   | Reduced-channel models | Not yet run |
| 6   | Per-subject per-class accuracy analysis | Not started |
| 7   | Real-time data collection | Future |

---

## Session goals

1. Run feature-based MI channel clustering to select the best 8 of 24 channels.
2. Add SVM_W to the cross-subject LOSO baseline (was missing; needed for fair
   comparison with the 8-channel efficient run).
3. Expand `efficient.ipynb` to support both within-subject (3-fold CV) and
   cross-subject (LOSO) protocols, for both deep learning and feature-based models.
4. Run all four efficient-model configurations:
   - Within-subject × deep learning (EMG_TCN, EEGNet, ShallowConvNet)
   - Within-subject × feature-based (SVM_W)
   - Cross-subject LOSO × deep learning (EMG_TCN, EEGNet, ShallowConvNet)
   - Cross-subject LOSO × feature-based (SVM_W)
5. Write a summary analysis comparing 8-channel vs 24-channel accuracy across all
   conditions. Save to `docs/`.

---

## Implementation plan — step by step

### Step 1 — Run `clustering.ipynb`

**Goal:** Produce `clustering & analysis/feat_representative_channels.npy` (8 channel
indices) and `feat_channel_mi_scores.npy`.

**Method:** Mutual Information (sklearn `mutual_info_classif`) computed per-channel
using per-channel libemg feature extraction (1-channel calls, 87 scalars/channel).
All 44 subjects pooled before MI scoring, so channel selection is global (same 8
channels used by every subject).

**Outputs:**
- `clustering & analysis/feat_representative_channels.npy` — (8,) 0-indexed channel IDs
- `clustering & analysis/feat_channel_mi_scores.npy` — (24,) MI scores
- `clustering & analysis/mi_channel_scores.png`
- `clustering & analysis/ring_diagram.png`
- `clustering & analysis/channel_correlation_heatmap.png`

**Implementation status:** Notebook fully written; needs execution.

---

### Step 2 — Run baseline SVM_W LOSO

**Goal:** Produce `results/results_baseline_SVM_W.txt` so the efficient comparison
cell has a 24-channel SVM_W LOSO baseline to compare against.

**Why it was missing:** The baseline.ipynb already supports `MODEL_TYPE='SVM_W'` with
`window_level=(MODEL_TYPE == 'SVM_W')` in the run cell. It was simply never executed.

**Method:** `run_svm_loso(..., window_level=True)` — train SVM on all 26 windows per
rep as separate samples across 43 training subjects; majority-vote 26 per-window
predictions per test rep. RBF kernel, C=50, StandardScaler fit on training pool only.

**Outputs:**
- `results/results_baseline_SVM_W.txt`
- `weights/baseline/SVM_W/SVM_W_XX.pkl` (44 files)

**Implementation status:** No code changes needed; execute `baseline.ipynb` with
`PROTOCOL='loso', APPROACH='feature_based', MODEL_TYPE='SVM_W'`.

---

### Step 3 — Expand `efficient.ipynb`

**Goal:** Add `PROTOCOL = 'loso' | 'within'` config and within-subject training
sections, so one notebook covers all four experiment variants.

**Changes to `efficient.ipynb`:**

**Cell 1 (Config)** — add `PROTOCOL` variable:
```python
PROTOCOL   = 'loso'           # 'loso'  |  'within'
APPROACH   = 'deep_learning'  # 'deep_learning'  |  'feature_based'
MODEL_TYPE = 'EMG_TCN'        # deep_learning: EMG_TCN | EEGNet | ShallowConvNet
                               # feature_based: SVM_W
```
Derived output paths:
- LOSO DL:      `weights/efficient/{MODEL_TYPE}/`       → `results/results_efficient_{MODEL_TYPE}.txt`
- LOSO SVM_W:   `weights/efficient/SVM_W/`              → `results/results_efficient_SVM_W.txt`
- Within DL:    `weights/efficient/within/{MODEL_TYPE}/` → `results/results_efficient_within_DL.txt`
- Within SVM_W: `weights/efficient/within/SVM_W/`        → logged via `_write_all_svm_results`

**Cell 3 (Data loading)** — unchanged; the 8-channel slicing already works for all
protocols (DL: `X[:, :, channels, :]`; feature-based: `col_indices` slicing on flat
feature vector).

**Cell 4 (Run)** — replace the current LOSO-only run cell with a four-branch
conditional that matches `baseline.ipynb`'s structure:
```
loso  × deep_learning   → run_loso(...)
loso  × feature_based   → run_svm_loso(..., window_level=True)
within × deep_learning  → run_within_subject(...)
within × feature_based  → run_svm_within_subject(..., window_level=True)
```

**Cell 5 (Comparison)** — update `_parse_mean` to handle both LOSO result formats
and within-subject result formats; print 8-ch vs 24-ch delta for all conditions.

**Implementation status:** Needs code edits before execution.

---

### Step 4 — Execute all efficient-model configurations

Run `efficient.ipynb` once per config combination below. Completed folds are skipped
automatically (checkpointing), so runs can be interrupted and resumed.

| # | PROTOCOL | APPROACH       | MODEL_TYPE     | Est. time |
|---|----------|----------------|----------------|-----------|
| A | within   | deep_learning  | EMG_TCN        | ~2 h      |
| B | within   | deep_learning  | EEGNet         | ~2 h      |
| C | within   | deep_learning  | ShallowConvNet | ~1.5 h    |
| D | within   | feature_based  | SVM_W          | ~0.5 h    |
| E | loso     | deep_learning  | EMG_TCN        | ~4 h      |
| F | loso     | deep_learning  | EEGNet         | ~4 h      |
| G | loso     | deep_learning  | ShallowConvNet | ~3 h      |
| H | loso     | feature_based  | SVM_W          | ~2 h      |

Step 2 (baseline SVM_W LOSO) must complete before Step 4H so the comparison cell
has a 24-channel reference.

---

### Step 5 — Write analysis

**Output:** `docs/efficient_analysis.txt`

**Contents:**
- Selected channels: IDs, ring distribution, MI scores
- 8-channel vs 24-channel accuracy table for all 8 conditions above
- Accuracy delta (pp) and % retention per model
- Key observations: which models degrade most / least with channel reduction

---

## Decisions made this session

| Decision | Rationale |
|----------|-----------|
| Feature-based MI only (no raw-signal clustering) | Raw-signal clustering was found less informative in prior experiments |
| Same 8 channels for all subjects | MI scored on pooled 44-subject data; channel selection is a global design choice, not per-subject |
| Add SVM_W to baseline LOSO | Needed for fair 8-channel comparison; SVM_W is the best feature-based within-subject model and deserves a LOSO counterpart |
| All 3 DL models for efficient within-subject | Keeps comparison complete and consistent with Phase 1 baseline |
| Skip anatomical validation (Phase 4) | Out of scope for this session; can be added later |
