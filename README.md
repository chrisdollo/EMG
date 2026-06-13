# EMG Gesture Recognition — putEMG Dataset

Two parallel pipelines for hand gesture recognition from surface electromyography (sEMG) signals, built on the [putEMG dataset](https://biolab.put.poznan.pl/putemg-dataset/):

- **Deep learning** — CNN/TCN models on raw time-series (input: 1 × 24 × 1500)
- **Feature-based** — 41 single-scalar libemg features per channel × 24 channels = 984-dim per window, SVM / FeatureMLP

Each pipeline is evaluated under two protocols: **within-subject** (3-fold CV per subject) and **cross-subject** (LOSO — leave-one-subject-out).

---

## Dataset

The **putEMG** dataset ([Kaczmarek et al., 2019](https://www.mdpi.com/1424-8220/19/16/3548)) is a public sEMG benchmark from Poznan University of Technology.

| Property      | Value                                                   |
|---------------|---------------------------------------------------------|
| Subjects      | 44 able-bodied participants                             |
| Electrodes    | 24 sEMG channels (3 rings × 8 electrodes, 45° spacing) |
| Sampling rate | 5120 Hz                                                 |
| Gestures      | 7 active: G1, G2, G3, G6, G7, G8, G9                   |
| Repetitions   | ~40 per gesture per subject (2 sessions, ~280 total)    |
| File format   | CSV per recording session                               |

---

## Project Structure

```
putEMG prime/
├── baseline.ipynb              # all baseline experiments (configure PROTOCOL/APPROACH/MODEL)
├── clustering.ipynb            # Phase 3: feature-space 24→8 channel selection
├── efficient.ipynb             # Phase 5: 8-channel models
├── k.ipynb                     # Colab feature extraction (batch, faster than local)
│
├── src/                        # shared Python modules
│   ├── deep_learning_models.py # EEGNet, ShallowConvNet, EMG_TCN
│   ├── feature_based_models.py # FeatureMLP
│   ├── feature_extraction.py   # libemg 41-feature extractor (batch_extract_features)
│   ├── housekeeping.py         # loaders, fold splits, majority vote, results writer
│   ├── preprocessing.py        # bandpass / notch / resample / z-score
│   ├── runners.py              # run_within, run_cross
│   └── trainer.py              # train, evaluate
│
├── scripts/
│   ├── run_baseline.py         # entry point for all baseline experiments
│   ├── run_clustering.py       # Phase 3 runner
│   └── run_efficient.py        # Phase 5 runner
│
├── data_preprocessing/
│   ├── matlab/                 # CSV → per-subject combinedCell .mat (put_emg_driver.m)
│   └── driver.ipynb            # preprocessing entry point (Google Colab)
│
├── csv_data/                   # raw CSV recordings, one folder per subject
│   └── subject_NN/
│
├── data/
│   ├── processed gestures/     # {train,eval}/emg_gestures_SS_U.npz  (N × 24 × 1500)
│   ├── features/               # {train,eval}/features_SS_flat_rep.npz  (N × 25584)
│   └── unprocessed gestures/   # intermediate .mat outputs
│
├── weights/
│   ├── baseline/
│   │   ├── within_dl/          # per-subject .pt  (EEGNet, ShallowConvNet, EMG_TCN)
│   │   ├── within_feat/        # per-subject .pt  (SVM, FeatureMLP)
│   │   ├── cross_dl/           # per-LOSO-fold .pt  (EEGNet, ShallowConvNet, EMG_TCN)
│   │   └── cross_feat/         # per-LOSO-fold .pt  (SGD_SVM, FeatureMLP)
│   └── efficient/              # Phase 5 weights (8-channel models)
│
├── results/
│   ├── baseline/               # results_baseline_{within,cross}_{DL,feature}_{model}.txt
│   └── efficient/              # Phase 5 results
│
├── clustering & analysis/      # Phase 3/4 outputs
│   ├── feat_representative_channels.npy   # (8,) selected channel indices [after Phase 3]
│   └── feat_cluster_labels.npy            # (24,) cluster assignments
│
└── docs/
    ├── plan.txt                # full 7-phase research plan with status
    └── project_summary.txt     # results, pipeline details, pending work
```

---

## Pipeline

```
csv_data/subject_NN/
   │
   ▼  [MATLAB]  data_preprocessing/matlab/put_emg_driver.m
      CSV → detect gesture blocks → extract 24 channels → combinedCell .mat
   │
   ▼  [Python]  data_preprocessing/driver.ipynb + src/preprocessing.py
      Bandpass 20–700 Hz (Butterworth order 4, zero-phase)
      Notch filters: 30, 50, 60, 90, 150 Hz  (IIR notch Q=30, zero-phase)
      Resample → 1500 samples/rep
      Z-score per channel
      → data/processed gestures/{train,eval}/emg_gestures_SS_U.npz  (N × 24 × 1500)
   │
   ├── [Deep Learning]  src/runners.py → run_within / run_cross
   │   Input shape: (batch, 1, 24, 1500)
   │   Models: EEGNet, ShallowConvNet, EMG_TCN
   │
   └── [Feature Extraction]  src/feature_extraction.py
       Sliding window: size=250 samples (~49 ms), shift=50 → 26 windows/rep
       41 single-scalar libemg features × 24 channels = 984-dim/window  (channel-major)
       → data/features/{train,eval}/features_SS_flat_rep.npz  (N × 25584)
           │
           └── [Feature-Based]  src/runners.py → run_within / run_cross
               Within: SVM (RBF, C=50) + FeatureMLP — window-level, majority vote
               Cross:  SGD_SVM (hinge loss) + FeatureMLP — window-level, majority vote
```

---

## Results

### Within-Subject (3-fold CV, 44 subjects)

| Model              | Mean Acc   | Std      | Best fold          | Worst fold         |
|--------------------|------------|----------|--------------------|--------------------|
| **EMG_TCN**        | **96.44%** | ±2.91%   | Subj 42 (99.63%)   | Subj 06 (83.99%)   |
| **SVM** (RBF)      | **96.31%** | ±4.01%   | Subj 24 (99.64%)   | Subj 06 (76.62%)   |
| ShallowConvNet     | 92.19%     | ±4.46%   | Subj 49 (96.69%)   | Subj 06 (72.38%)   |
| EEGNet             | 85.80%     | ±8.28%   | Subj 48 (95.66%)   | Subj 06 (45.07%)   |
| FeatureMLP         | 71.82%     | ±11.62%  | Subj 29 (89.28%)   | Subj 51 (40.61%)   |
| putEMG paper       | ~90%       | —        | —                  | —                  |

### Cross-Subject LOSO (44/44 folds)

| Model              | Mean Acc   | Std      | Best fold          | Worst fold         |
|--------------------|------------|----------|--------------------|--------------------|
| **EMG_TCN**        | **84.80%** | ±11.56%  | Subj 53 (98.52%)   | Subj 17 (55.94%)   |
| ShallowConvNet     | 83.92%     | ±10.13%  | Subj 33 (98.18%)   | Subj 26 (57.04%)   |
| EEGNet             | 83.41%     | ±11.07%  | Subj 38 (98.48%)   | Subj 47 (52.16%)   |
| **SGD_SVM**        | **69.07%** | ±14.79%  | Subj 33 (92.73%)   | Subj 06 (35.00%)   |
| FeatureMLP         | 67.89%     | ±13.47%  | Subj 14 (94.10%)   | Subj 06 (39.67%)   |

Key observations:
- DL models cluster tightly (~84%); all exceed within-subject putEMG benchmark cross-subject
- ~10–12 pp std across all models: some subjects near-perfect (>98%), others near-chance (<55%)
- Subject 06 is hardest across all models; Subject 33 is easiest
- Feature-based cross-subject gap is large (SGD_SVM 69% vs SVM within 96%) — inter-subject variance not handled by a global linear boundary without per-subject normalization

---

## How to Run

### 1. MATLAB Preprocessing
```matlab
% Set data paths in put_emg_driver.m, then:
run('data_preprocessing/matlab/put_emg_driver.m')
```

### 2. Python Signal Preprocessing
Run `data_preprocessing/driver.ipynb` (Colab or local).

### 3. Feature Extraction
Run `k.ipynb` on Colab (faster) or locally via `test.py`.

### 4. Baselines (DL + Feature-based)
```bash
# edit scripts/run_baseline.py to select protocol/model, then:
python scripts/run_baseline.py
```
Or use `baseline.ipynb` for interactive Colab runs.

---

## Requirements

**MATLAB:** R2019b or later

**Python:**
```bash
pip install scipy numpy matplotlib scikit-learn torch libemg seaborn
```

---

## References

- Kaczmarek, P., Mankowski, T., Tomczynski, J. (2019). **putEMG — A Surface Electromyography Hand Gesture Recognition Dataset.** *Sensors, 19*(16), 3548. https://doi.org/10.3390/s19163548
- Lawhern, V. J., et al. (2018). **EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.** *Journal of Neural Engineering.* https://arxiv.org/abs/1611.08024
- Schirrmeister, R. T., et al. (2017). **Deep Learning with CNNs for EEG Motor Imagery Decoding.** *Human Brain Mapping.* https://doi.org/10.1002/hbm.23730
- putEMG dataset: https://biolab.put.poznan.pl/putemg-dataset/
- libemg: https://github.com/libemg/libemg
