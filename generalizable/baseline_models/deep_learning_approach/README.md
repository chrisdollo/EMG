# Deep Learning Approach

CNN/TCN-based gesture recognition on the putEMG dataset.  
Input: time-series EMG signals (1500 samples × 24 channels per gesture repetition).

## Folder Structure

```
deep_learning_approach/
├── model/
│   ├── model.ipynb          # Training & evaluation of all architectures
│   ├── models.py            # Model class definitions (EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN)
│   ├── emg_datahandler.py   # Data loading, splitting, DataLoaders, train/eval functions
│   └── weights/
│       ├── EMG_TCN_best.pt  # Best model checkpoint (93.57% test accuracy)
│       └── ...
└── README.md
```

## Prerequisites

Run `data_preprocessing/` first to produce the model-ready dataset:
`data/X/model_ready_5/model_ready_5_sub.mat`

## How to Run

Open and run `model/model.ipynb`.  
Adjust `MAX_EPOCHS`, `DROPOUT`, `LR` in the config cell.  
Best weights are saved automatically to `model/weights/`.

## Results (5 subjects)

| Model          | Test Accuracy |
|----------------|--------------|
| EMG_TCN        | **93.57%**   |
| EEGNet         | 86.43%       |
| ShallowConvNet | 81.07%       |
| DeepConvNet    | 59.64%       |
| CNN_LSTM       | 51.43%       |
| putEMG paper (SVM+RMS) | ~90% |

## Model Input/Output

All models accept `(batch, 1, 24, 1500)` and output `(batch, 7)` logits.

## Shared Modules

**`models.py`** — import model classes:
```python
from models import EEGNet, ShallowConvNet, DeepConvNet, CNN_LSTM, EMG_TCN
```

**`emg_datahandler.py`** — data loading and training utilities:
```python
from emg_datahandler import train, evaluate
from emg_datahandler import X_train, X_dev, X_test, train_loader, dev_loader, test_loader
```
