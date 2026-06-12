"""
Run efficient (8-channel) models.

Usage:
  python scripts/run_efficient.py --protocol loso   --approach deep_learning --model EMG_TCN
  python scripts/run_efficient.py --protocol loso   --approach deep_learning --model EEGNet
  python scripts/run_efficient.py --protocol loso   --approach deep_learning --model ShallowConvNet
  python scripts/run_efficient.py --protocol loso   --approach feature_based
  python scripts/run_efficient.py --protocol within --approach deep_learning
  python scripts/run_efficient.py --protocol within --approach feature_based
  python scripts/run_efficient.py --all   (runs all 6 configs sequentially)
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

import numpy as np
import torch

from emg_loader import load_subjects_combined, load_feature_subjects_combined
from deep_learning_models import EMG_TCN, EEGNet, ShallowConvNet
from trainer import run_loso, run_within_subject, run_svm_loso, run_svm_within_subject

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_TRAIN    = os.path.join(ROOT, 'data', 'processed gestures', 'train')
DATA_EVAL     = os.path.join(ROOT, 'data', 'processed gestures', 'eval')
FEAT_TRAIN    = os.path.join(ROOT, 'data', 'features', 'train')
FEAT_EVAL     = os.path.join(ROOT, 'data', 'features', 'eval')
CHANNELS_FILE = os.path.join(ROOT, 'clustering & analysis', 'feat_representative_channels.npy')
N_CHANNELS    = 24

MAX_EPOCHS = 20
PATIENCE   = 5
DROPOUT    = 0.1
BATCH_SIZE = 16
LR         = 1e-3
SVM_C      = 50.0

ALL_CONFIGS = [
    ('loso',   'deep_learning', 'EMG_TCN'),
    ('loso',   'deep_learning', 'EEGNet'),
    ('loso',   'deep_learning', 'ShallowConvNet'),
    ('loso',   'feature_based', 'SVM_W'),
    ('within', 'deep_learning', None),
    ('within', 'feature_based', 'SVM_W'),
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def run(protocol, approach, model_type):
    device   = get_device()
    channels = np.load(CHANNELS_FILE)
    print(f'\n{"="*60}')
    print(f'  Protocol: {protocol}  |  Approach: {approach}  |  Model: {model_type}')
    print(f'  Device  : {device}  |  Channels: {channels.tolist()}')
    print(f'{"="*60}\n')

    # ── Load data ─────────────────────────────────────────────────────────────
    if approach == 'deep_learning':
        all_subjects = load_subjects_combined(DATA_TRAIN, DATA_EVAL)
        subjects = [(name, X[:, :, channels, :], y) for name, X, y in all_subjects]
        print(f'DL input shape: {subjects[0][1].shape}')
    else:
        all_subjects = load_feature_subjects_combined(FEAT_TRAIN, FEAT_EVAL, mode='flat_rep')
        feat_per_win = all_subjects[0][1].shape[2]
        F_PER_CH     = feat_per_win // N_CHANNELS
        per_win_cols = np.concatenate([
            np.arange(c * F_PER_CH, (c + 1) * F_PER_CH) for c in channels
        ])
        subjects = [(name, X[:, :, per_win_cols], y) for name, X, y in all_subjects]
        print(f'Feature shape: {subjects[0][1].shape}')

    # ── Derived paths ─────────────────────────────────────────────────────────
    results_dir = os.path.join(ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)

    if protocol == 'loso':
        weights_dir  = os.path.join(ROOT, 'weights', 'efficient', model_type)
        results_file = os.path.join(results_dir, f'results_efficient_cross-subject_{model_type}.txt')
        os.makedirs(weights_dir, exist_ok=True)
    else:
        weights_dir = os.path.join(ROOT, 'weights', 'efficient', 'within')
        os.makedirs(weights_dir, exist_ok=True)

    # ── Run ───────────────────────────────────────────────────────────────────
    DL_MODELS = {
        'EMG_TCN':        lambda: EMG_TCN(dropout_rate=DROPOUT),
        'EEGNet':         lambda: EEGNet(dropout_rate=DROPOUT),
        'ShallowConvNet': lambda: ShallowConvNet(dropout_rate=DROPOUT),
    }
    DL_CLS = {'EMG_TCN': EMG_TCN, 'EEGNet': EEGNet, 'ShallowConvNet': ShallowConvNet}

    if protocol == 'loso' and approach == 'deep_learning':
        run_loso(
            subjects, DL_CLS[model_type], model_type,
            weights_dir=weights_dir, log_path=results_file, device=device,
            dropout=DROPOUT, batch_size=BATCH_SIZE, lr=LR,
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )

    elif protocol == 'loso' and approach == 'feature_based':
        run_svm_loso(
            subjects, weights_dir=weights_dir, log_path=results_file,
            C=SVM_C, window_level=True,
        )

    elif protocol == 'within' and approach == 'deep_learning':
        run_within_subject(
            subjects, model_registry=DL_MODELS,
            weights_dir=weights_dir, device=device,
            results_dir=results_dir,
            log_filename='results_efficient_within_DL.txt',
            max_epochs=MAX_EPOCHS, patience=PATIENCE, lr=LR,
        )

    elif protocol == 'within' and approach == 'feature_based':
        run_svm_within_subject(
            subjects, weights_dir=weights_dir, results_dir=results_dir,
            C=SVM_C, window_level=True,
            log_filename='results_efficient_within_SVM.txt',
        )

    print(f'\nDone: {protocol} / {approach} / {model_type}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['loso', 'within'])
    parser.add_argument('--approach', choices=['deep_learning', 'feature_based'])
    parser.add_argument('--model',    default=None)
    parser.add_argument('--all',      action='store_true', help='Run all 6 configs sequentially')
    args = parser.parse_args()

    assert os.path.exists(CHANNELS_FILE), (
        f'Channel file not found: {CHANNELS_FILE}\n'
        'Run scripts/run_clustering.py first.'
    )

    if args.all:
        for protocol, approach, model in ALL_CONFIGS:
            run(protocol, approach, model)
    else:
        if not args.protocol or not args.approach:
            parser.error('--protocol and --approach are required (or use --all)')
        run(args.protocol, args.approach, args.model)


if __name__ == '__main__':
    main()
