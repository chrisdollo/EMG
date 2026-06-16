"""
Phase 5 — Reduced-Channel Models

Loads the 8 representative channels selected in Phase 3
(clustering & analysis/clustering_results.json) and retrains baseline models
restricted to those channels only.

Usage:
  python scripts/run_efficient.py
"""

import sys
import os
import json

os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import runners

CLUSTER_RESULTS = os.path.join('clustering & analysis', 'clustering_results.json')


def _load_selected_channels():
    with open(CLUSTER_RESULTS) as f:
        results = json.load(f)
    return results['selected_channels']


def run_within_dl(FORCE_RERUN=False):
    channels = _load_selected_channels()
    print(f'Selected channels (0-indexed): {channels}')
    runners.run_within(
        type='deep_learning',
        FORCE_RERUN=FORCE_RERUN,
        input_folder='data/processed gestures',
        output_weight_folder='weights/efficient/within_dl',
        result_folder='results/efficient',
        experiment='efficient',
        channels=channels,
    )


def run_within_feat(FORCE_RERUN=False):
    channels = _load_selected_channels()
    print(f'Selected channels (0-indexed): {channels}')
    runners.run_within(
        type='feature_based',
        FORCE_RERUN=FORCE_RERUN,
        input_folder='data/features',
        output_weight_folder='weights/efficient/within_feat',
        result_folder='results/efficient',
        experiment='efficient',
        channels=channels,
    )


def run_cross_dl(FORCE_RERUN=False):
    channels = _load_selected_channels()
    print(f'Selected channels (0-indexed): {channels}')
    runners.run_cross(
        type='deep_learning',
        FORCE_RERUN=FORCE_RERUN,
        input_folder='data/processed gestures',
        output_weight_folder='weights/efficient/cross_dl',
        result_folder='results/efficient',
        experiment='efficient',
        channels=channels,
    )


def run_cross_feat(FORCE_RERUN=False):
    # Mirrors baseline cross-subject feature protocol: SGD_SVM (scales to the full
    # window pool) + FeatureMLP. LinearSVC/SVM_W is O(n^2)-ish and intractable here too.
    channels = _load_selected_channels()
    print(f'Selected channels (0-indexed): {channels}')
    runners.run_cross(
        type='feature_based',
        FORCE_RERUN=FORCE_RERUN,
        input_folder='data/features',
        output_weight_folder='weights/efficient/cross_feat',
        result_folder='results/efficient',
        experiment='efficient',
        channels=channels,
        models=['SGD_SVM', 'FeatureMLP'],
    )


if __name__ == '__main__':
    run_within_dl()
    run_within_feat()
    run_cross_dl()
    run_cross_feat()
