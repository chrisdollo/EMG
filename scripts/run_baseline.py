import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import runners

runners.run_within(
    type='deep_learning',
    FORCE_RERUN=True,
    input_folder='data/processed gestures',
    output_weight_folder='weights/baseline/within_dl',
    result_folder='results/baseline',
    experiment='baseline',
)

runners.run_within(
    type='feature_based',
    FORCE_RERUN=True,
    input_folder='data/features',
    output_weight_folder='weights/baseline/within_feat',
    result_folder='results/baseline',
    experiment='baseline',
)

runners.run_cross(
    type='deep_learning',
    FORCE_RERUN=True,
    input_folder='data/processed gestures',
    output_weight_folder='weights/baseline/cross_dl',
    result_folder='results/baseline',
    experiment='baseline',
)

runners.run_cross(
    type='feature_based',
    FORCE_RERUN=True,
    input_folder='data/features',
    output_weight_folder='weights/baseline/cross_feat',
    result_folder='results/baseline',
    experiment='baseline',
)
