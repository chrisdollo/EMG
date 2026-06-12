import sys
sys.path.insert(0, 'src')

# --- Feature extraction: 41 single-scalar features, channel-major ---
# force=False skips subjects already extracted; set force=True to re-extract all.
from feature_extraction import batch_extract_split_dirs

batch_extract_split_dirs(
    input_root='data/processed gestures',
    output_root='data/features',
    force=True,
)

# --- Within-subject DL training (uncomment to run) ---
# import runners
# runners.run_within(
#     type='deep_learning',
#     FORCE_RERUN=False,
#     input_folder='data/processed gestures',
#     output_weight_folder='weights/test',
#     result_folder='results/test',
#     max_epochs=20, patience=5, min_delta=0.0, batch_size=16, lr=1e-3,
# )

# --- Within-subject feature-based training (uncomment to run) ---
# SVM (RBF, C=50) + FeatureMLP, window-level + majority vote. input_folder is data/features.
# import runners
# runners.run_within(
#     type='feature_based',
#     FORCE_RERUN=False,
#     input_folder='data/features',
#     output_weight_folder='weights/test_feat',
#     result_folder='results/test_feat',
#     max_epochs=20, patience=5, min_delta=0.0, batch_size=16, lr=1e-3, svm_c=50.0,
# )

# --- Cross-subject LOSO DL training (uncomment to run) ---
# import runners
# runners.run_cross(
#     type='deep_learning',
#     FORCE_RERUN=False,
#     input_folder='data/processed gestures',
#     output_weight_folder='weights/cross_dl',
#     result_folder='results',
#     val_frac=0.10, max_epochs=20, patience=5, min_delta=0.0, batch_size=16, lr=1e-3,
# )

# --- Cross-subject LOSO feature-based training (uncomment to run) ---
# SVM mean-pooled; FeatureMLP window-level + majority vote (raise feat_mlp_batch_size on GPU).
# import runners
# runners.run_cross(
#     type='feature_based',
#     FORCE_RERUN=False,
#     input_folder='data/features',
#     output_weight_folder='weights/cross_feat',
#     result_folder='results',
#     val_frac=0.10, max_epochs=20, patience=5, min_delta=0.0, lr=1e-3, svm_c=50.0,
#     feat_mlp_batch_size=512,
# )