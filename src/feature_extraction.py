import os
import re
import glob
import warnings
import numpy as np
import libemg

warnings.filterwarnings('ignore', category=UserWarning, module='pywt')

GESTURE_NAMES = ["G1", "G2", "G3", "G6", "G7", "G8", "G9"]
N_CHANNELS    = 24

_fe = libemg.feature_extractor.FeatureExtractor()

# Restrict to the 41 single-scalar-per-channel libemg features (exactly one number
# per channel). The excluded groups do not yield a clean 1-per-channel block:
#   multi-scalar within-channel : AR, CC, DFTR, WENG, WV, WWL, WENT
#   cross-channel               : RMSPHASOR, WLPHASOR
EXCLUDED_FEATURES = {'AR', 'CC', 'DFTR', 'WENG', 'WV', 'WWL', 'WENT',
                     'RMSPHASOR', 'WLPHASOR'}
FEATURE_LIST      = [f for f in _fe.get_feature_list() if f not in EXCLUDED_FEATURES]
N_FEATURES_PER_CH = len(FEATURE_LIST)   # 41


def _subject_id(filename):
    # emg_gestures_03_U.npz → '03'
    m = re.search(r'emg_gestures_(\d+)', os.path.splitext(filename)[0])
    return m.group(1) if m else os.path.splitext(filename)[0]


def _slide_windows(ch_first, window_size, window_shift):
    # ch_first: (24, 1500) → (n_wins, 24, window_size)
    starts = range(0, ch_first.shape[1] - window_size + 1, window_shift)
    return np.stack([ch_first[:, s:s + window_size] for s in starts])


def _to_channel_major(feats):
    # libemg array output is feature-major: [f0_ch0..f0_ch23, f1_ch0..f1_ch23, ...].
    # Each single-scalar feature contributes exactly N_CHANNELS columns, so reshape
    # to (TW, n_features, n_channels), swap axes, and flatten to channel-major:
    # [ch0_f0..ch0_f40, ch1_f0..ch1_f40, ...]. Channel c → cols [c*41 : (c+1)*41].
    tw = feats.shape[0]
    return (feats.reshape(tw, N_FEATURES_PER_CH, N_CHANNELS)
                 .transpose(0, 2, 1)
                 .reshape(tw, N_FEATURES_PER_CH * N_CHANNELS))


def batch_extract_features(input_dir, output_dir, window_size=250, window_shift=50, force=False):
    # batch_extract_features: for each subject .npz, slide windows, extract the 41
    # features channel-major, and save one flat (N, n_wins * 41 * 24) feature file.
    os.makedirs(output_dir, exist_ok=True)
    fe        = libemg.feature_extractor.FeatureExtractor()
    npz_files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))
    if not npz_files:
        print(f'No .npz files found in: {input_dir}')
        return

    n_files = len(npz_files)
    print(f'Found {n_files} file(s) — {N_FEATURES_PER_CH} features '
          f'× {N_CHANNELS} ch (channel-major)\n')

    for i, input_path in enumerate(npz_files, start=1):
        sid      = _subject_id(os.path.basename(input_path))
        out_path = os.path.join(output_dir, f'features_{sid}_flat_rep.npz')
        if not force and os.path.exists(out_path):
            print(f'  [{i}/{n_files}] [SKIP] subject {sid} — already extracted', flush=True)
            continue

        data = np.load(input_path)                       # X: (N, 24, 1500)
        # slide each rep into windows, dropping any rep shorter than one window
        reps = [(_slide_windows(x.astype(np.float64), window_size, window_shift), int(g))
                for x, g in zip(data['X'], data['y']) if x.shape[1] >= window_size]
        wins_list, y_list = zip(*reps)
        n_wins   = wins_list[0].shape[0]
        all_wins = np.concatenate(wins_list, axis=0)     # (n_reps * n_wins, 24, window_size)
        print(f'  [{i}/{n_files}] subject {sid}: extracting {len(all_wins)} windows '
              f'({len(y_list)} reps × {n_wins} wins)...', flush=True)

        # extract 41 features/window, reorder to channel-major, flatten per rep
        feats = _to_channel_major(fe.extract_features(FEATURE_LIST, all_wins, array=True))
        X = feats.reshape(len(y_list), -1).astype(np.float32)   # (n_reps, n_wins * 41 * 24)
        y = np.array(y_list, dtype=np.int64)

        np.savez(out_path, X=X, y=y,
                 feature_list      = np.array(FEATURE_LIST),
                 n_windows_per_rep = n_wins,
                 n_features_per_ch = N_FEATURES_PER_CH,
                 n_channels        = N_CHANNELS,
                 layout            = 'channel_major',
                 window_size       = window_size,
                 window_shift      = window_shift)
        print(f'  [{i}/{n_files}] subject {sid}: saved → X={X.shape}', flush=True)

    print(f'\nDone → {output_dir}')


def batch_extract_split_dirs(input_root, output_root, splits=('train', 'eval'),
                              window_size=250, window_shift=50, force=False):
    # batch_extract_split_dirs: run batch_extract_features over each train/eval split
    for split in splits:
        input_dir  = os.path.join(input_root,  split)
        output_dir = os.path.join(output_root, split)
        if not os.path.isdir(input_dir):
            print(f"Skipping '{split}/' — not found in {input_root}")
            continue
        print(f"\n{'='*60}\n  Extracting features: {split}\n{'='*60}")
        batch_extract_features(input_dir, output_dir, window_size, window_shift, force)
