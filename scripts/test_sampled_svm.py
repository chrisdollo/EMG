import os, sys, time
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from housekeeping import (
    _list_subject_files,
    _load_combined_feature_subject,
    _stratified_sample_windows,
    _reps_to_windows,
    _majority_vote,
)

train_dir = 'data/features/train'
eval_dir  = 'data/features/eval'

# ── 1. Discover subjects ──────────────────────────────────────────────────────
print('[1] Discovering subjects...')
sids = sorted(set(_list_subject_files(train_dir)) | set(_list_subject_files(eval_dir)))
print(f'    Found {len(sids)} subjects: {sids[:5]} ...')

# ── 2. Load all subjects ──────────────────────────────────────────────────────
print('[2] Loading all subjects into cache...')
t = time.time()
cache = {sid: _load_combined_feature_subject(train_dir, eval_dir, sid) for sid in sids}
print(f'    Done in {time.time()-t:.1f}s')
sid = sids[0]
X0, y0 = cache[sid]
print(f'    Subject {sid}: X={X0.shape}  y={y0.shape}  classes={np.unique(y0)}')

# ── 3. Stratified sampling ────────────────────────────────────────────────────
print('[3] Running _stratified_sample_windows (n_samples=5000)...')
t = time.time()
Xtr, ytr = _stratified_sample_windows(cache, sid, 300000, seed=10)
print(f'    Done in {time.time()-t:.1f}s')
print(f'    Xtr={Xtr.shape}  dtype={Xtr.dtype}  classes={np.unique(ytr)}  any_nan={np.isnan(Xtr).any()}')

# ── 4. Test window expansion ──────────────────────────────────────────────────
print('[4] Expanding test reps to windows...')
t = time.time()
Xte_w, _ = _reps_to_windows(X0, y0)
print(f'    Done in {time.time()-t:.1f}s')
print(f'    Xte_w={Xte_w.shape}')

# ── 5. Fit StandardScaler + SGDClassifier ────────────────────────────────────
print('[5] Fitting StandardScaler...')
t = time.time()
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(Xtr)
print(f'    Done in {time.time()-t:.1f}s')

print('[6] Fitting SGDClassifier(loss=hinge)...')
t = time.time()
clf = SGDClassifier(loss='hinge', max_iter=200, random_state=42, n_jobs=-1)
clf.fit(Xtr_scaled, ytr)
print(f'    Done in {time.time()-t:.1f}s  n_iter_={clf.n_iter_}')

# ── 7. Predict + majority vote ────────────────────────────────────────────────
print('[7] Predicting + majority vote...')
t = time.time()
Xte_scaled = scaler.transform(Xte_w)
preds_w    = clf.predict(Xte_scaled)
rep_pred   = _majority_vote(preds_w, X0.shape[1])
acc        = float((rep_pred == y0).mean())
print(f'    Done in {time.time()-t:.1f}s')
print(f'    Test acc subject {sid}: {acc*100:.2f}%')

print('\n[OK] Full pipeline passed.')
