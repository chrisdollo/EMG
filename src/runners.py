import os
import datetime
import numpy as np
import torch

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from deep_learning_models import EEGNet, EMG_TCN, ShallowConvNet
from feature_based_models import FeatureMLP
from trainer import train, evaluate
from housekeeping import (
    _list_subject_files,
    _load_combined_subject,
    _load_combined_feature_subject,
    _reps_to_windows,
    _majority_vote,
    _mlp_window_preds,
    _pool_except,
    _get_device,
    _stratified_folds,
    _stratified_val_split,
    _loader,
    _write_results,
)

DL_MODELS = {
    'EEGNet':         EEGNet,
    'ShallowConvNet': ShallowConvNet,
    'EMG_TCN':        EMG_TCN,
}

WITHIN_FEATURE_MODELS = ['SVM', 'FeatureMLP']
CROSS_FEATURE_MODELS  = ['SVM_W', 'FeatureMLP']


def _cpu_state(state_dict):
    return {k: v.cpu() for k, v in state_dict.items()}


def run_within(type, FORCE_RERUN, input_folder, output_weight_folder, result_folder, *,
               n_folds=3, val_frac=0.15, batch_size=16, lr=1e-3,
               max_epochs=20, patience=5, min_delta=0.0, dropout=0.1,
               svm_c=50.0, channels=None, feature_idx=None, experiment='baseline',
               seed=42, device=None):
    # run_within: within-subject training with per-model checkpoint subfolders.
    # Folder layout: output_weight_folder/{model_name}/{sid}.pt
    # Each checkpoint stores fold-level weights (fold_states for DL/MLP, fold_clfs for SVM).
    os.makedirs(result_folder, exist_ok=True)
    device = device or _get_device()

    train_dir = os.path.join(input_folder, 'train')
    eval_dir  = os.path.join(input_folder, 'eval')

    if type == 'deep_learning':
        subject_ids = sorted(set(_list_subject_files(train_dir)) | set(_list_subject_files(eval_dir)))
        print(f'Found {len(subject_ids)} subjects  |  device: {device}')

        model_folders = {}
        for model_name in DL_MODELS:
            folder = os.path.join(output_weight_folder, model_name)
            os.makedirs(folder, exist_ok=True)
            model_folders[model_name] = folder

        for sid in subject_ids:
            models_to_run = {
                name: cls for name, cls in DL_MODELS.items()
                if FORCE_RERUN or not os.path.exists(os.path.join(model_folders[name], f'{sid}.pt'))
            }
            if not models_to_run:
                print(f'[SKIP] subject {sid} — all models complete')
                continue

            X, y = _load_combined_subject(train_dir, eval_dir, sid, channels=channels)
            print(f'\n=== subject {sid}: X={X.shape}  y={y.shape} ===')
            folds = _stratified_folds(y, n_folds, seed)

            for model_name, model_cls in models_to_run.items():
                fold_accs, fold_vals, fold_epochs, fold_states = [], [], [], []
                for k in range(n_folds):
                    test_idx = folds[k]
                    pool_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                    tr_rel, val_rel = _stratified_val_split(y[pool_idx], val_frac, seed)
                    train_loader = _loader(X[pool_idx[tr_rel]], y[pool_idx[tr_rel]], batch_size, True)
                    val_loader   = _loader(X[pool_idx[val_rel]], y[pool_idx[val_rel]], batch_size, False)
                    test_loader  = _loader(X[test_idx], y[test_idx], batch_size, False)

                    print(f'[{model_name}] subject {sid} fold {k+1}/{n_folds} '
                          f'(train={len(tr_rel)} val={len(val_rel)} test={len(test_idx)})')
                    model = model_cls(num_channels=X.shape[2], dropout_rate=dropout).to(device)
                    best_state, best_val, best_epoch = train(
                        model, train_loader, val_loader, device,
                        max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr,
                    )
                    model.load_state_dict(best_state)
                    acc = evaluate(model, test_loader, device)
                    print(f'    fold test acc = {acc*100:.2f}%  (val {best_val*100:.2f}% @ ep {best_epoch})')
                    fold_accs.append(acc)
                    fold_vals.append(best_val)
                    fold_epochs.append(best_epoch)
                    fold_states.append(_cpu_state(best_state))

                print(f'  → {model_name} mean = {np.mean(fold_accs)*100:.2f}%')
                ckpt = os.path.join(model_folders[model_name], f'{sid}.pt')
                torch.save({
                    'subject':     sid,
                    'model':       model_name,
                    'fold_accs':   fold_accs,
                    'fold_vals':   fold_vals,
                    'fold_epochs': fold_epochs,
                    'fold_states': fold_states,      # list[state_dict] — one per CV fold
                    'mean_acc':    float(np.mean(fold_accs)),
                    'mean_val':    float(np.mean(fold_vals)),
                    'mean_epoch':  float(np.mean(fold_epochs)),
                    'n_folds':     n_folds,
                    'date':        datetime.date.today().isoformat(),
                }, ckpt)
                _write_results(model_folders[model_name], result_folder, type,
                               len(subject_ids), model_name,
                               experiment=experiment, protocol='within',
                               channels=channels or 24)

    elif type == 'feature_based':
        subject_ids = sorted(set(_list_subject_files(train_dir)) | set(_list_subject_files(eval_dir)))
        print(f'Found {len(subject_ids)} subjects  |  device: {device}')

        svm_folder = os.path.join(output_weight_folder, 'SVM')
        mlp_folder = os.path.join(output_weight_folder, 'FeatureMLP')
        os.makedirs(svm_folder, exist_ok=True)
        os.makedirs(mlp_folder, exist_ok=True)

        for sid in subject_ids:
            svm_ckpt = os.path.join(svm_folder, f'{sid}.pt')
            mlp_ckpt = os.path.join(mlp_folder, f'{sid}.pt')
            run_svm = FORCE_RERUN or not os.path.exists(svm_ckpt)
            run_mlp = FORCE_RERUN or not os.path.exists(mlp_ckpt)
            if not run_svm and not run_mlp:
                print(f'[SKIP] subject {sid} — all models complete')
                continue

            X, y = _load_combined_feature_subject(train_dir, eval_dir, sid,
                                                  channels=channels, feature_idx=feature_idx)
            n_wins = X.shape[1]
            print(f'\n=== subject {sid}: X={X.shape}  y={y.shape} ===')
            folds = _stratified_folds(y, n_folds, seed)

            if run_svm:
                fold_accs, fold_clfs = [], []
                for k in range(n_folds):
                    test_idx = folds[k]
                    pool_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                    Xtr_w, ytr_w = _reps_to_windows(X[pool_idx], y[pool_idx])
                    Xte_w, _     = _reps_to_windows(X[test_idx], y[test_idx])
                    print(f'[SVM] subject {sid} fold {k+1}/{n_folds} '
                          f'(train={len(Xtr_w)} win  test={len(test_idx)} reps)')
                    clf = make_pipeline(StandardScaler(), SVC(C=svm_c, kernel='rbf'))
                    clf.fit(Xtr_w, ytr_w)
                    rep_pred = _majority_vote(clf.predict(Xte_w), n_wins)
                    acc = float((rep_pred == y[test_idx]).mean())
                    print(f'    fold test acc = {acc*100:.2f}%')
                    fold_accs.append(acc)
                    fold_clfs.append(clf)
                print(f'  → SVM mean = {np.mean(fold_accs)*100:.2f}%')
                torch.save({
                    'subject':    sid,
                    'model':      'SVM',
                    'fold_accs':  fold_accs,
                    'fold_clfs':  fold_clfs,        # list[sklearn Pipeline] — one per CV fold
                    'mean_acc':   float(np.mean(fold_accs)),
                    'mean_val':   None,
                    'mean_epoch': None,
                    'n_folds':    n_folds,
                    'date':       datetime.date.today().isoformat(),
                }, svm_ckpt)
                _write_results(svm_folder, result_folder, type, len(subject_ids), 'SVM',
                               experiment=experiment, protocol='within', channels=channels or 24)

            if run_mlp:
                fold_accs, fold_vals, fold_epochs, fold_states = [], [], [], []
                for k in range(n_folds):
                    test_idx = folds[k]
                    pool_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
                    tr_rel, val_rel = _stratified_val_split(y[pool_idx], val_frac, seed)
                    train_idx = pool_idx[tr_rel]
                    val_idx   = pool_idx[val_rel]
                    Xtr_w, ytr_w   = _reps_to_windows(X[train_idx], y[train_idx])
                    Xval_w, yval_w = _reps_to_windows(X[val_idx],   y[val_idx])
                    Xte_w, _       = _reps_to_windows(X[test_idx],  y[test_idx])
                    train_loader = _loader(Xtr_w,  ytr_w,  batch_size, True)
                    val_loader   = _loader(Xval_w, yval_w, batch_size, False)

                    print(f'[FeatureMLP] subject {sid} fold {k+1}/{n_folds} '
                          f'(train={len(Xtr_w)} win  val={len(Xval_w)} win  test={len(test_idx)} reps)')
                    model = FeatureMLP(input_size=X.shape[2], dropout_rate=dropout).to(device)
                    best_state, best_val, best_epoch = train(
                        model, train_loader, val_loader, device,
                        max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr,
                    )
                    model.load_state_dict(best_state)
                    rep_pred = _majority_vote(_mlp_window_preds(model, Xte_w, device), n_wins)
                    acc = float((rep_pred == y[test_idx]).mean())
                    print(f'    fold test acc = {acc*100:.2f}%  (val {best_val*100:.2f}% @ ep {best_epoch})')
                    fold_accs.append(acc)
                    fold_vals.append(best_val)
                    fold_epochs.append(best_epoch)
                    fold_states.append(_cpu_state(best_state))
                print(f'  → FeatureMLP mean = {np.mean(fold_accs)*100:.2f}%')
                torch.save({
                    'subject':     sid,
                    'model':       'FeatureMLP',
                    'fold_accs':   fold_accs,
                    'fold_vals':   fold_vals,
                    'fold_epochs': fold_epochs,
                    'fold_states': fold_states,      # list[state_dict] — one per CV fold
                    'mean_acc':    float(np.mean(fold_accs)),
                    'mean_val':    float(np.mean(fold_vals)),
                    'mean_epoch':  float(np.mean(fold_epochs)),
                    'n_folds':     n_folds,
                    'date':        datetime.date.today().isoformat(),
                }, mlp_ckpt)
                _write_results(mlp_folder, result_folder, type, len(subject_ids), 'FeatureMLP',
                               experiment=experiment, protocol='within', channels=channels or 24)

    else:
        raise ValueError(f"type must be 'deep_learning' or 'feature_based', got {type!r}")


def run_cross(type, FORCE_RERUN, input_folder, output_weight_folder, result_folder, *,
              val_frac=0.10, batch_size=16, lr=1e-3, max_epochs=20, patience=5,
              min_delta=0.0, dropout=0.1, svm_c=50.0, feat_mlp_batch_size=512,
              channels=None, feature_idx=None, experiment='baseline',
              seed=42, device=None):
    # run_cross: LOSO cross-subject training with per-model checkpoint subfolders.
    # Folder layout: output_weight_folder/{model_name}/{sid}.pt  (one per LOSO fold)
    #                output_weight_folder/{model_name}/final.pt  (trained on all subjects)
    os.makedirs(result_folder, exist_ok=True)
    device = device or _get_device()

    train_dir = os.path.join(input_folder, 'train')
    eval_dir  = os.path.join(input_folder, 'eval')
    subject_ids = sorted(set(_list_subject_files(train_dir)) | set(_list_subject_files(eval_dir)))

    if type == 'deep_learning':
        print(f'Loading {len(subject_ids)} subjects (DL)  |  device: {device}')
        cache = {sid: _load_combined_subject(train_dir, eval_dir, sid, channels=channels)
                 for sid in subject_ids}

        for model_name, model_cls in DL_MODELS.items():
            model_folder = os.path.join(output_weight_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)

            for sid in subject_ids:
                ckpt = os.path.join(model_folder, f'{sid}.pt')
                if not FORCE_RERUN and os.path.exists(ckpt):
                    print(f'[SKIP] {model_name} test subject {sid}')
                    continue

                test_X, test_y = cache[sid]
                pool_X, pool_y = _pool_except(cache, sid)
                tr_rel, val_rel = _stratified_val_split(pool_y, val_frac, seed)
                train_loader = _loader(pool_X[tr_rel], pool_y[tr_rel], batch_size, True)
                val_loader   = _loader(pool_X[val_rel], pool_y[val_rel], batch_size, False)
                test_loader  = _loader(test_X, test_y, batch_size, False)

                print(f'\n[{model_name}] LOSO test subject {sid}  '
                      f'(train={len(tr_rel)} val={len(val_rel)} test={len(test_X)})')
                model = model_cls(num_channels=pool_X.shape[2], dropout_rate=dropout).to(device)
                best_state, best_val, best_epoch = train(
                    model, train_loader, val_loader, device,
                    max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr,
                )
                model.load_state_dict(best_state)
                acc = evaluate(model, test_loader, device)
                print(f'  → test acc = {acc*100:.2f}%  (val {best_val*100:.2f}% @ ep {best_epoch})')

                torch.save({
                    'subject':    sid,
                    'model':      model_name,
                    'mean_acc':   float(acc),
                    'mean_val':   float(best_val),
                    'mean_epoch': float(best_epoch),
                    'state_dict': _cpu_state(best_state),
                    'protocol':   'cross',
                    'date':       datetime.date.today().isoformat(),
                }, ckpt)
                _write_results(model_folder, result_folder, type, len(subject_ids), model_name,
                               experiment=experiment, protocol='cross', channels=channels or 24)

            # Final model: trained on all subjects — for deployment / fine-tuning
            final_ckpt = os.path.join(model_folder, 'final.pt')
            if FORCE_RERUN or not os.path.exists(final_ckpt):
                all_X = np.concatenate([X for X, _ in cache.values()], axis=0)
                all_y = np.concatenate([y for _, y in cache.values()], axis=0)
                tr_rel, val_rel = _stratified_val_split(all_y, val_frac, seed)
                train_loader = _loader(all_X[tr_rel], all_y[tr_rel], batch_size, True)
                val_loader   = _loader(all_X[val_rel], all_y[val_rel], batch_size, False)
                print(f'\n[{model_name}] Training final model on all {len(subject_ids)} subjects...')
                model = model_cls(num_channels=all_X.shape[2], dropout_rate=dropout).to(device)
                best_state, best_val, best_epoch = train(
                    model, train_loader, val_loader, device,
                    max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr,
                )
                torch.save({
                    'subject':    'final',
                    'model':      model_name,
                    'mean_val':   float(best_val),
                    'mean_epoch': float(best_epoch),
                    'state_dict': _cpu_state(best_state),
                    'n_subjects': len(subject_ids),
                    'protocol':   'cross',
                    'date':       datetime.date.today().isoformat(),
                }, final_ckpt)
                print(f'  → final model val {best_val*100:.2f}% @ epoch {best_epoch}')

    elif type == 'feature_based':
        print(f'Loading {len(subject_ids)} subjects (features)  |  device: {device}')
        cache = {sid: _load_combined_feature_subject(train_dir, eval_dir, sid,
                                                     channels=channels, feature_idx=feature_idx)
                 for sid in subject_ids}
        n_wins = next(iter(cache.values()))[0].shape[1]

        # --- SVM_W (LinearSVC, window-level + majority vote) ---
        svm_folder = os.path.join(output_weight_folder, 'SVM_W')
        os.makedirs(svm_folder, exist_ok=True)

        for sid in subject_ids:
            ckpt = os.path.join(svm_folder, f'{sid}.pt')
            if not FORCE_RERUN and os.path.exists(ckpt):
                print(f'[SKIP] SVM_W test subject {sid}')
                continue
            test_X, test_y = cache[sid]
            pool_X, pool_y = _pool_except(cache, sid)
            Xtr_w, ytr_w = _reps_to_windows(pool_X, pool_y)
            Xte_w, _     = _reps_to_windows(test_X, test_y)
            print(f'\n[SVM_W] test subject {sid}  (train={len(Xtr_w)} win  test={len(test_X)} reps)')
            clf = make_pipeline(StandardScaler(), LinearSVC(C=svm_c, max_iter=2000))
            clf.fit(Xtr_w, ytr_w)
            rep_pred = _majority_vote(clf.predict(Xte_w), n_wins)
            acc = float((rep_pred == test_y).mean())
            print(f'  → SVM_W test acc = {acc*100:.2f}%')
            torch.save({
                'subject':    sid,
                'model':      'SVM_W',
                'mean_acc':   acc,
                'mean_val':   None,
                'mean_epoch': None,
                'clf':        clf,                  # sklearn Pipeline (StandardScaler + LinearSVC)
                'protocol':   'cross',
                'date':       datetime.date.today().isoformat(),
            }, ckpt)
            _write_results(svm_folder, result_folder, type, len(subject_ids), 'SVM_W',
                           experiment=experiment, protocol='cross', channels=channels or 24)

        svm_final = os.path.join(svm_folder, 'final.pt')
        if FORCE_RERUN or not os.path.exists(svm_final):
            all_X = np.concatenate([X for X, _ in cache.values()], axis=0)
            all_y = np.concatenate([y for _, y in cache.values()], axis=0)
            Xtr_w, ytr_w = _reps_to_windows(all_X, all_y)
            print(f'\n[SVM_W] Training final model on all {len(subject_ids)} subjects '
                  f'({len(Xtr_w)} windows)...')
            clf = make_pipeline(StandardScaler(), LinearSVC(C=svm_c, max_iter=2000))
            clf.fit(Xtr_w, ytr_w)
            torch.save({
                'subject':    'final',
                'model':      'SVM_W',
                'clf':        clf,
                'n_subjects': len(subject_ids),
                'protocol':   'cross',
                'date':       datetime.date.today().isoformat(),
            }, svm_final)
            print('  → SVM_W final model saved')

        # --- FeatureMLP (window-level + majority vote) ---
        mlp_folder = os.path.join(output_weight_folder, 'FeatureMLP')
        os.makedirs(mlp_folder, exist_ok=True)

        for sid in subject_ids:
            ckpt = os.path.join(mlp_folder, f'{sid}.pt')
            if not FORCE_RERUN and os.path.exists(ckpt):
                print(f'[SKIP] FeatureMLP test subject {sid}')
                continue
            test_X, test_y = cache[sid]
            pool_X, pool_y = _pool_except(cache, sid)
            tr_rel, val_rel = _stratified_val_split(pool_y, val_frac, seed)
            Xtr_w, ytr_w   = _reps_to_windows(pool_X[tr_rel],  pool_y[tr_rel])
            Xval_w, yval_w = _reps_to_windows(pool_X[val_rel], pool_y[val_rel])
            Xte_w, _       = _reps_to_windows(test_X, test_y)
            train_loader = _loader(Xtr_w,  ytr_w,  feat_mlp_batch_size, True)
            val_loader   = _loader(Xval_w, yval_w, feat_mlp_batch_size, False)
            print(f'\n[FeatureMLP] test subject {sid}  '
                  f'(train={len(Xtr_w)} win  val={len(Xval_w)} win  test={len(test_X)} reps)')
            model = FeatureMLP(input_size=test_X.shape[2], dropout_rate=dropout).to(device)
            best_state, best_val, best_epoch = train(
                model, train_loader, val_loader, device,
                max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr,
            )
            model.load_state_dict(best_state)
            rep_pred = _majority_vote(_mlp_window_preds(model, Xte_w, device), n_wins)
            acc = float((rep_pred == test_y).mean())
            print(f'  → FeatureMLP test acc = {acc*100:.2f}%  (val {best_val*100:.2f}% @ ep {best_epoch})')
            torch.save({
                'subject':    sid,
                'model':      'FeatureMLP',
                'mean_acc':   float(acc),
                'mean_val':   float(best_val),
                'mean_epoch': float(best_epoch),
                'state_dict': _cpu_state(best_state),
                'protocol':   'cross',
                'date':       datetime.date.today().isoformat(),
            }, ckpt)
            _write_results(mlp_folder, result_folder, type, len(subject_ids), 'FeatureMLP',
                           experiment=experiment, protocol='cross', channels=channels or 24)

        mlp_final = os.path.join(mlp_folder, 'final.pt')
        if FORCE_RERUN or not os.path.exists(mlp_final):
            all_X = np.concatenate([X for X, _ in cache.values()], axis=0)
            all_y = np.concatenate([y for _, y in cache.values()], axis=0)
            tr_rel, val_rel = _stratified_val_split(all_y, val_frac, seed)
            Xtr_w, ytr_w   = _reps_to_windows(all_X[tr_rel], all_y[tr_rel])
            Xval_w, yval_w = _reps_to_windows(all_X[val_rel], all_y[val_rel])
            train_loader = _loader(Xtr_w,  ytr_w,  feat_mlp_batch_size, True)
            val_loader   = _loader(Xval_w, yval_w, feat_mlp_batch_size, False)
            print(f'\n[FeatureMLP] Training final model on all {len(subject_ids)} subjects...')
            model = FeatureMLP(input_size=all_X.shape[2], dropout_rate=dropout).to(device)
            best_state, best_val, best_epoch = train(
                model, train_loader, val_loader, device,
                max_epochs=max_epochs, patience=patience, min_delta=min_delta, lr=lr,
            )
            torch.save({
                'subject':    'final',
                'model':      'FeatureMLP',
                'mean_val':   float(best_val),
                'mean_epoch': float(best_epoch),
                'state_dict': _cpu_state(best_state),
                'n_subjects': len(subject_ids),
                'protocol':   'cross',
                'date':       datetime.date.today().isoformat(),
            }, mlp_final)
            print(f'  → FeatureMLP final model val {best_val*100:.2f}% @ epoch {best_epoch}')

    else:
        raise ValueError(f"type must be 'deep_learning' or 'feature_based', got {type!r}")
