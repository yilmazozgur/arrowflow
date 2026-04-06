"""
Sample Efficiency Experiment (C.3) — ArrowFlow vs Baselines at Small N
======================================================================
Tests whether ArrowFlow's ordinal inductive bias provides an advantage
when training data is scarce.

Protocol:
    1. Take all 7 UCI datasets.
    2. Subsample training data to N ∈ {20, 50, 100, 200, full}.
    3. Train ArrowFlow (best config) and baselines (GridSearchCV) at each size.
    4. Evaluate on the full, fixed test set.

Run:
    python -m test.experiments.exp_sample_efficiency
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import csv
import time
import warnings
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from arrowflow.benchmark import (
    ArrowFlowConfig, load_dataset, run_arrowflow_ensemble,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Best ArrowFlow configs (from UCI sweep)
# ---------------------------------------------------------------------------

BEST_CONFIGS = {
    'iris': {
        'filters': [64, 128], 'layer_types': ['sort', 'sort', 'sort'],
        'embed_dim': 16, 'pol_deg': 3, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 3,
        'learning_rate': 0.1, 'last_layer_update': True,
    },
    'wine': {
        'filters': [128], 'layer_types': ['sort', 'sort'],
        'embed_dim': 64, 'pol_deg': 1, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': False, 'n_iters': 200, 'n_classes': 3,
        'learning_rate': 0.1, 'last_layer_update': True,
    },
    'breast_cancer': {
        'filters': [64, 128], 'layer_types': ['sort', 'sort', 'sort'],
        'embed_dim': 32, 'pol_deg': 2, 'projection_strategy': 'random',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 2,
        'learning_rate': 0.1, 'last_layer_update': True,
    },
    'wine_quality': {
        'filters': [128], 'layer_types': ['sort', 'sort'],
        'embed_dim': 16, 'pol_deg': 3, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 3,
        'learning_rate': 0.1, 'last_layer_update': True,
    },
    'vehicle': {
        'filters': [64], 'layer_types': ['sort', 'sort'],
        'embed_dim': 32, 'pol_deg': 2, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 4,
        'learning_rate': 0.1, 'last_layer_update': True,
    },
    'segment': {
        'filters': [128, 256], 'layer_types': ['sort', 'sort', 'sort'],
        'embed_dim': 32, 'pol_deg': 2, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 7,
        'learning_rate': 0.1, 'last_layer_update': True,
    },
    'digits': {
        'filters': [256], 'layer_types': ['sort', 'sort'],
        'embed_dim': 64, 'pol_deg': 1, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': False, 'n_iters': 200, 'n_classes': 10,
        'learning_rate': 0.2, 'last_layer_update': True,
    },
}

BASELINE_METHODS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200], 'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(64,), (128,), (64, 64)],
        'learning_rate_init': [0.001, 0.01], 'max_iter': [1000],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [1, 3, 5, 7], 'weights': ['uniform', 'distance'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }),
}

TRAIN_SIZES = [20, 50, 100, 200]  # plus 'full'
N_AF_SIMS = 3
DATASETS = ['iris', 'wine', 'breast_cancer', 'wine_quality',
            'vehicle', 'segment', 'digits']


def subsample_stratified(X, y, n, rng):
    """Subsample to n samples with stratified class balance."""
    classes = np.unique(y)
    n_classes = len(classes)
    # Ensure at least 2 per class for train/val split
    per_class = max(2, n // n_classes)
    actual_n = per_class * n_classes

    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
        indices.extend(chosen)

    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


def train_baseline(name, X_train, y_train, X_test, y_test):
    """Train a single baseline. Returns error rate."""
    label, cls, param_grid = BASELINE_METHODS[name]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    n_classes = len(np.unique(y_train))
    min_class_count = min(np.bincount(y_train.astype(int)))
    cv_folds = min(3, min_class_count)

    if cv_folds >= 2 and param_grid and len(X_train) >= 10:
        try:
            model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
            grid = GridSearchCV(model, param_grid, cv=cv_folds,
                                scoring='accuracy', n_jobs=4)
            grid.fit(X_tr_s, y_train)
            best_model = grid.best_estimator_
        except Exception:
            best_model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
            best_model.fit(X_tr_s, y_train)
    else:
        best_model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
        best_model.fit(X_tr_s, y_train)

    y_pred = best_model.predict(X_te_s)
    return float(1.0 - accuracy_score(y_test, y_pred))


def train_arrowflow(X_train, y_train, X_test, y_test, n_classes, cfg, seed=42):
    """Train ArrowFlow ensemble and return mean error."""
    af_config = ArrowFlowConfig(
        no_of_filters=cfg['filters'],
        layer_types=cfg['layer_types'],
        no_of_iters=cfg['n_iters'],
        moe_no_of_networks=1,
        no_of_embedding_dim=cfg['embed_dim'],
        poly_expansion=(cfg['pol_deg'] > 1),
        pol_deg=cfg['pol_deg'],
        n_ensemble_views=cfg['n_views'],
        projection_strategy=cfg['projection_strategy'],
        use_augmentation=cfg['augment'],
        n_augmentations=1 if cfg['augment'] else 0,
        max_swaps=2,
        lda_ratio=0.3,
        learning_rate=cfg['learning_rate'],
        last_layer_update=cfg['last_layer_update'],
        verbose=0,
    )
    errors = []
    for sim in range(N_AF_SIMS):
        sim_seed = seed + sim * 12345
        err = run_arrowflow_ensemble(
            X_train, y_train, X_test, y_test, n_classes,
            af_config, seed=sim_seed
        )
        errors.append(err)
    return float(np.mean(errors))


def run_experiment():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'sample_efficiency_{timestamp}.csv')

    csv_fields = ['dataset', 'method', 'train_size', 'error_pct']
    csv_rows = []

    print("=" * 80)
    print("Sample Efficiency Experiment: ArrowFlow vs Baselines at Small N")
    print(f"Training sizes: {TRAIN_SIZES} + full")
    print(f"ArrowFlow sims: {N_AF_SIMS}")
    print("=" * 80)

    for dataset_name in DATASETS:
        cfg = BEST_CONFIGS[dataset_name]
        X_train_full, y_train_full, X_test, y_test, n_classes = load_dataset(
            dataset_name, random_state=42
        )

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name} — {len(X_train_full)} train, "
              f"{len(X_test)} test, {n_classes} classes")
        print(f"{'=' * 70}")

        sizes = [s for s in TRAIN_SIZES if s < len(X_train_full)]
        sizes.append(len(X_train_full))  # 'full'

        for n_train in sizes:
            is_full = (n_train == len(X_train_full))
            size_label = 'full' if is_full else str(n_train)

            if is_full:
                X_tr, y_tr = X_train_full, y_train_full
            else:
                rng = np.random.RandomState(42)
                X_tr, y_tr = subsample_stratified(X_train_full, y_train_full,
                                                   n_train, rng)

            print(f"\n  --- N={size_label} (actual={len(X_tr)}) ---")

            # Baselines
            for bl_name in BASELINE_METHODS:
                t0 = time.time()
                err = train_baseline(bl_name, X_tr, y_tr, X_test, y_test)
                elapsed = time.time() - t0
                label = BASELINE_METHODS[bl_name][0]
                print(f"    {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
                csv_rows.append({
                    'dataset': dataset_name, 'method': label,
                    'train_size': len(X_tr), 'error_pct': round(100 * err, 2),
                })

            # ArrowFlow
            t0 = time.time()
            af_err = train_arrowflow(X_tr, y_tr, X_test, y_test,
                                     n_classes, cfg, seed=42)
            elapsed = time.time() - t0
            print(f"    {'ArrowFlow':20s}: {100*af_err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'dataset': dataset_name, 'method': 'ArrowFlow',
                'train_size': len(X_tr), 'error_pct': round(100 * af_err, 2),
            })

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to: {csv_path}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Error (%) at each training size")
    print("=" * 80)
    for dataset_name in DATASETS:
        ds_rows = [r for r in csv_rows if r['dataset'] == dataset_name]
        sizes_in_ds = sorted(set(r['train_size'] for r in ds_rows))
        methods_in_ds = sorted(set(r['method'] for r in ds_rows),
                               key=lambda m: m != 'ArrowFlow')
        print(f"\n--- {dataset_name} ---")
        header = f"  {'Method':20s}" + "".join(f"  N={s:<5}" for s in sizes_in_ds)
        print(header)
        for method in methods_in_ds:
            row_str = f"  {method:20s}"
            for sz in sizes_in_ds:
                match = [r for r in ds_rows
                         if r['method'] == method and r['train_size'] == sz]
                if match:
                    row_str += f"  {match[0]['error_pct']:5.1f}%"
                else:
                    row_str += "     — "
            print(row_str)

    return csv_rows


if __name__ == '__main__':
    run_experiment()
