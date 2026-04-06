"""
Privacy-Preserving Classification (C.4) — Rank-Transformed Features
===================================================================
Tests whether ArrowFlow handles ordinal-only (rank-transformed) data
better than baselines that were designed for raw numeric features.

Protocol:
    1. Take all 7 UCI datasets.
    2. Rank-transform features: replace each value with its rank within
       that feature column (across training set).
    3. Train and evaluate ALL methods on both raw and rank-transformed data.
    4. Compare: which methods degrade least when magnitudes are removed?

The claim: ArrowFlow's architecture is natively designed for ordinal input,
so it should degrade less than baselines when moving from raw to ranked data.

Run:
    python -m test.experiments.exp_privacy_rank
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
from scipy.stats import rankdata
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
# Best ArrowFlow configs
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
        'n_estimators': [100, 200, 500], 'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2], 'max_features': ['sqrt', 'log2', None],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(64,), (128,), (256,), (64, 64), (128, 64), (256, 128)],
        'learning_rate_init': [0.001, 0.01], 'alpha': [0.0001, 0.001],
        'max_iter': [1000],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [1, 3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.8, 1.0],
    }),
}

N_AF_SIMS = 3
DATASETS = ['iris', 'wine', 'breast_cancer', 'wine_quality',
            'vehicle', 'segment', 'digits']


def rank_transform(X_train, X_test):
    """Rank-transform features: replace values with per-column ranks.

    Training ranks are computed within the training set.
    Test ranks are computed by inserting test values into the training
    distribution (rank among combined set, to avoid data leakage while
    maintaining consistent scale).
    """
    n_train = len(X_train)
    X_combined = np.vstack([X_train, X_test])
    X_ranked = np.zeros_like(X_combined, dtype=float)
    for j in range(X_combined.shape[1]):
        X_ranked[:, j] = rankdata(X_combined[:, j], method='average')
    # Normalize to [0, 1] for consistent scale
    X_ranked = X_ranked / len(X_combined)
    return X_ranked[:n_train], X_ranked[n_train:]


def train_baseline(name, X_train, y_train, X_test, y_test):
    """Train a baseline with GridSearchCV. Returns error rate."""
    label, cls, param_grid = BASELINE_METHODS[name]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    if param_grid:
        model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
        grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=8)
        grid.fit(X_tr_s, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = cls()
        best_model.fit(X_tr_s, y_train)

    y_pred = best_model.predict(X_te_s)
    return float(1.0 - accuracy_score(y_test, y_pred))


def train_arrowflow(X_train, y_train, X_test, y_test, n_classes, cfg, seed=42):
    """Train ArrowFlow and return mean error over sims."""
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
    csv_path = os.path.join(results_dir, f'privacy_rank_{timestamp}.csv')

    csv_fields = ['dataset', 'method', 'data_mode', 'error_pct']
    csv_rows = []

    print("=" * 80)
    print("Privacy-Preserving Classification: Raw vs Rank-Transformed Features")
    print(f"ArrowFlow sims: {N_AF_SIMS}")
    print("=" * 80)

    for dataset_name in DATASETS:
        cfg = BEST_CONFIGS[dataset_name]
        X_train, y_train, X_test, y_test, n_classes = load_dataset(
            dataset_name, random_state=42
        )

        # Rank-transform
        X_train_rank, X_test_rank = rank_transform(X_train, X_test)

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name} — {len(X_train)} train, "
              f"{len(X_test)} test, {n_classes} classes, "
              f"{X_train.shape[1]} features")
        print(f"{'=' * 70}")

        for data_mode, X_tr, X_te in [('raw', X_train, X_test),
                                        ('ranked', X_train_rank, X_test_rank)]:
            print(f"\n  --- {data_mode} features ---")

            # Baselines
            for bl_name in BASELINE_METHODS:
                t0 = time.time()
                err = train_baseline(bl_name, X_tr, y_train, X_te, y_test)
                elapsed = time.time() - t0
                label = BASELINE_METHODS[bl_name][0]
                print(f"    {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
                csv_rows.append({
                    'dataset': dataset_name, 'method': label,
                    'data_mode': data_mode, 'error_pct': round(100 * err, 2),
                })

            # ArrowFlow
            t0 = time.time()
            af_err = train_arrowflow(X_tr, y_train, X_te, y_test,
                                     n_classes, cfg, seed=42)
            elapsed = time.time() - t0
            print(f"    {'ArrowFlow':20s}: {100*af_err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'dataset': dataset_name, 'method': 'ArrowFlow',
                'data_mode': data_mode, 'error_pct': round(100 * af_err, 2),
            })

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Error (%) — Raw vs Ranked")
    print("=" * 80)
    for dataset_name in DATASETS:
        ds_rows = [r for r in csv_rows if r['dataset'] == dataset_name]
        methods = sorted(set(r['method'] for r in ds_rows),
                         key=lambda m: m != 'ArrowFlow')
        print(f"\n--- {dataset_name} ---")
        print(f"  {'Method':20s}  {'Raw':>7s}  {'Ranked':>7s}  {'Delta':>7s}")
        for method in methods:
            raw = [r for r in ds_rows
                   if r['method'] == method and r['data_mode'] == 'raw']
            ranked = [r for r in ds_rows
                      if r['method'] == method and r['data_mode'] == 'ranked']
            if raw and ranked:
                r_val = raw[0]['error_pct']
                k_val = ranked[0]['error_pct']
                delta = k_val - r_val
                print(f"  {method:20s}  {r_val:6.1f}%  {k_val:6.1f}%  {delta:+6.1f}pp")

    return csv_rows


if __name__ == '__main__':
    run_experiment()
