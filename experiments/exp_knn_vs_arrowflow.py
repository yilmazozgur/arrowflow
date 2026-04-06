"""
kNN vs ArrowFlow on Encoded Permutations
=========================================
Tests whether ArrowFlow's learned sort filters improve over a simple kNN
baseline operating in the SAME permutation space with the SAME encoding.

If ArrowFlow beats kNN-on-encoded, it proves that the sort-layer learning
(permutation-matrix filter updates) genuinely learns better representations
than mere nearest-neighbor lookup in permutation distance space.

Uses the best config per dataset from the UCI sweep (embed_dim, pol_deg,
projection_strategy, n_views) to ensure a fair comparison.

Run:
    python -m test.experiments.exp_knn_vs_arrowflow
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import time
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier

from arrowflow.benchmark import (
    ArrowFlowConfig, load_dataset, run_arrowflow_experiment,
)
from arrowflow.arrowflow import DataGraph


# ---------------------------------------------------------------------------
# Encoding functions — mirrors the benchmark harness exactly
# ---------------------------------------------------------------------------

def encode_view(X_tr_poly, y_train, X_te_poly, strategy, embed_dim,
                lda_ratio, seed):
    """Encode one view using the same pipeline as ArrowFlow."""
    if strategy == 'target_aware':
        perm_train, perm_test = DataGraph.target_aware_encode(
            X_tr_poly, y_train, X_te_poly,
            embed_dim=embed_dim, lda_ratio=lda_ratio, seed=seed
        )
    elif strategy == 'calibrated':
        perm_train, perm_test = DataGraph.calibrated_encode(
            X_tr_poly, y_train, X_te_poly,
            embed_dim=embed_dim, seed=seed, calibration='standardize'
        )
    else:  # 'random'
        rng = np.random.RandomState(seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_poly)
        X_te_s = scaler.transform(X_te_poly)
        W = rng.randn(X_tr_s.shape[1], embed_dim)
        perm_train = np.argsort(X_tr_s @ W, axis=1).astype(float)
        perm_test = np.argsort(X_te_s @ W, axis=1).astype(float)
    return perm_train, perm_test


def get_strategies(n_views, projection_strategy):
    """Get per-view projection strategies (same cycling as ArrowFlow)."""
    if projection_strategy == 'diverse':
        cycle = ['target_aware', 'random', 'calibrated']
        return [cycle[v % 3] for v in range(n_views)]
    return [projection_strategy] * n_views


def knn_multiview_ensemble(X_train, y_train, X_test, y_test, n_classes,
                            embed_dim, pol_deg, projection_strategy,
                            lda_ratio, n_views, k, seed):
    """Multi-view kNN ensemble with majority vote on encoded permutations."""
    # Polynomial expansion (shared, same as ArrowFlow)
    if pol_deg > 1:
        poly = PolynomialFeatures(degree=pol_deg)
        X_tr_poly = poly.fit_transform(X_train)
        X_te_poly = poly.transform(X_test)
    else:
        X_tr_poly = X_train.copy()
        X_te_poly = X_test.copy()

    strategies = get_strategies(n_views, projection_strategy)
    all_predictions = []

    for v in range(n_views):
        view_seed = seed + v * 9999
        perm_train, perm_test = encode_view(
            X_tr_poly, y_train, X_te_poly, strategies[v],
            embed_dim, lda_ratio, view_seed
        )
        knn = KNeighborsClassifier(
            n_neighbors=min(k, len(perm_train) - 1),
            metric='manhattan', n_jobs=1
        )
        knn.fit(perm_train, y_train)
        all_predictions.append(knn.predict(perm_test))

    # Majority vote (same as ArrowFlow ensemble)
    predictions = np.array(all_predictions)
    final_preds = np.array([
        np.bincount(predictions[:, i].astype(int), minlength=n_classes).argmax()
        for i in range(len(y_test))
    ])
    return float(np.sum(final_preds != y_test) / len(y_test))


# ---------------------------------------------------------------------------
# Best configs per dataset (from UCI sweep results)
# ---------------------------------------------------------------------------

BEST_CONFIGS = {
    'iris': {
        'filters': [64, 128], 'layer_types': ['sort', 'sort', 'sort'],
        'embed_dim': 16, 'pol_deg': 3, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 3,
        'learning_rate': 0.1, 'last_layer_update': True,
        'best_error': 2.7,
    },
    'wine': {
        'filters': [128], 'layer_types': ['sort', 'sort'],
        'embed_dim': 64, 'pol_deg': 1, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': False, 'n_iters': 200, 'n_classes': 3,
        'learning_rate': 0.1, 'last_layer_update': True,
        'best_error': 2.8,
    },
    'breast_cancer': {
        'filters': [64, 128], 'layer_types': ['sort', 'sort', 'sort'],
        'embed_dim': 32, 'pol_deg': 2, 'projection_strategy': 'random',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 2,
        'learning_rate': 0.1, 'last_layer_update': True,
        'best_error': 2.8,
    },
    'wine_quality': {
        'filters': [128], 'layer_types': ['sort', 'sort'],
        'embed_dim': 16, 'pol_deg': 3, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 3,
        'learning_rate': 0.1, 'last_layer_update': True,
        'best_error': 35.9,
    },
    'vehicle': {
        'filters': [64], 'layer_types': ['sort', 'sort'],
        'embed_dim': 32, 'pol_deg': 2, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 4,
        'learning_rate': 0.1, 'last_layer_update': True,
        'best_error': 17.4,
    },
    'segment': {
        'filters': [128, 256], 'layer_types': ['sort', 'sort', 'sort'],
        'embed_dim': 32, 'pol_deg': 2, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': True, 'n_iters': 200, 'n_classes': 7,
        'learning_rate': 0.1, 'last_layer_update': True,
        'best_error': 5.2,
    },
    'digits': {
        'filters': [256], 'layer_types': ['sort', 'sort'],
        'embed_dim': 64, 'pol_deg': 1, 'projection_strategy': 'diverse',
        'n_views': 7, 'augment': False, 'n_iters': 200, 'n_classes': 10,
        'learning_rate': 0.2, 'last_layer_update': True,
        'best_error': 4.6,
    },
}


def run_comparison():
    """Run kNN vs ArrowFlow on all datasets using best ArrowFlow config."""
    n_sims = 5
    k_values = [1, 3, 5, 7]
    datasets = ['iris', 'wine', 'breast_cancer', 'wine_quality',
                'vehicle', 'segment', 'digits']

    print("=" * 80)
    print("kNN on Encoded Permutations vs ArrowFlow — Same Encoding Pipeline")
    print("=" * 80)

    for dataset_name in datasets:
        cfg = BEST_CONFIGS[dataset_name]

        X_train, y_train, X_test, y_test, n_classes = load_dataset(
            dataset_name, random_state=42
        )

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name} — {len(X_train)} train, "
              f"{len(X_test)} test, {n_classes} classes, "
              f"{X_train.shape[1]} features")
        print(f"Best ArrowFlow config: f={cfg['filters']} e={cfg['embed_dim']} "
              f"pol={cfg['pol_deg']} strat={cfg['projection_strategy']} "
              f"v={cfg['n_views']} lr={cfg['learning_rate']}")
        print(f"ArrowFlow best error (from sweep): {cfg['best_error']:.1f}%")
        print(f"{'=' * 70}")

        # --- Single-view kNN (same encoding, one projection) ---
        print(f"\n  --- Single-view kNN (L1/Manhattan on encoded perms) ---")
        for k in k_values:
            errors = []
            for sim in range(n_sims):
                sim_seed = 42 + sim * 1000
                err = knn_multiview_ensemble(
                    X_train, y_train, X_test, y_test, n_classes,
                    embed_dim=cfg['embed_dim'], pol_deg=cfg['pol_deg'],
                    projection_strategy=cfg['projection_strategy'],
                    lda_ratio=0.3, n_views=1, k=k, seed=sim_seed
                )
                errors.append(err)
            mean_err = np.mean(errors)
            std_err = np.std(errors) / np.sqrt(n_sims)
            print(f"    kNN(k={k}, 1 view):  {100*mean_err:.1f}% ± {100*std_err:.1f}%")

        # --- Multi-view kNN ensemble (same encoding + same # views) ---
        print(f"\n  --- Multi-view kNN ensemble ({cfg['n_views']} views, majority vote) ---")
        for k in k_values:
            errors = []
            t0 = time.time()
            for sim in range(n_sims):
                sim_seed = 42 + sim * 1000
                err = knn_multiview_ensemble(
                    X_train, y_train, X_test, y_test, n_classes,
                    embed_dim=cfg['embed_dim'], pol_deg=cfg['pol_deg'],
                    projection_strategy=cfg['projection_strategy'],
                    lda_ratio=0.3, n_views=cfg['n_views'], k=k, seed=sim_seed
                )
                errors.append(err)
            elapsed = time.time() - t0
            mean_err = np.mean(errors)
            std_err = np.std(errors) / np.sqrt(n_sims)
            min_err = np.min(errors)
            print(f"    kNN(k={k}, {cfg['n_views']}v): "
                  f"{100*mean_err:.1f}% ± {100*std_err:.1f}% "
                  f"(min={100*min_err:.1f}%) [{elapsed:.1f}s]")

        # --- ArrowFlow with best config (re-run for fair seed comparison) ---
        print(f"\n  --- ArrowFlow (best config, {n_sims} sims) ---")
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
        t0 = time.time()
        af_result = run_arrowflow_experiment(
            dataset_name, af_config, n_sims=n_sims, random_state=42
        )
        elapsed = time.time() - t0
        print(f"    ArrowFlow:          {100*af_result['test_error_mean']:.1f}% "
              f"± {100*af_result['test_error_stderr']:.1f}% "
              f"(min={100*af_result['test_error_min']:.1f}%) [{elapsed:.1f}s]")


if __name__ == '__main__':
    run_comparison()
