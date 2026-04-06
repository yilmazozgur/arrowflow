"""
UCI Tabular Benchmarks — Multi-View Ensemble ArrowFlow
=======================================================
Explores ArrowFlow architecture choices (width, depth, embedding dim,
number of views, projection strategy) on UCI classification datasets.

All ArrowFlow configs use the multi-view ensemble algorithm:
  - Each view trains an independent network on a different random projection
  - Final prediction via majority vote across views
  - Polynomial feature expansion for low-dimensional data
  - Diverse projection strategies (LDA + random + calibrated) for multi-class

Baselines use GridSearchCV(cv=3) for fair hyperparameter selection.

Run:
    python -m test.experiments.exp_uci_tabular [--quick] [--adaptive-only]

Results are logged to test/experiments/results/uci_tabular_*.csv
"""

import sys
import copy
import time
import itertools
from arrowflow.benchmark import (
    ArrowFlowConfig, run_benchmark, run_learning_curve, log_results,
    run_arrowflow_experiment, run_baseline, load_dataset,
    BASELINE_CONFIGS,
)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DATASETS = {
    'iris':          {'n': 150,   'features': 4,  'classes': 3,  'notes': 'Tiny classic'},
    'wine':          {'n': 178,   'features': 13, 'classes': 3,  'notes': 'Wine variety classification'},
    'breast_cancer': {'n': 569,   'features': 30, 'classes': 2,  'notes': 'Medical, moderate dim'},
    'wine_quality':  {'n': 1599,  'features': 11, 'classes': 3,  'notes': 'Red wine quality (binned)'},
    'vehicle':       {'n': 846,   'features': 18, 'classes': 4,  'notes': 'Vehicle silhouettes'},
    'segment':       {'n': 2310,  'features': 19, 'classes': 7,  'notes': 'Image segmentation'},
    'digits':        {'n': 5620,  'features': 64, 'classes': 10, 'notes': 'Pixel sums in 4x4 blocks'},
}

# Reference accuracy (%) from literature for context.
SOTA_REFERENCE = {
    'iris':          {'accuracy': 98.0, 'method': 'SVM-RBF',  'source': 'LIBSVM / UCI literature'},
    'wine':          {'accuracy': 99.0, 'method': 'SVM-RBF',  'source': 'LIBSVM / UCI literature'},
    'breast_cancer': {'accuracy': 97.5, 'method': 'SVM-RBF',  'source': 'Mangasarian et al. / sklearn'},
    'wine_quality':  {'accuracy': 62.0, 'method': 'SVM/RF',   'source': 'Cortez et al. 2009 (3-class binned)'},
    'vehicle':       {'accuracy': 85.0, 'method': 'SVM-RBF',  'source': 'UCI literature'},
    'segment':       {'accuracy': 97.5, 'method': 'RF/SVM',   'source': 'UCI literature'},
    'digits':        {'accuracy': 99.0, 'method': 'SVM-RBF',  'source': 'LIBSVM guide (8x8 digits)'},
}

# Baseline GridSearchCV results (test error %) — cached to avoid re-running.
# Generated 2026-04-03 with GridSearchCV(cv=3) on standard 80/20 split (random_state=42).
# RF: n_estimators[100,200,500] max_depth[5,10,None] min_samples_leaf[1,2] max_features[sqrt,log2,None]
# SVM: C[0.01..100] gamma[scale,auto,0.001,0.01,0.1]
# MLP: hidden[(64,)..(256,128)] lr_init[0.001,0.01] alpha[0.0001,0.001] max_iter=1000
# KNN: n_neighbors[1..15] weights[uniform,distance] metric[euclidean,manhattan]
# XGBoost: n_estimators[100,200,500] max_depth[3,5,7,10] lr[0.01,0.05,0.1] subsample[0.8,1.0]
BASELINE_SWEEP = {
    'iris': {
        'rf':  {'error_pct': 3.3, 'time': 2.7},
        'svm': {'error_pct': 3.3, 'time': 0.0},
        'mlp': {'error_pct': 3.3, 'time': 0.8},
        'knn': {'error_pct': 3.3, 'time': 0.0},
        'xgb': {'error_pct': 3.3, 'time': 6.1},
    },
    'wine': {
        'rf':  {'error_pct': 0.0, 'time': 2.3},
        'svm': {'error_pct': 2.8, 'time': 0.0},
        'mlp': {'error_pct': 2.8, 'time': 0.4},
        'knn': {'error_pct': 0.0, 'time': 0.0},
        'xgb': {'error_pct': 2.8, 'time': 9.8},
    },
    'breast_cancer': {
        'rf':  {'error_pct': 4.4, 'time': 5.8},
        'svm': {'error_pct': 1.8, 'time': 0.0},
        'mlp': {'error_pct': 4.4, 'time': 1.3},
        'knn': {'error_pct': 2.6, 'time': 0.1},
        'xgb': {'error_pct': 4.4, 'time': 17.8},
    },
    'wine_quality': {
        'rf':  {'error_pct': 25.9, 'time': 8.3},
        'svm': {'error_pct': 30.6, 'time': 0.3},
        'mlp': {'error_pct': 30.3, 'time': 12.8},
        'knn': {'error_pct': 29.7, 'time': 0.1},
        'xgb': {'error_pct': 24.1, 'time': 64.2},
    },
    'vehicle': {
        'rf':  {'error_pct': 27.1, 'time': 5.8},
        'svm': {'error_pct': 14.1, 'time': 0.1},
        'mlp': {'error_pct': 12.4, 'time': 6.0},
        'knn': {'error_pct': 27.1, 'time': 0.1},
        'xgb': {'error_pct': 21.8, 'time': 55.3},
    },
    'segment': {
        'rf':  {'error_pct': 2.8, 'time': 13.2},
        'svm': {'error_pct': 3.2, 'time': 0.3},
        'mlp': {'error_pct': 3.7, 'time': 9.3},
        'knn': {'error_pct': 4.1, 'time': 0.1},
        'xgb': {'error_pct': 2.6, 'time': 207.5},
    },
    'digits': {
        'rf':  {'error_pct': 2.8, 'time': 11.4},
        'svm': {'error_pct': 1.7, 'time': 0.3},
        'mlp': {'error_pct': 1.9, 'time': 3.8},
        'knn': {'error_pct': 2.2, 'time': 0.1},
        'xgb': {'error_pct': 4.2, 'time': 250.4},
    },
}

# ---------------------------------------------------------------------------
# Multi-view ensemble configs — sweep architecture choices
# ---------------------------------------------------------------------------
# All configs use n_ensemble_views >= 1 so they route through
# run_arrowflow_ensemble() which applies poly expansion, diverse
# projections, augmentation, and majority voting.

def _make_ensemble_config(filters, layer_types, embed_dim, n_views=7,
                          pol_deg=None, projection_strategy=None,
                          augment=None, n_iters=200, n_classes=None,
                          learning_rate=0.1, lr_tensor=0.01, batch_size=32,
                          last_layer_update=True):
    """Helper to build a multi-view ensemble ArrowFlowConfig.

    If pol_deg/projection_strategy/augment are None, dataset-adaptive
    defaults are used based on embed_dim and n_classes.
    """
    # Adaptive defaults based on embedding dimension (proxy for feature count)
    if pol_deg is None:
        if embed_dim <= 16:
            pol_deg = 3
        elif embed_dim <= 32:
            pol_deg = 2
        else:
            pol_deg = 1

    if projection_strategy is None:
        projection_strategy = 'diverse' if (n_classes is None or n_classes >= 3) else 'random'

    if augment is None:
        augment = (embed_dim <= 32)

    return ArrowFlowConfig(
        no_of_filters=filters,
        layer_types=layer_types,
        no_of_iters=n_iters,
        moe_no_of_networks=1,
        no_of_embedding_dim=embed_dim,
        poly_expansion=(pol_deg > 1),
        pol_deg=pol_deg,
        n_ensemble_views=n_views,
        projection_strategy=projection_strategy,
        use_augmentation=augment,
        n_augmentations=1 if augment else 0,
        max_swaps=2,
        lda_ratio=0.3,
        learning_rate=learning_rate,
        lr_tensor=lr_tensor,
        batch_size=batch_size,
        last_layer_update=last_layer_update,
        verbose=0,
    )


# --- Quick configs: small sweep for fast iteration ---
CONFIGS_QUICK = {
    'iris': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_views=7, n_classes=3),
    ],
    'wine': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7, n_classes=3),
    ],
    'breast_cancer': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7, n_classes=2),
    ],
    'wine_quality': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7, n_classes=3),
    ],
    'vehicle': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7, n_classes=4),
    ],
    'segment': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7, n_classes=7),
    ],
    'digits': [
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3, n_classes=10, n_iters=100, pol_deg=1),
    ],
}

# --- Full sweep: explore architecture choices per dataset ---
# Each dataset gets configs tailored to its properties (feature count,
# number of classes) while sweeping width, depth, and embed_dim.

CONFIGS_FULL = {
    # ===================================================================
    # IRIS: 4 features, 3 classes, 150 samples
    # ===================================================================
    'iris': [
        # --- Width sweep (baseline: embed=16, v=7, pol=3, diverse, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=16, n_classes=3),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=16, n_classes=3),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=16, n_classes=3),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=16, n_classes=3),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=16, n_classes=3),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=8,  n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_views=3,  n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_views=5,  n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_views=11, n_classes=3),
        # --- Augmentation ablation ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=3, augment=False),
        # --- Projection strategy ablation ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=3,
                              projection_strategy='random'),
        # --- Learning rate sweep (on best: [64,128] d=2) ---
        _make_ensemble_config([64, 128], ['sort', 'sort', 'sort'], embed_dim=16, n_classes=3,
                              learning_rate=0.05),
        _make_ensemble_config([64, 128], ['sort', 'sort', 'sort'], embed_dim=16, n_classes=3,
                              learning_rate=0.2),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=3,
                              last_layer_update=False),
        # --- Hybrid architecture: tensor→sort (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=16, n_classes=3,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=16, n_classes=3,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=16, n_classes=3,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=16, n_classes=3,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],

    # ===================================================================
    # WINE: 13 features, 3 classes, 178 samples
    # ===================================================================
    'wine': [
        # --- Width sweep (baseline: embed=32, v=7, pol=2, diverse, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=32, n_classes=3),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=32, n_classes=3),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=32, n_classes=3),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=32, n_classes=3),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=3),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=3),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3,  n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=11, n_classes=3),
        # --- Learning rate sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3,
                              learning_rate=0.05),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3,
                              learning_rate=0.2),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3,
                              last_layer_update=False),
        # --- Hybrid architecture (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],

    # ===================================================================
    # BREAST CANCER: 30 features, 2 classes, 569 samples
    # ===================================================================
    'breast_cancer': [
        # --- Width sweep (baseline: embed=32, v=7, pol=2, random, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=32, n_classes=2),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=32, n_classes=2),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=32, n_classes=2),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=32, n_classes=2),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=2),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=2),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=2),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3,  n_classes=2),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=11, n_classes=2),
        # --- Augmentation ablation ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=2, augment=False),
        # --- Learning rate sweep (on best: [64,128] d=2) ---
        _make_ensemble_config([64, 128], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=2,
                              learning_rate=0.05),
        _make_ensemble_config([64, 128], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=2,
                              learning_rate=0.2),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=2,
                              last_layer_update=False),
        # --- Hybrid architecture (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=2,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=2,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=2,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=2,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],

    # ===================================================================
    # WINE QUALITY: 11 features, 3 classes, 1599 samples
    # ===================================================================
    'wine_quality': [
        # --- Width sweep (baseline: embed=32, v=7, pol=2, diverse, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=32, n_classes=3),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=32, n_classes=3),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=32, n_classes=3),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=32, n_classes=3),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=3),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=3),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3,  n_classes=3),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=11, n_classes=3),
        # --- Learning rate sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3,
                              learning_rate=0.05),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3,
                              learning_rate=0.2),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=3,
                              last_layer_update=False),
        # --- Hybrid architecture (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=3,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],

    # ===================================================================
    # VEHICLE: 18 features, 4 classes, 846 samples
    # ===================================================================
    'vehicle': [
        # --- Width sweep (baseline: embed=32, v=7, pol=2, diverse, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=32, n_classes=4),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=32, n_classes=4),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=32, n_classes=4),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=32, n_classes=4),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=4),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=4),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=4),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3,  n_classes=4),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=11, n_classes=4),
        # --- Learning rate sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=4,
                              learning_rate=0.05),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=4,
                              learning_rate=0.2),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=4,
                              last_layer_update=False),
        # --- Hybrid architecture (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=4,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=4,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=4,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=4,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],

    # ===================================================================
    # SEGMENT: 19 features, 7 classes, 2310 samples
    # ===================================================================
    'segment': [
        # --- Width sweep (baseline: embed=32, v=7, pol=2, diverse, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=32, n_classes=7),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=32, n_classes=7),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=32, n_classes=7),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=32, n_classes=7),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=7),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=7),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=7),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3,  n_classes=7),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=11, n_classes=7),
        # --- Learning rate sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=7,
                              learning_rate=0.05),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=7,
                              learning_rate=0.2),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=7,
                              last_layer_update=False),
        # --- Hybrid architecture (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=7,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=7,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=7,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=7,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],

    # ===================================================================
    # DIGITS: 64 features, 10 classes, 1797 samples
    # Best prior result: embed=64, pol=1, no aug → 5.8%
    # ===================================================================
    'digits': [
        # --- Width sweep (baseline: embed=32, v=7, pol=2, diverse, aug) ---
        _make_ensemble_config([64],   ['sort', 'sort'], embed_dim=32, n_classes=10),
        _make_ensemble_config([128],  ['sort', 'sort'], embed_dim=32, n_classes=10),
        _make_ensemble_config([256],  ['sort', 'sort'], embed_dim=32, n_classes=10),
        _make_ensemble_config([512],  ['sort', 'sort'], embed_dim=32, n_classes=10),
        # --- Depth sweep ---
        _make_ensemble_config([64, 128],  ['sort', 'sort', 'sort'], embed_dim=32, n_classes=10),
        _make_ensemble_config([128, 256], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=10),
        _make_ensemble_config([256, 512], ['sort', 'sort', 'sort'], embed_dim=32, n_classes=10),
        # --- Embed dim sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=16, n_classes=10),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=10),
        # --- View count sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=3,  n_classes=10),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=11, n_classes=10),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=13, n_classes=10),
        # --- Iterations sweep ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7,
                              n_classes=10, n_iters=300),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_views=7,
                              n_classes=10, n_iters=400),
        # --- Learning rate sweep (on best config: embed=64, pol=1) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=10,
                              learning_rate=0.05),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=10,
                              learning_rate=0.2),
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=64, n_classes=10,
                              learning_rate=0.01),
        # --- Width sweep at best embed_dim=64 ---
        _make_ensemble_config([64],  ['sort', 'sort'], embed_dim=64, n_classes=10),
        _make_ensemble_config([256], ['sort', 'sort'], embed_dim=64, n_classes=10),
        # --- last_layer_update ablation (sort-only) ---
        _make_ensemble_config([128], ['sort', 'sort'], embed_dim=32, n_classes=10,
                              last_layer_update=False),
        # --- Hybrid architecture (llu=False) ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=10,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=10,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        _make_ensemble_config([64, 128], ['tensor', 'sort', 'sort'], embed_dim=32, n_classes=10,
                              lr_tensor=0.01, last_layer_update=False, batch_size=16, n_iters=500),
        # --- Hybrid with last_layer_update=True ---
        _make_ensemble_config([64],  ['tensor', 'sort'], embed_dim=32, n_classes=10,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([128], ['tensor', 'sort'], embed_dim=32, n_classes=10,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
        _make_ensemble_config([64, 128], ['tensor', 'sort', 'sort'], embed_dim=32, n_classes=10,
                              lr_tensor=0.01, last_layer_update=True, batch_size=16, n_iters=500),
    ],
}


# Baselines to compare against (all use GridSearchCV)
BASELINES = ['rf', 'svm', 'mlp', 'knn', 'xgb']

# Learning curve subsample fractions
LEARNING_CURVE_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_full_benchmark(quick=False, adaptive_only=False):
    """Run ArrowFlow + baselines on all UCI datasets.

    Baselines are run once per dataset (in parallel), then all ArrowFlow
    configs are swept with views parallelized across cores.

    Args:
        quick: If True, run minimal configs for fast iteration.
        adaptive_only: If True, skip the architecture sweep and only run
            the adaptive config (auto-selected based on dataset properties).
    """
    datasets = list(DATASETS.keys())
    n_sims = 3 if quick else 5

    all_results = []
    for dataset_name in datasets:
        ds_info = DATASETS[dataset_name]
        ref = SOTA_REFERENCE.get(dataset_name, {})

        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name} — {ds_info['notes']}")
        print(f"  n={ds_info['n']}, features={ds_info['features']}, classes={ds_info['classes']}")
        if ref:
            print(f"  SOTA ref: {ref['accuracy']:.1f}% ({ref['method']})")
        print(f"{'='*60}")

        # Load data once for this dataset
        X_train, y_train, X_test, y_test, n_classes = load_dataset(
            dataset_name, random_state=42
        )
        print(f"Data: {len(X_train)} train, {len(X_test)} test, {n_classes} classes, {X_train.shape[1]} features")

        # --- Baselines: use cached results if available, otherwise run ---
        cached = BASELINE_SWEEP.get(dataset_name)
        if cached:
            print(f"\n--- Baselines (cached from GridSearchCV sweep) ---")
            bl_label_map = {k: v[0] for k, v in BASELINE_CONFIGS.items()}
            for bl_name in BASELINES:
                if bl_name in cached:
                    info = cached[bl_name]
                    label = bl_label_map.get(bl_name, bl_name)
                    print(f"  {label}: {info['error_pct']:.1f}% (cached)")
                    all_results.append({
                        'method': label,
                        'dataset': dataset_name,
                        'n_train': len(X_train),
                        'n_test': len(X_test),
                        'n_classes': n_classes,
                        'test_error_mean': info['error_pct'] / 100.0,
                        'test_error_std': 0.0,
                        'test_error_min': info['error_pct'] / 100.0,
                        'test_error_max': info['error_pct'] / 100.0,
                        'test_error_stderr': 0.0,
                        'time_mean': info['time'],
                        'n_sims': 1,
                        'subsample': None,
                        'config': {'source': 'cached_gridsearch'},
                    })
        else:
            print(f"\n--- Running {len(BASELINES)} baselines ---")
            t0_bl = time.time()
            for bl_name in BASELINES:
                label = BASELINE_CONFIGS[bl_name][0]
                try:
                    bl_result = run_baseline(bl_name, X_train, y_train, X_test, y_test,
                                             random_state=42)
                    bl_result['dataset'] = dataset_name
                    bl_result['subsample'] = None
                    bl_result['n_classes'] = n_classes
                    print(f"  {label}: {100*bl_result['test_error_mean']:.1f}% ({bl_result['time_mean']:.1f}s)")
                    all_results.append(bl_result)
                except Exception as e:
                    print(f"  {label}: FAILED: {e}")
            print(f"  Baselines done in {time.time() - t0_bl:.1f}s")

        # --- Sweep ArrowFlow configs ---
        if adaptive_only:
            configs = [ArrowFlowConfig(adaptive_mode=True, verbose=1)]
        elif quick:
            configs = CONFIGS_QUICK.get(dataset_name, [])
        else:
            configs = CONFIGS_FULL.get(dataset_name, [])

        for i, config in enumerate(configs):
            lr_str = f" lr={config.learning_rate}" if config.learning_rate != 0.1 else ""
            arch_str = "→".join(config.layer_types[:-1])  # e.g. "sort" or "tensor→sort"
            llu_str = f" llu={config.last_layer_update}" if ('tensor' in config.layer_types or not config.last_layer_update) else ""
            print(f"\n  --- ArrowFlow config {i+1}/{len(configs)}: "
                  f"f={config.no_of_filters} d={len(config.layer_types)-1} "
                  f"arch={arch_str} "
                  f"e={config.no_of_embedding_dim} v={config.n_ensemble_views} "
                  f"pol={config.pol_deg} strat={config.projection_strategy} "
                  f"aug={config.use_augmentation} iters={config.no_of_iters}{lr_str}{llu_str} ---")
            t0_af = time.time()
            try:
                af_result = run_arrowflow_experiment(
                    dataset_name, config, n_sims=n_sims, random_state=42
                )
                af_result['subsample'] = None
                elapsed = time.time() - t0_af
                print(f"  -> {100*af_result['test_error_mean']:.1f}% "
                      f"+/- {100*af_result['test_error_stderr']:.1f}% "
                      f"(min={100*af_result['test_error_min']:.1f}%) [{elapsed:.1f}s]")
                all_results.append(af_result)
            except Exception as e:
                print(f"  -> FAILED: {e}")
                import traceback
                traceback.print_exc()

    filename = log_results(all_results, filename=None)
    return all_results, filename


def run_learning_curves(quick=False):
    """Run learning curve experiments on selected datasets."""
    n_sims = 3 if quick else 5
    datasets = ['iris', 'digits'] if quick else ['iris', 'wine', 'digits']
    fractions = [0.25, 0.5, 1.0] if quick else LEARNING_CURVE_FRACTIONS

    all_results = []
    for dataset_name in datasets:
        ds_info = DATASETS[dataset_name]
        # Use adaptive config for learning curves
        config = ArrowFlowConfig(adaptive_mode=True, verbose=1)

        print(f"\n{'#'*60}")
        print(f"# Learning Curve: {dataset_name}")
        print(f"{'#'*60}")
        results = run_learning_curve(
            dataset_name, config,
            subsample_fractions=fractions,
            n_sims=n_sims,
            baselines=['rf', 'svm', 'xgb']
        )
        all_results.extend(results)

    filename = log_results(all_results, filename=None)
    return all_results, filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    quick = '--quick' in sys.argv
    adaptive_only = '--adaptive-only' in sys.argv
    mode = 'learning_curve' if '--learning-curve' in sys.argv else 'benchmark'

    if mode == 'learning_curve':
        run_learning_curves(quick=quick)
    else:
        run_full_benchmark(quick=quick, adaptive_only=adaptive_only)
