"""
Natively Ordinal Data (C.5) — Sushi Preference Dataset
=======================================================
Tests ArrowFlow on data that IS natively ordinal: user rankings of sushi types.
ArrowFlow uses native encoding (rankings fed directly to sort layers).
Baselines treat rankings as numeric features.

Dataset: Sushi Preference Data Set (Kamishima, 2003)
    - 5000 users each ranking 10 sushi types
    - Task: predict user's region (East/West Japan) from sushi preferences

Run:
    python -m test.experiments.exp_sushi_ordinal
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import io
import csv
import time
import zipfile
import warnings
import numpy as np
from datetime import datetime
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from arrowflow.benchmark import (
    ArrowFlowConfig, run_arrowflow_ensemble,
)
from arrowflow.arrowflow import DataGraph

warnings.filterwarnings("ignore")

SUSHI_URL = "http://www.kamishima.net/asset/sushi3-2016.zip"
SUSHI_CACHE = os.path.join(os.path.dirname(__file__), 'results', 'sushi3-2016.zip')

BASELINE_METHODS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200, 500], 'max_depth': [5, 10, None],
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
        'n_neighbors': [1, 3, 5, 7, 11], 'weights': ['uniform', 'distance'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }),
}

N_AF_SIMS = 5


def download_sushi():
    """Download and cache the Sushi dataset."""
    if os.path.exists(SUSHI_CACHE):
        print(f"  Using cached sushi data: {SUSHI_CACHE}")
        return SUSHI_CACHE

    print(f"  Downloading sushi dataset from {SUSHI_URL}...")
    os.makedirs(os.path.dirname(SUSHI_CACHE), exist_ok=True)
    try:
        urlretrieve(SUSHI_URL, SUSHI_CACHE)
        print(f"  Downloaded to {SUSHI_CACHE}")
        return SUSHI_CACHE
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def load_sushi():
    """Load Sushi dataset: rankings + user attributes.

    Returns:
        X_rankings: (N, 10) array — each row is a ranking (permutation of 0-9)
        y_region: (N,) array — 0=East, 1=West Japan
    """
    zip_path = download_sushi()
    if zip_path is None:
        return None, None

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # List files to find the right ones
        names = zf.namelist()

        # Find order file (rankings) — sushi3a.5000.10.order or similar
        order_file = [n for n in names if 'order' in n.lower() and n.endswith('.order')]
        if not order_file:
            order_file = [n for n in names if '5000.10' in n]
        # Find user data
        udata_file = [n for n in names if 'udata' in n.lower()]

        if not order_file or not udata_file:
            print(f"  Could not find expected files. Available: {names[:20]}")
            return None, None

        # Parse order file
        # Format: each line has variable columns, last 10 are the ranking
        rankings = []
        with zf.open(order_file[0]) as f:
            for line in io.TextIOWrapper(f, encoding='utf-8'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                # The ranking is the last 10 values
                if len(parts) >= 10:
                    ranking = [int(x) for x in parts[-10:]]
                    rankings.append(ranking)

        # Parse user data
        # Format: user_id, gender, age, ...prefecture_id(15 regions), ...
        # Column 4 is prefecture_id (region)
        regions = []
        with zf.open(udata_file[0]) as f:
            for line in io.TextIOWrapper(f, encoding='utf-8'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    # Prefecture ID — binarize: 0-23 = East, 24+ = West
                    # (rough geographic split of Japan's 47 prefectures)
                    pref = int(parts[4])
                    regions.append(0 if pref < 24 else 1)

    n = min(len(rankings), len(regions))
    X = np.array(rankings[:n])
    y = np.array(regions[:n])

    print(f"  Loaded {n} users, {X.shape[1]} sushi types, "
          f"classes: {np.bincount(y)} (East/West)")
    return X, y


def rankings_to_arrowflow_data(X, y, vocab_size=10):
    """Convert ranking matrix to ArrowFlow sorted-list format."""
    data = []
    for i in range(len(X)):
        ordered_list = [str(x) for x in X[i]]
        label = str(int(y[i]))
        data.append([ordered_list, label, 1.0])
    adj_list = [str(x) for x in range(vocab_size)]
    return data, adj_list


def train_baseline(name, X_train, y_train, X_test, y_test):
    """Train a baseline. Returns error rate."""
    label, cls, param_grid = BASELINE_METHODS[name]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train.astype(float))
    X_te_s = scaler.transform(X_test.astype(float))

    model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=8)
    grid.fit(X_tr_s, y_train)
    y_pred = grid.best_estimator_.predict(X_te_s)
    return float(1.0 - accuracy_score(y_test, y_pred))


def run_experiment():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'sushi_ordinal_{timestamp}.csv')

    print("=" * 80)
    print("Natively Ordinal Data: Sushi Preference Dataset")
    print("=" * 80)

    X, y = load_sushi()
    if X is None:
        print("\nSushi dataset unavailable. Skipping experiment.")
        return []

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_classes = len(np.unique(y))

    print(f"\n{len(X_train)} train, {len(X_test)} test, {n_classes} classes")
    print(f"Feature type: rankings (permutations of 0-9)")

    csv_rows = []

    # --- Baselines (rankings as numeric features) ---
    print("\n--- Baselines (rankings as numeric features) ---")
    for bl_name in BASELINE_METHODS:
        t0 = time.time()
        err = train_baseline(bl_name, X_train, y_train, X_test, y_test)
        elapsed = time.time() - t0
        label = BASELINE_METHODS[bl_name][0]
        print(f"  {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
        csv_rows.append({'dataset': 'sushi', 'method': label,
                         'encoding': 'numeric', 'error_pct': round(100 * err, 2)})

    # --- ArrowFlow with native encoding (rankings directly as permutations) ---
    print("\n--- ArrowFlow with native encoding ---")
    data_train, adj_list = rankings_to_arrowflow_data(X_train, y_train)
    data_test, _ = rankings_to_arrowflow_data(X_test, y_test)

    for n_views in [1, 7]:
        for filters_desc, filters, layer_types in [
            ('[64]', [64], ['sort', 'sort']),
            ('[128]', [128], ['sort', 'sort']),
            ('[64,32]', [64, 32], ['sort', 'sort', 'sort']),
        ]:
            af_config = ArrowFlowConfig(
                no_of_filters=filters,
                layer_types=layer_types,
                no_of_iters=300,
                moe_no_of_networks=1,
                no_of_embedding_dim=10,
                encoding_mode='native',
                n_ensemble_views=n_views,
                use_augmentation=True,
                n_augmentations=1,
                max_swaps=2,
                learning_rate=0.1,
                last_layer_update=True,
                verbose=0,
            )

            errors = []
            t0 = time.time()
            for sim in range(N_AF_SIMS):
                seed = 42 + sim * 12345
                err = run_arrowflow_ensemble(
                    X_train.astype(float), y_train, X_test.astype(float), y_test,
                    n_classes, af_config, seed=seed,
                    preencoded_data=(data_train, data_test, adj_list)
                )
                errors.append(err)
            elapsed = time.time() - t0
            mean_err = np.mean(errors)
            std_err = np.std(errors) / np.sqrt(len(errors))
            config_str = f"AF native {filters_desc} {n_views}v"
            print(f"  {config_str:30s}: {100*mean_err:5.1f}% ± {100*std_err:.1f}% [{elapsed:.1f}s]")
            csv_rows.append({'dataset': 'sushi', 'method': config_str,
                             'encoding': 'native', 'error_pct': round(100 * mean_err, 2)})

    # --- ArrowFlow with projection encoding (rankings → standardize → project → argsort) ---
    print("\n--- ArrowFlow with projection encoding ---")
    af_config = ArrowFlowConfig(
        no_of_filters=[128],
        layer_types=['sort', 'sort'],
        no_of_iters=200,
        moe_no_of_networks=1,
        no_of_embedding_dim=16,
        poly_expansion=True,
        pol_deg=2,
        n_ensemble_views=7,
        projection_strategy='diverse',
        use_augmentation=True,
        n_augmentations=1,
        max_swaps=2,
        lda_ratio=0.3,
        learning_rate=0.1,
        last_layer_update=True,
        verbose=0,
    )
    errors = []
    t0 = time.time()
    for sim in range(N_AF_SIMS):
        seed = 42 + sim * 12345
        err = run_arrowflow_ensemble(
            X_train.astype(float), y_train, X_test.astype(float), y_test,
            n_classes, af_config, seed=seed
        )
        errors.append(err)
    elapsed = time.time() - t0
    mean_err = np.mean(errors)
    std_err = np.std(errors) / np.sqrt(len(errors))
    print(f"  {'AF projection 7v':30s}: {100*mean_err:5.1f}% ± {100*std_err:.1f}% [{elapsed:.1f}s]")
    csv_rows.append({'dataset': 'sushi', 'method': 'AF projection 7v',
                     'encoding': 'projection', 'error_pct': round(100 * mean_err, 2)})

    # Write CSV
    csv_fields = ['dataset', 'method', 'encoding', 'error_pct']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nResults saved to: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in csv_rows:
        print(f"  {r['method']:30s} ({r['encoding']:10s}): {r['error_pct']:5.1f}%")

    return csv_rows


if __name__ == '__main__':
    run_experiment()
