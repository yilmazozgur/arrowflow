"""
MNIST via PCA + ArrowFlow — Isolating the Sort-Layer Classifier
================================================================
Tests whether ArrowFlow's sort layers add value as a classifier by giving
both ArrowFlow and baselines the *exact same* PCA-reduced features.

Pipeline:
  MNIST 784px → PCA(n) → argsort → ArrowFlow multi-view ensemble
  MNIST 784px → PCA(n) →           GridSearchCV baselines (RF, SVM, MLP, KNN, XGBoost)

This isolates the sort-layer classifier from the argsort encoding: any
accuracy difference is purely due to ArrowFlow's ordinal learning vs.
conventional classifiers on the same feature representation.

Axes:
  - PCA components: {16, 32, 64}  (controls information available)
  - Training size: {1000, 5000, 10000} (controls sample efficiency)

ArrowFlow configs sweep architecture and view count explicitly.

Run:
    python -m test.experiments.exp_mnist_pca [--quick]
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import gc
import sys
import csv
import time
import warnings
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from arrowflow.benchmark import ArrowFlowConfig, run_arrowflow_ensemble

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PCA_COMPONENTS = [16, 32, 64]
TRAIN_SIZES = [1000, 5000, 10000]
N_AF_SIMS = 3
N_CLASSES = 10

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

BASELINE_METHODS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200, 500], 'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto'],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(128,), (256,), (128, 64)],
        'learning_rate_init': [0.001, 0.01], 'max_iter': [500],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }),
}

# ArrowFlow architecture sweep — views explicit per config
AF_CONFIGS = [
    # (description, filters, layer_types, n_views, learning_rate)
    ('AF [128] 2L 1v',     [128],      ['sort', 'sort'],               1, 0.1),
    ('AF [128] 2L 7v',     [128],      ['sort', 'sort'],               7, 0.1),
    ('AF [256] 2L 7v',     [256],      ['sort', 'sort'],               7, 0.1),
    ('AF [256] 2L 7v lr.2',[256],      ['sort', 'sort'],               7, 0.2),
    ('AF [512] 2L 7v',     [512],      ['sort', 'sort'],               7, 0.1),
    ('AF [128,64] 3L 7v',  [128, 64],  ['sort', 'sort', 'sort'],       7, 0.1),
    ('AF [256,128] 3L 7v', [256, 128], ['sort', 'sort', 'sort'],       7, 0.1),
    ('AF [512,64] 3L 7v', [512, 64], ['sort', 'sort', 'sort'],       7, 0.1),
    ('AF [1024,64] 3L 7v', [1024, 64], ['sort', 'sort', 'sort'],       7, 0.1),
    ('AF [1024,128] 3L 7v', [1024, 128], ['sort', 'sort', 'sort'],       7, 0.1),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mnist():
    """Load MNIST via torchvision."""
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root="./data", train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(
        root="./data", train=False, transform=transform, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 784).astype(float) / 255.0
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy().reshape(-1, 784).astype(float) / 255.0
    y_test = test_ds.targets.numpy()

    print(f"  MNIST: {len(y_train)} train, {len(y_test)} test, {N_CLASSES} classes")
    return X_train, y_train, X_test, y_test


def apply_pca(X_train, X_test, n_components):
    """Standardize and apply PCA with whitening."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    variance = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA({n_components}): explained variance = {100*variance:.1f}%")
    return X_train_pca, X_test_pca


def subsample_stratified(X, y, n, rng):
    """Subsample to n with stratified class balance."""
    classes = np.unique(y)
    per_class = max(2, n // len(classes))
    indices = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
        indices.extend(chosen)
    indices = np.array(indices)
    rng.shuffle(indices)
    return X[indices], y[indices]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def get_baseline_grid(name, n_train):
    """Get parameter grid, with scaling safeguards for large N."""
    label, cls, full_grid = BASELINE_METHODS[name]

    # SVM kernel matrix = N^2 x 8 bytes: skip for N > 10000
    if name == 'svm' and n_train > 10000:
        return label, cls, None
    # KNN: slow on large N x high-dim
    if name == 'knn' and n_train > 10000:
        return label, cls, None

    if n_train <= 5000:
        return label, cls, full_grid

    # Reduced grids for large N
    reduced = {
        'rf': {'n_estimators': [200, 500], 'max_depth': [20, None],
               'min_samples_leaf': [1]},
        'svm': {'C': [1, 10], 'gamma': ['scale']},
        'mlp': {'hidden_layer_sizes': [(128,), (256,)],
                'learning_rate_init': [0.001], 'max_iter': [500]},
        'knn': {'n_neighbors': [5, 7], 'weights': ['distance']},
        'xgb': {'n_estimators': [200], 'max_depth': [5, 7],
                'learning_rate': [0.1]},
    }
    return label, cls, reduced.get(name, full_grid)


def train_baselines(X_train, y_train, X_test, y_test, n_train):
    """Train all baselines with GridSearchCV."""
    results = {}
    cv_folds = 3 if n_train <= 5000 else 2
    n_jobs = 4 if n_train <= 2000 else 2 if n_train <= 10000 else 1

    for name in BASELINE_METHODS:
        label, cls, param_grid = get_baseline_grid(name, n_train)
        if param_grid is None:
            print(f"      {label:20s}: SKIPPED (N={n_train} too large)")
            continue
        t0 = time.time()
        try:
            model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
            grid = GridSearchCV(model, param_grid, cv=cv_folds,
                                scoring='accuracy', n_jobs=n_jobs)
            grid.fit(X_train, y_train)
            y_pred = grid.best_estimator_.predict(X_test)
            err = float(1.0 - accuracy_score(y_test, y_pred))
            del grid
        except Exception as e:
            print(f"      {label} FAILED: {e}")
            err = float('nan')
        elapsed = time.time() - t0
        results[name] = (label, err, elapsed)
        print(f"      {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
        gc.collect()
    return results


# ---------------------------------------------------------------------------
# ArrowFlow
# ---------------------------------------------------------------------------

def run_arrowflow_pca(X_train_pca, y_train, X_test_pca, y_test,
                       filters, layer_types, n_views, n_iters, embed_dim,
                       learning_rate=0.1):
    """Run ArrowFlow with projection encoding on PCA features."""
    config = ArrowFlowConfig(
        no_of_filters=filters,
        layer_types=layer_types,
        no_of_iters=n_iters,
        moe_no_of_networks=1,
        no_of_embedding_dim=embed_dim,
        poly_expansion=False,
        pol_deg=1,
        n_ensemble_views=n_views,
        projection_strategy='diverse',
        use_augmentation=True,
        n_augmentations=1,
        max_swaps=2,
        lda_ratio=0.3,
        learning_rate=learning_rate,
        last_layer_update=False,
        verbose=0,
    )
    errors = []
    for sim in range(N_AF_SIMS):
        seed = 42 + sim * 12345
        err = run_arrowflow_ensemble(
            X_train_pca, y_train, X_test_pca, y_test,
            N_CLASSES, config, seed=seed
        )
        errors.append(err)
    return errors


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(quick=False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(RESULTS_DIR, f'mnist_pca_{timestamp}.csv')

    csv_fields = ['pca', 'n_train', 'method', 'error_pct',
                  'error_std', 'time_s', 'config']
    csv_rows = []

    if quick:
        pca_list = [16, 32]
        train_list = [1000, 5000]
    else:
        pca_list = PCA_COMPONENTS
        train_list = TRAIN_SIZES

    print("=" * 80)
    print("MNIST + PCA: Isolating the Sort-Layer Classifier")
    print(f"PCA components: {pca_list}")
    print(f"Training sizes: {train_list}")
    print("=" * 80)

    # Load MNIST once
    X_train_full, y_train_full, X_test, y_test = load_mnist()

    for n_pca in pca_list:
        print(f"\n{'=' * 70}")
        print(f"PCA = {n_pca}")
        print(f"{'=' * 70}")

        # Apply PCA to full data
        X_train_pca_full, X_test_pca = apply_pca(X_train_full, X_test, n_pca)

        for n_train in train_list:
            actual_n = min(n_train, len(y_train_full))
            print(f"\n  --- N_train = {actual_n}, PCA = {n_pca} ---")

            # Subsample training data
            if actual_n < len(y_train_full):
                rng = np.random.RandomState(42)
                X_tr, y_tr = subsample_stratified(
                    X_train_pca_full, y_train_full, actual_n, rng)
            else:
                X_tr, y_tr = X_train_pca_full, y_train_full

            # ---- Baselines (same PCA features) ----
            print(f"\n    Baselines (on PCA({n_pca}) features):")
            bl_results = train_baselines(X_tr, y_tr, X_test_pca, y_test, actual_n)
            for name, (label, err, elapsed) in bl_results.items():
                csv_rows.append({
                    'pca': n_pca, 'n_train': actual_n, 'method': label,
                    'error_pct': round(100 * err, 2), 'error_std': 0,
                    'time_s': round(elapsed, 1), 'config': '',
                })

            # ---- ArrowFlow (PCA → projection → argsort → ensemble) ----
            print(f"\n    ArrowFlow (PCA({n_pca}) → projection → argsort):")
            # Iterations scale with data: more data → more iters, capped
            n_iters = min(500, max(100, actual_n // 25))
            # Embedding dim: match or exceed PCA dim
            embed_dim = max(n_pca, 32)

            for desc, filters, ltypes, n_views, lr in AF_CONFIGS:
                t0 = time.time()
                errors = run_arrowflow_pca(
                    X_tr, y_tr, X_test_pca, y_test,
                    filters, ltypes, n_views, n_iters, embed_dim,
                    learning_rate=lr,
                )
                elapsed = time.time() - t0
                mean_err = np.mean(errors)
                std_err = np.std(errors) / np.sqrt(len(errors))
                print(f"      {desc:28s}: {100*mean_err:5.1f}% ± {100*std_err:.1f}% "
                      f"[{elapsed:.1f}s] (iters={n_iters})")
                csv_rows.append({
                    'pca': n_pca, 'n_train': actual_n, 'method': desc,
                    'error_pct': round(100 * mean_err, 2),
                    'error_std': round(100 * std_err, 2),
                    'time_s': round(elapsed, 1),
                    'config': f'filters={filters} iters={n_iters} views={n_views} '
                              f'embed={embed_dim} lr={lr}',
                })
                gc.collect()

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to: {csv_path}")

    # ---- Summary per PCA dimension ----
    for n_pca in pca_list:
        print(f"\n{'=' * 80}")
        print(f"SUMMARY: PCA = {n_pca} — Error (%)")
        print("=" * 80)

        pca_rows = [r for r in csv_rows if r['pca'] == n_pca]
        methods_seen = []
        for r in pca_rows:
            if r['method'] not in methods_seen:
                methods_seen.append(r['method'])

        sizes_in_data = sorted(set(r['n_train'] for r in pca_rows))
        header = f"  {'Method':25s}" + "".join(f"  N={s:<6}" for s in sizes_in_data)
        print(header)
        print("  " + "-" * (25 + 8 * len(sizes_in_data)))
        for method in methods_seen:
            row_str = f"  {method:25s}"
            for sz in sizes_in_data:
                match = [r for r in pca_rows
                         if r['method'] == method and r['n_train'] == sz]
                if match:
                    row_str += f"  {match[0]['error_pct']:5.1f}%"
                else:
                    row_str += "     —  "
            print(row_str)

    return csv_rows


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    run_experiment(quick=quick)
