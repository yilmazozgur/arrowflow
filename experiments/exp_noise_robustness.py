"""
Noise Robustness Experiment — ArrowFlow vs Classical Baselines
==============================================================
Tests whether ArrowFlow's argsort encoding provides inherent robustness
to feature noise compared to methods operating on raw continuous values.

Hypothesis: argsort is a natural denoiser — small perturbations in feature
space do not change relative orderings, so ArrowFlow's encoded representation
is invariant to small noise. Methods like SVM/MLP/RF see completely different
input values and should degrade faster.

Protocol:
    1. Train ALL methods on clean training data.
    2. At test time, add Gaussian noise: X_test_noisy = X_test + N(0, sigma * std_j)
       where std_j is the per-feature training-set standard deviation.
    3. Evaluate test error at each noise level (sigma), averaged over
       multiple noise realizations.
    4. Output results table + CSV for plotting.

Run:
    python -m test.experiments.exp_noise_robustness
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import csv
import time
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


# ---------------------------------------------------------------------------
# Best ArrowFlow config per dataset (from UCI sweep)
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


# ---------------------------------------------------------------------------
# Baseline configs (same grids as exp_uci_tabular / benchmark_arrowflow)
# ---------------------------------------------------------------------------

BASELINE_METHODS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(64,), (128,), (256,), (64, 64), (128, 64), (256, 128)],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001],
        'max_iter': [1000],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [1, 3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
    }),
}


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

NOISE_SIGMAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
N_NOISE_REALIZATIONS = 5
N_ARROWFLOW_SIMS = 3       # ArrowFlow sims per noise realization
DATASETS = ['iris', 'wine', 'breast_cancer', 'wine_quality',
            'vehicle', 'segment', 'digits']


def train_baselines_on_clean(X_train, y_train):
    """Train all baseline methods on clean data via GridSearchCV.

    Returns dict: method_name -> (fitted_model, fitted_scaler).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    trained = {}
    for name, (label, cls, param_grid) in BASELINE_METHODS.items():
        t0 = time.time()
        if param_grid:
            model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
            grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=8)
            grid.fit(X_train_s, y_train)
            best_model = grid.best_estimator_
        else:
            best_model = cls()
            best_model.fit(X_train_s, y_train)
        elapsed = time.time() - t0
        trained[name] = (best_model, scaler, label)
        print(f"    {label}: trained in {elapsed:.1f}s")
    return trained


def evaluate_baselines_noisy(trained_models, X_test_noisy, y_test):
    """Evaluate pre-trained baseline models on noisy test data.

    Returns dict: method_name -> error_rate.
    """
    results = {}
    for name, (model, scaler, label) in trained_models.items():
        X_test_s = scaler.transform(X_test_noisy)
        y_pred = model.predict(X_test_s)
        error = 1.0 - accuracy_score(y_test, y_pred)
        results[name] = float(error)
    return results


def evaluate_arrowflow_noisy(X_train, y_train, X_test_noisy, y_test,
                              n_classes, cfg, seed=42):
    """Train ArrowFlow on clean data, evaluate on noisy test data.

    ArrowFlow is re-trained each call but training uses only X_train (clean).
    The noisy X_test_noisy goes through the encoding pipeline at prediction.
    """
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
    # Average over multiple ArrowFlow simulations
    errors = []
    for sim in range(N_ARROWFLOW_SIMS):
        sim_seed = seed + sim * 12345
        err = run_arrowflow_ensemble(
            X_train, y_train, X_test_noisy, y_test, n_classes,
            af_config, seed=sim_seed
        )
        errors.append(err)
    return float(np.mean(errors))


def add_noise(X_test, feature_stds, sigma, rng):
    """Add Gaussian noise: X_noisy = X + N(0, sigma * std_j) per feature."""
    noise = rng.randn(*X_test.shape) * (sigma * feature_stds[np.newaxis, :])
    return X_test + noise


def run_noise_experiment():
    """Run the full noise robustness experiment across all datasets."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'noise_robustness_{timestamp}.csv')

    csv_fields = ['dataset', 'method', 'sigma', 'error_mean', 'error_std',
                  'error_stderr', 'n_realizations']
    csv_rows = []

    print("=" * 80)
    print("Noise Robustness Experiment: ArrowFlow vs Classical Baselines")
    print(f"Noise levels (sigma): {NOISE_SIGMAS}")
    print(f"Noise realizations: {N_NOISE_REALIZATIONS}")
    print(f"ArrowFlow sims per realization: {N_ARROWFLOW_SIMS}")
    print("=" * 80)

    for dataset_name in DATASETS:
        cfg = BEST_CONFIGS[dataset_name]
        X_train, y_train, X_test, y_test, n_classes = load_dataset(
            dataset_name, random_state=42
        )

        # Per-feature standard deviation (computed on clean training data)
        feature_stds = np.std(X_train, axis=0)
        # Avoid zero std (constant features) — set floor to small value
        feature_stds = np.maximum(feature_stds, 1e-8)

        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name} — {len(X_train)} train, "
              f"{len(X_test)} test, {n_classes} classes, "
              f"{X_train.shape[1]} features")
        print(f"ArrowFlow config: f={cfg['filters']} e={cfg['embed_dim']} "
              f"pol={cfg['pol_deg']} strat={cfg['projection_strategy']}")
        print(f"{'=' * 70}")

        # --- Train baselines on clean data (once per dataset) ---
        print("\n  Training baselines on clean data...")
        trained_models = train_baselines_on_clean(X_train, y_train)

        # --- Evaluate at each noise level ---
        # Collect: method -> sigma -> list of errors across realizations
        all_methods = list(BASELINE_METHODS.keys()) + ['arrowflow']
        method_labels = {k: v[0] for k, v in BASELINE_METHODS.items()}
        method_labels['arrowflow'] = 'ArrowFlow'

        for sigma in NOISE_SIGMAS:
            print(f"\n  --- sigma = {sigma} ---")
            method_errors = {m: [] for m in all_methods}

            for r in range(N_NOISE_REALIZATIONS):
                rng = np.random.RandomState(42 + r * 777 + int(sigma * 1000))

                if sigma == 0.0:
                    X_test_noisy = X_test.copy()
                else:
                    X_test_noisy = add_noise(X_test, feature_stds, sigma, rng)

                # Baselines
                bl_results = evaluate_baselines_noisy(
                    trained_models, X_test_noisy, y_test
                )
                for name, err in bl_results.items():
                    method_errors[name].append(err)

                # ArrowFlow
                af_seed = 42 + r * 777
                af_err = evaluate_arrowflow_noisy(
                    X_train, y_train, X_test_noisy, y_test,
                    n_classes, cfg, seed=af_seed
                )
                method_errors['arrowflow'].append(af_err)

                # For sigma=0.0, one realization is enough (deterministic noise=0)
                if sigma == 0.0:
                    # Replicate the single result for consistent stats
                    for m in all_methods:
                        method_errors[m] = method_errors[m] * N_NOISE_REALIZATIONS
                    break

            # Print and log results
            for method in all_methods:
                errs = np.array(method_errors[method])
                mean_err = np.mean(errs)
                std_err = np.std(errs)
                stderr = std_err / np.sqrt(len(errs))
                label = method_labels[method]

                print(f"    {label:20s}: {100*mean_err:6.1f}% ± {100*stderr:.1f}%")

                csv_rows.append({
                    'dataset': dataset_name,
                    'method': label,
                    'sigma': sigma,
                    'error_mean': round(100 * mean_err, 2),
                    'error_std': round(100 * std_err, 2),
                    'error_stderr': round(100 * stderr, 2),
                    'n_realizations': len(errs),
                })

    # --- Write CSV ---
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to: {csv_path}")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY: Error (%) at each noise level")
    print("=" * 80)

    for dataset_name in DATASETS:
        ds_rows = [r for r in csv_rows if r['dataset'] == dataset_name]
        methods_in_ds = sorted(set(r['method'] for r in ds_rows),
                               key=lambda m: m != 'ArrowFlow')  # ArrowFlow first
        print(f"\n--- {dataset_name} ---")
        header = f"  {'Method':20s}" + "".join(f"  σ={s:<6}" for s in NOISE_SIGMAS)
        print(header)
        for method in methods_in_ds:
            row_str = f"  {method:20s}"
            for sigma in NOISE_SIGMAS:
                match = [r for r in ds_rows
                         if r['method'] == method and r['sigma'] == sigma]
                if match:
                    row_str += f"  {match[0]['error_mean']:5.1f}%"
                else:
                    row_str += "     — "
            print(row_str)

    return csv_rows


if __name__ == '__main__':
    run_noise_experiment()
