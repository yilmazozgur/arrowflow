"""
Gene Expression Cancer Classification — ArrowFlow vs Baselines
===============================================================
Tests ArrowFlow on rank-transformed gene expression data (TCGA PANCAN).

Motivation: Gene expression absolute values vary wildly across experiments
and batches, but the *relative ordering* of gene expression levels is
conserved. This is the principle behind Top Scoring Pairs (TSP), an
established rank-based method in bioinformatics. ArrowFlow's ordinal
processing is natively batch-effect invariant.

Dataset: UCI Gene Expression Cancer RNA-Seq (TCGA PANCAN)
    - 801 samples, 20531 genes, 5 cancer types
    - Classes: BRCA, KIRC, COAD, LUAD, PRAD

Pipeline:
    1. Select top-K most discriminative genes (mutual information)
    2. Option A: rank-transform each sample → native permutation (ArrowFlow native)
    3. Option B: use raw values with standard pipeline (ArrowFlow projection)
    4. Compare against baselines on both raw and rank-transformed data

Run:
    python -m test.experiments.exp_gene_expression
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
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from arrowflow.benchmark import ArrowFlowConfig, run_arrowflow_ensemble

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), 'results')
GENE_PANELS = [10, 15, 20]   # Number of top genes to select
N_AF_SIMS = 5

BASELINE_METHODS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200, 500], 'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(64,), (128,), (64, 64)],
        'learning_rate_init': [0.001, 0.01], 'max_iter': [1000],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [1, 3, 5, 7, 11], 'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200], 'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
    }),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tcga():
    """Load TCGA PANCAN gene expression dataset.

    Tries ucimlrepo first, then falls back to manual CSV download.

    Returns:
        X: (801, 20531) gene expression values
        y: (801,) integer class labels
        label_names: list of cancer type names
    """
    # Try ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        print("  Loading via ucimlrepo...")
        data = fetch_ucirepo(id=401)
        X = data.data.features.values.astype(float)
        y_raw = data.data.targets.values.ravel()
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} genes, "
              f"{len(le.classes_)} classes: {list(le.classes_)}")
        return X, y, list(le.classes_)
    except Exception as e:
        print(f"  ucimlrepo failed ({e}), trying manual download...")

    # Manual download from UCI
    try:
        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/"
        data_file = os.path.join(DATA_DIR, 'TCGA-PANCAN-HiSeq-801x20531')

        if not os.path.exists(data_file + '.data.csv'):
            from urllib.request import urlretrieve
            os.makedirs(DATA_DIR, exist_ok=True)
            print("  Downloading TCGA data (this may take a minute)...")
            urlretrieve(base_url + "TCGA-PANCAN-HiSeq-801x20531.tar.gz",
                        data_file + ".tar.gz")
            import tarfile
            with tarfile.open(data_file + ".tar.gz", "r:gz") as tar:
                tar.extractall(DATA_DIR)
            print("  Extracted.")

        # Read the extracted files
        X_df = pd.read_csv(os.path.join(DATA_DIR,
                           'TCGA-PANCAN-HiSeq-801x20531', 'data.csv'),
                           index_col=0)
        y_df = pd.read_csv(os.path.join(DATA_DIR,
                           'TCGA-PANCAN-HiSeq-801x20531', 'labels.csv'),
                           index_col=0)

        X = X_df.values.astype(float)
        le = LabelEncoder()
        y = le.fit_transform(y_df.values.ravel())
        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} genes, "
              f"{len(le.classes_)} classes: {list(le.classes_)}")
        return X, y, list(le.classes_)

    except Exception as e:
        print(f"  Manual download also failed: {e}")
        print("  Please install ucimlrepo: pip install ucimlrepo")
        return None, None, None


def select_top_genes(X_train, y_train, n_genes):
    """Select top-n genes by mutual information with the target."""
    print(f"  Selecting top {n_genes} genes by mutual information...")
    mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=5)
    top_idx = np.argsort(mi)[-n_genes:][::-1]
    return top_idx


def rank_transform(X):
    """Convert each sample to a permutation by ranking its features.

    For a sample [3.1, 0.5, 7.2], the rank transform gives [1, 0, 2]
    (argsort of argsort = rank). This is the native input for ArrowFlow.
    """
    return np.apply_along_axis(
        lambda x: np.argsort(np.argsort(-x)), axis=1, arr=X
    )


def rankings_to_arrowflow_data(X_ranked, y, vocab_size):
    """Convert rank-transformed matrix to ArrowFlow sorted-list format."""
    # Each row of X_ranked is a permutation: the rank of each gene
    # We need to convert to the ordered list: which gene is rank 0, rank 1, ...
    data = []
    for i in range(len(X_ranked)):
        # argsort of ranks gives the ordering
        ordered_list = [str(x) for x in np.argsort(X_ranked[i])]
        label = str(int(y[i]))
        data.append([ordered_list, label, 1.0])
    adj_list = [str(x) for x in range(vocab_size)]
    return data, adj_list


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def train_baseline(name, X_train, y_train, X_test, y_test):
    """Train a GridSearchCV baseline. Returns (error_rate, elapsed_time)."""
    label, cls, param_grid = BASELINE_METHODS[name]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
    grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=8)
    t0 = time.time()
    grid.fit(X_tr_s, y_train)
    y_pred = grid.best_estimator_.predict(X_te_s)
    elapsed = time.time() - t0
    err = 1.0 - accuracy_score(y_test, y_pred)
    return err, elapsed


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'gene_expression_{timestamp}.csv')

    print("=" * 80)
    print("Gene Expression Cancer Classification (TCGA PANCAN)")
    print("ArrowFlow rank-transform vs baselines")
    print("=" * 80)

    X_all, y_all, label_names = load_tcga()
    if X_all is None:
        print("\nDataset unavailable. Exiting.")
        return []

    # Train/test split (stratified)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    n_classes = len(np.unique(y_all))

    print(f"\n{len(X_train_full)} train, {len(X_test_full)} test, "
          f"{n_classes} classes: {label_names}")

    csv_rows = []

    for n_genes in GENE_PANELS:
        print(f"\n{'='*70}")
        print(f"Gene panel size: {n_genes}")
        print(f"{'='*70}")

        # Select top genes on training data only
        top_idx = select_top_genes(X_train_full, y_train, n_genes)
        X_train = X_train_full[:, top_idx]
        X_test = X_test_full[:, top_idx]

        # Also create rank-transformed versions
        X_train_ranked = rank_transform(X_train)
        X_test_ranked = rank_transform(X_test)

        # === Baselines on RAW values ===
        print(f"\n  --- Baselines on raw gene expression (top {n_genes} genes) ---")
        for bl_name in BASELINE_METHODS:
            err, elapsed = train_baseline(bl_name, X_train, y_train, X_test, y_test)
            label = BASELINE_METHODS[bl_name][0]
            print(f"    {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_genes': n_genes, 'method': label, 'encoding': 'raw',
                'error_pct': round(100 * err, 2), 'time_s': round(elapsed, 1),
            })

        # === Baselines on RANK-TRANSFORMED values ===
        print(f"\n  --- Baselines on rank-transformed expression ---")
        for bl_name in BASELINE_METHODS:
            err, elapsed = train_baseline(bl_name,
                                          X_train_ranked.astype(float), y_train,
                                          X_test_ranked.astype(float), y_test)
            label = BASELINE_METHODS[bl_name][0]
            print(f"    {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_genes': n_genes, 'method': label, 'encoding': 'ranked',
                'error_pct': round(100 * err, 2), 'time_s': round(elapsed, 1),
            })

        # === ArrowFlow NATIVE (rank-transform → permutation) ===
        print(f"\n  --- ArrowFlow native (rank-transform → permutation) ---")
        data_train, adj_list = rankings_to_arrowflow_data(
            X_train_ranked, y_train, n_genes)
        data_test, _ = rankings_to_arrowflow_data(
            X_test_ranked, y_test, n_genes)

        af_native_configs = [
            ('[64] 1v',   [64],      ['sort', 'sort'], 1),
            ('[64] 7v',   [64],      ['sort', 'sort'], 7),
            ('[128] 1v',  [128],     ['sort', 'sort'], 1),
            ('[128] 7v',  [128],     ['sort', 'sort'], 7),
            ('[64,32] 7v', [64, 32], ['sort', 'sort', 'sort'], 7),
        ]

        for desc, filters, ltypes, n_views in af_native_configs:
            af_config = ArrowFlowConfig(
                no_of_filters=filters,
                layer_types=ltypes,
                no_of_iters=300,
                moe_no_of_networks=1,
                no_of_embedding_dim=n_genes,
                encoding_mode='native',
                n_ensemble_views=n_views,
                use_augmentation=True,
                n_augmentations=1,
                max_swaps=2,
                learning_rate=0.1,
                last_layer_update=False,
                verbose=0,
            )
            errors = []
            t0 = time.time()
            for sim in range(N_AF_SIMS):
                seed = 42 + sim * 12345
                err = run_arrowflow_ensemble(
                    X_train_ranked.astype(float), y_train,
                    X_test_ranked.astype(float), y_test,
                    n_classes, af_config, seed=seed,
                    preencoded_data=(data_train, data_test, adj_list),
                )
                errors.append(err)
            elapsed = time.time() - t0
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            config_str = f"AF native {desc}"
            print(f"    {config_str:30s}: {100*mean_err:5.1f}% "
                  f"± {100*std_err:.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_genes': n_genes, 'method': config_str,
                'encoding': 'native',
                'error_pct': round(100 * mean_err, 2),
                'time_s': round(elapsed, 1),
            })

        # === ArrowFlow PROJECTION (raw values → poly → project → argsort) ===
        print(f"\n  --- ArrowFlow projection (raw → poly → project → argsort) ---")
        af_proj_configs = [
            ('[128] p1 7v', [128], 1, 7),
            ('[128] p2 7v', [128], 2, 7),
        ]

        for desc, filters, pdeg, n_views in af_proj_configs:
            af_config = ArrowFlowConfig(
                no_of_filters=filters,
                layer_types=['sort', 'sort'],
                no_of_iters=200,
                moe_no_of_networks=1,
                no_of_embedding_dim=max(n_genes, 16),
                encoding_mode='projection',
                poly_expansion=(pdeg > 1),
                pol_deg=pdeg,
                n_ensemble_views=n_views,
                projection_strategy='diverse',
                use_augmentation=True,
                n_augmentations=1,
                max_swaps=2,
                lda_ratio=0.3,
                learning_rate=0.1,
                last_layer_update=False,
                verbose=0,
            )
            errors = []
            t0 = time.time()
            for sim in range(N_AF_SIMS):
                seed = 42 + sim * 12345
                err = run_arrowflow_ensemble(
                    X_train, y_train, X_test, y_test,
                    n_classes, af_config, seed=seed,
                )
                errors.append(err)
            elapsed = time.time() - t0
            mean_err = np.mean(errors)
            std_err = np.std(errors)
            config_str = f"AF proj {desc}"
            print(f"    {config_str:30s}: {100*mean_err:5.1f}% "
                  f"± {100*std_err:.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_genes': n_genes, 'method': config_str,
                'encoding': 'projection',
                'error_pct': round(100 * mean_err, 2),
                'time_s': round(elapsed, 1),
            })

    # === Simulated batch effect experiments ===
    # Two types of batch effects, reflecting real-world gene expression variation:
    #
    # Type 1: MONOTONE PER-SAMPLE transforms (global scaling, log, power).
    #   All genes in a sample are transformed identically.
    #   Within-sample ranking is PERFECTLY preserved → ArrowFlow is immune.
    #   Baselines on raw values are NOT immune.
    #
    # Type 2: PER-GENE multiplicative shifts (ComBat model).
    #   Each gene gets a different scale factor across batches (probe affinity,
    #   amplification efficiency). This CAN change within-sample rankings.
    #   Modeled as log-normal: scale_i = exp(N(0, σ)), with realistic σ.
    #   σ=0.3 → 95% of factors in [0.55, 1.82] (mild, routine batch effect)
    #   σ=0.7 → 95% of factors in [0.25, 4.06] (moderate, cross-platform)
    #   σ=1.2 → 95% of factors in [0.09, 11.0] (severe, different technology)

    n_genes_batch = 15
    top_idx = select_top_genes(X_train_full, y_train, n_genes_batch)
    X_train_sel = X_train_full[:, top_idx]
    X_test_sel = X_test_full[:, top_idx]

    # Train models ONCE on clean data
    scaler_raw = StandardScaler()
    X_tr_raw = scaler_raw.fit_transform(X_train_sel)

    X_train_ranked_sel = rank_transform(X_train_sel)
    scaler_ranked = StandardScaler()
    X_tr_ranked_s = scaler_ranked.fit_transform(X_train_ranked_sel.astype(float))

    svm_raw = SVC(C=10, gamma='scale', random_state=42)
    svm_raw.fit(X_tr_raw, y_train)
    rf_raw = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_raw.fit(X_tr_raw, y_train)
    svm_ranked = SVC(C=10, gamma='scale', random_state=42)
    svm_ranked.fit(X_tr_ranked_s, y_train)
    rf_ranked = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_ranked.fit(X_tr_ranked_s, y_train)

    # Pre-train ArrowFlow on clean ranked data
    data_train_b, adj_list_b = rankings_to_arrowflow_data(
        X_train_ranked_sel, y_train, n_genes_batch)

    # ---- Type 1: Monotone per-sample transforms ----
    print(f"\n{'='*70}")
    print("Batch Effect Type 1: Monotone Per-Sample Transforms")
    print("(Same transform applied to ALL genes in a sample)")
    print("ArrowFlow ranking is perfectly invariant to these.")
    print(f"{'='*70}")

    monotone_transforms = [
        ("none",          lambda X: X),
        ("log(1+x)",      lambda X: np.log1p(np.abs(X)) * np.sign(X)),
        ("sqrt(|x|)",     lambda X: np.sqrt(np.abs(X)) * np.sign(X)),
        ("x^2 (signed)",  lambda X: X * np.abs(X)),
        ("×0.01 global",  lambda X: X * 0.01),
        ("×100 global",   lambda X: X * 100),
    ]

    for tf_name, tf_func in monotone_transforms:
        X_test_tf = tf_func(X_test_sel)
        X_test_ranked_tf = rank_transform(X_test_tf)

        # Raw baselines
        X_te_raw = scaler_raw.transform(X_test_tf)
        svm_r_err = 1.0 - accuracy_score(y_test, svm_raw.predict(X_te_raw))
        rf_r_err = 1.0 - accuracy_score(y_test, rf_raw.predict(X_te_raw))

        # Rank-based SVM (rank-transform the transformed data, then classify)
        X_te_ranked_s = scaler_ranked.transform(X_test_ranked_tf.astype(float))
        svm_rk_err = 1.0 - accuracy_score(y_test, svm_ranked.predict(X_te_ranked_s))

        # ArrowFlow native
        data_test_b, _ = rankings_to_arrowflow_data(
            X_test_ranked_tf, y_test, n_genes_batch)
        af_config = ArrowFlowConfig(
            no_of_filters=[128], layer_types=['sort', 'sort'],
            no_of_iters=300, moe_no_of_networks=1,
            no_of_embedding_dim=n_genes_batch, encoding_mode='native',
            n_ensemble_views=7, use_augmentation=True, n_augmentations=1,
            max_swaps=2, learning_rate=0.1, last_layer_update=False, verbose=0,
        )
        af_errors = []
        for sim in range(N_AF_SIMS):
            seed = 42 + sim * 12345
            err = run_arrowflow_ensemble(
                X_train_ranked_sel.astype(float), y_train,
                X_test_ranked_tf.astype(float), y_test,
                n_classes, af_config, seed=seed,
                preencoded_data=(data_train_b, data_test_b, adj_list_b),
            )
            af_errors.append(err)
        af_err = np.mean(af_errors)

        print(f"\n  Transform: {tf_name}")
        print(f"    SVM (raw):    {100*svm_r_err:5.1f}%   "
              f"RF (raw):    {100*rf_r_err:5.1f}%   "
              f"SVM (ranked): {100*svm_rk_err:5.1f}%   "
              f"AF native:    {100*af_err:5.1f}%")

        for method, err_val in [('SVM-raw', svm_r_err), ('RF-raw', rf_r_err),
                                ('SVM-ranked', svm_rk_err), ('AF-native', af_err)]:
            csv_rows.append({
                'n_genes': n_genes_batch,
                'method': f'{method} (mono: {tf_name})',
                'encoding': 'batch_monotone',
                'error_pct': round(100 * err_val, 2), 'time_s': 0,
            })

    # ---- Type 2: Per-gene multiplicative shifts (ComBat model) ----
    print(f"\n{'='*70}")
    print("Batch Effect Type 2: Per-Gene Log-Normal Scaling (ComBat model)")
    print("scale_i = exp(N(0, σ)) per gene. Changes within-sample rankings")
    print("when scaling overwhelms the gap between genes (Theorem 4).")
    print(f"{'='*70}")

    rng = np.random.RandomState(42)
    batch_sigmas = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]

    for sigma in batch_sigmas:
        if sigma == 0.0:
            X_test_batch = X_test_sel.copy()
            sigma_label = "σ=0 (clean)"
        else:
            # Log-normal per-gene scaling: realistic batch effect model
            log_scales = rng.normal(0, sigma, size=n_genes_batch)
            scale_factors = np.exp(log_scales)
            X_test_batch = X_test_sel * scale_factors[np.newaxis, :]
            factor_range = (np.min(scale_factors), np.max(scale_factors))
            sigma_label = f"σ={sigma} [{factor_range[0]:.2f}-{factor_range[1]:.2f}×]"

        X_test_ranked_batch = rank_transform(X_test_batch)

        # Measure how many within-sample rankings changed
        X_test_ranked_clean = rank_transform(X_test_sel)
        rank_agreement = np.mean(X_test_ranked_batch == X_test_ranked_clean)

        # Raw baselines
        X_te_raw = scaler_raw.transform(X_test_batch)
        svm_r_err = 1.0 - accuracy_score(y_test, svm_raw.predict(X_te_raw))
        rf_r_err = 1.0 - accuracy_score(y_test, rf_raw.predict(X_te_raw))

        # Rank-based baselines
        X_te_ranked_s = scaler_ranked.transform(X_test_ranked_batch.astype(float))
        svm_rk_err = 1.0 - accuracy_score(y_test, svm_ranked.predict(X_te_ranked_s))
        rf_rk_err = 1.0 - accuracy_score(y_test, rf_ranked.predict(X_te_ranked_s))

        # ArrowFlow native
        data_test_b, _ = rankings_to_arrowflow_data(
            X_test_ranked_batch, y_test, n_genes_batch)
        af_config = ArrowFlowConfig(
            no_of_filters=[128], layer_types=['sort', 'sort'],
            no_of_iters=300, moe_no_of_networks=1,
            no_of_embedding_dim=n_genes_batch, encoding_mode='native',
            n_ensemble_views=7, use_augmentation=True, n_augmentations=1,
            max_swaps=2, learning_rate=0.1, last_layer_update=False, verbose=0,
        )
        af_errors = []
        for sim in range(N_AF_SIMS):
            seed = 42 + sim * 12345
            err = run_arrowflow_ensemble(
                X_train_ranked_sel.astype(float), y_train,
                X_test_ranked_batch.astype(float), y_test,
                n_classes, af_config, seed=seed,
                preencoded_data=(data_train_b, data_test_b, adj_list_b),
            )
            af_errors.append(err)
        af_err = np.mean(af_errors)

        print(f"\n  {sigma_label}  (rank agreement: {100*rank_agreement:.1f}%)")
        print(f"    SVM (raw):    {100*svm_r_err:5.1f}%   "
              f"RF (raw):    {100*rf_r_err:5.1f}%   "
              f"SVM (ranked): {100*svm_rk_err:5.1f}%   "
              f"RF (ranked):  {100*rf_rk_err:5.1f}%   "
              f"AF native:    {100*af_err:5.1f}%")

        for method, err_val in [('SVM-raw', svm_r_err), ('RF-raw', rf_r_err),
                                ('SVM-ranked', svm_rk_err), ('RF-ranked', rf_rk_err),
                                ('AF-native', af_err)]:
            csv_rows.append({
                'n_genes': n_genes_batch,
                'method': f'{method} (gene σ={sigma})',
                'encoding': 'batch_pergene',
                'error_pct': round(100 * err_val, 2), 'time_s': 0,
            })

    # Write CSV
    csv_fields = ['n_genes', 'method', 'encoding', 'error_pct', 'time_s']
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
        genes = r.get('n_genes', '?')
        print(f"  [{genes:>2} genes] {r['method']:35s} ({r['encoding']:10s}): "
              f"{r['error_pct']:5.1f}%")

    return csv_rows


if __name__ == '__main__':
    run_experiment()
