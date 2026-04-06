"""
Phase 2.2: IMDB Scaling Experiment
====================================
Extends existing IMDB results to publication quality.
Tests ArrowFlow vs baselines at varying training set sizes and vocabulary sizes.
The key hypothesis: ArrowFlow has a low-data advantage that diminishes at large N.

Run:
    python -m test.experiments.exp_imdb_scaling [--quick]

Results are logged to test/experiments/results/imdb_scaling_*.csv
"""

import os
import sys
import copy
import time
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from arrowflow.arrowflow import SortFlowHybridNetwork, SortFlow_MoE, DataGraph
from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config, log_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_imdb_raw():
    """Load raw IMDB data."""
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'IMDB_Dataset.csv'))
    df = pd.read_csv(csv_path)
    df["Category"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"], df['Category'], test_size=0.2, random_state=42
    )
    return X_train_text, X_test_text, y_train.to_numpy(), y_test.to_numpy()


def prepare_imdb_arrowflow(X_train_text, X_test_text, y_train, y_test,
                           vocabulary_size=1000, embed_dim=32, n_train=None):
    """Prepare IMDB data in ArrowFlow format using TF-IDF + random projection.

    Uses the same project_random_space pipeline as tabular data to produce
    valid permutations (the old BoW index encoding produced non-permutation data).
    """
    tfidf = TfidfVectorizer(max_features=vocabulary_size)
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray().astype(float)
    X_test_tfidf = tfidf.transform(X_test_text).toarray().astype(float)

    # Subsample training data
    if n_train is not None and n_train < len(X_train_tfidf):
        idx = np.random.RandomState(42).choice(len(X_train_tfidf), n_train, replace=False)
        X_train_tfidf = X_train_tfidf[idx]
        y_train = y_train[idx]
        X_test_tfidf = X_test_tfidf[:max(200, n_train // 4)]
        y_test = y_test[:max(200, n_train // 4)]

    # Convert to ArrowFlow sorted list format via random projection
    data_graph = DataGraph('imdb')
    data_train, adj_list_input, W_random = data_graph.project_random_space(
        X_train_tfidf, y_train, None,
        poly_expansion=False, expand_diff_features=False,
        pol_deg=3, no_dimensions=embed_dim
    )
    data_test, _, _ = data_graph.project_random_space(
        X_test_tfidf, y_test, W_random,
        poly_expansion=False, expand_diff_features=False,
        pol_deg=3, no_dimensions=embed_dim
    )

    return data_train, data_test, adj_list_input, X_train_tfidf, X_test_tfidf, y_train, y_test


def prepare_imdb_baselines(X_train_text, X_test_text, y_train, y_test,
                           vocabulary_size=1000, n_train=None):
    """Prepare IMDB data for baseline classifiers."""
    # TF-IDF features (stronger than raw BoW for baselines)
    tfidf = TfidfVectorizer(max_features=vocabulary_size)
    X_train_tfidf = tfidf.fit_transform(X_train_text).toarray()
    X_test_tfidf = tfidf.transform(X_test_text).toarray()

    if n_train is not None and n_train < len(X_train_tfidf):
        idx = np.random.RandomState(42).choice(len(X_train_tfidf), n_train, replace=False)
        X_train_tfidf = X_train_tfidf[idx]
        y_train_sub = y_train[idx]
        X_test_tfidf = X_test_tfidf[:max(200, n_train // 4)]
        y_test_sub = y_test[:max(200, n_train // 4)]
    else:
        y_train_sub = y_train
        y_test_sub = y_test

    return X_train_tfidf, X_test_tfidf, y_train_sub, y_test_sub


def run_arrowflow_imdb(data_train, data_test, adj_list_input, config, n_sims=5):
    """Run ArrowFlow on IMDB data, return test errors."""
    sf_config = _build_sortnet_config(config, 2)

    test_errors = []
    for sim in range(n_sims):
        seed = (sim + 1) * 54321
        np.random.seed(seed)

        if config.moe_no_of_networks > 1:
            net = SortFlow_MoE('sf' + str(sim), adj_list_input, 2, 'imdb_exp', sf_config)
            errors = net.train([data_train, data_test], sf_config)
        else:
            net = SortFlowHybridNetwork('sf' + str(sim), adj_list_input, 2, 'imdb_exp', sf_config)
            errors = net.train([data_train, data_test], sf_config)

        test_errors.append(errors[3])
        print(f"    Sim {sim+1}/{n_sims}: test_error={100*errors[3]:.1f}%")

    return np.array(test_errors)


def run_baselines_imdb(X_train, X_test, y_train, y_test):
    """Run baseline classifiers on IMDB TF-IDF features."""
    results = {}

    # Logistic Regression (strong baseline for text)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results['Logistic Regression'] = 1.0 - accuracy_score(y_test, lr.predict(X_test))

    # Naive Bayes
    # Use GaussianNB since TF-IDF can have negative values after scaling
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    results['Naive Bayes'] = 1.0 - accuracy_score(y_test, nb.predict(X_test))

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                        learning_rate_init=0.01, random_state=42)
    mlp.fit(X_train, y_train)
    results['MLP [64]'] = 1.0 - accuracy_score(y_test, mlp.predict(X_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    results['Random Forest'] = 1.0 - accuracy_score(y_test, rf.predict(X_test))

    return results


# ---------------------------------------------------------------------------
# Main experiment: vocabulary size x training size grid
# ---------------------------------------------------------------------------

def run_scaling_experiment(quick=False):
    """Run the full IMDB scaling experiment."""
    X_train_text, X_test_text, y_train, y_test = load_imdb_raw()

    if quick:
        vocab_sizes = [1000]
        train_sizes = [1000, 5000]
        n_sims = 2
    else:
        vocab_sizes = [500, 1000, 2000]
        train_sizes = [500, 1000, 2000, 5000, 10000, 20000, 40000]
        n_sims = 5

    af_config = ArrowFlowConfig(
        no_of_filters=[128],
        layer_types=['sort', 'sort'],
        no_of_iters=300,  # will be scaled per train_size
        moe_no_of_networks=5,
        batch_size=32,
        learning_rate=0.1,
        no_of_embedding_dim=32,
        pol_deg=3,
        verbose=0,
    )

    all_results = []

    for vocab_size in vocab_sizes:
        for n_train in train_sizes:
            print(f"\n{'='*60}")
            print(f"Vocab={vocab_size}, N_train={n_train}")
            print(f"{'='*60}")

            # Scale iterations with data size
            iters = max(100, min(3000, n_train // 10 * 3))
            af_config_iter = copy.deepcopy(af_config)
            af_config_iter.no_of_iters = iters

            # Prepare ArrowFlow data
            try:
                (data_train, data_test, adj_list_input,
                 X_tr, X_te, y_tr, y_te) = prepare_imdb_arrowflow(
                    X_train_text, X_test_text, y_train, y_test,
                    vocabulary_size=vocab_size, embed_dim=32, n_train=n_train
                )
            except Exception as e:
                print(f"  Data prep failed: {e}")
                continue

            # Run ArrowFlow
            print(f"  ArrowFlow: filters={af_config_iter.no_of_filters}, iters={iters}, moe={af_config_iter.moe_no_of_networks}")
            t0 = time.time()
            try:
                af_errors = run_arrowflow_imdb(data_train, data_test, adj_list_input,
                                               af_config_iter, n_sims=n_sims)
                af_time = time.time() - t0
                af_result = {
                    'timestamp': datetime.now().isoformat(),
                    'dataset': 'imdb',
                    'method': 'ArrowFlow',
                    'vocab_size': vocab_size,
                    'n_train': n_train,
                    'n_test': len(y_te),
                    'n_classes': 2,
                    'n_sims': n_sims,
                    'test_error_mean': float(np.mean(af_errors)),
                    'test_error_std': float(np.std(af_errors)),
                    'test_error_min': float(np.min(af_errors)),
                    'test_error_max': float(np.max(af_errors)),
                    'test_error_stderr': float(np.std(af_errors) / np.sqrt(n_sims)),
                    'time_mean': af_time / n_sims,
                    'config': af_config_iter.to_dict(),
                }
                all_results.append(af_result)
                print(f"  ArrowFlow: {100*np.mean(af_errors):.1f}% +/- {100*np.std(af_errors)/np.sqrt(n_sims):.1f}%")
            except Exception as e:
                print(f"  ArrowFlow FAILED: {e}")

            # Run baselines
            try:
                X_bl_train, X_bl_test, y_bl_train, y_bl_test = prepare_imdb_baselines(
                    X_train_text, X_test_text, y_train, y_test,
                    vocabulary_size=vocab_size, n_train=n_train
                )
                bl_results = run_baselines_imdb(X_bl_train, X_bl_test, y_bl_train, y_bl_test)
                for bl_name, bl_error in bl_results.items():
                    bl_result = {
                        'timestamp': datetime.now().isoformat(),
                        'dataset': 'imdb',
                        'method': bl_name,
                        'vocab_size': vocab_size,
                        'n_train': n_train,
                        'n_test': len(y_bl_test),
                        'n_classes': 2,
                        'n_sims': 1,
                        'test_error_mean': float(bl_error),
                        'test_error_std': 0.0,
                        'test_error_min': float(bl_error),
                        'test_error_max': float(bl_error),
                        'test_error_stderr': 0.0,
                        'time_mean': 0.0,
                        'config': {'vocab_size': vocab_size},
                    }
                    all_results.append(bl_result)
                    print(f"  {bl_name}: {100*bl_error:.1f}%")
            except Exception as e:
                print(f"  Baselines FAILED: {e}")

    # Save results
    filename = os.path.join(RESULTS_DIR,
                            f"imdb_scaling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    import csv
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for r in all_results:
                row = {k: str(v) if isinstance(v, dict) else v for k, v in r.items()}
                writer.writerow(row)
        print(f"\nResults saved to {filename}")

    return all_results


if __name__ == '__main__':
    quick = '--quick' in sys.argv
    run_scaling_experiment(quick=quick)
