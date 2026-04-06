"""
IMDB Sentiment Classification — ArrowFlow vs Baselines
=======================================================
Tests ArrowFlow with CountVectorizer nonzero encoding against GridSearchCV-
tuned ML baselines and a learned-embedding neural baseline.

Baseline results for N ≤ 5000 are hard-coded from a previous run.

Parallelism: uses mp.get_context('spawn') to avoid fork-based memory
duplication. Each ArrowFlow config runs in its own spawned worker, using
~80% of CPU cores. Data is loaded once per worker (not inherited via fork).

Run:
    python -m test.experiments.exp_imdb_sentiment
"""

import os
import sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import gc
import csv
import copy
import time
import traceback
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import multiprocessing as mp

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_CPUS = os.cpu_count() or 4
MAX_WORKERS = max(1, int(_N_CPUS * 0.8))   # 80% of cores

TRAIN_SIZES = [1000, 5000, 10000, 25000]
N_TEST = 5000
CV_VOCAB = 1000
CV_VECTOR_SIZE = 64

# ---------------------------------------------------------------------------
# Hard-coded baseline results (from previous run, test set = 5000)
# ---------------------------------------------------------------------------

CACHED_BASELINES = {
    500: {
        'lr':  ('Logistic Regression', 19.3, 0.6),
        'rf':  ('Random Forest',       20.9, 4.2),
        'svm': ('SVM-RBF',             19.5, 1.5),
        'mlp': ('MLP',                 21.0, 8.3),
        'knn': ('KNN',                 37.7, 0.3),
        'xgb': ('XGBoost',             25.5, 22.3),
        'emb': ('Embedding+MLP',       28.8, 1.5),
    },
    1000: {
        'lr':  ('Logistic Regression', 17.7, 0.6),
        'rf':  ('Random Forest',       18.7, 6.6),
        'svm': ('SVM-RBF',             17.2, 4.6),
        'mlp': ('MLP',                 19.0, 15.4),
        'knn': ('KNN',                 35.5, 0.5),
        'xgb': ('XGBoost',             21.9, 53.6),
        'emb': ('Embedding+MLP',       25.2, 0.3),
    },
    2000: {
        'lr':  ('Logistic Regression', 16.9, 0.8),
        'rf':  ('Random Forest',       18.3, 16.0),
        'svm': ('SVM-RBF',             16.5, 22.4),
        'mlp': ('MLP',                 17.3, 26.8),
        'knn': ('KNN',                 32.5, 1.0),
        'xgb': ('XGBoost',             19.5, 122.2),
        'emb': ('Embedding+MLP',       20.8, 0.6),
    },
    5000: {
        'lr':  ('Logistic Regression', 14.3, 1.3),
        'rf':  ('Random Forest',       15.7, 87.6),
        'svm': ('SVM-RBF',             14.2, 184.1),
        'mlp': ('MLP',                 17.0, 106.2),
        'knn': ('KNN',                 30.9, 5.3),
        'xgb': ('XGBoost',             17.3, 748.8),
        'emb': ('Embedding+MLP',       17.7, 0.7),
    },
}

# ---------------------------------------------------------------------------
# Baseline method definitions (only used for N > 5000)
# ---------------------------------------------------------------------------

BASELINE_METHODS = {
    'lr': ('Logistic Regression', LogisticRegression, {
        'C': [0.1, 1, 10], 'max_iter': [1000],
    }),
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [200, 500], 'max_depth': [20, None],
        'min_samples_leaf': [1],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(128,), (256,)],
        'learning_rate_init': [0.001], 'max_iter': [500],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [200], 'max_depth': [5, 7],
        'learning_rate': [0.1],
    }),
}

TFIDF_VOCAB = 2000

# ---------------------------------------------------------------------------
# ArrowFlow configs
# ---------------------------------------------------------------------------

AF_CV_CONFIGS = [
    # (description, filters, layer_types, n_views, n_iters_base)
    ('AF [512,64] 3L 1v',        [512, 64],       ['sort']*3,  1, 200),
    ('AF [1024,64] 3L 1v',       [1024, 64],      ['sort']*3,  1, 200),
    ('AF [1024,128] 3L 1v',      [1024, 128],     ['sort']*3,  1, 200),
    ('AF [1024,256] 3L 1v',      [1024, 256],     ['sort']*3,  1, 200),
    ('AF [1024,256,64] 4L 1v',   [1024, 256, 64], ['sort']*4,  1, 200),
]


# ---------------------------------------------------------------------------
# Spawned ArrowFlow worker — self-contained, no inherited parent state
# ---------------------------------------------------------------------------

def _train_arrowflow_worker(args):
    """Spawned worker: loads data from scratch, trains one ArrowFlow config.

    Using spawn context means this process starts clean — no inherited
    pandas DataFrames or other parent memory. Memory usage is only what
    this worker actually needs.

    Returns (desc, error, elapsed).
    """
    desc, filters, layer_types, n_views, n_iters, n_train, seed = args

    # Fresh imports inside spawned worker
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np
    import copy
    import torch as _torch
    _torch.cuda.is_available = lambda: False

    from arrowflow.benchmark import (
        ArrowFlowConfig, _load_imdb_nonzero, _build_sortnet_config,
    )
    from arrowflow.arrowflow import SortFlowHybridNetwork, DataGraph

    t0 = time.time()

    # Load data fresh (no fork inheritance)
    data_train_all, data_test_all, adj_list_input, n_classes, y_train, y_test = \
        _load_imdb_nonzero(vocabulary_size=CV_VOCAB, vector_size=CV_VECTOR_SIZE,
                           subsample=n_train, random_state=42)

    n_test = min(N_TEST, len(data_test_all))
    data_test = data_test_all[:n_test]
    y_test_sub = y_test[:n_test]

    # Train n_views models and majority vote
    all_predictions = []
    for v in range(n_views):
        view_seed = seed + v * 9999
        np.random.seed(view_seed)

        data_train = list(data_train_all)
        data_train = DataGraph.augment_permutation_data(
            data_train, n_augmentations=1, max_swaps=2, seed=view_seed
        )

        config = ArrowFlowConfig(
            no_of_filters=filters,
            layer_types=layer_types,
            no_of_iters=n_iters,
            moe_no_of_networks=1,
            no_of_embedding_dim=CV_VECTOR_SIZE,
            learning_rate=0.1,
            last_layer_update=False,
            verbose=0,
        )
        sf_config = _build_sortnet_config(config, n_classes)
        net = SortFlowHybridNetwork(
            f'sf_v{v}', adj_list_input, n_classes,
            'benchmark_ensemble', sf_config
        )
        net.train([data_train, list(data_test)], sf_config)
        net.graph = copy.deepcopy(net.optimal_model)
        _, preds = net.evaluate(list(data_test), 'supervised', 'classification')
        all_predictions.append(np.array(preds).astype(int))

        # Free memory between views
        del net, data_train
        gc.collect()

    # Majority vote if multiple views
    gt = np.array([int(dp[1]) for dp in data_test])
    if len(all_predictions) == 1:
        final_preds = all_predictions[0]
    else:
        predictions_array = np.stack(all_predictions)
        final_preds = np.array([
            np.bincount(predictions_array[:, i], minlength=n_classes).argmax()
            for i in range(len(gt))
        ])

    err = float(np.sum(final_preds != gt) / len(gt))
    elapsed = time.time() - t0
    return (desc, err, elapsed)


# ---------------------------------------------------------------------------
# Data loading (for baselines only — ArrowFlow workers load their own)
# ---------------------------------------------------------------------------

def load_imdb_raw():
    """Load raw IMDB text data."""
    csv_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'docs', 'IMDB_Dataset.csv'))
    df = pd.read_csv(csv_path)
    df["Category"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"], df['Category'], test_size=0.2, random_state=42
    )
    return X_train_text, X_test_text, y_train.to_numpy(), y_test.to_numpy()


def subsample_text(X_text, y, n, rng):
    """Subsample text data with stratification."""
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    n_per_class = n // 2
    chosen_pos = rng.choice(idx_pos, size=min(n_per_class, len(idx_pos)), replace=False)
    chosen_neg = rng.choice(idx_neg, size=min(n_per_class, len(idx_neg)), replace=False)
    idx = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(idx)
    return X_text.iloc[idx], y[idx], idx


# ---------------------------------------------------------------------------
# Traditional ML baselines (only for uncached sizes)
# ---------------------------------------------------------------------------

def train_baselines(X_train_tfidf, y_train, X_test_tfidf, y_test, n_train):
    """Train baselines with GridSearchCV on TF-IDF features."""
    results = {}
    n_jobs = 2 if n_train <= 10000 else 1
    cv_folds = 2

    for name, (label, cls, param_grid) in BASELINE_METHODS.items():
        t0 = time.time()
        try:
            model = cls(random_state=42) if 'random_state' in cls().get_params() else cls()
            grid = GridSearchCV(model, param_grid, cv=cv_folds,
                                scoring='accuracy', n_jobs=n_jobs)
            grid.fit(X_train_tfidf, y_train)
            y_pred = grid.best_estimator_.predict(X_test_tfidf)
            err = float(1.0 - accuracy_score(y_test, y_pred))
        except Exception as e:
            print(f"      {label} FAILED: {e}")
            err = float('nan')
        elapsed = time.time() - t0
        results[name] = (label, err, elapsed)
        print(f"      {label:25s}: {100*err:5.1f}% [{elapsed:.1f}s]")
    return results


# ---------------------------------------------------------------------------
# Embedding baseline
# ---------------------------------------------------------------------------

class EmbeddingClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = torch.relu(self.fc1(self.dropout(pooled)))
        return self.fc2(self.dropout(h))


def train_embedding_baseline(X_train_text, y_train, X_test_text, y_test,
                              vocab_size=10000, max_len=256, embed_dim=128,
                              hidden_dim=128, epochs=10, batch_size=64, lr=0.001):
    """Train embedding baseline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from collections import Counter
    word_counts = Counter()
    for text in X_train_text:
        word_counts.update(text.lower().split())

    most_common = word_counts.most_common(vocab_size - 2)
    word2idx = {w: i + 2 for i, (w, _) in enumerate(most_common)}

    def tokenize(texts, max_len):
        tokens = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            words = text.lower().split()[:max_len]
            for j, w in enumerate(words):
                tokens[i, j] = word2idx.get(w, 1)
        return tokens

    X_train_tok = tokenize(X_train_text, max_len)
    X_test_tok = tokenize(X_test_text, max_len)

    train_x = torch.tensor(X_train_tok, device=device)
    train_y = torch.tensor(y_train, dtype=torch.long, device=device)
    test_x = torch.tensor(X_test_tok, device=device)

    model = EmbeddingClassifier(vocab_size, embed_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n = len(train_x)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(train_x[idx])
            loss = criterion(logits, train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for start in range(0, len(test_x), batch_size):
            logits = model(test_x[start:start + batch_size])
            preds.append(logits.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)

    return float(1.0 - accuracy_score(y_test, y_pred))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'imdb_sentiment_{timestamp}.csv')

    csv_fields = ['n_train', 'method', 'encoding', 'error_pct',
                  'error_std', 'time_s', 'config']
    csv_rows = []

    print("=" * 80)
    print("IMDB Sentiment Classification: ArrowFlow vs Baselines")
    print(f"Training sizes: {TRAIN_SIZES}")
    print(f"Test set: {N_TEST} samples")
    print(f"Workers: 2 (spawn context, memory-safe)")
    print("=" * 80)
    sys.stdout.flush()

    # Load raw data once (for baselines only)
    X_train_text_all, X_test_text_all, y_train_all, y_test_all = load_imdb_raw()
    test_idx = np.arange(min(N_TEST, len(y_test_all)))
    X_test_text = X_test_text_all.iloc[test_idx]
    y_test = y_test_all[test_idx]

    # Use spawn context — workers start clean, no fork memory inheritance
    ctx = mp.get_context('spawn')

    for n_train in TRAIN_SIZES:
        gc.collect()
        print(f"\n{'=' * 70}")
        print(f"N_train = {n_train}")
        print(f"{'=' * 70}")
        sys.stdout.flush()

        # Subsample training data (for baselines)
        rng = np.random.RandomState(42)
        if n_train < len(y_train_all):
            X_train_text, y_train, train_idx = subsample_text(
                X_train_text_all, y_train_all, n_train, rng)
        else:
            X_train_text = X_train_text_all
            y_train = y_train_all

        # ---- Baselines: use cache or compute ----
        if n_train in CACHED_BASELINES:
            print(f"\n  --- Baselines (cached) ---")
            for key, (label, err_pct, t_s) in CACHED_BASELINES[n_train].items():
                encoding = 'learned-embed' if key == 'emb' else f'tfidf-{TFIDF_VOCAB}'
                csv_rows.append({
                    'n_train': n_train, 'method': label,
                    'encoding': encoding,
                    'error_pct': err_pct, 'error_std': 0,
                    'time_s': t_s, 'config': 'cached',
                })
                print(f"      {label:25s}: {err_pct:5.1f}% (cached)")
        else:
            print(f"\n  Building TF-IDF features (vocab={TFIDF_VOCAB})...")
            tfidf = TfidfVectorizer(max_features=TFIDF_VOCAB)
            X_train_tfidf = tfidf.fit_transform(X_train_text).toarray().astype(float)
            X_test_tfidf = tfidf.transform(X_test_text).toarray().astype(float)

            print(f"\n  --- Traditional ML Baselines ---")
            bl_results = train_baselines(X_train_tfidf, y_train,
                                          X_test_tfidf, y_test, n_train)
            for name, (label, err, elapsed) in bl_results.items():
                csv_rows.append({
                    'n_train': n_train, 'method': label,
                    'encoding': f'tfidf-{TFIDF_VOCAB}',
                    'error_pct': round(100 * err, 2),
                    'error_std': 0, 'time_s': round(elapsed, 1),
                    'config': '',
                })

            print(f"\n  --- Embedding Baseline ---")
            t0 = time.time()
            emb_epochs = 10 if n_train <= 10000 else 5
            emb_err = train_embedding_baseline(
                X_train_text, y_train, X_test_text, y_test,
                vocab_size=10000, max_len=256, embed_dim=128, hidden_dim=128,
                epochs=emb_epochs, batch_size=64, lr=0.001
            )
            elapsed = time.time() - t0
            print(f"      {'Embedding+MLP':25s}: {100*emb_err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_train': n_train, 'method': 'Embedding+MLP',
                'encoding': 'learned-embed',
                'error_pct': round(100 * emb_err, 2),
                'error_std': 0, 'time_s': round(elapsed, 1),
                'config': f'vocab=10000 embed=128 hidden=128 epochs={emb_epochs}',
            })

        # ---- ArrowFlow: run configs in parallel (memory-safe) ----
        # Each spawn worker loads IMDB + builds model. Memory per worker
        # grows with n_train and n_filters. Conservative cap: 2 concurrent.
        n_workers = 2
        print(f"\n  --- ArrowFlow ({n_workers} parallel workers) ---")
        sys.stdout.flush()

        worker_args = []
        for desc, filters, ltypes, nv, niters_base in AF_CV_CONFIGS:
            n_iters = min(500, max(100, niters_base * n_train // 2000))
            worker_args.append((desc, filters, ltypes, nv, n_iters, n_train, 42))

        t0_all = time.time()
        try:
            with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
                results = pool.map(_train_arrowflow_worker, worker_args)
        except Exception as e:
            print(f"      POOL FAILED: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            continue

        for desc, err, elapsed in results:
            print(f"      {desc:30s}: {100*err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_train': n_train, 'method': desc,
                'encoding': f'cv-{CV_VOCAB}-nonzero',
                'error_pct': round(100 * err, 2),
                'error_std': 0,
                'time_s': round(elapsed, 1),
                'config': f'filters= iters= views=',
            })
        print(f"      (all configs: {time.time()-t0_all:.0f}s wall)")
        sys.stdout.flush()

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n\nResults saved to: {csv_path}")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY: Error (%) by Training Size")
    print("=" * 80)

    methods_seen = []
    for r in csv_rows:
        if r['method'] not in methods_seen:
            methods_seen.append(r['method'])

    header = f"  {'Method':30s}" + "".join(f"  N={s:<6}" for s in TRAIN_SIZES)
    print(header)
    print("  " + "-" * (30 + 8 * len(TRAIN_SIZES)))
    for method in methods_seen:
        row_str = f"  {method:30s}"
        for sz in TRAIN_SIZES:
            match = [r for r in csv_rows
                     if r['method'] == method and r['n_train'] == sz]
            if match:
                row_str += f"  {match[0]['error_pct']:5.1f}%"
            else:
                row_str += "     —  "
        print(row_str)

    return csv_rows


if __name__ == '__main__':
    run_experiment()
