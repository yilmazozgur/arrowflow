"""
Glasgow City Council 2007 STV Elections — ArrowFlow vs Baselines
=================================================================
Tests ArrowFlow on natively ordinal data: real ranked ballots from the
2007 Glasgow City Council elections (Single Transferable Vote system).

Task: Classify which ward a ballot came from, based on the voter's
ranking of candidates. Different wards have different political profiles,
so the whole ranking pattern encodes community identity.

Dataset: PrefLib 00008 — Glasgow City Council 2007
    - 21 wards, 8-13 candidates per ward
    - 5,199-12,744 ballots per ward
    - STV partial rankings (voters rank as many candidates as they wish)

Strategy: Group wards by candidate count to keep permutation length fixed.
The largest group is 10-candidate wards (typically 5-8 wards, ~40K+ ballots).
For wards with partial rankings, unranked candidates are placed at the end
in a canonical order.

Run:
    python -m test.experiments.exp_glasgow_elections
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import csv
import io
import re
import time
import zipfile
import warnings
import numpy as np
from datetime import datetime
from collections import defaultdict
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
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

GLASGOW_URL = "https://github.com/PrefLib/PrefLib-Data/releases/download/v1.0/00008_glasgow.zip"
GLASGOW_CACHE = os.path.join(os.path.dirname(__file__), 'results', 'glasgow.zip')
N_AF_SIMS = 5
MAX_SAMPLES_PER_WARD = 5000  # Cap per ward to balance classes

BASELINE_METHODS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200], 'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 5],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(64,), (128,), (64, 64)],
        'learning_rate_init': [0.001, 0.01], 'max_iter': [500],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [1, 3, 5, 7, 11], 'weights': ['uniform', 'distance'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200], 'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }),
}


# ---------------------------------------------------------------------------
# Data loading — PrefLib SOI/TOC parser
# ---------------------------------------------------------------------------

def download_glasgow():
    """Download and cache the Glasgow dataset."""
    if os.path.exists(GLASGOW_CACHE):
        print(f"  Using cached Glasgow data: {GLASGOW_CACHE}")
        return GLASGOW_CACHE

    os.makedirs(os.path.dirname(GLASGOW_CACHE), exist_ok=True)
    print(f"  Downloading Glasgow dataset from PrefLib...")
    try:
        urlretrieve(GLASGOW_URL, GLASGOW_CACHE)
        print(f"  Downloaded to {GLASGOW_CACHE}")
        return GLASGOW_CACHE
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def parse_preflib_soi(text):
    """Parse a PrefLib SOI (Strict Orderings — Incomplete) file.

    Returns:
        n_alternatives: int
        ballots: list of (count, ranking) where ranking is a list of ints
    """
    lines = text.strip().split('\n')
    # First line: number of alternatives
    n_alternatives = int(lines[0].strip())

    # Next n_alternatives lines: candidate names (id, name)
    # Then a summary line: total_votes, unique_ballots, ...
    # Then ballot lines: count, c1, c2, c3, ...

    # Find the summary line (after candidate names)
    candidate_lines = []
    ballot_start = 1
    for i in range(1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        # Candidate lines have format: "id, name" or "id,name"
        parts = line.split(',')
        # If first part is small int and second is text, it's a candidate
        try:
            cid = int(parts[0].strip())
            if cid <= n_alternatives and len(parts) >= 2:
                candidate_lines.append(line)
                ballot_start = i + 1
                continue
        except ValueError:
            pass
        # If it looks like summary (3 comma-separated ints), skip it
        try:
            nums = [int(x.strip()) for x in parts]
            if len(nums) <= 4 and nums[0] > n_alternatives:
                ballot_start = i + 1
                continue
        except ValueError:
            pass
        break

    ballot_start = 1 + n_alternatives + 1  # candidates + summary line

    ballots = []
    for i in range(ballot_start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split(',')
        try:
            nums = [int(x.strip()) for x in parts]
            count = nums[0]
            ranking = nums[1:]
            if len(ranking) > 0:
                ballots.append((count, ranking))
        except ValueError:
            continue

    return n_alternatives, ballots


def complete_partial_ranking(ranking, n_alternatives):
    """Complete a partial ranking by appending unranked candidates at the end.

    Unranked candidates are added in canonical (sorted) order.
    """
    ranked_set = set(ranking)
    unranked = sorted([c for c in range(1, n_alternatives + 1)
                       if c not in ranked_set])
    return list(ranking) + unranked


def load_glasgow():
    """Load Glasgow STV election data, grouped by number of candidates.

    Returns:
        ward_groups: dict mapping n_candidates → list of
            (ward_name, n_alternatives, X_rankings, y_ward_label)
    """
    zip_path = download_glasgow()
    if zip_path is None:
        return None

    ward_groups = defaultdict(list)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find all .soi or .toc files
        soi_files = sorted([n for n in zf.namelist()
                            if n.endswith('.soi') or n.endswith('.toc')])
        # Prefer .soi over .toc for the same ward
        # Extract ward name from filename
        seen_wards = set()
        files_to_process = []
        for fn in soi_files:
            # Extract a ward identifier from the filename
            base = os.path.basename(fn)
            ward_id = base.split('.')[0]  # e.g., "00008-00000001"
            if ward_id not in seen_wards:
                seen_wards.add(ward_id)
                files_to_process.append(fn)

        print(f"  Found {len(files_to_process)} ward files")

        for ward_idx, fn in enumerate(files_to_process):
            with zf.open(fn) as f:
                text = f.read().decode('utf-8', errors='replace')

            n_alt, ballots = parse_preflib_soi(text)
            if n_alt < 3 or len(ballots) < 50:
                continue

            # Expand ballot counts and complete partial rankings
            all_rankings = []
            for count, ranking in ballots:
                completed = complete_partial_ranking(ranking, n_alt)
                # Convert to 0-indexed
                completed_0 = [c - 1 for c in completed]
                for _ in range(count):
                    all_rankings.append(completed_0)

            if len(all_rankings) < 100:
                continue

            ward_name = os.path.basename(fn).replace('.soi', '').replace('.toc', '')
            X = np.array(all_rankings)
            ward_groups[n_alt].append((ward_name, n_alt, X))
            print(f"    Ward {ward_name}: {n_alt} candidates, "
                  f"{len(all_rankings)} ballots")

    return ward_groups


def prepare_classification_data(ward_list, max_per_ward):
    """Merge wards with the same candidate count into one classification dataset.

    Args:
        ward_list: list of (ward_name, n_alt, X_rankings)
        max_per_ward: max samples per ward for class balance

    Returns:
        X, y, ward_names
    """
    X_parts, y_parts, names = [], [], []
    for label_idx, (ward_name, n_alt, X_ward) in enumerate(ward_list):
        n = min(len(X_ward), max_per_ward)
        # Random subsample for balance
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_ward), size=n, replace=False)
        X_parts.append(X_ward[idx])
        y_parts.append(np.full(n, label_idx))
        names.append(ward_name)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    return X[perm], y[perm], names


def rankings_to_arrowflow_data(X, y, vocab_size):
    """Convert ranking matrix to ArrowFlow sorted-list format.

    Each row of X is already a permutation (ordering of candidates).
    """
    data = []
    for i in range(len(X)):
        ordered_list = [str(x) for x in X[i]]
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
    X_tr_s = scaler.fit_transform(X_train.astype(float))
    X_te_s = scaler.transform(X_test.astype(float))

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
    csv_path = os.path.join(results_dir, f'glasgow_elections_{timestamp}.csv')

    print("=" * 80)
    print("Glasgow City Council 2007 STV Elections")
    print("ArrowFlow native ordinal classification vs baselines")
    print("=" * 80)

    ward_groups = load_glasgow()
    if ward_groups is None:
        print("\nDataset unavailable. Exiting.")
        return []

    csv_rows = []

    # Process each group of wards with the same candidate count
    # Focus on groups with >= 3 wards (enough for a meaningful classification task)
    for n_cand in sorted(ward_groups.keys()):
        wards = ward_groups[n_cand]
        if len(wards) < 2:
            print(f"\n  Skipping {n_cand}-candidate wards "
                  f"(only {len(wards)} ward, need >= 2)")
            continue

        print(f"\n{'='*70}")
        print(f"{n_cand}-candidate wards: {len(wards)} wards")
        for wn, na, X in wards:
            print(f"  {wn}: {len(X)} ballots")
        print(f"{'='*70}")

        X, y, ward_names = prepare_classification_data(wards, MAX_SAMPLES_PER_WARD)
        n_classes = len(np.unique(y))
        total_samples = len(X)

        print(f"\nClassification: {n_classes} classes, {total_samples} samples, "
              f"permutation length {n_cand}")
        print(f"Ward names: {ward_names}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"{len(X_train)} train, {len(X_test)} test")

        # === Baselines (rankings as numeric features) ===
        print(f"\n  --- Baselines (rankings as numeric features) ---")
        for bl_name in BASELINE_METHODS:
            err, elapsed = train_baseline(bl_name, X_train, y_train,
                                          X_test, y_test)
            label = BASELINE_METHODS[bl_name][0]
            print(f"    {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")
            csv_rows.append({
                'n_candidates': n_cand, 'n_wards': n_classes,
                'n_samples': total_samples,
                'method': label, 'encoding': 'numeric',
                'error_pct': round(100 * err, 2),
                'time_s': round(elapsed, 1),
            })

        # === ArrowFlow native (ballots directly as permutations) ===
        print(f"\n  --- ArrowFlow native (ballots as permutations) ---")
        data_train, adj_list = rankings_to_arrowflow_data(
            X_train, y_train, n_cand)
        data_test, _ = rankings_to_arrowflow_data(
            X_test, y_test, n_cand)

        af_configs = [
            ('[64] 1v',   [64],      ['sort', 'sort'], 1, 300),
            ('[64] 7v',   [64],      ['sort', 'sort'], 7, 300),
            ('[128] 1v',  [128],     ['sort', 'sort'], 1, 300),
            ('[128] 7v',  [128],     ['sort', 'sort'], 7, 300),
            ('[64,32] 7v', [64, 32], ['sort', 'sort', 'sort'], 7, 300),
            ('[128,64] 7v', [128, 64], ['sort', 'sort', 'sort'], 7, 400),
        ]

        for desc, filters, ltypes, n_views, n_iters in af_configs:
            af_config = ArrowFlowConfig(
                no_of_filters=filters,
                layer_types=ltypes,
                no_of_iters=n_iters,
                moe_no_of_networks=1,
                no_of_embedding_dim=n_cand,
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
                    X_train.astype(float), y_train,
                    X_test.astype(float), y_test,
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
                'n_candidates': n_cand, 'n_wards': n_classes,
                'n_samples': total_samples,
                'method': config_str, 'encoding': 'native',
                'error_pct': round(100 * mean_err, 2),
                'time_s': round(elapsed, 1),
            })

    # Write CSV
    csv_fields = ['n_candidates', 'n_wards', 'n_samples', 'method',
                  'encoding', 'error_pct', 'time_s']
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
        print(f"  [{r['n_candidates']}cand, {r['n_wards']}wards] "
              f"{r['method']:35s} ({r['encoding']:8s}): {r['error_pct']:5.1f}%")

    return csv_rows


if __name__ == '__main__':
    run_experiment()
