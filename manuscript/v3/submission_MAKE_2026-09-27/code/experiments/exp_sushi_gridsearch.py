"""
Sushi ArrowFlow Grid Search — Fair Comparison
==============================================
Systematic hyperparameter search for ArrowFlow native encoding on Sushi,
matching the GridSearchCV treatment given to baselines.

Grid:
    - filters: [32], [64], [128], [256], [64,32], [128,64], [64,128]
    - embed_dim: 5, 10, 16, 32
    - learning_rate: 0.05, 0.1, 0.2
    - n_iters: 200, 300, 500
    - augment: True/False
    - max_swaps: 1, 2, 3 (when augment=True)
    - last_layer_update: True/False
    - n_views: 1, 3, 7

Phase 1: Coarse search with 1 view, 1 sim (fast screening)
Phase 2: Top configs re-evaluated with 7 views, 5 sims

Run:
    python -m test.experiments.exp_sushi_gridsearch
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
from itertools import product

from arrowflow.benchmark import ArrowFlowConfig, run_arrowflow_ensemble
from experiments.exp_sushi_ordinal import (
    load_sushi, rankings_to_arrowflow_data, train_baseline, BASELINE_METHODS,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def run_gridsearch():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f'sushi_gridsearch_{timestamp}.csv')

    print("=" * 80)
    print("Sushi ArrowFlow Grid Search")
    print("=" * 80)

    X, y = load_sushi()
    if X is None:
        print("Sushi dataset unavailable.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    n_classes = len(np.unique(y))
    data_train, adj_list = rankings_to_arrowflow_data(X_train, y_train)
    data_test, _ = rankings_to_arrowflow_data(X_test, y_test)

    print(f"{len(X_train)} train, {len(X_test)} test, {n_classes} classes\n")

    # ---- Phase 1: Coarse search (1 view, 1 sim) ----
    architectures = [
        ('[32]',    [32],      ['sort', 'sort']),
        ('[64]',    [64],      ['sort', 'sort']),
        ('[128]',   [128],     ['sort', 'sort']),
        ('[256]',   [256],     ['sort', 'sort']),
        ('[64,32]', [64, 32],  ['sort', 'sort', 'sort']),
        ('[128,64]',[128, 64], ['sort', 'sort', 'sort']),
        ('[64,128]',[64, 128], ['sort', 'sort', 'sort']),
    ]
    embed_dims = [5, 10, 16, 32]
    learning_rates = [0.05, 0.1, 0.2]
    n_iters_list = [200, 300, 500]
    augment_configs = [
        (False, 0, 0),      # no augmentation
        (True, 1, 1),       # mild augmentation
        (True, 1, 2),       # moderate augmentation
        (True, 1, 3),       # strong augmentation
    ]
    last_layer_opts = [True, False]

    # Build grid
    grid = []
    for arch, edim, lr, niters, (aug, naug, mswap), llu in product(
        architectures, embed_dims, learning_rates, n_iters_list,
        augment_configs, last_layer_opts
    ):
        grid.append({
            'arch_desc': arch[0], 'filters': arch[1], 'layer_types': arch[2],
            'embed_dim': edim, 'lr': lr, 'n_iters': niters,
            'augment': aug, 'n_augmentations': naug, 'max_swaps': mswap,
            'last_layer_update': llu,
        })

    total = len(grid)
    print(f"Phase 1: Screening {total} configs (1 view, 1 sim each)")
    print("-" * 60)

    phase1_results = []
    t_start = time.time()

    for i, cfg in enumerate(grid):
        af_config = ArrowFlowConfig(
            no_of_filters=cfg['filters'],
            layer_types=cfg['layer_types'],
            no_of_iters=cfg['n_iters'],
            moe_no_of_networks=1,
            no_of_embedding_dim=cfg['embed_dim'],
            encoding_mode='native',
            n_ensemble_views=1,
            use_augmentation=cfg['augment'],
            n_augmentations=cfg['n_augmentations'],
            max_swaps=cfg['max_swaps'],
            learning_rate=cfg['lr'],
            last_layer_update=cfg['last_layer_update'],
            verbose=0,
        )
        err = run_arrowflow_ensemble(
            X_train.astype(float), y_train, X_test.astype(float), y_test,
            n_classes, af_config, seed=42,
            preencoded_data=(data_train, data_test, adj_list)
        )
        phase1_results.append((err, cfg))

        if (i + 1) % 50 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            best_so_far = min(r[0] for r in phase1_results)
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, "
                  f"best so far: {100*best_so_far:.1f}%")

    # Sort by error
    phase1_results.sort(key=lambda x: x[0])

    print(f"\nPhase 1 complete. Top 10 configs:")
    for rank, (err, cfg) in enumerate(phase1_results[:10]):
        aug_str = f"aug(sw={cfg['max_swaps']})" if cfg['augment'] else "no_aug"
        llu_str = "llu=T" if cfg['last_layer_update'] else "llu=F"
        print(f"  #{rank+1}: {100*err:5.1f}% — {cfg['arch_desc']} "
              f"e={cfg['embed_dim']} lr={cfg['lr']} it={cfg['n_iters']} "
              f"{aug_str} {llu_str}")

    # ---- Phase 2: Top 15 configs with 7 views, 5 sims ----
    top_n = 15
    top_configs = phase1_results[:top_n]

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Re-evaluating top {top_n} configs (7 views, 5 sims)")
    print("=" * 60)

    phase2_results = []
    for rank, (phase1_err, cfg) in enumerate(top_configs):
        errors = []
        t0 = time.time()
        for sim in range(5):
            seed = 42 + sim * 12345
            af_config = ArrowFlowConfig(
                no_of_filters=cfg['filters'],
                layer_types=cfg['layer_types'],
                no_of_iters=cfg['n_iters'],
                moe_no_of_networks=1,
                no_of_embedding_dim=cfg['embed_dim'],
                encoding_mode='native',
                n_ensemble_views=7,
                use_augmentation=cfg['augment'],
                n_augmentations=cfg['n_augmentations'],
                max_swaps=cfg['max_swaps'],
                learning_rate=cfg['lr'],
                last_layer_update=cfg['last_layer_update'],
                verbose=0,
            )
            err = run_arrowflow_ensemble(
                X_train.astype(float), y_train, X_test.astype(float), y_test,
                n_classes, af_config, seed=seed,
                preencoded_data=(data_train, data_test, adj_list)
            )
            errors.append(err)
        elapsed = time.time() - t0
        mean_err = np.mean(errors)
        std_err = np.std(errors) / np.sqrt(len(errors))

        aug_str = f"aug(sw={cfg['max_swaps']})" if cfg['augment'] else "no_aug"
        llu_str = "llu=T" if cfg['last_layer_update'] else "llu=F"
        config_str = (f"{cfg['arch_desc']} e={cfg['embed_dim']} lr={cfg['lr']} "
                      f"it={cfg['n_iters']} {aug_str} {llu_str}")
        print(f"  #{rank+1}: {100*mean_err:5.1f}% ± {100*std_err:.1f}% — "
              f"{config_str} [{elapsed:.1f}s]")

        phase2_results.append({
            'rank': rank + 1, 'config': config_str,
            'mean_err': round(100 * mean_err, 2),
            'std_err': round(100 * std_err, 2),
            'phase1_err': round(100 * phase1_err, 2),
            **cfg,
        })

    # ---- Baselines for reference ----
    print(f"\n{'=' * 60}")
    print("Baselines (for reference)")
    print("=" * 60)
    for bl_name in BASELINE_METHODS:
        t0 = time.time()
        err = train_baseline(bl_name, X_train, y_train, X_test, y_test)
        elapsed = time.time() - t0
        label = BASELINE_METHODS[bl_name][0]
        print(f"  {label:20s}: {100*err:5.1f}% [{elapsed:.1f}s]")

    # Write CSV
    csv_fields = ['rank', 'config', 'mean_err', 'std_err', 'phase1_err',
                  'arch_desc', 'embed_dim', 'lr', 'n_iters',
                  'augment', 'n_augmentations', 'max_swaps', 'last_layer_update']
    with open(csv_path, 'w', newline='') as f:
        # Remove non-serializable fields
        rows_out = []
        for r in phase2_results:
            row = {k: r[k] for k in csv_fields if k in r}
            rows_out.append(row)
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nResults saved to: {csv_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL RANKING (7 views, 5 sims)")
    print("=" * 80)
    phase2_results.sort(key=lambda x: x['mean_err'])
    for r in phase2_results:
        print(f"  {r['mean_err']:5.1f}% ± {r['std_err']:.1f}% — {r['config']}")

    return phase2_results


if __name__ == '__main__':
    run_gridsearch()
