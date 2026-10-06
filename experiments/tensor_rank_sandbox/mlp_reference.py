"""The MLP reference on the splits of confirm.py and core_confirm.py: the paper's MLP comparator
(comparisons.conventional_factory('mlp'): imputer, scaler, sklearn MLPClassifier), tuned exactly as the paper tunes it
(CONVENTIONAL_GRIDS['mlp'], budget 24, candidate seed 41071, 3-fold stratified CV on the training portion, mean accuracy,
ties to the lowest config_id), refitted on the training portion with 5 seeds.

    python -m experiments.tensor_rank_sandbox.mlp_reference [--workers 12]
"""
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from experiments.make_revision.comparisons import CONVENTIONAL_GRIDS, conventional_factory
from experiments.make_revision.evaluation import candidate_grid, config_id
from .confirm import OPENML, load_any
from .sandbox import DATASETS

SEEDS = (0, 1, 2, 3, 4)
RUNS = Path(__file__).with_name('runs')


def job(args):
    name, seed = args
    X, y = load_any(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
    candidates = candidate_grid(CONVENTIONAL_GRIDS['mlp'], 24, 41071)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr))
    scores = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for c in candidates:
            scores[config_id(c)] = np.mean([np.mean(conventional_factory('mlp', c, seed).fit(Xtr[a], ytr[a]).predict(Xtr[b])
                                                    == ytr[b]) for a, b in folds])
        best = min(candidates, key=lambda c: (-scores[config_id(c)], config_id(c)))
        acc = float(np.mean(conventional_factory('mlp', best, seed).fit(Xtr, ytr).predict(Xte) == yte))
    return {'dataset': name, 'seed': seed, 'acc': acc, 'selected': best}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    names = list(DATASETS) + list(OPENML)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, [(n, s) for n in names for s in SEEDS]))
    RUNS.mkdir(exist_ok=True)
    (RUNS/'mlp_reference.json').write_text(json.dumps(results, indent=1) + '\n')
    core = json.loads((RUNS/'core_confirm.json').read_text())['results']
    logistic = {r['dataset']: r['rank_acc'] for r in json.loads((RUNS/'confirm.json').read_text())['results']
                if r['arm'] == 'logistic'}
    def mean_core(n, arm, arch):
        return 100 * np.mean([r['acc'] for r in core if r['dataset'] == n and r['arm'] == arm and r['arch'] == arch])
    print('held-out test accuracy (%), mean over 5 seeds (logistic: one fit)')
    print(f"{'dataset':16s}{'hybrid target':>14s}{'hybrid repair':>14s}{'hybrid frozen':>14s}{'MLP (tuned)':>13s}{'logistic':>10s}")
    rows = []
    for n in names:
        mlp = 100 * np.mean([r['acc'] for r in results if r['dataset'] == n])
        row = (mean_core(n, 'target', 'core_mlp'), mean_core(n, 'core_detached_flipped', 'core_mlp'),
               mean_core(n, 'frozen', 'core_mlp'), mlp, 100 * logistic[n])
        rows.append(row)
        print(f'{n:16s}' + ''.join(f'{v:14.1f}' for v in row[:3]) + f'{row[3]:13.1f}{row[4]:10.1f}')
    m = np.mean(rows, axis=0)
    print(f"{'mean of 12':16s}" + ''.join(f'{v:14.1f}' for v in m[:3]) + f'{m[3]:13.1f}{m[4]:10.1f}')
    print(f'hybrid target vs MLP: higher on {sum(r[0] > r[3] for r in rows)}/12, lower on {sum(r[0] < r[3] for r in rows)}/12')


if __name__ == '__main__':
    main()
