"""Confirmation of ONE configuration, fixed from the dev search (stage 2, best mean dev accuracy) before any test score:
the held-out 30% test portions of the four search datasets, and eight cached OpenML datasets the search never used.

Arms (identical except for what the tensor layer receives):
    trained    the tensor layer is trained by the rank layer's converted displacement signal
    frozen     the tensor layer keeps its random initialisation; the rank layer trains the same way
    scrambled  the same signal with its coordinates shuffled in every example (equal size, wrong alignment)
    logistic   logistic regression on the same standardized inputs (reference for a linear first layer)
Five model seeds per dataset; one stratified 70/30 split per dataset (split seed 0, as in the search).

    python -m experiments.tensor_rank_sandbox.confirm [--workers 12]
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
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .sandbox import DATASETS, Features, TensorRank, load

FINAL = {'V': 64, 'convert': 'target', 'eta': .01, 'contrast': .5, 'eta_decay': True, 'epochs': 150}
ARMS = {'trained': FINAL, 'frozen': {**FINAL, 'train_tensor': False}, 'scrambled': {**FINAL, 'scramble': True}}
OPENML = {'vehicle': 54, 'segment': 36, 'balance_scale': 11, 'mfeat_zernike': 22, 'ionosphere': 59, 'diabetes': 37,
          'vertebra_column': 1523, 'steel_plates': 40982}
SEEDS = (0, 1, 2, 3, 4)
RUNS = Path(__file__).with_name('runs')


def load_any(name):
    if name in DATASETS:
        return load(name)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bunch = fetch_openml(data_id=OPENML[name], as_frame=False, parser='auto')
    X = np.asarray(bunch.data, dtype=float)
    return X, np.unique(np.asarray(bunch.target), return_inverse=True)[1]


def job(args):
    name, arm, seed = args
    X, y = load_any(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
    if arm == 'logistic':
        f = Features('linear').fit(Xtr)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = LogisticRegression(max_iter=5000).fit(f.transform(Xtr), ytr)
        return {'dataset': name, 'arm': arm, 'seed': seed, 'rank_acc': float(np.mean(model.predict(f.transform(Xte)) == yte)),
                'knn_acc': None}
    m = TensorRank(ARMS[arm], seed=seed).fit(Xtr, ytr)
    return {'dataset': name, 'arm': arm, 'seed': seed, 'rank_acc': float(np.mean(m.predict(Xte, 'rank') == yte)),
            'knn_acc': float(np.mean(m.predict(Xte, 'knn') == yte))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    names = list(DATASETS) + list(OPENML)
    jobs = [(n, arm, s) for n in names for arm in (*ARMS, 'logistic') for s in (SEEDS if arm != 'logistic' else (0,))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, jobs))
    RUNS.mkdir(exist_ok=True)
    (RUNS/'confirm.json').write_text(json.dumps({'final': FINAL, 'results': results}, indent=1) + '\n')
    acc = {(r['dataset'], r['arm'], r['seed']): r['rank_acc'] for r in results}
    knn = {(r['dataset'], r['arm'], r['seed']): r['knn_acc'] for r in results}
    print('held-out test accuracy (%), rank readout, mean over 5 seeds; paired differences in points (seeds better)')
    print(f"{'dataset':16s}{'trained':>9s}{'frozen':>9s}{'scrambl':>9s}{'logreg':>9s}   {'trained-frozen':>16s}{'scrambled-frozen':>18s}"
          f"{'kNN trained/frozen':>21s}")
    for group, members in (('search datasets (dev-tuned)', list(DATASETS)), ('new datasets (never seen)', list(OPENML))):
        print(f'-- {group}')
        for n in members:
            mean = {arm: 100 * np.mean([acc[(n, arm, s)] for s in SEEDS]) for arm in ARMS}
            d = [100 * (acc[(n, 'trained', s)] - acc[(n, 'frozen', s)]) for s in SEEDS]
            e = [100 * (acc[(n, 'scrambled', s)] - acc[(n, 'frozen', s)]) for s in SEEDS]
            kt = 100 * np.mean([knn[(n, 'trained', s)] for s in SEEDS])
            kf = 100 * np.mean([knn[(n, 'frozen', s)] for s in SEEDS])
            print(f"{n:16s}{mean['trained']:9.1f}{mean['frozen']:9.1f}{mean['scrambled']:9.1f}{100 * acc[(n, 'logistic', 0)]:9.1f}"
                  f"   {np.mean(d):+8.1f} ({sum(x > 0 for x in d)}/5){np.mean(e):+10.1f} ({sum(x > 0 for x in e)}/5)"
                  f"{kt:12.1f} /{kf:5.1f}")


if __name__ == '__main__':
    main()
