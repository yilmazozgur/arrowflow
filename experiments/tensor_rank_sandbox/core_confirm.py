"""The conversion ported into the core (core_hybrid.py), on the same 12 datasets, splits and seeds as confirm.py.

Fixed before any score: V = 64, the core's defaults otherwise (Adam at lr_tensor 0.01 with its schedule, rank lr 0.1,
p_correct 0.1, batch 32, no validation checkpoint), 60 epochs' worth of batch updates. Arms:
    frozen                 no tensor update
    core_original          the core's conversion, verbatim (all examples, as the core does)
    core_detached_flipped  the minimal repair: target detached and sign flipped (all examples, as the core)
    target                 the sandbox's conversion (beta 1, contrast 0.5, gate: wrong or p_correct)
    target_scrambled       the same with the signal's coordinates shuffled
Two tensor layers: 'linear' (as in the sandbox) and 'core_mlp' (the core's TensorNet). Readout: the core's rank layer.
    python -m experiments.tensor_rank_sandbox.core_confirm [--workers 12]
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from .confirm import OPENML, load_any
from .sandbox import DATASETS, Features

ARMS = {'frozen': {}, 'core_original': {}, 'core_detached_flipped': {'gate': 'all'},
        'target': {'gate': 'accepted', 'contrast': .5}, 'target_scrambled': {'gate': 'accepted', 'contrast': .5}}
ARCHS = ('linear', 'core_mlp')
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 60
RUNS = Path(__file__).with_name('runs')


def job(args):
    name, arm, arch, seed = args
    from .core_hybrid import fit_predict
    X, y = load_any(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, stratify=y, random_state=0)
    f = Features('linear').fit(Xtr)
    iterations = EPOCHS * int(np.ceil(len(ytr) / 32))
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pred, net = fit_predict(f.transform(Xtr), ytr, f.transform(Xte), arm, seed=seed, V=64, iterations=iterations,
                                arch=arch, **ARMS[arm])
    return {'dataset': name, 'arm': arm, 'arch': arch, 'seed': seed, 'acc': float(np.mean(pred == yte)),
            'iterations': iterations, 'seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    names = list(DATASETS) + list(OPENML)
    jobs = [(n, arm, arch, s) for n in names for arch in ARCHS for arm in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, jobs))
    RUNS.mkdir(exist_ok=True)
    (RUNS/'core_confirm.json').write_text(json.dumps({'arms': ARMS, 'epochs': EPOCHS, 'results': results}, indent=1) + '\n')
    acc = {(r['dataset'], r['arm'], r['arch'], r['seed']): r['acc'] for r in results}
    for arch in ARCHS:
        print(f'\n== tensor layer: {arch}. Held-out test accuracy (%), mean over 5 seeds; target - frozen (seeds better)')
        print(f"{'dataset':16s}" + ''.join(f'{a[:13]:>14s}' for a in ARMS) + f"{'target-frozen':>16s}")
        for n in names:
            means = {a: 100 * np.mean([acc[(n, a, arch, s)] for s in SEEDS]) for a in ARMS}
            d = [100 * (acc[(n, 'target', arch, s)] - acc[(n, 'frozen', arch, s)]) for s in SEEDS]
            print(f'{n:16s}' + ''.join(f'{means[a]:14.1f}' for a in ARMS) + f'{np.mean(d):+10.1f} ({sum(x > 0 for x in d)}/5)')
        overall = {a: 100 * np.mean([acc[(n, a, arch, s)] for n in names for s in SEEDS]) for a in ARMS}
        print(f"{'mean of 12':16s}" + ''.join(f'{overall[a]:14.1f}' for a in ARMS))


if __name__ == '__main__':
    main()
