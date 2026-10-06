"""Staged configuration search on the DEV portions (3-fold CV x 2 seeds); the 30% test portions are never read here.

    python -m experiments.tensor_rank_sandbox.search STAGE [--workers 12]
Results are appended to experiments/tensor_rank_sandbox/runs/search.jsonl.
"""
import os
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from .sandbox import DATASETS, Features, dev_cv_score, dev_test

RUNS = Path(__file__).with_name('runs')


def stage_configs(stage):
    if stage == 0:
        configs = [('frozen tensor, rank trained', {'train_tensor': False})]
        for convert in ('target', 'sign', 'linear'):
            for eta in (.003, .01, .03, .1, .3):
                configs.append((f'{convert} eta={eta}', {'convert': convert, 'eta': eta}))
        return configs
    if stage == 1:
        base = {'convert': 'target', 'eta': .01}
        variants = [('base: target eta=.01', {}), ('epochs 150', {'epochs': 150}), ('p_correct 1.0', {'p_correct': 1.}),
                    ('p_correct 0.3', {'p_correct': .3}), ('gate all', {'gate': 'all'}), ('beta 0.5', {'beta': .5}),
                    ('contrast 0.5', {'contrast': .5}), ('contrast 1.0', {'contrast': 1.}), ('V 16', {'V': 16}),
                    ('V 64', {'V': 64}), ('phi poly2', {'phi': 'poly2'}), ('adam eta .003', {'optimizer': 'adam', 'eta': .003}),
                    ('adam eta .01', {'optimizer': 'adam', 'eta': .01}), ('init class_borda', {'init': 'class_borda'}),
                    ('rank_repel', {'rank_repel': True}), ('row_norm', {'row_norm': True}),
                    ('rank frozen (random filters)', {'train_rank': False}),
                    ('rank frozen (class_borda filters)', {'train_rank': False, 'init': 'class_borda'}),
                    ('FROZEN tensor, class_borda', {'train_tensor': False, 'init': 'class_borda'}),
                    ('FROZEN tensor, V 64', {'train_tensor': False, 'V': 64})]
        return [(label, {**base, **v}) for label, v in variants]
    if stage == 2:
        variants = []
        for V in (32, 64):
            variants += [(f'V{V} FROZEN tensor', {'V': V, 'train_tensor': False}),
                         (f'V{V} FROZEN tensor + ckpt', {'V': V, 'train_tensor': False, 'val_ratio': .15, 'epochs': 100}),
                         (f'V{V} target c.5', {'V': V, 'contrast': .5}),
                         (f'V{V} target c.5 decay e150', {'V': V, 'contrast': .5, 'eta_decay': True, 'epochs': 150}),
                         (f'V{V} target c.5 ckpt e100', {'V': V, 'contrast': .5, 'val_ratio': .15, 'epochs': 100}),
                         (f'V{V} target c.5 ckpt decay e150', {'V': V, 'contrast': .5, 'val_ratio': .15, 'eta_decay': True, 'epochs': 150}),
                         (f'V{V} target c.5 repel ckpt decay', {'V': V, 'contrast': .5, 'rank_repel': True, 'val_ratio': .15, 'eta_decay': True, 'epochs': 150}),
                         (f'V{V} target c.5 eta.003 ckpt decay', {'V': V, 'contrast': .5, 'eta': .003, 'val_ratio': .15, 'eta_decay': True, 'epochs': 150}),
                         (f'V{V} target c.5 ckpt decay SCRAMBLED', {'V': V, 'contrast': .5, 'val_ratio': .15, 'eta_decay': True, 'epochs': 150, 'scramble': True})]
        return [(label, {'convert': 'target', 'eta': .01, **v}) for label, v in variants]
    raise ValueError(f'unknown stage {stage}')


def logistic_reference(name, seeds=(0, 1), folds=3):
    Xd, _, yd, _ = dev_test(name)
    acc = []
    for s in seeds:
        for a, b in StratifiedKFold(folds, shuffle=True, random_state=100 + s).split(Xd, yd):
            f = Features('linear').fit(Xd[a])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = LogisticRegression(max_iter=2000).fit(f.transform(Xd[a]), yd[a])
            acc.append(np.mean(model.predict(f.transform(Xd[b])) == yd[b]))
    return float(np.mean(acc))


def job(args):
    label, cfg, name = args
    start = time.perf_counter()
    rank, knn = dev_cv_score(name, cfg)
    return {'label': label, 'cfg': cfg, 'dataset': name, 'rank_acc': rank, 'knn_acc': knn,
            'seconds': time.perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', type=int)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    configs = stage_configs(args.stage)
    names = list(DATASETS)
    jobs = [(label, cfg, name) for label, cfg in configs for name in names]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(job, jobs))
    with open(RUNS/'search.jsonl', 'a') as log:
        for r in results:
            log.write(json.dumps({'stage': args.stage, **r}) + '\n')
    table = {}
    for r in results:
        table.setdefault(r['label'], {})[r['dataset']] = (r['rank_acc'], r['knn_acc'])
    print(f"dev-CV accuracy (%): rank readout / kNN readout    stage {args.stage}")
    print(f"{'config':32s}" + ''.join(f'{n:>18s}' for n in names) + f"{'mean rank':>11s}")
    ref = {n: logistic_reference(n) for n in names}
    print(f"{'logistic regression (reference)':32s}" + ''.join(f'{100 * ref[n]:18.1f}' for n in names)
          + f'{100 * np.mean(list(ref.values())):11.1f}')
    for label, _ in configs:
        row = table[label]
        print(f'{label:32s}' + ''.join(f'{100 * row[n][0]:10.1f} /{100 * row[n][1]:5.1f}' for n in names)
              + f'{100 * np.mean([row[n][0] for n in names]):11.1f}')


if __name__ == '__main__':
    main()
