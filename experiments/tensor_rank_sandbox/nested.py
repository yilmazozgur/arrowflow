"""Nested cross-validation of the tensor -> rank hybrid under the paper's protocol, paired with the paper's recorded runs.

DESIGN (fixed 2026-09-24 before any nested score; committed before the run)
- Data and folds: the prepared data.npz and splits.json of the paper's own runs (2026-09-12-bridge-knn for the seven
  benchmark datasets; 2026-09-14-newdata-batch1/2 for the ten further datasets): 5 x 3 outer folds (split seed 27183),
  each with its 3 inner folds. So every outer fold is identical to the paper's.
- Arms (core_hybrid.py; the core's ['tensor', 'sort'] network with the repaired conversion installed):
    hybrid_target_mlp     the core's TensorNet trained by the target conversion   grid V {64, 128} x lr_tensor {.003, .01}
    hybrid_frozen_mlp     the same network, tensor layer never updated (control)   grid V {64, 128}
    hybrid_target_linear  a linear tensor layer trained by the target conversion   grid V {64, 128} x lr_tensor {.003, .01}
  Common: 60 epochs of batch updates (batch 32), contrast 0.5, beta 1, gate (wrong, or p_correct 0.1), rank lr 0.1,
  no validation checkpoint; inputs imputed by the training mean and standardized on the training rows.
- Selection: mean accuracy of the rank readout over the fold's 3 inner splits at fit seed 8129; ties to the lowest
  config_id. The chosen configuration is refitted on the outer training rows with fit seeds 8129, 19391 and 39019, as
  the paper does, and scored on the outer test rows.
- Readouts of every fit: 'rank' (the core's rank classification layer) and 'knn' (a footrule kNN on the tensor
  layer's ranking, neighbours and weighting chosen by multiview.select_knn_readout on the training rows: the readout
  ArrowFlow-kNN uses, read here from the tensor layer instead of a hidden rank layer).
- Primary contrasts (per dataset, paired over the 15 outer folds of fold accuracy averaged over the 3 fit seeds,
  corrected resampled t with test/train ratio 0.25 and 14 df, two-sided; Holm across the 17 datasets per contrast):
    P1  hybrid_target_mlp[rank]  - hybrid_frozen_mlp[rank]   does the rank signal train the tensor layer?
    P2  hybrid_target_mlp[rank]  - mlp                        against the paper's tuned MLP comparator
    P3  hybrid_target_mlp[knn]   - arrowflow_full_knn         against the paper's method (the same readout family)
  Everything else is descriptive.

    python -m experiments.tensor_rank_sandbox.nested run --output DIR [--workers 14]
    python -m experiments.tensor_rank_sandbox.nested analyse --output DIR
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import time
import warnings
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

RUNS_ROOT = Path(__file__).resolve().parents[3]/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'
SOURCES = {'2026-09-12-bridge-knn': ('iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits'),
           '2026-09-14-newdata-batch1': ('balance_scale', 'banknote_authentication', 'diabetes', 'ionosphere', 'qsar_biodeg'),
           '2026-09-14-newdata-batch2': ('climate_model_simulation_crashes', 'hcv_egyptian_patients', 'mfeat_zernike',
                                         'steel_plates_fault', 'vertebra_column')}
DATASETS = [d for group in SOURCES.values() for d in group]
FIT_SEEDS = (8129, 19391, 39019)
EPOCHS = 60
COMMON = {'contrast': .5, 'beta': 1., 'gate': 'accepted'}
BUILD = {'lr_rank': .1, 'p_correct': .1, 'batch_size': 32}
ARMS = {'hybrid_target_mlp': ('target', 'core_mlp', [{'V': V, 'lr_tensor': lr} for V in (64, 128) for lr in (.003, .01)]),
        'hybrid_frozen_mlp': ('frozen', 'core_mlp', [{'V': V, 'lr_tensor': .01} for V in (64, 128)]),
        'hybrid_target_linear': ('target', 'linear', [{'V': V, 'lr_tensor': lr} for V in (64, 128) for lr in (.003, .01)])}
REGISTERED = ('arrowflow_full_knn', 'mlp', 'svc_rbf', 'random_forest', 'gradient_boosting', 'numeric_knn', 'dummy')
PRIMARY = (('P1', ('hybrid_target_mlp', 'rank'), ('hybrid_frozen_mlp', 'rank')),
           ('P2', ('hybrid_target_mlp', 'rank'), ('mlp', None)),
           ('P3', ('hybrid_target_mlp', 'knn'), ('arrowflow_full_knn', None)))


def source_of(dataset):
    return next(run for run, group in SOURCES.items() if dataset in group)


def load(dataset):
    base = RUNS_ROOT/source_of(dataset)/dataset
    data = np.load(base/'data.npz', allow_pickle=True)
    return np.asarray(data['X'], dtype=float), np.asarray(data['y']), json.loads((base/'splits.json').read_text())


class Hybrid:
    def __init__(self, mode, arch, V, lr_tensor, seed):
        self.mode, self.arch, self.V, self.lr_tensor, self.seed = mode, arch, V, lr_tensor, seed

    def _inputs(self, X):
        return self.scaler.transform(self.imputer.transform(X))

    def tensor_positions(self, Z):
        """Descending-rank positions of the tensor outputs: the input ranking the rank layer sees (the core negates
        a tensor layer's input). A batch call equals the core's per-row call (InstanceNorm1d normalizes each row)."""
        import torch
        from .core_hybrid import descending_positions
        with torch.no_grad():
            h = self.net.graph.vertex_list['hyb_ly0'](torch.tensor(-Z, dtype=torch.float32)).numpy()
        return descending_positions(h)

    def fit(self, X, y):
        from experiments.make_revision.comparisons import StableFootruleKNN, derive_seed
        from experiments.make_revision.models import NumericImputer
        from experiments.make_revision.multiview import select_knn_readout
        from .core_hybrid import build, install, samples
        self.imputer = NumericImputer().fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        Z = self._inputs(X)
        self.classes_, yi = np.unique(y, return_inverse=True)
        iterations = EPOCHS * int(np.ceil(len(yi) / 32))
        self.net, config = build(Z.shape[1], len(self.classes_), V=self.V, iterations=iterations, lr_tensor=self.lr_tensor,
                                 arch=self.arch, seed=self.seed, **BUILD)
        install(self.net, self.mode, p_correct=BUILD['p_correct'], seed=self.seed, **COMMON)
        train = samples(Z, yi)
        self.net.train([train, train[:1]], config)
        pos = self.tensor_positions(Z)
        selection = select_knn_readout(pos, yi, seed=derive_seed(self.seed, 'hybrid_knn_readout'))
        self.knn = StableFootruleKNN(**selection['config'], input_kind='positions').fit(pos, yi)
        return self

    def predict(self, X, readout='rank'):
        from .core_hybrid import samples
        Z = self._inputs(X)
        if readout == 'rank':
            _, prediction = self.net.evaluate(samples(Z, np.zeros(len(Z), dtype=int)), 'supervised', 'classification')
            return self.classes_[np.asarray([int(p) for p in prediction])]
        return self.classes_[self.knn.predict(self.tensor_positions(Z))]


def config_id(config):
    from experiments.make_revision.evaluation import config_id as cid
    return cid(config)


def run_job(dataset, arm, index):
    X, y, splits = load(dataset)
    split = splits[index]
    mode, arch, candidates = ARMS[arm]
    train, test = np.asarray(split['train']), np.asarray(split['test'])
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores = {}
        for c in candidates:
            accs = []
            for inner in split['inner']:
                a, b = np.asarray(inner['train']), np.asarray(inner['validation'])
                model = Hybrid(mode, arch, c['V'], c['lr_tensor'], FIT_SEEDS[0]).fit(X[a], y[a])
                accs.append(float(np.mean(model.predict(X[b], 'rank') == y[b])))
            scores[config_id(c)] = float(np.mean(accs))
        best = min(candidates, key=lambda c: (-scores[config_id(c)], config_id(c)))
        fits = []
        for seed in FIT_SEEDS:
            model = Hybrid(mode, arch, best['V'], best['lr_tensor'], seed).fit(X[train], y[train])
            pr, pk = model.predict(X[test], 'rank'), model.predict(X[test], 'knn')
            fits.append({'seed': seed, 'rank_acc': float(np.mean(pr == y[test])), 'knn_acc': float(np.mean(pk == y[test])),
                         'rank_pred': pr.tolist(), 'knn_pred': pk.tolist()})
    return {'dataset': dataset, 'arm': arm, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
            'selected': best, 'inner_scores': scores, 'fits': fits, 'seconds': time.perf_counter() - start}


def _worker(args):
    dataset, arm, index, out = args
    path = Path(out)/'results'/f'{dataset}__{arm}__i{index:02d}.json'
    if path.exists():
        return path.name, 'exists'
    record = run_job(dataset, arm, index)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return path.name, f"{record['seconds']:.0f}s"


def run(out, workers):
    out = Path(out)
    (out/'results').mkdir(parents=True, exist_ok=True)
    (out/'design.json').write_text(json.dumps({'arms': {a: [m, arch, c] for a, (m, arch, c) in ARMS.items()}, 'epochs': EPOCHS,
                                               'common': COMMON, 'build': BUILD, 'fit_seeds': FIT_SEEDS, 'sources': SOURCES},
                                              indent=1) + '\n')
    sizes = {d: len(load(d)[1]) for d in DATASETS}
    jobs = sorted(((d, a, i, str(out)) for d in DATASETS for a in ARMS for i in range(15)),
                  key=lambda j: -sizes[j[0]] * len(ARMS[j[1]][2]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            name, status = future.result()
            print(f'[{done}/{len(jobs)}] {name} {status}', flush=True)


# ------------------------------------------------------------------------------------------------ analysis

def registered_fold_accuracy(dataset, model):
    base = RUNS_ROOT/source_of(dataset)/'results'
    out = {}
    for path in base.glob(f'{dataset}__{model}__r*f*.json'):
        record = json.loads(path.read_text())
        if record.get('status') != 'ok':
            raise RuntimeError(f'{path.name}: status {record.get("status")}')
        m = record['models'][0]
        out[(m['outer_repeat'], m['outer_fold'])] = float(np.mean([x['accuracy'] for x in record['models']]))
    return out


def hybrid_fold_accuracy(out_dir, dataset, arm, readout):
    out = {}
    for path in (Path(out_dir)/'results').glob(f'{dataset}__{arm}__i*.json'):
        record = json.loads(path.read_text())
        out[(record['outer_repeat'], record['outer_fold'])] = float(np.mean([f[f'{readout}_acc'] for f in record['fits']]))
    return out


def corrected_t(diff, ratio=.25):
    d = np.asarray(diff, dtype=float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), (1. if d.mean() == 0 else 0.)
    t = d.mean() / np.sqrt((1 / J + ratio) * var)
    return float(d.mean()), float(2 * stats.t.sf(abs(t), J - 1))


def holm(p):
    order = np.argsort(p)
    adjusted, running = np.empty(len(p)), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def analyse(out_dir):
    out_dir = Path(out_dir)
    acc = {}
    for d in DATASETS:
        for arm in ARMS:
            for readout in ('rank', 'knn'):
                acc[(d, arm, readout)] = hybrid_fold_accuracy(out_dir, d, arm, readout)
        for model in REGISTERED:
            acc[(d, model, None)] = registered_fold_accuracy(d, model)
    missing = [(d, a, r) for (d, a, r), v in acc.items() if len(v) != 15]
    if missing:
        raise SystemExit(f'{len(missing)} incomplete (dataset, model, readout) cells, e.g. {missing[:3]}')
    columns = [('hybrid_target_mlp', 'rank'), ('hybrid_target_mlp', 'knn'), ('hybrid_frozen_mlp', 'rank'),
               ('hybrid_frozen_mlp', 'knn'), ('hybrid_target_linear', 'rank'), ('hybrid_target_linear', 'knn'),
               ('arrowflow_full_knn', None), ('mlp', None), ('svc_rbf', None), ('random_forest', None),
               ('gradient_boosting', None), ('numeric_knn', None), ('dummy', None)]
    means = {(d, m, r): 100 * np.mean(list(acc[(d, m, r)].values())) for d in DATASETS for m, r in columns}
    lines = ['# Nested CV: mean outer-fold accuracy (%), 15 folds (fit seeds averaged)', '',
             '| dataset | ' + ' | '.join(f'{m}[{r}]' if r else m for m, r in columns) + ' |',
             '|---|' + '---|' * len(columns)]
    for d in DATASETS:
        lines.append(f'| {d} | ' + ' | '.join(f'{means[(d, m, r)]:.1f}' for m, r in columns) + ' |')
    lines.append('| **mean of 17** | ' + ' | '.join(f'{np.mean([means[(d, m, r)] for d in DATASETS]):.1f}' for m, r in columns) + ' |')
    result = {'means': {f'{d}|{m}|{r}': v for (d, m, r), v in means.items()}, 'contrasts': {}}
    for name, (a, ra), (b, rb) in PRIMARY:
        rows = []
        for d in DATASETS:
            keys = sorted(acc[(d, a, ra)])
            diff = [100 * (acc[(d, a, ra)][k] - acc[(d, b, rb)][k]) for k in keys]
            mean, p = corrected_t(diff)
            rows.append({'dataset': d, 'mean_difference': mean, 'p': p, 'folds_better': int(sum(x > 0 for x in diff))})
        for row, adj in zip(rows, holm([r['p'] for r in rows])):
            row['p_holm'] = float(adj)
        result['contrasts'][name] = rows
        pos = sum(r['mean_difference'] > 0 for r in rows)
        sig_pos = sum(r['mean_difference'] > 0 and r['p_holm'] < .05 for r in rows)
        sig_neg = sum(r['mean_difference'] < 0 and r['p_holm'] < .05 for r in rows)
        lines += ['', f"## {name}: {a}[{ra}] - {b}{'[' + rb + ']' if rb else ''} (points; Holm over 17)", '',
                  f'Higher mean on {pos}/17; significantly higher on {sig_pos}, significantly lower on {sig_neg}.', '',
                  '| dataset | difference | p | Holm p | folds better |', '|---|---|---|---|---|']
        for r in rows:
            lines.append(f"| {r['dataset']} | {r['mean_difference']:+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r['folds_better']}/15 |")
    (out_dir/'analysis.json').write_text(json.dumps(result, indent=1) + '\n')
    (out_dir/'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'analyse'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.output, args.workers)
    else:
        analyse(args.output)


if __name__ == '__main__':
    main()
