"""Encoder swap: ArrowFlow-kNN at the paper's own per-fold selections, with only the encoder replaced.

DESIGN (fixed 2026-09-25 before any swap score; committed before the run)
- Data, folds and seeds: the paper's registered runs (2026-09-12-bridge-knn; 2026-09-14-newdata-batch1/2): 17 datasets,
  5 x 3 outer folds (split seed 27183), fit seeds 8129, 19391 and 39019.
- Configuration: in every outer fold, the configuration the paper's nested CV selected for arrowflow_full_knn in that
  fold (the registered result's selection), resolved on the outer training rows by bridge.resolve_selected (embed_dim,
  degree, augment), exactly as the paper's component ablation reconstructs it. Nothing is re-tuned.
- Arms. The seven-view pipeline is identical in every arm; only each view's encoder differs:
    fixed    the paper's encoder (OrdinalEncoder: imputation, polynomial degree, standardisation, the view's
             projection strategy (target-aware, random or calibrated), argsort). Its outer predictions are the
             registered ones; a sample is refitted here and must reproduce them exactly (the 'check' command).
    learned  per view, the core's TensorNet with V = the selected embed_dim, trained for 60 epochs as the hybrid's
             tensor layer (core_hybrid: target conversion, contrast 0.5, lr_tensor 0.01, rank lr 0.1, p_correct 0.1)
             on the view's training rows, seeded by the view seed. Its descending argsort is the view's input ranking.
    random   the same TensorNet at its seeded random initialisation, never trained: the control separating
             "learned" from "MLP-shaped".
  The pipeline is the same everywhere: hidden rank layer(s), learning rate, iterations, validation checkpoint,
  augmentation, the kNN readout chosen on training rows, and the majority vote. With the learned and random encoders,
  the views differ only by seed (there are no projection strategies), and the polynomial degree does not apply.
- Primary contrast: learned - fixed per dataset, paired over the 15 outer folds of fold accuracy (mean of the 3 fit
  seeds); corrected resampled t (test/train ratio 0.25, 14 df), two-sided; Holm across the 17 datasets.
- Reading, fixed now: "improves ArrowFlow" if the mean over the 17 is higher and more datasets are significantly
  higher than significantly lower; "worsens" if the reverse; otherwise "no difference shown".
- Secondary, descriptive: learned - random, random - fixed, learned - the paper's MLP, learned - the nested hybrid's
  kNN readout.

    python -m experiments.tensor_rank_sandbox.encoder_swap check              reproduction and timing on a sample
    python -m experiments.tensor_rank_sandbox.encoder_swap run --output DIR [--workers 14]
    python -m experiments.tensor_rank_sandbox.encoder_swap analyse --output DIR [--nested DIR]
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
from sklearn.preprocessing import StandardScaler
from .nested import DATASETS, RUNS_ROOT, corrected_t, holm, load, registered_fold_accuracy, source_of

FIT_SEEDS = (8129, 19391, 39019)
ARMS = ('learned', 'random')
ENCODER = {'epochs': 60, 'lr_tensor': .01, 'lr_rank': .1, 'p_correct': .1, 'contrast': .5, 'beta': 1.,
           'gate': 'accepted', 'arch': 'core_mlp'}
REFERENCE_MODEL = 'arrowflow_full_knn'
CHECK_SAMPLE = (('iris', 0), ('segment', 0), ('balance_scale', 0), ('mfeat_zernike', 3))


class LearnedTensorEncoder:
    """A view's encoder: the core's TensorNet (optionally trained as the hybrid's tensor layer) and a descending argsort
    of its outputs. fit/transform mirror OrdinalEncoder's; transform returns item orders, as OrdinalEncoder does."""
    def __init__(self, embed_dim, seed, train=True):
        self.embed_dim, self.seed, self.train = int(embed_dim), int(seed), bool(train)

    def fit(self, X, y):
        from experiments.make_revision.models import NumericImputer
        from .core_hybrid import build, install, samples
        self.imputer = NumericImputer().fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        Z = self.scaler.transform(self.imputer.transform(X))
        classes, yi = np.unique(y, return_inverse=True)
        iterations = ENCODER['epochs'] * int(np.ceil(len(yi) / 32))
        net, config = build(Z.shape[1], len(classes), V=self.embed_dim, iterations=iterations,
                            lr_tensor=ENCODER['lr_tensor'], arch=ENCODER['arch'], seed=self.seed, lr_rank=ENCODER['lr_rank'],
                            p_correct=ENCODER['p_correct'], batch_size=32)
        if self.train:
            install(net, 'target', p_correct=ENCODER['p_correct'], seed=self.seed, contrast=ENCODER['contrast'],
                    beta=ENCODER['beta'], gate=ENCODER['gate'])
            train = samples(Z, yi)
            net.train([train, train[:1]], config)
        self.layer = net.graph.vertex_list['hyb_ly0']
        self.training_updates_ = int(getattr(net, 'update_iter', 0))
        return self

    def transform(self, X):
        import torch
        from arrowflow.ranking import score_order
        Z = self.scaler.transform(self.imputer.transform(X))
        with torch.no_grad():
            h = self.layer(torch.tensor(-Z, dtype=torch.float32)).numpy().astype(float)   # the core negates the input
        return score_order(-h)                                                              # descending order of h


def learned_model_class(train):
    from experiments.make_revision.comparisons import derive_seed
    from experiments.make_revision.models import ArrowFlowEstimator
    from experiments.make_revision.multiview import MultiViewArrowFlowKNN

    class MultiViewTensorEncoderKNN(MultiViewArrowFlowKNN):
        """MultiViewArrowFlowKNN whose views are encoded by LearnedTensorEncoder instead of OrdinalEncoder. The loop is
        MultiViewArrowFlow.fit's own, line for line, with the encoder line replaced; the network, its seed and the
        readout selection are unchanged."""
        def fit(self, X, y):
            self.readouts_, self.readout_selections_ = [], []
            self.readout_seconds_ = 0.
            self.classes_ = np.unique(y)
            self.views_ = []
            encoding = training = 0.
            for v in range(self.n_views):
                seed_v = derive_seed(self.seed, 'view', v)
                start = time.perf_counter()
                enc = LearnedTensorEncoder(self.embed_dim, seed_v, train=train).fit(X, y)
                orders = enc.transform(X)
                encoding += time.perf_counter() - start
                net = ArrowFlowEstimator(embed_dim=self.embed_dim, degree=self.degree, widths=self.widths,
                                         iterations=self.iterations, learning_rate=self.learning_rate,
                                         batch_size=self.batch_size, last_layer_update=self.last_layer_update,
                                         ratio_data_backprop=self.ratio_data_backprop,
                                         motion_normalization_mult=self.motion_normalization_mult,
                                         p_correct=self.p_correct, seed=seed_v,
                                         validation_ratio=self.validation_ratio, augment=self.augment,
                                         n_augmentations=self.n_augmentations, max_swaps=self.max_swaps)
                net.fit_orders(orders, y)
                training += net.training_seconds_
                self.views_.append((enc, net))
                self._fit_view_readout(enc, net, orders, y, seed_v)
            self.encoding_seconds_ = encoding
            self.training_seconds_ = training
            return self

    return MultiViewTensorEncoderKNN


def registered_record(dataset, split):
    path = RUNS_ROOT/source_of(dataset)/'results'/f"{dataset}__{REFERENCE_MODEL}__r{split['outer_repeat']}f{split['outer_fold']}.json"
    record = json.loads(path.read_text())
    if record.get('status') != 'ok':
        raise RuntimeError(f'{path.name}: status {record.get("status")}')
    return record


def registered_predictions(record, seed, test):
    rows = {p['sample_id']: p['y_pred'] for p in record['predictions'] if p['model_seed'] == seed}
    return np.asarray([rows[int(i)] for i in test])


def fit_arm(arm, params, seed, X_train, y_train, X_test):
    from threadpoolctl import threadpool_limits
    from experiments.make_revision.models import seed_fit
    from experiments.make_revision.multiview import MultiViewArrowFlowKNN
    from experiments.make_revision.secondary_studies import majority
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seed_fit(seed)
        cls = MultiViewArrowFlowKNN if arm == 'fixed' else learned_model_class(train=(arm == 'learned'))
        model = cls(**params, seed=seed).fit(X_train, y_train)
        knn_views, _ = model.predict_views(X_test)
    return majority(knn_views), np.stack(knn_views)


def run_job(dataset, arm, index):
    from experiments.make_revision.bridge import resolve_selected
    X, y, splits = load(dataset)
    split = splits[index]
    train, test = np.asarray(split['train']), np.asarray(split['test'])
    record = registered_record(dataset, split)
    params = resolve_selected(record['selection']['config'], X.shape[1], len(train))
    start = time.perf_counter()
    fits = []
    for seed in FIT_SEEDS:
        pred, views = fit_arm(arm, params, seed, X[train], y[train], X[test])
        fits.append({'seed': seed, 'acc': float(np.mean(pred == y[test])), 'pred': np.asarray(pred).tolist(),
                     'view_acc': [float(np.mean(v == y[test])) for v in views]})
    return {'dataset': dataset, 'arm': arm, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
            'config': record['selection']['config'], 'params': params, 'fits': fits,
            'seconds': time.perf_counter() - start}


def check():
    """Refit the FIXED arm on a sample of (dataset, fold) at all three seeds: it must reproduce the registered outer
    predictions exactly, which shows this module rebuilds the paper's pipeline. Then time one learned and one random fit.
    Accuracies of the learned and random arms are not printed."""
    from experiments.make_revision.bridge import resolve_selected
    for dataset, index in CHECK_SAMPLE:
        X, y, splits = load(dataset)
        split = splits[index]
        train, test = np.asarray(split['train']), np.asarray(split['test'])
        record = registered_record(dataset, split)
        params = resolve_selected(record['selection']['config'], X.shape[1], len(train))
        for seed in FIT_SEEDS:
            start = time.perf_counter()
            pred, _ = fit_arm('fixed', params, seed, X[train], y[train], X[test])
            expected = registered_predictions(record, seed, test)
            differing = int(np.sum(np.asarray(pred) != expected))
            print(f'{dataset:14s} fold {index:2d} seed {seed}: fixed refit vs registered: {differing} of {len(test)} differ '
                  f'({time.perf_counter() - start:.0f}s)', flush=True)
            if differing:
                raise SystemExit('the fixed arm does not reproduce the registered predictions')
    X, y, splits = load('segment')
    split = splits[0]
    train, test = np.asarray(split['train']), np.asarray(split['test'])
    params = resolve_selected(registered_record('segment', split)['selection']['config'], X.shape[1], len(train))
    for arm in ARMS:
        start = time.perf_counter()
        fit_arm(arm, params, FIT_SEEDS[0], X[train], y[train], X[test])
        print(f'timing: segment fold 0, one seed, {arm}: {time.perf_counter() - start:.0f}s (embed_dim {params["embed_dim"]})')


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
    (out/'design.json').write_text(json.dumps({'arms': ARMS, 'encoder': ENCODER, 'fit_seeds': FIT_SEEDS,
                                               'reference_model': REFERENCE_MODEL}, indent=1) + '\n')
    sizes = {d: len(load(d)[1]) for d in DATASETS}
    jobs = sorted(((d, a, i, str(out)) for d in DATASETS for a in ARMS for i in range(15)),
                  key=lambda j: (-sizes[j[0]] * (2 if j[1] == 'learned' else 1)))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            name, status = future.result()
            print(f'[{done}/{len(jobs)}] {name} {status}', flush=True)


def swap_fold_accuracy(out_dir, dataset, arm):
    out = {}
    for path in (Path(out_dir)/'results').glob(f'{dataset}__{arm}__i*.json'):
        record = json.loads(path.read_text())
        out[(record['outer_repeat'], record['outer_fold'])] = float(np.mean([f['acc'] for f in record['fits']]))
    return out


def analyse(out_dir, nested_dir=None):
    out_dir = Path(out_dir)
    acc = {}
    for d in DATASETS:
        acc[(d, 'fixed')] = registered_fold_accuracy(d, REFERENCE_MODEL)
        acc[(d, 'mlp')] = registered_fold_accuracy(d, 'mlp')
        for arm in ARMS:
            acc[(d, arm)] = swap_fold_accuracy(out_dir, d, arm)
        if nested_dir:
            from .nested import hybrid_fold_accuracy
            acc[(d, 'hybrid_knn')] = hybrid_fold_accuracy(nested_dir, d, 'hybrid_target_mlp', 'knn')
    incomplete = [k for k, v in acc.items() if len(v) != 15]
    if incomplete:
        raise SystemExit(f'{len(incomplete)} incomplete cells, e.g. {incomplete[:3]}')
    columns = ['fixed', 'learned', 'random', 'mlp'] + (['hybrid_knn'] if nested_dir else [])
    means = {(d, c): 100 * np.mean(list(acc[(d, c)].values())) for d in DATASETS for c in columns}
    lines = ['# Encoder swap: mean outer-fold accuracy (%), 15 folds (fit seeds averaged)', '',
             '| dataset | ' + ' | '.join(columns) + ' |', '|---|' + '---|' * len(columns)]
    for d in DATASETS:
        lines.append(f'| {d} | ' + ' | '.join(f'{means[(d, c)]:.1f}' for c in columns) + ' |')
    lines.append('| **mean of 17** | ' + ' | '.join(f'{np.mean([means[(d, c)] for d in DATASETS]):.1f}' for c in columns) + ' |')
    result = {'means': {f'{d}|{c}': v for (d, c), v in means.items()}, 'contrasts': {}}
    contrasts = [('PRIMARY', 'learned', 'fixed'), ('learned - random', 'learned', 'random'),
                 ('random - fixed', 'random', 'fixed'), ('learned - mlp', 'learned', 'mlp')]
    if nested_dir:
        contrasts.append(('learned - hybrid_knn', 'learned', 'hybrid_knn'))
    for name, a, b in contrasts:
        rows = []
        for d in DATASETS:
            keys = sorted(acc[(d, a)])
            diff = [100 * (acc[(d, a)][k] - acc[(d, b)][k]) for k in keys]
            mean, p = corrected_t(diff)
            rows.append({'dataset': d, 'mean_difference': mean, 'p': p, 'folds_better': int(sum(x > 0 for x in diff))})
        for row, adj in zip(rows, holm([r['p'] for r in rows])):
            row['p_holm'] = float(adj)
        up = sum(r['mean_difference'] > 0 and r['p_holm'] < .05 for r in rows)
        down = sum(r['mean_difference'] < 0 and r['p_holm'] < .05 for r in rows)
        mean_all = float(np.mean([r['mean_difference'] for r in rows]))
        verdict = ('improves ArrowFlow' if mean_all > 0 and up > down else 'worsens ArrowFlow' if mean_all < 0 and down > up
                   else 'no difference shown') if name == 'PRIMARY' else None
        result['contrasts'][name] = {'rows': rows, 'mean_difference': mean_all, 'significantly_higher': up,
                                     'significantly_lower': down, 'verdict': verdict}
        lines += ['', f'## {name}: {a} - {b} (points; Holm over 17)', '',
                  f'Mean difference {mean_all:+.2f}; higher on {sum(r["mean_difference"] > 0 for r in rows)}/17; '
                  f'significantly higher on {up}, significantly lower on {down}.' + (f' Verdict: **{verdict}**.' if verdict else ''),
                  '', '| dataset | difference | p | Holm p | folds better |', '|---|---|---|---|---|']
        for r in rows:
            lines.append(f"| {r['dataset']} | {r['mean_difference']:+.2f} | {r['p']:.4f} | {r['p_holm']:.4f} | {r['folds_better']}/15 |")
    (out_dir/'analysis.json').write_text(json.dumps(result, indent=1) + '\n')
    (out_dir/'tables.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'run', 'analyse'))
    parser.add_argument('--output')
    parser.add_argument('--nested')
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args()
    if args.command == 'check':
        check()
    elif args.command == 'run':
        run(args.output, args.workers)
    else:
        analyse(args.output, args.nested)


if __name__ == '__main__':
    main()
