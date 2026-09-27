"""Nested selection with factories that own all fold-local preprocessing."""
from dataclasses import dataclass
import hashlib
import json
import resource
import time
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from threadpoolctl import threadpool_limits


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def config_id(config):
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:16]


def dataset_fingerprint(X, y, feature_names, label_map):
    X = np.asarray(X, dtype='<f8').copy()
    X[np.isnan(X)] = np.nan  # normalize NaN payloads
    h = hashlib.sha256(canonical_json({'shape': X.shape, 'feature_names': list(feature_names),
                                      'label_map': list(label_map)}).encode())
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(canonical_json(np.asarray(y).tolist()).encode())
    return h.hexdigest()


def candidate_grid(grid, budget=24, seed=41071):
    """Sample unique configurations before fitting, with a canonical tie key."""
    candidates = {config_id(c): c for c in ParameterGrid(grid)}
    ids = sorted(candidates)
    if budget < 1:
        raise ValueError('Candidate budget must be positive')
    if len(ids) > budget:
        ids = sorted(np.random.RandomState(seed).choice(ids, budget, replace=False))
    return [candidates[i] for i in ids]


def make_splits(y, outer_folds=5, repeats=3, inner_folds=3, seed=27183):
    y = np.asarray(y)
    if min(np.unique(y, return_counts=True)[1]) < outer_folds:
        raise ValueError('Insufficient class counts for outer stratification')
    splits = []
    for repeat in range(repeats):
        outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y)):
            if min(np.unique(y[train], return_counts=True)[1]) < inner_folds:
                raise ValueError('Insufficient class counts for inner stratification')
            inner_seed = seed + 10000 + repeat * outer_folds + fold
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=inner_seed)
            splits.append({'outer_repeat': repeat, 'outer_fold': fold, 'split_seed': seed + repeat,
                           'inner_seed': inner_seed, 'train': train.tolist(), 'test': test.tolist(),
                           'inner': [{'train': train[a].tolist(), 'validation': train[b].tolist()}
                                     for a, b in inner.split(np.zeros(len(train)), y[train])]})
    return splits


def validate_split(split, n):
    def indices(values):
        if any(not isinstance(v, (int, np.integer)) or v < 0 or v >= n for v in values):
            raise ValueError('Invalid split indices')
        if len(set(values)) != len(values):
            raise ValueError('Duplicate split indices')
        return set(values)
    train, test = indices(split['train']), indices(split['test'])
    if train & test or train | test != set(range(n)):
        raise ValueError('Outer split must be a disjoint partition')
    validations = []
    for inner in split['inner']:
        a, b = indices(inner['train']), indices(inner['validation'])
        if a & b or a | b != train:
            raise ValueError('Inner split leaks or omits outer training rows')
        validations += inner['validation']
    if sorted(validations) != sorted(train):
        raise ValueError('Inner validations must partition outer training')


@dataclass
class ModelSpec:
    model_id: str
    factory: object  # factory(config: dict, seed: int) -> fresh unfitted estimator
    candidates: list
    stochastic: bool

    def __post_init__(self):
        ids = [config_id(c) for c in self.candidates]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Supply a nonempty unique candidate list')


def metric_values(y, predictions):
    return {'accuracy': float(accuracy_score(y, predictions)),
            'error': float(1 - accuracy_score(y, predictions)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predictions)),
            'macro_f1': float(f1_score(y, predictions, average='macro', zero_division=0))}


def _fit_predict(spec, config, seed, X_train, y_train, X_query):
    # No held-out labels cross this interface. CPU processes, never thread workers.
    from .models import seed_fit
    with threadpool_limits(limits=1):
        seed_fit(seed)
        estimator = spec.factory(config.copy(), seed)
        start = time.perf_counter()
        estimator.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        start = time.perf_counter()
        predictions = np.asarray(estimator.predict(X_query))
        prediction_time = time.perf_counter() - start
    encoding = getattr(estimator, 'encoding_seconds_', None)
    query_encoding = getattr(estimator, 'last_encoding_seconds_', None)
    timing = {'fit_seconds': fit_time, 'encoding_seconds': encoding,
              'classifier_fit_seconds': getattr(estimator, 'classifier_fit_seconds_',
                                                getattr(estimator, 'training_seconds_', None)),
              'fit_warnings': getattr(estimator, 'fit_warnings_', []),
              'representation_metadata': getattr(estimator, 'representation_metadata_', None),
              'inference_state_array_bytes': getattr(estimator, 'inference_array_bytes_', None),
              'predict_seconds': prediction_time, 'query_encoding_seconds': query_encoding,
              'inference_seconds': prediction_time - (query_encoding or 0),
              'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'training_sample_count': len(y_train),
              'preprocessing_settings': (estimator.encoder_.get_params(deep=False)
                                         if hasattr(estimator, 'encoder_') else repr(estimator)),
              'training_tie_rate': getattr(getattr(estimator, 'encoder_', None), 'training_tie_rate_', None)}
    return predictions, timing


class SelectionFailed(RuntimeError):
    """Terminal selection failure carrying every attempted inner fit."""
    def __init__(self, message, fits):
        super().__init__(message)
        self.fits = fits


def select_model(X, y, split, spec, seeds=(8129, 19391, 39019), score='accuracy', sink=None):
    """Screen all candidates, rerank best 3 stochastic candidates over 3 seeds.

    Returned fit log includes failures. A candidate failing any required fit is
    ineligible; an all-failed selection raises after emitting logs to sink.
    """
    X, y = np.asarray(X), np.asarray(y)
    validate_split(split, len(y))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError('Exactly three distinct predetermined fit seeds required')
    if score not in ('accuracy', 'balanced_accuracy', 'macro_f1'):
        raise ValueError('Unknown selection metric')
    fits = []
    def fit_candidate(config, seed, inner_fold, inner):
        row = {'stage': 'inner', 'model_id': spec.model_id, 'config_id': config_id(config),
               'config': config, 'model_seed': seed, 'inner_fold': inner_fold,
               'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
               'fit_rows': inner['train'], 'validation_rows': inner['validation']}
        try:
            pred, timing = _fit_predict(spec, config, seed, X[inner['train']], y[inner['train']], X[inner['validation']])
            row.update(timing, status='ok', score=metric_values(y[inner['validation']], pred)[score])
        except Exception as exc:
            row.update(status='failed', score=None, exception=f'{type(exc).__name__}: {exc}')
        fits.append(row)
        if sink is not None:
            sink(row)
    for config in spec.candidates:
        for i, inner in enumerate(split['inner']):
            fit_candidate(config, seeds[0], i, inner)
    def ranking(configs):
        scores = []
        for c in configs:
            rows = [r for r in fits if r['config_id'] == config_id(c)]
            if all(r['status'] == 'ok' for r in rows):
                scores.append((-float(np.mean([r['score'] for r in rows])), config_id(c), c))
        return sorted(scores, key=lambda row: row[:2])
    screened = ranking(spec.candidates)
    if not screened:
        raise SelectionFailed(f'All candidates failed for {spec.model_id}; inspect fit log', fits)
    finalists = [r[2] for r in screened[:3]] if spec.stochastic else [r[2] for r in screened]
    if spec.stochastic:
        for config in finalists:
            for seed in seeds[1:]:
                for i, inner in enumerate(split['inner']):
                    fit_candidate(config, seed, i, inner)
    reranked = ranking(finalists)
    if not reranked:
        raise SelectionFailed(f'All finalists failed for {spec.model_id}; inspect fit log', fits)
    neg_score, cid, config = reranked[0]
    return {'config': config, 'config_id': cid, 'inner_score': -neg_score, 'fits': fits,
            'finalist_ids': [config_id(c) for c in finalists]}


def evaluate_fold(X, y, split, spec, seeds=(8129, 19391, 39019), *, dataset_id,
                  dataset_hash, code_revision, score='accuracy', sink=None):
    X, y = np.asarray(X), np.asarray(y)
    try:
        selection = select_model(X, y, split, spec, seeds, score, sink)
    except SelectionFailed as exc:
        models = [{'dataset_id': dataset_id, 'dataset_hash': dataset_hash,
                   'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
                   'model_id': spec.model_id, 'model_seed': seed, 'code_revision': code_revision,
                   'stage': 'outer', 'status': 'failed_selection', 'config_id': None,
                   'fit_rows': split['train'], 'test_rows': split['test'], 'exception': str(exc)}
                  for seed in (seeds if spec.stochastic else seeds[:1])]
        for row in models:
            if sink is not None:
                sink(row)
        return {'status': 'failed_selection', 'selection': {'status': 'failed', 'fits': exc.fits},
                'models': models, 'predictions': []}
    predictions, models = [], []
    common = {'dataset_id': dataset_id, 'dataset_hash': dataset_hash,
              'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
              'model_id': spec.model_id, 'config_id': selection['config_id'],
              'code_revision': code_revision, 'view_id': 'ensemble', 'condition': 'clean',
              'perturbation_seed': None}
    for seed in (seeds if spec.stochastic else seeds[:1]):
        row = dict(common, model_seed=seed, config=selection['config'], stage='outer',
                   fit_rows=split['train'], test_rows=split['test'])
        try:
            pred, timing = _fit_predict(spec, selection['config'], seed, X[split['train']], y[split['train']], X[split['test']])
            row.update(timing, status='ok', **metric_values(y[split['test']], pred))
            predictions.extend(dict(common, model_seed=seed, sample_id=int(sample_id),
                                    y_true=y[sample_id].item(), y_pred=np.asarray(label).item())
                               for sample_id, label in zip(split['test'], pred))
        except Exception as exc:
            row.update(status='failed', exception=f'{type(exc).__name__}: {exc}')
        models.append(row)
        if sink is not None:
            sink(row)
    return {'status': 'ok' if all(r['status'] == 'ok' for r in models) else 'failed_outer',
            'selection': selection, 'models': models, 'predictions': predictions}


def paired_corrected_interval(rows, model_a, model_b, metric='accuracy', q=.25, confidence=.95,
                              *, expected_folds, expected_seeds):
    """Average fitting seeds within fold; approximate Nadeau–Bengio t interval."""
    rows = list(rows)
    validate_outer_schedule(rows, expected_folds=expected_folds, expected_seeds=expected_seeds)
    if model_a == model_b or not {model_a, model_b} <= expected_seeds.keys():
        raise ValueError('Declare both distinct models in the expected schedule')
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        if row['model_id'] not in (model_a, model_b):
            continue
        if row.get('status', 'ok') != 'ok':
            raise ValueError('Failed folds must be resolved/reported before inference')
        key = (row['outer_repeat'], row['outer_fold'], row['model_id'])
        seed_key = key + (row['model_seed'],)
        if seed_key in seen:
            raise ValueError('Duplicate fold/seed record')
        seen.add(seed_key)
        groups[key].append(row[metric])
    a = {k[:2]: float(np.mean(v)) for k, v in groups.items() if k[2] == model_a}
    b = {k[:2]: float(np.mean(v)) for k, v in groups.items() if k[2] == model_b}
    if a.keys() != b.keys() or len(a) < 2 or q <= 0 or not 0 < confidence < 1:
        raise ValueError('Need >=2 complete paired folds and valid ratio/confidence')
    differences = np.array([a[k] - b[k] for k in sorted(a)])
    mean = float(np.mean(differences))
    se = float(np.sqrt((1 / len(a) + q) * np.var(differences, ddof=1)))
    half = float(stats.t.ppf((1 + confidence) / 2, len(a) - 1) * se)
    p = float(2 * stats.t.sf(abs(mean / se), len(a) - 1)) if se else (1. if mean == 0 else 0.)
    return {'mean_difference': mean, 'standard_error': se, 'ci_low': mean-half, 'ci_high': mean+half,
            'n_folds': len(a), 'df': len(a)-1, 'test_train_ratio': q, 'p_approximate': p,
            'method': 'seed-averaged corrected resampled t approximation'}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite p values in [0,1]')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def summarize_outer(rows, model_id, metric='accuracy', *, expected_folds, expected_seeds):
    """Outer-fold means/SD; fitting-seed variation is a separate quantity."""
    rows = list(rows)
    validate_outer_schedule(rows, expected_folds=expected_folds,
                            expected_seeds={model_id: expected_seeds})
    groups = defaultdict(dict)
    for row in rows:
        if row['model_id'] != model_id:
            continue
        if row.get('status', 'ok') != 'ok':
            raise ValueError('Failed outer fits preclude a complete summary')
        key = (row['outer_repeat'], row['outer_fold'])
        seed = row['model_seed']
        if seed in groups[key]:
            raise ValueError('Duplicate fold/seed record')
        groups[key][seed] = row[metric]
    if not groups:
        raise ValueError('No model records')
    seed_sets = [set(v) for v in groups.values()]
    if any(s != seed_sets[0] for s in seed_sets):
        raise ValueError('Incomplete fitting seed schedule across folds')
    means = [np.mean(list(v.values())) for v in groups.values()]
    seed_sd = [np.std(list(v.values()), ddof=1) for v in groups.values() if len(v) > 1]
    return {'model_id': model_id, 'metric': metric, 'mean': float(np.mean(means)),
            'outer_fold_sd': float(np.std(means, ddof=1)) if len(means) > 1 else None,
            'mean_within_fold_seed_sd': float(np.mean(seed_sd)) if seed_sd else None,
            'n_folds': len(means), 'seeds_per_fold': len(seed_sets[0])}


def expected_schedule(protocol, registry):
    """Declared design for one dataset, never inferred from available results."""
    return {'expected_folds': [(repeat, fold) for repeat in range(protocol['outer_repeats'])
                               for fold in range(protocol['outer_folds'])],
            'expected_seeds': {name: list(protocol['fit_seeds'] if spec.stochastic
                                          else protocol['fit_seeds'][:1])
                               for name, spec in registry.items()}}


def validate_outer_schedule(rows, *, expected_folds, expected_seeds):
    """Require every planned fold/seed exactly once for each specified model.

    Call on one dataset at a time. Unselected models are ignored, but schedule
    rows may not span multiple datasets. Failures cannot become missing evidence.
    """
    folds = [tuple(key) for key in expected_folds]
    if not folds or len(set(folds)) != len(folds) or any(len(k) != 2 for k in folds):
        raise ValueError('Expected schedule needs unique (repeat, fold) keys')
    if not expected_seeds or any(not seeds or len(set(seeds)) != len(seeds)
                                 for seeds in expected_seeds.values()):
        raise ValueError('Expected schedule needs nonempty unique model seeds')
    expected = {(repeat, fold, model, seed) for repeat, fold in folds
                for model, seeds in expected_seeds.items() for seed in seeds}
    observed = set()
    datasets = set()
    for row in rows:
        if row['model_id'] not in expected_seeds:
            continue
        key = (row['outer_repeat'], row['outer_fold'], row['model_id'], row['model_seed'])
        if key in observed:
            raise ValueError('Duplicate fold/seed record in expected schedule')
        if row.get('status', 'ok') != 'ok':
            raise ValueError('Failed outer fits preclude complete scheduled inference')
        observed.add(key)
        datasets.add(row.get('dataset_id'))
    if len(datasets) > 1:
        raise ValueError('Validate one dataset schedule at a time')
    if observed != expected:
        raise ValueError(f'Incomplete or unexpected fold/seed schedule: '
                         f'{len(expected-observed)} missing, {len(observed-expected)} unexpected')
