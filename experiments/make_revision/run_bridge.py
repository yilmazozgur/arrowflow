"""Selected-configuration ablation and paired contrasts for the v3 bridge family.

prepare   --protocol P --bridge-source B --output O        seal the per-fold selected configurations
smoke     --protocol P --output O [--workers 2]             synthetic end-to-end exercise; never evidence
pilot     --protocol P --bridge-source B --output O         training-only timing on the pilot datasets
ablation  --protocol P --output O --workers 16              fit every variant per dataset, fold and seed
summary   --output O                                        verify every planned record; ablation_summary.json
contrasts --ablation-source O [--matched-source M] --output C   the 14 prespecified contrasts

The selected configuration of every outer fold is reconstructed from the bridge
run's complete inner fit history (reporting.validate_result_records) and resolved
to concrete embed_dim/degree/augment from the outer training partition's shape.
No outer-fold score chooses anything here: the ArrowFlow variants inherit the
bridge selection, and the kNN control selects n_neighbors/weights on inner folds.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from .bridge import ABLATION_VARIANTS, FIXED, ablation_variants, arrowflow_full_factory, resolve_selected
from .evaluation import (ModelSpec, canonical_json, candidate_grid, config_id, dataset_fingerprint,
                         evaluate_fold, holm_adjust, make_splits, metric_values, paired_corrected_interval,
                         summarize_outer, validate_outer_schedule, validate_split)
from .matched import ArtifactWriter, architecture_id, model_seed_schedule, select_symmetric_probe
from .models import array_hash, seed_fit
from .multiview import MultiViewArrowFlow, MultiViewFootruleKNN, borda_aggregate
from .reporting import validate_result_records
from .run_revision import code_revision, environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

PROTOCOL = Path(__file__).with_name('protocols')/'2026-09-12'/'ablation.json'
BRIDGE_PROTOCOL = Path(__file__).with_name('protocols')/'2026-09-12'/'bridge.json'
SOURCE_MODULES = ['experiments.make_revision.bridge', 'experiments.make_revision.multiview',
                  'experiments.make_revision.models', 'experiments.make_revision.comparisons',
                  'experiments.make_revision.evaluation', 'experiments.make_revision.secondary_studies',
                  'experiments.make_revision.matched', 'experiments.make_revision.reporting',
                  'experiments.make_revision.run_revision', 'experiments.make_revision.datasets']
BRIDGE_MODEL = 'arrowflow_full'
CONTROL = 'multiview_footrule_knn'
PREFIXES = {'views1': 1, 'views3': 3, 'views7': 7}       # majority over the first k of the seven fitted views
SEPARATE = ('no_checkpoint', 'no_augment', 'single_view_no_checkpoint_no_augment')
METRICS = ('accuracy', 'error', 'balanced_accuracy', 'macro_f1')
PREDICTION_KEYS = ('dataset_id', 'outer_repeat', 'outer_fold', 'variant_id', 'model_seed', 'sample_id',
                   'y_true', 'y_pred', 'config_id', 'code_revision')
SELECTION_COLUMNS = ('dataset_id', 'model_id', 'outer_repeat', 'outer_fold', 'config_id', 'config', 'fitting_seeds')
SUMMARY_COLUMNS = ('dataset_id', 'variant_id', 'metric', 'mean', 'outer_fold_sd', 'mean_within_fold_seed_sd',
                   'n_folds', 'seeds_per_fold')
CONTRAST_COLUMNS = ('family_index', 'dataset_id', 'kind', 'source', 'model_a', 'model_b', 'metric', 'status',
                    'n_folds', 'df', 'test_train_ratio', 'confidence', 'mean_difference', 'standard_error',
                    'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate', 'method', 'note')


def environment():
    return environment_record(__package__ + '.run_bridge:environment')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, header, rows):
    """Refuse to replace a CSV whose content differs (same rule as write_json)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    content = buffer.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite {path}; use a new output directory')
        return
    with path.open('x') as stream:
        stream.write(content)


def knn_candidates(p):
    grid = {'n_neighbors': list(p['knn_neighbors']), 'weights': list(p['knn_weights'])}
    return candidate_grid(grid, budget=len(grid['n_neighbors']) * len(grid['weights']))


def fold_schedule(p):
    return [(repeat, fold) for repeat in range(p['outer_repeats']) for fold in range(p['outer_folds'])]


# ----------------------------------------------------------------------------- bridge source

def load_bridge(source, *, allow_smoke=False):
    source = Path(source)
    protocol = json.loads((source/'protocol.json').read_text())
    saved_environment = json.loads((source/'environment.json').read_text())
    candidates = json.loads((source/'candidates.json').read_text())
    if not protocol.get('frozen') and not allow_smoke:
        raise ValueError('The bridge source must be a frozen production run')
    if BRIDGE_MODEL not in candidates:
        raise ValueError(f'The bridge source has no {BRIDGE_MODEL} candidate registry')
    entry = candidates[BRIDGE_MODEL]
    spec = ModelSpec(BRIDGE_MODEL, arrowflow_full_factory, entry['candidates'], entry['stochastic'])
    if entry['config_ids'] != [config_id(c) for c in spec.candidates]:
        raise ValueError('Bridge candidate IDs disagree with their configurations')
    files = {name: sha256_file(source/name) for name in ('protocol.json', 'environment.json', 'candidates.json')}
    return {'directory': source, 'protocol': protocol, 'environment': saved_environment, 'spec': spec, 'files': files}


def check_bridge_protocol(p, bridge_protocol):
    declared = p['bridge_source']
    if (bridge_protocol.get('protocol_id') != declared['protocol_id']
            or bridge_protocol.get('production_family') != declared['family']):
        raise ValueError('Bridge source protocol identity differs from the ablation protocol declaration')
    for key in ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds'):
        if bridge_protocol[key] != p[key]:
            raise ValueError(f'Bridge source and ablation protocols disagree on {key}')
    if any(name not in bridge_protocol['datasets'] for name in p['datasets']):
        raise ValueError('Every ablation dataset must belong to the bridge panel')


def selection_record(bridge, name, split, y, manifest):
    """One fold's selected configuration, reconstructed from the complete inner history."""
    stem = f'{name}__{BRIDGE_MODEL}__r{split["outer_repeat"]}f{split["outer_fold"]}'
    path = bridge['directory']/'results'/f'{stem}.json'
    log = path.with_suffix('.fits.jsonl')
    result = json.loads(path.read_text())
    events = [json.loads(line) for line in log.read_text().splitlines()]
    if result['status'] != 'ok':
        raise ValueError(f'{stem}: bridge job status {result["status"]}')
    if canonical_json(events) != canonical_json(result['selection']['fits'] + result['models']):
        raise ValueError(f'{stem}: bridge fit log and result disagree')
    job = {'dataset_id': name, 'model_id': BRIDGE_MODEL,
           'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold']}
    verified = validate_result_records(result, job, split, y, manifest, bridge['spec'], bridge['protocol'],
                                       bridge['environment']['code_revision'])
    selection = result['selection']
    by_seed = defaultdict(dict)
    for row in result['predictions']:
        by_seed[row['model_seed']][row['sample_id']] = row['y_pred']
    seeds = list(bridge['protocol']['fit_seeds'])
    hashes = {str(seed): array_hash(np.asarray([by_seed[seed][sample] for sample in split['test']])) for seed in seeds}
    return {'dataset_id': name, 'model_id': BRIDGE_MODEL, 'outer_repeat': split['outer_repeat'],
            'outer_fold': split['outer_fold'], 'config_id': selection['config_id'], 'config': selection['config'],
            'fitting_seeds': seeds, 'finalist_ids': selection['finalist_ids'], 'inner_score': selection['inner_score'],
            'result_file': f'results/{stem}.json', 'result_sha256': sha256_file(path), 'log_sha256': sha256_file(log),
            'bridge_prediction_hashes': hashes,
            'bridge_accuracy': {str(row['model_seed']): row['accuracy'] for row in verified}}


def selection_rows(selections):
    return [[s['dataset_id'], s['model_id'], s['outer_repeat'], s['outer_fold'], s['config_id'],
             canonical_json(s['config']), canonical_json(s['fitting_seeds'])] for s in selections]


def fit_sources(variants):
    """How each variant's predictions arise from fits; a separately named variant whose
    parameters coincide with views7 (no_augment where augmentation is already off) is not refitted."""
    params = dict(variants)
    sources = {'views7': 'fitted', 'views1': 'prefix_of_views7', 'views3': 'prefix_of_views7',
               'borda_views7': 'borda_of_views7', CONTROL: 'inner_selected'}
    for variant in SEPARATE:
        sources[variant] = 'identical_to_views7' if params[variant] == params['views7'] else 'separate'
    return sources


def planned_job(name, split, record, n_features):
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    variants = ablation_variants(selected)
    return {'dataset_id': name, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
            'stem': f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}', 'config_id': record['config_id'],
            'config': record['config'], 'selected': selected, 'model_seeds': list(record['fitting_seeds']),
            'variants': [{'variant_id': v, 'params': params} for v, params in variants],
            'fit_sources': fit_sources(variants)}


def prepare(output, p, bridge_source, *, allow_smoke=False, purpose='confirmatory'):
    output = Path(output)
    bridge = load_bridge(bridge_source, allow_smoke=allow_smoke)
    check_bridge_protocol(p, bridge['protocol'])
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs = [], []
    for name in p['datasets']:
        X, y, manifest, splits = load_prepared(bridge['directory'], name)
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        if not (output/name/'data.npz').exists():
            shutil.copyfile(bridge['directory']/name/'data.npz', output/name/'data.npz')
        copied_X, copied_y, _, _ = load_prepared(output, name)        # hash-checked against the manifest
        if not (np.array_equal(copied_X, X, equal_nan=True) and np.array_equal(copied_y, y)):
            raise ValueError(f'{name}: copied dataset differs from the bridge source')
        for split in splits:
            validate_split(split, len(y))
            record = selection_record(bridge, name, split, y, manifest)
            selections.append(record)
            jobs.append(planned_job(name, split, record, X.shape[1]))
    write_csv(output/'bridge_selected_configurations.csv', SELECTION_COLUMNS, selection_rows(selections))
    write_json(output/'bridge_selections.json', selections)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'protocol_hash': config_id(p), 'datasets': list(p['datasets']),
        'bridge_source': {'directory': str(bridge['directory'].resolve()),
                          'protocol_id': bridge['protocol'].get('protocol_id'),
                          'protocol_hash': config_id(bridge['protocol']),
                          'code_revision': bridge['environment']['code_revision'],
                          'file_sha256': bridge['files'], 'model_id': BRIDGE_MODEL,
                          'selected_folds': len(selections)},
        'variants': list(ABLATION_VARIANTS), 'knn_candidates': knn_candidates(p)})
    write_json(output/'planned_jobs.json', jobs)
    return jobs


def verify(output, *, allow_smoke=False, environment_check='full'):
    """environment_check 'full' (run: HEAD, software and sources must equal the prepared record) or
    'sources' (summary: like reporting.collect_verified_results, a later manuscript-only commit may
    change HEAD, but every sealed scientific source must be identical)."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    if not p['frozen'] and not (allow_smoke and manifest['purpose'] == 'synthetic_smoke_only'):
        raise ValueError('A frozen reviewed protocol is required')
    if manifest['protocol_hash'] != config_id(p):
        raise ValueError('Protocol seal changed')
    saved, current = json.loads((output/'environment.json').read_text()), environment()
    if environment_check == 'sources':
        saved, current = saved['source_hashes'], current['source_hashes']
    elif environment_check != 'full':
        raise ValueError('environment_check must be full or sources')
    elif allow_smoke:
        saved.pop('code_revision', None)
        current.pop('code_revision', None)
    if saved != current:
        raise ValueError('Source/environment seal changed')
    if (manifest['datasets'] != list(p['datasets']) or manifest['variants'] != list(ABLATION_VARIANTS)
            or manifest['knn_candidates'] != knn_candidates(p)):
        raise ValueError('Prepared manifest disagrees with the protocol')
    selections = json.loads((output/'bridge_selections.json').read_text())
    saved_csv = list(csv.reader(io.StringIO((output/'bridge_selected_configurations.csv').read_text())))
    if saved_csv != [list(SELECTION_COLUMNS)] + [[str(v) for v in row] for row in selection_rows(selections)]:
        raise ValueError('Selected-configuration CSV disagrees with the sealed selections')
    index = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s for s in selections}
    if len(index) != len(selections):
        raise ValueError('Duplicate sealed selections')
    expected = []
    for name in p['datasets']:
        X, y, data, splits = load_prepared(output, name)
        if splits != make_splits(y, p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['split_seed']):
            raise ValueError('Prepared splits differ from the declared fold schedule')
        for split in splits:
            record = index.get((name, split['outer_repeat'], split['outer_fold']))
            if record is None or record['fitting_seeds'] != list(p['fit_seeds']):
                raise ValueError(f'Missing or inconsistent sealed selection for {name} '
                                 f'r{split["outer_repeat"]}f{split["outer_fold"]}')
            expected.append(planned_job(name, split, record, X.shape[1]))
    if json.loads((output/'planned_jobs.json').read_text()) != expected:
        raise ValueError('Planned job schedule changed')
    return p, manifest, expected


# ----------------------------------------------------------------------------- fitting

def fit_views7(params, seed, X_train, y_train, X_test):
    """The seven views fitted once; per-view predictions and class rankings on the query rows."""
    seed_fit(seed)
    start = time.perf_counter()
    model = MultiViewArrowFlow(**params, seed=seed).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    view_predictions, orders = model.predict_views(X_test)
    rankings = [net.predict_class_ranking(o) for (enc, net), o in zip(model.views_, orders)]
    return model, fit_seconds, np.stack(view_predictions), np.stack(rankings)


def derived_predictions(view_predictions, rankings, classes):
    """views1/views3/views7: majority over the first k fitted views (what MultiViewArrowFlow(n_views=k)
    predicts, since view v depends only on derive_seed(seed, 'view', v)); borda_views7: Borda over all seven."""
    views = np.asarray(view_predictions)
    if views.shape[0] != 7:
        raise ValueError('Expected the seven fitted views')
    out = {name: majority(views[:k]) for name, k in PREFIXES.items()}
    out['borda_views7'] = borda_aggregate(list(np.asarray(rankings)), np.asarray(classes))
    return out


def knn_inner_rows(X, y, split, params, seed, candidates):
    """Inner-fold validation accuracy of every kNN candidate; encoders fitted once per inner fold."""
    rows = []
    max_k = max(c['n_neighbors'] for c in candidates)
    for i, inner in enumerate(split['inner']):
        train, validation = inner['train'], inner['validation']
        try:
            seed_fit(seed)
            base = MultiViewFootruleKNN(**params, n_neighbors=max_k, weights='uniform', seed=seed).fit(
                X[train], y[train], sample_ids=train)
            caches = [knn.kneighbors(enc.transform(X[validation])) for enc, knn in base.views_]
        except Exception as exc:
            rows.extend(dict(config=c, config_id=config_id(c), inner_fold=i, model_seed=seed, state='input',
                             status='failed', score=None, exception=f'{type(exc).__name__}: {exc}')
                        for c in candidates)
            continue
        for cfg in candidates:
            row = dict(config=cfg, config_id=config_id(cfg), inner_fold=i, model_seed=seed, state='input',
                       n_validation=len(validation))
            try:
                votes = []
                for (enc, knn), (distances, indices) in zip(base.views_, caches):
                    knn.n_neighbors, knn.weights = cfg['n_neighbors'], cfg['weights']
                    votes.append(knn.predict_neighbors(distances, indices))
                pred = majority(votes)
                row.update(status='ok', score=float(np.mean(pred == y[validation])), prediction_hash=array_hash(pred))
            except Exception as exc:
                row.update(status='failed', score=None, exception=f'{type(exc).__name__}: {exc}')
            rows.append(row)
    return rows


def evaluate_job(X, y, split, p, job, *, dataset_hash, code_revision, artifact_root=None, sink=None):
    writer = ArtifactWriter(artifact_root)
    name = job['dataset_id']
    train, test = split['train'], split['test']
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    params = {v['variant_id']: v['params'] for v in job['variants']}
    sources, seeds, candidates = job['fit_sources'], list(p['fit_seeds']), knn_candidates(p)
    common = {'dataset_id': name, 'dataset_hash': dataset_hash, 'outer_repeat': split['outer_repeat'],
              'outer_fold': split['outer_fold'], 'config_id': job['config_id'], 'code_revision': code_revision}
    result = {'status': 'running', 'fits': [], 'selection': {}, 'models': [], 'predictions': [], 'events': [],
              'provenance': dict(common, split_hash=config_id(split), train_ids=train, test_ids=test,
                                 raw_train_hash=array_hash(X_train), raw_test_hash=array_hash(X_test),
                                 training_labels_hash=array_hash(y_train), config=job['config'],
                                 selected=job['selected'], variants=job['variants'], fit_sources=sources,
                                 model_seeds=seeds, knn_candidates=candidates)}

    def emit(stage, record):
        event = {'stage': stage, 'record': record}
        result['events'].append(event)
        if sink is not None:
            sink(event)

    def record_fit(variant, seed, fitted_params, **extra):
        fit = dict(fit_id=f'{variant}__s{seed}', variant_id=variant, model_seed=seed, fit_source=sources[variant],
                   params=fitted_params, training_sample_count=len(y_train), status='ok', **extra)
        result['fits'].append(fit)
        emit('fit', fit)
        return fit

    emit('provenance', result['provenance'])
    predictions, fitted = {}, {}
    try:
        with threadpool_limits(limits=1):
            inner_rows = []
            for seed in seeds:
                model, fit_seconds, views, rankings = fit_views7(params['views7'], seed, X_train, y_train, X_test)
                derived = derived_predictions(views, rankings, model.classes_)
                writer.save(f'views/s{seed}.npz', {'view_predictions': views, 'rankings': rankings,
                                                   'classes': model.classes_})
                timing = dict(fit_seconds=fit_seconds, encoding_seconds=model.encoding_seconds_,
                              training_seconds=model.training_seconds_,
                              validation_ratio_nominal=params['views7']['validation_ratio'])
                # Derived and identical variants reuse the views7 fit: zero timings, so cost sums never double count.
                reused = dict(fit_seconds=0., encoding_seconds=0., training_seconds=0., reused_from=f'views7__s{seed}',
                              validation_ratio_nominal=params['views7']['validation_ratio'])
                for variant in ('views1', 'views3', 'views7', 'borda_views7'):
                    predictions[(variant, seed)] = derived[variant]
                    extra = dict(timing, views_hash=array_hash(views), rankings_hash=array_hash(rankings)) \
                        if variant == 'views7' else reused
                    fitted[(variant, seed)] = record_fit(variant, seed, params[variant], **extra)
                del model
                for variant in SEPARATE:
                    if sources[variant] == 'identical_to_views7':
                        predictions[(variant, seed)] = derived['views7']
                        fitted[(variant, seed)] = record_fit(variant, seed, params[variant], **reused)
                        continue
                    seed_fit(seed)
                    start = time.perf_counter()
                    separate = MultiViewArrowFlow(**params[variant], seed=seed).fit(X_train, y_train)
                    elapsed = time.perf_counter() - start
                    predictions[(variant, seed)] = separate.predict(X_test)
                    fitted[(variant, seed)] = record_fit(
                        variant, seed, params[variant], fit_seconds=elapsed,
                        encoding_seconds=separate.encoding_seconds_, training_seconds=separate.training_seconds_,
                        validation_ratio_nominal=params[variant]['validation_ratio'])
                    del separate
                inner_rows.extend(knn_inner_rows(X, y, split, params[CONTROL], seed, candidates))
            for row in inner_rows:
                emit('inner_score', row)
            chosen = select_symmetric_probe(inner_rows, candidates, inner_folds=len(split['inner']),
                                            seeds=seeds, states=('input',))
            result['selection'] = {CONTROL: dict(chosen, fits=inner_rows)}
            emit('selection', {'model_id': CONTROL, **chosen})
            for seed in seeds:
                seed_fit(seed)
                start = time.perf_counter()
                knn = MultiViewFootruleKNN(**params[CONTROL], **chosen['config'], seed=seed).fit(
                    X_train, y_train, sample_ids=train)
                elapsed = time.perf_counter() - start
                predictions[(CONTROL, seed)] = knn.predict(X_test)
                fitted[(CONTROL, seed)] = record_fit(CONTROL, seed, {**params[CONTROL], **chosen['config']},
                                                     fit_seconds=elapsed, config=chosen['config'],
                                                     config_id=chosen['config_id'])
        for (variant, seed), pred in predictions.items():
            pred = np.asarray(pred)
            fit = fitted[(variant, seed)]
            row = dict(common, variant_id=variant, model_id=variant, model_seed=seed, params=fit['params'],
                       fit_source=fit['fit_source'], stage='outer', status='ok', training_sample_count=len(y_train),
                       prediction_hash=array_hash(pred), **metric_values(y_test, pred))
            result['models'].append(row)
            emit('model', row)
            result['predictions'].extend(
                {'dataset_id': name, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
                 'variant_id': variant, 'model_seed': seed, 'sample_id': int(sample), 'y_true': y[sample].item(),
                 'y_pred': np.asarray(label).item(), 'config_id': job['config_id'], 'code_revision': code_revision}
                for sample, label in zip(test, pred))
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = 'failed'
        result['exception'] = f'{type(exc).__name__}: {exc}'
        emit('terminal_failure', {'exception': result['exception']})
        present = {(r['variant_id'], r['model_seed']) for r in result['models']}
        for variant in ABLATION_VARIANTS:
            for seed in seeds:
                if (variant, seed) not in present:
                    row = dict(common, variant_id=variant, model_id=variant, model_seed=seed, stage='outer',
                               status='failed', exception=result['exception'])
                    result['models'].append(row)
                    emit('model', row)
    result['artifacts'] = writer.files
    return result


def worker(arguments):
    output, job = arguments
    output, stem = Path(output), job['stem']
    log, result_path = output/'logs'/f'{stem}.jsonl', output/'results'/f'{stem}.json'
    prediction_path, root = output/'predictions'/f'{stem}.jsonl', output/'artifacts'/stem
    if any(path.exists() for path in (log, result_path, prediction_path, root)):
        raise FileExistsError(f'Existing ablation job {stem}')
    for path in (log, result_path, prediction_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (job['outer_repeat'], job['outer_fold']))
    p = json.loads((output/'protocol.json').read_text())
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    with log.open('x') as stream:
        def sink(event):
            stream.write(canonical_json(event) + '\n')
            stream.flush()
        result = evaluate_job(X, y, split, p, job, dataset_hash=data['dataset_hash'], code_revision=revision,
                              artifact_root=root, sink=sink)
    records = result.pop('predictions')
    with prediction_path.open('x') as stream:
        for record in records:
            stream.write(canonical_json(record) + '\n')
    result['prediction_file'] = {'path': f'predictions/{stem}.jsonl', 'records': len(records),
                                 'sha256': sha256_file(prediction_path)}
    write_json(result_path, result)
    return str(result_path)


# ----------------------------------------------------------------------------- verification

def validate_job(result, events, job, p, X, y, split, data, revision, root, prediction_path, sealed):
    """Bind every record to the sealed plan, the saved views and the per-example predictions."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def logged(stage):
        return [e['record'] for e in events if e['stage'] == stage]

    require(events == result['events'], 'events differ from log')
    require(result['status'] == 'ok', 'failed terminal job')
    train, test, seeds, candidates = split['train'], split['test'], list(p['fit_seeds']), knn_candidates(p)
    provenance = {'dataset_id': job['dataset_id'], 'dataset_hash': data['dataset_hash'],
                  'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'], 'config_id': job['config_id'],
                  'code_revision': revision, 'split_hash': config_id(split), 'train_ids': train, 'test_ids': test,
                  'raw_train_hash': array_hash(X[train]), 'raw_test_hash': array_hash(X[test]),
                  'training_labels_hash': array_hash(y[train]), 'config': job['config'], 'selected': job['selected'],
                  'variants': job['variants'], 'fit_sources': job['fit_sources'], 'model_seeds': seeds,
                  'knn_candidates': candidates}
    require(result['provenance'] == provenance, 'job provenance disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'],
            'planned configuration disagrees with the sealed bridge selection')
    require(logged('provenance') == [provenance] and logged('fit') == result['fits']
            and logged('model') == result['models'] and logged('inner_score') == result['selection'][CONTROL]['fits'],
            'records differ from log')
    require({e['stage'] for e in events} <= {'provenance', 'fit', 'inner_score', 'selection', 'model'},
            'unexpected event stage')
    expected_keys = {(variant, seed) for variant in ABLATION_VARIANTS for seed in seeds}
    require(Counter((f['variant_id'], f['model_seed']) for f in result['fits']) == Counter({k: 1 for k in expected_keys}),
            'incomplete fit schedule')
    require(Counter((r['variant_id'], r['model_seed']) for r in result['models']) == Counter({k: 1 for k in expected_keys}),
            'incomplete model rows')
    require(prediction_path.is_file() and sha256_file(prediction_path) == result['prediction_file']['sha256']
            and result['prediction_file']['path'] == f'predictions/{job["stem"]}.jsonl', 'prediction file hash')
    records = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    require(len(records) == result['prediction_file']['records'] == len(expected_keys) * len(test),
            'prediction record count')
    vectors, labels = defaultdict(list), set(np.asarray(y).tolist())
    for r in records:
        require(sorted(r) == sorted(PREDICTION_KEYS), 'prediction record schema')
        require(r['dataset_id'] == job['dataset_id'] and (r['outer_repeat'], r['outer_fold']) == (job['outer_repeat'], job['outer_fold'])
                and r['config_id'] == job['config_id'] and r['code_revision'] == revision, 'prediction identity')
        require(r['y_true'] == y[r['sample_id']] and r['y_pred'] in labels, 'prediction truth/label')
        vectors[(r['variant_id'], r['model_seed'])].append((r['sample_id'], r['y_pred']))
    require(set(vectors) == expected_keys, 'prediction variant/seed coverage')
    for items in vectors.values():
        require([sample for sample, _ in items] == test, 'prediction sample order')
    predictions = {k: np.asarray([label for _, label in items]) for k, items in vectors.items()}
    fits = {(f['variant_id'], f['model_seed']): f for f in result['fits']}
    params = {v['variant_id']: v['params'] for v in job['variants']}
    chosen = result['selection'][CONTROL]
    for row in result['models']:
        key = (row['variant_id'], row['model_seed'])
        fit, pred = fits[key], predictions[key]
        require(row['status'] == 'ok' and fit['status'] == 'ok' and row['model_id'] == row['variant_id']
                and row['code_revision'] == revision and row['dataset_hash'] == data['dataset_hash']
                and row['config_id'] == job['config_id'] and row['training_sample_count'] == len(train) == fit['training_sample_count'],
                'model row identity')
        require(row['fit_source'] == fit['fit_source'] == job['fit_sources'][row['variant_id']], 'fit source disagrees with plan')
        expected_params = params[row['variant_id']] if row['variant_id'] != CONTROL else {**params[CONTROL], **chosen['config']}
        require(row['params'] == expected_params and fit['params'] == expected_params,
                'variant parameters disagree with plan/selection')
        require(row['prediction_hash'] == array_hash(pred), 'prediction hash')
        if fit['fit_source'] in ('prefix_of_views7', 'borda_of_views7', 'identical_to_views7'):
            require(fit['fit_seconds'] == fit['encoding_seconds'] == fit['training_seconds'] == 0
                    and fit.get('reused_from') == f'views7__s{row["model_seed"]}',
                    'reused fit must carry zero timing and name its source fit')
        metrics = metric_values(y[test], pred)
        require(all(np.isclose(row[k], v, rtol=0, atol=1e-12) for k, v in metrics.items()), 'metric/prediction disagreement')
    artifacts = {a['path']: a['sha256'] for a in result['artifacts']}
    require(set(artifacts) == {f'views/s{seed}.npz' for seed in seeds} and len(artifacts) == len(result['artifacts']),
            'artifact schedule')
    for path, digest in artifacts.items():
        require((root/path).is_file() and sha256_file(root/path) == digest, 'changed artifact ' + path)
    for seed in seeds:
        with np.load(root/f'views/s{seed}.npz', allow_pickle=False) as arrays:
            views, rankings, classes = arrays['view_predictions'], arrays['rankings'], arrays['classes']
        require(views.shape == (7, len(test)) and rankings.shape == (7, len(test), len(classes)), 'view artifact shape')
        require(fits[('views7', seed)]['views_hash'] == array_hash(views)
                and fits[('views7', seed)]['rankings_hash'] == array_hash(rankings), 'view artifact hashes')
        for variant, expected in derived_predictions(views, rankings, classes).items():
            require(np.array_equal(predictions[(variant, seed)], expected), f'{variant} does not derive from the saved views')
        for variant in SEPARATE:
            if job['fit_sources'][variant] == 'identical_to_views7':
                require(params[variant] == params['views7'], 'identical marker without identical parameters')
                require(np.array_equal(predictions[(variant, seed)], predictions[('views7', seed)]),
                        f'{variant} marked identical to views7 but differs')
            else:
                require(params[variant] != params['views7'], 'separate fit with parameters identical to views7')
    inner = chosen['fits']
    expected_inner = Counter((config_id(c), i, s) for c in candidates for i in range(len(split['inner'])) for s in seeds)
    require(Counter((r['config_id'], r['inner_fold'], r['model_seed']) for r in inner) == expected_inner,
            'incomplete kNN inner schedule')
    require(all(r['config_id'] == config_id(r['config']) and r['state'] == 'input' and r['status'] in ('ok', 'failed')
                for r in inner), 'kNN inner record identity')
    actual = select_symmetric_probe(inner, candidates, inner_folds=len(split['inner']), seeds=seeds, states=('input',))
    require(all(chosen[k] == v for k, v in actual.items()), 'kNN selection disagrees with its inner scores')
    require(logged('selection') == [{'model_id': CONTROL, **actual}], 'selection log')
    for seed in seeds:
        fit = fits[(CONTROL, seed)]
        require(fit['config'] == chosen['config'] and fit['config_id'] == chosen['config_id'], 'kNN outer fit config')
    return {str(seed): array_hash(predictions[('views7', seed)]) == sealed['bridge_prediction_hashes'][str(seed)]
            for seed in seeds}


def collect_results(output, *, allow_smoke=False):
    """Every planned job, log, prediction file and artifact reconciled; failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs = verify(output, allow_smoke=allow_smoke, environment_check='sources')
    selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                  for s in json.loads((output/'bridge_selections.json').read_text())}
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    prepared = {name: load_prepared(output, name) for name in p['datasets']}
    rows, knn, reproduction, issues = defaultdict(list), defaultdict(list), defaultdict(dict), []
    for job in jobs:
        stem = job['stem']
        result_path, log = output/'results'/f'{stem}.json', output/'logs'/f'{stem}.jsonl'
        prediction_path = output/'predictions'/f'{stem}.jsonl'
        missing = [str(path.relative_to(output)) for path in (result_path, log, prediction_path) if not path.exists()]
        if missing:
            issues.append(f'missing {stem}: {", ".join(missing)}')
            continue
        try:
            result = json.loads(result_path.read_text())
            events = [json.loads(line) for line in log.read_text().splitlines()]
            X, y, data, splits = prepared[job['dataset_id']]
            split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (job['outer_repeat'], job['outer_fold']))
            matches = validate_job(result, events, job, p, X, y, split, data, revision, output/'artifacts'/stem,
                                   prediction_path, selections[(job['dataset_id'], job['outer_repeat'], job['outer_fold'])])
            rows[job['dataset_id']].extend(result['models'])
            chosen = result['selection'][CONTROL]
            knn[job['dataset_id']].append({'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'],
                                           'config': chosen['config'], 'config_id': chosen['config_id'],
                                           'inner_score': chosen['inner_score']})
            reproduction[job['dataset_id']][stem] = matches
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    for name in p['datasets']:
        try:
            validate_outer_schedule(rows[name], expected_folds=fold_schedule(p),
                                    expected_seeds={v: list(p['fit_seeds']) for v in ABLATION_VARIANTS})
        except ValueError as exc:
            issues.append(f'{name}: {exc}')
    if issues:
        raise ValueError('Incomplete ablation evidence: ' + '; '.join(issues))
    return {'rows': dict(rows), 'knn_selection': dict(knn), 'reproduction': {k: dict(v) for k, v in reproduction.items()},
            'jobs': jobs, 'protocol': p, 'manifest': manifest, 'code_revision': revision}


def run(output, workers=1, *, allow_smoke=False):
    output = Path(output)
    p, manifest, jobs = verify(output, allow_smoke=allow_smoke)
    if not 1 <= workers <= p['max_workers']:
        raise ValueError('Worker count exceeds the shared limit')
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in pool.map(worker, [(str(output), job) for job in jobs]):
            print(path, flush=True)
    return collect_results(output, allow_smoke=allow_smoke)


def summary(output, *, allow_smoke=False):
    """Seed-within-fold means, outer-fold mean/SD and within-fold seed SD per variant and dataset."""
    collected = collect_results(output, allow_smoke=allow_smoke)
    p = collected['protocol']
    folds, seeds, q = fold_schedule(p), list(p['fit_seeds']), p['test_train_ratio']
    summaries, flat = {}, []
    for name in p['datasets']:
        records, table = collected['rows'][name], {}
        for variant in ABLATION_VARIANTS:
            entry = {'metrics': {m: summarize_outer(records, variant, m, expected_folds=folds, expected_seeds=seeds)
                                 for m in METRICS}}
            if variant != 'views7':
                entry['change_from_views7'] = {}
                for m in ('accuracy', 'error'):
                    interval = paired_corrected_interval(records, variant, 'views7', metric=m, q=q, confidence=p['confidence'],
                                                         expected_folds=folds, expected_seeds={variant: seeds, 'views7': seeds})
                    interval.pop('p_approximate')     # descriptive interval only; the prespecified family carries the tests
                    entry['change_from_views7'][m] = interval
            table[variant] = entry
            flat.extend({'dataset_id': name, 'variant_id': variant, 'metric': m,
                         **{k: entry['metrics'][m][k] for k in SUMMARY_COLUMNS[3:]}} for m in METRICS)
        jobs = [j for j in collected['jobs'] if j['dataset_id'] == name]
        matches = collected['reproduction'][name]
        summaries[name] = {
            'variants': table, 'knn_selection': collected['knn_selection'][name],
            'resolved_configurations': [
                {'outer_repeat': j['outer_repeat'], 'outer_fold': j['outer_fold'], 'config_id': j['config_id'],
                 **{k: j['selected'][k] for k in ('widths', 'learning_rate', 'embed_dim', 'degree', 'augment')},
                 'fit_sources': j['fit_sources']} for j in jobs],
            'views7_reproduces_bridge': {
                'matching_fold_seeds': int(sum(sum(v.values()) for v in matches.values())),
                'total_fold_seeds': int(sum(len(v) for v in matches.values())),
                'mismatches': [f'{stem} seed {seed}' for stem, v in matches.items() for seed, ok in v.items() if not ok]}}
    report = {'purpose': 'selected_configuration_component_ablation', 'code_revision': collected['code_revision'],
              'protocol_id': p.get('protocol_id'), 'bridge_source': collected['manifest']['bridge_source'],
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; '
                             'within-fold seed SD reported separately',
              'change_from_views7': 'seed-averaged corrected resampled t interval of each variant against views7; '
                                    'descriptive, without p values; the prespecified 14-member family (contrasts) '
                                    'carries the tests',
              'views7_reproduces_bridge': 'per fold and seed, whether the views7 refit reproduces the bridge run\'s '
                                          'outer predictions exactly (diagnostic only)',
              'inferential_significance_claims': False, 'summaries': summaries, 'model_rows': collected['rows']}
    return report, flat


# ----------------------------------------------------------------------------- contrasts

def declare_family(datasets, *, probe_architecture=(128,)):
    """The 14 prespecified members: per dataset, views7 - multiview_footrule_knn (ablation) and the
    trained-vs-initial hidden probe of the primary architecture (matched study v3)."""
    datasets = list(datasets)
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError('Declare a nonempty unique dataset panel')
    arch, depth = architecture_id(list(probe_architecture)), len(probe_architecture)
    output = [{'dataset': d, 'kind': 'output_vs_knn', 'source': 'ablation', 'model_a': 'views7',
               'model_b': CONTROL, 'metric': 'accuracy'} for d in datasets]
    probes = [{'dataset': d, 'kind': 'trained_vs_initial', 'source': 'matched_v3',
               'model_a': f'{arch}_d{depth}_trained', 'model_b': f'{arch}_d{depth}_untrained', 'metric': 'accuracy'}
              for d in datasets]
    return output + probes


def compute_contrasts(family, ablation_rows, matched, *, q, confidence, folds, ablation_seeds):
    """matched is None (members pending, never fabricated) or {'rows': {dataset: rows}, 'seeds': {model: seeds}}."""
    out = []
    for index, member in enumerate(family, 1):
        row = {'family_index': index, 'dataset_id': member['dataset'],
               **{k: member[k] for k in ('kind', 'source', 'model_a', 'model_b', 'metric')}}
        if member['kind'] == 'output_vs_knn':
            rows = ablation_rows[member['dataset']]
            seeds = {member['model_a']: list(ablation_seeds), member['model_b']: list(ablation_seeds)}
        elif matched is None:
            out.append(dict(row, status='pending', note='matched_v3 source not supplied; member not computed'))
            continue
        else:
            rows = matched['rows'][member['dataset']]
            seeds = {m: list(matched['seeds'][m]) for m in (member['model_a'], member['model_b'])}
        interval = paired_corrected_interval(rows, member['model_a'], member['model_b'], metric=member['metric'],
                                             q=q, confidence=confidence, expected_folds=folds, expected_seeds=seeds)
        out.append(dict(row, status='computed', confidence=confidence, **interval))
    complete = all(r['status'] == 'computed' for r in out)
    adjusted = holm_adjust([r['p_approximate'] for r in out]) if complete else [None] * len(out)
    for r, h in zip(out, adjusted):
        r['holm_p_approximate'] = h
    return out, complete


def load_matched(source, family, *, folds, q):
    """Trained-vs-initial rows of the matched study v3, cross-checked against its verified summary."""
    source = Path(source)
    protocol = json.loads((source/'protocol.json').read_text())
    saved_environment = json.loads((source/'environment.json').read_text())
    summary_path = source/'summary.json'
    if not summary_path.is_file():
        raise ValueError('The matched source has no verified summary.json; run run_studies summary first')
    if not protocol.get('frozen'):
        raise ValueError('The matched source protocol is not frozen')
    content = json.loads(summary_path.read_text())
    if not isinstance(content, dict) or not isinstance(content.get('primary_contrasts'), list):
        raise ValueError('The matched summary.json has no primary_contrasts block')
    published = content['primary_contrasts']
    widths = protocol['architectures'][protocol['primary_architecture_index']]
    members = [m for m in family if m['kind'] == 'trained_vs_initial']
    expected_a = f'{architecture_id(widths)}_d{len(widths)}_trained'
    if any(m['model_a'] != expected_a for m in members):
        raise ValueError(f'The matched primary architecture ({expected_a}) differs from the declared '
                         f'trained-vs-initial family ({members[0]["model_a"] if members else "none"})')
    if fold_schedule(protocol) != folds or protocol['test_train_ratio'] != q:
        raise ValueError('Matched fold schedule or test/train ratio differs from the contrast family')
    schedule = model_seed_schedule(protocol)
    rows = {}
    for member in members:
        name, pair = member['dataset'], (member['model_a'], member['model_b'])
        collected = []
        for repeat, fold in folds:
            path = source/'results'/f'{name}__r{repeat}f{fold}.json'
            if not path.is_file():
                raise ValueError(f'matched result missing: results/{path.name}')
            result = json.loads(path.read_text())
            if not isinstance(result, dict) or result.get('status') != 'ok' or not isinstance(result.get('models'), list):
                raise ValueError(f'matched job {name} r{repeat}f{fold} is not a complete ok result')
            collected.extend(row for row in result['models'] if row.get('model_id') in pair)
        rows[name] = collected
        matches = [c for c in published if (c.get('dataset_id'), c.get('model_a'), c.get('model_b')) == (name, *pair)]
        if len(matches) != 1:
            raise ValueError(f'The matched summary lacks the {name} trained-vs-initial contrast')
        actual = paired_corrected_interval(collected, *pair, metric='accuracy', q=q, expected_folds=folds,
                                           expected_seeds={m: schedule[m] for m in pair})
        for key in ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_approximate'):
            if key not in matches[0]:
                raise ValueError(f'The matched summary row for {name} lacks {key}')
            if not np.isclose(actual[key], matches[0][key], rtol=0, atol=1e-12):
                raise ValueError(f'matched {name} contrast disagrees with the verified summary ({key})')
    return {'rows': rows, 'seeds': schedule,
            'provenance': {'directory': str(source.resolve()), 'protocol_id': protocol.get('protocol_id'),
                           'protocol_hash': config_id(protocol), 'code_revision': saved_environment['code_revision'],
                           'summary_sha256': sha256_file(summary_path), 'primary_architecture': widths}}


def contrasts(ablation_source, matched_source, output):
    ablation_source, output = Path(ablation_source), Path(output)
    summary_path = ablation_source/'ablation_summary.json'
    if not summary_path.is_file():
        raise ValueError('The ablation source has no verified ablation_summary.json; run summary first')
    report = json.loads(summary_path.read_text())
    p = json.loads((ablation_source/'protocol.json').read_text())
    manifest = json.loads((ablation_source/'manifest.json').read_text())
    if not p.get('frozen') and manifest['purpose'] != 'synthetic_smoke_only':
        raise ValueError('Contrasts require a frozen ablation protocol')
    if report.get('code_revision') != json.loads((ablation_source/'environment.json').read_text())['code_revision']:
        raise ValueError('Ablation summary revision differs from its environment seal')
    if not isinstance(report.get('model_rows'), dict):
        raise ValueError('The ablation summary has no model_rows block')
    folds, q, confidence = fold_schedule(p), p['test_train_ratio'], p['confidence']
    declared = p['contrast_family']
    family = declare_family(p['datasets'], probe_architecture=declared['trained_vs_initial']['probe_architecture'])
    if len(family) != declared['size']:
        raise ValueError('Declared family size differs from the dataset panel')
    matched = load_matched(matched_source, family, folds=folds, q=q) if matched_source else None
    rows, complete = compute_contrasts(family, report['model_rows'], matched, q=q, confidence=confidence,
                                       folds=folds, ablation_seeds=p['fit_seeds'])
    result = {'purpose': 'prespecified_primary_contrast_family_v3', 'family_size': len(family),
              'computed': sum(r['status'] == 'computed' for r in rows), 'pending': sum(r['status'] == 'pending' for r in rows),
              'holm_applied': complete, 'q': q, 'confidence': confidence,
              'sources': {'ablation': {'directory': str(ablation_source.resolve()), 'protocol_id': p.get('protocol_id'),
                                       'summary_sha256': sha256_file(summary_path), 'code_revision': report['code_revision']},
                          'matched_v3': matched['provenance'] if matched else None},
              'contrasts': rows}
    write_json(output/'primary_contrasts_v3.json', result)
    write_csv(output/'primary_contrasts_v3.csv', CONTRAST_COLUMNS,
              [['' if r.get(c) is None else r.get(c) for c in CONTRAST_COLUMNS] for r in rows])
    return result


# ----------------------------------------------------------------------------- pilot and smoke

def runtime_pilot(output, p, bridge_source):
    """Training-only timing of every variant on the pilot datasets' first outer training partition."""
    output = Path(output)
    jobs = prepare(output, p, bridge_source, purpose='training_runtime_only')
    seed, candidates, records = p['fit_seeds'][0], knn_candidates(p), []
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and (j['outer_repeat'], j['outer_fold']) == (0, 0))
        X, y, data, splits = load_prepared(output, name)
        split = splits[0]
        train = split['train']
        query = train[::4]                     # training rows only; the outer test fold is never touched
        params = {v['variant_id']: v['params'] for v in job['variants']}
        seconds, network_fits = {}, {}
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            fit_views7(params['views7'], seed, X[train], y[train], X[query])
            seconds['views7'], network_fits['views7'] = time.perf_counter() - start, 7
            for variant in SEPARATE:
                if job['fit_sources'][variant] == 'identical_to_views7':
                    seconds[variant], network_fits[variant] = 0., 0
                    continue
                seed_fit(seed)
                start = time.perf_counter()
                MultiViewArrowFlow(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, params[variant]['n_views']
            start = time.perf_counter()
            inner = knn_inner_rows(X, y, split, params[CONTROL], seed, candidates)
            chosen = select_symmetric_probe(inner, candidates, inner_folds=len(split['inner']), seeds=[seed], states=('input',))
            MultiViewFootruleKNN(**params[CONTROL], **chosen['config'], seed=seed).fit(X[train], y[train], sample_ids=train).predict(X[query])
            seconds[CONTROL], network_fits[CONTROL] = time.perf_counter() - start, 0
        per_seed = sum(seconds.values())
        records.append({'dataset_id': name, 'dataset_hash': data['dataset_hash'], 'train_ids': train, 'query_ids': query,
                        'config_id': job['config_id'], 'selected': job['selected'], 'fit_sources': job['fit_sources'],
                        'model_seed': seed, 'seconds_by_variant': seconds, 'network_fits_by_variant': network_fits,
                        'network_fits_per_seed': sum(network_fits.values()), 'seconds_per_seed': per_seed,
                        'seconds_per_job_estimate': per_seed * len(p['fit_seeds']), 'status': 'ok'})
    piloted = {r['dataset_id']: r for r in records}
    slowest = max(r['seconds_per_job_estimate'] for r in records)
    projection = {}
    for name in p['datasets']:
        per_job = piloted[name]['seconds_per_job_estimate'] if name in piloted else slowest
        projection[name] = {'jobs': len(fold_schedule(p)), 'seconds_per_job': per_job,
                            'basis': 'piloted' if name in piloted else 'slowest piloted dataset (not a bound)'}
    serial = sum(v['jobs'] * v['seconds_per_job'] for v in projection.values())
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'records': records, 'projection': projection,
              'serial_hours': serial / 3600, 'hours_at_16_workers_ideal': serial / 3600 / 16,
              'wallclock_cap_hours': p['wallclock_cap_hours'],
              'estimate_limitations': 'one seed on one training partition per pilot dataset; unpiloted datasets '
                                      'priced at the slowest piloted dataset; contention under 16 workers not included'}
    write_json(output/'pilot.json', report)
    print(json.dumps({k: report[k] for k in ('serial_hours', 'hours_at_16_workers_ideal', 'wallclock_cap_hours')}, indent=2))
    return report


def synthetic_bridge_source(directory, *, samples=240, seed=33):
    """A tiny bridge run in the production format (selection history, outer fits, predictions)."""
    directory = Path(directory)
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], samples // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    features, labels = [f'x{i}' for i in range(4)], ['0', '1', '2']
    manifest = {'dataset_id': 'synthetic', 'purpose': 'synthetic_smoke_only', 'feature_names': features,
                'label_map': labels, 'shape': [len(y), 4], 'class_counts': [len(y) // 3] * 3,
                'dataset_hash': dataset_fingerprint(X, y, features, labels)}
    protocol = dict(json.loads(BRIDGE_PROTOCOL.read_text()), datasets=['synthetic'], outer_folds=3,
                    outer_repeats=1, inner_folds=2, frozen=False, purpose='synthetic_smoke_only')
    candidates = [{**FIXED, 'widths': [4], 'learning_rate': .1, 'iterations': 1, 'embed_scale': 1, 'degree_offset': 0}]
    spec = ModelSpec(BRIDGE_MODEL, arrowflow_full_factory, candidates, True)
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'candidates.json', {BRIDGE_MODEL: {'stochastic': True, 'candidates': candidates,
                                                             'config_ids': [config_id(c) for c in candidates]}})
    write_json(directory/'environment.json', environment_record('experiments.make_revision.bridge:bridge_registry'))
    splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
    manifest['splits_hash'] = config_id(splits)
    write_json(directory/'synthetic'/'manifest.json', manifest)
    write_json(directory/'synthetic'/'splits.json', splits)
    if not (directory/'synthetic'/'data.npz').exists():
        np.savez_compressed(directory/'synthetic'/'data.npz', X=X, y=y)
    revision = code_revision()
    for split in splits:
        stem = f'synthetic__{BRIDGE_MODEL}__r{split["outer_repeat"]}f{split["outer_fold"]}'
        destination = directory/'results'/f'{stem}.json'
        log = destination.with_suffix('.fits.jsonl')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            def sink(row):
                stream.write(canonical_json(row) + '\n')
            result = evaluate_fold(X, y, split, spec, protocol['fit_seeds'], dataset_id='synthetic',
                                   dataset_hash=manifest['dataset_hash'], code_revision=revision,
                                   score=protocol['selection_metric'], sink=sink)
        write_json(destination, result)
    return directory


def smoke(output, p, workers=1):
    """Synthetic end-to-end exercise of prepare, run and summary; never manuscript evidence."""
    output = Path(output)
    source = synthetic_bridge_source(output/'synthetic_bridge')
    bridge_protocol = json.loads((source/'protocol.json').read_text())
    tiny = dict(p, datasets=['synthetic'], outer_folds=bridge_protocol['outer_folds'],
                outer_repeats=bridge_protocol['outer_repeats'], inner_folds=bridge_protocol['inner_folds'],
                pilot_datasets=['synthetic'], contrast_family={**p['contrast_family'], 'size': 2}, frozen=False)
    prepare(output, tiny, source, allow_smoke=True, purpose='synthetic_smoke_only')
    run(output, workers, allow_smoke=True)
    report, flat = summary(output, allow_smoke=True)
    write_json(output/'ablation_summary.json', report)
    write_csv(output/'ablation_summary.csv', SUMMARY_COLUMNS, [[r[c] for c in SUMMARY_COLUMNS] for r in flat])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'smoke', 'pilot', 'ablation', 'summary', 'contrasts'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--bridge-source', type=Path)
    parser.add_argument('--ablation-source', type=Path)
    parser.add_argument('--matched-source', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'contrasts':
        if args.ablation_source is None:
            raise ValueError('contrasts requires --ablation-source')
        contrasts(args.ablation_source, args.matched_source, args.output)
        return
    if args.command == 'summary':
        report, flat = summary(args.output)
        write_json(args.output/'ablation_summary.json', report)
        write_csv(args.output/'ablation_summary.csv', SUMMARY_COLUMNS, [[r[c] for c in SUMMARY_COLUMNS] for r in flat])
        return
    p = json.loads(args.protocol.read_text())
    if args.command == 'prepare':
        if args.bridge_source is None:
            raise ValueError('prepare requires --bridge-source')
        prepare(args.output, p, args.bridge_source)
    elif args.command == 'ablation':
        if not p.get('frozen'):
            raise ValueError('The ablation run requires a frozen reviewed protocol')
        if p != json.loads((args.output/'protocol.json').read_text()):
            raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
        if args.bridge_source is not None:
            manifest = json.loads((args.output/'manifest.json').read_text())
            if str(Path(args.bridge_source).resolve()) != manifest['bridge_source']['directory']:
                raise ValueError('--bridge-source differs from the prepared bridge source')
        run(args.output, args.workers)
    elif args.command == 'pilot':
        if args.bridge_source is None:
            raise ValueError('pilot requires --bridge-source')
        with execution_lock():
            runtime_pilot(args.output, p, args.bridge_source)
    else:
        smoke(args.output, p, args.workers)


if __name__ == '__main__':
    main()
