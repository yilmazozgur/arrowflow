"""Component ablation of ArrowFlow-kNN at its reconstructed per-fold selections (Task 20B; descriptive).

prepare   --protocol P --reference-source K --output O   seal the per-fold selections and the reference predictions
smoke     --protocol P --output O [--workers 3]           synthetic end-to-end exercise; never evidence
pilot     --protocol P --reference-source K --output O   training-only timing on the pilot datasets
ablation  --protocol P --output O --workers 16           fit every variant per dataset, outer fold and seed
summary   --output O                                     verify every planned record; knn_ablation_summary.json/.csv

The reference is the bridge_knn production run (arrowflow_full_knn). Every outer fold's selected configuration is
reconstructed from that run's complete inner fit history (reporting.validate_result_records) and resolved to concrete
embed_dim/degree/augment from the outer training partition's shape (bridge.resolve_selected). views7 refits ArrowFlow-kNN
at that configuration and must reproduce the reference outer predictions exactly: a mismatch fails its job, cancels the
pending jobs and blocks the summary, which also re-derives every sealed selection from the reference run. No outer-fold
score chooses anything here; every kNN readout is tuned inside its fit on training rows only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from .bridge import arrowflow_full_knn_factory, resolve_selected
from .evaluation import (ModelSpec, _fit_predict, canonical_json, config_id, make_splits, metric_values,
                         paired_corrected_interval, summarize_outer, validate_outer_schedule, validate_split)
from .knn_controls import (MultiViewInputKNN, UntrainedMultiViewArrowFlowKNN, depth_split, pooled_depth_split,
                           reference_pins, synthetic_reference_run, validate_depths)
from .matched import ArtifactWriter
from .models import array_hash, seed_fit
from .multiview import MultiViewArrowFlowKNN
from .reporting import validate_result_records
from .run_bridge import fold_schedule, sha256_file, write_csv
from .run_revision import environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

PROTOCOL = Path(__file__).with_name('protocols')/'2026-09-12'/'knn_ablation.json'
SOURCE_MODULES = ['experiments.make_revision.bridge', 'experiments.make_revision.multiview',
                  'experiments.make_revision.knn_controls', 'experiments.make_revision.models',
                  'experiments.make_revision.comparisons', 'experiments.make_revision.evaluation',
                  'experiments.make_revision.secondary_studies', 'experiments.make_revision.matched',
                  'experiments.make_revision.reporting', 'experiments.make_revision.run_revision',
                  'experiments.make_revision.run_bridge', 'experiments.make_revision.datasets']
REFERENCE_MODEL = 'arrowflow_full_knn'
VARIANTS = ('views7', 'views1', 'views3', 'no_checkpoint', 'no_augment', 'prototype_readout', 'untrained', 'input_knn')
PREFIXES = {'views1': 1, 'views3': 3, 'views7': 7}       # majority over the first k of the seven per-view kNN votes
DERIVED = ('views7', 'views1', 'views3', 'prototype_readout')
SEPARATE = ('no_checkpoint', 'no_augment')
CONTROLS = {'untrained': UntrainedMultiViewArrowFlowKNN, 'input_knn': MultiViewInputKNN}
REUSED_SOURCES = ('prefix_of_views7', 'output_rule_of_views7', 'identical_to_views7')
UNTRAINED_KEYS = ('n_views', 'strategy', 'embed_dim', 'degree', 'widths', 'aggregation')
INPUT_KEYS = ('n_views', 'strategy', 'embed_dim', 'degree', 'aggregation')
METRICS = ('accuracy', 'error', 'balanced_accuracy', 'macro_f1')
DESIGN_KEYS = ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds')
PREDICTION_KEYS = ('dataset_id', 'outer_repeat', 'outer_fold', 'variant_id', 'model_seed', 'sample_id', 'y_true', 'y_pred',
                   'config_id', 'code_revision')
SELECTION_COLUMNS = ('dataset_id', 'model_id', 'outer_repeat', 'outer_fold', 'config_id', 'config', 'selected_widths',
                     'fitting_seeds', 'reference_prediction_hashes')
SUMMARY_COLUMNS = ('dataset_id', 'variant_id', 'metric', 'mean', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds',
                   'seeds_per_fold')


class ReproductionError(RuntimeError):
    """views7 did not reproduce the reference arrowflow_full_knn outer predictions."""


def environment():
    return environment_record(__package__ + '.run_knn_ablation:environment')


# ----------------------------------------------------------------------------- reference source

def load_reference(source, *, allow_smoke=False):
    source = Path(source)
    names = ('protocol.json', 'environment.json', 'candidates.json', 'summary.json')
    missing = [name for name in names if not (source/name).is_file()]
    if missing:
        raise ValueError(f'The reference source {source} is not a complete run (missing {", ".join(missing)})')
    protocol, saved_environment, candidates = (json.loads((source/name).read_text()) for name in names[:3])
    if not protocol.get('frozen') and not allow_smoke:
        raise ValueError('The reference source must be a frozen production run')
    if REFERENCE_MODEL not in candidates:
        raise ValueError(f'The reference source has no {REFERENCE_MODEL} candidate registry')
    entry = candidates[REFERENCE_MODEL]
    spec = ModelSpec(REFERENCE_MODEL, arrowflow_full_knn_factory, entry['candidates'], entry['stochastic'])
    if entry['config_ids'] != [config_id(c) for c in spec.candidates]:
        raise ValueError('Reference candidate IDs disagree with their configurations')
    return {'directory': source, 'protocol': protocol, 'environment': saved_environment, 'spec': spec,
            'files': {name: sha256_file(source/name) for name in names}}


def check_reference(p, reference):
    """The protocol's reference_source pins must name exactly this run, with the same nested design."""
    declared = p['reference_source']
    observed = {'protocol_id': reference['protocol'].get('protocol_id'), 'protocol_sha256': reference['files']['protocol.json'],
                'code_revision': reference['environment']['code_revision'],
                'summary_sha256': reference['files']['summary.json'], 'model_id': REFERENCE_MODEL,
                'family': reference['protocol'].get('production_family')}
    wrong = [f'{key}: declared {declared.get(key)!r}, reference has {value!r}'
             for key, value in observed.items() if declared.get(key) != value]
    if wrong:
        raise ValueError('The reference source does not match the protocol pins: ' + '; '.join(wrong))
    for key in DESIGN_KEYS:
        if reference['protocol'][key] != p[key]:
            raise ValueError(f'Reference and ablation protocols disagree on {key}')
    if any(name not in reference['protocol']['datasets'] for name in p['datasets']):
        raise ValueError('Every ablation dataset must belong to the reference panel')
    return observed


def selection_record(reference, name, split, y, manifest):
    """One fold's selected configuration and reference outer predictions, reconstructed from the complete history."""
    stem = f'{name}__{REFERENCE_MODEL}__r{split["outer_repeat"]}f{split["outer_fold"]}'
    path = reference['directory']/'results'/f'{stem}.json'
    log = path.with_suffix('.fits.jsonl')
    result = json.loads(path.read_text())
    events = [json.loads(line) for line in log.read_text().splitlines()]
    if result['status'] != 'ok':
        raise ValueError(f'{stem}: reference job status {result["status"]}')
    if canonical_json(events) != canonical_json(result['selection']['fits'] + result['models']):
        raise ValueError(f'{stem}: reference fit log and result disagree')
    job = {'dataset_id': name, 'model_id': REFERENCE_MODEL,
           'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold']}
    verified = validate_result_records(result, job, split, y, manifest, reference['spec'], reference['protocol'],
                                       reference['environment']['code_revision'])
    selection = result['selection']
    by_seed = defaultdict(dict)
    for row in result['predictions']:
        by_seed[row['model_seed']][row['sample_id']] = row['y_pred']
    seeds = list(reference['protocol']['fit_seeds'])
    labels = {str(seed): [by_seed[seed][sample] for sample in split['test']] for seed in seeds}
    return {'dataset_id': name, 'model_id': REFERENCE_MODEL, 'outer_repeat': split['outer_repeat'],
            'outer_fold': split['outer_fold'], 'config_id': selection['config_id'], 'config': selection['config'],
            'selected_widths': list(selection['config']['widths']), 'fitting_seeds': seeds,
            'finalist_ids': selection['finalist_ids'], 'inner_score': selection['inner_score'],
            'result_file': f'results/{stem}.json', 'result_sha256': sha256_file(path), 'log_sha256': sha256_file(log),
            'reference_prediction_hashes': {seed: array_hash(np.asarray(values)) for seed, values in labels.items()},
            'reference_predictions': labels,
            'reference_accuracy': {str(row['model_seed']): row['accuracy'] for row in verified},
            'reference_outer_seconds': {str(row['model_seed']): row['fit_seconds'] + row['predict_seconds']
                                        for row in result['models']}}


def selection_rows(selections):
    return [[s['dataset_id'], s['model_id'], s['outer_repeat'], s['outer_fold'], s['config_id'], canonical_json(s['config']),
             canonical_json(s['selected_widths']), canonical_json(s['fitting_seeds']),
             canonical_json(s['reference_prediction_hashes'])] for s in selections]


# ----------------------------------------------------------------------------- plan

def knn_ablation_variants(selected):
    """Eight (variant_id, params) pairs at one resolved ArrowFlow-kNN configuration (resolve_selected output)."""
    base = {k: v for k, v in selected.items() if k not in ('embed_scale', 'degree_offset')}
    missing = {'n_views', 'strategy', 'embed_dim', 'degree', 'widths', 'augment', 'validation_ratio', 'aggregation'} - base.keys()
    if missing:
        raise ValueError(f'Resolve the selected configuration before the ablation; missing {sorted(missing)}')
    if base['n_views'] != 7 or base['aggregation'] != 'majority':
        raise ValueError('The kNN ablation variants are defined for the seven-view majority-vote selected configuration')
    return [('views7', dict(base)), ('views1', {**base, 'n_views': 1}), ('views3', {**base, 'n_views': 3}),
            ('no_checkpoint', {**base, 'validation_ratio': 0}), ('no_augment', {**base, 'augment': False}),
            ('prototype_readout', dict(base)), ('untrained', {k: base[k] for k in UNTRAINED_KEYS}),
            ('input_knn', {k: base[k] for k in INPUT_KEYS})]


def fit_sources(variants):
    """How each variant's predictions arise; a separately named variant whose parameters coincide with views7
    (no_augment where augmentation is already off) is not refitted."""
    params = dict(variants)
    sources = {'views7': 'fitted', 'views1': 'prefix_of_views7', 'views3': 'prefix_of_views7',
               'prototype_readout': 'output_rule_of_views7', 'untrained': 'separate_untrained', 'input_knn': 'separate_input'}
    for variant in SEPARATE:
        sources[variant] = 'identical_to_views7' if params[variant] == params['views7'] else 'separate'
    return sources


def planned_job(name, split, record, n_features):
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    variants = knn_ablation_variants(selected)
    return {'dataset_id': name, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
            'stem': f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}', 'config_id': record['config_id'],
            'config': record['config'], 'selected': selected, 'selected_widths': list(selected['widths']),
            'model_seeds': list(record['fitting_seeds']),
            'variants': [{'variant_id': v, 'params': params} for v, params in variants], 'fit_sources': fit_sources(variants)}


def prepare(output, p, reference_source, *, allow_smoke=False, purpose='confirmatory'):
    output = Path(output)
    reference = load_reference(reference_source, allow_smoke=allow_smoke)
    pins = check_reference(p, reference)
    validate_depths(p['depth_split']['depths'])
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs = [], []
    for name in p['datasets']:
        X, y, manifest, splits = load_prepared(reference['directory'], name)
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        if not (output/name/'data.npz').exists():
            shutil.copyfile(reference['directory']/name/'data.npz', output/name/'data.npz')
        copied_X, copied_y, _, _ = load_prepared(output, name)        # hash-checked against the manifest
        if not (np.array_equal(copied_X, X, equal_nan=True) and np.array_equal(copied_y, y)):
            raise ValueError(f'{name}: copied dataset differs from the reference source')
        for split in splits:
            validate_split(split, len(y))
            record = selection_record(reference, name, split, y, manifest)
            selections.append(record)
            jobs.append(planned_job(name, split, record, X.shape[1]))
    write_csv(output/'reference_selected_configurations.csv', SELECTION_COLUMNS, selection_rows(selections))
    write_json(output/'reference_selections.json', selections)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'protocol_hash': config_id(p), 'datasets': list(p['datasets']), 'variants': list(VARIANTS),
        'reference_source': {'directory': str(reference['directory'].resolve()), **pins,
                             'protocol_hash': config_id(reference['protocol']), 'file_sha256': reference['files'],
                             'selected_folds': len(selections)}})
    write_json(output/'planned_jobs.json', jobs)
    return jobs


def verify(output, *, allow_smoke=False, environment_check='full'):
    """environment_check 'full' (run: HEAD, software and sources must equal the prepared record) or 'sources'
    (summary: a later manuscript-only commit may change HEAD, but every sealed scientific source must be identical)."""
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
    if manifest['datasets'] != list(p['datasets']) or manifest['variants'] != list(VARIANTS):
        raise ValueError('Prepared manifest disagrees with the protocol')
    validate_depths(p['depth_split']['depths'])
    selections = json.loads((output/'reference_selections.json').read_text())
    saved_csv = list(csv.reader(io.StringIO((output/'reference_selected_configurations.csv').read_text())))
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
            if (record is None or record['fitting_seeds'] != list(p['fit_seeds'])
                    or sorted(record['reference_predictions']) != sorted(map(str, p['fit_seeds']))):
                raise ValueError(f'Missing or inconsistent sealed selection for {name} '
                                 f'r{split["outer_repeat"]}f{split["outer_fold"]}')
            for seed, labels in record['reference_predictions'].items():
                if len(labels) != len(split['test']) or array_hash(np.asarray(labels)) != record['reference_prediction_hashes'][seed]:
                    raise ValueError(f'Sealed reference predictions of {name} r{split["outer_repeat"]}f{split["outer_fold"]} '
                                     'disagree with their hashes')
            expected.append(planned_job(name, split, record, X.shape[1]))
    if json.loads((output/'planned_jobs.json').read_text()) != expected:
        raise ValueError('Planned job schedule changed')
    return p, manifest, expected


# ----------------------------------------------------------------------------- fitting

def fit_views7(params, seed, X_train, y_train, X_test):
    """ArrowFlow-kNN at the selected configuration, seeded as the reference harness seeds it: the fitted model, its fit
    seconds, the per-view kNN votes and the per-view output-rule predictions of the same trained networks."""
    seed_fit(seed)
    start = time.perf_counter()
    model = MultiViewArrowFlowKNN(**params, seed=seed).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    knn_views, orders = model.predict_views(X_test)
    output_views = [net.predict_orders(o) for (enc, net), o in zip(model.views_, orders)]
    return model, fit_seconds, np.stack(knn_views), np.stack(output_views)


def derived_predictions(knn_views, output_views):
    """views1/views3/views7: majority over the first k kNN votes (what MultiViewArrowFlowKNN(n_views=k) predicts, since
    view v depends only on derive_seed(seed, 'view', v)); prototype_readout: majority over the seven output rules."""
    knn_views, output_views = np.asarray(knn_views), np.asarray(output_views)
    if knn_views.shape[0] != 7 or output_views.shape != knn_views.shape:
        raise ValueError('Expected the seven fitted views (kNN votes and output-rule predictions of the same networks)')
    out = {name: majority(knn_views[:k]) for name, k in PREFIXES.items()}
    out['prototype_readout'] = majority(output_views)
    return out


def readout_choices(model):
    return [{'view': v, 'config': s['config'], 'config_id': s['config_id'], 'inner_score': s['inner_score']}
            for v, s in enumerate(model.readout_selections_)]


def reproduction_check(predictions, sealed, seed):
    expected = np.asarray(sealed['reference_predictions'][str(seed)])
    predictions = np.asarray(predictions)
    differing = int(np.sum(predictions != expected)) if predictions.shape == expected.shape else len(expected)
    check = {'model_seed': seed, 'n_test': len(expected), 'n_differing': differing,
             'views7_prediction_hash': array_hash(predictions),
             'reference_prediction_hash': sealed['reference_prediction_hashes'][str(seed)]}
    check['reproduced'] = differing == 0 and check['views7_prediction_hash'] == check['reference_prediction_hash']
    return check


def evaluate_job(X, y, split, p, job, sealed, *, dataset_hash, code_revision, artifact_root=None, sink=None):
    writer = ArtifactWriter(artifact_root)
    name = job['dataset_id']
    train, test = split['train'], split['test']
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    params = {v['variant_id']: v['params'] for v in job['variants']}
    sources, seeds = job['fit_sources'], list(p['fit_seeds'])
    common = {'dataset_id': name, 'dataset_hash': dataset_hash, 'outer_repeat': split['outer_repeat'],
              'outer_fold': split['outer_fold'], 'config_id': job['config_id'], 'code_revision': code_revision}
    result = {'status': 'running', 'fits': [], 'reproduction': [], 'models': [], 'predictions': [], 'events': [],
              'provenance': dict(common, split_hash=config_id(split), train_ids=train, test_ids=test,
                                 raw_train_hash=array_hash(X_train), raw_test_hash=array_hash(X_test),
                                 training_labels_hash=array_hash(y_train), config=job['config'], selected=job['selected'],
                                 variants=job['variants'], fit_sources=sources, model_seeds=seeds,
                                 reference_prediction_hashes=sealed['reference_prediction_hashes'])}

    def emit(stage, record):
        event = {'stage': stage, 'record': record}
        result['events'].append(event)
        if sink is not None:
            sink(event)

    def record_fit(variant, seed, **extra):
        fit = dict(fit_id=f'{variant}__s{seed}', variant_id=variant, model_seed=seed, fit_source=sources[variant],
                   params=params[variant], training_sample_count=len(y_train), status='ok', **extra)
        result['fits'].append(fit)
        emit('fit', fit)
        return fit

    emit('provenance', result['provenance'])
    predictions, fitted = {}, {}
    try:
        with threadpool_limits(limits=1):
            for seed in seeds:
                model, fit_seconds, knn_views, output_views = fit_views7(params['views7'], seed, X_train, y_train, X_test)
                derived = derived_predictions(knn_views, output_views)
                check = reproduction_check(derived['views7'], sealed, seed)
                result['reproduction'].append(check)
                emit('reproduction', check)
                if not check['reproduced']:
                    raise ReproductionError(f'views7 does not reproduce the reference {REFERENCE_MODEL} outer predictions '
                                            f'for seed {seed}: {check["n_differing"]} of {check["n_test"]} test rows differ')
                writer.save(f'views/s{seed}.npz', {'knn_view_predictions': knn_views, 'output_view_predictions': output_views,
                                                   'classes': model.classes_})
                timing = dict(fit_seconds=fit_seconds, encoding_seconds=model.encoding_seconds_,
                              training_seconds=model.training_seconds_, readout_seconds=model.readout_seconds_,
                              readout_choices=readout_choices(model), knn_views_hash=array_hash(knn_views),
                              output_views_hash=array_hash(output_views))
                # Derived and identical variants reuse the views7 fit: zero timings, so cost sums never double count.
                reused = dict(fit_seconds=0., encoding_seconds=0., training_seconds=0., readout_seconds=0.,
                              reused_from=f'views7__s{seed}')
                for variant in DERIVED:
                    predictions[(variant, seed)] = derived[variant]
                    fitted[(variant, seed)] = record_fit(variant, seed, **(timing if variant == 'views7' else reused))
                del model
                for variant in SEPARATE:
                    if sources[variant] == 'identical_to_views7':
                        predictions[(variant, seed)] = derived['views7']
                        fitted[(variant, seed)] = record_fit(variant, seed, **reused)
                        continue
                    seed_fit(seed)
                    start = time.perf_counter()
                    separate = MultiViewArrowFlowKNN(**params[variant], seed=seed).fit(X_train, y_train)
                    elapsed = time.perf_counter() - start
                    predictions[(variant, seed)] = separate.predict(X_test)
                    fitted[(variant, seed)] = record_fit(
                        variant, seed, fit_seconds=elapsed, encoding_seconds=separate.encoding_seconds_,
                        training_seconds=separate.training_seconds_, readout_seconds=separate.readout_seconds_,
                        readout_choices=readout_choices(separate))
                    del separate
                for variant, estimator in CONTROLS.items():
                    seed_fit(seed)
                    start = time.perf_counter()
                    control = estimator(**params[variant], seed=seed).fit(X_train, y_train)
                    elapsed = time.perf_counter() - start
                    predictions[(variant, seed)] = control.predict(X_test)
                    fitted[(variant, seed)] = record_fit(
                        variant, seed, fit_seconds=elapsed, encoding_seconds=control.encoding_seconds_, training_seconds=0.,
                        readout_seconds=control.readout_seconds_, readout_choices=readout_choices(control))
                    del control
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
        result['reproduction_failed'] = isinstance(exc, ReproductionError)
        emit('terminal_failure', {'exception': result['exception'], 'reproduction_failed': result['reproduction_failed']})
        present = {(r['variant_id'], r['model_seed']) for r in result['models']}
        for variant in VARIANTS:
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
        raise FileExistsError(f'Existing kNN ablation job {stem}')
    for path in (log, result_path, prediction_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    key = (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == key[1:])
    sealed = next(s for s in json.loads((output/'reference_selections.json').read_text())
                  if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == key)
    p = json.loads((output/'protocol.json').read_text())
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    with log.open('x') as stream:
        def sink(event):
            stream.write(canonical_json(event) + '\n')
            stream.flush()
        result = evaluate_job(X, y, split, p, job, sealed, dataset_hash=data['dataset_hash'], code_revision=revision,
                              artifact_root=root, sink=sink)
    records = result.pop('predictions')
    with prediction_path.open('x') as stream:
        for record in records:
            stream.write(canonical_json(record) + '\n')
    result['prediction_file'] = {'path': f'predictions/{stem}.jsonl', 'records': len(records),
                                 'sha256': sha256_file(prediction_path)}
    write_json(result_path, result)
    return str(result_path), result['status'], bool(result.get('reproduction_failed'))


# ----------------------------------------------------------------------------- verification

def validate_job(result, events, job, p, X, y, split, data, revision, root, prediction_path, sealed):
    """Bind every record to the sealed plan, the saved views, the per-example predictions and the reference predictions;
    returns the number of fitting seeds whose views7 predictions equal the reference exactly (all, or it raises)."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def logged(stage):
        return [e['record'] for e in events if e['stage'] == stage]

    require(events == result['events'], 'events differ from log')
    require(result['status'] == 'ok', 'failed terminal job')
    train, test, seeds = split['train'], split['test'], list(p['fit_seeds'])
    provenance = {'dataset_id': job['dataset_id'], 'dataset_hash': data['dataset_hash'],
                  'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'], 'config_id': job['config_id'],
                  'code_revision': revision, 'split_hash': config_id(split), 'train_ids': train, 'test_ids': test,
                  'raw_train_hash': array_hash(X[train]), 'raw_test_hash': array_hash(X[test]),
                  'training_labels_hash': array_hash(y[train]), 'config': job['config'], 'selected': job['selected'],
                  'variants': job['variants'], 'fit_sources': job['fit_sources'], 'model_seeds': seeds,
                  'reference_prediction_hashes': sealed['reference_prediction_hashes']}
    require(result['provenance'] == provenance, 'job provenance disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'],
            'planned configuration disagrees with the sealed reference selection')
    require(logged('provenance') == [provenance] and logged('fit') == result['fits'] and logged('model') == result['models']
            and logged('reproduction') == result['reproduction'], 'records differ from log')
    require({e['stage'] for e in events} <= {'provenance', 'reproduction', 'fit', 'model'}, 'unexpected event stage')
    require([r['model_seed'] for r in result['reproduction']] == seeds
            and all(r == reproduction_check(sealed['reference_predictions'][str(r['model_seed'])], sealed, r['model_seed'])
                    for r in result['reproduction']), 'views7 reproduction records')
    expected_keys = {(variant, seed) for variant in VARIANTS for seed in seeds}
    require(Counter((f['variant_id'], f['model_seed']) for f in result['fits']) == Counter({k: 1 for k in expected_keys}),
            'incomplete fit schedule')
    require(Counter((r['variant_id'], r['model_seed']) for r in result['models']) == Counter({k: 1 for k in expected_keys}),
            'incomplete model rows')
    require(prediction_path.is_file() and sha256_file(prediction_path) == result['prediction_file']['sha256']
            and result['prediction_file']['path'] == f'predictions/{job["stem"]}.jsonl', 'prediction file hash')
    records = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    require(len(records) == result['prediction_file']['records'] == len(expected_keys) * len(test), 'prediction record count')
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
    for row in result['models']:
        key = (row['variant_id'], row['model_seed'])
        fit, pred = fits[key], predictions[key]
        require(row['status'] == 'ok' and fit['status'] == 'ok' and row['model_id'] == row['variant_id']
                and row['code_revision'] == revision and row['dataset_hash'] == data['dataset_hash']
                and row['config_id'] == job['config_id'] and row['training_sample_count'] == len(train) == fit['training_sample_count'],
                'model row identity')
        require(row['fit_source'] == fit['fit_source'] == job['fit_sources'][row['variant_id']], 'fit source disagrees with plan')
        require(row['params'] == fit['params'] == params[row['variant_id']], 'variant parameters disagree with plan')
        require(row['prediction_hash'] == array_hash(pred), 'prediction hash')
        if fit['fit_source'] in REUSED_SOURCES:
            require(fit['fit_seconds'] == fit['encoding_seconds'] == fit['training_seconds'] == fit['readout_seconds'] == 0
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
            knn_views, output_views = arrays['knn_view_predictions'], arrays['output_view_predictions']
        require(knn_views.shape == output_views.shape == (7, len(test)), 'view artifact shape')
        require(fits[('views7', seed)]['knn_views_hash'] == array_hash(knn_views)
                and fits[('views7', seed)]['output_views_hash'] == array_hash(output_views), 'view artifact hashes')
        for variant, expected in derived_predictions(knn_views, output_views).items():
            require(np.array_equal(predictions[(variant, seed)], expected), f'{variant} does not derive from the saved views')
        for variant in SEPARATE:
            if job['fit_sources'][variant] == 'identical_to_views7':
                require(params[variant] == params['views7'], 'identical marker without identical parameters')
                require(np.array_equal(predictions[(variant, seed)], predictions[('views7', seed)]),
                        f'{variant} marked identical to views7 but differs')
            else:
                require(params[variant] != params['views7'], 'separate fit with parameters identical to views7')
        require(reproduction_check(predictions[('views7', seed)], sealed, seed)['reproduced'],
                f'views7 does not reproduce the reference {REFERENCE_MODEL} predictions (seed {seed})')
    return len(seeds)


def collect_results(output, *, allow_smoke=False):
    """Every planned job, log, prediction file and artifact reconciled, and every sealed selection re-derived from the
    reference run; failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs = verify(output, allow_smoke=allow_smoke, environment_check='sources')
    reference = load_reference(manifest['reference_source']['directory'], allow_smoke=allow_smoke)
    check_reference(p, reference)
    selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                  for s in json.loads((output/'reference_selections.json').read_text())}
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    prepared = {name: load_prepared(output, name) for name in p['datasets']}
    rows, reproduced, issues = defaultdict(list), defaultdict(int), []
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
            sealed = selections[(job['dataset_id'], job['outer_repeat'], job['outer_fold'])]
            if selection_record(reference, job['dataset_id'], split, y, data) != sealed:
                raise ValueError('the sealed selection differs from the one re-derived from the reference run')
            reproduced[job['dataset_id']] += validate_job(result, events, job, p, X, y, split, data, revision,
                                                          output/'artifacts'/stem, prediction_path, sealed)
            rows[job['dataset_id']].extend(result['models'])
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    for name in p['datasets']:
        try:
            validate_outer_schedule(rows[name], expected_folds=fold_schedule(p),
                                    expected_seeds={v: list(p['fit_seeds']) for v in VARIANTS})
        except ValueError as exc:
            issues.append(f'{name}: {exc}')
    if issues:
        raise ValueError('Incomplete kNN ablation evidence: ' + '; '.join(issues))
    return {'rows': dict(rows), 'reproduced': dict(reproduced), 'jobs': jobs, 'protocol': p, 'manifest': manifest,
            'code_revision': revision}


def run(output, workers=1, *, allow_smoke=False):
    """Every planned job on spawned single-thread workers; a views7 reproduction failure cancels the pending jobs."""
    output = Path(output)
    p, manifest, jobs = verify(output, allow_smoke=allow_smoke)
    if not 1 <= workers <= p['max_workers']:
        raise ValueError('Worker count exceeds the shared limit')
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, (str(output), job)) for job in jobs]
        try:
            for future in as_completed(futures):
                path, status, reproduction_failed = future.result()
                print(path, status, flush=True)
                if reproduction_failed:
                    raise ReproductionError(f'{path}: views7 did not reproduce the reference predictions; pending jobs '
                                            'are cancelled and this run cannot be summarized')
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return collect_results(output, allow_smoke=allow_smoke)


def summary(output, *, allow_smoke=False):
    """Seed-within-fold means, outer-fold mean/SD, descriptive intervals against views7 and the depth split."""
    collected = collect_results(output, allow_smoke=allow_smoke)
    p = collected['protocol']
    folds, seeds, q, confidence = fold_schedule(p), list(p['fit_seeds']), p['test_train_ratio'], p['confidence']
    depths = validate_depths(p['depth_split']['depths'])
    summaries, flat, split_by_dataset = {}, [], {}
    for name in p['datasets']:
        records, table = collected['rows'][name], {}
        for variant in VARIANTS:
            entry = {'metrics': {m: summarize_outer(records, variant, m, expected_folds=folds, expected_seeds=seeds)
                                 for m in METRICS}}
            if variant != 'views7':
                entry['change_from_views7'] = {}
                for m in ('accuracy', 'error'):
                    interval = paired_corrected_interval(records, variant, 'views7', metric=m, q=q, confidence=confidence,
                                                         expected_folds=folds, expected_seeds={variant: seeds, 'views7': seeds})
                    interval.pop('p_approximate')     # descriptive interval only
                    entry['change_from_views7'][m] = interval
            table[variant] = entry
            flat.extend({'dataset_id': name, 'variant_id': variant, 'metric': m,
                         **{k: entry['metrics'][m][k] for k in SUMMARY_COLUMNS[3:]}} for m in METRICS)
        jobs = [j for j in collected['jobs'] if j['dataset_id'] == name]
        widths = {(j['outer_repeat'], j['outer_fold']): j['selected_widths'] for j in jobs}
        split_by_dataset[name] = depth_split(records, 'untrained', 'views7', widths, depths=depths, folds=folds,
                                             seeds={'untrained': seeds, 'views7': seeds}, q=q, confidence=confidence)
        summaries[name] = {
            'variants': table,
            'resolved_configurations': [
                {'outer_repeat': j['outer_repeat'], 'outer_fold': j['outer_fold'], 'config_id': j['config_id'],
                 **{k: j['selected'][k] for k in ('widths', 'learning_rate', 'embed_dim', 'degree', 'augment')},
                 'fit_sources': j['fit_sources']} for j in jobs],
            'views7_reproduces_reference': {'matching_fold_seeds': collected['reproduced'][name],
                                            'total_fold_seeds': len(jobs) * len(seeds)}}
    report = {'purpose': 'knn_anchored_component_ablation', 'code_revision': collected['code_revision'],
              'protocol_id': p.get('protocol_id'), 'reference_source': collected['manifest']['reference_source'],
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; within-fold seed SD '
                             'reported separately',
              'change_from_views7': 'seed-averaged corrected resampled t interval of each variant minus views7; '
                                    'descriptive, without p values or multiplicity adjustment',
              'views7_reproduces_reference': 'hard check: every fold and seed of views7 reproduced the reference '
                                             f'{REFERENCE_MODEL} outer predictions exactly (the summary refuses otherwise)',
              'depth_split': {'difference': 'untrained minus views7 accuracy, fitting seeds averaged within fold',
                              'grouping': 'the hidden widths of the reconstructed selection of each outer fold',
                              'status': 'descriptive; no p values', 'depths': depths, 'by_dataset': split_by_dataset,
                              'pooled': pooled_depth_split(split_by_dataset, depths)},
              'inferential_significance_claims': False, 'summaries': summaries, 'model_rows': collected['rows']}
    return report, flat


def write_summary(output, *, allow_smoke=False):
    report, flat = summary(output, allow_smoke=allow_smoke)
    write_json(Path(output)/'knn_ablation_summary.json', report)
    write_csv(Path(output)/'knn_ablation_summary.csv', SUMMARY_COLUMNS, [[r[c] for c in SUMMARY_COLUMNS] for r in flat])
    return report


# ----------------------------------------------------------------------------- pilot and smoke

def reproduction_probe(reference, name):
    """The first saved inner fit of the reference run's first outer fold of `name`, refitted and compared exactly: the
    inner validation score and every view's readout selection record. Inner rows are outer training rows only."""
    X, y, manifest, splits = load_prepared(reference['directory'], name)
    split = splits[0]
    stem = f'{name}__{REFERENCE_MODEL}__r{split["outer_repeat"]}f{split["outer_fold"]}'
    fit = json.loads((reference['directory']/'results'/f'{stem}.json').read_text())['selection']['fits'][0]
    inner = split['inner'][fit['inner_fold']]
    if fit['fit_rows'] != inner['train'] or fit['validation_rows'] != inner['validation']:
        raise ValueError(f'{stem}: the saved inner fit rows differ from the prepared inner split')
    start = time.perf_counter()
    predictions, timing = _fit_predict(reference['spec'], fit['config'], fit['model_seed'], X[inner['train']],
                                       y[inner['train']], X[inner['validation']])
    score = metric_values(y[inner['validation']], predictions)[reference['protocol']['selection_metric']]
    identical = canonical_json(timing['representation_metadata']['views']) == canonical_json(fit['representation_metadata']['views'])
    return {'dataset_id': name, 'result_file': f'results/{stem}.json', 'config_id': fit['config_id'],
            'model_seed': fit['model_seed'], 'inner_fold': fit['inner_fold'], 'reference_score': fit['score'],
            'refit_score': score, 'readout_selections_identical': identical,
            'reproduced': score == fit['score'] and identical, 'seconds': time.perf_counter() - start}


def runtime_pilot(output, p, reference_source):
    """Training-only timing of every variant on the pilot datasets' first outer training partition (one seed), projected
    two ways: the idle pilot seconds (unpiloted datasets at the slowest piloted one) and the reference run's realized
    outer-fit seconds for the same configurations under 16 workers, scaled by each piloted all-variants/views7 ratio."""
    output = Path(output)
    jobs = prepare(output, p, reference_source, purpose='training_runtime_only')
    reference = load_reference(reference_source)
    selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                  for s in json.loads((output/'reference_selections.json').read_text())}
    seed, records = p['fit_seeds'][0], []
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and (j['outer_repeat'], j['outer_fold']) == (0, 0))
        X, y, data, splits = load_prepared(output, name)
        train = splits[0]['train']
        query = train[::4]                     # training rows only; the outer test fold is never touched
        params = {v['variant_id']: v['params'] for v in job['variants']}
        seconds, network_fits = {}, {}
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            _, _, knn_views, output_views = fit_views7(params['views7'], seed, X[train], y[train], X[query])
            derived_predictions(knn_views, output_views)
            seconds['views7'], network_fits['views7'] = time.perf_counter() - start, 7
            for variant in SEPARATE:
                if job['fit_sources'][variant] == 'identical_to_views7':
                    seconds[variant], network_fits[variant] = 0., 0
                    continue
                seed_fit(seed)
                start = time.perf_counter()
                MultiViewArrowFlowKNN(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, 7
            for variant, estimator in CONTROLS.items():
                seed_fit(seed)
                start = time.perf_counter()
                estimator(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, 0
        per_seed = sum(seconds.values())
        records.append({'dataset_id': name, 'dataset_hash': data['dataset_hash'], 'train_ids': train, 'query_ids': query,
                        'config_id': job['config_id'], 'selected': job['selected'], 'fit_sources': job['fit_sources'],
                        'model_seed': seed, 'seconds_by_variant': seconds, 'network_fits_by_variant': network_fits,
                        'network_fits_per_seed': sum(network_fits.values()), 'seconds_per_seed': per_seed,
                        'seconds_per_job_estimate': per_seed * len(p['fit_seeds']),
                        'all_variants_to_views7_ratio': per_seed / seconds['views7'], 'status': 'ok'})
    piloted = {r['dataset_id']: r for r in records}
    slowest = max(r['seconds_per_job_estimate'] for r in records)
    harness, calibrated = {}, {}
    for name in p['datasets']:
        name_jobs = [j for j in jobs if j['dataset_id'] == name]
        per_job = piloted[name]['seconds_per_job_estimate'] if name in piloted else slowest
        harness[name] = {'jobs': len(name_jobs), 'seconds_per_job': per_job,
                         'basis': 'piloted' if name in piloted else 'slowest piloted dataset (not a bound)'}
        augment_source = name_jobs[0]['fit_sources']['no_augment']
        alike = [r['all_variants_to_views7_ratio'] for r in records if r['fit_sources']['no_augment'] == augment_source]
        ratio = (piloted[name]['all_variants_to_views7_ratio'] if name in piloted
                 else max(alike or [r['all_variants_to_views7_ratio'] for r in records]))
        realized = sum(sum(selections[(name, j['outer_repeat'], j['outer_fold'])]['reference_outer_seconds'].values())
                       for j in name_jobs)
        calibrated[name] = {'jobs': len(name_jobs), 'reference_views7_seconds': realized, 'all_variants_to_views7_ratio': ratio,
                            'seconds': realized * ratio,
                            'basis': 'piloted ratio' if name in piloted else
                                     f'largest piloted ratio among datasets whose no_augment is {augment_source}'}
    serial = sum(v['jobs'] * v['seconds_per_job'] for v in harness.values())
    calibrated_serial = sum(v['seconds'] for v in calibrated.values())
    probe = reproduction_probe(reference, p['pilot_datasets'][0])
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'records': records,
              'harness_projection': {'datasets': harness, 'serial_hours': serial / 3600,
                                     'hours_at_16_workers_ideal': serial / 3600 / 16},
              'calibrated_projection': {'datasets': calibrated, 'serial_hours': calibrated_serial / 3600,
                                        'hours_at_16_workers': calibrated_serial / 3600 / 16,
                                        'basis': 'the reference run\'s realized outer fit and predict seconds of every '
                                                 'fold and seed (the views7 configuration and rows, measured under 16 '
                                                 'workers) times the piloted all-variants/views7 time ratio'},
              'wallclock_cap_hours': p['wallclock_cap_hours'], 'reproduction_probe': probe,
              'estimate_limitations': 'one seed on one training partition per pilot dataset on an idle machine; the '
                                      'harness projection prices unpiloted datasets at the slowest piloted one and '
                                      'includes no contention; both projections assume ideal packing of jobs on 16 workers'}
    write_json(output/'pilot.json', report)
    print(json.dumps({'harness_hours_at_16_workers': report['harness_projection']['hours_at_16_workers_ideal'],
                      'calibrated_hours_at_16_workers': report['calibrated_projection']['hours_at_16_workers'],
                      'wallclock_cap_hours': p['wallclock_cap_hours'], 'reproduction_probe': probe['reproduced']}, indent=2))
    return report


SMOKE_CANDIDATES = [{'aggregation': 'majority', 'batch_size': 32, 'degree_offset': 0, 'embed_scale': 1, 'iterations': 1,
                     'learning_rate': .1, 'n_views': 7, 'strategy': 'diverse', 'validation_ratio': .1, 'widths': widths}
                    for widths in ([4], [2, 4])]


def smoke(output, p, workers=3):
    """Synthetic reference run (240 rows, so augmentation is on), then prepare, run and summary; never evidence."""
    output = Path(output)
    with execution_lock():
        source = synthetic_reference_run(output/'synthetic_reference', SMOKE_CANDIDATES, workers=workers, samples=240)
    reference_protocol = json.loads((source/'protocol.json').read_text())
    tiny = dict(p, datasets=['synthetic'], pilot_datasets=['synthetic'], frozen=False, purpose='synthetic_smoke_only',
                **{key: reference_protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds')},
                reference_source={**p['reference_source'], **reference_pins(source),
                                  'family': reference_protocol['production_family']},
                depth_split={**p['depth_split'], 'depths': [c['widths'] for c in SMOKE_CANDIDATES]})
    prepare(output, tiny, source, allow_smoke=True, purpose='synthetic_smoke_only')
    run(output, workers, allow_smoke=True)
    return write_summary(output, allow_smoke=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'smoke', 'pilot', 'ablation', 'summary'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--reference-source', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'summary':
        report = write_summary(args.output)
        for name, entry in report['summaries'].items():
            counts = entry['views7_reproduces_reference']
            print(f"{name}: views7 reproduced {counts['matching_fold_seeds']}/{counts['total_fold_seeds']} fold-seeds")
        return
    p = json.loads(args.protocol.read_text())
    if args.command == 'prepare':
        if args.reference_source is None:
            raise ValueError('prepare requires --reference-source')
        prepare(args.output, p, args.reference_source)
    elif args.command == 'ablation':
        if not p.get('frozen'):
            raise ValueError('The kNN ablation run requires a frozen reviewed protocol')
        if p != json.loads((args.output/'protocol.json').read_text()):
            raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
        if args.reference_source is not None:
            manifest = json.loads((args.output/'manifest.json').read_text())
            if str(Path(args.reference_source).resolve()) != manifest['reference_source']['directory']:
                raise ValueError('--reference-source differs from the prepared reference source')
        run(args.output, args.workers)
    elif args.command == 'pilot':
        if args.reference_source is None:
            raise ValueError('pilot requires --reference-source')
        with execution_lock():
            runtime_pilot(args.output, p, args.reference_source)
    else:
        if not 1 <= args.workers <= 16:
            raise ValueError('Worker count must be between 1 and 16')
        smoke(args.output, p, args.workers)


if __name__ == '__main__':
    main()
