"""Gene study companions: selector prepass, memo-aware pilot projection, frozen-model corruption.

python -m experiments.make_revision.run_gene pilot_projection --output runs/2026-09-12-gene-pilot
python -m experiments.make_revision.run_gene selector_prepass --output runs/2026-09-12-gene --workers 16
python -m experiments.make_revision.run_gene gene_corruption --output runs/2026-09-12-gene --workers 16
python -m experiments.make_revision.run_gene summary --output runs/2026-09-12-gene

selector_prepass computes and stores the gene ranking of every training partition of the prepared splits
(outer and inner) into the shared selector cache before `run`; export ARROWFLOW_GENE_MI_CACHE to that directory
for `run` and `gene_corruption`. gene_corruption consumes a complete, verified run (frozen protocol, every
planned job and fit log): every outer model is refitted deterministically from its recorded selection and
fitting seed, its clean predictions must equal the recorded run predictions, and per-example predictions per
condition are written under <output>/corruption.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from .evaluation import (canonical_json, config_id, expected_schedule, metric_values, paired_corrected_interval,
                         summarize_outer, validate_split)
from .gene import CACHE_KEY_VERSION, GeneCorruptionBank, corruption_schedule, partition_ranking, uses_selector
from .models import array_hash, seed_fit
from .reporting import METRICS, collect_verified_results
from .run_revision import (code_revision, environment_record, execution_lock, get_registry, load_prepared,
                           planned_jobs, write_json)


def _dispatch(function, arguments, workers):
    if not 1 <= workers <= 16:
        raise ValueError('Worker count must be between 1 and 16')
    with execution_lock():
        if workers == 1:
            return [function(argument) for argument in arguments]
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            return list(pool.map(function, arguments))


# ----------------------------------------------------------------------------- selector prepass

def prepass_partitions(protocol):
    return ['outer'] + [f'inner{i}' for i in range(protocol['inner_folds'])]


def partition_rows(split, partition):
    return split['train'] if partition == 'outer' else split['inner'][int(partition[len('inner'):])]['train']


def prepass_worker(arguments):
    """Rank the genes of one training partition into the shared cache; no held-out row is touched."""
    output, name, index, partition, cache_dir = arguments
    previous = os.environ.get('ARROWFLOW_GENE_MI_CACHE')
    os.environ['ARROWFLOW_GENE_MI_CACHE'] = cache_dir
    try:
        X, y, data, splits = load_prepared(Path(output), name)
        split = splits[index]
        rows = partition_rows(split, partition)
        with threadpool_limits(limits=1):
            values, key, source, seconds, cache_file = partition_ranking(X[rows], y[rows])
        return {'dataset_id': name, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
                'partition': partition, 'row_count': len(rows), 'rows_hash': config_id(rows), 'cache_key': key,
                'ranking_hash': array_hash(np.argsort(-values, kind='stable')), 'selection_source': source,
                'selection_seconds': seconds, 'cache_file': cache_file}
    finally:
        if previous is None:
            os.environ.pop('ARROWFLOW_GENE_MI_CACHE', None)
        else:
            os.environ['ARROWFLOW_GENE_MI_CACHE'] = previous


def selector_prepass(output, workers=1, cache_dir=None):
    """Compute and store the selector ranking of every training partition (outer and inner) before `run`.

    Harness jobs are dispatched fold-major, so without the prepass the eight model jobs of a fold would all
    compute the same rankings concurrently. The factory still computes a missing partition itself and records
    selection_source='computed' in its fit row, so the prepass is a runtime measure, never a correctness one.
    """
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    manifest_path = output / 'selector_prepass.json'
    cache = Path(cache_dir or os.environ.get('ARROWFLOW_GENE_MI_CACHE') or output / 'mi_cache')
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        missing = [p['cache_key'] for p in manifest['partitions']
                   if not (Path(manifest['cache_directory']) / f"{p['cache_key']}.npz").is_file()]
        if missing:
            raise ValueError(f'{len(missing)} prepass cache files are missing; remove {manifest_path} to redo the prepass')
        return manifest
    arguments = [(str(output), name, index, partition, str(cache)) for name in protocol['datasets']
                 for index in range(protocol['outer_folds'] * protocol['outer_repeats'])
                 for partition in prepass_partitions(protocol)]
    partitions = _dispatch(prepass_worker, arguments, workers)
    manifest = {'purpose': 'selector_prepass_training_partitions_only_no_heldout_rows',
                'cache_directory': str(cache), 'cache_key_version': CACHE_KEY_VERSION,
                'sklearn': sklearn.__version__, 'numpy': np.__version__, 'code_revision': code_revision(),
                'partition_count': len(partitions),
                'computed': sum(p['selection_source'] == 'computed' for p in partitions),
                'use': 'export ARROWFLOW_GENE_MI_CACHE=<cache_directory> before run and gene_corruption',
                'partitions': partitions}
    write_json(manifest_path, manifest)
    return manifest


# ----------------------------------------------------------------------------- pilot projection

def pilot_projection(output):
    """Runtime projection from run_revision's training-only pilot that charges the selector explicitly.

    The selector cost is measured from fits (or prepass partitions) that computed it, then charged once per
    (job, training partition) for every family that selects genes: with the process memo alone that is
    jobs x (inner_folds + 1) x selector_seconds per family; after the prepass every job hits the shared cache
    and the selector is paid once per partition in the prepass.
    """
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    pilot = json.loads((output / 'pilot.json').read_text())
    registry = get_registry(json.loads((output / 'environment.json').read_text())['registry'], protocol)
    selects = {name: uses_selector(spec) for name, spec in registry.items()}
    jobs = protocol['outer_folds'] * protocol['outer_repeats'] * len(protocol['datasets'])
    partitions = protocol['inner_folds'] + 1
    cap = protocol['wallclock_cap_hours']

    def meta(row):
        return row.get('representation_metadata') or {}
    ok = [r for r in pilot['rows'] if r['status'] == 'ok']
    computed = [float(meta(r)['selection_seconds']) for r in ok if meta(r).get('selection_source') == 'computed']
    hits = [float(meta(r)['selection_seconds']) for r in ok if meta(r).get('selection_source') in ('memo', 'disk')]
    prepass = None
    if (output / 'selector_prepass.json').is_file():
        saved = json.loads((output / 'selector_prepass.json').read_text())
        measured = [float(p['selection_seconds']) for p in saved['partitions'] if p['selection_source'] == 'computed']
        prepass = {'partition_count': saved['partition_count'], 'expected_partitions': jobs * partitions,
                   'complete': saved['partition_count'] == jobs * partitions, 'computed': saved['computed'],
                   'selector_seconds_max': max(measured, default=None),
                   'selector_seconds_mean': float(np.mean(measured)) if measured else None}
    if computed:
        selector_seconds, selector_source = max(computed), 'pilot fits that computed the selector'
    elif prepass and prepass['selector_seconds_max'] is not None:
        selector_seconds, selector_source = prepass['selector_seconds_max'], 'selector_prepass.json computed partitions'
    elif any(selects.values()):
        raise ValueError('No pilot fit or prepass partition computed the selector; run the pilot without '
                         'ARROWFLOW_GENE_MI_CACHE or run selector_prepass first')
    else:
        selector_seconds, selector_source = 0., 'no family selects genes'
    cache_hit_seconds = max(hits, default=0.)
    models = {}
    for model, estimate in pilot['workload_estimates'].items():
        rows = [r for r in ok if r['model_id'] == model]
        if not rows:
            models[model] = {'status': 'no successful pilot fit'}
            continue
        fits = estimate['fits_per_outer']
        other = max(float(r['elapsed_seconds']) - float(meta(r).get('selection_seconds') or 0.) for r in rows)
        first = meta(rows[0]).get('selection_source')
        selector_partitions = partitions if selects.get(model) else 0
        models[model] = {'jobs': jobs, 'fits_per_job': fits, 'uses_selector': bool(selects.get(model)),
                         'first_fit_selection_source': first, 'fit_seconds_excluding_selector': other,
                         'selector_partitions_per_job': selector_partitions,
                         'serial_seconds_process_memo': jobs * (selector_partitions * selector_seconds + fits * other),
                         'serial_seconds_after_prepass': jobs * fits * (other + (cache_hit_seconds if selector_partitions else 0.))}
    measured_models = [m for m, v in models.items() if 'serial_seconds_process_memo' in v]
    selector_families = [m for m in measured_models if models[m]['uses_selector']]
    memo_serial = sum(models[m]['serial_seconds_process_memo'] for m in measured_models)
    prepass_serial = jobs * partitions * selector_seconds
    after_serial = prepass_serial + sum(models[m]['serial_seconds_after_prepass'] for m in measured_models)
    report = {'purpose': 'training_only_runtime_projection_no_heldout_scores', 'wallclock_cap_hours': cap,
              'selector_seconds': selector_seconds, 'selector_seconds_source': selector_source,
              'cache_hit_seconds': cache_hit_seconds, 'selector_partitions_per_job': partitions,
              'families_with_measured_selector': [m for m in selector_families if models[m]['first_fit_selection_source'] == 'computed'],
              'pilot_memo_cleared_between_families': all(models[m]['first_fit_selection_source'] != 'memo' for m in selector_families),
              'models': models, 'prepass_partitions': jobs * partitions, 'prepass': prepass,
              'prepass_serial_hours': prepass_serial / 3600, 'prepass_hours_at_16_workers': prepass_serial / 3600 / 16,
              'serial_hours_process_memo': memo_serial / 3600, 'hours_at_16_workers_process_memo': memo_serial / 3600 / 16,
              'serial_hours_after_prepass': after_serial / 3600, 'hours_at_16_workers_after_prepass': after_serial / 3600 / 16,
              'within_cap_process_memo': memo_serial / 3600 / 16 <= cap,
              'within_cap_after_prepass': after_serial / 3600 / 16 <= cap,
              'limitations': 'Three sampled candidates per family on one training partition; the selector is charged '
                             'once per (job, partition) at the largest measured cost; ideal 16-worker scaling; not a bound.'}
    write_json(output / 'pilot_projection.json', report)
    return report


# ----------------------------------------------------------------------------- frozen-model corruption

def corruption_root(output):
    return Path(output) / 'corruption'


def corruption_jobs(names, protocol, registry):
    jobs = []
    for job in planned_jobs(names, protocol, registry):
        stem = Path(job['result_file']).stem
        jobs.append({**job, 'stem': stem, 'bank': f'banks/{job["dataset_id"]}__r{job["outer_repeat"]}f{job["outer_fold"]}',
                     'prediction_file': f'corruption/predictions/{stem}.jsonl',
                     'corruption_result_file': f'corruption/results/{stem}.json'})
    return jobs


def prepare_corruption(output, registry_path=None):
    """Verify the finished run, build the shared per-fold banks and the job manifest."""
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    environment = json.loads((output / 'environment.json').read_text())
    registry_path = registry_path or environment['registry']
    registry = get_registry(registry_path, protocol)
    current = environment_record(registry_path)
    if current['source_hashes'] != environment['source_hashes']:
        raise ValueError('Scientific sources changed since the run; deterministic refits are not reproducible')
    schedule = corruption_schedule()
    if protocol.get('corruption') != schedule:
        raise ValueError('Protocol corruption block disagrees with the code schedule')
    collect_verified_results(output, protocol['datasets'], protocol, registry)
    root = corruption_root(output)
    for name in protocol['datasets']:
        X, y, manifest, splits = load_prepared(output, name)
        for split in splits:
            bank = GeneCorruptionBank.create(X[split['test']], dataset_id=name, outer_repeat=split['outer_repeat'],
                                             outer_fold=split['outer_fold'], base_seed=schedule['base_seed'],
                                             sigmas=schedule['lognormal_sigmas'])
            destination = root / 'banks' / f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}'
            if destination.exists():
                if json.loads((destination / 'manifest.json').read_text()) != bank.metadata():
                    raise ValueError(f'Saved corruption bank differs: {destination}')
            else:
                bank.save(destination)
    manifest = {'purpose': 'frozen_model_corruption_of_outer_test_partitions', 'schedule': schedule,
                'registry': registry_path, 'run_code_revision': environment['code_revision'],
                'code_revision': code_revision(), 'source_hashes': current['source_hashes'],
                'selector_cache': os.environ.get('ARROWFLOW_GENE_MI_CACHE'),
                'refit': 'seed_fit(seed); factory(selected config, seed).fit(outer training partition); '
                         'clean predictions must equal the recorded run predictions',
                'jobs': corruption_jobs(protocol['datasets'], protocol, registry)}
    write_json(root / 'manifest.json', manifest)
    return manifest


def corruption_worker(arguments):
    output, job = arguments
    output = Path(output)
    root = corruption_root(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    schedule = manifest['schedule']
    spec = get_registry(manifest['registry'], protocol)[job['model_id']]
    destination, predictions_path = output / job['corruption_result_file'], output / job['prediction_file']
    if destination.exists() or predictions_path.exists():
        raise FileExistsError(f'Existing corruption output: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    split = splits[job['outer_repeat'] * protocol['outer_folds'] + job['outer_fold']]
    validate_split(split, len(y))
    train, test = split['train'], split['test']
    result = json.loads((output / job['result_file']).read_text())
    selection, config = result['selection'], result['selection']['config']
    recorded = {(row['model_seed'], row['sample_id']): row['y_pred'] for row in result['predictions']}
    common = {'dataset_id': job['dataset_id'], 'dataset_hash': data['dataset_hash'], 'outer_repeat': job['outer_repeat'],
              'outer_fold': job['outer_fold'], 'model_id': job['model_id'], 'config_id': selection['config_id'],
              'code_revision': manifest['code_revision'], 'view_id': 'ensemble'}
    rank_input = job['model_id'] in schedule['rank_input_models']
    rows, lines, refits = [], [], []
    try:
        bank = GeneCorruptionBank.create(X[test], dataset_id=job['dataset_id'], outer_repeat=job['outer_repeat'],
                                         outer_fold=job['outer_fold'], base_seed=schedule['base_seed'],
                                         sigmas=schedule['lognormal_sigmas'])
        if bank.metadata() != json.loads((root / job['bank'] / 'manifest.json').read_text()):
            raise ValueError('Corruption bank differs from the prepared bank')
        for seed in job['model_seeds']:
            with threadpool_limits(limits=1):
                seed_fit(seed)
                estimator = spec.factory(dict(config), seed)
                start = time.perf_counter()
                estimator.fit(X[train], y[train])
                fit_seconds = time.perf_counter() - start
                refits.append({'model_seed': seed, 'fit_seconds': fit_seconds,
                               'representation_metadata': getattr(estimator, 'representation_metadata_', None)})
                clean = np.asarray(estimator.predict(bank.clean))
                if any(recorded[(seed, int(s))] != np.asarray(p).item() for s, p in zip(test, clean)):
                    raise ValueError(f'Refit predictions differ from the recorded run for seed {seed}')
                for case in bank.cases:
                    start = time.perf_counter()
                    pred = np.asarray(estimator.predict(case.raw))
                    elapsed = time.perf_counter() - start
                    identity = dict(common, model_seed=seed, condition=case.condition, corruption_family=case.family,
                                    severity=case.severity, perturbation_seed=case.draw_seed)
                    rows.append(dict(identity, config=config, raw_query_hash=case.raw_hash, fit_seconds=fit_seconds,
                                     predict_seconds=elapsed, status='ok', prediction_hash=array_hash(pred),
                                     agreement_with_clean=float(np.mean(pred == clean)),
                                     expected_exact_invariance=rank_input and case.family == 'monotone',
                                     **metric_values(y[test], pred)))
                    lines.extend(dict(identity, sample_id=int(s), y_true=y[s].item(), y_pred=np.asarray(p).item())
                                 for s, p in zip(test, pred))
                if not np.array_equal(np.asarray(estimator.predict(bank.clean)), clean):
                    raise ValueError('Frozen model changed under corruption')
        bank.assert_intact()
    except Exception as exc:
        write_json(destination, {'status': 'failed', 'job': job, 'refit_matches_recorded': False,
                                 'exception': f'{type(exc).__name__}: {exc}', 'rows': rows, 'refits': refits})
        return str(destination)
    with predictions_path.open('x') as stream:
        for line in lines:
            stream.write(canonical_json(line) + '\n')
    write_json(destination, {'status': 'ok', 'job': job, 'refit_matches_recorded': True, 'rows': rows,
                             'refits': refits, 'bank': bank.metadata()})
    return str(destination)


def summarize_corruption(output):
    """Recompute every condition metric from per-example predictions; every planned job is required."""
    output = Path(output)
    root = corruption_root(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    registry = get_registry(manifest['registry'], protocol)
    schedule = manifest['schedule']
    conditions = [c['condition'] for c in schedule['conditions']]
    families = {c['condition']: c for c in schedule['conditions']}
    if schedule != corruption_schedule() or protocol.get('corruption') != schedule:
        raise ValueError('Corruption schedule changed')
    if canonical_json(manifest['jobs']) != canonical_json(corruption_jobs(protocol['datasets'], protocol, registry)):
        raise ValueError('Missing or changed corruption job manifest')
    prepared = {name: load_prepared(output, name) for name in protocol['datasets']}
    issues, rows = [], defaultdict(list)
    for job in manifest['jobs']:
        result_path, prediction_path = output / job['corruption_result_file'], output / job['prediction_file']
        try:
            if not result_path.exists():
                raise ValueError('missing corruption result')
            result = json.loads(result_path.read_text())
            if result['status'] != 'ok' or result['refit_matches_recorded'] is not True:
                raise ValueError(result.get('exception', result['status']))
            if not prediction_path.exists():
                raise ValueError('missing per-example predictions')
            selected = json.loads((output / job['result_file']).read_text())['selection']['config_id']
            X, y, data, splits = prepared[job['dataset_id']]
            split = splits[job['outer_repeat'] * protocol['outer_folds'] + job['outer_fold']]
            test = split['test']
            test_set = set(test)
            predictions = {}
            for line in prediction_path.read_text().splitlines():
                record = json.loads(line)
                key = (record['model_seed'], record['condition'], record['sample_id'])
                if (key in predictions or record['sample_id'] not in test_set or record['y_true'] != y[record['sample_id']]
                        or record['model_id'] != job['model_id'] or record['dataset_hash'] != data['dataset_hash']
                        or (record['outer_repeat'], record['outer_fold']) != (job['outer_repeat'], job['outer_fold'])):
                    raise ValueError('invalid prediction record')
                predictions[key] = record['y_pred']
            if predictions.keys() != {(seed, c, s) for seed in job['model_seeds'] for c in conditions for s in test}:
                raise ValueError('incomplete per-example prediction schedule')
            recorded = {(r['model_seed'], r['condition']): r for r in result['rows']}
            if recorded.keys() != {(seed, c) for seed in job['model_seeds'] for c in conditions}:
                raise ValueError('incomplete condition rows')
            for (seed, condition), row in recorded.items():
                pred = np.asarray([predictions[(seed, condition, s)] for s in test])
                clean = np.asarray([predictions[(seed, 'clean', s)] for s in test])
                metrics = metric_values(y[test], pred)
                agreement = float(np.mean(pred == clean))
                if (any(not np.isclose(row[m], v, rtol=0, atol=1e-12) for m, v in metrics.items())
                        or not np.isclose(row['agreement_with_clean'], agreement, rtol=0, atol=1e-12)
                        or row['status'] != 'ok' or row['config_id'] != selected):
                    raise ValueError('condition metrics disagree with per-example predictions or the recorded selection')
                rows[job['dataset_id']].append(dict(row, **metrics, agreement_with_clean=agreement))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            issues.append(f'{job["stem"]}: {exc}')
    if issues:
        raise ValueError('Incomplete corruption evidence: ' + '; '.join(issues))
    design = expected_schedule(protocol, registry)
    folds = design['expected_folds']
    summaries, violations = {}, []
    for name, model_rows in rows.items():
        table = {}
        for model in registry:
            seeds = design['expected_seeds'][model]
            by_condition = defaultdict(list)
            for row in model_rows:
                if row['model_id'] == model:
                    by_condition[row['condition']].append(row)
            table[model] = {}
            for condition in conditions:
                subset = by_condition[condition]
                entry = {'family': families[condition]['family'], 'severity': families[condition]['severity'],
                         'metrics': {m: summarize_outer(subset, model, m, expected_folds=folds, expected_seeds=seeds)
                                     for m in METRICS},
                         'agreement_with_clean': summarize_outer(subset, model, 'agreement_with_clean',
                                                                 expected_folds=folds, expected_seeds=seeds),
                         'expected_exact_invariance': model in schedule['rank_input_models']
                                                      and families[condition]['family'] == 'monotone',
                         'exact_invariance_observed': all(r['agreement_with_clean'] == 1. for r in subset)}
                if condition != 'clean':
                    reference = [dict(r, model_id='clean_reference') for r in by_condition['clean']]
                    changed = [dict(r, model_id='condition') for r in subset]
                    interval = paired_corrected_interval(changed + reference, 'condition', 'clean_reference', metric='error',
                                                         q=protocol['test_train_ratio'], expected_folds=folds,
                                                         expected_seeds={'condition': seeds, 'clean_reference': seeds})
                    entry['change_from_clean'] = {'metric': 'error', **{k: v for k, v in interval.items() if k != 'p_approximate'}}
                if entry['expected_exact_invariance'] and not entry['exact_invariance_observed']:
                    violations.append({'dataset_id': name, 'model_id': model, 'condition': condition})
                table[model][condition] = entry
        summaries[name] = table
    report = {'purpose': 'frozen_model_corruption_metrics_recomputed_from_per_example_predictions',
              'code_revision': manifest['code_revision'], 'run_code_revision': manifest['run_code_revision'],
              'schedule': schedule, 'summaries': summaries, 'invariance_violations': violations,
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; change_from_clean '
                             'is a descriptive seed-averaged corrected resampled t interval on error (no p-values)',
              'inferential_significance_claims': False, 'model_rows': dict(rows)}
    write_json(root / 'summary.json', report)
    return report


def gene_corruption(output, workers=1, registry_path=None):
    output = Path(output)
    manifest = prepare_corruption(output, registry_path)
    for path in _dispatch(corruption_worker, [(str(output), job) for job in manifest['jobs']], workers):
        print(path, flush=True)
    return summarize_corruption(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['selector_prepass', 'pilot_projection', 'gene_corruption', 'summary'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--registry', default=None, help='gene_corruption: defaults to the registry recorded in environment.json')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='selector_prepass: shared selector cache; defaults to ARROWFLOW_GENE_MI_CACHE or <output>/mi_cache')
    args = parser.parse_args(argv)
    if args.command == 'selector_prepass':
        report = selector_prepass(args.output, args.workers, args.cache_dir)
        print(json.dumps({k: report[k] for k in ('cache_directory', 'partition_count', 'computed', 'use')}, indent=2))
    elif args.command == 'pilot_projection':
        report = pilot_projection(args.output)
        print(json.dumps({k: report[k] for k in ('selector_seconds', 'pilot_memo_cleared_between_families',
                                                  'serial_hours_process_memo', 'hours_at_16_workers_process_memo',
                                                  'prepass_hours_at_16_workers', 'serial_hours_after_prepass',
                                                  'hours_at_16_workers_after_prepass', 'wallclock_cap_hours',
                                                  'within_cap_after_prepass')}, indent=2))
    elif args.command == 'gene_corruption':
        report = gene_corruption(args.output, args.workers, args.registry)
        print(json.dumps({'invariance_violations': report['invariance_violations']}, indent=2))
    else:
        report = summarize_corruption(args.output)
        print(json.dumps({'invariance_violations': report['invariance_violations']}, indent=2))


if __name__ == '__main__':
    main()
