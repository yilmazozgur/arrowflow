"""G4 follow-up experiments (a) and (c): the component ablation of ArrowFlow-kNN on the dedup and artificial datasets, at its
reconstructed per-fold selections (descriptive).

prepare   --protocol P --reference R --output O               seal the per-fold selections and reference predictions of run R
ablation  --protocol P --reference R --output O --workers N   fit every variant per dataset, outer fold and seed
summary   --output O                                          verify every planned record; <family>_ablation_summary.json/.csv

The design is run_newdata_ablation's with one reference: the run of the same frozen family protocol, whose ablation block
declares the variants views7, views1, views3, no_checkpoint, no_augment, prototype_readout, untrained and input_knn, the fit
reuse, the kNN readout, the depth split and the reporting. The per-job functions are run_knn_ablation's, imported as
run_newdata_ablation imports them (load_reference, check_reference, selection_record, selection_rows, planned_job, worker,
validate_job); run_newdata_ablation.calibrated_projection prices the planned jobs at the run's realized outer seconds for the
manifest. What run_newdata_ablation binds to its two newdata batch references and the newdata pins is re-stated here for
the family run and the extra_data pins.

Each fold's selection is reconstructed from the complete inner fit history of the run (reporting.validate_result_records)
and resolved from the outer training partition's shape (bridge.resolve_selected). In production the protocol must be the
committed frozen family protocol, the run must hold it byte for byte, and every dataset is reloaded through its pinned
extra_data loader and must equal the run's prepared features (NaN equal to NaN) and labels. views7 refits ArrowFlow-kNN at
the selection and must reproduce the run's outer predictions exactly for every dataset, outer fold and fitting seed: a
mismatch fails its job, cancels the pending jobs and blocks the summary, which also re-derives every sealed selection from
the run. No outer-fold score chooses anything; every kNN readout is tuned inside its fit on training rows only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import zipfile
import numpy as np
from . import extra_runs as er
from . import run_knn_ablation as base
from . import run_newdata_ablation as rna
from .evaluation import canonical_json, config_id, make_splits, paired_corrected_interval, summarize_outer, validate_outer_schedule, validate_split
from .extra_data import DatasetIdentityError
from .knn_controls import depth_split, pooled_depth_split, validate_depths
from .run_bridge import fold_schedule, sha256_file, write_csv
from .run_revision import environment_record, execution_lock, load_prepared, write_json

SOURCE_MODULES = base.SOURCE_MODULES + ['experiments.make_revision.run_knn_ablation', 'experiments.make_revision.run_newdata_ablation',
                                        'experiments.make_revision.newdata', 'experiments.make_revision.projected_knn',
                                        'experiments.make_revision.extra_data', 'experiments.make_revision.extra_runs']
REFERENCE_MODEL = base.REFERENCE_MODEL
VARIANTS = base.VARIANTS
PIN_KEYS = ('protocol_id', 'protocol_sha256', 'code_revision', 'summary_sha256', 'model_id', 'family')
ABLATION_RECORDS = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json',
                    'reference_selected_configurations.csv')
REALIZED_RULE = ('informational, not a decision: each planned job priced at the family run\'s realized outer fit and predict seconds '
                 'of its fold and seeds times the piloted all-variants/views7 time ratio of its dataset (the frozen protocol\'s '
                 'projection.ablation_pilot), simulated on the family workers in planned order')


def environment():
    return environment_record(__package__ + '.extra_ablation:environment')


def summary_names(family):
    return f'{family}_ablation_summary.json', f'{family}_ablation_summary.csv'


def check_protocol(p, *, allow_smoke=False):
    """A family protocol whose ablation block declares the run_knn_ablation variants; in production the committed frozen
    protocol. Returns whether it is a synthetic smoke protocol."""
    smoke = allow_smoke and p.get('purpose') == 'synthetic_smoke_only'
    er.validate_extra_protocol(p)
    block = p['ablation']
    if tuple(block['variants']) != VARIANTS:
        raise ValueError(f'The ablation block must declare the variants {", ".join(VARIANTS)}')
    validate_depths(block['depth_split']['depths'])
    if not smoke:
        committed = er.PROTOCOL_FILES[p['production_family']]
        if p.get('purpose') is not None or not p.get('frozen'):
            raise ValueError('The component ablation requires the frozen production family protocol')
        if not committed.is_file() or canonical_json(json.loads(committed.read_text())) != canonical_json(p):
            raise ValueError(f'The protocol is not the committed frozen {committed.name}')
    return smoke


def family_reference(p, source, *, allow_smoke=False, declared=None):
    """(run_knn_ablation reference, pins) for the family run at `source`: it must hold this protocol and name its registry;
    run_knn_ablation.check_reference then compares the pins (`declared`, or those observed at prepare) and the design."""
    reference = base.load_reference(source, allow_smoke=allow_smoke)
    if canonical_json(reference['protocol']) != canonical_json(p):
        raise ValueError('The reference run does not hold this family protocol')
    if reference['environment'].get('registry') != p['registry']:
        raise ValueError(f'The reference environment names the registry {reference["environment"].get("registry")!r}, not {p["registry"]}')
    if not allow_smoke and reference['files']['protocol.json'] != sha256_file(er.PROTOCOL_FILES[p['production_family']]):
        raise ValueError('The reference run protocol.json is not the committed frozen protocol file')
    observed = {'protocol_id': p['protocol_id'], 'protocol_sha256': reference['files']['protocol.json'],
                'code_revision': reference['environment']['code_revision'], 'summary_sha256': reference['files']['summary.json'],
                'model_id': REFERENCE_MODEL, 'family': p['production_family']}
    pins = base.check_reference(dict(p, reference_source=declared or observed), reference)
    return reference, pins


def check_dataset(name, X, y, manifest, splits, p, *, production, loaders=None):
    """The run's prepared dataset: the declared nested splits, and in production the panel pins and a fresh load through the
    pinned extra_data loader of the same features (NaN equal to NaN) and labels."""
    if (splits != make_splits(y, p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['split_seed'])
            or config_id(splits) != manifest['splits_hash']):
        raise ValueError(f'{name}: the reference splits are not the declared nested splits')
    identity = {'dataset_hash': manifest['dataset_hash'], 'splits_hash': manifest['splits_hash'], 'pins_checked': production}
    if not production:
        return identity
    entry = er.panel_by_name(p)[name]
    if (manifest['dataset_hash'], manifest['splits_hash']) != (entry.get('dataset_hash'), entry.get('splits_hash')):
        raise DatasetIdentityError(f'{name}: the reference dataset or splits hash differs from the panel pins')
    loaded_X, loaded_y, loaded = er.load(name, p, loaders=loaders)
    if (not (np.array_equal(loaded_X, X, equal_nan=True) and np.array_equal(loaded_y, y))
            or loaded['dataset_hash'] != manifest['dataset_hash']):
        raise DatasetIdentityError(f'{name}: the pinned dataset differs from the reference prepared data')
    return {**identity, 'loader': loaded['loader']}


def prepare(output, p, source, *, allow_smoke=False, purpose='confirmatory', loaders=None):
    output = Path(output)
    smoke = check_protocol(p, allow_smoke=allow_smoke)
    reference, pins = family_reference(p, source, allow_smoke=smoke)
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs, identities = [], [], {}
    for name in p['datasets']:
        X, y, manifest, splits = load_prepared(reference['directory'], name)
        identities[name] = check_dataset(name, X, y, manifest, splits, p, production=not smoke, loaders=loaders)
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        if not (output/name/'data.npz').exists():
            shutil.copyfile(reference['directory']/name/'data.npz', output/name/'data.npz')
        copied_X, copied_y, _, _ = load_prepared(output, name)        # hash-checked against the manifest
        if not (np.array_equal(copied_X, X, equal_nan=True) and np.array_equal(copied_y, y)):
            raise ValueError(f'{name}: copied dataset differs from the reference source')
        for split in splits:
            validate_split(split, len(y))
            record = base.selection_record(reference, name, split, y, manifest)
            selections.append(record)
            jobs.append(base.planned_job(name, split, record, X.shape[1]))
    write_csv(output/'reference_selected_configurations.csv', base.SELECTION_COLUMNS, base.selection_rows(selections))
    write_json(output/'reference_selections.json', selections)
    record = {'purpose': purpose, 'family': p['production_family'], 'protocol_hash': config_id(p), 'datasets': list(p['datasets']),
              'variants': list(VARIANTS), 'dataset_identity': identities,
              'reference_source': {'directory': str(reference['directory'].resolve()), **pins,
                                   'protocol_hash': config_id(reference['protocol']), 'file_sha256': reference['files'],
                                   'selected_folds': len(selections)}}
    piloted = (p.get('projection') or {}).get('ablation_pilot')
    if piloted:
        index = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s for s in selections}
        records = [{'dataset_id': name, **entry} for name, entry in piloted.items()]
        record['realized_projection'] = rna.calibrated_projection({'decision_rule': REALIZED_RULE}, jobs, index, records,
                                                                  workers=p['workers'])
    write_json(output/'manifest.json', record)
    write_json(output/'planned_jobs.json', jobs)
    return jobs


# ----------------------------------------------------------------------------- verification

def verify(output, *, allow_smoke=False, environment_check='full'):
    """run_newdata_ablation.verify for the family run: environment_check 'full' (run) or 'sources' (summary)."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = check_protocol(p, allow_smoke=allow_smoke and manifest['purpose'] == 'synthetic_smoke_only')
    if manifest['protocol_hash'] != config_id(p):
        raise ValueError('Protocol seal changed')
    saved, current = json.loads((output/'environment.json').read_text()), environment()
    if environment_check == 'sources':
        saved, current = saved['source_hashes'], current['source_hashes']
    elif environment_check != 'full':
        raise ValueError('environment_check must be full or sources')
    elif smoke:
        saved.pop('code_revision', None)
        current.pop('code_revision', None)
    if saved != current:
        raise ValueError('Source/environment seal changed')
    if (manifest['datasets'] != list(p['datasets']) or manifest['variants'] != list(VARIANTS)
            or manifest.get('family') != p['production_family']):
        raise ValueError('Prepared manifest disagrees with the protocol')
    selections = json.loads((output/'reference_selections.json').read_text())
    saved_csv = list(csv.reader(io.StringIO((output/'reference_selected_configurations.csv').read_text())))
    if saved_csv != [list(base.SELECTION_COLUMNS)] + [[str(v) for v in row] for row in base.selection_rows(selections)]:
        raise ValueError('Selected-configuration CSV disagrees with the sealed selections')
    index = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s for s in selections}
    if len(index) != len(selections):
        raise ValueError('Duplicate sealed selections')
    entries, expected = er.panel_by_name(p), []
    for name in p['datasets']:
        X, y, data, splits = load_prepared(output, name)
        if splits != make_splits(y, p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['split_seed']):
            raise ValueError('Prepared splits differ from the declared fold schedule')
        if not smoke and (data['dataset_hash'], data['splits_hash']) != (entries[name]['dataset_hash'], entries[name]['splits_hash']):
            raise DatasetIdentityError(f'{name}: the prepared dataset or splits hash differs from the panel pins')
        for split in splits:
            record = index.get((name, split['outer_repeat'], split['outer_fold']))
            if (record is None or record['fitting_seeds'] != list(p['fit_seeds'])
                    or sorted(record['reference_predictions']) != sorted(map(str, p['fit_seeds']))):
                raise ValueError(f'Missing or inconsistent sealed selection for {name} r{split["outer_repeat"]}f{split["outer_fold"]}')
            for seed, labels in record['reference_predictions'].items():
                if len(labels) != len(split['test']) or base.array_hash(np.asarray(labels)) != record['reference_prediction_hashes'][seed]:
                    raise ValueError(f'Sealed reference predictions of {name} r{split["outer_repeat"]}f{split["outer_fold"]} '
                                     'disagree with their hashes')
            expected.append(base.planned_job(name, split, record, X.shape[1]))
    if json.loads((output/'planned_jobs.json').read_text()) != expected:
        raise ValueError('Planned job schedule changed')
    return p, manifest, expected, smoke


def collect_results(output, *, allow_smoke=False):
    """Every planned job, log, prediction file and artifact reconciled (run_knn_ablation.validate_job), the run still the one
    sealed at prepare, and every sealed selection re-derived from it; failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs, smoke = verify(output, allow_smoke=allow_smoke, environment_check='sources')
    sealed_pins = {key: manifest['reference_source'][key] for key in PIN_KEYS}
    reference, _ = family_reference(p, manifest['reference_source']['directory'], allow_smoke=smoke, declared=sealed_pins)
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
            if base.selection_record(reference, job['dataset_id'], split, y, data) != sealed:
                raise ValueError('the sealed selection differs from the one re-derived from the run')
            reproduced[job['dataset_id']] += base.validate_job(result, events, job, p, X, y, split, data, revision,
                                                               output/'artifacts'/stem, prediction_path, sealed)
            rows[job['dataset_id']].extend(result['models'])
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    for name in p['datasets']:
        try:
            validate_outer_schedule(rows[name], expected_folds=fold_schedule(p), expected_seeds={v: list(p['fit_seeds']) for v in VARIANTS})
        except ValueError as exc:
            issues.append(f'{name}: {exc}')
    if issues:
        raise ValueError(f'Incomplete {p["production_family"]} ablation evidence: ' + '; '.join(issues))
    return {'rows': dict(rows), 'reproduced': dict(reproduced), 'jobs': jobs, 'protocol': p, 'manifest': manifest,
            'code_revision': revision}


def run(output, workers=1, *, allow_smoke=False):
    """Every planned job on spawned single-thread workers (run_knn_ablation.worker); a views7 reproduction failure cancels
    the pending jobs."""
    output = Path(output)
    p, _, jobs, _ = verify(output, allow_smoke=allow_smoke)
    if not 1 <= workers <= p['ablation']['max_workers']:
        raise ValueError('Worker count exceeds the shared limit')
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(base.worker, (str(output), job)) for job in jobs]
        try:
            for future in as_completed(futures):
                path, status, reproduction_failed = future.result()
                print(path, status, flush=True)
                if reproduction_failed:
                    raise base.ReproductionError(f'{path}: views7 did not reproduce the reference predictions; pending jobs '
                                                 'are cancelled and this run cannot be summarized')
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return collect_results(output, allow_smoke=allow_smoke)


def summary(output, *, allow_smoke=False):
    """run_newdata_ablation.summary on this run: seed-within-fold means, outer-fold mean and SD, descriptive intervals against
    views7 and the depth split."""
    collected = collect_results(output, allow_smoke=allow_smoke)
    p = collected['protocol']
    folds, seeds, q, confidence = fold_schedule(p), list(p['fit_seeds']), p['test_train_ratio'], p['confidence']
    depths = validate_depths(p['ablation']['depth_split']['depths'])
    summaries, flat, split_by_dataset = {}, [], {}
    for name in p['datasets']:
        records, table = collected['rows'][name], {}
        for variant in VARIANTS:
            entry = {'metrics': {m: summarize_outer(records, variant, m, expected_folds=folds, expected_seeds=seeds) for m in base.METRICS}}
            if variant != 'views7':
                entry['change_from_views7'] = {}
                for m in ('accuracy', 'error'):
                    interval = paired_corrected_interval(records, variant, 'views7', metric=m, q=q, confidence=confidence,
                                                         expected_folds=folds, expected_seeds={variant: seeds, 'views7': seeds})
                    interval.pop('p_approximate')     # descriptive interval only
                    entry['change_from_views7'][m] = interval
            table[variant] = entry
            flat.extend({'dataset_id': name, 'variant_id': variant, 'metric': m,
                         **{k: entry['metrics'][m][k] for k in base.SUMMARY_COLUMNS[3:]}} for m in base.METRICS)
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
    family = p['production_family']
    report = {'purpose': f'knn_anchored_component_ablation_of_the_{family}_datasets', 'code_revision': collected['code_revision'],
              'protocol_id': p.get('protocol_id'), 'reference_source': collected['manifest']['reference_source'],
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; within-fold seed SD '
                             'reported separately',
              'change_from_views7': 'seed-averaged corrected resampled t interval of each variant minus views7; '
                                    'descriptive, without p values or multiplicity adjustment',
              'views7_reproduces_reference': 'hard check: every fold and seed of views7 reproduced the outer predictions of the '
                                             f'{family} run ({REFERENCE_MODEL}) exactly (the summary refuses otherwise)',
              'depth_split': {'difference': 'untrained minus views7 accuracy, fitting seeds averaged within fold',
                              'grouping': 'the hidden widths of the reconstructed selection of each outer fold',
                              'status': 'descriptive; no p values', 'depths': depths, 'by_dataset': split_by_dataset,
                              'pooled': pooled_depth_split(split_by_dataset, depths)},
              'inferential_significance_claims': False, 'summaries': summaries, 'model_rows': collected['rows']}
    return report, flat


def write_summary(output, *, allow_smoke=False):
    report, flat = summary(output, allow_smoke=allow_smoke)
    summary_json, summary_csv = summary_names(json.loads((Path(output)/'protocol.json').read_text())['production_family'])
    write_json(Path(output)/summary_json, report)
    write_csv(Path(output)/summary_csv, base.SUMMARY_COLUMNS, [[r[c] for c in base.SUMMARY_COLUMNS] for r in flat])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'ablation', 'summary'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--reference', type=Path, help='the complete family run (summary.json written)')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == 'summary':
        report = write_summary(args.output)
        for name, entry in report['summaries'].items():
            counts = entry['views7_reproduces_reference']
            print(f"{name}: views7 reproduced {counts['matching_fold_seeds']}/{counts['total_fold_seeds']} fold-seeds")
        return
    if args.protocol is None or args.reference is None:
        parser.error(f'{args.command} needs --protocol and --reference')
    p = json.loads(args.protocol.read_text())
    if args.command == 'prepare':
        prepare(args.output, p, args.reference)
        return
    if p != json.loads((args.output/'protocol.json').read_text()):
        raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
    manifest = json.loads((args.output/'manifest.json').read_text())
    if str(Path(args.reference).resolve()) != manifest['reference_source']['directory']:
        raise ValueError('--reference differs from the prepared reference source')
    run(args.output, args.workers)


if __name__ == '__main__':
    main()
