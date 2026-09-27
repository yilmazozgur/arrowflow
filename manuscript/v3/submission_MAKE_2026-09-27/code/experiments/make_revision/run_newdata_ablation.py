"""Task 24 Part 2: the component ablation of ArrowFlow-kNN on the ten further datasets, at its reconstructed per-fold
selections (descriptive).

draft     --output P                                           the unfrozen protocol (draft_protocol)
prepare   --protocol P --batch1 B1 --batch2 B2 --output O      seal the per-fold selections and the reference predictions
smoke     --protocol P --output O [--workers 3]                synthetic two-reference exercise; never evidence
pilot     --protocol P --batch1 B1 --batch2 B2 --output O      training-only timing on the pilot datasets and the projection
freeze    --draft P --pilot O/pilot.json --stages S --output F   the frozen protocol, only if the projection is within the cap
ablation  --protocol P --output O --workers 16                 fit every variant per dataset, outer fold and seed
summary   --output O                                           verify every planned record; newdata_ablation_summary.json/.csv

The design is run_knn_ablation's (knn_ablation.json, Task 20B): the variants views7, views1, views3, no_checkpoint,
no_augment, prototype_readout, untrained and input_knn, the same fit reuse, kNN readout, job records, verification and
summary. Its functions are imported wherever they take their inputs as arguments (load_reference, check_reference,
selection_record, planned_job, knn_ablation_variants, evaluate_job, worker, validate_job, reproduction_probe); what
run_knn_ablation binds to its single reference and its own sealed environment is re-stated here for two references.

The references are the two newdata batch runs (arrowflow-v3-newdata-batch1-1 and -batch2-1, arrowflow_full_knn): each
dataset's per-fold selection is reconstructed from the complete inner fit history of the batch run holding it
(reporting.validate_result_records) and resolved from the outer training partition's shape (bridge.resolve_selected).
Each dataset is loaded by newdata.load_newdata, which refuses any OpenML pin mismatch, and must equal that batch run's
prepared features and labels, with the pinned dataset and splits hashes. views7 refits ArrowFlow-kNN at the selection and
must reproduce the reference outer predictions exactly for every dataset, outer fold and fitting seed: a mismatch fails its
job, cancels the pending jobs and blocks the summary, which also re-derives every sealed selection from its batch run. No
outer-fold score chooses anything; every kNN readout is tuned inside its fit on training rows only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from . import run_knn_ablation as base
from .evaluation import canonical_json, config_id, make_splits, paired_corrected_interval, summarize_outer, validate_outer_schedule, validate_split
from .knn_controls import depth_split, pooled_depth_split, reference_pins, validate_depths
from .models import seed_fit
from .multiview import MultiViewArrowFlowKNN
from .newdata import (PANEL, PIN_BY_NAME, PROTOCOL_FILES, PROTOCOL_IDS, WORKSPACE_RUNS, DatasetIdentityError, load_newdata, makespan,
                      sha256_file, validate_newdata_protocol)
from .reporting import summarize_verified_results
from .run_bridge import fold_schedule, write_csv
from .run_revision import _worker, environment_record, execution_lock, get_registry, load_prepared, planned_jobs, write_json

PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-12'
PROTOCOL = PROTOCOLS/'newdata_ablation.json'
TEMPLATE = PROTOCOLS/'knn_ablation.json'
PROTOCOL_ID = 'arrowflow-v3-newdata-ablation-1'
SOURCE_MODULES = base.SOURCE_MODULES + ['experiments.make_revision.run_knn_ablation', 'experiments.make_revision.newdata',
                                        'experiments.make_revision.projected_knn']
REFERENCE_MODEL = base.REFERENCE_MODEL
VARIANTS = base.VARIANTS
BATCHES = ('1', '2')
CAP_HOURS = 3
WORKERS = 16
PILOT_DATASETS = ('mfeat_zernike', 'vertebra_column')          # the largest and the smallest further dataset (rows)
PROBE_DATASETS = {'1': 'ionosphere', '2': 'vertebra_column'}   # the smallest dataset of each batch
# The frozen newdata batch runs (Task 23B production at 6022f9b5e, summaries written 2026-09-14): protocol files committed
# at protocols/2026-09-12/newdata_batch{1,2}.json.
REFERENCE_CODE_REVISION = '6022f9b5e2312f80a96e96b2d0607c72b7d52138'
REFERENCE_PROTOCOL_SHA256 = {'1': 'f715ec2808b4b08c10e49ae23696070a471d7ea88eb351d63b059eecccc0d011',
                             '2': '188ff1e946b6dc8df712c5a5cfb01e9cf2f9e7022a7776d98ec95ae1f66fca49'}
REFERENCE_SUMMARY_SHA256 = {'1': 'f3ab23c5eeccbe36f036325fe94c4c28ce69bc9d35fa16e62cea813e977ec843',
                            '2': '50ebfbef88616f8ecd20737f45b3ef840ed19f75a5e5a3f8ceeb120e386d49db'}
REFERENCE_DIRECTORIES = {'1': WORKSPACE_RUNS/'2026-09-14-newdata-batch1', '2': WORKSPACE_RUNS/'2026-09-14-newdata-batch2'}
COPIED_KEYS = ('aggregation', 'confidence', 'depth_split', 'dropped_variants', 'failure_policy', 'fit_reuse', 'fit_seeds',
               'historical_results', 'inner_folds', 'max_workers', 'numeric_threads_per_worker', 'outer_folds', 'outer_repeats',
               'parallelism', 'report_metrics', 'selection_metric', 'split_seed', 'test_train_ratio', 'variant_definitions',
               'variants')
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'pilot_projection')
FROZEN_STATUS = 'reviewed_and_piloted_before_confirmatory_scoring'
SUMMARY_JSON, SUMMARY_CSV = 'newdata_ablation_summary.json', 'newdata_ablation_summary.csv'
SMOKE_DATASETS = {'1': 'synthetic_b1', '2': 'synthetic_b2'}
SMOKE_SAMPLES = {'1': 240, '2': 120}                            # augmentation on in batch 1 (160 training rows), off in batch 2
SMOKE_DESIGN = {'outer_folds': 3, 'outer_repeats': 1, 'inner_folds': 2}
REFERENCE_SMOKE_REGISTRY = 'experiments.make_revision.knn_controls:smoke_reference_registry'


def environment():
    return environment_record(__package__ + '.run_newdata_ablation:environment')


def _plain(value):
    return json.loads(canonical_json(value))


# ----------------------------------------------------------------------------- protocol

def batch_datasets():
    """{batch: its datasets in panel order}, as the committed frozen batch protocols list them."""
    return {batch: list(json.loads(PROTOCOL_FILES[int(batch)].read_text())['datasets']) for batch in BATCHES}


def draft_protocol():
    """The unfrozen protocol: knn_ablation.json's design with the ten further datasets and the two batch references."""
    template = json.loads(TEMPLATE.read_text())
    members = batch_datasets()
    protocol = {key: template[key] for key in COPIED_KEYS}
    protocol.update(
        protocol_id=PROTOCOL_ID, production_family='newdata_ablation', datasets=list(PANEL), pilot_datasets=list(PILOT_DATASETS),
        probe_datasets=dict(PROBE_DATASETS),
        reference_sources={batch: {'protocol_id': PROTOCOL_IDS[int(batch)], 'protocol_sha256': REFERENCE_PROTOCOL_SHA256[batch],
                                   'code_revision': REFERENCE_CODE_REVISION, 'summary_sha256': REFERENCE_SUMMARY_SHA256[batch],
                                   'model_id': REFERENCE_MODEL, 'family': 'newdata', 'datasets': members[batch],
                                   'protocol_file': f'experiments/make_revision/protocols/2026-09-12/newdata_batch{batch}.json',
                                   'directory': 'supplied by --batch1/--batch2 at prepare and recorded in manifest.json'}
                           for batch in BATCHES},
        reproduction='views7 must reproduce the outer predictions of the batch run holding the dataset exactly for every dataset, '
                     'outer fold and fitting seed (the same labels in the sealed test-sample order); checked inside every job, '
                     'again for every record by the summary, and every sealed selection is re-derived from its batch run at '
                     'summary time',
        selected_configuration='results/<dataset>__arrowflow_full_knn__r<repeat>f<fold>.json selection block of the batch run '
                               'holding the dataset, reconstructed from the complete inner fit history at prepare time '
                               '(reporting.validate_result_records) and resolved to embed_dim, degree and augment from the outer '
                               'training partition shape (bridge.resolve_selected); sealed in reference_selected_configurations.csv '
                               'and reference_selections.json with the reference per-example predictions',
        dataset_loading='newdata.load_newdata (every OpenML pin: identity, shape, missing and infinite cells, labels, class counts, '
                        'sha256 of X, y and feature names, harness fingerprint); the loaded arrays must equal the batch run\'s '
                        'prepared data.npz, and its dataset and splits hashes must equal newdata.PINS; the prepared data are copied '
                        'from the batch run',
        knn_readout={**template['knn_readout'], 'source': 'identical to arrowflow_full_knn (newdata_batch1.json and '
                                                          'newdata_batch2.json models.arrowflow_full_knn.readout)'},
        reporting={**template['reporting'], 'outputs': f'{SUMMARY_JSON} (metrics, intervals, depth split, reproduction counts, model '
                                                       f'rows) and {SUMMARY_CSV} (metrics per dataset, variant and metric)'},
        wallclock_cap_hours=CAP_HOURS,
        decision_rule=f'freeze only if the calibrated projection at {WORKERS} single-thread workers is at most {CAP_HOURS} h: the '
                      'simulated first-free-worker makespan of the planned jobs in planned order, each job priced at the batch run\'s '
                      'realized outer fit and predict seconds of its fold and seeds times the piloted all-variants/views7 time ratio '
                      '(the pilot dataset of the same no_augment fit source; the largest piloted ratio otherwise)',
        design_source={'template': 'protocols/2026-09-12/knn_ablation.json (sha256 in source_template_sha256)',
                       'identical_to_template': ', '.join(COPIED_KEYS),
                       'changed': 'protocol_id, production_family, datasets, pilot_datasets, knn_readout.source, reporting.outputs, '
                                  'wallclock_cap_hours, frozen, status, resource_decision, source_template_sha256',
                       'removed': 'reference_source (one bridge_knn run) and frozen_at_utc until the freeze',
                       'added': 'reference_sources (the two batch runs), probe_datasets, reproduction, selected_configuration, '
                                'dataset_loading, decision_rule, design_source'},
        source_template_sha256=sha256_file(TEMPLATE), frozen=False,
        status='drafted_awaiting_prepare_smoke_and_training_only_pilot',
        resource_decision='pending: prepare of the ten further datasets, synthetic two-reference smoke, and the training-only pilot on '
                          'mfeat_zernike (largest) and vertebra_column (smallest)')
    return _plain(protocol)


def validate_protocol(p):
    """A production protocol is the draft, or the draft with exactly the freeze fields set by `freeze`."""
    draft = draft_protocol()
    if {k: v for k, v in p.items() if k not in FREEZE_FIELDS} != {k: v for k, v in draft.items() if k not in FREEZE_FIELDS}:
        differing = sorted(k for k in set(p) | set(draft) if k not in FREEZE_FIELDS and p.get(k) != draft.get(k))
        raise ValueError(f'The protocol differs from run_newdata_ablation.draft_protocol() in {", ".join(differing)}')
    if not p.get('frozen'):
        if p != draft:
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('pilot_projection') or {}
    hours = projection.get('decision_hours')
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != CAP_HOURS or isinstance(hours, bool) or not isinstance(hours, (int, float))
            or not 0 < hours <= CAP_HOURS):
        raise ValueError(f'A frozen protocol records its freeze and a pilot projection within the {CAP_HOURS} h cap')
    return p


def batch_of(p):
    mapping = {}
    for batch, pins in p['reference_sources'].items():
        for name in pins['datasets']:
            if name in mapping:
                raise ValueError(f'{name} belongs to two reference batches')
            mapping[name] = batch
    if sorted(mapping) != sorted(p['datasets']) or sorted(p['reference_sources']) != list(BATCHES):
        raise ValueError('The reference batches must partition the protocol datasets')
    return mapping


# ----------------------------------------------------------------------------- references and datasets

def load_references(p, sources, *, allow_smoke=False):
    """{batch: run_knn_ablation reference} for the batch run directories {batch: path}; each must be the run the protocol pins."""
    if sorted(map(str, sources)) != sorted(p['reference_sources']):
        raise ValueError('Supply one reference run per protocol batch')
    references, observed = {}, {}
    for batch in sorted(p['reference_sources']):
        declared = p['reference_sources'][batch]
        reference = base.load_reference(sources[batch], allow_smoke=allow_smoke)
        observed[batch] = base.check_reference(dict(p, reference_source=declared, datasets=list(declared['datasets'])), reference)
        if not allow_smoke:
            protocol = reference['protocol']
            try:
                validate_newdata_protocol(protocol)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'The batch {batch} reference is not a newdata batch run: {exc}') from exc
            if (protocol.get('batch') != int(batch) or protocol['datasets'] != declared['datasets']
                    or reference['files']['protocol.json'] != sha256_file(PROTOCOL_FILES[int(batch)])):
                raise ValueError(f'The batch {batch} reference does not hold the committed frozen batch {batch} protocol')
        references[batch] = reference
    return references, observed


def check_dataset(name, X, y, manifest, splits, p, *, production, loader=None):
    """The reference's prepared dataset: the declared nested splits, and in production every newdata pin and a fresh
    newdata.load_newdata of the same features and labels."""
    if splits != make_splits(y, p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['split_seed']) or config_id(splits) != manifest['splits_hash']:
        raise ValueError(f'{name}: the reference splits are not the declared nested splits')
    identity = {'dataset_hash': manifest['dataset_hash'], 'splits_hash': manifest['splits_hash'], 'pins_checked': production}
    if not production:
        return identity
    pin = PIN_BY_NAME.get(name)
    if pin is None or (manifest['dataset_hash'], manifest['splits_hash']) != (pin['dataset_hash'], pin['splits_hash']):
        raise DatasetIdentityError(f'{name}: the reference dataset or splits hash differs from newdata.PINS')
    loaded_X, loaded_y, loaded = (loader or load_newdata)(name)
    if not (np.array_equal(loaded_X, X) and np.array_equal(loaded_y, y)) or loaded['dataset_hash'] != manifest['dataset_hash']:
        raise DatasetIdentityError(f'{name}: the pinned OpenML dataset differs from the reference prepared data')
    return {**identity, 'openml': loaded['openml'], 'identity': loaded['identity']}


def prepare(output, p, sources, *, allow_smoke=False, purpose='confirmatory', loader=None):
    output = Path(output)
    references, pins = load_references(p, sources, allow_smoke=allow_smoke)
    batches = batch_of(p)
    validate_depths(p['depth_split']['depths'])
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs, identities = [], [], {}
    for name in p['datasets']:
        reference = references[batches[name]]
        X, y, manifest, splits = load_prepared(reference['directory'], name)
        identities[name] = check_dataset(name, X, y, manifest, splits, p, production=not allow_smoke, loader=loader)
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
    write_json(output/'manifest.json', {
        'purpose': purpose, 'protocol_hash': config_id(p), 'datasets': list(p['datasets']), 'variants': list(VARIANTS),
        'dataset_identity': identities,
        'reference_sources': {batch: {'directory': str(references[batch]['directory'].resolve()), **pins[batch],
                                      'datasets': list(p['reference_sources'][batch]['datasets']),
                                      'protocol_hash': config_id(references[batch]['protocol']),
                                      'file_sha256': references[batch]['files'],
                                      'selected_folds': sum(1 for s in selections if batches[s['dataset_id']] == batch)}
                              for batch in sorted(references)}})
    write_json(output/'planned_jobs.json', jobs)
    return jobs


# ----------------------------------------------------------------------------- verification

def verify(output, *, allow_smoke=False, environment_check='full'):
    """run_knn_ablation.verify for two references: environment_check 'full' (run) or 'sources' (summary)."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = allow_smoke and manifest['purpose'] == 'synthetic_smoke_only'
    if not p['frozen'] and not smoke:
        raise ValueError('A frozen reviewed protocol is required')
    if not smoke:
        validate_protocol(p)
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
    batches = batch_of(p)
    if sorted(manifest['reference_sources']) != list(BATCHES) or any(
            manifest['reference_sources'][batch]['datasets'] != p['reference_sources'][batch]['datasets'] for batch in BATCHES):
        raise ValueError('Prepared reference batches disagree with the protocol')
    validate_depths(p['depth_split']['depths'])
    selections = json.loads((output/'reference_selections.json').read_text())
    saved_csv = list(csv.reader(io.StringIO((output/'reference_selected_configurations.csv').read_text())))
    if saved_csv != [list(base.SELECTION_COLUMNS)] + [[str(v) for v in row] for row in base.selection_rows(selections)]:
        raise ValueError('Selected-configuration CSV disagrees with the sealed selections')
    index = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s for s in selections}
    if len(index) != len(selections):
        raise ValueError('Duplicate sealed selections')
    expected = []
    for name in p['datasets']:
        X, y, data, splits = load_prepared(output, name)
        if splits != make_splits(y, p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['split_seed']):
            raise ValueError('Prepared splits differ from the declared fold schedule')
        if not smoke and (data['dataset_hash'], data['splits_hash']) != (PIN_BY_NAME[name]['dataset_hash'], PIN_BY_NAME[name]['splits_hash']):
            raise DatasetIdentityError(f'{name}: the prepared dataset or splits hash differs from newdata.PINS')
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
    return p, manifest, expected


def collect_results(output, *, allow_smoke=False):
    """Every planned job, log, prediction file and artifact reconciled (run_knn_ablation.validate_job), and every sealed
    selection re-derived from the batch run holding its dataset; failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs = verify(output, allow_smoke=allow_smoke, environment_check='sources')
    sources = {batch: entry['directory'] for batch, entry in manifest['reference_sources'].items()}
    references, _ = load_references(p, sources, allow_smoke=allow_smoke)
    batches = batch_of(p)
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
            reference = references[batches[job['dataset_id']]]
            if base.selection_record(reference, job['dataset_id'], split, y, data) != sealed:
                raise ValueError('the sealed selection differs from the one re-derived from its batch run')
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
        raise ValueError('Incomplete newdata ablation evidence: ' + '; '.join(issues))
    return {'rows': dict(rows), 'reproduced': dict(reproduced), 'jobs': jobs, 'protocol': p, 'manifest': manifest,
            'code_revision': revision}


def run(output, workers=1, *, allow_smoke=False):
    """Every planned job on spawned single-thread workers (run_knn_ablation.worker); a views7 reproduction failure cancels
    the pending jobs."""
    output = Path(output)
    p, manifest, jobs = verify(output, allow_smoke=allow_smoke)
    if not 1 <= workers <= p['max_workers']:
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
    """run_knn_ablation.summary on this run: seed-within-fold means, outer-fold mean and SD, descriptive intervals against
    views7 and the depth split."""
    collected = collect_results(output, allow_smoke=allow_smoke)
    p = collected['protocol']
    folds, seeds, q, confidence = fold_schedule(p), list(p['fit_seeds']), p['test_train_ratio'], p['confidence']
    depths = validate_depths(p['depth_split']['depths'])
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
    report = {'purpose': 'knn_anchored_component_ablation_of_the_further_datasets', 'code_revision': collected['code_revision'],
              'protocol_id': p.get('protocol_id'), 'reference_sources': collected['manifest']['reference_sources'],
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; within-fold seed SD '
                             'reported separately',
              'change_from_views7': 'seed-averaged corrected resampled t interval of each variant minus views7; '
                                    'descriptive, without p values or multiplicity adjustment',
              'views7_reproduces_reference': 'hard check: every fold and seed of views7 reproduced the outer predictions of the '
                                             f'batch run holding the dataset ({REFERENCE_MODEL}) exactly (the summary refuses otherwise)',
              'depth_split': {'difference': 'untrained minus views7 accuracy, fitting seeds averaged within fold',
                              'grouping': 'the hidden widths of the reconstructed selection of each outer fold',
                              'status': 'descriptive; no p values', 'depths': depths, 'by_dataset': split_by_dataset,
                              'pooled': pooled_depth_split(split_by_dataset, depths)},
              'inferential_significance_claims': False, 'summaries': summaries, 'model_rows': collected['rows']}
    return report, flat


def write_summary(output, *, allow_smoke=False):
    report, flat = summary(output, allow_smoke=allow_smoke)
    write_json(Path(output)/SUMMARY_JSON, report)
    write_csv(Path(output)/SUMMARY_CSV, base.SUMMARY_COLUMNS, [[r[c] for c in base.SUMMARY_COLUMNS] for r in flat])
    return report


# ----------------------------------------------------------------------------- pilot, projection and freeze

def calibrated_projection(p, jobs, selections, records, workers=WORKERS):
    """Per job: the reference's realized outer fit and predict seconds of its fold and fitting seeds (measured under the batch
    run's 16 workers) times the all-variants/views7 ratio of the pilot dataset with the same no_augment fit source (the
    largest such ratio; the largest piloted ratio when no pilot dataset shares the source); per dataset and in total the
    serial hours, and the simulated first-free-worker makespan of the jobs in planned order."""
    piloted = {r['dataset_id']: r for r in records}
    job_seconds, datasets = [], {}
    for job in jobs:
        name, source = job['dataset_id'], job['fit_sources']['no_augment']
        alike = [r['all_variants_to_views7_ratio'] for r in records if r['fit_sources']['no_augment'] == source]
        ratio = (piloted[name]['all_variants_to_views7_ratio'] if name in piloted
                 else max(alike or [r['all_variants_to_views7_ratio'] for r in records]))
        realized = sum(selections[(name, job['outer_repeat'], job['outer_fold'])]['reference_outer_seconds'].values())
        job_seconds.append(realized * ratio)
        entry = datasets.setdefault(name, {'jobs': 0, 'reference_views7_seconds': 0., 'seconds': 0., 'ratios': set(),
                                           'basis': 'piloted ratio' if name in piloted else
                                                    ('largest piloted ratio with the same no_augment fit source' if alike else
                                                     'largest piloted ratio (no pilot dataset shares the no_augment fit source)')})
        entry['jobs'] += 1
        entry['reference_views7_seconds'] += realized
        entry['seconds'] += realized * ratio
        entry['ratios'].add(ratio)
    for entry in datasets.values():
        entry['all_variants_to_views7_ratio'] = sorted(entry.pop('ratios'))
        entry['serial_hours'] = entry['seconds'] / 3600
    serial = sum(job_seconds)
    return {'datasets': datasets, 'serial_hours': serial / 3600, 'serial_hours_over_workers': serial / 3600 / workers,
            'simulated_makespan_hours': makespan(job_seconds, workers) / 3600, 'longest_job_hours': max(job_seconds) / 3600,
            'workers': workers, 'basis': p['decision_rule']}


def runtime_pilot(output, p, sources, *, allow_smoke=False):
    """Training-only timing of every variant on each pilot dataset's first outer training partition (one seed; predictions on
    training rows only), the harness projection (idle pilot seconds, unpiloted datasets at the slowest piloted one), the
    calibrated projection that decides the freeze, and one reproduction probe per reference batch."""
    output = Path(output)
    jobs = prepare(output, p, sources, allow_smoke=allow_smoke, purpose='training_runtime_only')
    references, _ = load_references(p, sources, allow_smoke=allow_smoke)
    batches = batch_of(p)
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
            _, _, knn_views, output_views = base.fit_views7(params['views7'], seed, X[train], y[train], X[query])
            base.derived_predictions(knn_views, output_views)
            seconds['views7'], network_fits['views7'] = time.perf_counter() - start, 7
            for variant in base.SEPARATE:
                if job['fit_sources'][variant] == 'identical_to_views7':
                    seconds[variant], network_fits[variant] = 0., 0
                    continue
                seed_fit(seed)
                start = time.perf_counter()
                MultiViewArrowFlowKNN(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, 7
            for variant, estimator in base.CONTROLS.items():
                seed_fit(seed)
                start = time.perf_counter()
                estimator(**params[variant], seed=seed).fit(X[train], y[train]).predict(X[query])
                seconds[variant], network_fits[variant] = time.perf_counter() - start, 0
        per_seed = sum(seconds.values())
        records.append({'dataset_id': name, 'batch': batches[name], 'dataset_hash': data['dataset_hash'], 'train_ids': train,
                        'query_ids': query, 'config_id': job['config_id'], 'selected': job['selected'],
                        'fit_sources': job['fit_sources'], 'model_seed': seed, 'seconds_by_variant': seconds,
                        'network_fits_by_variant': network_fits, 'network_fits_per_seed': sum(network_fits.values()),
                        'seconds_per_seed': per_seed, 'seconds_per_job_estimate': per_seed * len(p['fit_seeds']),
                        'all_variants_to_views7_ratio': per_seed / seconds['views7'], 'status': 'ok'})
    piloted = {r['dataset_id']: r for r in records}
    slowest = max(r['seconds_per_job_estimate'] for r in records)
    harness = {}
    for name in p['datasets']:
        count = sum(1 for j in jobs if j['dataset_id'] == name)
        per_job = piloted[name]['seconds_per_job_estimate'] if name in piloted else slowest
        harness[name] = {'jobs': count, 'seconds_per_job': per_job,
                         'basis': 'piloted' if name in piloted else 'slowest piloted dataset (not a bound)'}
    serial = sum(v['jobs'] * v['seconds_per_job'] for v in harness.values())
    calibrated = calibrated_projection(p, jobs, selections, records)
    probes = {batch: base.reproduction_probe(references[batch], name) for batch, name in sorted(p['probe_datasets'].items())}
    decision = calibrated['simulated_makespan_hours']
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'protocol_hash': config_id(p), 'protocol_id': p['protocol_id'],
              'records': records,
              'harness_projection': {'datasets': harness, 'serial_hours': serial / 3600, 'hours_at_16_workers_ideal': serial / 3600 / WORKERS},
              'calibrated_projection': calibrated, 'reproduction_probes': probes,
              'decision': {'rule': p['decision_rule'], 'hours': decision, 'cap_hours': p['wallclock_cap_hours'],
                           'within_cap': decision <= p['wallclock_cap_hours'],
                           'probes_reproduced': all(probe['reproduced'] for probe in probes.values())},
              'estimate_limitations': 'one seed on one training partition per pilot dataset on an idle machine; the harness '
                                      'projection prices unpiloted datasets at the slowest piloted one and includes no contention; '
                                      'the calibrated projection assumes the batch runs\' realized per-fold seconds carry over and '
                                      'that a job takes its fold\'s three seeds in sequence'}
    write_json(output/'pilot.json', report)
    print(json.dumps({'harness_hours_at_16_workers': report['harness_projection']['hours_at_16_workers_ideal'],
                      'calibrated_serial_hours': calibrated['serial_hours'],
                      'calibrated_hours_at_16_workers': calibrated['serial_hours_over_workers'],
                      'calibrated_makespan_hours_at_16_workers': decision, 'wallclock_cap_hours': p['wallclock_cap_hours'],
                      'within_cap': report['decision']['within_cap'],
                      'reproduction_probes': {batch: probe['reproduced'] for batch, probe in probes.items()}}, indent=2))
    return report


def freeze(draft_path, pilot_path, stages_path, output_path, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the pilot of that draft projects within the cap and every
    reproduction probe held; an existing output must be the draft itself, which the frozen protocol then replaces."""
    draft = json.loads(Path(draft_path).read_text())
    if draft != draft_protocol():
        raise ValueError('The draft differs from run_newdata_ablation.draft_protocol()')
    pilot, stages = json.loads(Path(pilot_path).read_text()), json.loads(Path(stages_path).read_text())
    if pilot.get('protocol_hash') != config_id(draft):
        raise ValueError('The pilot did not run with this draft')
    decision = pilot['decision']
    if not decision['within_cap'] or decision['hours'] > CAP_HOURS or not decision['probes_reproduced']:
        raise ValueError(f'Not frozen: projection {decision["hours"]:.2f} h against the {CAP_HOURS} h cap, probes reproduced '
                         f'{decision["probes_reproduced"]}')
    calibrated, harness = pilot['calibrated_projection'], pilot['harness_projection']
    record = {'cap_hours': CAP_HOURS, 'workers': WORKERS, 'decision_hours': decision['hours'], 'decision_rule': decision['rule'],
              'calibrated': {key: calibrated[key] for key in ('serial_hours', 'serial_hours_over_workers', 'simulated_makespan_hours',
                                                              'longest_job_hours')},
              'calibrated_per_dataset_hours': {name: entry['serial_hours'] for name, entry in calibrated['datasets'].items()},
              'harness': {key: harness[key] for key in ('serial_hours', 'hours_at_16_workers_ideal')},
              'pilot_ratios': {r['dataset_id']: r['all_variants_to_views7_ratio'] for r in pilot['records']},
              'reproduction_probes': {batch: {key: probe[key] for key in ('dataset_id', 'result_file', 'config_id', 'model_seed',
                                                                          'inner_fold', 'reference_score', 'refit_score',
                                                                          'readout_selections_identical', 'reproduced')}
                                      for batch, probe in pilot['reproduction_probes'].items()},
              'pilot_sha256': sha256_file(pilot_path), 'stages': stages}
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    text = (f"Task 24 Part 2 (author decision 3 and its scope ruling, 2026-09-14): drafted from knn_ablation.json with the "
            f"references the two newdata batch runs; {stages['summary']}; projected at {WORKERS} single-thread workers: calibrated "
            f"simulated makespan {decision['hours']:.2f} h (serial {calibrated['serial_hours']:.2f} h, serial over workers "
            f"{calibrated['serial_hours_over_workers']:.2f} h), harness {harness['hours_at_16_workers_ideal']:.2f} h; cap {CAP_HOURS} h; "
            f"reproduction probes held on {', '.join(probe['dataset_id'] for probe in pilot['reproduction_probes'].values())}; "
            f"frozen after the pilot")
    protocol = validate_protocol(_plain(dict(draft, frozen=True, frozen_at_utc=frozen_at, status=FROZEN_STATUS,
                                             resource_decision=text, pilot_projection=record)))
    output_path = Path(output_path)
    content = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if output_path.exists() and json.loads(output_path.read_text()) not in (draft, protocol):
        raise FileExistsError(f'Refusing to replace {output_path}: it is neither the draft nor this frozen protocol')
    temporary = output_path.with_name(output_path.name + '.freezing')
    temporary.write_text(content)
    os.replace(temporary, output_path)
    return protocol


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

def write_synthetic_dataset(directory, protocol, name, *, samples, seed):
    """A three-class, four-feature dataset in run_revision's prepared layout (knn_controls.write_synthetic_dataset with a name)."""
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], samples // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    features, labels = [f'x{i}' for i in range(4)], ['0', '1', '2']
    splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
    from .evaluation import dataset_fingerprint
    manifest = {'dataset_id': name, 'purpose': 'synthetic_smoke_only', 'source': 'synthetic smoke dataset', 'feature_names': features,
                'label_map': labels, 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                'sample_order': 'source row order; zero-based sample_id', 'dataset_hash': dataset_fingerprint(X, y, features, labels),
                'splits_hash': config_id(splits)}
    write_json(Path(directory)/name/'manifest.json', manifest)
    write_json(Path(directory)/name/'splits.json', splits)
    if not (Path(directory)/name/'data.npz').exists():
        np.savez_compressed(Path(directory)/name/'data.npz', X=X, y=y)


def synthetic_reference_batch(directory, batch, candidates, *, workers=1):
    """A complete synthetic batch run of ArrowFlow-kNN alone on one synthetic dataset, in the production layout with
    summary.json, through run_revision's worker and reporting (knn_controls.smoke_reference_registry)."""
    directory, name = Path(directory), SMOKE_DATASETS[batch]
    protocol = dict(json.loads((PROTOCOLS/'bridge_knn.json').read_text()), **SMOKE_DESIGN, datasets=[name], primary_family_size=1,
                    frozen=True, purpose='synthetic_smoke_only', production_family='newdata', batch=int(batch),
                    protocol_id=f'{PROTOCOL_IDS[int(batch)]}-synthetic-smoke', registry=REFERENCE_SMOKE_REGISTRY,
                    smoke_reference_candidates=list(candidates))
    registry = get_registry(REFERENCE_SMOKE_REGISTRY, protocol)
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'candidates.json', {model: {'stochastic': spec.stochastic, 'candidates': spec.candidates,
                                                     'config_ids': [config_id(c) for c in spec.candidates]}
                                             for model, spec in registry.items()})
    write_json(directory/'environment.json', environment_record(REFERENCE_SMOKE_REGISTRY))
    write_synthetic_dataset(directory, protocol, name, samples=SMOKE_SAMPLES[batch], seed=30 + int(batch))
    write_json(directory/'planned_jobs.json', planned_jobs([name], protocol, registry))
    jobs = [(str(directory), name, index, model, REFERENCE_SMOKE_REGISTRY)
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for _ in pool.map(_worker, jobs):
            pass
    write_json(directory/'summary.json', summarize_verified_results(directory))
    return directory


def smoke_protocol(p, references):
    """The synthetic smoke form of a protocol whose references are the given synthetic batch runs {batch: directory}."""
    reference_protocol = json.loads((references['1']/'protocol.json').read_text())
    return _plain(dict(p, datasets=[SMOKE_DATASETS[batch] for batch in BATCHES], pilot_datasets=[SMOKE_DATASETS[batch] for batch in BATCHES],
                       probe_datasets=dict(SMOKE_DATASETS), frozen=False, purpose='synthetic_smoke_only',
                       **{key: reference_protocol[key] for key in SMOKE_DESIGN},
                       reference_sources={batch: {**p['reference_sources'][batch], **reference_pins(references[batch]),
                                                  'family': 'newdata', 'datasets': [SMOKE_DATASETS[batch]]} for batch in BATCHES},
                       depth_split={**p['depth_split'], 'depths': [c['widths'] for c in base.SMOKE_CANDIDATES]}))


def smoke(output, p, workers=3):
    """Two synthetic reference batch runs (augmentation on in batch 1, off in batch 2), then prepare, run and summary; never
    evidence."""
    output = Path(output)
    with execution_lock():
        references = {batch: synthetic_reference_batch(output/f'synthetic_reference_batch{batch}', batch, base.SMOKE_CANDIDATES,
                                                       workers=workers) for batch in BATCHES}
    tiny = smoke_protocol(p, references)
    prepare(output, tiny, references, allow_smoke=True, purpose='synthetic_smoke_only')
    run(output, workers, allow_smoke=True)
    return write_summary(output, allow_smoke=True)


# ----------------------------------------------------------------------------- command

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['draft', 'prepare', 'smoke', 'pilot', 'freeze', 'ablation', 'summary'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--batch1', type=Path, default=REFERENCE_DIRECTORIES['1'])
    parser.add_argument('--batch2', type=Path, default=REFERENCE_DIRECTORIES['2'])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--draft', type=Path, default=PROTOCOL)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--stages', type=Path)
    args = parser.parse_args(argv)
    sources = {'1': args.batch1, '2': args.batch2}
    if args.command == 'draft':
        write_json(args.output, draft_protocol())
        return
    if args.command == 'summary':
        report = write_summary(args.output)
        for name, entry in report['summaries'].items():
            counts = entry['views7_reproduces_reference']
            print(f"{name}: views7 reproduced {counts['matching_fold_seeds']}/{counts['total_fold_seeds']} fold-seeds")
        return
    if args.command == 'freeze':
        if args.pilot is None or args.stages is None:
            parser.error('freeze needs --pilot and --stages')
        protocol = freeze(args.draft, args.pilot, args.stages, args.output)
        print(f"frozen at {protocol['frozen_at_utc']}: decision {protocol['pilot_projection']['decision_hours']:.2f} h")
        return
    p = json.loads(args.protocol.read_text())
    if args.command == 'smoke':
        if not 1 <= args.workers <= WORKERS:
            raise ValueError('Worker count must be between 1 and 16')
        smoke(args.output, validate_protocol(p), args.workers)
        return
    validate_protocol(p)
    if args.command == 'prepare':
        prepare(args.output, p, sources)
    elif args.command == 'pilot':
        with execution_lock():
            runtime_pilot(args.output, p, sources)
    else:
        if not p.get('frozen'):
            raise ValueError('The newdata ablation run requires a frozen reviewed protocol')
        if p != json.loads((args.output/'protocol.json').read_text()):
            raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
        manifest = json.loads((args.output/'manifest.json').read_text())
        if any(str(Path(sources[batch]).resolve()) != manifest['reference_sources'][batch]['directory'] for batch in BATCHES):
            raise ValueError('--batch1/--batch2 differ from the prepared reference sources')
        run(args.output, args.workers)


if __name__ == '__main__':
    main()
