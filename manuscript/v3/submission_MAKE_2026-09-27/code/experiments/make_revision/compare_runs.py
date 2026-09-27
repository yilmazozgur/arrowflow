"""Cross-run comparison: arrowflow_full_knn (bridge_knn run) versus arrowflow_full (frozen bridge run).

python -m experiments.make_revision.compare_runs knn --knn-source K --bridge-source B --output O
python -m experiments.make_revision.compare_runs training --training-source T --knn-source K --output O
python -m experiments.make_revision.compare_runs training-pairing --training-source T --knn-source K
python -m experiments.make_revision.compare_runs projected --projected-source P --knn-source K --training-source T --output O
python -m experiments.make_revision.compare_runs projected-pairing --projected-source P --knn-source K --training-source T

Both runs must be complete, and every saved record of both is re-verified before anything is compared, with the
harness validators that do not compare analysis source hashes against the current tree (reporting's
collect_verified_results without that one check): run_revision.load_prepared (dataset and splits hashes), the
protocol's declared nested splits (evaluation.make_splits), run_revision.collect_confirmatory_results (planned jobs,
result files, fit logs, result status, outer schedule) and reporting.validate_result_records for every planned job
(complete inner selection history, per-example predictions against the prepared truth, outer metrics recomputed from
the predictions). Each run's summary.json model_rows must equal the re-verified model rows field for field, and its
summaries must be recomputable from them. The knn protocol's knn_readout.reference must pin the bridge run
(summary.json sha256, code revision, protocol sha256, protocol ID, model ID), and the runs must share the dataset
panel, dataset and splits hashes, the outer schedule, the fitting seeds and the ArrowFlow candidates. Anything else is
refused (exit status 2) before an output is written.

Outputs (all or none; an existing file with different content is never replaced):
  knn_vs_full_contrasts.csv  per dataset, accuracy of arrowflow_full_knn minus arrowflow_full: fitting seeds
                             averaged within each outer fold, corrected resampled t over the outer folds
                             (evaluation.paired_corrected_interval), Holm across the datasets (evaluation.holm_adjust)
  knn_vs_full_summary.json   the family, the contrasts, definitions (with the exact reproduction criterion), the
                             sha256 of the CSV, provenance of both runs and the comparator reproduction record
  main_table.json            per dataset, mean error, outer-fold SD and within-fold seed SD: arrowflow_full from the
                             bridge run; arrowflow_full_knn and the refitted comparators from the knn run
The refitted comparators are compared with the bridge run's per-example predictions and selected configurations
for every dataset, outer fold and fitting seed; mismatches are listed, never raised.

training (Task 20A): arrowflow_full_knn (bridge_knn run) versus its two training controls, arrowflow_knn_untrained and
input_footrule_knn (knn_training run). Both runs are loaded and re-verified exactly as above. The training protocol's
training_controls.reference must pin the knn run; the runs must share the dataset panel, dataset and splits hashes, the
nested design keys and the fitting seeds; each control's candidates must be the projection of the knn run's
arrowflow_full_knn candidates onto the keys that act without training (knn_controls.CANDIDATE_KEYS); every source both
environment records seal must be byte-identical, and the network, encoder, readout, resolution and harness sources
(SHARED_SOURCES) must be among them. Outputs (all or none):
  training_contrasts.csv     the prespecified family: per dataset, arrowflow_full_knn minus arrowflow_knn_untrained and
                             arrowflow_full_knn minus input_footrule_knn accuracy, seed-averaged corrected resampled t,
                             one Holm adjustment across all members (two per dataset)
  training_contrasts.json    the family, the contrasts, definitions, the sha256 of the CSV and provenance of both runs
  training_error_table.json  per dataset, mean error, outer-fold SD and within-fold seed SD of the three models
  training_depth_split.json  arrowflow_full_knn minus arrowflow_knn_untrained by the hidden widths arrowflow_full_knn
                             selected in each outer fold; descriptive, outside the Holm family, no p values
training-pairing checks a prepared knn_training directory against the complete knn run before its jobs start (reference
pins and pairing only) and writes nothing.

projected (Task 23A): projected_numeric_knn (knn_projected run) against arrowflow_full_knn and numeric_knn (knn run) and
input_footrule_knn and arrowflow_knn_untrained (knn_training run); projected-pairing checks a prepared knn_projected
directory before its jobs start. Both are implemented, with their pairing rules and outputs, in compare_projected.py.
"""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import numpy as np
from .evaluation import (ModelSpec, canonical_json, config_id, expected_schedule, holm_adjust, make_splits,
                         paired_corrected_interval, summarize_outer)
from .knn_controls import (CANDIDATE_KEYS, CONTROL_MODELS, PRIMARY_CONTRASTS as TRAINING_CONTRASTS, depth_split,
                           pooled_depth_split, project_candidates, selected_widths, validate_depths)
from .reporting import METRICS, validate_result_records
from .run_revision import collect_confirmatory_results, load_prepared, planned_jobs

KNN_MODEL = 'arrowflow_full_knn'
FULL_MODEL = 'arrowflow_full'
CONTRAST_ID = f'{KNN_MODEL}_vs_{FULL_MODEL}'
CONTRAST_METRIC = 'accuracy'
TABLE_METRIC = 'error'
RUN_RECORDS = ('protocol.json', 'environment.json', 'candidates.json', 'planned_jobs.json', 'summary.json')
DESIGN_KEYS = ('datasets', 'outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds',
               'selection_metric', 'test_train_ratio', 'confidence')
CANDIDATE_FIELDS = ('stochastic', 'candidates', 'config_ids')
CONTRAST_COLUMNS = ('dataset', 'model_a', 'model_b', 'mean_difference', 'standard_error', 'ci_low', 'ci_high',
                    'p_approximate', 'holm_p_approximate', 'n_folds', 'df')
CONTRASTS_CSV, SUMMARY_JSON, TABLE_JSON = 'knn_vs_full_contrasts.csv', 'knn_vs_full_summary.json', 'main_table.json'
ANALYSIS_SOURCES = ('compare_runs.py', 'evaluation.py', 'reporting.py', 'run_revision.py')
VALIDATORS = ('run_revision.load_prepared', 'evaluation.make_splits (the declared nested splits)',
              'run_revision.collect_confirmatory_results', 'reporting.validate_result_records (every planned job)')
TOLERANCE = 1e-12
REPRODUCTION_CRITERION = (
    'A comparator reproduces the bridge run exactly when (1) its candidate definitions are identical in both runs '
    '(candidates.json stochastic flag, candidate list and config_ids) and (2) for every dataset, outer fold and fitting '
    'seed of its schedule, both runs selected the same config_id and made identical per-example predictions: the same '
    'test sample IDs, each with the same predicted label. Any difference sets comparators_reproduced to false and is '
    'listed in reproduction.definition_mismatches or reproduction.mismatches.')
DEFINITIONS = {
    'mean_difference': f'{KNN_MODEL} (knn run) minus {FULL_MODEL} (bridge run) accuracy: fitting seeds averaged within '
                       'each outer fold, then the mean over the outer folds; positive favours the kNN readout',
    'p_approximate': 'two-sided corrected resampled t over the outer folds: standard error '
                     'sqrt((1/n_folds + test_train_ratio) * variance (ddof 1) of the fold differences), df = n_folds - 1',
    'holm_p_approximate': 'Holm adjustment of p_approximate across the datasets of the family (primary_family_size)',
    'comparators_reproduced': REPRODUCTION_CRITERION,
    'reproduced_by_dataset': 'per comparator and dataset: criterion (1), and criterion (2) on the outer folds and '
                             'fitting seeds of that dataset',
    'verified': 'every saved record of both runs passed ' + '; '.join(VALIDATORS) + '. summary.json model_rows equal '
                'the re-verified model rows field for field and its summaries were recomputed from them. The analysis '
                'source hashes in environment.json are not compared with the current tree; each run\'s recorded code '
                'revision is in provenance.',
}


class RunComparisonError(ValueError):
    """A run is incomplete or unverified, the runs cannot be paired, or the outputs cannot be written."""


@dataclass
class Run:
    label: str
    path: Path
    protocol: dict
    environment: dict
    candidates: dict
    jobs: list
    summary: dict
    manifests: dict
    registry: dict          # {model_id: ModelSpec rebuilt from candidates.json}, in planned-job (registry) order
    schedule: dict
    protocol_sha256: str
    summary_sha256: str

    @property
    def models(self):
        return list(self.registry)

    def provenance(self):
        return {'run': str(self.path), 'protocol_id': self.protocol.get('protocol_id'),
                'protocol_sha256': self.protocol_sha256, 'code_revision': self.environment['code_revision'],
                'summary_sha256': self.summary_sha256, 'registry': self.environment.get('registry'),
                'planned_jobs': len(self.jobs)}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analysis_sources(names=ANALYSIS_SOURCES):
    here = Path(__file__).resolve()
    return {f'experiments/make_revision/{name}': sha256_file(here.with_name(name)) for name in names}


def _reason(exc):
    return str(exc) if type(exc) is ValueError else f'{type(exc).__name__}: {exc}'


def _read(path, where):
    """(parsed JSON, sha256 of the exact bytes)."""
    try:
        data = Path(path).read_bytes()
        return json.loads(data), hashlib.sha256(data).hexdigest()
    except (OSError, ValueError) as exc:
        raise RunComparisonError(f'{where}: unreadable JSON ({exc})') from exc


def _same_summary(recomputed, published):
    if not isinstance(published, dict) or set(recomputed) != set(published):
        return False
    for key, value in recomputed.items():
        other = published[key]
        if isinstance(value, float) and isinstance(other, (int, float)) and not isinstance(other, bool):
            if not np.isclose(value, other, rtol=0, atol=TOLERANCE):
                return False
        elif value != other:
            return False
    return True


# ----------------------------------------------------------------------------- one run

def load_run(source, label):
    """The saved records of one complete run; its evidence is re-verified by verify_run."""
    path = Path(source).resolve()
    if not path.is_dir():
        raise RunComparisonError(f'{label} run {path}: no such run directory')
    missing = [name for name in RUN_RECORDS if not (path/name).is_file()]
    if missing:
        raise RunComparisonError(f'{label} run {path} is incomplete or unverified: missing {", ".join(missing)} '
                                 '(reporting writes summary.json only after verifying every planned job)')
    (protocol, protocol_sha256), (environment, _), (candidates, _), (jobs, _), (summary, summary_sha256) = (
        _read(path/name, f'{label} run {name}') for name in RUN_RECORDS)
    try:
        return _checked_run(label, path, protocol, environment, candidates, jobs, summary, protocol_sha256, summary_sha256)
    except RunComparisonError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
        raise RunComparisonError(f'{label} run {path}: malformed run records ({exc!r})') from exc


def _checked_run(label, path, protocol, environment, candidates, jobs, summary, protocol_sha256, summary_sha256):
    if not protocol.get('frozen'):
        raise RunComparisonError(f'{label} run protocol is not frozen')
    models = list(dict.fromkeys(job['model_id'] for job in jobs))          # registry order, as planned
    if not models or set(models) != set(candidates):
        raise RunComparisonError(f'{label} run planned_jobs.json and candidates.json declare different model families')
    registry = {}
    for model in models:
        entry = candidates[model]
        if (not isinstance(entry['stochastic'], bool)
                or entry['config_ids'] != [config_id(config) for config in entry['candidates']]):
            raise RunComparisonError(f'{label} run candidates.json entry for {model} is not a candidate registry record')
        registry[model] = ModelSpec(model, None, entry['candidates'], entry['stochastic'])
    if canonical_json(jobs) != canonical_json(planned_jobs(protocol['datasets'], protocol, registry)):
        raise RunComparisonError(f'{label} run planned_jobs.json does not declare the complete protocol panel '
                                 '(every dataset x outer fold x model family)')
    absent = [job[key] for job in jobs for key in ('result_file', 'log_file') if not (path/job[key]).is_file()]
    if absent:
        raise RunComparisonError(f'{label} run {path} is incomplete: {len(absent)} of {2*len(jobs)} planned result '
                                 f'files and fit logs are missing (first: {absent[0]})')
    if summary.get('code_revision') != environment['code_revision']:
        raise RunComparisonError(f'{label} run summary.json code revision differs from its environment seal')
    datasets = protocol['datasets']
    for block in ('summaries', 'model_rows'):
        if not isinstance(summary.get(block), dict) or sorted(summary[block]) != sorted(datasets):
            raise RunComparisonError(f'{label} run summary.json {block} does not cover the protocol datasets')
    schedule = expected_schedule(protocol, registry)
    manifests = {}
    for name in datasets:
        if not (path/name/'manifest.json').is_file():
            raise RunComparisonError(f'{label} run {path} is incomplete: missing {name}/manifest.json')
        manifests[name], _ = _read(path/name/'manifest.json', f'{label} run {name}/manifest.json')
        rows = summary['model_rows'][name]
        published = {(row['model_id'], row['metric']): row for row in summary['summaries'][name]}
        if (len(published) != len(summary['summaries'][name])
                or set(published) != {(model, metric) for model in models for metric in METRICS}):
            raise RunComparisonError(f'{label} run summary.json summaries for {name} do not list every model and metric once')
        for model in models:
            for metric in METRICS:
                try:
                    recomputed = summarize_outer(rows, model, metric, expected_folds=schedule['expected_folds'],
                                                 expected_seeds=schedule['expected_seeds'][model])
                except ValueError as exc:
                    raise RunComparisonError(f'{label} run summary.json model_rows for {name} {model}: {exc}') from exc
                if not _same_summary(recomputed, published[model, metric]):
                    raise RunComparisonError(f'{label} run summary.json summaries for {name} {model} {metric} '
                                             'differ from its model_rows')
    return Run(label, path, protocol, environment, candidates, jobs, summary, manifests, registry, schedule,
               protocol_sha256, summary_sha256)


def verify_run(run):
    """Re-verify every saved record of one run with the harness validators; summary.json must be that evidence.

    Returns (re-verified model rows by dataset, prediction cells), where the cells are
    {(dataset, model, outer_repeat, outer_fold, seed): (selected config_id, test sample IDs, predicted labels)}.
    """
    protocol, names = run.protocol, run.protocol['datasets']
    prepared = {}
    for name in names:
        try:
            _, y, manifest, splits = load_prepared(run.path, name)
            declared = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'],
                                   protocol['split_seed'])
        except (OSError, EOFError, zipfile.BadZipFile, KeyError, TypeError, ValueError) as exc:
            raise RunComparisonError(f'{run.label} run prepared data for {name} is unreadable or does not match its '
                                     f'manifest ({_reason(exc)})') from exc
        if manifest.get('dataset_id', name) != name:
            raise RunComparisonError(f'{run.label} run {name}/manifest.json names another dataset')
        if canonical_json(splits) != canonical_json(declared):
            raise RunComparisonError(f'{run.label} run {name}/splits.json is not the nested splits the protocol declares '
                                     '(outer_folds, outer_repeats, inner_folds, split_seed)')
        prepared[name] = (y, manifest, splits)
    try:
        collect_confirmatory_results(run.path, names, protocol, run.registry)
    except (OSError, KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
        raise RunComparisonError(f'{run.label} run fails run_revision.collect_confirmatory_results: {_reason(exc)}') from exc
    rows, cells = {name: [] for name in names}, {}
    for job in run.jobs:
        y, manifest, splits = prepared[job['dataset_id']]
        split = splits[job['outer_repeat']*protocol['outer_folds'] + job['outer_fold']]
        where = f'{run.label} run {job["result_file"]}'
        result, _ = _read(run.path/job['result_file'], where)
        try:
            rows[job['dataset_id']].extend(validate_result_records(result, job, split, y, manifest,
                                                                   run.registry[job['model_id']], protocol,
                                                                   run.environment['code_revision']))
            cells.update(_prediction_cells(result, job, split['test']))
        except (KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
            raise RunComparisonError(f'{where} fails reporting.validate_result_records: {_reason(exc)}') from exc
    for name in names:
        _check_model_rows(run, name, rows[name])
    return rows, cells


def _prediction_cells(result, job, test):
    """The verified per-example predictions of one job, per fitting seed, in test-sample order."""
    identity = tuple(job[key] for key in ('dataset_id', 'model_id', 'outer_repeat', 'outer_fold'))
    labels = {seed: {} for seed in job['model_seeds']}
    for row in result['predictions']:
        labels[row['model_seed']][row['sample_id']] = row['y_pred']
    selected = result['selection']['config_id']
    return {identity + (seed,): (selected, tuple(test), tuple(by_sample[sample] for sample in test))
            for seed, by_sample in labels.items()}


def _differing_fields(verified, published):
    if not isinstance(published, dict):
        return ['(not an object)']
    return [field for field in sorted(set(verified) | set(published))
            if field not in verified or field not in published
            or canonical_json(verified[field]) != canonical_json(published[field])]


def _check_model_rows(run, name, verified):
    """summary.json model_rows of one dataset must be the re-verified model rows: same rows, order and fields."""
    published = run.summary['model_rows'][name]
    if not isinstance(published, list) or len(published) != len(verified):
        count = len(published) if isinstance(published, list) else 'no list of'
        raise RunComparisonError(f'{run.label} run summary.json model_rows for {name} hold {count} rows; the model rows '
                                 f're-verified from the saved records are {len(verified)}')
    for index, (ours, theirs) in enumerate(zip(verified, published)):
        fields = _differing_fields(ours, theirs)
        if fields:
            cell = ' '.join(f'{key} {ours[key]}' for key in ('model_id', 'outer_repeat', 'outer_fold', 'model_seed'))
            raise RunComparisonError(f'{run.label} run summary.json model_rows for {name} differ from the model rows '
                                     f're-verified from the saved records (row {index}, {cell}: {", ".join(fields)})')


# ----------------------------------------------------------------------------- the pair

def check_reference(knn_protocol, bridge):
    """The knn protocol's knn_readout.reference must pin exactly this bridge run."""
    readout = knn_protocol.get('knn_readout')
    if readout is not None and not isinstance(readout, dict):
        raise RunComparisonError('The knn protocol knn_readout must be an object holding the reference block, '
                                 f'not {type(readout).__name__}')
    reference = (readout or {}).get('reference')
    if not isinstance(reference, dict):
        raise RunComparisonError('The knn protocol declares no knn_readout.reference block')
    observed = {'summary_sha256': bridge.summary_sha256, 'code_revision': bridge.environment['code_revision'],
                'protocol_sha256': bridge.protocol_sha256, 'protocol_id': bridge.protocol.get('protocol_id'),
                'model_id': FULL_MODEL}
    wrong = [f'{key}: declared {reference.get(key)!r}, bridge run has {value!r}'
             for key, value in observed.items() if reference.get(key) != value]
    if wrong:
        raise RunComparisonError('The knn protocol reference block does not match the bridge run: ' + '; '.join(wrong))
    return observed


def check_pairing(knn, bridge):
    """Identical panel, dataset and splits hashes, outer schedule, fitting seeds and ArrowFlow candidates."""
    if KNN_MODEL not in knn.registry or FULL_MODEL not in bridge.registry:
        raise RunComparisonError(f'The knn run must hold {KNN_MODEL} and the bridge run {FULL_MODEL}')
    contrasts = knn.protocol.get('primary_contrasts')
    if not isinstance(contrasts, list) or CONTRAST_ID not in contrasts:
        raise RunComparisonError(f'The knn protocol primary_contrasts do not declare {CONTRAST_ID}')
    for key in DESIGN_KEYS:
        if (key not in knn.protocol or key not in bridge.protocol
                or canonical_json(knn.protocol[key]) != canonical_json(bridge.protocol[key])):
            raise RunComparisonError(f'Protocol {key} differs between the runs '
                                     f'(knn {knn.protocol.get(key)!r}, bridge {bridge.protocol.get(key)!r})')
    datasets = knn.protocol['datasets']
    if 'primary_family_size' not in knn.protocol:
        raise RunComparisonError('The knn protocol declares no primary_family_size (the Holm family size is required)')
    size = knn.protocol['primary_family_size']
    if type(size) is not int or size != len(datasets):
        raise RunComparisonError(f'The knn protocol primary_family_size {size!r} differs from its panel of '
                                 f'{len(datasets)} datasets')
    for field in CANDIDATE_FIELDS:
        if canonical_json(knn.candidates[KNN_MODEL][field]) != canonical_json(bridge.candidates[FULL_MODEL][field]):
            raise RunComparisonError(f'{KNN_MODEL} (knn run) and {FULL_MODEL} (bridge run) candidates differ ({field})')
    if knn.schedule['expected_seeds'][KNN_MODEL] != bridge.schedule['expected_seeds'][FULL_MODEL]:
        raise RunComparisonError(f'{KNN_MODEL} and {FULL_MODEL} fitting seed schedules differ')
    comparators = [model for model in knn.models if model != KNN_MODEL]
    bridge_comparators = [model for model in bridge.models if model != FULL_MODEL]
    if sorted(comparators) != sorted(bridge_comparators):
        raise RunComparisonError('The runs hold different comparator families (knn run only: '
                                 f'{sorted(set(comparators) - set(bridge_comparators))}, bridge run only: '
                                 f'{sorted(set(bridge_comparators) - set(comparators))})')
    for name in datasets:
        for key, description in (('dataset_hash', 'dataset hash'), ('splits_hash', 'splits hash')):
            ours, theirs = knn.manifests[name].get(key), bridge.manifests[name].get(key)
            if not ours or ours != theirs:
                raise RunComparisonError(f'The {description} of {name} differs between the runs (knn {ours}, bridge {theirs})')
    return {'datasets': list(datasets), 'dataset_hash': {name: knn.manifests[name]['dataset_hash'] for name in datasets},
            'splits_hash': {name: knn.manifests[name]['splits_hash'] for name in datasets},
            **{key: knn.protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds')},
            'candidate_config_ids': list(knn.candidates[KNN_MODEL]['config_ids']), 'comparators': comparators}


def comparator_reproduction(knn, bridge, comparators, knn_cells, bridge_cells):
    """Every refitted comparator cell against the bridge run under REPRODUCTION_CRITERION; reports, never raises."""
    datasets = knn.protocol['datasets']
    reproduced = {model: dict.fromkeys(datasets, True) for model in comparators}
    definition_mismatches, mismatches, compared = [], [], 0
    for model in comparators:
        for field in CANDIDATE_FIELDS:
            if canonical_json(knn.candidates[model][field]) != canonical_json(bridge.candidates[model][field]):
                definition_mismatches.append({'model_id': model, 'field': field})
                reproduced[model] = dict.fromkeys(datasets, False)
    for name in datasets:
        for model in comparators:
            seeds = list(dict.fromkeys(knn.schedule['expected_seeds'][model] + bridge.schedule['expected_seeds'][model]))
            for repeat, fold in knn.schedule['expected_folds']:
                for seed in seeds:
                    compared += 1
                    key = (name, model, repeat, fold, seed)
                    identity = {'dataset_id': name, 'model_id': model, 'outer_repeat': repeat, 'outer_fold': fold,
                                'model_seed': seed}
                    if key not in knn_cells or key not in bridge_cells:
                        mismatches.append(dict(identity, missing_in='knn run' if key not in knn_cells else 'bridge run'))
                        reproduced[model][name] = False
                        continue
                    (config_knn, samples, labels), (config_bridge, bridge_samples, bridge_labels) = knn_cells[key], bridge_cells[key]
                    theirs = dict(zip(bridge_samples, bridge_labels))
                    differing = [sample for sample, label in zip(samples, labels) if sample not in theirs or theirs[sample] != label]
                    if differing or config_knn != config_bridge or sorted(samples) != sorted(bridge_samples):
                        mismatches.append(dict(identity, n_test=len(samples), n_differing=len(differing), sample_ids=differing,
                                               config_id_knn=config_knn, config_id_bridge=config_bridge))
                        reproduced[model][name] = False
    return {'comparators_reproduced': not mismatches and not definition_mismatches, 'comparators': list(comparators),
            'criterion': REPRODUCTION_CRITERION, 'cells_compared': compared, 'cells_mismatched': len(mismatches),
            'definition_mismatches': definition_mismatches, 'mismatches': mismatches, 'reproduced_by_dataset': reproduced}


def knn_vs_full_contrasts(knn, bridge):
    """Per dataset, arrowflow_full_knn (knn run) minus arrowflow_full (bridge run); Holm across the panel."""
    q, confidence = knn.protocol['test_train_ratio'], knn.protocol['confidence']
    seeds = {KNN_MODEL: knn.schedule['expected_seeds'][KNN_MODEL], FULL_MODEL: bridge.schedule['expected_seeds'][FULL_MODEL]}
    rows = []
    for name in knn.protocol['datasets']:
        combined = ([row for row in knn.summary['model_rows'][name] if row['model_id'] == KNN_MODEL]
                    + [row for row in bridge.summary['model_rows'][name] if row['model_id'] == FULL_MODEL])
        interval = paired_corrected_interval(combined, KNN_MODEL, FULL_MODEL, metric=CONTRAST_METRIC, q=q,
                                             confidence=confidence, expected_folds=knn.schedule['expected_folds'],
                                             expected_seeds=seeds)
        rows.append({'dataset': name, 'model_a': KNN_MODEL, 'run_a': knn.label, 'model_b': FULL_MODEL,
                     'run_b': bridge.label, 'metric': CONTRAST_METRIC, 'confidence': confidence, **interval})
    for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
        row['holm_p_approximate'] = adjusted
    return rows


def main_table(knn, bridge, comparators, reproduction, sources):
    def entry(run, name, model, reproduced):
        row = next(r for r in run.summary['summaries'][name] if (r['model_id'], r['metric']) == (model, TABLE_METRIC))
        return {'model_id': model, 'source_run': run.label, 'mean_error': row['mean'], 'outer_fold_sd': row['outer_fold_sd'],
                'mean_within_fold_seed_sd': row['mean_within_fold_seed_sd'], 'n_folds': row['n_folds'],
                'seeds_per_fold': row['seeds_per_fold'], 'reproduced_bridge_exactly': reproduced}
    datasets = knn.protocol['datasets']
    return {'purpose': 'main_benchmark_table_arrowflow_full_from_the_bridge_run_other_families_from_the_knn_run',
            'metric': TABLE_METRIC, 'datasets': list(datasets), 'models': [FULL_MODEL, KNN_MODEL, *comparators],
            'comparators_reproduced': reproduction['comparators_reproduced'],
            'definitions': {'mean_error': 'mean over outer folds of the fitting-seed-averaged error',
                            'outer_fold_sd': 'SD (ddof 1) of those outer-fold means',
                            'mean_within_fold_seed_sd': 'mean over outer folds of the within-fold SD across fitting seeds; '
                                                        'null for deterministic families',
                            'reproduced_bridge_exactly': 'comparators only: the reproduction criterion '
                                                         f'({SUMMARY_JSON} definitions.comparators_reproduced) holds on '
                                                         'every outer fold and fitting seed of this dataset; null for '
                                                         'the ArrowFlow families'},
            'sources': {'bridge': bridge.provenance(), 'knn': knn.provenance()},
            'analysis_sources': sources,
            'rows': {name: [entry(bridge, name, FULL_MODEL, None), entry(knn, name, KNN_MODEL, None)]
                           + [entry(knn, name, model, reproduction['reproduced_by_dataset'][model][name]) for model in comparators]
                     for name in datasets}}


# ----------------------------------------------------------------------------- outputs and command

def _json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'      # run_revision.write_json format


def _csv_text(rows, columns=CONTRAST_COLUMNS):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows([row[column] for column in columns] for row in rows)
    return buffer.getvalue()


def check_output(output, names=(CONTRASTS_CSV, SUMMARY_JSON, TABLE_JSON)):
    """The output location must be absent or a directory, and each output name absent or a regular file."""
    output = Path(output)
    try:
        if output.exists() and not output.is_dir():
            raise RunComparisonError(f'Output {output} exists and is not a directory')
        for name in names:
            if (output/name).exists() and not (output/name).is_file():
                raise RunComparisonError(f'Output {output/name} exists and is not a regular file')
    except OSError as exc:
        raise RunComparisonError(f'Output {output} cannot be inspected ({_reason(exc)})') from exc


def write_outputs(output, contents):
    """All outputs or none; an existing file whose content differs is never replaced."""
    output = Path(output)
    check_output(output, contents)                  # again at writing time: the location may have changed meanwhile
    payloads = {name: text.encode('utf-8') for name, text in contents.items()}
    try:
        conflicts = [name for name, data in payloads.items() if (output/name).exists() and (output/name).read_bytes() != data]
    except OSError as exc:
        raise RunComparisonError(f'Output {output} cannot be inspected ({_reason(exc)})') from exc
    if conflicts:
        raise FileExistsError(f'Refusing to overwrite {", ".join(conflicts)} in {output}; use a new output directory')
    created = []
    try:
        output.mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            if not (output/name).exists():
                with (output/name).open('xb') as stream:
                    created.append(output/name)
                    stream.write(data)
    except OSError as exc:
        for path in created:
            path.unlink(missing_ok=True)
        raise RunComparisonError(f'Output {output} could not be written ({_reason(exc)}); the files this call created '
                                 'were removed') from exc


def compare_knn(knn_source, bridge_source, output):
    check_output(output)                            # an unusable output location is refused before any run is read
    knn, bridge = load_run(knn_source, 'knn'), load_run(bridge_source, 'bridge')
    reference = check_reference(knn.protocol, bridge)
    pairing = check_pairing(knn, bridge)
    knn_rows, knn_cells = verify_run(knn)
    bridge_rows, bridge_cells = verify_run(bridge)
    comparators = pairing['comparators']
    reproduction = comparator_reproduction(knn, bridge, comparators, knn_cells, bridge_cells)
    contrasts = knn_vs_full_contrasts(knn, bridge)
    datasets, folds = knn.protocol['datasets'], knn.schedule['expected_folds']
    sources = analysis_sources()
    contrasts_csv = _csv_text(contrasts)
    summary = {
        'purpose': 'cross_run_primary_contrast_family_arrowflow_full_knn_minus_arrowflow_full',
        'contrast': CONTRAST_ID,
        'family': {'metric': CONTRAST_METRIC, 'model_a': KNN_MODEL, 'run_a': knn.label, 'model_b': FULL_MODEL,
                   'run_b': bridge.label, 'datasets': list(datasets), 'size': len(datasets), 'multiplicity': 'holm',
                   'test_train_ratio': knn.protocol['test_train_ratio'], 'confidence': knn.protocol['confidence'],
                   'n_folds': len(folds), 'df': len(folds) - 1, 'fitting_seeds': list(knn.protocol['fit_seeds']),
                   'method': 'fitting seeds averaged within each outer fold; corrected resampled t over the outer folds '
                             '(evaluation.paired_corrected_interval on the knn run arrowflow_full_knn model_rows and the '
                             'bridge run arrowflow_full model_rows of each summary.json, both equal to the model rows '
                             're-verified from the saved records); Holm across the datasets (evaluation.holm_adjust)'},
        'definitions': dict(DEFINITIONS),
        'contrasts': contrasts,
        'comparators_reproduced': reproduction['comparators_reproduced'],
        'reproduction': reproduction,
        'outputs': {CONTRASTS_CSV: {'sha256': hashlib.sha256(contrasts_csv.encode('utf-8')).hexdigest(),
                                    'rows': len(contrasts)}},
        'provenance': {'knn': knn.provenance(), 'bridge': bridge.provenance(), 'reference': reference, 'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS),
                                        **{run.label: {'jobs_verified': len(run.jobs),
                                                       'model_rows_verified': sum(map(len, rows.values()))}
                                           for run, rows in ((knn, knn_rows), (bridge, bridge_rows))}},
                       'analysis_sources': sources},
    }
    table = main_table(knn, bridge, comparators, reproduction, sources)
    write_outputs(output, {CONTRASTS_CSV: contrasts_csv, SUMMARY_JSON: _json_text(summary), TABLE_JSON: _json_text(table)})
    return {'contrasts': contrasts, 'summary': summary, 'main_table': table}


# ----------------------------------------------------------------------------- training controls (Task 20A)

UNTRAINED_MODEL, INPUT_MODEL = CONTROL_MODELS
TRAINING_CONTRASTS_CSV, TRAINING_CONTRASTS_JSON = 'training_contrasts.csv', 'training_contrasts.json'
TRAINING_TABLE_JSON, TRAINING_DEPTH_JSON = 'training_error_table.json', 'training_depth_split.json'
TRAINING_OUTPUTS = (TRAINING_CONTRASTS_CSV, TRAINING_CONTRASTS_JSON, TRAINING_TABLE_JSON, TRAINING_DEPTH_JSON)
TRAINING_CONTRAST_COLUMNS = ('family_index', 'dataset', 'contrast', 'model_a', 'run_a', 'model_b', 'run_b', 'mean_difference',
                             'standard_error', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate', 'n_folds', 'df')
TRAINING_ANALYSIS_SOURCES = ANALYSIS_SOURCES + ('knn_controls.py',)
SHARED_SOURCES = ('arrowflow/arrowflow.py', 'arrowflow/benchmark.py', 'arrowflow/config.py', 'arrowflow/ranking.py',
                  'experiments/make_revision/bridge.py', 'experiments/make_revision/comparisons.py',
                  'experiments/make_revision/datasets.py', 'experiments/make_revision/evaluation.py',
                  'experiments/make_revision/models.py', 'experiments/make_revision/multiview.py',
                  'experiments/make_revision/reporting.py', 'experiments/make_revision/run_revision.py',
                  'experiments/make_revision/secondary_studies.py')
TRAINING_DEFINITIONS = {
    'mean_difference': f'{KNN_MODEL} (knn run) minus the control (training run) accuracy: fitting seeds averaged within '
                       'each outer fold, then the mean over the outer folds; positive favours the trained ArrowFlow-kNN',
    'p_approximate': DEFINITIONS['p_approximate'],
    'holm_p_approximate': 'Holm adjustment of p_approximate across every member of the family (primary_family_size: '
                          'two contrasts per dataset)',
    'family_index': 'position in the family: datasets in panel order, the untrained control before the input control',
    'candidates': 'each control\'s candidates equal the knn run\'s arrowflow_full_knn candidates projected onto '
                  'knn_controls.CANDIDATE_KEYS, duplicates removed, in canonical config_id order',
    'shared_sources': 'every source hash both environment records carry is identical, and SHARED_SOURCES are among them',
    'verified': DEFINITIONS['verified'],
}


def load_prepared_run(source, label):
    """The prepared records of a run whose jobs have not started: protocol, environment, candidates and manifests."""
    path = Path(source).resolve()
    names = ('protocol.json', 'environment.json', 'candidates.json')
    missing = [name for name in names if not (path/name).is_file()]
    if missing:
        raise RunComparisonError(f'{label} directory {path} is not a prepared run (missing {", ".join(missing)})')
    (protocol, protocol_sha256), (environment, _), (candidates, _) = (_read(path/name, f'{label} {name}') for name in names)
    try:
        registry = {}
        for model, entry in candidates.items():
            if (not isinstance(entry['stochastic'], bool)
                    or entry['config_ids'] != [config_id(config) for config in entry['candidates']]):
                raise RunComparisonError(f'{label} candidates.json entry for {model} is not a candidate registry record')
            registry[model] = ModelSpec(model, None, entry['candidates'], entry['stochastic'])
        manifests = {}
        for name in protocol['datasets']:
            if not (path/name/'manifest.json').is_file():
                raise RunComparisonError(f'{label} directory {path} is not prepared: missing {name}/manifest.json')
            manifests[name], _ = _read(path/name/'manifest.json', f'{label} {name}/manifest.json')
        schedule = expected_schedule(protocol, registry)
    except RunComparisonError:
        raise
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as exc:
        raise RunComparisonError(f'{label} directory {path}: malformed prepared records ({exc!r})') from exc
    return SimpleNamespace(label=label, path=path, protocol=protocol, environment=environment, candidates=candidates,
                           manifests=manifests, registry=registry, schedule=schedule, protocol_sha256=protocol_sha256)


def check_training_reference(training_protocol, knn):
    """The training protocol's training_controls.reference must pin exactly this knn run."""
    block = training_protocol.get('training_controls')
    reference = block.get('reference') if isinstance(block, dict) else None
    if not isinstance(reference, dict):
        raise RunComparisonError('The training protocol declares no training_controls.reference block')
    observed = {'summary_sha256': knn.summary_sha256, 'code_revision': knn.environment['code_revision'],
                'protocol_sha256': knn.protocol_sha256, 'protocol_id': knn.protocol.get('protocol_id'), 'model_id': KNN_MODEL}
    wrong = [f'{key}: declared {reference.get(key)!r}, knn run has {value!r}'
             for key, value in observed.items() if reference.get(key) != value]
    if wrong:
        raise RunComparisonError('The training protocol reference block does not match the knn run: ' + '; '.join(wrong))
    return observed


def check_training_pairing(training, knn):
    """Identical panel, dataset and splits hashes, nested design and fitting seeds; the controls' candidates are
    projections of arrowflow_full_knn's; every source both runs seal is byte-identical."""
    if KNN_MODEL not in knn.registry:
        raise RunComparisonError(f'The knn run holds no {KNN_MODEL}')
    if sorted(training.registry) != sorted(CONTROL_MODELS):
        raise RunComparisonError(f'The training run must hold exactly {list(CONTROL_MODELS)}, not {sorted(training.registry)}')
    if training.protocol.get('primary_contrasts') != TRAINING_CONTRASTS:
        raise RunComparisonError(f'The training protocol primary_contrasts must be {TRAINING_CONTRASTS}')
    for key in DESIGN_KEYS:
        if (key not in training.protocol or key not in knn.protocol
                or canonical_json(training.protocol[key]) != canonical_json(knn.protocol[key])):
            raise RunComparisonError(f'Protocol {key} differs between the runs '
                                     f'(training {training.protocol.get(key)!r}, knn {knn.protocol.get(key)!r})')
    datasets = training.protocol['datasets']
    size = training.protocol.get('primary_family_size')
    if type(size) is not int or size != len(CONTROL_MODELS) * len(datasets):
        raise RunComparisonError(f'The training protocol primary_family_size {size!r} is not two contrasts for each of '
                                 f'its {len(datasets)} datasets')
    seeds = knn.schedule['expected_seeds'][KNN_MODEL]
    for model in CONTROL_MODELS:
        if training.schedule['expected_seeds'][model] != seeds:
            raise RunComparisonError(f'{model} and {KNN_MODEL} fitting seed schedules differ')
        expected = project_candidates(knn.candidates[KNN_MODEL]['candidates'], CANDIDATE_KEYS[model])
        if canonical_json(training.candidates[model]['candidates']) != canonical_json(expected):
            raise RunComparisonError(f'{model} candidates are not the projection of the knn run {KNN_MODEL} candidates '
                                     f'onto {list(CANDIDATE_KEYS[model])}')
    for name in datasets:
        for key, description in (('dataset_hash', 'dataset hash'), ('splits_hash', 'splits hash')):
            ours, theirs = training.manifests[name].get(key), knn.manifests[name].get(key)
            if not ours or ours != theirs:
                raise RunComparisonError(f'The {description} of {name} differs between the runs (training {ours}, knn {theirs})')
    ours, theirs = training.environment.get('source_hashes') or {}, knn.environment.get('source_hashes') or {}
    absent = [source for source in SHARED_SOURCES if source not in ours or source not in theirs]
    if absent:
        raise RunComparisonError('Both environment records must seal ' + ', '.join(absent))
    differing = sorted(source for source in set(ours) & set(theirs) if ours[source] != theirs[source])
    if differing:
        raise RunComparisonError('Sources sealed by both runs differ: ' + ', '.join(differing))
    try:
        depths = validate_depths(((training.protocol.get('training_controls') or {}).get('depth_split') or {}).get('depths'))
    except ValueError as exc:
        raise RunComparisonError(f'The training protocol training_controls.{exc}') from exc
    return {'datasets': list(datasets), 'dataset_hash': {name: training.manifests[name]['dataset_hash'] for name in datasets},
            'splits_hash': {name: training.manifests[name]['splits_hash'] for name in datasets},
            **{key: training.protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds')},
            'candidate_config_ids': {model: list(training.candidates[model]['config_ids']) for model in CONTROL_MODELS},
            'reference_candidate_config_ids': list(knn.candidates[KNN_MODEL]['config_ids']),
            'shared_sources': {source: ours[source] for source in sorted(set(ours) & set(theirs))}, 'depths': depths}


def check_training_prepared(training_source, knn_source):
    """Before a knn_training run: its prepared records against the complete knn run (reference pins and pairing)."""
    training, knn = load_prepared_run(training_source, 'training'), load_run(knn_source, 'knn')
    return {'reference': check_training_reference(training.protocol, knn), 'pairing': check_training_pairing(training, knn)}


def _model_rows(run, name, model):
    return [row for row in run.summary['model_rows'][name] if row['model_id'] == model]


def training_contrasts(training, knn):
    """Per dataset, arrowflow_full_knn (knn run) minus each control (training run); one Holm adjustment across all."""
    q, confidence = training.protocol['test_train_ratio'], training.protocol['confidence']
    rows = []
    for name in training.protocol['datasets']:
        for model, contrast in zip(CONTROL_MODELS, TRAINING_CONTRASTS):
            seeds = {KNN_MODEL: knn.schedule['expected_seeds'][KNN_MODEL], model: training.schedule['expected_seeds'][model]}
            interval = paired_corrected_interval(_model_rows(knn, name, KNN_MODEL) + _model_rows(training, name, model),
                                                 KNN_MODEL, model, metric=CONTRAST_METRIC, q=q, confidence=confidence,
                                                 expected_folds=training.schedule['expected_folds'], expected_seeds=seeds)
            rows.append({'family_index': len(rows) + 1, 'dataset': name, 'contrast': contrast, 'model_a': KNN_MODEL,
                         'run_a': knn.label, 'model_b': model, 'run_b': training.label, 'metric': CONTRAST_METRIC,
                         'confidence': confidence, **interval})
    for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
        row['holm_p_approximate'] = adjusted
    return rows


def training_error_table(training, knn, sources):
    def entry(run, name, model):
        row = next(r for r in run.summary['summaries'][name] if (r['model_id'], r['metric']) == (model, TABLE_METRIC))
        return {'model_id': model, 'source_run': run.label, 'mean_error': row['mean'], 'outer_fold_sd': row['outer_fold_sd'],
                'mean_within_fold_seed_sd': row['mean_within_fold_seed_sd'], 'n_folds': row['n_folds'],
                'seeds_per_fold': row['seeds_per_fold']}
    datasets = training.protocol['datasets']
    return {'purpose': 'mean_outer_error_of_arrowflow_knn_and_its_training_controls', 'metric': TABLE_METRIC,
            'datasets': list(datasets), 'models': [KNN_MODEL, *CONTROL_MODELS],
            'definitions': {'mean_error': 'mean over outer folds of the fitting-seed-averaged error',
                            'outer_fold_sd': 'SD (ddof 1) of those outer-fold means',
                            'mean_within_fold_seed_sd': 'mean over outer folds of the within-fold SD across fitting seeds'},
            'sources': {'knn': knn.provenance(), 'training': training.provenance()}, 'analysis_sources': sources,
            'rows': {name: [entry(knn, name, KNN_MODEL)] + [entry(training, name, model) for model in CONTROL_MODELS]
                     for name in datasets}}


def training_depth_split(training, knn, depths, sources):
    """arrowflow_full_knn minus arrowflow_knn_untrained by the widths arrowflow_full_knn selected per outer fold."""
    q, confidence = training.protocol['test_train_ratio'], training.protocol['confidence']
    seeds = {KNN_MODEL: knn.schedule['expected_seeds'][KNN_MODEL], UNTRAINED_MODEL: training.schedule['expected_seeds'][UNTRAINED_MODEL]}
    by_dataset = {}
    for name in training.protocol['datasets']:
        trained, untrained = _model_rows(knn, name, KNN_MODEL), _model_rows(training, name, UNTRAINED_MODEL)
        entries = depth_split(trained + untrained, KNN_MODEL, UNTRAINED_MODEL, selected_widths(trained, KNN_MODEL),
                              depths=depths, folds=training.schedule['expected_folds'], seeds=seeds, q=q, confidence=confidence)
        own = selected_widths(untrained, UNTRAINED_MODEL)
        for entry in entries:
            entry['untrained_selected_the_same_widths'] = sum(own[tuple(fold)] == entry['widths'] for fold in entry['folds'])
        by_dataset[name] = entries
    return {'purpose': 'training_effect_by_the_depth_arrowflow_full_knn_selected_descriptive',
            'difference': f'{KNN_MODEL} (knn run) minus {UNTRAINED_MODEL} (training run) accuracy, fitting seeds averaged within fold',
            'grouping': f'the hidden widths of the configuration {KNN_MODEL} selected on the inner folds of each outer fold',
            'status': 'descriptive; outside the Holm family; no p values', 'depths': depths,
            'definitions': {'interval': 'corrected resampled t over the folds of the group only (at least two folds; '
                                        'q = test_train_ratio, df = n_folds - 1); descriptive',
                            'untrained_selected_the_same_widths': f'folds of the group in which {UNTRAINED_MODEL} selected '
                                                                  'the same widths on its own inner folds',
                            'pooled': 'across datasets per group: the dataset-fold count and the mean and SD of their '
                                      'differences; no interval'},
            'by_dataset': by_dataset, 'pooled': pooled_depth_split(by_dataset, depths),
            'sources': {'knn': knn.provenance(), 'training': training.provenance()}, 'analysis_sources': sources}


def compare_training(training_source, knn_source, output):
    check_output(output, TRAINING_OUTPUTS)          # an unusable output location is refused before any run is read
    training, knn = load_run(training_source, 'training'), load_run(knn_source, 'knn')
    reference = check_training_reference(training.protocol, knn)
    pairing = check_training_pairing(training, knn)
    training_rows, _ = verify_run(training)
    knn_rows, _ = verify_run(knn)
    sources = analysis_sources(TRAINING_ANALYSIS_SOURCES)
    contrasts = training_contrasts(training, knn)
    contrasts_csv = _csv_text(contrasts, TRAINING_CONTRAST_COLUMNS)
    folds = training.schedule['expected_folds']
    summary = {
        'purpose': 'cross_run_primary_contrast_family_arrowflow_full_knn_minus_its_training_controls',
        'contrasts_declared': list(TRAINING_CONTRASTS),
        'family': {'metric': CONTRAST_METRIC, 'model_a': KNN_MODEL, 'run_a': knn.label, 'models_b': list(CONTROL_MODELS),
                   'run_b': training.label, 'datasets': list(training.protocol['datasets']), 'size': len(contrasts),
                   'multiplicity': 'holm', 'test_train_ratio': training.protocol['test_train_ratio'],
                   'confidence': training.protocol['confidence'], 'n_folds': len(folds), 'df': len(folds) - 1,
                   'fitting_seeds': list(training.protocol['fit_seeds']),
                   'method': 'fitting seeds averaged within each outer fold; corrected resampled t over the outer folds '
                             '(evaluation.paired_corrected_interval on the knn run arrowflow_full_knn model_rows and the '
                             'training run control model_rows of each summary.json, both equal to the model rows '
                             're-verified from the saved records); Holm across all members (evaluation.holm_adjust)'},
        'definitions': dict(TRAINING_DEFINITIONS),
        'contrasts': contrasts,
        'outputs': {TRAINING_CONTRASTS_CSV: {'sha256': hashlib.sha256(contrasts_csv.encode('utf-8')).hexdigest(),
                                             'rows': len(contrasts)}},
        'provenance': {'training': training.provenance(), 'knn': knn.provenance(), 'reference': reference, 'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS),
                                        **{run.label: {'jobs_verified': len(run.jobs),
                                                       'model_rows_verified': sum(map(len, rows.values()))}
                                           for run, rows in ((training, training_rows), (knn, knn_rows))}},
                       'analysis_sources': sources},
    }
    table = training_error_table(training, knn, sources)
    depth = training_depth_split(training, knn, pairing['depths'], sources)
    write_outputs(output, {TRAINING_CONTRASTS_CSV: contrasts_csv, TRAINING_CONTRASTS_JSON: _json_text(summary),
                           TRAINING_TABLE_JSON: _json_text(table), TRAINING_DEPTH_JSON: _json_text(depth)})
    return {'contrasts': contrasts, 'summary': summary, 'error_table': table, 'depth_split': depth}


def _main_knn(parser, args):
    try:
        result = compare_knn(args.knn_source, args.bridge_source, args.output)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_runs knn refused: {exc}\n')
    for row in result['contrasts']:
        print(f"{row['dataset']}: {KNN_MODEL} - {FULL_MODEL} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} Holm p={row['holm_p_approximate']:.3g}")
    reproduction = result['summary']['reproduction']
    print(f"comparators_reproduced: {reproduction['comparators_reproduced']} ({reproduction['cells_mismatched']} of "
          f"{reproduction['cells_compared']} comparator cells differ; {len(reproduction['definition_mismatches'])} "
          f"definition mismatches)")


def _main_training(parser, args):
    try:
        result = compare_training(args.training_source, args.knn_source, args.output)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_runs training refused: {exc}\n')
    for row in result['contrasts']:
        print(f"{row['dataset']}: {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} Holm p={row['holm_p_approximate']:.3g}")
    for entry in result['depth_split']['pooled']:
        mean = 'none' if entry['mean_difference'] is None else f"{entry['mean_difference']:+.4f}"
        print(f"widths {entry['widths']}: {KNN_MODEL} - {UNTRAINED_MODEL} mean {mean} over {entry['n_dataset_folds']} "
              'dataset-folds (descriptive)')


def _main_training_pairing(parser, args):
    try:
        record = check_training_prepared(args.training_source, args.knn_source)
    except RunComparisonError as exc:
        parser.exit(2, f'compare_runs training-pairing refused: {exc}\n')
    pairing = record['pairing']
    print(f"paired with {record['reference']['protocol_id']} at {record['reference']['code_revision']}: "
          f"{len(pairing['datasets'])} datasets, fitting seeds {pairing['fit_seeds']}, "
          f"{len(pairing['shared_sources'])} shared sealed sources identical")


def _main_projected(parser, args):
    from .compare_projected import main_projected           # Task 23A; imported on use because it imports this module
    main_projected(parser, args)


def _main_projected_pairing(parser, args):
    from .compare_projected import main_projected_pairing
    main_projected_pairing(parser, args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    knn = commands.add_parser('knn', help=f'{KNN_MODEL} (knn run) versus {FULL_MODEL} (bridge run)')
    knn.add_argument('--knn-source', type=Path, required=True, help='complete bridge_knn run directory (summary.json written)')
    knn.add_argument('--bridge-source', type=Path, required=True, help='complete bridge run directory (summary.json written)')
    knn.add_argument('--output', type=Path, required=True, help=f'directory for {CONTRASTS_CSV}, {SUMMARY_JSON}, {TABLE_JSON}')
    training = commands.add_parser('training', help=f'{KNN_MODEL} (knn run) versus its training controls (knn_training run)')
    training.add_argument('--training-source', type=Path, required=True,
                          help='complete knn_training run directory (summary.json written)')
    training.add_argument('--knn-source', type=Path, required=True, help='complete bridge_knn run directory (summary.json written)')
    training.add_argument('--output', type=Path, required=True, help='directory for ' + ', '.join(TRAINING_OUTPUTS))
    pairing = commands.add_parser('training-pairing', help='check a prepared knn_training directory against the complete '
                                                           'bridge_knn run before its jobs start; writes nothing')
    pairing.add_argument('--training-source', type=Path, required=True, help='prepared knn_training run directory')
    pairing.add_argument('--knn-source', type=Path, required=True, help='complete bridge_knn run directory (summary.json written)')
    projected = commands.add_parser('projected', help='projected_numeric_knn (knn_projected run) against the knn and '
                                                      'knn_training runs (compare_projected.py)')
    projected.add_argument('--projected-source', type=Path, required=True,
                           help='complete knn_projected run directory (summary.json written)')
    projected_pairing = commands.add_parser('projected-pairing', help='check a prepared knn_projected directory against the '
                                                                      'complete knn and knn_training runs; writes nothing')
    projected_pairing.add_argument('--projected-source', type=Path, required=True, help='prepared knn_projected run directory')
    for command in (projected, projected_pairing):
        command.add_argument('--knn-source', type=Path, required=True,
                             help='complete bridge_knn run directory (summary.json written)')
        command.add_argument('--training-source', type=Path, required=True,
                             help='complete knn_training run directory (summary.json written)')
    projected.add_argument('--output', type=Path, required=True, help='directory for the four compare_projected outputs')
    args = parser.parse_args(argv)
    {'knn': _main_knn, 'training': _main_training, 'training-pairing': _main_training_pairing,
     'projected': _main_projected, 'projected-pairing': _main_projected_pairing}[args.command](parser, args)


if __name__ == '__main__':
    main()
