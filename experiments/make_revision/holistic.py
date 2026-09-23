"""Task 24: the combined analysis of the seven benchmark datasets and the ten further datasets; no model is fitted.

The benchmark datasets come from the bridge_knn run (ArrowFlow = arrowflow_full_knn, the majority class and the five tuned
classical models), the knn_training run (the untrained ArrowFlow and tuned input footrule kNN) and the knn_projected run
(unsorted projected kNN); the further datasets come from the two newdata batch runs, which hold all ten models. Every run is
loaded and re-verified with compare_runs.load_run and compare_runs.verify_run (prepared data and splits hashes, the declared
nested splits, planned jobs and fit logs, every result record with its per-example predictions, summary.json against the
re-verified model rows), and the registered pairings are re-checked with the existing checks: compare_runs
check_training_reference and check_training_pairing, compare_projected check_projected_references and
check_projected_pairing, and compare_newdata.check_batches (the committed frozen batch protocols and the dataset pins). The
runs must share the nested design, fold schedule, fitting seeds and candidate records. Every number is recomputed from the
verified model rows with the functions that produced the published analyses; each published analysis must record the
provenance of the verified runs and must agree with the recomputation (absolute tolerance 1e-12). Anything else is refused
(exit status 2) before an output is written. Outputs are written all or none, and a file with different content is never
replaced.

python -m experiments.make_revision.holistic benchmark --output O [--runs R]
  main_table.csv            17 datasets x ArrowFlow, the five tuned classical models and the majority class: mean outer
                            error and outer-fold SD
  comparator_intervals.csv  ArrowFlow minus each tuned classical model, accuracy: seed-averaged corrected resampled t (q 0.25,
                            14 df), unadjusted; descriptive
  competitiveness.csv       per dataset: the best tuned classical model (majority class excluded), the gap, the
                            within-three-points and best-on flags, and the ceiling, near-majority and no-learning flags under
                            the fixed rules of RULES
  rank_matrix.csv, mean_ranks.csv   Friedman ranks over 17 datasets x 6 models (majority class excluded); descriptive
  complete_metrics.csv      17 datasets x the ten models: error, balanced accuracy and macro-F1, each with its outer-fold SD
                            and within-fold seed SD
  selected_widths.csv       ArrowFlow's selected hidden widths per outer fold (benchmark and further folds)
  duplicates.csv            the exact-duplicate share of test rows per dataset (both duplicate audits, recomputed)
  benchmark.json            every result with the Friedman, Iman-Davenport and Nemenyi record, definitions, counts, the
                            cross-checks and provenance
python -m experiments.make_revision.holistic training --output O [--runs R]
  training_controls.csv     17 rows: ArrowFlow minus the untrained ArrowFlow and minus tuned input footrule kNN, accuracy,
                            with the interval, the unadjusted p, the registered Holm p and the registered family
  holm34_sensitivity.csv    the 34 training contrasts under one Holm adjustment; post hoc, descriptive
  ladder.csv, ladder_detail.csv   mean error of raw numeric kNN, unsorted projected kNN, encoded-ranking kNN, untrained
                            ArrowFlow and ArrowFlow (the detail adds the SDs)
  depth_split.csv, depth_pooled.csv   ArrowFlow minus the untrained ArrowFlow by the hidden widths ArrowFlow selected;
                            descriptive
  training.json             every result, the prespecified stratum test and the Spearman result passed through from the
                            newdata analysis (and recomputed), definitions and provenance
python -m experiments.make_revision.holistic components --output O [--runs R]
  refuses (exit status 2, nothing written) until both ablation runs are complete (run_knn_ablation on the benchmark
  datasets, run_newdata_ablation on the further datasets); then re-verifies each with its runner's summary (every job, the
  views7 reproduction and every sealed selection re-derived from its reference) and requires the published summary JSON and
  CSV to equal the recomputation
  components.csv            every variant's change from views7 per dataset as ArrowFlow (views7) minus the variant, accuracy,
                            with the interval (the published variant-minus-views7 interval negated); descriptive
  components_depth.csv, components_depth_pooled.csv   untrained minus views7 accuracy by the selected hidden widths, with
                            views7 minus untrained beside it; descriptive
  components.json           every result, definitions and provenance
python -m experiments.make_revision.holistic ready --root D
  checks D/benchmark and D/training (every sealed output present and unchanged, sources committed, identical analysis
  sources) and writes D/READY with the commit hash and the output file list
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .compare_newdata import LABELS as BATCH_LABELS, check_batches, permutation_test, spearman_exact
from .compare_projected import check_projected_pairing, check_projected_references
from .compare_runs import (VALIDATORS, RunComparisonError, _json_text, _reason, check_output, check_training_pairing,
                           check_training_reference, load_run, verify_run, write_outputs)
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .knn_controls import (CONTROL_MODELS, INPUT_MODEL, TRAINED_MODEL, UNTRAINED_MODEL, depth_split, pooled_depth_split,
                           selected_widths, validate_depths)
from .newdata import LADDER, MODEL_ORDER, PROJECTED_MODEL
from .referee_analyses import (COUNT_FIELDS, RANK_MODELS, check_design, code_record, display_value, duplicate_groups, finish,
                               friedman_nemenyi, training_duplicate_flags)
from .run_revision import load_prepared

REPO = Path(__file__).resolve().parents[2]
WORKSPACE_RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'
RUN_LABELS = ('knn', 'training', 'projected', *BATCH_LABELS)
SOURCES = {'knn': '2026-09-12-bridge-knn', 'training': '2026-09-13-knn-training', 'projected': '2026-09-13-knn-projected',
           'batch1': '2026-09-14-newdata-batch1', 'batch2': '2026-09-14-newdata-batch2',
           'knn_vs_full': '2026-09-12-knn-vs-full', 'referee': '2026-09-13-referee-analyses',
           'training_compare': '2026-09-13-knn-training/compare_training',
           'projected_compare': '2026-09-13-knn-projected/compare_projected', 'newdata_analysis': '2026-09-14-newdata-analysis'}
PUBLISHED = {
    'main_table': ('knn_vs_full', 'main_table.json'),
    'comparator_csv': ('referee', 'comparators/comparator_contrasts.csv'),
    'comparator_json': ('referee', 'comparators/comparator_contrasts.json'),
    'ranks_json': ('referee', 'ranks/friedman_nemenyi.json'),
    'duplicate_csv': ('referee', 'duplicates/duplicate_groups.csv'),
    'duplicate_json': ('referee', 'duplicates/duplicate_sensitivity.json'),
    'training_csv': ('training_compare', 'training_contrasts.csv'),
    'training_json': ('training_compare', 'training_contrasts.json'),
    'training_depth_json': ('training_compare', 'training_depth_split.json'),
    'projected_ladder_json': ('projected_compare', 'projected_ladder_error_table.json'),
    'newdata_families_csv': ('newdata_analysis', 'newdata_families.csv'),
    'newdata_comparators_csv': ('newdata_analysis', 'newdata_comparators.csv'),
    'newdata_ladder_csv': ('newdata_analysis', 'newdata_ladder.csv'),
    'newdata_duplicates_csv': ('newdata_analysis', 'newdata_duplicates.csv'),
    'newdata_analysis_json': ('newdata_analysis', 'newdata_analysis.json'),
}
# Which run each published record names, and which CSV of it the record seals: (record, path to the run block, run label).
RECORDED_RUNS = (('main_table', ('sources', 'knn'), 'knn'), ('comparator_json', ('provenance', 'knn'), 'knn'),
                 ('ranks_json', ('provenance', 'knn'), 'knn'), ('duplicate_json', ('provenance', 'knn'), 'knn'),
                 ('training_json', ('provenance', 'training'), 'training'), ('training_json', ('provenance', 'knn'), 'knn'),
                 ('training_depth_json', ('sources', 'training'), 'training'), ('training_depth_json', ('sources', 'knn'), 'knn'),
                 ('projected_ladder_json', ('sources', 'projected'), 'projected'),
                 ('projected_ladder_json', ('sources', 'knn'), 'knn'), ('projected_ladder_json', ('sources', 'training'), 'training'),
                 ('newdata_analysis_json', ('provenance', 'batch1'), 'batch1'),
                 ('newdata_analysis_json', ('provenance', 'batch2'), 'batch2'))
SEALS = (('comparator_json', 'comparator_csv'), ('duplicate_json', 'duplicate_csv'), ('training_json', 'training_csv'),
         ('newdata_analysis_json', 'newdata_families_csv'), ('newdata_analysis_json', 'newdata_comparators_csv'),
         ('newdata_analysis_json', 'newdata_ladder_csv'), ('newdata_analysis_json', 'newdata_duplicates_csv'))
BENCHMARK_PUBLISHED = ('main_table', 'comparator_csv', 'comparator_json', 'ranks_json', 'duplicate_csv', 'duplicate_json',
                       'training_depth_json', 'newdata_comparators_csv', 'newdata_duplicates_csv', 'newdata_analysis_json')
TRAINING_PUBLISHED = ('training_csv', 'training_json', 'training_depth_json', 'projected_ladder_json', 'newdata_families_csv',
                      'newdata_ladder_csv', 'newdata_analysis_json')
ANALYSIS_SOURCES = ('holistic.py', 'compare_runs.py', 'compare_projected.py', 'compare_newdata.py', 'referee_analyses.py',
                    'newdata.py', 'knn_controls.py', 'projected_knn.py', 'evaluation.py', 'reporting.py', 'run_revision.py')

BENCHMARK, FURTHER = 'benchmark', 'further'
TRAINED, UNTRAINED, INPUT = TRAINED_MODEL, UNTRAINED_MODEL, INPUT_MODEL
CLASSICAL = ('svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting')
MAJORITY = 'dummy'
MAIN_MODELS = (TRAINED, *CLASSICAL, MAJORITY)
ALL_MODELS = tuple(MODEL_ORDER)
CONTROL_PREFIX = {UNTRAINED: 'untrained', INPUT: 'input'}
NEWDATA_FAMILY = {UNTRAINED: 'primary', INPUT: 'secondary'}
DESIGN_KEYS = ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds', 'selection_metric', 'test_train_ratio',
               'confidence')
RUN_FIELDS = ('protocol_id', 'protocol_sha256', 'code_revision', 'summary_sha256', 'registry', 'planned_jobs')
TOLERANCE = 1e-12
POINT_DECIMALS = 9
ALPHA = .05
DESCRIPTIVE = 'descriptive; no multiplicity adjustment'
POST_HOC = 'post hoc sensitivity analysis; descriptive'
RULES = {
    'unit': 'percentage points of mean outer error (the summary.json mean times 100), rounded to 9 decimals before any '
            'comparison so that binary floating-point noise cannot move a flag',
    'best_classical': 'the tuned classical model (svc_rbf, random_forest, mlp, numeric_knn, gradient_boosting; the majority '
                      'class excluded) with the lowest mean outer error; every model tied at that minimum is listed',
    'gap_points': 'ArrowFlow mean error minus the best tuned classical mean error; positive when ArrowFlow trails',
    'within_three_points': 'gap_points <= 3.0',
    'within_three_points_as_printed': 'the same rule on the gap printed with two decimals (gap_points_2dp) and on the gap of '
                                      'the two errors printed with one decimal (gap_points_from_1dp_errors); true only if both '
                                      'agree with within_three_points, so a count never contradicts a printed value',
    'best_on': 'ArrowFlow mean error strictly below the mean error of every tuned classical model (gap_points < 0)',
    'ceiling': 'the best tuned classical mean error is below 1.0 point',
    'near_majority': 'ArrowFlow mean error within 0.5 points (absolute difference, inclusive) of the majority-class mean error, '
                     'while the best tuned classical model is at least 2.0 points better than ArrowFlow (gap_points >= 2.0)',
    'no_learning': 'the majority class has the lowest mean error of all models: no model among ArrowFlow and the five tuned '
                   'classical models has a strictly lower mean error than the majority class',
    'degenerate': 'ceiling or near_majority or no_learning',
    'thresholds': {'within_points': 3.0, 'ceiling_points': 1.0, 'near_majority_points': .5, 'near_majority_advantage_points': 2.0},
}
THRESHOLDS = RULES['thresholds']


# ----------------------------------------------------------------------------- the verified panel

class Panel:
    """The five verified runs; the benchmark datasets in bridge_knn panel order, then the further datasets in newdata panel
    order. Benchmark models live in the knn run (ArrowFlow, classical, majority), the training run (the two controls) and
    the projected run; every model of a further dataset lives in the batch run holding the dataset."""
    def __init__(self, runs):
        self.runs = runs
        self.benchmark = list(runs['knn'].protocol['datasets'])
        self.further = list(runs['batch1'].protocol['analysis']['datasets'])
        self.datasets = self.benchmark + self.further
        self.group = {**dict.fromkeys(self.benchmark, BENCHMARK), **dict.fromkeys(self.further, FURTHER)}
        self.batch_of = {name: int(key) for key, members in runs['batch1'].protocol['batches'].items() for name in members}
        self.q = runs['knn'].protocol['test_train_ratio']
        self.confidence = runs['knn'].protocol['confidence']
        self.folds = [tuple(fold) for fold in runs['knn'].schedule['expected_folds']]

    def run_for(self, name, model):
        if self.group[name] == FURTHER:
            return self.runs[f'batch{self.batch_of[name]}']
        return self.runs['training' if model in CONTROL_MODELS else 'projected' if model == PROJECTED_MODEL else 'knn']

    def rows(self, name, model):
        return [row for row in self.run_for(name, model).summary['model_rows'][name] if row['model_id'] == model]

    def seeds(self, name, model):
        return list(self.run_for(name, model).schedule['expected_seeds'][model])

    def summary(self, name, model, metric):
        run = self.run_for(name, model)
        return next(row for row in run.summary['summaries'][name] if (row['model_id'], row['metric']) == (model, metric))

    def interval(self, name, model_a, model_b):
        """model_a minus model_b accuracy: seeds averaged within fold, corrected resampled t over the outer folds."""
        return paired_corrected_interval(self.rows(name, model_a) + self.rows(name, model_b), model_a, model_b,
                                         metric='accuracy', q=self.q, confidence=self.confidence, expected_folds=self.folds,
                                         expected_seeds={model_a: self.seeds(name, model_a), model_b: self.seeds(name, model_b)})


def source_paths(runs_root):
    root = Path(runs_root)
    return {key: root/relative for key, relative in SOURCES.items()}


def check_combination(panel):
    """Disjoint panels; the same nested design, outer folds, fitting seeds and candidate record of every model in the runs
    that hold it for the benchmark datasets and in both batch runs."""
    runs = panel.runs
    overlap = sorted(set(panel.benchmark) & set(panel.further))
    if overlap:
        raise RunComparisonError(f'The benchmark and further panels share datasets: {overlap}')
    if sorted(panel.further) != sorted(runs['batch1'].protocol['datasets'] + runs['batch2'].protocol['datasets']):
        raise RunComparisonError('The further panel is not the union of the two batch panels')
    absent = [model for model in MAIN_MODELS if model not in runs['knn'].registry]
    if absent:
        raise RunComparisonError(f'The knn run holds no {", ".join(absent)}')
    for key in DESIGN_KEYS:
        values = {label: canonical_json(run.protocol.get(key)) for label, run in runs.items()}
        if None in (run.protocol.get(key) for run in runs.values()) or len(set(values.values())) != 1:
            raise RunComparisonError(f'Protocol {key} differs between the runs ({values})')
    if any([tuple(fold) for fold in run.schedule['expected_folds']] != panel.folds for run in runs.values()):
        raise RunComparisonError('The outer fold schedules differ between the runs')
    holders = {model: 'training' if model in CONTROL_MODELS else 'projected' if model == PROJECTED_MODEL else 'knn'
               for model in ALL_MODELS}
    for model, label in holders.items():
        for batch in BATCH_LABELS:
            if model not in runs[label].candidates or canonical_json(runs[label].candidates[model]) != canonical_json(runs[batch].candidates[model]):
                raise RunComparisonError(f'The {model} candidate record differs between the {label} run and the {batch} run')
            if runs[label].schedule['expected_seeds'][model] != runs[batch].schedule['expected_seeds'][model]:
                raise RunComparisonError(f'The {model} fitting seeds differ between the {label} run and the {batch} run')
    return {'benchmark_datasets': list(panel.benchmark), 'further_datasets': list(panel.further),
            'batch_of': dict(panel.batch_of), 'design': {key: runs['knn'].protocol[key] for key in DESIGN_KEYS},
            'model_runs': {BENCHMARK: holders, FURTHER: 'the batch run holding the dataset'},
            'candidate_config_ids': {model: list(runs[label].candidates[model]['config_ids']) for model, label in holders.items()}}


def load_panel(paths, *, frozen_protocols=None):
    """Load the five runs, re-check their registered pairings and the combination, then re-verify every saved record."""
    runs = {label: load_run(paths[label], label) for label in RUN_LABELS}
    for run in runs.values():
        check_design(run)
    references = {'training': check_training_reference(runs['training'].protocol, runs['knn']),
                  'projected': check_projected_references(runs['projected'].protocol, runs['knn'], runs['training'])}
    pairing = {'training': check_training_pairing(runs['training'], runs['knn']),
               'projected': check_projected_pairing(runs['projected'], runs['knn'], runs['training']),
               'batches': check_batches({label: runs[label] for label in BATCH_LABELS}, frozen_protocols)}
    panel = Panel(runs)
    combination = check_combination(panel)
    verification = {'validators': list(VALIDATORS)}
    for label, run in runs.items():
        rows, _ = verify_run(run)
        verification[label] = {'jobs_verified': len(run.jobs), 'model_rows_verified': sum(map(len, rows.values()))}
    return panel, {'runs': {label: run.provenance() for label, run in runs.items()}, 'references': references,
                   'pairing': pairing, 'combination': combination, 'verification': verification}


# ----------------------------------------------------------------------------- published analyses and agreement

def read_published(paths, keys):
    records, files = {}, {}
    for key in keys:
        directory, relative = PUBLISHED[key]
        path = Path(paths[directory])/relative
        try:
            data = path.read_bytes()
            text = data.decode('utf-8')
            records[key] = json.loads(text) if relative.endswith('.json') else list(csv.DictReader(io.StringIO(text)))
        except (OSError, ValueError) as exc:
            raise RunComparisonError(f'The published analysis {path} is missing or unreadable ({_reason(exc)})') from exc
        files[key] = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
    return records, files


def check_published(records, files, panel):
    """Every published record names the verified runs, and every CSV it seals is the file on disk."""
    for key, route, label in RECORDED_RUNS:
        if key not in records:
            continue
        block = records[key]
        for step in route:
            block = block.get(step) if isinstance(block, dict) else None
        observed = panel.runs[label].provenance()
        wrong = [f'{field} recorded {(block or {}).get(field)!r}, verified {observed[field]!r}' for field in RUN_FIELDS
                 if not isinstance(block, dict) or block.get(field) != observed[field]]
        if wrong:
            raise RunComparisonError(f'{files[key]["path"]} was computed from another {label} run: ' + '; '.join(wrong))
    for record_key, csv_key in SEALS:
        if record_key in records and csv_key in records:
            name = Path(PUBLISHED[csv_key][1]).name
            declared = ((records[record_key].get('outputs') or {}).get(name) or {}).get('sha256')
            if declared != files[csv_key]['sha256']:
                raise RunComparisonError(f'{files[csv_key]["path"]} (sha256 {files[csv_key]["sha256"]}) is not the file '
                                         f'{files[record_key]["path"]} seals ({declared})')


def _parse(text):
    if text == '':
        return None
    if text in ('True', 'False'):
        return text == 'True'
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def same(published, computed):
    if isinstance(published, str) and not isinstance(computed, str):
        published = _parse(published)
    if published is None or computed is None:
        return published is None and computed is None
    if isinstance(published, bool) or isinstance(computed, bool):
        return type(published) is type(computed) and published == computed
    if isinstance(published, (int, float)) and isinstance(computed, (int, float)):
        return bool(np.isclose(float(published), float(computed), rtol=0, atol=TOLERANCE))
    if isinstance(published, (list, tuple)) and isinstance(computed, (list, tuple)):
        return len(published) == len(computed) and all(same(a, b) for a, b in zip(published, computed))
    if isinstance(published, dict) and isinstance(computed, dict):
        return set(published) == set(computed) and all(same(published[key], computed[key]) for key in published)
    return published == computed


def agree(published, computed, fields, where):
    wrong = [f'{field} published {None if published is None else published.get(field)!r}, recomputed {computed.get(field)!r}'
             for field in fields if published is None or not same(published.get(field), computed.get(field))]
    if wrong:
        raise RunComparisonError(f'{where} differs from the recomputation from the verified runs: ' + '; '.join(wrong))


def index_rows(rows, keys, where, expected):
    indexed = {}
    for row in rows:
        key = tuple(_parse(row[k]) if isinstance(row[k], str) and k in ('family_index',) else row[k] for k in keys)
        if key in indexed:
            raise RunComparisonError(f'{where} repeats the row {key}')
        indexed[key] = row
    if set(indexed) != set(expected):
        raise RunComparisonError(f'{where} does not hold exactly the expected rows ({len(indexed)} held, {len(expected)} expected)')
    return indexed


# ----------------------------------------------------------------------------- benchmark: tables

MAIN_COLUMNS = ('dataset', 'group', 'model_id', 'source_run', 'mean_error', 'outer_fold_sd', 'mean_within_fold_seed_sd',
                'n_folds', 'seeds_per_fold')
INTERVAL_KEYS = ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_unadjusted', 'n_folds', 'df')
COMPARATOR_COLUMNS = ('status', 'dataset', 'group', 'model_a', 'model_b', *INTERVAL_KEYS, 'mean_error_a', 'mean_error_b',
                      'best_classical')
COMPETITIVENESS_COLUMNS = ('status', 'dataset', 'group', 'arrowflow_error', 'best_classical', 'best_classical_error',
                           'gap_points', 'gap_points_2dp', 'gap_points_from_1dp_errors', 'within_three_points',
                           'within_three_points_as_printed', 'best_on', 'majority_class_error', 'majority_distance_points',
                           'ceiling', 'near_majority', 'no_learning', 'degenerate')
RANK_MATRIX_COLUMNS = ('status', 'dataset', 'group', 'model_id', 'mean_error', 'rank', 'mean_error_percent_as_displayed',
                       'rank_as_displayed')
MEAN_RANK_COLUMNS = ('status', 'model_id', 'mean_rank', 'rank_sum', 'n_datasets', 'mean_rank_as_displayed')
METRIC_FIELDS = ('error', 'balanced_accuracy', 'macro_f1', 'accuracy')
COMPLETE_COLUMNS = ('dataset', 'group', 'model_id', 'source_run',
                    *(f'{metric}{suffix}' for metric in METRIC_FIELDS for suffix in ('', '_outer_fold_sd', '_within_fold_seed_sd')),
                    'n_folds', 'seeds_per_fold')
WIDTH_COLUMNS = ('dataset', 'group', 'outer_repeat', 'outer_fold', 'config_id', 'widths', 'hidden_layers')
DUPLICATE_COLUMNS = ('status', 'dataset', 'group', *COUNT_FIELDS, 'test_rows', 'test_rows_with_training_duplicate',
                     'share_with_training_duplicate', 'fold_share_min', 'fold_share_max', 'source_audit')
BENCHMARK_OUTPUTS = ('main_table.csv', 'comparator_intervals.csv', 'competitiveness.csv', 'rank_matrix.csv', 'mean_ranks.csv',
                     'complete_metrics.csv', 'selected_widths.csv', 'duplicates.csv', 'benchmark.json')


def main_table_rows(panel, published):
    table = published['main_table']
    if list(table.get('datasets') or []) != panel.benchmark:
        raise RunComparisonError('main_table.json does not cover the benchmark panel of the knn run')
    rows = []
    for name in panel.datasets:
        for model in MAIN_MODELS:
            entry = panel.summary(name, model, 'error')
            row = {'dataset': name, 'group': panel.group[name], 'model_id': model, 'source_run': panel.run_for(name, model).label,
                   'mean_error': entry['mean'], **{key: entry[key] for key in MAIN_COLUMNS[5:]}}
            rows.append(row)
            if panel.group[name] == BENCHMARK:
                recorded = next((r for r in table['rows'].get(name, []) if r.get('model_id') == model), None)
                agree(recorded, row, MAIN_COLUMNS[4:], f'main_table.json {name} {model}')
    return rows


def comparator_rows(panel, published):
    """ArrowFlow minus each tuned classical model per dataset, with the published intervals (the majority class included
    there) checked against the recomputation."""
    sources = {BENCHMARK: ('comparator_contrasts.csv', published['comparator_csv']),
               FURTHER: ('newdata_comparators.csv', published['newdata_comparators_csv'])}
    indexed = {group: index_rows(rows, ('dataset', 'model_b'), where,
                                 [(name, model) for name in panel.datasets if panel.group[name] == group
                                  for model in (*CLASSICAL, MAJORITY)])
               for group, (where, rows) in sources.items()}
    rows = []
    for name in panel.datasets:
        group = panel.group[name]
        errors = {model: panel.summary(name, model, 'error')['mean'] for model in MAIN_MODELS}
        lowest = min(errors[model] for model in CLASSICAL)
        for model in (*CLASSICAL, MAJORITY):
            interval = panel.interval(name, TRAINED, model)
            row = {'status': DESCRIPTIVE, 'dataset': name, 'group': group, 'model_a': TRAINED, 'model_b': model,
                   **{key: interval[key] for key in INTERVAL_KEYS if key != 'p_unadjusted'},
                   'p_unadjusted': interval['p_approximate'], 'mean_error_a': errors[TRAINED], 'mean_error_b': errors[model],
                   'best_classical': model != MAJORITY and errors[model] == lowest}
            agree(indexed[group][(name, model)], row, (*INTERVAL_KEYS, 'mean_error_a', 'mean_error_b'),
                  f'{sources[group][0]} {name} {model}')
            if model != MAJORITY:
                rows.append(row)
    return rows


def points(value):
    return round(100 * float(value), POINT_DECIMALS)


def competitiveness(errors):
    """The fixed rules of RULES on one dataset's mean errors {model: error as a fraction} of MAIN_MODELS."""
    arrowflow, majority = points(errors[TRAINED]), points(errors[MAJORITY])
    classical = {model: points(errors[model]) for model in CLASSICAL}
    best_error = min(classical.values())
    gap = round(arrowflow - best_error, POINT_DECIMALS)
    printed_gap = float(f'{gap:.2f}')
    printed_errors_gap = round(float(f'{arrowflow:.1f}') - float(f'{best_error:.1f}'), 1)
    within = gap <= THRESHOLDS['within_points']
    distance = round(abs(arrowflow - majority), POINT_DECIMALS)
    ceiling = best_error < THRESHOLDS['ceiling_points']
    near_majority = distance <= THRESHOLDS['near_majority_points'] and gap >= THRESHOLDS['near_majority_advantage_points']
    no_learning = majority <= min(arrowflow, *classical.values())
    return {'arrowflow_error': errors[TRAINED], 'best_classical': [model for model in CLASSICAL if classical[model] == best_error],
            'best_classical_error': errors[min(CLASSICAL, key=lambda model: (classical[model], CLASSICAL.index(model)))],
            'gap_points': gap, 'gap_points_2dp': printed_gap, 'gap_points_from_1dp_errors': printed_errors_gap,
            'within_three_points': within,
            'within_three_points_as_printed': within and printed_gap <= THRESHOLDS['within_points']
                                              and printed_errors_gap <= THRESHOLDS['within_points'],
            'printed_precision_agrees': (printed_gap <= THRESHOLDS['within_points']) == within
                                        == (printed_errors_gap <= THRESHOLDS['within_points']),
            'best_on': arrowflow < best_error, 'majority_class_error': errors[MAJORITY], 'majority_distance_points': distance,
            'ceiling': ceiling, 'near_majority': near_majority, 'no_learning': no_learning,
            'degenerate': ceiling or near_majority or no_learning}


def competitiveness_rows(panel):
    rows = []
    for name in panel.datasets:
        result = competitiveness({model: panel.summary(name, model, 'error')['mean'] for model in MAIN_MODELS})
        rows.append({'status': 'descriptive; fixed rules (benchmark.json rules)', 'dataset': name, 'group': panel.group[name],
                     **result, 'best_classical': '+'.join(result['best_classical'])})
    return rows


def competitiveness_counts(rows):
    def names(condition, group=None):
        return [row['dataset'] for row in rows if condition(row) and group in (None, row['group'])]
    within = lambda row: row['within_three_points']
    counts = {'datasets': len(rows), 'best_on': names(lambda row: row['best_on']),
              'trails_best_classical': names(lambda row: row['gap_points'] > 0),
              'ties_best_classical': names(lambda row: row['gap_points'] == 0),
              'within_three_points': {'all': names(within), BENCHMARK: names(within, BENCHMARK), FURTHER: names(within, FURTHER)},
              'within_three_points_and_degenerate': names(lambda row: within(row) and row['degenerate']),
              'within_three_points_and_not_degenerate': names(lambda row: within(row) and not row['degenerate']),
              'degenerate': {flag: names(lambda row, flag=flag: row[flag]) for flag in ('ceiling', 'near_majority', 'no_learning')},
              'printed_precision_disagreements': names(lambda row: not row['printed_precision_agrees'])}
    counts['summary'] = (f"within three points on {len(counts['within_three_points']['all'])} of {len(rows)} datasets "
                         f"({len(counts['within_three_points'][BENCHMARK])} benchmark, {len(counts['within_three_points'][FURTHER])} "
                         f"further); {len(counts['within_three_points_and_degenerate'])} of them carry a degenerate flag "
                         f"({', '.join(counts['within_three_points_and_degenerate']) or 'none'}); best on "
                         f"{len(counts['best_on'])} of {len(rows)}")
    return counts


def rank_analysis(panel, datasets=None):
    """Friedman ranks (referee_analyses.friedman_nemenyi) over the mean outer errors of ArrowFlow and the five tuned
    classical models, with the same computation at the printed precision of the main table."""
    datasets = list(panel.datasets if datasets is None else datasets)
    matrix = [[panel.summary(name, model, 'error')['mean'] for model in RANK_MODELS] for name in datasets]
    return rank_record(matrix, datasets)


def rank_record(matrix, datasets, models=RANK_MODELS):
    shown = [[display_value(value) for value in row] for row in matrix]
    test, displayed = friedman_nemenyi(matrix), friedman_nemenyi(shown)
    mean_ranks, cd = dict(zip(models, test['mean_ranks'])), test['nemenyi_cd']
    pairs = [{'model_a': a, 'model_b': b, 'mean_rank_difference': abs(mean_ranks[a] - mean_ranks[b])}
             for index, a in enumerate(models) for b in models[index + 1:]]
    return {'status': 'descriptive', 'metric': 'error', 'models': list(models), 'datasets': datasets,
            'matrix': {name: dict(zip(models, row)) for name, row in zip(datasets, matrix)},
            'ranks': {name: dict(zip(models, row)) for name, row in zip(datasets, test['ranks'])},
            'ranks_as_displayed': {name: dict(zip(models, row)) for name, row in zip(datasets, displayed['ranks'])},
            'mean_ranks': mean_ranks, 'rank_sums': dict(zip(models, test['rank_sums'])),
            'mean_ranks_as_displayed': dict(zip(models, displayed['mean_ranks'])),
            'friedman': {key: test[key] for key in ('n_datasets', 'n_models', 'chi2', 'df', 'p', 'tie_sum', 'chi2_tie_corrected',
                                                    'p_tie_corrected', 'iman_davenport_f', 'iman_davenport_df', 'iman_davenport_p')},
            'nemenyi': {'alpha': test['alpha'], 'q_alpha': test['q_alpha'], 'critical_difference': cd,
                        'pairs_exceeding_critical_difference': [pair for pair in pairs if pair['mean_rank_difference'] > cd],
                        'largest_mean_rank_difference': max(pairs, key=lambda pair: pair['mean_rank_difference'])},
            'ties': {'datasets_with_tied_errors': [name for name, row in zip(datasets, matrix) if len(set(row)) < len(models)],
                     'datasets_with_tied_errors_as_displayed': [name for name, row in zip(datasets, shown) if len(set(row)) < len(models)],
                     'ranks_as_displayed_equal_ranks': displayed['ranks'] == test['ranks'],
                     'friedman_as_displayed': {key: displayed[key] for key in ('chi2', 'p', 'chi2_tie_corrected', 'p_tie_corrected',
                                                                               'iman_davenport_f', 'iman_davenport_p')}}}


def check_published_ranks(panel, published):
    """The seven-dataset Friedman record of the referee analyses equals the same computation on the verified knn run."""
    recorded = published['ranks_json']
    if list(recorded.get('models') or []) != list(RANK_MODELS) or list(recorded.get('datasets') or []) != panel.benchmark:
        raise RunComparisonError('friedman_nemenyi.json does not hold the benchmark panel and the rank models')
    ours = rank_analysis(panel, panel.benchmark)
    agree(recorded, ours, ('matrix', 'ranks', 'mean_ranks'), 'friedman_nemenyi.json')
    agree(recorded['friedman'], ours['friedman'], tuple(ours['friedman']), 'friedman_nemenyi.json friedman')
    agree(recorded['nemenyi'], ours['nemenyi'], ('q_alpha', 'critical_difference'), 'friedman_nemenyi.json nemenyi')


def rank_rows(record, panel):
    matrix = [{'status': record['status'], 'dataset': name, 'group': panel.group[name], 'model_id': model,
               'mean_error': record['matrix'][name][model], 'rank': record['ranks'][name][model],
               'mean_error_percent_as_displayed': display_value(record['matrix'][name][model]),
               'rank_as_displayed': record['ranks_as_displayed'][name][model]}
              for name in record['datasets'] for model in record['models']]
    means = [{'status': record['status'], 'model_id': model, 'mean_rank': record['mean_ranks'][model],
              'rank_sum': record['rank_sums'][model], 'n_datasets': len(record['datasets']),
              'mean_rank_as_displayed': record['mean_ranks_as_displayed'][model]} for model in record['models']]
    return matrix, means


def complete_metric_rows(panel):
    rows = []
    for name in panel.datasets:
        for model in ALL_MODELS:
            row = {'dataset': name, 'group': panel.group[name], 'model_id': model, 'source_run': panel.run_for(name, model).label}
            for metric in METRIC_FIELDS:
                entry = panel.summary(name, model, metric)
                row.update({metric: entry['mean'], f'{metric}_outer_fold_sd': entry['outer_fold_sd'],
                            f'{metric}_within_fold_seed_sd': entry['mean_within_fold_seed_sd']})
                row.update(n_folds=entry['n_folds'], seeds_per_fold=entry['seeds_per_fold'])
            rows.append(row)
    return rows


def selected_width_rows(panel, depths):
    rows, by_dataset = [], {}
    for name in panel.datasets:
        model_rows = panel.rows(name, TRAINED)
        widths = selected_widths(model_rows, TRAINED)
        configs = {}
        for row in model_rows:
            if configs.setdefault((row['outer_repeat'], row['outer_fold']), row['config_id']) != row['config_id']:
                raise RunComparisonError(f'{name}: {TRAINED} rows of one outer fold disagree on the selected configuration')
        outside = sorted({tuple(widths[fold]) for fold in panel.folds} - {tuple(depth) for depth in depths})
        if outside:
            raise RunComparisonError(f'{name}: {TRAINED} selected widths outside the declared depths: {outside}')
        for fold in panel.folds:
            rows.append({'dataset': name, 'group': panel.group[name], 'outer_repeat': fold[0], 'outer_fold': fold[1],
                         'config_id': configs[fold], 'widths': json.dumps(widths[fold]), 'hidden_layers': len(widths[fold])})
        by_dataset[name] = {json.dumps(depth): sum(list(widths[fold]) == depth for fold in panel.folds) for depth in depths}
    totals = {scope: {json.dumps(depth): sum(by_dataset[name][json.dumps(depth)] for name in names) for depth in depths}
              for scope, names in (('all', panel.datasets), (BENCHMARK, panel.benchmark), (FURTHER, panel.further))}
    folds = {scope: len(names) * len(panel.folds) for scope, names in (('all', panel.datasets), (BENCHMARK, panel.benchmark),
                                                                        (FURTHER, panel.further))}
    return rows, {'by_dataset': by_dataset, 'totals': totals, 'outer_folds': folds}


def check_published_widths(panel, published):
    recorded = published['training_depth_json']['by_dataset']
    for name in panel.benchmark:
        widths = selected_widths(panel.rows(name, TRAINED), TRAINED)
        for entry in recorded.get(name, []):
            ours = [list(fold) for fold in panel.folds if list(widths[fold]) == entry['widths']]
            if ours != entry['folds']:
                raise RunComparisonError(f'training_depth_split.json {name} {entry["widths"]}: the folds differ from the '
                                         'widths ArrowFlow selected in the verified knn run')


def duplicate_rows(panel, published):
    sources = {BENCHMARK: ('duplicate_groups.csv', published['duplicate_csv']),
               FURTHER: ('newdata_duplicates.csv', published['newdata_duplicates_csv'])}
    indexed = {group: index_rows(rows, ('dataset',), where, [(name,) for name in panel.datasets if panel.group[name] == group])
               for group, (where, rows) in sources.items()}
    rows = []
    for name in panel.datasets:
        run = panel.run_for(name, TRAINED)
        try:
            X, y, _, splits = load_prepared(run.path, name)
            vector, counts = duplicate_groups(X, y)
        except (OSError, ValueError, KeyError) as exc:
            raise RunComparisonError(f'{name}: the prepared features of the {run.label} run cannot be audited ({_reason(exc)})') from exc
        tested, flagged, shares = 0, 0, []
        for repeat, fold in panel.folds:
            split = splits[repeat * run.protocol['outer_folds'] + fold]
            flags = training_duplicate_flags(vector, split['train'], split['test'])
            tested, flagged = tested + len(split['test']), flagged + int(sum(flags))
            shares.append(sum(flags) / len(split['test']))
        group = panel.group[name]
        row = {'status': 'descriptive', 'dataset': name, 'group': group, **{key: counts[key] for key in COUNT_FIELDS},
               'test_rows': tested, 'test_rows_with_training_duplicate': flagged, 'share_with_training_duplicate': flagged / tested,
               'fold_share_min': float(min(shares)), 'fold_share_max': float(max(shares)), 'source_audit': sources[group][0]}
        agree(indexed[group][(name,)], row, (*COUNT_FIELDS, *DUPLICATE_COLUMNS[len(COUNT_FIELDS) + 3:-1]),
              f'{sources[group][0]} {name}')
        rows.append(row)
    return rows


def benchmark(paths, output, *, frozen_protocols=None):
    check_output(output, BENCHMARK_OUTPUTS)            # an unusable output location is refused before any run is read
    panel, provenance = load_panel(paths, frozen_protocols=frozen_protocols)
    published, files = read_published(paths, BENCHMARK_PUBLISHED)
    check_published(published, files, panel)
    depths = validate_depths(((panel.runs['training'].protocol.get('training_controls') or {}).get('depth_split') or {}).get('depths'))
    try:
        main = main_table_rows(panel, published)
        intervals = comparator_rows(panel, published)
        competitive = competitiveness_rows(panel)
        counts = competitiveness_counts(competitive)
        check_published_ranks(panel, published)
        ranks = rank_analysis(panel)
        matrix_rows, mean_rows = rank_rows(ranks, panel)
        metrics = complete_metric_rows(panel)
        check_published_widths(panel, published)
        widths, width_counts = selected_width_rows(panel, depths)
        duplicates = duplicate_rows(panel, published)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified runs do not support the combined benchmark: {_reason(exc)}') from exc
    record = {
        'purpose': 'task24_combined_benchmark_of_the_benchmark_and_further_datasets',
        'status': 'no model fitted; every number recomputed from the verified runs and checked against the published '
                  'analyses; the paired intervals, the Friedman test and every count are descriptive',
        'datasets': {'all': panel.datasets, BENCHMARK: panel.benchmark, FURTHER: panel.further},
        'models': {'arrowflow': TRAINED, 'classical': list(CLASSICAL), 'majority_class': MAJORITY, 'complete_metrics': list(ALL_MODELS)},
        'definitions': {
            'mean_error': 'mean over the outer folds of the fitting-seed-averaged error (the summary.json mean of the run holding '
                          'the model on the dataset)',
            'outer_fold_sd': 'SD (ddof 1) of the outer-fold means', 'within_fold_seed_sd': 'mean over the outer folds of the '
                             'within-fold SD across fitting seeds; null for a deterministic model',
            'comparator_intervals': f'{TRAINED} minus the tuned classical model, accuracy: fitting seeds averaged within each '
                                    'outer fold, corrected resampled t over the outer folds (evaluation.paired_corrected_interval, '
                                    f'q = {panel.q}, {panel.confidence:.0%}, df = outer folds - 1), unadjusted two-sided p; '
                                    'positive favours ArrowFlow; ' + DESCRIPTIVE,
            'friedman': 'referee_analyses.friedman_nemenyi on the dataset x model matrix of mean outer error of ArrowFlow and the '
                        'five tuned classical models (the majority class excluded): average ranks, Friedman chi-square with its '
                        'tie-corrected form, Iman-Davenport F and the Nemenyi critical difference at alpha 0.05; descriptive',
            'selected_widths': 'the hidden widths of the configuration ArrowFlow selected on the inner folds of each outer fold '
                               '(the config of its verified outer model rows)',
            'duplicates': 'referee_analyses.duplicate_groups on the prepared features and training_duplicate_flags per outer '
                          'fold: test rows with an exact feature copy in the training partition, pooled over the outer folds',
            'groups': f'{BENCHMARK}: the datasets of the bridge_knn run; {FURTHER}: the datasets of the newdata batch runs'},
        'rules': RULES, 'main_table': main, 'comparator_intervals': intervals, 'competitiveness': competitive, 'counts': counts,
        'ranks': ranks, 'selected_widths': width_counts, 'duplicates': duplicates,
        'cross_checks': {
            'main_table.json': 'benchmark rows (mean error, outer-fold SD, seed SD, folds, seeds) equal the verified knn run',
            'comparator_contrasts.csv, newdata_comparators.csv': 'every interval (all six comparators) and both mean errors '
                                                                 'equal the recomputation',
            'friedman_nemenyi.json': 'the benchmark-panel matrix, ranks, Friedman and Nemenyi values equal the recomputation',
            'training_depth_split.json': 'the benchmark folds of each depth equal the widths ArrowFlow selected',
            'duplicate_groups.csv, newdata_duplicates.csv': 'every count and share equals the audit recomputed from the prepared '
                                                            'features',
            'recorded_runs': 'each published record names the verified runs (protocol, sha256, code revision, summary, registry, '
                             'planned jobs) and seals the CSV on disk'},
        'provenance': {**provenance, 'published_analyses': files, **code_record(ANALYSIS_SOURCES)},
    }
    tables = {'main_table.csv': (main, MAIN_COLUMNS), 'comparator_intervals.csv': (intervals, COMPARATOR_COLUMNS),
              'competitiveness.csv': (competitive, COMPETITIVENESS_COLUMNS), 'rank_matrix.csv': (matrix_rows, RANK_MATRIX_COLUMNS),
              'mean_ranks.csv': (mean_rows, MEAN_RANK_COLUMNS), 'complete_metrics.csv': (metrics, COMPLETE_COLUMNS),
              'selected_widths.csv': (widths, WIDTH_COLUMNS), 'duplicates.csv': (duplicates, DUPLICATE_COLUMNS)}
    finish(output, tables, 'benchmark.json', record)
    return record


# ----------------------------------------------------------------------------- training: tables

CONTROL_KEYS = ('difference', 'standard_error', 'ci_low', 'ci_high', 'p_unadjusted', 'registered_holm_p', 'registered_family',
                'registered_family_size')
TRAINING_COLUMNS = ('dataset', 'group', *(f'{CONTROL_PREFIX[control]}_{key}' for control in CONTROL_MODELS for key in CONTROL_KEYS),
                    'n_folds', 'df')
HOLM_COLUMNS = ('status', 'dataset', 'group', 'control', 'contrast', 'mean_difference', 'ci_low', 'ci_high', 'p_unadjusted',
                'registered_family', 'registered_holm_p', 'holm34_p', 'holm34_below_alpha')
LADDER_COLUMNS = ('dataset', 'group', *(rung for rung, _ in LADDER))
LADDER_DETAIL_COLUMNS = ('dataset', 'group', 'rung', 'model_id', 'source_run', 'mean_error', 'outer_fold_sd',
                         'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')
DEPTH_COLUMNS = ('status', 'dataset', 'group', 'widths', 'n_folds', 'mean_difference', 'sd', 'min', 'max', 'ci_low', 'ci_high',
                 'untrained_selected_the_same_widths')
DEPTH_POOLED_COLUMNS = ('status', 'scope', 'widths', 'n_dataset_folds', 'mean_difference', 'sd', 'datasets')
TRAINING_OUTPUTS = ('training_controls.csv', 'holm34_sensitivity.csv', 'ladder.csv', 'ladder_detail.csv', 'depth_split.csv',
                    'depth_pooled.csv', 'training.json')
FAMILY_FIELDS = ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_approximate', 'holm_p_approximate', 'n_folds', 'df')


def registered_families(panel):
    training, first = panel.runs['training'].protocol, panel.runs['batch1'].protocol
    analysis = first['analysis']
    batch_ids = [panel.runs[label].protocol['protocol_id'] for label in BATCH_LABELS]
    return {
        'knn_training_primary': {'protocols': [training['protocol_id']], 'block': 'primary_contrasts',
                                 'contrasts': list(training['primary_contrasts']), 'size': training['primary_family_size'],
                                 'multiplicity': training['multiplicity'], 'datasets': panel.benchmark,
                                 'members': 'per benchmark dataset in panel order, the untrained control before the input control'},
        'newdata_primary': {'protocols': batch_ids, 'block': 'analysis.primary_family', 'contrasts': [analysis['primary_family']['contrast']],
                            'size': analysis['primary_family']['size'], 'multiplicity': analysis['primary_family']['multiplicity'],
                            'datasets': panel.further, 'members': 'one per further dataset in panel order'},
        'newdata_secondary': {'protocols': batch_ids, 'block': 'analysis.secondary_family',
                              'contrasts': [analysis['secondary_family']['contrast']], 'size': analysis['secondary_family']['size'],
                              'multiplicity': analysis['secondary_family']['multiplicity'], 'datasets': panel.further,
                              'members': 'one per further dataset in panel order'}}


def training_contrast_rows(panel, published):
    """ArrowFlow minus each control on every dataset, with the registered Holm p of its family recomputed and checked
    against the published family tables."""
    families = registered_families(panel)
    rows = []
    for name in panel.datasets:
        for control in CONTROL_MODELS:
            interval = panel.interval(name, TRAINED, control)
            family = 'knn_training_primary' if panel.group[name] == BENCHMARK else f'newdata_{NEWDATA_FAMILY[control]}'
            rows.append({'dataset': name, 'group': panel.group[name], 'control': control,
                         'contrast': f'{TRAINED}_vs_{control}', 'model_a': TRAINED, 'model_b': control, **interval,
                         'p_unadjusted': interval['p_approximate'], 'registered_family': family,
                         'registered_family_size': families[family]['size']})
    for family, definition in families.items():
        members = [row for row in rows if row['registered_family'] == family]
        if len(members) != definition['size']:
            raise RunComparisonError(f'The registered family {family} declares {definition["size"]} members; the panel gives '
                                     f'{len(members)}')
        for index, (row, adjusted) in enumerate(zip(members, holm_adjust([row['p_unadjusted'] for row in members])), start=1):
            row.update(registered_holm_p=adjusted, registered_family_index=index, holm_p_approximate=adjusted)
    benchmark_rows = index_rows(published['training_csv'], ('family_index',), 'training_contrasts.csv',
                                [(index,) for index in range(1, families['knn_training_primary']['size'] + 1)])
    further_rows = index_rows(published['newdata_families_csv'], ('family', 'dataset'), 'newdata_families.csv',
                              [(label, name) for label in ('primary', 'secondary') for name in panel.further])
    for row in rows:
        if row['registered_family'] == 'knn_training_primary':
            recorded, where = benchmark_rows[(row['registered_family_index'],)], 'training_contrasts.csv'
        else:
            recorded, where = further_rows[(NEWDATA_FAMILY[row['control']], row['dataset'])], 'newdata_families.csv'
            agree(recorded, {**row, 'family_index': row['registered_family_index']}, ('family_index',), f'{where} {row["dataset"]}')
        agree(recorded, row, ('dataset', 'contrast', 'model_a', 'model_b', *FAMILY_FIELDS),
              f'{where} {row["dataset"]} {row["control"]}')
    return rows, families


def holm34(rows, alpha=ALPHA):
    """One Holm adjustment over every training contrast (post hoc; descriptive)."""
    adjusted = holm_adjust([row['p_unadjusted'] for row in rows])
    table = [{'status': POST_HOC, **{key: row[key] for key in ('dataset', 'group', 'control', 'contrast', 'mean_difference', 'ci_low',
                                                                 'ci_high', 'p_unadjusted', 'registered_family', 'registered_holm_p')},
              'holm34_p': value, 'holm34_below_alpha': value < alpha} for row, value in zip(rows, adjusted)]
    survivors = [{'dataset': row['dataset'], 'control': row['control'], 'holm34_p': row['holm34_p']} for row in table
                 if row['holm34_below_alpha']]
    return table, {'status': POST_HOC, 'members': len(rows), 'alpha': alpha,
                   'definition': 'evaluation.holm_adjust over the unadjusted p of every contrast of the three registered '
                                 'families together; not a registered analysis',
                   'below_alpha': survivors}


def training_table(rows, panel):
    table = []
    for name in panel.datasets:
        entry = {'dataset': name, 'group': panel.group[name]}
        for control in CONTROL_MODELS:
            row = next(r for r in rows if (r['dataset'], r['control']) == (name, control))
            prefix = CONTROL_PREFIX[control]
            entry.update({f'{prefix}_difference': row['mean_difference'], f'{prefix}_standard_error': row['standard_error'],
                          f'{prefix}_ci_low': row['ci_low'], f'{prefix}_ci_high': row['ci_high'],
                          f'{prefix}_p_unadjusted': row['p_unadjusted'], f'{prefix}_registered_holm_p': row['registered_holm_p'],
                          f'{prefix}_registered_family': row['registered_family'],
                          f'{prefix}_registered_family_size': row['registered_family_size']})
            entry.update(n_folds=row['n_folds'], df=row['df'])
        table.append(entry)
    return table


def ladder_rows(panel, published):
    projected = published['projected_ladder_json']
    further = index_rows(published['newdata_ladder_csv'], ('dataset', 'model_id'), 'newdata_ladder.csv',
                         [(name, model) for name in panel.further for _, model in LADDER])
    detail, wide = [], []
    for name in panel.datasets:
        entry = {'dataset': name, 'group': panel.group[name]}
        for rung, model in LADDER:
            summary = panel.summary(name, model, 'error')
            row = {'dataset': name, 'group': panel.group[name], 'rung': rung, 'model_id': model,
                   'source_run': panel.run_for(name, model).label, 'mean_error': summary['mean'],
                   **{key: summary[key] for key in LADDER_DETAIL_COLUMNS[6:]}}
            if panel.group[name] == BENCHMARK:
                recorded = next((r for r in (projected.get('rows') or {}).get(name, []) if r.get('model_id') == model), None)
                where = f'projected_ladder_error_table.json {name} {model}'
            else:
                recorded, where = further[(name, model)], f'newdata_ladder.csv {name} {model}'
            agree(recorded, row, LADDER_DETAIL_COLUMNS[5:], where)
            detail.append(row)
            entry[rung] = summary['mean']
        wide.append(entry)
    return wide, detail


def stratum_passthrough(panel, rows, published, files):
    """The prespecified moderator test and the descriptive Spearman correlation of the newdata analysis, recomputed from the
    verified further-dataset training effects and passed through with the provenance of their record."""
    record = published['newdata_analysis_json']
    declared = panel.runs['batch1'].protocol['analysis']
    if canonical_json(record.get('analysis_declaration')) != canonical_json(declared):
        raise RunComparisonError('newdata_analysis.json declares another analysis than the frozen batch protocols')
    effects = {row['dataset']: row['mean_difference'] for row in rows if row['group'] == FURTHER and row['control'] == UNTRAINED}
    test = permutation_test(effects, panel.further, declared['moderator_test']['strata']['H'])
    gaps = {entry['name']: entry['external_gap']['points'] for entry in panel.runs['batch1'].protocol['panel']}
    with_gap = list(declared['spearman']['datasets'])
    spearman = spearman_exact([gaps[name] for name in with_gap], [effects[name] for name in with_gap])
    moderator = {key: value for key, value in record['moderator_test'].items() if key != 'null_distribution'}
    agree(moderator, test, ('statistic', 'mean_h', 'mean_c', 'splits', 'at_least_observed', 'p_one_sided', 'larger_than_observed'),
          'newdata_analysis.json moderator_test')
    agree(record['spearman'], spearman, ('n', 'rho', 'p_one_sided', 'p_two_sided', 'permutations'), 'newdata_analysis.json spearman')
    return {'status': 'passed through from the frozen newdata analysis and recomputed from the verified batch runs: the moderator '
                      'test is prespecified, the Spearman correlation descriptive',
            'source': files['newdata_analysis_json'], 'moderator_test': moderator, 'spearman': record['spearman'],
            'notes': record.get('notes'), 'declaration': {'moderator_test': declared['moderator_test'], 'spearman': declared['spearman']},
            'record_provenance': record.get('provenance')}


def depth_rows(panel, published, depths):
    by_dataset = {}
    for name in panel.datasets:
        trained, untrained = panel.rows(name, TRAINED), panel.rows(name, UNTRAINED)
        entries = depth_split(trained + untrained, TRAINED, UNTRAINED, selected_widths(trained, TRAINED), depths=depths,
                              folds=panel.folds, seeds={TRAINED: panel.seeds(name, TRAINED), UNTRAINED: panel.seeds(name, UNTRAINED)},
                              q=panel.q, confidence=panel.confidence)
        own = selected_widths(untrained, UNTRAINED)
        for entry in entries:
            entry['untrained_selected_the_same_widths'] = sum(own[tuple(fold)] == entry['widths'] for fold in entry['folds'])
        by_dataset[name] = entries
        if panel.group[name] == BENCHMARK:
            recorded = published['training_depth_json']['by_dataset'].get(name)
            if recorded is None or len(recorded) != len(entries):
                raise RunComparisonError(f'training_depth_split.json holds no depth split of {name}')
            for published_entry, entry in zip(recorded, entries):
                agree(published_entry, entry, tuple(entry), f'training_depth_split.json {name} {entry["widths"]}')
    if canonical_json(published['training_depth_json'].get('depths')) != canonical_json(depths):
        raise RunComparisonError('training_depth_split.json declares other depths than the training protocol')
    table = []
    for name in panel.datasets:
        for entry in by_dataset[name]:
            interval = entry['interval'] or {}
            table.append({'status': 'descriptive; no p values', 'dataset': name, 'group': panel.group[name],
                          'widths': json.dumps(entry['widths']), **{key: entry[key] for key in ('n_folds', 'mean_difference', 'sd', 'min', 'max')},
                          'ci_low': interval.get('ci_low'), 'ci_high': interval.get('ci_high'),
                          'untrained_selected_the_same_widths': entry['untrained_selected_the_same_widths']})
    pooled = {scope: pooled_depth_split({name: by_dataset[name] for name in names}, depths)
              for scope, names in (('all', panel.datasets), (BENCHMARK, panel.benchmark), (FURTHER, panel.further))}
    pooled_rows = [{'status': 'descriptive; no interval', 'scope': scope, 'widths': json.dumps(entry['widths']),
                    'n_dataset_folds': entry['n_dataset_folds'], 'mean_difference': entry['mean_difference'], 'sd': entry['sd'],
                    'datasets': ' '.join(entry['datasets'])} for scope, entries in pooled.items() for entry in entries]
    return by_dataset, table, pooled, pooled_rows


def training(paths, output, *, frozen_protocols=None):
    check_output(output, TRAINING_OUTPUTS)             # an unusable output location is refused before any run is read
    panel, provenance = load_panel(paths, frozen_protocols=frozen_protocols)
    published, files = read_published(paths, TRAINING_PUBLISHED)
    check_published(published, files, panel)
    depths = validate_depths(((panel.runs['training'].protocol.get('training_controls') or {}).get('depth_split') or {}).get('depths'))
    try:
        rows, families = training_contrast_rows(panel, published)
        sensitivity, sensitivity_record = holm34(rows)
        table = training_table(rows, panel)
        ladder, ladder_detail = ladder_rows(panel, published)
        stratum = stratum_passthrough(panel, rows, published, files)
        by_dataset, depth_table, pooled, pooled_rows = depth_rows(panel, published, depths)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified runs do not support the combined training analysis: {_reason(exc)}') from exc
    counts = {'datasets': len(panel.datasets),
              'higher_mean_accuracy_than_untrained': [row['dataset'] for row in rows if row['control'] == UNTRAINED and row['mean_difference'] > 0],
              'registered_holm_below_alpha': [{'dataset': row['dataset'], 'control': row['control'], 'registered_family': row['registered_family'],
                                               'registered_holm_p': row['registered_holm_p']} for row in rows if row['registered_holm_p'] < ALPHA],
              'unadjusted_interval_excludes_zero': [{'dataset': row['dataset'], 'control': row['control']} for row in rows
                                                    if row['ci_low'] > 0 or row['ci_high'] < 0],
              'depth_folds': {scope: {json.dumps(entry['widths']): entry['n_dataset_folds'] for entry in entries} for scope, entries in pooled.items()}}
    record = {
        'purpose': 'task24_combined_training_controls_of_the_benchmark_and_further_datasets',
        'status': 'no model fitted; every contrast recomputed from the verified runs and checked against the published families; '
                  'each registered Holm p is the adjustment within its registered family; the 34-contrast Holm adjustment is a '
                  'post hoc sensitivity analysis; the ladder and the depth split are descriptive',
        'datasets': {'all': panel.datasets, BENCHMARK: panel.benchmark, FURTHER: panel.further},
        'definitions': {
            'difference': f'{TRAINED} minus the control, accuracy: fitting seeds averaged within each outer fold, corrected '
                          f'resampled t over the outer folds (q = {panel.q}, {panel.confidence:.0%}, df = outer folds - 1); '
                          'two-sided unadjusted p; positive favours the trained ArrowFlow',
            'registered_holm_p': 'evaluation.holm_adjust within the registered family of the contrast (families block), equal to '
                                 'the published family table',
            'controls': {UNTRAINED: 'untrained ArrowFlow (seeded initial filters, the same kNN readout selection)',
                         INPUT: 'tuned input footrule kNN on the encoded input ranking of the same encoders'},
            'ladder': 'mean outer error of each rung from the verified summary of the run holding it: ' +
                      ', '.join(f'{rung} = {model}' for rung, model in LADDER),
            'depth_split': f'{TRAINED} minus {UNTRAINED} accuracy by the hidden widths {TRAINED} selected in each outer fold '
                           '(knn_controls.depth_split): per group its folds, mean, SD, min, max and the corrected resampled t '
                           'interval over the group\'s folds when there are at least two; pooled across datasets per depth the '
                           'dataset-fold count, mean and SD (knn_controls.pooled_depth_split); descriptive, no p values'},
        'families': families, 'contrasts': rows, 'holm34': {**sensitivity_record, 'table': sensitivity}, 'ladder': ladder_detail,
        'stratum_test': stratum, 'depth_split': {'depths': depths, 'by_dataset': by_dataset, 'pooled': pooled}, 'counts': counts,
        'cross_checks': {
            'training_contrasts.csv, newdata_families.csv': 'every member (difference, interval, p, Holm p, index) equals the '
                                                             'recomputation within its registered family',
            'projected_ladder_error_table.json, newdata_ladder.csv': 'every rung (mean error, SDs, folds, seeds) equals the '
                                                                     'verified summaries',
            'newdata_analysis.json': 'the analysis declaration equals the frozen batch protocols; the moderator test and the '
                                     'Spearman correlation equal the recomputation from the verified training effects',
            'training_depth_split.json': 'the benchmark depth split equals the recomputation',
            'recorded_runs': 'each published record names the verified runs and seals the CSV on disk'},
        'provenance': {**provenance, 'published_analyses': files, **code_record(ANALYSIS_SOURCES)},
    }
    tables = {'training_controls.csv': (table, TRAINING_COLUMNS), 'holm34_sensitivity.csv': (sensitivity, HOLM_COLUMNS),
              'ladder.csv': (ladder, LADDER_COLUMNS), 'ladder_detail.csv': (ladder_detail, LADDER_DETAIL_COLUMNS),
              'depth_split.csv': (depth_table, DEPTH_COLUMNS), 'depth_pooled.csv': (pooled_rows, DEPTH_POOLED_COLUMNS)}
    finish(output, tables, 'training.json', record)
    return record


# ----------------------------------------------------------------------------- components (after both ablation runs)

ABLATION_SOURCES = {'knn_ablation': '2026-09-13-knn-ablation', 'newdata_ablation': '2026-09-14-newdata-ablation'}
ABLATION_SUMMARIES = {'knn_ablation': ('knn_ablation_summary.json', 'knn_ablation_summary.csv'),
                      'newdata_ablation': ('newdata_ablation_summary.json', 'newdata_ablation_summary.csv')}
ABLATION_PROTOCOL_IDS = {'knn_ablation': 'arrowflow-v3-knn-ablation-1', 'newdata_ablation': 'arrowflow-v3-newdata-ablation-1'}
ABLATION_GROUPS = {'knn_ablation': BENCHMARK, 'newdata_ablation': FURTHER}
ABLATION_RECORDS = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json')
COMPONENT_VARIANTS = ('views1', 'views3', 'no_checkpoint', 'no_augment', 'prototype_readout', 'untrained', 'input_knn')
COMPONENT_STATUS = 'descriptive; no p values; no multiplicity adjustment'
COMPONENT_COLUMNS = ('status', 'dataset', 'group', 'variant', 'arrowflow_minus_variant', 'standard_error', 'ci_low', 'ci_high',
                     'n_folds', 'df', 'identical_to_views7_folds', 'variant_accuracy', 'views7_accuracy', 'variant_error',
                     'views7_error', 'source_run')
COMPONENT_DEPTH_COLUMNS = ('status', 'dataset', 'group', 'widths', 'n_folds', 'untrained_minus_views7', 'sd', 'min', 'max',
                           'ci_low', 'ci_high', 'views7_minus_untrained', 'views7_minus_untrained_ci_low',
                           'views7_minus_untrained_ci_high')
COMPONENT_POOLED_COLUMNS = ('status', 'scope', 'widths', 'n_dataset_folds', 'untrained_minus_views7', 'views7_minus_untrained',
                            'sd', 'datasets')
COMPONENT_OUTPUTS = ('components.csv', 'components_depth.csv', 'components_depth_pooled.csv', 'components.json')
COMPONENT_ANALYSIS_SOURCES = ANALYSIS_SOURCES + ('run_knn_ablation.py', 'run_newdata_ablation.py')


def ablation_paths(runs_root):
    return {key: Path(runs_root)/name for key, name in ABLATION_SOURCES.items()}


def _negated(value):
    return None if value is None else 0. - value


def ablation_gate(paths):
    """Refuse unless both ablation runs are complete: their records, their summary JSON and CSV, and every planned job's
    result, log and prediction file. Only planned_jobs.json is parsed, so no score is read before both are complete."""
    problems = []
    for key in ABLATION_SOURCES:
        path = Path(paths[key])
        if not path.is_dir():
            problems.append(f'{key} run {path}: no such run directory')
            continue
        missing = [name for name in (*ABLATION_RECORDS, *ABLATION_SUMMARIES[key]) if not (path/name).is_file()]
        absent, first = 0, None
        if (path/'planned_jobs.json').is_file():
            try:
                for job in json.loads((path/'planned_jobs.json').read_text()):
                    for relative in (f"results/{job['stem']}.json", f"logs/{job['stem']}.jsonl", f"predictions/{job['stem']}.jsonl"):
                        if not (path/relative).is_file():
                            absent, first = absent + 1, first or relative
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f'{key} run {path}: unreadable planned_jobs.json ({_reason(exc)})')
                continue
        if missing or absent:
            parts = ([f'missing {", ".join(missing)}'] if missing else []) + (
                [f'{absent} planned job files missing (first: {first})'] if absent else [])
            problems.append(f'{key} run {path} is not complete: ' + '; '.join(parts))
    if problems:
        raise RunComparisonError('The component table is written only after both ablation runs are complete: ' + ' | '.join(problems))


def _summary_csv_text(flat, columns):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows([row[column] for column in columns] for row in flat)
    return buffer.getvalue()


def verified_ablation(key, path, runner, summary_columns, *, allow_smoke):
    """One ablation run re-verified by its runner's summary (every job, the views7 reproduction, every sealed selection
    re-derived from its reference run); the published summary JSON and CSV must equal the recomputation."""
    path = Path(path)
    summary_name, csv_name = ABLATION_SUMMARIES[key]
    try:
        protocol = json.loads((path/'protocol.json').read_text())
    except (OSError, ValueError) as exc:
        raise RunComparisonError(f'{key} run {path}: unreadable protocol ({_reason(exc)})') from exc
    if not allow_smoke and (protocol.get('protocol_id') != ABLATION_PROTOCOL_IDS[key] or protocol.get('frozen') is not True):
        raise RunComparisonError(f'{key} run {path} does not hold the frozen {ABLATION_PROTOCOL_IDS[key]} protocol')
    try:
        report, flat = runner.summary(path, allow_smoke=allow_smoke)
        published, published_csv = json.loads((path/summary_name).read_text()), (path/csv_name).read_text()
    except (OSError, KeyError, TypeError, ValueError, IndexError, StopIteration, EOFError, zipfile.BadZipFile) as exc:
        raise RunComparisonError(f'{key} run {path} fails its re-verification: {_reason(exc)}') from exc
    if canonical_json(published) != canonical_json(report):
        raise RunComparisonError(f'{path/summary_name} differs from the recomputation from the verified {key} run')
    if published_csv != _summary_csv_text(flat, summary_columns):
        raise RunComparisonError(f'{path/csv_name} differs from the recomputation from the verified {key} run')
    files = {name: {'path': str(path/name), 'sha256': sha256_file(path/name)}
             for name in ('protocol.json', 'manifest.json', summary_name, csv_name)}
    return protocol, report, files


def component_rows(reports):
    rows, datasets = [], []
    for key, report in reports.items():
        for name, entry in report['summaries'].items():
            datasets.append(name)
            identical = Counter(variant for configuration in entry['resolved_configurations']
                                for variant, source in configuration['fit_sources'].items() if source == 'identical_to_views7')
            views7 = entry['variants']['views7']['metrics']
            for variant in COMPONENT_VARIANTS:
                change, metrics = entry['variants'][variant]['change_from_views7']['accuracy'], entry['variants'][variant]['metrics']
                rows.append({'status': COMPONENT_STATUS, 'dataset': name, 'group': ABLATION_GROUPS[key], 'variant': variant,
                             'arrowflow_minus_variant': _negated(change['mean_difference']), 'standard_error': change['standard_error'],
                             'ci_low': _negated(change['ci_high']), 'ci_high': _negated(change['ci_low']), 'n_folds': change['n_folds'],
                             'df': change['df'], 'identical_to_views7_folds': identical[variant],
                             'variant_accuracy': metrics['accuracy']['mean'], 'views7_accuracy': views7['accuracy']['mean'],
                             'variant_error': metrics['error']['mean'], 'views7_error': views7['error']['mean'], 'source_run': key})
    if len(set(datasets)) != len(datasets):
        raise RunComparisonError('The two ablation runs share datasets')
    return rows, datasets


def component_depth(reports):
    declared = [report['depth_split']['depths'] for report in reports.values()]
    if len({canonical_json(depths) for depths in declared}) != 1:
        raise RunComparisonError('The two ablation runs declare different depths')
    depths = validate_depths(declared[0])
    by_dataset, groups, rows = {}, {}, []
    for key, report in reports.items():
        for name in report['summaries']:
            by_dataset[name], groups[name] = report['depth_split']['by_dataset'][name], ABLATION_GROUPS[key]
            for entry in by_dataset[name]:
                interval = entry['interval'] or {}
                rows.append({'status': 'descriptive; no p values', 'dataset': name, 'group': groups[name],
                             'widths': json.dumps(entry['widths']), 'n_folds': entry['n_folds'],
                             'untrained_minus_views7': entry['mean_difference'], 'sd': entry['sd'], 'min': entry['min'],
                             'max': entry['max'], 'ci_low': interval.get('ci_low'), 'ci_high': interval.get('ci_high'),
                             'views7_minus_untrained': _negated(entry['mean_difference']),
                             'views7_minus_untrained_ci_low': _negated(interval.get('ci_high')),
                             'views7_minus_untrained_ci_high': _negated(interval.get('ci_low'))})
    scopes = {'all': list(by_dataset), BENCHMARK: [name for name in by_dataset if groups[name] == BENCHMARK],
              FURTHER: [name for name in by_dataset if groups[name] == FURTHER]}
    pooled = {scope: pooled_depth_split({name: by_dataset[name] for name in names}, depths) for scope, names in scopes.items()}
    pooled_rows = [{'status': 'descriptive; no interval', 'scope': scope, 'widths': json.dumps(entry['widths']),
                    'n_dataset_folds': entry['n_dataset_folds'], 'untrained_minus_views7': entry['mean_difference'],
                    'views7_minus_untrained': _negated(entry['mean_difference']), 'sd': entry['sd'],
                    'datasets': ' '.join(entry['datasets'])} for scope, entries in pooled.items() for entry in entries]
    return depths, by_dataset, rows, pooled, pooled_rows


def components(paths, output, *, allow_smoke=False):
    check_output(output, COMPONENT_OUTPUTS)            # an unusable output location is refused before any run is read
    ablation_gate(paths)
    from . import run_knn_ablation, run_newdata_ablation     # imported on use: the runners seal their own environments
    if tuple(run_knn_ablation.VARIANTS) != ('views7', *COMPONENT_VARIANTS) or tuple(run_newdata_ablation.VARIANTS) != tuple(run_knn_ablation.VARIANTS):
        raise RunComparisonError('The ablation runners declare other variants than the component table')
    runners = {'knn_ablation': run_knn_ablation, 'newdata_ablation': run_newdata_ablation}
    protocols, reports, files = {}, {}, {}
    for key, runner in runners.items():
        protocols[key], reports[key], files[key] = verified_ablation(key, paths[key], runner, run_knn_ablation.SUMMARY_COLUMNS,
                                                                     allow_smoke=allow_smoke)
    try:
        rows, datasets = component_rows(reports)
        depths, by_dataset, depth_table, pooled, pooled_rows = component_depth(reports)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified ablation summaries do not support the component table: {_reason(exc)}') from exc
    counts = {variant: {
        'arrowflow_more_accurate': [r['dataset'] for r in rows if r['variant'] == variant and r['identical_to_views7_folds'] < r['n_folds']
                                    and r['arrowflow_minus_variant'] > 0],
        'variant_more_accurate': [r['dataset'] for r in rows if r['variant'] == variant and r['identical_to_views7_folds'] < r['n_folds']
                                  and r['arrowflow_minus_variant'] < 0],
        'identical_to_views7_in_every_fold': [r['dataset'] for r in rows if r['variant'] == variant and r['identical_to_views7_folds'] == r['n_folds']],
        'interval_excludes_zero': [r['dataset'] for r in rows if r['variant'] == variant and r['identical_to_views7_folds'] < r['n_folds']
                                   and (r['ci_low'] > 0 or r['ci_high'] < 0)]} for variant in COMPONENT_VARIANTS}
    groups = {name: row['group'] for row in rows for name in [row['dataset']]}
    record = {
        'purpose': 'task24_combined_component_ablation_of_the_benchmark_and_further_datasets',
        'status': 'descriptive: no p values and no multiplicity adjustment; nothing fitted here; both ablation runs re-verified by '
                  'their runners, and their published summaries equal the recomputation',
        'datasets': {'all': datasets, BENCHMARK: [name for name in datasets if groups[name] == BENCHMARK],
                     FURTHER: [name for name in datasets if groups[name] == FURTHER]},
        'variants': list(COMPONENT_VARIANTS),
        'definitions': {
            'arrowflow_minus_variant': 'views7 (ArrowFlow at its reconstructed per-fold selection, which reproduces the reference '
                                       'ArrowFlow outer predictions exactly) minus the variant, accuracy: the published '
                                       'change_from_views7 accuracy interval (variant minus views7; fitting seeds averaged within '
                                       'each outer fold, corrected resampled t over the outer folds, q = test_train_ratio, df = '
                                       'folds - 1) negated with its bounds swapped; positive when ArrowFlow is more accurate, the '
                                       'sign of Table 4; for untrained it is trained minus untrained',
            'identical_to_views7_folds': 'outer folds in which the variant coincides with views7 by construction (no_augment where '
                                         'the adaptive rule already switches augmentation off); its difference there is zero',
            'counts': 'datasets per variant, from the differences and unadjusted intervals of this table; descriptive',
            'depth': 'untrained minus views7 accuracy grouped by the hidden widths of the reconstructed selection of each outer fold, '
                     'as each ablation summary records it, with views7 minus untrained beside it; pooled across datasets per depth: '
                     'the dataset-fold count, mean and SD (knn_controls.pooled_depth_split); descriptive, no p values'},
        'components': rows, 'counts': counts,
        'depth_split': {'depths': depths, 'by_dataset': by_dataset, 'pooled': pooled},
        'provenance': {'runs': {key: {'run': str(Path(paths[key]).resolve()), 'protocol_id': protocols[key].get('protocol_id'),
                                      'frozen': protocols[key].get('frozen'), 'code_revision': reports[key]['code_revision'],
                                      'references': reports[key].get('reference_source') or reports[key].get('reference_sources'),
                                      'jobs_verified': sum(len(entry['resolved_configurations']) for entry in reports[key]['summaries'].values()),
                                      'views7_reproduces_reference': {name: entry['views7_reproduces_reference']
                                                                      for name, entry in reports[key]['summaries'].items()},
                                      'files': files[key]} for key in runners},
                       **code_record(COMPONENT_ANALYSIS_SOURCES)}}
    tables = {'components.csv': (rows, COMPONENT_COLUMNS), 'components_depth.csv': (depth_table, COMPONENT_DEPTH_COLUMNS),
              'components_depth_pooled.csv': (pooled_rows, COMPONENT_POOLED_COLUMNS)}
    finish(output, tables, 'components.json', record)
    return record


# ----------------------------------------------------------------------------- readiness

READY_RECORDS = {'benchmark': 'benchmark.json', 'training': 'training.json'}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ready(root, *, now=None):
    """READY for the manuscript: the benchmark and training outputs exist exactly as sealed, from committed and identical
    analysis sources; lists the commit and every output file with its sha256."""
    root = Path(root)
    listing, code = [], {}
    for directory, name in READY_RECORDS.items():
        try:
            record = json.loads((root/directory/name).read_text())
            present = sorted(path.name for path in (root/directory).iterdir())
        except (OSError, ValueError) as exc:
            raise RunComparisonError(f'{root/directory/name} is missing or unreadable ({_reason(exc)})') from exc
        outputs = record.get('outputs') or {}
        expected = sorted([*outputs, name])
        if present != expected:
            raise RunComparisonError(f'{root/directory} holds {present}; its record declares {expected}')
        changed = [csv_name for csv_name, seal in outputs.items() if sha256_file(root/directory/csv_name) != seal.get('sha256')]
        if changed:
            raise RunComparisonError(f'{root/directory}: {", ".join(changed)} differ from the sealed outputs')
        provenance = record.get('provenance') or {}
        if provenance.get('sources_committed_at_revision') is not True:
            raise RunComparisonError(f'{root/directory/name} was not written from committed analysis sources')
        code[directory] = {'code_revision': provenance.get('code_revision'), 'analysis_sources': provenance.get('analysis_sources')}
        listing += [{'path': f'{directory}/{file}', 'sha256': sha256_file(root/directory/file)} for file in expected]
    if len({canonical_json(entry['analysis_sources']) for entry in code.values()}) != 1:
        raise RunComparisonError('The benchmark and training outputs were written from different analysis sources')
    revisions = sorted({entry['code_revision'] for entry in code.values()})
    content = {'purpose': 'task24_holistic_outputs_ready_for_the_manuscript', 'commit': revisions[0] if len(revisions) == 1 else revisions,
               'code_revisions': {directory: entry['code_revision'] for directory, entry in code.items()},
               'analysis_sources': code['benchmark']['analysis_sources'], 'outputs': listing,
               'written_utc': now or datetime.now(timezone.utc).isoformat(timespec='seconds')}
    write_outputs(root, {'READY': _json_text(content)})
    return content


# ----------------------------------------------------------------------------- command

def _print_benchmark(record):
    counts, ranks = record['counts'], record['ranks']
    friedman, nemenyi = ranks['friedman'], ranks['nemenyi']
    print(f"Friedman over {friedman['n_datasets']} datasets x {friedman['n_models']} models: chi2 = {friedman['chi2']:.4f} on "
          f"{friedman['df']} df, p = {friedman['p']:.4f}; Iman-Davenport F = {friedman['iman_davenport_f']:.4f}, "
          f"p = {friedman['iman_davenport_p']:.4f}; Nemenyi CD = {nemenyi['critical_difference']:.4f}; pairs separated: "
          f"{len(nemenyi['pairs_exceeding_critical_difference'])} (descriptive)")
    print('mean ranks: ' + ', '.join(f'{model} {rank:.2f}' for model, rank in sorted(ranks['mean_ranks'].items(), key=lambda item: item[1])))
    print(counts['summary'])
    print('degenerate flags: ' + '; '.join(f"{flag} {', '.join(names) or 'none'}" for flag, names in counts['degenerate'].items()))
    print('selected widths: ' + json.dumps(record['selected_widths']['totals']))


def _print_training(record):
    for row in record['contrasts']:
        print(f"{row['dataset']} ({row['group']}): {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_unadjusted']:.3g} registered Holm p={row['registered_holm_p']:.3g} "
              f"({row['registered_family']})")
    holm = record['holm34']
    print(f"Holm over all {holm['members']} contrasts (post hoc): below {holm['alpha']}: " +
          (', '.join(f"{entry['dataset']} vs {entry['control']} ({entry['holm34_p']:.3g})" for entry in holm['below_alpha']) or 'none'))
    moderator, spearman = record['stratum_test']['moderator_test'], record['stratum_test']['spearman']
    print(f"stratum test: {moderator['statistic']:+.4f}, exact one-sided p = {moderator['at_least_observed']}/{moderator['splits']}; "
          f"Spearman rho {spearman['rho']:.4f} (n = {spearman['n']}), one-sided p {spearman['p_one_sided']:.4f}")
    print('depth folds: ' + json.dumps(record['counts']['depth_folds']))


def _print_components(record):
    total = len(record['datasets']['all'])
    for variant, counts in record['counts'].items():
        print(f"{variant}: ArrowFlow more accurate on {len(counts['arrowflow_more_accurate'])}, the variant on "
              f"{len(counts['variant_more_accurate'])}, identical by construction on {len(counts['identical_to_views7_in_every_fold'])} "
              f"of {total} datasets; interval excludes zero on {len(counts['interval_excludes_zero'])} (descriptive)")
    print('depth folds: ' + json.dumps({scope: {json.dumps(entry['widths']): entry['n_dataset_folds'] for entry in entries}
                                        for scope, entries in record['depth_split']['pooled'].items()}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, text in (('benchmark', 'the combined main benchmark'), ('training', 'the combined training controls')):
        sub = commands.add_parser(command, help=text)
        sub.add_argument('--runs', type=Path, default=WORKSPACE_RUNS, help='directory holding the source runs under their names')
        sub.add_argument('--output', type=Path, required=True, help='new output directory')
    components_parser = commands.add_parser('components', help='the combined component ablation (after both ablation runs)')
    components_parser.add_argument('--runs', type=Path, default=WORKSPACE_RUNS, help='directory holding both ablation runs')
    components_parser.add_argument('--output', type=Path, required=True, help='new output directory')
    ready_parser = commands.add_parser('ready', help='check the benchmark and training outputs and write READY')
    ready_parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'benchmark':
            _print_benchmark(benchmark(source_paths(args.runs), args.output))
        elif args.command == 'training':
            _print_training(training(source_paths(args.runs), args.output))
        elif args.command == 'components':
            _print_components(components(ablation_paths(args.runs), args.output))
        else:
            content = ready(args.root)
            print(f"READY at {content['commit']}: {len(content['outputs'])} output files")
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'holistic {args.command} refused: {exc}\n')


if __name__ == '__main__':
    main()
