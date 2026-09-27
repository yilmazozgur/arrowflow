"""Report-ready tables and figure data from frozen training-diagnostics outputs (post-processing only: no model is fitted,
nothing is selected, no test is made).

tables --diagnostics DIAG_DIR [--diagnostics DIAG_DIR2 ...] --ablation NAME=ABLATION_DIR ... --output OUT [--allow-smoke]

DIAG_DIR is a complete training_diagnostics run directory (the five CSVs, provenance.json, diagnostics_summary.json, the sealed
plan, job records and artifacts). NAME is a reference recorded in that run's provenance.json (bridge_knn, newdata_batch1,
newdata_batch2 for arrowflow-v3-training-diagnostics-1) and ABLATION_DIR the component ablation run that sealed its per-fold
selections; every reference of every diagnostics run needs one --ablation.

Refused (exit status 2, nothing written):
- provenance.json missing, or reporting a failed or incomplete check (its check totals, any job's checks, or
  diagnostics_summary.json disagreeing with them or missing);
- a file whose sha256 differs from provenance.json (the five CSVs, protocol.json, every job record and artifact);
- a run directory that training_diagnostics.verify refuses (frozen protocol, seals, unchanged sources); --allow-smoke admits a
  synthetic smoke run, whose tables are marked as never evidence;
- an ablation run whose pins differ from those the diagnostics run recorded (training_diagnostics.load_ablation), that its own
  family verify refuses, or whose job record for a used fold fails run_knn_ablation.validate_job;
- a 7-view (or per-view) kNN accuracy at the checkpoint network that differs from the reference seed accuracy (the sealed
  reference predictions, the ablation's views7 record and its per-view votes);
- inconsistent tables (for example a checkpoint-0 view whose checkpoint displacement is not 0), or an existing output file
  with different content.

Outputs (all or none; an existing file with different content is never replaced): the CSVs in COLUMNS and
diagnostics_tables.json with every column definition, the random references, the verified inputs and the sha256 of each CSV.
"""
import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
import numpy as np
from . import run_knn_ablation as base
from . import run_newdata_ablation as newdata_ablation
from . import training_diagnostics as td
from .evaluation import canonical_json
from .run_revision import load_prepared

OUTPUT_JSON = 'diagnostics_tables.json'
TOLERANCE = 1e-12                       # accuracies of the same predictions are equal; a real difference is >= 1 / n_test
TIE_REFERENCE = {'seed': 20260915, 'filter_sets': 200, 'rows_per_set': 25}
INITIAL_KNN = 'initial filters, final readout setting'
INITIAL = 'initial filters'
AFTER = 'after updates'
AT_CHECKPOINT = 'each view at its checkpoint'
ABLATION_VERIFIERS = {'knn_ablation': base.verify, 'newdata_ablation': newdata_ablation.verify}
SCRATCH_ERRORS = (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile)

GROUP = (('scope', 'dataset: one dataset; pooled: every dataset of the diagnostics protocol, each contributing all its outer folds '
                   'and views (datasets of different protocols are never pooled); pooled_by_architecture (t4 only): pooled over '
                   'datasets within one (widths, embed_dim)'),
         ('protocol_id', 'the diagnostics protocol of the row'),
         ('dataset_id', 'the dataset; empty in pooled rows'),
         ('reference', 'the reference run of the dataset (provenance.json references); empty in pooled rows'))
LAYER = (('widths', 'hidden widths of the view network, canonical JSON (for example [128] or [64,128])'),
         ('layer', 'hidden_l (l the 0-based position among the hidden layers) or output'),
         ('n', 'permutation length: the number of items each filter of the layer orders; embed_dim for hidden_0, the previous '
               'hidden width for hidden_l with l > 0, the last hidden width for the output layer'))
COUNT = (('n_datasets', 'datasets contributing to the row'),)
FOLD_STATS = (('n_folds', 'outer folds contributing to the row'),
              ('views_per_fold', 'views averaged within each fold before averaging over folds (7 for per_view, 1 for '
                                 'seven_view_majority)'))
LATER = 'views with checkpoint > 0 only (checkpoint-0 views are excluded, never averaged in); empty when there is none'
COLUMNS = {
    't1_checkpoints.csv': GROUP + COUNT + (
        ('n_folds', 'outer folds'), ('n_views', 'views (outer folds x 7)'),
        ('iterations_min', 'smallest number of training updates T of a view'), ('iterations_max', 'largest T'),
        ('n_validation_samples_min', 'smallest number of core validation rows deciding a checkpoint: the checkpoint is the last '
                                     'strict improvement of the output-rule error on these rows, not a kNN criterion'),
        ('n_validation_samples_max', 'largest number of core validation rows'),
        ('n_views_checkpoint_0', 'views whose checkpoint is iteration 0 (the core returned the initial filters)'),
        ('share_checkpoint_0', 'n_views_checkpoint_0 / n_views; on such a view the network equals its initial network'),
        ('checkpoint_median', 'median checkpoint iteration over all views (numpy.percentile, linear)'),
        ('checkpoint_q25', 'first quartile over all views'), ('checkpoint_q75', 'third quartile over all views'),
        ('checkpoint_iqr', 'checkpoint_q75 - checkpoint_q25 over all views'),
        ('n_views_later_checkpoint', 'views with checkpoint > 0'),
        ('later_checkpoint_median', 'median checkpoint iteration over ' + LATER),
        ('later_checkpoint_q25', 'first quartile over ' + LATER), ('later_checkpoint_q75', 'third quartile over ' + LATER),
        ('later_checkpoint_iqr', 'interquartile range over ' + LATER)),
    't1_checkpoint_counts.csv': GROUP + (
        ('n_views_total', 'views in the group'), ('iterations_max', 'largest T in the group (histogram range)'),
        ('checkpoint_iteration', 'a checkpoint iteration that occurs (values without a view are omitted)'),
        ('n_views', 'views with this checkpoint iteration'), ('share_of_views', 'n_views / n_views_total')),
    't2_displacement.csv': GROUP + COUNT + LAYER + (
        ('n_views', 'views in the group (every checkpoint)'), ('n_views_checkpoint_0', 'views whose checkpoint is iteration 0'),
        ('share_checkpoint_0', 'n_views_checkpoint_0 / n_views'), ('n_views_later_checkpoint', 'views with checkpoint > 0'),
        ('displacement_mean_later', 'mean over ' + LATER + ' of the checkpoint displacement from the initial filters: per view the '
                                    'mean over the layer filters of footrule(filter at checkpoint, initial filter) / floor(n^2 / 2)'),
        ('displacement_sd_later', 'sample SD (ddof 1) over the same views; empty with fewer than 2'),
        ('changed_share_mean_later', 'mean over the same views of the share of the layer filters that differ from their initial '
                                     'ordering'),
        ('random_displacement_reference', 'exact random reference (n^2 - 1) / 3 / floor(n^2 / 2): the expected footrule between two '
                                          'independent uniform permutations of length n over the maximum footrule')),
    't3_ties.csv': GROUP + COUNT + LAYER + (
        ('n_filters', 'W: the layer width of a hidden layer; the number of classes of the output layer'),
        ('n_views', 'views in the group'),
        ('tied_response_share_initial', 'hidden layers: mean over views of the share of (row, filter) responses on the outer test '
                                        'rows equal to another filter response in the same row, initial filters (iteration 0)'),
        ('tied_response_share_checkpoint', 'the same with the checkpoint filters'),
        ('random_tied_response_share', 'random-filter reference: the same tie share for uniform random input permutations of '
                                       'length n against W uniform random filters (seeded simulation)'),
        ('distinct_response_ratio_initial', 'hidden layers: mean over views of the mean over rows of distinct responses / W, '
                                            'initial filters'),
        ('distinct_response_ratio_checkpoint', 'the same with the checkpoint filters'),
        ('random_distinct_response_ratio', 'random-filter reference of the distinct-response ratio (the same simulation)'),
        ('distinct_values_bound', 'floor(n^2 / 4) + 1: a footrule between permutations of length n is even and at most '
                                  'floor(n^2 / 2), so a hidden response takes at most this many values'),
        ('distinct_response_ratio_bound', 'min(1, distinct_values_bound / W)'),
        ('tied_nearest_share_initial', 'output layer: mean over views of the share of outer test rows whose smallest response is '
                                       'attained by more than one class filter, initial filters'),
        ('tied_nearest_share_checkpoint', 'the same with the checkpoint filters')),
    't4_relabel.csv': GROUP + COUNT + (
        ('widths', 'hidden widths (pooled_by_architecture rows only)'),
        ('embed_dim', 'permutation length of hidden_0 (pooled_by_architecture rows only)'),
        ('level', 'view: every view of every fold (unit = fold x view); seven_view_majority: the 7-view majority (unit = fold)'),
        ('n_units', 'units in the row'), ('draws_per_unit', 'relabeling draws per unit'),
        ('checkpoint_accuracy_mean', 'mean over units of the outer-test kNN accuracy at the checkpoint network, not relabeled'),
        ('changed_share_mean', 'mean over units and draws of the share of outer test predictions that change'),
        ('changed_share_max', 'largest share over units and draws'),
        ('share_of_draws_with_changes', 'draws that change at least one prediction / all draws'),
        ('accuracy_change_mean', 'mean over units and draws of relabeled minus checkpoint accuracy'),
        ('accuracy_change_min', 'smallest accuracy change'), ('accuracy_change_max', 'largest accuracy change'),
        ('abs_accuracy_change_max', 'largest absolute accuracy change')),
    't5_learning_curves.csv': GROUP + (
        ('snapshot', 'scheduled: the network after `iteration` updates; checkpoint: each view at its own checkpoint'),
        ('iteration', 'scheduled snapshot iteration (0, every 10 updates, T); empty in checkpoint rows'),
        ('point', f'"{INITIAL_KNN}" for kNN measures at iteration 0 (never the untrained ArrowFlow); "{INITIAL}" for the output '
                  f'rule and the core validation error at 0; "{AFTER}" for iteration > 0; "{AT_CHECKPOINT}"'),
        ('level', 'seven_view_majority or per_view'),
        ('measure', 'knn_test_accuracy, output_rule_test_accuracy, output_rule_training_accuracy (all outer training rows, '
                    'including the core validation rows) or core_validation_error'),
    ) + FOLD_STATS + (
        ('mean', 'mean over folds of the fold value (per_view: the mean over the fold views first)'),
        ('sd_over_folds', 'sample SD over folds of the fold value; empty with fewer than 2 folds'),
        ('min_over_folds', 'smallest fold value'), ('max_over_folds', 'largest fold value')),
    't5_validation_curves.csv': GROUP + (('iteration', 'update 0..T (0: the initial filters)'),) + FOLD_STATS + (
        ('validation_error_mean', 'mean over folds of the fold mean over views of the core validation error after `iteration` '
                                  'updates'),
        ('validation_error_sd_over_folds', 'sample SD over folds of the fold mean; empty with fewer than 2 folds'),
        ('running_minimum_mean', 'mean over folds and views of the running minimum of the core validation error'),
        ('improved_share', 'share of views whose error at this iteration is strictly below the running minimum before it; '
                           'empty at iteration 0')),
    't5_displacement_curves.csv': GROUP + LAYER + (
        ('iteration', 'scheduled snapshot iteration of the training trajectory'),
        ('n_folds', 'outer folds contributing'), ('n_views', 'views contributing (all views: every view trains for T updates, '
                                                             'whatever checkpoint the core returns)'),
        ('displacement_from_initial_mean', 'mean over views of the normalized footrule displacement of the network after '
                                           '`iteration` updates from the initial filters'),
        ('changed_share_from_initial_mean', 'mean over views of the share of filters that differ from the initial filters'),
        ('displacement_from_previous_mean', 'the same against the previous scheduled snapshot; empty at iteration 0'),
        ('changed_share_from_previous_mean', 'share of filters changed since the previous scheduled snapshot; empty at 0'),
        ('random_displacement_reference', '(n^2 - 1) / 3 / floor(n^2 / 2)')),
    't6_trained_untrained.csv': GROUP + (
        ('model_seed', 'the fitting seed of the diagnostics (one of the three registered seeds)'),
        ('n_folds', 'outer folds'),
        ('trained_accuracy_mean', 'mean over folds of the 7-view kNN outer-test accuracy at the checkpoint network (diagnostics); '
                                  'equal on every fold to the reference ArrowFlow-kNN accuracy of this seed (checked)'),
        ('untrained_accuracy_mean', 'mean over the same folds of the component ablation untrained variant accuracy for the same '
                                    'seed and selected configuration'),
        ('difference_mean', 'mean over folds of trained minus untrained accuracy'),
        ('difference_sd', 'sample SD over folds; empty with fewer than 2'),
        ('difference_min', 'smallest fold difference'), ('difference_max', 'largest fold difference')),
    't6_trained_untrained_folds.csv': (
        ('protocol_id', 'the diagnostics protocol'), ('dataset_id', 'the dataset'), ('reference', 'the reference run'),
        ('outer_repeat', 'outer repeat'), ('outer_fold', 'outer fold'), ('model_seed', 'the fitting seed'),
        ('widths', 'hidden widths of the fold selection'), ('embed_dim', 'embed_dim of the fold selection'),
        ('n_test', 'outer test rows'), ('n_views_checkpoint_0', 'views of the fold whose checkpoint is iteration 0'),
        ('trained_accuracy', '7-view kNN outer-test accuracy at the checkpoint network (diagnostics)'),
        ('reference_accuracy', 'accuracy of the sealed reference predictions for the seed (equal to trained_accuracy, checked)'),
        ('untrained_accuracy', 'accuracy of the component ablation untrained variant for the seed'),
        ('difference', 'trained_accuracy - untrained_accuracy')),
}
FILES = tuple(COLUMNS)
VERIFY_ERRORS = (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile)
READOUT_MEASURES = {'knn_test_accuracy', 'output_rule_test_accuracy', 'output_rule_training_accuracy'}


class Refusal(RuntimeError):
    """An input or output the tables refuse: the command exits with status 2 and writes nothing."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path):
    return sha256_bytes(Path(path).read_bytes())


# ----------------------------------------------------------------------------- random references

def random_displacement_reference(n):
    """E footrule(sigma, tau) / floor(n^2 / 2) for independent uniform permutations sigma and tau of length n, where
    E footrule = (n^2 - 1) / 3 (Diaconis and Graham 1977); rounded once from the exact fraction."""
    n = int(n)
    if n < 2:
        raise ValueError('A permutation length of at least 2 is required')
    return float(Fraction(n * n - 1, 3 * (n * n // 2)))


def distinct_values_bound(n):
    """floor(n^2 / 4) + 1: a footrule between two permutations of length n is even and at most floor(n^2 / 2)."""
    return int(n) * int(n) // 4 + 1


def random_tie_reference(n, width, *, seed=TIE_REFERENCE['seed'], filter_sets=TIE_REFERENCE['filter_sets'],
                         rows_per_set=TIE_REFERENCE['rows_per_set']):
    """The random-filter tie reference of a hidden layer that ranks n items with `width` filters: filter_sets draws of `width`
    uniform random filters (position vectors of length n), each against rows_per_set uniform random input permutations, the
    responses by training_diagnostics.network_forward (cityblock) and the tie measures by training_diagnostics.response_ties,
    averaged over the filter sets with their Monte Carlo standard errors. The generator is seeded by (seed, n, width), so a
    reference does not depend on which other layers a run holds."""
    n, width, filter_sets, rows_per_set = int(n), int(width), int(filter_sets), int(rows_per_set)
    if n < 1 or width < 1 or filter_sets < 2 or rows_per_set < 1:
        raise ValueError('Expected n >= 1, width >= 1, at least two filter sets and one row per set')
    rng = np.random.default_rng([int(seed), n, width])
    tied, distinct = [], []
    for _ in range(filter_sets):
        filters = rng.permuted(np.tile(np.arange(n), (width, 1)), axis=1)
        rows = rng.permuted(np.tile(np.arange(n), (rows_per_set, 1)), axis=1)
        share, ratio = td.response_ties(td.network_forward(rows, [filters])[0][0])
        tied.append(share)
        distinct.append(ratio)
    return {'n': n, 'n_filters': width, 'seed': int(seed), 'filter_sets': filter_sets, 'rows_per_set': rows_per_set,
            'tied_response_share': float(np.mean(tied)),
            'tied_response_share_mc_se': float(np.std(tied, ddof=1) / math.sqrt(filter_sets)),
            'distinct_response_ratio': float(np.mean(distinct)),
            'distinct_response_ratio_mc_se': float(np.std(distinct, ddof=1) / math.sqrt(filter_sets))}


# ----------------------------------------------------------------------------- statistics and CSV text

def mean(values):
    return float(np.mean(values)) if len(values) else None


def sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def quartiles(values):
    """(median, q25, q75, iqr) by numpy.percentile (linear); four None without values."""
    if not len(values):
        return None, None, None, None
    q25, median, q75 = (float(x) for x in np.percentile(np.asarray(values, dtype=float), [25, 50, 75]))
    return median, q25, q75, q75 - q25


def cell(value):
    if value is None:
        return ''
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('A table value is not finite')
        return value
    return value


def render_csv(name, rows):
    """CSV text of one output table: the COLUMNS header, then each row dict (a missing column is an empty cell)."""
    columns = [column for column, _ in COLUMNS[name]]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    for row in rows:
        unknown = sorted(set(row) - set(columns))
        if unknown:
            raise ValueError(f'{name}: unknown columns {unknown}')
        writer.writerow([cell(row.get(column)) for column in columns])
    return buffer.getvalue()


# ----------------------------------------------------------------------------- diagnostics runs

RELABEL_FIELDS = ('checkpoint_accuracy', 'changed_share_mean', 'changed_share_max', 'accuracy_change_mean', 'accuracy_change_min',
                  'accuracy_change_max', 'abs_accuracy_change_max')
IDENTITY = ('protocol_id', 'dataset_id', 'reference', 'outer_repeat', 'outer_fold', 'model_seed', 'widths', 'widths_text', 'embed_dim')


def provenance_problems(provenance):
    """The failed or incomplete checks a diagnostics provenance.json reports (empty when every check passed)."""
    totals = provenance.get('check_totals')
    if not isinstance(totals, dict) or not totals:
        return ['no check totals']
    problems = [f'the check totals lack {name}' for name in td.JOB_CHECKS if name not in totals]
    for name, total in sorted(totals.items()):
        total = total if isinstance(total, dict) else {}
        if not total or total.get('passed') != total.get('performed'):
            problems.append(f'{name} passed {total.get("passed")} of {total.get("performed")}')
    jobs = provenance.get('jobs')
    if not isinstance(jobs, dict) or not jobs or provenance.get('planned_jobs') != len(jobs):
        return problems + ['the job entries do not cover the planned jobs']
    if (totals.get('reference_predictions') or {}).get('performed') != len(jobs):
        problems.append('the reference-prediction check was not performed on every job')
    for stem, job in sorted(jobs.items()):
        checks = job.get('checks') if isinstance(job, dict) else None
        if not isinstance(checks, dict) or any(name not in checks for name in td.JOB_CHECKS):
            problems.append(f'{stem}: incomplete checks')
            continue
        for name, check in sorted(checks.items()):
            performed, passed = check.get('performed'), check.get('passed')
            if (performed is True and passed is not True) or (performed is not True and passed is not None):
                problems.append(f'{stem}: {name} performed {performed}, passed {passed}')
    return problems


def verified_texts(directory, provenance):
    """{diagnostics CSV name: text} once every file whose sha256 provenance.json records (the five CSVs, protocol.json, every job
    record and artifact) matched it."""
    tables = provenance.get('tables')
    if not isinstance(tables, dict) or sorted(tables) != sorted(td.TABLES):
        raise Refusal(f'{directory/td.PROVENANCE_FILE} does not record the sha256 of the five diagnostics tables')
    expected = [(name, tables[name]) for name in td.TABLES] + [('protocol.json', provenance.get('protocol_sha256'))]
    for stem, job in sorted(provenance['jobs'].items()):
        expected += [(f'jobs/{stem}.json', job.get('record_sha256')), (f'artifacts/{stem}.npz', job.get('artifact_sha256'))]
    texts, problems = {}, []
    for relative, digest in expected:
        path = directory/relative
        if not path.is_file():
            problems.append(f'{relative} is missing')
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != digest:
            problems.append(f'{relative} differs from its recorded sha256')
        elif relative in td.TABLES:
            texts[relative] = data.decode('utf-8')
    if problems:
        raise Refusal(f'{directory}: {len(problems)} file(s) differ from provenance.json: ' + '; '.join(problems[:8]))
    return texts


def verified_summary(directory, provenance):
    """sha256 of diagnostics_summary.json, which must exist and agree with provenance.json."""
    path = directory/td.SUMMARY_FILE
    if not path.is_file():
        raise Refusal(f'{path} is missing: the diagnostics summary stage (it re-verifies every record and re-derives every '
                      'selection) has not completed')
    try:
        summary = json.loads(path.read_text())
    except ValueError as exc:
        raise Refusal(f'{path} is not valid JSON: {exc}') from exc
    wrong = [key for key in ('protocol_id', 'datasets', 'planned_jobs', 'check_totals')
             if not isinstance(summary, dict) or summary.get(key) != provenance.get(key)]
    if wrong:
        raise Refusal(f'{path} disagrees with provenance.json on {", ".join(wrong)}')
    return sha256_path(path)


def load_run(directory, *, allow_smoke=False):
    """One verified diagnostics run: provenance, plan and its folds with their views joined from the five tables."""
    directory = Path(directory).resolve()
    path = directory/td.PROVENANCE_FILE
    if not path.is_file():
        raise Refusal(f'{path} is missing')
    try:
        provenance = json.loads(path.read_text())
    except ValueError as exc:
        raise Refusal(f'{path} is not valid JSON: {exc}') from exc
    problems = provenance_problems(provenance) if isinstance(provenance, dict) else ['not a JSON object']
    if problems:
        raise Refusal(f'{path} reports failed or incomplete checks: ' + '; '.join(problems[:8]))
    if provenance.get('frozen') is not True and not allow_smoke:
        raise Refusal(f'{directory} is not a frozen diagnostics run (--allow-smoke admits a synthetic smoke run, never evidence)')
    texts = verified_texts(directory, provenance)
    try:
        p, manifest, jobs, selections = td.verify(directory, allow_smoke=allow_smoke)
    except VERIFY_ERRORS as exc:
        raise Refusal(f'{directory}: training_diagnostics.verify refuses the run: {type(exc).__name__}: {exc}') from exc
    if (provenance.get('protocol_id') != p['protocol_id'] or provenance.get('protocol_hash') != manifest['protocol_hash']
            or provenance.get('datasets') != manifest['datasets'] or provenance.get('references') != manifest['references']
            or sorted(provenance['jobs']) != sorted(job['stem'] for job in jobs)):
        raise Refusal(f'{path} does not describe the sealed plan of {directory}')
    summary_sha256 = verified_summary(directory, provenance)
    try:
        folds = parse_run(texts, p, jobs)
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise Refusal(f'{directory}: inconsistent diagnostics tables: {type(exc).__name__}: {exc}') from exc
    return {'directory': directory, 'protocol': p, 'protocol_id': p['protocol_id'], 'model_seed': int(p['model_seed']),
            'smoke': manifest.get('purpose') == 'synthetic_smoke_only', 'references': provenance['references'],
            'selections': selections, 'datasets': list(manifest['datasets']), 'folds': folds,
            'views': [view for fold in folds for view in fold['views']],
            'record': {'directory': str(directory), 'protocol_id': p['protocol_id'], 'protocol_sha256': provenance['protocol_sha256'],
                       'frozen': provenance['frozen'], 'purpose': manifest.get('purpose'),
                       'code_revision': provenance.get('code_revision'), 'datasets': list(manifest['datasets']),
                       'planned_jobs': len(jobs), 'check_totals': provenance['check_totals'],
                       'provenance_sha256': sha256_path(path), 'summary_sha256': summary_sha256,
                       'tables_sha256': provenance['tables'],
                       'references': {name: {'datasets': entry['datasets'], 'ablation': entry['ablation']}
                                      for name, entry in sorted(provenance['references'].items())}}}


def rows_of(texts, name):
    reader = csv.reader(io.StringIO(texts[name]))
    header = tuple(next(reader, ()))
    if header != td.TABLES[name]:
        raise ValueError(f'{name}: the columns differ from training_diagnostics.TABLES')
    for row in reader:
        if len(row) != len(header):
            raise ValueError(f'{name}: a row has {len(row)} cells, expected {len(header)}')
        yield dict(zip(header, row))


def optional_float(text):
    return None if text == '' else float(text)


def parse_run(texts, p, jobs):
    """The folds of one run in planned order, each with its views joined from the five tables (check_fold on each)."""
    every, n_views, seed = int(p['snapshots']['every']), int(p['n_views']), int(p['model_seed'])
    folds, views = {}, {}
    for job in jobs:
        key = (job['dataset_id'], int(job['outer_repeat']), int(job['outer_fold']))
        widths = tuple(int(w) for w in job['selected']['widths'])
        folds[key] = {'protocol_id': p['protocol_id'], 'dataset_id': key[0], 'reference': job['reference'], 'outer_repeat': key[1],
                      'outer_fold': key[2], 'model_seed': seed, 'widths': widths, 'widths_text': canonical_json(list(widths)),
                      'embed_dim': int(job['selected']['embed_dim']), 'majority_scheduled': {}, 'majority_checkpoint': None,
                      'relabel': None, 'views': []}

    def fold_of(row):
        return folds[(row['dataset_id'], int(row['outer_repeat']), int(row['outer_fold']))]

    def view_of(row):
        return views[(row['dataset_id'], int(row['outer_repeat']), int(row['outer_fold']), int(row['view']))]

    for row in rows_of(texts, 'checkpoints.csv'):
        fold = fold_of(row)
        key = (fold['dataset_id'], fold['outer_repeat'], fold['outer_fold'], int(row['view']))
        checkpoint = int(row['checkpoint_iteration'])
        if key in views or row['reference'] != fold['reference'] or int(row['model_seed']) != seed:
            raise ValueError(f'{key}: a duplicate view or an identity that differs from the planned job')
        if tuple(json.loads(row['widths'])) != fold['widths'] or int(row['embed_dim']) != fold['embed_dim']:
            raise ValueError(f'{key}: widths or embed_dim differ from the planned selection')
        if (row['returned_initial_filters'] == '1') != (checkpoint == 0) or row['orders_verified'] != '1':
            raise ValueError(f'{key}: checkpoint flags')
        widths = fold['widths']
        items = [fold['embed_dim'], *widths]
        names = [f'hidden_{index}' for index in range(len(widths))] + ['output']
        view = {**{k: fold[k] for k in IDENTITY}, 'view': key[3], 'iterations': int(row['iterations']),
                'n_validation_samples': int(row['n_validation_samples']), 'checkpoint': checkpoint,
                'checkpoint_validation_error': float(row['checkpoint_validation_error']),
                'layers': {name: {'layer': name, 'n': items[index], 'n_filters': widths[index] if index < len(widths) else None,
                                  'at_checkpoint': {}, 'trajectory': defaultdict(dict), 'ties': {}}
                           for index, name in enumerate(names)},
                'scheduled': defaultdict(dict), 'at_checkpoint': {}, 'checkpoint_iterations': set(), 'validation': {}, 'relabel': None}
        views[key] = view
        fold['views'].append(view)
    for row in rows_of(texts, 'snapshots.csv'):
        kind, measure, value = row['snapshot'], row['measure'], float(row['value'])
        if kind not in ('scheduled', 'checkpoint'):
            raise ValueError(f'snapshot kind {kind!r}')
        if row['view'] == 'majority':
            fold = fold_of(row)
            if row['layer'] != 'readout' or measure != 'knn_test_accuracy':
                raise ValueError('a majority row that is not the kNN test accuracy')
            if kind == 'checkpoint':
                if row['iteration'] != '' or fold['majority_checkpoint'] is not None:
                    raise ValueError('a second or iterated majority checkpoint row')
                fold['majority_checkpoint'] = value
            elif fold['majority_scheduled'].setdefault(int(row['iteration']), value) is not value:
                raise ValueError('a duplicate majority row')
            continue
        view, iteration = view_of(row), int(row['iteration'])
        if kind == 'checkpoint':
            view['checkpoint_iterations'].add(iteration)
        if row['layer'] == 'readout':
            target = view['scheduled'][iteration] if kind == 'scheduled' else view['at_checkpoint']
        else:
            layer = view['layers'][row['layer']]
            target = layer['trajectory'][iteration] if kind == 'scheduled' else layer['at_checkpoint']
        if measure in target:
            raise ValueError(f'a duplicate {measure} row')
        target[measure] = value
    for row in rows_of(texts, 'validation_curves.csv'):
        view, iteration = view_of(row), int(row['iteration'])
        if iteration in view['validation']:
            raise ValueError('a duplicate validation row')
        view['validation'][iteration] = (float(row['validation_error']), float(row['running_minimum']),
                                         None if row['improved'] == '' else int(row['improved']))
    for row in rows_of(texts, 'ties.csv'):
        view, network = view_of(row), row['network']
        layer = view['layers'][row['layer']]
        if network not in ('initial', 'checkpoint') or network in layer['ties']:
            raise ValueError(f'tie rows of network {network!r}')
        if layer['n_filters'] is None:
            layer['n_filters'] = int(row['n_filters'])            # the output layer: the number of classes
        if int(row['n_filters']) != layer['n_filters']:
            raise ValueError(f'{row["layer"]} has {row["n_filters"]} filters')
        layer['ties'][network] = {'iteration': int(row['iteration']),
                                  **{field: optional_float(row[field]) for field in
                                     ('tied_response_share', 'distinct_response_ratio', 'tied_nearest_share')}}
    for row in rows_of(texts, 'relabel.csv'):
        target = fold_of(row) if row['view'] == 'majority' else view_of(row)
        if target['relabel'] is not None:
            raise ValueError('a duplicate relabel row')
        target['relabel'] = {'draws': int(row['draws']), 'draws_with_changes': int(row['draws_with_changes']),
                             **{field: float(row[field]) for field in RELABEL_FIELDS}}
    for key, fold in folds.items():
        check_fold(key, fold, every, n_views)
    return list(folds.values())


def check_fold(key, fold, every, n_views):
    """What the tables rely on: the fold's views 0..n_views-1 with one schedule, every readout, displacement, validation, tie and
    relabel row, and no movement at the checkpoint of a checkpoint-0 view."""
    fold['views'].sort(key=lambda view: view['view'])
    if [view['view'] for view in fold['views']] != list(range(n_views)):
        raise ValueError(f'{key}: views {[view["view"] for view in fold["views"]]}')
    schedules = {tuple(td.scheduled_iterations(view['iterations'], every)) for view in fold['views']}
    schedule = list(schedules.pop()) if len(schedules) == 1 else None
    if schedule is None or sorted(fold['majority_scheduled']) != schedule or fold['majority_checkpoint'] is None or fold['relabel'] is None:
        raise ValueError(f'{key}: the schedule or the majority rows')
    for view in fold['views']:
        c, T, where = view['checkpoint'], view['iterations'], f'{key} view {view["view"]}'
        if view['checkpoint_iterations'] != {c} or not 0 <= c <= T or sorted(view['validation']) != list(range(T + 1)):
            raise ValueError(f'{where}: the checkpoint snapshot or the validation curve')
        if (sorted(view['scheduled']) != schedule or any(set(view['scheduled'][t]) != READOUT_MEASURES for t in schedule)
                or set(view['at_checkpoint']) != READOUT_MEASURES or view['relabel'] is None):
            raise ValueError(f'{where}: the readout or relabel rows')
        for layer in view['layers'].values():
            trajectory, hidden = layer['trajectory'], layer['layer'] != 'output'
            moved = (layer['at_checkpoint'].get('displacement_from_initial'), layer['at_checkpoint'].get('changed_share_from_initial'))
            if None in moved or sorted(trajectory) != schedule or any(
                    not {'displacement_from_initial', 'changed_share_from_initial'} <= set(trajectory[t])
                    or ('displacement_from_previous' in trajectory[t]) != (t > 0) for t in schedule):
                raise ValueError(f'{where} {layer["layer"]}: the displacement rows')
            if c == 0 and moved != (0.0, 0.0):
                raise ValueError(f'{where} {layer["layer"]}: a checkpoint-0 view has a non-zero checkpoint displacement')
            ties = layer['ties']
            if set(ties) != {'initial', 'checkpoint'} or ties['initial']['iteration'] != 0 or ties['checkpoint']['iteration'] != c or any(
                    (entry['tied_response_share'] is None) == hidden or (entry['distinct_response_ratio'] is None) == hidden
                    or (entry['tied_nearest_share'] is None) != hidden for entry in ties.values()):
                raise ValueError(f'{where} {layer["layer"]}: the tie rows')


# ----------------------------------------------------------------------------- component ablations (the untrained comparison)

def accuracy(predicted, truth):
    return float(np.mean(np.asarray(predicted) == np.asarray(truth)))


def parse_ablation(text):
    """NAME=ABLATION_DIR -> (NAME, Path(ABLATION_DIR))."""
    name, separator, directory = str(text).partition('=')
    if not name or not separator or not directory:
        raise argparse.ArgumentTypeError('--ablation takes NAME=ABLATION_DIR')
    return name, Path(directory)


def family_verify(directory, allow_smoke):
    """The ablation run's own directory verification (its family verify, environment_check='sources'): (protocol, planned jobs
    by fold, code revision, family)."""
    try:
        family = json.loads((directory/'protocol.json').read_text()).get('production_family')
    except (OSError, ValueError, AttributeError) as exc:
        raise Refusal(f'{directory}: no readable ablation protocol.json: {exc}') from exc
    verifier = ABLATION_VERIFIERS.get(family)
    if verifier is None:
        raise Refusal(f'{directory}: ablation family {family!r} is not supported ({", ".join(sorted(ABLATION_VERIFIERS))})')
    try:
        p, _, jobs = verifier(directory, allow_smoke=allow_smoke, environment_check='sources')
        revision = json.loads((directory/'environment.json').read_text())['code_revision']
    except VERIFY_ERRORS as exc:
        raise Refusal(f'{directory}: the {family} verification refuses the ablation run: {type(exc).__name__}: {exc}') from exc
    return p, {(job['dataset_id'], int(job['outer_repeat']), int(job['outer_fold'])): job for job in jobs}, revision, family


def ablation_fold(directory, key, job, sealed, diagnostics_sealed, p, revision, seed, prepared):
    """One fold of the component ablation after run_knn_ablation.validate_job: the seed's untrained and views7 accuracies, the
    accuracy of the sealed reference predictions and the per-view kNN accuracies of the saved views7 votes."""
    if sealed != diagnostics_sealed:
        raise ValueError('the sealed selection differs from the one the diagnostics run sealed')
    if key[0] not in prepared:
        prepared[key[0]] = load_prepared(directory, key[0])
    X, y, data, splits = prepared[key[0]]
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == key[1:])
    stem = job['stem']
    result = json.loads((directory/'results'/f'{stem}.json').read_text())
    events = [json.loads(line) for line in (directory/'logs'/f'{stem}.jsonl').read_text().splitlines()]
    base.validate_job(result, events, job, p, X, y, split, data, revision, directory/'artifacts'/stem,
                      directory/'predictions'/f'{stem}.jsonl', sealed)
    truth = y[split['test']]
    rows = {(row['variant_id'], row['model_seed']): row for row in result['models']}
    with np.load(directory/'artifacts'/stem/'views'/f's{seed}.npz', allow_pickle=False) as arrays:
        votes = arrays['knn_view_predictions']
    return {'n_test': int(len(truth)), 'untrained_accuracy': float(rows[('untrained', seed)]['accuracy']),
            'views7_accuracy': float(rows[('views7', seed)]['accuracy']),
            'reference_accuracy': accuracy(sealed['reference_predictions'][str(seed)], truth),
            'sealed_reference_accuracy': float(sealed['reference_accuracy'][str(seed)]),
            'view_accuracies': [accuracy(view, truth) for view in votes]}


def verify_ablations(runs, sources, *, allow_smoke=False):
    """({fold key: ablation_fold}, {name: record}). Every reference of every run needs exactly one --ablation, whose pins must be
    the ones the diagnostics run recorded (training_diagnostics.load_ablation: pins, design, anchoring on the reference run, views7
    reproduction in its summary), whose directory must pass its family verify, and whose job record of every used fold must pass
    run_knn_ablation.validate_job."""
    used = defaultdict(list)
    for run in runs:
        for name, entry in sorted(run['references'].items()):
            used[name].append((run, entry))
    missing, unknown = sorted(set(used) - set(sources)), sorted(set(sources) - set(used))
    if missing or unknown:
        raise Refusal('--ablation must name every diagnostics reference once (missing: ' + (', '.join(missing) or 'none')
                      + '; not a reference: ' + (', '.join(unknown) or 'none') + ')')
    verified, folds, records = {}, {}, {}
    for name in sorted(used):
        directory = Path(sources[name]).resolve()
        if directory not in verified:
            verified[directory] = family_verify(directory, allow_smoke)
        p, planned, revision, family = verified[directory]
        prepared = {}
        record = {'directory': str(directory), 'family': family, 'protocol_id': p.get('protocol_id'), 'code_revision': revision,
                  'pins': None, 'datasets': [], 'validated_job_records': 0}
        for run, entry in used[name]:
            seed = run['model_seed']
            try:
                ablation = td.load_ablation(directory, entry['ablation'], {'files': entry['run_file_sha256']}, entry['datasets'],
                                            run['protocol'], allow_smoke=allow_smoke)
            except VERIFY_ERRORS as exc:
                raise Refusal(f'--ablation {name}={directory} is not the ablation run that sealed the selections of '
                              f'{run["directory"]}: {type(exc).__name__}: {exc}') from exc
            if seed not in p['fit_seeds']:
                raise Refusal(f'--ablation {name}={directory} holds no fits for the diagnostics seed {seed}')
            for fold in run['folds']:
                if fold['dataset_id'] not in entry['datasets']:
                    continue
                key = (fold['dataset_id'], fold['outer_repeat'], fold['outer_fold'])
                try:
                    folds[key] = ablation_fold(directory, key, planned[key], ablation['selections'][key], run['selections'][key],
                                               p, revision, seed, prepared)
                except VERIFY_ERRORS as exc:
                    raise Refusal(f'--ablation {name}={directory}: the record of {key[0]} r{key[1]}f{key[2]} fails its verification: '
                                  f'{type(exc).__name__}: {exc}') from exc
                record['validated_job_records'] += 1
            record['pins'] = ablation['observed']
            record['datasets'] += list(entry['datasets'])
        records[name] = record
    return folds, records


# ----------------------------------------------------------------------------- table builders (pure; view and fold records)

PER_VIEW_MEASURES = ('knn_test_accuracy', 'output_rule_test_accuracy', 'output_rule_training_accuracy', 'core_validation_error')
TRAJECTORY_MEASURES = ('displacement_from_initial', 'changed_share_from_initial', 'displacement_from_previous',
                       'changed_share_from_previous')


def fold_key(record):
    return record['protocol_id'], record['dataset_id'], record['outer_repeat'], record['outer_fold']


def dataset_groups(records):
    """[(group fields, records)]: one group per dataset in first-seen order, then one pooled group per protocol."""
    by_dataset, by_protocol = {}, {}
    for record in records:
        by_dataset.setdefault((record['protocol_id'], record['dataset_id']), []).append(record)
        by_protocol.setdefault(record['protocol_id'], []).append(record)
    return ([({'scope': 'dataset', 'protocol_id': protocol, 'dataset_id': dataset, 'reference': items[0]['reference']}, items)
             for (protocol, dataset), items in by_dataset.items()]
            + [({'scope': 'pooled', 'protocol_id': protocol}, items) for protocol, items in by_protocol.items()])


def layer_order(item):
    (widths, name, *rest), _ = item
    return (len(widths), widths, (1, 0) if name == 'output' else (0, int(name.split('_', 1)[1])), *rest)


def layer_groups(views, *extra):
    """{(widths, layer, n[, n_filters]): [(view, layer record)]} in layer_order."""
    groups = defaultdict(list)
    for view in views:
        for layer in view['layers'].values():
            groups[(view['widths'], layer['layer'], layer['n'], *(layer[field] for field in extra))].append((view, layer))
    return sorted(groups.items(), key=layer_order)


def t1_rows(views):
    """t1_checkpoints.csv and t1_checkpoint_counts.csv rows."""
    rows, counts = [], []
    for fields, items in dataset_groups(views):
        checkpoints = [view['checkpoint'] for view in items]
        later = [c for c in checkpoints if c > 0]
        median, q25, q75, iqr = quartiles(checkpoints)
        later_median, later_q25, later_q75, later_iqr = quartiles(later)
        iterations, validation = [view['iterations'] for view in items], [view['n_validation_samples'] for view in items]
        rows.append({**fields, 'n_datasets': len({view['dataset_id'] for view in items}), 'n_folds': len({fold_key(v) for v in items}),
                     'n_views': len(items), 'iterations_min': min(iterations), 'iterations_max': max(iterations),
                     'n_validation_samples_min': min(validation), 'n_validation_samples_max': max(validation),
                     'n_views_checkpoint_0': len(items) - len(later), 'share_checkpoint_0': (len(items) - len(later)) / len(items),
                     'checkpoint_median': median, 'checkpoint_q25': q25, 'checkpoint_q75': q75, 'checkpoint_iqr': iqr,
                     'n_views_later_checkpoint': len(later), 'later_checkpoint_median': later_median, 'later_checkpoint_q25': later_q25,
                     'later_checkpoint_q75': later_q75, 'later_checkpoint_iqr': later_iqr})
        tally = defaultdict(int)
        for c in checkpoints:
            tally[c] += 1
        counts += [{**fields, 'n_views_total': len(items), 'iterations_max': max(iterations), 'checkpoint_iteration': c,
                    'n_views': tally[c], 'share_of_views': tally[c] / len(items)} for c in sorted(tally)]
    return rows, counts


def t2_rows(views):
    """t2_displacement.csv: the checkpoint displacement over views with a later checkpoint, beside the checkpoint-0 share."""
    rows = []
    for fields, items in dataset_groups(views):
        for (widths, name, n), members in layer_groups(items):
            later = [layer['at_checkpoint'] for view, layer in members if view['checkpoint'] > 0]
            displacement = [entry['displacement_from_initial'] for entry in later]
            rows.append({**fields, 'n_datasets': len({view['dataset_id'] for view, _ in members}), 'widths': canonical_json(list(widths)),
                         'layer': name, 'n': n, 'n_views': len(members), 'n_views_checkpoint_0': len(members) - len(later),
                         'share_checkpoint_0': (len(members) - len(later)) / len(members), 'n_views_later_checkpoint': len(later),
                         'displacement_mean_later': mean(displacement), 'displacement_sd_later': sd(displacement),
                         'changed_share_mean_later': mean([entry['changed_share_from_initial'] for entry in later]),
                         'random_displacement_reference': random_displacement_reference(n)})
    return rows


def tie_references(views):
    """{(n, W): random_tie_reference} for every hidden layer shape in the views."""
    shapes = sorted({(layer['n'], layer['n_filters']) for view in views for layer in view['layers'].values() if layer['layer'] != 'output'})
    return {shape: random_tie_reference(*shape) for shape in shapes}


def t3_rows(views, references):
    """t3_ties.csv: hidden-layer tie shares by (widths, layer, n, W), initial and checkpoint beside the random-filter reference;
    output-layer nearest-class ties."""
    rows = []
    for fields, items in dataset_groups(views):
        for (widths, name, n, width), members in layer_groups(items, 'n_filters'):
            ties = [layer['ties'] for _, layer in members]
            row = {**fields, 'n_datasets': len({view['dataset_id'] for view, _ in members}), 'widths': canonical_json(list(widths)),
                   'layer': name, 'n': n, 'n_filters': width, 'n_views': len(members)}
            if name == 'output':
                row.update({f'tied_nearest_share_{network}': mean([entry[network]['tied_nearest_share'] for entry in ties])
                            for network in ('initial', 'checkpoint')})
            else:
                bound = distinct_values_bound(n)
                row.update({f'{measure}_{network}': mean([entry[network][measure] for entry in ties])
                            for measure in ('tied_response_share', 'distinct_response_ratio') for network in ('initial', 'checkpoint')})
                row.update(random_tied_response_share=references[(n, width)]['tied_response_share'],
                           random_distinct_response_ratio=references[(n, width)]['distinct_response_ratio'],
                           distinct_values_bound=bound, distinct_response_ratio_bound=min(1.0, bound / width))
            rows.append(row)
    return rows


def relabel_block(fields, level, entries):
    draws = {entry['draws'] for entry in entries}
    if len(draws) != 1:
        raise ValueError('the relabeling draws per unit differ')
    draws = draws.pop()
    return {**fields, 'level': level, 'n_units': len(entries), 'draws_per_unit': draws,
            'checkpoint_accuracy_mean': mean([entry['checkpoint_accuracy'] for entry in entries]),
            'changed_share_mean': mean([entry['changed_share_mean'] for entry in entries]),
            'changed_share_max': max(entry['changed_share_max'] for entry in entries),
            'share_of_draws_with_changes': sum(entry['draws_with_changes'] for entry in entries) / (draws * len(entries)),
            'accuracy_change_mean': mean([entry['accuracy_change_mean'] for entry in entries]),
            'accuracy_change_min': min(entry['accuracy_change_min'] for entry in entries),
            'accuracy_change_max': max(entry['accuracy_change_max'] for entry in entries),
            'abs_accuracy_change_max': max(entry['abs_accuracy_change_max'] for entry in entries)}


def t4_rows(folds):
    """t4_relabel.csv: per dataset, pooled per protocol and pooled by (widths, embed_dim); views and the 7-view majority."""
    rows = []
    for fields, items in dataset_groups(folds):
        fields = {**fields, 'n_datasets': len({fold['dataset_id'] for fold in items})}
        rows.append(relabel_block(fields, 'view', [view['relabel'] for fold in items for view in fold['views']]))
        rows.append(relabel_block(fields, 'seven_view_majority', [fold['relabel'] for fold in items]))
    architectures = defaultdict(list)
    for fold in folds:
        architectures[(fold['protocol_id'], len(fold['widths']), fold['widths'], fold['embed_dim'])].append(fold)
    for (protocol, _, widths, embed_dim), items in sorted(architectures.items()):
        fields = {'scope': 'pooled_by_architecture', 'protocol_id': protocol, 'n_datasets': len({fold['dataset_id'] for fold in items}),
                  'widths': canonical_json(list(widths)), 'embed_dim': embed_dim}
        rows.append(relabel_block(fields, 'view', [view['relabel'] for fold in items for view in fold['views']]))
        rows.append(relabel_block(fields, 'seven_view_majority', [fold['relabel'] for fold in items]))
    return rows


def view_value(view, snapshot, iteration, measure):
    if measure == 'core_validation_error':
        return view['validation'][iteration][0] if snapshot == 'scheduled' else view['checkpoint_validation_error']
    return (view['scheduled'][iteration] if snapshot == 'scheduled' else view['at_checkpoint'])[measure]


def fold_statistics(values):
    return {'n_folds': len(values), 'mean': mean(values), 'sd_over_folds': sd(values), 'min_over_folds': min(values),
            'max_over_folds': max(values)}


def t5_learning_rows(folds):
    """t5_learning_curves.csv: per dataset, one contiguous curve per (level, measure) over the scheduled iterations, then its
    checkpoint point."""
    rows = []
    for fields, items in dataset_groups(folds):
        if fields['scope'] != 'dataset':
            continue
        schedule = sorted({t for fold in items for t in fold['majority_scheduled']})
        curves = [('seven_view_majority', 'knn_test_accuracy')] + [('per_view', measure) for measure in PER_VIEW_MEASURES]
        for level, measure in curves:
            for t in schedule:
                members = [fold for fold in items if t in fold['majority_scheduled']]
                if level == 'seven_view_majority':
                    values, per_fold = [fold['majority_scheduled'][t] for fold in members], 1
                else:
                    values = [mean([view_value(view, 'scheduled', t, measure) for view in fold['views']]) for fold in members]
                    per_fold = len(members[0]['views'])
                point = AFTER if t else (INITIAL_KNN if measure == 'knn_test_accuracy' else INITIAL)
                rows.append({**fields, 'snapshot': 'scheduled', 'iteration': t, 'point': point, 'level': level, 'measure': measure,
                             'views_per_fold': per_fold, **fold_statistics(values)})
            if level == 'seven_view_majority':
                values, per_fold = [fold['majority_checkpoint'] for fold in items], 1
            else:
                values = [mean([view_value(view, 'checkpoint', None, measure) for view in fold['views']]) for fold in items]
                per_fold = len(items[0]['views'])
            rows.append({**fields, 'snapshot': 'checkpoint', 'iteration': None, 'point': AT_CHECKPOINT, 'level': level,
                         'measure': measure, 'views_per_fold': per_fold, **fold_statistics(values)})
    return rows


def t5_validation_rows(folds):
    """t5_validation_curves.csv: the core validation error after every update, per dataset."""
    rows = []
    for fields, items in dataset_groups(folds):
        if fields['scope'] != 'dataset':
            continue
        for t in range(max(view['iterations'] for fold in items for view in fold['views']) + 1):
            members = [fold for fold in items if all(t in view['validation'] for view in fold['views'])]
            errors = [mean([view['validation'][t][0] for view in fold['views']]) for fold in members]
            present = [view['validation'][t] for fold in members for view in fold['views']]
            rows.append({**fields, 'iteration': t, 'n_folds': len(members), 'views_per_fold': len(members[0]['views']),
                         'validation_error_mean': mean(errors), 'validation_error_sd_over_folds': sd(errors),
                         'running_minimum_mean': mean([entry[1] for entry in present]),
                         'improved_share': None if t == 0 else mean([entry[2] for entry in present])})
    return rows


def t5_displacement_rows(views):
    """t5_displacement_curves.csv: the training trajectory's displacement per dataset, layer and scheduled iteration."""
    rows = []
    for fields, items in dataset_groups(views):
        if fields['scope'] != 'dataset':
            continue
        for (widths, name, n), members in layer_groups(items):
            for t in sorted({t for _, layer in members for t in layer['trajectory']}):
                present = [(view, layer['trajectory'][t]) for view, layer in members if t in layer['trajectory']]
                row = {**fields, 'widths': canonical_json(list(widths)), 'layer': name, 'n': n, 'iteration': t,
                       'n_folds': len({fold_key(view) for view, _ in present}), 'n_views': len(present),
                       'random_displacement_reference': random_displacement_reference(n)}
                row.update({f'{measure}_mean': mean([entry[measure] for _, entry in present if measure in entry])
                            for measure in TRAJECTORY_MEASURES})
                rows.append(row)
    return rows


def t6_rows(folds, ablation_folds):
    """t6_trained_untrained.csv and its fold rows; a Refusal when a checkpoint accuracy differs from the reference seed accuracy
    (sealed predictions, sealed record, ablation views7) or a per-view accuracy from the ablation views7 votes."""
    dataset_rows, fold_rows, problems = [], [], []
    for fields, items in dataset_groups(folds):
        if fields['scope'] != 'dataset':
            continue
        trained, untrained, differences = [], [], []
        for fold in items:
            where, ablation = f'{fold["dataset_id"]} r{fold["outer_repeat"]}f{fold["outer_fold"]}', ablation_folds.get(fold_key(fold)[1:])
            if ablation is None:
                problems.append(f'{where}: no verified ablation record')
                continue
            value = fold['majority_checkpoint']
            if any(abs(value - ablation[name]) > TOLERANCE for name in ('reference_accuracy', 'sealed_reference_accuracy', 'views7_accuracy')):
                problems.append(f'{where}: the 7-view kNN accuracy at the checkpoint network ({value!r}) differs from the reference '
                                f'seed accuracy ({ablation["reference_accuracy"]!r})')
            views = [view['at_checkpoint']['knn_test_accuracy'] for view in fold['views']]
            if len(views) != len(ablation['view_accuracies']) or any(abs(a - b) > TOLERANCE for a, b in zip(views, ablation['view_accuracies'])):
                problems.append(f'{where}: a per-view kNN accuracy at the checkpoint differs from the ablation views7 votes')
            difference = value - ablation['untrained_accuracy']
            fold_rows.append({**{k: fold[k] for k in ('protocol_id', 'dataset_id', 'reference', 'outer_repeat', 'outer_fold', 'model_seed',
                                                     'embed_dim')},
                              'widths': fold['widths_text'], 'n_test': ablation['n_test'],
                              'n_views_checkpoint_0': sum(1 for view in fold['views'] if view['checkpoint'] == 0),
                              'trained_accuracy': value, 'reference_accuracy': ablation['reference_accuracy'],
                              'untrained_accuracy': ablation['untrained_accuracy'], 'difference': difference})
            trained.append(value)
            untrained.append(ablation['untrained_accuracy'])
            differences.append(difference)
        if differences:
            dataset_rows.append({**fields, 'model_seed': items[0]['model_seed'], 'n_folds': len(differences),
                                 'trained_accuracy_mean': mean(trained), 'untrained_accuracy_mean': mean(untrained),
                                 'difference_mean': mean(differences), 'difference_sd': sd(differences),
                                 'difference_min': min(differences), 'difference_max': max(differences)})
    if problems:
        raise Refusal('The diagnostics do not describe the reference model: ' + '; '.join(problems[:8]))
    return dataset_rows, fold_rows


# ----------------------------------------------------------------------------- the command

GENERAL = {
    'descriptive': 'Descriptive only: nothing is selected, no test or interval is made, and no value feeds back into any fit, readout, '
                   'checkpoint, dataset, fold or analysis choice.',
    'iteration_0': f'Iteration-0 kNN points are "{INITIAL_KNN}": the initial filters read with the kNN readout setting the fit selected '
                   'on the trained hidden rankings. They are not the untrained ArrowFlow, whose readout setting is selected on its own '
                   'initial hidden rankings. The trained-versus-untrained comparison is t6, from the component ablation untrained '
                   'predictions for the same seed, never checkpoint minus iteration 0.',
    'seed': 'One fitting seed (the diagnostics model_seed, 8129): these accuracies are that seed\'s values, not the registered '
            'three-seed means, and are never put in one table with those without saying so.',
    'checkpoint': 'The checkpoint is the last update whose core validation error (the output rule on the core\'s int(validation_ratio '
                  'x training samples) held-out rows) is strictly below the running minimum before it; 0 means the core returned the '
                  'initial filters. On a checkpoint-0 view the network equals its initial network, so the trained and untrained '
                  'ArrowFlow at one configuration and seed differ only through the views with a later checkpoint.',
    'permutation_length': 'n is the number of items the filters of a layer order: embed_dim for hidden_0, the previous hidden width for '
                          'a later hidden layer, the last hidden width for the output layer. Ties and displacement are reported by n '
                          'because the parity and range of the footrule fix much of both.',
    'displacement_reference': '(n^2 - 1) / 3 is the expected Spearman footrule between two independent uniform permutations of length '
                              'n (Diaconis and Graham 1977) and floor(n^2 / 2) its maximum, so the reference is 2/3 for odd n and '
                              '2 (n^2 - 1) / (3 n^2) for even n (0.6641 at n = 16, 0.6660 at 32, 0.6665 at 64, 0.6666 at 128).',
    'tie_reference': f'Hidden layers: {TIE_REFERENCE["filter_sets"]} sets of W uniform random filters (position vectors of length n), '
                     f'each against {TIE_REFERENCE["rows_per_set"]} uniform random input permutations; cityblock responses '
                     '(training_diagnostics.network_forward) and the tie measures of training_diagnostics.response_ties, averaged '
                     f'over the sets; numpy default_rng seeded by ({TIE_REFERENCE["seed"]}, n, W). Monte Carlo standard errors are in '
                     'random_references.ties.references.',
    'relabel': 'Sensitivity of the kNN readout to filter-ID tie-breaking in the trained network: at inference on the checkpoint '
               'network, the hidden filter IDs are permuted with consistent next-layer relabeling; the output layer is not relabeled.',
    'curves': 'Curves use the scheduled snapshots only; checkpoint rows (empty iteration) are separate points, never part of a curve. '
              'The output-rule training accuracy uses all outer training rows, including the core validation rows.',
    'verification': [
        'diagnostics: provenance.json with every check performed and passed in its totals and in every job; diagnostics_summary.json '
        'present and equal to it on protocol, datasets, planned jobs and check totals',
        'diagnostics: the sha256 of the five CSVs, protocol.json, every job record and every artifact equal to provenance.json',
        'diagnostics: training_diagnostics.verify (a frozen protocol unless a declared synthetic smoke, the protocol and plan seals, '
        'unchanged sealed sources)',
        'diagnostics: the joined tables hold seven views per planned fold with one schedule and every snapshot, validation, tie and '
        'relabel row, and every checkpoint-0 view has zero checkpoint displacement',
        'ablations: training_diagnostics.load_ablation against the pins the diagnostics run recorded (protocol, manifest, sealed '
        'selections and summary sha256; design; anchoring on the reference run; views7 reproducing every fold and seed)',
        'ablations: the family verify (run_knn_ablation or run_newdata_ablation, sealed sources) and run_knn_ablation.validate_job '
        'for every used fold, whose sealed selection must equal the diagnostics run\'s',
        't6: on every fold the 7-view checkpoint accuracy equals the accuracy of the sealed reference predictions, the sealed reference '
        'accuracy and the ablation views7 accuracy for the seed, and every per-view checkpoint accuracy equals that of the ablation '
        'views7 votes (tolerance 1e-12)']}


def write_outputs(output, contents):
    """All or none (training_diagnostics.write_all): refuse before writing anything when an existing file differs."""
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise Refusal(f'{output} is not a directory')
    differing = [name for name, text in contents.items() if (output/name).exists() and (output/name).read_bytes() != text.encode('utf-8')]
    if differing:
        raise Refusal(f'Refusing to replace {", ".join(differing)} in {output}; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    try:
        td.write_all(output, contents)
    except FileExistsError as exc:
        raise Refusal(str(exc)) from exc


def tables(diagnostics, ablations, output, *, allow_smoke=False):
    """Verify every input, build every table, then write all outputs or none; returns the diagnostics_tables.json document."""
    runs = [load_run(directory, allow_smoke=allow_smoke) for directory in diagnostics]
    if len({run['directory'] for run in runs}) != len(runs):
        raise Refusal('A diagnostics directory is given twice')
    datasets = [name for run in runs for name in run['datasets']]
    if len(set(datasets)) != len(datasets):
        raise Refusal('A dataset belongs to two diagnostics runs')
    ablation_folds, ablation_records = verify_ablations(runs, dict(ablations), allow_smoke=allow_smoke)
    folds = [fold for run in runs for fold in run['folds']]
    views = [view for run in runs for view in run['views']]
    t6, t6_folds = t6_rows(folds, ablation_folds)
    try:
        t1, counts = t1_rows(views)
        references = tie_references(views)
        rows = {'t1_checkpoints.csv': t1, 't1_checkpoint_counts.csv': counts, 't2_displacement.csv': t2_rows(views),
                't3_ties.csv': t3_rows(views, references), 't4_relabel.csv': t4_rows(folds),
                't5_learning_curves.csv': t5_learning_rows(folds), 't5_validation_curves.csv': t5_validation_rows(folds),
                't5_displacement_curves.csv': t5_displacement_rows(views), 't6_trained_untrained.csv': t6,
                't6_trained_untrained_folds.csv': t6_folds}
        contents = {name: render_csv(name, rows[name]) for name in FILES}
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        raise Refusal(f'The tables cannot be built from the verified inputs: {type(exc).__name__}: {exc}') from exc
    smoke = any(run['smoke'] for run in runs)
    lengths = sorted({layer['n'] for view in views for layer in view['layers'].values()})
    document = {
        'purpose': 'report-ready tables and figure data of the ArrowFlow-kNN training diagnostics; post-processing of frozen outputs '
                   'only: no model is fitted and nothing is selected',
        'synthetic_smoke_only': smoke,
        'allow_smoke': bool(allow_smoke),
        'evidence': 'never evidence: synthetic smoke outputs' if smoke else 'descriptive tables of the verified frozen runs in inputs',
        'module': {'path': 'experiments/make_revision/diagnostics_tables.py', 'sha256': sha256_path(__file__)},
        'general': GENERAL,
        'tables': {name: {'rows': len(rows[name]), 'sha256': sha256_bytes(contents[name].encode('utf-8')),
                          'columns': [{'column': column, 'definition': definition} for column, definition in COLUMNS[name]]}
                   for name in FILES},
        'random_references': {
            'displacement': {'formula': '(n^2 - 1) / 3 / floor(n^2 / 2)',
                             'by_n': {str(n): random_displacement_reference(n) for n in lengths if n >= 2}},
            'ties': {'method': GENERAL['tie_reference'], **TIE_REFERENCE, 'distinct_values_bound': 'floor(n^2 / 4) + 1',
                     'references': list(references.values())}},
        'checks': {'diagnostics_runs': len(runs),
                   'ablation_job_records_validated': sum(record['validated_job_records'] for record in ablation_records.values()),
                   'folds_with_checkpoint_accuracy_equal_to_the_reference': len(t6_folds),
                   'views_with_checkpoint_accuracy_equal_to_the_ablation_votes': sum(len(fold['views']) for fold in folds),
                   'checkpoint_zero_views_with_zero_checkpoint_displacement': sum(1 for view in views if view['checkpoint'] == 0),
                   'tolerance': TOLERANCE},
        'inputs': {'diagnostics': [run['record'] for run in runs], 'ablations': ablation_records}}
    contents[OUTPUT_JSON] = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n'
    write_outputs(output, contents)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['tables'])
    parser.add_argument('--diagnostics', type=Path, action='append', required=True, metavar='DIAG_DIR')
    parser.add_argument('--ablation', type=parse_ablation, action='append', required=True, metavar='NAME=ABLATION_DIR')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true', help='admit synthetic smoke runs (the tables are never evidence)')
    args = parser.parse_args(argv)
    try:
        names = [name for name, _ in args.ablation]
        if len(set(names)) != len(names):
            raise Refusal('--ablation names a reference twice')
        document = tables(args.diagnostics, dict(args.ablation), args.output, allow_smoke=args.allow_smoke)
    except Refusal as exc:
        print(f'REFUSED (nothing written): {exc}', file=sys.stderr)
        return 2
    for name, entry in document['tables'].items():
        print(f'{args.output/name}: {entry["rows"]} rows, sha256 {entry["sha256"]}')
    print(f'{args.output/OUTPUT_JSON}' + (' (synthetic smoke; never evidence)' if document['synthetic_smoke_only'] else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
