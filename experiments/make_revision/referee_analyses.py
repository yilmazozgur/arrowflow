"""Task 22: descriptive analyses a simulated referee asked for (P6 M8a and M8b), from frozen run outputs only.

No model is fitted. Every input run is loaded and re-verified with compare_runs.load_run and compare_runs.verify_run
(the harness validators: prepared data and splits hashes, the declared nested splits, planned jobs and fit logs, every
result record with its per-example predictions, and summary.json against the re-verified model rows) before anything is
computed. Outputs are written all or none, and a file with different content is never replaced
(compare_runs.write_outputs). Refusals exit with status 2.

python -m experiments.make_revision.referee_analyses comparators --knn-source K --output O
  Per dataset and comparator of the bridge_knn run: arrowflow_full_knn minus the comparator accuracy, fitting seeds
  averaged within each outer fold, corrected resampled t over the 15 outer folds (evaluation.paired_corrected_interval,
  q = 0.25, 95%, df = 14) with its unadjusted p. The comparator with the lowest mean outer error on each dataset is
  flagged, for display only.
  Outputs: comparator_contrasts.csv, comparator_contrasts.json

python -m experiments.make_revision.referee_analyses duplicates --knn-source K --bridge-source B --output O
  Exact-duplicate sensitivity. Per dataset: groups of rows with identical raw feature vectors (the prepared data.npz
  every fit read); for every outer fold, the test rows with an exact feature duplicate in that fold's training
  partition; seed-averaged outer-fold accuracy of arrowflow_full_knn and every comparator (knn run) and of
  arrowflow_full (bridge run) on all test rows, which must reproduce the recorded accuracies exactly, and on the test
  rows without a training duplicate; arrowflow_full_knn minus arrowflow_full on both row sets, with corrected resampled
  t intervals, paired only on identical test rows.
  Outputs: duplicate_groups.csv, duplicate_folds.csv, duplicate_accuracy.csv, duplicate_readout.csv,
  duplicate_sensitivity.json

python -m experiments.make_revision.referee_analyses ranks --knn-source K --output O
  Friedman test over the dataset x model matrix of mean outer error of arrowflow_full_knn and the five tuned
  comparators (the majority class excluded): mean ranks (ties share their average rank), the Friedman statistic with
  its p value, and the Nemenyi critical difference at alpha = 0.05.
  Outputs: rank_matrix.csv, mean_ranks.csv, friedman_nemenyi.json

Labels: comparators and duplicates 'descriptive; not a registered family; no multiplicity adjustment'; ranks
'descriptive'. Every JSON record carries the input runs' provenance (summary.json and protocol sha256, code revision),
the analysis code revision with the sha256 of its sources, the definitions and the sha256 of each CSV.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import zipfile
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from .compare_runs import (FULL_MODEL, KNN_MODEL, VALIDATORS, RunComparisonError, _csv_text, _json_text,
                           analysis_sources, check_output, check_pairing, check_reference, load_run, verify_run,
                           write_outputs)
from .evaluation import canonical_json, paired_corrected_interval, summarize_outer
from .run_revision import DATASETS, code_revision, load_prepared

STATUS = 'descriptive; not a registered family; no multiplicity adjustment'
RANK_STATUS = 'descriptive'
METRIC = 'accuracy'
TEST_TRAIN_RATIO, CONFIDENCE, OUTER_FOLDS = .25, .95, 15
ANALYSIS_SOURCES = ('referee_analyses.py', 'compare_runs.py', 'evaluation.py', 'reporting.py', 'run_revision.py')
INTERVAL_FIELDS = ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'p_unadjusted', 'n_folds', 'df')
REPO = Path(__file__).resolve().parents[2]


class ReproductionError(RunComparisonError):
    """An accuracy recomputed on all test rows differs from the accuracy a run recorded."""


def code_record(names=ANALYSIS_SOURCES):
    """The analysis code revision, whether its sources are committed unchanged at it, and their sha256."""
    paths = [f'experiments/make_revision/{name}' for name in names]
    try:
        changed = subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], cwd=REPO, text=True)
        revision = code_revision()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunComparisonError(f'The analysis code revision cannot be read ({exc})') from exc
    return {'code_revision': revision, 'sources_committed_at_revision': not changed.strip(),
            'analysis_sources': analysis_sources(names)}


def check_design(run):
    """These analyses are defined for the registered nested design: q = 0.25, 95% intervals, 15 outer folds."""
    declared = (run.protocol.get('test_train_ratio'), run.protocol.get('confidence'), len(run.schedule['expected_folds']))
    if declared != (TEST_TRAIN_RATIO, CONFIDENCE, OUTER_FOLDS):
        raise RunComparisonError(f'The {run.label} run declares test_train_ratio, confidence and outer fold count '
                                 f'{declared}; these analyses are defined for {(TEST_TRAIN_RATIO, CONFIDENCE, OUTER_FOLDS)}')


def published_mean(run, name, model, metric):
    """summary.json summaries mean (load_run recomputed it from the model rows that verify_run re-verifies)."""
    return next(row['mean'] for row in run.summary['summaries'][name] if (row['model_id'], row['metric']) == (model, metric))


def interval_fields(interval):
    """The CSV/JSON fields of one evaluation.paired_corrected_interval result; its p is unadjusted."""
    return {**{field: interval[field] for field in INTERVAL_FIELDS if field != 'p_unadjusted'},
            'p_unadjusted': interval['p_approximate']}


def verification_record(*verified):
    return {'validators': list(VALIDATORS),
            **{run.label: {'jobs_verified': len(run.jobs), 'model_rows_verified': sum(map(len, rows.values()))}
               for run, rows in verified}}


def finish(output, tables, record_name, record):
    """Seal each CSV's sha256 in the JSON record, then write every output, all or none."""
    contents = {name: _csv_text(rows, columns) for name, (rows, columns) in tables.items()}
    record['outputs'] = {name: {'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'rows': len(tables[name][0])}
                         for name, text in contents.items()}
    contents[record_name] = _json_text(record)
    write_outputs(output, contents)


# ----------------------------------------------------------------------------- A: ArrowFlow minus each comparator

COMPARATOR_CSV, COMPARATOR_JSON = 'comparator_contrasts.csv', 'comparator_contrasts.json'
COMPARATOR_COLUMNS = ('status', 'dataset', 'model_a', 'model_b', *INTERVAL_FIELDS, 'mean_error_a', 'mean_error_b',
                      'best_comparator')
COMPARATOR_DEFINITIONS = {
    'mean_difference': f'{KNN_MODEL} minus the comparator accuracy, both from the knn run: fitting seeds averaged within '
                       'each outer fold, then the mean over the outer folds; positive favours ArrowFlow',
    'interval': 'evaluation.paired_corrected_interval on the re-verified model rows: standard error '
                'sqrt((1/n_folds + q) * variance (ddof 1) of the fold differences), q = test_train_ratio = 0.25, '
                't quantile with df = n_folds - 1 = 14, 95%',
    'p_unadjusted': 'two-sided corrected resampled t p over the outer folds (approximate); no multiplicity adjustment',
    'mean_error_a, mean_error_b': 'mean outer error of each model: the summary.json summaries mean, i.e. the mean over the '
                                  'outer folds of the fitting-seed-averaged error (the values of the main benchmark table)',
    'best_comparator': 'display only: the comparator with the lowest mean outer error on the dataset (every comparator '
                       'tied at that minimum is flagged)',
    'comparators': f'every model family of the knn run other than {KNN_MODEL}, the majority class included',
    'status': STATUS,
}


def comparator_contrasts(knn, rows):
    """Per dataset and comparator: arrowflow_full_knn minus the comparator (knn run), with the best comparator flagged."""
    comparators = [model for model in knn.models if model != KNN_MODEL]
    contrasts, best = [], {}
    for name in knn.protocol['datasets']:
        errors = {model: published_mean(knn, name, model, 'error') for model in knn.models}
        lowest = min(errors[model] for model in comparators)
        best[name] = [model for model in comparators if errors[model] == lowest]
        for model in comparators:
            seeds = {KNN_MODEL: knn.schedule['expected_seeds'][KNN_MODEL], model: knn.schedule['expected_seeds'][model]}
            interval = paired_corrected_interval(rows[name], KNN_MODEL, model, metric=METRIC, q=TEST_TRAIN_RATIO,
                                                 confidence=CONFIDENCE, expected_folds=knn.schedule['expected_folds'],
                                                 expected_seeds=seeds)
            contrasts.append({'status': STATUS, 'dataset': name, 'model_a': KNN_MODEL, 'model_b': model,
                              **interval_fields(interval), 'mean_error_a': errors[KNN_MODEL],
                              'mean_error_b': errors[model], 'best_comparator': model in best[name]})
    return contrasts, best


def analyse_comparators(knn_source, output):
    check_output(output, (COMPARATOR_CSV, COMPARATOR_JSON))     # an unusable output location is refused first
    knn = load_run(knn_source, 'knn')
    if KNN_MODEL not in knn.registry or len(knn.models) < 2:
        raise RunComparisonError(f'The knn run must hold {KNN_MODEL} and at least one comparator')
    check_design(knn)
    rows, _ = verify_run(knn)
    contrasts, best = comparator_contrasts(knn, rows)
    record = {'status': STATUS, 'purpose': 'referee_M8a_arrowflow_full_knn_minus_each_comparator_paired_intervals',
              'metric': METRIC, 'model_a': KNN_MODEL, 'run': knn.label,
              'comparators': [model for model in knn.models if model != KNN_MODEL],
              'datasets': list(knn.protocol['datasets']), 'test_train_ratio': TEST_TRAIN_RATIO, 'confidence': CONFIDENCE,
              'n_folds': OUTER_FOLDS, 'df': OUTER_FOLDS - 1, 'fitting_seeds': list(knn.protocol['fit_seeds']),
              'definitions': dict(COMPARATOR_DEFINITIONS), 'contrasts': contrasts, 'best_comparator': best,
              'provenance': {'knn': knn.provenance(), 'verification': verification_record((knn, rows)), **code_record()}}
    finish(output, {COMPARATOR_CSV: (contrasts, COMPARATOR_COLUMNS)}, COMPARATOR_JSON, record)
    return record


# ----------------------------------------------------------------------------- B: exact-duplicate sensitivity

def duplicate_groups(X, y):
    """Rows with identical raw feature vectors: exact float64 equality of every feature (-0.0 equals 0.0; NaN refused).

    Returns (per row, the index of its distinct feature vector in first-occurrence order; the counts and the groups,
    i.e. the vectors held by at least two rows, listed by their first row with the sample IDs and labels of every copy).
    """
    X, y = np.asarray(X, dtype='<f8'), np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError('Expected a 2-D feature matrix with one label per row')
    if np.isnan(X).any():
        raise ValueError('NaN features: exact feature identity is undefined')
    keys = {}
    vector = np.array([keys.setdefault(row.tobytes(), len(keys)) for row in np.ascontiguousarray(X + 0.)], dtype=np.int64)
    members = {}
    for sample, key in enumerate(vector.tolist()):
        members.setdefault(key, []).append(sample)
    groups = [{'sample_ids': ids, 'labels': y[ids].tolist()} for ids in members.values() if len(ids) > 1]
    sizes = Counter(len(group['sample_ids']) for group in groups)
    return vector, {'n_rows': len(X), 'n_distinct_rows': len(members), 'duplicate_rows': len(X) - len(members),
                    'rows_in_duplicate_groups': sum(size * count for size, count in sizes.items()),
                    'duplicate_groups': len(groups),
                    'label_conflicting_groups': sum(len(set(group['labels'])) > 1 for group in groups),
                    'largest_group': max(sizes, default=1),
                    'group_sizes': {str(size): sizes[size] for size in sorted(sizes)}, 'groups': groups}


def training_duplicate_flags(vector, train, test):
    """Per test row, in test order: an exact feature duplicate of it lies in the training partition."""
    in_training = set(np.asarray(vector)[np.asarray(train, dtype=np.int64)].tolist())
    return [int(vector[sample]) in in_training for sample in test]


def split_folds(flags, folds):
    """(folds keeping at least one test row without a training duplicate, folds keeping none)."""
    remaining = [tuple(fold) for fold in folds if not all(flags[tuple(fold)])]
    return remaining, [list(fold) for fold in folds if all(flags[tuple(fold)])]


def cell_accuracies(run, name, cells, y, splits, flags):
    """Accuracy of every recorded (model, repeat, fold, seed) cell of one dataset on all test rows and on the test rows
    without a training duplicate, from the verified per-example predictions.

    Hard check: every recorded model row has verified predictions on exactly the saved split's test rows, records that
    split's training rows as its fit rows, and its all-rows accuracy equals the accuracy summary.json model_rows records,
    with exact float equality. Anything else raises ReproductionError.
    """
    y, outer_folds, result = np.asarray(y), run.protocol['outer_folds'], {}
    for row in run.summary['model_rows'][name]:
        key = (row['model_id'], row['outer_repeat'], row['outer_fold'], row['model_seed'])
        where = f'{run.label} run {name} {key[0]} r{key[1]}f{key[2]} seed {key[3]}'
        if (name, *key) not in cells or key in result:
            raise ReproductionError(f'{where}: no verified predictions, or a repeated model row')
        split = splits[key[1] * outer_folds + key[2]]
        _, test, labels = cells[(name, *key)]
        if list(test) != split['test'] or row['test_rows'] != split['test'] or row['fit_rows'] != split['train']:
            raise ReproductionError(f'{where}: the predictions or the recorded partition differ from the saved split')
        truth, predicted = y[list(test)], np.asarray(labels)
        accuracy = float(accuracy_score(truth, predicted))
        if accuracy != row[METRIC]:
            raise ReproductionError(f'{where}: accuracy on all test rows {accuracy!r} differs from the recorded '
                                    f'{row[METRIC]!r}')
        keep = ~np.asarray(flags[key[1], key[2]], dtype=bool)
        result[key] = {'all': accuracy,
                       'restricted': float(accuracy_score(truth[keep], predicted[keep])) if keep.any() else None,
                       'rows': {'all': tuple(test), 'restricted': tuple(np.asarray(test)[keep].tolist())}}
    if {cell[1:] for cell in cells if cell[0] == name} != set(result):
        raise ReproductionError(f'{run.label} run {name}: the verified prediction cells and the recorded model rows differ')
    return result


def row_set_rows(run, name, model, accuracies, row_set, folds):
    """Model rows carrying the accuracy on one row set, in summary.json model_rows order, for the given folds."""
    keep = {tuple(fold) for fold in folds}
    return [{'dataset_id': name, 'model_id': model, 'outer_repeat': row['outer_repeat'], 'outer_fold': row['outer_fold'],
             'model_seed': row['model_seed'], 'status': 'ok',
             METRIC: accuracies[model, row['outer_repeat'], row['outer_fold'], row['model_seed']][row_set]}
            for row in run.summary['model_rows'][name]
            if row['model_id'] == model and (row['outer_repeat'], row['outer_fold']) in keep]


def row_set_summary(run, name, model, accuracies, row_set, folds):
    """evaluation.summarize_outer on one row set over the given folds (seed-averaged within fold); None without folds."""
    if not folds:
        return None
    return summarize_outer(row_set_rows(run, name, model, accuracies, row_set, folds), model, METRIC,
                           expected_folds=list(folds), expected_seeds=run.schedule['expected_seeds'][model])


def reproduced_summary(run, name, model, accuracies):
    """Hard check: the all-rows mean accuracy equals the summary.json summaries mean, exactly."""
    summary = row_set_summary(run, name, model, accuracies, 'all', run.schedule['expected_folds'])
    recorded = published_mean(run, name, model, METRIC)
    if summary['mean'] != recorded:
        raise ReproductionError(f'{run.label} run {name} {model}: mean accuracy on all test rows {summary["mean"]!r} '
                                f'differs from the recorded {recorded!r}')
    return summary, recorded


def paired_readout(knn, bridge, name, knn_accuracies, bridge_accuracies, row_set, folds):
    """arrowflow_full_knn (knn run) minus arrowflow_full (bridge run) on one row set over the given folds.

    Only identical row sets are paired: in every fold and fitting seed both readouts must have been scored on the same
    test rows. With fewer than two folds the mean difference (if any fold remains) is given without an interval.
    """
    seeds = knn.schedule['expected_seeds'][KNN_MODEL]
    if bridge.schedule['expected_seeds'][FULL_MODEL] != seeds:
        raise RunComparisonError(f'{KNN_MODEL} and {FULL_MODEL} fitting seed schedules differ')
    for repeat, fold in folds:
        for seed in seeds:
            if (knn_accuracies[KNN_MODEL, repeat, fold, seed]['rows'][row_set]
                    != bridge_accuracies[FULL_MODEL, repeat, fold, seed]['rows'][row_set]):
                raise RunComparisonError(f'{name} r{repeat}f{fold} seed {seed}: the two readouts were scored on different '
                                         f'{row_set} test rows; only identical row sets are paired')
    if len(folds) >= 2:
        rows = (row_set_rows(knn, name, KNN_MODEL, knn_accuracies, row_set, folds)
                + row_set_rows(bridge, name, FULL_MODEL, bridge_accuracies, row_set, folds))
        return interval_fields(paired_corrected_interval(rows, KNN_MODEL, FULL_MODEL, metric=METRIC, q=TEST_TRAIN_RATIO,
                                                         confidence=CONFIDENCE, expected_folds=list(folds),
                                                         expected_seeds={KNN_MODEL: seeds, FULL_MODEL: seeds}))
    differences = [np.mean([knn_accuracies[KNN_MODEL, repeat, fold, seed][row_set] for seed in seeds])
                   - np.mean([bridge_accuracies[FULL_MODEL, repeat, fold, seed][row_set] for seed in seeds])
                   for repeat, fold in folds]
    return {**dict.fromkeys(INTERVAL_FIELDS), 'mean_difference': float(np.mean(differences)) if differences else None,
            'n_folds': len(folds)}


DUPLICATE_GROUPS_CSV, DUPLICATE_FOLDS_CSV = 'duplicate_groups.csv', 'duplicate_folds.csv'
DUPLICATE_ACCURACY_CSV, DUPLICATE_READOUT_CSV = 'duplicate_accuracy.csv', 'duplicate_readout.csv'
DUPLICATE_JSON = 'duplicate_sensitivity.json'
DUPLICATE_OUTPUTS = (DUPLICATE_GROUPS_CSV, DUPLICATE_FOLDS_CSV, DUPLICATE_ACCURACY_CSV, DUPLICATE_READOUT_CSV, DUPLICATE_JSON)
COUNT_FIELDS = ('n_rows', 'n_distinct_rows', 'duplicate_rows', 'rows_in_duplicate_groups', 'duplicate_groups',
                'label_conflicting_groups', 'largest_group')
GROUP_COLUMNS = ('status', 'dataset', *COUNT_FIELDS, 'test_rows', 'test_rows_with_training_duplicate',
                 'share_with_training_duplicate', 'fold_share_min', 'fold_share_max', 'folds_without_remaining_rows')
FOLD_COLUMNS = ('status', 'dataset', 'outer_repeat', 'outer_fold', 'n_test', 'n_with_training_duplicate',
                'n_without_training_duplicate')
ACCURACY_COLUMNS = ('status', 'dataset', 'model_id', 'source_run', 'accuracy_all_test_rows', 'recorded_accuracy',
                    'reproduced_exactly', 'accuracy_without_training_duplicates', 'change', 'n_folds_all',
                    'n_folds_without', 'seeds_per_fold')
READOUT_COLUMNS = ('status', 'dataset', 'row_set', 'model_a', 'run_a', 'model_b', 'run_b', *INTERVAL_FIELDS,
                   'folds_without_remaining_rows')
ROW_SETS = (('all_test_rows', 'all'), ('test_rows_without_training_duplicate', 'restricted'))
DUPLICATE_DEFINITIONS = {
    'raw_features': 'X of each run\'s prepared data.npz: the matrix run_revision.load_dataset returned from the pinned '
                    'loaders (run_revision.DATASETS), before any encoding, which every fit read through '
                    'run_revision.load_prepared; checked against the manifest dataset_hash, identical in both runs, and '
                    'for the pinned datasets of the shape and class counts run_revision.DATASETS declares',
    'exact_duplicate': 'two rows whose every feature is equal as float64 (-0.0 equals 0.0; NaN features are refused)',
    'n_distinct_rows': 'number of distinct feature vectors',
    'duplicate_rows': 'n_rows minus n_distinct_rows: the copies beyond the first of every duplicated feature vector',
    'rows_in_duplicate_groups': 'rows whose feature vector occurs at least twice, every copy counted',
    'duplicate_groups': 'distinct feature vectors that occur at least twice',
    'label_conflicting_groups': 'duplicate groups whose rows carry more than one label (the labels the harness fitted)',
    'training_duplicate': 'a test row of an outer fold has a training duplicate when an exact feature duplicate of it lies '
                          'in that fold\'s training partition (the train rows of the saved splits.json). The partition '
                          'is the same for every fitting seed: every recorded model row of the fold carries fit_rows and '
                          'test_rows equal to the saved split, which is checked',
    'share_with_training_duplicate': 'test rows with a training duplicate over all test rows, pooled over the 15 outer '
                                     'folds (every row is a test row once per repeat); fold_share_min and fold_share_max '
                                     'give the range over the folds',
    'accuracy_all_test_rows': 'accuracy_score on all test rows of every recorded outer fold and fitting seed, from the '
                              'verified per-example predictions; fitting seeds averaged within each outer fold, then the '
                              'mean over the outer folds (evaluation.summarize_outer)',
    'reproduced_exactly': 'hard check, for every model of both runs: in every outer fold and fitting seed the all-rows '
                          'accuracy equals the accuracy summary.json model_rows records, and the mean equals the '
                          'summary.json summaries mean, with exact float equality; any difference refuses the analysis',
    'accuracy_without_training_duplicates': 'the same on the test rows without a training duplicate, over the outer folds '
                                            'that keep at least one such row',
    'change': 'accuracy_without_training_duplicates minus accuracy_all_test_rows',
    'folds_without_remaining_rows': 'outer folds all of whose test rows have a training duplicate. The row sets depend '
                                    'only on the split and the features, never on the model, so such a fold is dropped '
                                    'from the restricted row set of every model: restricted means and intervals use the '
                                    'remaining folds (df = remaining folds - 1), and with fewer than two remaining folds '
                                    'the mean difference is given without an interval',
    'readout_difference': f'{KNN_MODEL} (knn run) minus {FULL_MODEL} (bridge run) accuracy: fitting seeds averaged within '
                          'each outer fold, paired fold by fold, and only where both readouts were scored on the '
                          'identical test rows (checked for every fold and fitting seed); evaluation.'
                          'paired_corrected_interval with q = 0.25, 95%, df = n_folds - 1, unadjusted p. On all test rows '
                          'it equals the registered knn-versus-full contrast before that family\'s Holm adjustment',
    'q_on_restricted_rows': 'q stays at the protocol value 0.25 because the training partitions are unchanged; the '
                            'restricted test sets are smaller, so the actual ratio of test to training rows is lower and '
                            'this interval is wider than one computed with that ratio',
    'status': STATUS,
}


def prepared_features(knn, bridge, name):
    """X, y, manifest and splits of one dataset: identical in both runs and of the identity run_revision.DATASETS pins."""
    try:
        X, y, manifest, splits = load_prepared(knn.path, name)
        other_X, other_y, _, other_splits = load_prepared(bridge.path, name)
    except (OSError, EOFError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise RunComparisonError(f'The prepared data of {name} are unreadable or do not match the manifest ({exc})') from exc
    if not (np.array_equal(X, other_X) and np.array_equal(y, other_y)
            and canonical_json(splits) == canonical_json(other_splits)):
        raise RunComparisonError(f'The prepared features, labels or splits of {name} differ between the runs')
    pinned = DATASETS.get(name)
    if pinned is not None and (list(X.shape) != list(pinned[1]) or np.bincount(y).tolist() != list(pinned[2])):
        raise RunComparisonError(f'The prepared data of {name} lack the shape and class counts run_revision.DATASETS pins')
    return X, y, manifest, splits, pinned is not None


def dataset_sensitivity(knn, bridge, name, knn_cells, bridge_cells):
    """(groups CSV row, fold CSV rows, accuracy CSV rows, readout CSV rows, JSON record) of one dataset."""
    X, y, manifest, splits, pinned = prepared_features(knn, bridge, name)
    try:
        vector, counts = duplicate_groups(X, y)
    except ValueError as exc:
        raise RunComparisonError(f'{name}: {exc}') from exc
    folds, outer_folds = [tuple(fold) for fold in knn.schedule['expected_folds']], knn.protocol['outer_folds']
    flags, fold_rows = {}, []
    for repeat, fold in folds:
        split = splits[repeat * outer_folds + fold]
        flags[repeat, fold] = training_duplicate_flags(vector, split['train'], split['test'])
        flagged = [sample for sample, flag in zip(split['test'], flags[repeat, fold]) if flag]
        fold_rows.append({'status': STATUS, 'dataset': name, 'outer_repeat': repeat, 'outer_fold': fold,
                          'n_test': len(split['test']), 'n_with_training_duplicate': len(flagged),
                          'n_without_training_duplicate': len(split['test']) - len(flagged),
                          'test_rows_with_training_duplicate': flagged})
    remaining, empty = split_folds(flags, folds)
    accuracies = {'knn': cell_accuracies(knn, name, knn_cells, y, splits, flags),
                  'bridge': cell_accuracies(bridge, name, bridge_cells, y, splits, flags)}
    summaries = {}
    for run in (knn, bridge):                      # the hard check covers every model of both runs
        for model in run.models:
            full, recorded = reproduced_summary(run, name, model, accuracies[run.label])
            summaries[run.label, model] = full, recorded, row_set_summary(run, name, model, accuracies[run.label],
                                                                          'restricted', remaining)
    table = []
    for run, model in [(knn, KNN_MODEL), (bridge, FULL_MODEL), *((knn, model) for model in knn.models if model != KNN_MODEL)]:
        full, recorded, restricted = summaries[run.label, model]
        table.append({'status': STATUS, 'dataset': name, 'model_id': model, 'source_run': run.label,
                      'accuracy_all_test_rows': full['mean'], 'recorded_accuracy': recorded,
                      'reproduced_exactly': full['mean'] == recorded,
                      'accuracy_without_training_duplicates': restricted['mean'] if restricted else None,
                      'change': restricted['mean'] - full['mean'] if restricted else None, 'n_folds_all': full['n_folds'],
                      'n_folds_without': restricted['n_folds'] if restricted else 0, 'seeds_per_fold': full['seeds_per_fold']})
    readout = [{'status': STATUS, 'dataset': name, 'row_set': row_set, 'model_a': KNN_MODEL, 'run_a': knn.label,
                'model_b': FULL_MODEL, 'run_b': bridge.label,
                **paired_readout(knn, bridge, name, accuracies['knn'], accuracies['bridge'], key,
                                 folds if key == 'all' else remaining),
                'folds_without_remaining_rows': 0 if key == 'all' else len(empty)} for row_set, key in ROW_SETS]
    n_test, n_flagged = sum(row['n_test'] for row in fold_rows), sum(row['n_with_training_duplicate'] for row in fold_rows)
    shares = [row['n_with_training_duplicate'] / row['n_test'] for row in fold_rows]
    group_row = {'status': STATUS, 'dataset': name, **{field: counts[field] for field in COUNT_FIELDS}, 'test_rows': n_test,
                 'test_rows_with_training_duplicate': n_flagged, 'share_with_training_duplicate': n_flagged / n_test,
                 'fold_share_min': min(shares), 'fold_share_max': max(shares), 'folds_without_remaining_rows': len(empty)}
    record = {'dataset': name, 'source': manifest.get('source'), 'dataset_hash': manifest['dataset_hash'],
              'splits_hash': manifest['splits_hash'], 'pinned_shape_and_class_counts_checked': pinned,
              'counts': {field: counts[field] for field in COUNT_FIELDS}, 'group_sizes': counts['group_sizes'],
              'groups': counts['groups'],
              'test_rows': {field: group_row[field] for field in GROUP_COLUMNS[len(COUNT_FIELDS) + 2:]},
              'folds_without_remaining_rows': empty, 'folds': fold_rows, 'accuracy': table, 'readout': readout,
              'cells_reproduced': {label: len(cells) for label, cells in accuracies.items()},
              'model_means_reproduced': len(summaries)}
    return group_row, fold_rows, table, readout, record


def analyse_duplicates(knn_source, bridge_source, output):
    check_output(output, DUPLICATE_OUTPUTS)         # an unusable output location is refused first
    knn, bridge = load_run(knn_source, 'knn'), load_run(bridge_source, 'bridge')
    reference = check_reference(knn.protocol, bridge)
    pairing = check_pairing(knn, bridge)
    for run in (knn, bridge):
        check_design(run)
    knn_rows, knn_cells = verify_run(knn)
    bridge_rows, bridge_cells = verify_run(bridge)
    parts = [dataset_sensitivity(knn, bridge, name, knn_cells, bridge_cells) for name in knn.protocol['datasets']]
    groups, datasets = [part[0] for part in parts], [part[4] for part in parts]
    folds, accuracy, readout = ([row for part in parts for row in part[index]] for index in (1, 2, 3))
    record = {'status': STATUS, 'purpose': 'referee_M8b_exact_duplicate_sensitivity_from_saved_predictions',
              'metric': METRIC, 'models': {'knn': list(knn.models), 'bridge': [FULL_MODEL]},
              'readout_difference': f'{KNN_MODEL} (knn run) minus {FULL_MODEL} (bridge run)',
              'row_sets': [row_set for row_set, _ in ROW_SETS], 'test_train_ratio': TEST_TRAIN_RATIO,
              'confidence': CONFIDENCE, 'definitions': dict(DUPLICATE_DEFINITIONS),
              'reproduction': {'all_rows_reproduced_exactly': True,
                               'cells_checked': sum(sum(entry['cells_reproduced'].values()) for entry in datasets),
                               'model_means_checked': sum(entry['model_means_reproduced'] for entry in datasets),
                               'criterion': DUPLICATE_DEFINITIONS['reproduced_exactly']},
              'datasets': datasets,
              'provenance': {'knn': knn.provenance(), 'bridge': bridge.provenance(), 'reference': reference,
                             'pairing': pairing, 'verification': verification_record((knn, knn_rows), (bridge, bridge_rows)),
                             **code_record()}}
    finish(output, {DUPLICATE_GROUPS_CSV: (groups, GROUP_COLUMNS), DUPLICATE_FOLDS_CSV: (folds, FOLD_COLUMNS),
                    DUPLICATE_ACCURACY_CSV: (accuracy, ACCURACY_COLUMNS), DUPLICATE_READOUT_CSV: (readout, READOUT_COLUMNS)},
           DUPLICATE_JSON, record)
    return record


# ----------------------------------------------------------------------------- C: Friedman ranks and the Nemenyi difference

RANK_MODELS = (KNN_MODEL, 'svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting')
RANK_METRIC, ALPHA, DISPLAY_DECIMALS = 'error', .05, 1
RANK_MATRIX_CSV, MEAN_RANKS_CSV, RANKS_JSON = 'rank_matrix.csv', 'mean_ranks.csv', 'friedman_nemenyi.json'
RANK_OUTPUTS = (RANK_MATRIX_CSV, MEAN_RANKS_CSV, RANKS_JSON)
RANK_MATRIX_COLUMNS = ('status', 'dataset', 'model_id', 'mean_error', 'rank', 'mean_error_percent_as_displayed',
                       'rank_as_displayed')
MEAN_RANK_COLUMNS = ('status', 'model_id', 'mean_rank', 'rank_sum', 'n_datasets', 'mean_rank_as_displayed')
REFEREE_M8A = {'source': 'simulated referee P6, concern M8a (passes/P6-report.md), computed from the main benchmark table',
               'mean_ranks': {'svc_rbf': 2.29, 'mlp': 3.14, KNN_MODEL: 3.43, 'random_forest': 3.43, 'numeric_knn': 4.29,
                              'gradient_boosting': 4.43},
               'chi2': 6.18, 'p': .29, 'nemenyi_cd': 2.85, 'decimals': 2}
RANK_DEFINITIONS = {
    'matrix': f'rows: the datasets of the knn run; columns: {KNN_MODEL} and the five tuned comparators (the majority class '
              'excluded); cell: mean outer error, the summary.json summaries mean (the mean over the outer folds of the '
              'fitting-seed-averaged error, the values of the main benchmark table)',
    'rank': 'within each dataset, 1 for the lowest mean error; tied errors (exact float equality) share their average rank '
            '(scipy.stats.rankdata, method average)',
    'chi2': 'Friedman statistic 12N/(k(k+1)) * (sum of squared mean ranks - k(k+1)^2/4) for N datasets and k models, on '
            'k - 1 df, with its chi-square p (Demsar 2006)',
    'chi2_tie_corrected': 'chi2 divided by 1 - sum over datasets and tied groups of (t^3 - t) / (N k (k^2 - 1)), as '
                          'scipy.stats.friedmanchisquare computes it; equal to chi2 when no errors tie',
    'iman_davenport_f': '(N - 1) chi2 / (N (k - 1) - chi2) on (k - 1, (k - 1)(N - 1)) df (Demsar 2006); null under '
                        'perfect agreement',
    'nemenyi_cd': 'q_alpha * sqrt(k (k + 1) / (6 N)) at alpha = 0.05, q_alpha the studentized range quantile for k groups '
                  'and infinite df divided by sqrt(2); two models whose mean ranks differ by more than it are separated',
    'as_displayed': f'the same after rounding each mean error to {DISPLAY_DECIMALS} decimal in percent, as the main '
                    'benchmark table prints it',
    'referee_m8a': "the simulated referee's figures, and this computation rounded to the referee's precision",
    'status': 'descriptive: not a registered analysis; the p value and the critical difference describe this panel only',
}


def friedman_nemenyi(matrix, alpha=ALPHA):
    """Friedman test over an N x k matrix (rows: datasets; columns: models; lower is better) and the Nemenyi critical
    difference, after Demsar (2006); RANK_DEFINITIONS gives every formula."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all() or not 0 < alpha < 1:
        raise ValueError('Expected a finite matrix of at least two datasets and two models, and 0 < alpha < 1')
    n, k = matrix.shape
    ranks = np.vstack([stats.rankdata(row, method='average') for row in matrix])
    mean_ranks = ranks.mean(axis=0)
    chi2 = max(0., float(12 * n / (k * (k + 1)) * (np.sum(mean_ranks ** 2) - k * (k + 1) ** 2 / 4)))
    tie_sum = int(sum(t ** 3 - t for row in matrix for t in np.unique(row, return_counts=True)[1].tolist()))
    correction = 1 - tie_sum / (n * k * (k * k - 1))
    corrected = chi2 / correction if correction > 0 else None
    denominator = n * (k - 1) - chi2
    f_value = (n - 1) * chi2 / denominator if denominator > 1e-12 else None
    q_alpha = float(stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))
    return {'n_datasets': n, 'n_models': k, 'ranks': ranks.tolist(), 'mean_ranks': mean_ranks.tolist(),
            'rank_sums': ranks.sum(axis=0).tolist(), 'chi2': chi2, 'df': k - 1, 'p': float(stats.chi2.sf(chi2, k - 1)),
            'tie_sum': tie_sum, 'chi2_tie_corrected': corrected,
            'p_tie_corrected': None if corrected is None else float(stats.chi2.sf(corrected, k - 1)),
            'iman_davenport_f': f_value, 'iman_davenport_df': [k - 1, (k - 1) * (n - 1)],
            'iman_davenport_p': None if f_value is None else float(stats.f.sf(f_value, k - 1, (k - 1) * (n - 1))),
            'alpha': alpha, 'q_alpha': q_alpha, 'nemenyi_cd': q_alpha * float(np.sqrt(k * (k + 1) / (6 * n)))}


def display_value(value):
    """A mean error as the main benchmark table prints it: percent with DISPLAY_DECIMALS decimals."""
    return float(f'{100 * float(value):.{DISPLAY_DECIMALS}f}')


def analyse_ranks(knn_source, output):
    check_output(output, RANK_OUTPUTS)             # an unusable output location is refused first
    knn = load_run(knn_source, 'knn')
    missing = [model for model in RANK_MODELS if model not in knn.registry]
    if missing:
        raise RunComparisonError(f'The knn run holds no {", ".join(missing)}')
    rows, _ = verify_run(knn)
    datasets = list(knn.protocol['datasets'])
    matrix = [[published_mean(knn, name, model, RANK_METRIC) for model in RANK_MODELS] for name in datasets]
    shown = [[display_value(value) for value in row] for row in matrix]
    test, displayed = friedman_nemenyi(matrix), friedman_nemenyi(shown)
    mean_ranks, cd = dict(zip(RANK_MODELS, test['mean_ranks'])), test['nemenyi_cd']
    pairs = [{'model_a': a, 'model_b': b, 'mean_rank_difference': abs(mean_ranks[a] - mean_ranks[b])}
             for index, a in enumerate(RANK_MODELS) for b in RANK_MODELS[index + 1:]]
    matrix_rows = [{'status': RANK_STATUS, 'dataset': name, 'model_id': model, 'mean_error': matrix[i][j],
                    'rank': test['ranks'][i][j], 'mean_error_percent_as_displayed': shown[i][j],
                    'rank_as_displayed': displayed['ranks'][i][j]}
                   for i, name in enumerate(datasets) for j, model in enumerate(RANK_MODELS)]
    mean_rows = [{'status': RANK_STATUS, 'model_id': model, 'mean_rank': test['mean_ranks'][j], 'rank_sum': test['rank_sums'][j],
                  'n_datasets': len(datasets), 'mean_rank_as_displayed': displayed['mean_ranks'][j]}
                 for j, model in enumerate(RANK_MODELS)]
    decimals = REFEREE_M8A['decimals']
    computed = {'mean_ranks': {model: round(rank, decimals) for model, rank in mean_ranks.items()},
                'chi2': round(test['chi2'], decimals), 'p': round(test['p'], decimals), 'nemenyi_cd': round(cd, decimals)}
    record = {'status': RANK_STATUS,
              'purpose': 'referee_M8a_friedman_test_and_nemenyi_critical_difference_over_mean_outer_error',
              'metric': RANK_METRIC, 'models': list(RANK_MODELS),
              'excluded_models': [model for model in knn.models if model not in RANK_MODELS], 'datasets': datasets,
              'definitions': dict(RANK_DEFINITIONS),
              'matrix': {name: dict(zip(RANK_MODELS, row)) for name, row in zip(datasets, matrix)},
              'ranks': {name: dict(zip(RANK_MODELS, row)) for name, row in zip(datasets, test['ranks'])},
              'mean_ranks': mean_ranks,
              'friedman': {key: test[key] for key in ('n_datasets', 'n_models', 'chi2', 'df', 'p', 'tie_sum', 'chi2_tie_corrected',
                                                      'p_tie_corrected', 'iman_davenport_f', 'iman_davenport_df',
                                                      'iman_davenport_p')},
              'nemenyi': {'alpha': ALPHA, 'q_alpha': test['q_alpha'], 'critical_difference': cd,
                          'pairs_exceeding_critical_difference': [pair for pair in pairs if pair['mean_rank_difference'] > cd],
                          'largest_mean_rank_difference': max(pairs, key=lambda pair: pair['mean_rank_difference'])},
              'ties': {'datasets_with_tied_errors': sum(len(set(row)) < len(RANK_MODELS) for row in matrix),
                       'datasets_with_tied_errors_as_displayed': sum(len(set(row)) < len(RANK_MODELS) for row in shown),
                       'ranks_as_displayed_equal_ranks': displayed['ranks'] == test['ranks'],
                       'friedman_as_displayed': {'chi2': displayed['chi2'], 'p': displayed['p']}},
              'referee_m8a': {'reported': REFEREE_M8A, 'computed_at_reported_precision': computed,
                              'reproduced': computed['mean_ranks'] == REFEREE_M8A['mean_ranks']
                                            and all(computed[key] == REFEREE_M8A[key] for key in ('chi2', 'p', 'nemenyi_cd'))},
              'provenance': {'knn': knn.provenance(), 'verification': verification_record((knn, rows)), **code_record()}}
    finish(output, {RANK_MATRIX_CSV: (matrix_rows, RANK_MATRIX_COLUMNS), MEAN_RANKS_CSV: (mean_rows, MEAN_RANK_COLUMNS)},
           RANKS_JSON, record)
    return record


# ----------------------------------------------------------------------------- command

def _signed(value):
    return 'none' if value is None else f'{value:+.4f}'


def _print_comparators(record):
    for row in record['contrasts']:
        best = ' (best comparator)' if row['best_comparator'] else ''
        print(f"{row['dataset']}: {row['model_a']} - {row['model_b']}{best} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_unadjusted']:.3g}")
    print(f"({record['status']})")


def _print_duplicates(record):
    for entry in record['datasets']:
        counts, test = entry['counts'], entry['test_rows']
        print(f"{entry['dataset']}: {counts['duplicate_rows']} duplicate rows in {counts['duplicate_groups']} groups "
              f"({counts['label_conflicting_groups']} label-conflicting); {test['share_with_training_duplicate']:.2%} of test "
              f"rows have a training duplicate; {len(entry['folds_without_remaining_rows'])} folds without remaining rows")
        for row in entry['readout']:
            interval = '' if row['ci_low'] is None else f" [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            print(f"  {row['row_set']}: {KNN_MODEL} - {FULL_MODEL} accuracy {_signed(row['mean_difference'])}{interval} "
                  f"over {row['n_folds']} folds")
    reproduction = record['reproduction']
    print(f"all-rows accuracies reproduced exactly: {reproduction['cells_checked']} cells, "
          f"{reproduction['model_means_checked']} means ({record['status']})")


def _print_ranks(record):
    friedman, nemenyi = record['friedman'], record['nemenyi']
    print('mean ranks: ' + ', '.join(f'{model} {rank:.2f}' for model, rank in sorted(record['mean_ranks'].items(),
                                                                                    key=lambda item: item[1])))
    print(f"Friedman chi2 = {friedman['chi2']:.4f} on {friedman['df']} df, p = {friedman['p']:.4f}; Nemenyi critical "
          f"difference (alpha {nemenyi['alpha']}) = {nemenyi['critical_difference']:.4f}; referee figures reproduced: "
          f"{record['referee_m8a']['reproduced']} ({record['status']})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    helps = {'comparators': f'{KNN_MODEL} minus each comparator of the knn run (descriptive)',
             'duplicates': 'exact-duplicate sensitivity of the knn and bridge runs (descriptive)',
             'ranks': 'Friedman ranks and the Nemenyi critical difference over the knn run (descriptive)'}
    for command, text in helps.items():
        sub = commands.add_parser(command, help=text)
        sub.add_argument('--knn-source', type=Path, required=True, help='complete bridge_knn run directory (summary.json written)')
        if command == 'duplicates':
            sub.add_argument('--bridge-source', type=Path, required=True, help='complete bridge run directory (summary.json written)')
        sub.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'comparators':
            record, show = analyse_comparators(args.knn_source, args.output), _print_comparators
        elif args.command == 'duplicates':
            record, show = analyse_duplicates(args.knn_source, args.bridge_source, args.output), _print_duplicates
        else:
            record, show = analyse_ranks(args.knn_source, args.output), _print_ranks
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'referee_analyses {args.command} refused: {exc}\n')
    show(record)


if __name__ == '__main__':
    main()
