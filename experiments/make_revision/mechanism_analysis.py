"""Mechanism analysis of the two results that survive the Holm adjustment; no model is fitted.

python -m experiments.make_revision.mechanism_analysis analyse --runs R --output O

Datasets: balance_scale and mfeat_zernike, the two whose training gains survive the paper's Holm correction over all 34
training contrasts. Models: the ten of newdata.MODEL_ORDER - ArrowFlow with the kNN readout, the untrained ArrowFlow,
tuned input footrule kNN, unsorted projected numeric kNN, raw numeric kNN, the five tuned classical models (raw numeric
kNN is one of them) and the majority class.

Every number comes from the saved per-example predictions, the re-verified outer model rows and the prepared features of
the five completed runs that hold those datasets and models (the bridge_knn, knn_training and knn_projected runs and the
two newdata batch runs). They are loaded and re-verified exactly as holistic.load_panel loads them - compare_runs
load_run and verify_run (prepared data and splits hashes, the declared nested splits, planned jobs and fit logs, every
result record with its per-example predictions, summary.json against the re-verified model rows), the registered
pairings (compare_runs check_training_reference and check_training_pairing, compare_projected check_projected_references
and check_projected_pairing, compare_newdata check_batches) and holistic.check_combination - except that the per-example
prediction cells verify_run returns are kept instead of discarded, and every rate below is recomputed from them. Each
model's fold/seed accuracy recomputed from those predictions must equal the accuracy its verified model row records.
Anything that does not verify is refused (exit status 2) before an output is written; the outputs are written all or
none and a file with different content is never replaced.

Analyses

1 class_rates.csv        the full confusion matrix and per-class recall of every model on each dataset, pooled over the
                         outer folds and fitting seeds and again within each outer fold, every row carrying the outer
                         fits and the predictions behind it. The models do NOT all have the same number of fits: the
                         majority class, the SVC and numeric kNN are deterministic and ran one fitting seed, so their
                         rates rest on a third of the predictions of the seven seed-averaged models, and no pooled rate
                         may be compared with another without stating both counts. Pooled rates repeat the same
                         underlying rows across the outer repeats and the fitting seeds; they are descriptive and are
                         never an independent sample for an interval or a test.
2 representation_ladder.csv  the five rungs of newdata.LADDER (raw numeric kNN, unsorted projected numeric kNN on the
                         pre-sort scores, tuned input footrule kNN on the sorted encoded input, the untrained ArrowFlow
                         and ArrowFlow), each with its mean outer error and its per-class recall, so that the rung at
                         which a class becomes recoverable can be read off.
3 paired_contrasts.csv   post hoc: ArrowFlow minus unsorted projected numeric kNN, accuracy, fitting seeds averaged
                         within each outer fold, corrected resampled t over the 15 outer folds (q = 0.25, 95%, 14 df).
                         The two registered contrasts (ArrowFlow minus the untrained ArrowFlow and minus tuned input
                         footrule kNN) are recomputed beside them and must equal the published newdata family values to
                         REPRODUCTION_TOLERANCE, or the analysis refuses.
4 torque_structure.csv   balance_scale only: the label rule (left weight x left distance against right weight x right
                         distance) checked against every prepared row, the absolute torque difference of each row, and
                         each model's recall inside the declared buckets of that difference. The buckets are a property
                         of the data; nothing here shows that any model represents or learned the physical rule.
5 architecture_matching.csv  the hidden widths ArrowFlow and the untrained control selected in each outer fold, and the
                         folds and fits where they differ; the registered contrast is architecture-matched only on the
                         folds where they agree.
6 mfeat_reading.csv      items 2 and 3 stated as arithmetic: how much error the sorted ordinal representation costs
                         against the pre-sort projection, how much of that cost training recovers, and whether the
                         trained model passes the pre-sort rung.

mechanism_analysis.json holds every result, the definitions, the per-model normalization counts, the provenance and
verification of each source run, and the sha256 of each CSV.
"""
import argparse
from pathlib import Path
import numpy as np
from .compare_newdata import LABELS as BATCH_LABELS, check_batches
from .compare_projected import check_projected_pairing, check_projected_references
from .compare_runs import (VALIDATORS, RunComparisonError, _reason, check_output, check_training_pairing,
                           check_training_reference, load_run, verify_run)
from .holistic import (ALL_MODELS, CLASSICAL, MAJORITY, Panel, RUN_LABELS, WORKSPACE_RUNS, check_combination,
                       check_published, read_published, source_paths)
from .knn_controls import INPUT_MODEL, TRAINED_MODEL, UNTRAINED_MODEL, selected_widths
from .newdata import LADDER
from .projected_knn import PROJECTED_MODEL, RAW_MODEL
from .referee_analyses import check_design, code_record, finish
from .run_revision import load_prepared

TRAINED, UNTRAINED, INPUT, PROJECTED, RAW = TRAINED_MODEL, UNTRAINED_MODEL, INPUT_MODEL, PROJECTED_MODEL, RAW_MODEL
DATASETS = ('balance_scale', 'mfeat_zernike')
MODELS = tuple(ALL_MODELS)
METRIC = 'accuracy'
TOLERANCE = 1e-12
REPRODUCTION_TOLERANCE = 1e-12
PUBLISHED_KEYS = ('newdata_families_csv', 'newdata_analysis_json')
ANALYSIS_SOURCES = ('mechanism_analysis.py', 'holistic.py', 'compare_runs.py', 'compare_projected.py',
                    'compare_newdata.py', 'referee_analyses.py', 'newdata.py', 'knn_controls.py', 'projected_knn.py',
                    'evaluation.py', 'reporting.py', 'run_revision.py')

POOLED, PER_FOLD = 'pooled', 'outer_fold'
DESCRIPTIVE = ('descriptive; pooled over the outer folds, repeats and fitting seeds, which repeat the same underlying '
               'rows; never an independent sample for an interval or a test')
POST_HOC = 'post hoc; descriptive; not a registered family; no multiplicity adjustment'
REGISTERED_STATUS = 'registered family, recomputed here only as a reproduction check of the published value'
REGISTERED_CONTRASTS = ((TRAINED, UNTRAINED, 'primary'), (TRAINED, INPUT, 'secondary'))
POST_HOC_CONTRASTS = ((TRAINED, PROJECTED),)

# The label rule of balance-scale, declared before it is checked; the analysis refuses unless it reproduces every row.
TORQUE_SPECS = {
    'balance_scale': {'left': ('left-weight', 'left-distance'), 'right': ('right-weight', 'right-distance'),
                      'greater_label': 'L', 'less_label': 'R', 'equal_label': 'B', 'expected_rows': 625,
                      'expected_class_counts': {'B': 49, 'L': 288, 'R': 288},
                      'rule': 'left weight x left distance against right weight x right distance: greater tips left '
                              '(L), smaller tips right (R), equal balances (B)'},
}
BUCKETS = ((0, 0), (1, 2), (3, 5), (6, 10), (11, None))
TEST_ROW, BUCKET_RECALL = 'test_row', 'bucket_recall'

CLASS_COLUMNS = ('status', 'dataset', 'model_id', 'source_run', 'n_fitting_seeds', 'n_outer_folds', 'n_outer_fits',
                 'n_predictions', 'scope', 'outer_repeat', 'outer_fold', 'fits_in_scope', 'predictions_in_scope',
                 'true_label', 'n_true_in_scope', 'predicted_label', 'n_predicted_as', 'share_of_true')
LADDER_COLUMNS = ('status', 'dataset', 'rung_index', 'rung', 'model_id', 'source_run', 'n_fitting_seeds',
                  'n_outer_fits', 'n_predictions', 'mean_error', 'error_outer_fold_sd', 'pooled_accuracy',
                  'true_label', 'n_true_pooled', 'pooled_recall', 'fold_recall_mean', 'fold_recall_sd',
                  'fold_recall_min', 'fold_recall_max', 'n_folds_with_class')
CONTRAST_COLUMNS = ('status', 'dataset', 'contrast_id', 'model_a', 'model_b', 'mean_difference', 'standard_error',
                    'ci_low', 'ci_high', 'p_unadjusted', 'n_folds', 'df', 'test_train_ratio', 'confidence',
                    'n_outer_fits_a', 'n_outer_fits_b', 'n_predictions_a', 'n_predictions_b', 'registered',
                    'published_family', 'published_family_index', 'published_mean_difference', 'published_ci_low',
                    'published_ci_high', 'published_p_approximate', 'published_holm_p_approximate',
                    'agrees_with_published')
TORQUE_COLUMNS = ('status', 'record', 'dataset', 'sample_id', 'left_weight', 'left_distance', 'right_weight',
                  'right_distance', 'left_torque', 'right_torque', 'torque_difference', 'abs_torque_difference',
                  'true_label', 'rule_label', 'rule_agrees', 'bucket_index', 'bucket', 'model_id', 'source_run',
                  'n_fitting_seeds', 'rows_in_bucket', 'predictions_in_bucket', 'correct_in_bucket', 'recall')
ARCHITECTURE_COLUMNS = ('status', 'dataset', 'outer_repeat', 'outer_fold', 'arrowflow_config_id', 'arrowflow_widths',
                        'arrowflow_hidden_layers', 'untrained_config_id', 'untrained_widths',
                        'untrained_hidden_layers', 'widths_match', 'arrowflow_fits_in_fold', 'untrained_fits_in_fold')
READING_COLUMNS = ('status', 'dataset', 'raw_error', 'projected_error', 'input_error', 'untrained_error',
                   'arrowflow_error', 'sorting_cost_input_minus_projected', 'random_layer_cost_untrained_minus_input',
                   'representation_cost_untrained_minus_projected', 'training_gain_untrained_minus_arrowflow',
                   'arrowflow_minus_projected_error', 'recovered_share_of_representation_cost',
                   'contrast_mean_difference', 'contrast_ci_low', 'contrast_ci_high', 'contrast_p_unadjusted',
                   'ahead_of_projected', 'behind_projected', 'reading')

CLASS_CSV, LADDER_CSV, CONTRAST_CSV = 'class_rates.csv', 'representation_ladder.csv', 'paired_contrasts.csv'
TORQUE_CSV, ARCHITECTURE_CSV, READING_CSV = 'torque_structure.csv', 'architecture_matching.csv', 'mfeat_reading.csv'
RECORD_JSON = 'mechanism_analysis.json'
OUTPUTS = (CLASS_CSV, LADDER_CSV, CONTRAST_CSV, TORQUE_CSV, ARCHITECTURE_CSV, READING_CSV, RECORD_JSON)

AHEAD = 'beats_the_pre_sort_projection'
RECOVERS = 'recovers_a_representation_cost_without_reaching_the_pre_sort_projection'
BEHIND = 'behind_the_pre_sort_projection'
NO_COST = 'no_representation_cost_to_recover'
READINGS = {
    AHEAD: 'ArrowFlow has the lower mean outer error of the two and the paired interval of ArrowFlow minus the '
           'unsorted projected kNN excludes zero in ArrowFlow\'s favour',
    RECOVERS: 'the ordinal rungs cost error against the unsorted projected kNN, training removes part of that cost, '
              'and the paired interval of ArrowFlow minus the unsorted projected kNN contains zero, so the trained '
              'model recovers a loss its own representation caused rather than beating the pre-sort scores',
    BEHIND: 'the paired interval of ArrowFlow minus the unsorted projected kNN excludes zero against ArrowFlow',
    NO_COST: 'the ordinal rungs cost no error against the unsorted projected kNN, so there is nothing to recover',
}

DEFINITIONS = {
    'source': 'no model is fitted; every rate is recomputed from the per-example predictions verify_run re-verifies, '
              'every mean error from the re-verified outer model rows, and the balance_scale label rule from the '
              'prepared features every fit read',
    'prediction_reproduction': 'for every dataset, model, outer fold and fitting seed, the accuracy recomputed from '
                               'the saved per-example predictions against the prepared labels equals the accuracy the '
                               f'verified outer model row records (absolute tolerance {TOLERANCE})',
    'n_outer_fits': 'outer folds x the fitting seeds of that model: 15 x 3 = 45 for a stochastic model and 15 x 1 = 15 '
                    'for a deterministic one (the majority class, the SVC and numeric kNN)',
    'n_predictions': 'the test rows of every outer fold, counted once per fitting seed; a one-seed model therefore has '
                     'a third of the predictions of a three-seed model on the same dataset',
    'pooled_rates': DESCRIPTIVE,
    'recall': 'predictions of that true class that carry that class as the predicted label, divided by the '
              'predictions of that true class, inside the scope of the row',
    'confusion_matrix': 'n_predicted_as counts predictions whose true label is true_label and whose predicted label is '
                        'predicted_label; the row of a true label sums to n_true_in_scope',
    'mean_error': 'the mean over the outer folds of the fitting-seed-averaged error, from summary.json of the run '
                  'holding the model (recomputed from the model rows load_run re-verifies)',
    'pooled_accuracy': 'correct predictions divided by all predictions of that model, pooled over folds, repeats and '
                       'seeds; it equals 1 - mean_error only when every outer fold has the same number of test rows',
    'ladder': 'the five rungs of newdata.LADDER in order: raw numeric kNN on the raw features, numeric kNN on the '
              'unsorted projected scores, tuned input footrule kNN on the sorted encoded input, the untrained '
              'ArrowFlow and ArrowFlow',
    'contrast': 'model_a minus model_b accuracy: fitting seeds averaged within each outer fold, then the corrected '
                'resampled t over the outer folds (evaluation.paired_corrected_interval, standard error '
                'sqrt((1/n_folds + q) * variance (ddof 1) of the fold differences), q = test_train_ratio, df = '
                'n_folds - 1); positive favours model_a',
    'registered_reproduction': 'the two registered contrasts must equal the published newdata family values '
                               f'(mean difference, standard error, interval and p) to {REPRODUCTION_TOLERANCE}',
    'torque_rule': 'declared before it is checked; the analysis refuses unless the rule reproduces the label of every '
                   'prepared row and the row and class counts are the declared ones',
    'torque_buckets': 'half-open in neither direction: each bucket holds the rows whose absolute torque difference '
                      'lies between its bounds inclusive; the first bucket is exact equality, which is the balanced '
                      'class itself, so its recall is that class\'s recall',
    'torque_scope': 'a bucket is a property of the data, not of any model; a model recovering the equality bucket is '
                    'not evidence that it represents or learned the physical rule',
    'selected_widths': 'the hidden widths of the configuration the model selected on the inner folds of an outer fold; '
                       'one configuration is selected per outer fold and shared by that fold\'s fitting seeds',
    'architecture_matching': 'the folds where ArrowFlow and the untrained control selected different widths; on those '
                             'folds the registered contrast compares different architectures',
    'readings': READINGS,
    'verified': 'every saved record of all five runs passed ' + '; '.join(VALIDATORS) + '. summary.json model_rows '
                'equal the re-verified model rows field for field and its summaries were recomputed from them. The '
                'analysis source hashes in environment.json are not compared with the current tree; each run\'s '
                'recorded code revision is in provenance.',
}


# ----------------------------------------------------------------------------- the verified panel, with predictions

def load_verified_panel(paths, *, frozen_protocols=None):
    """holistic.load_panel's sequence, keeping the per-example prediction cells verify_run returns.

    Returns (panel, {run label: verify_run cells}, provenance). The cells of one run are
    {(dataset, model, outer_repeat, outer_fold, seed): (selected config_id, test sample IDs, predicted labels)}.
    """
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
    verification, cells = {'validators': list(VALIDATORS)}, {}
    for label, run in runs.items():
        rows, run_cells = verify_run(run)
        cells[label] = run_cells
        verification[label] = {'jobs_verified': len(run.jobs), 'model_rows_verified': sum(map(len, rows.values()))}
    return panel, cells, {'runs': {label: run.provenance() for label, run in runs.items()}, 'references': references,
                          'pairing': pairing, 'combination': combination, 'verification': verification}


def check_datasets(panel, datasets):
    if not datasets:
        raise RunComparisonError('No dataset was declared for the mechanism analysis')
    absent = [name for name in datasets if name not in panel.datasets]
    if absent:
        raise RunComparisonError(f'The verified panel holds no {", ".join(absent)}; it holds {panel.datasets}')


def prepared_labels(panel, name):
    """The prepared features, labels and manifest of one dataset, from the run that holds it."""
    run = panel.run_for(name, TRAINED)
    try:
        X, y, manifest, _ = load_prepared(run.path, name)
    except (OSError, EOFError, KeyError, TypeError, ValueError) as exc:
        raise RunComparisonError(f'The prepared data of {name} in the {run.label} run is unreadable ({_reason(exc)})') from exc
    return X, y, manifest, run


def model_cells(panel, cells, name, model):
    """{(outer_repeat, outer_fold, seed): (sample IDs, predicted labels)} of one dataset and model, folds in schedule order."""
    run = panel.run_for(name, model)
    seeds = panel.seeds(name, model)
    held = cells[run.label]
    chosen = {}
    for repeat, fold in panel.folds:
        for seed in seeds:
            key = (name, model, repeat, fold, seed)
            if key not in held:
                raise RunComparisonError(f'The {run.label} run holds no verified predictions for {key}')
            _, samples, predicted = held[key]
            chosen[repeat, fold, seed] = (np.asarray(samples, dtype=int), np.asarray(predicted, dtype=int))
    return chosen


def check_prediction_accuracy(panel, cells, name, model, y):
    """Every fold/seed accuracy recomputed from the saved predictions equals the verified model row's accuracy."""
    recorded = {(row['outer_repeat'], row['outer_fold'], row['model_seed']): row[METRIC]
                for row in panel.rows(name, model)}
    for key, (samples, predicted) in model_cells(panel, cells, name, model).items():
        if key not in recorded:
            raise RunComparisonError(f'{name} {model} has predictions for {key} but no verified outer model row')
        accuracy = float(np.mean(predicted == y[samples]))
        if not np.isclose(accuracy, recorded[key], rtol=0, atol=TOLERANCE):
            raise RunComparisonError(f'{name} {model} outer fit {key}: the accuracy recomputed from the saved '
                                     f'predictions is {accuracy!r}, the verified model row records {recorded[key]!r}')
    if len(recorded) != len(panel.folds) * len(panel.seeds(name, model)):
        raise RunComparisonError(f'{name} {model} has {len(recorded)} verified outer model rows; the schedule declares '
                                 f'{len(panel.folds)} outer folds x {len(panel.seeds(name, model))} fitting seeds')


def normalization(panel, cells, name, model, y):
    """The fits and the predictions behind every rate of one dataset and model."""
    chosen = model_cells(panel, cells, name, model)
    seeds = panel.seeds(name, model)
    predictions = int(sum(len(samples) for samples, _ in chosen.values()))
    correct = int(sum(int(np.sum(predicted == y[samples])) for samples, predicted in chosen.values()))
    return {'dataset': name, 'model_id': model, 'source_run': panel.run_for(name, model).label,
            'fitting_seeds': list(seeds), 'n_fitting_seeds': len(seeds), 'n_outer_folds': len(panel.folds),
            'n_outer_fits': len(chosen), 'n_predictions': predictions, 'n_correct': correct,
            'pooled_accuracy': float(correct / predictions) if predictions else None,
            'deterministic': len(seeds) == 1}


# ----------------------------------------------------------------------------- 1: confusion matrices and recall

def confusion(chosen, y, n_classes, folds=None):
    """Counts[true, predicted] over the given (repeat, fold, seed) cells; `folds` restricts to those outer folds."""
    counts = np.zeros((n_classes, n_classes), dtype=np.int64)
    fits = 0
    for (repeat, fold, _), (samples, predicted) in chosen.items():
        if folds is not None and (repeat, fold) not in folds:
            continue
        fits += 1
        np.add.at(counts, (y[samples], predicted), 1)
    return counts, fits


def recall_of(counts):
    """Per-class recall, None where the class has no rows in scope."""
    totals = counts.sum(axis=1)
    return [float(counts[k, k] / totals[k]) if totals[k] else None for k in range(len(totals))]


def class_rate_rows(panel, datasets, labels, counts_by):
    """One row per (dataset, model, scope, true label, predicted label); pooled first, then each outer fold."""
    rows = []
    for name in datasets:
        names = labels[name]
        for model in MODELS:
            counts = counts_by[name, model]
            base = {'status': DESCRIPTIVE, 'dataset': name, 'model_id': model,
                    'source_run': panel.run_for(name, model).label,
                    'n_fitting_seeds': counts['normalization']['n_fitting_seeds'],
                    'n_outer_folds': counts['normalization']['n_outer_folds'],
                    'n_outer_fits': counts['normalization']['n_outer_fits'],
                    'n_predictions': counts['normalization']['n_predictions']}
            scopes = [(POOLED, None, None, counts['pooled'], counts['pooled_fits'])]
            scopes += [(PER_FOLD, repeat, fold, counts['folds'][repeat, fold], counts['fold_fits'][repeat, fold])
                       for repeat, fold in panel.folds]
            for scope, repeat, fold, matrix, fits in scopes:
                totals = matrix.sum(axis=1)
                for true_index, true_label in enumerate(names):
                    for predicted_index, predicted_label in enumerate(names):
                        count = int(matrix[true_index, predicted_index])
                        rows.append({**base, 'scope': scope, 'outer_repeat': repeat, 'outer_fold': fold,
                                     'fits_in_scope': fits, 'predictions_in_scope': int(matrix.sum()),
                                     'true_label': true_label, 'n_true_in_scope': int(totals[true_index]),
                                     'predicted_label': predicted_label, 'n_predicted_as': count,
                                     'share_of_true': float(count / totals[true_index]) if totals[true_index] else None})
    return rows


def recall_dispersion(panel, counts, class_index):
    """Mean, SD, min and max over the outer folds of one class's recall; folds without the class are skipped."""
    values = [recall_of(counts['folds'][fold])[class_index] for fold in panel.folds]
    present = [value for value in values if value is not None]
    return {'fold_recall': values, 'n_folds_with_class': len(present),
            'fold_recall_mean': float(np.mean(present)) if present else None,
            'fold_recall_sd': float(np.std(present, ddof=1)) if len(present) > 1 else None,
            'fold_recall_min': float(min(present)) if present else None,
            'fold_recall_max': float(max(present)) if present else None}


# ----------------------------------------------------------------------------- 2: the representation ladder

def ladder_rows(panel, datasets, labels, counts_by):
    rows, record = [], {}
    for name in datasets:
        names = labels[name]
        record[name] = []
        for index, (rung, model) in enumerate(LADDER):
            counts = counts_by[name, model]
            summary = panel.summary(name, model, 'error')
            pooled = recall_of(counts['pooled'])
            entry = {'rung_index': index, 'rung': rung, 'model_id': model,
                     'source_run': panel.run_for(name, model).label, 'mean_error': summary['mean'],
                     'error_outer_fold_sd': summary['outer_fold_sd'],
                     'pooled_accuracy': counts['normalization']['pooled_accuracy'],
                     'n_fitting_seeds': counts['normalization']['n_fitting_seeds'],
                     'n_outer_fits': counts['normalization']['n_outer_fits'],
                     'n_predictions': counts['normalization']['n_predictions'], 'classes': []}
            for class_index, label in enumerate(names):
                dispersion = recall_dispersion(panel, counts, class_index)
                entry['classes'].append({'true_label': label,
                                         'n_true_pooled': int(counts['pooled'].sum(axis=1)[class_index]),
                                         'pooled_recall': pooled[class_index], **dispersion})
                rows.append({'status': DESCRIPTIVE, 'dataset': name,
                             **{key: entry[key] for key in ('rung_index', 'rung', 'model_id', 'source_run',
                                                            'n_fitting_seeds', 'n_outer_fits', 'n_predictions',
                                                            'mean_error', 'error_outer_fold_sd', 'pooled_accuracy')},
                             'true_label': label,
                             'n_true_pooled': int(counts['pooled'].sum(axis=1)[class_index]),
                             'pooled_recall': pooled[class_index],
                             **{key: dispersion[key] for key in ('fold_recall_mean', 'fold_recall_sd',
                                                                 'fold_recall_min', 'fold_recall_max',
                                                                 'n_folds_with_class')}})
            record[name].append(entry)
    return rows, record


# ----------------------------------------------------------------------------- 3: paired contrasts

def published_families(published):
    """{(dataset, model_a, model_b): the published newdata family row}."""
    return {(row['dataset'], row['model_a'], row['model_b']): row for row in published['newdata_families_csv']}


def contrast_rows(panel, datasets, published, normalizations):
    families = published_families(published)
    rows, record = [], []
    for name in datasets:
        for model_a, model_b, family in [(*pair, None) for pair in POST_HOC_CONTRASTS] + list(REGISTERED_CONTRASTS):
            interval = panel.interval(name, model_a, model_b)
            registered = family is not None
            key = (name, model_a, model_b)
            reference = families.get(key) if registered else None
            if registered and reference is None:
                raise RunComparisonError(f'The published newdata families hold no {model_a} minus {model_b} row for '
                                         f'{name}; the registered contrasts cannot be reproduced')
            agrees = None
            if reference is not None:
                fields = (('mean_difference', 'mean_difference'), ('standard_error', 'standard_error'),
                          ('ci_low', 'ci_low'), ('ci_high', 'ci_high'), ('p_approximate', 'p_approximate'))
                wrong = [f'{name_published} published {reference[name_published]}, recomputed {interval[name_computed]!r}'
                         for name_published, name_computed in fields
                         if not np.isclose(float(reference[name_published]), interval[name_computed], rtol=0,
                                           atol=REPRODUCTION_TOLERANCE)]
                if wrong:
                    raise RunComparisonError(f'The registered contrast {model_a} minus {model_b} on {name} does not '
                                             'reproduce the published newdata family value: ' + '; '.join(wrong))
                if reference['family'] != family:
                    raise RunComparisonError(f'The published {model_a} minus {model_b} row for {name} is in family '
                                             f'{reference["family"]!r}, not {family!r}')
                agrees = True
            row = {'status': REGISTERED_STATUS if registered else POST_HOC, 'dataset': name,
                   'contrast_id': f'{model_a}_vs_{model_b}', 'model_a': model_a, 'model_b': model_b,
                   'mean_difference': interval['mean_difference'], 'standard_error': interval['standard_error'],
                   'ci_low': interval['ci_low'], 'ci_high': interval['ci_high'],
                   'p_unadjusted': interval['p_approximate'], 'n_folds': interval['n_folds'], 'df': interval['df'],
                   'test_train_ratio': panel.q, 'confidence': panel.confidence,
                   'n_outer_fits_a': normalizations[name, model_a]['n_outer_fits'],
                   'n_outer_fits_b': normalizations[name, model_b]['n_outer_fits'],
                   'n_predictions_a': normalizations[name, model_a]['n_predictions'],
                   'n_predictions_b': normalizations[name, model_b]['n_predictions'], 'registered': registered,
                   'published_family': reference['family'] if reference else None,
                   'published_family_index': int(reference['family_index']) if reference else None,
                   'published_mean_difference': float(reference['mean_difference']) if reference else None,
                   'published_ci_low': float(reference['ci_low']) if reference else None,
                   'published_ci_high': float(reference['ci_high']) if reference else None,
                   'published_p_approximate': float(reference['p_approximate']) if reference else None,
                   'published_holm_p_approximate': float(reference['holm_p_approximate']) if reference else None,
                   'agrees_with_published': agrees}
            rows.append(row)
            record.append(row)
    return rows, record


# ----------------------------------------------------------------------------- 4: the balance-scale label structure

def bucket_index(value):
    for index, (low, high) in enumerate(BUCKETS):
        if value >= low and (high is None or value <= high):
            return index
    raise ValueError(f'No declared bucket holds {value!r}')


def bucket_name(index):
    low, high = BUCKETS[index]
    return f'{low}' if low == high else (f'>={low}' if high is None else f'{low}-{high}')


def torque_table(X, y, manifest, spec):
    """The declared label rule applied to every prepared row; no check, no claim about any model."""
    features = list(manifest['feature_names'])
    names = list(manifest['label_map'])
    missing = [name for name in spec['left'] + spec['right'] if name not in features]
    unknown = [spec[key] for key in ('greater_label', 'less_label', 'equal_label') if spec[key] not in names]
    if missing or unknown:
        raise RunComparisonError(f'The prepared data does not carry the declared torque rule: missing features '
                                 f'{missing}, labels not in the label map {unknown}')
    column = {name: np.asarray(X[:, features.index(name)], dtype=float) for name in spec['left'] + spec['right']}
    left = column[spec['left'][0]] * column[spec['left'][1]]
    right = column[spec['right'][0]] * column[spec['right'][1]]
    difference = left - right
    index = {spec['greater_label']: names.index(spec['greater_label']),
             spec['less_label']: names.index(spec['less_label']),
             spec['equal_label']: names.index(spec['equal_label'])}
    rule = np.where(difference > 0, index[spec['greater_label']],
                    np.where(difference < 0, index[spec['less_label']], index[spec['equal_label']]))
    agrees = rule == y
    rows = [{'sample_id': int(sample), 'left_weight': float(column[spec['left'][0]][sample]),
             'left_distance': float(column[spec['left'][1]][sample]),
             'right_weight': float(column[spec['right'][0]][sample]),
             'right_distance': float(column[spec['right'][1]][sample]), 'left_torque': float(left[sample]),
             'right_torque': float(right[sample]), 'torque_difference': float(difference[sample]),
             'abs_torque_difference': float(abs(difference[sample])), 'true_label': names[y[sample]],
             'rule_label': names[rule[sample]], 'rule_agrees': bool(agrees[sample]),
             'bucket_index': bucket_index(abs(float(difference[sample])))} for sample in range(len(y))]
    class_counts = {label: int(np.sum(y == position)) for position, label in enumerate(names)}
    return {'rows': rows, 'n_rows': int(len(y)), 'n_agreeing': int(np.sum(agrees)),
            'agreement_share': float(np.mean(agrees)), 'class_counts': class_counts,
            'rule': spec['rule'], 'buckets': [{'bucket_index': index, 'bucket': bucket_name(index),
                                               'low': BUCKETS[index][0], 'high': BUCKETS[index][1],
                                               'n_rows': sum(1 for row in rows if row['bucket_index'] == index),
                                               'class_counts': {label: sum(1 for row in rows
                                                                           if row['bucket_index'] == index
                                                                           and row['true_label'] == label)
                                                                for label in names}}
                                              for index in range(len(BUCKETS))]}


def check_torque(table, spec, name):
    """The declared rule must reproduce every row, and the row and class counts must be the declared ones."""
    if table['n_agreeing'] != table['n_rows']:
        raise RunComparisonError(f'The declared label rule reproduces {table["n_agreeing"]} of {table["n_rows"]} '
                                 f'{name} rows; the analysis is defined only where it reproduces every row')
    if table['n_rows'] != spec['expected_rows'] or table['class_counts'] != spec['expected_class_counts']:
        raise RunComparisonError(f'{name} holds {table["n_rows"]} rows with class counts {table["class_counts"]}; the '
                                 f'declared structure is {spec["expected_rows"]} rows with '
                                 f'{spec["expected_class_counts"]}')


def torque_rows(panel, cells, name, table, y):
    """The per-row torque records, then every model's recall inside each declared bucket."""
    bucket_of = np.array([row['bucket_index'] for row in table['rows']], dtype=int)
    rows = [{'status': DESCRIPTIVE, 'record': TEST_ROW, 'dataset': name, **row, 'bucket': bucket_name(row['bucket_index']),
             **dict.fromkeys(('model_id', 'source_run', 'n_fitting_seeds', 'rows_in_bucket', 'predictions_in_bucket',
                              'correct_in_bucket', 'recall'))} for row in table['rows']]
    record = []
    for model in MODELS:
        chosen = model_cells(panel, cells, name, model)
        seeds = panel.seeds(name, model)
        totals = np.zeros(len(BUCKETS), dtype=np.int64)
        correct = np.zeros(len(BUCKETS), dtype=np.int64)
        for samples, predicted in chosen.values():
            np.add.at(totals, bucket_of[samples], 1)
            np.add.at(correct, bucket_of[samples], (predicted == y[samples]).astype(np.int64))
        for index in range(len(BUCKETS)):
            entry = {'status': DESCRIPTIVE, 'record': BUCKET_RECALL, 'dataset': name,
                     **dict.fromkeys(('sample_id', 'left_weight', 'left_distance', 'right_weight', 'right_distance',
                                      'left_torque', 'right_torque', 'torque_difference', 'abs_torque_difference',
                                      'true_label', 'rule_label', 'rule_agrees')),
                     'bucket_index': index, 'bucket': bucket_name(index), 'model_id': model,
                     'source_run': panel.run_for(name, model).label, 'n_fitting_seeds': len(seeds),
                     'rows_in_bucket': table['buckets'][index]['n_rows'],
                     'predictions_in_bucket': int(totals[index]), 'correct_in_bucket': int(correct[index]),
                     'recall': float(correct[index] / totals[index]) if totals[index] else None}
            rows.append(entry)
            record.append({key: entry[key] for key in ('model_id', 'source_run', 'n_fitting_seeds', 'bucket_index',
                                                       'bucket', 'rows_in_bucket', 'predictions_in_bucket',
                                                       'correct_in_bucket', 'recall')})
    return rows, record


# ----------------------------------------------------------------------------- 5: architecture matching

def architecture_rows(panel, datasets):
    rows, record = [], {}
    for name in datasets:
        chosen = {model: selected_widths(panel.rows(name, model), model) for model in (TRAINED, UNTRAINED)}
        configs = {model: {(row['outer_repeat'], row['outer_fold']): row['config_id'] for row in panel.rows(name, model)}
                   for model in (TRAINED, UNTRAINED)}
        seeds = {model: len(panel.seeds(name, model)) for model in (TRAINED, UNTRAINED)}
        differing = []
        for repeat, fold in panel.folds:
            widths = {model: list(chosen[model][repeat, fold]) for model in (TRAINED, UNTRAINED)}
            match = widths[TRAINED] == widths[UNTRAINED]
            if not match:
                differing.append([repeat, fold])
            rows.append({'status': DESCRIPTIVE, 'dataset': name, 'outer_repeat': repeat, 'outer_fold': fold,
                         'arrowflow_config_id': configs[TRAINED][repeat, fold],
                         'arrowflow_widths': str(widths[TRAINED]), 'arrowflow_hidden_layers': len(widths[TRAINED]),
                         'untrained_config_id': configs[UNTRAINED][repeat, fold],
                         'untrained_widths': str(widths[UNTRAINED]), 'untrained_hidden_layers': len(widths[UNTRAINED]),
                         'widths_match': match, 'arrowflow_fits_in_fold': seeds[TRAINED],
                         'untrained_fits_in_fold': seeds[UNTRAINED]})
        record[name] = {
            'n_outer_folds': len(panel.folds),
            'widths': {model: {str(list(widths)): sum(1 for fold in panel.folds if list(chosen[model][fold]) == list(widths))
                               for widths in sorted({tuple(v) for v in chosen[model].values()})}
                       for model in (TRAINED, UNTRAINED)},
            'folds_with_different_widths': differing, 'n_folds_with_different_widths': len(differing),
            'n_arrowflow_fits_on_those_folds': len(differing) * seeds[TRAINED],
            'n_untrained_fits_on_those_folds': len(differing) * seeds[UNTRAINED],
            'n_arrowflow_fits': len(panel.folds) * seeds[TRAINED],
            'n_untrained_fits': len(panel.folds) * seeds[UNTRAINED],
            'architecture_matched': not differing}
    return rows, record


# ----------------------------------------------------------------------------- 6: the ladder reading

def reading_rows(panel, datasets, ladder_record, contrast_record):
    rows = []
    for name in datasets:
        error = {entry['model_id']: entry['mean_error'] for entry in ladder_record[name]}
        contrast = next(entry for entry in contrast_record
                        if entry['dataset'] == name and (entry['model_a'], entry['model_b']) == (TRAINED, PROJECTED))
        sorting = error[INPUT] - error[PROJECTED]
        random_layers = error[UNTRAINED] - error[INPUT]
        representation = error[UNTRAINED] - error[PROJECTED]
        gain = error[UNTRAINED] - error[TRAINED]
        residual = error[TRAINED] - error[PROJECTED]
        ahead = error[TRAINED] < error[PROJECTED] and contrast['ci_low'] > 0
        behind = error[TRAINED] > error[PROJECTED] and contrast['ci_high'] < 0
        if ahead:
            reading = AHEAD
        elif representation <= 0:
            reading = NO_COST
        elif behind:
            reading = BEHIND
        else:
            reading = RECOVERS
        rows.append({'status': POST_HOC, 'dataset': name, 'raw_error': error[RAW],
                     'projected_error': error[PROJECTED], 'input_error': error[INPUT],
                     'untrained_error': error[UNTRAINED], 'arrowflow_error': error[TRAINED],
                     'sorting_cost_input_minus_projected': sorting,
                     'random_layer_cost_untrained_minus_input': random_layers,
                     'representation_cost_untrained_minus_projected': representation,
                     'training_gain_untrained_minus_arrowflow': gain,
                     'arrowflow_minus_projected_error': residual,
                     'recovered_share_of_representation_cost': gain / representation if representation else None,
                     'contrast_mean_difference': contrast['mean_difference'], 'contrast_ci_low': contrast['ci_low'],
                     'contrast_ci_high': contrast['ci_high'], 'contrast_p_unadjusted': contrast['p_unadjusted'],
                     'ahead_of_projected': ahead, 'behind_projected': behind, 'reading': reading})
    return rows


# ----------------------------------------------------------------------------- the command

def analyse(paths, output, *, datasets=DATASETS, torque_specs=None, frozen_protocols=None):
    check_output(output, OUTPUTS)             # an unusable output location is refused before any run is read
    torque_specs = TORQUE_SPECS if torque_specs is None else torque_specs
    panel, cells, provenance = load_verified_panel(paths, frozen_protocols=frozen_protocols)
    published, files = read_published(paths, PUBLISHED_KEYS)
    check_published(published, files, panel)
    datasets = list(datasets)
    check_datasets(panel, datasets)
    try:
        labels, truth, counts_by, normalizations = {}, {}, {}, {}
        for name in datasets:
            X, y, manifest, run = prepared_labels(panel, name)
            labels[name], truth[name] = list(manifest['label_map']), (X, y, manifest)
            for model in MODELS:
                check_prediction_accuracy(panel, cells, name, model, y)
                chosen = model_cells(panel, cells, name, model)
                pooled, pooled_fits = confusion(chosen, y, len(labels[name]))
                per_fold = {fold: confusion(chosen, y, len(labels[name]), folds={fold}) for fold in panel.folds}
                folds = {fold: counts for fold, (counts, _) in per_fold.items()}
                fold_fits = {fold: fits for fold, (_, fits) in per_fold.items()}
                normalizations[name, model] = normalization(panel, cells, name, model, y)
                counts_by[name, model] = {'pooled': pooled, 'pooled_fits': pooled_fits, 'folds': folds,
                                          'fold_fits': fold_fits, 'normalization': normalizations[name, model]}
        class_rows = class_rate_rows(panel, datasets, labels, counts_by)
        ladder, ladder_record = ladder_rows(panel, datasets, labels, counts_by)
        contrasts, contrast_record = contrast_rows(panel, datasets, published, normalizations)
        torque, torque_record = [], {}
        for name in datasets:
            spec = torque_specs.get(name)
            if spec is None:
                continue
            X, y, manifest = truth[name]
            table = torque_table(X, y, manifest, spec)
            check_torque(table, spec, name)
            rows, buckets = torque_rows(panel, cells, name, table, y)
            torque += rows
            torque_record[name] = {**{key: table[key] for key in ('n_rows', 'n_agreeing', 'agreement_share',
                                                                  'class_counts', 'rule', 'buckets')},
                                   'bucket_recall': buckets}
        architecture, architecture_record = architecture_rows(panel, datasets)
        reading = reading_rows(panel, datasets, ladder_record, contrast_record)
        confusion_record = {name: {model: {'labels': labels[name],
                                           'confusion_matrix': counts_by[name, model]['pooled'].tolist(),
                                           'pooled_recall': recall_of(counts_by[name, model]['pooled']),
                                           'n_true_pooled': counts_by[name, model]['pooled'].sum(axis=1).tolist(),
                                           'fold_recall': {f'r{repeat}f{fold}':
                                                           recall_of(counts_by[name, model]['folds'][repeat, fold])
                                                           for repeat, fold in panel.folds},
                                           **normalizations[name, model]}
                                   for model in MODELS} for name in datasets}
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified runs do not support the mechanism analysis: {_reason(exc)}') from exc
    record = {
        'purpose': 'mechanism_analysis_of_the_two_holm34_surviving_training_gains',
        'status': 'no model fitted; every number recomputed from the verified runs; the class rates and the buckets '
                  'are descriptive and the paired contrast against the unsorted projected kNN is post hoc',
        'datasets': datasets, 'models': {'all': list(MODELS), 'arrowflow': TRAINED, 'untrained_arrowflow': UNTRAINED,
                                         'input_footrule_knn': INPUT, 'projected_numeric_knn': PROJECTED,
                                         'raw_numeric_knn': RAW, 'classical': list(CLASSICAL),
                                         'majority_class': MAJORITY},
        'definitions': DEFINITIONS,
        'normalization': {'note': 'the fits and the predictions behind every rate of every model; a rate of a '
                                  'one-seed model rests on a third of the predictions of a three-seed model on the '
                                  'same dataset and the two must never be compared without stating both counts',
                          'by_model': {name: {model: normalizations[name, model] for model in MODELS}
                                       for name in datasets}},
        'class_rates': {'pooled': confusion_record,
                        'per_outer_fold_cells': f'the full per-fold confusion cells are in {CLASS_CSV}, sealed by its '
                                                'sha256 in outputs'},
        'representation_ladder': ladder_record, 'paired_contrasts': contrast_record,
        'torque_structure': torque_record or 'no declared torque dataset in this panel',
        'architecture_matching': architecture_record, 'reading': reading,
        'provenance': {**provenance, 'published_analyses': files, **code_record(ANALYSIS_SOURCES)},
    }
    tables = {CLASS_CSV: (class_rows, CLASS_COLUMNS), LADDER_CSV: (ladder, LADDER_COLUMNS),
              CONTRAST_CSV: (contrasts, CONTRAST_COLUMNS), TORQUE_CSV: (torque, TORQUE_COLUMNS),
              ARCHITECTURE_CSV: (architecture, ARCHITECTURE_COLUMNS), READING_CSV: (reading, READING_COLUMNS)}
    finish(output, tables, RECORD_JSON, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    analyse_parser = sub.add_parser('analyse', help='the mechanism analysis of the two Holm-34 surviving gains')
    analyse_parser.add_argument('--runs', default=str(WORKSPACE_RUNS), help='the workspace runs directory')
    analyse_parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        analyse(source_paths(args.runs), Path(args.output))
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'mechanism_analysis {args.command} refused: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
