"""Task 22: referee_analyses, three descriptive analyses computed from frozen run outputs only (no model is fitted).

The run fixtures are test_compare_runs.build_pair runs: the saved layout of a complete bridge run and of a bridge_knn run
that pins it, with synthetic per-example predictions. Here dataset alpha has copied feature rows, so some test rows have
an exact training duplicate, and one copied group carries two labels; dataset beta has no duplicates.
"""
import csv
import dataclasses
import hashlib
import json
import math
import shutil
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import stats
import test_compare_runs as tcr
from experiments.make_revision import referee_analyses as ra
from experiments.make_revision.compare_runs import FULL_MODEL, KNN_MODEL, RunComparisonError, load_run, verify_run
from experiments.make_revision.run_revision import load_prepared

COPIES = {'alpha': {1: 0, 2: 0, 12: 11, 19: 5}}      # row: the row whose features it copies (row 5 class 0, row 19 class 1)
COMPARATOR_OUTPUTS = ('comparator_contrasts.csv', 'comparator_contrasts.json')
DUPLICATE_OUTPUTS = ('duplicate_groups.csv', 'duplicate_folds.csv', 'duplicate_accuracy.csv', 'duplicate_readout.csv',
                     'duplicate_sensitivity.json')
RANK_OUTPUTS = ('rank_matrix.csv', 'mean_ranks.csv', 'friedman_nemenyi.json')
FOLDS4 = [(0, 0), (0, 1), (0, 2), (0, 3)]
DEMSAR_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


# ----------------------------------------------------------------------------- duplicate detection and restriction

def test_duplicate_groups_use_exact_float_equality_and_count_label_conflicts():
    X = np.array([[1., 2.], [1., 2.], [0., -0.], [0., 0.], [1., 2. + 1e-12], [3., 4.], [3., 4.], [3., 4.]])
    vector, counts = ra.duplicate_groups(X, np.array([0, 0, 1, 0, 1, 2, 2, 2]))
    assert vector.tolist() == [0, 0, 1, 1, 2, 3, 3, 3]            # -0.0 equals 0.0; 2 + 1e-12 is another vector
    assert counts.pop('groups') == [{'sample_ids': [0, 1], 'labels': [0, 0]}, {'sample_ids': [2, 3], 'labels': [1, 0]},
                                    {'sample_ids': [5, 6, 7], 'labels': [2, 2, 2]}]
    assert counts == {'n_rows': 8, 'n_distinct_rows': 4, 'duplicate_rows': 4, 'rows_in_duplicate_groups': 7,
                      'duplicate_groups': 3, 'label_conflicting_groups': 1, 'largest_group': 3,
                      'group_sizes': {'2': 2, '3': 1}}
    _, none = ra.duplicate_groups(np.eye(3), [0, 1, 2])
    assert (none['duplicate_rows'], none['duplicate_groups'], none['largest_group'], none['groups']) == (0, 0, 1, [])
    with pytest.raises(ValueError, match='NaN'):
        ra.duplicate_groups(np.array([[np.nan], [np.nan]]), [0, 0])


def test_a_test_row_is_flagged_only_when_a_copy_of_it_lies_in_the_training_partition():
    vector = np.array([0, 1, 1, 2, 2, 0, 2])                       # groups {0, 5}, {1, 2}, {3, 4, 6}
    assert ra.training_duplicate_flags(vector, [0, 4, 6], [1, 2, 3, 5]) == [False, False, True, True]
    assert ra.training_duplicate_flags(vector, [1, 2, 3], [0, 4, 5, 6]) == [False, True, False, True]


def fake_runs(knn_values, bridge_values, rows, seeds=(1, 2)):
    """Stand-in runs holding only what the pairing reads: model rows, seed schedules and per-cell accuracies."""
    def run(label, model, values):
        model_rows = [{'model_id': model, 'outer_repeat': r, 'outer_fold': f, 'model_seed': s} for r, f in FOLDS4 for s in seeds]
        accuracies = {(model, r, f, s): {'restricted': values[r, f][i], 'rows': {'restricted': rows[label][r, f]}}
                      for r, f in FOLDS4 for i, s in enumerate(seeds)}
        return SimpleNamespace(label=label, summary={'model_rows': {'d': model_rows}},
                               schedule={'expected_seeds': {model: list(seeds)}}), accuracies
    (knn, knn_accuracies), (bridge, bridge_accuracies) = run('knn', KNN_MODEL, knn_values), run('bridge', FULL_MODEL, bridge_values)
    return knn, bridge, knn_accuracies, bridge_accuracies


def test_folds_without_remaining_rows_are_dropped_and_only_identical_row_sets_are_paired():
    flags = {(0, 0): [True, False, False], (0, 1): [True, True, True], (0, 2): [False, True, False], (0, 3): [False] * 3}
    remaining, empty = ra.split_folds(flags, FOLDS4)
    assert remaining == [(0, 0), (0, 2), (0, 3)] and empty == [[0, 1]]
    knn_values = {(0, 0): [.5, 1.], (0, 1): [None, None], (0, 2): [1., 1.], (0, 3): [2 / 3, 1 / 3]}
    bridge_values = {(0, 0): [.5, .5], (0, 1): [None, None], (0, 2): [.5, 0.], (0, 3): [1 / 3, 1 / 3]}
    rows = {(0, 0): (4, 7), (0, 1): (), (0, 2): (1, 9), (0, 3): (2, 5, 8)}
    knn, bridge, knn_accuracies, bridge_accuracies = fake_runs(knn_values, bridge_values, {'knn': rows, 'bridge': rows})
    result = ra.paired_readout(knn, bridge, 'd', knn_accuracies, bridge_accuracies, 'restricted', remaining)
    differences = np.array([.75 - .5, 1. - .25, .5 - 1 / 3])       # seed-averaged per remaining fold
    se = math.sqrt((1 / 3 + .25) * differences.var(ddof=1))
    half = stats.t.ppf(.975, 2) * se
    assert (result['n_folds'], result['df']) == (3, 2)
    assert result['mean_difference'] == pytest.approx(differences.mean(), abs=1e-12)
    assert (result['ci_low'], result['ci_high']) == pytest.approx((differences.mean() - half, differences.mean() + half), abs=1e-12)
    assert result['p_unadjusted'] == pytest.approx(2 * stats.t.sf(differences.mean() / se, 2), abs=1e-12)
    one = ra.paired_readout(knn, bridge, 'd', knn_accuracies, bridge_accuracies, 'restricted', [(0, 2)])
    assert one == dict.fromkeys(ra.INTERVAL_FIELDS) | {'mean_difference': .75, 'n_folds': 1}
    none = ra.paired_readout(knn, bridge, 'd', knn_accuracies, bridge_accuracies, 'restricted', [])
    assert none == dict.fromkeys(ra.INTERVAL_FIELDS) | {'n_folds': 0}
    other = {**rows, (0, 2): (1, 10)}
    knn, bridge, knn_accuracies, bridge_accuracies = fake_runs(knn_values, bridge_values, {'knn': rows, 'bridge': other})
    with pytest.raises(RunComparisonError, match='different restricted test rows'):
        ra.paired_readout(knn, bridge, 'd', knn_accuracies, bridge_accuracies, 'restricted', remaining)


# ----------------------------------------------------------------------------- Friedman and Nemenyi

def test_friedman_statistic_and_nemenyi_difference_on_matrices_with_known_values():
    agreement = ra.friedman_nemenyi([[.1, .2, .3]] * 4)            # four datasets rank three models alike
    assert agreement['mean_ranks'] == [1., 2., 3.] and agreement['df'] == 2
    assert agreement['chi2'] == pytest.approx(8.) and agreement['p'] == pytest.approx(math.exp(-4.))  # sf on 2 df: exp(-x/2)
    assert agreement['iman_davenport_f'] is None                    # N(k - 1) - chi2 = 0 under perfect agreement
    assert agreement['nemenyi_cd'] == pytest.approx(2.343 * math.sqrt(3 * 4 / (6 * 4)), abs=1e-3)
    latin = ra.friedman_nemenyi([[1, 2, 3], [1, 3, 2], [2, 1, 3]])  # rank sums 4, 6, 8
    assert latin['chi2'] == pytest.approx(8 / 3) and latin['p'] == pytest.approx(math.exp(-4 / 3))
    assert latin['iman_davenport_f'] == pytest.approx(1.6) and latin['iman_davenport_p'] == pytest.approx(stats.f.sf(1.6, 2, 4))
    tied = np.array([[.1, .1, .3], [.2, .1, .3], [.3, .2, .1], [.1, .2, .2]])
    result = ra.friedman_nemenyi(tied)
    assert result['ranks'] == [[1.5, 1.5, 3.], [2., 1., 3.], [3., 2., 1.], [1., 2.5, 2.5]]
    assert result['tie_sum'] == 12 and result['chi2'] == pytest.approx(.875)
    reference = stats.friedmanchisquare(*tied.T)
    assert result['chi2_tie_corrected'] == pytest.approx(reference.statistic) == pytest.approx(1.)
    assert result['p_tie_corrected'] == pytest.approx(reference.pvalue)
    for k, q in DEMSAR_Q05.items():                                 # Demsar (2006), Table 5a
        assert ra.friedman_nemenyi(np.tile(np.arange(k, dtype=float), (2, 1)))['q_alpha'] == pytest.approx(q, abs=1e-3)
    with pytest.raises(ValueError):
        ra.friedman_nemenyi([[1., np.nan], [1., 2.]])


# ----------------------------------------------------------------------------- run fixtures

@pytest.fixture(scope='session')
def pair(tmp_path_factory):
    """A valid bridge and bridge_knn pair (test_compare_runs.build_pair) whose dataset alpha holds copied feature rows."""
    original = tcr.synthetic_dataset

    def with_copies(name, shift=0.):
        X, y = original(name, shift)
        for row, source in COPIES.get(name, {}).items():
            X[row] = X[source]
        return X, y
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tcr, 'synthetic_dataset', with_copies)
        return tcr.build_pair(tmp_path_factory.mktemp('referee_pair'))


@pytest.fixture(scope='session')
def comparators(pair, tmp_path_factory):
    out = tmp_path_factory.mktemp('comparators')/'out'
    return SimpleNamespace(record=ra.analyse_comparators(pair.knn, out), out=out)


@pytest.fixture(scope='session')
def duplicates(pair, tmp_path_factory):
    out = tmp_path_factory.mktemp('duplicates')/'out'
    return SimpleNamespace(record=ra.analyse_duplicates(pair.knn, pair.bridge, out), out=out)


def read_csv(path):
    with open(path, newline='') as stream:
        return list(csv.DictReader(stream))


def assert_sealed_and_labelled(out, record, names, status):
    """Every CSV row carries the label, and the JSON record seals each CSV's sha256 and is the returned record."""
    assert sorted(path.name for path in out.iterdir()) == sorted(names)
    for name in names:
        if name.endswith('.csv'):
            assert {row['status'] for row in read_csv(out/name)} == {status}
            assert record['outputs'][name]['sha256'] == hashlib.sha256((out/name).read_bytes()).hexdigest()
    assert json.loads((out/next(name for name in names if name.endswith('.json'))).read_text()) == record
    assert record['status'] == status and record['provenance']['code_revision']


def seed_averaged(rows, model):
    return {fold: float(np.mean(values)) for fold, values in tcr.fold_means(rows, model, 'accuracy').items()}


# ----------------------------------------------------------------------------- A: ArrowFlow minus each comparator

def test_comparator_contrasts_pair_outer_folds_average_seeds_and_flag_the_best_comparator(pair, comparators):
    record, names = comparators.record, list(tcr.COMPARATORS)
    assert record['comparators'] == names and (record['n_folds'], record['df'], record['test_train_ratio']) == (15, 14, .25)
    assert [(row['dataset'], row['model_b']) for row in record['contrasts']] == [(d, m) for d in tcr.DATASETS for m in names]
    summaries = json.loads((pair.knn/'summary.json').read_text())['summaries']
    for row in record['contrasts']:
        ours, theirs = seed_averaged(pair.knn_rows[row['dataset']], KNN_MODEL), seed_averaged(pair.knn_rows[row['dataset']], row['model_b'])
        differences = np.array([ours[fold] - theirs[fold] for fold in tcr.FOLDS])
        se = math.sqrt((1 / 15 + .25) * differences.var(ddof=1))
        half = stats.t.ppf(.975, 14) * se
        assert (row['status'], row['model_a'], row['n_folds'], row['df']) == (ra.STATUS, KNN_MODEL, 15, 14)
        assert row['mean_difference'] == pytest.approx(differences.mean(), abs=1e-12)
        assert (row['ci_low'], row['ci_high']) == pytest.approx((differences.mean() - half, differences.mean() + half), abs=1e-12)
        assert row['p_unadjusted'] == pytest.approx(2 * stats.t.sf(abs(differences.mean()) / se, 14), abs=1e-12)
        errors = {r['model_id']: r['mean'] for r in summaries[row['dataset']] if r['metric'] == 'error'}
        assert (row['mean_error_a'], row['mean_error_b']) == (errors[KNN_MODEL], errors[row['model_b']])
        assert row['best_comparator'] == (errors[row['model_b']] == min(errors[model] for model in names))
        assert row['best_comparator'] == (row['model_b'] in record['best_comparator'][row['dataset']])
    assert_sealed_and_labelled(comparators.out, record, COMPARATOR_OUTPUTS, ra.STATUS)


# ----------------------------------------------------------------------------- B: exact-duplicate sensitivity

def saved_predictions(root, name, model):
    """{(repeat, fold, seed): {sample: predicted label}}, read straight from the saved result files."""
    labels = {}
    for path in sorted((root/'results').glob(f'{name}__{model}__r*f*.json')):
        for row in json.loads(path.read_text())['predictions']:
            labels.setdefault((row['outer_repeat'], row['outer_fold'], row['model_seed']), {})[row['sample_id']] = row['y_pred']
    return labels


def independent_marking(root, name):
    """Truth and, per outer fold, (test rows, training-duplicate flags) from the saved data and splits by byte keys."""
    X, y, _, splits = load_prepared(root, name)
    keys = [row.tobytes() for row in np.ascontiguousarray(X + 0.)]
    return y, {(s['outer_repeat'], s['outer_fold']): (s['test'], [keys[t] in {keys[r] for r in s['train']} for t in s['test']])
               for s in splits}


def independent_fold_accuracies(labels, y, marking, restricted):
    folds = {}
    for (repeat, fold, seed), predicted in sorted(labels.items()):
        test, flags = marking[repeat, fold]
        rows = [t for t, flag in zip(test, flags) if not (restricted and flag)]
        if rows:
            folds.setdefault((repeat, fold), []).append(np.mean([predicted[t] == y[t] for t in rows]))
    return {fold: float(np.mean(values)) for fold, values in folds.items()}


def test_duplicate_sensitivity_follows_from_the_saved_data_splits_and_predictions(pair, duplicates):
    record = duplicates.record
    alpha, beta = record['datasets']
    assert alpha['counts'] == {'n_rows': 20, 'n_distinct_rows': 16, 'duplicate_rows': 4, 'rows_in_duplicate_groups': 7,
                               'duplicate_groups': 3, 'label_conflicting_groups': 1, 'largest_group': 3}
    assert alpha['groups'] == [{'sample_ids': [0, 1, 2], 'labels': [0, 0, 0]}, {'sample_ids': [5, 19], 'labels': [0, 1]},
                               {'sample_ids': [11, 12], 'labels': [1, 1]}]
    assert (beta['counts']['duplicate_rows'], beta['groups'], beta['test_rows']['test_rows_with_training_duplicate']) == (0, [], 0)
    for entry in (alpha, beta):
        name = entry['dataset']
        y, marking = independent_marking(pair.knn, name)
        assert [fold['test_rows_with_training_duplicate'] for fold in entry['folds']] == [
            [t for t, flag in zip(*marking[fold]) if flag] for fold in tcr.FOLDS]
        remaining = [fold for fold in tcr.FOLDS if not all(marking[fold][1])]
        assert entry['folds_without_remaining_rows'] == [list(fold) for fold in tcr.FOLDS if fold not in remaining]
        flagged = sum(sum(marking[fold][1]) for fold in tcr.FOLDS)
        assert entry['test_rows']['share_with_training_duplicate'] == flagged / sum(len(marking[fold][0]) for fold in tcr.FOLDS)
        fold_accuracies = {}
        for row in entry['accuracy']:
            root = pair.knn if row['source_run'] == 'knn' else pair.bridge
            labels = saved_predictions(root, name, row['model_id'])
            full, restricted = (independent_fold_accuracies(labels, y, marking, flag) for flag in (False, True))
            fold_accuracies[row['model_id']] = full, restricted
            assert row['reproduced_exactly'] is True and row['accuracy_all_test_rows'] == row['recorded_accuracy']
            assert row['accuracy_all_test_rows'] == pytest.approx(np.mean(list(full.values())), abs=1e-12)
            assert row['accuracy_without_training_duplicates'] == pytest.approx(np.mean(list(restricted.values())), abs=1e-12)
            assert (row['n_folds_all'], row['n_folds_without']) == (15, len(remaining))
        assert [row['model_id'] for row in entry['accuracy']] == [KNN_MODEL, FULL_MODEL, *tcr.COMPARATORS]
        for readout, index in zip(entry['readout'], (0, 1)):
            folds = tcr.FOLDS if index == 0 else remaining
            differences = np.array([fold_accuracies[KNN_MODEL][index][f] - fold_accuracies[FULL_MODEL][index][f] for f in folds])
            half = stats.t.ppf(.975, len(folds) - 1) * math.sqrt((1 / len(folds) + .25) * differences.var(ddof=1))
            assert (readout['n_folds'], readout['df']) == (len(folds), len(folds) - 1)
            assert (readout['mean_difference'], readout['ci_low'], readout['ci_high']) == pytest.approx(
                (differences.mean(), differences.mean() - half, differences.mean() + half), abs=1e-12)
    assert alpha['test_rows']['test_rows_with_training_duplicate'] > 0
    assert any(row['accuracy_without_training_duplicates'] != row['accuracy_all_test_rows'] for row in alpha['accuracy'])
    for row in beta['accuracy']:                                     # no duplicates: the two row sets coincide exactly
        assert row['accuracy_without_training_duplicates'] == row['accuracy_all_test_rows'] and row['change'] == 0
    assert {k: v for k, v in beta['readout'][0].items() if k != 'row_set'} == {k: v for k, v in beta['readout'][1].items() if k != 'row_set'}
    registered = {row['dataset']: row for row in tcr.cr.knn_vs_full_contrasts(load_run(pair.knn, 'knn'), load_run(pair.bridge, 'bridge'))}
    for entry in (alpha, beta):                                      # all rows: the registered contrast before Holm
        registered_row = registered[entry['dataset']]
        assert {field: entry['readout'][0][field] for field in ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'n_folds', 'df')} == {
            field: registered_row[field] for field in ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'n_folds', 'df')}
        assert entry['readout'][0]['p_unadjusted'] == registered_row['p_approximate']
    assert record['reproduction']['cells_checked'] == sum(len(rows) for rows in pair.knn_rows.values()) + sum(len(rows) for rows in pair.bridge_rows.values())
    assert_sealed_and_labelled(duplicates.out, record, DUPLICATE_OUTPUTS, ra.STATUS)
    assert len(read_csv(duplicates.out/'duplicate_accuracy.csv')) == 2 * 8 and len(read_csv(duplicates.out/'duplicate_folds.csv')) == 30


def test_the_all_rows_reproduction_check_is_exact_and_refuses_any_difference(pair):
    knn = load_run(pair.knn, 'knn')
    _, cells = verify_run(knn)
    X, y, _, splits = load_prepared(knn.path, 'alpha')
    vector, _ = ra.duplicate_groups(X, y)
    flags = {(s['outer_repeat'], s['outer_fold']): ra.training_duplicate_flags(vector, s['train'], s['test']) for s in splits}
    accuracies = ra.cell_accuracies(knn, 'alpha', cells, y, splits, flags)
    recorded = knn.summary['model_rows']['alpha']
    assert len(accuracies) == len(recorded) and all(
        accuracies[r['model_id'], r['outer_repeat'], r['outer_fold'], r['model_seed']]['all'] == r['accuracy'] for r in recorded)
    assert all(ra.reproduced_summary(knn, 'alpha', model, accuracies)[0]['mean'] == ra.published_mean(knn, 'alpha', model, 'accuracy')
               for model in knn.models)
    rows = json.loads(json.dumps(knn.summary['model_rows']))         # one recorded accuracy one float step away
    rows['alpha'][3]['accuracy'] = float(np.nextafter(rows['alpha'][3]['accuracy'], 2.))
    with pytest.raises(ra.ReproductionError, match='differs from the recorded'):
        ra.cell_accuracies(dataclasses.replace(knn, summary=dict(knn.summary, model_rows=rows)), 'alpha', cells, y, splits, flags)
    key = next(cell for cell in cells if cell[0] == 'alpha')          # one predicted label changed
    config, test, labels = cells[key]
    changed = dict(cells)
    changed[key] = (config, test, ((int(labels[0]) + 1) % 2,) + tuple(labels[1:]))
    with pytest.raises(ra.ReproductionError, match='differs from the recorded'):
        ra.cell_accuracies(knn, 'alpha', changed, y, splits, flags)
    del changed[key]                                                  # one verified cell missing
    with pytest.raises(ra.ReproductionError, match='no verified predictions'):
        ra.cell_accuracies(knn, 'alpha', changed, y, splits, flags)
    summaries = json.loads(json.dumps(knn.summary['summaries']))      # one summary mean one float step away
    entry = next(r for r in summaries['alpha'] if (r['model_id'], r['metric']) == (KNN_MODEL, 'accuracy'))
    entry['mean'] = float(np.nextafter(entry['mean'], 2.))
    with pytest.raises(ra.ReproductionError, match='mean accuracy on all test rows'):
        ra.reproduced_summary(dataclasses.replace(knn, summary=dict(knn.summary, summaries=summaries)), 'alpha', KNN_MODEL, accuracies)


# ----------------------------------------------------------------------------- C: Friedman ranks, and the command

@pytest.fixture(scope='session')
def ranks(pair, tmp_path_factory):
    out = tmp_path_factory.mktemp('ranks')/'out'
    return SimpleNamespace(record=ra.analyse_ranks(pair.knn, out), out=out)


def test_ranks_use_the_mean_outer_errors_of_arrowflow_and_the_five_tuned_comparators(pair, ranks):
    record = ranks.record
    assert record['models'] == list(ra.RANK_MODELS) and record['excluded_models'] == ['dummy']
    summaries = json.loads((pair.knn/'summary.json').read_text())['summaries']
    matrix = [[next(r['mean'] for r in summaries[name] if (r['model_id'], r['metric']) == (model, 'error'))
               for model in ra.RANK_MODELS] for name in tcr.DATASETS]
    assert [[record['matrix'][name][model] for model in ra.RANK_MODELS] for name in tcr.DATASETS] == matrix
    assert record['mean_ranks'] == dict(zip(ra.RANK_MODELS, np.vstack([stats.rankdata(row) for row in matrix]).mean(axis=0).tolist()))
    assert record['friedman']['chi2'] == ra.friedman_nemenyi(matrix)['chi2'] and record['friedman']['df'] == 5
    assert record['nemenyi']['critical_difference'] == pytest.approx(2.850 * math.sqrt(6 * 7 / (6 * 2)), abs=1e-3)
    assert record['referee_m8a']['reported'] == ra.REFEREE_M8A and isinstance(record['referee_m8a']['reproduced'], bool)
    assert len(read_csv(ranks.out/'rank_matrix.csv')) == 2 * 6 and len(read_csv(ranks.out/'mean_ranks.csv')) == 6
    assert_sealed_and_labelled(ranks.out, record, RANK_OUTPUTS, ra.RANK_STATUS)


def test_commands_write_the_same_labelled_outputs_and_refuse_changed_outputs_or_incomplete_runs(
        pair, comparators, duplicates, ranks, tmp_path, capsys):
    cases = {'comparators': (['--knn-source', str(pair.knn)], comparators.out, COMPARATOR_OUTPUTS),
             'duplicates': (['--knn-source', str(pair.knn), '--bridge-source', str(pair.bridge)], duplicates.out, DUPLICATE_OUTPUTS),
             'ranks': (['--knn-source', str(pair.knn)], ranks.out, RANK_OUTPUTS)}
    for command, (arguments, reference, names) in cases.items():
        ra.main([command, *arguments, '--output', str(tmp_path/command)])
        assert all((tmp_path/command/name).read_bytes() == (reference/name).read_bytes() for name in names)
        ra.main([command, *arguments, '--output', str(tmp_path/command)])    # the same evidence again: identical files
    printed = capsys.readouterr().out
    assert ra.STATUS in printed and 'reproduced exactly' in printed and 'Friedman chi2' in printed
    (tmp_path/'ranks'/'mean_ranks.csv').write_text('changed\n')
    with pytest.raises(SystemExit) as refused:
        ra.main(['ranks', '--knn-source', str(pair.knn), '--output', str(tmp_path/'ranks')])
    assert refused.value.code == 2 and 'overwrite' in capsys.readouterr().err
    shutil.copytree(pair.knn, tmp_path/'incomplete')
    (tmp_path/'incomplete'/'summary.json').unlink()
    with pytest.raises(SystemExit) as refused:
        ra.main(['comparators', '--knn-source', str(tmp_path/'incomplete'), '--output', str(tmp_path/'refused')])
    assert refused.value.code == 2 and 'incomplete' in capsys.readouterr().err and not (tmp_path/'refused').exists()
