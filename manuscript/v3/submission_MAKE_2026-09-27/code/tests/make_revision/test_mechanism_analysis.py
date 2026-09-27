"""mechanism_analysis: the artifact-only mechanism analysis of the two Holm-34 surviving training gains.

The label rule, the buckets, the confusion counts and the dispersion are checked on hand-built inputs with known
answers. The command is checked end to end on test_holistic's fixture chain, which holds every source run and every
published analysis the analysis reads in the registered layout; tests that alter records work on a private copy. The
only test that touches the real workspace runs reads one prepared dataset and is skipped when it is absent.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pytest
import test_compare_runs as tcr
from test_holistic import chain, copy                                     # noqa: F401  (fixtures used by these tests)
from experiments.make_revision import holistic as h
from experiments.make_revision import mechanism_analysis as ma
from experiments.make_revision import newdata as nd
from experiments.make_revision.compare_runs import RunComparisonError
from experiments.make_revision.run_revision import load_prepared

BATCH1 = list(nd.SMOKE_BATCHES['1'])
SPEC = ma.TORQUE_SPECS['balance_scale']
REAL = h.WORKSPACE_RUNS/h.SOURCES['batch1']/'balance_scale'
needs_real_balance_scale = pytest.mark.skipif(not REAL.is_dir(), reason='the real balance_scale run is not on this machine')


def read_csv(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text())))


def load(path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------- the declared buckets

def test_every_absolute_difference_falls_in_exactly_one_declared_bucket():
    for value in range(0, 60):
        holding = [index for index, (low, high) in enumerate(ma.BUCKETS) if value >= low and (high is None or value <= high)]
        assert holding == [ma.bucket_index(value)]
    assert ma.bucket_index(0) == 0 and ma.bucket_index(1) != 0            # the first bucket is exact equality only
    assert [ma.bucket_name(index) for index in range(len(ma.BUCKETS))] == ['0', '1-2', '3-5', '6-10', '>=11']
    with pytest.raises(ValueError, match='No declared bucket'):
        ma.bucket_index(-1)


# ----------------------------------------------------------------------------- the torque rule, hand built

def lattice(values=(1, 2, 3), label_map=('B', 'L', 'R')):
    """Every weight/distance combination over `values`, labelled by the declared rule."""
    rows, labels = [], []
    index = {name: position for position, name in enumerate(label_map)}
    for lw in values:
        for ld in values:
            for rw in values:
                for rd in values:
                    rows.append([lw, ld, rw, rd])
                    difference = lw*ld - rw*rd
                    labels.append(index['L'] if difference > 0 else index['R'] if difference < 0 else index['B'])
    manifest = {'feature_names': ['left-weight', 'left-distance', 'right-weight', 'right-distance'],
                'label_map': list(label_map)}
    return np.array(rows, dtype=float), np.array(labels, dtype=int), manifest


def test_the_declared_rule_reproduces_a_hand_built_lattice_with_its_torques_and_buckets():
    X, y, manifest = lattice()
    table = ma.torque_table(X, y, manifest, SPEC)
    assert (table['n_rows'], table['n_agreeing'], table['agreement_share']) == (81, 81, 1.)
    assert table['class_counts'] == {'B': int(np.sum(y == 0)), 'L': int(np.sum(y == 1)), 'R': int(np.sum(y == 2))}
    assert sum(table['class_counts'].values()) == 81 and table['class_counts']['L'] == table['class_counts']['R']
    first = table['rows'][0]                                              # 1*1 against 1*1: balanced, difference 0
    assert (first['left_torque'], first['right_torque'], first['torque_difference']) == (1., 1., 0.)
    assert (first['true_label'], first['rule_label'], first['bucket_index']) == ('B', 'B', 0)
    biggest = max(table['rows'], key=lambda row: row['abs_torque_difference'])
    assert biggest['abs_torque_difference'] == 8. and biggest['bucket_index'] == ma.bucket_index(8)
    assert all(row['rule_agrees'] for row in table['rows'])
    assert all(row['abs_torque_difference'] == abs(row['left_torque'] - row['right_torque']) for row in table['rows'])
    equality = [row for row in table['rows'] if row['bucket_index'] == 0]
    assert {row['true_label'] for row in equality} == {'B'} and len(equality) == table['class_counts']['B']
    assert [entry['n_rows'] for entry in table['buckets']] == [sum(1 for row in table['rows'] if row['bucket_index'] == index)
                                                               for index in range(len(ma.BUCKETS))]
    assert table['buckets'][0]['class_counts'] == {'B': table['class_counts']['B'], 'L': 0, 'R': 0}


def test_check_torque_refuses_a_lattice_the_rule_does_not_reproduce():
    X, y, manifest = lattice()
    y[0] = 1                                                              # a balanced row relabelled as left-heavy
    table = ma.torque_table(X, y, manifest, SPEC)
    assert table['n_agreeing'] == 80 and table['agreement_share'] == pytest.approx(80/81)
    with pytest.raises(RunComparisonError, match='reproduces 80 of 81'):
        ma.check_torque(table, dict(SPEC, expected_rows=81, expected_class_counts=table['class_counts']), 'hand_built')


def test_check_torque_refuses_row_or_class_counts_that_differ_from_the_declared_structure():
    X, y, manifest = lattice()
    table = ma.torque_table(X, y, manifest, SPEC)
    with pytest.raises(RunComparisonError, match='declared structure is 625 rows'):
        ma.check_torque(table, SPEC, 'hand_built')                        # the balance-scale spec, a different lattice
    wrong = dict(SPEC, expected_rows=81, expected_class_counts={**table['class_counts'], 'B': 0})
    with pytest.raises(RunComparisonError, match='declared structure is 81 rows'):
        ma.check_torque(table, wrong, 'hand_built')
    ma.check_torque(table, dict(SPEC, expected_rows=81, expected_class_counts=table['class_counts']), 'hand_built')


def test_torque_table_refuses_features_or_labels_the_prepared_data_does_not_carry():
    X, y, manifest = lattice()
    with pytest.raises(RunComparisonError, match="missing features \\['left-weight'\\]"):
        ma.torque_table(X, y, {**manifest, 'feature_names': ['a', 'left-distance', 'right-weight', 'right-distance']}, SPEC)
    with pytest.raises(RunComparisonError, match='labels not in the label map'):
        ma.torque_table(X, y, {**manifest, 'label_map': ['B', 'L', 'X']}, SPEC)


@needs_real_balance_scale
def test_the_real_balance_scale_lattice_is_the_declared_625_row_structure():
    X, y, manifest, _ = load_prepared(REAL.parent, 'balance_scale')
    table = ma.torque_table(X, y, manifest, SPEC)
    ma.check_torque(table, SPEC, 'balance_scale')
    assert (table['n_rows'], table['n_agreeing'], table['agreement_share']) == (625, 625, 1.)
    assert table['class_counts'] == {'B': 49, 'L': 288, 'R': 288}
    assert table['buckets'][0]['n_rows'] == 49 and table['buckets'][0]['class_counts'] == {'B': 49, 'L': 0, 'R': 0}


# ----------------------------------------------------------------------------- confusion, recall and dispersion

def cells_of(assignments, samples):
    """{(repeat, fold, seed): (sample IDs, predictions)} from {(repeat, fold, seed): predicted labels}."""
    return {key: (np.asarray(samples, dtype=int), np.asarray(values, dtype=int)) for key, values in assignments.items()}


def test_confusion_counts_every_prediction_once_and_recall_is_its_diagonal_share():
    y = np.array([0, 0, 1, 1, 2])
    chosen = cells_of({(0, 0, 8129): [0, 1, 1, 1, 2], (0, 0, 19391): [0, 0, 1, 2, 2], (1, 0, 8129): [1, 1, 1, 1, 1]},
                      [0, 1, 2, 3, 4])
    pooled, fits = ma.confusion(chosen, y, 3)
    assert fits == 3 and pooled.sum() == 15 and pooled.tolist() == [[3, 3, 0], [0, 5, 1], [0, 1, 2]]
    assert ma.recall_of(pooled) == [.5, 5/6, 2/3]
    first, fold_fits = ma.confusion(chosen, y, 3, folds={(0, 0)})
    assert fold_fits == 2 and first.sum() == 10 and ma.recall_of(first) == [.75, .75, 1.]
    empty, none = ma.confusion(chosen, y, 3, folds={(2, 4)})
    assert none == 0 and ma.recall_of(empty) == [None, None, None]


def test_recall_dispersion_skips_the_folds_without_the_class():
    class Fake:
        folds = [(0, 0), (0, 1), (0, 2)]
    counts = {'folds': {(0, 0): np.array([[2, 0], [0, 2]]), (0, 1): np.array([[1, 1], [0, 2]]),
                        (0, 2): np.array([[0, 0], [1, 3]])}}
    dispersion = ma.recall_dispersion(Fake(), counts, 0)
    assert dispersion['fold_recall'] == [1., .5, None] and dispersion['n_folds_with_class'] == 2
    assert dispersion['fold_recall_mean'] == .75 and dispersion['fold_recall_sd'] == pytest.approx(np.std([1., .5], ddof=1))
    assert (dispersion['fold_recall_min'], dispersion['fold_recall_max']) == (.5, 1.)
    single = ma.recall_dispersion(Fake(), {'folds': {**counts['folds'], (0, 1): np.array([[0, 0], [0, 2]]),
                                                     (0, 2): np.array([[0, 0], [1, 3]])}}, 0)
    assert single['n_folds_with_class'] == 1 and single['fold_recall_sd'] is None


# ----------------------------------------------------------------------------- the command, end to end

@pytest.fixture(scope='session')
def analysed(chain, tmp_path_factory):                                    # noqa: F811
    output = tmp_path_factory.mktemp('mechanism')/'out'
    record = ma.analyse(chain.paths, output, datasets=BATCH1, torque_specs={}, frozen_protocols=chain.frozen)
    return record, output


def test_analyse_writes_every_output_and_seals_each_csv(analysed):
    record, output = analysed
    assert sorted(path.name for path in output.iterdir()) == sorted(ma.OUTPUTS)
    for name, seal in record['outputs'].items():
        assert seal['sha256'] == tcr.sha256(output/name) and seal['rows'] == len(read_csv(output/name))
    assert record['datasets'] == BATCH1 and record['models']['all'] == list(ma.MODELS)
    assert set(record['provenance']['runs']) == set(h.RUN_LABELS)
    assert record['provenance']['verification']['validators'] and 'no model fitted' in record['status']
    assert set(record['provenance']['published_analyses']) == set(ma.PUBLISHED_KEYS)


def test_every_rate_carries_the_fits_and_predictions_behind_it_and_the_seed_counts_differ(analysed):
    record, output = analysed
    counts = record['normalization']['by_model']
    deterministic = [model for model in ma.MODELS if counts[BATCH1[0]][model]['n_fitting_seeds'] == 1]
    assert sorted(deterministic) == sorted(['dummy', 'svc_rbf', 'numeric_knn'])
    for name in BATCH1:
        seeded = counts[name][ma.TRAINED]
        assert seeded['n_fitting_seeds'] == 3 and seeded['n_outer_fits'] == 3 * seeded['n_outer_folds']
        for model in deterministic:
            single = counts[name][model]
            assert single['n_outer_fits'] == single['n_outer_folds'] == 15
            assert 3 * single['n_predictions'] == seeded['n_predictions']     # a third of the predictions
    rows = read_csv(output/ma.CLASS_CSV)
    assert {row['status'] for row in rows} == {ma.DESCRIPTIVE} and 'never an independent sample' in ma.DESCRIPTIVE
    for row in rows:
        stated = counts[row['dataset']][row['model_id']]
        assert (int(row['n_outer_fits']), int(row['n_predictions'])) == (stated['n_outer_fits'], stated['n_predictions'])
        assert int(row['n_fitting_seeds']) == stated['n_fitting_seeds']
    pooled = [row for row in rows if row['scope'] == ma.POOLED]
    per_fold = [row for row in rows if row['scope'] == ma.PER_FOLD]
    assert len(per_fold) == 15 * len(pooled) and {row['outer_repeat'] for row in pooled} == {''}
    for row in pooled:                                                    # every pooled cell counts every prediction once
        stated = counts[row['dataset']][row['model_id']]
        assert int(row['predictions_in_scope']) == stated['n_predictions']
        assert int(row['fits_in_scope']) == stated['n_outer_fits']
    by_cell = {(row['dataset'], row['model_id'], row['true_label'], row['predicted_label']): row for row in pooled}
    for key, row in by_cell.items():
        folds = [int(other['n_predicted_as']) for other in per_fold
                 if (other['dataset'], other['model_id'], other['true_label'], other['predicted_label']) == key]
        assert sum(folds) == int(row['n_predicted_as']) and len(folds) == 15


def test_pooled_recall_matches_the_confusion_matrix_and_the_verified_predictions(analysed):
    record, output = analysed
    rows = read_csv(output/ma.CLASS_CSV)
    for name in BATCH1:
        for model in ma.MODELS:
            block = record['class_rates']['pooled'][name][model]
            matrix = np.array(block['confusion_matrix'])
            assert matrix.sum() == block['n_predictions'] and matrix.trace() == block['n_correct']
            assert block['pooled_accuracy'] == pytest.approx(block['n_correct']/block['n_predictions'])
            assert block['pooled_recall'] == ma.recall_of(matrix)
            for index, label in enumerate(block['labels']):
                cell = next(row for row in rows if row['scope'] == ma.POOLED and row['dataset'] == name
                            and row['model_id'] == model and row['true_label'] == label and row['predicted_label'] == label)
                assert float(cell['share_of_true']) == pytest.approx(block['pooled_recall'][index])
                assert int(cell['n_true_in_scope']) == block['n_true_pooled'][index]


def test_the_ladder_holds_the_five_rungs_with_the_verified_mean_errors_and_the_reading_follows_them(analysed):
    record, output = analysed
    rows = read_csv(output/ma.LADDER_CSV)
    for name in BATCH1:
        entries = record['representation_ladder'][name]
        assert [(entry['rung'], entry['model_id']) for entry in entries] == [(rung, model) for rung, model in nd.LADDER]
        for entry in entries:
            written = [row for row in rows if row['dataset'] == name and row['model_id'] == entry['model_id']]
            assert len(written) == len(entry['classes'])
            assert all(float(row['mean_error']) == pytest.approx(entry['mean_error']) for row in written)
    reading = {row['dataset']: row for row in read_csv(output/ma.READING_CSV)}
    for name in BATCH1:
        error = {entry['model_id']: entry['mean_error'] for entry in record['representation_ladder'][name]}
        row = reading[name]
        assert float(row['arrowflow_error']) == pytest.approx(error[ma.TRAINED])
        assert float(row['representation_cost_untrained_minus_projected']) == pytest.approx(error[ma.UNTRAINED] - error[ma.PROJECTED])
        assert float(row['training_gain_untrained_minus_arrowflow']) == pytest.approx(error[ma.UNTRAINED] - error[ma.TRAINED])
        assert float(row['arrowflow_minus_projected_error']) == pytest.approx(error[ma.TRAINED] - error[ma.PROJECTED])
        assert row['reading'] in ma.READINGS and row['status'] == ma.POST_HOC


def test_the_registered_contrasts_reproduce_the_published_family_values(analysed):
    record, output = analysed
    published = {(row['dataset'], row['model_a'], row['model_b']): row
                 for row in read_csv(Path(record['provenance']['published_analyses']['newdata_families_csv']['path']))}
    rows = read_csv(output/ma.CONTRAST_CSV)
    assert len(rows) == 3 * len(BATCH1)
    for row in rows:
        key = (row['dataset'], row['model_a'], row['model_b'])
        if row['registered'] == 'True':
            assert row['agrees_with_published'] == 'True' and row['status'] == ma.REGISTERED_STATUS
            for field in ('mean_difference', 'ci_low', 'ci_high', 'p_approximate'):
                mine = row['p_unadjusted'] if field == 'p_approximate' else row[field]
                assert float(mine) == pytest.approx(float(published[key][field]), abs=ma.REPRODUCTION_TOLERANCE)
                assert float(row[f'published_{field}']) == float(published[key][field])
        else:
            assert (row['model_a'], row['model_b']) == (ma.TRAINED, ma.PROJECTED) and row['status'] == ma.POST_HOC
            assert row['agrees_with_published'] == '' and row['published_family'] == ''
        assert (int(row['n_folds']), int(row['df'])) == (15, 14) and float(row['test_train_ratio']) == .25
        for side in ('a', 'b'):
            model = row[f'model_{side}']
            assert int(row[f'n_outer_fits_{side}']) == record['normalization']['by_model'][row['dataset']][model]['n_outer_fits']


def widths_from_summary(paths, label, name, model):
    """{(repeat, fold): widths} read straight from the run's saved model rows, independently of the analysis."""
    chosen = {}
    for row in load(paths[label]/'summary.json')['model_rows'][name]:
        if row['model_id'] == model:
            chosen.setdefault((row['outer_repeat'], row['outer_fold']), []).append(list(row['config']['widths']))
    assert all(len(set(map(str, values))) == 1 for values in chosen.values())   # one selection per outer fold
    return {fold: values[0] for fold, values in chosen.items()}


def test_architecture_matching_names_the_folds_where_the_two_selections_differ(chain, analysed):   # noqa: F811
    record, output = analysed
    rows = read_csv(output/ma.ARCHITECTURE_CSV)
    assert len(rows) == 15 * len(BATCH1)
    for name in BATCH1:
        entry = record['architecture_matching'][name]
        trained = widths_from_summary(chain.paths, 'batch1', name, ma.TRAINED)
        untrained = widths_from_summary(chain.paths, 'batch1', name, ma.UNTRAINED)
        expected = sorted(fold for fold in trained if trained[fold] != untrained[fold])
        differing = sorted(tuple(fold) for fold in entry['folds_with_different_widths'])
        assert differing == expected and 0 < len(differing) < 15     # the chain gives ArrowFlow [64, 128] on some folds
        assert entry['n_folds_with_different_widths'] == len(differing) and not entry['architecture_matched']
        assert entry['n_untrained_fits_on_those_folds'] == 3 * len(differing)
        assert entry['widths'][ma.TRAINED] == {str(widths): sum(1 for value in trained.values() if value == widths)
                                               for widths in sorted(map(list, {tuple(v) for v in trained.values()}))}
        written = [row for row in rows if row['dataset'] == name]
        assert sorted((int(row['outer_repeat']), int(row['outer_fold'])) for row in written if row['widths_match'] == 'False') == differing
        for row in written:
            fold = (int(row['outer_repeat']), int(row['outer_fold']))
            assert row['arrowflow_widths'] == str(trained[fold]) and row['untrained_widths'] == str(untrained[fold])
            assert int(row['arrowflow_hidden_layers']) == len(trained[fold])


def test_the_torque_table_is_absent_when_the_panel_holds_no_declared_lattice(analysed):
    record, output = analysed
    assert record['torque_structure'] == 'no declared torque dataset in this panel'
    assert read_csv(output/ma.TORQUE_CSV) == [] and (output/ma.TORQUE_CSV).read_text().strip() == ','.join(ma.TORQUE_COLUMNS)


def test_load_verified_panel_is_holistic_load_panel_with_the_prediction_cells_kept(chain):   # noqa: F811
    panel, cells, provenance = ma.load_verified_panel(chain.paths, frozen_protocols=chain.frozen)
    reference_panel, reference = h.load_panel(chain.paths, frozen_protocols=chain.frozen)
    assert provenance == reference and panel.datasets == reference_panel.datasets
    assert set(cells) == set(h.RUN_LABELS)
    key = (BATCH1[0], ma.TRAINED, *panel.folds[0], panel.seeds(BATCH1[0], ma.TRAINED)[0])
    assert key in cells['batch1'] and len(cells['batch1'][key]) == 3


# ----------------------------------------------------------------------------- refusals and all-or-none writing

def reseal(paths, rewrite):
    """Rewrite newdata_families.csv and update the sha256 newdata_analysis.json seals, so only the value changes."""
    csv_path = paths['newdata_analysis']/'newdata_families.csv'
    csv_path.write_text(rewrite(csv_path.read_text()))
    tcr.rewrite(paths['newdata_analysis']/'newdata_analysis.json',
                lambda record: record['outputs']['newdata_families.csv'].update(
                    sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest()))


def test_analyse_refuses_a_registered_contrast_that_differs_from_the_published_value(copy):   # noqa: F811
    def bend(text):
        rows = list(csv.DictReader(io.StringIO(text)))
        target = next(row for row in rows if row['dataset'] == BATCH1[0] and row['model_b'] == ma.UNTRAINED)
        target['mean_difference'] = repr(float(target['mean_difference']) + 1e-6)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()
    reseal(copy.paths, bend)
    with pytest.raises(RunComparisonError, match='does not reproduce the published newdata family value'):
        ma.analyse(copy.paths, copy.out, datasets=BATCH1, torque_specs={}, frozen_protocols=copy.frozen)
    assert not copy.out.exists()


def test_analyse_refuses_a_published_family_csv_that_its_record_does_not_seal(copy):          # noqa: F811
    path = copy.paths['newdata_analysis']/'newdata_families.csv'
    path.write_text(path.read_text().replace('primary', 'primary '))
    with pytest.raises(RunComparisonError, match='is not the file'):
        ma.analyse(copy.paths, copy.out, datasets=BATCH1, torque_specs={}, frozen_protocols=copy.frozen)
    assert not copy.out.exists()


def test_analyse_refuses_a_run_whose_saved_predictions_do_not_verify(copy):                   # noqa: F811
    job = next(job for job in load(copy.paths['batch1']/'planned_jobs.json')
               if job['dataset_id'] == BATCH1[0] and job['model_id'] == ma.TRAINED)
    tcr.rewrite(copy.paths['batch1']/job['result_file'],
                lambda result: result['predictions'][0].update(y_pred=(result['predictions'][0]['y_pred'] + 1) % 3))
    with pytest.raises(RunComparisonError):
        ma.analyse(copy.paths, copy.out, datasets=BATCH1, torque_specs={}, frozen_protocols=copy.frozen)
    assert not copy.out.exists()


def test_analyse_refuses_a_missing_run_and_the_command_exits_2_without_writing(copy, tmp_path, capsys):   # noqa: F811
    (copy.paths['projected']/'summary.json').unlink()
    with pytest.raises(RunComparisonError, match='incomplete or unverified'):
        ma.analyse(copy.paths, copy.out, datasets=BATCH1, torque_specs={}, frozen_protocols=copy.frozen)
    assert not copy.out.exists()
    with pytest.raises(SystemExit) as refused:
        ma.main(['analyse', '--runs', str(tmp_path/'absent'), '--output', str(tmp_path/'cli')])
    assert refused.value.code == 2 and 'mechanism_analysis analyse refused' in capsys.readouterr().err
    assert not (tmp_path/'cli').exists()


def test_analyse_refuses_an_unusable_output_before_it_reads_any_run(chain, tmp_path, monkeypatch):   # noqa: F811
    read = []
    monkeypatch.setattr(ma, 'load_run', lambda *args, **kwargs: read.append(args))
    blocked = tmp_path/'out'
    blocked.write_text('not a directory')
    with pytest.raises(RunComparisonError, match='exists and is not a directory'):
        ma.analyse(chain.paths, blocked, datasets=BATCH1, torque_specs={}, frozen_protocols=chain.frozen)
    occupied = tmp_path/'other'
    occupied.mkdir()
    (occupied/ma.CONTRAST_CSV).mkdir()
    with pytest.raises(RunComparisonError, match='exists and is not a regular file'):
        ma.analyse(chain.paths, occupied, datasets=BATCH1, torque_specs={}, frozen_protocols=chain.frozen)
    assert read == []


def test_analyse_refuses_a_dataset_the_verified_panel_does_not_hold(chain, tmp_path):        # noqa: F811
    with pytest.raises(RunComparisonError, match='The verified panel holds no not_a_dataset'):
        ma.analyse(chain.paths, tmp_path/'out', datasets=(BATCH1[0], 'not_a_dataset'), torque_specs={},
                   frozen_protocols=chain.frozen)
    with pytest.raises(RunComparisonError, match='No dataset was declared'):
        ma.analyse(chain.paths, tmp_path/'out2', datasets=(), torque_specs={}, frozen_protocols=chain.frozen)
    assert not (tmp_path/'out').exists() and not (tmp_path/'out2').exists()


def test_the_outputs_are_written_all_or_none_and_a_differing_file_is_never_replaced(chain, analysed, tmp_path):   # noqa: F811
    record, written = analysed
    output = tmp_path/'out'
    output.mkdir()
    (output/ma.CONTRAST_CSV).write_text('another analysis\n')
    with pytest.raises(FileExistsError, match=ma.CONTRAST_CSV):
        ma.analyse(chain.paths, output, datasets=BATCH1, torque_specs={}, frozen_protocols=chain.frozen)
    assert [path.name for path in output.iterdir()] == [ma.CONTRAST_CSV]
    assert (output/ma.CONTRAST_CSV).read_text() == 'another analysis\n'
    identical = tmp_path/'again'
    identical.mkdir()
    for name in ma.OUTPUTS:                                               # rewriting the same content is accepted
        (identical/name).write_bytes((written/name).read_bytes())
    assert ma.analyse(chain.paths, identical, datasets=BATCH1, torque_specs={},
                      frozen_protocols=chain.frozen)['outputs'] == record['outputs']
