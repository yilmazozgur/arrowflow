"""diagnostics_tables: the random references, the pure table builders on hand-built view records, and the tables command on a
synthetic training-diagnostics smoke run with its synthetic ablations (built once per session with two workers; never evidence).
Refusal and output tests work on copies."""
from collections import Counter
import csv
from fractions import Fraction
import io
import itertools
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.make_revision import diagnostics_tables as dt
from experiments.make_revision import training_diagnostics as td


# ----------------------------------------------------------------------------- random references

def test_random_displacement_reference_is_the_exact_expected_footrule():
    for n in range(2, 6):                                     # two independent uniform permutations, every pair
        permutations = [np.asarray(p) for p in itertools.permutations(range(n))]
        total = sum(int(np.abs(a - b).sum()) for a in permutations for b in permutations)
        assert Fraction(total, len(permutations) ** 2) == Fraction(n * n - 1, 3)
        assert dt.random_displacement_reference(n) == float(Fraction(total, len(permutations) ** 2) / (n * n // 2))
    for n in (6, 7):                                          # against a fixed permutation the distribution is the same
        values = [int(np.abs(np.asarray(p) - np.arange(n)).sum()) for p in itertools.permutations(range(n))]
        assert Fraction(sum(values), len(values)) == Fraction(n * n - 1, 3) and max(values) == n * n // 2
        assert len(set(values)) == dt.distinct_values_bound(n) and all(value % 2 == 0 for value in values)
    assert [dt.random_displacement_reference(n) for n in (16, 32, 64, 128)] == [85 / 128, 341 / 512, 1365 / 2048, 5461 / 8192]
    assert dt.random_displacement_reference(7) == dt.random_displacement_reference(9) == 2 / 3
    with pytest.raises(ValueError):
        dt.random_displacement_reference(1)


def exhaustive_ties(n, width):
    """Every input permutation against every width-tuple of filters: the exact expected tie measures under the
    training_diagnostics definitions (responses are cityblock distances of position vectors)."""
    permutations = np.asarray(list(itertools.permutations(range(n))))
    distances = np.abs(permutations[:, None, :] - permutations[None, :, :]).sum(axis=2)      # input x filter
    rows = [distances[x, list(combo)] for x in range(len(permutations))
            for combo in itertools.product(range(len(permutations)), repeat=width)]
    return td.response_ties(np.asarray(rows, dtype=float)), Counter(distances[0].tolist())


@pytest.mark.parametrize('n, width', [(3, 2), (3, 4), (4, 3)])
def test_tie_reference_simulation_is_deterministic_and_agrees_with_the_tie_definition(n, width):
    (tied, distinct), footrules = exhaustive_ties(n, width)
    p = np.asarray(list(footrules.values()), dtype=float) / sum(footrules.values())
    assert tied == pytest.approx(1 - np.sum(p * (1 - p) ** (width - 1)), abs=1e-12)          # W i.i.d. responses per row
    assert distinct == pytest.approx(np.sum(1 - (1 - p) ** width) / width, abs=1e-12)
    simulated = dt.random_tie_reference(n, width, seed=7, filter_sets=400, rows_per_set=50)
    assert simulated == dt.random_tie_reference(n, width, seed=7, filter_sets=400, rows_per_set=50)
    assert simulated != dt.random_tie_reference(n, width, seed=8, filter_sets=400, rows_per_set=50)
    assert abs(simulated['tied_response_share'] - tied) <= 5 * simulated['tied_response_share_mc_se'] + 1e-9
    assert abs(simulated['distinct_response_ratio'] - distinct) <= 5 * simulated['distinct_response_ratio_mc_se'] + 1e-9


def test_default_tie_reference_reproduces_the_ruled_figures():
    shares = [dt.random_tie_reference(n, 128)['tied_response_share'] for n in (16, 32, 64)]
    assert shares == pytest.approx([.95, .79, .47], abs=.01)
    assert dt.random_tie_reference(16, 128) == dt.random_tie_reference(16, 128)
    assert dt.distinct_values_bound(16) == 65


# ----------------------------------------------------------------------------- builders on hand-built records

def view_record(dataset, fold, view, checkpoint, displacements, *, widths=(8,), embed_dim=16, iterations=40):
    names = [f'hidden_{index}' for index in range(len(widths))] + ['output']
    items = [embed_dim, *widths]
    return {'protocol_id': 'p', 'dataset_id': dataset, 'reference': 'r', 'outer_repeat': 0, 'outer_fold': fold, 'model_seed': 8129,
            'widths': tuple(widths), 'widths_text': json.dumps(list(widths), separators=(',', ':')), 'embed_dim': embed_dim,
            'view': view, 'iterations': iterations, 'n_validation_samples': 12, 'checkpoint': checkpoint,
            'layers': {name: {'layer': name, 'n': items[index], 'n_filters': widths[index] if index < len(widths) else 3,
                              'at_checkpoint': {'displacement_from_initial': displacements[index],
                                                'changed_share_from_initial': 1.0 if displacements[index] else 0.0}}
                       for index, name in enumerate(names)}}


def test_t1_and_t2_exclude_checkpoint_zero_views():
    views = [view_record('a', 0, 0, 0, [0., 0.]), view_record('a', 0, 1, 30, [.3, .1]), view_record('a', 1, 0, 0, [0., 0.]),
             view_record('a', 1, 1, 10, [.5, .2]), view_record('b', 0, 0, 20, [.4, .3], embed_dim=32)]
    rows, counts = dt.t1_rows(views)
    a = next(row for row in rows if row['dataset_id'] == 'a')
    assert (a['n_folds'], a['n_views'], a['n_views_checkpoint_0'], a['share_checkpoint_0'], a['n_views_later_checkpoint']) == (2, 4, 2, .5, 2)
    assert (a['checkpoint_median'], a['checkpoint_q25'], a['checkpoint_q75'], a['checkpoint_iqr']) == (5., 0., 15., 15.)
    assert (a['later_checkpoint_median'], a['later_checkpoint_q25'], a['later_checkpoint_q75'], a['later_checkpoint_iqr']) == (20., 15., 25., 10.)
    pooled = next(row for row in rows if row['scope'] == 'pooled')
    assert (pooled['n_datasets'], pooled['n_views'], pooled['n_views_checkpoint_0'], pooled['later_checkpoint_median']) == (2, 5, 2, 20.)
    assert [(row['checkpoint_iteration'], row['n_views'], row['share_of_views']) for row in counts if row.get('dataset_id') == 'a'] == [
        (0, 2, .5), (10, 1, .25), (30, 1, .25)]
    rows = dt.t2_rows(views)
    hidden = next(row for row in rows if row['dataset_id'] == 'a' and row['layer'] == 'hidden_0')
    assert (hidden['n'], hidden['n_views'], hidden['n_views_checkpoint_0'], hidden['share_checkpoint_0'], hidden['n_views_later_checkpoint']) == (
        16, 4, 2, .5, 2)
    assert hidden['displacement_mean_later'] == pytest.approx(.4) and hidden['displacement_sd_later'] == pytest.approx(np.std([.3, .5], ddof=1))
    assert hidden['changed_share_mean_later'] == 1. and hidden['random_displacement_reference'] == 85 / 128
    output = next(row for row in rows if row['dataset_id'] == 'a' and row['layer'] == 'output')
    assert output['n'] == 8 and output['displacement_mean_later'] == pytest.approx(.15)
    assert [(row['n'], row['n_views'], row['n_views_later_checkpoint']) for row in rows if row['scope'] == 'pooled' and row['layer'] == 'hidden_0'] == [
        (16, 4, 2), (32, 1, 1)]
    only_zero = dt.t2_rows([view_record('c', 0, 0, 0, [0., 0.])])
    assert only_zero[0]['displacement_mean_later'] is None and only_zero[0]['share_checkpoint_0'] == 1. and only_zero[0]['n_views_later_checkpoint'] == 0
    assert 'displacement_mean_later,' in dt.render_csv('t2_displacement.csv', only_zero) and ',,,' in dt.render_csv('t2_displacement.csv', only_zero)


def test_t3_reports_ties_by_permutation_length_beside_the_random_reference():
    views = [view_record('a', 0, 0, 0, [0., 0.]), view_record('a', 0, 1, 5, [.1, .1], embed_dim=32), view_record('a', 1, 0, 5, [.1, .1])]
    for view, (initial, checkpoint) in zip(views, [(.9, .8), (.7, .6), (.5, .4)]):
        for layer in view['layers'].values():
            hidden = layer['layer'] != 'output'
            layer['ties'] = {network: {'tied_response_share': value if hidden else None, 'distinct_response_ratio': .5 if hidden else None,
                                       'tied_nearest_share': None if hidden else .1 * (network == 'checkpoint')}
                             for network, value in (('initial', initial), ('checkpoint', checkpoint))}
    assert sorted(dt.tie_references(views)) == [(16, 8), (32, 8)]
    references = {(16, 8): {'tied_response_share': .95, 'distinct_response_ratio': .2}, (32, 8): {'tied_response_share': .75, 'distinct_response_ratio': .4}}
    rows = dt.t3_rows(views, references)
    hidden = [row for row in rows if row['scope'] == 'dataset' and row['layer'] == 'hidden_0']
    assert [(row['n'], row['n_filters'], row['n_views'], row['tied_response_share_initial'], row['tied_response_share_checkpoint'],
             row['random_tied_response_share']) for row in hidden] == [(16, 8, 2, pytest.approx(.7), pytest.approx(.6), .95), (32, 8, 1, .7, .6, .75)]
    assert hidden[0]['distinct_values_bound'] == 65 and hidden[0]['distinct_response_ratio_bound'] == 1. and 'tied_nearest_share_initial' not in hidden[0]
    output = [row for row in rows if row['scope'] == 'dataset' and row['layer'] == 'output']
    assert len(output) == 1 and (output[0]['n'], output[0]['n_filters'], output[0]['tied_nearest_share_initial'],
                                 output[0]['tied_nearest_share_checkpoint']) == (8, 3, 0., pytest.approx(.1))


def fold_record(accuracy, view_accuracies, checkpoints):
    return {'protocol_id': 'p', 'dataset_id': 'a', 'reference': 'r', 'outer_repeat': 0, 'outer_fold': 0, 'model_seed': 8129, 'widths': (8,),
            'widths_text': '[8]', 'embed_dim': 16, 'majority_checkpoint': accuracy,
            'views': [{'checkpoint': c, 'at_checkpoint': {'knn_test_accuracy': value}} for c, value in zip(checkpoints, view_accuracies)]}


def test_t6_requires_the_reference_seed_accuracy_and_reports_trained_minus_untrained():
    fold = fold_record(.8, [.7, .75, .8], [0, 3, 0])
    ablation = {'n_test': 20, 'untrained_accuracy': .75, 'views7_accuracy': .8, 'reference_accuracy': .8, 'sealed_reference_accuracy': .8,
                'view_accuracies': [.7, .75, .8]}
    rows, fold_rows = dt.t6_rows([fold], {('a', 0, 0): ablation})
    assert (rows[0]['n_folds'], rows[0]['difference_sd']) == (1, None) and rows[0]['difference_mean'] == pytest.approx(.05)
    assert (fold_rows[0]['n_views_checkpoint_0'], fold_rows[0]['trained_accuracy'], fold_rows[0]['reference_accuracy']) == (2, .8, .8)
    for change in ({'reference_accuracy': .75}, {'views7_accuracy': .85}, {'sealed_reference_accuracy': .7}, {'view_accuracies': [.7, .8, .8]}):
        with pytest.raises(dt.Refusal, match='reference model'):
            dt.t6_rows([fold], {('a', 0, 0): {**ablation, **change}})
    with pytest.raises(dt.Refusal, match='no verified ablation record'):
        dt.t6_rows([fold], {})


# ----------------------------------------------------------------------------- the command on the synthetic smoke run

@pytest.fixture(scope='session')
def smoke(tmp_path_factory):
    """training_diagnostics.smoke with two workers (never evidence); DIAGNOSTICS_TABLES_SMOKE_ROOT names a prebuilt root instead,
    which the tests only read."""
    prebuilt = os.environ.get('DIAGNOSTICS_TABLES_SMOKE_ROOT')
    root = Path(prebuilt) if prebuilt else tmp_path_factory.mktemp('diagnostics_tables_smoke')
    if not prebuilt:
        td.smoke(root, td.draft_protocol(), workers=2)
    return SimpleNamespace(root=root, diagnostics=root/'diagnostics',
                           ablations={'smoke_bridge_knn': root/'synthetic_knn_ablation', 'smoke_newdata_batch1': root/'synthetic_newdata_ablation',
                                      'smoke_newdata_batch2': root/'synthetic_newdata_ablation'})


def command(diagnostics, ablations, output, *extra):
    argv = ['tables', '--diagnostics', str(diagnostics), '--output', str(output)]
    for name, directory in ablations.items():
        argv += ['--ablation', f'{name}={directory}']
    return dt.main(argv + list(extra))


def read(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text())))


@pytest.fixture(scope='session')
def tables_output(smoke, tmp_path_factory):
    output = tmp_path_factory.mktemp('diagnostics_tables')/'tables'
    assert command(smoke.diagnostics, smoke.ablations, output, '--allow-smoke') == 0
    return output


def test_the_tables_of_the_smoke_run(smoke, tables_output):
    document = json.loads((tables_output/dt.OUTPUT_JSON).read_text())
    assert document['synthetic_smoke_only'] is True and document['allow_smoke'] is True and sorted(path.name for path in tables_output.iterdir()) == sorted(dt.FILES + (dt.OUTPUT_JSON,))
    for name in dt.FILES:
        assert document['tables'][name]['sha256'] == td.sha256_file(tables_output/name)
        assert next(csv.reader(io.StringIO((tables_output/name).read_text()))) == [column for column, _ in dt.COLUMNS[name]]
    assert document['checks']['ablation_job_records_validated'] == 9 == document['checks']['folds_with_checkpoint_accuracy_equal_to_the_reference']
    checkpoints, snapshots = read(smoke.diagnostics/'checkpoints.csv'), read(smoke.diagnostics/'snapshots.csv')
    t1 = read(tables_output/'t1_checkpoints.csv')
    for dataset in ('synthetic', 'synthetic_b1', 'synthetic_b2'):
        values = [int(row['checkpoint_iteration']) for row in checkpoints if row['dataset_id'] == dataset]
        later, row = [value for value in values if value > 0], next(row for row in t1 if row['dataset_id'] == dataset)
        assert int(row['n_views']) == len(values) == 21 and int(row['n_views_checkpoint_0']) == values.count(0)
        assert float(row['checkpoint_median']) == np.median(values) and float(row['later_checkpoint_median']) == np.median(later)
    assert sum(int(row['n_views']) for row in read(tables_output/'t1_checkpoint_counts.csv') if row['scope'] == 'pooled') == 63
    moved = {(row['dataset_id'], row['outer_repeat'], row['outer_fold'], row['view']) for row in checkpoints if row['checkpoint_iteration'] != '0'}
    direct = [float(row['value']) for row in snapshots if row['dataset_id'] == 'synthetic_b2' and row['snapshot'] == 'checkpoint'
              and row['layer'] == 'hidden_0' and row['measure'] == 'displacement_from_initial'
              and (row['dataset_id'], row['outer_repeat'], row['outer_fold'], row['view']) in moved]
    t2 = read(tables_output/'t2_displacement.csv')
    row = next(row for row in t2 if row['dataset_id'] == 'synthetic_b2' and row['layer'] == 'hidden_0')
    assert int(row['n_views_later_checkpoint']) == len(direct) > 0 and float(row['displacement_mean_later']) == pytest.approx(np.mean(direct))
    items = set()
    for path in (smoke.diagnostics/'jobs').iterdir():                  # the permutation length n is each layer's n_items
        record = json.loads(path.read_text())
        items |= {(record['identity']['dataset_id'], json.dumps(view['widths'], separators=(',', ':')), layer['layer'], layer['n_items'])
                  for view in record['views'] for layer in view['snapshots'][0]['layers']}
    assert {(row['dataset_id'], row['widths'], row['layer'], int(row['n'])) for row in t2 if row['scope'] == 'dataset'} == items
    t3 = read(tables_output/'t3_ties.csv')
    assert all((row['random_tied_response_share'] != '') == (row['layer'] != 'output') != (row['tied_nearest_share_initial'] != '') for row in t3)
    ties = read(smoke.diagnostics/'ties.csv')
    row = next(row for row in t3 if row['scope'] == 'pooled' and row['widths'] == '[4]' and row['layer'] == 'hidden_0')
    direct = [float(tie['tied_response_share']) for tie in ties if tie['dataset_id'] in ('synthetic_b1', 'synthetic_b2')
              and tie['layer'] == 'hidden_0' and tie['network'] == 'checkpoint']
    assert (int(row['n']), int(row['n_filters']), int(row['n_views'])) == (16, 4, 42) and float(row['tied_response_share_checkpoint']) == pytest.approx(np.mean(direct))
    assert float(row['random_tied_response_share']) == dt.random_tie_reference(16, 4)['tied_response_share']
    t4 = read(tables_output/'t4_relabel.csv')
    assert [(row['level'], row['n_units']) for row in t4 if row['dataset_id'] == 'synthetic'] == [('view', '21'), ('seven_view_majority', '3')]
    curves = read(tables_output/'t5_learning_curves.csv')
    initial = [row for row in curves if row['iteration'] == '0']
    assert {row['point'] for row in initial if row['measure'] == 'knn_test_accuracy'} == {dt.INITIAL_KNN}
    assert {row['point'] for row in initial if row['measure'] != 'knn_test_accuracy'} == {dt.INITIAL}
    assert [row['iteration'] for row in curves if row['dataset_id'] == 'synthetic' and row['level'] == 'seven_view_majority'] == ['0', '10', '20', '23', '']
    majority = {(row['dataset_id'], row['outer_repeat'], row['outer_fold']): float(row['value']) for row in snapshots
                if row['view'] == 'majority' and row['snapshot'] == 'checkpoint'}
    folds = read(tables_output/'t6_trained_untrained_folds.csv')
    assert {(row['dataset_id'], row['outer_repeat'], row['outer_fold']): float(row['trained_accuracy']) for row in folds} == majority
    for row in folds:
        result = json.loads((smoke.ablations[row['reference']]/'results'/f"{row['dataset_id']}__r{row['outer_repeat']}f{row['outer_fold']}.json").read_text())
        untrained = next(model['accuracy'] for model in result['models'] if (model['variant_id'], model['model_seed']) == ('untrained', 8129))
        assert float(row['untrained_accuracy']) == untrained and float(row['difference']) == pytest.approx(float(row['trained_accuracy']) - untrained)
    t6 = read(tables_output/'t6_trained_untrained.csv')
    assert [row['dataset_id'] for row in t6] == ['synthetic', 'synthetic_b1', 'synthetic_b2'] and all(row['n_folds'] == '3' for row in t6)
    validation = read(tables_output/'t5_validation_curves.csv')
    initial_errors = [float(row['initial_validation_error']) for row in checkpoints if row['dataset_id'] == 'synthetic']
    assert float(next(row for row in validation if row['dataset_id'] == 'synthetic' and row['iteration'] == '0')['validation_error_mean']) == pytest.approx(np.mean(initial_errors))


def copied(source, target):
    return Path(shutil.copytree(source, target))


def edit_json(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def refused(capsys, diagnostics, ablations, output, match, *extra):
    assert command(diagnostics, ablations, output, *extra) == 2
    assert match in capsys.readouterr().err and not Path(output).exists()


def test_refuses_a_missing_or_failing_provenance(smoke, tmp_path, capsys):
    missing = copied(smoke.diagnostics, tmp_path/'missing')
    (missing/td.PROVENANCE_FILE).unlink()
    refused(capsys, missing, smoke.ablations, tmp_path/'out', 'provenance.json is missing', '--allow-smoke')
    totals = copied(smoke.diagnostics, tmp_path/'totals')
    edit_json(totals/td.PROVENANCE_FILE, lambda p: p['check_totals']['rng_untouched'].update(passed=8))
    refused(capsys, totals, smoke.ablations, tmp_path/'out', 'rng_untouched passed 8 of 9', '--allow-smoke')
    job = copied(smoke.diagnostics, tmp_path/'job')
    edit_json(job/td.PROVENANCE_FILE, lambda p: p['jobs']['synthetic_b1__r0f1']['checks']['checkpoint_orders'].update(passed=False))
    refused(capsys, job, smoke.ablations, tmp_path/'out', 'synthetic_b1__r0f1: checkpoint_orders', '--allow-smoke')
    refused(capsys, smoke.diagnostics, smoke.ablations, tmp_path/'out', 'not a frozen diagnostics run')


def test_refuses_files_that_differ_from_the_provenance(smoke, tmp_path, capsys):
    table = copied(smoke.diagnostics, tmp_path/'table')
    (table/'ties.csv').write_text((table/'ties.csv').read_text().replace(',80,', ',81,', 1))
    refused(capsys, table, smoke.ablations, tmp_path/'out', 'ties.csv differs from its recorded sha256', '--allow-smoke')
    record = copied(smoke.diagnostics, tmp_path/'record')
    with (record/'jobs'/'synthetic__r0f2.json').open('a') as stream:
        stream.write(' ')
    refused(capsys, record, smoke.ablations, tmp_path/'out', 'jobs/synthetic__r0f2.json differs', '--allow-smoke')
    artifact = copied(smoke.diagnostics, tmp_path/'artifact')
    (artifact/'artifacts'/'synthetic_b2__r0f0.npz').unlink()
    refused(capsys, artifact, smoke.ablations, tmp_path/'out', 'artifacts/synthetic_b2__r0f0.npz is missing', '--allow-smoke')
    summary = copied(smoke.diagnostics, tmp_path/'summary')
    (summary/td.SUMMARY_FILE).unlink()
    refused(capsys, summary, smoke.ablations, tmp_path/'out', 'diagnostics_summary.json is missing', '--allow-smoke')


def test_refuses_an_ablation_that_fails_its_verification(smoke, tmp_path, capsys):
    ablation = copied(smoke.ablations['smoke_bridge_knn'], tmp_path/'ablation')
    edit_json(ablation/'results'/'synthetic__r0f1.json',
              lambda r: next(m for m in r['models'] if (m['variant_id'], m['model_seed']) == ('untrained', 8129)).update(accuracy=1.))
    refused(capsys, smoke.diagnostics, {**smoke.ablations, 'smoke_bridge_knn': ablation}, tmp_path/'out',
            'the record of synthetic r0f1 fails its verification', '--allow-smoke')
    refused(capsys, smoke.diagnostics, {**smoke.ablations, 'smoke_bridge_knn': smoke.ablations['smoke_newdata_batch1']}, tmp_path/'out',
            'is not the ablation run that sealed the selections', '--allow-smoke')
    refused(capsys, smoke.diagnostics, {name: path for name, path in smoke.ablations.items() if name != 'smoke_newdata_batch2'},
            tmp_path/'out', 'missing: smoke_newdata_batch2', '--allow-smoke')


def test_outputs_are_all_or_none_and_never_replace_different_content(smoke, tables_output, tmp_path, capsys):
    again = copied(tables_output, tmp_path/'again')
    assert command(smoke.diagnostics, smoke.ablations, again, '--allow-smoke') == 0
    assert all((again/name).read_bytes() == (tables_output/name).read_bytes() for name in dt.FILES + (dt.OUTPUT_JSON,))
    changed = (again/'t3_ties.csv').read_text() + 'x\n'
    (again/'t3_ties.csv').write_text(changed)
    (again/'t5_learning_curves.csv').unlink()
    assert command(smoke.diagnostics, smoke.ablations, again, '--allow-smoke') == 2 and 't3_ties.csv' in capsys.readouterr().err
    assert (again/'t3_ties.csv').read_text() == changed and not (again/'t5_learning_curves.csv').exists()
    assert sorted(path.name for path in again.iterdir()) == sorted(set(dt.FILES + (dt.OUTPUT_JSON,)) - {'t5_learning_curves.csv'})
    fresh = tmp_path/'fresh'
    fresh.mkdir()
    (fresh/'t4_relabel.csv').write_text('x\n')
    assert command(smoke.diagnostics, smoke.ablations, fresh, '--allow-smoke') == 2
    assert [path.name for path in fresh.iterdir()] == ['t4_relabel.csv'] and (fresh/'t4_relabel.csv').read_text() == 'x\n'
