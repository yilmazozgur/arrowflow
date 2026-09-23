"""Training diagnostics of ArrowFlow-kNN (training_diagnostics): the pure numpy measures, the relabeling, the non-invasive
recorder, the synthetic smoke run (built once per session; tests that alter records work on copies) and the protocol.
The real reference runs and cached OpenML downloads are used where they are on this machine; no real job is fitted."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.datasets import get_data_home
from arrowflow.ranking import inverse_positions
from experiments.make_revision import newdata as nd
from experiments.make_revision import training_diagnostics as td
from experiments.make_revision.comparisons import StableFootruleKNN
from experiments.make_revision.knn_controls import reference_pins
from experiments.make_revision.models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from experiments.make_revision.multiview import MultiViewArrowFlowKNN
from experiments.make_revision.run_revision import load_prepared

REPO = Path(__file__).resolve().parents[2]
REAL = {name: tuple(Path(d) for d in directories) for name, directories in td.DEFAULT_SOURCES.items()}
needs_real = pytest.mark.skipif(not all((run/'summary.json').is_file() and (ablation/'reference_selections.json').is_file()
                                        for run, ablation in REAL.values()), reason='the reference runs are not on this machine')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tiny_data(n=90, seed=5):
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], n // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    return X, y


def tiny_network(widths=(6, 5), iterations=12, validation_ratio=.2):
    X, y = tiny_data()
    orders = OrdinalEncoder('random', 8, 1, .3, 11).fit(X, y).transform(X)
    net = ArrowFlowEstimator(embed_dim=8, widths=list(widths), iterations=iterations, learning_rate=.3, seed=17,
                             validation_ratio=validation_ratio)
    return net, orders, y


# ----------------------------------------------------------------------------- pure numpy measures

def test_network_forward_is_the_core_rule():
    net, orders, y = tiny_network()
    net.fit_orders(orders, y)
    layers = td.network_forward(inverse_positions(orders), td.filter_copy(net.network_)['matrices'])
    depth = net.transform_orders_by_depth(orders)
    assert len(layers) == 3 and all(np.array_equal(layers[l][2], depth[l]) for l in range(2))
    assert np.array_equal(net.classes_[layers[-1][1][:, 0]], net.predict_orders(orders))


def test_displacement_is_the_normalized_footrule_and_the_changed_share():
    identity = np.tile(np.arange(7, dtype=float), (3, 1))
    reversed_ = identity[:, ::-1].copy()
    moved = identity.copy()
    moved[0, [0, 1]] = [1, 0]
    assert td.displacement(identity, identity) == (0.0, 0.0)
    assert np.abs(reversed_ - identity).sum(axis=1).tolist() == [7 * 7 // 2] * 3 and td.displacement(reversed_, identity) == (1.0, 1.0)
    assert td.displacement(moved, identity) == pytest.approx((2 / 24 / 3, 1 / 3))
    with pytest.raises(ValueError, match='same layer'):
        td.displacement(identity, identity[:, :6])


def test_tie_measures_on_constructed_responses():
    tied, distinct = td.response_ties(np.asarray([[1., 2., 2., 5.], [3., 3., 3., 3.], [0., 1., 2., 3.]]))
    assert tied == pytest.approx(6 / 12) and distinct == pytest.approx((3 / 4 + 1 / 4 + 1) / 3)
    assert td.nearest_ties(np.asarray([[1., 1., 3.], [0., 2., 2.], [2., 2., 2.]])) == pytest.approx(2 / 3)


def test_checkpoint_is_the_last_strict_improvement_and_the_schedule():
    assert td.checkpoint_from_curve(.5, [.6, .5, .5]) == (0, [], [.5, .5, .5, .5])
    assert td.checkpoint_from_curve(.5, [.4, .4, .3, .35, .3]) == (3, [1, 3], [.5, .4, .4, .3, .3, .3])
    assert td.scheduled_iterations(23) == [0, 10, 20, 23] and td.scheduled_iterations(200) == list(range(0, 201, 10))
    assert td.previous_iteration(15, [0, 10, 20]) == 10 and td.previous_iteration(20, [0, 10, 20]) == 10
    assert td.previous_iteration(0, [0, 10]) is None


# ----------------------------------------------------------------------------- relabeling

def tie_free_rows(matrices, rng, count):
    rows = []
    while len(rows) < count:
        row = rng.permutation(matrices[0].shape[1])
        if all(td.response_ties(responses)[0] == 0 for responses, _, _ in td.network_forward(row[None], matrices)):
            rows.append(row)
    return np.stack(rows)


def test_relabeling_changes_nothing_without_ties_and_changes_tie_broken_orders_with_ties():
    rng = np.random.RandomState(3)
    matrices = [np.stack([rng.permutation(16) for _ in range(5)]).astype(float),
                np.stack([rng.permutation(5) for _ in range(3)]).astype(float)]
    positions = tie_free_rows(matrices, rng, 12)
    y = np.asarray([0, 1] * 6)
    layers = td.network_forward(positions, matrices)
    original = StableFootruleKNN(n_neighbors=3, input_kind='positions').fit(layers[-1][2][:8], y[:8]).predict(layers[-1][2][8:])
    identity = {'dataset_id': 'constructed', 'outer_repeat': 0, 'outer_fold': 0, 'model_seed': 8129}
    for draw in range(10):
        permutations = td.relabel_permutations(td.RELABEL_SEED, identity, 0, draw, [5, 3])
        assert [sorted(p.tolist()) for p in permutations] == [list(range(5)), list(range(3))]
        relabeled = td.network_forward(positions, td.relabel_matrices(matrices, permutations))
        for (responses, _, hidden), (new_responses, _, new_hidden), permutation in zip(layers, relabeled, permutations):
            inverse = np.argsort(permutation)
            assert np.array_equal(new_responses, responses[:, inverse]) and np.array_equal(new_hidden, hidden[:, inverse])
        knn = StableFootruleKNN(n_neighbors=3, input_kind='positions').fit(relabeled[-1][2][:8], y[:8]).predict(relabeled[-1][2][8:])
        assert np.array_equal(knn, original)
    assert all(np.array_equal(a, b) for a, b in zip(td.relabel_matrices(matrices, [np.arange(5), np.arange(3)]), matrices))
    tied = [matrices[0].copy(), matrices[1]]
    tied[0][1] = tied[0][0]                                   # filters 0 and 1 respond identically to every row
    before = td.network_forward(positions, tied)
    swap = [np.asarray([1, 0, 2, 3, 4]), np.arange(3)]
    after = td.network_forward(positions, td.relabel_matrices(tied, swap))
    assert np.array_equal(after[0][0][:, swap[0]], before[0][0])           # the same responses, relabeled
    assert not np.array_equal(after[0][2][:, swap[0]], before[0][2])       # but the ranking completed by filter ID changed
    assert np.all(after[0][2][:, swap[0]][:, [0, 1]] == before[0][2][:, [1, 0]])


# ----------------------------------------------------------------------------- the recorder

def test_recorder_is_non_invasive_and_verifies_its_checkpoint():
    plain, orders, y = tiny_network()
    plain.fit_orders(orders, y)
    net, _, _ = tiny_network()
    net.initialize_orders(orders, y)
    recorder = td.TrainingRecorder(net, every=5)
    with recorder.installed():
        net.train_initialized(orders, y)
    assert 'update_network' not in vars(net.network_) and net.state_hash() == plain.state_hash()
    verified = recorder.verify()
    assert verified['training_record']['passed'] and verified['checkpoint_orders']['passed'] and all(recorder.rng_checks)
    assert sorted(recorder.snapshots) == [0, 5, 10, 12] and len(recorder.errors) == 12 and verified['n_validation'] == 18
    assert td.copies_equal(td.filter_copy(net.network_), recorder.checkpoint_copy)
    assert recorder.checkpoint_copy_iteration == verified['checkpoint']
    for t, snapshot in recorder.snapshots.items():
        assert td.output_error(recorder.validation, snapshot['matrices']) == (recorder.initial_error if t == 0 else recorder.errors[t - 1])


def test_recorder_guard_detects_rng_use_and_restores_the_state(monkeypatch):
    plain, orders, y = tiny_network(iterations=4)
    plain.fit_orders(orders, y)
    net, _, _ = tiny_network(iterations=4)
    net.initialize_orders(orders, y)
    recorder = td.TrainingRecorder(net, every=2)
    copy = td.filter_copy

    def consuming(network, orders=True):
        np.random.rand()
        return copy(network, orders)

    monkeypatch.setattr(td, 'filter_copy', consuming)
    with recorder.installed():
        net.train_initialized(orders, y)
    assert not all(recorder.rng_checks) and net.state_hash() == plain.state_hash()
    other, other_orders, other_y = tiny_network(iterations=4)
    other.initialize_orders(other_orders, other_y)
    other.network_.eval_period = 2
    with pytest.raises(td.CheckFailed, match='eval_period'):
        td.TrainingRecorder(other)


def test_instrumented_fit_is_the_estimator_fit_step_for_step():
    X, y = tiny_data(150)
    params = {'n_views': 7, 'strategy': 'diverse', 'embed_dim': 8, 'degree': 2, 'widths': [5], 'iterations': 6, 'learning_rate': .2,
              'batch_size': 32, 'validation_ratio': .1, 'aggregation': 'majority', 'augment': True}
    seed_fit(8129)
    plain = MultiViewArrowFlowKNN(**params, seed=8129).fit(X, y)
    seed_fit(8129)
    model, recorders = td.instrumented_fit(params, 8129, X, y, every=5)
    assert [net.state_hash() for _, net in model.views_] == [net.state_hash() for _, net in plain.views_]
    assert np.array_equal(model.predict(X), plain.predict(X)) and model.readout_selections_ == plain.readout_selections_
    assert len(recorders) == 7 and all(r.verify()['training_record']['passed'] for r in recorders)


# ----------------------------------------------------------------------------- the synthetic smoke run

@pytest.fixture(scope='session')
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('training_diagnostics_smoke')
    report = td.smoke(root, td.draft_protocol(), workers=3)
    return SimpleNamespace(root=root, output=root/'diagnostics', report=report)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def sealed_for(output, job):
    key = (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
    return next(s for s in json.loads((output/'reference_selections.json').read_text())
                if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == key)


def test_smoke_exercises_every_reference_measure_and_check(smoke_run):
    report, output = smoke_run.report, smoke_run.output
    jobs = json.loads((output/'planned_jobs.json').read_text())
    assert report['datasets'] == ['synthetic', 'synthetic_b1', 'synthetic_b2'] and report['planned_jobs'] == len(jobs) == 9
    assert report['check_totals']['uninstrumented_state_hash'] == {'performed': 3, 'passed': 3}
    assert all(report['check_totals'][name] == {'performed': 9, 'passed': 9} for name in td.JOB_CHECKS if name != 'uninstrumented_state_hash')
    assert report['inferential_significance_claims'] is False and set(report['summaries']) == set(report['datasets'])
    assert sorted(json.loads((output/'manifest.json').read_text())['references']) == ['smoke_bridge_knn', 'smoke_newdata_batch1',
                                                                                     'smoke_newdata_batch2']
    assert [job['check_uninstrumented'] for job in jobs] == [True, False, False] * 3
    depths = {job['dataset_id']: {len(j['selected_widths']) for j in jobs if j['dataset_id'] == job['dataset_id']} for job in jobs}
    assert depths['synthetic'] == {2} and {1} in (depths['synthetic_b1'], depths['synthetic_b2'])
    assert {job['selected']['augment'] for job in jobs} == {True, False}
    provenance = json.loads((output/td.PROVENANCE_FILE).read_text())
    assert provenance['check_totals'] == report['check_totals'] and all(provenance['tables'][name] == sha256(output/name) for name in td.TABLES)
    records = [json.loads((output/'jobs'/f"{job['stem']}.json").read_text()) for job in jobs]
    iterations = {record['identity']['dataset_id']: record['views'][0]['iterations'] for record in records}
    assert iterations == {'synthetic': 23, 'synthetic_b1': 1, 'synthetic_b2': 1}
    assert len(read_csv(output/'validation_curves.csv')) == sum(7 * (iterations[job['dataset_id']] + 1) for job in jobs)
    checkpoints = read_csv(output/'checkpoints.csv')
    assert len(checkpoints) == 63 and all(row['orders_verified'] == '1' for row in checkpoints)
    assert all((row['checkpoint_iteration'] == '0') == (row['returned_initial_filters'] == '1') for row in checkpoints)
    snapshots = read_csv(output/'snapshots.csv')
    assert {row['measure'] for row in snapshots} == {'displacement_from_initial', 'changed_share_from_initial', 'displacement_from_previous',
                                                     'changed_share_from_previous', 'knn_test_accuracy', 'output_rule_test_accuracy',
                                                     'output_rule_training_accuracy'}
    curve = [row['iteration'] for row in snapshots if row['dataset_id'] == 'synthetic' and row['view'] == '0' and row['layer'] == 'readout'
             and row['measure'] == 'knn_test_accuracy' and row['snapshot'] == 'scheduled']
    assert curve == ['0', '10', '20', '23'] * 3
    assert len([row for row in snapshots if row['view'] == 'majority']) == sum(len(r['majority']['snapshots']) for r in records)
    ties = read_csv(output/'ties.csv')
    assert len(ties) == sum(7 * 2 * (len(job['selected_widths']) + 1) for job in jobs)
    assert all((row['tied_nearest_share'] == '') == (row['layer'] != 'output') for row in ties)
    assert {row['layer'] for row in ties} == {'hidden_0', 'hidden_1', 'output'}
    assert {row['layer'] for row in snapshots} == {'hidden_0', 'hidden_1', 'output', 'readout'}
    relabel = read_csv(output/'relabel.csv')
    assert len(relabel) == 9 * 8 and all(row['draws'] == '20' for row in relabel)
    assert {row['view'] for row in relabel} == {str(v) for v in range(7)} | {'majority'}


def test_a_job_fails_when_its_predictions_differ_from_the_reference(smoke_run):
    output = smoke_run.output
    job = json.loads((output/'planned_jobs.json').read_text())[1]
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (job['outer_repeat'], job['outer_fold']))
    tampered = json.loads(json.dumps(sealed_for(output, job)))
    labels = tampered['reference_predictions']['8129']
    labels[0] = (labels[0] + 1) % 3
    tampered['reference_prediction_hashes']['8129'] = array_hash(np.asarray(labels))
    record, arrays = td.evaluate_job(X, y, split, job, tampered, json.loads((output/'protocol.json').read_text()),
                                     dataset_hash=data['dataset_hash'], code_revision='test', protocol_hash='test')
    assert record['status'] == 'failed' and record['check_failure'] is True and arrays is None
    assert record['checks']['reference_predictions']['passed'] is False and record['checks']['reference_predictions']['n_differing'] == 1


def test_a_failed_job_cancels_the_run_and_no_table_is_written(smoke_run, tmp_path):
    output, target = smoke_run.output, tmp_path/'copy'
    target.mkdir()
    for name in ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json'):
        shutil.copyfile(output/name, target/name)
    for name in json.loads((output/'manifest.json').read_text())['datasets']:
        shutil.copytree(output/name, target/name)
    selections = json.loads((target/'reference_selections.json').read_text())
    labels = selections[1]['reference_predictions']['8129']
    labels[0] = (labels[0] + 1) % 3
    selections[1]['reference_prediction_hashes']['8129'] = array_hash(np.asarray(labels))
    (target/'reference_selections.json').write_text(json.dumps(selections, indent=2, sort_keys=True) + '\n')
    with pytest.raises(ValueError, match='sealed selections changed'):
        td.verify(target, allow_smoke=True)
    manifest = json.loads((target/'manifest.json').read_text())
    manifest['reference_selections_sha256'] = sha256(target/'reference_selections.json')          # a deliberate reseal
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    _, _, jobs, _ = td.verify(target, allow_smoke=True)
    with pytest.raises(td.CheckFailed, match='job failed'):
        td.execute(target, jobs[:2], 1)
    with pytest.raises(ValueError, match='Incomplete training diagnostics evidence'):
        td.write_tables(target, allow_smoke=True)
    assert not any((target/name).exists() for name in [*td.TABLES, td.PROVENANCE_FILE])


def test_summary_refuses_tampered_tables_records_and_artifacts(smoke_run, tmp_path):
    stem = json.loads((smoke_run.output/'planned_jobs.json').read_text())[0]['stem']
    table = shutil.copytree(smoke_run.output, tmp_path/'table')
    rows = (table/'relabel.csv').read_text().splitlines()
    rows[1] = rows[1].replace(',20,', ',19,', 1)
    (table/'relabel.csv').write_text('\n'.join(rows) + '\n')
    with pytest.raises(ValueError, match='relabel.csv'):
        td.summary(table, allow_smoke=True)
    record_copy = shutil.copytree(smoke_run.output, tmp_path/'record')
    record = json.loads((record_copy/'jobs'/f'{stem}.json').read_text())
    record['views'][0]['snapshots'][0]['knn_accuracy'] += .5
    (record_copy/'jobs'/f'{stem}.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='scheduled accuracies'):
        td.summary(record_copy, allow_smoke=True)
    artifact_copy = shutil.copytree(smoke_run.output, tmp_path/'artifact')
    with np.load(artifact_copy/'artifacts'/f'{stem}.npz') as stored:
        arrays = {key: stored[key] for key in stored.files}
    arrays['knn_checkpoint'][0, 0] = (arrays['knn_checkpoint'][0, 0] + 1) % 3
    np.savez_compressed(artifact_copy/'artifacts'/f'{stem}.npz', **arrays)
    with pytest.raises(ValueError, match='artifact hash'):
        td.summary(artifact_copy, allow_smoke=True)
    assert td.summary(shutil.copytree(smoke_run.output, tmp_path/'clean'), allow_smoke=True) == smoke_run.report


def test_outputs_are_all_or_none_and_a_used_directory_is_refused(smoke_run, tmp_path):
    td.write_all(tmp_path, {'a.csv': 'x\n', 'b.csv': 'y\n'})
    td.write_all(tmp_path, {'a.csv': 'x\n', 'b.csv': 'y\n'})
    with pytest.raises(FileExistsError, match='b.csv'):
        td.write_all(tmp_path, {'a.csv': 'x\n', 'b.csv': 'z\n', 'c.csv': 'w\n'})
    assert (tmp_path/'b.csv').read_text() == 'y\n' and sorted(p.name for p in tmp_path.iterdir()) == ['a.csv', 'b.csv']
    with pytest.raises(FileExistsError, match='not empty'):
        td.run(smoke_run.output, json.loads((smoke_run.output/'protocol.json').read_text()), {}, allow_smoke=True)
    with pytest.raises(ValueError, match='frozen'):
        td.run(tmp_path/'new', td.draft_protocol(), td.DEFAULT_SOURCES)


# ----------------------------------------------------------------------------- protocol, projection and freeze

def test_the_committed_protocol_is_the_draft_or_its_freeze():
    p = json.loads(td.PROTOCOL.read_text())
    assert td.validate_protocol(p) is p
    strip = lambda q: {k: v for k, v in q.items() if k not in td.FREEZE_FIELDS}
    assert strip(p) == strip(td.draft_protocol())
    if p['frozen']:
        assert p['status'] == td.FROZEN_STATUS and 0 < p['pilot_projection']['decision_hours'] <= td.CAP_HOURS
        assert p['pilot_projection']['workers'] == td.WORKERS
    else:
        assert p == td.draft_protocol()


def test_the_draft_declares_the_ruled_design_and_refuses_edits():
    p = td.draft_protocol()
    assert len(p['datasets']) == 17 and p['datasets'][:7] == ['iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits']
    assert sorted(p['datasets'][7:]) == sorted(nd.PANEL) and p['pilot_datasets'] == ['iris', 'segment'] and p['frozen'] is False
    assert (p['model_seed'], p['n_views'], p['snapshots']['every'], p['relabel'], p['wallclock_cap_hours'], p['workers']) == (
        8129, 7, 10, {'draws': 20, 'seed': td.RELABEL_SEED}, 4, 8)
    assert set(p['checks']) == set(td.JOB_CHECKS) and p['selection_statement'].startswith('Nothing is selected')
    for entry in p['references'].values():
        assert all(sha256(REPO/entry[block]['protocol_file']) == entry[block]['protocol_sha256'] for block in ('run', 'ablation'))
    with pytest.raises(ValueError, match='snapshots'):
        td.validate_protocol({**p, 'snapshots': {**p['snapshots'], 'every': 5}})
    with pytest.raises(ValueError, match='must equal the draft'):
        td.validate_protocol({**p, 'status': 'edited'})
    frozen = {**p, 'frozen': True, 'frozen_at_utc': 'x', 'status': td.FROZEN_STATUS, 'resource_decision': 'x',
              'pilot_projection': {'cap_hours': 4, 'workers': 8, 'decision_hours': 1.}}
    assert td.validate_protocol(frozen) is frozen
    for change in ({'pilot_projection': {'cap_hours': 4, 'workers': 8, 'decision_hours': 4.5}}, {'frozen_at_utc': None},
                   {'pilot_projection': {'cap_hours': 4, 'workers': 16, 'decision_hours': 1.}}):
        with pytest.raises(ValueError, match='frozen protocol'):
            td.validate_protocol({**frozen, **change})
    references = json.loads(json.dumps(p['references']))
    references['newdata_batch2']['datasets'].append('iris')
    with pytest.raises(ValueError, match='two references'):
        td.reference_of({**p, 'references': references})
    extra = {**td.REFERENCES, 'dedup': {**td.REFERENCES['bridge_knn'], 'datasets': ['wine_quality_dedup']}}
    later = td.draft_protocol(references=extra, protocol_id='arrowflow-v3-training-diagnostics-dedup-1', pilot_datasets=['wine_quality_dedup'])
    assert td.validate_protocol(later) is later and td.reference_of(later)['wine_quality_dedup'] == 'dedup'
    assert td.parse_reference('dedup=/runs/a,/runs/b') == ('dedup', (Path('/runs/a'), Path('/runs/b')))
    with pytest.raises(Exception, match='NAME=RUN_DIR,ABLATION_DIR'):
        td.parse_reference('dedup=/runs/a')


def test_the_projection_prices_every_job_and_adds_the_first_fold_refit():
    jobs = ([{'dataset_id': 'a', 'reference_outer_seconds': 100., 'check_uninstrumented': i == 0} for i in range(3)]
            + [{'dataset_id': 'b', 'reference_outer_seconds': 50., 'check_uninstrumented': i == 0} for i in range(2)])
    projection = td.calibrated_projection(jobs, [{'dataset_id': 'a', 'job_to_fit_ratio': 2.}], workers=2)
    assert projection['serial_hours'] == pytest.approx(950 / 3600) and projection['datasets']['b']['basis'] == 'largest piloted ratio'
    assert projection['simulated_makespan_hours'] == pytest.approx(500 / 3600) and projection['longest_job_hours'] == pytest.approx(300 / 3600)


def test_freeze_requires_the_draft_a_passing_pilot_and_a_passing_smoke(tmp_path):
    from experiments.make_revision.evaluation import config_id
    draft = td.draft_protocol()
    probe = {'dataset_id': 'iris', 'result_file': 'r', 'config_id': 'c', 'model_seed': 8129, 'inner_fold': 0, 'reference_score': 1.,
             'refit_score': 1., 'readout_selections_identical': True, 'reproduced': True}
    pilot = {'protocol_hash': config_id(draft), 'code_revision': 'abc', 'records': [{'dataset_id': 'iris', 'job_to_fit_ratio': 1.1, 'timing': {}}],
             'calibrated_projection': {'serial_hours': 8., 'serial_hours_over_workers': 1., 'simulated_makespan_hours': 1.2,
                                       'longest_job_hours': .1, 'datasets': {'iris': {'serial_hours': .5}}},
             'harness_projection': {'serial_hours': 6., 'hours_at_workers_ideal': .75}, 'reproduction_probes': {'bridge_knn': probe},
             'decision': {'rule': draft['decision_rule'], 'hours': 1.2, 'cap_hours': 4, 'workers': 8, 'within_cap': True,
                          'checks_passed': True, 'probes_reproduced': True}}
    stages = {'smoke': {'status': 'ok'}, 'summary': 'stages'}
    for index, (pilot_change, stages_change) in enumerate([({'within_cap': False}, {}), ({'hours': 4.5}, {}), ({'workers': 16}, {}),
                                                          ({'checks_passed': False}, {}), ({}, {'smoke': {'status': 'failed'}})]):
        paths = [tmp_path/f'{name}{index}.json' for name in ('draft', 'pilot', 'stages')]
        for path, value in zip(paths, (draft, dict(pilot, decision={**pilot['decision'], **pilot_change}), {**stages, **stages_change})):
            path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            td.freeze(*paths, paths[0])
        assert json.loads(paths[0].read_text()) == draft
    paths = [tmp_path/f'{name}.json' for name in ('draft', 'pilot', 'stages')]
    for path, value in zip(paths, (draft, pilot, stages)):
        path.write_text(json.dumps(value))
    frozen = td.freeze(*paths, paths[0], frozen_at_utc='2026-09-14T00:00:00+00:00')
    assert json.loads(paths[0].read_text()) == frozen and td.validate_protocol(frozen) is frozen
    assert frozen['pilot_projection']['decision_hours'] == 1.2 and frozen['pilot_projection']['stages'] == stages
    with pytest.raises(ValueError, match='already frozen'):
        td.freeze(*paths, paths[0])


# ----------------------------------------------------------------------------- the real references (read only; nothing is fitted)

def _cached(pin):
    home = Path(get_data_home())/'openml'/'openml.org'
    return all(path.is_file() for path in (home/'api'/'v1'/'json'/'data'/f"{pin['data_id']}.gz",
                                           home/'api'/'v1'/'json'/'data'/'features'/f"{pin['data_id']}.gz",
                                           home/'data'/'v1'/'download'/f"{pin['file_id']}.gz"))


@needs_real
def test_the_pins_match_the_real_references_and_prepare_reconstructs_their_selections(tmp_path):
    p = td.draft_protocol()
    for name, (run, ablation) in REAL.items():
        declared = p['references'][name]
        assert reference_pins(run) == {key: declared['run'][key] for key in ('model_id', 'protocol_id', 'protocol_sha256', 'summary_sha256',
                                                                             'code_revision')}
        assert td.ablation_pins(ablation, declared['ablation']['summary_file']) == {key: declared['ablation'][key] for key in td.ABLATION_PINS}
    jobs, references = td.prepare(tmp_path/'iris', p, REAL, ['iris'])
    assert [job['stem'] for job in jobs] == [f'iris__r{r}f{f}' for r in range(3) for f in range(5)] and list(references) == ['bridge_knn']
    assert [job['check_uninstrumented'] for job in jobs] == [True] + [False] * 14
    identity = json.loads((tmp_path/'iris'/'manifest.json').read_text())['dataset_identity']['iris']
    assert identity['fresh_load']['loader'] == td.BENCHMARK_LOADER
    sealed = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
              for s in json.loads((REAL['bridge_knn'][1]/'reference_selections.json').read_text())}
    assert json.loads((tmp_path/'iris'/'reference_selections.json').read_text()) == [sealed[('iris', j['outer_repeat'], j['outer_fold'])] for j in jobs]
    assert not (tmp_path/'iris'/'jobs').exists()
    with pytest.raises(ValueError, match='Supply --reference for newdata_batch1'):
        td.prepare(tmp_path/'other', p, {'bridge_knn': REAL['bridge_knn']}, ['iris', 'diabetes'])


@needs_real
@pytest.mark.skipif(not _cached(nd.PIN_BY_NAME['vertebra_column']), reason='the vertebra_column download is not cached here')
def test_prepare_loads_a_further_dataset_with_its_pins(tmp_path):
    jobs, _ = td.prepare(tmp_path/'vertebra', td.draft_protocol(), REAL, ['vertebra_column'])
    identity = json.loads((tmp_path/'vertebra'/'manifest.json').read_text())['dataset_identity']['vertebra_column']
    assert identity['fresh_load'] == {'loader': td.NEWDATA_LOADER, 'dataset_hash': nd.PIN_BY_NAME['vertebra_column']['dataset_hash'],
                                      'pinned_splits_hash': nd.PIN_BY_NAME['vertebra_column']['splits_hash']}
    assert len(jobs) == 15 and {job['reference'] for job in jobs} == {'newdata_batch2'}
