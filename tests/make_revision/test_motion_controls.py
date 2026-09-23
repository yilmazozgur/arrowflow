"""E1 matched motion-signal controls (motion_controls, compare_motion): the non-invasive arm wrapper and its
transcription fidelity, each arm's construction and the matching it preserves, the measures, the protocol and freeze
rules, and the analysis gate and tables. Small synthetic networks only; no real job is fitted."""
import json
import random
from pathlib import Path
import numpy as np
import pytest
from experiments.make_revision import compare_motion as cm
from experiments.make_revision import motion_controls as mc
from experiments.make_revision import training_diagnostics as td
from experiments.make_revision.comparisons import StableFootruleKNN
from experiments.make_revision.compare_runs import RunComparisonError
from experiments.make_revision.models import seed_fit
from experiments.make_revision.multiview import MultiViewArrowFlowKNN

SEED = 8129
IDENTITY = {'dataset_id': 'synthetic', 'outer_repeat': 0, 'outer_fold': 0, 'model_seed': SEED}


def tiny_data(n=90, seed=5):
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], n // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    return X, y


def tiny_params(widths=(8, 12), iterations=10):
    return dict(n_views=2, strategy='diverse', embed_dim=8, degree=1, widths=list(widths), learning_rate=.2,
                iterations=iterations, batch_size=16, validation_ratio=.1, augment=False, aggregation='majority')


@pytest.fixture(scope='module')
def data():
    return tiny_data()


@pytest.fixture(scope='module')
def fits(data):
    """One fit of every arm at two hidden layers, and the uninstrumented reference fit, built once."""
    X, y = data
    params = tiny_params()
    seed_fit(SEED)
    plain = MultiViewArrowFlowKNN(**params, seed=SEED).fit(X, y)
    reference = {'state': [net.state_hash() for _, net in plain.views_],
                 'predictions': mc._majority(plain.predict_views(X)[0])}
    out = {}
    for arm in mc.ARMS:
        seed_fit(SEED)
        model, arms = mc.arm_fit(arm, params, SEED, X, y, identity=IDENTITY, hidden_layers=2)
        out[arm] = {'model': model, 'arms': arms, 'state': [net.state_hash() for _, net in model.views_],
                    'predictions': mc._majority(model.predict_views(X)[0]),
                    'displacement': mc._mean_displacement([a.displacement_by_layer() for a in arms]),
                    'matching': [a.matching() for a in arms]}
    return {'reference': reference, 'arms': out, 'params': params}


# ----------------------------------------------------------------------------- non-invasiveness

def test_views7_through_the_wrapper_equals_an_uninstrumented_fit(fits):
    assert fits['arms']['views7']['state'] == fits['reference']['state']
    assert np.array_equal(fits['arms']['views7']['predictions'], fits['reference']['predictions'])


def test_views7_at_one_hidden_layer_equals_an_uninstrumented_fit(data):
    X, y = data
    params = tiny_params(widths=(10,))
    seed_fit(SEED)
    plain = MultiViewArrowFlowKNN(**params, seed=SEED).fit(X, y)
    seed_fit(SEED)
    model, arms = mc.arm_fit('views7', params, SEED, X, y, identity=IDENTITY, hidden_layers=1)
    assert [n.state_hash() for _, n in model.views_] == [n.state_hash() for _, n in plain.views_]
    assert all(len(a.displacement_by_layer()) == 2 for a in arms)          # one hidden layer plus the output layer


def test_every_arm_leaves_the_global_rngs_alone_and_removes_its_wrapper(fits):
    for arm, entry in fits['arms'].items():
        assert all(all(a.rng_checks) for a in entry['arms']), arm
        assert all(a.wrapper_removed() for a in entry['arms']), arm


def test_the_wrapper_restores_the_numpy_and_python_rng_states(data):
    X, y = data
    net, motion = _initialized(X, y, 'permuted_alignment')     # initialize_orders reseeds; the wrapper must not
    np.random.seed(11)
    random.seed(11)
    before = (np.random.get_state(), random.getstate())
    with motion.installed():
        motion._backward({}, [])                               # a wrapper call with no hidden layer to process
    assert mc.numpy_states_equal(before[0], np.random.get_state()) and before[1] == random.getstate()
    assert all(motion.rng_checks)


def test_a_second_wrapper_on_the_same_network_is_refused(data):
    X, y = data
    net, motion = _initialized(X, y, 'views7')
    with motion.installed():
        with pytest.raises(ValueError, match='already carries'):
            mc.MotionArm(net, 'frozen', seed=1, hidden_layers=2)


def test_the_pinned_core_sources_are_unchanged():
    check = mc.check_core_sources()
    assert check['passed'] and check['observed'] == mc.CORE_SOURCES


def test_a_changed_core_source_fails_the_check(monkeypatch):
    monkeypatch.setitem(mc.CORE_SOURCES, 'Vertex.apply_motion', '0' * 64)
    check = mc.check_core_sources()
    assert not check['passed'] and check['changed'] == ['Vertex.apply_motion']


def _initialized(X, y, arm, widths=(8, 12)):
    from experiments.make_revision.models import ArrowFlowEstimator, OrdinalEncoder
    seed_fit(SEED)
    enc = OrdinalEncoder('random', 8, 1, .3, SEED).fit(X, y)
    net = ArrowFlowEstimator(embed_dim=8, degree=1, widths=list(widths), iterations=4, learning_rate=.2,
                             batch_size=16, validation_ratio=.1, seed=SEED)
    net.initialize_orders(enc.transform(X), y)
    return net, mc.MotionArm(net, arm, seed=7, hidden_layers=len(widths))


# ----------------------------------------------------------------------------- the arms

def test_arm_layers_cover_the_declared_restrictions():
    assert mc.arm_layers('views7', 2) == (0, 1) and mc.arm_layers('views7', 1) == (0,)
    assert mc.arm_layers('frozen', 2) == ()
    assert mc.arm_layers('permuted_alignment', 2) == (0, 1) and mc.arm_layers('random_direction', 2) == (0, 1)
    assert mc.arm_layers('single_layer_first', 2) == (0,) and mc.arm_layers('single_layer_last', 2) == (1,)
    assert mc.arm_layers('single_layer_first', 1) == mc.arm_layers('single_layer_last', 1) == (0,)
    with pytest.raises(ValueError):
        mc.arm_layers('other', 1)


def test_frozen_holds_every_hidden_filter_still_and_still_moves_the_output_layer(fits):
    hidden = [e for e in fits['arms']['frozen']['displacement'] if e['kind'] == 'hidden']
    assert len(hidden) == 2
    assert all(e['mean_normalized_footrule'] == 0. and e['unchanged_share'] == 1. for e in hidden)
    output = [e for e in fits['arms']['frozen']['displacement'] if e['kind'] == 'output'][0]
    assert output['mean_normalized_footrule'] > 0


def test_frozen_equals_accumulating_and_then_discarding_the_accumulator(data, monkeypatch):
    """The early stop is an optimization: running the whole hidden pass and resetting each accumulator instead of
    applying it leaves the identical network."""
    X, y = data
    params = tiny_params()
    seed_fit(SEED)
    fast, _ = mc.arm_fit('frozen', params, SEED, X, y, identity=IDENTITY, hidden_layers=2)
    original = mc.MotionArm.__init__

    def slow_init(self, net, arm, *, seed, hidden_layers):
        original(self, net, arm, seed=seed, hidden_layers=hidden_layers)
        self.stop_after = 0                       # process every hidden layer, apply none
    monkeypatch.setattr(mc.MotionArm, '__init__', slow_init)
    seed_fit(SEED)
    slow, _ = mc.arm_fit('frozen', params, SEED, X, y, identity=IDENTITY, hidden_layers=2)
    assert [n.state_hash() for _, n in fast.views_] == [n.state_hash() for _, n in slow.views_]


def test_single_layer_arms_move_exactly_their_own_layer(fits):
    first = fits['arms']['single_layer_first']['displacement']
    last = fits['arms']['single_layer_last']['displacement']
    assert first[0]['mean_normalized_footrule'] > 0 and first[1]['mean_normalized_footrule'] == 0.
    assert last[0]['mean_normalized_footrule'] == 0. and last[1]['mean_normalized_footrule'] > 0


def test_every_control_arm_differs_from_views7(fits):
    for arm in mc.ARMS:
        if arm == 'views7':
            continue
        assert fits['arms'][arm]['state'] != fits['reference']['state'], arm


def test_permuted_alignment_rekeys_almost_every_accepted_slot_and_keeps_the_motion(fits):
    summary = mc._matching_summary([fits['arms']['permuted_alignment']['matching']])
    for layer, width in (('0', 8), ('1', 12)):
        entry = summary[layer]
        assert entry['mass_before'] == entry['mass_after']
        assert entry['positive_before'] == entry['positive_after']
        assert 0.5 < entry['changed_share'] <= 1.        # a uniform permutation fixes about 1 of every `width` slots
    assert summary['0']['changed_share'] == pytest.approx(1 - 1 / 8, abs=.12)
    assert summary['1']['changed_share'] == pytest.approx(1 - 1 / 12, abs=.12)


def test_random_direction_preserves_mass_magnitudes_and_sign_counts(fits):
    summary = mc._matching_summary([fits['arms']['random_direction']['matching']])
    for entry in summary.values():
        assert entry['mass_before'] == entry['mass_after']
        assert entry['effective_before'] == entry['effective_after']
        assert entry['positive_before'] == entry['positive_after']
        assert entry['changed_slots'] > 0


def test_every_arm_reports_a_matched_transform(fits):
    for arm, entry in fits['arms'].items():
        assert all(view['passed'] for view in entry['matching']), arm
        for view in entry['matching']:
            for layer in view['layers'].values():
                assert layer['max_mass_deviation'] == 0.
                assert all(layer['checks'].values())


def test_arms_are_deterministic_given_the_recorded_seed(data):
    X, y = data
    params = tiny_params()
    hashes = []
    for _ in range(2):
        seed_fit(SEED)
        model, _ = mc.arm_fit('random_direction', params, SEED, X, y, identity=IDENTITY, hidden_layers=2)
        hashes.append([n.state_hash() for _, n in model.views_])
    assert hashes[0] == hashes[1]


def test_arm_seeds_are_distinct_per_arm_fold_seed_and_view():
    seeds = {mc.arm_seed(arm, dict(IDENTITY, outer_fold=fold, model_seed=seed), view)
             for arm in mc.ARMS for fold in (0, 1) for seed in (8129, 19391) for view in range(7)}
    assert len(seeds) == len(mc.ARMS) * 2 * 2 * 7
    assert mc.arm_seed('frozen', IDENTITY, 0) == mc.arm_seed('frozen', IDENTITY, 0)


def test_a_broken_transform_is_refused_at_the_batch_that_breaks_it(data):
    X, y = data
    net, motion = _initialized(X, y, 'random_direction')
    layer = motion.network.graph.vertex_list[f'{motion.network.id}_ly1']
    keys = list(layer.graph.vertex_list)
    records = [[keys, np.linspace(-1., 1., len(keys))]]
    assert motion._transform(1, layer, records) is not records          # a real, matched transform passes
    motion._randomize_direction = lambda accepted, given: ([[k, np.asarray(m) * 2] for k, m in given], 0)
    with pytest.raises(mc.CheckFailed, match='mass_matched'):
        motion._transform(1, layer, records)


def test_require_matched_names_what_broke():
    net = object.__new__(mc.MotionArm)
    net.arm = 'random_direction'
    before = {'slots': 4, 'effective': 4, 'mass': 10., 'positive': 2, 'negative': 2, 'magnitudes': 'a'}
    assert net._require_matched(0, before, dict(before)) == 0.
    with pytest.raises(mc.CheckFailed, match='accepted_mass'):
        net._require_matched(0, before, dict(before, mass=11.))
    with pytest.raises(mc.CheckFailed, match='sign_counts'):
        net._require_matched(0, before, dict(before, positive=3, negative=1))
    with pytest.raises(mc.CheckFailed, match='magnitude_multiset'):
        net._require_matched(0, before, dict(before, magnitudes='b'))


def test_every_arm_starts_from_the_same_initial_filters(fits):
    hashes = {arm: [mc._filters_hash(a.initial) for a in entry['arms']] for arm, entry in fits['arms'].items()}
    assert all(value == hashes['views7'] for value in hashes.values())


def test_every_arm_sees_the_same_signal_at_the_first_batch(fits):
    """At the first batch no filter has moved, so every arm sees views7's signal at the last hidden layer. The identity
    transforms leave the deeper layer's signal alone too; the randomizing ones are expected to change it."""
    views7 = [a.first_batch_signal() for a in fits['arms']['views7']['arms']]
    for arm in ('permuted_alignment', 'random_direction', 'single_layer_first', 'single_layer_last'):
        for theirs, ours in zip(views7, [a.first_batch_signal() for a in fits['arms'][arm]['arms']]):
            assert theirs['1'] == ours['1'], arm
            if arm == 'permuted_alignment':
                assert theirs['0'] != ours['0'], arm        # re-keying changes which filters are accepted
            elif '0' in ours:
                assert theirs['0'] == ours['0'], arm        # a sign permutation leaves the propagated mean alone


# ----------------------------------------------------------------------------- measures

def test_drop_self_removes_the_query_row_from_its_own_neighbourhood():
    indices = np.array([[0, 5, 7], [3, 1, 9], [8, 4, 6]])
    out = mc._drop_self(indices, [0, 1, 2])
    assert out.tolist() == [[5, 7], [3, 9], [8, 4]]        # row 2 is absent, so the last neighbour goes


def test_neighborhood_purity_counts_same_class_neighbours():
    positions = np.array([[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3],
                          [3, 2, 1, 0], [2, 3, 1, 0], [3, 1, 2, 0]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    readout = StableFootruleKNN(n_neighbors=1, input_kind='positions').fit(positions, labels)
    assert mc.neighborhood_purity(readout, labels, positions, labels, k=1) == 1.
    assert mc.neighborhood_purity(readout, labels, positions, labels, k=1, exclude_rows=np.arange(6)) == 1.
    assert mc.neighborhood_purity(readout, labels, positions, 1 - labels, k=1) == 0.
    assert mc.neighborhood_purity(readout, labels, positions, labels, k=5, exclude_rows=np.arange(6)) == pytest.approx(.4)
    assert mc.neighborhood_purity(readout, labels, np.empty((0, 4), dtype=int), np.empty(0), k=1) is None


def test_purity_record_reports_both_row_sets_and_the_view_k(fits, data):
    X, y = data
    model = fits['arms']['views7']['model']
    orders = [enc.transform(X) for enc, _ in model.views_]
    record = mc.purity_record(model, orders, orders, y, y)
    assert 0 <= record['test'] <= 1 and 0 <= record['train'] <= 1
    assert record['k_by_view'] == [min(r.n_neighbors, len(y)) for r in model.readouts_]
    assert record['train_query_rows'] == len(y) and record['train_query_step'] == 1


def test_purity_record_subsamples_a_large_training_partition(fits, data):
    X, y = data
    model = fits['arms']['views7']['model']
    orders = [enc.transform(X) for enc, _ in model.views_]
    record = mc.purity_record(model, orders, orders, y, y, query_cap=20)
    assert record['train_query_step'] == 5 and record['train_query_rows'] == 18


def test_per_class_recall_is_aligned_with_the_dataset_classes():
    y = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    assert mc.per_class_recall(y, predicted, [0, 1, 2]) == {'0': .5, '1': 1., '2': .5}


def test_mean_displacement_labels_the_output_layer():
    rows = mc._mean_displacement([[(0.2, 0.5), (0.4, 1.0)], [(0.4, 0.5), (0.6, 1.0)]])
    assert [r['kind'] for r in rows] == ['hidden', 'output']
    assert rows[0]['mean_normalized_footrule'] == pytest.approx(0.3)
    assert rows[0]['unchanged_share'] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- protocol

def test_the_draft_protocol_validates_and_declares_the_design():
    p = mc.draft_protocol()
    assert mc.validate_protocol(p) == p
    assert len(p['datasets']) == 17 and sorted(p['references']) == ['bridge_knn', 'newdata_batch1', 'newdata_batch2']
    assert p['arms'] == list(mc.ARMS) and not p['frozen']
    assert p['analysis']['primary_family']['arms'] == list(mc.PRIMARY_ARMS)
    assert p['analysis']['named_subset']['datasets'] == list(mc.NAMED_SUBSET)
    assert 'label-permutation' in p['supervision_disclosure']
    assert 'never trains at all' in p['differs_from_untrained']


def test_the_committed_protocol_file_is_this_draft_or_its_freeze():
    if not mc.PROTOCOL.is_file():
        pytest.skip('the protocol has not been drafted into the repository yet')
    p = json.loads(mc.PROTOCOL.read_text())
    assert mc.validate_protocol(p) == p


def test_a_changed_protocol_is_refused():
    p = dict(mc.draft_protocol(), split_seed=1)
    with pytest.raises(ValueError, match='differs from motion_controls.draft_protocol'):
        mc.validate_protocol(p)


def test_a_frozen_protocol_needs_its_projection_within_the_cap():
    draft = mc.draft_protocol()
    frozen = dict(draft, frozen=True, frozen_at_utc='2026-09-14T00:00:00Z', status=mc.FROZEN_STATUS,
                  resource_decision='x', pilot_projection={'cap_hours': mc.CAP_HOURS, 'workers': mc.WORKERS,
                                                           'decision_hours': 1.5})
    assert mc.validate_protocol(frozen) == frozen
    for broken in ({'decision_hours': mc.CAP_HOURS + .1}, {'workers': 1}, {'cap_hours': 99}):
        with pytest.raises(ValueError, match='frozen protocol records'):
            mc.validate_protocol(dict(frozen, pilot_projection={**frozen['pilot_projection'], **broken}))


def test_fit_sources_reuse_views7_only_at_one_hidden_layer():
    assert mc.fit_sources(1) == {**{arm: 'fitted' for arm in mc.ARMS},
                                 'single_layer_first': 'identical_to_views7', 'single_layer_last': 'identical_to_views7'}
    assert set(mc.fit_sources(2).values()) == {'fitted'}


def test_freeze_refuses_a_projection_over_the_cap_or_a_missing_smoke(tmp_path):
    draft_path = tmp_path/'draft.json'
    draft = mc.draft_protocol()
    draft_path.write_text(json.dumps(draft, indent=2, sort_keys=True) + '\n')
    stages = tmp_path/'stages.json'
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'ok'}}))
    from experiments.make_revision.evaluation import config_id
    pilot = tmp_path/'pilot.json'

    def write_pilot(hours, checks=True, probes=True):
        pilot.write_text(json.dumps({'protocol_hash': config_id(draft), 'calibrated_projection': {
            'serial_hours': 1., 'serial_hours_over_workers': .1, 'simulated_makespan_hours': hours,
            'longest_job_hours': .1, 'datasets': {}},
            'records': [], 'reproduction_probes': {},
            'decision': {'hours': hours, 'within_cap': hours <= mc.CAP_HOURS, 'workers': mc.WORKERS,
                         'checks_passed': checks, 'probes_reproduced': probes, 'rule': 'r'}}))
    write_pilot(mc.CAP_HOURS + 1)
    with pytest.raises(ValueError, match='Not frozen'):
        mc.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    write_pilot(1., checks=False)
    with pytest.raises(ValueError, match='Not frozen'):
        mc.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    write_pilot(1.)
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'failed'}}))
    with pytest.raises(ValueError, match='passing synthetic smoke'):
        mc.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'ok'}}))
    frozen = mc.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    assert frozen['frozen'] and frozen['pilot_projection']['decision_hours'] == 1.
    assert mc.validate_protocol(json.loads((tmp_path/'frozen.json').read_text())) == frozen


def test_freeze_never_replaces_another_protocol(tmp_path):
    from experiments.make_revision.evaluation import config_id
    draft_path, output = tmp_path/'draft.json', tmp_path/'frozen.json'
    draft = mc.draft_protocol()
    draft_path.write_text(json.dumps(draft, indent=2, sort_keys=True) + '\n')
    pilot, stages = tmp_path/'pilot.json', tmp_path/'stages.json'
    pilot.write_text(json.dumps({'protocol_hash': config_id(draft), 'records': [], 'reproduction_probes': {},
                                 'calibrated_projection': {'serial_hours': 1., 'serial_hours_over_workers': .1,
                                                           'simulated_makespan_hours': 1., 'longest_job_hours': .1,
                                                           'datasets': {}},
                                 'decision': {'hours': 1., 'within_cap': True, 'workers': mc.WORKERS,
                                              'checks_passed': True, 'probes_reproduced': True, 'rule': 'r'}}))
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'ok'}}))
    output.write_text(json.dumps({'something': 'else'}))
    with pytest.raises(FileExistsError, match='Refusing to replace'):
        mc.freeze(draft_path, pilot, stages, output)
    assert json.loads(output.read_text()) == {'something': 'else'}


def test_planned_job_records_the_depth_the_arm_seeds_and_the_reuse():
    p = mc.draft_protocol()
    record = {'config': {'aggregation': 'majority', 'batch_size': 32, 'degree_offset': 0, 'embed_scale': 1,
                         'iterations': 200, 'learning_rate': .1, 'n_views': 7, 'strategy': 'diverse',
                         'validation_ratio': .1, 'widths': [64, 128]},
              'config_id': 'abc', 'reference_prediction_hashes': {str(s): 'h' for s in p['fit_seeds']},
              'reference_outer_seconds': {str(s): 1. for s in p['fit_seeds']}}
    split = {'outer_repeat': 0, 'outer_fold': 0, 'train': list(range(200)), 'test': list(range(200, 250))}
    from experiments.make_revision.bridge import resolve_selected
    selected = resolve_selected(record['config'], 8, 200)
    reference = {'name': 'r', 'ablation': {'jobs': {('d', 0, 0): {'config_id': 'abc', 'selected': selected}}}}
    job = mc.planned_job(p, reference, 'd', 0, split, record, 8)
    assert job['hidden_layers'] == 2 and set(job['fit_sources'].values()) == {'fitted'}
    assert list(job['arm_seeds']) == list(mc.ARMS) and len(job['arm_seeds']['frozen'][str(p['fit_seeds'][0])]) == 7
    assert job['check_uninstrumented'] and job['reference_outer_seconds'] == 3.


# ----------------------------------------------------------------------------- the analysis

def synthetic_report(datasets=('a', 'b', 'c'), folds=15, seeds=(1, 2, 3), two_layer=5, gap=.02, rng_seed=3):
    """A motion_controls.summary-shaped report over `datasets`, with views7 better than every control arm by `gap`."""
    rng = np.random.RandomState(rng_seed)
    rows, summaries = {}, {}
    for name in datasets:
        entries = []
        for index in range(folds):
            repeat, fold = divmod(index, 5)
            for arm in mc.ARMS:
                advantage = 0. if arm == 'views7' else gap
                for seed in seeds:
                    entries.append({'dataset_id': name, 'model_id': arm, 'arm_id': arm, 'outer_repeat': repeat,
                                    'outer_fold': fold, 'model_seed': seed, 'status': 'ok',
                                    'accuracy': .8 - advantage + rng.normal(0, .004),
                                    'error': .2 + advantage, 'balanced_accuracy': .8, 'macro_f1': .8,
                                    'hidden_layers': 2 if index < two_layer else 1,
                                    'per_class_recall': {'0': .8, '1': .8},
                                    'neighborhood_purity_test': .7, 'neighborhood_purity_train': .75,
                                    'displacement': [{'layer': 0, 'kind': 'hidden', 'mean_normalized_footrule': .1,
                                                      'mean_changed_share': .9, 'unchanged_share': .1},
                                                     {'layer': 1, 'kind': 'output', 'mean_normalized_footrule': .2,
                                                      'mean_changed_share': .9, 'unchanged_share': .1}]})
        rows[name] = entries
        summaries[name] = {'arms': {arm: {'metrics': {'accuracy': {'mean': float(np.mean(
            [r['accuracy'] for r in entries if r['model_id'] == arm]))}},
            'neighborhood_purity': {'test': {'mean': .7, 'sd': .01, 'n': folds},
                                    'train': {'mean': .75, 'sd': .01, 'n': folds}},
            'per_class_recall': {'0': {'mean': .8, 'sd': .01, 'n': folds}, '1': {'mean': .8, 'sd': .01, 'n': folds}},
            'displacement': [{'layer': 0, 'kind': 'hidden',
                              'mean_normalized_footrule': {'mean': .1, 'sd': .01, 'n': folds},
                              'unchanged_share': {'mean': .1, 'sd': .01, 'n': folds}}]} for arm in mc.ARMS},
            'views7_reproduces_reference': {'matching_fold_seeds': folds * len(seeds),
                                            'total_fold_seeds': folds * len(seeds)}}
    return {'model_rows': rows, 'summaries': summaries}


def synthetic_protocol(datasets, seeds=(1, 2, 3)):
    return dict(mc.draft_protocol(), datasets=list(datasets), fit_seeds=list(seeds), outer_folds=5, outer_repeats=3,
                test_train_ratio=.25, confidence=.95)


def test_the_primary_family_is_holm_adjusted_within_each_arm():
    names = tuple('abcdefghijklmnopq')          # seventeen datasets
    report, p = synthetic_report(names), synthetic_protocol(names)
    rows = cm.family_rows(report, p, mc.PRIMARY_ARMS, label='primary')
    assert len(rows) == len(mc.PRIMARY_ARMS) * len(names)
    assert {row['arm'] for row in rows} == set(mc.PRIMARY_ARMS)
    for arm in mc.PRIMARY_ARMS:
        block = [row for row in rows if row['arm'] == arm]
        assert all(row['n_folds'] == 15 and row['df'] == 14 and row['test_train_ratio'] == .25 for row in block)
        assert all(row['mean_difference'] > 0 for row in block)          # views7 minus the arm, positive favours ArrowFlow
        assert all(row['holm_p_approximate'] >= row['p_approximate'] - 1e-12 for row in block)
        assert max(row['holm_p_approximate'] for row in block) <= 1.


def test_the_named_subset_carries_the_family_adjustment(fits=None):
    names = tuple('abcdefghijklmno') + mc.NAMED_SUBSET
    report, p = synthetic_report(names), synthetic_protocol(names)
    rows = cm.family_rows(report, p, mc.PRIMARY_ARMS, label='primary')
    subset = cm.named_subset_rows(rows, mc.NAMED_SUBSET)
    assert {row['dataset'] for row in subset} == set(mc.NAMED_SUBSET)
    assert len(subset) == len(mc.PRIMARY_ARMS) * len(mc.NAMED_SUBSET)
    assert all('not adjusted again' in row['status'] for row in subset)
    for row in subset:
        parent = next(r for r in rows if (r['arm'], r['dataset']) == (row['arm'], row['dataset']))
        assert row['holm_p_approximate'] == parent['holm_p_approximate']


def test_single_layer_rows_use_only_the_two_hidden_layer_folds():
    names = ('a', 'b')
    report, p = synthetic_report(names, two_layer=5), synthetic_protocol(names)
    jobs = [{'dataset_id': name, 'outer_repeat': index // 5, 'outer_fold': index % 5,
             'hidden_layers': 2 if index < 5 else 1} for name in names for index in range(15)]
    rows = cm.single_layer_rows(report, p, jobs)
    assert {row['arm'] for row in rows} == set(mc.DEPTH_ARMS)
    for row in rows:
        assert row['two_hidden_layer_folds'] == 5 and row['n_folds'] == 5 and row['df'] == 4
        assert row['outer_folds'] == 15 and 'not Holm-adjusted' in row['status']


def test_single_layer_rows_report_a_one_layer_dataset_as_identical():
    names = ('a',)
    report, p = synthetic_report(names, two_layer=0), synthetic_protocol(names)
    jobs = [{'dataset_id': 'a', 'outer_repeat': index // 5, 'outer_fold': index % 5, 'hidden_layers': 1}
            for index in range(15)]
    rows = cm.single_layer_rows(report, p, jobs)
    assert all(row['n_folds'] == 0 and row['ci_low'] is None and 'identical to views7' in row['status'] for row in rows)


def test_the_descriptive_tables_cover_every_dataset_and_arm():
    names = ('a', 'b')
    report, p = synthetic_report(names), synthetic_protocol(names)
    assert len(cm.purity_rows(report, p)) == len(names) * len(mc.ARMS) * 2
    assert len(cm.recall_rows(report, p)) == len(names) * len(mc.ARMS) * 2
    assert len(cm.displacement_rows(report, p)) == len(names) * len(mc.ARMS)


def test_the_analysis_refuses_an_incomplete_run_and_writes_nothing(tmp_path):
    run = tmp_path/'run'
    run.mkdir()
    with pytest.raises(RunComparisonError, match='is not complete'):
        cm.analyse(run, tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()
    for name in cm.RUN_FILES[:-1] + (mc.SUMMARY_JSON, mc.SUMMARY_CSV):
        (run/name).write_text('{}')
    (run/'planned_jobs.json').write_text(json.dumps([{'stem': 'a__r0f0', 'dataset_id': 'a'}]))
    with pytest.raises(RunComparisonError) as info:
        cm.analyse(run, tmp_path/'analysis')
    assert 'planned job files missing' in str(info.value) and 'reference_selections.json' in str(info.value)
    assert not (tmp_path/'analysis').exists()


def test_the_analysis_command_exits_two_when_it_refuses(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        cm.main(['analyse', '--run', str(tmp_path/'absent'), '--output', str(tmp_path/'analysis')])
    assert info.value.code == 2
    assert 'refused' in capsys.readouterr().err
    assert not (tmp_path/'analysis').exists()


def test_check_protocol_requires_the_declared_arms_and_design():
    p = dict(mc.draft_protocol(), frozen=True)
    assert cm.check_protocol(p, smoke=True)['primary_family']['arms'] == list(mc.PRIMARY_ARMS)
    with pytest.raises(RunComparisonError, match='six arms'):
        cm.check_protocol(dict(p, arms=['views7']), smoke=True)
    with pytest.raises(RunComparisonError, match='q = 0.25'):
        cm.check_protocol(dict(p, test_train_ratio=.1), smoke=True)


def test_the_analysis_outputs_are_all_or_none(tmp_path):
    from experiments.make_revision.compare_runs import write_outputs
    (tmp_path/'motion_primary_family.csv').parent.mkdir(exist_ok=True)
    (tmp_path/'motion_primary_family.csv').write_text('different\n')
    with pytest.raises(FileExistsError):
        write_outputs(tmp_path, {'motion_primary_family.csv': 'a\n', 'motion_analysis.json': '{}\n'})
    assert not (tmp_path/'motion_analysis.json').exists()
