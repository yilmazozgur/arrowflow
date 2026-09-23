"""The readout-matched representation test (representation_test, compare_representation): the Kendall-kernel SVC readout
and its selection, the three representations and their identities, the job checks, the protocol and freeze rules, the
projection, and the analysis gate, families, fixed interpretation and tables. Small synthetic networks only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import stats
from experiments.make_revision import compare_representation as cr
from experiments.make_revision import multiview
from experiments.make_revision import neighbour_baselines as nb
from experiments.make_revision import representation_test as rt
from experiments.make_revision import run_knn_ablation as base
from experiments.make_revision import training_diagnostics as td
from arrowflow.ranking import inverse_positions
from experiments.make_revision.compare_runs import RunComparisonError
from experiments.make_revision.comparisons import derive_seed
from experiments.make_revision.evaluation import config_id
from experiments.make_revision.knn_controls import MultiViewInputKNN, UntrainedMultiViewArrowFlowKNN
from experiments.make_revision.models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from experiments.make_revision.multiview import MultiViewArrowFlowKNN, select_knn_readout, view_strategy
from experiments.make_revision.secondary_studies import majority

SEED = 8129
REPO = Path(rt.__file__).resolve().parents[2]


def tiny_data(n=96, seed=5):
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], n // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    return X, y


def tiny_params(widths=(6,), iterations=8):
    selected = dict(n_views=7, strategy='diverse', embed_dim=8, degree=1, widths=list(widths), learning_rate=.2,
                    iterations=iterations, batch_size=16, validation_ratio=.1, augment=False, aggregation='majority')
    variants = dict(base.knn_ablation_variants(selected))
    return {variant: variants[variant] for variant in rt.ABLATION_VARIANTS}


def random_orders(rows, items, seed):
    rng = np.random.RandomState(seed)
    return np.stack([rng.permutation(items) for _ in range(rows)])


@pytest.fixture(scope='module')
def data():
    X, y = tiny_data()
    train, test = list(range(0, 96, 4)) + list(range(1, 96, 4)) + list(range(2, 96, 4)), list(range(3, 96, 4))
    return X, y, sorted(train), test


@pytest.fixture(scope='module')
def fitted(data):
    X, y, train, test = data
    params = tiny_params(widths=(6, 8))
    return params, rt.fit_representations(params, SEED, X[train], y[train], X[test])


# ----------------------------------------------------------------------------- the Kendall kernel on hidden rankings

def test_a_hidden_ranking_enters_the_kernel_as_the_order_of_its_filters():
    positions = inverse_positions(random_orders(12, 9, seed=2))     # rows: the position of each of 9 filters
    orders = inverse_positions(positions)                           # the filters from nearest to farthest
    kernel = nb.kendall_kernel(orders)
    i, j = np.triu_indices(9, 1)
    signs = np.sign(positions[:, j] - positions[:, i]).astype(float)    # V4's pair features on the positions
    assert np.array_equal(kernel, signs @ signs.T / len(i))
    for a in range(4):
        for b in range(4):
            assert kernel[a, b] == pytest.approx(stats.kendalltau(positions[a], positions[b]).statistic, abs=1e-12)


def test_the_svc_selection_uses_the_splits_that_choose_the_knn_readout(monkeypatch):
    seen = {}

    class Recording(multiview.StratifiedKFold):
        def split(self, X, y=None, groups=None):
            folds = [(a.tolist(), b.tolist()) for a, b in super().split(X, y, groups)]
            seen.setdefault(self.random_state, []).append(folds)
            return iter([(np.asarray(a), np.asarray(b)) for a, b in folds])

    monkeypatch.setattr(multiview, 'StratifiedKFold', Recording)
    monkeypatch.setattr(rt, 'StratifiedKFold', Recording)
    y = np.tile([0, 1, 2], 20)
    orders = random_orders(len(y), 7, seed=3)
    seed = derive_seed(derive_seed(SEED, 'view', 2), 'readout_selection')
    select_knn_readout(inverse_positions(orders), y, seed=seed)
    rt.select_svc_penalty(nb.kendall_kernel(orders), y, seed=seed)
    assert len(seen[seed]) == 2 and seen[seed][0] == seen[seed][1]


def test_each_split_reads_exactly_its_own_block_of_the_gram_matrix():
    orders = random_orders(40, 8, seed=4)
    gram = nb.kendall_kernel(orders)
    a, b = np.arange(0, 40, 2), np.arange(1, 40, 2)
    assert np.array_equal(gram[np.ix_(a, a)], nb.kendall_kernel(orders[a]))
    assert np.array_equal(gram[np.ix_(b, a)], nb.kendall_kernel(orders[b], orders[a]))


def test_the_svc_penalty_is_the_best_mean_score_with_ties_to_the_smallest_c(monkeypatch):
    class Stub:
        def __init__(self, kernel, C, probability, max_iter):
            self.C = C

        def fit(self, gram, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, gram):
            right = self.C >= 10                                        # C = 10 and C = 100 tie at the best score
            return np.zeros(len(gram), dtype=int) if right else np.ones(len(gram), dtype=int)

    monkeypatch.setattr(rt, 'SVC', Stub)
    y = np.array([0] * 12 + [1] * 6)
    selection = rt.select_svc_penalty(np.eye(len(y)), y, seed=1)
    assert selection['C'] == 10 and selection['folds'] == 3
    assert selection['candidate_scores']['10'] == selection['candidate_scores']['100'] == selection['inner_score']
    assert selection['candidate_scores']['0.1'] < selection['inner_score']


def test_the_svc_penalty_grid_is_the_kendall_svc_baselines():
    assert rt.SVC_C == nb.SVC_C == [.1, 1, 10, 100]


def test_the_selection_needs_two_rows_of_every_class():
    with pytest.raises(ValueError, match='at least two rows per class'):
        rt.select_svc_penalty(np.eye(5), np.array([0, 0, 1, 1, 2]), seed=1)
    assert rt.selection_folds(np.array([0, 0, 1, 1, 1, 1])) == 2 and rt.selection_folds(np.tile([0, 1], 9)) == 3


def test_the_kendall_svc_readout_records_its_own_selection():
    y = np.tile([0, 1, 2], 16)
    orders = random_orders(len(y), 6, seed=5)
    predicted, record = rt.kendall_svc_readout(orders, y, orders[:9], seed=7)
    assert predicted.shape == (9,) and set(predicted) <= {0, 1, 2}
    assert record['C'] in rt.SVC_C and record['candidate_scores'][rt.penalty_key(record['C'])] == record['inner_score']
    assert record['inner_score'] == max(record['candidate_scores'].values())
    assert record['vocabulary'] == 6 and record['pairs'] == 15 and record['libsvm_fit_status'] == 0


def test_a_kernel_above_the_cap_is_refused_never_subsampled():
    y = np.tile([0, 1], 6)
    with pytest.raises(nb.KernelTooLarge, match='never subsampled'):
        rt.kendall_svc_readout(random_orders(len(y), 5, seed=1), y, random_orders(2, 5, seed=2), seed=1,
                               cap={'max_training_rows': 10, 'max_feature_entries': 10 ** 9})


def test_the_views_are_combined_with_arrowflows_tie_rule():
    assert rt.majority is majority
    assert majority(np.array([[2, 1], [1, 2]])).tolist() == [1, 1]         # a tie goes to the lowest class label


# ----------------------------------------------------------------------------- the three representations

def test_every_arm_has_seven_views_of_test_predictions(fitted, data):
    _, fit = fitted
    assert set(fit['views']) == set(rt.ARMS)
    assert all(views.shape == (7, len(data[3])) for views in fit['views'].values())
    assert all(len(fit['readouts'][arm]) == 7 for arm in rt.ARMS)


def test_knn_trained_is_arrowflow_knn_refitted(fitted, data):
    X, y, train, test = data
    params, fit = fitted
    seed_fit(SEED)
    model = MultiViewArrowFlowKNN(**params['views7'], seed=SEED).fit(X[train], y[train])
    assert np.array_equal(fit['views']['knn_trained'], np.stack(model.predict_views(X[test])[0]))


def test_knn_untrained_and_knn_input_are_the_registered_controls(fitted, data):
    X, y, train, test = data
    params, fit = fitted
    seed_fit(SEED)
    untrained = UntrainedMultiViewArrowFlowKNN(**params['untrained'], seed=SEED).fit(X[train], y[train])
    control = MultiViewInputKNN(**params['input_knn'], seed=SEED).fit(X[train], y[train])
    assert np.array_equal(fit['views']['knn_untrained'], np.stack(untrained.predict_views(X[test])[0]))
    assert np.array_equal(fit['views']['knn_input'], np.stack(control.predict_views(X[test])))


def test_every_view_check_passes(fitted):
    _, fit = fitted
    for check in rt.VIEW_CHECKS:
        assert len(fit['checks'][check]) == 7 and all(entry['passed'] for entry in fit['checks'][check]), check


def test_the_untrained_filters_are_where_the_trained_network_began(data):
    X, y, train, test = data
    params = tiny_params(widths=(6, 8))['views7']
    v = 3
    seed_v = derive_seed(SEED, 'view', v)
    encoder = OrdinalEncoder(view_strategy('diverse', v), 8, 1, .3, seed_v).fit(X[train], y[train])
    orders = encoder.transform(X[train])
    net = ArrowFlowEstimator(embed_dim=8, degree=1, widths=params['widths'], iterations=params['iterations'],
                             learning_rate=params['learning_rate'], batch_size=params['batch_size'], seed=seed_v,
                             validation_ratio=params['validation_ratio'])
    net.initialize_orders(orders, y[train])
    initial = td.filter_copy(net.network_)
    seed_fit(SEED)
    untrained = UntrainedMultiViewArrowFlowKNN(**tiny_params(widths=(6, 8))['untrained'], seed=SEED).fit(X[train], y[train])
    assert td.copies_equal(initial, td.filter_copy(untrained.views_[v][1].network_))
    # The same start with no validation checkpoint (the checkpoint may return the initial filters of a short fit): the
    # final iterate has moved away from the filters the untrained network keeps.
    moving = ArrowFlowEstimator(embed_dim=8, degree=1, widths=params['widths'], iterations=20,
                                learning_rate=params['learning_rate'], batch_size=params['batch_size'], seed=seed_v)
    moving.initialize_orders(orders, y[train])
    assert td.copies_equal(initial, td.filter_copy(moving.network_))
    moving.train_initialized(orders, y[train])
    assert not td.copies_equal(initial, td.filter_copy(moving.network_))


def test_the_svc_reads_the_hidden_ranking_the_knn_readout_stores(data):
    X, y, train, test = data
    params = tiny_params()
    model, _, _, _ = base.fit_views7(params['views7'], SEED, X[train], y[train], X[test])
    encoder, net = model.views_[0]
    hidden = net.transform_orders(encoder.transform(X[train]))
    assert np.array_equal(model.readouts_[0].positions_, hidden)
    assert np.array_equal(hidden, net.transform_orders_by_depth(encoder.transform(X[train]))[-1])


def test_training_changes_the_hidden_representation(fitted):
    _, fit = fitted
    assert not np.array_equal(fit['views']['svc_trained'], fit['views']['svc_untrained']) or \
        not np.array_equal(fit['views']['knn_trained'], fit['views']['knn_untrained'])


def test_a_changed_initial_state_fails_the_initial_state_check(data, monkeypatch):
    X, y, train, test = data

    class Shifted(ArrowFlowEstimator):
        def initialize_orders(self, orders, y):
            self.seed = self.seed + 1
            return super().initialize_orders(orders, y)

    monkeypatch.setattr(rt, 'ArrowFlowEstimator', Shifted)
    fit = rt.fit_representations(tiny_params(), SEED, X[train], y[train], X[test])
    assert not any(entry['passed'] for entry in fit['checks']['initial_state'])


# ----------------------------------------------------------------------------- one job

def job_for(data, params, fit=None, *, tamper=None):
    """A planned job and sealed record for the tiny split whose stored hashes are those of `fit` (a first fit)."""
    X, y, train, test = data
    views = fit['views']
    predictions = {arm: majority(views[arm]) for arm in rt.ARMS}
    hashes = {str(SEED): {'knn_views': array_hash(views['knn_trained']), 'views7': array_hash(predictions['knn_trained']),
                          'untrained': array_hash(predictions['knn_untrained']),
                          'input_knn': array_hash(predictions['knn_input'])}}
    if tamper:
        hashes[str(SEED)][tamper] = '0' * 64
    sealed = {'reference_predictions': {str(SEED): predictions['knn_trained'].tolist()},
              'reference_prediction_hashes': {str(SEED): hashes[str(SEED)]['views7']}}
    if tamper == 'views7':
        sealed['reference_predictions'][str(SEED)] = [(label + 1) % 3 for label in sealed['reference_predictions'][str(SEED)]]
    job = {'dataset_id': 'synthetic', 'reference': 'r', 'panel': 'further', 'outer_repeat': 0, 'outer_fold': 0,
           'stem': 'synthetic__r0f0', 'config_id': 'c', 'config': {}, 'selected': {}, 'selected_widths': [6],
           'hidden_layers': 1, 'params': params, 'arms': list(rt.ARMS), 'model_seeds': [SEED], 'first_fold': True,
           'reference_prediction_hashes': sealed['reference_prediction_hashes'],
           'ablation': {'result_file': 'x', 'result_sha256': 'y', 'hashes': hashes}}
    split = {'outer_repeat': 0, 'outer_fold': 0, 'train': train, 'test': test}
    return job, sealed, split


@pytest.fixture(scope='module')
def one_layer(data):
    X, y, train, test = data
    params = tiny_params()
    return params, rt.fit_representations(params, SEED, X[train], y[train], X[test])


def test_a_job_that_reproduces_every_stored_prediction_passes_every_check(data, one_layer):
    X, y, _, test = data
    params, fit = one_layer
    job, sealed, split = job_for(data, params, fit)
    result, arrays = rt.evaluate_job(X, y, split, {'fit_seeds': [SEED]}, job, sealed, dataset_hash='h', code_revision='c')
    assert result['status'] == 'ok', result.get('exception')
    assert all(result['checks'][check]['performed'] and result['checks'][check]['passed'] for check in rt.JOB_CHECKS)
    assert sorted(arrays) == sorted(f'{arm}__s{SEED}' for arm in rt.ARMS)
    assert len(result['models']) == len(rt.ARMS) and len(result['predictions']) == len(rt.ARMS) * len(test)
    assert result['reproduction'][0]['reproduced'] and result['reproduction'][0]['reference_views']
    for row in result['models']:
        assert row['prediction_hash'] == array_hash(majority(arrays[f'{row["arm_id"]}__s{SEED}']))


@pytest.mark.parametrize('tamper, check', [('views7', 'reference_predictions'), ('knn_views', 'reference_views'),
                                           ('untrained', 'untrained_knn_reproduced'), ('input_knn', 'input_knn_reproduced')])
def test_a_job_that_does_not_reproduce_a_stored_prediction_fails(data, one_layer, tamper, check):
    X, y, _, _ = data
    params, fit = one_layer
    job, sealed, split = job_for(data, params, fit, tamper=tamper)
    result, arrays = rt.evaluate_job(X, y, split, {'fit_seeds': [SEED]}, job, sealed, dataset_hash='h', code_revision='c')
    assert result['status'] == 'failed' and result['check_failed'] and arrays is None
    assert result['checks'][check]['passed'] is False and check in result['exception']
    assert all(row['status'] == 'failed' for row in result['models'])


def test_the_pilot_skips_only_the_checks_against_stored_outer_predictions(data, one_layer):
    X, y, train, _ = data
    params, fit = one_layer
    job, sealed, split = job_for(data, params, fit)
    result, _ = rt.evaluate_job(X, y, split, {'fit_seeds': [SEED]}, job, sealed, dataset_hash='h', code_revision='c',
                                query=train[::4])
    assert result['status'] == 'ok' and result['identity']['query'] == 'training_rows_only'
    assert all(result['checks'][check]['performed'] is False for check in rt.REFERENCE_CHECKS)
    assert all(result['checks'][check]['passed'] is True for check in rt.VIEW_CHECKS)
    assert set(result['timing']['seconds_by_part']) == set(rt.PARTS)


def test_a_failed_view_check_fails_the_job(data, one_layer, monkeypatch):
    X, y, _, _ = data
    params, fit = one_layer
    job, sealed, split = job_for(data, params, fit)
    monkeypatch.setattr(rt.td, 'copies_equal', lambda a, b, orders=True: False)
    result, arrays = rt.evaluate_job(X, y, split, {'fit_seeds': [SEED]}, job, sealed, dataset_hash='h', code_revision='c')
    assert result['status'] == 'failed' and result['check_failed'] and 'initial_state' in result['exception']
    assert result['checks']['initial_state']['passed'] is False and arrays is None


# ----------------------------------------------------------------------------- stored predictions and the plan

def write_ablation(directory, name, seeds, hashes, *, views7=None):
    directory.mkdir(parents=True, exist_ok=True)
    fits = [{'fit_id': f'views7__s{seed}', 'knn_views_hash': hashes[seed]['knn_views']} for seed in seeds]
    models = [{'variant_id': variant, 'model_seed': seed, 'prediction_hash': hashes[seed][variant]}
              for seed in seeds for variant in rt.ABLATION_VARIANTS]
    (directory/'results').mkdir(exist_ok=True)
    (directory/'results'/f'{name}__r0f0.json').write_text(json.dumps({'status': 'ok', 'fits': fits, 'models': models}))
    rows = {name: [dict(row, outer_repeat=0, outer_fold=0) for row in models]}
    if views7 is not None:
        rows[name][0]['prediction_hash'] = views7
    return rows


def test_the_ablation_record_reads_every_stored_hash(tmp_path):
    seeds = [1, 2]
    hashes = {seed: {'knn_views': f'v{seed}', 'views7': f'r{seed}', 'untrained': f'u{seed}', 'input_knn': f'i{seed}'}
              for seed in seeds}
    rows = write_ablation(tmp_path, 'd', seeds, hashes)
    reference = {'ablation': {'directory': str(tmp_path)}}
    sealed = {'reference_prediction_hashes': {'1': 'r1', '2': 'r2'}}
    record = rt.ablation_record(reference, 'd', {'outer_repeat': 0, 'outer_fold': 0}, seeds, rows, sealed)
    assert record['hashes'] == {'1': {'knn_views': 'v1', 'views7': 'r1', 'untrained': 'u1', 'input_knn': 'i1'},
                                '2': {'knn_views': 'v2', 'views7': 'r2', 'untrained': 'u2', 'input_knn': 'i2'}}
    assert record['result_sha256'] == hashlib.sha256((tmp_path/'results'/'d__r0f0.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='does not reproduce the registered'):
        rt.ablation_record(reference, 'd', {'outer_repeat': 0, 'outer_fold': 0}, seeds, rows,
                           {'reference_prediction_hashes': {'1': 'r1', '2': 'other'}})
    disagreeing = write_ablation(tmp_path/'b', 'd', seeds, hashes, views7='changed')
    with pytest.raises(ValueError, match='disagree'):
        rt.ablation_record({'ablation': {'directory': str(tmp_path/'b')}}, 'd', {'outer_repeat': 0, 'outer_fold': 0},
                           seeds, disagreeing, sealed)


def test_the_planned_job_carries_the_ablations_parameters_and_refuses_others():
    p = rt.draft_protocol()
    record = {'config': {'aggregation': 'majority', 'batch_size': 32, 'degree_offset': 0, 'embed_scale': 1,
                         'iterations': 200, 'learning_rate': .1, 'n_views': 7, 'strategy': 'diverse',
                         'validation_ratio': .1, 'widths': [64, 128]},
              'config_id': 'abc', 'reference_prediction_hashes': {str(s): 'h' for s in p['fit_seeds']},
              'reference_outer_seconds': {str(s): 1. for s in p['fit_seeds']}}
    split = {'outer_repeat': 0, 'outer_fold': 0, 'train': list(range(200)), 'test': list(range(200, 250))}
    from experiments.make_revision.bridge import resolve_selected
    selected = resolve_selected(record['config'], 8, 200)
    variants = [{'variant_id': v, 'params': params} for v, params in base.knn_ablation_variants(selected)]
    reference = {'name': 'newdata_batch1', 'ablation': {'jobs': {('ionosphere', 0, 0): {
        'config_id': 'abc', 'selected': selected, 'variants': variants}}}}
    job = rt.planned_job(p, reference, 'ionosphere', 0, split, record, 8, {'hashes': {}})
    assert job['panel'] == 'further' and job['hidden_layers'] == 2 and job['reference_outer_seconds'] == 3.
    assert job['params'] == {v['variant_id']: v['params'] for v in variants if v['variant_id'] in rt.ABLATION_VARIANTS}
    variants[6]['params'] = dict(variants[6]['params'], widths=[128])
    with pytest.raises(ValueError, match='differ from the ablation plan'):
        rt.planned_job(p, reference, 'ionosphere', 0, split, record, 8, {'hashes': {}})


def test_the_projection_prices_a_piloted_dataset_by_its_own_ratio_and_the_rest_by_the_largest():
    p = rt.draft_protocol()
    jobs = [{'dataset_id': 'a', 'reference_outer_seconds': 100.}, {'dataset_id': 'b', 'reference_outer_seconds': 100.}]
    records = [{'dataset_id': 'a', 'job_to_views7_ratio': 1.5}, {'dataset_id': 'c', 'job_to_views7_ratio': 2.}]
    plan = rt.calibrated_projection(p, jobs, records, workers=2)
    assert plan['datasets']['a']['seconds'] == 150. and plan['datasets']['b']['seconds'] == 200.
    assert plan['serial_hours'] == pytest.approx(350 / 3600) and plan['simulated_makespan_hours'] == pytest.approx(200 / 3600)


# ----------------------------------------------------------------------------- the protocol

def test_the_draft_protocol_validates_and_declares_the_design():
    p = rt.draft_protocol()
    assert rt.validate_protocol(p) == p
    assert len(p['datasets']) == 17 and p['development_datasets'] == list(rt.DEVELOPMENT)
    assert p['further_datasets'] == list(rt.FURTHER) and len(rt.FURTHER) == 10 and not set(rt.FURTHER) & set(rt.DEVELOPMENT)
    assert p['arms'] == list(rt.ARMS) and not p['frozen'] and p['wallclock_cap_hours'] == 4 and p['workers'] == 16
    block = p['analysis']
    assert block['primary_family']['datasets'] == list(rt.FURTHER) and block['primary_family']['size'] == 10
    assert (block['primary_family']['model_a'], block['primary_family']['model_b']) == ('svc_trained', 'svc_untrained')
    assert (block['secondary_family']['model_a'], block['secondary_family']['model_b']) == ('svc_trained', 'svc_input')
    assert block['descriptive']['development_datasets']['label'] == 'screened before this protocol'
    rule = block['interpretation']
    assert rule['primary']['if_met'] == 'the paper states that training improves the representation itself'
    assert rule['primary']['otherwise'] == ('the paper states that training improves the nearest-neighbour '
                                            'neighbourhoods but not the representation read by a strong fixed readout')
    assert rule['primary']['most_threshold'] == 6 and 'softened' in rule['status']
    assert p['readouts']['svc']['grid'] == {'C': [.1, 1, 10, 100]} and '0.54 pp' in p['screening_disclosure']['screen']


def test_the_screening_disclosure_names_the_committed_screen():
    root = REPO.parent
    for relative, digest in rt.SCREENING['files'].items():
        path = root/relative
        if not path.is_file():
            pytest.skip('the review records are not in this checkout')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_the_committed_protocol_file_is_this_draft_or_its_freeze():
    if not rt.PROTOCOL.is_file():
        pytest.skip('the protocol has not been drafted into the repository yet')
    p = json.loads(rt.PROTOCOL.read_text())
    assert rt.validate_protocol(p) == p


@pytest.mark.parametrize('change', [
    {'split_seed': 1}, {'further_datasets': list(rt.FURTHER[:9])}, {'arms': list(rt.ARMS[:3])},
    {'readouts': 'changed'}, {'analysis': 'changed'}])
def test_a_changed_protocol_is_refused(change):
    with pytest.raises(ValueError, match='differs from representation_test.draft_protocol'):
        rt.validate_protocol(dict(rt.draft_protocol(), **change))


def test_the_production_references_require_the_registered_panels():
    p = rt.draft_protocol(development_reference='newdata_batch1')
    with pytest.raises(ValueError, match='seven development and the ten further'):
        rt.validate_protocol(p)


def test_every_dataset_is_within_the_kernel_cap():
    capacity = rt.kernel_capacity()
    assert sorted(capacity) == sorted(rt.DEVELOPMENT + rt.FURTHER)
    assert all(entry['within_cap'] and entry['vocabulary'] == 128 for entry in capacity.values())


def test_a_frozen_protocol_needs_its_projection_within_the_cap():
    frozen = dict(rt.draft_protocol(), frozen=True, frozen_at_utc='2026-09-23T00:00:00Z', status=rt.FROZEN_STATUS,
                  resource_decision='x', pilot_projection={'cap_hours': 4, 'workers': 16, 'decision_hours': 1.5})
    assert rt.validate_protocol(frozen) == frozen
    for broken in ({'decision_hours': 4.1}, {'workers': 8}, {'cap_hours': 6}):
        with pytest.raises(ValueError, match='frozen protocol records'):
            rt.validate_protocol(dict(frozen, pilot_projection={**frozen['pilot_projection'], **broken}))


def write_pilot(path, draft, hours, *, checks=True, probes=True):
    path.write_text(json.dumps({'protocol_hash': config_id(draft), 'code_revision': 'r', 'records': [],
                                'reproduction_probes': {}, 'calibrated_projection': {
                                    'serial_hours': 1., 'serial_hours_over_workers': .1, 'simulated_makespan_hours': hours,
                                    'longest_job_hours': .1, 'datasets': {}},
                                'decision': {'hours': hours, 'within_cap': hours <= 4, 'workers': 16,
                                             'checks_passed': checks, 'probes_reproduced': probes, 'rule': 'r'}}))


def test_freeze_refuses_a_projection_over_the_cap_failed_checks_or_a_missing_smoke(tmp_path):
    draft = rt.draft_protocol()
    draft_path, pilot, stages = tmp_path/'draft.json', tmp_path/'pilot.json', tmp_path/'stages.json'
    draft_path.write_text(json.dumps(draft, indent=2, sort_keys=True) + '\n')
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'ok'}}))
    write_pilot(pilot, draft, 4.5)
    with pytest.raises(ValueError, match='Not frozen'):
        rt.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    write_pilot(pilot, draft, 1., checks=False)
    with pytest.raises(ValueError, match='Not frozen'):
        rt.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    write_pilot(pilot, draft, 1., probes=False)
    with pytest.raises(ValueError, match='Not frozen'):
        rt.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    write_pilot(pilot, draft, 1.)
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'failed'}}))
    with pytest.raises(ValueError, match='passing synthetic smoke'):
        rt.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'ok'}}))
    frozen = rt.freeze(draft_path, pilot, stages, tmp_path/'frozen.json')
    assert frozen['frozen'] and frozen['pilot_projection']['decision_hours'] == 1.
    assert rt.validate_protocol(json.loads((tmp_path/'frozen.json').read_text())) == frozen


def test_freeze_never_replaces_another_protocol(tmp_path):
    draft = rt.draft_protocol()
    draft_path, pilot, stages, output = tmp_path/'draft.json', tmp_path/'pilot.json', tmp_path/'stages.json', tmp_path/'f.json'
    draft_path.write_text(json.dumps(draft, indent=2, sort_keys=True) + '\n')
    write_pilot(pilot, draft, 1.)
    stages.write_text(json.dumps({'summary': 's', 'smoke': {'status': 'ok'}}))
    output.write_text(json.dumps({'something': 'else'}))
    with pytest.raises(FileExistsError, match='Refusing to replace'):
        rt.freeze(draft_path, pilot, stages, output)
    assert json.loads(output.read_text()) == {'something': 'else'}


def test_the_summary_lines_carry_every_reproduction_count():
    report = {'summaries': {'iris': {'reproduction': {check: {'matching_fold_seeds': 45, 'total_fold_seeds': 45}
                                                      for check in rt.REFERENCE_CHECKS}}}}
    assert rt.summary_lines(report) == ['iris: views7 reproduced 45/45 fold-seeds; views 45/45; untrained kNN 45/45; '
                                        'input kNN 45/45']


# ----------------------------------------------------------------------------- the analysis

def synthetic_report(datasets, gaps, folds=15, seeds=(1, 2, 3), rng_seed=3, noise=.004):
    """A representation_test.summary-shaped report: svc_trained beats svc_untrained and svc_input by gaps[name]."""
    rng = np.random.RandomState(rng_seed)
    rows, summaries = {}, {}
    for name in datasets:
        entries = []
        for index in range(folds):
            repeat, fold = divmod(index, 5)
            for arm in rt.ARMS:
                shift = gaps[name] if arm == 'svc_trained' else 0.
                for seed in seeds:
                    value = .8 + shift + rng.normal(0, noise)
                    entries.append({'dataset_id': name, 'model_id': arm, 'arm_id': arm, 'outer_repeat': repeat,
                                    'outer_fold': fold, 'model_seed': seed, 'status': 'ok', 'accuracy': value,
                                    'error': 1 - value, 'balanced_accuracy': value, 'macro_f1': value})
        rows[name] = entries
        summaries[name] = {'arms': {arm: {'metrics': {m: {
            'mean': float(np.mean([r[m] for r in entries if r['model_id'] == arm])), 'outer_fold_sd': .01,
            'mean_within_fold_seed_sd': .01, 'n_folds': folds, 'seeds_per_fold': len(seeds)} for m in rt.METRICS}}
            for arm in rt.ARMS},
            'readout_settings': {arm: ({'views': 315, 'mean_selection_score': .8, 'selected_C': {'1': 315},
                                        'libsvm_fit_status_nonzero': 0, 'fit_warnings': 0, 'selection_warnings': 0,
                                        'mean_support_vectors': 40., 'vocabulary': [128]} if arm.startswith('svc')
                                       else {'views': 315, 'mean_selection_score': .8, 'selected_setting': {'k=5,uniform': 315}})
                                 for arm in rt.ARMS}}
    return {'model_rows': rows, 'summaries': summaries}


def synthetic_protocol():
    return dict(rt.draft_protocol(), fit_seeds=[1, 2, 3])


def test_the_primary_family_is_holm_adjusted_across_the_ten_further_datasets():
    gaps = {name: .02 for name in rt.FURTHER}
    report, p = synthetic_report(rt.FURTHER, gaps), synthetic_protocol()
    rows = cr.family_rows(report, p, rt.FURTHER, *rt.PRIMARY, family='primary')
    assert [row['dataset'] for row in rows] == list(rt.FURTHER) and all(row['panel'] == 'further' for row in rows)
    assert all(row['n_folds'] == 15 and row['df'] == 14 and row['test_train_ratio'] == .25 for row in rows)
    assert all(row['mean_difference'] > 0 and row['holm_p_approximate'] >= row['p_approximate'] - 1e-12 for row in rows)
    from experiments.make_revision.evaluation import holm_adjust
    assert [row['holm_p_approximate'] for row in rows] == holm_adjust([row['p_approximate'] for row in rows])


@pytest.mark.parametrize('gaps, met', [
    ([.03, .001, .001, .001, .001, .001, -.001, -.001, -.001, -.001], True),      # one significant, six higher
    ([.03, .001, .001, .001, .001, -.001, -.001, -.001, -.001, -.001], False),    # one significant, five higher
    ([.0002] * 10, False),                                                        # higher on most, none significant
    ([-.03] + [.0002] * 9, False)])                                               # higher on most, only a significant loss
def test_the_fixed_interpretation_rule(gaps, met):
    report, p = synthetic_report(rt.FURTHER, dict(zip(rt.FURTHER, gaps)), noise=.0005), synthetic_protocol()
    rows = cr.family_rows(report, p, rt.FURTHER, *rt.PRIMARY, family='primary')
    outcome = cr.statements(cr.interpretation(rows), cr.interpretation(rows))
    if gaps[1] == .0002:                   # the rule fails for want of a significant gain, not for want of higher means
        assert outcome['primary']['holm_significant_positive'] == [] and outcome['primary']['higher_mean_count'] >= 6
    assert outcome['primary']['met'] is met
    assert outcome['primary']['statement'] == rt.STATEMENTS['met' if met else 'not_met']
    if gaps[0] < 0:
        assert outcome['primary']['holm_significant_negative'] == [rt.FURTHER[0]]


def test_the_descriptive_tables_carry_no_p_values_and_label_the_development_datasets():
    names = rt.DEVELOPMENT + rt.FURTHER
    report, p = synthetic_report(names, {name: .01 for name in names}), synthetic_protocol()
    rows = cr.descriptive_rows(report, p, rt.DEVELOPMENT, (rt.PRIMARY, rt.SECONDARY), table='development')
    assert len(rows) == 14 and all(row['label'] == 'screened before this protocol' for row in rows)
    assert not {'p_approximate', 'holm_p_approximate'} & set(cr.DESCRIPTIVE_COLUMNS)
    assert len(cr.metric_rows(report, p)) == 17 * len(rt.ARMS) * len(rt.METRICS)
    assert len(cr.settings_rows(report, p)) == 17 * len(rt.ARMS)


def test_the_analysis_refuses_an_incomplete_run_and_writes_nothing(tmp_path):
    run = tmp_path/'run'
    run.mkdir()
    with pytest.raises(RunComparisonError, match='is not complete'):
        cr.analyse(run, tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()
    for name in cr.RUN_FILES[:-1] + (rt.SUMMARY_JSON, rt.SUMMARY_CSV):
        (run/name).write_text('{}')
    (run/'planned_jobs.json').write_text(json.dumps([{'stem': 'a__r0f0', 'dataset_id': 'a'}]))
    with pytest.raises(RunComparisonError) as info:
        cr.analyse(run, tmp_path/'analysis')
    assert 'planned job files missing' in str(info.value) and 'reference_selections.json' in str(info.value)
    assert not (tmp_path/'analysis').exists()


def test_the_analysis_command_exits_two_when_it_refuses(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        cr.main(['analyse', '--run', str(tmp_path/'absent'), '--output', str(tmp_path/'analysis')])
    assert info.value.code == 2 and 'refused' in capsys.readouterr().err
    assert not (tmp_path/'analysis').exists()


def test_check_protocol_requires_the_declared_families_and_design():
    p = dict(rt.draft_protocol(), frozen=True)
    assert cr.check_protocol(p, smoke=True)['primary_family']['model_b'] == 'svc_untrained'
    with pytest.raises(RunComparisonError, match='six arms'):
        cr.check_protocol(dict(p, arms=['svc_trained']), smoke=True)
    with pytest.raises(RunComparisonError, match='q = 0.25'):
        cr.check_protocol(dict(p, test_train_ratio=.1), smoke=True)
    changed = json.loads(json.dumps(p))
    changed['analysis']['primary_family']['model_b'] = 'svc_input'
    with pytest.raises(RunComparisonError, match='declared families'):
        cr.check_protocol(changed, smoke=True)
    with pytest.raises(RunComparisonError, match='not a valid representation-test protocol'):
        cr.check_protocol(dict(p, split_seed=2), smoke=False)


def test_the_analysis_outputs_are_all_or_none(tmp_path):
    from experiments.make_revision.compare_runs import write_outputs
    (tmp_path/'representation_primary_family.csv').write_text('different\n')
    with pytest.raises(FileExistsError):
        write_outputs(tmp_path, {'representation_primary_family.csv': 'a\n', 'representation_analysis.json': '{}\n'})
    assert not (tmp_path/'representation_analysis.json').exists()
