"""Task 20A: the training controls of ArrowFlow-kNN (knn_controls) and the knn_training protocol."""
import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np
import pytest
from sklearn.datasets import load_iris
from experiments.make_revision import bridge
from experiments.make_revision import knn_controls as kc
from experiments.make_revision.comparisons import StableFootruleKNN, derive_seed
from experiments.make_revision.evaluation import (ModelSpec, _fit_predict, canonical_json, config_id, evaluate_fold,
                                                  make_splits, paired_corrected_interval)
from experiments.make_revision.models import ArrowFlowEstimator, seed_fit
from experiments.make_revision.multiview import (MultiViewArrowFlow, MultiViewArrowFlowKNN, MultiViewFootruleKNN,
                                                 select_knn_readout)
from experiments.make_revision.secondary_studies import majority
from arrowflow.ranking import inverse_positions

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'/'2026-09-12'
REAL_KNN = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'/'2026-09-12-bridge-knn'


def small(n=90, seed=0):
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(seed).permutation(len(y))[:n]
    return X[idx], y[idx]


def layer_arrays(net):
    return {key: value for key, value in net.state_snapshot().items() if key.startswith('layer_') or key.startswith('rng_')}


# ----------------------------------------------------------------------------- candidates

def test_control_candidates_are_the_distinct_projections_of_the_bridge_grid_in_canonical_order():
    untrained, inputs = kc.control_candidates(kc.UNTRAINED_MODEL), kc.control_candidates(kc.INPUT_MODEL)
    assert len(untrained) == 8 and len(inputs) == 4
    for candidates, keys in ((untrained, kc.CANDIDATE_KEYS[kc.UNTRAINED_MODEL]), (inputs, kc.CANDIDATE_KEYS[kc.INPUT_MODEL])):
        ids = [config_id(c) for c in candidates]
        assert ids == sorted(ids) and len(set(ids)) == len(ids) and all(set(c) == set(keys) for c in candidates)
        assert {config_id({k: c[k] for k in keys}) for c in bridge.bridge_candidates()} == set(ids)
    assert {(tuple(c['widths']), c['embed_scale'], c['degree_offset']) for c in untrained} == set(
        product(((128,), (64, 128)), (1, 2), (0, -1)))
    assert {(c['embed_scale'], c['degree_offset']) for c in inputs} == set(product((1, 2), (0, -1)))
    assert all(c['n_views'] == 7 and c['strategy'] == 'diverse' and c['aggregation'] == 'majority' for c in untrained + inputs)
    assert not any(key in c for c in untrained + inputs for key in ('learning_rate', 'iterations', 'batch_size', 'validation_ratio'))
    half = [c for c in bridge.bridge_candidates() if c['learning_rate'] == .1]        # the learning rate never matters
    assert kc.control_candidates(kc.UNTRAINED_MODEL, half) == untrained and kc.control_candidates(kc.INPUT_MODEL, half) == inputs
    with pytest.raises(ValueError, match='lacks'):
        kc.project_candidates([{'widths': [128]}], kc.CANDIDATE_KEYS[kc.UNTRAINED_MODEL])


# ----------------------------------------------------------------------------- the untrained network

def test_untrained_hidden_transform_equals_the_trained_estimators_hidden_transform_before_its_first_update(monkeypatch):
    X, y = small()
    captured = []
    original = ArrowFlowEstimator.train_initialized

    def spy(self, orders, labels):                      # the trained estimator's state just before its first update
        captured.append({'seed': self.seed, 'arrays': {k: v.copy() for k, v in layer_arrays(self).items()},
                         'hidden': self.transform_orders(orders), 'update_iter': self.network_.update_iter})
        return original(self, orders, labels)
    monkeypatch.setattr(ArrowFlowEstimator, 'train_initialized', spy)
    params = dict(n_views=3, strategy='diverse', embed_dim=16, degree=2, widths=(16, 32), seed=7)
    seed_fit(1)
    trained = MultiViewArrowFlowKNN(**params, iterations=3, learning_rate=.2, validation_ratio=.1, augment=True).fit(X, y)
    monkeypatch.setattr(ArrowFlowEstimator, 'train_initialized', original)
    seed_fit(2)                                          # another global RNG state: every view reseeds from its own seed
    untrained = kc.UntrainedMultiViewArrowFlowKNN(**params).fit(X, y)
    assert len(captured) == len(untrained.views_) == 3
    for v, ((enc, net), before, (enc_t, net_t)) in enumerate(zip(untrained.views_, captured, trained.views_)):
        assert net.seed == before['seed'] == derive_seed(7, 'view', v) and before['update_iter'] == 0
        assert np.array_equal(enc.transform(X), enc_t.transform(X))                  # the same encoder
        assert all(np.array_equal(value, before['arrays'][key]) for key, value in layer_arrays(net).items())
        assert np.array_equal(net.transform_orders(enc.transform(X)), before['hidden'])
        assert net.network_.update_iter == 0 and net_t.network_.update_iter > 0
    assert any(not np.array_equal(net_t.transform_orders(enc_t.transform(X)), before['hidden'])
               for (enc_t, net_t), before in zip(trained.views_, captured))            # training moved the filters


@pytest.mark.parametrize('setting', [{'learning_rate': .2}, {'iterations': 7}, {'batch_size': 8}, {'validation_ratio': .1},
                                     {'augment': True, 'n_augmentations': 2, 'max_swaps': 3}, {'p_correct': .5}])
def test_initial_filters_ignore_every_setting_that_acts_only_during_training(setting):
    X, y = small()
    from experiments.make_revision.models import OrdinalEncoder
    orders = OrdinalEncoder('random', 16, 1, .3, 11).fit(X, y).transform(X)
    base = ArrowFlowEstimator(embed_dim=16, widths=(16, 32), seed=11).initialize_orders(orders, y)
    other = ArrowFlowEstimator(embed_dim=16, widths=(16, 32), seed=11, **setting).initialize_orders(orders, y)
    assert all(np.array_equal(value, layer_arrays(other)[key]) for key, value in layer_arrays(base).items())
    assert np.array_equal(base.transform_orders(orders), other.transform_orders(orders))


def test_untrained_readout_is_the_arrowflow_knn_readout_selection_on_the_untrained_hidden_rankings():
    X, y = small()
    Xtr, ytr, Xte = X[:60], y[:60], X[60:]
    seed_fit(3)
    model = kc.UntrainedMultiViewArrowFlowKNN(n_views=3, embed_dim=16, degree=1, widths=(32,), seed=5).fit(Xtr, ytr)
    expected_votes = []
    for v, ((enc, net), readout, selection) in enumerate(zip(model.views_, model.readouts_, model.readout_selections_)):
        hidden = net.transform_orders(enc.transform(Xtr))
        assert selection == select_knn_readout(hidden, ytr, seed=derive_seed(derive_seed(5, 'view', v), 'readout_selection'))
        assert isinstance(readout, StableFootruleKNN) and readout.input_kind == 'positions'
        assert (readout.n_neighbors, readout.weights) == (selection['config']['n_neighbors'], selection['config']['weights'])
        expected_votes.append(readout.predict(net.transform_orders(enc.transform(Xte))))
    assert np.array_equal(model.predict(Xte), majority(expected_votes))
    assert all(net.network_.update_iter == 0 for _, net in model.views_)                 # prediction trains nothing
    record = model.readout_record()
    assert record['readout'] == 'knn_hidden' and record['network_state'].startswith('seeded_initial_filters')
    assert [v['config_id'] for v in record['views']] == [s['config_id'] for s in model.readout_selections_]
    seed_fit(99)
    again = kc.UntrainedMultiViewArrowFlowKNN(n_views=3, embed_dim=16, degree=1, widths=(32,), seed=5).fit(Xtr, ytr)
    assert again.readout_selections_ == model.readout_selections_ and np.array_equal(again.predict(Xte), model.predict(Xte))
    assert 'learning_rate' not in model.get_params() and 'validation_ratio' not in model.get_params()
    with pytest.raises(ValueError, match='majority'):
        kc.UntrainedMultiViewArrowFlowKNN(n_views=1, embed_dim=16, degree=1, widths=(8,), aggregation='borda').fit(Xtr, ytr)


# ----------------------------------------------------------------------------- the input control

def test_input_knn_keeps_the_multiview_encoders_and_tunes_each_readout_on_the_encoded_input_positions(monkeypatch):
    from experiments.make_revision import comparisons
    X, y = small()
    Xtr, ytr, Xte = X[:60], y[:60], X[60:]
    params = dict(n_views=3, strategy='diverse', embed_dim=16, degree=2, seed=9)
    fits = []
    original = comparisons.StableFootruleKNN.fit

    def spy(self, X_fit, y_fit, sample_ids=None):
        fits.append((len(y_fit), self.n_neighbors, self.input_kind))
        return original(self, X_fit, y_fit, sample_ids=sample_ids)
    monkeypatch.setattr(comparisons.StableFootruleKNN, 'fit', spy)
    model = kc.MultiViewInputKNN(**params).fit(Xtr, ytr)
    monkeypatch.setattr(comparisons.StableFootruleKNN, 'fit', original)
    reference = MultiViewFootruleKNN(**params).fit(Xtr, ytr)
    trained = MultiViewArrowFlowKNN(**params, widths=(8,), iterations=1).fit(Xtr, ytr)
    votes = []
    for v, ((enc, readout), (enc_f, _), (enc_t, _), selection) in enumerate(
            zip(model.views_, reference.views_, trained.views_, model.readout_selections_)):
        orders = enc.transform(Xtr)
        assert np.array_equal(orders, enc_f.transform(Xtr)) and np.array_equal(orders, enc_t.transform(Xtr))
        assert selection == select_knn_readout(inverse_positions(orders), ytr,
                                               seed=derive_seed(derive_seed(9, 'view', v), 'readout_selection'))
        assert len(selection['candidate_scores']) == 10
        assert (readout.n_neighbors, readout.weights, readout.input_kind) == (
            selection['config']['n_neighbors'], selection['config']['weights'], 'positions')
        votes.append(readout.predict(inverse_positions(enc.transform(Xte))))
    assert np.array_equal(model.predict(Xte), majority(votes)) and set(model.predict(Xte)) <= set(ytr)
    per_view = kc.KNN_SELECTION_FOLDS + 1
    assert len(fits) == 3 * per_view and all(kind == 'positions' for _, _, kind in fits)
    assert all(rows < len(ytr) and k == 21 for rows, k, _ in fits[v * per_view:(v + 1) * per_view - 1] for v in range(3))
    assert [fits[(v + 1) * per_view - 1][0] for v in range(3)] == [len(ytr)] * 3          # the refit: every training row
    assert model.readout_record()['readout'] == 'knn_input' and model.training_seconds_ == 0


# ----------------------------------------------------------------------------- wrappers, registry and harness

def test_adaptive_controls_resolve_from_the_training_partition_and_log_their_readout_choices():
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(1).permutation(len(y))
    for model_id, factory in ((kc.UNTRAINED_MODEL, kc.untrained_factory), (kc.INPUT_MODEL, kc.input_factory)):
        config = next(c for c in kc.control_candidates(model_id) if c['embed_scale'] == 1 and c['degree_offset'] == 0)
        spec = ModelSpec(model_id, factory, [config], True)
        predictions, record = _fit_predict(spec, config, 8129, X[idx[:60]], y[idx[:60]], X[idx[60:70]])
        assert predictions.shape == (10,) and set(predictions) <= set(y)
        assert record['preprocessing_settings']['strategy'] == 'target_aware'
        assert record['preprocessing_settings']['embed_dim'] == 16 and record['preprocessing_settings']['degree'] == 3
        assert record['classifier_fit_seconds'] == 0 and record['encoding_seconds'] > 0
        meta = record['representation_metadata']
        assert len(meta['views']) == 7 and meta['readout_seconds'] > 0
        assert all(len(v['candidate_scores']) == 10 and v['candidate_scores'][v['config_id']] == v['inner_score'] for v in meta['views'])
        canonical_json(record)
        estimator = factory(config, 8129).fit(X[idx[:60]], y[idx[:60]])
        resolved = bridge.resolve(config, 4, 60)
        assert estimator.resolved_ == {'embed_dim': resolved['embed_dim'], 'degree': resolved['degree']}
        assert estimator.encoder_ is estimator.model_.views_[0][0]
    with pytest.raises(ValueError, match='hold exactly'):
        kc.untrained_factory({**kc.control_candidates(kc.UNTRAINED_MODEL)[0], 'learning_rate': .1}, 1).fit(X[:30], y[:30])
    with pytest.raises(ValueError, match='hold exactly'):
        kc.input_factory({**kc.control_candidates(kc.INPUT_MODEL)[0], 'widths': [128]}, 1).fit(X[:30], y[:30])


def test_registry_holds_the_two_stochastic_controls_through_run_revision():
    from experiments.make_revision.run_revision import get_registry
    protocol = json.loads((PROTOCOLS/'knn_training.json').read_text())
    registry = get_registry(protocol['registry'], protocol)
    assert list(registry) == [kc.UNTRAINED_MODEL, kc.INPUT_MODEL]
    assert [len(s.candidates) for s in registry.values()] == [8, 4] and all(s.stochastic for s in registry.values())
    assert isinstance(registry[kc.UNTRAINED_MODEL].factory(registry[kc.UNTRAINED_MODEL].candidates[0], 1), kc.AdaptiveUntrainedKNN)
    assert isinstance(registry[kc.INPUT_MODEL].factory(registry[kc.INPUT_MODEL].candidates[0], 1), kc.AdaptiveInputKNN)
    assert set(kc.SOURCE_MODULES) >= set(bridge.SOURCE_MODULES) and 'experiments.make_revision.bridge' in kc.SOURCE_MODULES


def test_nested_fold_evaluation_selects_and_refits_both_controls():
    from experiments.make_revision.reporting import validate_result_records
    X, y = load_iris(return_X_y=True)
    split = make_splits(y, 3, 1, 2, 27183)[0]
    protocol = {'fit_seeds': [8129, 19391, 39019]}
    manifest = {'dataset_hash': 'fixture'}
    for model_id, factory in ((kc.UNTRAINED_MODEL, kc.untrained_factory), (kc.INPUT_MODEL, kc.input_factory)):
        candidates = [dict(c, n_views=3) for c in kc.control_candidates(model_id) if c['embed_scale'] == 1
                      and c.get('widths', [128]) == [128]]
        spec = ModelSpec(model_id, factory, candidates, True)
        result = evaluate_fold(X, y, split, spec, protocol['fit_seeds'], dataset_id='iris', dataset_hash='fixture',
                               code_revision='test')
        assert result['status'] == 'ok' and len(result['models']) == 3
        job = {'dataset_id': 'iris', 'model_id': model_id, 'outer_repeat': 0, 'outer_fold': 0}
        verified = validate_result_records(result, job, split, y, manifest, spec, protocol, 'test')
        assert len(verified) == 3 and all(0 <= row['accuracy'] <= 1 for row in verified)
        canonical_json(result)


# ----------------------------------------------------------------------------- protocol

def test_knn_training_protocol_pins_the_bridge_knn_design_and_tolerates_the_freeze():
    old = json.loads((PROTOCOLS/'bridge_knn.json').read_text())
    new = json.loads((PROTOCOLS/'knn_training.json').read_text())
    template_hash = hashlib.sha256((PROTOCOLS/'bridge_knn.json').read_bytes()).hexdigest()
    provenance = {'frozen', 'frozen_at_utc', 'source_template_sha256', 'resource_decision', 'status'}
    removed = {'augmentation', 'internal_validation_ratio', 'knn_readout', 'secondary_studies'}
    added = {'training_controls', 'design_source'}
    changed = {'protocol_id', 'production_family', 'registry', 'primary_contrasts', 'primary_family_size', 'multiplicity',
               'wallclock_cap_hours'}

    def pin(p):
        assert set(p) - provenance == (set(old) - provenance - removed) | added
        assert {k for k in set(old) - provenance - removed if old[k] != p[k]} == changed
        assert p['protocol_id'] == 'arrowflow-v3-knn-training-1' and p['production_family'] == 'knn_training'
        assert p['registry'] == 'experiments.make_revision.knn_controls:knn_training_registry'
        assert p['primary_contrasts'] == kc.PRIMARY_CONTRASTS and p['primary_family_size'] == 14 == 2 * len(p['datasets'])
        assert p['wallclock_cap_hours'] == 3 and p['datasets'] == old['datasets'] and p['fit_seeds'] == [8129, 19391, 39019]
        assert (p['split_seed'], p['outer_folds'], p['outer_repeats'], p['inner_folds']) == (27183, 5, 3, 3)
        assert set(p['design_source']['removed_from_template']) == removed
        block = p['training_controls']
        assert block['readout']['grid'] == old['knn_readout']['grid'] and block['readout']['selection'] == old['knn_readout']['selection']
        assert block['reference']['protocol_sha256'] == template_hash == p['source_template_sha256']
        assert block['reference']['protocol_id'] == old['protocol_id'] and block['reference']['model_id'] == kc.TRAINED_MODEL
        assert block['depth_split']['depths'] == [[128], [64, 128]] and 'outside the Holm family' in block['depth_split']['status']
        assert set(block['models'][kc.UNTRAINED_MODEL]['dropped']) == {'learning_rate', 'iterations', 'batch_size',
                                                                        'validation_ratio', 'augment'}
        assert set(block['models'][kc.INPUT_MODEL]['dropped']) == set(block['models'][kc.UNTRAINED_MODEL]['dropped']) | {'widths'}
        kc.validate_training_protocol(p)
        assert ('frozen_at_utc' in p) == bool(p['frozen'])
        if p['frozen']:
            assert p['frozen_at_utc'] >= '2026-09-13'
    pin(new)
    unfrozen = {k: v for k, v in new.items() if k != 'frozen_at_utc'}
    pin(dict(unfrozen, frozen=False, status='drafted_awaiting_training_only_pilot'))
    pin(dict(unfrozen, frozen=True, frozen_at_utc='2026-09-13T12:00:00+00:00', status='reviewed_and_piloted'))
    with pytest.raises(AssertionError):
        pin(dict(unfrozen, frozen=True))                                             # a freeze without its timestamp
    with pytest.raises(AssertionError):
        pin(dict(new, candidate_budget=16))
    with pytest.raises(AssertionError):
        pin(dict(new, fit_seeds=[1, 2, 3]))


@pytest.mark.skipif(not REAL_KNN.is_dir(), reason='the bridge_knn run is not on this machine')
def test_knn_training_protocol_reference_pins_the_real_bridge_knn_run():
    reference = json.loads((PROTOCOLS/'knn_training.json').read_text())['training_controls']['reference']
    assert reference == {**reference, **kc.reference_pins(REAL_KNN)}


@pytest.mark.parametrize('mutation, message', [
    (lambda p: p.update(primary_contrasts=p['primary_contrasts'][:1]), 'primary_contrasts'),
    (lambda p: p.update(primary_family_size=7), 'primary_family_size'),
    (lambda p: p['training_controls']['models'].pop(kc.INPUT_MODEL), 'models'),
    (lambda p: p['training_controls']['models'][kc.UNTRAINED_MODEL].update(candidates=16), kc.UNTRAINED_MODEL),
    (lambda p: p['training_controls']['models'][kc.INPUT_MODEL].update(candidate_keys=['widths']), kc.INPUT_MODEL),
    (lambda p: p['training_controls']['readout'].update(grid={'n_neighbors': [5], 'weights': ['uniform']}), 'readout'),
    (lambda p: p['training_controls']['reference'].update(summary_sha256=''), 'reference'),
    (lambda p: p['training_controls']['reference'].update(model_id='arrowflow_full'), 'reference'),
    (lambda p: p['training_controls']['depth_split'].update(depths=[[128], [128]]), 'depths'),
])
def test_validate_training_protocol_refuses_inconsistent_declarations(mutation, message):
    protocol = json.loads((PROTOCOLS/'knn_training.json').read_text())
    mutation(protocol)
    with pytest.raises(ValueError, match=message):
        kc.knn_training_registry(protocol)


# ----------------------------------------------------------------------------- depth split

def split_rows(differences, *, base=.8, noise=None):
    """One dataset's rows of models a and b over the given folds, three seeds each; a - b equals the fold's value."""
    rows = []
    for (repeat, fold), difference in differences.items():
        for i, seed in enumerate((1, 2, 3)):
            jitter = 0. if noise is None else noise[(repeat, fold)][i]
            rows.append({'dataset_id': 'd', 'model_id': 'a', 'outer_repeat': repeat, 'outer_fold': fold, 'model_seed': seed,
                         'accuracy': base + difference + jitter, 'status': 'ok'})
            rows.append({'dataset_id': 'd', 'model_id': 'b', 'outer_repeat': repeat, 'outer_fold': fold, 'model_seed': seed,
                         'accuracy': base, 'status': 'ok'})
    return rows


def test_depth_split_groups_folds_by_the_selected_widths_and_gives_descriptive_intervals_only():
    folds = [(0, f) for f in range(5)] + [(1, 0)]
    differences = dict(zip(folds, (.02, .04, .06, -.01, .03, .05)))
    widths = {(0, 0): [128], (0, 1): [64, 128], (0, 2): [128], (0, 3): [64, 128], (0, 4): [128], (1, 0): [64, 128]}
    rows = split_rows(differences)
    seeds = {'a': [1, 2, 3], 'b': [1, 2, 3]}
    entries = kc.depth_split(rows, 'a', 'b', widths, depths=[[128], [64, 128]], folds=folds, seeds=seeds, q=.25, confidence=.95)
    assert [e['widths'] for e in entries] == [[128], [64, 128]]
    shallow, deep = entries
    assert shallow['folds'] == [[0, 0], [0, 2], [0, 4]] and deep['folds'] == [[0, 1], [0, 3], [1, 0]]
    assert np.allclose(shallow['fold_differences'], [.02, .06, .03]) and np.allclose(deep['fold_differences'], [.04, -.01, .05])
    assert np.isclose(shallow['mean_difference'], np.mean([.02, .06, .03])) and np.isclose(deep['sd'], np.std([.04, -.01, .05], ddof=1))
    subset = [r for r in rows if [r['outer_repeat'], r['outer_fold']] in deep['folds']]
    expected = paired_corrected_interval(subset, 'a', 'b', q=.25, expected_folds=[tuple(f) for f in deep['folds']], expected_seeds=seeds)
    assert 'p_approximate' not in deep['interval'] and deep['interval']['df'] == 2
    assert all(np.isclose(deep['interval'][k], expected[k]) for k in ('mean_difference', 'ci_low', 'ci_high', 'standard_error'))
    one = kc.depth_split(rows, 'a', 'b', {**widths, (0, 1): [128], (0, 3): [128]}, depths=[[128], [64, 128]], folds=folds,
                         seeds=seeds, q=.25, confidence=.95)[1]
    assert one['n_folds'] == 1 and one['interval'] is None and one['sd'] is None
    none = kc.depth_split(rows, 'a', 'b', {fold: [128] for fold in folds}, depths=[[128], [64, 128]], folds=folds, seeds=seeds,
                          q=.25, confidence=.95)[1]
    assert none['n_folds'] == 0 and none['mean_difference'] is None and none['interval'] is None
    with pytest.raises(ValueError, match='outside the declared depths'):
        kc.depth_split(rows, 'a', 'b', {**widths, (0, 0): [32]}, depths=[[128], [64, 128]], folds=folds, seeds=seeds, q=.25, confidence=.95)
    pooled = kc.pooled_depth_split({'d': entries, 'e': [dict(shallow, fold_differences=[.1]), dict(deep, fold_differences=[], n_folds=0)]},
                                   [[128], [64, 128]])
    assert pooled[0]['n_dataset_folds'] == 4 and np.isclose(pooled[0]['mean_difference'], np.mean([.02, .06, .03, .1]))
    assert pooled[1]['n_dataset_folds'] == 3 and pooled[1]['datasets'] == ['d']
    assert kc.selected_widths([{'model_id': 'a', 'outer_repeat': 0, 'outer_fold': 0, 'config': {'widths': [128]}}], 'a') == {(0, 0): [128]}
    with pytest.raises(ValueError, match='disagree'):
        kc.selected_widths([{'model_id': 'a', 'outer_repeat': 0, 'outer_fold': 0, 'config': {'widths': w}} for w in ([128], [64, 128])], 'a')


# ----------------------------------------------------------------------------- smoke

def test_smoke_runs_the_family_through_the_harness_and_compare_runs_training(tmp_path):
    protocol = json.loads((PROTOCOLS/'knn_training.json').read_text())
    record = kc.smoke(tmp_path, protocol, workers=2)
    assert record['purpose'] == 'synthetic_smoke_only_not_paper_evidence' and len(record['contrasts']) == 2
    assert [r['model_b'] for r in record['contrasts']] == [kc.UNTRAINED_MODEL, kc.INPUT_MODEL]
    assert all(r['n_folds'] == 3 and r['holm_p_approximate'] is not None for r in record['contrasts'])
    training = json.loads((tmp_path/'training'/'candidates.json').read_text())
    assert [len(training[m]['candidates']) for m in (kc.UNTRAINED_MODEL, kc.INPUT_MODEL)] == [8, 4]
    assert sorted(p.name for p in (tmp_path/'compare').iterdir()) == sorted(
        ['training_contrasts.csv', 'training_contrasts.json', 'training_error_table.json', 'training_depth_split.json'])
    depth = json.loads((tmp_path/'compare'/'training_depth_split.json').read_text())
    assert sum(e['n_folds'] for e in depth['by_dataset']['synthetic']) == 3
