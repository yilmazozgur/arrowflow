"""Task 23A: projected_numeric_knn (projected_knn) and the knn_projected protocol."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from experiments.make_revision import bridge, models, multiview
from experiments.make_revision import knn_controls as kc
from experiments.make_revision import projected_knn as pk
from experiments.make_revision.comparisons import CONVENTIONAL_GRIDS, derive_seed
from experiments.make_revision.evaluation import ModelSpec, _fit_predict, canonical_json, config_id, evaluate_fold, make_splits
from experiments.make_revision.models import OrdinalEncoder, seed_fit
from experiments.make_revision.multiview import KNN_READOUT_GRID, select_knn_readout
from experiments.make_revision.secondary_studies import majority
from arrowflow.ranking import inverse_positions, score_order

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'/'2026-09-12'
RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'
REAL_KNN, REAL_TRAINING = RUNS/'2026-09-12-bridge-knn', RUNS/'2026-09-13-knn-training'


def small(n=90, seed=0):
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(seed).permutation(len(y))[:n]
    return X[idx], y[idx]


# ----------------------------------------------------------------------------- the pre-sort array

@pytest.mark.parametrize('degree', [1, 2])
@pytest.mark.parametrize('strategy', ['random', 'target_aware', 'calibrated'])
def test_projected_scores_are_the_array_the_encoder_sorts_without_further_standardization(monkeypatch, strategy, degree):
    X, y = small()
    X = X.copy()
    X[3, 1] = np.nan                                                     # imputation belongs to the encoder
    enc = OrdinalEncoder(strategy, 16, degree, .3, 11).fit(X[:60], y[:60])
    sorted_arrays, original = [], models.score_order

    def spy(scores, items=None):
        sorted_arrays.append(np.array(scores, copy=True))
        return original(scores, items)
    monkeypatch.setattr(models, 'score_order', spy)
    orders = enc.transform(X)
    monkeypatch.setattr(models, 'score_order', original)
    scores = pk.projected_scores(enc, X)
    assert len(sorted_arrays) == 1 and scores.dtype == np.float64
    assert np.array_equal(scores, sorted_arrays[0]) and np.array_equal(score_order(scores), orders)
    restandardized = StandardScaler().fit(scores[:60]).transform(scores[:60])
    if strategy == 'calibrated':                                         # the encoder's own calibration scaler
        assert np.allclose(scores[:60].mean(axis=0), 0) and np.allclose(scores[:60].std(axis=0), 1)
    else:
        assert not np.allclose(restandardized, scores[:60])


def test_scores_that_do_not_sort_into_the_encoded_ranking_are_refused(monkeypatch):
    X, y = small()
    enc = OrdinalEncoder('random', 8, 1, .3, 3).fit(X, y)
    original = OrdinalEncoder.transform
    monkeypatch.setattr(OrdinalEncoder, 'transform', lambda self, rows: original(self, rows)[:, ::-1])
    with pytest.raises(pk.PresortMismatch):
        pk.projected_scores(enc, X)
    with pytest.raises(pk.PresortMismatch):
        pk.MultiViewProjectedKNN(n_views=1, embed_dim=8, degree=1).fit(X, y)


def test_every_view_classifier_is_fitted_on_the_array_its_encoder_sorts_and_tuned_on_training_rows_only(monkeypatch):
    X, y = small()
    Xtr, ytr = X[:60], y[:60]
    sorted_arrays, fits, original_order, original_fit = [], [], models.score_order, pk.StableNumericKNN.fit

    def order_spy(scores, items=None):
        sorted_arrays.append(np.array(scores, copy=True))
        return original_order(scores, items)

    def fit_spy(self, rows, labels, sample_ids=None):
        fits.append((np.array(rows, copy=True), self.n_neighbors, self.p, None if sample_ids is None else np.array(sample_ids)))
        return original_fit(self, rows, labels, sample_ids=sample_ids)
    monkeypatch.setattr(models, 'score_order', order_spy)
    monkeypatch.setattr(pk.StableNumericKNN, 'fit', fit_spy)
    model = pk.MultiViewProjectedKNN(n_views=3, embed_dim=16, degree=2, seed=4).fit(Xtr, ytr)
    monkeypatch.undo()
    assert len(sorted_arrays) == 3 and len(fits) == 3 * (3 * 2 + 1)    # per view: 3 splits x 2 orders p, then the refit
    for v in range(3):
        view_fits, presort = fits[v * 7:(v + 1) * 7], sorted_arrays[v]
        for rows, k, p, ids in view_fits[:-1]:                           # the selection caches: training rows only
            assert k == 21 and len(rows) < len(ytr) and np.array_equal(rows, presort[ids])
        assert sorted(p for _, _, p, _ in view_fits[:-1]) == [1, 1, 1, 2, 2, 2]
        refit, _, _, ids = view_fits[-1]
        assert ids is None and np.array_equal(refit, presort)            # the refit: exactly the array the encoder sorted
        assert np.array_equal(model.views_[v][1].positions_, presort)


# ----------------------------------------------------------------------------- encoders, readout and vote

def test_projected_knn_keeps_the_input_controls_encoders_and_votes_by_majority_over_the_views(monkeypatch):
    X, y = small()
    Xtr, ytr, Xte = X[:60], y[:60], X[60:]
    params = dict(n_views=3, strategy='diverse', embed_dim=16, degree=2, seed=9)
    seed_fit(1)
    model = pk.MultiViewProjectedKNN(**params).fit(Xtr, ytr)
    seed_fit(2)                                                          # another global RNG state changes nothing
    inputs = kc.MultiViewInputKNN(**params).fit(Xtr, ytr)
    votes = []
    for v, ((enc, readout), (enc_i, _), selection) in enumerate(zip(model.views_, inputs.views_, model.readout_selections_)):
        seed_v = derive_seed(9, 'view', v)
        assert enc.seed == seed_v and enc.strategy == multiview.CYCLE[v]
        scores = pk.projected_scores(enc, Xtr)
        assert np.array_equal(score_order(scores), enc_i.transform(Xtr))       # argsorting the input gives the ranking
        assert np.array_equal(score_order(pk.projected_scores(enc, Xte)), enc_i.transform(Xte))
        assert selection == pk.select_numeric_readout(scores, ytr, seed=derive_seed(seed_v, 'readout_selection'))
        assert len(selection['candidate_scores']) == 20
        assert (readout.n_neighbors, readout.weights, readout.p) == tuple(selection['config'][k] for k in ('n_neighbors', 'weights', 'p'))
        votes.append(readout.predict(pk.projected_scores(enc, Xte)))
    assert np.array_equal(model.predict(Xte), majority(votes)) and set(model.predict(Xte)) <= set(ytr)
    scripted = [np.array([0, 1, 1, 2, 2]), np.array([1, 1, 0, 2, 1]), np.array([1, 0, 0, 0, 0])]   # the views disagree
    monkeypatch.setattr(pk.MultiViewProjectedKNN, 'predict_views', lambda self, rows: scripted)
    assert model.predict(Xte[:5]).tolist() == majority(scripted).tolist() == [1, 1, 0, 2, 0]   # a three-way tie: lowest
    monkeypatch.undo()
    record = model.readout_record()
    assert record['readout'] == 'knn_projected_scores' and record['representation'] == pk.PRESORT_REPRESENTATION
    assert record['grid'] == pk.NUMERIC_READOUT_GRID and len(record['views']) == 3 and model.training_seconds_ == 0
    canonical_json(record)
    with pytest.raises(ValueError, match='majority'):
        pk.MultiViewProjectedKNN(n_views=1, embed_dim=8, degree=1, aggregation='borda').fit(Xtr, ytr)


def test_numeric_readout_selection_equals_refitting_every_candidate_on_the_footrule_selections_splits(monkeypatch):
    X, y = small(120, 3)
    enc = OrdinalEncoder('target_aware', 16, 2, .3, 4).fit(X, y)
    scores = pk.projected_scores(enc, X)
    seed = derive_seed(derive_seed(7, 'view', 0), 'readout_selection')
    recorded = []

    class Recording(StratifiedKFold):
        def split(self, X, y=None, groups=None):
            splits = [(a.copy(), b.copy()) for a, b in super().split(X, y, groups)]
            recorded.append(((self.n_splits, self.shuffle, self.random_state), splits))
            return iter(splits)
    monkeypatch.setattr(pk, 'StratifiedKFold', Recording)
    monkeypatch.setattr(multiview, 'StratifiedKFold', Recording)
    selection = pk.select_numeric_readout(scores, y, seed=seed)
    select_knn_readout(inverse_positions(score_order(scores)), y, seed=seed)
    monkeypatch.undo()
    (ours, splits), (footrule, footrule_splits) = recorded
    assert ours == footrule == (3, True, seed)
    assert all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(splits, footrule_splits))
    candidates = pk.numeric_readout_candidates()
    assert len(candidates) == 20 and [config_id(c) for c in candidates] == sorted(config_id(c) for c in candidates)
    expected = {config_id(c): float(np.mean([float(np.mean(pk.StableNumericKNN(**c).fit(scores[a], y[a]).predict(scores[b]) == y[b]))
                                             for a, b in splits])) for c in candidates}
    assert selection['candidate_scores'] == expected and selection['folds'] == 3
    best = max(expected.values())
    assert selection['config_id'] == min(cid for cid, value in expected.items() if value == best) and selection['inner_score'] == best
    rows = np.random.RandomState(5).randn(12, 4)
    labels = np.array([0] * 10 + [1] * 2)
    assert pk.select_numeric_readout(rows, labels, seed=1)['folds'] == 2
    with pytest.raises(ValueError, match='two rows per class'):
        pk.select_numeric_readout(rows[:11], labels[:11], seed=1)


def test_stable_numeric_knn_is_minkowski_knn_with_the_footrule_readouts_tie_and_vote_rules():
    rng = np.random.RandomState(0)
    A, b, Q = rng.randn(150, 5), rng.randint(0, 3, 150), rng.randn(40, 5)
    for c in pk.numeric_readout_candidates():                            # no exact ties: sklearn's brute-force kNN
        expected = KNeighborsClassifier(algorithm='brute', **c).fit(A, b).predict(Q)
        assert np.array_equal(pk.StableNumericKNN(**c).fit(A, b).predict(Q), expected), c
    for p, distance in ((1, np.abs(Q[0] - A).sum(axis=1)), (2, np.sqrt(((Q[0] - A) ** 2).sum(axis=1)))):
        assert np.allclose(pk.StableNumericKNN(n_neighbors=4, p=p).fit(A, b).kneighbors(Q[:1])[0][0], np.sort(distance)[:4])
    ring, origin = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]), np.zeros((1, 2))
    labels = np.array([1, 1, 0, 0])
    assert pk.StableNumericKNN(n_neighbors=2, p=1).fit(ring, labels).predict(origin)[0] == 1          # cutoff ties: earlier rows
    assert pk.StableNumericKNN(n_neighbors=2, p=1).fit(ring, labels, sample_ids=np.array([4, 3, 2, 1])).predict(origin)[0] == 0
    centre = np.vstack([origin, ring])
    assert pk.StableNumericKNN(n_neighbors=5, weights='uniform').fit(centre, [0, 1, 1, 1, 1]).predict(origin)[0] == 1
    assert pk.StableNumericKNN(n_neighbors=5, weights='distance').fit(centre, [0, 1, 1, 1, 1]).predict(origin)[0] == 0
    assert pk.StableNumericKNN(n_neighbors=2).fit(ring[:1].repeat(2, axis=0) * [[1], [-1]], [1, 0]).predict(origin)[0] == 0
    for bad in ({'p': 3}, {'p': True}, {'weights': 'rank'}):
        with pytest.raises(ValueError):
            pk.StableNumericKNN(**bad).fit(A, b)
    with pytest.raises(ValueError, match='finite'):
        pk.StableNumericKNN().fit(np.array([[np.nan, 1.]]), [0])


def test_numeric_readout_grid_is_the_numeric_knn_comparator_grid_and_the_footrule_readout_grid_with_p():
    assert pk.NUMERIC_READOUT_GRID == CONVENTIONAL_GRIDS['numeric_knn']
    assert {k: v for k, v in pk.NUMERIC_READOUT_GRID.items() if k != 'p'} == KNN_READOUT_GRID and pk.NUMERIC_READOUT_GRID['p'] == [1, 2]


# ----------------------------------------------------------------------------- wrapper, registry and harness

def test_adaptive_projected_knn_resolves_from_the_training_partition_and_logs_the_presort_array_and_readout_choices():
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(1).permutation(len(y))
    assert pk.projected_candidates() == kc.control_candidates(kc.INPUT_MODEL) and len(pk.projected_candidates()) == 4
    config = next(c for c in pk.projected_candidates() if c['embed_scale'] == 1 and c['degree_offset'] == 0)
    spec = ModelSpec(pk.PROJECTED_MODEL, pk.projected_factory, [config], True)
    predictions, record = _fit_predict(spec, config, 8129, X[idx[:60]], y[idx[:60]], X[idx[60:70]])
    assert predictions.shape == (10,) and set(predictions) <= set(y)
    settings = record['preprocessing_settings']
    assert (settings['strategy'], settings['embed_dim'], settings['degree']) == ('target_aware', 16, 3)
    assert record['classifier_fit_seconds'] == 0 and record['encoding_seconds'] > 0
    meta = record['representation_metadata']
    assert meta['representation'] == pk.PRESORT_REPRESENTATION and meta['presort_check'] == pk.PRESORT_CHECK
    assert len(meta['views']) == 7 and meta['score_dimension'] == 16
    assert all(len(v['candidate_scores']) == 20 and v['candidate_scores'][v['config_id']] == v['inner_score'] for v in meta['views'])
    canonical_json(record)
    estimator = pk.projected_factory(config, 8129).fit(X[idx[:60]], y[idx[:60]])
    control = kc.input_factory(config, 8129).fit(X[idx[:60]], y[idx[:60]])
    resolved = bridge.resolve(config, 4, 60)
    assert estimator.resolved_ == control.resolved_ == {'embed_dim': resolved['embed_dim'], 'degree': resolved['degree']}
    assert all(np.array_equal(score_order(pk.projected_scores(enc, X)), enc_i.transform(X))
               for (enc, _), (enc_i, _) in zip(estimator.model_.views_, control.model_.views_))
    with pytest.raises(ValueError, match='hold exactly'):
        pk.projected_factory({**config, 'widths': [128]}, 1).fit(X[:30], y[:30])


def test_registry_holds_one_stochastic_model_and_its_run_seals_the_shared_sources():
    from experiments.make_revision.compare_runs import SHARED_SOURCES
    from experiments.make_revision.run_revision import environment_record, get_registry
    protocol = json.loads((PROTOCOLS/'knn_projected.json').read_text())
    registry = get_registry(protocol['registry'], protocol)
    spec = registry[pk.PROJECTED_MODEL]
    assert list(registry) == [pk.PROJECTED_MODEL] and spec.stochastic and spec.candidates == kc.control_candidates(kc.INPUT_MODEL)
    assert isinstance(spec.factory(spec.candidates[0], 1), pk.AdaptiveProjectedKNN)
    sealed = environment_record(protocol['registry'])['source_hashes']
    assert set(sealed) == set(SHARED_SOURCES) | {'experiments/make_revision/knn_controls.py', 'experiments/make_revision/projected_knn.py'}


def test_nested_fold_evaluation_selects_and_refits_the_projected_model_without_test_rows():
    from experiments.make_revision.reporting import validate_result_records
    X, y = load_iris(return_X_y=True)
    split = make_splits(y, 3, 1, 2, 27183)[0]
    seeds = [8129, 19391, 39019]
    spec = ModelSpec(pk.PROJECTED_MODEL, pk.projected_factory,
                     [dict(c, n_views=3) for c in pk.projected_candidates() if c['embed_scale'] == 1], True)
    result = evaluate_fold(X, y, split, spec, seeds, dataset_id='iris', dataset_hash='fixture', code_revision='test')
    assert result['status'] == 'ok' and len(result['models']) == 3
    job = {'dataset_id': 'iris', 'model_id': pk.PROJECTED_MODEL, 'outer_repeat': 0, 'outer_fold': 0}
    verified = validate_result_records(result, job, split, y, {'dataset_hash': 'fixture'}, spec, {'fit_seeds': seeds}, 'test')
    assert len(verified) == 3 and all(0 <= row['accuracy'] <= 1 for row in verified)
    assert all(set(row['fit_rows']).isdisjoint(split['test']) for row in result['selection']['fits'] + result['models'])
    canonical_json(result)


# ----------------------------------------------------------------------------- protocol

def test_knn_projected_protocol_copies_the_knn_training_design_and_tolerates_the_freeze():
    from experiments.make_revision.compare_projected import DESIGN_COPY_KEYS
    old = json.loads((PROTOCOLS/'knn_training.json').read_text())
    new = json.loads((PROTOCOLS/'knn_projected.json').read_text())
    template_hash = hashlib.sha256((PROTOCOLS/'knn_training.json').read_bytes()).hexdigest()
    provenance = {'frozen', 'frozen_at_utc', 'source_template_sha256', 'resource_decision', 'status'}
    removed, added = {'training_controls'}, {'projected_control'}
    changed = {'protocol_id', 'production_family', 'registry', 'primary_contrasts', 'wallclock_cap_hours', 'design_source'}

    def pin(p):
        assert set(p) - provenance == (set(old) - provenance - removed) | added
        assert {k for k in set(old) - provenance - removed if old[k] != p[k]} == changed
        assert all(canonical_json(p[k]) == canonical_json(old[k]) for k in DESIGN_COPY_KEYS)
        assert p['design_source']['identical_to_template'] == ', '.join(DESIGN_COPY_KEYS)
        assert set(p['design_source']['removed_from_template']) == removed
        assert p['protocol_id'] == 'arrowflow-v3-knn-projected-1' and p['production_family'] == 'knn_projected'
        assert p['registry'] == 'experiments.make_revision.projected_knn:knn_projected_registry'
        assert p['primary_contrasts'] == ['input_footrule_knn_vs_projected_numeric_knn', 'arrowflow_full_knn_vs_projected_numeric_knn']
        assert p['primary_family_size'] == 14 == 2 * len(p['datasets']) and p['wallclock_cap_hours'] == 1
        assert p['source_template_sha256'] == template_hash
        assert (p['split_seed'], p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['fit_seeds']) == (27183, 5, 3, 3, [8129, 19391, 39019])
        block = p['projected_control']
        assert block['readout']['grid'] == CONVENTIONAL_GRIDS['numeric_knn'] and block['model']['representation'] == pk.PRESORT_REPRESENTATION
        assert block['model']['dropped'] == old['training_controls']['models'][kc.INPUT_MODEL]['dropped']
        assert block['references']['knn']['protocol_sha256'] == old['training_controls']['reference']['protocol_sha256']
        assert block['references']['training']['protocol_sha256'] == template_hash
        assert 'outside the Holm family' in block['descriptive']['status'] and block['ladder'] == pk.ladder_declaration()
        pk.validate_projected_protocol(p)
        assert ('frozen_at_utc' in p) == bool(p['frozen'])
        if p['frozen']:
            assert p['frozen_at_utc'] >= '2026-09-13'
    pin(new)
    unfrozen = {k: v for k, v in new.items() if k != 'frozen_at_utc'}
    pin(dict(unfrozen, frozen=False, status='drafted_awaiting_training_only_pilot'))
    pin(dict(unfrozen, frozen=True, frozen_at_utc='2026-09-13T20:00:00+00:00', status='reviewed_and_piloted'))
    for broken in (dict(unfrozen, frozen=True), dict(new, split_seed=1), dict(new, failure_policy='stop_early'),
                   dict(new, fit_seeds=[1, 2, 3])):
        with pytest.raises(AssertionError):
            pin(broken)


@pytest.mark.skipif(not (REAL_KNN.is_dir() and REAL_TRAINING.is_dir()), reason='the knn and knn_training runs are not on this machine')
def test_knn_projected_protocol_references_pin_the_real_knn_and_training_runs():
    references = json.loads((PROTOCOLS/'knn_projected.json').read_text())['projected_control']['references']
    for label, directory in (('knn', REAL_KNN), ('training', REAL_TRAINING)):
        assert references[label] == {**references[label], **pk.run_pins(directory, label)}


@pytest.mark.parametrize('mutation, message', [
    (lambda p: p.pop('projected_control'), 'projected_control block'),
    (lambda p: p.update(primary_contrasts=p['primary_contrasts'][::-1]), 'primary_contrasts'),
    (lambda p: p.update(primary_family_size=7), 'primary_family_size'),
    (lambda p: p['projected_control']['model'].update(candidates=8), 'projected_control.model'),
    (lambda p: p['projected_control']['model'].update(candidate_keys=['widths']), 'projected_control.model'),
    (lambda p: p['projected_control']['readout'].update(grid=KNN_READOUT_GRID), 'readout'),
    (lambda p: p['projected_control']['descriptive'].update(status='a member of the family'), 'descriptive'),
    (lambda p: p['projected_control'].update(ladder=p['projected_control']['ladder'][::-1]), 'ladder'),
    (lambda p: p['projected_control']['references']['training'].update(summary_sha256=''), 'references.training'),
    (lambda p: p['projected_control']['references']['knn'].update(model_ids=['arrowflow_full_knn']), 'references.knn'),
])
def test_validate_projected_protocol_refuses_inconsistent_declarations(mutation, message):
    protocol = json.loads((PROTOCOLS/'knn_projected.json').read_text())
    mutation(protocol)
    with pytest.raises(ValueError, match=message):
        pk.knn_projected_registry(protocol)


# ----------------------------------------------------------------------------- smoke

def test_smoke_runs_the_three_families_through_the_harness_and_compare_runs_projected(tmp_path):
    from experiments.make_revision.compare_projected import PROJECTED_OUTPUTS
    protocol = json.loads((PROTOCOLS/'knn_projected.json').read_text())
    record = pk.smoke(tmp_path, protocol, workers=2)
    assert record['purpose'] == 'synthetic_smoke_only_not_paper_evidence'
    assert [(r['model_a'], r['model_b']) for r in record['contrasts']] == [(kc.INPUT_MODEL, pk.PROJECTED_MODEL),
                                                                           (kc.TRAINED_MODEL, pk.PROJECTED_MODEL)]
    assert all(r['n_folds'] == 3 and r['df'] == 2 and r['holm_p_approximate'] is not None for r in record['contrasts'])
    assert [r['model_a'] for r in record['descriptive']] == [pk.RAW_MODEL]
    assert not {'p_approximate', 'holm_p_approximate'} & set(record['descriptive'][0])
    assert sorted(p.name for p in (tmp_path/'compare').iterdir()) == sorted(PROJECTED_OUTPUTS)
    assert [row['model_id'] for row in record['ladder'][kc.SMOKE_DATASET]] == [model for _, model, _ in pk.LADDER]
    candidates = json.loads((tmp_path/'projected'/'candidates.json').read_text())
    assert candidates[pk.PROJECTED_MODEL]['candidates'] == pk.projected_candidates()
