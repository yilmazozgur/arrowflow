# tests/make_revision/test_multiview.py
import numpy as np, pytest
from sklearn.datasets import load_iris
from experiments.make_revision.models import ArrowFlowEstimator
from experiments.make_revision.multiview import (MultiViewArrowFlow, MultiViewFootruleKNN,
                                                 borda_aggregate, view_strategy)

def small():
    X, y = load_iris(return_X_y=True)
    rng = np.random.RandomState(0); idx = rng.permutation(len(y))[:90]
    return X[idx], y[idx]

def test_class_ranking_is_a_permutation_of_classes_and_agrees_with_predict():
    X, y = small()
    est = ArrowFlowEstimator(embed_dim=16, degree=1, widths=(16,), iterations=5, learning_rate=.1, seed=1).fit(X, y)
    orders = est.encoder_.transform(X)
    ranking = est.predict_class_ranking(orders)
    assert ranking.shape == (len(X), 3)
    assert all(sorted(row) == [0, 1, 2] for row in ranking.tolist())
    assert np.array_equal(ranking[:, 0], est.predict_orders(orders))

def test_validation_checkpoint_and_augmentation_run_and_change_training_sample_count():
    X, y = small()
    plain = ArrowFlowEstimator(embed_dim=16, degree=1, widths=(16,), iterations=5, seed=1).fit(X, y)
    aug = ArrowFlowEstimator(embed_dim=16, degree=1, widths=(16,), iterations=5, seed=1,
                             validation_ratio=.2, augment=True, n_augmentations=1, max_swaps=2).fit(X, y)
    assert plain.training_sample_count_ == 90
    assert aug.training_sample_count_ == 2 * 72          # 20% held out, then doubled by augmentation
    assert aug.validation_sample_count_ == 18

def test_view_strategy_cycle():
    assert [view_strategy('diverse', v) for v in range(4)] == ['target_aware', 'random', 'calibrated', 'target_aware']
    assert view_strategy('random', 5) == 'random'

def test_borda_aggregate_prefers_consistently_second_class_over_split_first():
    classes = np.array([0, 1, 2])
    rankings = [np.array([[0, 1, 2]]), np.array([[2, 1, 0]]), np.array([[1, 0, 2]])]
    assert borda_aggregate(rankings, classes).tolist() == [1]      # scores 0:3, 1:4, 2:2

def test_multiview_fit_predict_shapes_and_view_seeds_differ():
    X, y = small()
    m = MultiViewArrowFlow(n_views=3, embed_dim=16, degree=1, widths=(16,), iterations=5, seed=7).fit(X, y)
    assert len(m.views_) == 3
    seeds = {enc.seed for enc, net in m.views_}
    assert len(seeds) == 3
    assert m.predict(X).shape == (len(X),)
    assert len(m.predict_views(X)[0]) == 3
    m.set_params(aggregation='borda')
    assert m.predict(X).shape == (len(X),)

def test_multiview_knn_uses_identical_encoders():
    X, y = small()
    af = MultiViewArrowFlow(n_views=3, embed_dim=16, degree=1, widths=(16,), iterations=2, seed=7).fit(X, y)
    knn = MultiViewFootruleKNN(n_views=3, strategy='diverse', embed_dim=16, degree=1, n_neighbors=3, weights='uniform', seed=7).fit(X, y)
    for (enc_a, _), (enc_k, _) in zip(af.views_, knn.views_):
        assert np.array_equal(enc_a.transform(X), enc_k.transform(X))
    assert knn.predict(X).shape == (len(X),)

def test_core_validation_split_matches_reported_counts_after_augmentation(monkeypatch):
    # The core splits validation as int(val_data_ratio * len(list)) on the list it receives; after
    # augmentation the estimator must still hand it exactly its n_val held-out rows at the head.
    from arrowflow.arrowflow import DataGraph, SortFlowHybridNetwork
    X, y = small()
    splits, augmented = [], []
    original_split = SortFlowHybridNetwork.split_into_train_validation_dataset
    original_augment = DataGraph.augment_permutation_data
    def spy_split(self, data_train, ratio_validation=0.2):
        train, validation = original_split(self, data_train, ratio_validation)
        splits.append((len(data_train), validation, train))
        return train, validation
    def spy_augment(data_train, **kwargs):
        out = original_augment(data_train, **kwargs)
        augmented.append(out)
        return out
    monkeypatch.setattr(SortFlowHybridNetwork, 'split_into_train_validation_dataset', spy_split)
    monkeypatch.setattr(DataGraph, 'augment_permutation_data', staticmethod(spy_augment))
    est = ArrowFlowEstimator(embed_dim=16, degree=1, widths=(16,), iterations=5, seed=1,
                             validation_ratio=.2, augment=True, n_augmentations=1, max_swaps=2).fit(X, y)
    assert len(splits) == 1 and len(augmented) == 1
    total, validation, train = splits[0]
    assert (total, len(validation), len(train)) == (162, 18, 144)
    assert (len(validation), len(train)) == (est.validation_sample_count_, est.training_sample_count_)
    augmented_ids = {id(row) for row in augmented[0]}
    assert not any(id(row) in augmented_ids for row in validation)   # no held-out row entered augmentation
    assert all(id(row) in augmented_ids for row in train)             # training part is exactly its output

def test_multiview_knn_forwards_sample_ids_to_every_view(monkeypatch):
    # The ablation runner passes the global training-row IDs of the saved split so that
    # k-nearest-neighbour cutoff ties are broken by source row, not by partition position.
    from experiments.make_revision import comparisons
    X, y = small()
    ids = np.random.RandomState(3).permutation(1000)[:len(y)]        # unique, unsorted global row IDs
    received = []
    original = comparisons.StableFootruleKNN.fit
    def spy(self, X, y, sample_ids=None):
        received.append(sample_ids)
        return original(self, X, y, sample_ids=sample_ids)
    monkeypatch.setattr(comparisons.StableFootruleKNN, 'fit', spy)
    knn = MultiViewFootruleKNN(n_views=3, embed_dim=16, degree=1, n_neighbors=3, seed=7).fit(X, y, sample_ids=ids)
    assert len(received) == 3 and all(r is not None and np.array_equal(r, ids) for r in received)
    for _, view in knn.views_:
        assert np.array_equal(view.source_order_, np.argsort(ids, kind='stable'))
    assert knn.predict(X).shape == (len(X),)
    received.clear()
    MultiViewFootruleKNN(n_views=2, embed_dim=16, degree=1, n_neighbors=3, seed=7).fit(X, y)
    assert received == [None, None]                                   # omitted IDs keep the partition order


# ----------------------------------------------------------------------------- kNN readout on the hidden ranking (Task 10)

def split_small():
    X, y = small()
    return X[:60], y[:60], X[60:], y[60:]

def test_knn_readout_grid_is_the_laboratory_grid_in_canonical_order():
    from experiments.make_revision.evaluation import config_id
    from experiments.make_revision.multiview import KNN_READOUT_GRID, knn_readout_candidates
    cands = knn_readout_candidates()
    assert KNN_READOUT_GRID == {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance']}
    assert len(cands) == 10 and [config_id(c) for c in cands] == sorted(config_id(c) for c in cands)
    assert {(c['n_neighbors'], c['weights']) for c in cands} == {(k, w) for k in (1, 3, 5, 11, 21) for w in ('uniform', 'distance')}

def test_select_knn_readout_scores_every_candidate_on_training_splits_only_and_breaks_ties_canonically():
    from sklearn.model_selection import StratifiedKFold
    from experiments.make_revision.comparisons import StableFootruleKNN
    from experiments.make_revision.evaluation import config_id
    from experiments.make_revision.multiview import knn_readout_candidates, select_knn_readout
    rng = np.random.RandomState(0)
    y = np.repeat([0, 1, 2], 20)
    hidden = np.stack([rng.permutation(12) for _ in y])            # positions of a 12-filter hidden ranking
    chosen = select_knn_readout(hidden, y, seed=11)
    cands = knn_readout_candidates()
    # the same rule recomputed candidate by candidate with plain fits on the same stratified splits
    means = {}
    for c in cands:
        scores = []
        for a, b in StratifiedKFold(3, shuffle=True, random_state=11).split(np.zeros(len(y)), y):
            knn = StableFootruleKNN(**c, input_kind='positions').fit(hidden[a], y[a], sample_ids=a)
            scores.append(float(np.mean(knn.predict(hidden[b]) == y[b])))
        means[config_id(c)] = float(np.mean(scores))
    assert chosen['folds'] == 3 and chosen['candidate_scores'] == means
    best = min((-s, cid) for cid, s in means.items())
    assert (chosen['config_id'], chosen['inner_score']) == (best[1], -best[0])
    assert chosen['config'] in cands and config_id(chosen['config']) == chosen['config_id']
    # exact ties: constant predictions make every candidate score the same; the lowest config_id wins
    tied = select_knn_readout(np.tile(np.arange(12), (len(y), 1)), y, seed=11)
    assert len(set(tied['candidate_scores'].values())) == 1 and tied['config_id'] == min(tied['candidate_scores'])
    # fewer rows in a class than folds: the split count shrinks; one row per class cannot be selected on
    assert select_knn_readout(hidden[:42], y[:42], seed=3)['folds'] == 2                 # class 2 has two rows
    with pytest.raises(ValueError, match='two rows'):
        select_knn_readout(hidden[:41], y[:41], seed=3)

def test_multiview_knn_readout_keeps_the_output_rule_networks_and_votes_by_majority():
    from experiments.make_revision.comparisons import StableFootruleKNN
    from experiments.make_revision.evaluation import config_id
    from experiments.make_revision.models import seed_fit
    from experiments.make_revision.multiview import MultiViewArrowFlowKNN, knn_readout_candidates
    from experiments.make_revision.secondary_studies import majority
    Xtr, ytr, Xte, yte = split_small()
    params = dict(n_views=2, embed_dim=16, degree=1, widths=(16,), iterations=3, seed=7)
    seed_fit(7); knn = MultiViewArrowFlowKNN(**params).fit(Xtr, ytr)
    seed_fit(7); ref = MultiViewArrowFlow(**params).fit(Xtr, ytr)
    assert len(knn.views_) == len(knn.readouts_) == len(knn.readout_selections_) == 2
    for (enc_k, net_k), (enc_r, net_r) in zip(knn.views_, ref.views_):
        assert enc_k.seed == enc_r.seed and net_k.state_hash() == net_r.state_hash()      # the readout is the only change
    ids = {config_id(c) for c in knn_readout_candidates()}
    for readout, selection in zip(knn.readouts_, knn.readout_selections_):
        assert isinstance(readout, StableFootruleKNN) and readout.input_kind == 'positions'
        assert (readout.n_neighbors, readout.weights) == (selection['config']['n_neighbors'], selection['config']['weights'])
        assert selection['config_id'] in ids and 0 <= selection['inner_score'] <= 1
    predictions, orders = knn.predict_views(Xte)
    expected = [r.predict(net.transform_orders(enc.transform(Xte))) for (enc, net), r in zip(knn.views_, knn.readouts_)]
    assert all(np.array_equal(p, e) for p, e in zip(predictions, expected))
    assert all(np.array_equal(o, enc.transform(Xte)) for o, (enc, _) in zip(orders, knn.views_))
    out = knn.predict(Xte)
    assert out.shape == (len(Xte),) and np.array_equal(out, majority(expected)) and set(out) <= set(ytr)
    assert knn.encoding_seconds_ > 0 and knn.training_seconds_ > 0 and knn.readout_seconds_ > 0 and knn.last_encoding_seconds_ > 0
    knn.set_params(aggregation='borda')
    with pytest.raises(ValueError, match='majority'):
        knn.predict(Xte)

def test_multiview_knn_readout_selection_never_touches_rows_outside_the_training_partition(monkeypatch):
    from experiments.make_revision import comparisons
    from experiments.make_revision.evaluation import config_id
    from experiments.make_revision.multiview import KNN_SELECTION_FOLDS, MultiViewArrowFlowKNN
    Xtr, ytr, Xte, yte = split_small()
    calls = []
    original = comparisons.StableFootruleKNN.fit
    def spy(self, X, y, sample_ids=None):
        calls.append({'X': np.array(X, copy=True), 'y': np.array(y, copy=True), 'sample_ids': sample_ids,
                      'n_neighbors': self.n_neighbors, 'weights': self.weights, 'input_kind': self.input_kind})
        return original(self, X, y, sample_ids=sample_ids)
    monkeypatch.setattr(comparisons.StableFootruleKNN, 'fit', spy)
    model = MultiViewArrowFlowKNN(n_views=2, embed_dim=16, degree=1, widths=(16,), iterations=3, seed=7).fit(Xtr, ytr)
    per_view = KNN_SELECTION_FOLDS + 1
    assert len(calls) == 2 * per_view and all(c['input_kind'] == 'positions' for c in calls)
    for v, (enc, net) in enumerate(model.views_):
        train_hidden = net.transform_orders(enc.transform(Xtr))
        rows = {tuple(r) for r in train_hidden.tolist()}
        view_calls = calls[v * per_view:(v + 1) * per_view]
        for c in view_calls[:-1]:                                    # the selection fits: strict subsets of the training rows
            assert c['n_neighbors'] == 21 and len(c['y']) < len(ytr) and set(map(tuple, c['X'].tolist())) <= rows
            assert c['sample_ids'] is not None and np.array_equal(np.sort(c['sample_ids']), c['sample_ids'])
            assert np.array_equal(c['X'], train_hidden[c['sample_ids']]) and np.array_equal(c['y'], ytr[c['sample_ids']])
        held_out = np.concatenate([np.setdiff1d(np.arange(len(ytr)), c['sample_ids']) for c in view_calls[:-1]])
        assert np.array_equal(np.sort(held_out), np.arange(len(ytr)))  # the selection splits hold every training row out exactly once
        refit = view_calls[-1]                                       # the refit: every training row, the chosen setting
        chosen = model.readout_selections_[v]['config']
        assert np.array_equal(refit['X'], train_hidden) and np.array_equal(refit['y'], ytr) and refit['sample_ids'] is None
        assert (refit['n_neighbors'], refit['weights']) == (chosen['n_neighbors'], chosen['weights'])
        assert config_id(chosen) == model.readout_selections_[v]['config_id']
    fitted_rows = {tuple(r) for c in calls for r in c['X'].tolist()}
    train_rows = {tuple(r) for enc, net in model.views_ for r in net.transform_orders(enc.transform(Xtr)).tolist()}
    assert fitted_rows <= train_rows
    n = len(calls)
    model.predict(Xte)
    assert len(calls) == n                                           # prediction fits nothing

def test_multiview_knn_readout_selection_derives_its_split_seed_from_the_view_seed_and_is_reproducible():
    # The frozen protocol (bridge_knn.json knn_readout.selection) names random_state = derive_seed(view_seed, 'readout_selection');
    # the logged readout choices re-derive only if every fit reproduces them whatever the global RNG state.
    from experiments.make_revision.comparisons import derive_seed
    from experiments.make_revision.models import seed_fit
    from experiments.make_revision.multiview import MultiViewArrowFlowKNN, select_knn_readout
    X, y = small()
    params = dict(n_views=3, embed_dim=16, degree=1, widths=(16,), iterations=3, seed=7)
    seed_fit(1); first = MultiViewArrowFlowKNN(**params).fit(X, y)       # two different global RNG states before the fit
    seed_fit(2); second = MultiViewArrowFlowKNN(**params).fit(X, y)
    assert len(first.readout_selections_) == 3 and first.readout_selections_ == second.readout_selections_
    bare_view_seed = []
    for v, (enc, net) in enumerate(first.views_):
        hidden = net.transform_orders(enc.transform(X))
        view_seed = derive_seed(7, 'view', v)
        assert first.readout_selections_[v] == select_knn_readout(hidden, y, seed=derive_seed(view_seed, 'readout_selection'))
        bare_view_seed.append(select_knn_readout(hidden, y, seed=view_seed))
    assert bare_view_seed != first.readout_selections_                  # the fixture separates the protocol's seed from the bare view seed
