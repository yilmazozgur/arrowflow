# tests/make_revision/test_permlvq.py
"""Permutation LVQ layer (Task 8, B2): exact footrule median, ArrowFlow accumulator semantics (attraction,
repulsion, reset), the stacked classifier's shapes and readouts, and the readout rules on hand-built prototypes."""
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from arrowflow.permlvq import PermutationLVQ, PermutationLVQClassifier, footrule_median
from arrowflow.ranking import inverse_positions, score_order


def inv(order): return np.argsort(order)


def test_footrule_median_is_exact_on_the_three_item_example():
    # profile ABC, ABC, BCA, CBA (positions 0..2): Borda gives BAC, footrule median gives ABC
    profile = np.asarray([inv([0,1,2]), inv([0,1,2]), inv([1,2,0]), inv([2,1,0])])
    weights = np.ones(4)
    assert footrule_median(profile, weights, prior=None).tolist() == [0, 1, 2]


def test_attraction_moves_prototype_toward_input_and_reset_clears_votes():
    rng = np.random.RandomState(0)
    X = np.asarray([inv([3,2,1,0])] * 32); y = np.zeros(32, dtype=int)
    model = PermutationLVQ(prototypes_per_class=1, iterations=1, batch_size=32, learning_rate=5., p_correct=1., repulsion=False, seed=0)
    model.fit(X, y)
    assert model.prototypes_[0].tolist() == inv([3,2,1,0]).tolist()
    assert np.array_equal(model.accumulators_[0], np.eye(4))


def test_repulsion_pushes_wrong_class_prototype_toward_reversal():
    X = np.asarray([inv([0,1,2,3])] * 32); y = np.ones(32, dtype=int)
    model = PermutationLVQ(prototypes_per_class=1, iterations=1, batch_size=32, learning_rate=5., p_correct=1., repulsion=True, classes=(0, 1), seed=0)
    model.fit(X, y)
    wrong = model.prototypes_[model.labels_ == 0][0]
    assert wrong.tolist() == inv([3,2,1,0]).tolist()


def test_stacked_classifier_shapes_and_readouts():
    rng = np.random.RandomState(1)
    orders = np.asarray([rng.permutation(10) for _ in range(120)]); y = rng.randint(0, 3, 120)
    clf = PermutationLVQClassifier(layers=({'prototypes_per_class': 4}, {'prototypes_per_class': 3}), iterations=3, seed=2).fit_orders(orders, y)
    assert clf.transform_orders(orders).shape == (120, 9)        # 3 classes × 3 prototypes in layer 2
    for readout in ('nearest', 'plurality', 'borda'):
        clf.set_params(readout=readout)
        assert clf.predict_orders(orders).shape == (120,)


# ----------------------------------------------------------------------------- further pins

def test_footrule_median_prior_and_weights_enter_the_cost():
    swap = np.asarray([inv([1, 0])])
    assert footrule_median(swap, np.asarray([.5]), prior=np.asarray([0, 1])).tolist() == [0, 1]   # prior wins
    assert footrule_median(swap, np.asarray([2.]), prior=np.asarray([0, 1])).tolist() == [1, 0]   # vote wins
    assert footrule_median(swap, np.asarray([1.]), prior=None).tolist() == [1, 0]
    with pytest.raises(ValueError):
        footrule_median(swap, np.asarray([1., 1.]), prior=None)


def test_borda_and_footrule_median_aggregation_disagree_on_the_three_item_profile():
    # the same four accepted votes (weight 100 each; identity prior weight 1) reorder one prototype
    X = np.asarray([inv([0, 1, 2]), inv([0, 1, 2]), inv([1, 2, 0]), inv([2, 1, 0])]); y = np.zeros(4, dtype=int)
    shared = dict(prototypes_per_class=1, iterations=1, batch_size=4, learning_rate=100., p_correct=1., repulsion=False, seed=3)
    borda = PermutationLVQ(aggregation='borda', **shared).fit(X, y)
    median = PermutationLVQ(aggregation='footrule_median', **shared).fit(X, y)
    assert borda.prototypes_[0].tolist() == inv([1, 0, 2]).tolist()      # BAC by mean position
    assert median.prototypes_[0].tolist() == inv([0, 1, 2]).tolist()     # ABC by minimum-cost assignment
    assert np.array_equal(median.accumulators_[0], np.eye(3))


def test_votes_land_in_the_accumulator_in_the_prototype_order_with_the_identity_prior():
    X = np.asarray([inv([2, 0, 1])]); y = np.zeros(1, dtype=int)
    model = PermutationLVQ(prototypes_per_class=1, iterations=0, seed=5).fit(X, y)
    order = model.prototype_orders_[0].copy()
    model._vote(0, X[0], 2.5)
    expected = np.eye(3)
    expected[np.arange(3), X[0][order]] += 2.5              # row p = item order[p]; column = its target position
    assert np.array_equal(model.accumulators_[0], expected)
    model._vote(0, X[0], -1.)                                 # negative weight: reversed target, weight |a|
    expected[np.arange(3), 2 - X[0][order]] += 1.
    assert np.array_equal(model.accumulators_[0], expected)


def hand_built(prototypes, labels):
    V = prototypes.shape[1]
    model = PermutationLVQ(prototypes_per_class=1, iterations=0, classes=tuple(sorted(set(labels))), seed=0)
    model.fit(np.asarray([np.arange(V)]), np.asarray([labels[0]]))
    model.prototypes_ = np.asarray(prototypes, dtype=np.int64)
    model.prototype_orders_ = inverse_positions(model.prototypes_)
    model.labels_ = np.asarray(labels)
    model.accumulators_ = [np.eye(V) for _ in prototypes]
    return model


def test_readout_rules_on_hand_built_prototypes():
    # distances from the query [0,1,2,3]: 2, 2, 2, 4, 6, 6 -> ranked by (distance, prototype ID)
    prototypes = np.asarray([[1, 0, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2], [2, 1, 0, 3], [3, 1, 2, 0], [1, 2, 3, 0]])
    labels = np.asarray([1, 0, 0, 0, 1, 1])
    model = hand_built(prototypes, labels)
    query = np.asarray([[0, 1, 2, 3]])
    assert model.predict(query, readout='nearest').tolist() == [1]
    assert model.predict(query, readout='plurality', k=5).tolist() == [0]      # 3 of the 5 nearest
    assert model.predict(query, readout='borda', k=5).tolist() == [0]          # 4+3+2 = 9 against 5+1 = 6
    assert model.predict(query, readout='plurality', k=1).tolist() == [1]
    assert model.predict(query, readout='plurality', k=2).tolist() == [0]      # tie -> lowest label
    assert model.predict(query, readout='borda', k=2).tolist() == [1]          # 2 against 1
    assert model.predict(query, readout='plurality', k=50).tolist() == [0]     # k clipped to the prototype count
    ranking = model.transform(query)
    assert ranking.shape == (1, 6) and ranking.tolist() == [[0, 1, 2, 3, 4, 5]]
    with pytest.raises(ValueError):
        model.predict(query, readout='mode')
    with pytest.raises(ValueError):
        model.predict(np.asarray([[0, 1, 2]]))


def test_transform_is_the_inverse_positions_of_the_prototype_ranking():
    rng = np.random.RandomState(7)
    X = np.asarray([rng.permutation(6) for _ in range(40)]); y = rng.randint(0, 2, 40)
    model = PermutationLVQ(prototypes_per_class=3, iterations=4, batch_size=8, seed=9).fit(X, y)
    assert model.prototypes_.shape == (6, 6) and model.labels_.tolist() == [0, 1, 0, 1, 0, 1]
    assert all(sorted(p) == list(range(6)) for p in model.prototypes_.tolist())
    distances = cdist(X, model.prototypes_, metric='cityblock')
    assert np.array_equal(model.transform(X), inverse_positions(score_order(distances)))
    assert model.predict(X, readout='nearest').tolist() == model.labels_[np.argmin(distances, axis=1)].tolist()


def test_fit_is_reproducible_and_validates_inputs():
    rng = np.random.RandomState(4)
    X = np.asarray([rng.permutation(8) for _ in range(60)]); y = rng.randint(0, 3, 60)
    a = PermutationLVQ(prototypes_per_class=2, iterations=5, batch_size=16, seed=11).fit(X, y)
    b = PermutationLVQ(prototypes_per_class=2, iterations=5, batch_size=16, seed=11).fit(X, y)
    assert np.array_equal(a.prototypes_, b.prototypes_) and a.fit_seconds_ >= 0
    c = PermutationLVQ(prototypes_per_class=2, iterations=5, batch_size=16, seed=12).fit(X, y)
    assert not np.array_equal(a.prototypes_, c.prototypes_)
    with pytest.raises(ValueError):
        PermutationLVQ().fit(X + 1, y)                                # not permutations of 0..V-1
    with pytest.raises(ValueError):
        PermutationLVQ(classes=(0, 1)).fit(X, y)                      # label 2 outside the fixed classes
    with pytest.raises(ValueError):
        PermutationLVQ(aggregation='kemeny').fit(X, y)
    with pytest.raises(ValueError):
        PermutationLVQ(prototypes_per_class=0).fit(X, y)
    with pytest.raises(ValueError):
        PermutationLVQ(p_correct=1.5).fit(X, y)
    with pytest.raises(ValueError):
        PermutationLVQ().fit(X, y[:-1])


def test_classifier_stacks_greedily_and_exposes_the_orders_interface():
    rng = np.random.RandomState(2)
    orders = np.asarray([rng.permutation(7) for _ in range(90)]); y = rng.randint(0, 3, 90)
    clf = PermutationLVQClassifier(layers=({'prototypes_per_class': 2}, {'prototypes_per_class': 2, 'aggregation': 'footrule_median'}),
                                   iterations=2, batch_size=16, seed=5).fit_orders(orders, y)
    assert len(clf.layers_) == 2 and clf.training_seconds_ >= 0 and clf.classes_.tolist() == [0, 1, 2]
    assert clf.layers_[0].aggregation == 'borda' and clf.layers_[1].aggregation == 'footrule_median'
    positions = inverse_positions(orders)
    first = clf.layers_[0].transform(positions)
    assert np.array_equal(clf.transform(positions), clf.layers_[1].transform(first))
    assert np.array_equal(clf.transform_orders(orders), clf.transform(positions))
    assert np.array_equal(clf.predict_orders(orders), clf.layers_[1].predict(first, readout='plurality', k=5))
    again = PermutationLVQClassifier(**clf.get_params()).fit(positions, y)
    assert np.array_equal(again.predict(positions), clf.predict(positions))
    assert clf.set_params(readout='borda', k=3).predict_orders(orders).shape == (90,)
    with pytest.raises(ValueError):
        PermutationLVQClassifier(layers=()).fit_orders(orders, y)
    with pytest.raises(ValueError):
        PermutationLVQClassifier(layers=({'prototypes_per_class': 2, 'seed': 1},)).fit_orders(orders, y)
    with pytest.raises(ValueError):
        PermutationLVQClassifier(readout='mode').fit_orders(orders, y)


def test_prior_weight_scales_the_identity_prior_for_both_rules():
    # one prototype, one class; a batch of two identical samples = two consistent votes of weight lr toward the
    # reversal of the initial prototype. Under the median an item moves only when the vote mass exceeds the prior.
    V, lr = 4, .1
    initial = PermutationLVQ(prototypes_per_class=1, iterations=0, seed=6).fit(np.asarray([np.arange(V)]), [0]).prototypes_[0]
    target = V - 1 - initial
    X = np.asarray([target, target]); y = np.zeros(2, dtype=int)

    def fitted(aggregation, prior_weight):
        return PermutationLVQ(prototypes_per_class=1, iterations=1, batch_size=2, learning_rate=lr, p_correct=1., repulsion=False,
                              aggregation=aggregation, seed=6, prior_weight=prior_weight).fit(X, y)
    for aggregation in ('borda', 'footrule_median'):
        assert fitted(aggregation, 1.).prototypes_[0].tolist() == initial.tolist()            # prior 1 outweighs 0.2
        model = fitted(aggregation, lr)
        assert model.prototypes_[0].tolist() == target.tolist()                              # prior lr: two votes move it
        assert np.array_equal(model.accumulators_[0], lr * np.eye(V))                        # reset to the weighted prior
    # the module-level median takes the same weighted prior
    profile = np.asarray([target, target]); weights = np.full(2, lr)
    assert footrule_median(profile, weights, prior=initial, prior_weight=1.).tolist() == initial.tolist()
    assert footrule_median(profile, weights, prior=initial, prior_weight=lr).tolist() == target.tolist()
    assert PermutationLVQClassifier(prior_weight=lr).get_params()['prior_weight'] == lr
    clf = PermutationLVQClassifier(layers=({'prototypes_per_class': 1}, {'prototypes_per_class': 1, 'prior_weight': .5}),
                                   iterations=1, prior_weight=lr, seed=6).fit(X, y)
    assert [layer.prior_weight for layer in clf.layers_] == [lr, .5]
    for bad in (0., -1., np.inf):
        with pytest.raises(ValueError):
            PermutationLVQ(prior_weight=bad).fit(X, y)
        with pytest.raises(ValueError):                    # the module function applies the same rule as the layer
            footrule_median(profile, weights, prior=initial, prior_weight=bad)


def recorded_votes(monkeypatch):
    """Every _vote call as (prototype, weight), in order."""
    calls = []
    original = PermutationLVQ._vote

    def wrapped(self, m, target, weight):
        calls.append((int(m), float(weight)))
        return original(self, m, target, weight)
    monkeypatch.setattr(PermutationLVQ, '_vote', wrapped)
    return calls


def test_acceptance_is_wrong_nearest_label_or_the_p_correct_draw(monkeypatch):
    V = 6
    common = dict(prototypes_per_class=1, classes=(0, 1), seed=3, learning_rate=.5)
    initial = PermutationLVQ(iterations=0, **common).fit(np.asarray([np.arange(V)]), [0])
    p0, p1 = initial.prototypes_                       # the prototypes of labels 0 and 1 (same seed: same initialisation)
    calls = recorded_votes(monkeypatch)
    # (i) p_correct = 0: correctly decided samples never vote ...
    model = PermutationLVQ(iterations=1, batch_size=3, p_correct=0., **common).fit(np.asarray([p0, p0, p0]), np.zeros(3, dtype=int))
    assert model.vote_count_ == 0 and calls == [] and np.array_equal(model.prototypes_, initial.prototypes_)
    # ... while every misclassified sample (a row equal to p1 labelled 0) votes exactly once for `same` and, with
    # repulsion, once for `other`
    X, y = np.asarray([p0, p1, p1, p0, p1]), np.zeros(5, dtype=int)
    model = PermutationLVQ(iterations=1, batch_size=5, p_correct=0., repulsion=True, **common).fit(X, y)
    assert model.vote_count_ == 3 and calls == [(0, .5), (1, -.5)] * 3
    calls.clear()
    model = PermutationLVQ(iterations=1, batch_size=5, p_correct=0., repulsion=False, **common).fit(X, y)
    assert model.vote_count_ == 3 and calls == [(0, .5)] * 3
    calls.clear()
    # (ii) p_correct = 1: every sample votes, correctly decided or not
    model = PermutationLVQ(iterations=1, batch_size=4, p_correct=1., repulsion=True, **common).fit(np.asarray([p0] * 4), np.zeros(4, dtype=int))
    assert model.vote_count_ == 4 and calls == [(0, .5), (1, -.5)] * 4
    calls.clear()
    # (iii) p_correct = .5: the accepted count is the number of draws below .5 in the model's own RNG stream, which
    # after the prototype initialisation and the batch draw hands out one uniform per correctly decided sample
    N = 40
    single = dict(prototypes_per_class=1, classes=(0,), seed=3, learning_rate=.5)
    rng = np.random.RandomState(3)
    rng.permutation(V)                                 # one prototype
    rng.choice(N, size=N, replace=False)               # the batch
    expected = int((rng.rand(N) < .5).sum())
    model = PermutationLVQ(iterations=1, batch_size=N, p_correct=.5, **single).fit(np.asarray([p0] * N), np.zeros(N, dtype=int))
    assert 0 < expected < N and model.vote_count_ == expected and len(calls) == expected
    calls.clear()
    # (iv) one class with repulsion: no `other` prototype exists, so only attraction votes are cast
    rng = np.random.RandomState(9)
    X = np.asarray([rng.permutation(V) for _ in range(20)])
    model = PermutationLVQ(iterations=1, batch_size=20, p_correct=1., repulsion=True, **single).fit(X, np.zeros(20, dtype=int))
    assert model.vote_count_ == 20 and len(calls) == 20 and all(weight > 0 for _, weight in calls)
