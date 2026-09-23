# tests/make_revision/test_readouts.py
import numpy as np
import pytest
from arrowflow.readouts import KPrototypeBorda
from arrowflow.ranking import inverse_positions, score_order


def two_clusters_per_class(V=8, per_cluster=20, seed=0):
    rng = np.random.RandomState(seed)
    centers = {0: [np.arange(V), np.arange(V)[::-1]], 1: [np.roll(np.arange(V), 3), np.roll(np.arange(V), -3)]}
    X, y = [], []
    for label, cs in centers.items():
        for c in cs:
            for _ in range(per_cluster):
                noisy = c + rng.normal(0, .3, V)
                X.append(np.argsort(np.argsort(noisy)))   # inverse positions
                y.append(label)
    return np.asarray(X), np.asarray(y)


def test_kprototype_borda_recovers_two_clusters_per_class():
    rng = np.random.RandomState(0)
    V = 8
    centers = {0: [np.arange(V), np.arange(V)[::-1]], 1: [np.roll(np.arange(V), 3), np.roll(np.arange(V), -3)]}
    X, y = [], []
    for label, cs in centers.items():
        for c in cs:
            for _ in range(20):
                noisy = c + rng.normal(0, .3, V)
                X.append(np.argsort(np.argsort(noisy)))   # inverse positions
                y.append(label)
    X, y = np.asarray(X), np.asarray(y)
    model = KPrototypeBorda(k=2, seed=1).fit(X, y)
    assert model.prototypes_.shape == (4, V)
    assert all(sorted(p) == list(range(V)) for p in model.prototypes_.tolist())
    assert (model.predict(X) == y).mean() > .95


def test_one_prototype_per_class_equals_the_borda_classifier():
    from experiments.make_revision.comparisons import BordaClassifier
    X, y = two_clusters_per_class(seed=3)
    model = KPrototypeBorda(k=1, seed=0).fit(X, y)
    borda = BordaClassifier().fit(inverse_positions(X), y)        # BordaClassifier takes orders; X holds positions
    assert np.array_equal(model.prototypes_, borda.prototype_positions_)
    assert np.array_equal(model.prototype_labels_, borda.classes_)
    assert np.array_equal(model.predict(X), borda.predict(inverse_positions(X)))


def test_prototypes_are_borda_centroids_of_seeded_kmeans_groups_and_predictions_use_footrule():
    from scipy.spatial.distance import cdist
    from sklearn.cluster import KMeans
    X, y = two_clusters_per_class(seed=5)
    model = KPrototypeBorda(k=2, seed=7).fit(X, y)
    assert model.classes_.tolist() == [0, 1] and model.prototype_labels_.tolist() == [0, 0, 1, 1]
    expected = []
    for label in (0, 1):
        rows = X[y == label]
        groups = KMeans(n_clusters=2, n_init=4, random_state=7).fit_predict(rows.astype(float))
        for g in range(2):
            expected.append(inverse_positions(score_order(rows[groups == g].mean(axis=0)[None, :]))[0])
    assert np.array_equal(model.prototypes_, np.asarray(expected))
    assert model.prototype_orders_.tolist() == inverse_positions(model.prototypes_).tolist()
    nearest = np.argmin(cdist(X, model.prototypes_, metric='cityblock'), axis=1)
    assert np.array_equal(model.predict(X), model.prototype_labels_[nearest])
    again = KPrototypeBorda(k=2, seed=7).fit(X, y)
    assert np.array_equal(again.prototypes_, model.prototypes_)          # seeded, hence reproducible


def test_small_classes_yield_one_prototype_per_row_and_inputs_are_validated():
    V = 8
    X = np.array([np.roll(np.arange(V), i) for i in range(6)] + [np.roll(np.arange(V)[::-1], i) for i in range(6)])
    y = np.array([0] * 6 + [1] * 6)                                   # six distinct rows per class, fewer than k
    model = KPrototypeBorda(k=8, seed=0).fit(X, y)
    assert model.prototypes_.shape == (12, V) and model.prototype_labels_.tolist() == [0] * 6 + [1] * 6
    assert sorted(map(tuple, model.prototypes_.tolist())) == sorted(map(tuple, X.tolist()))   # one prototype per row
    assert (model.predict(X) == y).mean() == 1.0
    duplicated = KPrototypeBorda(k=4, seed=0).fit(np.repeat(X[:1], 5, axis=0), np.zeros(5, dtype=int))
    assert duplicated.prototypes_.shape[0] <= 4 and duplicated.prototypes_.shape[1] == V   # empty groups are skipped
    with pytest.raises(ValueError):
        KPrototypeBorda(k=0).fit(X, y)
    with pytest.raises(ValueError):
        KPrototypeBorda(k=2).fit(X[:, :4], y)                          # not complete permutations of 0..V-1
    with pytest.raises(ValueError):
        KPrototypeBorda(k=2).fit(X, y).predict(np.tile(np.arange(5), (3, 1)))   # vocabulary changed
