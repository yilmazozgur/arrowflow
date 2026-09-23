"""Ranking order and independent inverse-position metric oracles."""
import numpy as np
import pytest
from arrowflow.arrowflow import DataGraph
from experiments import exp_knn_vs_arrowflow as comparison


def footrule_oracle(a, b):
    assert len(set(a)) == len(a) and set(a) == set(b)
    return sum(abs(a.index(item) - b.index(item)) for item in a)


def test_actual_comparison_knn_uses_inverse_positions():
    a, b = [0, 1, 3, 2], [0, 2, 3, 1]
    assert sum(abs(x-y) for x, y in zip(a, b)) == 2
    assert footrule_oracle(a, b) == 4
    assert hasattr(comparison, 'fit_ordinal_knn'), 'comparison needs one shared ordinal fit path'
    knn = comparison.fit_ordinal_knn([a], [0], 1)
    assert knn.kneighbors([b], return_distance=True)[0][0, 0] == 4
    relabel = {0: 2, 1: 0, 2: 3, 3: 1}
    ar, br = [[relabel[x] for x in row] for row in [a, b]]
    knn = comparison.fit_ordinal_knn([ar], [0], 1)
    assert knn.kneighbors([br], return_distance=True)[0][0, 0] == 4


def test_numeric_encoder_keeps_equal_coordinates_and_numeric_ties():
    graph = DataGraph('encoder')
    # Cached identity projection and no scaler expose the actual ranking boundary.
    data, vocab, _ = graph.project_random_space(np.array([[2, 2, 1, 4]]), [0],
        W_random=[[np.eye(4)], None, 1, 4], poly_expansion=False, pol_deg=1, no_dimensions=4)
    assert data[0][0] == ['3', '1', '2', '4']
    assert set(vocab) == {'1', '2', '3', '4'}
    data, _, _ = graph.project_random_space(np.zeros((1, 20)), [0],
        W_random=[[np.eye(20)], None, 1, 20], poly_expansion=False, pol_deg=1, no_dimensions=20)
    assert data[0][0] == list(map(str, range(1, 21)))


def test_nonfinite_numeric_features_require_explicit_imputation():
    with pytest.raises(ValueError, match='finite|imput'):
        DataGraph('encoder').project_random_space(np.array([[np.nan, 2.]]), [0],
            W_random=[[np.eye(2)], None, 1, 2], poly_expansion=False, pol_deg=1, no_dimensions=2)


def test_comparison_encoding_imputes_from_training_only():
    train = np.array([[0., np.nan], [2., 4.], [4., 8.]])
    test = np.array([[np.nan, 6.]])
    actual = comparison.encode_view(train, [0, 1, 1], test, 'random', 8, .3, 42)
    expected = comparison.encode_view(np.array([[0., 6.], [2., 4.], [4., 8.]]),
        [0, 1, 1], np.array([[2., 6.]]), 'random', 8, .3, 42)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


def test_comparison_ensemble_uses_corrected_neighbor_path(monkeypatch):
    # Encoding is fixed to isolate the actual comparison metric, prediction and vote.
    # Query b is footrule 4 from a, 2 from c; direct order-array L1 picks a.
    a, b, c = [0, 1, 3, 2], [0, 2, 3, 1], [0, 3, 2, 1]
    monkeypatch.setattr(comparison, 'encode_view',
        lambda *args: (np.array([a, c]), np.array([b])))
    assert comparison.knn_multiview_ensemble(np.zeros((2, 1)), [0, 1],
        np.zeros((1, 1)), np.array([1]), 2, 4, 1, 'random', .3, 1, 1, 42) == 0


@pytest.mark.parametrize('orders', [[[0, 0, 2]], [[0, 1, 3]], [[.5, 1, 2]]])
def test_knn_rejects_nonpermutations(orders):
    with pytest.raises(ValueError, match='permutation'):
        comparison.fit_ordinal_knn(orders, [0], 1)


def test_all_missing_training_feature_is_zero_without_using_test_values():
    from arrowflow.ranking import impute_numeric
    train, test = impute_numeric([[np.nan, 2], [np.nan, 4]], [[np.nan, np.nan], [99, 9]])
    np.testing.assert_array_equal(train, [[0, 2], [0, 4]])
    np.testing.assert_array_equal(test, [[0, 3], [99, 9]])


def test_knn_and_ensemble_vote_ties_choose_lowest_class_id():
    model = comparison.fit_ordinal_knn([[0, 1, 2], [0, 1, 2]], [10, 2], 2)
    assert model.predict([[0, 1, 2]]).tolist() == [2]
