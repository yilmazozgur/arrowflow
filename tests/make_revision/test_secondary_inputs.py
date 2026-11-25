"""Scientific contracts for shared corruptions and the controlled degree study."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.make_revision.models import array_hash
from experiments.make_revision.secondary_inputs import (
    CorruptionBank, NestedDegreeEncoder, degree_columns, degree_schedule,
)


def bank(train, query, **kwargs):
    return CorruptionBank.create(
        train, query, dataset_id="fixture", outer_repeat=0, outer_fold=1, **kwargs
    )


def case_for(value, family, severity, seed=104729):
    return next(
        c for c in value.cases
        if (c.family, c.severity, c.base_seed) == (family, severity, seed)
    )


def test_zero_severity_is_byte_identical_and_arrays_are_immutable():
    train = np.array([[0., 2.], [3., 8.], [6., 11.]])
    query = np.array([[np.nextafter(1., 2.), 4.], [2., np.nan]])
    value = bank(train, query)
    zeros = [c for c in value.cases if c.severity == 0]
    assert len(value.cases) == 66
    assert len(zeros) == 14
    assert all(c.raw_hash == array_hash(query) for c in zeros)
    for c in value.cases:
        assert not c.raw.flags.writeable
    with pytest.raises(ValueError):
        value.cases[0].raw[0, 0] = 500.
    value.assert_intact()


def test_corruption_stats_use_only_training_rows_and_zero_scale_is_one():
    train = np.array([[1., 7.], [3., 7.]])
    a = bank(train, np.array([[100., 9.]]))
    b = bank(train, np.array([[-1000., -20.]]))
    np.testing.assert_array_equal(a.means, [2., 7.])
    np.testing.assert_array_equal(a.scales, [1., 1.])
    np.testing.assert_array_equal(a.means, b.means)
    np.testing.assert_array_equal(a.scales, b.scales)
    assert a.training_hash == b.training_hash
    assert a.query_hash != b.query_hash


def test_shared_draws_reproduce_across_studies_and_scale_across_severity():
    rng = np.random.RandomState(4)
    train, query = rng.normal(size=(40, 5)), rng.normal(size=(20, 5))
    a, b = bank(train, query), bank(train, query)
    assert a.metadata() == b.metadata()
    for ca, cb in zip(a.cases, b.cases):
        np.testing.assert_array_equal(ca.raw, cb.raw)
    for family in ("gaussian_isotropic", "gaussian_heterogeneous", "gaussian_correlated"):
        low = case_for(a, family, .1).raw - query
        high = case_for(a, family, .5).raw - query
        np.testing.assert_allclose(high, 5 * low, atol=2e-15)
    assert a.normal_seeds[104729] != a.mask_seeds[104729]
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(a.normal_draws[104729], a.normal_draws[130363])


def test_gaussian_covariances_match_declared_families_in_standardized_units():
    train = np.array([[-2.] * 4, [2.] * 4])
    value = bank(
        train, np.zeros((80000, 4)), base_seeds=(104729,),
        gaussian_levels=(1.,), quantization_steps=(), masking_probabilities=(),
    )
    noises = {
        family: case_for(value, family, 1.).raw / value.scales
        for family in ("gaussian_isotropic", "gaussian_heterogeneous", "gaussian_correlated")
    }
    cov = {family: np.cov(x, rowvar=False) for family, x in noises.items()}
    np.testing.assert_allclose(cov["gaussian_isotropic"], np.eye(4), atol=.02)
    np.testing.assert_allclose(
        cov["gaussian_correlated"], .5 * np.eye(4) + .5 * np.ones((4, 4)), atol=.02
    )
    var = np.diag(cov["gaussian_heterogeneous"])
    assert np.all(np.diff(var) > 0)
    assert var[-1] / var[0] == pytest.approx(9., abs=.2)
    assert np.mean(var) == pytest.approx(1., abs=.02)
    assert all(np.max(np.abs(x.mean(axis=0))) < .015 for x in noises.values())


def test_quantization_uses_nearest_even_at_half_steps_in_training_units():
    value = bank(
        np.array([[-1.], [1.]]), np.array([[-.75], [-.25], [.25], [.75]]),
        base_seeds=(104729,), gaussian_levels=(), quantization_steps=(.5,),
        masking_probabilities=(),
    )
    quantized = case_for(value, "quantization", .5, None)
    np.testing.assert_array_equal(quantized.raw[:, 0], [-1., 0., 0., 1.])
    assert quantized.draw_seed is None


def test_masks_are_nested_and_preserve_unmasked_raw_entries():
    query = np.arange(10000., dtype=float).reshape(2000, 5)
    value = bank(np.vstack([-np.ones(5), np.ones(5)]), query)
    previous = np.zeros_like(query, dtype=bool)
    for severity in (0., .05, .1, .2, .4):
        masked = case_for(value, "masking", severity).raw
        current = np.isnan(masked)
        assert np.all(~previous | current)
        np.testing.assert_array_equal(masked[~current], query[~current])
        previous = current
    assert abs(previous.mean() - .4) < .02


def test_saved_bank_reconciles_every_case_and_detects_tampering(tmp_path):
    value = bank(np.array([[0., 2.], [2., 4.]]), np.array([[1., 3.], [2., 8.]]))
    destination = tmp_path / "bank"
    value.save(destination)
    metadata = json.loads((destination / "manifest.json").read_text())
    with np.load(destination / "arrays.npz", allow_pickle=False) as saved:
        assert len(metadata["cases"]) == len(value.cases)
        for record, original in zip(metadata["cases"], value.cases):
            assert array_hash(saved[record["array_key"]]) == original.raw_hash
        np.testing.assert_array_equal(saved["means"], value.means)
        np.testing.assert_array_equal(saved["scales"], value.scales)
    with pytest.raises(FileExistsError):
        value.save(destination)
    value.cases[0].raw.setflags(write=True)
    value.cases[0].raw[0, 0] += 1
    with pytest.raises(ValueError, match="changed"):
        value.assert_intact()


@pytest.mark.parametrize("kwargs", [
    {"gaussian_levels": (-.1,)},
    {"quantization_steps": (np.nan,)},
    {"masking_probabilities": (1.1,)},
    {"base_seeds": (104729, 104729)},
    {"gaussian_levels": (.1, .1)},
])
def test_invalid_or_duplicate_conditions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        bank(np.ones((2, 2)), np.ones((1, 2)), **kwargs)


def test_degree_schedule_excludes_oversized_expansion_before_fitting():
    assert degree_columns(64, 3) == 47905
    assert degree_columns(4, 1) == 5
    rows = degree_schedule(64)
    assert [(r["degree"], r["columns"], r["eligible"]) for r in rows] == [
        (1, 65, True), (2, 2145, True), (3, 47905, False)
    ]
    with pytest.raises(ValueError, match="47905"):
        NestedDegreeEncoder(degree=3).fit(np.ones((2, 64)))


def test_common_monomials_have_identical_projection_rows_and_fitted_scaling():
    rng = np.random.RandomState(42)
    train = rng.normal(size=(30, 4))
    encoders = [NestedDegreeEncoder(degree=d, seed=313, embed_dim=32).fit(train)
                for d in (1, 2, 3)]
    assert encoders[0].poly_.include_bias
    for low, high in zip(encoders, encoders[1:]):
        width = low.projection_.shape[0]
        np.testing.assert_array_equal(low.poly_.powers_, high.poly_.powers_[:width])
        np.testing.assert_array_equal(low.projection_, high.projection_[:width])
        np.testing.assert_allclose(low.scaler_.mean_, high.scaler_.mean_[:width], atol=0)
        np.testing.assert_allclose(low.scaler_.scale_, high.scaler_.scale_[:width], atol=0)
        np.testing.assert_array_equal(
            low.scaler_.transform(low.poly_.transform(train))[:, 0], 0
        )


def test_degree_encoder_is_label_independent_fold_fitted_and_returns_full_orders():
    train = np.array([[0., 3.], [2., 5.], [4., 7.], [6., 9.]])
    query = np.array([[100., -100.], [np.nan, 5.]])
    a = NestedDegreeEncoder(degree=1, embed_dim=8, seed=17).fit(train, [0, 0, 1, 1])
    b = NestedDegreeEncoder(degree=1, embed_dim=8, seed=17).fit(train, [1, 0, 1, 0])
    before = (a.scaler_.mean_.copy(), a.scaler_.scale_.copy(), a.projection_.copy())
    orders = a.transform(query)
    np.testing.assert_array_equal(orders, b.transform(query))
    np.testing.assert_array_equal(np.sort(orders, axis=1), np.tile(np.arange(8), (2, 1)))
    for saved, actual in zip(before, (a.scaler_.mean_, a.scaler_.scale_, a.projection_)):
        np.testing.assert_array_equal(saved, actual)
    ties = NestedDegreeEncoder(degree=2, embed_dim=8, seed=17).fit(np.ones((4, 2)))
    np.testing.assert_array_equal(ties.transform(np.ones((1, 2)))[0], np.arange(8))
