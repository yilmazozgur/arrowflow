"""Shared query perturbations and nested polynomial encodings for E06/E07.

No function accepts query labels. Fit statistics use the supplied training
rows; callers must pass the saved outer/inner partitions and preserve row order.
"""
from dataclasses import dataclass
import json
from math import comb
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .comparisons import derive_seed
from .models import NumericImputer, OrdinalEncoder, array_hash, numeric

SOURCE_MODULES = [
    "experiments.make_revision.models",
    "experiments.make_revision.comparisons",
]
CORRUPTION_SEEDS = (104729, 130363, 155921)
GAUSSIAN_LEVELS = (0., .1, .25, .5, 1.)
QUANTIZATION_STEPS = (0., .1, .25, .5, 1.)
MASKING_PROBABILITIES = (0., .05, .1, .2, .4)


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _levels(values, name, maximum=None):
    result = tuple(float(v) for v in values)
    if (len(set(result)) != len(result)
            or any(not np.isfinite(v) or v < 0 for v in result)
            or (maximum is not None and any(v > maximum for v in result))):
        raise ValueError(f"Invalid or duplicate {name}")
    return result


@dataclass(frozen=True)
class CorruptionCase:
    family: str
    severity: float
    base_seed: int | None
    draw_seed: int | None
    raw: np.ndarray
    raw_hash: str

    @property
    def case_id(self):
        return f"{self.family}__level{self.severity:g}__seed{self.base_seed}"


@dataclass(frozen=True)
class CorruptionBank:
    dataset_id: str
    outer_repeat: int
    outer_fold: int
    training_hash: str
    query_hash: str
    means: np.ndarray
    scales: np.ndarray
    cases: tuple[CorruptionCase, ...]
    normal_draws: object
    mask_draws: object
    normal_seeds: object
    mask_seeds: object
    _state_hashes: object

    @classmethod
    def create(
        cls, X_train, X_query, *, dataset_id, outer_repeat, outer_fold,
        base_seeds=CORRUPTION_SEEDS, gaussian_levels=GAUSSIAN_LEVELS,
        quantization_steps=QUANTIZATION_STEPS,
        masking_probabilities=MASKING_PROBABILITIES,
    ):
        train, query = numeric(X_train), numeric(X_query)
        if not train.shape[0] or not train.shape[1] or query.shape[1] != train.shape[1]:
            raise ValueError("Nonempty training data and matching feature counts required")
        seeds = tuple(base_seeds)
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(not isinstance(s, Integral) or isinstance(s, bool)
                       or not 0 <= s < 2**32 for s in seeds)):
            raise ValueError("Distinct uint32 base corruption seeds required")
        seeds = tuple(int(s) for s in seeds)
        gaussian_levels = _levels(gaussian_levels, "Gaussian levels")
        quantization_steps = _levels(quantization_steps, "quantization steps")
        masking_probabilities = _levels(masking_probabilities, "masking probabilities", 1)
        imputer = NumericImputer().fit(train)
        scaler = StandardScaler().fit(imputer.transform(train))
        means, scales = _readonly(scaler.mean_), _readonly(scaler.scale_)
        normal_draws, mask_draws, normal_seeds, mask_seeds = {}, {}, {}, {}
        for seed in seeds:
            parts = (str(dataset_id), int(outer_repeat), int(outer_fold))
            normal_seed = derive_seed(seed, *parts, "corruption_gaussian")
            mask_seed = derive_seed(seed, *parts, "corruption_mask")
            normal_seeds[seed], mask_seeds[seed] = normal_seed, mask_seed
            normal_draws[seed] = _readonly(
                np.random.RandomState(normal_seed).normal(size=(len(query), query.shape[1] + 1))
            )
            mask_draws[seed] = _readonly(
                np.random.RandomState(mask_seed).uniform(size=query.shape)
            )

        cases = []

        def add(family, severity, seed, draw_seed, raw):
            raw = _readonly(raw)
            cases.append(CorruptionCase(family, severity, seed, draw_seed, raw, array_hash(raw)))

        add("clean", 0., None, None, query)
        multipliers = np.linspace(.5, 1.5, query.shape[1])
        multipliers /= np.sqrt(np.mean(multipliers**2))
        for seed in seeds:
            g = normal_draws[seed]
            noises = {
                "gaussian_isotropic": g[:, 1:],
                "gaussian_heterogeneous": g[:, 1:] * multipliers,
                "gaussian_correlated": np.sqrt(.5) * (g[:, 1:] + g[:, :1]),
            }
            for family, noise in noises.items():
                for severity in gaussian_levels:
                    # Bypass round-trip standardization at zero for exact equality.
                    raw = query if severity == 0 else query + severity * scales * noise
                    add(family, severity, seed, normal_seeds[seed], raw)
        standardized = (query - means) / scales
        for step in quantization_steps:
            raw = query if step == 0 else means + scales * step * np.rint(standardized / step)
            add("quantization", step, None, None, raw)
        for seed in seeds:
            uniforms = mask_draws[seed]
            for severity in masking_probabilities:
                raw = query.copy()
                raw[uniforms < severity] = np.nan
                add("masking", severity, seed, mask_seeds[seed], raw)
        hashes = {"means": array_hash(means), "scales": array_hash(scales)}
        hashes.update({f"normal_{s}": array_hash(v) for s, v in normal_draws.items()})
        hashes.update({f"mask_{s}": array_hash(v) for s, v in mask_draws.items()})
        return cls(
            str(dataset_id), int(outer_repeat), int(outer_fold),
            array_hash(train), array_hash(query), means, scales, tuple(cases),
            MappingProxyType(normal_draws), MappingProxyType(mask_draws),
            MappingProxyType(normal_seeds), MappingProxyType(mask_seeds),
            MappingProxyType(hashes),
        )

    @property
    def clean(self):
        return self.cases[0].raw

    def assert_intact(self):
        arrays = {"means": self.means, "scales": self.scales}
        arrays.update({f"normal_{s}": v for s, v in self.normal_draws.items()})
        arrays.update({f"mask_{s}": v for s, v in self.mask_draws.items()})
        for name, value in arrays.items():
            if array_hash(value) != self._state_hashes[name]:
                raise ValueError(f"Shared corruption state changed: {name}")
        for case in self.cases:
            if array_hash(case.raw) != case.raw_hash:
                raise ValueError(f"Shared corruption array changed: {case.case_id}")

    def metadata(self):
        self.assert_intact()
        return {
            "dataset_id": self.dataset_id, "outer_repeat": self.outer_repeat,
            "outer_fold": self.outer_fold, "training_hash": self.training_hash,
            "query_hash": self.query_hash, "query_shape": list(self.clean.shape),
            "state_hashes": dict(self._state_hashes),
            "normal_seeds": {str(k): v for k, v in self.normal_seeds.items()},
            "mask_seeds": {str(k): v for k, v in self.mask_seeds.items()},
            "seed_derivation": "derive_seed(base,dataset,outer_repeat,outer_fold,stream)",
            "normal_stream": "corruption_gaussian",
            "mask_stream": "corruption_mask",
            "normal_columns": "shared component first, then one independent component per feature",
            "quantization": "training-standardized, zero origin, np.rint nearest-even",
            "cases": [
                {"case_id": c.case_id, "family": c.family, "severity": c.severity,
                 "base_seed": c.base_seed, "draw_seed": c.draw_seed, "raw_hash": c.raw_hash,
                 "array_key": f"case_{i:03d}"}
                for i, c in enumerate(self.cases)
            ],
        }

    def save(self, destination):
        """Save all shared raw arrays/draws; refuse existing destinations."""
        metadata = self.metadata()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        arrays = {"means": self.means, "scales": self.scales}
        arrays.update({f"normal_{s}": v for s, v in self.normal_draws.items()})
        arrays.update({f"mask_{s}": v for s, v in self.mask_draws.items()})
        arrays.update({f"case_{i:03d}": c.raw for i, c in enumerate(self.cases)})
        with (destination / "arrays.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        with (destination / "manifest.json").open("x") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


def degree_columns(n_features, degree):
    if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 1
           for v in (n_features, degree)):
        raise ValueError("Positive integer feature count and degree required")
    return comb(int(n_features) + int(degree), int(degree))


def degree_schedule(n_features, degrees=(1, 2, 3), column_cap=10000):
    if (not isinstance(column_cap, Integral) or column_cap < 1
            or len(set(degrees)) != len(degrees)):
        raise ValueError("Positive column cap and distinct degrees required")
    return [
        {"degree": int(d), "columns": degree_columns(n_features, d),
         "eligible": degree_columns(n_features, d) <= column_cap,
         "column_cap": int(column_cap)}
        for d in degrees
    ]


class NestedDegreeEncoder(OrdinalEncoder):
    """Random polynomial encoding with bias and shared monomial row ordering.

    Callers use the SAME E03 degree-two seed key for every degree in a partition.
    The row-major Gaussian matrix then shares coefficients for common monomials.
    This controlled degree-one path intentionally differs from E02's encoder.
    """
    def __init__(self, embed_dim=32, degree=2, seed=8129, column_cap=10000):
        super().__init__(strategy="random", embed_dim=embed_dim, degree=degree, seed=seed)
        self.column_cap = column_cap

    def fit(self, X, y=None):
        X = numeric(X)
        columns = degree_columns(X.shape[1], self.degree)
        if columns > self.column_cap:
            raise ValueError(
                f"Degree {self.degree} requires {columns} columns; cap is {self.column_cap}"
            )
        if self.embed_dim < 1:
            raise ValueError("Positive embedding width required")
        self.n_features_in_ = X.shape[1]
        self.imputer_ = NumericImputer().fit(X)
        self.poly_ = PolynomialFeatures(self.degree, include_bias=True)
        expanded = numeric(self.poly_.fit_transform(self.imputer_.transform(X)))
        self.scaler_ = StandardScaler().fit(expanded)
        scaled = self.scaler_.transform(expanded)
        self.projection_ = np.random.RandomState(self.seed).randn(columns, self.embed_dim)
        self.lda_ = self.calibration_ = None
        self.lda_scale_ = self.random_scale_ = 1.
        scores = self._scores_from_scaled(scaled)
        self.training_tie_rate_ = float(
            np.mean(np.any(np.diff(np.sort(scores, axis=1), axis=1) == 0, axis=1))
        )
        return self
