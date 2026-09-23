"""Fold-fitted numeric/ordinal transforms and a seeded sort-only adapter."""
from dataclasses import dataclass
import hashlib
import random
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from arrowflow.ranking import score_order, inverse_positions


def numeric(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or np.isinf(X).any():
        raise ValueError('Expected a numeric matrix without infinity')
    return X


class NumericImputer(TransformerMixin, BaseEstimator):
    """Training column means, zero for wholly missing columns; preserve width."""
    def fit(self, X, y=None):
        X = numeric(X)
        count = np.sum(~np.isnan(X), axis=0)
        self.means_ = np.divide(np.nansum(X, axis=0), count,
                                out=np.zeros(X.shape[1]), where=count > 0)
        return self

    def transform(self, X):
        check_is_fitted(self, 'means_')
        X = numeric(X)
        if X.shape[1] != len(self.means_):
            raise ValueError('Feature count changed')
        return np.where(np.isnan(X), self.means_, X)


class FittedECDF(TransformerMixin, BaseEstimator):
    """Mid-distribution F(x)=(#training<x + #training<=x)/(2N)."""
    def fit(self, X, y=None):
        self.imputer_ = NumericImputer().fit(X)
        self.sorted_ = np.sort(self.imputer_.transform(X), axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, 'sorted_')
        X = self.imputer_.transform(X)
        return np.column_stack([(np.searchsorted(col, X[:, j], side='left') +
                                 np.searchsorted(col, X[:, j], side='right')) /
                                (2 * len(col)) for j, col in enumerate(self.sorted_.T)])


class OrdinalEncoder(TransformerMixin, BaseEstimator):
    """Frozen legacy projection path; coordinate IDs ranked with stable ties.

    For rank-deficient LDA, fail rather than silently shrink the requested
    vocabulary or substitute a different representation.
    """
    def __init__(self, strategy='random', embed_dim=16, degree=1, lda_ratio=.3, seed=8129):
        self.strategy = strategy
        self.embed_dim = embed_dim
        self.degree = degree
        self.lda_ratio = lda_ratio
        self.seed = seed

    def fit(self, X, y=None):
        if self.strategy not in ('random', 'target_aware', 'calibrated'):
            raise ValueError('Unknown projection strategy')
        if self.embed_dim < 1 or self.degree < 1:
            raise ValueError('Positive embedding width and polynomial degree required')
        self.imputer_ = NumericImputer().fit(X)
        X = self.imputer_.transform(X)
        self.poly_ = PolynomialFeatures(self.degree, include_bias=True) if self.degree > 1 else None
        if self.poly_ is not None:
            X = self.poly_.fit_transform(X)
        self.scaler_ = StandardScaler().fit(X)
        X = self.scaler_.transform(X)
        rng = np.random.RandomState(self.seed)
        self.lda_ = None
        self.calibration_ = None
        self.lda_scale_ = self.random_scale_ = 1.
        n_lda = 0
        if self.strategy == 'target_aware':
            if y is None or len(np.unique(y)) < 2:
                raise ValueError('Target-aware encoding requires at least two training classes')
            n_lda = max(1, min(int(self.embed_dim * self.lda_ratio),
                              len(np.unique(y)) - 1, X.shape[1], self.embed_dim))
            self.lda_ = LinearDiscriminantAnalysis(n_components=n_lda).fit(X, y)
            lda = self.lda_.transform(X)
            if lda.shape[1] != n_lda:
                raise ValueError('Effective LDA rank cannot honor requested embedding vocabulary')
        self.projection_ = rng.randn(X.shape[1], self.embed_dim - n_lda)
        if self.lda_ is not None and self.projection_.shape[1]:
            self.lda_scale_ = np.std(lda) + 1e-10
            self.random_scale_ = np.std(X @ self.projection_) + 1e-10
        projected = self._project(X)
        if self.strategy == 'calibrated':
            self.calibration_ = StandardScaler().fit(projected)
        scores = self._scores_from_scaled(X)
        self.training_tie_rate_ = float(np.mean(np.any(np.diff(np.sort(scores, axis=1), axis=1) == 0, axis=1)))
        return self

    def _project(self, X):
        random_block = (X @ self.projection_) / self.random_scale_
        if self.lda_ is None:
            return random_block
        lda_block = self.lda_.transform(X) / self.lda_scale_
        return np.hstack([lda_block, random_block])

    def _scores_from_scaled(self, X):
        scores = self._project(X)
        return self.calibration_.transform(scores) if self.calibration_ is not None else scores

    def transform(self, X):
        check_is_fitted(self, 'projection_')
        X = self.imputer_.transform(X)
        if self.poly_ is not None:
            X = self.poly_.transform(X)
        return score_order(self._scores_from_scaled(self.scaler_.transform(X)))


def array_hash(X):
    X = np.ascontiguousarray(X)
    return hashlib.sha256(str((X.shape, X.dtype.str)).encode() + X.tobytes()).hexdigest()


@dataclass(frozen=True)
class SharedEncoding:
    """Exact, immutable arrays delivered to matched classifier controls."""
    train: np.ndarray
    test: np.ndarray
    train_hash: str
    test_hash: str
    raw_train_hash: str
    raw_test_hash: str

    @classmethod
    def create(cls, fitted_encoder, X_train, X_test):
        train = fitted_encoder.transform(X_train)
        test = fitted_encoder.transform(X_test)
        train.setflags(write=False)
        test.setflags(write=False)
        return cls(train, test, array_hash(train), array_hash(test),
                   array_hash(np.asarray(X_train)), array_hash(np.asarray(X_test)))

    def save(self, path):
        from pathlib import Path
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, train=self.train, test=self.test,
                                train_hash=self.train_hash, test_hash=self.test_hash,
                                raw_train_hash=self.raw_train_hash, raw_test_hash=self.raw_test_hash)

    def assert_identical(self, train, test):
        if (array_hash(train) != self.train_hash or array_hash(test) != self.test_hash
                or not np.array_equal(train, self.train) or not np.array_equal(test, self.test)):
            raise ValueError('Matched classifiers received different encodings')


def seed_fit(seed):
    import torch
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.set_num_threads(1)


class ArrowFlowEstimator(ClassifierMixin, TransformerMixin, BaseEstimator):
    """One view; optional validation checkpoint and augmentation (both off by default).

    transform returns inverse positions of the last hidden ranking for probes.
    A probe pipeline must fit this learner inside every inner fit.
    fit_orders/predict_orders allow exact shared-encoding comparisons.
    """
    def __init__(self, embed_dim=16, degree=1, strategy='random', lda_ratio=.3,
                 widths=(64,), iterations=100, learning_rate=.5, batch_size=32,
                 last_layer_update=True, ratio_data_backprop=.5,
                 motion_normalization_mult=.125, p_correct=.01, seed=8129,
                 validation_ratio=0.0, augment=False, n_augmentations=1, max_swaps=2):
        self.embed_dim = embed_dim
        self.degree = degree
        self.strategy = strategy
        self.lda_ratio = lda_ratio
        self.widths = widths
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.last_layer_update = last_layer_update
        self.ratio_data_backprop = ratio_data_backprop
        self.motion_normalization_mult = motion_normalization_mult
        self.p_correct = p_correct
        self.seed = seed
        self.validation_ratio = validation_ratio
        self.augment = augment
        self.n_augmentations = n_augmentations
        self.max_swaps = max_swaps

    def fit(self, X, y):
        start = time.perf_counter()
        self.encoder_ = OrdinalEncoder(self.strategy, self.embed_dim, self.degree, self.lda_ratio, self.seed).fit(X, y)
        orders = self.encoder_.transform(X)
        self.encoding_seconds_ = time.perf_counter() - start
        return self.fit_orders(orders, y)

    def fit_orders(self, orders, y):
        return self.initialize_orders(orders, y).train_initialized(orders, y)

    def initialize_orders(self, orders, y):
        """Initialize once without training; safe to deepcopy for matched controls."""
        from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config
        from arrowflow.arrowflow import SortFlowHybridNetwork
        inverse_positions(orders)  # validate before native string/integer conversion
        check_consistent_length(orders, y)
        seed_fit(self.seed)
        self.classes_, labels = np.unique(y, return_inverse=True)
        if np.asarray(orders).shape[1] != self.embed_dim:
            raise ValueError('Encoding vocabulary does not match embed_dim')
        cfg = ArrowFlowConfig(no_of_filters=list(self.widths), layer_types=['sort'] * (len(self.widths) + 1),
                              no_of_iters=self.iterations, batch_size=self.batch_size,
                              learning_rate=self.learning_rate, val_data_ratio=self.validation_ratio, device='cpu',
                              verbose=0, evaluate_train_data=False, last_layer_update=self.last_layer_update,
                              ratio_data_backprop=self.ratio_data_backprop,
                              motion_normalization_mult=self.motion_normalization_mult,
                              change_probability_when_decision_correct=self.p_correct)
        self.config_ = _build_sortnet_config(cfg, len(self.classes_))
        self.network_ = SortFlowHybridNetwork('revision', list(map(str, range(1, self.embed_dim + 1))),
                                              len(self.classes_), 'revision', self.config_)
        self.training_encoding_hash_ = array_hash(np.asarray(orders))
        self.training_labels_hash_ = array_hash(np.asarray(y))
        self._training_rng_state_ = np.random.get_state()
        self.initial_state_hash_ = self.state_hash()
        return self

    def train_initialized(self, orders, y):
        check_is_fitted(self, 'network_')
        if (self.network_.update_iter != 0 or array_hash(np.asarray(orders)) != self.training_encoding_hash_
                or array_hash(np.asarray(y)) != self.training_labels_hash_):
            raise ValueError('Training must start from its matching initial state and training partition')
        samples = self._samples(orders, np.searchsorted(self.classes_, y))
        np.random.set_state(self._training_rng_state_)
        if self.validation_ratio > 0:
            # The core takes the FIRST fraction as validation; shuffle first with the seeded stream.
            samples = [samples[i] for i in np.random.permutation(len(samples))]
        n_val = int(self.validation_ratio * len(samples))
        if self.augment and self.n_augmentations > 0:
            from arrowflow.arrowflow import DataGraph
            head, tail = samples[:n_val], samples[n_val:]
            tail = DataGraph.augment_permutation_data(tail, n_augmentations=self.n_augmentations,
                                                      max_swaps=self.max_swaps, seed=self.seed)
            samples = head + tail
        if int(self.config_.val_data_ratio * len(samples)) != n_val:
            # The core splits by int(val_data_ratio * len(list)); augmentation lengthened the list,
            # so pass a ratio that reproduces exactly the n_val held-out rows at the head.
            self.config_.val_data_ratio = (n_val + .5) / len(samples)
        self.validation_sample_count_ = n_val
        self.training_sample_count_ = len(samples) - n_val
        start = time.perf_counter()
        self.network_.train([samples, samples], self.config_)
        self.training_seconds_ = time.perf_counter() - start
        self._training_rng_state_ = np.random.get_state()
        return self

    def state_snapshot(self):
        """Copies of permutation/accumulator/cache arrays, classes and RNG state."""
        check_is_fitted(self, 'network_')
        arrays = {'classes': self.classes_.copy(),
                  'training_scalars': np.asarray([self.network_.learning_rate,
                      self.network_.update_iter, self.network_.num_of_epochs]),
                  'rng_keys': self._training_rng_state_[1].copy(),
                  'rng_meta': np.asarray(self._training_rng_state_[2:])}
        for i, layer in enumerate(self.network_.graph.vertex_list.values()):
            vertices = list(layer.graph.vertex_list.values())
            arrays[f'layer_{i}_orders'] = np.asarray([v.adjacency_list for v in vertices])
            arrays[f'layer_{i}_accumulators'] = np.asarray([v.permutation_matrix_accumulate for v in vertices])
            arrays[f'layer_{i}_positions'] = layer.index_matrix.copy()
        return arrays

    def state_hash(self):
        h = hashlib.sha256()
        for key, value in sorted(self.state_snapshot().items()):
            h.update(key.encode()); h.update(array_hash(value).encode())
        return h.hexdigest()

    @staticmethod
    def _samples(orders, labels=None):
        inverse_positions(orders)
        if labels is None:
            labels = np.zeros(len(orders), dtype=int)
        return [[list(map(str, (np.asarray(row, dtype=int) + 1))), str(int(label)), 1.] for row, label in zip(orders, labels)]

    def predict_orders(self, orders):
        check_is_fitted(self, 'network_')
        pred = self.network_.forward_propagate(self._samples(orders), 'supervised', 'classification', evaluate_only=True)[3]
        return self.classes_[np.asarray(pred, dtype=int)]

    def predict_class_ranking(self, orders):
        """Class labels per row from the nearest to the farthest output filter, shape (N, C)."""
        check_is_fitted(self, 'network_')
        self.network_.forward_propagate(self._samples(orders), 'supervised', 'classification', evaluate_only=True)
        return self.classes_[self.network_.last_output_rankings_]

    def predict(self, X):
        start = time.perf_counter()
        orders = self.encoder_.transform(X)
        self.last_encoding_seconds_ = time.perf_counter() - start
        return self.predict_orders(orders)

    def transform_orders(self, orders):
        return self.transform_orders_by_depth(orders)[-1]

    def transform_orders_by_depth(self, orders):
        check_is_fitted(self, 'network_')
        positions = inverse_positions(orders)
        if not self.widths:
            raise ValueError('Hidden representation requires at least one hidden layer')
        depths = []
        for i in range(len(self.widths)):
            layer = self.network_.graph.vertex_list[f'revision_ly{i}']
            positions = inverse_positions(score_order(cdist(positions, layer.index_matrix, metric='cityblock')))
            depths.append(positions)
        return depths

    def transform(self, X):
        return self.transform_orders(self.encoder_.transform(X))
