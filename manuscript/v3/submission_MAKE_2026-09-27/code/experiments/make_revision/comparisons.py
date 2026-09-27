"""Prespecified conventional and native ordinal controls for MAKE.

E02 independently tunes full numeric pipelines. E03/E04 will reuse the ordinal
estimators on exact shared arrays; this module does not implement that study.
"""
import hashlib
from functools import partial
import time
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted, check_consistent_length
from arrowflow.ranking import inverse_positions, score_order
from .evaluation import ModelSpec, candidate_grid, canonical_json
from .models import NumericImputer, ArrowFlowEstimator, array_hash

SOURCE_MODULES = ['experiments.make_revision.datasets', 'experiments.make_revision.models',
                  'experiments.make_revision.evaluation', 'experiments.make_revision.run_revision']


def derive_seed(base_seed, *stream_parts):
    """Specified SHA-256 first-four-byte, big-endian uint32 stream seed."""
    digest = hashlib.sha256(canonical_json([base_seed, *stream_parts]).encode()).digest()
    return int.from_bytes(digest[:4], 'big')


class InversePositionTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        inverse_positions(X)
        self.vocabulary_size_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'vocabulary_size_')
        positions = inverse_positions(X)
        if positions.shape[1] != self.vocabulary_size_:
            raise ValueError('Permutation vocabulary changed')
        return positions


class TimedPipeline(Pipeline):
    """Ordinary fold-local sklearn steps with recorded timing and fit warnings."""
    def fit(self, X, y=None, **params):
        if params:
            raise ValueError('This unit-weight evaluation does not route fit metadata')
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            start = time.perf_counter()
            self.preprocessing_ = Pipeline(self.steps[:-1])
            transformed = self.preprocessing_.fit_transform(X, y)
            self.encoding_seconds_ = time.perf_counter() - start
            start = time.perf_counter()
            self.steps[-1][1].fit(transformed, y)
            self.classifier_fit_seconds_ = time.perf_counter() - start
        self.fit_warnings_ = [{'category': w.category.__name__, 'message': str(w.message)} for w in captured]
        return self

    def predict(self, X):
        check_is_fitted(self, 'preprocessing_')
        start = time.perf_counter()
        transformed = self.preprocessing_.transform(X)
        self.last_encoding_seconds_ = time.perf_counter() - start
        return self.steps[-1][1].predict(transformed)


def conventional_factory(family, config, seed, native=False):
    constructors = {'svc_rbf': SVC, 'random_forest': RandomForestClassifier,
                    'mlp': MLPClassifier, 'numeric_knn': KNeighborsClassifier,
                    'gradient_boosting': GradientBoostingClassifier}
    settings = dict(config)
    if family == 'svc_rbf':
        settings.update(kernel='rbf', probability=False, max_iter=1000000)
    if family in ('random_forest', 'mlp', 'gradient_boosting'):
        settings['random_state'] = seed
    if family in ('random_forest', 'numeric_knn'):
        settings['n_jobs'] = 1
    if family == 'mlp':
        settings.update(solver='adam', early_stopping=False)
    steps = [('positions', InversePositionTransformer())] if native else []
    steps.append(('imputer', NumericImputer()))
    if family in ('svc_rbf', 'mlp', 'numeric_knn'):
        steps.append(('scaler', StandardScaler()))
    steps.append(('classifier', constructors[family](**settings)))
    return TimedPipeline(steps)


CONVENTIONAL_GRIDS = {
    'svc_rbf': {'C': [.1, 1, 10, 100], 'gamma': ['scale', .01, .1, 1]},
    'random_forest': {'n_estimators': [100, 300], 'max_features': ['sqrt', .5, 1.],
                      'min_samples_leaf': [1, 2, 5], 'max_depth': [None, 10]},
    'mlp': {'hidden_layer_sizes': [(64,), (128,), (64, 32)], 'alpha': [.0001, .01],
            'learning_rate_init': [.001, .01], 'max_iter': [200, 400]},
    'numeric_knn': {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance'], 'p': [1, 2]},
    'gradient_boosting': {'n_estimators': [100, 200], 'learning_rate': [.03, .1],
                          'max_depth': [1, 2, 3], 'min_samples_leaf': [1, 5]},
}


def conventional_registry(protocol, native=False):
    result = {}
    for family, grid in CONVENTIONAL_GRIDS.items():
        name = f'native_{family}' if native else family
        result[name] = ModelSpec(name, partial(conventional_factory, family, native=native),
                                 candidate_grid(grid, protocol['candidate_budget'], protocol['candidate_seed']),
                                 family in ('random_forest', 'mlp', 'gradient_boosting'))
    return result


def e02_registry(protocol):
    from .run_revision import default_registry
    return {**default_registry(protocol), **conventional_registry(protocol)}


class StableFootruleKNN(ClassifierMixin, BaseEstimator):
    """Exact full-cutoff ties by ascending source row ID, then lowest class ID.

    When sample_ids are omitted, input rows must already be in source-row order
    (the saved protocol splits are ascending). input_kind='positions' is for
    hidden representation probes; the default accepts item-order permutations.
    """
    def __init__(self, n_neighbors=5, weights='uniform', input_kind='orders', batch_size=128):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.input_kind = input_kind
        self.batch_size = batch_size

    def _positions(self, X):
        positions = inverse_positions(X)
        if self.input_kind == 'orders':
            return positions
        if self.input_kind == 'positions':
            return np.asarray(X, dtype=np.int64)
        raise ValueError('input_kind must be orders or positions')

    def fit(self, X, y, sample_ids=None):
        if self.n_neighbors < 1 or self.batch_size < 1 or self.weights not in ('uniform', 'distance'):
            raise ValueError('Positive neighbor/batch counts and uniform/distance weights required')
        start = time.perf_counter()
        positions = self._positions(X)
        self.encoding_seconds_ = time.perf_counter() - start
        check_consistent_length(positions, y)
        if not len(positions):
            raise ValueError('Training samples must be nonempty')
        ids = np.arange(len(positions)) if sample_ids is None else np.asarray(sample_ids)
        if ids.shape != (len(positions),) or len(np.unique(ids)) != len(ids) or ids.dtype.kind not in 'iu':
            raise ValueError('sample_ids must be unique integer source row IDs')
        start = time.perf_counter()
        self.classes_, self.labels_ = np.unique(y, return_inverse=True)
        self.source_order_ = np.argsort(ids, kind='stable')
        self.positions_ = positions[self.source_order_]
        self.classifier_fit_seconds_ = time.perf_counter() - start
        return self

    def _neighbors(self, X):
        check_is_fitted(self, 'positions_')
        start = time.perf_counter()
        query = self._positions(X)
        if query.shape[1] != self.positions_.shape[1]:
            raise ValueError('Permutation vocabulary changed')
        self.last_encoding_seconds_ = time.perf_counter() - start
        k = min(self.n_neighbors, len(self.positions_))
        for begin in range(0, len(query), self.batch_size):
            distances = cdist(query[begin:begin+self.batch_size], self.positions_, metric='cityblock')
            nearest = np.argsort(distances, axis=1, kind='stable')[:, :k]
            yield np.take_along_axis(distances, nearest, axis=1), self.source_order_[nearest]

    def kneighbors(self, X, return_distance=True):
        batches = list(self._neighbors(X))
        if not batches:
            k = min(self.n_neighbors, len(self.positions_))
            distances, indices = np.empty((0, k)), np.empty((0, k), dtype=int)
        else:
            distances, indices = map(np.concatenate, zip(*batches))
        return (distances, indices) if return_distance else indices

    def predict_neighbors(self, distances, indices):
        """Vote on an already stable, largest-k neighbor cache from this fit."""
        check_is_fitted(self, 'positions_')
        distances, indices = np.asarray(distances), np.asarray(indices)
        k = min(self.n_neighbors, len(self.positions_))
        if (distances.ndim != 2 or distances.shape != indices.shape or distances.shape[1] < k
                or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= len(self.labels_))
                or np.any(~np.isfinite(distances)) or np.any(distances < 0)):
            raise ValueError('Expected a valid stable neighbor cache with at least k neighbors')
        predictions = []
        for d, neighbors in zip(distances[:, :k], indices[:, :k]):
            if self.weights == 'uniform':
                weights = np.ones(len(d))
            elif np.any(d == 0):
                weights = (d == 0).astype(float)
            else:
                weights = 1. / d
            votes = np.bincount(self.labels_[neighbors], weights=weights, minlength=len(self.classes_))
            predictions.append(self.classes_[np.argmax(votes)])
        return np.asarray(predictions, dtype=self.classes_.dtype)

    def predict(self, X):
        predictions = [self.predict_neighbors(distances, indices) for distances, indices in self._neighbors(X)]
        return np.concatenate(predictions) if predictions else np.empty(0, dtype=self.classes_.dtype)


class BordaClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        start = time.perf_counter()
        positions = inverse_positions(X)
        self.encoding_seconds_ = time.perf_counter() - start
        check_consistent_length(positions, y)
        self.classes_, labels = np.unique(y, return_inverse=True)
        if not len(self.classes_):
            raise ValueError('Training samples must be nonempty')
        start = time.perf_counter()
        means = np.array([positions[labels == c].mean(axis=0) for c in range(len(self.classes_))])
        self.prototype_orders_ = score_order(means)
        self.prototype_positions_ = inverse_positions(self.prototype_orders_)
        self.classifier_fit_seconds_ = time.perf_counter() - start
        return self

    def predict(self, X):
        check_is_fitted(self, 'prototype_positions_')
        start = time.perf_counter()
        positions = inverse_positions(X)
        self.last_encoding_seconds_ = time.perf_counter() - start
        distances = cdist(positions, self.prototype_positions_, metric='cityblock')
        return self.classes_[np.argmin(distances, axis=1)]


def normalize_rows(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms != 0)


class OrderedPositionHDC(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Bipolar item × ordered-position binding; float64 normalized centroids.

    Codes are int8, bundle accumulation int32, normalization/centroids float64.
    Zero vectors remain zero. No N×V×D allocation or binarized bundle is used.
    fit_encoder/transform/fit_bundles support Task5b's explicit bundle cache.
    """
    def __init__(self, dimension=1024, seed=8129, item_seed=None, position_seed=None, batch_size=128):
        self.dimension = dimension
        self.seed = seed
        self.item_seed = item_seed
        self.position_seed = position_seed
        self.batch_size = batch_size

    def fit_encoder(self, X):
        inverse_positions(X)
        self.vocabulary_size_ = np.asarray(X).shape[1]
        if self.vocabulary_size_ < 2 or self.dimension < 2 or self.dimension % 2 or self.batch_size < 1:
            raise ValueError('HDC requires at least two items, positive batch size, and positive even dimension')
        start = time.perf_counter()
        self.item_seed_ = self.item_seed if self.item_seed is not None else derive_seed(self.seed, 'hdc_item', self.vocabulary_size_, self.dimension)
        self.position_seed_ = self.position_seed if self.position_seed is not None else derive_seed(self.seed, 'hdc_position', self.vocabulary_size_, self.dimension)
        item_rng = np.random.RandomState(self.item_seed_)
        position_rng = np.random.RandomState(self.position_seed_)
        self.item_keys_ = (2*item_rng.randint(0, 2, (self.vocabulary_size_, self.dimension))-1).astype(np.int8)
        base = (2*position_rng.randint(0, 2, self.dimension)-1).astype(np.int8)
        coordinates = position_rng.permutation(self.dimension)
        self.position_codes_ = np.tile(base, (self.vocabulary_size_, 1))
        for position in range(self.vocabulary_size_):
            flips = position*self.dimension // (2*(self.vocabulary_size_-1))
            self.position_codes_[position, coordinates[:flips]] *= -1
        self.encoder_setup_seconds_ = time.perf_counter() - start
        self.max_bundle_batch_rows_ = 0
        self.representation_metadata_ = {'dimension': self.dimension, 'vocabulary_size': self.vocabulary_size_,
            'item_seed': self.item_seed_, 'position_seed': self.position_seed_,
            'item_keys_hash': array_hash(self.item_keys_), 'position_codes_hash': array_hash(self.position_codes_),
            'code_dtype': 'int8', 'bundle_accumulator_dtype': 'int32', 'normalized_dtype': 'float64',
            'bundle_batch_size': self.batch_size}
        return self

    def _bundles(self, X):
        check_is_fitted(self, 'item_keys_')
        inverse_positions(X)
        orders = np.asarray(X, dtype=np.int64)
        if orders.shape[1] != self.vocabulary_size_:
            raise ValueError('HDC vocabulary changed')
        for begin in range(0, len(orders), self.batch_size):
            block = orders[begin:begin+self.batch_size]
            self.max_bundle_batch_rows_ = max(self.max_bundle_batch_rows_, len(block))
            bundle = np.zeros((len(block), self.dimension), dtype=np.int32)
            for position in range(self.vocabulary_size_):
                bundle += self.item_keys_[block[:, position]] * self.position_codes_[position]
            yield begin, normalize_rows(bundle)

    def transform(self, X):
        result = np.empty((len(X), self.dimension), dtype=np.float64)
        for begin, bundle in self._bundles(X):
            result[begin:begin+len(bundle)] = bundle
        return result

    def fit_bundles(self, bundles, y):
        bundles = np.asarray(bundles, dtype=np.float64)
        check_consistent_length(bundles, y)
        if bundles.ndim != 2 or bundles.shape[1] != self.dimension or not np.isfinite(bundles).all():
            raise ValueError('Expected finite HDC bundles of declared dimension')
        self.classes_, labels = np.unique(y, return_inverse=True)
        if not len(self.classes_):
            raise ValueError('Training samples must be nonempty')
        normalized = normalize_rows(bundles)
        means = np.array([normalized[labels == c].mean(axis=0) for c in range(len(self.classes_))])
        self.class_prototypes_ = normalize_rows(means)
        return self

    def fit(self, X, y):
        check_consistent_length(X, y)
        self.fit_encoder(X)
        self.classes_, labels = np.unique(y, return_inverse=True)
        if not len(self.classes_):
            raise ValueError('Training samples must be nonempty')
        sums = np.zeros((len(self.classes_), self.dimension), dtype=np.float64)
        encoding_seconds = self.encoder_setup_seconds_
        start = time.perf_counter()
        iterator = iter(self._bundles(X))
        while True:
            encode_start = time.perf_counter()
            try:
                begin, bundle = next(iterator)
            except StopIteration:
                break
            encoding_seconds += time.perf_counter() - encode_start
            batch_labels = labels[begin:begin+len(bundle)]
            for c in np.unique(batch_labels):
                sums[c] += bundle[batch_labels == c].sum(axis=0)
        self.class_prototypes_ = normalize_rows(sums / np.bincount(labels)[:, None])
        self.classifier_fit_seconds_ = max(0., time.perf_counter()-start-(encoding_seconds-self.encoder_setup_seconds_))
        self.encoding_seconds_ = encoding_seconds
        self.inference_array_bytes_ = self.item_keys_.nbytes+self.position_codes_.nbytes+self.class_prototypes_.nbytes
        return self

    def predict_bundles(self, bundles):
        check_is_fitted(self, 'class_prototypes_')
        bundles = np.asarray(bundles, dtype=np.float64)
        if bundles.ndim != 2 or bundles.shape[1] != self.dimension or not np.isfinite(bundles).all():
            raise ValueError('Expected finite HDC bundles of declared dimension')
        return self.classes_[np.argmax(normalize_rows(bundles) @ self.class_prototypes_.T, axis=1)]

    def predict(self, X):
        check_is_fitted(self, 'class_prototypes_')
        predictions = []
        encode_seconds = 0.
        iterator = iter(self._bundles(X))
        while True:
            start = time.perf_counter()
            try:
                _, bundle = next(iterator)
            except StopIteration:
                break
            encode_seconds += time.perf_counter()-start
            predictions.append(self.classes_[np.argmax(bundle @ self.class_prototypes_.T, axis=1)])
        self.last_encoding_seconds_ = encode_seconds
        return np.concatenate(predictions) if predictions else np.empty(0, dtype=self.classes_.dtype)


class NativeArrowFlow(ArrowFlowEstimator):
    """Fixed native ten-item vocabulary; numerical preprocessing is bypassed."""
    def __init__(self, widths=(64,32), iterations=200, learning_rate=.5, validation_ratio=0., p_correct=.01, seed=8129):
        super().__init__(embed_dim=10, widths=widths, iterations=iterations, learning_rate=learning_rate,
                         validation_ratio=validation_ratio, p_correct=p_correct, seed=seed)

    def fit(self, X, y):
        self.encoding_seconds_ = 0.
        return self.fit_orders(X, y)

    def predict(self, X):
        self.last_encoding_seconds_ = 0.
        return self.predict_orders(X)

    def transform(self, X):
        return self.transform_orders(X)


def ordinal_factory(estimator_class, config, seed, stochastic=False):
    return estimator_class(**config, **({'seed': seed} if stochastic else {}))


# v3 native grid: 36 combinations, sampled to the shared candidate budget by candidate_grid
# with the protocol's candidate seed, exactly as the conventional and footrule families are.
NATIVE_ARROWFLOW_GRID_V3 = {'widths': [[64], [128], [64, 32]], 'learning_rate': [.05, .1, .2],
                            'validation_ratio': [0, .1], 'p_correct': [.01, .1]}


def native_registry(protocol):
    from .run_revision import dummy_factory
    registry = conventional_registry(protocol, native=True)
    registry['native_dummy'] = ModelSpec('native_dummy', dummy_factory, [{}], False)
    registry['native_footrule_knn'] = ModelSpec('native_footrule_knn', partial(ordinal_factory, StableFootruleKNN),
        candidate_grid({'n_neighbors': [1,3,5,11,21], 'weights': ['uniform','distance']},
                       protocol['candidate_budget'], protocol['candidate_seed']), False)
    registry['native_borda'] = ModelSpec('native_borda', partial(ordinal_factory, BordaClassifier), [{}], False)
    registry['native_hdc'] = ModelSpec('native_hdc', partial(ordinal_factory, OrderedPositionHDC, stochastic=True),
                                     [{'dimension': 1024}, {'dimension': 10000}], True)
    grid = protocol.get('native_arrowflow_grid')
    if grid == 'v3':
        registry['native_arrowflow'] = ModelSpec('native_arrowflow', partial(ordinal_factory, NativeArrowFlow, stochastic=True),
            candidate_grid(NATIVE_ARROWFLOW_GRID_V3, protocol['candidate_budget'], protocol['candidate_seed']), True)
    elif grid is None:
        # 2026-09-11 design: three fixed architectures, one untuned candidate each.
        for widths in ([64], [64,32], [230]):
            name = 'native_arrowflow_' + '_'.join(map(str, widths))
            registry[name] = ModelSpec(name, partial(ordinal_factory, NativeArrowFlow, stochastic=True),
                                       [{'widths': widths, 'iterations': 200}], True)
    else:
        raise ValueError(f'Unknown native_arrowflow_grid {grid!r}; expected "v3" or an absent key')
    return registry
