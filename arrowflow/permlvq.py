"""Permutation LVQ: labeled permutation prototypes trained by rank aggregation (v3 laboratory, B2).

A PermutationLVQ layer holds M = prototypes_per_class x C prototype permutations over a vocabulary of V
items, stored as inverse positions (row m, column v = position of item v in prototype m). Distances are the
footrule (cityblock on positions). One batch step is a labeled version of ArrowFlow's filter update:

* for every sample of the batch, `same` is the nearest prototype with the sample's label and `other` the
  nearest with a different label; the sample is accepted when the nearest prototype overall has the wrong
  label, or otherwise with probability p_correct (one uniform draw per correctly decided sample);
* an accepted sample casts a vote of weight learning_rate for `same` toward the sample's ranking and, with
  repulsion, a vote of weight -learning_rate for `other` (a negative weight votes for the reversed ranking);
* votes accumulate in one V x V matrix per prototype, initialised to prior_weight times the identity in the
  prototype's current order (row p is the item at position p; the identity is the prior "stay in place",
  weight 1 by default — exactly ArrowFlow's accumulator): a vote of weight a toward the target positions t
  adds |a| at [p, t(item_p)] for every row p — exactly ArrowFlow's accumulate_perm_inplace;
* after the batch every voted-on prototype is reordered by the aggregation rule and its matrix is reset:
  'borda' sorts the rows by their weighted mean column (ArrowFlow's compute_adj_list_with_permutation; the
  prior enters the mean with weight prior_weight), ties by ascending item ID; 'footrule_median' solves the
  minimum-cost assignment of rows to positions under cost[p, q] = sum_c A[p, c] |c - q| = sum_t w_t
  |pos_t(item_p) - q| + prior_weight |p - q| (the exact footrule median of the weighted profile plus the
  weighted identity prior; Dwork et al. 2001), so both rules read the same accumulator. Under the median an
  item moves only when the vote mass pulling it exceeds prior_weight; with prior_weight = learning_rate one
  consistent vote ties the prior and two consistent votes move the item.

transform(positions) ranks the prototype IDs by footrule distance and returns that ranking as inverse
positions, the input of a next layer whose vocabulary is this layer's prototype set. predict offers three
readouts on the ranked prototype labels: 'nearest', 'plurality' (majority among the k nearest, lowest label
on ties) and 'borda' (score k - j for the j-th nearest, lowest label on ties).

PermutationLVQClassifier stacks layers greedily, each fitted on the previous layer's frozen transform, and
reads out from the last layer; fit_orders/predict_orders/transform_orders take rankings of item IDs.
"""
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from .ranking import inverse_positions, score_order

AGGREGATIONS = ('borda', 'footrule_median')
READOUTS = ('nearest', 'plurality', 'borda')


# ----------------------------------------------------------------------------- aggregation rules

def assignment_positions(cost):
    """Positions of a minimum-cost assignment of items (rows) to positions (columns): positions[v] = q.
    Ties between equal-cost assignments resolve as scipy's linear_sum_assignment does (deterministic)."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or cost.shape[0] == 0 or not np.isfinite(cost).all():
        raise ValueError('Expected a finite square cost matrix')
    rows, cols = linear_sum_assignment(cost)
    positions = np.empty(cost.shape[0], dtype=np.int64)
    positions[rows] = cols
    return positions


def footrule_median(profile, weights, prior=None, prior_weight=1.0):
    """Exact footrule median of a weighted profile of rankings, as inverse positions.

    profile: (T, V) inverse positions of T rankings; weights: (T,) nonnegative vote weights; prior: optional
    (V,) inverse positions of a ranking that votes with weight prior_weight (the layer's identity prior).
    Minimises sum_t w_t * footrule(profile_t, result) (+ prior_weight * footrule(prior, result)) as the
    minimum-cost assignment on cost[v, q] = sum_t w_t |profile[t, v] - q| (+ prior_weight |prior[v] - q|)
    (Dwork et al. 2001)."""
    profile = np.asarray(profile)
    weights = np.asarray(weights, dtype=np.float64)
    if profile.ndim != 2 or weights.shape != (profile.shape[0],) or (weights < 0).any() or not np.isfinite(weights).all():
        raise ValueError('Expected one finite nonnegative weight per ranking of the profile')
    if not np.isfinite(prior_weight) or prior_weight <= 0:          # the same rule as the layer
        raise ValueError('prior_weight must be a positive finite number')
    inverse_positions(profile)               # complete permutations of 0..V-1
    V = profile.shape[1]
    grid = np.arange(V)
    cost = np.einsum('t,tvq->vq', weights, np.abs(profile[:, :, None] - grid[None, None, :]))
    if prior is not None:
        prior = np.asarray(prior)
        inverse_positions(prior[None, :])
        if prior.shape != (V,):
            raise ValueError('The prior must be a ranking of the same vocabulary')
        cost += float(prior_weight) * np.abs(prior[:, None] - grid[None, :])
    return assignment_positions(cost)


def _borda_order(accumulator, order):
    """ArrowFlow's row rule: ascending weighted mean column, ties by ascending item ID (score_order)."""
    V = accumulator.shape[0]
    means = accumulator @ np.arange(V) / accumulator.sum(axis=1)
    return order[np.lexsort((order, means))]


def _footrule_median_order(accumulator, order, column_distance):
    """Minimum-cost assignment of the rows (items in the current order) to positions."""
    positions = assignment_positions(accumulator @ column_distance)
    new_order = np.empty_like(order)
    new_order[positions] = order
    return new_order


# ----------------------------------------------------------------------------- the layer

class PermutationLVQ(ClassifierMixin, BaseEstimator):
    """One permutation LVQ layer; see the module docstring for the update semantics."""
    def __init__(self, prototypes_per_class=8, iterations=200, batch_size=32, learning_rate=.1, p_correct=.01,
                 repulsion=True, aggregation='borda', classes=None, seed=0, prior_weight=1.0):
        self.prototypes_per_class = prototypes_per_class
        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.p_correct = p_correct
        self.repulsion = repulsion
        self.aggregation = aggregation
        self.classes = classes
        self.seed = seed
        self.prior_weight = prior_weight

    # ---------------------------------------------------------------- validation
    @staticmethod
    def _positions(X):
        positions = np.asarray(X)
        inverse_positions(positions)             # validates complete permutations of 0..V-1
        return positions.astype(np.int64)

    @staticmethod
    def _positive_int(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
        return int(value)

    def _check_params(self):
        self._positive_int(self.prototypes_per_class, 'prototypes_per_class')
        self._positive_int(self.iterations, 'iterations', minimum=0)
        self._positive_int(self.batch_size, 'batch_size')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be a positive finite number')
        if not np.isfinite(self.p_correct) or not 0 <= self.p_correct <= 1:
            raise ValueError('p_correct must lie in [0, 1]')
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
        if not np.isfinite(self.prior_weight) or self.prior_weight <= 0:
            raise ValueError('prior_weight must be a positive finite number')

    def _query(self, X):
        check_is_fitted(self, 'prototypes_')
        positions = self._positions(X)
        if positions.shape[1] != self.vocabulary_size_:
            raise ValueError('Permutation vocabulary changed')
        return positions

    # ---------------------------------------------------------------- accumulator
    def _vote(self, m, target, weight):
        """A vote of weight `weight` for prototype m toward the target positions (reversed when negative):
        adds |weight| at [p, pos_target(item_p)] for every row p of the prototype's accumulator."""
        target = np.asarray(target)
        if weight < 0:
            target = self.vocabulary_size_ - 1 - target
        self.accumulators_[m][self._rows, target[self.prototype_orders_[m]]] += abs(float(weight))

    def _reset(self, m):
        accumulator = self.accumulators_[m]
        accumulator.fill(0.)
        np.fill_diagonal(accumulator, float(self.prior_weight))

    def _reorder(self, m):
        order = self.prototype_orders_[m]
        if self.aggregation == 'borda':
            new_order = _borda_order(self.accumulators_[m], order)
        else:
            new_order = _footrule_median_order(self.accumulators_[m], order, self._column_distance)
        self.prototype_orders_[m] = new_order
        self.prototypes_[m, new_order] = self._rows
        self._reset(m)

    # ---------------------------------------------------------------- fitting
    def fit(self, X, y):
        self._check_params()
        positions = self._positions(X)
        y = np.asarray(y)
        check_consistent_length(positions, y)
        if self.classes is None:
            self.classes_ = np.unique(y)
        else:
            self.classes_ = np.unique(np.asarray(self.classes))
            if len(self.classes_) != len(self.classes):
                raise ValueError('classes must be distinct')
        if not len(self.classes_):
            raise ValueError('Training samples must be nonempty')
        class_index = np.searchsorted(self.classes_, y)
        if (class_index >= len(self.classes_)).any() or (self.classes_[np.minimum(class_index, len(self.classes_) - 1)] != y).any():
            raise ValueError('Training labels outside the class set')
        start = time.perf_counter()
        N, V = positions.shape
        C = len(self.classes_)
        M = int(self.prototypes_per_class) * C
        rng = np.random.RandomState(self.seed)
        self.random_state_ = rng
        self.vocabulary_size_ = V
        self._rows = np.arange(V)
        self._column_distance = np.abs(self._rows[:, None] - self._rows[None, :]).astype(np.float64)
        self.labels_ = self.classes_[np.arange(M) % C]
        self.prototype_orders_ = np.asarray([rng.permutation(V) for _ in range(M)], dtype=np.int64)
        self.prototypes_ = inverse_positions(self.prototype_orders_)
        self.accumulators_ = [float(self.prior_weight) * np.eye(V) for _ in range(M)]
        same_label = self.labels_[None, :] == y[:, None]          # (N, M)
        batch_size = min(int(self.batch_size), N)
        self.vote_count_ = 0
        for _ in range(int(self.iterations)):
            batch = rng.choice(N, size=batch_size, replace=False)
            distances = cdist(positions[batch], self.prototypes_, metric='cityblock')
            nearest = np.argmin(distances, axis=1)
            mask = same_label[batch]
            same = np.argmin(np.where(mask, distances, np.inf), axis=1)
            other_distances = np.where(mask, np.inf, distances)
            other = np.argmin(other_distances, axis=1)
            has_other = np.isfinite(other_distances[np.arange(batch_size), other])
            correct = self.labels_[nearest] == y[batch]
            accepted = ~correct
            accepted[correct] = rng.rand(int(correct.sum())) < self.p_correct
            touched = set()
            for i in np.flatnonzero(accepted):
                sample = positions[batch[i]]
                self._vote(same[i], sample, self.learning_rate)
                touched.add(int(same[i]))
                if self.repulsion and has_other[i]:
                    self._vote(other[i], sample, -self.learning_rate)
                    touched.add(int(other[i]))
            self.vote_count_ += int(accepted.sum())
            for m in sorted(touched):
                self._reorder(m)
        self.fit_seconds_ = time.perf_counter() - start
        return self

    # ---------------------------------------------------------------- inference
    def _ranking(self, positions):
        return score_order(cdist(positions, self.prototypes_, metric='cityblock'))

    def transform(self, X):
        """Ranking of the prototype IDs by ascending footrule distance (ties by lowest ID) as inverse positions."""
        return inverse_positions(self._ranking(self._query(X)))

    def predict(self, X, readout='plurality', k=5):
        if readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}')
        positions = self._query(X)
        ranking = self._ranking(positions)
        if readout == 'nearest':
            return self.labels_[ranking[:, 0]]
        k = min(self._positive_int(k, 'k'), ranking.shape[1])
        top = np.searchsorted(self.classes_, self.labels_[ranking[:, :k]])        # class indices (N, k)
        weights = np.ones(k) if readout == 'plurality' else np.arange(k, 0, -1).astype(np.float64)
        scores = np.zeros((len(positions), len(self.classes_)))
        np.add.at(scores, (np.repeat(np.arange(len(positions)), k), top.ravel()), np.tile(weights, len(positions)))
        return self.classes_[np.argmax(scores, axis=1)]                        # argmax: lowest class on ties


# ----------------------------------------------------------------------------- the stack

LAYER_KEYS = ('prototypes_per_class', 'iterations', 'batch_size', 'learning_rate', 'p_correct', 'repulsion', 'aggregation',
              'prior_weight')


class PermutationLVQClassifier(ClassifierMixin, BaseEstimator):
    """Greedy stack of PermutationLVQ layers with a readout on the last layer.

    layers: one dict per layer; keys are PermutationLVQ parameters other than classes and seed (a layer's
    dict overrides the shared values). Layer i is fitted on layer i-1's frozen transform; every layer shares
    the class set of y. Layer seeds are drawn from RandomState(seed)."""
    def __init__(self, layers=({'prototypes_per_class': 8},), readout='plurality', k=5, iterations=200, batch_size=32,
                 learning_rate=.1, p_correct=.01, repulsion=True, aggregation='borda', seed=0, prior_weight=1.0):
        self.layers = layers
        self.readout = readout
        self.k = k
        self.iterations = iterations
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.p_correct = p_correct
        self.repulsion = repulsion
        self.aggregation = aggregation
        self.seed = seed
        self.prior_weight = prior_weight

    def _shared(self):
        return {key: getattr(self, key) for key in LAYER_KEYS if key != 'prototypes_per_class'}

    def fit(self, X, y):
        layers = list(self.layers)
        if not layers or not all(isinstance(spec, dict) and set(spec) <= set(LAYER_KEYS) for spec in layers):
            raise ValueError(f'layers must be a nonempty sequence of dicts with keys among {LAYER_KEYS}')
        if self.readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}')
        positions = PermutationLVQ._positions(X)
        y = np.asarray(y)
        check_consistent_length(positions, y)
        self.classes_ = np.unique(y)
        rng = np.random.RandomState(self.seed)
        seeds = rng.randint(0, 2**31 - 1, size=len(layers))
        start = time.perf_counter()
        self.layers_ = []
        current = positions
        for spec, seed in zip(layers, seeds):
            layer = PermutationLVQ(**{**self._shared(), **spec}, classes=tuple(self.classes_.tolist()), seed=int(seed))
            layer.fit(current, y)
            self.layers_.append(layer)
            if len(self.layers_) < len(layers):
                current = layer.transform(current)
        self.training_seconds_ = time.perf_counter() - start
        return self

    def _hidden(self, X):
        check_is_fitted(self, 'layers_')
        current = PermutationLVQ._positions(X)
        for layer in self.layers_[:-1]:
            current = layer.transform(current)
        return current

    def transform(self, X):
        return self.layers_[-1].transform(self._hidden(X))

    def predict(self, X):
        return self.layers_[-1].predict(self._hidden(X), readout=self.readout, k=self.k)

    def fit_orders(self, orders, y):
        return self.fit(inverse_positions(orders), y)

    def transform_orders(self, orders):
        return self.transform(inverse_positions(orders))

    def predict_orders(self, orders):
        return self.predict(inverse_positions(orders))
