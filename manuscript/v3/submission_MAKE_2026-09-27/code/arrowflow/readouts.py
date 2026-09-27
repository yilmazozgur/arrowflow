"""Readouts on hidden position vectors: the k-prototype Borda classifier (v3 laboratory, B1)."""
import time
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from .ranking import inverse_positions, score_order


class KPrototypeBorda(ClassifierMixin, BaseEstimator):
    """k Borda prototypes per class on inverse-position vectors; nearest prototype by footrule.

    fit(positions, y) clusters every class's position vectors into k groups with seeded k-means
    (Euclidean on positions, n_init=4) and takes each group's Borda centroid: the ranking that sorts
    the items by ascending mean position (score_order), stored as its inverse positions.
    predict(positions) returns the label of the nearest prototype under the footrule (cityblock)
    distance; ties resolve to the lowest prototype index (classes in sorted order, groups in
    k-means label order).

    prototypes_ has shape (k*C, V) when every class has at least k training rows; a class with
    fewer rows contributes one prototype per row, and a k-means group left empty contributes none.
    """
    def __init__(self, k=4, seed=0):
        self.k = k
        self.seed = seed

    @staticmethod
    def _positions(X):
        positions = np.asarray(X)
        inverse_positions(positions)           # validates complete permutations of 0..V-1
        return positions.astype(np.int64)

    def fit(self, X, y):
        if isinstance(self.k, bool) or not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError('k must be a positive integer')
        positions = self._positions(X)
        check_consistent_length(positions, y)
        self.encoding_seconds_ = 0.
        self.classes_, labels = np.unique(y, return_inverse=True)
        if not len(self.classes_):
            raise ValueError('Training samples must be nonempty')
        start = time.perf_counter()
        orders, owners = [], []
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            for c in range(len(self.classes_)):
                rows = positions[labels == c]
                n_clusters = min(int(self.k), len(rows))
                if n_clusters > 1:
                    groups = KMeans(n_clusters=n_clusters, n_init=4, random_state=self.seed).fit_predict(
                        rows.astype(np.float64))
                else:
                    groups = np.zeros(len(rows), dtype=int)
                for g in range(n_clusters):
                    members = rows[groups == g]
                    if len(members):
                        orders.append(score_order(members.mean(axis=0)))
                        owners.append(c)
        self.fit_warnings_ = [{'category': w.category.__name__, 'message': str(w.message)} for w in captured]
        self.prototype_orders_ = np.asarray(orders, dtype=np.int64)
        self.prototypes_ = inverse_positions(self.prototype_orders_)
        self.prototype_labels_ = self.classes_[np.asarray(owners, dtype=int)]
        self.vocabulary_size_ = positions.shape[1]
        self.classifier_fit_seconds_ = time.perf_counter() - start
        return self

    def predict(self, X):
        check_is_fitted(self, 'prototypes_')
        positions = self._positions(X)
        if positions.shape[1] != self.vocabulary_size_:
            raise ValueError('Permutation vocabulary changed')
        self.last_encoding_seconds_ = 0.
        distances = cdist(positions, self.prototypes_, metric='cityblock')
        return self.prototype_labels_[np.argmin(distances, axis=1)]
