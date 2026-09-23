"""MAKE v2 full-ranking boundaries; positions are zero based."""
import numpy as np
from sklearn.impute import SimpleImputer


def numeric_ids(items):
    """Native integer IDs or the numeric suffix of generated filter IDs."""
    return np.asarray([int(str(item).rsplit('_', 1)[-1]) for item in items])


def validate_permutation(order, vocabulary):
    if len(vocabulary) == 0 or len(order) != len(vocabulary) or len(set(order)) != len(order) or set(order) != set(vocabulary):
        raise ValueError('Expected a complete permutation of the layer vocabulary')
    ids = numeric_ids(order)
    if len(set(ids)) != len(ids):
        raise ValueError('Permutation items must have distinct canonical numeric IDs')


def score_order(scores, items=None):
    """Ascending score, then ascending numeric ID (last axis)."""
    values = np.asarray(scores)
    if not np.isfinite(values).all():
        raise ValueError('Ranking scores must be finite; impute numeric features first')
    if items is None:
        return np.argsort(values, axis=-1, kind='stable')
    return np.lexsort((numeric_ids(items), values))


def inverse_positions(orders):
    """Convert rows of coordinate-ID orders into position vectors for Manhattan kNN."""
    values = np.asarray(orders)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('Expected a matrix of complete permutations')
    vocabulary = np.arange(values.shape[1])
    if not np.all(np.sort(values, axis=1) == vocabulary):
        raise ValueError('Expected complete permutations of coordinate IDs 0..d-1')
    return np.argsort(values, axis=1, kind='stable')


def impute_numeric(train, test):
    """Training-column means for NaNs; all-missing training columns use zero."""
    train, test = np.asarray(train, dtype=float), np.asarray(test, dtype=float)
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError('Numeric features must be finite or NaN for explicit imputation')
    imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
    return imputer.fit_transform(train), imputer.transform(test)
