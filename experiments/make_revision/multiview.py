"""Full ArrowFlow method: several encoded views, one network per view, vote aggregation."""
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from .models import ArrowFlowEstimator, OrdinalEncoder
from .comparisons import derive_seed, StableFootruleKNN
from .evaluation import candidate_grid, config_id
from .secondary_studies import majority

CYCLE = ('target_aware', 'random', 'calibrated')


def view_strategy(strategy, v):
    return CYCLE[v % 3] if strategy == 'diverse' else strategy


def borda_aggregate(rankings, classes):
    classes = np.asarray(classes)
    index = {c: i for i, c in enumerate(classes.tolist())}
    n = len(rankings[0]); C = len(classes)
    scores = np.zeros((n, C))
    for ranking in rankings:
        for position in range(C):
            col = np.vectorize(index.get)(ranking[:, position])
            scores[np.arange(n), col] += C - 1 - position
    return classes[np.argmax(scores, axis=1)]          # argmax returns the lowest index on ties


class MultiViewArrowFlow(ClassifierMixin, BaseEstimator):
    def __init__(self, n_views=7, strategy='diverse', embed_dim=32, degree=2, widths=(128,),
                 learning_rate=.1, iterations=200, batch_size=32, validation_ratio=.1,
                 augment=False, n_augmentations=1, max_swaps=2, aggregation='majority',
                 lda_ratio=.3, p_correct=.01, ratio_data_backprop=.5,
                 motion_normalization_mult=.125, last_layer_update=True, seed=8129):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.views_ = []
        encoding = training = 0.
        for v in range(self.n_views):
            seed_v = derive_seed(self.seed, 'view', v)
            start = time.perf_counter()
            enc = OrdinalEncoder(view_strategy(self.strategy, v), self.embed_dim, self.degree,
                                 self.lda_ratio, seed_v).fit(X, y)
            orders = enc.transform(X)
            encoding += time.perf_counter() - start
            net = ArrowFlowEstimator(embed_dim=self.embed_dim, degree=self.degree, widths=self.widths,
                                     iterations=self.iterations, learning_rate=self.learning_rate,
                                     batch_size=self.batch_size, last_layer_update=self.last_layer_update,
                                     ratio_data_backprop=self.ratio_data_backprop,
                                     motion_normalization_mult=self.motion_normalization_mult,
                                     p_correct=self.p_correct, seed=seed_v,
                                     validation_ratio=self.validation_ratio, augment=self.augment,
                                     n_augmentations=self.n_augmentations, max_swaps=self.max_swaps)
            net.fit_orders(orders, y)
            training += net.training_seconds_
            self.views_.append((enc, net))
            self._fit_view_readout(enc, net, orders, y, seed_v)
        self.encoding_seconds_ = encoding
        self.training_seconds_ = training
        return self

    def _fit_view_readout(self, enc, net, orders, y, seed_v):
        """Hook for a subclass that replaces the network's output rule by a readout on the view's hidden
        ranking; the output rule itself needs nothing beyond the trained network."""

    def predict_views(self, X):
        check_is_fitted(self, 'views_')
        start = time.perf_counter()
        orders = [enc.transform(X) for enc, _ in self.views_]
        self.last_encoding_seconds_ = time.perf_counter() - start
        return [net.predict_orders(o) for (enc, net), o in zip(self.views_, orders)], orders

    def predict(self, X):
        predictions, orders = self.predict_views(X)
        if self.aggregation == 'majority':
            return majority(predictions)
        if self.aggregation == 'borda':
            rankings = [net.predict_class_ranking(o) for (enc, net), o in zip(self.views_, orders)]
            return borda_aggregate(rankings, self.classes_)
        raise ValueError('aggregation must be majority or borda')


# ----------------------------------------------------------------------------- kNN readout on the trained hidden ranking

KNN_READOUT_GRID = {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance']}
KNN_SELECTION_FOLDS = 3


def knn_readout_candidates():
    """The ten readout settings of the inner-fold laboratory's knn_hidden grid, in canonical order (config_id)."""
    return candidate_grid(KNN_READOUT_GRID, budget=10)


def select_knn_readout(hidden, y, *, seed, candidates=None, folds=KNN_SELECTION_FOLDS):
    """Choose n_neighbors and weights for a footrule kNN on `hidden`, the inverse positions of the final hidden
    ranking of the rows given (a training partition), by mean accuracy over stratified splits of those rows only:
    StratifiedKFold(folds, shuffle=True, random_state=seed), at most `folds` splits and fewer when the smallest
    class has fewer rows. One neighbour cache per split at the largest k serves every candidate
    (StableFootruleKNN.predict_neighbors, the laboratory's procedure); ties go to the lowest canonical config_id."""
    hidden, y = np.asarray(hidden), np.asarray(y)
    candidates = knn_readout_candidates() if candidates is None else list(candidates)
    folds = min(int(folds), int(np.unique(y, return_counts=True)[1].min()))
    if folds < 2:
        raise ValueError('Readout selection needs at least two rows per class in the training partition')
    largest = max(c['n_neighbors'] for c in candidates)
    scores = {config_id(c): [] for c in candidates}
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        cache = StableFootruleKNN(n_neighbors=largest, input_kind='positions').fit(hidden[a], y[a], sample_ids=a)
        distances, indices = cache.kneighbors(hidden[b])
        for config in candidates:
            cache.n_neighbors, cache.weights = config['n_neighbors'], config['weights']
            scores[config_id(config)].append(float(np.mean(cache.predict_neighbors(distances, indices) == y[b])))
    means = {cid: float(np.mean(s)) for cid, s in scores.items()}
    negative, cid, config = min((-means[config_id(c)], config_id(c), c) for c in candidates)
    return {'config': config, 'config_id': cid, 'inner_score': -negative, 'folds': folds, 'candidate_scores': means}


class MultiViewArrowFlowKNN(MultiViewArrowFlow):
    """MultiViewArrowFlow whose per-view prediction is a footrule kNN on the view's final hidden ranking.

    The networks are trained exactly as in MultiViewArrowFlow (same derived view seeds, checkpoint and
    augmentation; each view reseeds from its own seed), so a view's network equals the output-rule model's
    for the same parameters, seed and training rows. After training, the view's training rows pass through
    the trained hidden layers (ArrowFlowEstimator.transform_orders) and a StableFootruleKNN on those
    positions replaces the output layer: n_neighbors and weights are chosen by select_knn_readout on
    stratified splits of the training rows (random_state derived from the view seed), then the readout is
    refitted on all of them. The per-view votes are combined by majority; Borda is not defined here.
    """
    def fit(self, X, y):
        self.readouts_, self.readout_selections_ = [], []
        self.readout_seconds_ = 0.
        return super().fit(X, y)

    def _fit_view_readout(self, enc, net, orders, y, seed_v):
        start = time.perf_counter()
        hidden = net.transform_orders(orders)
        selection = select_knn_readout(hidden, y, seed=derive_seed(seed_v, 'readout_selection'))
        readout = StableFootruleKNN(**selection['config'], input_kind='positions').fit(hidden, y)
        self.readouts_.append(readout)
        self.readout_selections_.append(selection)
        self.readout_seconds_ += time.perf_counter() - start

    def predict_views(self, X):
        check_is_fitted(self, 'readouts_')
        start = time.perf_counter()
        orders = [enc.transform(X) for enc, _ in self.views_]
        self.last_encoding_seconds_ = time.perf_counter() - start
        predictions = [readout.predict(net.transform_orders(o))
                       for (enc, net), readout, o in zip(self.views_, self.readouts_, orders)]
        return predictions, orders

    def predict(self, X):
        if self.aggregation != 'majority':
            raise ValueError('MultiViewArrowFlowKNN combines the per-view kNN votes by majority only')
        return majority(self.predict_views(X)[0])


class MultiViewFootruleKNN(ClassifierMixin, BaseEstimator):
    """Same per-view encoders as MultiViewArrowFlow (same derived seeds), kNN per view, majority vote."""
    def __init__(self, n_views=7, strategy='diverse', embed_dim=32, degree=2, n_neighbors=5,
                 weights='uniform', lda_ratio=.3, seed=8129):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)

    def fit(self, X, y, sample_ids=None):
        """sample_ids: global source row IDs of X, forwarded to every per-view kNN so that
        neighbour cutoff ties are broken by source row (StableFootruleKNN semantics)."""
        self.classes_ = np.unique(y)
        self.views_ = []
        for v in range(self.n_views):
            seed_v = derive_seed(self.seed, 'view', v)
            enc = OrdinalEncoder(view_strategy(self.strategy, v), self.embed_dim, self.degree,
                                 self.lda_ratio, seed_v).fit(X, y)
            knn = StableFootruleKNN(n_neighbors=self.n_neighbors, weights=self.weights).fit(
                enc.transform(X), y, sample_ids=sample_ids)
            self.views_.append((enc, knn))
        return self

    def predict(self, X):
        check_is_fitted(self, 'views_')
        return majority([knn.predict(enc.transform(X)) for enc, knn in self.views_])
