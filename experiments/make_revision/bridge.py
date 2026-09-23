"""Bridge benchmark: the full ArrowFlow method (7 views, checkpoint, adaptive encoding) under nested CV."""
from itertools import product
from math import comb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from .evaluation import ModelSpec
from .comparisons import conventional_registry
from .multiview import KNN_READOUT_GRID, KNN_SELECTION_FOLDS, MultiViewArrowFlow, MultiViewArrowFlowKNN

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core.
SOURCE_MODULES = ['experiments.make_revision.multiview', 'experiments.make_revision.comparisons',
                  'experiments.make_revision.datasets', 'experiments.make_revision.secondary_studies']

FIXED = dict(n_views=7, strategy='diverse', iterations=200, batch_size=32, validation_ratio=.1,
             aggregation='majority')


def adaptive_defaults(n_features):
    if n_features <= 10:
        return {'embed_dim': 16, 'degree': 3, 'augment': True}
    if n_features <= 30:
        return {'embed_dim': 32, 'degree': 2, 'augment': True}
    return {'embed_dim': 64, 'degree': 1, 'augment': False}


def resolve(config, n_features, n_train, column_cap=10_000):
    base = adaptive_defaults(n_features)
    embed_dim = int(np.clip(round(base['embed_dim'] * config['embed_scale']), 8, 128))
    degree = max(1, base['degree'] + config['degree_offset'])
    while degree > 1 and comb(n_features + degree, degree) > column_cap:
        degree -= 1
    return {'embed_dim': embed_dim, 'degree': degree, 'augment': bool(base['augment'] and n_train >= 150)}


def bridge_candidates():
    # 16 candidates: embed_scale 0.5 was dropped after the training-only pilot (protocol resource_decision).
    grid = product(([128], [64, 128]), (.1, .2), (1, 2), (0, -1))
    return [{**FIXED, 'widths': w, 'learning_rate': lr, 'embed_scale': s, 'degree_offset': d}
            for w, lr, s, d in grid]


class AdaptiveMultiView(ClassifierMixin, BaseEstimator):
    """Resolves embed_dim/degree/augment from the training partition's shape, then trains MultiViewArrowFlow."""
    model_class = MultiViewArrowFlow

    def __init__(self, config=None, seed=8129):
        self.config = config
        self.seed = seed

    def fit(self, X, y):
        cfg = dict(self.config)
        self.resolved_ = resolve(cfg, X.shape[1], len(y))
        params = {k: v for k, v in cfg.items() if k not in ('embed_scale', 'degree_offset')}
        self.model_ = self.model_class(**params, **self.resolved_, seed=self.seed).fit(X, y)
        self.encoder_ = self.model_.views_[0][0]
        self.classes_ = self.model_.classes_
        self.encoding_seconds_ = self.model_.encoding_seconds_
        self.training_seconds_ = self.model_.training_seconds_
        return self

    def predict(self, X):
        out = self.model_.predict(X)
        self.last_encoding_seconds_ = self.model_.last_encoding_seconds_
        return out


class AdaptiveMultiViewKNN(AdaptiveMultiView):
    """The same resolution from the training partition's shape; the estimator is MultiViewArrowFlowKNN, whose
    per-view prediction is a footrule kNN on the view's trained hidden ranking (the readout adopted in the
    inner-fold laboratory). The fit record carries every view's chosen readout setting and the selection score of
    every readout setting, so the nested readout choice can be re-derived from the fit log."""
    model_class = MultiViewArrowFlowKNN

    def fit(self, X, y):
        super().fit(X, y)
        self.readout_seconds_ = self.model_.readout_seconds_
        self.readout_selections_ = self.model_.readout_selections_
        self.representation_metadata_ = {
            'readout': 'knn_hidden', 'grid': KNN_READOUT_GRID, 'selection_folds': KNN_SELECTION_FOLDS,
            'selection': 'mean_accuracy_over_stratified_splits_of_the_training_partition; ties lowest_canonical_config_id',
            'readout_seconds': self.readout_seconds_,
            'views': [{'view': v, 'config': s['config'], 'config_id': s['config_id'], 'inner_score': s['inner_score'],
                       'folds': s['folds'], 'candidate_scores': s['candidate_scores']}
                      for v, s in enumerate(self.readout_selections_)]}
        return self


def arrowflow_full_factory(config, seed):
    return AdaptiveMultiView(config=config, seed=seed)


def arrowflow_full_knn_factory(config, seed):
    return AdaptiveMultiViewKNN(config=config, seed=seed)


def bridge_registry(protocol):
    from .run_revision import dummy_factory
    registry = {'arrowflow_full': ModelSpec('arrowflow_full', arrowflow_full_factory, bridge_candidates(), True),
                'dummy': ModelSpec('dummy', dummy_factory, [{}], False)}
    registry.update(conventional_registry(protocol))
    return registry


def bridge_knn_registry(protocol):
    """The bridge registry with arrowflow_full replaced by arrowflow_full_knn: the same 16 abstract candidates
    (stochastic) with the kNN readout on the trained hidden ranking, beside the bridge run's dummy and conventional
    comparators, under the identical nested protocol. The output rule is not refitted: the training-only pilot of
    both ArrowFlow families projected beyond the 10 h cap, so the primary contrast pairs arrowflow_full_knn with the
    arrowflow_full outer-fold results of the frozen bridge run, which used the same datasets, folds, fit seeds and
    candidates and unchanged network-training sources (protocol knn_readout.reference)."""
    registry = bridge_registry(protocol)
    del registry['arrowflow_full']
    return {'arrowflow_full_knn': ModelSpec('arrowflow_full_knn', arrowflow_full_knn_factory, bridge_candidates(), True),
            **registry}


ABSTRACT_KEYS = ('embed_scale', 'degree_offset')
ABLATION_VARIANTS = ('views1', 'views3', 'views7', 'no_checkpoint', 'no_augment', 'borda_views7',
                     'single_view_no_checkpoint_no_augment', 'multiview_footrule_knn')


def resolve_selected(config, n_features, n_train):
    """The concrete MultiViewArrowFlow parameters of an abstract bridge candidate, exactly as
    AdaptiveMultiView.fit derives them from the training partition's shape."""
    params = {k: v for k, v in config.items() if k not in ABSTRACT_KEYS}
    return {**params, **resolve(config, n_features, n_train)}


def ablation_variants(selected):
    """Eight (variant_id, params) pairs at one resolved seven-view configuration.

    `selected` must already carry the concrete embed_dim/degree/augment (resolve_selected);
    the abstract keys are dropped so every params dict is a MultiViewArrowFlow keyword set,
    except the kNN control, whose params are the MultiViewFootruleKNN encoder settings.
    """
    base = {k: v for k, v in selected.items() if k not in ABSTRACT_KEYS}
    missing = {'n_views', 'strategy', 'embed_dim', 'degree', 'augment', 'validation_ratio', 'aggregation'} - base.keys()
    if missing:
        raise ValueError(f'Resolve the selected configuration before ablation; missing {sorted(missing)}')
    if base['n_views'] != 7:
        raise ValueError('The ablation variants are defined for the seven-view selected configuration')
    control = {k: base[k] for k in ('n_views', 'strategy', 'embed_dim', 'degree')}
    return [('views1', {**base, 'n_views': 1}),
            ('views3', {**base, 'n_views': 3}),
            ('views7', dict(base)),
            ('no_checkpoint', {**base, 'validation_ratio': 0}),
            ('no_augment', {**base, 'augment': False}),
            ('borda_views7', {**base, 'aggregation': 'borda'}),
            ('single_view_no_checkpoint_no_augment', {**base, 'n_views': 1, 'validation_ratio': 0, 'augment': False}),
            ('multiview_footrule_knn', control)]
