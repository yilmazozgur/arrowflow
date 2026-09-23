"""Nearest-baseline comparators for the deployed ArrowFlow-kNN readout (external review 2026-09-14, R3 and R8).

ArrowFlow's deployed classifier is a nearest-neighbour rule on a learned representation, so the sharpest alternative
explanation of its accuracy is that the supervised encoder, and not the ranking-filter update, carries the discrimination,
and that established supervised neighbourhood learning would do as well or better. The registered comparators (SVC, random
forest, MLP, gradient boosting and numeric kNN) do not test that. This family adds the four nearest baselines on all
seventeen datasets under the nested design of newdata_batch1.json, copied unchanged (5 x 3 outer folds, 3 inner folds,
split seed 27183, fitting seeds 8129, 19391 and 39019, candidate budget 24, candidate seed 41071, the tie rule, three
stochastic finalists, the report metrics, failure policy and freeze requirement):

lda_knn      LinearDiscriminantAnalysis as a supervised dimensionality reduction, then a numeric kNN on the discriminant
             coordinates. The component count is resolved fold-locally from component_scale (bounded by classes minus one
             and by the features); the neighbour count, the weighting and the Minkowski exponent are the numeric_knn
             comparator's grid.
pca_knn      the label-free counterpart: the same pipeline and the same grids with PCA in place of LDA, its component
             count bounded by the features, so the supervised part of the encoder can be separated from the projection.
nca_knn      NeighborhoodComponentsAnalysis (bounded max_iter, every convergence warning recorded per fit) then a numeric
             kNN; the component count, the neighbour count and the weighting are tuned.
kendall_svc  an SVC on a precomputed Kendall kernel (Jiao and Vert, 2015) over ArrowFlow's own encoded rankings, at the
             encoder settings of the input-kNN control (the four candidates of embed_scale x degree_offset, seven views,
             majority vote), so the baseline reads exactly the representation ArrowFlow's ranking layers receive. C is the
             svc_rbf comparator's grid.

ArrowFlow-kNN is not refitted here: the prespecified analysis (compare_baselines.py) pairs these outer folds with the
arrowflow_full_knn outer predictions of the registered runs (bridge_knn for the seven benchmark datasets, the two newdata
batches for the ten further datasets), which used the same rows, folds and fitting seeds.

python -m experiments.make_revision.neighbour_baselines draft --output P
    the unfrozen stage protocol
python -m experiments.make_revision.neighbour_baselines prepare --protocol P --output O [--dataset NAME ...]
    run_revision's prepared layout (protocol, candidates, environment, manifest, splits, data) for the pinned datasets
python -m experiments.make_revision.neighbour_baselines smoke --output O [--workers 3]
    synthetic, never evidence: a synthetic ArrowFlow-kNN reference run and a synthetic baselines run through
    run_revision's worker and reporting, then compare_baselines analyse
python -m experiments.make_revision.neighbour_baselines pilot --protocol P --output O [--dataset NAME ...]
    training-only runtime on the first outer training partition of every dataset: three evenly spaced candidates of every
    model (as run_revision.runtime_pilot), predictions on training rows only; no score
python -m experiments.make_revision.neighbour_baselines project --protocol P --pilot O/pilot.json --output PROJECTION.json
    the calibrated projection of the run at 16 workers, against the cap
python -m experiments.make_revision.neighbour_baselines freeze --draft P --projection PROJECTION.json --stages STAGES.json [--output F]
    the frozen protocol, only if the calibrated projection is within the cap
python -m experiments.make_revision.neighbour_baselines run --protocol F --output O [--workers N]
    run_revision's run stage for the frozen protocol (the same checks, worker, lock and verification)
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'  # as run_revision: spawned workers import this -m module before any numeric library
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
import json
import multiprocessing
from pathlib import Path
import time
import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from arrowflow.ranking import inverse_positions
from . import newdata as nd
from .bridge import resolve
from .comparisons import CONVENTIONAL_GRIDS, TimedPipeline, derive_seed
from .evaluation import ModelSpec, canonical_json, candidate_grid, config_id, dataset_fingerprint, make_splits
from .knn_controls import ABSTRACT_KEYS, CANDIDATE_KEYS as CONTROL_CANDIDATE_KEYS, INPUT_MODEL, TRAINED_MODEL, control_candidates
from .models import NumericImputer, OrdinalEncoder
from .multiview import view_strategy
from .secondary_studies import majority

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core: every module the
# bridge_knn and newdata runs sealed for arrowflow_full_knn and its controls, and newdata itself (the design and the pins).
SOURCE_MODULES = nd.SOURCE_MODULES + ['experiments.make_revision.newdata']

FAMILY = 'neighbour_baselines'
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-14'
PROTOCOL_FILE = PROTOCOLS/'neighbour_baselines.json'
PROTOCOL_ID = 'arrowflow-v3-neighbour-baselines-1'
TEMPLATE = nd.PROTOCOLS/'newdata_batch1.json'
REGISTRY = 'experiments.make_revision.neighbour_baselines:baselines_registry'
SMOKE_REGISTRY = 'experiments.make_revision.neighbour_baselines:smoke_baselines_registry'
WORKSPACE_RUNS = nd.WORKSPACE_RUNS
CAP_HOURS = 6
WORKERS = 16
MAX_WORKERS = 16
STAGE_OVERHEAD_HOURS = .25
KINDS = {'central': 'pooled', 'upper': 'max'}
DRAFT_STATUS = 'drafted_awaiting_prepare_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_confirmatory_scoring'
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'projection')
SMOKE_FIELDS = (*FREEZE_FIELDS, 'protocol_id', 'purpose', 'registry', 'datasets', 'panel', 'reference', 'kendall_kernel',
                'primary_family_size', 'analysis', 'outer_folds', 'outer_repeats', 'inner_folds')

LDA_MODEL, PCA_MODEL, NCA_MODEL, KENDALL_MODEL = 'lda_knn', 'pca_knn', 'nca_knn', 'kendall_svc'
MODEL_ORDER = (LDA_MODEL, PCA_MODEL, NCA_MODEL, KENDALL_MODEL)
SUPERVISED_MODELS = (LDA_MODEL, NCA_MODEL, KENDALL_MODEL)
PRIMARY_CONTRASTS = [f'{TRAINED_MODEL}_vs_{model}' for model in MODEL_ORDER]
ALPHA = .05
NCA_MAX_ITER = 50
COMPONENT_SCALES = [.5, 1.]
NEIGHBOUR_GRID = dict(CONVENTIONAL_GRIDS['numeric_knn'])
SVC_C = list(CONVENTIONAL_GRIDS['svc_rbf']['C'])
SVC_MAX_ITER = 1000000
KENDALL_KEYS = (*CONTROL_CANDIDATE_KEYS[INPUT_MODEL], 'C')
BASELINE_GRIDS = {LDA_MODEL: {'component_scale': COMPONENT_SCALES, **NEIGHBOUR_GRID},
                  PCA_MODEL: {'component_scale': COMPONENT_SCALES, **NEIGHBOUR_GRID},
                  NCA_MODEL: {'component_scale': COMPONENT_SCALES,
                              **{key: value for key, value in NEIGHBOUR_GRID.items() if key != 'p'}}}
STOCHASTIC = {LDA_MODEL: False, PCA_MODEL: False, NCA_MODEL: True, KENDALL_MODEL: True}

# The registered ArrowFlow-kNN runs whose outer predictions this family is paired with. Recorded here, not read from the
# workspace, so that protocol validation is pure and a detached production worktree needs no run directory.
REFERENCE_LABELS = ('knn', 'batch1', 'batch2')
REFERENCE_RUNS = {
    'knn': {'directory': '2026-09-12-bridge-knn', 'protocol_id': 'arrowflow-v3-bridge-knn-1',
            'protocol_sha256': '6335d8c4a1b16fb4044c3badb1fff1448f4083d4cc89611cd54b276d60057ff1',
            'summary_sha256': 'f5f0c7016864ce354c70f148bfecc31f8dbdbff5bc4f41d74e63e22a6ff36fda',
            'code_revision': '70fb9bf31092cb64e2bd349403ad090699a3494d'},
    'batch1': {'directory': '2026-09-14-newdata-batch1', 'protocol_id': 'arrowflow-v3-newdata-batch1-1',
               'protocol_sha256': 'f715ec2808b4b08c10e49ae23696070a471d7ea88eb351d63b059eecccc0d011',
               'summary_sha256': 'f3ab23c5eeccbe36f036325fe94c4c28ce69bc9d35fa16e62cea813e977ec843',
               'code_revision': '6022f9b5e2312f80a96e96b2d0607c72b7d52138'},
    'batch2': {'directory': '2026-09-14-newdata-batch2', 'protocol_id': 'arrowflow-v3-newdata-batch2-1',
               'protocol_sha256': '188ff1e946b6dc8df712c5a5cfb01e9cf2f9e7022a7776d98ec95ae1f66fca49',
               'summary_sha256': '50ebfbef88616f8ecd20737f45b3ef840ed19f75a5e5a3f8ceeb120e386d49db',
               'code_revision': '6022f9b5e2312f80a96e96b2d0607c72b7d52138'}}

# The seven benchmark datasets of run_revision.DATASETS, pinned by the dataset and splits hashes of the bridge_knn run's
# prepared manifests (the ten further datasets carry newdata's own pins).
BENCHMARK_PINS = (
    ('iris', [150, 4], 3, '33e362815d948773b41930da32bab6bc926bc3e27695071215897ada704a1d71', 'be679133de1d7551'),
    ('wine', [178, 13], 3, '25af908ff72aab7f44f1b69f12d1ae4231a1be2be49aaf38f1fca7736dd83adb', '8b6b40df532ebf02'),
    ('breast_cancer', [569, 30], 2, 'cc52cc8a9e05f5e3aceb9a5149af91a01ec49f05a392e9ac68da8d87d494c1fa', '93d4c06be906e07b'),
    ('wine_quality', [1599, 11], 3, 'ce355ca6e4b5ba0392eba133945437af61776a27eea18ff0e0c3055f7b1c0f66', 'ce17432feae50e58'),
    ('vehicle', [846, 18], 4, 'ba259412e447e5b657edf1313b1b8b6eebf214a72e24c6619aaf21371dd93886', '9f0639a14f5ee800'),
    ('segment', [2310, 19], 7, 'a2a1dbbbaf3c2983c8fd53be626118de95e52a8a0fb2dcd992f2a0de1544ed24', '48209cb10a3f57f2'),
    ('digits', [1797, 64], 10, '40521dcb7380cd2e327e0ed9e208fc5a08c78aa74eaf35d2570838c24d2c419b', '04a91fdff1945f71'))
BENCHMARK = tuple(pin[0] for pin in BENCHMARK_PINS)
FURTHER = tuple(nd.PANEL)
PANEL_NAMES = BENCHMARK + FURTHER

# The Kendall kernel is quadratic in the training rows and in the vocabulary; the cap is checked for every dataset before
# the freeze and again inside every fit, and a dataset above it is refused, never subsampled.
KERNEL_CAP = {'max_training_rows': 3000, 'max_feature_entries': 50000000,
              'rule': 'kendall_svc refuses a training partition with more than max_training_rows rows, or whose per-view '
                      'concordance feature matrix (training rows x V(V-1)/2 for the resolved vocabulary V) holds more than '
                      'max_feature_entries entries; the refusal is recorded in this protocol and in the fit log, and the '
                      'kernel is never computed on a subsample of the training partition'}
KENDALL_KERNEL_DEFINITION = (
    'K(s, s′) = (concordant - discordant) / (V choose 2) over the (V choose 2) unordered item pairs of two encoded '
    'rankings of the same V-item vocabulary: a pair {i, j} is concordant when both rankings place the same item first and '
    'discordant otherwise. The encoded rankings are total orders, so concordant + discordant = (V choose 2) and '
    'K = 1 - 2 * Kendall-tau-distance / (V choose 2), K(s, s) = 1 and K takes values in [-1, 1]; K is exactly Kendall’s '
    'tau between the two rankings. This is the Kendall kernel of Jiao and Vert (2015); it is positive semi-definite because '
    'K(s, s′) = <phi(s), phi(s′)> for the explicit feature map phi(s)_{i<j} = sign(position_j - position_i) / '
    'sqrt(V choose 2). neighbour_baselines.kendall_features builds that +-1 feature matrix from '
    'arrowflow.ranking.inverse_positions and kendall_kernel is the inner product: the partial sums are integers of modulus '
    'at most (V choose 2) < 2**24, so the float32 matrix product is exact, and the division by (V choose 2) is float64.')


def _plain(value):
    """The JSON form of a declaration (tuples become lists), so it compares equal to a protocol read from disk."""
    return json.loads(canonical_json(value))


# ----------------------------------------------------------------------------- the Kendall kernel

class KernelTooLarge(ValueError):
    """A training partition exceeds the declared Kendall kernel cap; the fit is refused rather than subsampled."""


def kendall_features(orders):
    """The +-1 concordance features of the Kendall kernel: for items i < j, the sign of position_j - position_i.

    `orders` holds rows of complete permutations of the coordinate IDs 0..V-1 (an ArrowFlow encoded ranking);
    arrowflow.ranking.inverse_positions turns each into the position vector and refuses anything else."""
    positions = inverse_positions(orders)
    vocabulary = positions.shape[1]
    if vocabulary < 2:
        raise ValueError('The Kendall kernel needs a vocabulary of at least two items')
    i, j = np.triu_indices(vocabulary, 1)
    return np.sign(positions[:, j] - positions[:, i]).astype(np.float32)


def kendall_pairs(vocabulary):
    return int(vocabulary) * (int(vocabulary) - 1) // 2


def kendall_kernel(orders_a, orders_b=None):
    """The normalized concordance between every pair of rankings: (concordant - discordant) / (V choose 2)."""
    features_a = kendall_features(orders_a)
    features_b = features_a if orders_b is None else kendall_features(orders_b)
    if features_a.shape[1] != features_b.shape[1]:
        raise ValueError('The Kendall kernel compares rankings of one vocabulary')
    vocabulary = np.asarray(orders_a).shape[1]
    return np.asarray(features_a @ features_b.T, dtype=np.float64) / kendall_pairs(vocabulary)


def kernel_cost(training_rows, vocabulary):
    """The kernel and feature sizes one fit of kendall_svc allocates for one view."""
    rows, pairs = int(training_rows), kendall_pairs(vocabulary)
    return {'training_rows': rows, 'vocabulary': int(vocabulary), 'pairs': pairs, 'kernel_entries': rows * rows,
            'feature_entries': rows * pairs, 'kernel_bytes': 8 * rows * rows, 'feature_bytes': 4 * rows * pairs}


def within_cap(cost, cap=None):
    cap = cap or KERNEL_CAP
    return cost['training_rows'] <= cap['max_training_rows'] and cost['feature_entries'] <= cap['max_feature_entries']


def check_kernel_cap(training_rows, vocabulary, cap=None):
    cost = kernel_cost(training_rows, vocabulary)
    if not within_cap(cost, cap):
        raise KernelTooLarge(f'The Kendall kernel of {cost["training_rows"]} training rows over a {cost["vocabulary"]}-item '
                             f'vocabulary ({cost["feature_entries"]} feature entries) exceeds the declared cap '
                             f'{cap or KERNEL_CAP}; the dataset is refused, never subsampled')
    return cost


def kendall_capacity(panel, cap=None, outer_folds=5):
    """Per dataset, the largest Kendall kernel the design can ask for: the outer training partition bound n - n // folds
    and the largest vocabulary the four encoder candidates resolve to on that partition."""
    rows = {}
    for entry in panel:
        samples, features = entry['shape']
        training = samples - samples // outer_folds
        vocabulary = max(resolve(config, features, training)['embed_dim'] for config in control_candidates(INPUT_MODEL))
        cost = kernel_cost(training, vocabulary)
        rows[entry['name']] = {**cost, 'training_rows_basis': 'n - n // outer_folds (an upper bound on any outer training '
                                                              'partition of the declared stratified design)',
                               'within_cap': within_cap(cost, cap)}
    return rows


# ----------------------------------------------------------------------------- fold-local projections

def resolve_components(component_scale, maximum):
    """The component count of one fold, as bridge.resolve derives an embedding width: round the scale times the maximum
    the training partition allows, then clip into [1, maximum]."""
    if not isinstance(component_scale, (int, float)) or isinstance(component_scale, bool) or not 0 < component_scale <= 1:
        raise ValueError('component_scale must be a fraction in (0, 1]')
    if int(maximum) < 1:
        raise ValueError('A projection needs at least one component')
    return int(np.clip(round(component_scale * int(maximum)), 1, int(maximum)))


class FoldLocalProjection(TransformerMixin, BaseEstimator):
    """A dimensionality reduction whose component count is resolved from the training partition of its own fold."""
    projection_id = None

    def __init__(self, component_scale=1.):
        self.component_scale = component_scale

    def maximum_components(self, X, y):
        raise NotImplementedError

    def make(self, n_components):
        raise NotImplementedError

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError('Expected a nonempty finite training matrix; impute and scale first')
        self.max_components_ = int(self.maximum_components(X, y))
        self.n_components_ = resolve_components(self.component_scale, self.max_components_)
        self.estimator_ = self.make(self.n_components_).fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'estimator_')
        return self.estimator_.transform(np.asarray(X, dtype=np.float64))

    def record(self):
        check_is_fitted(self, 'estimator_')
        return {'projection': self.projection_id, 'component_scale': float(self.component_scale),
                'max_components': int(self.max_components_), 'n_components': int(self.n_components_)}


class ScaledLDA(FoldLocalProjection):
    """LinearDiscriminantAnalysis (svd solver) as a supervised dimensionality reduction; at most classes minus one."""
    projection_id = 'linear_discriminant_analysis'

    def maximum_components(self, X, y):
        classes = len(np.unique(np.asarray(y)))
        if classes < 2:
            raise ValueError('Linear discriminant analysis needs at least two training classes')
        return max(1, min(classes - 1, X.shape[1]))

    def make(self, n_components):
        return LinearDiscriminantAnalysis(solver='svd', n_components=n_components)


class ScaledPCA(FoldLocalProjection):
    """PCA (full SVD, so the fit does not depend on any random state) as the label-free counterpart of ScaledLDA."""
    projection_id = 'principal_component_analysis'

    def maximum_components(self, X, y=None):
        return max(1, min(X.shape[0], X.shape[1]))

    def make(self, n_components):
        return PCA(n_components=n_components, svd_solver='full', whiten=False)


class ScaledNCA(FoldLocalProjection):
    """NeighborhoodComponentsAnalysis with a bounded max_iter; n_iter_ and the convergence status are recorded.

    sklearn raises its own ConvergenceWarning for a truncated NCA only when verbose is set, and verbose also prints a
    progress table, which a spawned worker must not do. A fit that stops at the bounded max_iter therefore raises the
    ConvergenceWarning here, so that it is recorded in that fit's fit_warnings exactly as the MLP comparator's warnings
    are; record() carries n_iter and reached_max_iter as well, so the truncation count never depends on a warning."""
    projection_id = 'neighborhood_components_analysis'

    def __init__(self, component_scale=1., max_iter=NCA_MAX_ITER, random_state=None):
        super().__init__(component_scale)
        self.max_iter = max_iter
        self.random_state = random_state

    def maximum_components(self, X, y=None):
        return max(1, X.shape[1])

    def make(self, n_components):
        if not isinstance(self.max_iter, int) or isinstance(self.max_iter, bool) or self.max_iter < 1:
            raise ValueError('NeighborhoodComponentsAnalysis needs a positive bounded max_iter')
        return NeighborhoodComponentsAnalysis(n_components=n_components, init='auto', max_iter=self.max_iter,
                                              random_state=self.random_state)

    def iterations(self):
        return int(getattr(self.estimator_, 'n_iter_', 0))

    def fit(self, X, y=None):
        super().fit(X, y)
        if self.iterations() >= int(self.max_iter):
            warnings.warn(f'NeighborhoodComponentsAnalysis stopped at the bounded max_iter={int(self.max_iter)} after '
                          f'{self.iterations()} iterations without converging', ConvergenceWarning)
        return self

    def record(self):
        iterations = self.iterations()
        return {**super().record(), 'max_iter': int(self.max_iter), 'n_iter': iterations,
                'reached_max_iter': iterations >= int(self.max_iter), 'init': 'auto',
                'random_state': None if self.random_state is None else int(self.random_state)}


PROJECTIONS = {LDA_MODEL: ScaledLDA, PCA_MODEL: ScaledPCA, NCA_MODEL: ScaledNCA}


class BaselinePipeline(TimedPipeline):
    """comparisons.TimedPipeline (fold-local steps, recorded timing and every fit warning) that also publishes the
    projection resolved inside the fit, so the selected component count of every fit is in the saved record."""
    def fit(self, X, y=None, **params):
        super().fit(X, y, **params)
        self.representation_metadata_ = self.named_steps['projection'].record()
        return self


def projection_factory(family, config, seed):
    """One fold-local pipeline: training-row imputation and scaling, the family's projection, then a numeric kNN."""
    settings = dict(config)
    scale = settings.pop('component_scale')
    parameters = {'component_scale': scale}
    if family == NCA_MODEL:
        parameters.update(max_iter=NCA_MAX_ITER, random_state=seed)
    return BaselinePipeline([('imputer', NumericImputer()), ('scaler', StandardScaler()),
                             ('projection', PROJECTIONS[family](**parameters)),
                             ('classifier', KNeighborsClassifier(**settings, n_jobs=1))])


# ----------------------------------------------------------------------------- the Kendall kernel SVC

class MultiViewKendallSVC(ClassifierMixin, BaseEstimator):
    """The seven encoders of input_footrule_knn with an SVC on a precomputed Kendall kernel per view; majority vote.

    View v is encoded exactly as in MultiViewInputKNN and MultiViewArrowFlowKNN: OrdinalEncoder(view_strategy(strategy, v),
    embed_dim, degree, lda_ratio, derive_seed(seed, 'view', v)) fitted on the training rows, whose transform is the encoded
    input ranking ArrowFlow's ranking layers receive. The view's classifier is SVC(kernel='precomputed') on
    kendall_kernel of those rankings. Only the encoded training orders are kept; the +-1 concordance features are rebuilt
    on demand, so a fitted view holds V integers per training row instead of (V choose 2)."""
    def __init__(self, n_views=7, strategy='diverse', embed_dim=32, degree=2, aggregation='majority', lda_ratio=.3, C=1.,
                 seed=8129, kernel_cap=None):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)

    def fit(self, X, y):
        if self.aggregation != 'majority':
            raise ValueError('MultiViewKendallSVC combines the per-view SVC votes by majority only')
        if not isinstance(self.C, (int, float)) or isinstance(self.C, bool) or self.C <= 0:
            raise ValueError('The SVC penalty C must be positive')
        self.classes_ = np.unique(y)
        self.views_, self.view_records_ = [], []
        self.readout_seconds_ = encoding = 0.
        self.kernel_cost_ = check_kernel_cap(len(np.asarray(y)), self.embed_dim, self.kernel_cap)
        for v in range(self.n_views):
            seed_v = derive_seed(self.seed, 'view', v)
            start = time.perf_counter()
            encoder = OrdinalEncoder(view_strategy(self.strategy, v), self.embed_dim, self.degree, self.lda_ratio,
                                     seed_v).fit(X, y)
            orders = np.asarray(encoder.transform(X), dtype=np.int32)
            gram = kendall_kernel(orders)
            encoding += time.perf_counter() - start
            start = time.perf_counter()
            classifier = SVC(kernel='precomputed', C=self.C, probability=False, max_iter=SVC_MAX_ITER).fit(gram, y)
            self.readout_seconds_ += time.perf_counter() - start
            self.views_.append((encoder, orders, classifier))
            self.view_records_.append({'view': v, 'strategy': view_strategy(self.strategy, v), 'view_seed': int(seed_v),
                                       'vocabulary': int(orders.shape[1]), 'pairs': kendall_pairs(orders.shape[1]),
                                       'support_vectors': int(sum(map(int, classifier.n_support_))),
                                       'libsvm_fit_status': int(classifier.fit_status_)})
        self.encoding_seconds_ = encoding
        self.training_seconds_ = 0.
        return self

    def predict_views(self, X):
        check_is_fitted(self, 'views_')
        start = time.perf_counter()
        grams = [kendall_kernel(np.asarray(encoder.transform(X), dtype=np.int32), orders)
                 for encoder, orders, _ in self.views_]
        self.last_encoding_seconds_ = time.perf_counter() - start
        return [classifier.predict(gram) for (_, _, classifier), gram in zip(self.views_, grams)]

    def predict(self, X):
        return majority(self.predict_views(X))

    def readout_record(self):
        return {'readout': 'kendall_kernel_svc', 'kernel': KENDALL_KERNEL_DEFINITION, 'C': float(self.C),
                'svc_max_iter': SVC_MAX_ITER, 'aggregation': self.aggregation,
                'representation': "each view's encoded input ranking, the representation ArrowFlow's ranking layers receive",
                'kernel_cost_per_view': self.kernel_cost_, 'readout_seconds': self.readout_seconds_,
                'views': list(self.view_records_)}


class AdaptiveKendallSVC(ClassifierMixin, BaseEstimator):
    """Resolves embed_dim and degree from the training partition's shape as bridge.AdaptiveMultiView does, then fits
    MultiViewKendallSVC. The resolved augment is dropped: nothing is trained and no training row is duplicated."""
    def __init__(self, config=None, seed=8129, kernel_cap=None):
        self.config = config
        self.seed = seed
        self.kernel_cap = kernel_cap

    def fit(self, X, y):
        cfg = dict(self.config)
        if set(cfg) != set(KENDALL_KEYS):
            raise ValueError(f'{KENDALL_MODEL} configurations hold exactly {sorted(KENDALL_KEYS)}, not {sorted(cfg)}')
        resolved = resolve(cfg, X.shape[1], len(y))
        self.resolved_ = {'embed_dim': resolved['embed_dim'], 'degree': resolved['degree']}
        params = {key: value for key, value in cfg.items() if key not in ABSTRACT_KEYS}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            self.model_ = MultiViewKendallSVC(**params, **self.resolved_, seed=self.seed,
                                              kernel_cap=self.kernel_cap).fit(X, y)
        self.fit_warnings_ = [{'category': w.category.__name__, 'message': str(w.message)} for w in captured]
        self.encoder_ = self.model_.views_[0][0]
        self.classes_ = self.model_.classes_
        self.encoding_seconds_ = self.model_.encoding_seconds_
        self.training_seconds_ = 0.
        self.readout_seconds_ = self.model_.readout_seconds_
        self.representation_metadata_ = {**self.model_.readout_record(), 'resolved': dict(self.resolved_)}
        return self

    def predict(self, X):
        out = self.model_.predict(X)
        self.last_encoding_seconds_ = self.model_.last_encoding_seconds_
        return out


def kendall_factory(config, seed):
    return AdaptiveKendallSVC(config=config, seed=seed)


def kendall_candidates(reference_candidates=None):
    """input_footrule_knn's four encoder candidates crossed with the svc_rbf comparator's C grid, in canonical order."""
    from .knn_controls import project_candidates
    base = (control_candidates(INPUT_MODEL) if reference_candidates is None
            else project_candidates(reference_candidates, CONTROL_CANDIDATE_KEYS[INPUT_MODEL]))
    candidates = {config_id(candidate): candidate
                  for candidate in ({**entry, 'C': penalty} for entry in base for penalty in SVC_C)}
    return [candidates[cid] for cid in sorted(candidates)]


# ----------------------------------------------------------------------------- the registry

def build_registry(protocol):
    """The four baselines in MODEL_ORDER, each tuned on the inner folds within the protocol's candidate budget."""
    budget, seed = protocol['candidate_budget'], protocol['candidate_seed']
    refused = list((protocol.get('kendall_kernel') or {}).get('refused') or [])
    if refused:
        raise ValueError(f'{KENDALL_MODEL} refuses {", ".join(refused)} under the declared kernel cap; a panel holding a '
                         'refused dataset must be split before it is run, and the kernel is never computed on a subsample')
    registry = {model: ModelSpec(model, partial(projection_factory, model),
                                 candidate_grid(BASELINE_GRIDS[model], budget, seed), STOCHASTIC[model])
                for model in (LDA_MODEL, PCA_MODEL, NCA_MODEL)}
    registry[KENDALL_MODEL] = ModelSpec(KENDALL_MODEL, kendall_factory, kendall_candidates(), STOCHASTIC[KENDALL_MODEL])
    if tuple(registry) != MODEL_ORDER:
        raise RuntimeError(f'Registry order {list(registry)} differs from MODEL_ORDER')
    for model, spec in registry.items():
        if len(spec.candidates) > budget:
            raise ValueError(f'{model} declares {len(spec.candidates)} candidates above the budget {budget}')
    return registry


def baselines_registry(protocol):
    """The production registry of a neighbour-baselines protocol (the stage draft or the frozen protocol)."""
    validate_protocol(protocol)
    if protocol.get('registry') != REGISTRY:
        raise ValueError(f'baselines_registry serves protocols declaring {REGISTRY}')
    return build_registry(protocol)


def smoke_baselines_registry(protocol):
    """Synthetic smoke only: the four baselines at their first two candidates."""
    if protocol.get('purpose') != 'synthetic_smoke_only':
        raise ValueError('smoke_baselines_registry serves synthetic smoke protocols only')
    validate_protocol(protocol)
    real = build_registry(protocol)
    return {model: ModelSpec(model, spec.factory, spec.candidates[:2], spec.stochastic) for model, spec in real.items()}


SMOKE_REFERENCE_MODEL = TRAINED_MODEL


def smoke_reference_registry(protocol):
    """Synthetic smoke only: the bridge_knn registry (arrowflow_full_knn at the smoke protocol's declared one-iteration
    candidates, every other model at its first two), so the smoke analysis has the whole rank panel to pair with."""
    from .bridge import bridge_knn_registry
    if protocol.get('purpose') != 'synthetic_smoke_only':
        raise ValueError('smoke_reference_registry serves synthetic smoke protocols only')
    real = bridge_knn_registry(protocol)
    registry = {SMOKE_REFERENCE_MODEL: ModelSpec(SMOKE_REFERENCE_MODEL, real[SMOKE_REFERENCE_MODEL].factory,
                                                 list(protocol['smoke_reference_candidates']), True)}
    registry.update((model, ModelSpec(model, spec.factory, spec.candidates[:2], spec.stochastic))
                    for model, spec in real.items() if model != SMOKE_REFERENCE_MODEL)
    return registry


# ----------------------------------------------------------------------------- the panel, the declarations, the protocol

def panel_declaration():
    """The seventeen datasets with their loader, reference run and pinned dataset and splits hashes."""
    entries = [{'name': name, 'panel': 'benchmark', 'loader': 'experiments.make_revision.run_revision:load_dataset',
                'reference': 'knn', 'shape': list(shape), 'n_classes': classes, 'dataset_hash': dataset_hash,
                'splits_hash': splits_hash}
               for name, shape, classes, dataset_hash, splits_hash in BENCHMARK_PINS]
    batches = {name: number for number, path in nd.PROTOCOL_FILES.items()
               for name in json.loads(Path(path).read_text())['datasets']}
    for name in FURTHER:
        pin = nd.PIN_BY_NAME[name]
        entries.append({'name': name, 'panel': 'further', 'loader': 'experiments.make_revision.newdata:load_newdata',
                        'reference': f'batch{batches[name]}', 'shape': list(pin['shape']),
                        'n_classes': len(pin['label_map']), 'dataset_hash': pin['dataset_hash'],
                        'splits_hash': pin['splits_hash']})
    return _plain(entries)


def panel_by_name(protocol):
    return {entry['name']: entry for entry in protocol['panel']}


def reference_declaration(panel):
    """The registered ArrowFlow-kNN runs this family is paired with, each pinned exactly as knn_controls pins its
    reference: protocol id, protocol sha256, code revision and summary sha256, with the datasets it supplies."""
    return _plain({label: {**REFERENCE_RUNS[label], 'model_id': TRAINED_MODEL,
                           'datasets': [entry['name'] for entry in panel if entry['reference'] == label]}
                   for label in REFERENCE_LABELS})


def models_declaration():
    grid_note = ('the neighbour count, the weighting and the Minkowski exponent of the numeric_knn comparator '
                 '(comparisons.CONVENTIONAL_GRIDS)')
    component_rule = ('component_scale is resolved inside every fit as int(clip(round(component_scale * max_components), '
                      '1, max_components)), exactly as bridge.resolve derives an embedding width; on a dataset whose '
                      'max_components is 1 (a two-class LDA) both scales resolve to the same single component')
    common = {'preprocessing': 'NumericImputer (training column means) then StandardScaler, both fitted on the training '
                               'rows of the fit only, as the conventional comparators are',
              'pipeline': 'experiments.make_revision.neighbour_baselines:BaselinePipeline (comparisons.TimedPipeline, '
                          'which records encoding, classifier fit and prediction seconds and every warning raised in the '
                          'fit)',
              'selection': 'the harness inner folds of the outer training partition (evaluation.select_model), mean '
                           'accuracy, ties to the lowest canonical config_id'}
    declaration = {
        LDA_MODEL: {**common, 'estimator': 'sklearn LinearDiscriminantAnalysis(solver="svd") as a supervised '
                                           'dimensionality reduction, then sklearn KNeighborsClassifier',
                    'grid': _plain(BASELINE_GRIDS[LDA_MODEL]), 'tuned': f'the component count and {grid_note}',
                    'component_bound': 'min(classes - 1, features) of the training partition', 'component_rule': component_rule,
                    'question': 'whether the label-aware projection alone, with a neighbour rule on top, explains the '
                                'accuracy of the deployed readout'},
        PCA_MODEL: {**common, 'estimator': 'sklearn PCA(svd_solver="full", whiten=False) as a label-free dimensionality '
                                           'reduction, then sklearn KNeighborsClassifier',
                    'grid': _plain(BASELINE_GRIDS[PCA_MODEL]), 'tuned': f'the component count and {grid_note}',
                    'component_bound': 'min(training rows, features)', 'component_rule': component_rule,
                    'question': 'the label-free counterpart of lda_knn, so the supervised part of the encoder is '
                                'separated from the projection itself; at component_scale 1 the projection is an '
                                'orthogonal rotation, which leaves the Euclidean neighbour order unchanged',
                    'determinism': 'svd_solver="full" is exact, so no random state enters the fit'},
        NCA_MODEL: {**common, 'estimator': f'sklearn NeighborhoodComponentsAnalysis(init="auto", max_iter={NCA_MAX_ITER}, '
                                           'random_state=the fitting seed), then sklearn KNeighborsClassifier '
                                           '(Minkowski exponent 2, the distance NCA optimizes)',
                    'grid': _plain(BASELINE_GRIDS[NCA_MODEL]),
                    'tuned': 'the component count, the neighbour count and the weighting',
                    'component_bound': 'the features', 'component_rule': component_rule,
                    'max_iter': NCA_MAX_ITER,
                    'convergence': 'every warning raised inside a fit is recorded in that fit\'s record (fit_warnings) and '
                                   'the resolved projection records n_iter and reached_max_iter; sklearn raises its own '
                                   'ConvergenceWarning for a truncated NCA only under verbose, which a spawned worker '
                                   'must not set, so ScaledNCA raises it from n_iter_ instead; the analysis reports both '
                                   'counts, which the review asked for and the registered MLP comparator does not carry',
                    'stochastic': 'the fitting seed is passed as random_state; with init="auto" the initialization is '
                                  'deterministic, so the three fitting seeds are expected to agree exactly and the '
                                  'recorded within-fold seed SD reports whether they do',
                    'question': 'whether established supervised neighbourhood learning, the method ArrowFlow\'s deployed '
                                'readout is closest to, does as well or better'},
        KENDALL_MODEL: {'estimator': 'experiments.make_revision.neighbour_baselines:MultiViewKendallSVC via '
                                     'AdaptiveKendallSVC: the seven encoders of input_footrule_knn with '
                                     'sklearn SVC(kernel="precomputed") on the Kendall kernel of each view\'s encoded '
                                     'ranking; majority vote',
                        'representation': "ArrowFlow's own encoder at the encoder settings of the input-kNN control, so "
                                          "the baseline reads exactly the representation ArrowFlow's ranking layers "
                                          'receive; embed_dim and degree are resolved from the training partition by '
                                          'bridge.resolve and the resolved augmentation is dropped, as in '
                                          'knn_controls.AdaptiveKNNControl',
                        'kernel': KENDALL_KERNEL_DEFINITION, 'reference': 'Jiao and Vert (2015), the Kendall kernel',
                        'candidate_source': "input_footrule_knn's four candidates (knn_controls.control_candidates) "
                                            'crossed with the C grid of the svc_rbf comparator',
                        'grid': _plain({'embed_scale': [1, 2], 'degree_offset': [0, -1], 'C': SVC_C}),
                        'tuned': 'the encoder setting (embed_scale and degree_offset) and C',
                        'svc_max_iter': SVC_MAX_ITER,
                        'preprocessing': "each view's OrdinalEncoder fits its own imputation, polynomial expansion, "
                                         'scaling and projection on the training rows of the fit only',
                        'selection': common['selection'],
                        'question': 'whether a permutation kernel over the same encoded rankings, with no ranking-filter '
                                    'update at all, reaches the accuracy of the trained readout'}}
    registry = build_registry(dict(nd.DESIGN, kendall_kernel={'refused': []}))
    for model in MODEL_ORDER:
        declaration[model].update(candidates=len(registry[model].candidates), stochastic=STOCHASTIC[model],
                                  candidate_budget=nd.DESIGN['candidate_budget'],
                                  timing='fit and prediction seconds of every fit, as the registered comparators record '
                                         'them (evaluation._fit_predict)')
    return _plain(declaration)


def analysis_declaration(panel):
    names = [entry['name'] for entry in panel]
    families = {model: {'contrast': f'{TRAINED_MODEL}_vs_{model}', 'model_a': TRAINED_MODEL, 'model_b': model,
                        'size': len(names), 'datasets': list(names),
                        'definition': f'per dataset, {TRAINED_MODEL} minus {model} accuracy',
                        'interval': nd.INTERVAL_RULE, 'alpha': ALPHA,
                        'multiplicity': f'Holm across the {len(names)} datasets of this baseline '
                                        '(evaluation.holm_adjust), adjusted separately from the other three baselines'}
                for model in MODEL_ORDER}
    return _plain({
        'status': 'prespecified before any outer score of these baselines exists; computed only by compare_baselines '
                  'analyse, which refuses (exit 2, nothing written) until the run is complete, then re-verifies the run '
                  'and every referenced ArrowFlow run with compare_runs.load_run and compare_runs.verify_run before any '
                  'score is read',
        'command': 'python -m experiments.make_revision.compare_baselines analyse --run <run> --output <directory> '
                   '[--runs <directory holding the registered runs>]',
        'requires_run_complete': True, 'metric': 'accuracy', 'datasets': list(names), 'models': list(MODEL_ORDER),
        'reference_model': TRAINED_MODEL,
        'reference_not_refitted': f'{TRAINED_MODEL} is not refitted for this family: its outer predictions are the '
                                  'registered runs\' (bridge_knn for the seven benchmark datasets, the two newdata '
                                  'batches for the ten further datasets), paired fold by fold. The pairing is verified '
                                  'the way compare_runs.check_training_pairing verifies a control run against the knn '
                                  'run (identical panel, dataset and splits hashes, nested design and fitting seeds, and '
                                  'byte-identical sealed sources) and the analysis refuses on any mismatch.',
        'primary_contrasts': list(PRIMARY_CONTRASTS), 'primary_family_size': len(names), 'families': families,
        'descriptive': {
            'status': 'descriptive; no multiplicity adjustment',
            'metrics_table': 'per dataset and baseline, the mean outer error (the mean over outer folds of the '
                             'fitting-seed-averaged error), the outer-fold SD (ddof 1) and the mean within-fold seed SD, '
                             'with balanced accuracy and macro-F1 in the same form, from the verified summary.json; a '
                             'non-stochastic baseline is fitted at one seed, so its seed SD is undefined by construction',
            'rank_panel': 'the rank of each baseline by mean outer error among the eleven models every dataset holds: '
                          'the four baselines and the seven models of the bridge_knn registry (ArrowFlow-kNN, the five '
                          'tuned classical comparators and the majority class); ties take the average rank. The three '
                          'ArrowFlow controls exist only on the ten further datasets and are therefore outside the rank '
                          'panel; their errors are listed beside it where they exist',
            'convergence_warnings': 'per dataset and baseline, the number of fits raising a warning and the count of each '
                                    'warning category, over the outer fits and over every fit of the complete fit logs '
                                    '(inner and outer); nca_knn additionally reports the fits that reached max_iter'},
        'interpretation': {
            'status': 'fixed in advance and recorded before any outer score exists',
            'rule': f'a baseline matches or beats ArrowFlow-kNN on a dataset when its mean outer error is at most '
                    f'{TRAINED_MODEL}\'s mean outer error there; it does so on most datasets when that holds on at least '
                    f'{len(names) // 2 + 1} of the {len(names)}',
            'if_matched': 'if a supervised neighbourhood baseline (lda_knn, nca_knn or kendall_svc) matches or beats '
                          'ArrowFlow on most datasets, the paper says so plainly in the results and in the discussion, '
                          'and the contribution is stated as a construction within permutation space rather than an '
                          'accuracy argument',
            'if_not_matched': 'if no supervised neighbourhood baseline matches or beats ArrowFlow on most datasets, the '
                              'paper reports the per-dataset comparison and the Holm-adjusted family in full, and still '
                              'states the contribution as a construction within permutation space; the baselines bound '
                              'the alternative explanation, they do not establish an accuracy claim',
            'supervised_baselines': list(SUPERVISED_MODELS)},
        'notes': {
            'what_is_tested': 'whether established supervised neighbourhood learning on the same rows (lda_knn, nca_knn) '
                              'and a permutation kernel on the same encoded rankings (kendall_svc) reach the accuracy of '
                              'the deployed ArrowFlow-kNN readout',
            'what_is_not_tested': 'nothing here isolates the ranking-filter update from the encoder: the alternative '
                                  'explanation is bounded by comparison, not by decomposition, and the component ablation '
                                  'and the untrained and input controls of the registered runs remain the decomposition',
            'lmnn': 'LMNN is out of scope: metric_learn is not installed in this environment and no package may be added '
                    'for this family, so the supervised metric learner is NCA alone',
            'polynomial': 'the three sklearn pipelines read the imputed and scaled features, with no polynomial '
                          'expansion; only kendall_svc reads ArrowFlow\'s own encoder, which expands and projects',
            'unpaired_nothing': 'every contrast is paired: the same rows, the same outer folds and the same fitting seeds '
                                'as the registered ArrowFlow-kNN runs'}})


def kendall_kernel_declaration(panel):
    capacity = kendall_capacity(panel)
    refused = sorted(name for name, entry in capacity.items() if not entry['within_cap'])
    return _plain({'definition': KENDALL_KERNEL_DEFINITION, 'reference': 'Jiao and Vert (2015), the Kendall kernel',
                   'implementation': 'experiments.make_revision.neighbour_baselines:kendall_features and kendall_kernel; '
                                     'no dependency beyond numpy, and the implementation is tested against a brute-force '
                                     'pairwise concordance count over every permutation of small vocabularies',
                   'cap': dict(KERNEL_CAP), 'capacity': capacity, 'refused': refused,
                   'refusal_effect': 'a panel holding a refused dataset is not run: build_registry raises, so the run '
                                     'cannot start and the kernel is never computed on a subsample'})


def design_source():
    return {'template': 'protocols/2026-09-12/newdata_batch1.json (sha256 in source_template_sha256)',
            'identical_to_template': ', '.join(nd.DESIGN_COPY_KEYS),
            'changed': 'protocol_id, production_family, registry, datasets, panel, models, model_order, '
                       'primary_contrasts, primary_family_size, multiplicity, analysis, reference, kendall_kernel, '
                       'selection_ruling, wallclock_cap_hours, frozen, status, resource_decision',
            'removed_from_template': {'batch, batches, batch_projection': 'one run over all seventeen datasets',
                                      'model_template_sha256, secondary_contrasts, secondary_family_size, production '
                                      'panel fields': 'this family declares its own models, panel and one family per '
                                                      'baseline',
                                      'frozen_at_utc': 'set at the freeze'},
            'added': 'reference, kendall_kernel, workers, projection', 'family': FAMILY}


SELECTION_RULING = (
    'External review of 2026-09-14 (REPORT.md R3 and R8, novelty-literature-audit items 3 and 4, and the referee request '
    '"Run the nearest conceptual comparators"): all seventeen registered datasets, every baseline reported whatever the '
    'result, with the interpretation fixed in this protocol before any outer score exists. The panel, the folds, the '
    'fitting seeds and the candidate budget are the registered design, so every contrast is paired with the registered '
    'ArrowFlow-kNN outer predictions and nothing is refitted.')


def draft_protocol():
    """The unfrozen stage protocol: newdata_batch1.json's nested design, the seventeen pinned datasets, the four
    baselines, the registered reference runs, the Kendall kernel cap and the prespecified analysis."""
    template = json.loads(TEMPLATE.read_text())
    protocol = {key: template[key] for key in nd.DESIGN_COPY_KEYS}
    if [key for key, value in nd.DESIGN.items() if protocol[key] != value]:
        raise ValueError('newdata_batch1.json no longer holds the newdata nested design')
    panel = panel_declaration()
    names = [entry['name'] for entry in panel]
    protocol.update(
        protocol_id=PROTOCOL_ID, production_family=FAMILY, registry=REGISTRY, datasets=names, panel=panel,
        models=models_declaration(), model_order=list(MODEL_ORDER), primary_contrasts=list(PRIMARY_CONTRASTS),
        primary_family_size=len(names),
        multiplicity=f'Holm_across_the_{len(names)}_datasets_within_each_baseline; the four baseline families are '
                     'adjusted separately',
        reference=reference_declaration(panel), kendall_kernel=kendall_kernel_declaration(panel),
        analysis=analysis_declaration(panel), selection_ruling=SELECTION_RULING, wallclock_cap_hours=CAP_HOURS,
        workers=WORKERS, design_source=design_source(), source_template_sha256=nd.sha256_file(TEMPLATE),
        frozen=False, status=DRAFT_STATUS,
        resource_decision='pending: prepare, the synthetic smoke (run, reporting and analysis) and the training-only '
                          'pilot of every dataset, then the calibrated projection against the cap',
        projection=None)
    return _plain(protocol)


def _validate_smoke(p):
    panel = p.get('panel')
    if (not isinstance(panel, list) or not panel or any(not isinstance(entry, dict) for entry in panel)
            or len({entry.get('name') for entry in panel}) != len(panel)
            or any(entry.get('reference') not in (p.get('reference') or {}) for entry in panel)):
        raise ValueError('A synthetic smoke panel holds uniquely named datasets, each naming a declared reference run')
    if (p.get('registry') != SMOKE_REGISTRY or p.get('datasets') != [entry['name'] for entry in panel]
            or p.get('primary_family_size') != len(panel) or p.get('analysis') != analysis_declaration(panel)
            or p.get('kendall_kernel') != kendall_kernel_declaration(panel) or p.get('frozen') is not True
            or not str(p.get('protocol_id', '')).endswith('-synthetic-smoke')
            or any(type(p.get(key)) is not int or p[key] < 2 for key in ('outer_folds', 'inner_folds'))
            or type(p.get('outer_repeats')) is not int or p['outer_repeats'] < 1):
        raise ValueError('A synthetic smoke protocol is frozen, names the smoke registry and declares its datasets, '
                         'family size, kernel capacity and analysis from its panel')
    return p


def validate_protocol(p):
    """A production protocol is the stage draft, or the draft with exactly the freeze fields set by `freeze`; a synthetic
    smoke protocol is the draft with its synthetic panel, design, references and analysis."""
    if not isinstance(p, dict) or p.get('production_family') != FAMILY:
        raise ValueError(f'production_family must be {FAMILY}')
    draft = draft_protocol()
    smoke = p.get('purpose') == 'synthetic_smoke_only'
    ignored = SMOKE_FIELDS if smoke else FREEZE_FIELDS
    differing = sorted(key for key in set(p) | set(draft)
                       if key not in ignored and canonical_json(p.get(key)) != canonical_json(draft.get(key)))
    if differing:
        raise ValueError(f'The protocol differs from neighbour_baselines.draft_protocol() in {", ".join(differing)}')
    if smoke:
        return _validate_smoke(p)
    if not p.get('frozen'):
        if canonical_json(p) != canonical_json(draft):
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('projection') or {}
    hours = projection.get('decision_hours')
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != CAP_HOURS or projection.get('workers') != WORKERS
            or isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= CAP_HOURS):
        raise ValueError(f'A frozen protocol records its freeze and a calibrated projection within the {CAP_HOURS} h cap '
                         f'at {WORKERS} workers')
    return p


# ----------------------------------------------------------------------------- loading, prepare and run

def load(name, protocol=None):
    """(X, y, manifest) of one panel dataset through the loader of the run that holds its ArrowFlow-kNN predictions."""
    from .run_revision import load_dataset
    if name in BENCHMARK:
        return load_dataset(name)
    if name in FURTHER:
        return nd.load_newdata(name)
    raise nd.DatasetIdentityError(f'{name} is not in the neighbour-baselines panel')


def prepare(output, protocol, names=None, *, loader=None):
    """run_revision.prepare for the pinned datasets: the same records and layout, each dataset loaded exactly as the run
    holding its ArrowFlow-kNN predictions loaded it, with its dataset and splits hashes checked against the panel pins."""
    from .run_revision import environment_record, get_registry, write_json
    validate_protocol(protocol)
    if protocol.get('purpose') is not None:
        raise ValueError('prepare serves production protocols; the smoke writes its own synthetic datasets')
    output = Path(output)
    names = list(protocol['datasets'] if names is None else names)
    outside = [name for name in names if name not in protocol['datasets']]
    if outside:
        raise ValueError(f'Datasets outside the protocol datasets: {outside}')
    registry = get_registry(protocol['registry'], protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', nd.candidate_record(registry))
    write_json(output/'environment.json', environment_record(protocol['registry']))
    entries = panel_by_name(protocol)
    for name in names:
        X, y, manifest = (loader or load)(name, protocol)
        splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'],
                             protocol['split_seed'])
        manifest['splits_hash'] = config_id(splits)
        if (manifest['dataset_hash'], manifest['splits_hash']) != (entries[name]['dataset_hash'], entries[name]['splits_hash']):
            raise nd.DatasetIdentityError(f'{name}: the dataset or splits hash differs from the panel pins, so the folds '
                                          'would not be the registered runs\' folds')
        write_json(output/name/'manifest.json', manifest)
        write_json(output/name/'splits.json', splits)
        destination = output/name/'data.npz'
        if not destination.exists():
            np.savez_compressed(destination, X=X, y=y)
    return registry


def run(protocol_path, output, workers):
    """run_revision's run stage for the frozen protocol, with the same checks in the same order."""
    from .reporting import collect_verified_results
    from .run_revision import (_worker, environment_record, execution_lock, get_registry, load_prepared, planned_jobs,
                               write_json)
    output = Path(output)
    protocol = json.loads(Path(protocol_path).read_text())
    if not protocol.get('frozen'):
        raise ValueError('Confirmatory run requires a reviewed frozen protocol')
    if protocol.get('purpose') is not None:
        raise ValueError('A production run needs the frozen production protocol, not a synthetic smoke protocol')
    validate_protocol(protocol)
    saved = json.loads((output/'protocol.json').read_text())
    if saved != protocol:
        raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
    registry_path = protocol['registry']
    registry = get_registry(registry_path, protocol)
    if canonical_json(json.loads((output/'candidates.json').read_text())) != canonical_json(nd.candidate_record(registry)):
        raise ValueError('Candidate registry changed after prepare')
    if json.loads((output/'environment.json').read_text()) != environment_record(registry_path):
        raise ValueError('Code revision, source, environment, or registry changed after prepare')
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError('Worker count must be between 1 and 16')
    names, entries = list(protocol['datasets']), panel_by_name(protocol)
    for name in names:
        manifest = load_prepared(output, name)[2]
        if (manifest['dataset_hash'], manifest['splits_hash']) != (entries[name]['dataset_hash'], entries[name]['splits_hash']):
            raise nd.DatasetIdentityError(f'{name}: the prepared dataset or splits hash differs from the panel pins')
    write_json(output/'planned_jobs.json', planned_jobs(names, protocol, registry))
    jobs = [(str(output), name, index, model, registry_path) for name in names
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for path in pool.map(_worker, jobs):
            print(path, flush=True)
    collect_verified_results(output, names, protocol, registry)


# ----------------------------------------------------------------------------- training-only pilot, projection and freeze

def runtime_pilot(output, protocol, names=None, *, loader=None):
    """run_revision.runtime_pilot for the pinned datasets: three evenly spaced candidates of every model on the first
    outer training partition, predictions on training rows only; no score, and pilot.json records the protocol hash."""
    from .evaluation import _fit_predict
    from .run_revision import execution_lock, load_prepared, write_json
    output = Path(output)
    names = list(protocol['datasets'] if names is None else names)
    with execution_lock():
        registry = prepare(output, protocol, names, loader=loader)
        rows, identity = [], {}
        for name in names:
            X, y, manifest, splits = load_prepared(output, name)
            identity[name] = {'dataset_hash': manifest['dataset_hash'], 'splits_hash': manifest['splits_hash']}
            train = splits[0]['train']  # never pass any outer-test sample or label
            for model, spec in registry.items():
                for index in sorted({0, len(spec.candidates) // 2, len(spec.candidates) - 1}):
                    config = spec.candidates[index]
                    start = time.perf_counter()
                    row = {'dataset_id': name, 'model_id': model, 'config': config, 'config_id': config_id(config),
                           'fit_rows': train, 'model_seed': protocol['fit_seeds'][0]}
                    try:
                        _, timing = _fit_predict(spec, config, protocol['fit_seeds'][0], X[train], y[train], X[train])
                        row.update(timing, status='ok', elapsed_seconds=time.perf_counter() - start)
                    except Exception as exc:
                        row.update(status='failed', exception=f'{type(exc).__name__}: {exc}',
                                   elapsed_seconds=time.perf_counter() - start)
                    rows.append(row)
    estimates = {}
    for model, spec in registry.items():
        durations = [r['elapsed_seconds'] for r in rows if r['model_id'] == model and r['status'] == 'ok']
        per_outer = nd.fits_per_outer(len(spec.candidates), spec.stochastic, protocol['inner_folds'])
        total = per_outer * protocol['outer_folds'] * protocol['outer_repeats'] * len(protocol['datasets'])
        estimates[model] = {'fits_per_outer': per_outer, 'panel_fit_count': total,
                            'observed_seconds_min': min(durations) if durations else None,
                            'observed_seconds_max': max(durations) if durations else None,
                            'serial_panel_seconds_using_observed_max': total * max(durations) if durations else None}
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'family': FAMILY, 'rows': rows,
              'workload_estimates': estimates,
              'estimate_limitations': 'Sampled configurations/datasets; max extrapolation is not a runtime bound; RSS is '
                                      'process lifetime high-water mark. This pilot ran while other production runs held '
                                      'the machine, so the observed seconds are inflated by contention.',
              'candidate_reduction': False, 'datasets_piloted': names, 'dataset_identity': identity,
              'protocol_id': protocol['protocol_id'], 'protocol_hash': config_id(protocol)}
    write_json(output/'pilot.json', report)
    return report


# The calibration of a pilot second into a production job second. These four models have no completed production run, so
# the factor is measured on the closest models that do: the tuned classical comparators of the bridge_knn run for the
# three sklearn pipelines (fold-local sklearn pipelines under the same harness), and input_footrule_knn of the
# knn_training run for kendall_svc (the same seven encoders on the same rows).
CALIBRATION_SOURCE = {LDA_MODEL: 'classical', PCA_MODEL: 'classical', NCA_MODEL: 'classical', KENDALL_MODEL: INPUT_MODEL}
CLASSICAL_CALIBRATION_MODELS = ('svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting')
JOB_OVERHEAD_SECONDS = nd.JOB_OVERHEAD_SECONDS


def calibration(calibration_runs=None):
    """{model or 'classical': {'pooled', 'max'}} from the realized bridge_knn and knn_training runs."""
    entries = calibration_runs or nd.CALIBRATION
    factors = {}
    for label, models in (('bridge_knn', CLASSICAL_CALIBRATION_MODELS), ('knn_training', (INPUT_MODEL,))):
        pilot = json.loads(Path(entries[label]['pilot']).read_text())
        factors.update(nd.calibration_factors(entries[label]['run'], pilot, models))
    classical = {kind: max(factors[model][kind] for model in CLASSICAL_CALIBRATION_MODELS) if kind == 'max'
                 else float(np.mean([factors[model]['pooled'] for model in CLASSICAL_CALIBRATION_MODELS]))
                 for kind in ('pooled', 'max')}
    return {**factors, 'classical': {**classical, 'models': list(CLASSICAL_CALIBRATION_MODELS),
                                     'definition': 'central: the mean pooled factor of the five tuned classical '
                                                   'comparators of the bridge_knn run; upper: the largest of their max '
                                                   'factors'}}


def hours(durations, workers):
    return {'serial_hours': sum(durations) / 3600, 'serial_over_workers_hours': sum(durations) / 3600 / workers,
            'simulated_makespan_hours': nd.makespan(durations, workers) / 3600}


def projection(protocol, pilot_path, calibration_runs=None, workers=WORKERS):
    """The calibrated projection: every job priced as newdata.projection prices a piloted dataset (fits_per_outer x mean
    pilot seconds x the calibration factor, central pooled and upper max, plus the job overhead); the decision is the
    central simulated makespan of the run jobs at `workers` plus STAGE_OVERHEAD_HOURS."""
    pilot_path = Path(pilot_path)
    pilot = json.loads(pilot_path.read_text())
    if pilot.get('protocol_hash') != config_id(protocol):
        raise ValueError('The pilot did not run with this protocol')
    factors = calibration(calibration_runs)
    names = list(protocol['datasets'])
    means = nd.pilot_means(pilot)
    registry = build_registry(protocol)
    per_outer = {model: nd.fits_per_outer(len(spec.candidates), spec.stochastic, protocol['inner_folds'])
                 for model, spec in registry.items()}
    missing = [f'{name} {model}' for name in names for model in MODEL_ORDER if (name, model) not in means]
    if missing:
        raise ValueError(f'The pilot must time every model on every dataset (missing {missing})')
    jobs = {(name, model): {kind: factors[CALIBRATION_SOURCE[model]][factor] * per_outer[model] * means[name, model]
                                  + JOB_OVERHEAD_SECONDS for kind, factor in KINDS.items()}
            for name in names for model in MODEL_ORDER}
    folds = protocol['outer_folds'] * protocol['outer_repeats']
    order = [jobs[name, model] for name in names for _ in range(folds) for model in MODEL_ORDER]
    run_hours = {kind: hours([entry[kind] for entry in order], workers) for kind in KINDS}
    total = {kind: run_hours[kind]['simulated_makespan_hours'] + STAGE_OVERHEAD_HOURS for kind in KINDS}
    harness = sum(pilot['workload_estimates'][model]['serial_panel_seconds_using_observed_max']
                  for model in MODEL_ORDER) / 3600 / workers
    return _plain({
        'purpose': f'{FAMILY}_calibrated_projection', 'family': FAMILY, 'protocol_id': protocol['protocol_id'],
        'protocol_hash': config_id(protocol), 'cap_hours': CAP_HOURS, 'workers': workers,
        'decision': f'calibrated central simulated makespan at {workers} workers of the run jobs plus '
                    f'{STAGE_OVERHEAD_HOURS} h of stage overhead (prepare, reporting and the analysis)',
        'decision_hours': total['central'], 'upper_hours': total['upper'], 'within_cap': total['central'] <= CAP_HOURS,
        'run': run_hours, 'stage_overhead_hours': STAGE_OVERHEAD_HOURS, 'harness_max_based_run_hours': harness,
        'per_dataset': {name: {'central_serial_hours': folds * sum(jobs[name, m]['central'] for m in MODEL_ORDER) / 3600,
                               'upper_serial_hours': folds * sum(jobs[name, m]['upper'] for m in MODEL_ORDER) / 3600,
                               'central_job_minutes': {m: jobs[name, m]['central'] / 60 for m in MODEL_ORDER},
                               'central_serial_hours_by_model': {m: folds * jobs[name, m]['central'] / 3600
                                                                 for m in MODEL_ORDER}} for name in names},
        'calibration': factors, 'calibration_source': dict(CALIBRATION_SOURCE), 'fits_per_outer': per_outer,
        'job_seconds': {f'{name}|{model}': jobs[name, model] for name in names for model in MODEL_ORDER},
        'assumptions': 'the calibration factors were measured under the 16 single-thread workers of the bridge_knn and '
                       'knn_training production runs and are applied here at the same worker count; these four models '
                       'have no completed production run, so the factor comes from the closest models that do; the pilot '
                       'itself ran while other production runs held the machine, so its seconds are already inflated by '
                       'contention and the projection is conservative in that direction',
        'sources': {'pilot': {'path': str(pilot_path), 'sha256': nd.sha256_file(pilot_path)},
                    **{label: {'run': str(entry['run']), 'pilot': str(entry['pilot']),
                               'pilot_sha256': nd.sha256_file(entry['pilot'])}
                       for label, entry in (calibration_runs or nd.CALIBRATION).items()}}})


def freeze(draft_path, projection_path, stages_path, output_path=None, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the calibrated projection of that draft's pilot is within
    the cap; an existing file with different content is never replaced."""
    from .run_revision import write_json
    draft = json.loads(Path(draft_path).read_text())
    if canonical_json(draft) != canonical_json(draft_protocol()):
        raise ValueError('The draft differs from neighbour_baselines.draft_protocol(); the stages must have used the '
                         'committed draft')
    plan, stages = json.loads(Path(projection_path).read_text()), json.loads(Path(stages_path).read_text())
    if plan.get('protocol_hash') != config_id(draft) or plan.get('family') != FAMILY or plan.get('workers') != WORKERS:
        raise ValueError('The projection was not computed from a pilot of this draft at the family workers')
    if not plan.get('within_cap') or not 0 < plan['decision_hours'] <= CAP_HOURS:
        raise ValueError(f'Not frozen: the calibrated projection {plan["decision_hours"]:.2f} h exceeds the {CAP_HOURS} h '
                         f'cap at {WORKERS} workers')
    refused = draft['kendall_kernel']['refused']
    if refused:
        raise ValueError(f'Not frozen: {KENDALL_MODEL} refuses {", ".join(refused)} under the declared kernel cap')
    record = {'cap_hours': CAP_HOURS, 'workers': WORKERS, 'decision': plan['decision'],
              'decision_hours': plan['decision_hours'], 'upper_hours': plan['upper_hours'], 'run': plan['run'],
              'stage_overhead_hours': plan['stage_overhead_hours'],
              'harness_max_based_run_hours': plan['harness_max_based_run_hours'],
              'per_dataset': {name: {key: entry[key] for key in ('central_serial_hours', 'upper_serial_hours')}
                              for name, entry in plan['per_dataset'].items()},
              'calibration_factors': {model: {'pooled': entry['pooled'], 'max': entry['max']}
                                      for model, entry in plan['calibration'].items()},
              'calibration_source': plan['calibration_source'], 'fits_per_outer': plan['fits_per_outer'],
              'assumptions': plan['assumptions'], 'projection_sha256': nd.sha256_file(projection_path), 'stages': stages}
    text = (f'Neighbour baselines (external review 2026-09-14): drafted from newdata_batch1.json with the identical '
            f'nested design over all seventeen registered datasets and the four nearest baselines; {stages["summary"]}; '
            f'projected at {WORKERS} single-thread workers as the calibrated central simulated makespan of the run '
            f'({plan["run"]["central"]["simulated_makespan_hours"]:.2f} h) plus {plan["stage_overhead_hours"]} h of '
            f'stage overhead: {plan["decision_hours"]:.2f} h (upper {plan["upper_hours"]:.2f} h; harness max-based run '
            f'{plan["harness_max_based_run_hours"]:.2f} h); cap {CAP_HOURS} h; every dataset within the Kendall kernel '
            'cap, so no dataset is refused; frozen after the pilot')
    protocol = validate_protocol(_plain(dict(draft, frozen=True,
                                             frozen_at_utc=frozen_at_utc or datetime.now(timezone.utc).isoformat(),
                                             status=FROZEN_STATUS, resource_decision=text, projection=record)))
    write_json(Path(output_path or PROTOCOL_FILE), protocol)
    return protocol


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

SMOKE_DESIGN = {'outer_folds': 3, 'outer_repeats': 1, 'inner_folds': 2}
SMOKE_PANEL = ({'name': 'syn_a', 'samples': 120, 'features': 4}, {'name': 'syn_b', 'samples': 90, 'features': 4})
SMOKE_REFERENCE_LABEL = 'knn'


def smoke_panel_declaration(panel=SMOKE_PANEL):
    return _plain([{'name': entry['name'], 'panel': 'benchmark', 'loader': 'synthetic smoke dataset',
                    'reference': SMOKE_REFERENCE_LABEL, 'shape': [entry['samples'], entry['features']], 'n_classes': 3,
                    'dataset_hash': None, 'splits_hash': None} for entry in panel])


def smoke_protocol(panel=SMOKE_PANEL, design=SMOKE_DESIGN):
    """A frozen synthetic smoke protocol (frozen only to pass reporting's gate; purpose synthetic_smoke_only)."""
    declared = smoke_panel_declaration(panel)
    reference = {SMOKE_REFERENCE_LABEL: {**REFERENCE_RUNS[SMOKE_REFERENCE_LABEL], 'model_id': TRAINED_MODEL,
                                         'datasets': [entry['name'] for entry in declared]}}
    protocol = dict(draft_protocol(), **design, protocol_id=f'{PROTOCOL_ID}-synthetic-smoke',
                    purpose='synthetic_smoke_only', registry=SMOKE_REGISTRY,
                    datasets=[entry['name'] for entry in declared], panel=declared, reference=_plain(reference),
                    kendall_kernel=kendall_kernel_declaration(declared), primary_family_size=len(declared),
                    analysis=analysis_declaration(declared), frozen=True, frozen_at_utc='2026-09-14T00:00:00+00:00',
                    status='synthetic_smoke_only_not_evidence', resource_decision='synthetic smoke only', projection=None)
    return validate_protocol(_plain(protocol))


def write_smoke_dataset(directory, entry, protocol, index):
    """A three-class synthetic dataset in run_revision's prepared layout."""
    from .run_revision import write_json
    rng = np.random.RandomState(71 + index)
    y = np.tile([0, 1, 2], entry['samples'] // 3)
    X = rng.randn(len(y), entry['features'])
    X[np.arange(len(y)), y % entry['features']] += 1.5
    features, labels = [f'x{i}' for i in range(entry['features'])], ['0', '1', '2']
    splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'],
                        protocol['split_seed'])
    manifest = {'dataset_id': entry['name'], 'purpose': 'synthetic_smoke_only', 'source': 'synthetic smoke dataset',
                'feature_names': features, 'label_map': labels, 'shape': list(X.shape),
                'class_counts': np.bincount(y).tolist(), 'sample_order': 'source row order; zero-based sample_id',
                'dataset_hash': dataset_fingerprint(X, y, features, labels), 'splits_hash': config_id(splits)}
    target = Path(directory)/entry['name']
    write_json(target/'manifest.json', manifest)
    write_json(target/'splits.json', splits)
    if not (target/'data.npz').exists():
        np.savez_compressed(target/'data.npz', X=X, y=y)


def run_smoke_family(output, protocol, registry_path, workers=1, panel=SMOKE_PANEL):
    """run_revision's prepare, run and reporting stages for one synthetic family, through the harness's own worker."""
    from .reporting import summarize_verified_results
    from .run_revision import _worker, environment_record, get_registry, planned_jobs, write_json
    output = Path(output)
    registry = get_registry(registry_path, protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', nd.candidate_record(registry))
    write_json(output/'environment.json', environment_record(registry_path))
    for index, entry in enumerate(panel):
        write_smoke_dataset(output, entry, protocol, index)
    write_json(output/'planned_jobs.json', planned_jobs(protocol['datasets'], protocol, registry))
    jobs = [(str(output), name, index, model, registry_path) for name in protocol['datasets']
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for _ in pool.map(_worker, jobs):
            pass
    write_json(output/'summary.json', summarize_verified_results(output))
    return output


def smoke_reference_protocol(protocol, candidates):
    """The synthetic ArrowFlow-kNN run the smoke analysis pairs with: the same synthetic datasets, folds and seeds."""
    return dict(protocol, protocol_id=f'{PROTOCOL_ID}-synthetic-smoke-reference',
                registry='experiments.make_revision.neighbour_baselines:smoke_reference_registry',
                smoke_reference_candidates=list(candidates))


def smoke_reference_candidates():
    """arrowflow_full_knn at the two one-iteration bridge candidates the newdata smoke uses."""
    from .bridge import bridge_candidates
    return [dict(candidate, iterations=1) for candidate in bridge_candidates()
            if candidate['learning_rate'] == .1 and candidate['widths'] == [128] and candidate['embed_scale'] == 1]


def smoke(output, workers=3, panel=SMOKE_PANEL, reference_candidates=None):
    """A synthetic reference ArrowFlow-kNN run, a synthetic baselines run and compare_baselines analyse; never evidence."""
    from .compare_baselines import analyse
    from .run_revision import execution_lock, write_json
    output = Path(output)
    protocol = smoke_protocol(panel)
    reference_candidates = smoke_reference_candidates() if reference_candidates is None else list(reference_candidates)
    with execution_lock():
        reference = run_smoke_family(output/'reference', smoke_reference_protocol(protocol, reference_candidates),
                                     'experiments.make_revision.neighbour_baselines:smoke_reference_registry', workers, panel)
        run_dir = run_smoke_family(output/'run', protocol, protocol['registry'], workers, panel)
    result = analyse(run_dir, output/'analysis', references={SMOKE_REFERENCE_LABEL: reference}, allow_smoke=True)
    record = {'purpose': 'synthetic_smoke_only_not_paper_evidence', 'reference': str(reference), 'run': str(run_dir),
              'analysis': str(output/'analysis'), 'families': {model: len(rows) for model, rows in result['families'].items()},
              'interpretation': result['interpretation'], 'outputs': sorted(result['outputs'])}
    write_json(output/'smoke.json', record)
    return record


def main(argv=None):
    from .run_revision import write_json
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    drafted = commands.add_parser('draft', help='write the unfrozen stage protocol')
    drafted.add_argument('--output', type=Path, required=True)
    prepared = commands.add_parser('prepare', help='prepare the pinned datasets in run_revision layout')
    prepared.add_argument('--protocol', type=Path, required=True)
    prepared.add_argument('--output', type=Path, required=True)
    prepared.add_argument('--dataset', nargs='+')
    smoked = commands.add_parser('smoke', help='synthetic run, reference run and analysis (never evidence)')
    smoked.add_argument('--output', type=Path, required=True)
    smoked.add_argument('--workers', type=int, default=3)
    piloted = commands.add_parser('pilot', help='training-only runtime pilot of every dataset')
    piloted.add_argument('--protocol', type=Path, required=True)
    piloted.add_argument('--output', type=Path, required=True)
    piloted.add_argument('--dataset', nargs='+')
    projected = commands.add_parser('project', help='the calibrated projection against the cap')
    projected.add_argument('--protocol', type=Path, required=True)
    projected.add_argument('--pilot', type=Path, required=True)
    projected.add_argument('--output', type=Path, required=True)
    frozen = commands.add_parser('freeze', help='write the frozen protocol if the projection is within the cap')
    frozen.add_argument('--draft', type=Path, required=True)
    frozen.add_argument('--projection', type=Path, required=True)
    frozen.add_argument('--stages', type=Path, required=True)
    frozen.add_argument('--output', type=Path)
    running = commands.add_parser('run', help='run the prepared frozen protocol')
    running.add_argument('--protocol', type=Path, required=True)
    running.add_argument('--output', type=Path, required=True)
    running.add_argument('--workers', type=int, default=WORKERS)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        write_json(args.output, draft_protocol())
    elif args.command == 'prepare':
        prepare(args.output, json.loads(args.protocol.read_text()), args.dataset)
    elif args.command == 'smoke':
        if not 1 <= args.workers <= MAX_WORKERS:
            raise ValueError('Worker count must be between 1 and 16')
        record = smoke(args.output, args.workers)
        print(json.dumps({key: record[key] for key in ('purpose', 'families', 'outputs')}, indent=2))
    elif args.command == 'pilot':
        report = runtime_pilot(args.output, json.loads(args.protocol.read_text()), args.dataset)
        print(json.dumps(report['workload_estimates'], indent=2))
    elif args.command == 'project':
        plan = projection(json.loads(args.protocol.read_text()), args.pilot)
        write_json(args.output, plan)
        print(f'decision {plan["decision_hours"]:.2f} h (upper {plan["upper_hours"]:.2f} h) against the {CAP_HOURS} h cap '
              f'at {plan["workers"]} workers: {"within" if plan["within_cap"] else "ABOVE"} the cap')
    elif args.command == 'freeze':
        protocol = freeze(args.draft, args.projection, args.stages, args.output)
        print(f'frozen {protocol["protocol_id"]} at {protocol["frozen_at_utc"]}')
    else:
        run(args.protocol, args.output, args.workers)


if __name__ == '__main__':
    main()
