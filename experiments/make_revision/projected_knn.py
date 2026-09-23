"""Projected-score kNN control for ArrowFlow-kNN (Task 23A; protocols/2026-09-12/knn_projected.json).

projected_numeric_knn  MultiViewProjectedKNN: the seven encoders of input_footrule_knn (MultiViewInputKNN, itself the
                       encoders of MultiViewFootruleKNN and ArrowFlow-kNN: view seeds derive_seed(seed, 'view', v), the
                       strategy cycle target_aware, random, calibrated, and embed_dim and degree resolved from the training
                       partition by bridge.resolve) with each view's nearest-neighbour classifier reading the real-valued
                       projected scores that the encoder would otherwise sort into a ranking; majority vote.

The pre-sort array. OrdinalEncoder.transform(X) returns arrowflow.ranking.score_order(S) with
    S = OrdinalEncoder._scores_from_scaled(scaler_.transform(poly_.transform(imputer_.transform(X))))
where poly_ is absent at degree 1 and _scores_from_scaled is the encoder's own projection: for target_aware views the LDA
block over lda_scale_ beside the random block over random_scale_, and for calibrated views the calibration StandardScaler
the encoder fitted on its training rows. projected_scores returns S itself, with no further standardization, and fails
unless score_order(S) equals the encoded ranking of the same rows exactly; the check runs on the training rows of every
fit and on every query.

Per view, StableNumericKNN (StableFootruleKNN's stable neighbour order and vote on real-valued rows under the Minkowski
distance of order p) is tuned by select_numeric_readout over NUMERIC_READOUT_GRID (n_neighbors x weights x p, the 20
settings of the numeric_knn comparator's grid) on stratified splits of the training rows only, with random_state
derive_seed(view seed, 'readout_selection') as ArrowFlow-kNN's readout selection derives it, then refitted on all
training rows. The candidates are input_footrule_knn's four (embed_scale x degree_offset). Nothing here depends on a
dataset: the encoder follows the training partition's shape and the classifier its rows.

python -m experiments.make_revision.projected_knn smoke --output O [--workers 3]
    synthetic end-to-end exercise: a synthetic knn run (ArrowFlow-kNN and numeric_knn), a synthetic knn_training run
    (both training controls) and a synthetic knn_projected run through run_revision's worker and reporting, then
    compare_runs projected; never evidence
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from arrowflow.ranking import score_order
from .bridge import arrowflow_full_knn_factory, bridge_candidates, resolve
from .comparisons import StableFootruleKNN, conventional_registry, derive_seed
from .evaluation import ModelSpec, candidate_grid, config_id
from .knn_controls import (ABSTRACT_KEYS, CANDIDATE_KEYS as CONTROL_CANDIDATE_KEYS, INPUT_MODEL, PROTOCOLS, SELECTION_RULE,
                           SMOKE_DATASET, SMOKE_DESIGN, TRAINED_MODEL, UNTRAINED_MODEL, control_candidates, reference_pins,
                           run_synthetic_family, sha256_file, view_selections)
from .models import OrdinalEncoder
from .multiview import KNN_SELECTION_FOLDS, view_strategy
from .secondary_studies import majority

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core: the sources the
# knn and knn_training runs sealed (knn_controls.py holds input_footrule_knn's candidates and readout record helpers).
SOURCE_MODULES = ['experiments.make_revision.bridge', 'experiments.make_revision.multiview',
                  'experiments.make_revision.comparisons', 'experiments.make_revision.datasets',
                  'experiments.make_revision.secondary_studies', 'experiments.make_revision.knn_controls']

PROJECTED_MODEL = 'projected_numeric_knn'
RAW_MODEL = 'numeric_knn'
CANDIDATE_KEYS = CONTROL_CANDIDATE_KEYS[INPUT_MODEL]
PRIMARY_CONTRASTS = [f'{INPUT_MODEL}_vs_{PROJECTED_MODEL}', f'{TRAINED_MODEL}_vs_{PROJECTED_MODEL}']
DESCRIPTIVE_CONTRAST = f'{RAW_MODEL}_vs_{PROJECTED_MODEL}'
LADDER = (('raw_numeric_knn', RAW_MODEL, 'knn'), ('projected_numeric_knn', PROJECTED_MODEL, 'projected'),
          ('encoded_ranking_knn', INPUT_MODEL, 'training'), ('untrained_arrowflow_knn', UNTRAINED_MODEL, 'training'),
          ('arrowflow_knn', TRAINED_MODEL, 'knn'))
REFERENCE_MODELS = {'knn': [TRAINED_MODEL, RAW_MODEL], 'training': [INPUT_MODEL, UNTRAINED_MODEL]}
REFERENCE_PINS = ('protocol_id', 'protocol_sha256', 'code_revision', 'summary_sha256')
NUMERIC_READOUT_GRID = {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
MINKOWSKI_METRICS = {1: 'cityblock', 2: 'euclidean'}
PRESORT_REPRESENTATION = (
    "per view, the real-valued score matrix the view's OrdinalEncoder passes to arrowflow.ranking.score_order in transform: "
    "_scores_from_scaled(scaler_.transform(poly_.transform(imputer_.transform(X)))), poly_ only when degree > 1; "
    "_scores_from_scaled is the encoder's projection (for target_aware views the LDA block over lda_scale_ beside the random "
    "block over random_scale_) followed, for calibrated views, by the encoder's own calibration StandardScaler fitted on the "
    "training rows; no further standardization")
PRESORT_CHECK = ('score_order of the scores equals OrdinalEncoder.transform of the same rows exactly, on the training rows of '
                 'every fit and on every query; otherwise the fit or prediction fails (PresortMismatch)')
CLASSIFIER_RULE = ('StableNumericKNN: scipy cdist cityblock (p = 1) or euclidean (p = 2) to the training scores; a stable sort '
                   'of each distance row, so a cutoff tie goes to the earlier training row in source order; the votes of '
                   'StableFootruleKNN.predict_neighbors (uniform, or inverse distance with exact zero distances taking all '
                   'the weight; a vote tie goes to the lowest class)')


class PresortMismatch(RuntimeError):
    """The scores handed to the classifier do not sort into the encoder's ranking of the same rows."""


def projected_scores(encoder, X):
    """The real-valued score matrix a fitted OrdinalEncoder sorts in transform, returned without further standardization.

    Raises PresortMismatch unless score_order of the matrix equals encoder.transform(X) exactly."""
    check_is_fitted(encoder, 'projection_')
    values = encoder.imputer_.transform(X)
    if encoder.poly_ is not None:
        values = encoder.poly_.transform(values)
    scores = encoder._scores_from_scaled(encoder.scaler_.transform(values))
    if not np.array_equal(score_order(scores), encoder.transform(X)):
        raise PresortMismatch('The projected scores do not sort into the encoded ranking of the same rows')
    return scores


class StableNumericKNN(StableFootruleKNN):
    """StableFootruleKNN's neighbour order and vote on real-valued rows under the Minkowski distance of order p.

    Distances are scipy cdist 'cityblock' (p = 1) or 'euclidean' (p = 2) to the training rows. Each distance row is
    sorted stably, so a cutoff tie goes to the training row that comes first in source order (ascending sample_ids when
    given, otherwise the order of the rows given), and the first k neighbours at a larger k are the k neighbours. Votes
    are StableFootruleKNN.predict_neighbors: uniform, or inverse distance where exact zero distances take all the
    weight; a vote tie goes to the lowest class."""
    def __init__(self, n_neighbors=5, weights='uniform', p=2, batch_size=128):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.batch_size = batch_size

    def _metric(self):
        if isinstance(self.p, bool) or not isinstance(self.p, (int, np.integer)) or int(self.p) not in MINKOWSKI_METRICS:
            raise ValueError('Minkowski order p must be 1 or 2')
        return MINKOWSKI_METRICS[int(self.p)]

    def _positions(self, X):
        """The rows as float64 scores (the parent stores its fitted rows under this name)."""
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError('Expected a nonempty finite real-valued matrix')
        return values

    def fit(self, X, y, sample_ids=None):
        self._metric()
        return super().fit(X, y, sample_ids=sample_ids)

    def _neighbors(self, X):
        check_is_fitted(self, 'positions_')
        metric = self._metric()
        start = time.perf_counter()
        query = self._positions(X)
        if query.shape[1] != self.positions_.shape[1]:
            raise ValueError('Score dimension changed')
        self.last_encoding_seconds_ = time.perf_counter() - start
        k = min(self.n_neighbors, len(self.positions_))
        for begin in range(0, len(query), self.batch_size):
            distances = cdist(query[begin:begin+self.batch_size], self.positions_, metric=metric)
            nearest = np.argsort(distances, axis=1, kind='stable')[:, :k]
            yield np.take_along_axis(distances, nearest, axis=1), self.source_order_[nearest]


def numeric_readout_candidates():
    """The 20 settings of NUMERIC_READOUT_GRID in canonical config_id order."""
    return candidate_grid(NUMERIC_READOUT_GRID, budget=20)


def select_numeric_readout(scores, y, *, seed, candidates=None, folds=KNN_SELECTION_FOLDS):
    """Choose n_neighbors, weights and p for StableNumericKNN on `scores`, the rows of a training partition, by mean
    accuracy over StratifiedKFold(folds, shuffle=True, random_state=seed) splits of those rows only: at most `folds`
    splits, fewer when the smallest class has fewer rows. Per split and p, one neighbour cache at the largest k serves
    every candidate of that p; ties go to the lowest canonical config_id. This is multiview.select_knn_readout with the
    Minkowski order added: the same splits for the same seed and labels, the same cache and the same tie rule."""
    scores, y = np.asarray(scores, dtype=np.float64), np.asarray(y)
    candidates = numeric_readout_candidates() if candidates is None else list(candidates)
    folds = min(int(folds), int(np.unique(y, return_counts=True)[1].min()))
    if folds < 2:
        raise ValueError('Readout selection needs at least two rows per class in the training partition')
    accuracy = {config_id(c): [] for c in candidates}
    for a, b in StratifiedKFold(folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        for p in sorted({c['p'] for c in candidates}):
            members = [c for c in candidates if c['p'] == p]
            cache = StableNumericKNN(n_neighbors=max(c['n_neighbors'] for c in members), p=p).fit(scores[a], y[a], sample_ids=a)
            distances, indices = cache.kneighbors(scores[b])
            for config in members:
                cache.n_neighbors, cache.weights = config['n_neighbors'], config['weights']
                accuracy[config_id(config)].append(float(np.mean(cache.predict_neighbors(distances, indices) == y[b])))
    means = {cid: float(np.mean(values)) for cid, values in accuracy.items()}
    negative, cid, config = min((-means[config_id(c)], config_id(c), c) for c in candidates)
    return {'config': config, 'config_id': cid, 'inner_score': -negative, 'folds': folds, 'candidate_scores': means}


class MultiViewProjectedKNN(ClassifierMixin, BaseEstimator):
    """The seven encoders of MultiViewInputKNN with a numeric kNN per view on the pre-sort projected scores; majority vote.

    View v: OrdinalEncoder(view_strategy(strategy, v), embed_dim, degree, lda_ratio, derive_seed(seed, 'view', v)) fitted
    on the training rows exactly as MultiViewInputKNN fits it. The view's classifier input is projected_scores(encoder, X),
    the matrix that encoder's transform would sort. Its readout is select_numeric_readout on the training rows with
    random_state derive_seed(view seed, 'readout_selection'), then StableNumericKNN at the chosen setting refitted on all
    training rows."""
    def __init__(self, n_views=7, strategy='diverse', embed_dim=32, degree=2, aggregation='majority', lda_ratio=.3,
                 seed=8129):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)

    def fit(self, X, y):
        if self.aggregation != 'majority':
            raise ValueError('MultiViewProjectedKNN combines the per-view kNN votes by majority only')
        self.classes_ = np.unique(y)
        self.views_, self.readout_selections_ = [], []
        self.readout_seconds_ = encoding = 0.
        for v in range(self.n_views):
            seed_v = derive_seed(self.seed, 'view', v)
            start = time.perf_counter()
            enc = OrdinalEncoder(view_strategy(self.strategy, v), self.embed_dim, self.degree, self.lda_ratio, seed_v).fit(X, y)
            scores = projected_scores(enc, X)
            encoding += time.perf_counter() - start
            start = time.perf_counter()
            selection = select_numeric_readout(scores, y, seed=derive_seed(seed_v, 'readout_selection'))
            readout = StableNumericKNN(**selection['config']).fit(scores, y)
            self.readout_seconds_ += time.perf_counter() - start
            self.views_.append((enc, readout))
            self.readout_selections_.append(selection)
        self.encoding_seconds_ = encoding
        self.training_seconds_ = 0.
        return self

    def predict_views(self, X):
        check_is_fitted(self, 'views_')
        start = time.perf_counter()
        scores = [projected_scores(enc, X) for enc, _ in self.views_]
        self.last_encoding_seconds_ = time.perf_counter() - start
        return [readout.predict(s) for (_, readout), s in zip(self.views_, scores)]

    def predict(self, X):
        return majority(self.predict_views(X))

    def readout_record(self):
        return {'readout': 'knn_projected_scores', 'representation': PRESORT_REPRESENTATION, 'presort_check': PRESORT_CHECK,
                'classifier': CLASSIFIER_RULE, 'grid': NUMERIC_READOUT_GRID, 'selection_folds': KNN_SELECTION_FOLDS,
                'selection': SELECTION_RULE, 'score_dimension': self.embed_dim, 'readout_seconds': self.readout_seconds_,
                'views': view_selections(self.readout_selections_)}


class AdaptiveProjectedKNN(ClassifierMixin, BaseEstimator):
    """Resolves embed_dim and degree from the training partition's shape as input_footrule_knn does (bridge.resolve), then
    fits MultiViewProjectedKNN. The resolved augment is dropped: nothing is trained."""
    def __init__(self, config=None, seed=8129):
        self.config = config
        self.seed = seed

    def fit(self, X, y):
        cfg = dict(self.config)
        if set(cfg) != set(CANDIDATE_KEYS):
            raise ValueError(f'{PROJECTED_MODEL} configurations hold exactly {sorted(CANDIDATE_KEYS)}, not {sorted(cfg)}')
        resolved = resolve(cfg, X.shape[1], len(y))
        self.resolved_ = {'embed_dim': resolved['embed_dim'], 'degree': resolved['degree']}
        params = {key: value for key, value in cfg.items() if key not in ABSTRACT_KEYS}
        self.model_ = MultiViewProjectedKNN(**params, **self.resolved_, seed=self.seed).fit(X, y)
        self.encoder_ = self.model_.views_[0][0]
        self.classes_ = self.model_.classes_
        self.encoding_seconds_ = self.model_.encoding_seconds_
        self.training_seconds_ = 0.
        self.readout_seconds_ = self.model_.readout_seconds_
        self.readout_selections_ = self.model_.readout_selections_
        self.representation_metadata_ = self.model_.readout_record()
        return self

    def predict(self, X):
        out = self.model_.predict(X)
        self.last_encoding_seconds_ = self.model_.last_encoding_seconds_
        return out


# ----------------------------------------------------------------------------- candidates, protocol and registry

def projected_candidates(reference_candidates=None):
    """input_footrule_knn's candidates: ArrowFlow-kNN's candidates (default: the 16 bridge candidates) projected onto the
    encoder keys, duplicates removed, in canonical config_id order (knn_controls.control_candidates)."""
    return control_candidates(INPUT_MODEL, reference_candidates)


def projected_factory(config, seed):
    return AdaptiveProjectedKNN(config=config, seed=seed)


def ladder_declaration():
    return [{'rung': rung, 'model_id': model, 'run': run} for rung, model, run in LADDER]


def validate_projected_protocol(p):
    """Refuse a protocol whose declared model, readout, contrasts, descriptive contrast, ladder or references differ from
    this module."""
    block = p.get('projected_control')
    if not isinstance(block, dict):
        raise ValueError('A knn_projected protocol declares a projected_control block')
    if p.get('primary_contrasts') != PRIMARY_CONTRASTS:
        raise ValueError(f'primary_contrasts must be {PRIMARY_CONTRASTS}')
    datasets = p.get('datasets')
    if (not isinstance(datasets, list) or not datasets or type(p.get('primary_family_size')) is not int
            or p['primary_family_size'] != len(PRIMARY_CONTRASTS) * len(datasets)):
        raise ValueError('primary_family_size must be two contrasts per dataset')
    model = block.get('model') or {}
    if (model.get('model_id') != PROJECTED_MODEL or model.get('candidate_keys') != list(CANDIDATE_KEYS)
            or model.get('candidates') != len(projected_candidates()) or model.get('stochastic') is not True):
        raise ValueError('projected_control.model disagrees with the registered model and candidates')
    readout = block.get('readout') or {}
    if readout.get('grid') != NUMERIC_READOUT_GRID or readout.get('selection_folds') != KNN_SELECTION_FOLDS:
        raise ValueError('projected_control.readout must declare the numeric readout grid and selection folds')
    descriptive = block.get('descriptive') or {}
    if descriptive.get('contrast') != DESCRIPTIVE_CONTRAST or 'outside the Holm family' not in str(descriptive.get('status')):
        raise ValueError(f'projected_control.descriptive must declare {DESCRIPTIVE_CONTRAST} outside the Holm family')
    if block.get('ladder') != ladder_declaration():
        raise ValueError('projected_control.ladder must list the five rungs in order')
    references = block.get('references') or {}
    for label, models in REFERENCE_MODELS.items():
        pins = references.get(label) or {}
        if pins.get('model_ids') != models or any(not isinstance(pins.get(key), str) or not pins[key] for key in REFERENCE_PINS):
            raise ValueError(f'projected_control.references.{label} must pin the {label} run ({", ".join(REFERENCE_PINS)}) '
                             f'and name {models}')
    return p


def knn_projected_registry(protocol):
    """projected_numeric_knn, stochastic (its encoders follow the fit seed), under run_revision's nested harness."""
    validate_projected_protocol(protocol)
    return {PROJECTED_MODEL: ModelSpec(PROJECTED_MODEL, projected_factory, projected_candidates(), True)}


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

SMOKE_KNN_REGISTRY = 'experiments.make_revision.projected_knn:smoke_knn_registry'


def smoke_knn_registry(protocol):
    """Synthetic smoke only: ArrowFlow-kNN at the smoke protocol's declared candidates beside the numeric_knn comparator."""
    if protocol.get('purpose') != 'synthetic_smoke_only':
        raise ValueError('smoke_knn_registry serves synthetic smoke protocols only')
    return {TRAINED_MODEL: ModelSpec(TRAINED_MODEL, arrowflow_full_knn_factory, protocol['smoke_reference_candidates'], True),
            RAW_MODEL: conventional_registry(protocol)[RAW_MODEL]}


def run_pins(directory, label):
    """The projected_control.references values that pin one complete run."""
    directory = Path(directory)
    return {'model_ids': list(REFERENCE_MODELS[label]),
            'protocol_id': json.loads((directory/'protocol.json').read_text())['protocol_id'],
            'protocol_sha256': sha256_file(directory/'protocol.json'), 'summary_sha256': sha256_file(directory/'summary.json'),
            'code_revision': json.loads((directory/'environment.json').read_text())['code_revision']}


def smoke_protocol(template, **fields):
    """A synthetic smoke protocol; frozen true only to pass reporting's gate, purpose synthetic_smoke_only."""
    return dict(template, **{**SMOKE_DESIGN, 'datasets': [SMOKE_DATASET], 'frozen': True, 'purpose': 'synthetic_smoke_only',
                             **fields})


def smoke(output, protocol, workers=3):
    """Three synthetic runs in the production layout, each summarized by reporting, then compare_runs projected: knn
    (ArrowFlow-kNN at the eight bridge candidates with learning rate 0.1 and one iteration, and numeric_knn), knn_training
    (both training controls at their real candidates) and knn_projected (projected_numeric_knn at its real candidates)."""
    from .compare_projected import compare_projected
    from .run_revision import execution_lock, write_json
    output = Path(output)
    training_template = json.loads((PROTOCOLS/'knn_training.json').read_text())
    reference_candidates = [dict(c, iterations=1) for c in bridge_candidates() if c['learning_rate'] == .1]
    with execution_lock():
        knn_protocol = smoke_protocol(json.loads((PROTOCOLS/'bridge_knn.json').read_text()), primary_family_size=1,
                                      protocol_id='arrowflow-v3-bridge-knn-1-synthetic-smoke', registry=SMOKE_KNN_REGISTRY,
                                      smoke_reference_candidates=reference_candidates)
        knn = run_synthetic_family(output/'knn', knn_protocol, SMOKE_KNN_REGISTRY, workers)
        controls = training_template['training_controls']
        training_protocol = smoke_protocol(training_template, primary_family_size=2,
                                           protocol_id=training_template['protocol_id'] + '-synthetic-smoke',
                                           training_controls=dict(controls, reference={**controls['reference'],
                                                                                       **reference_pins(knn)}))
        training = run_synthetic_family(output/'training', training_protocol, training_template['registry'], workers)
        block = protocol['projected_control']
        references = {label: {**block['references'][label], **run_pins(directory, label)}
                      for label, directory in (('knn', knn), ('training', training))}
        projected_protocol = smoke_protocol(protocol, primary_family_size=len(PRIMARY_CONTRASTS),
                                            protocol_id=protocol['protocol_id'] + '-synthetic-smoke',
                                            projected_control=dict(block, references=references))
        projected = run_synthetic_family(output/'projected', projected_protocol, protocol['registry'], workers)
        result = compare_projected(projected, knn, training, output/'compare')
    record = {'purpose': 'synthetic_smoke_only_not_paper_evidence', 'knn': str(knn), 'training': str(training),
              'projected': str(projected), 'comparison': str(output/'compare'), 'contrasts': result['contrasts'],
              'descriptive': result['descriptive'], 'ladder': result['ladder']['rows']}
    write_json(output/'smoke.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['smoke'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOLS/'knn_projected.json')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise ValueError('Worker count must be between 1 and 16')
    record = smoke(args.output, json.loads(args.protocol.read_text()), args.workers)
    for row in record['contrasts'] + record['descriptive']:
        holm = f" Holm p={row['holm_p_approximate']:.3g}" if 'holm_p_approximate' in row else ' (descriptive)'
        print(f"{row['dataset']}: {row['model_a']} - {row['model_b']} {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]{holm}")


if __name__ == '__main__':
    main()
