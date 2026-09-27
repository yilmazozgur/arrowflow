"""Training controls for ArrowFlow-kNN (Task 20A; protocols/2026-09-12/knn_training.json).

arrowflow_knn_untrained  UntrainedMultiViewArrowFlowKNN: the seven views of ArrowFlow-kNN (MultiViewArrowFlowKNN) with
                         every network kept at its seeded initial filters, no training update of any layer, and the
                         identical kNN readout selection on the untrained hidden rankings; majority vote.
input_footrule_knn       MultiViewInputKNN: the same seven encoders (the view seeds and strategy cycle of
                         MultiViewFootruleKNN) with each view's footrule kNN readout chosen by the same
                         select_knn_readout on the encoded input positions instead of a fixed k = 5; majority vote.

Both are selected under the nested design of bridge_knn.json among the distinct configurations of ArrowFlow-kNN's
candidate grid that still act without training: widths x embed_scale x degree_offset (8) for the untrained network and
embed_scale x degree_offset (4) for the input control. compare_runs training pairs them with ArrowFlow-kNN's outer
predictions from the bridge_knn run; depth_split describes a difference by the hidden widths selected in each outer fold.

python -m experiments.make_revision.knn_controls smoke --output O [--workers 3]
    synthetic end-to-end exercise of the family: a synthetic ArrowFlow-kNN reference run and the two controls through
    run_revision's worker and reporting, then compare_runs training; never evidence
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from arrowflow.ranking import inverse_positions
from .bridge import arrowflow_full_knn_factory, bridge_candidates, resolve
from .comparisons import StableFootruleKNN, derive_seed
from .evaluation import ModelSpec, config_id, dataset_fingerprint, make_splits, paired_corrected_interval
from .models import ArrowFlowEstimator, OrdinalEncoder
from .multiview import KNN_READOUT_GRID, KNN_SELECTION_FOLDS, MultiViewArrowFlowKNN, select_knn_readout, view_strategy
from .secondary_studies import majority

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core: the modules the
# reference bridge_knn run sealed through bridge.py, and bridge.py itself (candidate grid and resolution).
SOURCE_MODULES = ['experiments.make_revision.bridge', 'experiments.make_revision.multiview',
                  'experiments.make_revision.comparisons', 'experiments.make_revision.datasets',
                  'experiments.make_revision.secondary_studies']

TRAINED_MODEL = 'arrowflow_full_knn'
UNTRAINED_MODEL = 'arrowflow_knn_untrained'
INPUT_MODEL = 'input_footrule_knn'
CONTROL_MODELS = (UNTRAINED_MODEL, INPUT_MODEL)
PRIMARY_CONTRASTS = [f'{TRAINED_MODEL}_vs_{model}' for model in CONTROL_MODELS]
ABSTRACT_KEYS = ('embed_scale', 'degree_offset')
CANDIDATE_KEYS = {UNTRAINED_MODEL: ('n_views', 'strategy', 'aggregation', 'widths', 'embed_scale', 'degree_offset'),
                  INPUT_MODEL: ('n_views', 'strategy', 'aggregation', 'embed_scale', 'degree_offset')}
REFERENCE_PINS = ('model_id', 'protocol_id', 'protocol_sha256', 'code_revision', 'summary_sha256')
SELECTION_RULE = 'mean_accuracy_over_stratified_splits_of_the_training_partition; ties lowest_canonical_config_id'
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-12'


# ----------------------------------------------------------------------------- candidates

def project_candidates(candidates, keys):
    """The distinct projections of candidate configurations onto `keys`, in canonical config_id order."""
    projected = {}
    for config in candidates:
        missing = sorted(set(keys) - set(config))
        if missing:
            raise ValueError(f'Candidate {config_id(config)} lacks {missing}')
        value = {key: config[key] for key in keys}
        projected[config_id(value)] = value
    return [projected[cid] for cid in sorted(projected)]


def control_candidates(model, reference_candidates=None):
    """One control's candidates: ArrowFlow-kNN's candidates (default: the 16 bridge candidates) projected onto the keys
    that still act without training. learning_rate, iterations, batch_size and validation_ratio (the checkpoint) act
    only while a network trains, and the resolved augment only on training samples; the input control has no network,
    so widths go too."""
    return project_candidates(bridge_candidates() if reference_candidates is None else reference_candidates,
                              CANDIDATE_KEYS[model])


# ----------------------------------------------------------------------------- estimators

def view_selections(selections):
    return [{'view': v, 'config': s['config'], 'config_id': s['config_id'], 'inner_score': s['inner_score'],
             'folds': s['folds'], 'candidate_scores': s['candidate_scores']} for v, s in enumerate(selections)]


class UntrainedMultiViewArrowFlowKNN(MultiViewArrowFlowKNN):
    """MultiViewArrowFlowKNN whose networks keep their seeded initial filters.

    View v is encoded exactly as in MultiViewArrowFlow (OrdinalEncoder with view_strategy(strategy, v) and seed
    derive_seed(seed, 'view', v)). Its network is ArrowFlowEstimator.initialize_orders with that seed, which is the state
    MultiViewArrowFlow's fit_orders starts training from, and no layer is ever updated. The readout is
    MultiViewArrowFlowKNN's own (_fit_view_readout): select_knn_readout on the untrained hidden positions of the training
    rows with random_state derive_seed(view seed, 'readout_selection'), refitted on all of them; majority vote.
    Settings that act only during training (learning rate, iterations, batch size, validation checkpoint, augmentation)
    are not parameters: each network carries ArrowFlowEstimator's defaults, which a network that is never trained
    never reads.
    """
    def __init__(self, n_views=7, strategy='diverse', embed_dim=32, degree=2, widths=(128,), aggregation='majority',
                 lda_ratio=.3, seed=8129):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)

    def fit(self, X, y):
        if self.aggregation != 'majority':
            raise ValueError('UntrainedMultiViewArrowFlowKNN combines the per-view kNN votes by majority only')
        self.classes_ = np.unique(y)
        self.views_, self.readouts_, self.readout_selections_ = [], [], []
        self.readout_seconds_ = encoding = initialization = 0.
        for v in range(self.n_views):
            seed_v = derive_seed(self.seed, 'view', v)
            start = time.perf_counter()
            enc = OrdinalEncoder(view_strategy(self.strategy, v), self.embed_dim, self.degree, self.lda_ratio, seed_v).fit(X, y)
            orders = enc.transform(X)
            encoding += time.perf_counter() - start
            start = time.perf_counter()
            net = ArrowFlowEstimator(embed_dim=self.embed_dim, degree=self.degree, widths=self.widths,
                                     seed=seed_v).initialize_orders(orders, y)
            initialization += time.perf_counter() - start
            self.views_.append((enc, net))
            self._fit_view_readout(enc, net, orders, y, seed_v)
        if any(net.network_.update_iter != 0 for _, net in self.views_):
            raise RuntimeError('An untrained view network was updated')
        self.encoding_seconds_ = encoding
        self.initialization_seconds_ = initialization
        self.training_seconds_ = 0.
        return self

    def readout_record(self):
        return {'readout': 'knn_hidden', 'network_state': 'seeded_initial_filters_no_training_update',
                'representation': 'inverse positions of the final hidden ranking of each untrained view',
                'grid': KNN_READOUT_GRID, 'selection_folds': KNN_SELECTION_FOLDS, 'selection': SELECTION_RULE,
                'initialization_seconds': self.initialization_seconds_, 'readout_seconds': self.readout_seconds_,
                'views': view_selections(self.readout_selections_)}


class MultiViewInputKNN(ClassifierMixin, BaseEstimator):
    """The seven encoders of MultiViewFootruleKNN (same derived view seeds and strategy cycle) with each view's footrule
    kNN readout on the inverse positions of the encoded input ranking chosen as ArrowFlow-kNN chooses its readout on the
    hidden ranking: select_knn_readout with random_state derive_seed(view seed, 'readout_selection'), refitted on all
    training rows; majority vote."""
    def __init__(self, n_views=7, strategy='diverse', embed_dim=32, degree=2, aggregation='majority', lda_ratio=.3,
                 seed=8129):
        for name, value in locals().items():
            if name != 'self':
                setattr(self, name, value)

    def fit(self, X, y):
        if self.aggregation != 'majority':
            raise ValueError('MultiViewInputKNN combines the per-view kNN votes by majority only')
        self.classes_ = np.unique(y)
        self.views_, self.readout_selections_ = [], []
        self.readout_seconds_ = encoding = 0.
        for v in range(self.n_views):
            seed_v = derive_seed(self.seed, 'view', v)
            start = time.perf_counter()
            enc = OrdinalEncoder(view_strategy(self.strategy, v), self.embed_dim, self.degree, self.lda_ratio, seed_v).fit(X, y)
            positions = inverse_positions(enc.transform(X))
            encoding += time.perf_counter() - start
            start = time.perf_counter()
            selection = select_knn_readout(positions, y, seed=derive_seed(seed_v, 'readout_selection'))
            readout = StableFootruleKNN(**selection['config'], input_kind='positions').fit(positions, y)
            self.readout_seconds_ += time.perf_counter() - start
            self.views_.append((enc, readout))
            self.readout_selections_.append(selection)
        self.encoding_seconds_ = encoding
        self.training_seconds_ = 0.
        return self

    def predict_views(self, X):
        check_is_fitted(self, 'views_')
        start = time.perf_counter()
        positions = [inverse_positions(enc.transform(X)) for enc, _ in self.views_]
        self.last_encoding_seconds_ = time.perf_counter() - start
        return [readout.predict(p) for (_, readout), p in zip(self.views_, positions)]

    def predict(self, X):
        return majority(self.predict_views(X))

    def readout_record(self):
        return {'readout': 'knn_input', 'representation': 'inverse positions of each view\'s encoded input ranking',
                'grid': KNN_READOUT_GRID, 'selection_folds': KNN_SELECTION_FOLDS, 'selection': SELECTION_RULE,
                'readout_seconds': self.readout_seconds_, 'views': view_selections(self.readout_selections_)}


class AdaptiveKNNControl(ClassifierMixin, BaseEstimator):
    """Resolves embed_dim and degree from the training partition's shape as AdaptiveMultiView does (bridge.resolve),
    then fits the control. The resolved augment is dropped: nothing is trained."""
    model_class = None
    model_id = None

    def __init__(self, config=None, seed=8129):
        self.config = config
        self.seed = seed

    def fit(self, X, y):
        cfg = dict(self.config)
        keys = CANDIDATE_KEYS[self.model_id]
        if set(cfg) != set(keys):
            raise ValueError(f'{self.model_id} configurations hold exactly {sorted(keys)}, not {sorted(cfg)}')
        resolved = resolve(cfg, X.shape[1], len(y))
        self.resolved_ = {'embed_dim': resolved['embed_dim'], 'degree': resolved['degree']}
        params = {key: value for key, value in cfg.items() if key not in ABSTRACT_KEYS}
        self.model_ = self.model_class(**params, **self.resolved_, seed=self.seed).fit(X, y)
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


class AdaptiveUntrainedKNN(AdaptiveKNNControl):
    model_class = UntrainedMultiViewArrowFlowKNN
    model_id = UNTRAINED_MODEL


class AdaptiveInputKNN(AdaptiveKNNControl):
    model_class = MultiViewInputKNN
    model_id = INPUT_MODEL


def untrained_factory(config, seed):
    return AdaptiveUntrainedKNN(config=config, seed=seed)


def input_factory(config, seed):
    return AdaptiveInputKNN(config=config, seed=seed)


# ----------------------------------------------------------------------------- protocol and registry

def validate_depths(depths):
    if (not isinstance(depths, list) or not depths or any(not isinstance(d, list) or not d for d in depths)
            or any(type(w) is not int or w < 1 for d in depths for w in d) or len({tuple(d) for d in depths}) != len(depths)):
        raise ValueError('depth_split.depths must be a nonempty list of distinct nonempty lists of positive widths')
    return [list(d) for d in depths]


def validate_training_protocol(p):
    """Refuse a protocol whose declared controls, readout, contrasts or reference block differ from this module."""
    block = p.get('training_controls')
    if not isinstance(block, dict):
        raise ValueError('A knn_training protocol declares a training_controls block')
    if p.get('primary_contrasts') != PRIMARY_CONTRASTS:
        raise ValueError(f'primary_contrasts must be {PRIMARY_CONTRASTS}')
    if p.get('primary_family_size') != len(PRIMARY_CONTRASTS) * len(p['datasets']):
        raise ValueError('primary_family_size must be two contrasts per dataset')
    models = block.get('models')
    if not isinstance(models, dict) or sorted(models) != sorted(CONTROL_MODELS):
        raise ValueError(f'training_controls.models must declare exactly {list(CONTROL_MODELS)}')
    for model in CONTROL_MODELS:
        declared = models[model]
        if (declared.get('candidate_keys') != list(CANDIDATE_KEYS[model])
                or declared.get('candidates') != len(control_candidates(model)) or declared.get('stochastic') is not True):
            raise ValueError(f'training_controls.models.{model} disagrees with the registered candidates')
    readout = block.get('readout') or {}
    if readout.get('grid') != KNN_READOUT_GRID or readout.get('selection_folds') != KNN_SELECTION_FOLDS:
        raise ValueError('training_controls.readout must declare the ArrowFlow-kNN readout grid and selection folds')
    reference = block.get('reference') or {}
    if reference.get('model_id') != TRAINED_MODEL or any(not isinstance(reference.get(k), str) or not reference[k]
                                                         for k in REFERENCE_PINS):
        raise ValueError(f'training_controls.reference must pin the {TRAINED_MODEL} run ({", ".join(REFERENCE_PINS)})')
    validate_depths((block.get('depth_split') or {}).get('depths'))
    return p


def knn_training_registry(protocol):
    """The two training controls of ArrowFlow-kNN, both stochastic (their encoders and initial filters follow the fit
    seed), under run_revision's nested harness."""
    validate_training_protocol(protocol)
    return {UNTRAINED_MODEL: ModelSpec(UNTRAINED_MODEL, untrained_factory, control_candidates(UNTRAINED_MODEL), True),
            INPUT_MODEL: ModelSpec(INPUT_MODEL, input_factory, control_candidates(INPUT_MODEL), True)}


# ----------------------------------------------------------------------------- depth split (descriptive)

def fold_means(rows, model, metric='accuracy'):
    groups = {}
    for row in rows:
        if row['model_id'] == model:
            groups.setdefault((row['outer_repeat'], row['outer_fold']), []).append(row[metric])
    return {fold: float(np.mean(values)) for fold, values in groups.items()}


def selected_widths(rows, model):
    """{(outer_repeat, outer_fold): the hidden widths of the configuration `model` selected for that outer fit}."""
    widths = {}
    for row in rows:
        if row['model_id'] == model:
            fold, value = (row['outer_repeat'], row['outer_fold']), list(row['config']['widths'])
            if widths.setdefault(fold, value) != value:
                raise ValueError(f'{model} rows of outer fold {fold} disagree on the selected widths')
    return widths


def depth_split(rows, model_a, model_b, widths, *, depths, folds, seeds, q, confidence, metric='accuracy'):
    """model_a minus model_b by the hidden widths selected in each outer fold; descriptive, no p values.

    rows: one dataset's outer model rows of both models; widths: {(outer_repeat, outer_fold): widths}; seeds:
    {model: fitting seeds}. One entry per declared depth: its folds, their seed-averaged differences, n_folds, mean, SD
    (ddof 1), min, max and, with at least two folds, the corrected resampled t interval over those folds only."""
    folds = [tuple(fold) for fold in folds]
    depths = validate_depths(depths)
    unexpected = sorted({tuple(widths[fold]) for fold in folds} - {tuple(d) for d in depths})
    if unexpected:
        raise ValueError(f'Selected widths outside the declared depths: {unexpected}')
    means = {model: fold_means(rows, model, metric) for model in (model_a, model_b)}
    entries = []
    for depth in depths:
        chosen = [fold for fold in folds if list(widths[fold]) == depth]
        differences = [means[model_a][fold] - means[model_b][fold] for fold in chosen]
        entry = {'widths': depth, 'n_folds': len(chosen), 'folds': [list(fold) for fold in chosen],
                 'fold_differences': differences, 'mean_difference': float(np.mean(differences)) if chosen else None,
                 'sd': float(np.std(differences, ddof=1)) if len(chosen) > 1 else None,
                 'min': float(min(differences)) if chosen else None, 'max': float(max(differences)) if chosen else None,
                 'interval': None}
        if len(chosen) >= 2:
            members = set(chosen)
            subset = [row for row in rows if row['model_id'] in (model_a, model_b)
                      and (row['outer_repeat'], row['outer_fold']) in members]
            interval = paired_corrected_interval(subset, model_a, model_b, metric=metric, q=q, confidence=confidence,
                                                 expected_folds=chosen, expected_seeds=seeds)
            interval.pop('p_approximate')
            entry['interval'] = interval
        entries.append(entry)
    return entries


def pooled_depth_split(by_dataset, depths):
    """Across datasets, per depth: the number of dataset-folds and the mean and SD of their differences (no interval:
    folds of different datasets are not one resampling design)."""
    pooled = []
    for depth in validate_depths(depths):
        values = [d for entries in by_dataset.values() for e in entries if e['widths'] == depth for d in e['fold_differences']]
        pooled.append({'widths': depth, 'n_dataset_folds': len(values),
                       'datasets': sorted(name for name, entries in by_dataset.items()
                                          if any(e['widths'] == depth and e['n_folds'] for e in entries)),
                       'mean_difference': float(np.mean(values)) if values else None,
                       'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return pooled


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

SMOKE_DATASET = 'synthetic'
SMOKE_DESIGN = {'outer_folds': 3, 'outer_repeats': 1, 'inner_folds': 2}
REFERENCE_SMOKE_REGISTRY = 'experiments.make_revision.knn_controls:smoke_reference_registry'


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_reference_registry(protocol):
    """Synthetic smoke only: ArrowFlow-kNN (bridge.arrowflow_full_knn_factory) at the smoke protocol's declared candidates."""
    if protocol.get('purpose') != 'synthetic_smoke_only':
        raise ValueError('smoke_reference_registry serves synthetic smoke protocols only')
    return {TRAINED_MODEL: ModelSpec(TRAINED_MODEL, arrowflow_full_knn_factory, protocol['smoke_reference_candidates'], True)}


def write_synthetic_dataset(directory, protocol, *, samples=120, seed=33):
    """A three-class, four-feature dataset in run_revision's prepared layout (manifest, splits, data.npz)."""
    from .run_revision import write_json
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], samples // 3)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    features, labels = [f'x{i}' for i in range(4)], ['0', '1', '2']
    splits = make_splits(y, protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed'])
    manifest = {'dataset_id': SMOKE_DATASET, 'purpose': 'synthetic_smoke_only', 'source': 'synthetic smoke dataset',
                'feature_names': features, 'label_map': labels, 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                'sample_order': 'source row order; zero-based sample_id',
                'dataset_hash': dataset_fingerprint(X, y, features, labels), 'splits_hash': config_id(splits)}
    write_json(Path(directory)/SMOKE_DATASET/'manifest.json', manifest)
    write_json(Path(directory)/SMOKE_DATASET/'splits.json', splits)
    if not (Path(directory)/SMOKE_DATASET/'data.npz').exists():
        np.savez_compressed(Path(directory)/SMOKE_DATASET/'data.npz', X=X, y=y)


def run_synthetic_family(output, protocol, registry_path, workers=1, samples=120):
    """run_revision's prepare, run and reporting stages for one family on the synthetic dataset, through the harness's
    own worker and validators. A synthetic smoke protocol carries frozen true only to pass reporting's gate."""
    from .reporting import summarize_verified_results
    from .run_revision import _worker, environment_record, get_registry, planned_jobs, write_json
    output = Path(output)
    registry = get_registry(registry_path, protocol)
    write_json(output/'protocol.json', protocol)
    write_json(output/'candidates.json', {name: {'stochastic': spec.stochastic, 'candidates': spec.candidates,
                                                 'config_ids': [config_id(c) for c in spec.candidates]}
                                          for name, spec in registry.items()})
    write_json(output/'environment.json', environment_record(registry_path))
    write_synthetic_dataset(output, protocol, samples=samples)
    write_json(output/'planned_jobs.json', planned_jobs([SMOKE_DATASET], protocol, registry))
    jobs = [(str(output), SMOKE_DATASET, index, model, registry_path)
            for index in range(protocol['outer_folds'] * protocol['outer_repeats']) for model in registry]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for _ in pool.map(_worker, jobs):
            pass
    write_json(output/'summary.json', summarize_verified_results(output))
    return output


def synthetic_reference_run(directory, candidates, *, workers=1, samples=120):
    """A complete synthetic bridge_knn run (ArrowFlow-kNN only) in the production layout, with summary.json."""
    protocol = dict(json.loads((PROTOCOLS/'bridge_knn.json').read_text()), **SMOKE_DESIGN, datasets=[SMOKE_DATASET],
                    primary_family_size=1, frozen=True, purpose='synthetic_smoke_only',
                    protocol_id='arrowflow-v3-bridge-knn-1-synthetic-smoke', registry=REFERENCE_SMOKE_REGISTRY,
                    smoke_reference_candidates=list(candidates))
    return run_synthetic_family(directory, protocol, REFERENCE_SMOKE_REGISTRY, workers, samples)


def reference_pins(directory):
    """The training_controls.reference values that pin one complete ArrowFlow-kNN run."""
    directory = Path(directory)
    return {'model_id': TRAINED_MODEL, 'protocol_id': json.loads((directory/'protocol.json').read_text())['protocol_id'],
            'protocol_sha256': sha256_file(directory/'protocol.json'), 'summary_sha256': sha256_file(directory/'summary.json'),
            'code_revision': json.loads((directory/'environment.json').read_text())['code_revision']}


def smoke(output, protocol, workers=3):
    """Synthetic reference run (ArrowFlow-kNN at the eight bridge candidates with learning rate 0.1, one iteration), the
    two controls with their real candidates, both summarized by reporting, then compare_runs training."""
    from .compare_runs import compare_training
    from .run_revision import execution_lock, write_json
    output = Path(output)
    reference_candidates = [dict(c, iterations=1) for c in bridge_candidates() if c['learning_rate'] == .1]
    with execution_lock():
        reference = synthetic_reference_run(output/'reference', reference_candidates, workers=workers)
        block = dict(protocol['training_controls'], reference={**protocol['training_controls']['reference'],
                                                                **reference_pins(reference)})
        tiny = dict(protocol, **SMOKE_DESIGN, datasets=[SMOKE_DATASET], primary_family_size=len(PRIMARY_CONTRASTS),
                    frozen=True, purpose='synthetic_smoke_only', protocol_id=protocol['protocol_id'] + '-synthetic-smoke',
                    training_controls=block)
        training = run_synthetic_family(output/'training', tiny, protocol['registry'], workers)
        result = compare_training(training, reference, output/'compare')
    record = {'purpose': 'synthetic_smoke_only_not_paper_evidence', 'reference': str(reference), 'training': str(training),
              'comparison': str(output/'compare'), 'contrasts': result['contrasts'], 'depth_split': result['depth_split']}
    write_json(output/'smoke.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['smoke'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOLS/'knn_training.json')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise ValueError('Worker count must be between 1 and 16')
    record = smoke(args.output, json.loads(args.protocol.read_text()), args.workers)
    for row in record['contrasts']:
        print(f"{row['dataset']}: {row['model_a']} - {row['model_b']} {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] Holm p={row['holm_p_approximate']:.3g}")


if __name__ == '__main__':
    main()
