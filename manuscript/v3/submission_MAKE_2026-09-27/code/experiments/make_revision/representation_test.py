"""The readout-matched representation test of ArrowFlow-kNN (simulated referee panel of 2026-09-23, workstream 3).

draft    --output P                                                   the unfrozen protocol (draft_protocol)
prepare  --protocol P [--reference NAME=RUN,ABLATION ...] --output O  seal the selections and every stored prediction checked
smoke    --protocol P --output O [--workers 3]                        synthetic references, a complete run, its summary and its
                                                                      prespecified analysis; never evidence
pilot    --protocol P [--reference ...] --output O                    training-only timing and the projection at 16 workers
freeze   --draft P --pilot O/pilot.json --stages S --output F         the frozen protocol, only if the projection is within the cap
run      --protocol P [--reference ...] --output O --workers 16       every planned job (dataset x outer fold, three fitting seeds)
summary  --output O                                                   re-verify every planned record; representation_test_summary.json
                                                                      and .csv, all or none
The prespecified analysis is compare_representation.analyse.

The question. ArrowFlow predicts with a nearest-neighbour (kNN) readout on its last hidden ranking, and training improves that
readout over an untrained network. Does training produce a better REPRESENTATION, or only neighbourhoods that suit kNN? The
same representations are read with a strong fixed readout, a support vector machine on the Kendall kernel
(neighbour_baselines.kendall_kernel), at ArrowFlow-kNN's own reconstructed per-fold selections:

  (a) input      each view's encoded input ranking (the encoder of knn_controls.MultiViewInputKNN)
  (b) untrained  each view's last hidden ranking at its seeded initial filters, never updated (the networks of
                 knn_controls.UntrainedMultiViewArrowFlowKNN, read by ArrowFlowEstimator.transform_orders_by_depth)
  (c) trained    each view's last hidden ranking at its checkpoint filters (MultiViewArrowFlowKNN refitted deterministically by
                 run_knn_ablation.fit_views7, read the same way)

Both readouts are fitted on every representation of the outer training rows and predict the outer test rows; the seven views
are combined by plurality vote with ArrowFlow's own tie rule (secondary_studies.majority):

  svc_input, svc_untrained, svc_trained   a Kendall-kernel SVC, C from the kendall_svc baseline's grid chosen on stratified
                                          splits of the training rows only (the very splits that choose the kNN readout)
  knn_input, knn_untrained, knn_trained   the registered kNN readout; knn_trained is ArrowFlow-kNN itself (the reference), and
                                          knn_untrained and knn_input are the component ablation's untrained and input_knn
                                          variants

Every job fails, cancels the pending jobs and blocks the summary when a check fails:
  reference_predictions    knn_trained's seven-view predictions equal the registered run's outer predictions, every seed
  reference_views          knn_trained's seven per-view predictions equal the component ablation's stored per-view predictions
  untrained_knn_reproduced knn_untrained's predictions equal the ablation's untrained variant, every seed
  input_knn_reproduced     knn_input's predictions equal the ablation's input_knn variant, every seed
  initial_state            per view: a fresh initialization with the trained network's own parameters has the trained network's
                           initial state hash, and its filters equal the untrained network's, so (b) is exactly where (c) began
  untrained_unchanged      per view: the untrained network's state hash is still its initial state hash at the end of the fit
  encoder_identity         per view: the untrained and input controls encode every training and test row exactly as the trained
                           view does, so the three representations share one encoder
  readout_representation   per view: each kNN readout stores exactly the training representation the SVC reads
  kernel_cap               per view and representation: the Kendall kernel is within the declared cap (never subsampled)
Outputs are all or none and never replace a file with different content.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import warnings
import zipfile
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from arrowflow.ranking import inverse_positions
from . import neighbour_baselines as nb
from . import run_knn_ablation as base
from . import training_diagnostics as td
from .bridge import bridge_candidates, resolve, resolve_selected
from .comparisons import derive_seed
from .evaluation import (canonical_json, config_id, metric_values, paired_corrected_interval, summarize_outer,
                         validate_outer_schedule, validate_split)
from .knn_controls import MultiViewInputKNN, UntrainedMultiViewArrowFlowKNN
from .models import ArrowFlowEstimator, array_hash, seed_fit
from .multiview import KNN_READOUT_GRID, KNN_SELECTION_FOLDS
from .newdata import makespan, sha256_file
from .run_bridge import fold_schedule
from .run_revision import environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-23'
PROTOCOL = PROTOCOLS/'representation_test.json'
PROTOCOL_ID = 'arrowflow-v3-representation-test-1'
FAMILY = 'representation_test'
SOURCE_MODULES = td.SOURCE_MODULES + ['experiments.make_revision.training_diagnostics',
                                      'experiments.make_revision.neighbour_baselines']
REFERENCE_MODEL = base.REFERENCE_MODEL
REFERENCES = td.REFERENCES
DEFAULT_SOURCES = td.DEFAULT_SOURCES
DEVELOPMENT_REFERENCE = 'bridge_knn'
DEVELOPMENT = tuple(REFERENCES[DEVELOPMENT_REFERENCE]['datasets'])
FURTHER = ('balance_scale', 'ionosphere', 'diabetes', 'banknote_authentication', 'qsar_biodeg', 'mfeat_zernike',
           'vertebra_column', 'steel_plates_fault', 'climate_model_simulation_crashes', 'hcv_egyptian_patients')

READOUTS = ('svc', 'knn')
REPRESENTATIONS = ('input', 'untrained', 'trained')
ARMS = tuple(f'{readout}_{representation}' for readout in READOUTS for representation in REPRESENTATIONS)
REFERENCE_ARM = 'knn_trained'
ABLATION_VARIANTS = ('views7', 'untrained', 'input_knn')
ARM_SOURCES = {
    'svc_input': 'neighbour_baselines.kendall_kernel SVC on the input control\'s encoded rankings',
    'svc_untrained': 'neighbour_baselines.kendall_kernel SVC on the untrained networks\' last hidden rankings',
    'svc_trained': 'neighbour_baselines.kendall_kernel SVC on the trained networks\' last hidden rankings',
    'knn_input': 'knn_controls.MultiViewInputKNN at the selection (the ablation\'s input_knn variant)',
    'knn_untrained': 'knn_controls.UntrainedMultiViewArrowFlowKNN at the selection (the ablation\'s untrained variant)',
    'knn_trained': 'multiview.MultiViewArrowFlowKNN at the selection, refitted by run_knn_ablation.fit_views7 (ArrowFlow-kNN)'}
PRIMARY = ('svc_trained', 'svc_untrained')
SECONDARY = ('svc_trained', 'svc_input')
CONTRASTS = {'trained_minus_untrained_svc': PRIMARY, 'trained_minus_input_svc': SECONDARY,
             'svc_minus_knn_input': ('svc_input', 'knn_input'),
             'svc_minus_knn_untrained': ('svc_untrained', 'knn_untrained'),
             'svc_minus_knn_trained': ('svc_trained', 'knn_trained'),
             'trained_minus_untrained_knn': ('knn_trained', 'knn_untrained'),
             'trained_minus_input_knn': ('knn_trained', 'knn_input')}
CONTRAST_METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1')
METRICS = ('accuracy', 'error', 'balanced_accuracy', 'macro_f1')
SVC_C = list(nb.SVC_C)
SVC_MAX_ITER = nb.SVC_MAX_ITER
KERNEL_CAP = dict(nb.KERNEL_CAP)
SELECTION_FOLDS = KNN_SELECTION_FOLDS
ALPHA = .05
CAP_HOURS = 4
WORKERS = 16
MAX_WORKERS = 16
PILOT_DATASETS = ('wine', 'digits', 'segment')   # two hidden layers at the first fold; 128 input items; the most rows
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'pilot_projection')
DRAFT_STATUS = 'drafted_awaiting_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_any_outer_score'
SUMMARY_JSON, SUMMARY_CSV = 'representation_test_summary.json', 'representation_test_summary.csv'
SUMMARY_COLUMNS = ('dataset_id', 'panel', 'arm_id', 'metric', 'mean', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds',
                   'seeds_per_fold')
PREDICTION_KEYS = ('dataset_id', 'outer_repeat', 'outer_fold', 'arm_id', 'model_seed', 'sample_id', 'y_true', 'y_pred',
                   'config_id', 'code_revision')
REFERENCE_CHECKS = ('reference_predictions', 'reference_views', 'untrained_knn_reproduced', 'input_knn_reproduced')
VIEW_CHECKS = ('initial_state', 'untrained_unchanged', 'encoder_identity', 'readout_representation', 'kernel_cap')
JOB_CHECKS = REFERENCE_CHECKS + VIEW_CHECKS
STATEMENTS = {'met': 'training improves the representation itself',
              'not_met': 'training improves the nearest-neighbour neighbourhoods but not the representation read by a strong '
                         'fixed readout'}
SECONDARY_STATEMENTS = {'met': 'under the same strong fixed readout, the trained hidden rankings are a better representation '
                               'than the encoded input rankings',
                        'not_met': 'under the same strong fixed readout, the trained hidden rankings are not shown to be a '
                                   'better representation than the encoded input rankings'}
SCREENING = {
    'status': 'The seven development datasets were screened for this question before this protocol was written, so they are '
              'descriptive only and every table labels them "screened before this protocol". The ten further datasets were '
              'never screened for it and are the only confirmatory scope.',
    'screen': 'Simulated-referee verification V4 (2026-09-23), section 1.4c: a Kendall-kernel SVC and a footrule kNN read the '
              'encoded input rankings and the untrained and trained hidden rankings of the matched study\'s STORED '
              'single-view networks (random projection, a fixed configuration, hidden widths [128]; 15 outer folds x 3 '
              'fitting seeds) on the seven development datasets only. Under the SVC the trained hidden rankings had 0.54 pp '
              'more error than the untrained ones on average and less on only 2 of the 7.',
    'further_datasets': 'The screen scored no outer fold of the ten further datasets with any readout; on them it read only '
                        'training-partition readout scores, vote counts and displacements (V4 report, section 0).',
    'files': {'arrowflow_repo/manuscript/v3/review/simulated-referees-2026-09-23/verification/v4/V4-report.md':
              'fa81f816ba7bf28dea9bb985314ba5b2374a0a76334e6c47a8ef093e7c22ba29',
              'arrowflow_repo/manuscript/v3/review/simulated-referees-2026-09-23/verification/v4/screen_readouts.py':
              '06328d26f3c092849788c78d1f78bb5577de63a2420f0199eb43afc6a9cbcb7a'},
    'committed_at': '878a0cc9c5883df97637113a103f06bae8ac5325'}


class CheckFailed(RuntimeError):
    """A reproduction, identity or consistency check of a representation-test job failed."""


def environment():
    return environment_record(__package__ + '.representation_test:environment')


def _plain(value):
    return json.loads(canonical_json(td.native(value)))


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def arm_parts(arm):
    """(readout, representation) of an arm id."""
    readout, _, representation = arm.partition('_')
    if readout not in READOUTS or representation not in REPRESENTATIONS:
        raise ValueError(f'Unknown arm {arm!r}')
    return readout, representation


# ----------------------------------------------------------------------------- the Kendall-kernel SVC readout

def penalty_key(C):
    """The JSON key of one C value of the grid ('0.1', '1', '10', '100')."""
    return canonical_json(C)


def selection_folds(y, folds=SELECTION_FOLDS):
    """select_knn_readout's rule: at most `folds` stratified splits, fewer when the smallest class has fewer rows."""
    folds = min(int(folds), int(np.unique(np.asarray(y), return_counts=True)[1].min()))
    if folds < 2:
        raise ValueError('Readout selection needs at least two rows per class in the training partition')
    return folds


def select_svc_penalty(gram, y, *, seed, grid=None, folds=SELECTION_FOLDS):
    """C for a precomputed-kernel SVC on the training rows whose Gram matrix is `gram`, by mean accuracy over
    StratifiedKFold(folds, shuffle=True, random_state=seed) of those rows only; ties to the smallest C. Each split fits on
    its training block of the Gram matrix and predicts its held-out rows from their block against the split's training
    rows, which equals computing the kernel per split because the kernel is elementwise."""
    grid = list(SVC_C if grid is None else grid)
    y, gram = np.asarray(y), np.asarray(gram)
    if gram.shape != (len(y), len(y)):
        raise ValueError('The Gram matrix must be square over the training rows')
    count = selection_folds(y, folds)
    scores = [[] for _ in grid]
    captured_count, categories = 0, Counter()
    for a, b in StratifiedKFold(count, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        for index, C in enumerate(grid):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model = SVC(kernel='precomputed', C=C, probability=False, max_iter=SVC_MAX_ITER).fit(gram[np.ix_(a, a)], y[a])
                predicted = model.predict(gram[np.ix_(b, a)])
            captured_count += len(captured)
            categories.update(w.category.__name__ for w in captured)
            scores[index].append(float(np.mean(predicted == y[b])))
    means = [float(np.mean(values)) for values in scores]
    best = min(range(len(grid)), key=lambda index: (-means[index], grid[index]))
    return {'C': grid[best], 'inner_score': means[best], 'folds': count,
            'candidate_scores': {penalty_key(C): means[index] for index, C in enumerate(grid)},
            'selection_warnings': captured_count, 'selection_warning_categories': dict(sorted(categories.items()))}


def kendall_svc_readout(train_orders, y_train, test_orders, *, seed, grid=None, folds=SELECTION_FOLDS, cap=None):
    """A Kendall-kernel SVC on one view's representation: the kernel of the training rankings (rows of complete
    permutations; for a hidden representation the ranking of the layer's filters), C chosen on stratified splits of the
    training rows (select_svc_penalty), the final SVC fitted on every training row, and the test rows predicted from their
    kernel against the training rows. Returns (predictions, record)."""
    y_train = np.asarray(y_train)
    train_orders, test_orders = np.asarray(train_orders), np.asarray(test_orders)
    cost = nb.check_kernel_cap(len(y_train), train_orders.shape[1], cap)
    gram = nb.kendall_kernel(train_orders)
    selection = select_svc_penalty(gram, y_train, seed=seed, grid=grid, folds=folds)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        classifier = SVC(kernel='precomputed', C=selection['C'], probability=False, max_iter=SVC_MAX_ITER).fit(gram, y_train)
    predicted = np.asarray(classifier.predict(nb.kendall_kernel(test_orders, train_orders)))
    record = {'C': selection['C'], 'inner_score': selection['inner_score'], 'folds': selection['folds'],
              'candidate_scores': selection['candidate_scores'], 'selection_seed': int(seed),
              'selection_warnings': selection['selection_warnings'],
              'selection_warning_categories': selection['selection_warning_categories'],
              'vocabulary': int(cost['vocabulary']), 'pairs': int(cost['pairs']), 'kernel_cost': cost,
              'support_vectors': int(sum(int(n) for n in classifier.n_support_)),
              'libsvm_fit_status': int(classifier.fit_status_),
              'fit_warnings': [{'category': w.category.__name__, 'message': str(w.message)} for w in captured]}
    return predicted, record


# ----------------------------------------------------------------------------- one fitting seed of one outer fold

def _knn_records(model):
    return [dict(choice, folds=int(selection['folds']))
            for choice, selection in zip(base.readout_choices(model), model.readout_selections_)]


def fit_representations(params, seed, X_train, y_train, X_test, *, cap=None):
    """One fitting seed: ArrowFlow-kNN refitted at the selection (run_knn_ablation.fit_views7), the untrained and input
    controls at the same selection (the ablation's own classes and parameters), and a Kendall-kernel SVC on each view's
    input, untrained and trained representation. Returns {'views': {arm: (views, test rows) predictions}, 'readouts':
    {arm: [per-view record]}, 'checks': {check: [per-view record]}, 'timing': {...}}."""
    timing = {'svc_input_seconds': 0., 'svc_untrained_seconds': 0., 'svc_trained_seconds': 0., 'representation_seconds': 0.}
    start = time.perf_counter()
    model, fit_seconds, trained_views, _ = base.fit_views7(params['views7'], seed, X_train, y_train, X_test)
    timing['views7_seconds'] = time.perf_counter() - start
    timing['views7_fit_seconds'] = float(fit_seconds)
    start = time.perf_counter()
    seed_fit(seed)
    untrained = UntrainedMultiViewArrowFlowKNN(**params['untrained'], seed=seed).fit(X_train, y_train)
    untrained_views = np.stack(untrained.predict_views(X_test)[0])
    timing['untrained_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    seed_fit(seed)
    control = MultiViewInputKNN(**params['input_knn'], seed=seed).fit(X_train, y_train)
    input_views = np.stack(control.predict_views(X_test))
    timing['input_seconds'] = time.perf_counter() - start
    n_views = len(model.views_)
    if len(untrained.views_) != n_views or len(control.views_) != n_views:
        raise CheckFailed('The trained model and the two controls do not hold the same number of views')
    svc_views = {representation: [] for representation in REPRESENTATIONS}
    svc_records = {representation: [] for representation in REPRESENTATIONS}
    checks = {name: [] for name in VIEW_CHECKS}
    for v in range(n_views):
        (encoder, net), (untrained_encoder, untrained_net), (input_encoder, input_readout) = (
            model.views_[v], untrained.views_[v], control.views_[v])
        seed_v = derive_seed(seed, 'view', v)
        start = time.perf_counter()
        orders_train, orders_test = encoder.transform(X_train), encoder.transform(X_test)
        same_encoding = all(np.array_equal(other.transform(rows), expected)
                            for other in (untrained_encoder, input_encoder)
                            for rows, expected in ((X_train, orders_train), (X_test, orders_test)))
        trained_train, trained_test = net.transform_orders(orders_train), net.transform_orders(orders_test)
        untrained_train, untrained_test = untrained_net.transform_orders(orders_train), untrained_net.transform_orders(orders_test)
        input_train, input_test = inverse_positions(orders_train), inverse_positions(orders_test)
        stored = {'trained': np.array_equal(model.readouts_[v].positions_, trained_train),
                  'untrained': np.array_equal(untrained.readouts_[v].positions_, untrained_train),
                  'input': np.array_equal(input_readout.positions_, input_train)}
        # A fresh initialization with the trained network's own parameters: its initial state hash must be the one the trained
        # network recorded before training, and its filters must be the untrained network's. initialize_orders reseeds every
        # global generator, which is harmless here: every fit of this seed is complete and nothing below draws from them.
        fresh = ArrowFlowEstimator(**net.get_params()).initialize_orders(orders_train, y_train)
        start_matches = fresh.initial_state_hash_ == net.initial_state_hash_
        filters_match = td.copies_equal(td.filter_copy(fresh.network_), td.filter_copy(untrained_net.network_))
        untouched = untrained_net.network_.update_iter == 0 and untrained_net.state_hash() == untrained_net.initial_state_hash_
        del fresh
        timing['representation_seconds'] += time.perf_counter() - start
        checks['initial_state'].append({'view': v, 'passed': bool(start_matches and filters_match),
                                        'initial_state_hash_matches': bool(start_matches),
                                        'untrained_filters_match': bool(filters_match)})
        checks['untrained_unchanged'].append({'view': v, 'passed': bool(untouched)})
        checks['encoder_identity'].append({'view': v, 'passed': bool(same_encoding), 'vocabulary': int(orders_train.shape[1])})
        checks['readout_representation'].append({'view': v, 'passed': all(stored.values()), **{k: bool(x) for k, x in stored.items()}})
        representations = {'input': (orders_train, orders_test),
                           'untrained': (inverse_positions(untrained_train), inverse_positions(untrained_test)),
                           'trained': (inverse_positions(trained_train), inverse_positions(trained_test))}
        costs = {}
        for representation in REPRESENTATIONS:
            train_orders, test_orders = representations[representation]
            start = time.perf_counter()
            predicted, record = kendall_svc_readout(train_orders, y_train, test_orders,
                                                    seed=derive_seed(seed_v, 'readout_selection'), cap=cap)
            timing[f'svc_{representation}_seconds'] += time.perf_counter() - start
            svc_views[representation].append(predicted)
            svc_records[representation].append({'view': v, 'view_seed': int(seed_v), **record})
            costs[representation] = record['kernel_cost']
        checks['kernel_cap'].append({'view': v, 'passed': all(nb.within_cap(cost, cap) for cost in costs.values()),
                                     'costs': costs})
    views = {'svc_input': np.stack(svc_views['input']), 'svc_untrained': np.stack(svc_views['untrained']),
             'svc_trained': np.stack(svc_views['trained']), 'knn_input': input_views, 'knn_untrained': untrained_views,
             'knn_trained': np.asarray(trained_views)}
    shapes = {arm: array.shape for arm, array in views.items()}
    if len(set(shapes.values())) != 1 or shapes[REFERENCE_ARM] != (n_views, len(X_test)):
        raise CheckFailed(f'The per-view predictions of the arms disagree in shape: {shapes}')
    readouts = {'svc_input': svc_records['input'], 'svc_untrained': svc_records['untrained'],
                'svc_trained': svc_records['trained'], 'knn_input': _knn_records(control),
                'knn_untrained': _knn_records(untrained), 'knn_trained': _knn_records(model)}
    timing['encoding_seconds'] = float(model.encoding_seconds_)
    timing['training_seconds'] = float(model.training_seconds_)
    del model, untrained, control
    return {'views': views, 'readouts': readouts, 'checks': checks, 'timing': timing}


def reproduction_record(predictions, trained_views, sealed, hashes, seed):
    """The four checks against stored outer predictions for one fitting seed: knn_trained against the registered run
    (run_knn_ablation.reproduction_check), its seven per-view predictions against the ablation's stored views, and
    knn_untrained and knn_input against the ablation's untrained and input_knn variants."""
    check = base.reproduction_check(np.asarray(predictions['knn_trained']), sealed, seed)
    views_hash = array_hash(np.asarray(trained_views))
    untrained_hash = array_hash(np.asarray(predictions['knn_untrained']))
    input_hash = array_hash(np.asarray(predictions['knn_input']))
    return {**check, 'reference_predictions': bool(check['reproduced']),
            'trained_views_hash': views_hash, 'ablation_views_hash': hashes['knn_views'],
            'reference_views': views_hash == hashes['knn_views'],
            'untrained_prediction_hash': untrained_hash, 'ablation_untrained_hash': hashes['untrained'],
            'untrained_knn_reproduced': untrained_hash == hashes['untrained'],
            'input_prediction_hash': input_hash, 'ablation_input_hash': hashes['input_knn'],
            'input_knn_reproduced': input_hash == hashes['input_knn']}


# ----------------------------------------------------------------------------- protocol

def kernel_capacity(cap=None, outer_folds=5):
    """Per panel dataset, the largest Kendall kernel any fit can ask for: the outer training partition bounded by
    n - n // folds, and the vocabulary bounded by the largest of the embed_dim any registered candidate resolves to on that
    partition and the largest last hidden width of the registered grid."""
    shapes = {entry['name']: entry['shape'] for entry in nb.panel_declaration()}
    hidden = max(candidate['widths'][-1] for candidate in bridge_candidates())
    rows = {}
    for name in (*DEVELOPMENT, *FURTHER):
        samples, features = shapes[name]
        training = samples - samples // outer_folds
        vocabulary = max(max(resolve(candidate, features, training)['embed_dim'] for candidate in bridge_candidates()), hidden)
        cost = nb.kernel_cost(training, vocabulary)
        rows[name] = {**cost, 'within_cap': nb.within_cap(cost, cap),
                      'basis': 'training rows n - n // outer_folds; vocabulary max(largest resolved embed_dim, largest last '
                               'hidden width)'}
    return rows


def interpretation_declaration(development, further):
    most = len(further) // 2 + 1
    rule = (f'met when BOTH hold on the {len(further)} further datasets: (1) at least one contrast is Holm-significant in '
            f'favour of model_a (Holm-adjusted p < {ALPHA}, two-sided corrected resampled t, with a positive mean '
            f'difference), and (2) model_a has the higher mean accuracy (a positive mean difference over the 15 outer '
            f'folds) on at least {most} of the {len(further)}, i.e. on most of them')
    return {
        'status': 'fixed in advance and recorded in this protocol before any outer score of these readouts exists; neither '
                  'outcome is softened after the fact',
        'primary': {'contrast': 'svc_trained minus svc_untrained, accuracy', 'rule': rule, 'most_threshold': most,
                    'alpha': ALPHA, 'if_met': f'the paper states that {STATEMENTS["met"]}',
                    'otherwise': f'the paper states that {STATEMENTS["not_met"]}',
                    'statements': dict(STATEMENTS)},
        'secondary': {'contrast': 'svc_trained minus svc_input, accuracy', 'rule': rule, 'most_threshold': most,
                      'alpha': ALPHA, 'if_met': f'the paper states that {SECONDARY_STATEMENTS["met"]}',
                      'otherwise': f'the paper states that {SECONDARY_STATEMENTS["not_met"]}',
                      'statements': dict(SECONDARY_STATEMENTS),
                      'status': 'reported in full beside the primary family; it never changes the primary statement'},
        'reporting': 'every Holm-significant difference of either family is reported with its sign, whichever way it points; '
                     'the development datasets are descriptive and never change either statement; no dataset, arm, fold, '
                     'seed or contrast is added, dropped or reweighted after an outer score is seen',
        'development_datasets': list(development)}


def analysis_declaration(development, further):
    development, further = list(development), list(further)
    everything = development + further
    interval = ('corrected resampled t (evaluation.paired_corrected_interval): fitting seeds averaged within each outer fold, '
                'standard error sqrt((1/15 + q) * variance (ddof 1) of the fold differences), q = test_train_ratio = 0.25, '
                't quantile with 14 df, 95%; the p value is two-sided')
    return _plain({
        'declared': 'frozen with this protocol, before any outer score of these readouts exists',
        'metric': 'accuracy',
        'interval': interval,
        'primary_family': {'contrast': 'svc_trained minus svc_untrained', 'model_a': PRIMARY[0], 'model_b': PRIMARY[1],
                           'metric': 'accuracy', 'datasets': further, 'size': len(further),
                           'adjustment': f'Holm across the {len(further)} further datasets (evaluation.holm_adjust)',
                           'alpha': ALPHA, 'direction': 'positive favours the trained hidden rankings',
                           'scope': 'the ten further datasets only: they were never screened for this question'},
        'secondary_family': {'contrast': 'svc_trained minus svc_input', 'model_a': SECONDARY[0], 'model_b': SECONDARY[1],
                             'metric': 'accuracy', 'datasets': further, 'size': len(further),
                             'adjustment': f'Holm across the same {len(further)} further datasets, separately from the '
                                           'primary family',
                             'alpha': ALPHA, 'direction': 'positive favours the trained hidden rankings'},
        'interpretation': interpretation_declaration(development, further),
        'descriptive': {
            'status': 'descriptive: corrected resampled t intervals without p values and without multiplicity adjustment',
            'development_datasets': {'datasets': development, 'label': 'screened before this protocol',
                                     'contrasts': ['svc_trained minus svc_untrained', 'svc_trained minus svc_input'],
                                     'why': 'an exploratory screen already read these datasets (screening_disclosure)'},
            'readout_comparison': {'datasets': everything,
                                   'contrasts': ['svc_input minus knn_input', 'svc_untrained minus knn_untrained',
                                                 'svc_trained minus knn_trained'],
                                   'meaning': 'the SVC against the kNN readout on each representation'},
            'knn_contrasts': {'datasets': everything,
                              'contrasts': ['knn_trained minus knn_untrained', 'knn_trained minus knn_input'],
                              'meaning': 'the same comparisons under the kNN readout; knn_untrained and knn_input equal the '
                                         'component ablation\'s untrained and input_knn variants prediction for prediction'},
            'other_metrics': {'datasets': everything, 'metrics': ['balanced_accuracy', 'macro_f1'],
                              'contrasts': ['svc_trained minus svc_untrained', 'svc_trained minus svc_input']},
            'metrics_table': 'per dataset and arm: the mean outer accuracy, error, balanced accuracy and macro-F1 (the mean '
                             'over outer folds of the fitting-seed averages), the outer-fold SD and the mean within-fold '
                             'seed SD',
            'readout_settings': 'per dataset and arm: the selected C (SVC) or neighbour count and weighting (kNN) counted over '
                                'views, folds and seeds, the mean training-partition selection score, the libsvm fit '
                                'status and every warning'},
        'verification': 'compare_representation.analyse refuses (exit 2, nothing written) until the run is complete, then '
                        're-verifies every reference production run with compare_runs.load_run and compare_runs.verify_run '
                        'and this run with representation_test.summary, and requires the published summary to equal the '
                        're-verified one, before any score is read'})


def design():
    """The fixed design every representation-test protocol declares (validate_protocol requires it verbatim)."""
    return {
        'purpose': 'the readout-matched representation test: does training produce a better representation, or only '
                   'neighbourhoods that suit the nearest-neighbour readout?',
        'question': 'ArrowFlow predicts with a nearest-neighbour (kNN) readout on its last hidden ranking, and training '
                    'improves that readout over an untrained network. Here the same representations are also read by a '
                    'strong fixed readout, a support vector machine on the Kendall kernel, at ArrowFlow-kNN\'s own per-fold '
                    'selections: (a) the encoded input rankings, (b) the untrained hidden rankings (the seeded initial '
                    'filters) and (c) the trained hidden rankings (the checkpoint filters). If training improves the '
                    'representation itself, (c) beats (b) under the fixed readout too.',
        'selection_statement': 'Nothing is selected on an outer-fold score. Every representation is built at the per-fold '
                               'selection sealed by the reference ablation run (reconstructed from the registered run\'s '
                               'complete inner fit history); every readout setting (the SVC penalty C, the kNN neighbour '
                               'count and weighting) is chosen on stratified splits of the outer training rows only; the '
                               'analysis is frozen with this protocol, before any outer score of these readouts exists.',
        'unit_of_work': 'every protocol dataset x every outer fold (5 folds x 3 repeats) x every fitting seed; one job per '
                        'dataset and outer fold fits its three fitting seeds in sequence',
        'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3, 'split_seed': 27183, 'fit_seeds': [8129, 19391, 39019],
        'selection_metric': 'accuracy', 'test_train_ratio': 0.25, 'confidence': 0.95,
        'representations': {
            'input': 'each view\'s encoded input ranking: OrdinalEncoder(view_strategy(strategy, v), embed_dim, degree, '
                     'lda_ratio, derive_seed(seed, "view", v)) fitted on the outer training rows, whose transform ranks the '
                     'embed_dim items the first ranking layer receives. It is the encoder of the input-kNN control '
                     '(knn_controls.MultiViewInputKNN) and must equal the trained view\'s encoder on every training and test '
                     'row (encoder_identity).',
            'untrained': 'each view\'s last hidden ranking at its seeded initial filters, never updated: the networks of '
                         'knn_controls.UntrainedMultiViewArrowFlowKNN at the selection (ArrowFlowEstimator(embed_dim, degree, '
                         'widths, seed=view seed).initialize_orders on the view\'s encoded training ranking), read by '
                         'ArrowFlowEstimator.transform_orders_by_depth (its last layer): the last hidden layer\'s filters '
                         'ordered by ascending footrule distance to the row, ties by filter index. These are exactly the '
                         'filters the trained view network started training from (initial_state).',
            'trained': 'each view\'s last hidden ranking at its checkpoint filters: MultiViewArrowFlowKNN at the selection, '
                       'refitted deterministically by run_knn_ablation.fit_views7 (the core restores the checkpoint filters '
                       'at the end of training), read the same way. This is the representation ArrowFlow\'s own kNN readout '
                       'reads.',
            'hidden_width': 'the last hidden layer has 128 filters at every candidate of the registered grid, so every hidden '
                            'representation is a ranking of 128 items (8128 item pairs)'},
        'readouts': {
            'svc': {'estimator': f'sklearn SVC(kernel="precomputed", C, probability=False, max_iter={SVC_MAX_ITER}) on the '
                                 'Kendall kernel of the view\'s representation of the outer training rows; the outer test '
                                 'rows are predicted from their kernel against the training rows',
                    'kernel': nb.KENDALL_KERNEL_DEFINITION,
                    'kernel_items': 'the kernel reads a ranking as the order of its items: the embed_dim coordinate IDs for '
                                    'the input representation, the last hidden layer\'s filter IDs for a hidden one (so K counts, '
                                    'over pairs of filters, whether two rows rank the pair the same way); '
                                    'neighbour_baselines.kendall_kernel is applied to the ranking orders, for a hidden ranking '
                                    'arrowflow.ranking.inverse_positions of the positions transform_orders_by_depth returns',
                    'grid': {'C': list(SVC_C)},
                    'grid_source': 'the kendall_svc baseline\'s grid (neighbour_baselines.SVC_C, the svc_rbf comparator\'s)',
                    'selection': 'mean accuracy over StratifiedKFold(3, shuffle=True, random_state=derive_seed(view seed, '
                                 '"readout_selection")) of the outer training rows only, fewer splits when the smallest '
                                 'class has fewer than three rows: the very splits on which the view\'s kNN readout is '
                                 'chosen. The training Gram matrix is computed once and each split reads its blocks (the '
                                 'kernel is elementwise, so this equals recomputing it per split). Ties go to the smallest C.',
                    'final_fit': 'on every outer training row at the selected C',
                    'cap': dict(KERNEL_CAP), 'capacity': kernel_capacity()},
            'knn': {'estimator': 'the registered ArrowFlow-kNN readout: StableFootruleKNN on the representation\'s position '
                                 'vectors, chosen by multiview.select_knn_readout on the same stratified splits (ties to the '
                                 'lowest canonical config_id), refitted on every outer training row',
                    'grid': dict(KNN_READOUT_GRID), 'selection_folds': SELECTION_FOLDS,
                    'identities': {'knn_trained': 'ArrowFlow-kNN itself (the reference)',
                                   'knn_untrained': 'the component ablation\'s untrained variant',
                                   'knn_input': 'the component ablation\'s input_knn variant'}}},
        'aggregation': 'per arm, the seven per-view predictions are combined by plurality vote with ArrowFlow\'s own tie rule '
                       '(secondary_studies.majority: among the tied classes, the lowest class label)',
        'arms': list(ARMS), 'arm_sources': dict(ARM_SOURCES), 'reference_arm': REFERENCE_ARM,
        'screening_disclosure': dict(SCREENING),
        'checks': {
            'reference_predictions': 'every job and fitting seed: knn_trained\'s seven-view predictions equal the registered '
                                     'run\'s recorded outer predictions exactly, in the sealed test order (the component '
                                     'ablation\'s views7 check)',
            'reference_views': 'every job and fitting seed: knn_trained\'s seven per-view predictions have the array hash the '
                               'component ablation recorded for its views7 fit (knn_views_hash)',
            'untrained_knn_reproduced': 'every job and fitting seed: knn_untrained\'s predictions have the prediction hash of '
                                        'the ablation\'s untrained variant (its sealed summary)',
            'input_knn_reproduced': 'every job and fitting seed: knn_input\'s predictions have the prediction hash of the '
                                    'ablation\'s input_knn variant (its sealed summary)',
            'initial_state': 'every job, fitting seed and view: ArrowFlowEstimator(**trained network parameters)'
                             '.initialize_orders on the view\'s encoded training rows has the initial_state_hash_ the trained '
                             'network recorded before training, and every layer\'s filters (index matrices and orders) equal '
                             'the untrained network\'s',
            'untrained_unchanged': 'every job, fitting seed and view: the untrained network never updated (update_iter 0) and '
                                   'its state_hash() is still its initial_state_hash_',
            'encoder_identity': 'every job, fitting seed and view: the untrained and input controls encode every training and '
                                'test row exactly as the trained view does',
            'readout_representation': 'every job, fitting seed and view: each kNN readout stores exactly the training '
                                      'representation the SVC reads',
            'kernel_cap': 'every job, fitting seed, view and representation: the Kendall kernel is within the declared cap; '
                          'a larger kernel fails the job and is never computed on a subsample'},
        'fit_reuse': 'per outer fold and fitting seed: one seven-view ArrowFlow-kNN fit (knn_trained and svc_trained read its '
                     'networks), one untrained seven-view fit (initialization only; knn_untrained and svc_untrained read its '
                     'networks), one input control fit (knn_input and svc_input read its encoders) and 21 SVC readouts',
        'measures': 'per arm, dataset, outer fold and fitting seed: outer-test accuracy, error, balanced accuracy and macro-F1 '
                    'of the seven-view plurality; each view\'s predictions (artifacts); each view\'s readout setting and '
                    'training-partition selection score; each SVC fit\'s libsvm status, support vectors and warnings',
        'report_metrics': list(METRICS),
        'dataset_loading': 'the reference run prepared data (hash-checked) must equal the ablation copy, its splits the declared '
                           'nested splits, and in production a fresh load by the reference loader (run_revision.load_dataset '
                           'or newdata.load_newdata with every pin) must return the same arrays and dataset hash',
        'failure_policy': 'log_all_failures; a failed check or fit fails its job, cancels the pending jobs and blocks the '
                          'summary; failures are never omitted; no adaptive stopping on any score',
        'pilot': 'training-only: every arm on the first outer training partition of each pilot dataset at the first fitting '
                 'seed, predicting on every fourth training row (the outer test fold is never touched, so the four checks '
                 'against stored outer predictions are not performed there), plus one reproduction probe '
                 '(run_knn_ablation.reproduction_probe) per reference',
        'decision_rule': f'freeze only if the calibrated projection at {WORKERS} single-thread workers is at most {CAP_HOURS} h: '
                         'the simulated first-free-worker makespan of the planned jobs in planned order, each job priced at the '
                         'reference run\'s realized outer fit and predict seconds of its fold and its three fitting seeds '
                         'times the piloted job/views7 time ratio (the dataset\'s own ratio when it was piloted, the largest '
                         'piloted ratio otherwise)',
        'wallclock_cap_hours': CAP_HOURS, 'workers': WORKERS, 'max_workers': MAX_WORKERS, 'numeric_threads_per_worker': 1,
        'parallelism': 'one spawned single-thread process per dataset and outer fold under the shared execution lock',
        'reporting': {'outputs': f'{SUMMARY_JSON} (metrics, descriptive intervals, readout settings, reproduction counts, check '
                                 f'totals and model rows) and {SUMMARY_CSV} (metrics per dataset, arm and metric), written '
                                 'all or none by the summary stage; the prespecified analysis is compare_representation.analyse',
                      'intervals': 'the summary\'s intervals are descriptive (no p values); the tested families live in '
                                   'compare_representation.analyse'},
        'reference_argument': '--reference NAME=RUN_DIR,ABLATION_DIR, one per protocol reference',
    }


def draft_protocol(references=None, protocol_id=PROTOCOL_ID, pilot_datasets=PILOT_DATASETS,
                   development_reference=DEVELOPMENT_REFERENCE):
    """The unfrozen protocol; the datasets follow the references in name order (a JSON protocol keeps no key order), the
    development panel is the development reference's datasets and the further panel every other dataset."""
    references = REFERENCES if references is None else references
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    development = list(references[development_reference]['datasets'])
    further = [name for name in datasets if name not in development]
    return _plain({**design(), 'protocol_id': protocol_id, 'production_family': FAMILY, 'datasets': datasets,
                   'references': references, 'development_reference': development_reference,
                   'development_datasets': development, 'further_datasets': further,
                   'analysis': analysis_declaration(development, further), 'pilot_datasets': list(pilot_datasets),
                   'frozen': False, 'status': DRAFT_STATUS,
                   'resource_decision': 'pending: synthetic smoke and the training-only pilot on ' + ', '.join(pilot_datasets)})


def validate_protocol(p):
    """A production protocol is draft_protocol for its references, or that draft with exactly the freeze fields set by
    freeze; with the production references it must declare the registered development and further panels."""
    if not isinstance(p, dict) or p.get('production_family') != FAMILY:
        raise ValueError(f'production_family must be {FAMILY}')
    references, development_reference = p.get('references'), p.get('development_reference')
    if not isinstance(references, dict) or development_reference not in references:
        raise ValueError('The protocol must declare its references and a development reference among them')
    draft = draft_protocol(references=references, protocol_id=p.get('protocol_id'),
                           pilot_datasets=p.get('pilot_datasets', ()), development_reference=development_reference)
    strip = lambda q: {k: v for k, v in q.items() if k not in FREEZE_FIELDS}
    if strip(p) != strip(draft):
        differing = sorted(k for k in set(p) | set(draft) if k not in FREEZE_FIELDS and p.get(k) != draft.get(k))
        raise ValueError(f'The protocol differs from representation_test.draft_protocol() in {", ".join(differing)}')
    td.reference_of(p)
    if not p['development_datasets'] or not p['further_datasets']:
        raise ValueError('Both the development and the further panel must hold datasets')
    if references == _plain(REFERENCES) and (development_reference != DEVELOPMENT_REFERENCE
                                             or p['development_datasets'] != list(DEVELOPMENT)
                                             or p['further_datasets'] != list(FURTHER)):
        raise ValueError('With the production references the panels must be the seven development and the ten further '
                         'datasets')
    if not p.get('frozen'):
        if p != draft:
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('pilot_projection') or {}
    hours = projection.get('decision_hours')
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != CAP_HOURS or projection.get('workers') != WORKERS
            or isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= CAP_HOURS):
        raise ValueError(f'A frozen protocol records its freeze and a pilot projection within the {CAP_HOURS} h cap '
                         f'at {WORKERS} workers')
    return p


def panel_of(p, name):
    if name in p['development_datasets']:
        return 'development'
    if name in p['further_datasets']:
        return 'further'
    raise ValueError(f'{name} is in neither panel')


# ----------------------------------------------------------------------------- the stored predictions each job is checked against

def ablation_rows(reference):
    """The model rows of the reference's component ablation summary, read after binding the file to its declared pin."""
    observed = reference['ablation']['observed']
    path = Path(reference['ablation']['directory'])/observed['summary_file']
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != observed['summary_sha256']:
        raise ValueError(f'{path}: the ablation summary differs from its pin')
    return json.loads(data)['model_rows']


def ablation_record(reference, name, split, seeds, rows, sealed):
    """For one outer fold: the ablation result file (sha256), and per fitting seed the hash of the ablation's seven per-view
    views7 kNN predictions (knn_views_hash) and the prediction hashes of its views7, untrained and input_knn variants (from
    its sealed summary, which must agree with the result file). The ablation's views7 must reproduce the registered run."""
    repeat, fold = split['outer_repeat'], split['outer_fold']
    stem = f'{name}__r{repeat}f{fold}'
    path = Path(reference['ablation']['directory'])/'results'/f'{stem}.json'
    result = json.loads(path.read_text())
    if result.get('status') != 'ok':
        raise ValueError(f'{stem}: the ablation job did not succeed')
    fits = {fit['fit_id']: fit for fit in result['fits']}
    models = {(row['variant_id'], row['model_seed']): row for row in result['models']}
    summary = {(row['variant_id'], row['model_seed']): row for row in rows[name]
               if (row['outer_repeat'], row['outer_fold']) == (repeat, fold)}
    hashes = {}
    for seed in seeds:
        entry = {'knn_views': fits[f'views7__s{seed}']['knn_views_hash']}
        for variant in ABLATION_VARIANTS:
            if models[(variant, seed)]['prediction_hash'] != summary[(variant, seed)]['prediction_hash']:
                raise ValueError(f'{stem}: the ablation result and its summary disagree on {variant} (seed {seed})')
            entry[variant] = summary[(variant, seed)]['prediction_hash']
        if entry['views7'] != sealed['reference_prediction_hashes'][str(seed)]:
            raise ValueError(f'{stem}: the ablation views7 does not reproduce the registered predictions (seed {seed})')
        hashes[str(seed)] = entry
    return {'result_file': f'results/{stem}.json', 'result_sha256': sha256_file(path), 'hashes': hashes}


# ----------------------------------------------------------------------------- plan

def planned_job(p, reference, name, index, split, record, n_features, ablation):
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    key = (name, split['outer_repeat'], split['outer_fold'])
    sealed_job = reference['ablation']['jobs'].get(key)
    if sealed_job is None or sealed_job['config_id'] != record['config_id'] or sealed_job['selected'] != selected:
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the resolved selection differs from the '
                         'ablation plan')
    variants = dict(base.knn_ablation_variants(selected))
    params = {variant: variants[variant] for variant in ABLATION_VARIANTS}
    planned = {entry['variant_id']: entry['params'] for entry in sealed_job['variants']}
    if any(planned.get(variant) != params[variant] for variant in ABLATION_VARIANTS):
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the views7, untrained or input_knn '
                         'parameters differ from the ablation plan')
    return {'dataset_id': name, 'reference': reference['name'], 'panel': panel_of(p, name),
            'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
            'stem': f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}', 'config_id': record['config_id'],
            'config': record['config'], 'selected': selected, 'selected_widths': list(selected['widths']),
            'hidden_layers': len(selected['widths']), 'params': params, 'arms': list(ARMS),
            'model_seeds': list(p['fit_seeds']), 'first_fold': index == 0,
            'reference_prediction_hashes': dict(record['reference_prediction_hashes']), 'ablation': ablation,
            'reference_outer_seconds': sum(record['reference_outer_seconds'].values())}


def prepare(output, p, sources, datasets=None, *, allow_smoke=False, purpose='confirmatory'):
    """Verify the references, copy the prepared data, reconstruct every selection (it must equal the sealed record and
    resolve to the ablation plan), read every stored prediction hash each job is checked against, and seal the plan."""
    output = Path(output)
    mapping = td.reference_of(p)
    chosen = list(p['datasets']) if datasets is None else [name for name in p['datasets'] if name in set(datasets)]
    if not chosen:
        raise ValueError('The datasets must be protocol datasets')
    needed = [name for name in p['references'] if any(mapping[d] == name for d in chosen)]
    missing = [name for name in needed if name not in sources]
    if missing:
        raise ValueError(f'Supply --reference for {", ".join(missing)}')
    references = td.load_references(p, {name: sources[name] for name in needed}, allow_smoke=allow_smoke)
    rows = {name: ablation_rows(reference) for name, reference in references.items()}
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs, identities = [], [], {}
    seeds = list(p['fit_seeds'])
    for name in chosen:
        reference = references[mapping[name]]
        X, y, manifest, splits, identities[name] = td.prepare_dataset(output, reference, name, p, production=not allow_smoke)
        for index, split in enumerate(splits):
            validate_split(split, len(y))
            record = base.selection_record(reference['run'], name, split, y, manifest)
            if record != reference['ablation']['selections'].get((name, split['outer_repeat'], split['outer_fold'])):
                raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the selection reconstructed from '
                                 'the reference run differs from the sealed record')
            ablation = ablation_record(reference, name, split, seeds, rows[mapping[name]], record)
            selections.append(record)
            jobs.append(planned_job(p, reference, name, index, split, record, X.shape[1], ablation))
    write_json(output/'reference_selections.json', selections)
    write_json(output/'planned_jobs.json', jobs)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'protocol_id': p['protocol_id'], 'protocol_hash': config_id(p), 'datasets': chosen,
        'arms': list(ARMS), 'fit_seeds': seeds, 'planned_jobs': len(jobs),
        'planned_jobs_sha256': sha256_file(output/'planned_jobs.json'),
        'reference_selections_sha256': sha256_file(output/'reference_selections.json'), 'dataset_identity': identities,
        'references': {name: {'run_directory': str(Path(sources[name][0]).resolve()),
                              'ablation_directory': str(Path(sources[name][1]).resolve()),
                              'run': reference['observed_run'], 'ablation': reference['ablation']['observed'],
                              'run_file_sha256': reference['run']['files'],
                              'datasets': [d for d in chosen if mapping[d] == name]}
                       for name, reference in references.items()}})
    return jobs, references


def verify(output, *, allow_smoke=False, environment_check='full'):
    """The sealed run directory: protocol (frozen unless a synthetic smoke), manifest seal, unchanged scientific sources,
    planned jobs, sealed selections and the stored prediction hashes of every job. environment_check 'full' (run) or
    'sources' (summary)."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    if not smoke:
        validate_protocol(p)
        if not p['frozen'] or manifest.get('purpose') != 'confirmatory':
            raise ValueError('A representation-test run directory with a frozen reviewed protocol is required')
    if manifest['protocol_hash'] != config_id(p):
        raise ValueError('Protocol seal changed')
    saved, current = json.loads((output/'environment.json').read_text()), environment()
    if environment_check == 'sources':
        saved, current = saved['source_hashes'], current['source_hashes']
    elif environment_check != 'full':
        raise ValueError('environment_check must be full or sources')
    elif smoke:
        saved, current = dict(saved), dict(current)
        saved.pop('code_revision', None)
        current.pop('code_revision', None)
    if saved != current:
        raise ValueError('Source/environment seal changed')
    if manifest['arms'] != list(ARMS) or manifest['fit_seeds'] != list(p['fit_seeds']):
        raise ValueError('Prepared manifest disagrees with the protocol')
    if (sha256_file(output/'planned_jobs.json') != manifest['planned_jobs_sha256']
            or sha256_file(output/'reference_selections.json') != manifest['reference_selections_sha256']):
        raise ValueError('The planned jobs or the sealed selections changed')
    jobs = json.loads((output/'planned_jobs.json').read_text())
    selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                  for s in json.loads((output/'reference_selections.json').read_text())}
    folds = p['outer_folds'] * p['outer_repeats']
    if (len(jobs) != manifest['planned_jobs'] or len(jobs) != folds * len(manifest['datasets'])
            or len({job['stem'] for job in jobs}) != len(jobs) or len(selections) != len(jobs)
            or [job['dataset_id'] for job in jobs] != [name for name in manifest['datasets'] for _ in range(folds)]):
        raise ValueError('The planned jobs do not cover every dataset and outer fold exactly once')
    seeds = sorted(map(str, p['fit_seeds']))
    for job in jobs:
        record = selections[(job['dataset_id'], job['outer_repeat'], job['outer_fold'])]
        if record['fitting_seeds'] != list(p['fit_seeds']) or sorted(record['reference_predictions']) != seeds:
            raise ValueError(f'Missing or inconsistent sealed selection for {job["stem"]}')
        for seed, labels in record['reference_predictions'].items():
            if array_hash(np.asarray(labels)) != record['reference_prediction_hashes'][seed]:
                raise ValueError(f'Sealed reference predictions of {job["stem"]} disagree with their hashes')
        hashes = job['ablation']['hashes']
        if (sorted(hashes) != seeds or any(sorted(entry) != sorted(('knn_views',) + ABLATION_VARIANTS)
                                           for entry in hashes.values())
                or any(hashes[seed]['views7'] != record['reference_prediction_hashes'][seed] for seed in seeds)):
            raise ValueError(f'The stored prediction hashes of {job["stem"]} are incomplete or inconsistent')
    return p, manifest, jobs, selections


# ----------------------------------------------------------------------------- one job

def evaluate_job(X, y, split, p, job, sealed, *, dataset_hash, code_revision, sink=None, query=None, cap=None):
    """Every arm on one outer fold, for every fitting seed of the job. `query` replaces the outer test rows (the
    training-only pilot); the checks against stored outer predictions are then not performed. Returns (result, arrays):
    arrays holds each arm's seven per-view predictions per seed and is None when the job failed."""
    name, seeds = job['dataset_id'], [int(seed) for seed in job['model_seeds']]
    train = [int(i) for i in split['train']]
    test = [int(i) for i in (split['test'] if query is None else query)]
    held_out = query is None
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    params = job['params']
    common = {'dataset_id': name, 'dataset_hash': dataset_hash, 'outer_repeat': split['outer_repeat'],
              'outer_fold': split['outer_fold'], 'config_id': job['config_id'], 'code_revision': code_revision}
    identity = dict(common, panel=job['panel'], split_hash=config_id(split), train_ids=train, query_ids=test,
                    query='outer_test_rows' if held_out else 'training_rows_only', raw_train_hash=array_hash(X_train),
                    raw_query_hash=array_hash(X_test), training_labels_hash=array_hash(y_train), config=job['config'],
                    selected=job['selected'], params=params, hidden_layers=job['hidden_layers'], arms=list(ARMS),
                    model_seeds=seeds, reference_prediction_hashes=job['reference_prediction_hashes'],
                    ablation=job['ablation'], protocol_hash=config_id(p))
    result = {'status': 'running', 'identity': identity, 'fits': [], 'reproduction': [], 'models': [], 'predictions': [],
              'events': [], 'checks': {}, 'timing': {}}

    def emit(stage, record):
        event = {'stage': stage, 'record': _plain(record)}
        result['events'].append(event)
        if sink is not None:
            sink(event)

    emit('identity', identity)
    checks = {check: {'performed': False, 'passed': None} for check in JOB_CHECKS}
    result['checks'] = checks
    arrays, per_view = {}, {check: [] for check in VIEW_CHECKS}
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            for seed in seeds:
                fitted = fit_representations(params, seed, X_train, y_train, X_test, cap=cap)
                result['timing'][f's{seed}'] = fitted['timing']
                for check in VIEW_CHECKS:
                    per_view[check].extend({'model_seed': seed, **entry} for entry in fitted['checks'][check])
                broken = sorted({check for check in VIEW_CHECKS for entry in fitted['checks'][check] if not entry['passed']})
                if broken:
                    for check in broken:
                        checks[check] = {'performed': True, 'passed': False, 'views': per_view[check]}
                    raise CheckFailed(f'{", ".join(broken)}: failed for seed {seed}')
                predictions = {arm: majority(fitted['views'][arm]) for arm in ARMS}
                if held_out:
                    record = reproduction_record(predictions, fitted['views'][REFERENCE_ARM], sealed,
                                                 job['ablation']['hashes'][str(seed)], seed)
                    result['reproduction'].append(_plain(record))
                    emit('reproduction', record)
                    failed = [check for check in REFERENCE_CHECKS if not record[check]]
                    if failed:
                        for check in failed:
                            checks[check] = {'performed': True, 'passed': False, 'model_seed': seed}
                        raise CheckFailed(f'{", ".join(failed)}: the refit does not reproduce the stored predictions for '
                                          f'seed {seed}')
                for arm in ARMS:
                    readout, representation = arm_parts(arm)
                    views = np.asarray(fitted['views'][arm])
                    arrays[f'{arm}__s{seed}'] = views
                    fit = {'fit_id': f'{arm}__s{seed}', 'arm_id': arm, 'model_seed': seed, 'readout': readout,
                           'representation': representation, 'source': ARM_SOURCES[arm], 'status': 'ok',
                           'training_sample_count': len(y_train), 'views': fitted['readouts'][arm],
                           'view_predictions_hash': array_hash(views)}
                    result['fits'].append(_plain(fit))
                    emit('fit', fit)
                    predicted = np.asarray(predictions[arm])
                    row = dict(common, arm_id=arm, model_id=arm, variant_id=arm, model_seed=seed, readout=readout,
                               representation=representation, panel=job['panel'], stage='outer', status='ok',
                               training_sample_count=len(y_train), selected_widths=list(job['selected_widths']),
                               hidden_layers=job['hidden_layers'], prediction_hash=array_hash(predicted),
                               view_predictions_hash=array_hash(views), **metric_values(y_test, predicted))
                    result['models'].append(_plain(row))
                    emit('model', row)
                    result['predictions'].extend(
                        {'dataset_id': name, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
                         'arm_id': arm, 'model_seed': seed, 'sample_id': int(sample), 'y_true': y[sample].item(),
                         'y_pred': np.asarray(label).item(), 'config_id': job['config_id'], 'code_revision': code_revision}
                        for sample, label in zip(test, predicted))
                del fitted
        for check in REFERENCE_CHECKS:
            checks[check] = {'performed': held_out, 'passed': True if held_out else None, 'fitting_seeds': seeds}
        for check in VIEW_CHECKS:
            checks[check] = {'performed': True, 'passed': True, 'views': per_view[check]}
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = 'failed'
        result['exception'] = f'{type(exc).__name__}: {exc}'
        result['check_failed'] = isinstance(exc, CheckFailed)
        emit('terminal_failure', {'exception': result['exception'], 'check_failed': result['check_failed']})
        present = {(row['arm_id'], row['model_seed']) for row in result['models']}
        for arm in ARMS:
            for seed in seeds:
                if (arm, seed) not in present:
                    row = dict(common, arm_id=arm, model_id=arm, variant_id=arm, model_seed=seed, stage='outer',
                               status='failed', exception=result['exception'])
                    result['models'].append(_plain(row))
                    emit('model', row)
        arrays = None
    result['timing']['job_seconds'] = time.perf_counter() - started
    result['timing']['seconds_by_part'] = _seconds_by_part(result['timing'], seeds)
    return result, arrays


# Where a job's time goes: the ArrowFlow-kNN refit (fit and predict), the two controls, the three SVC readouts, and the
# per-view representations (encodings, hidden rankings) with the identity checks built on them.
PARTS = ('views7_seconds', 'untrained_seconds', 'input_seconds', 'svc_input_seconds', 'svc_untrained_seconds',
         'svc_trained_seconds', 'representation_seconds')


def _seconds_by_part(timing, seeds):
    return {part: float(sum(timing[f's{seed}'][part] for seed in seeds if f's{seed}' in timing)) for part in PARTS}


def worker(arguments):
    output, job = arguments
    output, stem = Path(output), job['stem']
    log, result_path = output/'logs'/f'{stem}.jsonl', output/'results'/f'{stem}.json'
    prediction_path, artifact_path = output/'predictions'/f'{stem}.jsonl', output/'artifacts'/f'{stem}.npz'
    if any(path.exists() for path in (log, result_path, prediction_path, artifact_path)):
        raise FileExistsError(f'Existing representation-test job {stem}')
    for path in (log, result_path, prediction_path, artifact_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    key = (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == key[1:])
    sealed = next(s for s in json.loads((output/'reference_selections.json').read_text())
                  if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == key)
    p = json.loads((output/'protocol.json').read_text())
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    with log.open('x') as stream:
        def sink(event):
            stream.write(canonical_json(event) + '\n')
            stream.flush()
        result, arrays = evaluate_job(X, y, split, p, job, sealed, dataset_hash=data['dataset_hash'], code_revision=revision,
                                      sink=sink)
    records = result.pop('predictions')
    with prediction_path.open('x') as stream:
        for record in records:
            stream.write(canonical_json(record) + '\n')
    result['prediction_file'] = {'path': f'predictions/{stem}.jsonl', 'records': len(records),
                                 'sha256': sha256_file(prediction_path)}
    if arrays is not None:
        with artifact_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        result['artifact'] = {'path': f'artifacts/{stem}.npz', 'sha256': sha256_file(artifact_path),
                              'arrays': sorted(arrays)}
    write_json(result_path, _plain(result))
    return str(result_path), result['status'], bool(result.get('check_failed'))


def run(output, workers=1, *, allow_smoke=False):
    """Every planned job on spawned single-thread workers; a failed job cancels the pending jobs."""
    output = Path(output)
    p, manifest, jobs, _ = verify(output, allow_smoke=allow_smoke)
    if not 1 <= workers <= p['max_workers']:
        raise ValueError('Worker count exceeds the shared limit')
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, (str(output), job)) for job in jobs]
        try:
            for future in as_completed(futures):
                path, status, check_failed = future.result()
                print(path, status, flush=True)
                if status != 'ok':
                    raise CheckFailed(f'{path}: the job failed ({"a check" if check_failed else "an error"}); the pending '
                                      'jobs are cancelled and this run cannot be summarized')
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return collect_results(output, allow_smoke=allow_smoke)


# ----------------------------------------------------------------------------- verification and summary

def validate_job(result, events, job, p, X, y, split, data, revision, prediction_path, artifact_path, sealed):
    """Bind every record to the sealed plan, the per-example predictions, the per-view artifacts and the stored
    predictions; returns {check: fold-seeds reproduced} (every seed, or it raises)."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def logged(stage):
        return [e['record'] for e in events if e['stage'] == stage]

    require(events == result['events'], 'events differ from log')
    require(result['status'] == 'ok', f'failed terminal job: {result.get("exception")}')
    train, test, seeds = split['train'], [int(i) for i in split['test']], list(p['fit_seeds'])
    identity = {'dataset_id': job['dataset_id'], 'dataset_hash': data['dataset_hash'], 'outer_repeat': job['outer_repeat'],
                'outer_fold': job['outer_fold'], 'config_id': job['config_id'], 'code_revision': revision,
                'panel': job['panel'], 'split_hash': config_id(split), 'train_ids': [int(i) for i in train],
                'query_ids': test, 'query': 'outer_test_rows', 'raw_train_hash': array_hash(X[train]),
                'raw_query_hash': array_hash(X[test]), 'training_labels_hash': array_hash(y[train]),
                'config': job['config'], 'selected': job['selected'], 'params': job['params'],
                'hidden_layers': job['hidden_layers'], 'arms': list(ARMS), 'model_seeds': seeds,
                'reference_prediction_hashes': job['reference_prediction_hashes'], 'ablation': job['ablation'],
                'protocol_hash': config_id(p)}
    require(result['identity'] == _plain(identity), 'job identity disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'],
            'planned configuration disagrees with the sealed reference selection')
    require(logged('identity') == [result['identity']] and logged('fit') == result['fits']
            and logged('model') == result['models'] and logged('reproduction') == result['reproduction'],
            'records differ from log')
    require({e['stage'] for e in events} <= {'identity', 'reproduction', 'fit', 'model'}, 'unexpected event stage')
    require(sorted(result['checks']) == sorted(JOB_CHECKS), 'check schedule')
    for check in JOB_CHECKS:
        require(result['checks'][check]['performed'] is True and result['checks'][check]['passed'] is True,
                f'check {check}')
    for check in VIEW_CHECKS:
        entries = result['checks'][check]['views']
        require(len(entries) == len(seeds) * 7 and all(entry['passed'] is True for entry in entries), f'check {check} views')
    expected_keys = {(arm, seed) for arm in ARMS for seed in seeds}
    require(Counter((f['arm_id'], f['model_seed']) for f in result['fits']) == Counter({k: 1 for k in expected_keys}),
            'incomplete fit schedule')
    require(Counter((r['arm_id'], r['model_seed']) for r in result['models']) == Counter({k: 1 for k in expected_keys}),
            'incomplete model rows')
    require(prediction_path.is_file() and sha256_file(prediction_path) == result['prediction_file']['sha256']
            and result['prediction_file']['path'] == f'predictions/{job["stem"]}.jsonl', 'prediction file hash')
    records = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    require(len(records) == result['prediction_file']['records'] == len(expected_keys) * len(test), 'prediction record count')
    vectors, labels = defaultdict(list), set(np.asarray(y).tolist())
    for r in records:
        require(sorted(r) == sorted(PREDICTION_KEYS), 'prediction record schema')
        require(r['dataset_id'] == job['dataset_id']
                and (r['outer_repeat'], r['outer_fold']) == (job['outer_repeat'], job['outer_fold'])
                and r['config_id'] == job['config_id'] and r['code_revision'] == revision, 'prediction identity')
        require(r['y_true'] == y[r['sample_id']] and r['y_pred'] in labels, 'prediction truth/label')
        vectors[(r['arm_id'], r['model_seed'])].append((r['sample_id'], r['y_pred']))
    require(set(vectors) == expected_keys, 'prediction arm/seed coverage')
    for items in vectors.values():
        require([sample for sample, _ in items] == test, 'prediction sample order')
    predictions = {k: np.asarray([label for _, label in items]) for k, items in vectors.items()}
    artifact = result.get('artifact') or {}
    require(artifact.get('path') == f'artifacts/{job["stem"]}.npz' and artifact_path.is_file()
            and sha256_file(artifact_path) == artifact.get('sha256'), 'artifact hash')
    with np.load(artifact_path, allow_pickle=False) as stored:
        views = {key: stored[key] for key in stored.files}
    require(sorted(views) == sorted(f'{arm}__s{seed}' for arm, seed in expected_keys) == artifact['arrays'],
            'artifact schedule')
    fits = {(f['arm_id'], f['model_seed']): f for f in result['fits']}
    for row in result['models']:
        key = (row['arm_id'], row['model_seed'])
        fit, pred, array = fits[key], predictions[key], views[f'{key[0]}__s{key[1]}']
        readout, representation = arm_parts(row['arm_id'])
        require(row['status'] == 'ok' and fit['status'] == 'ok' and row['model_id'] == row['arm_id'] == row['variant_id']
                and row['code_revision'] == revision and row['dataset_hash'] == data['dataset_hash']
                and row['config_id'] == job['config_id'] and row['panel'] == job['panel']
                and (row['readout'], row['representation']) == (fit['readout'], fit['representation']) == (readout, representation)
                and row['training_sample_count'] == len(train) == fit['training_sample_count'], 'model row identity')
        require(row['hidden_layers'] == job['hidden_layers'] and row['selected_widths'] == list(job['selected_widths']),
                'model row depth disagrees with plan')
        require(array.shape == (7, len(test)) and len(fit['views']) == 7, 'per-view prediction shape')
        require(row['view_predictions_hash'] == fit['view_predictions_hash'] == array_hash(array), 'per-view hash')
        require(np.array_equal(majority(array), pred), 'the plurality of the stored views differs from the predictions')
        require(row['prediction_hash'] == array_hash(pred), 'prediction hash')
        metrics = metric_values(y[test], pred)
        require(all(np.isclose(row[k], v, rtol=0, atol=1e-12) for k, v in metrics.items()), 'metric/prediction disagreement')
        if readout == 'svc':
            require(all(view['C'] in SVC_C and view['candidate_scores'][penalty_key(view['C'])] == view['inner_score']
                        and view['inner_score'] == max(view['candidate_scores'].values()) for view in fit['views']),
                    'an SVC readout record is not its own selection')
    counts = Counter()
    require([r['model_seed'] for r in result['reproduction']] == seeds, 'reproduction seeds')
    for record in result['reproduction']:
        seed = record['model_seed']
        by_arm = {arm: predictions[(arm, seed)] for arm in ARMS}
        expected = _plain(reproduction_record(by_arm, views[f'{REFERENCE_ARM}__s{seed}'], sealed,
                                              job['ablation']['hashes'][str(seed)], seed))
        require(record == expected, f'reproduction record of seed {seed}')
        for check in REFERENCE_CHECKS:
            require(record[check] is True, f'{check} does not hold for seed {seed}')
            counts[check] += 1
    return dict(counts)


def collect_results(output, *, allow_smoke=False, rederive=True):
    """Every planned job, log, prediction file and artifact reconciled, and (rederive) every sealed selection and stored
    prediction hash re-derived from the reference runs. Failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs, selections = verify(output, allow_smoke=allow_smoke, environment_check='sources')
    references, rows = None, {}
    if rederive:
        sources = {name: (entry['run_directory'], entry['ablation_directory'])
                   for name, entry in manifest['references'].items()}
        references = td.load_references(p, sources, allow_smoke=allow_smoke)
        rows = {name: ablation_rows(reference) for name, reference in references.items()}
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    prepared = {name: load_prepared(output, name) for name in manifest['datasets']}
    table, reproduced, issues, records = defaultdict(list), defaultdict(Counter), [], {}
    for job in jobs:
        stem, key = job['stem'], (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
        paths = {'result': output/'results'/f'{stem}.json', 'log': output/'logs'/f'{stem}.jsonl',
                 'predictions': output/'predictions'/f'{stem}.jsonl', 'artifact': output/'artifacts'/f'{stem}.npz'}
        missing = [str(path.relative_to(output)) for path in paths.values() if not path.exists()]
        if missing:
            issues.append(f'missing {stem}: {", ".join(missing)}')
            continue
        try:
            result = json.loads(paths['result'].read_text())
            events = [json.loads(line) for line in paths['log'].read_text().splitlines()]
            X, y, data, splits = prepared[job['dataset_id']]
            index = next(i for i, s in enumerate(splits) if (s['outer_repeat'], s['outer_fold']) == key[1:])
            split, sealed = splits[index], selections[key]
            if references is not None:
                reference = references[job['reference']]
                rederived = base.selection_record(reference['run'], job['dataset_id'], split, y, data)
                if rederived != sealed or rederived != reference['ablation']['selections'].get(key):
                    raise ValueError('the sealed selection differs from the one re-derived from its reference')
                ablation = ablation_record(reference, job['dataset_id'], split, list(p['fit_seeds']), rows[job['reference']],
                                           rederived)
                if planned_job(p, reference, job['dataset_id'], index, split, rederived, X.shape[1], ablation) != job:
                    raise ValueError('the planned job differs from the re-derived plan and stored predictions')
            reproduced[job['dataset_id']].update(validate_job(result, events, job, p, X, y, split, data, revision,
                                                              paths['predictions'], paths['artifact'], sealed))
            table[job['dataset_id']].extend(result['models'])
            records[stem] = result
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    for name in manifest['datasets']:
        try:
            validate_outer_schedule(table[name], expected_folds=fold_schedule(p),
                                    expected_seeds={arm: list(p['fit_seeds']) for arm in ARMS})
        except ValueError as exc:
            issues.append(f'{name}: {exc}')
    if issues:
        raise ValueError('Incomplete representation-test evidence: ' + '; '.join(issues))
    return {'rows': dict(table), 'reproduced': {name: dict(counter) for name, counter in reproduced.items()}, 'jobs': jobs,
            'protocol': p, 'manifest': manifest, 'records': records, 'code_revision': revision}


def _readout_settings(records, name):
    """Per arm: the readout settings chosen over views, folds and seeds, and the SVC fits' status and warnings."""
    out = {}
    for arm in ARMS:
        readout, _ = arm_parts(arm)
        views = [view for record in records.values() if record['identity']['dataset_id'] == name
                 for fit in record['fits'] if fit['arm_id'] == arm for view in fit['views']]
        entry = {'views': len(views), 'mean_selection_score': float(np.mean([view['inner_score'] for view in views]))}
        if readout == 'svc':
            entry.update(selected_C=dict(sorted(Counter(penalty_key(view['C']) for view in views).items())),
                         libsvm_fit_status_nonzero=sum(1 for view in views if view['libsvm_fit_status'] != 0),
                         fit_warnings=sum(len(view['fit_warnings']) for view in views),
                         selection_warnings=sum(view['selection_warnings'] for view in views),
                         mean_support_vectors=float(np.mean([view['support_vectors'] for view in views])),
                         vocabulary=sorted({view['vocabulary'] for view in views}))
        else:
            entry.update(selected_setting=dict(sorted(Counter(f"k={view['config']['n_neighbors']},{view['config']['weights']}"
                                                              for view in views).items())))
        out[arm] = entry
    return out


def summary(output, *, allow_smoke=False):
    """Seed-within-fold means, outer-fold mean and SD, descriptive intervals for the declared contrasts, the readout
    settings, the reproduction counts and the check totals."""
    collected = collect_results(output, allow_smoke=allow_smoke)
    p = collected['protocol']
    folds, seeds = fold_schedule(p), list(p['fit_seeds'])
    q, confidence = p['test_train_ratio'], p['confidence']
    summaries, flat = {}, []
    for name in collected['manifest']['datasets']:
        rows, panel = collected['rows'][name], panel_of(p, name)
        arms = {arm: {'metrics': {m: summarize_outer(rows, arm, m, expected_folds=folds, expected_seeds=seeds)
                                  for m in METRICS}} for arm in ARMS}
        for arm in ARMS:
            flat.extend({'dataset_id': name, 'panel': panel, 'arm_id': arm, 'metric': m,
                         **{k: arms[arm]['metrics'][m][k] for k in SUMMARY_COLUMNS[4:]}} for m in METRICS)
        contrasts = {}
        for label, (a, b) in CONTRASTS.items():
            contrasts[label] = {'model_a': a, 'model_b': b}
            for m in CONTRAST_METRICS:
                interval = paired_corrected_interval(rows, a, b, metric=m, q=q, confidence=confidence, expected_folds=folds,
                                                     expected_seeds={a: seeds, b: seeds})
                interval.pop('p_approximate')             # descriptive here; the tested families live in the analysis
                contrasts[label][m] = interval
        jobs = [j for j in collected['jobs'] if j['dataset_id'] == name]
        total = len(jobs) * len(seeds)
        summaries[name] = {
            'panel': panel, 'label': 'screened before this protocol' if panel == 'development' else 'never screened',
            'arms': arms, 'contrasts': contrasts,
            'readout_settings': _readout_settings(collected['records'], name),
            'reproduction': {check: {'matching_fold_seeds': collected['reproduced'][name].get(check, 0),
                                     'total_fold_seeds': total} for check in REFERENCE_CHECKS},
            'resolved_configurations': [{'outer_repeat': j['outer_repeat'], 'outer_fold': j['outer_fold'],
                                         'config_id': j['config_id'], 'hidden_layers': j['hidden_layers'],
                                         **{k: j['selected'][k] for k in ('widths', 'learning_rate', 'embed_dim', 'degree',
                                                                          'augment')}} for j in jobs],
            'two_hidden_layer_folds': sum(1 for j in jobs if j['hidden_layers'] >= 2), 'outer_folds': len(jobs)}
    report = {'purpose': 'readout_matched_representation_test_of_arrowflow_knn', 'code_revision': collected['code_revision'],
              'protocol_id': p.get('protocol_id'), 'references': collected['manifest']['references'],
              'arms': list(ARMS), 'reference_arm': REFERENCE_ARM, 'development_datasets': list(p['development_datasets']),
              'further_datasets': list(p['further_datasets']), 'screening_disclosure': p['screening_disclosure'],
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; within-fold seed SD '
                             'reported separately',
              'contrasts': {label: {'model_a': a, 'model_b': b} for label, (a, b) in CONTRASTS.items()},
              'intervals': 'seed-averaged corrected resampled t intervals, descriptive here, without p values or multiplicity '
                           'adjustment; the tested families are compare_representation.analyse',
              'reproduction': 'hard checks: every fold and seed reproduced the registered knn_trained predictions, the '
                              'ablation\'s per-view predictions and its untrained and input_knn variants exactly (the '
                              'summary refuses otherwise)',
              'checks': _check_totals(collected['records']), 'timing': _timing_totals(collected['records']),
              'inferential_significance_claims': False, 'summaries': summaries, 'model_rows': collected['rows']}
    return _plain(report), flat


def _check_totals(records):
    totals = {check: Counter() for check in JOB_CHECKS}
    for record in records.values():
        for check in JOB_CHECKS:
            totals[check][str(record['checks'][check]['passed'])] += 1
    return {check: dict(counter) for check, counter in totals.items()}


def _timing_totals(records):
    parts = Counter()
    for record in records.values():
        parts.update(record['timing']['seconds_by_part'])
        parts['job_seconds'] += record['timing']['job_seconds']
    return {part: float(value) for part, value in sorted(parts.items())}


def write_summary(output, *, allow_smoke=False):
    """The summary JSON and CSV, all or none (compare_runs.write_outputs); a file with different content is never replaced."""
    from .compare_runs import _csv_text, _json_text, write_outputs
    report, flat = summary(output, allow_smoke=allow_smoke)
    write_outputs(Path(output), {SUMMARY_JSON: _json_text(report), SUMMARY_CSV: _csv_text(flat, SUMMARY_COLUMNS)})
    return report


# ----------------------------------------------------------------------------- pilot, projection and freeze

def calibrated_projection(p, jobs, records, workers=WORKERS):
    """Per planned job: the reference run's realized outer fit and predict seconds of its fold and its three fitting seeds
    times the piloted job/views7 ratio (the dataset's own when it was piloted, the largest piloted ratio otherwise)."""
    own = {record['dataset_id']: record['job_to_views7_ratio'] for record in records}
    largest = max(own.values())
    seconds, datasets = [], {}
    for job in jobs:
        ratio = own.get(job['dataset_id'], largest)
        realized = float(job['reference_outer_seconds'])
        cost = realized * ratio
        seconds.append(cost)
        entry = datasets.setdefault(job['dataset_id'], {'jobs': 0, 'reference_views7_seconds': 0., 'seconds': 0.,
                                                        'ratio': ratio, 'basis': 'piloted ratio' if job['dataset_id'] in own
                                                        else 'largest piloted ratio'})
        entry['jobs'] += 1
        entry['reference_views7_seconds'] += realized
        entry['seconds'] += cost
    for entry in datasets.values():
        entry['serial_hours'] = entry['seconds'] / 3600
    serial = sum(seconds)
    return {'datasets': datasets, 'serial_hours': serial / 3600, 'serial_hours_over_workers': serial / 3600 / workers,
            'simulated_makespan_hours': makespan(seconds, workers) / 3600, 'longest_job_hours': max(seconds) / 3600,
            'workers': workers, 'basis': p['decision_rule']}


def runtime_pilot(output, p, sources):
    """Training-only: prepare every protocol dataset, then the first outer fold of each pilot dataset at the first fitting
    seed with every fourth training row as the query rows (the outer test fold is never touched), the projection and one
    reproduction probe per reference."""
    output = Path(output)
    started = utc_now()
    jobs, references = prepare(output, p, sources, purpose='training_runtime_only')
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    sealed = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
              for s in json.loads((output/'reference_selections.json').read_text())}
    seed = p['fit_seeds'][0]
    records = []
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and (j['outer_repeat'], j['outer_fold']) == (0, 0))
        X, y, data, splits = load_prepared(output, name)
        split = splits[0]
        query = [int(i) for i in split['train'][::4]]
        result, _ = evaluate_job(X, y, split, p, dict(job, model_seeds=[seed]),
                                 sealed[(name, split['outer_repeat'], split['outer_fold'])], dataset_hash=data['dataset_hash'],
                                 code_revision=revision, query=query)
        if result['status'] != 'ok':
            raise CheckFailed(f'pilot job {name} failed: {result.get("exception")}')
        parts = result['timing']['seconds_by_part']
        records.append({'dataset_id': name, 'panel': job['panel'], 'hidden_layers': job['hidden_layers'],
                        'dataset_hash': data['dataset_hash'], 'train_ids': [int(i) for i in split['train']],
                        'query_ids': query, 'config_id': job['config_id'], 'selected': job['selected'], 'model_seed': seed,
                        'seconds_by_part': parts, 'job_seconds': result['timing']['job_seconds'],
                        'job_to_views7_ratio': float(result['timing']['job_seconds'] / parts['views7_seconds']),
                        'selected_C': {arm: [view['C'] for view in fit['views']] for fit in result['fits']
                                       for arm in (fit['arm_id'],) if arm.startswith('svc_')},
                        'checks': {key: {'performed': c['performed'], 'passed': c['passed']}
                                   for key, c in result['checks'].items()},
                        'status': 'ok'})
    probes = {name: base.reproduction_probe(reference['run'], td.smallest_dataset(reference))
              for name, reference in references.items()}
    calibrated = calibrated_projection(p, jobs, records, p['workers'])
    decision = calibrated['simulated_makespan_hours']
    checks_passed = all(c['passed'] is True for r in records for key, c in r['checks'].items() if key in VIEW_CHECKS)
    report = _plain({
        'purpose': 'training_only_runtime_no_heldout_scores', 'protocol_id': p['protocol_id'],
        'protocol_hash': config_id(p), 'code_revision': revision, 'started_utc': started, 'ended_utc': utc_now(),
        'records': records, 'calibrated_projection': calibrated, 'reproduction_probes': probes,
        'decision': {'rule': p['decision_rule'], 'hours': decision, 'cap_hours': p['wallclock_cap_hours'],
                     'workers': p['workers'], 'within_cap': decision <= p['wallclock_cap_hours'],
                     'checks_passed': checks_passed,
                     'probes_reproduced': all(probe['reproduced'] for probe in probes.values())},
        'estimate_limitations': 'one fitting seed on one training partition per pilot dataset, measured on the machine as it '
                                'was; the job/views7 ratio is a ratio of two times measured under the same conditions, so it is '
                                'far less sensitive to contention than either time; the calibrated projection assumes the '
                                'reference runs\' realized per-fold seconds (measured under 16 workers) carry over, prices '
                                'unpiloted datasets at the largest piloted ratio and assumes a job takes its three fitting '
                                'seeds in sequence; the query rows are training rows (every fourth), so the prediction costs '
                                'follow the outer test size only approximately'})
    write_json(output/'pilot.json', report)
    return report


def freeze(draft_path, pilot_path, stages_path, output_path, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the training-only pilot of that draft projects within the cap
    at the protocol workers, every performed pilot check and every reproduction probe held, and the stage record carries a
    passing smoke; an existing output must be the draft itself, which the frozen protocol then replaces."""
    draft = json.loads(Path(draft_path).read_text())
    if draft.get('frozen'):
        raise ValueError('The draft is already frozen')
    validate_protocol(draft)
    pilot, stages = json.loads(Path(pilot_path).read_text()), json.loads(Path(stages_path).read_text())
    if pilot.get('protocol_hash') != config_id(draft):
        raise ValueError('The pilot did not run with this draft')
    decision = pilot['decision']
    if not (decision['within_cap'] and 0 < decision['hours'] <= CAP_HOURS and decision['workers'] == WORKERS
            and decision['probes_reproduced'] and decision['checks_passed']):
        raise ValueError(f'Not frozen: projection {decision["hours"]:.2f} h against the {CAP_HOURS} h cap at '
                         f'{decision["workers"]} workers, checks {decision["checks_passed"]}, probes '
                         f'{decision["probes_reproduced"]}')
    if (stages.get('smoke') or {}).get('status') != 'ok' or not stages.get('summary'):
        raise ValueError('Not frozen: the stage record lacks a passing synthetic smoke and its summary')
    calibrated = pilot['calibrated_projection']
    record = {'cap_hours': CAP_HOURS, 'workers': WORKERS, 'decision_hours': decision['hours'],
              'decision_rule': decision['rule'],
              'calibrated': {key: calibrated[key] for key in ('serial_hours', 'serial_hours_over_workers',
                                                              'simulated_makespan_hours', 'longest_job_hours')},
              'calibrated_per_dataset_hours': {name: entry['serial_hours'] for name, entry in calibrated['datasets'].items()},
              'pilot_ratios': {r['dataset_id']: r['job_to_views7_ratio'] for r in pilot['records']},
              'pilot_seconds_by_part': {r['dataset_id']: r['seconds_by_part'] for r in pilot['records']},
              'reproduction_probes': {name: {key: probe[key] for key in ('dataset_id', 'result_file', 'config_id',
                                                                         'model_seed', 'inner_fold', 'reference_score',
                                                                         'refit_score', 'readout_selections_identical',
                                                                         'reproduced')}
                                      for name, probe in pilot['reproduction_probes'].items()},
              'pilot_code_revision': pilot['code_revision'], 'pilot_sha256': sha256_file(pilot_path), 'stages': stages}
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    text = (f"Readout-matched representation test (simulated referee panel of 2026-09-23, workstream 3, approved by the "
            f"author): drafted over all seventeen datasets with the primary family on the ten further datasets; "
            f"{stages['summary']}; projected at {WORKERS} single-thread workers: calibrated simulated makespan "
            f"{decision['hours']:.2f} h (serial {calibrated['serial_hours']:.2f} h, serial over workers "
            f"{calibrated['serial_hours_over_workers']:.2f} h, longest job {calibrated['longest_job_hours']:.2f} h); cap "
            f"{CAP_HOURS} h; reproduction probes held on "
            f"{', '.join(probe['dataset_id'] for probe in pilot['reproduction_probes'].values())}; frozen after the pilot")
    protocol = validate_protocol(_plain(dict(draft, frozen=True, frozen_at_utc=frozen_at, status=FROZEN_STATUS,
                                             resource_decision=text, pilot_projection=record)))
    output_path = Path(output_path)
    content = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if output_path.exists() and json.loads(output_path.read_text()) not in (draft, protocol):
        raise FileExistsError(f'Refusing to replace {output_path}: it is neither the draft nor this frozen protocol')
    temporary = output_path.with_name(output_path.name + '.freezing')
    temporary.write_text(content)
    os.replace(temporary, output_path)
    return protocol


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

SMOKE_DEVELOPMENT_REFERENCE = 'smoke_bridge_knn'


def smoke_protocol(p, sources):
    """The synthetic smoke form of a protocol whose references are the given synthetic runs {name: (run_dir, ablation_dir)}:
    the development panel is the synthetic bridge_knn-style run's dataset, the further panel the two batch runs'."""
    tiny = td.smoke_protocol(p, sources)
    development = list(tiny['references'][SMOKE_DEVELOPMENT_REFERENCE]['datasets'])
    further = [name for name in tiny['datasets'] if name not in development]
    return _plain(dict(tiny, development_reference=SMOKE_DEVELOPMENT_REFERENCE, development_datasets=development,
                       further_datasets=further, analysis=analysis_declaration(development, further),
                       purpose='synthetic_smoke_only'))


def smoke(output, p, workers=3):
    """Synthetic references built with the real harness (training_diagnostics.smoke's construction: a bridge_knn-style run
    with two hidden layers and its knn_ablation, and the two newdata-style batch runs with their newdata_ablation, so both
    depths and both augmentation settings reach every arm and every check), then a complete representation-test run, its
    summary and its prespecified analysis. Never evidence."""
    from . import compare_representation as cr
    from . import run_newdata_ablation as rn
    from .knn_controls import reference_pins, synthetic_reference_run
    output = Path(output)
    with execution_lock():
        bridge = synthetic_reference_run(output/'synthetic_bridge_knn', td.SMOKE_CANDIDATES, workers=workers, samples=240)
    template, reference_protocol = json.loads(base.PROTOCOL.read_text()), json.loads((bridge/'protocol.json').read_text())
    tiny = dict(template, datasets=['synthetic'], pilot_datasets=['synthetic'], frozen=False, purpose='synthetic_smoke_only',
                **{key: reference_protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds')},
                reference_source={**template['reference_source'], **reference_pins(bridge),
                                  'family': reference_protocol['production_family']},
                depth_split={**template['depth_split'], 'depths': [c['widths'] for c in td.SMOKE_CANDIDATES]})
    knn_ablation = output/'synthetic_knn_ablation'
    base.prepare(knn_ablation, tiny, bridge, allow_smoke=True, purpose='synthetic_smoke_only')
    base.run(knn_ablation, workers, allow_smoke=True)
    base.write_summary(knn_ablation, allow_smoke=True)
    newdata = output/'synthetic_newdata_ablation'
    rn.smoke(newdata, rn.draft_protocol(), workers)
    sources = {SMOKE_DEVELOPMENT_REFERENCE: (bridge, knn_ablation),
               'smoke_newdata_batch1': (newdata/'synthetic_reference_batch1', newdata),
               'smoke_newdata_batch2': (newdata/'synthetic_reference_batch2', newdata)}
    tiny_test = smoke_protocol(p, sources)
    run_directory = output/'run'
    prepare(run_directory, tiny_test, sources, allow_smoke=True, purpose='synthetic_smoke_only')
    run(run_directory, workers, allow_smoke=True)
    report = write_summary(run_directory, allow_smoke=True)
    analysis = cr.analyse(run_directory, output/'analysis', sources, allow_smoke=True)
    record = {'purpose': 'synthetic_smoke_only_not_paper_evidence', 'run': str(run_directory),
              'analysis': str(output/'analysis'), 'datasets': list(report['summaries']),
              'development_datasets': report['development_datasets'], 'further_datasets': report['further_datasets'],
              'planned_jobs': len(json.loads((run_directory/'planned_jobs.json').read_text())),
              'check_totals': report['checks'],
              'reproduction': {name: entry['reproduction'] for name, entry in report['summaries'].items()},
              'two_hidden_layer_folds': {name: entry['two_hidden_layer_folds'] for name, entry in report['summaries'].items()},
              'interpretation': analysis['interpretation'], 'analysis_outputs': sorted(analysis['outputs'])}
    write_json(output/'smoke.json', _plain(record))
    return record


# ----------------------------------------------------------------------------- command

def parse_reference(value):
    name, _, directories = value.partition('=')
    run_dir, _, ablation_dir = directories.partition(',')
    if not name or not run_dir or not ablation_dir:
        raise argparse.ArgumentTypeError('--reference takes NAME=RUN_DIR,ABLATION_DIR')
    return name, (Path(run_dir), Path(ablation_dir))


def parse_sources(values):
    sources = {}
    for name, directories in values or []:
        if name in sources:
            raise ValueError(f'--reference {name} is given twice')
        sources[name] = directories
    return sources or dict(DEFAULT_SOURCES)


def uncommitted_sources(protocol_path):
    """This module's sealed sources (its own environment record, not training_diagnostics') and the protocol file that
    differ from HEAD or are untracked (training_diagnostics.uncommitted_sources with those paths as extras)."""
    root = Path(__file__).resolve().parents[2]
    return td.uncommitted_sources((protocol_path, *(root/path for path in sorted(environment()['source_hashes']))))


def summary_lines(report):
    """One line per dataset with the four reproduction counts (the production script counts the complete ones)."""
    lines = []
    for name, entry in report['summaries'].items():
        counts = entry['reproduction']
        pair = lambda check: f"{counts[check]['matching_fold_seeds']}/{counts[check]['total_fold_seeds']}"
        lines.append(f"{name}: views7 reproduced {pair('reference_predictions')} fold-seeds; views "
                     f"{pair('reference_views')}; untrained kNN {pair('untrained_knn_reproduced')}; input kNN "
                     f"{pair('input_knn_reproduced')}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['draft', 'prepare', 'smoke', 'pilot', 'freeze', 'run', 'summary'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--reference', type=parse_reference, action='append', metavar='NAME=RUN_DIR,ABLATION_DIR')
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--draft', type=Path, default=PROTOCOL)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--stages', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        write_json(args.output, draft_protocol())
        return
    if args.command == 'summary':
        report = write_summary(args.output)
        for line in summary_lines(report):
            print(line)
        print(json.dumps(report['checks'], indent=2))
        return
    if args.command == 'freeze':
        if args.pilot is None or args.stages is None:
            parser.error('freeze needs --pilot and --stages')
        protocol = freeze(args.draft, args.pilot, args.stages, args.output)
        print(f"frozen at {protocol['frozen_at_utc']}: decision {protocol['pilot_projection']['decision_hours']:.2f} h")
        return
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f'Worker count must be between 1 and {MAX_WORKERS}')
    p = json.loads(args.protocol.read_text())
    if args.command == 'smoke':
        record = smoke(args.output, validate_protocol(p), args.workers)
        print(json.dumps({key: record[key] for key in ('check_totals', 'reproduction', 'interpretation')}, indent=2))
        return
    validate_protocol(p)
    sources = parse_sources(args.reference)
    if args.command == 'prepare':
        prepare(args.output, p, sources, args.dataset)
    elif args.command == 'pilot':
        with execution_lock():
            report = runtime_pilot(args.output, p, sources)
        print(json.dumps({'decision': report['decision'],
                          'pilot_ratios': {r['dataset_id']: r['job_to_views7_ratio'] for r in report['records']},
                          'calibrated_serial_hours': report['calibrated_projection']['serial_hours'],
                          'calibrated_makespan_hours': report['calibrated_projection']['simulated_makespan_hours']},
                         indent=2))
    else:
        if not p.get('frozen'):
            raise ValueError('The representation-test run requires a frozen reviewed protocol')
        dirty = uncommitted_sources(args.protocol)
        if dirty:
            raise ValueError(f'Commit the sealed sources and the protocol before the run: {", ".join(dirty)}')
        if p != json.loads((args.output/'protocol.json').read_text()):
            raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
        manifest = json.loads((args.output/'manifest.json').read_text())
        for name, entry in manifest['references'].items():
            if (str(Path(sources[name][0]).resolve()) != entry['run_directory']
                    or str(Path(sources[name][1]).resolve()) != entry['ablation_directory']):
                raise ValueError('--reference differs from the prepared reference sources')
        run(args.output, args.workers)


if __name__ == '__main__':
    main()
