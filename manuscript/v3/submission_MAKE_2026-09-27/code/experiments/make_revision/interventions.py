"""Two controlled interventions inside ArrowFlow-kNN, at its reconstructed per-fold selections (review response E4).

family depth        the same configuration with only the hidden widths varied: depth1 [128], depth2 [64, 128],
                    depth2_untrained_second ([64, 128] with the second hidden layer frozen at its initial filters) and
                    depth2_first_only ([64, 128] with only the first hidden layer updated: the second hidden layer and
                    the output layer stay at their initial filters, the latter through the library's last_layer_update).
family aggregation  the same architecture with only the hidden-layer order rule varied: borda (the unmodified update),
                    median (the footrule median of the same accumulator, prior weight 1) and median_mass_matched (the
                    footrule median with the prior entered as ballots, the multiplier chosen on the training-only pilot).

draft    --family F --output P                          the unfrozen protocol (draft_protocol)
prepare  --protocol P [--reference ...] --output O      seal the per-fold selections and the reference predictions
smoke    --protocol P --output O [--workers 3]          synthetic reference, ablation and a complete run; never evidence
pilot    --protocol P [--reference ...] --output O      training-only timing, the projection and the mass-matching ladder
freeze   --draft P --pilot O/pilot.json --stages S --output F   the frozen protocol, only within the cap
run      --protocol P --output O [--workers 16]         every planned job of the prepared directory
summary  --output O                                     verify every record, then the tables and <family>_summary.json
analyse  --run O --output A                             refuses (exit 2, nothing written) until the run is complete

Unit of work: one dataset, one outer fold, fitting seed 8129 and every arm of the family, each a seven-view
MultiViewArrowFlowKNN at the fold's reconstructed selection (bridge.resolve_selected) with nothing but the intervention
changed. The learning rate, the iteration count, the encoder, the view identities, the initial seeds, the vocabulary, the
degree, the augmentation, the checkpoint and the readout rule are the fold's own selection and are not re-tuned: this is
a controlled comparison at one configuration, not an estimate of the best attainable model of any arm.

Every job fails when a check fails: the arm whose parameters are the fold's own selection must reproduce the reference
run's outer predictions for seed 8129 exactly; on the first outer fold of each dataset its view state hashes and
predictions must equal an uninstrumented fit's; no instrumentation may survive the fit; the guarded reads must leave the
global RNG untouched; a frozen layer must not have moved; and the last depth probe must equal the view readout. A failed
job cancels the pending jobs and no table is written. Outputs are all or none and never replace a file with different
content. The prespecified analysis is written into the protocol before any outer score exists.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
from threadpoolctl import threadpool_limits
from . import run_knn_ablation as base
from . import training_diagnostics as td
from . import update_rules as ur
from .bridge import resolve_selected
from .comparisons import StableFootruleKNN, derive_seed
from .evaluation import (canonical_json, config_id, holm_adjust, make_splits, metric_values, paired_corrected_interval,
                         summarize_outer, validate_outer_schedule, validate_split)
from .knn_controls import depth_split, validate_depths
from .models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from .multiview import MultiViewArrowFlowKNN, view_strategy
from .newdata import makespan, sha256_file
from .run_bridge import fold_schedule, write_csv
from .run_revision import environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

FAMILIES = ('depth', 'aggregation')
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-14'
PROTOCOL_FILES = {family: PROTOCOLS/f'{family}.json' for family in FAMILIES}
PROTOCOL_IDS = {'depth': 'arrowflow-v3-depth-1', 'aggregation': 'arrowflow-v3-aggregation-1'}
SOURCE_MODULES = td.SOURCE_MODULES + ['experiments.make_revision.training_diagnostics',
                                      'experiments.make_revision.update_rules']
REFERENCE_MODEL = base.REFERENCE_MODEL
MODEL_SEED = td.MODEL_SEED
N_VIEWS = td.N_VIEWS
ARM_WIDTHS = {'depth1': [128], 'depth2': [64, 128]}
REFERENCE_ARM = {'depth': 'depth1', 'aggregation': 'borda'}
ARMS = {'depth': ('depth1', 'depth2', 'depth2_untrained_second', 'depth2_first_only'),
        'aggregation': ('borda', 'median', 'median_mass_matched')}
MASS_LADDER = (1, 2, 4, 8, 16)
MASS_DEFAULT = 4
INERT_THRESHOLD = .01
MIN_SUBSET_FOLDS = 3          # a named depth subset carries an interval only with at least this many folds
METRICS = base.METRICS
CAP_HOURS = {'depth': 4., 'aggregation': 4.}
WORKERS = 16
MAX_WORKERS = 16
PILOT_DATASETS = ('iris', 'ionosphere')
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'pilot_projection', 'mass_matching_choice')
DRAFT_STATUS = 'drafted_awaiting_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_any_outer_score'
PROVENANCE_FILE = 'provenance.json'
DEPTHS = [ARM_WIDTHS['depth1'], ARM_WIDTHS['depth2']]
PREDICTION_KEYS = ('dataset_id', 'outer_repeat', 'outer_fold', 'arm_id', 'model_seed', 'sample_id', 'y_true', 'y_pred',
                   'config_id', 'code_revision')
KEY_COLUMNS = ('family', 'dataset_id', 'reference', 'outer_repeat', 'outer_fold', 'model_seed', 'arm_id')
TABLES = {
    'arms.csv': KEY_COLUMNS + ('own_selection', 'widths', 'learning_rate', 'embed_dim', 'degree', 'augment', 'accuracy',
                               'error', 'balanced_accuracy', 'macro_f1', 'neighborhood_purity', 'fit_seconds'),
    'depth_probe.csv': KEY_COLUMNS + ('view', 'depth', 'hidden_layers', 'knn_accuracy', 'neighborhood_purity',
                                      'majority_knn_accuracy'),
    'movement.csv': KEY_COLUMNS + ('view', 'layer', 'layer_name', 'n_filters', 'vocabulary', 'batches', 'updates',
                                   'changed_share', 'mean_displacement', 'mean_votes', 'mean_vote_mass',
                                   'max_vote_mass', 'incoming_to_prior_ratio', 'median_solves',
                                   'tie_canonicalisation_incomplete')}
CONTRAST_COLUMNS = ('family', 'contrast', 'subset', 'dataset_id', 'arm_a', 'arm_b', 'metric', 'mean_difference',
                    'standard_error', 'sd', 'ci_low', 'ci_high', 'n_folds', 'df', 'p_approximate',
                    'holm_p_approximate', 'significant_after_holm')
JOB_CHECKS = ('reference_predictions', 'uninstrumented_state_hash', 'instrumentation_removed', 'rng_untouched',
              'frozen_layers_unchanged', 'depth_probe_matches_readout', 'arm_parameters')


class CheckFailed(td.CheckFailed):
    """A check of an intervention job failed."""


def environment():
    return environment_record(__package__ + '.interventions:environment')


def _plain(value):
    return td._plain(value)


def utc_now():
    return td.utc_now()


# ----------------------------------------------------------------------------- arms

def arm_specs(family, protocol=None):
    """{arm: how it is fitted}. widths_key names the arm's hidden widths in the depth family ('selected' means the
    fold's own selection); rule, prior_rule and prior_multiplier drive update_rules.ArmInstrumentation; frozen_hidden
    names hidden layers held at their initial filters; last_layer_update is the library's own output-layer switch."""
    multiplier = MASS_DEFAULT if protocol is None else (protocol.get('mass_matching_choice') or {}).get('prior_multiplier', MASS_DEFAULT)
    if family == 'depth':
        return {'depth1': {'widths_key': 'depth1', 'rule': 'borda', 'prior_rule': 'unit', 'prior_multiplier': 1,
                           'frozen_hidden': [], 'last_layer_update': True},
                'depth2': {'widths_key': 'depth2', 'rule': 'borda', 'prior_rule': 'unit', 'prior_multiplier': 1,
                           'frozen_hidden': [], 'last_layer_update': True},
                'depth2_untrained_second': {'widths_key': 'depth2', 'rule': 'borda', 'prior_rule': 'unit',
                                            'prior_multiplier': 1, 'frozen_hidden': [1], 'last_layer_update': True},
                'depth2_first_only': {'widths_key': 'depth2', 'rule': 'borda', 'prior_rule': 'unit',
                                      'prior_multiplier': 1, 'frozen_hidden': [1], 'last_layer_update': False}}
    if family == 'aggregation':
        return {'borda': {'widths_key': 'selected', 'rule': 'borda', 'prior_rule': 'unit', 'prior_multiplier': 1,
                          'frozen_hidden': [], 'last_layer_update': True},
                'median': {'widths_key': 'selected', 'rule': 'median', 'prior_rule': 'unit', 'prior_multiplier': 1,
                           'frozen_hidden': [], 'last_layer_update': True},
                'median_mass_matched': {'widths_key': 'selected', 'rule': 'median', 'prior_rule': 'one_ballot',
                                        'prior_multiplier': multiplier, 'frozen_hidden': [], 'last_layer_update': True}}
    raise ValueError(f'family must be one of {", ".join(FAMILIES)}')


def arm_widths(spec, selected, widths=None):
    widths = ARM_WIDTHS if widths is None else widths
    return list(selected['widths']) if spec['widths_key'] == 'selected' else list(widths[spec['widths_key']])


def arm_params(spec, selected, widths=None):
    """The MultiViewArrowFlowKNN keyword set of one arm at one resolved selection: the selection with the arm's hidden
    widths and output-layer switch, everything else untouched."""
    params = {k: v for k, v in selected.items() if k not in ('embed_scale', 'degree_offset')}
    missing = {'n_views', 'strategy', 'embed_dim', 'degree', 'widths', 'augment', 'validation_ratio', 'aggregation',
               'learning_rate', 'iterations', 'batch_size'} - params.keys()
    if missing:
        raise ValueError(f'Resolve the selected configuration before the arms; missing {sorted(missing)}')
    if params['n_views'] != N_VIEWS or params['aggregation'] != 'majority':
        raise ValueError('The arms are defined for the seven-view majority-vote selected configuration')
    params['widths'] = arm_widths(spec, selected, widths)
    if not spec['last_layer_update']:
        params['last_layer_update'] = False
    return params


def own_arm(family, specs, selected, widths=None):
    """The arm whose parameters are the fold's own selection: it must reproduce the reference outer predictions."""
    own = [arm for arm, spec in sorted(specs.items())
           if spec['rule'] == 'borda' and not spec['frozen_hidden'] and spec['last_layer_update']
           and arm_widths(spec, selected, widths) == list(selected['widths'])]
    if len(own) != 1:
        raise ValueError(f'The {family} family needs exactly one unmodified arm at the selected widths '
                         f'{list(selected["widths"])}, found {own}')
    return own[0]


# ----------------------------------------------------------------------------- fitting one arm

def fit_arm(params, seed, X, y, spec, *, instrument=True):
    """MultiViewArrowFlowKNN(**params, seed=seed).fit(X, y) step for step (MultiViewArrowFlowKNN.fit,
    MultiViewArrowFlow.fit and ArrowFlowEstimator.fit_orders = initialize_orders then train_initialized), with the arm's
    instrumentation installed on each view network for its training only. Returns (model, instrumentations)."""
    model = MultiViewArrowFlowKNN(**params, seed=seed)
    model.readouts_, model.readout_selections_ = [], []
    model.readout_seconds_ = 0.
    model.classes_ = np.unique(y)
    model.views_ = []
    encoding = training = 0.
    arms = []
    for v in range(model.n_views):
        seed_v = derive_seed(model.seed, 'view', v)
        start = time.perf_counter()
        enc = OrdinalEncoder(view_strategy(model.strategy, v), model.embed_dim, model.degree, model.lda_ratio, seed_v).fit(X, y)
        orders = enc.transform(X)
        encoding += time.perf_counter() - start
        net = ArrowFlowEstimator(embed_dim=model.embed_dim, degree=model.degree, widths=model.widths,
                                 iterations=model.iterations, learning_rate=model.learning_rate,
                                 batch_size=model.batch_size, last_layer_update=model.last_layer_update,
                                 ratio_data_backprop=model.ratio_data_backprop,
                                 motion_normalization_mult=model.motion_normalization_mult,
                                 p_correct=model.p_correct, seed=seed_v, validation_ratio=model.validation_ratio,
                                 augment=model.augment, n_augmentations=model.n_augmentations, max_swaps=model.max_swaps)
        net.initialize_orders(orders, y)
        if instrument:
            arm = ur.ArmInstrumentation(net, rule=spec['rule'], prior_rule=spec['prior_rule'],
                                        prior_multiplier=spec['prior_multiplier'], frozen_hidden=spec['frozen_hidden'])
            with arm.installed():
                net.train_initialized(orders, y)
            arms.append(arm)
        else:
            net.train_initialized(orders, y)
        training += net.training_seconds_
        model.views_.append((enc, net))
        model._fit_view_readout(enc, net, orders, y, seed_v)
    model.encoding_seconds_ = encoding
    model.training_seconds_ = training
    return model, arms


def neighborhood_purity(knn, positions, labels, truth):
    """The share of same-class neighbours of the query rows: the k nearest training rows of each query row under the
    footrule distance on the given ranking, k the readout's own n_neighbors."""
    _, indices = knn.kneighbors(positions)
    return float(np.mean(np.asarray(labels)[indices] == np.asarray(truth)[:, None]))


def view_probe(net, enc, selection, X_train, y_train, X_query, y_query):
    """The representation after each hidden layer (models.transform_orders_by_depth) read out by a footrule kNN at the
    view's own selected setting, refitted on the training rows at that depth: (records, predictions per depth)."""
    orders_train, orders_query = enc.transform(X_train), enc.transform(X_query)
    depths_train = net.transform_orders_by_depth(orders_train)
    depths_query = net.transform_orders_by_depth(orders_query)
    config = selection['config']
    records, predictions = [], []
    for depth, (train_positions, query_positions) in enumerate(zip(depths_train, depths_query)):
        knn = StableFootruleKNN(**config, input_kind='positions').fit(train_positions, y_train)
        predicted = np.asarray(knn.predict(query_positions))
        records.append({'depth': depth, 'hidden_layers': len(depths_train),
                        'knn_accuracy': td.accuracy(predicted, y_query),
                        'neighborhood_purity': neighborhood_purity(knn, query_positions, y_train, y_query)})
        predictions.append(predicted)
    return records, predictions


def evaluate_arm(arm, spec, params, seed, X_train, y_train, X_query, y_query, *, instrument=True):
    """One arm of one job: the fit, its per-view probes and movement, the seven-view majority and its metrics."""
    start = time.perf_counter()
    seed_fit(seed)
    model, instrumentations = fit_arm(params, seed, X_train, y_train, spec, instrument=instrument)
    fit_seconds = time.perf_counter() - start
    views, _ = model.predict_views(X_query)
    view_predictions = np.stack([np.asarray(view) for view in views])
    prediction = majority(view_predictions)
    records, depth_predictions, checks = [], [], {'depth_probe_matches_readout': True, 'frozen_layers_unchanged': True,
                                                  'instrumentation_removed': True, 'rng_untouched': True}
    for v, ((enc, net), selection) in enumerate(zip(model.views_, model.readout_selections_)):
        probes, predictions = view_probe(net, enc, selection, X_train, y_train, X_query, y_query)
        if not np.array_equal(predictions[-1], view_predictions[v]):
            checks['depth_probe_matches_readout'] = False
        depth_predictions.append(predictions)
        movement = instrumentations[v].summary() if instrument else None
        if instrument:
            checks['instrumentation_removed'] &= bool(movement['instrumentation_removed'])
            checks['rng_untouched'] &= bool(movement['rng_unchanged'])
            checks['frozen_layers_unchanged'] &= bool(instrumentations[v].frozen_unchanged())
            for layer in movement['layers']:
                layer['incoming_to_prior_ratio'] = incoming_to_prior_ratio(spec, layer)
        records.append({'view': v, 'strategy': enc.strategy, 'view_seed': int(net.seed),
                        'readout': {'config': selection['config'], 'config_id': selection['config_id'],
                                    'inner_score': selection['inner_score']},
                        'depths': probes, 'movement': movement})
    depths = len(records[0]['depths'])
    majority_by_depth = [majority(np.stack([depth_predictions[v][d] for v in range(len(records))])) for d in range(depths)]
    state_hashes = [net.state_hash() for _, net in model.views_]
    record = {'arm_id': arm, 'params': params, 'spec': spec, 'fit_seconds': fit_seconds,
              'encoding_seconds': model.encoding_seconds_, 'training_seconds': model.training_seconds_,
              'readout_seconds': model.readout_seconds_, 'widths': list(params['widths']),
              'hidden_layers': depths, 'views': records, 'state_hashes': state_hashes,
              'neighborhood_purity': float(np.mean([r['depths'][-1]['neighborhood_purity'] for r in records])),
              'majority_by_depth': [{'depth': d, 'knn_accuracy': td.accuracy(majority_by_depth[d], y_query)}
                                    for d in range(depths)],
              'prediction_hash': array_hash(np.asarray(prediction)), **metric_values(y_query, prediction)}
    if instrument:
        record['movement'] = aggregate_movement(spec, records)
    arrays = {'view_predictions': view_predictions, 'majority_by_depth': np.stack(majority_by_depth),
              'predictions': np.asarray(prediction)}
    return record, arrays, checks, prediction, state_hashes


def incoming_to_prior_ratio(spec, layer):
    """The realised ratio of incoming vote mass to the prior weight in one layer: the mass itself under the unit prior,
    and the ballot count over the multiplier under the one-ballot prior."""
    if layer['updates'] == 0 or layer['mean_votes'] is None:
        return None
    if spec['rule'] != 'median' or spec['prior_rule'] == 'unit':
        return float(layer['mean_vote_mass'])
    return float(layer['mean_votes']) / float(spec['prior_multiplier'])


def aggregate_movement(spec, view_records):
    """Movement over the views of one arm: per layer and over all hidden layers, the share of filters that changed per
    batch and the mean normalised displacement, with the realised incoming-to-prior mass ratio."""
    layers = defaultdict(list)
    for record in view_records:
        for layer in record['movement']['layers']:
            layers[layer['layer']].append(layer)
    per_layer = []
    for index in sorted(layers):
        entries = [e for e in layers[index] if e['updates']]
        updates = sum(e['updates'] for e in entries)
        per_layer.append({'layer': index, 'layer_name': layers[index][0]['layer_name'], 'views': len(layers[index]),
                          'updates': updates,
                          'changed_share': float(sum(e['changed_share'] * e['updates'] for e in entries) / updates) if updates else None,
                          'mean_displacement': float(sum(e['mean_displacement'] * e['updates'] for e in entries) / updates) if updates else None,
                          'mean_votes': float(sum(e['mean_votes'] * e['updates'] for e in entries) / updates) if updates else None,
                          'mean_vote_mass': float(sum(e['mean_vote_mass'] * e['updates'] for e in entries) / updates) if updates else None,
                          'incoming_to_prior_ratio': float(np.mean([e['incoming_to_prior_ratio'] for e in entries])) if updates else None,
                          'median_solves': int(sum(e['median_solves'] for e in layers[index])),
                          'tie_canonicalisation_incomplete': int(sum(e['tie_canonicalisation_incomplete'] for e in layers[index]))})
    updates = sum(entry['updates'] for entry in per_layer)
    changed = sum((entry['changed_share'] or 0.) * entry['updates'] for entry in per_layer)
    displacement = sum((entry['mean_displacement'] or 0.) * entry['updates'] for entry in per_layer)
    # Inertness is read on the layers the arm is allowed to move: a frozen layer is zero by construction, not inert.
    updatable = [entry for entry in per_layer if entry['layer'] not in spec['frozen_hidden']]
    free = sum(entry['updates'] for entry in updatable)
    free_changed = sum((entry['changed_share'] or 0.) * entry['updates'] for entry in updatable)
    return {'rule': spec['rule'], 'prior_rule': spec['prior_rule'], 'prior_multiplier': spec['prior_multiplier'],
            'frozen_hidden_layers': list(spec['frozen_hidden']), 'hidden_updates': updates,
            'changed_share': float(changed / updates) if updates else None,
            'mean_displacement': float(displacement / updates) if updates else None,
            'updatable_changed_share': float(free_changed / free) if free else None,
            'effectively_inert': bool(free and free_changed / free < INERT_THRESHOLD), 'layers': per_layer}


# ----------------------------------------------------------------------------- one job

def evaluate_job(X, y, split, job, sealed, p, *, dataset_hash, code_revision, protocol_hash, query=None):
    """One dataset, outer fold and seed: every arm of the family, its checks and measures. query=None scores the outer
    test rows; the pilot passes training rows instead (training-only timing, no reference-prediction check)."""
    family, seed = p['production_family'], int(p['model_seed'])
    specs = arm_specs(family, p)
    train = [int(i) for i in split['train']]
    rows = [int(i) for i in (split['test'] if query is None else query)]
    X_train, y_train, X_query, y_query = X[train], y[train], X[rows], y[rows]
    identity = {'family': family, 'dataset_id': job['dataset_id'], 'reference': job['reference'], 'dataset_hash': dataset_hash,
                'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'], 'split_hash': config_id(split),
                'query': 'outer_test_rows' if query is None else 'training_rows_for_timing',
                'query_ids_hash': array_hash(np.asarray(rows)), 'config_id': job['config_id'], 'config': job['config'],
                'selected': job['selected'], 'own_arm': job['own_arm'], 'model_seed': seed,
                'check_uninstrumented': bool(job['check_uninstrumented']), 'code_revision': code_revision,
                'protocol_hash': protocol_hash}
    checks, timing, arms = {}, {}, {}
    result = {'status': 'running', 'identity': identity, 'checks': checks, 'timing': timing}
    arrays = None
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            collected, stored, flags = {}, {}, defaultdict(lambda: True)
            for arm in p['arms']:
                planned = next(entry for entry in job['arms'] if entry['arm_id'] == arm)
                params = arm_params(specs[arm], job['selected'], p['arm_widths'])
                if params != planned['params']:
                    raise CheckFailed(f'{arm}: the fitted parameters differ from the planned ones')
                record, arm_arrays, arm_checks, prediction, hashes = evaluate_arm(
                    arm, specs[arm], params, seed, X_train, y_train, X_query, y_query)
                collected[arm], stored[arm] = record, arm_arrays
                arms[arm] = {'prediction': prediction, 'state_hashes': hashes}
                for name, value in arm_checks.items():
                    flags[name] &= bool(value)
                timing[f'{arm}_seconds'] = record['fit_seconds']
            checks['arm_parameters'] = {'performed': True, 'passed': True}
            for name in ('instrumentation_removed', 'rng_untouched', 'frozen_layers_unchanged', 'depth_probe_matches_readout'):
                checks[name] = {'performed': True, 'passed': bool(flags[name])}
            own = job['own_arm']
            if query is None:
                checks['reference_predictions'] = td.reference_check(arms[own]['prediction'], sealed, seed)
                checks['reference_predictions']['arm_id'] = own
                if not checks['reference_predictions']['passed']:
                    raise CheckFailed(f'{own} differs from the reference on '
                                      f'{checks["reference_predictions"]["n_differing"]} of '
                                      f'{checks["reference_predictions"]["n_test"]} outer test rows')
            else:
                checks['reference_predictions'] = {'performed': False, 'passed': None, 'arm_id': own,
                                                   'reason': 'training-only pilot: the query rows are outer training rows'}
            if job['check_uninstrumented']:
                start = time.perf_counter()
                seed_fit(seed)
                plain = MultiViewArrowFlowKNN(**arm_params(specs[own], job['selected'], p['arm_widths']), seed=seed).fit(X_train, y_train)
                plain_views = np.stack([np.asarray(view) for view in plain.predict_views(X_query)[0]])
                plain_hashes = [net.state_hash() for _, net in plain.views_]
                timing['uninstrumented_fit_seconds'] = time.perf_counter() - start
                equal_views = bool(np.array_equal(plain_views, stored[own]['view_predictions']))
                checks['uninstrumented_state_hash'] = {'performed': True, 'arm_id': own,
                                                       'passed': plain_hashes == arms[own]['state_hashes'] and equal_views,
                                                       'view_state_hashes': arms[own]['state_hashes'],
                                                       'uninstrumented_view_state_hashes': plain_hashes,
                                                       'view_predictions_equal': equal_views}
                del plain
                if not checks['uninstrumented_state_hash']['passed']:
                    raise CheckFailed('a view state hash or prediction of the unmodified arm differs from the uninstrumented fit')
            else:
                timing['uninstrumented_fit_seconds'] = 0.
                checks['uninstrumented_state_hash'] = {'performed': False, 'passed': None, 'arm_id': own,
                                                       'reason': 'performed on the first outer fold of each dataset'}
        failed = sorted(name for name, check in checks.items() if check['performed'] and not check['passed'])
        if failed:
            raise CheckFailed('checks failed: ' + ', '.join(failed))
        arrays = {'query_rows': np.asarray(rows), 'query_labels': np.asarray(y_query)}
        for arm in p['arms']:
            for name, value in stored[arm].items():
                arrays[f'{arm}__{name}'] = value
            for record in collected[arm]['views']:
                if record['movement'] is not None:
                    for layer in record['movement']['layers']:
                        arrays[f'{arm}__v{record["view"]}__l{layer["layer"]}__changed'] = np.asarray(layer.pop('batch_changed_share'))
                        arrays[f'{arm}__v{record["view"]}__l{layer["layer"]}__displacement'] = np.asarray(layer.pop('batch_mean_displacement'))
        result.update(status='ok', arms=[collected[arm] for arm in p['arms']])
    except Exception as exc:
        result.update(status='failed', exception=f'{type(exc).__name__}: {exc}', check_failure=isinstance(exc, td.CheckFailed))
        arrays = None
    timing['job_seconds'] = time.perf_counter() - started
    return td.native(result), arrays


# ----------------------------------------------------------------------------- protocol

def design(family):
    """The fixed design every protocol of this family declares (validate_protocol requires it verbatim)."""
    specs = arm_specs(family)
    return {
        'purpose': {'depth': 'a controlled depth comparison of ArrowFlow-kNN at one fixed configuration per outer fold',
                    'aggregation': 'a controlled aggregation comparison inside ArrowFlow-kNN: the Borda order of the '
                                   'hidden-layer update against the footrule median of the same votes'}[family],
        'selection_statement': 'Nothing is selected on outer-fold results. Every configuration is the per-fold selection '
                               'sealed by the reference ablation run; every kNN readout setting is the one the fit chose '
                               'on training rows; the mass-matching multiplier is chosen on the training-only pilot '
                               'before the run; no outer score feeds back into any fit, arm, dataset, fold or analysis '
                               'choice.',
        'unit_of_work': 'every protocol dataset x every outer fold (5 folds x 3 repeats) x fitting seed 8129 x every arm',
        'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3, 'split_seed': 27183, 'fit_seeds': [8129, 19391, 39019],
        'model_seed': MODEL_SEED, 'n_views': N_VIEWS, 'test_train_ratio': .25, 'confidence': .95,
        'fit_seed_statement': 'The reference design fits three seeds per outer fold; these controlled arms are fitted at '
                              'seed 8129 alone, because four (depth) and three (aggregation) full seven-view fits per '
                              'outer fold at three seeds project beyond the approved budget for this follow-up. The arms '
                              'of one fold therefore share one encoder draw and one initial state, so the comparison is '
                              'paired within seed; no within-fold seed spread is estimated and every interval is over the '
                              'fifteen outer folds of that one seed.',
        'arms': list(ARMS[family]), 'arm_definitions': arm_definitions(family), 'arm_widths': dict(ARM_WIDTHS),
        'reference_arm': REFERENCE_ARM[family],
        'own_arm': 'the arm whose parameters are the fold\'s own reconstructed selection (in the depth family depth1 or '
                   'depth2 according to the selected widths, in the aggregation family always borda); it must reproduce '
                   'the reference run\'s outer predictions for seed 8129 exactly',
        'fit': 'MultiViewArrowFlowKNN at bridge.resolve_selected(sealed config, n_features, n_train) on the outer '
               'training rows with the arm\'s hidden widths and update rule: for view v, OrdinalEncoder(view_strategy, '
               'embed_dim, degree, lda_ratio, derive_seed(8129, "view", v)), ArrowFlowEstimator.initialize_orders then '
               'train_initialized, then MultiViewArrowFlowKNN._fit_view_readout (select_knn_readout on the trained '
               'hidden ranking); majority vote over the seven per-view kNN readouts',
        'not_retuned': 'the learning rate, the iteration count, the batch size, the checkpoint ratio, the encoder, the '
                       'vocabulary, the degree and the augmentation are the fold\'s own selection and are not re-tuned '
                       'for any arm; this is a controlled comparison at one configuration and not an estimate of the '
                       'best attainable model of any arm',
        'instrumentation': 'instance-level wrappers installed on each view network between initialize_orders and '
                           'train_initialized and removed afterwards (update_rules.ArmInstrumentation): '
                           'Vertex.accumulate_motion counts the eligible votes and their mass, Vertex.apply_motion '
                           'records the movement of the filter and freezes it in a frozen layer, and in a median arm '
                           'Vertex.compute_adj_list_with_permutation returns the footrule-median order of the same '
                           'accumulator. The signal generation, the eligibility gate, the initial state, the data, the '
                           'randomization, the batch resets and the readout protocol are untouched; the wrapper around '
                           'update_network only closes the per-batch record inside a guard that restores the global '
                           'numpy and Python RNG states',
        'measures': {
            'outer_metrics': 'per arm, the seven-view majority on the outer test rows: accuracy, error, balanced '
                             'accuracy and macro-F1 (evaluation.metric_values)',
            'neighborhood_purity': 'per arm, the mean over views of the share of same-class neighbours of the outer '
                                   'test rows in the last hidden ranking: the k nearest training rows under the '
                                   'footrule distance, k the view readout\'s own n_neighbors',
            'depth_probe': 'per arm, view and hidden layer, the representation after that layer '
                           '(models.transform_orders_by_depth) read out by a footrule kNN at the view\'s own selected '
                           'setting refitted on the training rows at that depth, scored on the outer test rows, with '
                           'its neighbourhood purity and the seven-view majority at each depth; at the last hidden '
                           'layer the probe must equal the view readout',
            'movement': 'per arm, view and hidden layer, over the training batches: the share of filters that changed '
                        'and the mean normalised footrule displacement (footrule / floor(V^2 / 2)), with the number of '
                        'eligible votes, the incoming vote mass and the realised incoming-to-prior mass ratio',
            'inertness': f'an arm is flagged effectively inert on a fold when its changed-filter share over the hidden '
                         f'layers is below {INERT_THRESHOLD}; a contrast against an effectively inert arm is never read '
                         f'as evidence about the aggregation rule'},
        'mass_matching': {
            'status': 'the fairness rule of the aggregation family; fixed before the run and recorded in the frozen '
                      'protocol' if family == 'aggregation' else 'not applicable to this family',
            'problem': 'matching the step size alone is not a fair control: the accumulator carries the prior at weight '
                       'exactly one while one batch of votes carries a mass of order 0.1, so the footrule median can be '
                       'frozen at the prior while Borda, being a mean, still moves',
            'rule': 'median keeps the accumulator as the core builds it (prior weight 1, the plain substitution). '
                    'median_mass_matched gives the prior m times the mean incoming ballot weight (mass / votes), so the '
                    'incoming-to-prior mass ratio is votes / m to 1',
            'ladder': list(MASS_LADDER),
            'selection': 'm is the ladder value whose mean per-batch changed-filter share over the hidden layers, on the '
                         'training-only pilot fits of the pilot datasets, is closest to the borda arm\'s on the same '
                         'partitions; ties to the smallest m. No outer-fold score takes part',
            'reporting': 'the movement rates of every arm are reported, so a moving-against-inert comparison is never '
                         'read as evidence about independence of irrelevant alternatives'},
        'median_rule': {
            'objective': 'the hidden-layer update minimises, over the V! orders of the filter, the weighted sum of rank '
                         'distances to the prior (weight 1) and to each vote (weight |a_t|); the core uses the squared '
                         'rank distance, whose minimiser is the accumulator row mean (a weighted Borda count, verified '
                         'in review-math-actions.md section 2(a)), and the median arm uses the footrule distance',
            'computation': 'the footrule median is the linear assignment on C = A D with D[q, p] = |q - p| '
                           '(scipy.optimize.linear_sum_assignment), the exact minimiser over all orders',
            'ties': 'the solver\'s optimum is canonicalised by zero-cost transpositions toward the lexicographically '
                    'smallest position vector read in ascending item order; what remains is the deterministic choice of '
                    'the pinned solver among exactly tied optima. On 400 random small accumulators the returned order '
                    'attained the minimum in every case and equalled the lexicographically smallest minimiser in 393 '
                    '(66 instances had tied optima); the brute-force test records this',
            'scope': 'hidden layers only; the output layer keeps the core rule, so the checkpoint and the output rule '
                     'are unchanged'},
        'checks': {
            'reference_predictions': 'every job: the own arm\'s seven-view predictions on the outer test rows equal the '
                                     'reference run\'s recorded predictions for seed 8129 exactly, in the sealed order',
            'uninstrumented_state_hash': 'first outer fold of every dataset: the own arm\'s view state hashes and '
                                         'predictions equal MultiViewArrowFlowKNN(**selected, seed=8129).fit\'s on the '
                                         'same rows',
            'instrumentation_removed': 'every job: no vertex of any reachable graph keeps an instance-level wrapper',
            'rng_untouched': 'every job: every guarded read left the global numpy and Python RNG states unchanged',
            'frozen_layers_unchanged': 'every job: a frozen hidden layer\'s filters equal their initial ones and its '
                                       'recorded movement is exactly zero',
            'depth_probe_matches_readout': 'every job: the last depth probe equals the view readout',
            'arm_parameters': 'every job: the fitted parameters equal the planned ones'},
        'outputs': {'job_records': 'jobs/<dataset>__r<repeat>f<fold>.json with every measure and check, '
                                   'artifacts/<stem>.npz with the per-arm view, depth and majority predictions and the '
                                   'per-batch movement series, predictions/<stem>.jsonl with the per-example outer '
                                   'predictions of every arm',
                    'tables': 'arms.csv, depth_probe.csv, movement.csv and provenance.json, written only after every '
                              'planned job succeeded and verified (all or none)',
                    'summary': f'{family}_summary.json and {family}_summary.csv by the summary stage',
                    'analysis': f'{family}_contrasts.csv and {family}_analysis.json by the analyse stage',
                    'overwrite': 'an existing file with different content is never replaced'},
        'analysis': analysis_declaration(family),
        'dataset_loading': 'the reference run prepared data (hash-checked) must equal the ablation copy, its splits the '
                           'declared nested splits, and in production a fresh load by the reference loader must return '
                           'the same arrays and dataset hash',
        'pilot': 'training-only: the first outer fold of each pilot dataset with the outer test rows replaced by every '
                 'fourth outer training row (the outer test fold is never touched), every arm, measure and check except '
                 'the reference predictions; one reproduction probe (run_knn_ablation.reproduction_probe) of the '
                 'smallest dataset per reference; in the aggregation family the mass-matching ladder',
        'decision_rule': f'freeze only if the calibrated projection at {WORKERS} single-thread workers is at most '
                         f'{CAP_HOURS[family]} h: the simulated first-free-worker makespan of the planned jobs in '
                         'planned order, each job priced at the reference run\'s realized outer fit and predict seconds '
                         'of its fold and seed 8129 times the piloted job / uninstrumented-fit ratio (its own ratio for '
                         'a pilot dataset, the largest piloted ratio otherwise), plus the realized seconds once more for '
                         'the uninstrumented fit of a first fold. Both sides of the ratio are measured in the same '
                         'pilot, so machine contention largely cancels in it while the realized reference seconds carry '
                         'the absolute scale',
        'wallclock_cap_hours': CAP_HOURS[family], 'workers': WORKERS, 'max_workers': MAX_WORKERS,
        'numeric_threads_per_worker': 1,
        'parallelism': 'one spawned single-thread process per dataset and outer fold under the shared execution lock',
        'failure_policy': 'a failed check fails its job; a failed job cancels the pending jobs and no table is written; '
                          'failures are never omitted; no adaptive stopping on any value',
        'arm_rules': {arm: {key: specs[arm][key] for key in sorted(specs[arm]) if key != 'prior_multiplier'}
                      for arm in ARMS[family]},
    }


def arm_definitions(family):
    if family == 'depth':
        return {
            'depth1': 'the single hidden layer [128]: the unmodified rule and the unmodified architecture',
            'depth2': 'two hidden layers [64, 128]: the unmodified rule, one more ranking layer',
            'depth2_untrained_second': '[64, 128] with the second hidden layer frozen at its initial filters (its votes '
                                       'are accumulated and discarded, its error signal to the first layer is '
                                       'unchanged), so the extra layer adds structure without learning',
            'depth2_first_only': '[64, 128] with only the first hidden layer updated: the second hidden layer is frozen '
                                 'as above and the output layer is held at its initial filters through the library\'s '
                                 'last_layer_update=False, which still generates the error signal',
            'note': 'the four arms share the fold\'s encoder, view identities, initial seeds, learning rate, vocabulary, '
                    'degree, augmentation, checkpoint and readout rule; only the hidden widths and which layers are '
                    'updated differ'}
    return {
        'borda': 'the unmodified update: the hidden-layer order is the accumulator row mean with ties by ascending item '
                 'ID, the weighted Borda count of the votes and the prior',
        'median': 'the same votes, the same weights and the same prior weight 1, ordered by the exact footrule median '
                  '(the linear assignment on C = A D) instead of the row mean',
        'median_mass_matched': 'the footrule median with the prior entered as m ballots of the mean incoming weight, m '
                               'chosen on the training-only pilot so that the movement rate is comparable to borda\'s',
        'note': 'the three arms share the fold\'s selection, encoder, view identities, initial seeds, data, '
                'randomization, eligibility gate, batch resets, checkpoint and readout rule; the output layer keeps the '
                'core rule in every arm'}


def analysis_declaration(family):
    reference, variants = REFERENCE_ARM[family], [arm for arm in ARMS[family] if arm != REFERENCE_ARM[family]]
    block = {
        'status': 'prespecified before any outer score of this family exists; computed only by the analyse stage, which '
                  'refuses (exit 2, nothing written) until the run is complete, then re-verifies every job record, '
                  're-derives every sealed selection from its reference run and re-renders every table before any score '
                  'is read',
        'command': 'python -m experiments.make_revision.interventions analyse --run <run> --output <directory>',
        'metric': 'accuracy', 'reference_arm': reference, 'variant_arms': variants,
        'definition': f'per dataset and variant arm, {reference} minus the variant arm accuracy, the fitting seeds '
                      'averaged within each outer fold (one seed here, so the fold value is that seed\'s)',
        'interval': 'corrected resampled t over the 15 outer folds (evaluation.paired_corrected_interval, q = 0.25, '
                    '95 per cent, 14 degrees of freedom)',
        'multiplicity': 'Holm across the seventeen datasets within each variant arm (evaluation.holm_adjust); the arms '
                        'are adjusted separately and no adjustment crosses the two families',
        'alpha': .05, 'families': [{'arm': arm, 'size': 17} for arm in variants],
        'descriptive': 'per arm and dataset: accuracy, balanced accuracy, macro-F1 and the neighbourhood purity of the '
                       'last hidden ranking (outer-fold mean and SD); the depth probe at each hidden layer; the '
                       'movement rates; no p value and no adjustment for these',
        'interpretation': {'depth': 'if the second hidden layer does not help at a matched configuration, the paper says '
                                    'so and keeps the stack as a construction rather than an empirical benefit',
                           'aggregation': 'if the median matches Borda once the vote mass is matched, the paper says the '
                                          'aggregation choice is not the source of the learning effect'}[family],
        'limits': 'a controlled comparison at one configuration per fold, at one fitting seed, with the learning rate '
                  'and the iteration count not re-tuned; it cannot estimate the best attainable model of any arm, and '
                  'it cannot separate an arm that does not move from an arm that moves and does not help unless the '
                  'movement rates are read with it'}
    if family == 'depth':
        block['named_subset'] = {
            'status': 'a named subset of the same family, reported beside it; descriptive, not a new family, no p value '
                      'and no place in any Holm family',
            'definition': f'the same {reference} minus variant differences restricted to the outer folds whose own '
                          'reconstructed selection was two hidden layers, and to those whose selection was one '
                          '(knn_controls.depth_split with depths ' + canonical_json(DEPTHS) + ')',
            'folds': 'no dataset selected two hidden layers on all fifteen outer folds: seven of the seventeen selected '
                     'them on none and the rest on three to eleven of fifteen (70 of 255 folds). Every subset row '
                     'carries its own fold count, its degrees of freedom are n - 1, and a corrected resampled t interval '
                     'is reported only where the subset has at least three folds',
            'caveat': 'the fold sets differ by dataset and are chosen by the selection itself, so the subsets describe '
                      'where the two-layer selection happened and never carry an inferential claim. The prespecified '
                      f'{reference}-minus-variant contrast is defined on all fifteen outer folds of every dataset and is '
                      'unaffected by this'}
    else:
        block['inertness'] = {
            'status': 'read before any contrast of this family',
            'definition': f'an arm is effectively inert on a fold when its changed-filter share over the hidden layers '
                          f'is below {INERT_THRESHOLD}; the analysis reports the share of folds on which each arm is '
                          'effectively inert and states that a contrast against an effectively inert arm is evidence '
                          'about movement, not about the aggregation rule'}
    return block


def draft_protocol(family, references=None, protocol_id=None, pilot_datasets=PILOT_DATASETS):
    """The unfrozen protocol; the datasets follow the references in name order (a JSON protocol keeps no key order)."""
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {", ".join(FAMILIES)}')
    references = td.REFERENCES if references is None else references
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    return _plain({**design(family), 'protocol_id': protocol_id or PROTOCOL_IDS[family], 'production_family': family,
                   'datasets': datasets, 'references': references, 'pilot_datasets': list(pilot_datasets),
                   'frozen': False, 'status': DRAFT_STATUS, 'mass_matching_choice': None,
                   'resource_decision': 'pending: synthetic smoke and the training-only pilot on ' + ' and '.join(pilot_datasets)})


def validate_protocol(p):
    """A production protocol is draft_protocol for its family and references, or that draft with exactly the freeze
    fields set by freeze."""
    family = p.get('production_family')
    draft = draft_protocol(family if family in FAMILIES else FAMILIES[0], references=p.get('references'),
                           protocol_id=p.get('protocol_id'), pilot_datasets=p.get('pilot_datasets', ()))
    strip = lambda q: {k: v for k, v in q.items() if k not in FREEZE_FIELDS}
    if strip(p) != strip(draft):
        differing = sorted(k for k in set(p) | set(draft) if k not in FREEZE_FIELDS and p.get(k) != draft.get(k))
        raise ValueError(f'The protocol differs from interventions.draft_protocol({family!r}) in {", ".join(differing)}')
    td.reference_of(p)
    if not p.get('frozen'):
        if p != draft:
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('pilot_projection') or {}
    hours, cap = projection.get('decision_hours'), CAP_HOURS[family]
    matching = p.get('mass_matching_choice') or {}
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != cap or projection.get('workers') != WORKERS
            or isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= cap):
        raise ValueError(f'A frozen protocol records its freeze and a pilot projection within the {cap} h cap at {WORKERS} workers')
    if family == 'aggregation' and matching.get('prior_multiplier') not in MASS_LADDER:
        raise ValueError('A frozen aggregation protocol records the mass-matching multiplier chosen on the pilot')
    return p


# ----------------------------------------------------------------------------- prepare and verify

def planned_job(p, reference, name, index, split, record, n_features):
    specs = arm_specs(p['production_family'], p)
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    sealed_job = reference['ablation']['jobs'].get((name, split['outer_repeat'], split['outer_fold']))
    if sealed_job is None or sealed_job['config_id'] != record['config_id'] or sealed_job['selected'] != selected:
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the resolved selection differs from the ablation plan')
    if p['production_family'] == 'depth' and list(selected['widths']) not in list(p['arm_widths'].values()):
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the selected widths '
                         f'{list(selected["widths"])} are not one of the declared depth arms')
    seed = str(p['model_seed'])
    return {'dataset_id': name, 'reference': reference['name'], 'outer_repeat': split['outer_repeat'],
            'outer_fold': split['outer_fold'], 'stem': f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}',
            'config_id': record['config_id'], 'config': record['config'], 'selected': selected,
            'selected_widths': list(selected['widths']), 'model_seed': int(p['model_seed']),
            'check_uninstrumented': index == 0, 'own_arm': own_arm(p['production_family'], specs, selected, p['arm_widths']),
            'arms': [{'arm_id': arm, 'params': arm_params(specs[arm], selected, p['arm_widths']),
                      'widths': arm_widths(specs[arm], selected, p['arm_widths'])} for arm in p['arms']],
            'reference_prediction_hash': record['reference_prediction_hashes'][seed],
            'reference_outer_seconds': record['reference_outer_seconds'][seed]}


def prepare(output, p, sources, datasets=None, *, allow_smoke=False, purpose='intervention'):
    """Verify the references, copy the prepared data, reconstruct every selection (it must equal the sealed record and
    resolve to the ablation plan) and seal the planned jobs."""
    output = Path(output)
    mapping = td.reference_of(p)
    chosen = list(p['datasets']) if datasets is None else [name for name in p['datasets'] if name in set(datasets)]
    if not chosen:
        raise ValueError('The datasets must be distinct protocol datasets')
    needed = [name for name in p['references'] if any(mapping[d] == name for d in chosen)]
    missing = [name for name in needed if name not in sources]
    if missing:
        raise ValueError(f'Supply --reference for {", ".join(missing)}')
    references = td.load_references(p, {name: sources[name] for name in needed}, allow_smoke=allow_smoke)
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs, identities = [], [], {}
    for name in chosen:
        reference = references[mapping[name]]
        X, y, manifest, splits, identities[name] = td.prepare_dataset(output, reference, name, p, production=not allow_smoke)
        for index, split in enumerate(splits):
            validate_split(split, len(y))
            record = base.selection_record(reference['run'], name, split, y, manifest)
            if record != reference['ablation']['selections'].get((name, split['outer_repeat'], split['outer_fold'])):
                raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the selection reconstructed '
                                 'from the reference run differs from the sealed record')
            selections.append(record)
            jobs.append(planned_job(p, reference, name, index, split, record, X.shape[1]))
    write_json(output/'reference_selections.json', selections)
    write_json(output/'planned_jobs.json', jobs)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'family': p['production_family'], 'protocol_id': p['protocol_id'],
        'protocol_hash': config_id(p), 'datasets': chosen, 'arms': list(p['arms']), 'model_seed': p['model_seed'],
        'planned_jobs': len(jobs), 'planned_jobs_sha256': sha256_file(output/'planned_jobs.json'),
        'reference_selections_sha256': sha256_file(output/'reference_selections.json'), 'dataset_identity': identities,
        'references': {name: {'run_directory': str(Path(sources[name][0]).resolve()),
                              'ablation_directory': str(Path(sources[name][1]).resolve()),
                              'run': reference['observed_run'], 'ablation': reference['ablation']['observed'],
                              'run_file_sha256': reference['run']['files'],
                              'datasets': [d for d in chosen if mapping[d] == name]}
                       for name, reference in references.items()}})
    return jobs, references


def verify(output, *, allow_smoke=False):
    """The sealed run directory: protocol (frozen unless a synthetic smoke), manifest seal, unchanged scientific
    sources, planned jobs and sealed selections."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    if not smoke:
        validate_protocol(p)
        if not p['frozen'] or manifest.get('purpose') != 'intervention':
            raise ValueError('An intervention run directory with a frozen reviewed protocol is required')
    if manifest['protocol_hash'] != config_id(p) or manifest['family'] != p['production_family'] or manifest['arms'] != list(p['arms']):
        raise ValueError('Protocol seal changed')
    if json.loads((output/'environment.json').read_text())['source_hashes'] != environment()['source_hashes']:
        raise ValueError('Source seal changed')
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
    return p, manifest, jobs, selections


def worker(arguments):
    output, job = arguments
    output, stem = Path(output), job['stem']
    record_path, artifact_path = output/'jobs'/f'{stem}.json', output/'artifacts'/f'{stem}.npz'
    prediction_path = output/'predictions'/f'{stem}.jsonl'
    if any(path.exists() for path in (record_path, artifact_path, prediction_path)):
        raise FileExistsError(f'Existing intervention job {stem}')
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    key = (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == key[1:])
    sealed = next(s for s in json.loads((output/'reference_selections.json').read_text())
                  if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == key)
    p = json.loads((output/'protocol.json').read_text())
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    result, arrays = evaluate_job(X, y, split, job, sealed, p, dataset_hash=data['dataset_hash'], code_revision=revision,
                                  protocol_hash=config_id(p))
    if arrays is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        result['artifact'] = {'path': f'artifacts/{stem}.npz', 'sha256': sha256_file(artifact_path)}
        records = prediction_records(arrays, job, split, y, revision, p['arms'])
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        with prediction_path.open('x') as stream:
            for record in records:
                stream.write(canonical_json(record) + '\n')
        result['prediction_file'] = {'path': f'predictions/{stem}.jsonl', 'records': len(records),
                                     'sha256': sha256_file(prediction_path)}
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open('x') as stream:
        stream.write(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + '\n')
    return str(record_path), result['status']


def prediction_records(arrays, job, split, y, revision, arms):
    """The per-example outer predictions of every arm, in the sealed test-sample order."""
    rows = []
    for arm in arms:
        rows.extend({'dataset_id': job['dataset_id'], 'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'],
                     'arm_id': arm, 'model_seed': job['model_seed'], 'sample_id': int(sample),
                     'y_true': np.asarray(y[sample]).item(), 'y_pred': np.asarray(label).item(),
                     'config_id': job['config_id'], 'code_revision': revision}
                    for sample, label in zip(split['test'], arrays[f'{arm}__predictions']))
    return rows


def uncommitted_sources(extra=()):
    """Sealed source files of this family (and extra repository paths) that differ from HEAD or are untracked."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted(environment()['source_hashes'])
    for path in extra:
        resolved = Path(path).resolve()
        if root not in resolved.parents:
            raise ValueError(f'{resolved} is outside the repository; a production protocol is a committed repository file')
        paths.append(str(resolved.relative_to(root)))
    status = subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], text=True, cwd=root)
    return [line[3:] for line in status.splitlines() if line.strip()]


def execute(output, jobs, workers):
    """Every planned job on spawned single-thread workers; a failed job cancels the pending ones."""
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, (str(output), job)) for job in jobs]
        try:
            for future in as_completed(futures):
                path, status = future.result()
                print(path, status, flush=True)
                if status != 'ok':
                    raise CheckFailed(f'{path}: the job failed; the pending jobs are cancelled and no table is written')
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def run(output, workers=WORKERS, *, allow_smoke=False, protocol_path=None):
    """Every planned job of a prepared directory."""
    output = Path(output)
    p, manifest, jobs, _ = verify(output, allow_smoke=allow_smoke)
    if not 1 <= int(workers) <= int(p['max_workers']):
        raise ValueError(f'Worker count must be between 1 and {p["max_workers"]}')
    if not allow_smoke:
        dirty = uncommitted_sources(() if protocol_path is None else (protocol_path,))
        if dirty:
            raise ValueError(f'Commit the sealed sources and the protocol before the run: {", ".join(dirty)}')
    started = utc_now()
    with execution_lock():
        execute(output, jobs, int(workers))
    return {'started_utc': started, 'ended_utc': utc_now(), 'workers': int(workers), 'jobs': len(jobs)}


# ----------------------------------------------------------------------------- verification and tables

def validate_record(record, job, p, data, split, y, revision, output, sealed):
    """Bind one job record to the sealed plan and recompute its metrics, purity-free quantities and predictions from the
    saved arrays and the saved per-example predictions."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(record.get('status') == 'ok', f'job status {record.get("status")}: {record.get("exception")}')
    seed, test = int(p['model_seed']), [int(i) for i in split['test']]
    identity = {'family': p['production_family'], 'dataset_id': job['dataset_id'], 'reference': job['reference'],
                'dataset_hash': data['dataset_hash'], 'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'],
                'split_hash': config_id(split), 'query': 'outer_test_rows', 'query_ids_hash': array_hash(np.asarray(test)),
                'config_id': job['config_id'], 'config': job['config'], 'selected': job['selected'],
                'own_arm': job['own_arm'], 'model_seed': seed, 'check_uninstrumented': job['check_uninstrumented'],
                'code_revision': revision, 'protocol_hash': config_id(p)}
    require(record['identity'] == identity, 'job identity disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'],
            'the plan disagrees with the sealed selection')
    require(sorted(record['checks']) == sorted(JOB_CHECKS), 'check schedule')
    for name in JOB_CHECKS:
        performed = job['check_uninstrumented'] if name == 'uninstrumented_state_hash' else True
        check = record['checks'][name]
        require(check['performed'] is performed and check['passed'] is (True if performed else None), f'check {name}')
    artifact, stem = record['artifact'], job['stem']
    require(artifact['path'] == f'artifacts/{stem}.npz' and (output/artifact['path']).is_file()
            and sha256_file(output/artifact['path']) == artifact['sha256'], 'artifact hash')
    with np.load(output/artifact['path'], allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    truth = y[test]
    require(np.array_equal(arrays['query_rows'], test) and np.array_equal(arrays['query_labels'], truth),
            'artifact rows or labels')
    require([arm['arm_id'] for arm in record['arms']] == list(p['arms']), 'arm schedule')
    prediction_path = output/record['prediction_file']['path']
    require(record['prediction_file']['path'] == f'predictions/{stem}.jsonl' and prediction_path.is_file()
            and sha256_file(prediction_path) == record['prediction_file']['sha256'], 'prediction file hash')
    lines = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    require(len(lines) == record['prediction_file']['records'] == len(p['arms']) * len(test), 'prediction record count')
    cells = defaultdict(list)
    labels = set(np.asarray(y).tolist())
    for row in lines:
        require(sorted(row) == sorted(PREDICTION_KEYS), 'prediction record schema')
        require(row['dataset_id'] == job['dataset_id'] and (row['outer_repeat'], row['outer_fold']) ==
                (job['outer_repeat'], job['outer_fold']) and row['config_id'] == job['config_id']
                and row['code_revision'] == revision and row['model_seed'] == seed, 'prediction identity')
        require(row['y_true'] == y[row['sample_id']] and row['y_pred'] in labels, 'prediction truth/label')
        cells[row['arm_id']].append((row['sample_id'], row['y_pred']))
    require(set(cells) == set(p['arms']), 'prediction arm coverage')
    specs = arm_specs(p['production_family'], p)
    for arm in record['arms']:
        name = arm['arm_id']
        predicted = arrays[f'{name}__predictions']
        require([sample for sample, _ in cells[name]] == test, 'prediction sample order')
        require(np.array_equal([label for _, label in cells[name]], predicted), 'predictions disagree with the artifact')
        require(arm['params'] == next(e['params'] for e in job['arms'] if e['arm_id'] == name)
                and arm['widths'] == next(e['widths'] for e in job['arms'] if e['arm_id'] == name), 'arm parameters')
        require(arm['spec'] == _plain(specs[name]), 'arm specification')
        require(arm['prediction_hash'] == array_hash(np.asarray(predicted)), 'prediction hash')
        metrics = metric_values(truth, predicted)
        require(all(np.isclose(arm[key], value, rtol=0, atol=1e-12) for key, value in metrics.items()),
                'metric/prediction disagreement')
        views = arrays[f'{name}__view_predictions']
        require(views.shape == (p['n_views'], len(test)) and np.array_equal(majority(views), predicted),
                'the arm prediction is not the seven-view majority of its artifact')
        require(len(arm['views']) == p['n_views'], 'view count')
        by_depth = arrays[f'{name}__majority_by_depth']
        require(by_depth.shape == (arm['hidden_layers'], len(test)) and np.array_equal(by_depth[-1], predicted),
                'the last depth majority is not the arm prediction')
        for depth, entry in enumerate(arm['majority_by_depth']):
            require(entry['depth'] == depth and entry['knn_accuracy'] == td.accuracy(by_depth[depth], truth),
                    'majority depth accuracy')
        require(np.isclose(arm['neighborhood_purity'],
                           float(np.mean([view['depths'][-1]['neighborhood_purity'] for view in arm['views']])),
                           rtol=0, atol=1e-12), 'neighbourhood purity')
        for view in arm['views']:
            require(len(view['depths']) == arm['hidden_layers'] == len(arm['widths']), 'depth schedule')
            movement = view['movement']
            require(movement is not None and movement['rule'] == specs[name]['rule']
                    and movement['frozen_hidden_layers'] == list(specs[name]['frozen_hidden'])
                    and movement['instrumentation_removed'] and movement['rng_unchanged'], 'movement record')
            require(len(movement['layers']) == arm['hidden_layers'], 'movement layer schedule')
            for layer in movement['layers']:
                series = arrays[f'{name}__v{view["view"]}__l{layer["layer"]}__changed']
                require(len(series) == layer['batches'], 'movement series length')
                require(layer['updates'] == 0 or np.isclose(float(np.mean(series)), layer['changed_share'], rtol=0, atol=1e-9)
                        or layer['batches'] != len(series), 'movement series disagrees with its summary')
                if layer['layer'] in specs[name]['frozen_hidden']:
                    require(not layer['changed_share'] and not layer['mean_displacement'], 'a frozen layer moved')
    return True


def collect(output, *, allow_smoke=False, rederive=False):
    """Every planned job record verified; with rederive, every sealed selection and planned job re-derived from its
    reference. Failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs, selections = verify(output, allow_smoke=allow_smoke)
    saved_environment = json.loads((output/'environment.json').read_text())
    references = None
    if rederive:
        sources = {name: (entry['run_directory'], entry['ablation_directory']) for name, entry in manifest['references'].items()}
        references = td.load_references(p, sources, allow_smoke=allow_smoke)
    prepared = {name: load_prepared(output, name) for name in manifest['datasets']}
    records, issues = {}, []
    for job in jobs:
        stem, key = job['stem'], (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
        path = output/'jobs'/f'{stem}.json'
        if not path.is_file():
            issues.append(f'missing jobs/{stem}.json')
            continue
        try:
            record = json.loads(path.read_text())
            X, y, data, splits = prepared[job['dataset_id']]
            index = next(i for i, s in enumerate(splits) if (s['outer_repeat'], s['outer_fold']) == key[1:])
            split, sealed = splits[index], selections[key]
            if references is not None:
                reference = references[job['reference']]
                if load_prepared(reference['run']['directory'], job['dataset_id'])[2] != data:
                    raise ValueError('the prepared dataset differs from its reference run')
                rederived = base.selection_record(reference['run'], job['dataset_id'], split, y, data)
                if rederived != sealed or rederived != reference['ablation']['selections'].get(key):
                    raise ValueError('the sealed selection differs from the one re-derived from its reference')
                if planned_job(p, reference, job['dataset_id'], index, split, rederived, X.shape[1]) != job:
                    raise ValueError('the planned job differs from the re-derived plan')
            validate_record(record, job, p, data, split, y, saved_environment['code_revision'], output, sealed)
            records[stem] = record
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    if issues:
        raise ValueError(f'Incomplete {p["production_family"]} intervention evidence: ' + '; '.join(issues))
    return {'protocol': p, 'manifest': manifest, 'jobs': jobs, 'records': records,
            'code_revision': saved_environment['code_revision'], 'environment': saved_environment}


def model_rows(collected):
    """{dataset: outer model rows}, one per arm and fold, in the schema evaluation's summarizers expect."""
    rows = defaultdict(list)
    for job in collected['jobs']:
        record = collected['records'][job['stem']]
        for arm in record['arms']:
            rows[job['dataset_id']].append({
                'dataset_id': job['dataset_id'], 'model_id': arm['arm_id'], 'arm_id': arm['arm_id'],
                'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'],
                'model_seed': record['identity']['model_seed'], 'status': 'ok',
                'own_selection': arm['arm_id'] == job['own_arm'], 'widths': list(arm['widths']),
                'neighborhood_purity': arm['neighborhood_purity'], 'fit_seconds': arm['fit_seconds'],
                **{metric: arm[metric] for metric in METRICS}})
    return dict(rows)


def job_rows(record, job, p):
    """The rows one verified job contributes to each table."""
    identity = record['identity']
    key = [p['production_family'], identity['dataset_id'], identity['reference'], identity['outer_repeat'],
           identity['outer_fold'], identity['model_seed']]
    selected = identity['selected']
    rows = defaultdict(list)
    for arm in record['arms']:
        row = key + [arm['arm_id']]
        rows['arms.csv'].append(row + [int(arm['arm_id'] == job['own_arm']), canonical_json(arm['widths']),
                                       selected['learning_rate'], selected['embed_dim'], selected['degree'],
                                       int(bool(selected['augment']))]
                                + [arm[metric] for metric in METRICS]
                                + [arm['neighborhood_purity'], arm['fit_seconds']])
        majority_by_depth = {entry['depth']: entry['knn_accuracy'] for entry in arm['majority_by_depth']}
        for view in arm['views']:
            for depth in view['depths']:
                rows['depth_probe.csv'].append(row + [view['view'], depth['depth'], depth['hidden_layers'],
                                                      depth['knn_accuracy'], depth['neighborhood_purity'],
                                                      majority_by_depth[depth['depth']]])
            for layer in (view['movement'] or {}).get('layers', []):
                rows['movement.csv'].append(row + [view['view'], layer['layer'], layer['layer_name'], layer['n_filters'],
                                                   layer['vocabulary'], layer['batches'], layer['updates'],
                                                   layer['changed_share'], layer['mean_displacement'],
                                                   layer['mean_votes'], layer['mean_vote_mass'], layer['max_vote_mass'],
                                                   layer['incoming_to_prior_ratio'], layer['median_solves'],
                                                   layer['tie_canonicalisation_incomplete']])
    return rows


def render_tables(collected):
    """{file name: CSV text} over every verified job in planned order."""
    tables = {name: [] for name in TABLES}
    for job in collected['jobs']:
        for name, rows in job_rows(collected['records'][job['stem']], job, collected['protocol']).items():
            tables[name].extend(rows)
    contents = {}
    for name, header in TABLES.items():
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(header)
        writer.writerows([['' if value is None else value for value in row] for row in tables[name]])
        contents[name] = buffer.getvalue()
    return contents


def check_totals(records):
    totals = {name: {'performed': 0, 'passed': 0} for name in JOB_CHECKS}
    for record in records.values():
        for name in JOB_CHECKS:
            totals[name]['performed'] += int(record['checks'][name]['performed'])
            totals[name]['passed'] += int(record['checks'][name]['passed'] is True)
    return totals


def provenance_record(output, collected, contents, run_record):
    p, manifest, records = collected['protocol'], collected['manifest'], collected['records']
    return _plain({
        'purpose': p['purpose'], 'selection_statement': p['selection_statement'], 'protocol_id': p['protocol_id'],
        'production_family': p['production_family'], 'protocol_hash': manifest['protocol_hash'],
        'protocol_sha256': sha256_file(Path(output)/'protocol.json'), 'frozen': p['frozen'],
        'frozen_at_utc': p.get('frozen_at_utc'), 'code_revision': collected['code_revision'],
        'environment': collected['environment'], 'references': manifest['references'],
        'dataset_identity': manifest['dataset_identity'], 'datasets': manifest['datasets'], 'arms': list(p['arms']),
        'planned_jobs': len(collected['jobs']), 'check_totals': check_totals(records),
        'jobs': {job['stem']: {'identity': records[job['stem']]['identity'], 'checks': records[job['stem']]['checks'],
                               'timing': records[job['stem']]['timing'],
                               'record_sha256': sha256_file(Path(output)/'jobs'/f'{job["stem"]}.json'),
                               'artifact_sha256': records[job['stem']]['artifact']['sha256'],
                               'prediction_sha256': records[job['stem']]['prediction_file']['sha256']}
                 for job in collected['jobs']},
        'tables': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in contents.items()},
        'run': run_record})


def write_tables(output, *, allow_smoke=False, run_record=None):
    """The three tables and provenance.json of a completed run, all or none."""
    collected = collect(output, allow_smoke=allow_smoke)
    contents = render_tables(collected)
    provenance = provenance_record(output, collected, contents, run_record)
    td.write_all(output, {**contents, PROVENANCE_FILE: json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'})
    return provenance


# ----------------------------------------------------------------------------- summary

def _mean_or_none(values):
    return float(np.mean(values)) if values else None


def summarize(collected):
    """Per dataset and arm, the outer-fold summaries of every metric and of the purity, the depth probe, the movement
    rates and the inertness flags. Descriptive; the prespecified contrasts belong to the analyse stage."""
    p = collected['protocol']
    rows, folds, seeds = model_rows(collected), fold_schedule(p), [int(p['model_seed'])]
    by_stem = collected['records']
    summaries = {}
    for name in collected['manifest']['datasets']:
        jobs = [job for job in collected['jobs'] if job['dataset_id'] == name]
        records = [by_stem[job['stem']] for job in jobs]
        table = {}
        for arm in p['arms']:
            entry = {'metrics': {metric: summarize_outer(rows[name], arm, metric, expected_folds=folds, expected_seeds=seeds)
                                 for metric in METRICS}}
            values = [r['neighborhood_purity'] for r in rows[name] if r['model_id'] == arm]
            entry['neighborhood_purity'] = {'mean': float(np.mean(values)),
                                            'outer_fold_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                            'n_folds': len(values)}
            arms = [next(a for a in record['arms'] if a['arm_id'] == arm) for record in records]
            depths = sorted({entry_d['depth'] for a in arms for entry_d in a['majority_by_depth']})
            entry['depth_probe'] = [
                {'depth': depth,
                 'folds': sum(1 for a in arms if any(e['depth'] == depth for e in a['majority_by_depth'])),
                 'majority_knn_accuracy': float(np.mean([e['knn_accuracy'] for a in arms for e in a['majority_by_depth']
                                                         if e['depth'] == depth])),
                 'view_knn_accuracy': float(np.mean([v['depths'][depth]['knn_accuracy'] for a in arms for v in a['views']
                                                     if depth < len(v['depths'])])),
                 'neighborhood_purity': float(np.mean([v['depths'][depth]['neighborhood_purity'] for a in arms
                                                       for v in a['views'] if depth < len(v['depths'])]))}
                for depth in depths]
            movements = [a['movement'] for a in arms if a.get('movement')]
            if movements:
                shares = [m['changed_share'] for m in movements if m['changed_share'] is not None]
                updatable = [m['updatable_changed_share'] for m in movements if m['updatable_changed_share'] is not None]
                entry['movement'] = {
                    'folds': len(movements), 'changed_share': float(np.mean(shares)) if shares else None,
                    'updatable_changed_share': _mean_or_none(updatable),
                    'mean_displacement': float(np.mean([m['mean_displacement'] for m in movements
                                                        if m['mean_displacement'] is not None])) if shares else None,
                    'effectively_inert_folds': int(sum(1 for m in movements if m['effectively_inert'])),
                    'inert_threshold': INERT_THRESHOLD,
                    'incoming_to_prior_ratio': float(np.mean([layer['incoming_to_prior_ratio'] for m in movements
                                                              for layer in m['layers']
                                                              if layer['incoming_to_prior_ratio'] is not None]))
                    if any(layer['incoming_to_prior_ratio'] is not None for m in movements for layer in m['layers']) else None,
                    'by_layer': [{'layer': layer, 'changed_share': _mean_or_none(
                        [l['changed_share'] for m in movements for l in m['layers']
                         if l['layer'] == layer and l['changed_share'] is not None]),
                                  'mean_displacement': _mean_or_none(
                        [l['mean_displacement'] for m in movements for l in m['layers']
                         if l['layer'] == layer and l['mean_displacement'] is not None])}
                        for layer in sorted({l['layer'] for m in movements for l in m['layers']})]}
            table[arm] = entry
        selected = defaultdict(int)
        for job in jobs:
            selected[canonical_json(job['selected_widths'])] += 1
        summaries[name] = {'reference': jobs[0]['reference'], 'folds': len(jobs), 'arms': table,
                           'selected_widths': dict(sorted(selected.items())),
                           'own_arm': dict(sorted({job['own_arm']: sum(1 for j in jobs if j['own_arm'] == job['own_arm'])
                                                   for job in jobs}.items()))}
    return _plain({'purpose': p['purpose'], 'selection_statement': p['selection_statement'],
                   'protocol_id': p['protocol_id'], 'production_family': p['production_family'],
                   'protocol_hash': collected['manifest']['protocol_hash'], 'code_revision': collected['code_revision'],
                   'datasets': collected['manifest']['datasets'], 'arms': list(p['arms']),
                   'planned_jobs': len(collected['jobs']), 'check_totals': check_totals(collected['records']),
                   'aggregation': 'one fitting seed per outer fold; outer-fold mean and SD over the fifteen folds',
                   'inferential_significance_claims': False, 'summaries': summaries})


def summary(output, *, allow_smoke=False):
    """Re-verify the run, require the tables and provenance to equal their re-rendering, then write the summary."""
    output = Path(output)
    collected = collect(output, allow_smoke=allow_smoke, rederive=True)
    family = collected['protocol']['production_family']
    contents = render_tables(collected)
    for name, text in contents.items():
        if not (output/name).is_file() or (output/name).read_text() != text:
            raise ValueError(f'{name} is missing or differs from the verified job records')
    saved = json.loads((output/PROVENANCE_FILE).read_text())
    if saved != provenance_record(output, collected, contents, saved.get('run')):
        raise ValueError(f'{PROVENANCE_FILE} differs from the verified run')
    report = summarize(collected)
    write_json(output/f'{family}_summary.json', report)
    flat = []
    for name, entry in report['summaries'].items():
        for arm in report['arms']:
            metrics = entry['arms'][arm]['metrics']
            flat.extend([name, arm, metric, metrics[metric]['mean'], metrics[metric]['outer_fold_sd'],
                         metrics[metric]['n_folds']] for metric in METRICS)
            purity = entry['arms'][arm]['neighborhood_purity']
            flat.append([name, arm, 'neighborhood_purity', purity['mean'], purity['outer_fold_sd'], purity['n_folds']])
    write_csv(output/f'{family}_summary.csv', ('dataset_id', 'arm_id', 'metric', 'mean', 'outer_fold_sd', 'n_folds'), flat)
    return report


# ----------------------------------------------------------------------------- pilot, mass matching and freeze

def calibrated_projection(jobs, records, workers=WORKERS):
    """Per planned job, the reference run's realized outer fit and predict seconds (seed 8129, that fold) times the
    piloted job / uninstrumented-fit ratio (its own for a pilot dataset, the largest piloted otherwise), plus the
    realized seconds again for a first fold's uninstrumented fit; serial hours and the first-free-worker makespan."""
    ratios = {r['dataset_id']: r['job_to_fit_ratio'] for r in records}
    largest = max(ratios.values())
    seconds, datasets = [], {}
    for job in jobs:
        realized, ratio = float(job['reference_outer_seconds']), ratios.get(job['dataset_id'], largest)
        cost = realized * ratio + (realized if job['check_uninstrumented'] else 0.)
        seconds.append(cost)
        entry = datasets.setdefault(job['dataset_id'], {'jobs': 0, 'reference_seconds': 0., 'seconds': 0., 'ratio': ratio,
                                                        'basis': 'piloted ratio' if job['dataset_id'] in ratios else 'largest piloted ratio'})
        entry['jobs'] += 1
        entry['reference_seconds'] += realized
        entry['seconds'] += cost
    for entry in datasets.values():
        entry['serial_hours'] = entry['seconds'] / 3600
    serial = sum(seconds)
    return {'datasets': datasets, 'serial_hours': serial / 3600, 'serial_hours_over_workers': serial / 3600 / workers,
            'simulated_makespan_hours': makespan(seconds, workers) / 3600, 'longest_job_hours': max(seconds) / 3600,
            'workers': workers}


def mass_ladder(p, job, X, y, split, query, *, borda_share, piloted):
    """The mass-matching ladder on one training-only partition: the median arm at each prior multiplier, its mean
    per-batch changed-filter share over the hidden layers, and the distance to the borda arm's share."""
    specs = arm_specs(p['production_family'], p)
    seed = int(p['model_seed'])
    train = [int(i) for i in split['train']]
    X_train, y_train = X[train], y[train]
    X_query, y_query = X[query], y[query]
    rungs = []
    for multiplier in MASS_LADDER:
        if multiplier in piloted:
            share, seconds = piloted[multiplier], 0.
        else:
            spec = {**specs['median_mass_matched'], 'prior_multiplier': multiplier}
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                record, _, _, _, _ = evaluate_arm('median_mass_matched', spec,
                                                  arm_params(spec, job['selected'], p['arm_widths']), seed,
                                                  X_train, y_train, X_query, y_query)
            share, seconds = record['movement']['changed_share'], time.perf_counter() - start
        rungs.append({'prior_multiplier': multiplier, 'changed_share': share, 'seconds': seconds,
                      'distance_to_borda': abs(float(share) - float(borda_share))})
    return rungs


def choose_multiplier(ladders):
    """The ladder value whose mean distance to the borda arm's changed share over the pilot datasets is smallest; ties
    to the smallest multiplier."""
    distances = {}
    for multiplier in MASS_LADDER:
        values = [rung['distance_to_borda'] for ladder in ladders for rung in ladder if rung['prior_multiplier'] == multiplier]
        distances[multiplier] = float(np.mean(values)) if values else float('inf')
    chosen = min(MASS_LADDER, key=lambda m: (distances[m], m))
    return {'prior_multiplier': int(chosen), 'mean_distance_to_borda': distances[chosen],
            'distances': {str(m): distances[m] for m in MASS_LADDER},
            'rule': 'the ladder value whose mean per-batch changed-filter share over the hidden layers on the '
                    'training-only pilot fits is closest to the borda arm\'s on the same partitions; ties to the '
                    'smallest multiplier; no outer-fold score takes part'}


def runtime_pilot(output, p, sources):
    """Training-only: prepare every protocol dataset, then the first outer fold of each pilot dataset with every fourth
    outer training row as the query rows (the outer test fold is never touched), the projections, one reproduction probe
    per reference and, in the aggregation family, the mass-matching ladder."""
    output = Path(output)
    started = utc_now()
    jobs, references = prepare(output, p, sources, purpose='training_runtime_only')
    mapping, revision = td.reference_of(p), json.loads((output/'environment.json').read_text())['code_revision']
    sealed_records = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                      for s in json.loads((output/'reference_selections.json').read_text())}
    family, records, ladders = p['production_family'], [], []
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and j['check_uninstrumented'])
        X, y, data, splits = load_prepared(output, name)
        split = splits[0]
        query = [int(i) for i in split['train'][::4]]
        record, _ = evaluate_job(X, y, split, job, sealed_records[(name, split['outer_repeat'], split['outer_fold'])], p,
                                 dataset_hash=data['dataset_hash'], code_revision=revision, protocol_hash=config_id(p),
                                 query=query)
        if record['status'] != 'ok':
            raise CheckFailed(f'pilot job {name} failed: {record.get("exception")}')
        timing = record['timing']
        movement = {arm['arm_id']: arm['movement'] for arm in record['arms']}
        entry = {'dataset_id': name, 'reference': mapping[name], 'dataset_hash': data['dataset_hash'],
                 'train_ids': split['train'], 'query_ids': query, 'config_id': job['config_id'],
                 'selected': job['selected'], 'own_arm': job['own_arm'], 'model_seed': job['model_seed'],
                 'timing': timing, 'reference_outer_seconds': job['reference_outer_seconds'],
                 'job_to_fit_ratio': (timing['job_seconds'] - timing['uninstrumented_fit_seconds']) / timing['uninstrumented_fit_seconds'],
                 'checks': {key: {'performed': c['performed'], 'passed': c['passed']} for key, c in record['checks'].items()},
                 'movement': {arm: {key: value[key] for key in ('changed_share', 'mean_displacement', 'hidden_updates',
                                                                'effectively_inert')}
                              for arm, value in movement.items()},
                 'accuracy_on_training_query_rows': {arm['arm_id']: arm['accuracy'] for arm in record['arms']},
                 'status': 'ok'}
        if family == 'aggregation':
            piloted = {arm_specs(family, p)['median_mass_matched']['prior_multiplier']: movement['median_mass_matched']['changed_share']}
            entry['mass_ladder'] = mass_ladder(p, job, X, y, split, query,
                                               borda_share=movement['borda']['changed_share'], piloted=piloted)
            ladders.append(entry['mass_ladder'])
        records.append(entry)
    piloted = {r['dataset_id']: r for r in records}
    slowest_first = max(r['timing']['job_seconds'] for r in records)
    slowest_other = max(r['timing']['job_seconds'] - r['timing']['uninstrumented_fit_seconds'] for r in records)
    harness_seconds = sum((piloted[j['dataset_id']]['timing']['job_seconds'] - (0. if j['check_uninstrumented'] else
                           piloted[j['dataset_id']]['timing']['uninstrumented_fit_seconds'])) if j['dataset_id'] in piloted
                          else (slowest_first if j['check_uninstrumented'] else slowest_other) for j in jobs)
    probes = {name: base.reproduction_probe(reference['run'], td.smallest_dataset(reference))
              for name, reference in references.items()}
    calibrated = calibrated_projection(jobs, records, p['workers'])
    decision = calibrated['simulated_makespan_hours']
    checks_passed = all(c['passed'] is True for r in records for key, c in r['checks'].items() if key != 'reference_predictions')
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'protocol_id': p['protocol_id'],
              'production_family': family, 'protocol_hash': config_id(p), 'code_revision': revision,
              'started_utc': started, 'ended_utc': utc_now(), 'records': records,
              'harness_projection': {'serial_hours': harness_seconds / 3600,
                                     'hours_at_workers_ideal': harness_seconds / 3600 / p['workers'],
                                     'basis': 'pilot seconds per job; unpiloted datasets at the slowest piloted job (not a bound)'},
              'calibrated_projection': calibrated, 'reproduction_probes': probes,
              'mass_matching': choose_multiplier(ladders) if ladders else None,
              'decision': {'rule': p['decision_rule'], 'hours': decision, 'cap_hours': p['wallclock_cap_hours'],
                           'workers': p['workers'], 'within_cap': decision <= p['wallclock_cap_hours'],
                           'checks_passed': checks_passed,
                           'probes_reproduced': all(probe['reproduced'] for probe in probes.values())},
              'estimate_limitations': 'one training partition per pilot dataset on the machine as it was, with two '
                                      'production runs holding the other workers, so every pilot second is inflated by '
                                      'contention; the calibrated projection divides one contended measurement by '
                                      'another and multiplies the reference run\'s realized per-fold seconds (measured '
                                      'under 16 workers), so the inflation largely cancels; the query rows are training '
                                      'rows, so the readout and probe costs follow the outer test size only approximately'}
    write_json(output/'pilot.json', _plain(report))
    return report


def freeze(draft_path, pilot_path, stages_path, output_path, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the training-only pilot of that draft projects within the
    cap at the protocol workers, every pilot check and reproduction probe held, and the stage record carries a passing
    smoke; an existing output must be the draft itself, which the frozen protocol then replaces."""
    draft = json.loads(Path(draft_path).read_text())
    if draft.get('frozen'):
        raise ValueError('The draft is already frozen')
    validate_protocol(draft)
    family, cap = draft['production_family'], CAP_HOURS[draft['production_family']]
    pilot, stages = json.loads(Path(pilot_path).read_text()), json.loads(Path(stages_path).read_text())
    if pilot.get('protocol_hash') != config_id(draft) or pilot.get('production_family') != family:
        raise ValueError('The pilot did not run with this draft')
    decision = pilot['decision']
    if not (decision['within_cap'] and 0 < decision['hours'] <= cap and decision['workers'] == WORKERS
            and decision['probes_reproduced'] and decision['checks_passed']):
        raise ValueError(f'Not frozen: projection {decision["hours"]:.2f} h at {decision["workers"]} workers against the '
                         f'{cap} h cap, probes reproduced {decision["probes_reproduced"]}, pilot checks passed '
                         f'{decision["checks_passed"]}')
    if (stages.get('smoke') or {}).get('status') != 'ok' or not stages.get('summary'):
        raise ValueError('The stage record lacks a passing synthetic smoke and its summary')
    matching = pilot.get('mass_matching')
    if family == 'aggregation' and (not matching or matching['prior_multiplier'] not in MASS_LADDER):
        raise ValueError('The aggregation freeze needs the mass-matching ladder of its pilot')
    calibrated, harness = pilot['calibrated_projection'], pilot['harness_projection']
    record = {'cap_hours': cap, 'workers': WORKERS, 'decision_hours': decision['hours'], 'decision_rule': decision['rule'],
              'calibrated': {key: calibrated[key] for key in ('serial_hours', 'serial_hours_over_workers',
                                                              'simulated_makespan_hours', 'longest_job_hours')},
              'calibrated_per_dataset_hours': {name: entry['serial_hours'] for name, entry in calibrated['datasets'].items()},
              'harness': {key: harness[key] for key in ('serial_hours', 'hours_at_workers_ideal')},
              'pilot_ratios': {r['dataset_id']: r['job_to_fit_ratio'] for r in pilot['records']},
              'pilot_seconds': {r['dataset_id']: r['timing'] for r in pilot['records']},
              'pilot_movement': {r['dataset_id']: r['movement'] for r in pilot['records']},
              'reproduction_probes': {name: {key: probe[key] for key in ('dataset_id', 'result_file', 'config_id',
                                                                         'model_seed', 'inner_fold', 'reference_score',
                                                                         'refit_score', 'readout_selections_identical',
                                                                         'reproduced')}
                                      for name, probe in pilot['reproduction_probes'].items()},
              'pilot_code_revision': pilot['code_revision'], 'pilot_sha256': sha256_file(pilot_path), 'stages': stages}
    chosen = None
    if family == 'aggregation':
        chosen = {**matching, 'ladders': {r['dataset_id']: r['mass_ladder'] for r in pilot['records']},
                  'borda_changed_share': {r['dataset_id']: r['movement']['borda']['changed_share'] for r in pilot['records']},
                  'pilot_sha256': record['pilot_sha256']}
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    ratios = ', '.join(f"{name} {value:.2f}" for name, value in sorted(record['pilot_ratios'].items()))
    text = (f"Review response E4 ({family}; author decision of 2026-09-14 and the E4 aggregation ruling): "
            f"{stages['summary']}; projected at {WORKERS} single-thread workers: calibrated simulated makespan "
            f"{decision['hours']:.2f} h (serial {calibrated['serial_hours']:.2f} h, serial over workers "
            f"{calibrated['serial_hours_over_workers']:.2f} h), harness {harness['hours_at_workers_ideal']:.2f} h; cap "
            f"{cap} h. The pilot ran while two production runs held 32 workers, so its absolute seconds are inflated by "
            f"contention and only the job / uninstrumented-fit ratio carries forward ({ratios}); the rule prices every "
            f"unpiloted dataset at the largest of those, measured on the smallest training partition, where the per-arm "
            f"bookkeeping weighs most. The cap is set above the projection for both reasons and no outer score of this "
            f"family exists. Reproduction probes held on "
            f"{', '.join(probe['dataset_id'] for probe in pilot['reproduction_probes'].values())}"
            + (f"; mass-matching multiplier {chosen['prior_multiplier']} chosen on the training-only pilot ladder"
               if chosen else '') + '; frozen after the pilot')
    protocol = validate_protocol(_plain(dict(draft, frozen=True, frozen_at_utc=frozen_at, status=FROZEN_STATUS,
                                             resource_decision=text, pilot_projection=record, mass_matching_choice=chosen)))
    output_path = Path(output_path)
    content = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if output_path.exists() and json.loads(output_path.read_text()) not in (draft, protocol):
        raise FileExistsError(f'Refusing to replace {output_path}: it is neither the draft nor this frozen protocol')
    temporary = output_path.with_name(output_path.name + '.freezing')
    temporary.write_text(content)
    os.replace(temporary, output_path)
    return protocol


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

SMOKE_CANDIDATES = [{'aggregation': 'majority', 'batch_size': 16, 'degree_offset': 0, 'embed_scale': 1, 'iterations': 4,
                     'learning_rate': .2, 'n_views': 7, 'strategy': 'diverse', 'validation_ratio': .1, 'widths': widths}
                    for widths in ([6], [4, 6])]
SMOKE_WIDTHS = {'depth1': [6], 'depth2': [4, 6]}


def smoke_protocol(p, sources):
    """The synthetic smoke form of a protocol whose reference is the given synthetic run and its ablation."""
    references = {}
    for name, (run_dir, ablation_dir) in sources.items():
        run_protocol = json.loads((Path(run_dir)/'protocol.json').read_text())
        references[name] = {'datasets': list(run_protocol['datasets']), 'loader': None,
                            'run': {**td.reference_pins(run_dir), 'family': run_protocol['production_family']},
                            'ablation': td.ablation_pins(ablation_dir)}
    design_source = json.loads((Path(next(iter(sources.values()))[0])/'protocol.json').read_text())
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    return _plain(dict(p, references=references, datasets=datasets, pilot_datasets=datasets[:1], frozen=False,
                       status='synthetic_smoke_only', purpose='synthetic smoke only; never evidence',
                       arm_widths=dict(SMOKE_WIDTHS), **{key: design_source[key] for key in base.DESIGN_KEYS}))


def smoke(output, p, workers=3):
    """A synthetic ArrowFlow-kNN reference run at both smoke depths with its component ablation, then a complete
    intervention run over it, its tables and its summary. Never evidence."""
    from .knn_controls import synthetic_reference_run
    output = Path(output)
    with execution_lock():                               # the harness lock is per source tree and never held nested
        reference = synthetic_reference_run(output/'synthetic_reference', SMOKE_CANDIDATES, workers=workers, samples=240)
    template, reference_protocol = json.loads(base.PROTOCOL.read_text()), json.loads((reference/'protocol.json').read_text())
    tiny = dict(template, datasets=['synthetic'], pilot_datasets=['synthetic'], frozen=False,
                purpose='synthetic_smoke_only',
                **{key: reference_protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds')},
                reference_source={**template['reference_source'], **td.reference_pins(reference),
                                  'family': reference_protocol['production_family']},
                depth_split={**template['depth_split'], 'depths': [c['widths'] for c in SMOKE_CANDIDATES]})
    ablation = output/'synthetic_ablation'
    base.prepare(ablation, tiny, reference, allow_smoke=True, purpose='synthetic_smoke_only')
    base.run(ablation, workers, allow_smoke=True)
    base.write_summary(ablation, allow_smoke=True)
    sources = {'smoke_reference': (reference, ablation)}
    run_directory = output/'run'
    prepare(run_directory, smoke_protocol(p, sources), sources, allow_smoke=True, purpose='synthetic_smoke_only')
    run(run_directory, workers, allow_smoke=True)
    write_tables(run_directory, allow_smoke=True, run_record={'purpose': 'synthetic_smoke_only'})
    report = summary(run_directory, allow_smoke=True)
    return report


# ----------------------------------------------------------------------------- the prespecified analysis

ANALYSIS_SOURCES = ('interventions.py', 'update_rules.py', 'training_diagnostics.py', 'run_knn_ablation.py',
                    'evaluation.py', 'knn_controls.py')
RUN_FILES = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json',
             PROVENANCE_FILE)


class AnalysisRefused(RuntimeError):
    """The analysis refuses to read a score."""


def completeness_gate(run_directory):
    """Refuse before any record holding a score is read."""
    path = Path(run_directory)
    if not path.is_dir():
        raise AnalysisRefused(f'{path}: no such run directory')
    family = None
    try:
        family = json.loads((path/'protocol.json').read_text()).get('production_family')
    except (OSError, ValueError):
        pass
    if family not in FAMILIES:
        raise AnalysisRefused(f'{path} does not hold an intervention protocol')
    missing = [name for name in RUN_FILES + tuple(TABLES) + (f'{family}_summary.json', f'{family}_summary.csv')
               if not (path/name).is_file()]
    if missing:
        raise AnalysisRefused(f'{path} is incomplete: the run and its summary must be finished first (missing '
                              f'{", ".join(missing)})')
    jobs = json.loads((path/'planned_jobs.json').read_text())
    absent = [job['stem'] for job in jobs
              if not all((path/f'{kind}/{job["stem"]}.{suffix}').is_file()
                         for kind, suffix in (('jobs', 'json'), ('artifacts', 'npz'), ('predictions', 'jsonl')))]
    if absent:
        raise AnalysisRefused(f'{path} is incomplete: {len(absent)} of {len(jobs)} planned jobs are missing records '
                              f'(first: {absent[0]})')
    return family


def contrast_rows(rows, p, datasets, widths_by_fold):
    """The prespecified family: per variant arm and dataset, the reference arm minus the variant accuracy, Holm-adjusted
    across the datasets within the arm; then the same differences on the named depth subsets (descriptive)."""
    reference, folds, seeds = p['analysis']['reference_arm'], fold_schedule(p), [int(p['model_seed'])]
    q, confidence = p['test_train_ratio'], p['confidence']
    depths = list(p['arm_widths'].values())        # the protocol's own arm widths, which a synthetic smoke overrides
    contrasts, subsets = [], []
    for arm in p['analysis']['variant_arms']:
        entries = []
        for name in datasets:
            interval = paired_corrected_interval(rows[name], reference, arm, metric='accuracy', q=q,
                                                 confidence=confidence, expected_folds=folds,
                                                 expected_seeds={reference: seeds, arm: seeds})
            entries.append({'family': p['production_family'], 'contrast': f'{reference}_minus_{arm}', 'subset': 'all_folds',
                            'dataset_id': name, 'arm_a': reference, 'arm_b': arm, 'metric': 'accuracy',
                            'sd': None,
                            **{key: interval[key] for key in ('mean_difference', 'standard_error', 'ci_low', 'ci_high',
                                                              'n_folds', 'df', 'p_approximate')}})
        adjusted = holm_adjust([entry['p_approximate'] for entry in entries])
        for entry, value in zip(entries, adjusted):
            entry['holm_p_approximate'] = value
            entry['significant_after_holm'] = bool(value < p['analysis']['alpha'])
        contrasts.extend(entries)
        if p['production_family'] != 'depth':
            continue
        for name in datasets:
            split_by_depth = depth_split(rows[name], reference, arm, widths_by_fold[name], depths=depths, folds=folds,
                                         seeds={reference: seeds, arm: seeds}, q=q, confidence=confidence)
            for entry in split_by_depth:
                # A two- or one-fold subset carries no interval: the fold count is on the row and df = n - 1.
                interval = (entry.get('interval') or {}) if entry['n_folds'] >= MIN_SUBSET_FOLDS else {}
                subsets.append({'family': p['production_family'], 'contrast': f'{reference}_minus_{arm}',
                                'subset': 'own_selection_' + canonical_json(entry['widths']), 'dataset_id': name,
                                'arm_a': reference, 'arm_b': arm, 'metric': 'accuracy',
                                'mean_difference': entry['mean_difference'], 'standard_error': interval.get('standard_error'),
                                'ci_low': interval.get('ci_low'), 'ci_high': interval.get('ci_high'),
                                'n_folds': entry['n_folds'], 'df': max(entry['n_folds'] - 1, 0),
                                'sd': entry['sd'], 'p_approximate': None, 'holm_p_approximate': None,
                                'significant_after_holm': None})
    return contrasts, subsets


def analyse(run_directory, output, *, allow_smoke=False):
    """Refuse until the run is complete, re-verify every record, then compute the prespecified contrasts."""
    run_directory, output = Path(run_directory), Path(output)
    family = completeness_gate(run_directory)
    names = (f'{family}_contrasts.csv', f'{family}_analysis.json')
    from .compare_runs import check_output, write_outputs
    check_output(output, names)
    collected = collect(run_directory, allow_smoke=allow_smoke, rederive=True)
    p = collected['protocol']
    contents = render_tables(collected)
    for name, text in contents.items():
        if not (run_directory/name).is_file() or (run_directory/name).read_text() != text:
            raise AnalysisRefused(f'{name} differs from the verified job records')
    saved_provenance = json.loads((run_directory/PROVENANCE_FILE).read_text())
    if saved_provenance != provenance_record(run_directory, collected, contents, saved_provenance.get('run')):
        raise AnalysisRefused(f'{PROVENANCE_FILE} differs from the verified run')
    report = summarize(collected)
    if json.loads((run_directory/f'{family}_summary.json').read_text()) != report:
        raise AnalysisRefused(f'{family}_summary.json differs from the verified run')
    rows = model_rows(collected)
    datasets = list(collected['manifest']['datasets'])
    widths_by_fold = {name: {(job['outer_repeat'], job['outer_fold']): job['selected_widths']
                             for job in collected['jobs'] if job['dataset_id'] == name} for name in datasets}
    contrasts, subsets = contrast_rows(rows, p, datasets, widths_by_fold)
    movement = {arm: {name: report['summaries'][name]['arms'][arm].get('movement') for name in datasets}
                for arm in p['arms']}
    inert = {arm: {'folds': sum((entry or {}).get('folds', 0) for entry in movement[arm].values()),
                   'effectively_inert_folds': sum((entry or {}).get('effectively_inert_folds', 0) for entry in movement[arm].values()),
                   'mean_changed_share': float(np.mean([entry['changed_share'] for entry in movement[arm].values()
                                                        if entry and entry['changed_share'] is not None]))
                   if any(entry and entry['changed_share'] is not None for entry in movement[arm].values()) else None}
             for arm in p['arms']}
    record = _plain({
        'purpose': f'prespecified_analysis_of_the_{family}_intervention_family',
        'status': 'computed after the run was complete and re-verified (every job record, every sealed selection '
                  're-derived from its reference run, every table re-rendered); one contrast family per variant arm, '
                  'Holm-adjusted within the arm across the seventeen datasets; everything else descriptive',
        'protocol_id': p['protocol_id'], 'production_family': family, 'protocol_hash': collected['manifest']['protocol_hash'],
        'code_revision': collected['code_revision'], 'analysis_declaration': p['analysis'],
        'datasets': datasets, 'arms': list(p['arms']), 'reference_arm': p['analysis']['reference_arm'],
        'contrasts': contrasts, 'named_subsets': subsets, 'movement': movement, 'inertness': inert,
        'interpretation': p['analysis']['interpretation'], 'limits': p['analysis']['limits'],
        'summaries': report['summaries'], 'check_totals': report['check_totals'],
        'provenance': {'run_directory': str(run_directory.resolve()), 'run': saved_provenance['run'],
                       'references': collected['manifest']['references'], 'jobs_verified': len(collected['jobs']),
                       'tables': saved_provenance['tables'], **td_code_record()}})
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(CONTRAST_COLUMNS)
    writer.writerows([['' if row[column] is None else row[column] for column in CONTRAST_COLUMNS]
                      for row in contrasts + subsets])
    write_outputs(output, {names[0]: buffer.getvalue(),
                           names[1]: json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n'})
    return record


def td_code_record():
    from .referee_analyses import code_record
    return code_record(ANALYSIS_SOURCES)


# ----------------------------------------------------------------------------- command

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['draft', 'prepare', 'smoke', 'pilot', 'freeze', 'run', 'summary', 'analyse'])
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--reference', type=td.parse_reference, action='append', metavar='NAME=RUN_DIR,ABLATION_DIR')
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--stages', type=Path)
    parser.add_argument('--run', dest='run_directory', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        if args.family is None:
            parser.error('draft needs --family')
        write_json(args.output, draft_protocol(args.family))
        return
    if args.command == 'analyse':
        if args.run_directory is None:
            parser.error('analyse needs --run')
        try:
            record = analyse(args.run_directory, args.output)
        except (AnalysisRefused, FileExistsError, ValueError) as exc:
            parser.exit(2, f'interventions analyse refused: {exc}\n')
        for row in record['contrasts']:
            print(f"{row['contrast']} {row['dataset_id']}: {row['mean_difference']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
                  f"Holm p={row['holm_p_approximate']:.3g}")
        return
    if args.command == 'summary':
        report = summary(args.output)
        for name, entry in report['summaries'].items():
            line = '; '.join(f"{arm} {entry['arms'][arm]['metrics']['accuracy']['mean']:.4f}" for arm in report['arms'])
            print(f'{name}: {line}')
        return
    if args.command == 'freeze':
        if args.draft is None or args.pilot is None or args.stages is None:
            parser.error('freeze needs --draft, --pilot and --stages')
        protocol = freeze(args.draft, args.pilot, args.stages, args.output)
        print(f"frozen at {protocol['frozen_at_utc']}: decision {protocol['pilot_projection']['decision_hours']:.2f} h")
        return
    if args.protocol is None:
        parser.error(f'{args.command} needs --protocol')
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f'Worker count must be between 1 and {MAX_WORKERS}')
    p = validate_protocol(json.loads(args.protocol.read_text()))
    if args.command == 'smoke':
        report = smoke(args.output, p, args.workers)
        print(json.dumps(report['check_totals'], indent=2))
        return
    sources = td.parse_sources(args.reference)
    if args.command == 'prepare':
        jobs, _ = prepare(args.output, p, sources, args.dataset)
        print(f'{len(jobs)} planned jobs')
        return
    if args.command == 'pilot':
        with execution_lock():
            report = runtime_pilot(args.output, p, sources)
        print(json.dumps({'decision': report['decision'],
                          'pilot_ratios': {r['dataset_id']: r['job_to_fit_ratio'] for r in report['records']},
                          'calibrated_serial_hours': report['calibrated_projection']['serial_hours'],
                          'harness_hours_at_workers': report['harness_projection']['hours_at_workers_ideal'],
                          'mass_matching': report['mass_matching']}, indent=2))
        return
    record = run(args.output, args.workers, protocol_path=args.protocol)
    provenance = write_tables(args.output, run_record=record)
    print(json.dumps(provenance['check_totals'], indent=2))


if __name__ == '__main__':
    main()
