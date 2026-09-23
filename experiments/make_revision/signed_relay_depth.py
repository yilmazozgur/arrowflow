"""The depth study rerun with the corrected relay (review response to the simulated referee panel of 2026-09-23, V1-A).

family signed_relay_depth, four arms at each fold's reconstructed selection (the depth study arrowflow-v3-depth-1's
datasets, folds, selections, fitting seed and fixed configuration; only the hidden widths, the relay and the vote scale
vary):
    depth1                 [128]. A network with one hidden layer never relays, so it is identical under either relay:
                           reused verbatim from the depth run, and refitted WITH the corrected relay on the first outer
                           fold of every dataset, where it must reproduce the stored record exactly
    depth2_printed         [64, 128] with the printed relay (the reference arm of the secondary contrast): reused verbatim
                           from the depth run's depth2, refitted on the same folds in the printed mode, exactly as above
    depth2_signed          [64, 128] with the corrected relay (signed_relay.RelayedMotion): fitted
    depth2_signed_scaled   [64, 128] with the corrected relay and the first hidden layer's votes multiplied by the vote
                           scale s chosen on the training-only pilot (signed_relay.ScaledVote): fitted

draft    --output P                                          the unfrozen protocol (draft_protocol)
prepare  --protocol P [--reference ...] [--source-depth-run D] --output O   seal the selections, the plan and the source
smoke    --protocol P --output O [--workers 3]               a synthetic depth run and a complete run over it; never evidence
pilot    --protocol P [--reference ...] [--source-depth-run D] --output O   training-only timing, projection, scale ladder
freeze   --draft P --pilot O/pilot.json --stages S --output F   the frozen protocol, only within the cap
run      --protocol P --output O [--workers 16]              every planned job of the prepared directory, then the tables
summary  --output O                                          verify every record and the source run, then the summary
analyse  --run O --output A                                  refuses (exit 2, nothing written) until the run is complete

Every job fails when a check fails: the source depth record, artifact and predictions must match the hashes sealed from
the source run's verified provenance; the own arm (a reused arm) must carry the reference run's predictions for seed 8129;
on the first outer fold of each dataset both reused arms are refitted through this module and by the unmodified library
and must reproduce the stored record exactly; no instrumentation may survive a fit; the guarded reads must leave the
global RNG untouched; the corrected arms must have corrected every repulsion's relay and scaled every lower vote; the last
depth probe must equal the view readout. A failed job cancels the pending jobs and no table is written. Outputs are all
or none and never replace a file with different content. The prespecified analysis and its interpretation are written
into the protocol before any outer score of a corrected arm exists.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
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
from . import interventions as iv
from . import run_knn_ablation as base
from . import signed_relay as sr
from . import training_diagnostics as td
from .bridge import resolve_selected
from .comparisons import derive_seed
from .evaluation import (canonical_json, config_id, holm_adjust, metric_values, paired_corrected_interval,
                         summarize_outer, validate_split)
from .knn_controls import depth_split
from .models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from .multiview import MultiViewArrowFlowKNN, view_strategy
from .newdata import WORKSPACE_RUNS, makespan, sha256_file
from .run_bridge import fold_schedule, write_csv
from .run_revision import environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

FAMILY = 'signed_relay_depth'
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-23'
PROTOCOL = PROTOCOLS/'signed_relay_depth.json'
PROTOCOL_ID = 'arrowflow-v3-signed-relay-depth-1'
SOURCE_MODULES = iv.SOURCE_MODULES + ['experiments.make_revision.interventions', 'experiments.make_revision.signed_relay']
MODEL_SEED = iv.MODEL_SEED
N_VIEWS = iv.N_VIEWS
METRICS = iv.METRICS
ARM_WIDTHS = {'depth1': [128], 'depth2': [64, 128]}
ARMS = ('depth1', 'depth2_printed', 'depth2_signed', 'depth2_signed_scaled')
REUSED = {'depth1': 'depth1', 'depth2_printed': 'depth2'}          # arm -> its arm in the source depth run
FITTED = ('depth2_signed', 'depth2_signed_scaled')
REFERENCE_ARM = 'depth1'
PRIMARY_VARIANTS = ('depth2_signed', 'depth2_signed_scaled')
SECONDARY = ('depth2_signed', 'depth2_printed')                     # arm_a minus arm_b
SCALE_LADDER = (1, 2, 4, 8, 16, 32)
SCALE_PILOT = 8                     # 1 / c: the first layer's largest vote equals the top hidden layer's to within 1/128
CLEARING_TARGET = .5
MOST_DATASETS = 9                   # more than half of the seventeen
ALPHA = .05
CAP_HOURS = 4.
WORKERS = 16
MAX_WORKERS = 16
PILOT_DATASETS = ('iris', 'ionosphere')
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'pilot_projection', 'scale_choice')
DRAFT_STATUS = 'drafted_awaiting_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_any_outer_score_of_a_corrected_arm'
PROVENANCE_FILE = 'provenance.json'
SOURCE_FILES = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json',
                'provenance.json', 'depth_summary.json')
DEFAULT_SOURCE_DEPTH_RUN = WORKSPACE_RUNS/'2026-09-14-depth-aggregation'/'depth'/'run'
# The completed depth run arrowflow-v3-depth-1 (frozen commit 176d6cb99, run 2026-09-15), pinned by the sha256 of its
# files, computed 2026-09-23 after interventions.collect(rederive=True) re-verified all 255 records, the three tables,
# provenance.json and depth_summary.json; its protocol.json is byte-identical to the committed protocol_file.
SOURCE_DEPTH_RUN = {
    'family': 'depth', 'protocol_id': 'arrowflow-v3-depth-1',
    'protocol_file': 'experiments/make_revision/protocols/2026-09-14/depth.json',
    'code_revision': '176d6cb9918893cc7b170f0f30ad6f9448352e3c',
    'protocol_sha256': 'fa99e4b2378d8839904eec109a13dee0321a46da426c52db3535abfb3fbbd842',
    'environment_sha256': 'bc4d40adeb0dbe7bd82e468140ce833189f298dbacd5d4dc267662b25f9df8f3',
    'manifest_sha256': '3d2a40863c052ca1b8e58f2fcbaea70475dbc4c22f29f478e8a99f3360572454',
    'planned_jobs_sha256': '99e14a966901eb9fabaecc55d6df2003c9bebfc04d520f5ee881dbc372209498',
    'reference_selections_sha256': '535c203238deef743119fe8bd0068d16db5d26d5f87008fe814da0f4eb44c327',
    'provenance_sha256': '947947cf726a6108586aed7a955eaf75bed4206cebe33cdb8effa0a4199abdc8',
    'depth_summary_sha256': '6ece827cb064e8f80f5a769412d3a5dc0a292c84e3e403c29dc9affa438bc507'}
PIN_FILES = {'protocol_sha256': 'protocol.json', 'environment_sha256': 'environment.json',
             'manifest_sha256': 'manifest.json', 'planned_jobs_sha256': 'planned_jobs.json',
             'reference_selections_sha256': 'reference_selections.json', 'provenance_sha256': 'provenance.json',
             'depth_summary_sha256': 'depth_summary.json'}
PREDICTION_KEYS = iv.PREDICTION_KEYS
KEY_COLUMNS = ('family', 'dataset_id', 'reference', 'outer_repeat', 'outer_fold', 'model_seed', 'arm_id')
THRESHOLD_COLUMNS = ('movement_threshold', 'voted_updates', 'cleared_updates', 'cleared_share_of_voted',
                     'mean_threshold_units_voted')
RELAY_COLUMNS = ('relays_to_layer_below', 'vote_scale', 'calls', 'attractions', 'repulsions', 'zero_votes',
                 'returned_motions_corrected', 'verified_attractions', 'verified_repulsions', 'scaled_votes',
                 'unscaled_vote_mass', 'scaled_vote_mass')
TABLES = {
    'arms.csv': KEY_COLUMNS + ('own_selection', 'reused', 'relay', 'lower_vote_scale', 'widths', 'learning_rate',
                               'embed_dim', 'degree', 'augment', 'accuracy', 'error', 'balanced_accuracy', 'macro_f1',
                               'neighborhood_purity', 'fit_seconds'),
    'depth_probe.csv': KEY_COLUMNS + ('view', 'depth', 'hidden_layers', 'knn_accuracy', 'neighborhood_purity',
                                      'majority_knn_accuracy'),
    'movement.csv': KEY_COLUMNS + ('view', 'layer', 'layer_name', 'n_filters', 'vocabulary', 'batches', 'updates',
                                   'changed_share', 'mean_displacement', 'mean_votes', 'mean_vote_mass', 'max_vote_mass',
                                   'incoming_to_prior_ratio') + THRESHOLD_COLUMNS,
    'relay.csv': KEY_COLUMNS + ('view', 'layer', 'layer_name') + RELAY_COLUMNS,
    'reproduction.csv': ('family', 'dataset_id', 'reference', 'outer_repeat', 'outer_fold', 'model_seed', 'arm_id',
                         'source_arm_id', 'refit_relay', 'refit_reproduced', 'refit_differences',
                         'plain_state_hashes_equal', 'plain_view_predictions_equal', 'lower_cleared_share_of_voted')}
CONTRAST_COLUMNS = ('family', 'role', 'contrast', 'subset', 'dataset_id', 'arm_a', 'arm_b', 'metric', 'mean_difference',
                    'standard_error', 'sd', 'ci_low', 'ci_high', 'n_folds', 'df', 'p_approximate', 'holm_p_approximate',
                    'significant_after_holm')
JOB_CHECKS = ('source_records', 'reference_predictions', 'reused_arms_reproduced', 'arm_parameters',
              'instrumentation_removed', 'rng_untouched', 'depth_probe_matches_readout', 'intervention_applied')
TIMING_KEYS = ('fit_seconds', 'encoding_seconds', 'training_seconds', 'readout_seconds')


class CheckFailed(td.CheckFailed):
    """A check of a signed-relay depth job failed."""


class AnalysisRefused(RuntimeError):
    """The analysis refuses to read a score."""


def environment():
    return environment_record(__package__ + '.signed_relay_depth:environment')


def _plain(value):
    return td._plain(value)


def utc_now():
    return td.utc_now()


# ----------------------------------------------------------------------------- arms

def lower_vote_scale(p=None):
    """The vote scale of depth2_signed_scaled: the frozen protocol's choice, else the pilot value."""
    choice = None if p is None else p.get('scale_choice')
    return (choice or {}).get('lower_vote_scale', SCALE_PILOT)


def arm_specs(p=None):
    """{arm: how it is fitted}. widths_key and last_layer_update drive interventions.arm_params; relay and
    lower_vote_scale drive signed_relay.RelayInstrumentation; source_arm names the depth run's arm a reused arm copies."""
    common = {'rule': 'borda', 'prior_rule': 'unit', 'prior_multiplier': 1, 'frozen_hidden': [], 'last_layer_update': True}
    return {'depth1': {**common, 'widths_key': 'depth1', 'relay': 'signed', 'lower_vote_scale': 1, 'source_arm': 'depth1'},
            'depth2_printed': {**common, 'widths_key': 'depth2', 'relay': 'printed', 'lower_vote_scale': 1,
                               'source_arm': 'depth2'},
            'depth2_signed': {**common, 'widths_key': 'depth2', 'relay': 'signed', 'lower_vote_scale': 1,
                              'source_arm': None},
            'depth2_signed_scaled': {**common, 'widths_key': 'depth2', 'relay': 'signed',
                                     'lower_vote_scale': lower_vote_scale(p), 'source_arm': None}}


def own_arm(selected, arm_widths=None):
    """The arm whose parameters are the fold's own selection (a reused arm): it carries the reference predictions."""
    arm_widths = ARM_WIDTHS if arm_widths is None else arm_widths
    widths = list(selected['widths'])
    if widths == list(arm_widths['depth1']):
        return 'depth1'
    if widths == list(arm_widths['depth2']):
        return 'depth2_printed'
    raise ValueError(f'The selected widths {widths} are not one of the declared depth arms')


# ----------------------------------------------------------------------------- fitting one arm

def fit_arm(params, seed, X, y, spec):
    """MultiViewArrowFlowKNN(**params, seed=seed).fit(X, y) step for step, as interventions.fit_arm, with
    signed_relay.RelayInstrumentation installed on each view network for its training only."""
    model = MultiViewArrowFlowKNN(**params, seed=seed)
    model.readouts_, model.readout_selections_ = [], []
    model.readout_seconds_ = 0.
    model.classes_ = np.unique(y)
    model.views_ = []
    encoding = training = 0.
    instrumentations = []
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
        instrumentation = sr.RelayInstrumentation(net, relay=spec['relay'], lower_vote_scale=spec['lower_vote_scale'])
        with instrumentation.installed():
            net.train_initialized(orders, y)
        instrumentations.append(instrumentation)
        training += net.training_seconds_
        model.views_.append((enc, net))
        model._fit_view_readout(enc, net, orders, y, seed_v)
    model.encoding_seconds_ = encoding
    model.training_seconds_ = training
    return model, instrumentations


def aggregate_movement(spec, view_records):
    """interventions.aggregate_movement (the same arithmetic, so a refitted arm's record equals the depth run's), then the
    prior's threshold per layer pooled over the views and the relay counts per layer summed over the views."""
    out = iv.aggregate_movement(spec, view_records)
    for entry in out['layers']:
        layers = [layer for record in view_records for layer in record['movement']['layers'] if layer['layer'] == entry['layer']]
        voted = sum(layer['voted_updates'] for layer in layers)
        cleared = sum(layer['cleared_updates'] for layer in layers)
        units = sum(layer['mean_threshold_units_voted'] * layer['voted_updates'] for layer in layers if layer['voted_updates'])
        entry.update({'movement_threshold': layers[0]['movement_threshold'], 'voted_updates': int(voted),
                      'cleared_updates': int(cleared), 'cleared_share_of_voted': float(cleared / voted) if voted else None,
                      'mean_threshold_units_voted': float(units / voted) if voted else None})
    relay_layers = defaultdict(list)
    for record in view_records:
        for layer in record['movement']['relay_layers']:
            relay_layers[layer['layer']].append(layer)
    out['relay'], out['lower_vote_scale'] = spec['relay'], spec['lower_vote_scale']
    out['relay_layers'] = [{'layer': index, 'layer_name': entries[0]['layer_name'],
                            'relays_to_layer_below': entries[0]['relays_to_layer_below'],
                            'vote_scale': entries[0]['vote_scale'],
                            **{key: sum(e[key] for e in entries) for key in ('calls', 'attractions', 'repulsions',
                                                                            'zero_votes', 'returned_motions_corrected',
                                                                            'verified_attractions', 'verified_repulsions',
                                                                            'scaled_votes')},
                            **{key: float(sum(e[key] for e in entries)) for key in ('unscaled_vote_mass', 'scaled_vote_mass')}}
                           for index, entries in sorted(relay_layers.items())]
    return out


def evaluate_arm(arm, spec, params, seed, X_train, y_train, X_query, y_query):
    """One fitted arm: the fit, its per-view probes and movement, the seven-view majority and its metrics, as
    interventions.evaluate_arm; the per-batch movement series leave the record for the arrays."""
    start = time.perf_counter()
    seed_fit(seed)
    model, instrumentations = fit_arm(params, seed, X_train, y_train, spec)
    fit_seconds = time.perf_counter() - start
    views, _ = model.predict_views(X_query)
    view_predictions = np.stack([np.asarray(view) for view in views])
    prediction = majority(view_predictions)
    records, depth_predictions = [], []
    checks = {'depth_probe_matches_readout': True, 'instrumentation_removed': True, 'rng_untouched': True,
              'intervention_applied': True}
    series = {}
    for v, ((enc, net), selection) in enumerate(zip(model.views_, model.readout_selections_)):
        probes, predictions = iv.view_probe(net, enc, selection, X_train, y_train, X_query, y_query)
        if not np.array_equal(predictions[-1], view_predictions[v]):
            checks['depth_probe_matches_readout'] = False
        depth_predictions.append(predictions)
        movement = instrumentations[v].summary()
        checks['instrumentation_removed'] &= bool(movement['instrumentation_removed'])
        checks['rng_untouched'] &= bool(movement['rng_unchanged'])
        checks['intervention_applied'] &= bool(movement['intervention_consistent'] and movement['relay'] == spec['relay']
                                               and movement['lower_vote_scale'] == float(spec['lower_vote_scale']))
        for layer in movement['layers']:
            layer['incoming_to_prior_ratio'] = iv.incoming_to_prior_ratio(spec, layer)
            series[f'v{v}__l{layer["layer"]}__changed'] = np.asarray(layer.pop('batch_changed_share'))
            series[f'v{v}__l{layer["layer"]}__displacement'] = np.asarray(layer.pop('batch_mean_displacement'))
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
              'prediction_hash': array_hash(np.asarray(prediction)), **metric_values(y_query, prediction),
              'reused': False, 'source': None}
    record['movement'] = aggregate_movement(spec, records)
    arrays = {'view_predictions': view_predictions, 'majority_by_depth': np.stack(majority_by_depth),
              'predictions': np.asarray(prediction), **series}
    return record, arrays, checks, prediction, state_hashes


# ----------------------------------------------------------------------------- the reused arms

def contains(new, old):
    """Every key of `old` is in `new` with an equal value, recursively (a refitted record carries more keys)."""
    if isinstance(old, dict):
        return isinstance(new, dict) and all(key in new and contains(new[key], value) for key, value in old.items())
    if isinstance(old, list):
        return isinstance(new, list) and len(new) == len(old) and all(contains(a, b) for a, b in zip(new, old))
    return new == old


def source_arm(source_record, source_arrays, arm, spec, job):
    """A reused arm: the source depth record's arm copied verbatim under this family's arm name, with its source, and
    its arrays (predictions and movement series) renamed."""
    name = REUSED[arm]
    stored = next(entry for entry in source_record['arms'] if entry['arm_id'] == name)
    record = json.loads(json.dumps(stored))
    record.update({'arm_id': arm, 'spec': _plain(spec), 'reused': True,
                   'source': {'protocol_id': job['source']['protocol_id'], 'arm_id': name, 'stem': job['stem'], 'spec': stored['spec'],
                              'record_sha256': job['source']['record_sha256'],
                              'artifact_sha256': job['source']['artifact_sha256']}})
    prefix = f'{name}__'
    arrays = {key[len(prefix):]: value for key, value in source_arrays.items() if key.startswith(prefix)}
    return record, arrays


def reproduction_differences(stored, stored_arrays, refit, refit_arrays, *, outer):
    """The fields in which a refitted reused arm differs from the stored record; empty when it reproduces it. Training
    quantities always (state hashes, parameters, the movement record and its per-batch series); on outer test rows also
    every prediction, probe, readout selection and metric."""
    differences = [key for key in ('state_hashes', 'params', 'widths', 'hidden_layers') if refit[key] != stored[key]]
    if not contains(refit['movement'], stored['movement']):
        differences.append('movement')
    for old, new in zip(stored['views'], refit['views']):
        keys = ('view', 'strategy', 'view_seed', 'readout', 'depths') if outer else ('view', 'strategy', 'view_seed')
        differences.extend(f'views[{old["view"]}].{key}' for key in keys if new[key] != old[key])
        if not contains(new['movement'], old['movement']):
            differences.append(f'views[{old["view"]}].movement')
    names = sorted(key for key in stored_arrays if '__l' in key)
    if outer:
        names += ['view_predictions', 'majority_by_depth', 'predictions']
        differences.extend(key for key in ('prediction_hash', 'neighborhood_purity', 'majority_by_depth', *METRICS)
                           if refit[key] != stored[key])
    differences.extend(f'arrays.{key}' for key in names
                       if key not in refit_arrays or not np.array_equal(refit_arrays[key], stored_arrays[key]))
    if sorted(key for key in refit_arrays if '__l' in key) != sorted(key for key in stored_arrays if '__l' in key):
        differences.append('arrays.movement_series')
    return differences


def reproduce_reused(arm, spec, params, seed, stored, stored_arrays, X_train, y_train, X_query, y_query, *, outer):
    """A reused arm refitted twice: through this module (depth1 with the corrected relay, depth2_printed in the printed
    mode) and by the unmodified library (MultiViewArrowFlowKNN.fit); both must reproduce the stored record."""
    record, arrays, checks, _, _ = evaluate_arm(arm, spec, params, seed, X_train, y_train, X_query, y_query)
    differences = reproduction_differences(stored, stored_arrays, record, arrays, outer=outer)
    differences.extend(name for name, passed in checks.items() if not passed)
    start = time.perf_counter()
    seed_fit(seed)
    plain = MultiViewArrowFlowKNN(**params, seed=seed).fit(X_train, y_train)
    plain_views = np.stack([np.asarray(view) for view in plain.predict_views(X_query)[0]])
    plain_hashes = [net.state_hash() for _, net in plain.views_]
    plain_seconds = time.perf_counter() - start
    lower = next((layer for layer in record['movement']['layers'] if layer['layer'] == 0), {})
    entry = {'arm_id': arm, 'source_arm_id': REUSED[arm], 'refit_relay': spec['relay'],
             'refit_reproduced': not differences, 'refit_differences': differences,
             'plain_state_hashes_equal': plain_hashes == stored['state_hashes'],
             'plain_view_predictions_equal': bool(np.array_equal(plain_views, stored_arrays['view_predictions'])) if outer else None,
             'refit_seconds': record['fit_seconds'], 'plain_seconds': plain_seconds,
             'refit_relay_layers': record['movement']['relay_layers'],
             'lower_cleared_share_of_voted': lower.get('cleared_share_of_voted'),
             'refit_layers': [{key: layer[key] for key in ('layer', 'changed_share', 'mean_displacement') + THRESHOLD_COLUMNS}
                              for layer in record['movement']['layers']]}
    entry['passed'] = bool(entry['refit_reproduced'] and entry['plain_state_hashes_equal']
                           and entry['plain_view_predictions_equal'] is not False)
    return entry


# ----------------------------------------------------------------------------- one job

def evaluate_job(X, y, split, job, sealed, source, p, *, dataset_hash, code_revision, protocol_hash, query=None):
    """One dataset, outer fold and seed: the two reused arms from the verified source record, the two corrected arms
    fitted, and on the first outer fold of each dataset the reused arms refitted. query=None scores the outer test rows;
    the pilot passes training rows (training-only timing; the reused arms are then compared on training quantities)."""
    seed, specs = int(p['model_seed']), arm_specs(p)
    outer = query is None
    train = [int(i) for i in split['train']]
    rows = [int(i) for i in (split['test'] if outer else query)]
    X_train, y_train, X_query, y_query = X[train], y[train], X[rows], y[rows]
    identity = {'family': FAMILY, 'dataset_id': job['dataset_id'], 'reference': job['reference'],
                'dataset_hash': dataset_hash, 'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'],
                'split_hash': config_id(split), 'query': 'outer_test_rows' if outer else 'training_rows_for_timing',
                'query_ids_hash': array_hash(np.asarray(rows)), 'config_id': job['config_id'], 'config': job['config'],
                'selected': job['selected'], 'own_arm': job['own_arm'], 'model_seed': seed,
                'reuse_check': bool(job['reuse_check']), 'source': job['source'], 'code_revision': code_revision,
                'protocol_hash': protocol_hash}
    checks, timing = {}, {}
    result = {'status': 'running', 'identity': identity, 'checks': checks, 'timing': timing}
    arrays = None
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            collected, stored, flags = {}, {}, defaultdict(lambda: True)
            for arm in p['arms']:
                planned = next(entry for entry in job['arms'] if entry['arm_id'] == arm)
                params = iv.arm_params(specs[arm], job['selected'], p['arm_widths'])
                if params != planned['params'] or planned['spec'] != _plain(specs[arm]):
                    raise CheckFailed(f'{arm}: the fitted parameters or specification differ from the planned ones')
            checks['arm_parameters'] = {'performed': True, 'passed': True}
            record = source['record']
            source_checks = record['checks']
            if not (record['status'] == 'ok' and all(check['passed'] is not False for check in source_checks.values())
                    and all(source_checks[name]['passed'] is True for name in source_checks if source_checks[name]['performed'])):
                raise CheckFailed('the source depth record did not pass every check')
            for arm in REUSED:
                collected[arm], stored[arm] = source_arm(record, source['arrays'], arm, specs[arm], job)
            checks['source_records'] = {'performed': True, 'passed': True, 'stem': job['stem'],
                                        'record_sha256': job['source']['record_sha256'],
                                        'artifact_sha256': job['source']['artifact_sha256'],
                                        'prediction_sha256': job['source']['prediction_sha256']}
            for arm in FITTED:
                params = next(entry['params'] for entry in job['arms'] if entry['arm_id'] == arm)
                arm_record, arm_arrays, arm_checks, _, _ = evaluate_arm(arm, specs[arm], params, seed, X_train, y_train,
                                                                        X_query, y_query)
                collected[arm], stored[arm] = arm_record, arm_arrays
                for name, value in arm_checks.items():
                    flags[name] &= bool(value)
                timing[f'{arm}_seconds'] = arm_record['fit_seconds']
            for name in ('instrumentation_removed', 'rng_untouched', 'depth_probe_matches_readout', 'intervention_applied'):
                checks[name] = {'performed': True, 'passed': bool(flags[name])}
            own = job['own_arm']
            if outer:
                check = td.reference_check(stored[own]['predictions'], sealed, seed)
                source_check = source_checks['reference_predictions']
                check.update(arm_id=own, source_arm_id=REUSED[own],
                             source_check_passed=bool(source_check['passed'] is True and source_check['arm_id'] == REUSED[own]))
                check['passed'] = bool(check['passed'] and check['source_check_passed']
                                       and collected[own]['prediction_hash'] == job['reference_prediction_hash'])
                checks['reference_predictions'] = check
            else:
                checks['reference_predictions'] = {'performed': False, 'passed': None, 'arm_id': own,
                                                   'reason': 'training-only pilot: the query rows are outer training rows'}
            if job['reuse_check']:
                entries = []
                for arm in REUSED:
                    params = next(entry['params'] for entry in job['arms'] if entry['arm_id'] == arm)
                    start = time.perf_counter()
                    entries.append(reproduce_reused(arm, specs[arm], params, seed, collected[arm], stored[arm], X_train,
                                                    y_train, X_query, y_query, outer=outer))
                    timing[f'{arm}_reuse_seconds'] = time.perf_counter() - start
                    timing[f'{arm}_refit_seconds'] = entries[-1]['refit_seconds']
                    timing[f'{arm}_plain_seconds'] = entries[-1]['plain_seconds']
                checks['reused_arms_reproduced'] = {'performed': True, 'passed': all(e['passed'] for e in entries),
                                                    'outer_rows': outer, 'arms': entries}
            else:
                checks['reused_arms_reproduced'] = {'performed': False, 'passed': None,
                                                    'reason': 'performed on the first outer fold of each dataset'}
        failed = sorted(name for name, check in checks.items() if check['performed'] and not check['passed'])
        if failed:
            raise CheckFailed('checks failed: ' + ', '.join(failed))
        arrays = {'query_rows': np.asarray(rows), 'query_labels': np.asarray(y_query)}
        for arm in p['arms']:
            for name, value in stored[arm].items():
                arrays[f'{arm}__{name}'] = value
        result.update(status='ok', arms=[collected[arm] for arm in p['arms']])
    except Exception as exc:
        result.update(status='failed', exception=f'{type(exc).__name__}: {exc}', check_failure=isinstance(exc, td.CheckFailed))
        arrays = None
    timing['job_seconds'] = time.perf_counter() - started
    return td.native(result), arrays


# ----------------------------------------------------------------------------- protocol

def arm_definitions():
    return {
        'depth1': 'the single hidden layer [128]; a network with one hidden layer never relays, so it is the same network '
                  'under the printed and the corrected relay. Reused verbatim from the depth run (its depth1); on the '
                  'first outer fold of every dataset it is refitted with the corrected relay installed and must '
                  'reproduce the stored record exactly',
        'depth2_printed': 'two hidden layers [64, 128] with the printed (flawed) relay: the unmodified library. Reused '
                          'verbatim from the depth run (its depth2); on the first outer fold of every dataset it is '
                          'refitted through this module in the printed mode and must reproduce the stored record exactly',
        'depth2_signed': 'two hidden layers [64, 128] with the corrected relay: every relayed motion is sign(a_j) times '
                         'the motion toward the unreversed input',
        'depth2_signed_scaled': 'two hidden layers [64, 128] with the corrected relay and every first-layer vote '
                                'multiplied by the vote scale s of scale_choice, chosen on the training-only pilot',
        'note': 'the four arms share the fold\'s encoder, view identities, initial seeds, learning rate, iteration count, '
                'batch size, vocabulary, degree, augmentation, checkpoint and readout rule; only the hidden widths, the '
                'relay and the first layer\'s vote scale differ'}


def analysis_declaration():
    variants = list(PRIMARY_VARIANTS)
    return {
        'status': 'prespecified before any outer score of a corrected arm exists; computed only by the analyse stage, which '
                  'refuses (exit 2, nothing written) until the run is complete, then re-verifies every job record, every '
                  'sealed selection re-derived from its reference run, the source depth run and every table before any '
                  'score is read. The outer scores of depth1 and depth2_printed exist already (arrowflow-v3-depth-1, '
                  'reported in the manuscript); they are reused, not re-estimated',
        'command': 'python -m experiments.make_revision.signed_relay_depth analyse --run <run> --output <directory>',
        'metric': 'accuracy', 'reference_arm': REFERENCE_ARM, 'alpha': ALPHA,
        'primary': {'reference_arm': REFERENCE_ARM, 'variant_arms': variants,
                    'definition': 'per dataset and variant arm, depth1 minus the variant arm accuracy, the fitting seeds '
                                  'averaged within each outer fold (one seed here, so the fold value is that seed\'s)',
                    'interval': 'corrected resampled t over the 15 outer folds (evaluation.paired_corrected_interval, '
                                'q = 0.25, 95 per cent, 14 degrees of freedom)',
                    'multiplicity': 'Holm across the seventeen datasets within each variant arm (evaluation.holm_adjust): '
                                    'two families of seventeen, adjusted separately'},
        'secondary': {'arm_a': SECONDARY[0], 'arm_b': SECONDARY[1],
                      'definition': 'per dataset, depth2_signed minus depth2_printed accuracy, seeds averaged within fold',
                      'interval': 'the same corrected resampled t (q = 0.25, 95 per cent, 14 degrees of freedom)',
                      'multiplicity': 'Holm across the seventeen datasets (one family of seventeen)'},
        'families': [{'role': 'primary', 'contrast': f'{REFERENCE_ARM}_minus_{arm}', 'size': 17} for arm in variants]
        + [{'role': 'secondary', 'contrast': f'{SECONDARY[0]}_minus_{SECONDARY[1]}', 'size': 17}],
        'descriptive': 'per arm and dataset: accuracy, balanced accuracy, macro-F1 and the neighbourhood purity of the last '
                       'hidden ranking (outer-fold mean and SD); the kNN probe after each hidden layer; per hidden layer '
                       'the changed-filter share, the normalised displacement and, for the fitted arms, the share of voted '
                       'filter-batches that reach the prior\'s movement threshold; the relay counts; no p value and no '
                       'adjustment for these',
        'named_subset': {
            'status': 'descriptive, not a new family, no p value and no place in any Holm family',
            'definition': 'the primary and secondary differences restricted to the outer folds whose own reconstructed '
                          'selection was two hidden layers, and to those whose selection was one (knn_controls.depth_split '
                          'with depths ' + canonical_json(list(ARM_WIDTHS.values())) + '); an interval only with at '
                          f'least {iv.MIN_SUBSET_FOLDS} folds, df = n - 1'},
        'interpretation': {
            'rule': f'For each corrected arm A (depth2_signed, depth2_signed_scaled): A helps if (i) on at least one '
                    f'dataset depth1 minus A is negative and significant after Holm at alpha {ALPHA} within A\'s family, '
                    f'and (ii) A\'s mean accuracy over the fifteen outer folds is higher than depth1\'s (the mean '
                    f'difference depth1 minus A is negative) on at least {MOST_DATASETS} of the seventeen datasets '
                    '(most: more than half; an equal mean does not count). The outcome is "helps" if either corrected '
                    'arm helps and "does not help" otherwise; the secondary contrast is reported beside it and does not '
                    'change the outcome',
            'helps': 'the paper reports that a second hidden layer helps once the relay is correct, and the controller '
                     'raises with the author whether to rerun the main benchmark with the corrected relay',
            'does_not_help': 'the paper reports that even with a corrected relay a second hidden layer did not help at a '
                             'matched configuration'},
        'limits': 'a controlled comparison at one configuration per fold (the depth run\'s: the fold\'s own selection with '
                  'only the widths, the relay and the first layer\'s vote scale changed), at one fitting seed, with the '
                  'learning rate and the iteration count not re-tuned; it cannot estimate the best attainable two-layer '
                  'model with the corrected relay, and a scale fixed on two pilot datasets is one choice, not a search'}


def design():
    """The fixed design every protocol of this family declares (validate_protocol requires it verbatim)."""
    specs = arm_specs()
    return {
        'purpose': 'the depth study rerun with the corrected relay: whether a second hidden layer helps ArrowFlow-kNN once '
                   'the signal relayed to the lower layer after a repulsion vote is correctly signed, at one fixed '
                   'configuration per outer fold',
        'selection_statement': 'Nothing is selected on outer-fold results. Every configuration is the per-fold selection '
                               'sealed by the reference ablation run (the depth run\'s); every kNN readout setting is the '
                               'one the fit chose on training rows; the vote scale is chosen on the training-only pilot '
                               'before the run; no outer score feeds back into any fit, arm, dataset, fold or analysis '
                               'choice.',
        'unit_of_work': 'every protocol dataset x every outer fold (5 folds x 3 repeats) x fitting seed 8129 x every arm',
        'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3, 'split_seed': 27183, 'fit_seeds': [8129, 19391, 39019],
        'model_seed': MODEL_SEED, 'n_views': N_VIEWS, 'test_train_ratio': .25, 'confidence': .95,
        'fit_seed_statement': 'As in the depth run, the arms are fitted at seed 8129 alone (the reference design fits three '
                              'seeds per outer fold). The arms of one fold share one encoder draw and one initial state, '
                              'so every comparison is paired within seed; every interval is over the fifteen outer folds '
                              'of that one seed.',
        'arms': list(ARMS), 'arm_definitions': arm_definitions(), 'arm_widths': dict(ARM_WIDTHS),
        'reference_arm': REFERENCE_ARM, 'reused_arms': dict(REUSED), 'fitted_arms': list(FITTED),
        'own_arm': 'the arm whose parameters are the fold\'s own reconstructed selection (depth1 or depth2_printed by the '
                   'selected widths, a reused arm); it must carry the reference run\'s outer predictions for seed 8129',
        'source_depth_run': dict(SOURCE_DEPTH_RUN),
        'reuse': 'depth1 and depth2_printed are copied verbatim (renamed) from the depth run\'s job records, artifacts and '
                 'predictions, which prepare re-verifies (interventions.collect with every selection re-derived, every '
                 'table and provenance.json re-rendered) and pins by the sha256 recorded in its provenance; on the first '
                 'outer fold of every dataset both are refitted through this module (depth1 with the corrected relay '
                 'installed, depth2_printed in the printed mode) and by the unmodified library, and must reproduce the '
                 'stored record exactly',
        'relay': {
            'defect': 'Vertex.compute_distance reverses the comparison list and drops the sign for a negative vote '
                      '(arrowflow.py:746-749), Vertex.accumulate_motion returns the motion toward that reversed list '
                      '(873-884), and backward_propagate hands it, unsigned, to the layer below (1828): after a repulsion '
                      'the lower layer is sent m(r -> rev(pi)), which is orthogonal to the push-away direction',
            'printed': 'd_j = m(r_j -> tau~_j), tau~_j = pi for a_j > 0 and rev(pi) for a_j < 0; unsigned',
            'corrected': 'd_j = sign(a_j) m(r_j -> pi), m(r -> pi)[p] = pos_pi(r[p]) - p: the motion toward the unreversed '
                         'input, signed. For a_j > 0 and a_j = 0 it is the core\'s own return value; for a_j < 0 it is '
                         'm(r_j -> rev(pi))[p] + 2p - (V - 1) = -m(r_j -> pi)[p], pi a complete permutation of the '
                         'filter\'s V items',
            'unchanged': 'the filter\'s own vote and its accumulation (a repulsion still accumulates toward rev(pi)), the '
                         'eligibility gate, the first ceil(rho N) entries, the unweighted mean over the example\'s nonzero '
                         'votes, eq. (6) with c = 0.125, the output layer, the prior, the batch resets, the checkpoint and '
                         'the readout',
            'weighting': 'not |a|-weighted: the variant a_j m(r_j -> pi) would also reweight the attraction relays, so it '
                         'would not equal the printed relay on attraction votes and the secondary contrast would mix two '
                         'changes; V1\'s instrumented runs found the two variants alike (iris: 71.5 against 71.8 per cent '
                         'of repelled pairs moved away)',
            'implementation': 'signed_relay.RelayedMotion, an instance-level Vertex.accumulate_motion on every hidden '
                              'filter: the core\'s accumulate_motion runs unchanged and, in the corrected mode, a '
                              'repulsion\'s return value is replaced; backward_propagate uses that value only for the '
                              'relay, so the one change point is arrowflow.py:1828\'s input. The first 8 repulsions and '
                              'attractions of each layer and view are checked against the directly computed motion',
            'scope': 'only a hidden layer above the first relays; a network with one hidden layer is unchanged by '
                     'construction'},
        'vote_scale': {
            'arm': 'depth2_signed_scaled',
            'definition': 'every vote cast at a hidden layer below the top hidden layer ([64, 128]: the first hidden layer) '
                          'is multiplied by s before it is accumulated (signed_relay.ScaledVote); the top hidden layer\'s '
                          'votes, the relayed signal, the output layer and everything else are unchanged',
            'problem': 'eq. (6) rescales the relayed signal to at most c N = 8 (c = 0.125, N = 64 first-layer filters), '
                       'so a first-layer vote 2 eta u / N is at most 2 eta c = eta / 4, while a top hidden layer vote is up '
                       'to 2 eta 127/128; after every '
                       'batch reset the prior carries weight 1 and dictates a filter\'s order unless its batch vote mass '
                       'reaches 1 / (e - 1), e the first layer\'s vocabulary: 0.0667, 0.0323, 0.0159 and 0.0079 at '
                       'e = 16, 32, 64 and 128',
            'measure': 'the cleared share: over the first hidden layer\'s filter-batches with at least one nonzero vote, '
                       'in all seven view networks of a fit, the share whose batch vote mass M reaches the threshold, '
                       'M (e - 1) >= 1',
            'ladder': list(SCALE_LADDER), 'target': CLEARING_TARGET,
            'rule': f's is the smallest ladder value whose cleared share is at least {CLEARING_TARGET} on the training-only '
                    'pilot fit of every pilot dataset; if no ladder value reaches it, the largest. No outer-fold row, no '
                    'score and no accuracy takes part',
            'pilot_value': f'the pilot job fits depth2_signed_scaled at s = {SCALE_PILOT} (= 1/c, where the first layer\'s '
                           'largest vote 2 eta c s equals the top hidden layer\'s, 2 eta 127/128, to within 1/128) and '
                           'reads s = 1 from depth2_signed; the ladder refits the other values',
            'sealed': 'the chosen s and its ladders are recorded in the frozen protocol (scale_choice) and read by arm_specs',
            'caveat': 'one s for every fold and vocabulary: where the first layer already reaches the threshold at s = 1 '
                      '(larger vocabularies, higher training error) the scaled arm moves it further; the movement and '
                      'the cleared share are reported per fold'},
        'fit': 'MultiViewArrowFlowKNN at bridge.resolve_selected(sealed config, n_features, n_train) on the outer training '
               'rows with the arm\'s hidden widths: for view v, OrdinalEncoder(view_strategy, embed_dim, degree, lda_ratio, '
               'derive_seed(8129, "view", v)), ArrowFlowEstimator.initialize_orders, signed_relay.RelayInstrumentation '
               'installed, train_initialized, the instrumentation removed, then MultiViewArrowFlowKNN._fit_view_readout; '
               'majority vote over the seven per-view kNN readouts',
        'not_retuned': 'the learning rate, the iteration count, the batch size, the checkpoint ratio, the encoder, the '
                       'vocabulary, the degree and the augmentation are the fold\'s own selection and are not re-tuned '
                       'for any arm; this is a controlled comparison at one configuration and not an estimate of the '
                       'best attainable model of any arm',
        'instrumentation': 'signed_relay.RelayInstrumentation on each view network between initialize_orders and '
                           'train_initialized, removed afterwards: update_rules.ArmInstrumentation (the unmodified Borda '
                           'rule, no frozen layer: vote counter, movement record, RNG-guarded batch close) with the movement '
                           'record extended by the prior\'s threshold, RelayedMotion on every hidden filter (the printed '
                           'mode returns the core\'s value), and ScaledVote below the top hidden layer when s != 1',
        'measures': {
            'outer_metrics': 'per arm, the seven-view majority on the outer test rows: accuracy, error, balanced accuracy '
                             'and macro-F1 (evaluation.metric_values)',
            'neighborhood_purity': 'per arm, the mean over views of the share of same-class neighbours of the outer test '
                                   'rows in the last hidden ranking (k the view readout\'s own n_neighbors)',
            'depth_probe': 'per arm, view and hidden layer, the representation after that layer read out by a footrule kNN '
                           'at the view\'s own selected setting refitted on the training rows at that depth, scored on '
                           'the outer test rows, with the seven-view majority at each depth; at the last hidden layer the '
                           'probe must equal the view readout',
            'movement': 'per arm, view and hidden layer, over the training batches: the share of filters that changed and '
                        'the mean normalised footrule displacement (footrule / floor(V^2 / 2)), the eligible votes and '
                        'their mass; for the fitted arms also the voted filter-batches and the share of them whose mass '
                        'reaches the prior\'s threshold 1 / (V - 1)',
            'relay': 'per fitted arm, view and hidden layer: the votes by sign, the returned motions corrected, the '
                     'motions checked directly and the scaled vote mass'},
        'checks': {
            'source_records': 'every job: the source depth record, artifact and predictions file carry the sha256 sealed '
                              'from the source run\'s verified provenance, every source check passed, and the reused '
                              'arms are its arms verbatim (renamed)',
            'reference_predictions': 'every job: the own arm carries the reference run\'s recorded outer predictions for '
                                     'seed 8129 exactly, and its source reference check passed',
            'reused_arms_reproduced': 'first outer fold of every dataset: depth1 refitted with the corrected relay and '
                                      'depth2_printed refitted in the printed mode reproduce the stored record exactly '
                                      '(every view state hash, the view, depth and majority predictions, the readout '
                                      'selections, the probes, the metrics, the movement record and its per-batch series), '
                                      'and MultiViewArrowFlowKNN(**params, seed=8129).fit gives the same view state hashes '
                                      'and view predictions',
            'arm_parameters': 'every job: the fitted parameters and specifications equal the planned ones, and the planned '
                              'parameters equal the depth run\'s planned depth1 and depth2 parameters of the fold',
            'instrumentation_removed': 'every job: no vertex of any reachable graph keeps an instance-level wrapper',
            'rng_untouched': 'every job: every guarded read left the global numpy and Python RNG states unchanged',
            'depth_probe_matches_readout': 'every job: the last depth probe equals the view readout',
            'intervention_applied': 'every job: in each fitted arm every repulsion\'s returned motion was corrected (none '
                                    'in the printed mode), the direct checks held, and in the scaled arm every nonzero '
                                    'first-layer vote was scaled by s and its accumulated mass is s times the incoming one'},
        'outputs': {'job_records': 'jobs/<dataset>__r<repeat>f<fold>.json with every measure and check, '
                                   'artifacts/<stem>.npz with the per-arm view, depth and majority predictions and the '
                                   'per-batch movement series, predictions/<stem>.jsonl with the per-example outer '
                                   'predictions of every arm',
                    'tables': ', '.join(TABLES) + ' and provenance.json, written only after every planned job succeeded '
                              'and verified (all or none)',
                    'summary': f'{FAMILY}_summary.json and {FAMILY}_summary.csv by the summary stage',
                    'analysis': f'{FAMILY}_contrasts.csv and {FAMILY}_analysis.json by the analyse stage',
                    'overwrite': 'an existing file with different content is never replaced'},
        'analysis': analysis_declaration(),
        'dataset_loading': 'the reference run prepared data (hash-checked) must equal the ablation copy, its splits the '
                           'declared nested splits, and in production a fresh load by the reference loader must return '
                           'the same arrays and dataset hash; the selections, splits and plan must equal the depth run\'s',
        'pilot': 'training-only: the first outer fold of each pilot dataset with the outer test rows replaced by every '
                 'fourth outer training row (the outer test fold is never touched): every fitted arm, the reuse refits '
                 'compared with the stored records on training quantities only (state hashes, movement), the unmodified '
                 'library fits, the scale ladder, and one reproduction probe (run_knn_ablation.reproduction_probe) of the '
                 'smallest dataset per reference',
        'decision_rule': f'freeze only if the calibrated projection at {WORKERS} single-thread workers is at most '
                         f'{CAP_HOURS} h: the simulated first-free-worker makespan of the planned jobs in planned order, '
                         'each job priced at the reference run\'s realized outer fit and predict seconds of its fold and '
                         'seed 8129 times the piloted ratio of the job\'s seconds without the reuse reproduction to the own '
                         'arm\'s unmodified-library fit and predict seconds, plus, on a first outer fold, the piloted ratio '
                         'of the reuse reproduction\'s seconds (the refits and unmodified fits of both reused arms) to the '
                         'same (each dataset\'s own ratios for a pilot dataset, the largest piloted ones otherwise). Both '
                         'sides of each ratio are measured in the same pilot, so machine load largely cancels in it while '
                         'the realized reference seconds carry the absolute scale',
        'wallclock_cap_hours': CAP_HOURS, 'workers': WORKERS, 'max_workers': MAX_WORKERS, 'numeric_threads_per_worker': 1,
        'parallelism': 'one spawned single-thread process per dataset and outer fold under the shared execution lock',
        'failure_policy': 'a failed check fails its job; a failed job cancels the pending jobs and no table is written; '
                          'failures are never omitted; no adaptive stopping on any value',
        'arm_rules': {arm: {key: specs[arm][key] for key in sorted(specs[arm]) if key != 'lower_vote_scale'}
                      for arm in ARMS},
    }


def draft_protocol(references=None, protocol_id=PROTOCOL_ID, pilot_datasets=PILOT_DATASETS, source_depth_run=None):
    """The unfrozen protocol; the datasets follow the references in name order (a JSON protocol keeps no key order)."""
    references = td.REFERENCES if references is None else references
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    block = design()
    if source_depth_run is not None:
        block['source_depth_run'] = source_depth_run
    return _plain({**block, 'protocol_id': protocol_id, 'production_family': FAMILY, 'datasets': datasets,
                   'references': references, 'pilot_datasets': list(pilot_datasets), 'frozen': False,
                   'status': DRAFT_STATUS, 'scale_choice': None,
                   'resource_decision': 'pending: synthetic smoke and the training-only pilot on ' + ' and '.join(pilot_datasets)})


def validate_protocol(p):
    """A production protocol is draft_protocol for its references and source run, or that draft with exactly the freeze
    fields set by freeze."""
    if p.get('production_family') != FAMILY:
        raise ValueError(f'The protocol is not a {FAMILY} protocol')
    draft = draft_protocol(references=p.get('references'), protocol_id=p.get('protocol_id'),
                           pilot_datasets=p.get('pilot_datasets', ()), source_depth_run=p.get('source_depth_run'))
    strip = lambda q: {k: v for k, v in q.items() if k not in FREEZE_FIELDS}
    if strip(p) != strip(draft):
        differing = sorted(k for k in set(p) | set(draft) if k not in FREEZE_FIELDS and p.get(k) != draft.get(k))
        raise ValueError(f'The protocol differs from signed_relay_depth.draft_protocol() in {", ".join(differing)}')
    td.reference_of(p)
    if not p.get('frozen'):
        if p != draft:
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('pilot_projection') or {}
    hours = projection.get('decision_hours')
    choice = p.get('scale_choice') or {}
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != CAP_HOURS or projection.get('workers') != WORKERS
            or isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= CAP_HOURS):
        raise ValueError(f'A frozen protocol records its freeze and a pilot projection within the {CAP_HOURS} h cap at '
                         f'{WORKERS} workers')
    if choice.get('lower_vote_scale') not in SCALE_LADDER:
        raise ValueError('A frozen protocol records the vote scale chosen on the training-only pilot ladder')
    return p


# ----------------------------------------------------------------------------- the source depth run

def source_pins(directory):
    """The observed pins of one depth run directory."""
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    pins = {key: sha256_file(directory/name) for key, name in PIN_FILES.items()}
    pins.update({'family': protocol.get('production_family'), 'protocol_id': protocol.get('protocol_id'),
                 'code_revision': json.loads((directory/'environment.json').read_text()).get('code_revision')})
    return pins


def load_source_depth_run(directory, p, *, allow_smoke=False):
    """The completed depth run the reused arms come from: its pins, a full re-verification (interventions.collect with
    every selection re-derived; the tables, provenance.json and depth_summary.json equal to their re-rendering) and its
    design equal to this protocol's. Returns the verified run."""
    directory = Path(directory)
    missing = [name for name in SOURCE_FILES if not (directory/name).is_file()]
    if missing:
        raise ValueError(f'The source depth run {directory} is not a complete run (missing {", ".join(missing)})')
    declared, observed = p['source_depth_run'], source_pins(directory)
    wrong = [f'{key}: declared {declared.get(key)!r}, observed {value!r}' for key, value in observed.items()
             if declared.get(key) != value]
    if wrong:
        raise ValueError('The source depth run does not match the protocol pins: ' + '; '.join(wrong))
    collected = iv.collect(directory, allow_smoke=allow_smoke, rederive=True)
    contents = iv.render_tables(collected)
    for name, text in contents.items():
        if (directory/name).read_text() != text:
            raise ValueError(f'The source depth run\'s {name} differs from its verified records')
    provenance = json.loads((directory/iv.PROVENANCE_FILE).read_text())
    if provenance != iv.provenance_record(directory, collected, contents, provenance.get('run')):
        raise ValueError('The source depth run\'s provenance.json differs from its verified records')
    if json.loads((directory/'depth_summary.json').read_text()) != iv.summarize(collected):
        raise ValueError('The source depth run\'s depth_summary.json differs from its verified records')
    source = collected['protocol']
    if not allow_smoke:
        committed = json.loads((Path(__file__).resolve().parents[2]/declared['protocol_file']).read_text())
        if source != committed or not source.get('frozen'):
            raise ValueError('The source depth run\'s protocol is not the committed frozen depth protocol')
    for key in ('datasets', 'outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds', 'model_seed',
                'n_views', 'arm_widths', 'references'):
        if source[key] != p[key]:
            raise ValueError(f'The source depth run and this protocol disagree on {key}')
    if source['production_family'] != 'depth' or any(arm not in source['arms'] for arm in REUSED.values()):
        raise ValueError('The source run is not a depth run holding depth1 and depth2')
    return {'directory': directory, 'collected': collected, 'provenance': provenance, 'observed': observed}


def load_source_job(directory, job):
    """The source depth job of one planned job, every file checked against the sha256 sealed in the plan."""
    directory, pins = Path(directory), job['source']
    paths = {'record': directory/'jobs'/f'{job["stem"]}.json', 'artifact': directory/'artifacts'/f'{job["stem"]}.npz',
             'prediction': directory/'predictions'/f'{job["stem"]}.jsonl'}
    for kind, path in paths.items():
        if not path.is_file() or sha256_file(path) != pins[f'{kind}_sha256']:
            raise ValueError(f'{job["stem"]}: the source {kind} file differs from the sealed plan')
    record = json.loads(paths['record'].read_text())
    with np.load(paths['artifact'], allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    return {'record': record, 'arrays': arrays}


# ----------------------------------------------------------------------------- prepare and verify

def planned_job(p, reference, name, index, split, record, n_features, source):
    specs = arm_specs(p)
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    key = (name, split['outer_repeat'], split['outer_fold'])
    sealed_job = reference['ablation']['jobs'].get(key)
    if sealed_job is None or sealed_job['config_id'] != record['config_id'] or sealed_job['selected'] != selected:
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the resolved selection differs from the ablation plan')
    stem = f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}'
    source_job = source['jobs'].get(stem)
    if source_job is None or source_job['selected'] != selected or source_job['config_id'] != record['config_id']:
        raise ValueError(f'{stem}: the source depth run planned another selection')
    arms = [{'arm_id': arm, 'params': iv.arm_params(specs[arm], selected, p['arm_widths']),
             'widths': iv.arm_widths(specs[arm], selected, p['arm_widths']), 'spec': _plain(specs[arm])} for arm in p['arms']]
    source_params = {entry['arm_id']: entry['params'] for entry in source_job['arms']}
    for entry in arms:
        counterpart = REUSED.get(entry['arm_id'], 'depth2')
        if entry['params'] != source_params[counterpart]:
            raise ValueError(f'{stem}: {entry["arm_id"]} is not planned at the depth run\'s {counterpart} parameters')
    seed = str(p['model_seed'])
    own = own_arm(selected, p['arm_widths'])
    if REUSED[own] != source_job['own_arm']:
        raise ValueError(f'{stem}: the own arm differs from the depth run\'s')
    pins = source['provenance']['jobs'][stem]
    return {'dataset_id': name, 'reference': reference['name'], 'outer_repeat': split['outer_repeat'],
            'outer_fold': split['outer_fold'], 'stem': stem, 'config_id': record['config_id'], 'config': record['config'],
            'selected': selected, 'selected_widths': list(selected['widths']), 'model_seed': int(p['model_seed']),
            'reuse_check': index == 0, 'own_arm': own, 'arms': arms,
            'source': {'protocol_id': source['collected']['protocol']['protocol_id'], 'stem': stem,
                       'record_sha256': pins['record_sha256'], 'artifact_sha256': pins['artifact_sha256'],
                       'prediction_sha256': pins['prediction_sha256']},
            'reference_prediction_hash': record['reference_prediction_hashes'][seed],
            'reference_outer_seconds': record['reference_outer_seconds'][seed]}


def prepare(output, p, sources, source_directory, datasets=None, *, allow_smoke=False, purpose=FAMILY):
    """Verify the references and the source depth run, copy the prepared data, reconstruct every selection (it must equal
    the sealed record, the ablation plan and the depth run's) and seal the planned jobs with the source hashes."""
    output = Path(output)
    mapping = td.reference_of(p)
    chosen = list(p['datasets']) if datasets is None else [name for name in p['datasets'] if name in set(datasets)]
    if not chosen:
        raise ValueError('The datasets must be distinct protocol datasets')
    needed = [name for name in p['references'] if any(mapping[d] == name for d in chosen)]
    missing = [name for name in needed if name not in sources]
    if missing:
        raise ValueError(f'Supply --reference for {", ".join(missing)}')
    source = load_source_depth_run(source_directory, p, allow_smoke=allow_smoke)
    source['jobs'] = {job['stem']: job for job in source['collected']['jobs']}
    source_selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                         for s in json.loads((Path(source_directory)/'reference_selections.json').read_text())}
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
            key = (name, split['outer_repeat'], split['outer_fold'])
            if record != reference['ablation']['selections'].get(key) or record != source_selections.get(key):
                raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the selection reconstructed '
                                 'from the reference run differs from the sealed record or the depth run\'s')
            selections.append(record)
            jobs.append(planned_job(p, reference, name, index, split, record, X.shape[1], source))
    write_json(output/'reference_selections.json', selections)
    write_json(output/'planned_jobs.json', jobs)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'family': FAMILY, 'protocol_id': p['protocol_id'], 'protocol_hash': config_id(p),
        'datasets': chosen, 'arms': list(p['arms']), 'model_seed': p['model_seed'], 'planned_jobs': len(jobs),
        'planned_jobs_sha256': sha256_file(output/'planned_jobs.json'),
        'reference_selections_sha256': sha256_file(output/'reference_selections.json'), 'dataset_identity': identities,
        'source_depth_run': {'directory': str(Path(source_directory).resolve()), 'pins': source['observed']},
        'references': {name: {'run_directory': str(Path(sources[name][0]).resolve()),
                              'ablation_directory': str(Path(sources[name][1]).resolve()),
                              'run': reference['observed_run'], 'ablation': reference['ablation']['observed'],
                              'run_file_sha256': reference['run']['files'],
                              'datasets': [d for d in chosen if mapping[d] == name]}
                       for name, reference in references.items()}})
    return jobs, references, source


def verify(output, *, allow_smoke=False):
    """The sealed run directory: protocol (frozen unless a synthetic smoke), manifest seal, unchanged scientific sources,
    planned jobs and sealed selections."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    if not smoke:
        validate_protocol(p)
        if not p['frozen'] or manifest.get('purpose') != FAMILY:
            raise ValueError('A signed-relay depth run directory with a frozen reviewed protocol is required')
    if manifest['protocol_hash'] != config_id(p) or manifest['family'] != FAMILY or manifest['arms'] != list(p['arms']):
        raise ValueError('Protocol seal changed')
    if any(p['source_depth_run'].get(key) != value for key, value in manifest['source_depth_run']['pins'].items()):
        raise ValueError('The source depth run seal changed')
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
        raise FileExistsError(f'Existing signed-relay depth job {stem}')
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    key = (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == key[1:])
    sealed = next(s for s in json.loads((output/'reference_selections.json').read_text())
                  if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == key)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    source = load_source_job(manifest['source_depth_run']['directory'], job)
    result, arrays = evaluate_job(X, y, split, job, sealed, source, p, dataset_hash=data['dataset_hash'],
                                  code_revision=revision, protocol_hash=config_id(p))
    if arrays is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        result['artifact'] = {'path': f'artifacts/{stem}.npz', 'sha256': sha256_file(artifact_path)}
        records = iv.prediction_records(arrays, job, split, y, revision, p['arms'])
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
    """Bind one job record to the sealed plan and recompute its metrics and predictions from the saved arrays and the
    saved per-example predictions."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(record.get('status') == 'ok', f'job status {record.get("status")}: {record.get("exception")}')
    seed, test = int(p['model_seed']), [int(i) for i in split['test']]
    identity = {'family': FAMILY, 'dataset_id': job['dataset_id'], 'reference': job['reference'],
                'dataset_hash': data['dataset_hash'], 'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'],
                'split_hash': config_id(split), 'query': 'outer_test_rows', 'query_ids_hash': array_hash(np.asarray(test)),
                'config_id': job['config_id'], 'config': job['config'], 'selected': job['selected'],
                'own_arm': job['own_arm'], 'model_seed': seed, 'reuse_check': job['reuse_check'], 'source': job['source'],
                'code_revision': revision, 'protocol_hash': config_id(p)}
    require(record['identity'] == identity, 'job identity disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'],
            'the plan disagrees with the sealed selection')
    require(sorted(record['checks']) == sorted(JOB_CHECKS), 'check schedule')
    for name in JOB_CHECKS:
        performed = job['reuse_check'] if name == 'reused_arms_reproduced' else True
        check = record['checks'][name]
        require(check['performed'] is performed and check['passed'] is (True if performed else None), f'check {name}')
    if job['reuse_check']:
        entries = record['checks']['reused_arms_reproduced']['arms']
        require([entry['arm_id'] for entry in entries] == list(REUSED) and record['checks']['reused_arms_reproduced']['outer_rows']
                and all(entry['refit_reproduced'] and entry['plain_state_hashes_equal']
                        and entry['plain_view_predictions_equal'] for entry in entries), 'reuse reproduction')
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
    specs = arm_specs(p)
    for arm in record['arms']:
        name = arm['arm_id']
        planned = next(e for e in job['arms'] if e['arm_id'] == name)
        predicted = arrays[f'{name}__predictions']
        require([sample for sample, _ in cells[name]] == test, 'prediction sample order')
        require(np.array_equal([label for _, label in cells[name]], predicted), 'predictions disagree with the artifact')
        require(arm['params'] == planned['params'] and arm['widths'] == planned['widths'], 'arm parameters')
        require(arm['spec'] == _plain(specs[name]) == planned['spec'], 'arm specification')
        require(arm['reused'] is (name in REUSED), 'reuse flag')
        if name in REUSED:
            require(arm['source'] == {'protocol_id': job['source']['protocol_id'], 'arm_id': REUSED[name], 'stem': stem,
                                      'spec': arm['source']['spec'], 'record_sha256': job['source']['record_sha256'],
                                      'artifact_sha256': job['source']['artifact_sha256']}, 'reused arm source')
        else:
            require(arm['source'] is None, 'a fitted arm has no source')
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
            require(movement is not None and movement['rule'] == 'borda' and movement['frozen_hidden_layers'] == []
                    and movement['instrumentation_removed'] and movement['rng_unchanged'], 'movement record')
            if name not in REUSED:
                require(movement['relay'] == specs[name]['relay'] and movement['intervention_consistent']
                        and movement['lower_vote_scale'] == float(specs[name]['lower_vote_scale'])
                        and len(movement['relay_layers']) == arm['hidden_layers'], 'relay record')
            require(len(movement['layers']) == arm['hidden_layers'], 'movement layer schedule')
            for layer in movement['layers']:
                series = arrays[f'{name}__v{view["view"]}__l{layer["layer"]}__changed']
                require(len(series) == layer['batches'], 'movement series length')
                require(layer['updates'] == 0 or np.isclose(float(np.mean(series)), layer['changed_share'], rtol=0, atol=1e-9),
                        'movement series disagrees with its summary')
    return True


def collect(output, *, allow_smoke=False, rederive=False):
    """Every planned job record verified; with rederive, every sealed selection and planned job re-derived from its
    reference, the source depth run re-verified and every reused arm re-derived from it. Failures never become missing
    evidence."""
    output = Path(output)
    p, manifest, jobs, selections = verify(output, allow_smoke=allow_smoke)
    saved_environment = json.loads((output/'environment.json').read_text())
    references = source = None
    if rederive:
        sources = {name: (entry['run_directory'], entry['ablation_directory']) for name, entry in manifest['references'].items()}
        references = td.load_references(p, sources, allow_smoke=allow_smoke)
        source = load_source_depth_run(manifest['source_depth_run']['directory'], p, allow_smoke=allow_smoke)
        source['jobs'] = {job['stem']: job for job in source['collected']['jobs']}
    prepared = {name: load_prepared(output, name) for name in manifest['datasets']}
    specs = arm_specs(p)
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
                if planned_job(p, reference, job['dataset_id'], index, split, rederived, X.shape[1], source) != job:
                    raise ValueError('the planned job differs from the re-derived plan')
                source_job = load_source_job(manifest['source_depth_run']['directory'], job)
                with np.load(output/record['artifact']['path'], allow_pickle=False) as stored:
                    arrays = {name: stored[name] for name in stored.files}
                for arm in REUSED:
                    expected, expected_arrays = source_arm(source_job['record'], source_job['arrays'], arm, specs[arm], job)
                    if next(entry for entry in record['arms'] if entry['arm_id'] == arm) != td.native(expected):
                        raise ValueError(f'{arm} differs from the source depth record')
                    if any(not np.array_equal(arrays[f'{arm}__{name}'], value) for name, value in expected_arrays.items()) \
                            or sorted(name for name in arrays if name.startswith(f'{arm}__')) != sorted(f'{arm}__{name}' for name in expected_arrays):
                        raise ValueError(f'{arm} arrays differ from the source depth artifact')
            validate_record(record, job, p, data, split, y, saved_environment['code_revision'], output, sealed)
            records[stem] = record
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    if issues:
        raise ValueError(f'Incomplete {FAMILY} evidence: ' + '; '.join(issues))
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


def _blank(value):
    return '' if value is None else value


def job_rows(record, job):
    """The rows one verified job contributes to each table."""
    identity = record['identity']
    key = [FAMILY, identity['dataset_id'], identity['reference'], identity['outer_repeat'], identity['outer_fold'],
           identity['model_seed']]
    selected = identity['selected']
    rows = defaultdict(list)
    for arm in record['arms']:
        row = key + [arm['arm_id']]
        rows['arms.csv'].append(row + [int(arm['arm_id'] == job['own_arm']), int(arm['reused']), arm['spec']['relay'],
                                       arm['spec']['lower_vote_scale'], canonical_json(arm['widths']),
                                       selected['learning_rate'], selected['embed_dim'], selected['degree'],
                                       int(bool(selected['augment']))]
                                + [arm[metric] for metric in METRICS] + [arm['neighborhood_purity'], arm['fit_seconds']])
        majority_by_depth = {entry['depth']: entry['knn_accuracy'] for entry in arm['majority_by_depth']}
        for view in arm['views']:
            for depth in view['depths']:
                rows['depth_probe.csv'].append(row + [view['view'], depth['depth'], depth['hidden_layers'],
                                                      depth['knn_accuracy'], depth['neighborhood_purity'],
                                                      majority_by_depth[depth['depth']]])
            for layer in view['movement']['layers']:
                rows['movement.csv'].append(row + [view['view'], layer['layer'], layer['layer_name'], layer['n_filters'],
                                                   layer['vocabulary'], layer['batches'], layer['updates'],
                                                   layer['changed_share'], layer['mean_displacement'],
                                                   layer['mean_votes'], layer['mean_vote_mass'], layer['max_vote_mass'],
                                                   layer['incoming_to_prior_ratio']]
                                            + [layer.get(column) for column in THRESHOLD_COLUMNS])
            for layer in view['movement'].get('relay_layers', []):
                rows['relay.csv'].append(row + [view['view'], layer['layer'], layer['layer_name']]
                                         + [layer[column] for column in RELAY_COLUMNS])
    if job['reuse_check']:
        for entry in record['checks']['reused_arms_reproduced']['arms']:
            rows['reproduction.csv'].append(key + [entry['arm_id'], entry['source_arm_id'], entry['refit_relay'],
                                                   int(entry['refit_reproduced']), ';'.join(entry['refit_differences']),
                                                   int(entry['plain_state_hashes_equal']),
                                                   int(bool(entry['plain_view_predictions_equal'])),
                                                   entry['lower_cleared_share_of_voted']])
    return rows


def render_tables(collected):
    """{file name: CSV text} over every verified job in planned order."""
    tables = {name: [] for name in TABLES}
    for job in collected['jobs']:
        for name, rows in job_rows(collected['records'][job['stem']], job).items():
            tables[name].extend(rows)
    contents = {}
    for name, header in TABLES.items():
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(header)
        writer.writerows([[_blank(value) for value in row] for row in tables[name]])
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
        'production_family': FAMILY, 'protocol_hash': manifest['protocol_hash'],
        'protocol_sha256': sha256_file(Path(output)/'protocol.json'), 'frozen': p['frozen'],
        'frozen_at_utc': p.get('frozen_at_utc'), 'scale_choice': p.get('scale_choice'),
        'code_revision': collected['code_revision'], 'environment': collected['environment'],
        'references': manifest['references'], 'source_depth_run': manifest['source_depth_run'],
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
    """The five tables and provenance.json of a completed run, all or none."""
    collected = collect(output, allow_smoke=allow_smoke)
    contents = render_tables(collected)
    provenance = provenance_record(output, collected, contents, run_record)
    td.write_all(output, {**contents, PROVENANCE_FILE: json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'})
    return provenance


# ----------------------------------------------------------------------------- summary

def _mean_or_none(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def layer_movement(arms):
    """Per hidden layer over the folds of one arm: the mean changed share and displacement, and for the fitted arms the
    pooled cleared share of the voted filter-batches."""
    layers = sorted({layer['layer'] for arm in arms for layer in arm['movement']['layers']})
    out = []
    for index in layers:
        entries = [layer for arm in arms for layer in arm['movement']['layers'] if layer['layer'] == index]
        voted = sum(layer.get('voted_updates') or 0 for layer in entries)
        cleared = sum(layer.get('cleared_updates') or 0 for layer in entries)
        out.append({'layer': index, 'folds': len(entries),
                    'changed_share': _mean_or_none([layer['changed_share'] for layer in entries]),
                    'mean_displacement': _mean_or_none([layer['mean_displacement'] for layer in entries]),
                    'cleared_share_of_voted': float(cleared / voted) if voted else None,
                    'voted_updates': int(voted) if voted else None})
    return out


def summarize(collected):
    """Per dataset and arm, the outer-fold summaries of every metric and of the purity, the depth probe, the movement by
    layer, the relay counts and the reuse reproduction. Descriptive; the prespecified contrasts belong to analyse."""
    p = collected['protocol']
    rows, folds, seeds = model_rows(collected), fold_schedule(p), [int(p['model_seed'])]
    summaries = {}
    for name in collected['manifest']['datasets']:
        jobs = [job for job in collected['jobs'] if job['dataset_id'] == name]
        records = [collected['records'][job['stem']] for job in jobs]
        table = {}
        for arm in p['arms']:
            entry = {'metrics': {metric: summarize_outer(rows[name], arm, metric, expected_folds=folds, expected_seeds=seeds)
                                 for metric in METRICS}}
            values = [r['neighborhood_purity'] for r in rows[name] if r['model_id'] == arm]
            entry['neighborhood_purity'] = {'mean': float(np.mean(values)),
                                            'outer_fold_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                            'n_folds': len(values)}
            arms = [next(a for a in record['arms'] if a['arm_id'] == arm) for record in records]
            depths = sorted({e['depth'] for a in arms for e in a['majority_by_depth']})
            entry['depth_probe'] = [
                {'depth': depth,
                 'majority_knn_accuracy': float(np.mean([e['knn_accuracy'] for a in arms for e in a['majority_by_depth']
                                                         if e['depth'] == depth])),
                 'view_knn_accuracy': float(np.mean([v['depths'][depth]['knn_accuracy'] for a in arms for v in a['views']
                                                     if depth < len(v['depths'])])),
                 'neighborhood_purity': float(np.mean([v['depths'][depth]['neighborhood_purity'] for a in arms
                                                       for v in a['views'] if depth < len(v['depths'])]))}
                for depth in depths]
            entry['movement'] = {'changed_share': _mean_or_none([a['movement']['changed_share'] for a in arms]),
                                 'mean_displacement': _mean_or_none([a['movement']['mean_displacement'] for a in arms]),
                                 'effectively_inert_folds': int(sum(1 for a in arms if a['movement']['effectively_inert'])),
                                 'by_layer': layer_movement(arms)}
            if arm not in REUSED:
                entry['relay'] = [{'layer': index,
                                   **{key: int(sum(layer[key] for a in arms for layer in a['movement']['relay_layers']
                                                   if layer['layer'] == index))
                                      for key in ('attractions', 'repulsions', 'returned_motions_corrected', 'scaled_votes')}}
                                  for index in sorted({layer['layer'] for a in arms for layer in a['movement']['relay_layers']})]
            table[arm] = entry
        reproduction = [entry for record in records if record['identity']['reuse_check']
                        for entry in record['checks']['reused_arms_reproduced']['arms']]
        selected = defaultdict(int)
        for job in jobs:
            selected[canonical_json(job['selected_widths'])] += 1
        summaries[name] = {'reference': jobs[0]['reference'], 'folds': len(jobs), 'arms': table,
                           'selected_widths': dict(sorted(selected.items())),
                           'reuse_reproduced': {entry['arm_id']: bool(entry['passed']) for entry in reproduction},
                           'printed_refit_lower_cleared_share': next((entry['lower_cleared_share_of_voted'] for entry in reproduction
                                                                      if entry['arm_id'] == 'depth2_printed'), None)}
    return _plain({'purpose': p['purpose'], 'selection_statement': p['selection_statement'],
                   'protocol_id': p['protocol_id'], 'production_family': FAMILY,
                   'protocol_hash': collected['manifest']['protocol_hash'], 'code_revision': collected['code_revision'],
                   'scale_choice': p.get('scale_choice'), 'datasets': collected['manifest']['datasets'],
                   'arms': list(p['arms']), 'planned_jobs': len(collected['jobs']),
                   'check_totals': check_totals(collected['records']),
                   'aggregation': 'one fitting seed per outer fold; outer-fold mean and SD over the fifteen folds',
                   'inferential_significance_claims': False, 'summaries': summaries})


def summary(output, *, allow_smoke=False):
    """Re-verify the run and the source depth run, require the tables and provenance to equal their re-rendering, then
    write the summary."""
    output = Path(output)
    collected = collect(output, allow_smoke=allow_smoke, rederive=True)
    contents = render_tables(collected)
    for name, text in contents.items():
        if not (output/name).is_file() or (output/name).read_text() != text:
            raise ValueError(f'{name} is missing or differs from the verified job records')
    saved = json.loads((output/PROVENANCE_FILE).read_text())
    if saved != provenance_record(output, collected, contents, saved.get('run')):
        raise ValueError(f'{PROVENANCE_FILE} differs from the verified run')
    report = summarize(collected)
    write_json(output/f'{FAMILY}_summary.json', report)
    flat = []
    for name, entry in report['summaries'].items():
        for arm in report['arms']:
            metrics = entry['arms'][arm]['metrics']
            flat.extend([name, arm, metric, metrics[metric]['mean'], metrics[metric]['outer_fold_sd'],
                         metrics[metric]['n_folds']] for metric in METRICS)
            purity = entry['arms'][arm]['neighborhood_purity']
            flat.append([name, arm, 'neighborhood_purity', purity['mean'], purity['outer_fold_sd'], purity['n_folds']])
    write_csv(output/f'{FAMILY}_summary.csv', ('dataset_id', 'arm_id', 'metric', 'mean', 'outer_fold_sd', 'n_folds'), flat)
    return report


# ----------------------------------------------------------------------------- pilot, scale ladder and freeze

def cleared_share(record, layer=0):
    """The pooled cleared share of one fitted arm record's hidden layer (all views)."""
    entry = next((entry for entry in record['movement']['layers'] if entry['layer'] == layer), None)
    return None if entry is None else entry['cleared_share_of_voted']


def ladder_rung(scale, record, seconds):
    lower = next(entry for entry in record['movement']['layers'] if entry['layer'] == 0)
    upper = next(entry for entry in record['movement']['layers'] if entry['layer'] == 1)
    return {'lower_vote_scale': scale, 'cleared_share': lower['cleared_share_of_voted'],
            'voted_updates': lower['voted_updates'], 'cleared_updates': lower['cleared_updates'],
            'mean_threshold_units_voted': lower['mean_threshold_units_voted'],
            'lower_changed_share': lower['changed_share'], 'lower_mean_displacement': lower['mean_displacement'],
            'upper_changed_share': upper['changed_share'], 'upper_mean_displacement': upper['mean_displacement'],
            'upper_cleared_share': upper['cleared_share_of_voted'], 'seconds': seconds}


def scale_ladder(p, job, X, y, split, query, piloted):
    """Every ladder value on one training-only partition: depth2_signed_scaled at that s (read from `piloted` where the
    pilot job already fitted it), its first layer's cleared share and movement. No accuracy is recorded."""
    seed, specs = int(p['model_seed']), arm_specs(p)
    train = [int(i) for i in split['train']]
    X_train, y_train, X_query, y_query = X[train], y[train], X[query], y[query]
    params = next(entry['params'] for entry in job['arms'] if entry['arm_id'] == 'depth2_signed_scaled')
    rungs = []
    for scale in SCALE_LADDER:
        if scale in piloted:
            rungs.append(piloted[scale])
            continue
        spec = {**specs['depth2_signed_scaled'], 'lower_vote_scale': scale}
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            record, _, checks, _, _ = evaluate_arm('depth2_signed_scaled', spec, params, seed, X_train, y_train,
                                                   X_query, y_query)
        if not all(checks.values()):
            raise CheckFailed(f'ladder s = {scale}: checks failed: {sorted(k for k, v in checks.items() if not v)}')
        rungs.append(ladder_rung(scale, record, time.perf_counter() - start))
    return rungs


def choose_scale(ladders):
    """The smallest ladder value whose first-layer cleared share reaches the target on every pilot dataset; if none
    does, the largest."""
    reaching = [scale for scale in SCALE_LADDER
                if all(next(r for r in ladder if r['lower_vote_scale'] == scale)['cleared_share'] is not None
                       and next(r for r in ladder if r['lower_vote_scale'] == scale)['cleared_share'] >= CLEARING_TARGET
                       for ladder in ladders.values())]
    chosen = min(reaching) if reaching else max(SCALE_LADDER)
    return {'lower_vote_scale': int(chosen), 'reached_target': bool(reaching), 'target': CLEARING_TARGET,
            'cleared_share': {name: {str(r['lower_vote_scale']): r['cleared_share'] for r in ladder}
                              for name, ladder in ladders.items()},
            'rule': f'the smallest ladder value whose first-layer cleared share is at least {CLEARING_TARGET} on the '
                    'training-only pilot fit of every pilot dataset; if no ladder value reaches it, the largest; no '
                    'outer-fold row, score or accuracy takes part'}


def calibrated_projection(jobs, records, workers=WORKERS):
    """Per planned job, the reference run's realized outer fit and predict seconds (seed 8129, that fold) times the
    piloted ratio of the job's seconds without the reuse reproduction to the own arm's unmodified-library fit seconds,
    plus on a first outer fold the piloted ratio of the reuse reproduction's seconds; serial hours and the
    first-free-worker makespan."""
    fitted = {r['dataset_id']: r['fitted_ratio'] for r in records}
    reuse = {r['dataset_id']: r['reuse_ratio'] for r in records}
    largest_fitted, largest_reuse = max(fitted.values()), max(reuse.values())
    seconds, datasets = [], {}
    for job in jobs:
        realized = float(job['reference_outer_seconds'])
        ratio = fitted.get(job['dataset_id'], largest_fitted)
        extra = reuse.get(job['dataset_id'], largest_reuse) if job['reuse_check'] else 0.
        cost = realized * (ratio + extra)
        seconds.append(cost)
        entry = datasets.setdefault(job['dataset_id'], {'jobs': 0, 'reference_seconds': 0., 'seconds': 0.,
                                                        'fitted_ratio': ratio,
                                                        'reuse_ratio': reuse.get(job['dataset_id'], largest_reuse),
                                                        'basis': 'piloted ratios' if job['dataset_id'] in fitted
                                                        else 'largest piloted ratios'})
        entry['jobs'] += 1
        entry['reference_seconds'] += realized
        entry['seconds'] += cost
    for entry in datasets.values():
        entry['serial_hours'] = entry['seconds'] / 3600
    serial = sum(seconds)
    return {'datasets': datasets, 'serial_hours': serial / 3600, 'serial_hours_over_workers': serial / 3600 / workers,
            'simulated_makespan_hours': makespan(seconds, workers) / 3600, 'longest_job_hours': max(seconds) / 3600,
            'workers': workers}


def runtime_pilot(output, p, sources, source_directory):
    """Training-only: prepare every protocol dataset (and verify the source depth run), then the first outer fold of each
    pilot dataset with every fourth outer training row as the query rows (the outer test fold is never touched): the
    job, the unmodified-library timing, the scale ladder; the projection and one reproduction probe per reference."""
    output = Path(output)
    started = utc_now()
    jobs, references, _ = prepare(output, p, sources, source_directory, purpose='training_runtime_only')
    mapping, revision = td.reference_of(p), json.loads((output/'environment.json').read_text())['code_revision']
    sealed_records = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                      for s in json.loads((output/'reference_selections.json').read_text())}
    records, ladders = [], {}
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and j['reuse_check'])
        X, y, data, splits = load_prepared(output, name)
        split = splits[0]
        query = [int(i) for i in split['train'][::4]]
        source = load_source_job(source_directory, job)
        record, _ = evaluate_job(X, y, split, job, sealed_records[(name, split['outer_repeat'], split['outer_fold'])],
                                 source, p, dataset_hash=data['dataset_hash'], code_revision=revision,
                                 protocol_hash=config_id(p), query=query)
        if record['status'] != 'ok':
            raise CheckFailed(f'pilot job {name} failed: {record.get("exception")}')
        timing = record['timing']
        arms = {arm['arm_id']: arm for arm in record['arms']}
        own = job['own_arm']
        own_plain = timing[f'{own}_plain_seconds']
        reuse_seconds = sum(timing[f'{arm}_reuse_seconds'] for arm in REUSED)
        fitted_seconds = timing['job_seconds'] - reuse_seconds
        piloted = {1: ladder_rung(1, arms['depth2_signed'], timing['depth2_signed_seconds'])}
        piloted[lower_vote_scale(p)] = ladder_rung(lower_vote_scale(p), arms['depth2_signed_scaled'],
                                                   timing['depth2_signed_scaled_seconds'])
        ladders[name] = scale_ladder(p, job, X, y, split, query, piloted)
        reproduction = record['checks']['reused_arms_reproduced']
        records.append({
            'dataset_id': name, 'reference': mapping[name], 'dataset_hash': data['dataset_hash'],
            'train_ids': split['train'], 'query_ids': query, 'config_id': job['config_id'], 'selected': job['selected'],
            'own_arm': own, 'model_seed': job['model_seed'], 'timing': timing,
            'reference_outer_seconds': job['reference_outer_seconds'],
            'fitted_ratio': fitted_seconds / own_plain, 'reuse_ratio': reuse_seconds / own_plain,
            'checks': {key: {'performed': c['performed'], 'passed': c['passed']} for key, c in record['checks'].items()},
            'reuse_reproduction': [{key: entry[key] for key in ('arm_id', 'source_arm_id', 'refit_relay', 'refit_reproduced',
                                                                'refit_differences', 'plain_state_hashes_equal',
                                                                'lower_cleared_share_of_voted')}
                                   for entry in reproduction['arms']],
            'movement': {arm: {'changed_share': arms[arm]['movement']['changed_share'],
                               'by_layer': [{key: layer.get(key) for key in ('layer', 'changed_share', 'mean_displacement',
                                                                             'cleared_share_of_voted', 'voted_updates')}
                                            for layer in arms[arm]['movement']['layers']]}
                         for arm in FITTED},
            'relay': {arm: arms[arm]['movement']['relay_layers'] for arm in FITTED},
            'scale_ladder': ladders[name], 'status': 'ok'})
    piloted_records = {r['dataset_id']: r for r in records}
    probes = {name: base.reproduction_probe(reference['run'], td.smallest_dataset(reference))
              for name, reference in references.items()}
    calibrated = calibrated_projection(jobs, records, p['workers'])
    decision = calibrated['simulated_makespan_hours']
    checks_passed = all(c['passed'] is True for r in records for key, c in r['checks'].items() if c['performed'])
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'protocol_id': p['protocol_id'],
              'production_family': FAMILY, 'protocol_hash': config_id(p), 'code_revision': revision,
              'started_utc': started, 'ended_utc': utc_now(), 'records': records,
              'calibrated_projection': calibrated, 'reproduction_probes': probes,
              'scale_choice': choose_scale(ladders),
              'decision': {'rule': p['decision_rule'], 'hours': decision, 'cap_hours': p['wallclock_cap_hours'],
                           'workers': p['workers'], 'within_cap': decision <= p['wallclock_cap_hours'],
                           'checks_passed': checks_passed,
                           'reuse_reproduced_on_training_quantities': all(
                               entry['refit_reproduced'] and entry['plain_state_hashes_equal']
                               for r in records for entry in r['reuse_reproduction']),
                           'probes_reproduced': all(probe['reproduced'] for probe in probes.values())},
              'pilot_datasets': list(piloted_records),
              'estimate_limitations': 'one training partition per pilot dataset on the machine as it was; the calibrated '
                                      'projection divides one pilot measurement by another and multiplies the reference '
                                      'run\'s realized per-fold seconds (measured under 16 workers), so machine load '
                                      'largely cancels; the query rows are training rows, so the readout and probe costs '
                                      'follow the outer test size only approximately'}
    write_json(output/'pilot.json', _plain(report))
    return report


def freeze(draft_path, pilot_path, stages_path, output_path, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the training-only pilot of that draft projects within the
    cap at the protocol workers, every pilot check, reuse reproduction and probe held, and the stage record carries a
    passing smoke; the scale chosen on the pilot ladder is sealed. An existing output must be the draft itself."""
    draft = json.loads(Path(draft_path).read_text())
    if draft.get('frozen'):
        raise ValueError('The draft is already frozen')
    validate_protocol(draft)
    pilot, stages = json.loads(Path(pilot_path).read_text()), json.loads(Path(stages_path).read_text())
    if pilot.get('protocol_hash') != config_id(draft) or pilot.get('production_family') != FAMILY:
        raise ValueError('The pilot did not run with this draft')
    decision = pilot['decision']
    if not (decision['within_cap'] and 0 < decision['hours'] <= CAP_HOURS and decision['workers'] == WORKERS
            and decision['probes_reproduced'] and decision['checks_passed']
            and decision['reuse_reproduced_on_training_quantities']):
        raise ValueError(f'Not frozen: projection {decision["hours"]:.2f} h at {decision["workers"]} workers against the '
                         f'{CAP_HOURS} h cap, probes reproduced {decision["probes_reproduced"]}, pilot checks passed '
                         f'{decision["checks_passed"]}, reuse reproduced {decision["reuse_reproduced_on_training_quantities"]}')
    if (stages.get('smoke') or {}).get('status') != 'ok' or not stages.get('summary'):
        raise ValueError('The stage record lacks a passing synthetic smoke and its summary')
    ladders = {r['dataset_id']: r['scale_ladder'] for r in pilot['records']}
    if sorted(ladders) != sorted(draft['pilot_datasets']) or choose_scale(ladders) != pilot['scale_choice']:
        raise ValueError('The pilot\'s scale choice does not follow from its ladders')
    choice = {**pilot['scale_choice'], 'ladders': ladders, 'pilot_sha256': sha256_file(pilot_path)}
    calibrated = pilot['calibrated_projection']
    record = {'cap_hours': CAP_HOURS, 'workers': WORKERS, 'decision_hours': decision['hours'], 'decision_rule': decision['rule'],
              'calibrated': {key: calibrated[key] for key in ('serial_hours', 'serial_hours_over_workers',
                                                              'simulated_makespan_hours', 'longest_job_hours')},
              'calibrated_per_dataset_hours': {name: entry['serial_hours'] for name, entry in calibrated['datasets'].items()},
              'pilot_ratios': {r['dataset_id']: {'fitted': r['fitted_ratio'], 'reuse': r['reuse_ratio']} for r in pilot['records']},
              'pilot_seconds': {r['dataset_id']: r['timing'] for r in pilot['records']},
              'pilot_movement': {r['dataset_id']: r['movement'] for r in pilot['records']},
              'reuse_reproduction': {r['dataset_id']: r['reuse_reproduction'] for r in pilot['records']},
              'reproduction_probes': {name: {key: probe[key] for key in ('dataset_id', 'result_file', 'config_id',
                                                                         'model_seed', 'inner_fold', 'reference_score',
                                                                         'refit_score', 'readout_selections_identical',
                                                                         'reproduced')}
                                      for name, probe in pilot['reproduction_probes'].items()},
              'pilot_code_revision': pilot['code_revision'], 'pilot_sha256': sha256_file(pilot_path), 'stages': stages}
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    ratios = ', '.join(f"{name} {value['fitted']:.2f} (+{value['reuse']:.2f} on a first fold)"
                       for name, value in sorted(record['pilot_ratios'].items()))
    shares = '; '.join(f"{name}: " + ', '.join(f"s={r['lower_vote_scale']} {r['cleared_share']:.3f}" for r in ladder)
                       for name, ladder in sorted(ladders.items()))
    text = (f"Review response to the simulated referee panel of 2026-09-23 (V1-A, the relay sign flaw; author decision of "
            f"2026-09-23): {stages['summary']}; projected at {WORKERS} single-thread workers: calibrated simulated makespan "
            f"{decision['hours']:.2f} h (serial {calibrated['serial_hours']:.2f} h, serial over workers "
            f"{calibrated['serial_hours_over_workers']:.2f} h); cap {CAP_HOURS} h. Piloted time ratios to the own arm's "
            f"unmodified-library fit: {ratios}; the rule prices every unpiloted dataset at the largest of them. Vote scale "
            f"s = {choice['lower_vote_scale']} chosen on the training-only pilot ladder (first-layer cleared share of the "
            f"voted filter-batches: {shares}; target {CLEARING_TARGET} on every pilot dataset"
            + ('' if choice['reached_target'] else ', not reached by any ladder value, so the largest') + '). '
            f"The reused arms reproduced the stored training quantities on the pilot folds; reproduction probes held on "
            f"{', '.join(probe['dataset_id'] for probe in pilot['reproduction_probes'].values())}; no outer score of a "
            f"corrected arm exists; frozen after the pilot")
    protocol = validate_protocol(_plain(dict(draft, frozen=True, frozen_at_utc=frozen_at, status=FROZEN_STATUS,
                                             resource_decision=text, pilot_projection=record, scale_choice=choice)))
    output_path = Path(output_path)
    content = json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if output_path.exists() and json.loads(output_path.read_text()) not in (draft, protocol):
        raise FileExistsError(f'Refusing to replace {output_path}: it is neither the draft nor this frozen protocol')
    temporary = output_path.with_name(output_path.name + '.freezing')
    temporary.write_text(content)
    os.replace(temporary, output_path)
    return protocol


# ----------------------------------------------------------------------------- synthetic smoke (never evidence)

def smoke_protocol(p, sources, source_directory):
    """The synthetic smoke form of a protocol: the synthetic reference, its ablation and the synthetic depth run."""
    depth = iv.smoke_protocol(iv.draft_protocol('depth'), sources)
    return _plain(dict(p, references=depth['references'], datasets=depth['datasets'], pilot_datasets=depth['pilot_datasets'],
                       frozen=False, status='synthetic_smoke_only', purpose='synthetic smoke only; never evidence',
                       arm_widths=dict(iv.SMOKE_WIDTHS), source_depth_run=source_pins(source_directory),
                       **{key: depth[key] for key in base.DESIGN_KEYS}))


def smoke(output, p, workers=3):
    """A synthetic depth run (interventions.smoke: a synthetic ArrowFlow-kNN reference at both smoke depths, its
    component ablation and a complete depth family run), then a complete run of this family over it, its tables, its
    summary and its analysis. Never evidence."""
    output = Path(output)
    iv.smoke(output/'depth', iv.draft_protocol('depth'), workers)
    sources = {'smoke_reference': (output/'depth'/'synthetic_reference', output/'depth'/'synthetic_ablation')}
    source_directory = output/'depth'/'run'
    run_directory = output/'run'
    prepare(run_directory, smoke_protocol(p, sources, source_directory), sources, source_directory, allow_smoke=True,
            purpose='synthetic_smoke_only')
    run(run_directory, workers, allow_smoke=True)
    write_tables(run_directory, allow_smoke=True, run_record={'purpose': 'synthetic_smoke_only'})
    report = summary(run_directory, allow_smoke=True)
    analyse(run_directory, output/'analysis', allow_smoke=True)
    return report


# ----------------------------------------------------------------------------- the prespecified analysis

ANALYSIS_SOURCES = ('signed_relay_depth.py', 'signed_relay.py', 'interventions.py', 'update_rules.py',
                    'training_diagnostics.py', 'run_knn_ablation.py', 'evaluation.py', 'knn_controls.py')
RUN_FILES = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json',
             PROVENANCE_FILE)


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
    if family != FAMILY:
        raise AnalysisRefused(f'{path} does not hold a {FAMILY} protocol')
    missing = [name for name in RUN_FILES + tuple(TABLES) + (f'{FAMILY}_summary.json', f'{FAMILY}_summary.csv')
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


def contrast_family(rows, p, datasets, widths_by_fold, role, arm_a, arm_b):
    """arm_a minus arm_b accuracy per dataset, Holm-adjusted across the datasets; then the named subsets (descriptive)."""
    folds, seeds = fold_schedule(p), [int(p['model_seed'])]
    q, confidence = p['test_train_ratio'], p['confidence']
    depths = list(p['arm_widths'].values())
    label = f'{arm_a}_minus_{arm_b}'
    entries = []
    for name in datasets:
        interval = paired_corrected_interval(rows[name], arm_a, arm_b, metric='accuracy', q=q, confidence=confidence,
                                             expected_folds=folds, expected_seeds={arm_a: seeds, arm_b: seeds})
        entries.append({'family': FAMILY, 'role': role, 'contrast': label, 'subset': 'all_folds', 'dataset_id': name,
                        'arm_a': arm_a, 'arm_b': arm_b, 'metric': 'accuracy', 'sd': None,
                        **{key: interval[key] for key in ('mean_difference', 'standard_error', 'ci_low', 'ci_high',
                                                          'n_folds', 'df', 'p_approximate')}})
    for entry, value in zip(entries, holm_adjust([entry['p_approximate'] for entry in entries])):
        entry['holm_p_approximate'] = value
        entry['significant_after_holm'] = bool(value < p['analysis']['alpha'])
    subsets = []
    for name in datasets:
        for entry in depth_split(rows[name], arm_a, arm_b, widths_by_fold[name], depths=depths, folds=folds,
                                 seeds={arm_a: seeds, arm_b: seeds}, q=q, confidence=confidence):
            interval = (entry.get('interval') or {}) if entry['n_folds'] >= iv.MIN_SUBSET_FOLDS else {}
            subsets.append({'family': FAMILY, 'role': role, 'contrast': label,
                            'subset': 'own_selection_' + canonical_json(entry['widths']), 'dataset_id': name,
                            'arm_a': arm_a, 'arm_b': arm_b, 'metric': 'accuracy',
                            'mean_difference': entry['mean_difference'], 'standard_error': interval.get('standard_error'),
                            'ci_low': interval.get('ci_low'), 'ci_high': interval.get('ci_high'),
                            'n_folds': entry['n_folds'], 'df': max(entry['n_folds'] - 1, 0), 'sd': entry['sd'],
                            'p_approximate': None, 'holm_p_approximate': None, 'significant_after_holm': None})
    return entries, subsets


def contrast_rows(rows, p, datasets, widths_by_fold):
    """The prespecified families: depth1 minus each corrected arm (primary) and depth2_signed minus depth2_printed
    (secondary), each Holm-adjusted across the datasets; then the named subsets."""
    contrasts, subsets = [], []
    families = [('primary', p['analysis']['primary']['reference_arm'], arm) for arm in p['analysis']['primary']['variant_arms']]
    families.append(('secondary', p['analysis']['secondary']['arm_a'], p['analysis']['secondary']['arm_b']))
    for role, arm_a, arm_b in families:
        entries, named = contrast_family(rows, p, datasets, widths_by_fold, role, arm_a, arm_b)
        contrasts.extend(entries)
        subsets.extend(named)
    return contrasts, subsets


def interpretation_outcome(contrasts, p):
    """The interpretation rule of the protocol applied to the primary contrasts."""
    reference, decisions = p['analysis']['primary']['reference_arm'], {}
    for arm in p['analysis']['primary']['variant_arms']:
        rows = [row for row in contrasts if row['contrast'] == f'{reference}_minus_{arm}' and row['subset'] == 'all_folds']
        gains = sorted(row['dataset_id'] for row in rows if row['significant_after_holm'] and row['mean_difference'] < 0)
        higher = sorted(row['dataset_id'] for row in rows if row['mean_difference'] < 0)
        losses = sorted(row['dataset_id'] for row in rows if row['significant_after_holm'] and row['mean_difference'] > 0)
        decisions[arm] = {'datasets': len(rows), 'holm_significant_gains': gains, 'higher_mean_datasets': higher,
                          'n_higher_mean': len(higher), 'most_threshold': MOST_DATASETS,
                          'holm_significant_losses': losses, 'helps': bool(gains) and len(higher) >= MOST_DATASETS}
    helps = any(entry['helps'] for entry in decisions.values())
    statement = p['analysis']['interpretation']['helps' if helps else 'does_not_help']
    return {'rule': p['analysis']['interpretation']['rule'], 'arms': decisions,
            'outcome': 'helps' if helps else 'does_not_help', 'statement': statement}


def analyse(run_directory, output, *, allow_smoke=False):
    """Refuse until the run is complete, re-verify every record and the source depth run, then compute the prespecified
    contrasts and the interpretation."""
    run_directory, output = Path(run_directory), Path(output)
    completeness_gate(run_directory)
    names = (f'{FAMILY}_contrasts.csv', f'{FAMILY}_analysis.json')
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
    if json.loads((run_directory/f'{FAMILY}_summary.json').read_text()) != report:
        raise AnalysisRefused(f'{FAMILY}_summary.json differs from the verified run')
    rows = model_rows(collected)
    datasets = list(collected['manifest']['datasets'])
    widths_by_fold = {name: {(job['outer_repeat'], job['outer_fold']): job['selected_widths']
                             for job in collected['jobs'] if job['dataset_id'] == name} for name in datasets}
    contrasts, subsets = contrast_rows(rows, p, datasets, widths_by_fold)
    outcome = interpretation_outcome(contrasts, p)
    movement = {arm: {name: report['summaries'][name]['arms'][arm]['movement'] for name in datasets} for arm in p['arms']}
    from .referee_analyses import code_record
    record = _plain({
        'purpose': f'prespecified_analysis_of_the_{FAMILY}_family',
        'status': 'computed after the run was complete and re-verified (every job record, every sealed selection '
                  're-derived from its reference run, the source depth run re-verified and every reused arm re-derived '
                  'from it, every table re-rendered); two primary families and one secondary family, each Holm-adjusted '
                  'across the seventeen datasets; everything else descriptive',
        'protocol_id': p['protocol_id'], 'production_family': FAMILY, 'protocol_hash': collected['manifest']['protocol_hash'],
        'code_revision': collected['code_revision'], 'analysis_declaration': p['analysis'],
        'scale_choice': p.get('scale_choice'), 'datasets': datasets, 'arms': list(p['arms']),
        'contrasts': contrasts, 'named_subsets': subsets, 'interpretation': outcome, 'movement': movement,
        'limits': p['analysis']['limits'], 'summaries': report['summaries'], 'check_totals': report['check_totals'],
        'provenance': {'run_directory': str(run_directory.resolve()), 'run': saved_provenance['run'],
                       'references': collected['manifest']['references'],
                       'source_depth_run': collected['manifest']['source_depth_run'],
                       'jobs_verified': len(collected['jobs']), 'tables': saved_provenance['tables'],
                       **code_record(ANALYSIS_SOURCES)}})
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(CONTRAST_COLUMNS)
    writer.writerows([[_blank(row[column]) for column in CONTRAST_COLUMNS] for row in contrasts + subsets])
    write_outputs(output, {names[0]: buffer.getvalue(),
                           names[1]: json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n'})
    return record


# ----------------------------------------------------------------------------- command

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['draft', 'prepare', 'smoke', 'pilot', 'freeze', 'run', 'summary', 'analyse'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--reference', type=td.parse_reference, action='append', metavar='NAME=RUN_DIR,ABLATION_DIR')
    parser.add_argument('--source-depth-run', type=Path, default=DEFAULT_SOURCE_DEPTH_RUN)
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--draft', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--stages', type=Path)
    parser.add_argument('--run', dest='run_directory', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        write_json(args.output, draft_protocol())
        return
    if args.command == 'analyse':
        if args.run_directory is None:
            parser.error('analyse needs --run')
        try:
            record = analyse(args.run_directory, args.output)
        except (AnalysisRefused, FileExistsError, ValueError) as exc:
            parser.exit(2, f'signed_relay_depth analyse refused: {exc}\n')
        for row in record['contrasts']:
            print(f"{row['role']} {row['contrast']} {row['dataset_id']}: {row['mean_difference']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
                  f"Holm p={row['holm_p_approximate']:.3g}")
        print(f"interpretation: {record['interpretation']['outcome']}: {record['interpretation']['statement']}")
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
        print(f"frozen at {protocol['frozen_at_utc']}: decision {protocol['pilot_projection']['decision_hours']:.2f} h, "
              f"vote scale {protocol['scale_choice']['lower_vote_scale']}")
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
        jobs, _, _ = prepare(args.output, p, sources, args.source_depth_run, args.dataset)
        print(f'{len(jobs)} planned jobs')
        return
    if args.command == 'pilot':
        with execution_lock():
            report = runtime_pilot(args.output, p, sources, args.source_depth_run)
        print(json.dumps({'decision': report['decision'], 'scale_choice': report['scale_choice'],
                          'pilot_ratios': {r['dataset_id']: {'fitted': r['fitted_ratio'], 'reuse': r['reuse_ratio']}
                                           for r in report['records']},
                          'calibrated_serial_hours': report['calibrated_projection']['serial_hours']}, indent=2))
        return
    record = run(args.output, args.workers, protocol_path=args.protocol)
    provenance = write_tables(args.output, run_record=record)
    print(json.dumps(provenance['check_totals'], indent=2))


if __name__ == '__main__':
    main()
