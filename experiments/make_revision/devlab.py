"""Inner-fold laboratory on the bridge-selected single view: readouts (v3 Task 7, B1) and permutation LVQ (Task 8, B2).

readouts --bridge-source B --output O [--dataset ...] [--workers 16] [--skip-pilot-check]   prepare, run, summarize (B1)
permlvq  --bridge-source B --output O [--dataset ...] [--workers 16] [--skip-pilot-check]   prepare, run, summarize (B2)
pilot    --family F --bridge-source B --output O [--dataset ...] [--workers 16]             one cell per dataset, timed in this process
summary  --family F --output O                                                              <family>_summary.json from the saved records
smoke    --family F --output O [--workers 2]                                                synthetic end-to-end exercise; never evidence

The laboratory design (datasets, outer repeat, selection splits, grids, variants, verdict rule, caps) is read from
protocols/2026-09-12/devlab.json (`--protocol`) and copied into the laboratory root as protocol.json. The readouts
family lives in O itself; the permlvq family lives in O/permlvq (its own manifest, environment, plan and records)
so that both can share one output directory, and both summaries are written beside each other in O
(readouts_summary.json / permlvq_summary.json, with the *_rows.csv). `--family` is required for pilot, summary and
smoke; a laboratory command names its family itself (a `--family` given with it must agree).

Pilot gate: a family command with more than one worker needs pilot.json in its laboratory root, written by `pilot`
for the same family, covering every requested dataset with a successful cell, made from the same bridge source
(protocol hash and file SHA-256s) and the same laboratory protocol (hash), and whose conservative projection for the
requested worker count is within the family's cap: max(planned cells x mean cell seconds, sum over datasets of
planned cells x cell seconds) / workers x CONTENTION_FACTOR (1.5: start-up, tails, a shared machine). The gate
recomputes the projection from the pilot's cells; the run manifest records the authorising pilot's SHA-256 and
projection, or the explicit --skip-pilot-check escape. `pilot` overwrites pilot.json (the latest) and keeps every
pilot under pilots/pilot_<utc>.json.

One unit is one (dataset, outer fold of repeat 0, inner fold, fitting seed). The bridge-selected configuration of
that outer fold (run_bridge.selection_record, resolved by run_bridge.planned_job) is fitted with n_views=1 (view 0
of the diverse cycle: target-aware encoding) on the inner training partition with the selected validation_ratio.
The reference is the network's own output rule on the inner validation fold. Readouts act on the hidden
representation (ArrowFlowEstimator.transform_orders, the inverse positions of the final hidden ranking); the
permutation LVQ variants act on the view-0 encodings themselves (the LVQ stack replaces the network on the same
encodings, seeds and budget: iterations, batch size and learning rate of the selected configuration). Hyperparameters
are chosen one level deeper, on stratified splits of the inner training partition; the scored inner validation fold
never enters a selection, and no outer test fold is touched. Mean inner accuracies feed adoption_verdict; nothing
here is outer-fold evidence.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from arrowflow.permlvq import PermutationLVQClassifier
from arrowflow.ranking import inverse_positions
from arrowflow.readouts import KPrototypeBorda
from .comparisons import BordaClassifier, StableFootruleKNN, derive_seed
from .evaluation import candidate_grid, canonical_json, config_id, validate_split
from .matched import select_symmetric_probe
from .models import array_hash, seed_fit
from .multiview import MultiViewArrowFlow
from .run_bridge import BRIDGE_MODEL, load_bridge, planned_job, selection_record, synthetic_bridge_source, write_csv
from .run_revision import environment_record, execution_lock, load_prepared, write_json

SOURCE_MODULES = ['arrowflow.readouts', 'arrowflow.permlvq', 'experiments.make_revision.bridge',
                  'experiments.make_revision.multiview', 'experiments.make_revision.models',
                  'experiments.make_revision.comparisons', 'experiments.make_revision.evaluation',
                  'experiments.make_revision.matched', 'experiments.make_revision.reporting',
                  'experiments.make_revision.run_bridge', 'experiments.make_revision.run_revision',
                  'experiments.make_revision.secondary_studies']
PROTOCOL_FILE = Path(__file__).with_name('protocols')/'2026-09-12'/'devlab.json'
FAMILY_NAMES = ('readouts', 'permlvq')
REFERENCE = 'output_rule'
COLUMNS = ('dataset_id', 'outer_repeat', 'outer_fold', 'inner_fold', 'model_seed', 'readout_id', 'accuracy')
ENCODER_KEYS = ('embed_dim', 'degree')
NETWORK_KEYS = ('widths', 'learning_rate', 'iterations', 'batch_size', 'validation_ratio', 'augment')
PURPOSE = 'inner_fold_laboratory'
SMOKE_PURPOSE = 'synthetic_smoke_only'
CONTENTION_FACTOR = 1.5      # conservative wall-clock projection = serial hours / workers x this (start-up, tails, shared machine)
VARIANT_SPECS = {          # prior_weight: a number, or 'learning_rate' for the unit's selected learning rate
    'lvq1_borda_plurality': {'depth': 1, 'aggregation': 'borda', 'readout': 'plurality', 'prior_weight': 1.0},
    'lvq1_median_plurality': {'depth': 1, 'aggregation': 'footrule_median', 'readout': 'plurality', 'prior_weight': 1.0},
    'lvq2_borda_plurality': {'depth': 2, 'aggregation': 'borda', 'readout': 'plurality', 'prior_weight': 1.0},
    'lvq1_borda_nearest': {'depth': 1, 'aggregation': 'borda', 'readout': 'nearest', 'prior_weight': 1.0},
    'lvq1_borda_borda': {'depth': 1, 'aggregation': 'borda', 'readout': 'borda', 'prior_weight': 1.0},
    'lvq1_median_priorlr_plurality': {'depth': 1, 'aggregation': 'footrule_median', 'readout': 'plurality',
                                      'prior_weight': 'learning_rate'}}
K_READOUTS = ('plurality', 'borda')          # readouts whose size k is selected jointly with prototypes_per_class


# ----------------------------------------------------------------------------- protocol

def load_protocol(path=PROTOCOL_FILE):
    """The laboratory design file, checked for the keys the code reads."""
    protocol = json.loads(Path(path).read_text())
    required = {'protocol_id', 'datasets', 'outer_repeat', 'inner_folds', 'fit_seeds', 'selection', 'verdict_rule',
                'max_workers', 'families', 'bridge_source'}
    if not required <= set(protocol) or set(protocol['families']) != set(FAMILY_NAMES):
        raise ValueError(f'The laboratory protocol needs {sorted(required)} and the families {FAMILY_NAMES}')
    readouts, permlvq = protocol['families']['readouts'], protocol['families']['permlvq']
    if readouts['readouts'][0] != REFERENCE or permlvq['variants'][0] != REFERENCE:
        raise ValueError(f'Both families list the reference {REFERENCE} first')
    if set(permlvq['variants'][1:]) != set(VARIANT_SPECS):
        raise ValueError(f'permlvq variants must be exactly {sorted(VARIANT_SPECS)}')
    if not {'folds', 'seed'} <= set(protocol['selection']) or not {'minimum_improved', 'tolerance_pp'} <= set(protocol['verdict_rule']):
        raise ValueError('The selection needs folds and seed; the verdict rule needs minimum_improved and tolerance_pp')
    return protocol


DEFAULT_PROTOCOL = load_protocol()
READOUTS = tuple(DEFAULT_PROTOCOL['families']['readouts']['readouts'])
PERMLVQ_VARIANTS = tuple(DEFAULT_PROTOCOL['families']['permlvq']['variants'])
KNN_GRID = DEFAULT_PROTOCOL['families']['readouts']['knn_grid']
KPROTO_GRID = DEFAULT_PROTOCOL['families']['readouts']['kproto_grid']
PERMLVQ_GRID = DEFAULT_PROTOCOL['families']['permlvq']['prototype_grid']
OUTER_REPEAT = DEFAULT_PROTOCOL['outer_repeat']
SELECTION_FOLDS = DEFAULT_PROTOCOL['selection']['folds']
SELECTION_SEED = DEFAULT_PROTOCOL['selection']['seed']
VERDICT_RULE = dict(DEFAULT_PROTOCOL['verdict_rule'])
MAX_WORKERS = DEFAULT_PROTOCOL['max_workers']
WALLCLOCK_CAP_HOURS = {name: spec['wallclock_cap_hours'] for name, spec in DEFAULT_PROTOCOL['families'].items()}


def family_spec(family, protocol=None):
    if family not in FAMILY_NAMES:
        raise ValueError(f'Unknown laboratory family {family!r}')
    return (protocol or DEFAULT_PROTOCOL)['families'][family]


def family_readouts(family, protocol=None):
    spec = family_spec(family, protocol)
    return tuple(spec['readouts'] if family == 'readouts' else spec['variants'])


def environment():
    return environment_record(__package__ + '.devlab:environment')


def full_grid(grid):
    return candidate_grid(grid, budget=int(np.prod([len(v) for v in grid.values()])))


def knn_candidates(protocol=None):
    return full_grid(family_spec('readouts', protocol)['knn_grid'])


def kproto_candidates(protocol=None):
    return full_grid(family_spec('readouts', protocol)['kproto_grid'])


def permlvq_candidates(protocol=None, variant=None):
    """prototypes_per_class grid; for a plurality or borda variant the readout size k is gridded jointly."""
    spec = family_spec('permlvq', protocol)
    grid = dict(spec['prototype_grid'])
    if variant is not None and VARIANT_SPECS[variant]['readout'] in K_READOUTS:
        grid['k'] = list(spec['readout_k_grid'])
    return full_grid(grid)


def candidate_sets(family, protocol=None):
    """Selected hyperparameter grids per readout or variant of a family (the reference and borda1 have none)."""
    if family == 'readouts':
        return {'knn_hidden': knn_candidates(protocol), 'kproto_borda': kproto_candidates(protocol)}
    return {variant: permlvq_candidates(protocol, variant) for variant in family_readouts('permlvq', protocol)[1:]}


def laboratory_paths(output, family):
    """(laboratory root, summary destination): readouts live in the output directory, permlvq in its subdirectory."""
    output = Path(output)
    if family not in FAMILY_NAMES:
        raise ValueError(f'Unknown laboratory family {family!r}')
    return (output if family == 'readouts' else output/'permlvq'), output


def comparison_path(output, family):
    """The readouts summary a permlvq summary compares against (knn_hidden); None for the readouts family."""
    return Path(output)/'readouts_summary.json' if family == 'permlvq' else None


# ----------------------------------------------------------------------------- verdict

def adoption_verdict(reference, candidate, *, minimum_improved=5, tolerance_pp=1.0, minimum_datasets=5):
    """'adopt' when at least `minimum_improved` datasets improve and no dataset worsens by more than
    `tolerance_pp` percentage points (inputs are mean inner accuracies in percent); 'not_applicable' when fewer
    than `minimum_datasets` datasets are scored; otherwise 'reject'."""
    if not reference or set(reference) != set(candidate):
        raise ValueError('Reference and candidate must score the same nonempty dataset panel')
    changes = {name: float(candidate[name]) - float(reference[name]) for name in reference}
    if not all(np.isfinite(list(changes.values()))):
        raise ValueError('Verdict inputs must be finite')
    if len(changes) < minimum_datasets:
        return 'not_applicable'
    improved = sum(change > 0 for change in changes.values())
    return 'adopt' if improved >= minimum_improved and min(changes.values()) >= -tolerance_pp else 'reject'


def verdict_record(reference, candidate, *, rule=VERDICT_RULE):
    changes = {name: float(candidate[name]) - float(reference[name]) for name in reference}
    return {'verdict': adoption_verdict(reference, candidate, **rule),
            'improved': int(sum(c > 0 for c in changes.values())), 'worsened': int(sum(c < 0 for c in changes.values())),
            'datasets': len(changes), 'mean_change_pp': float(np.mean(list(changes.values()))),
            'worst_change_pp': float(min(changes.values())), 'change_pp': changes}


# ----------------------------------------------------------------------------- configuration

def laboratory_configuration(selected):
    """Encoder settings and network parameters of a resolved bridge selection (run_bridge.planned_job['selected'])."""
    missing = sorted(set(ENCODER_KEYS + NETWORK_KEYS) - set(selected))
    if missing:
        raise ValueError(f'Resolve the selected configuration first; missing {missing}')
    return {'encoder_settings': {k: selected[k] for k in ENCODER_KEYS},
            'network_params': {k: selected[k] for k in NETWORK_KEYS}}


def laboratory_params(encoder_settings, network_params):
    """MultiViewArrowFlow keywords of the laboratory model: one view, view 0 of the diverse cycle (target-aware)."""
    params = {'n_views': 1, 'strategy': 'diverse', **encoder_settings, **network_params, 'aggregation': 'majority'}
    unknown = sorted(set(params) - set(MultiViewArrowFlow().get_params()))
    if unknown:
        raise ValueError(f'Unknown laboratory parameters {unknown}')
    return params


def lvq_budget(network_params, protocol=None):
    """PermutationLVQ training budget of a unit: the selected configuration's iterations, batch size and learning
    rate; p_correct and repulsion from the protocol. prototypes_per_class and the readout k are selected."""
    spec = family_spec('permlvq', protocol)
    return {'iterations': int(network_params['iterations']), 'batch_size': int(network_params['batch_size']),
            'learning_rate': float(network_params['learning_rate']), 'p_correct': float(spec['p_correct']),
            'repulsion': bool(spec['repulsion'])}


def variant_model(variant_id, config, budget, seed):
    """The classifier of a variant: config carries prototypes_per_class and, for plurality/borda readouts, k
    (a nearest readout ignores k); the prior weight is the spec's number or the unit's learning rate."""
    spec = VARIANT_SPECS[variant_id]
    layers = tuple({'prototypes_per_class': int(config['prototypes_per_class'])} for _ in range(spec['depth']))
    prior = budget['learning_rate'] if spec['prior_weight'] == 'learning_rate' else float(spec['prior_weight'])
    return PermutationLVQClassifier(layers=layers, readout=spec['readout'], k=int(config.get('k', 1)),
                                    iterations=budget['iterations'], batch_size=budget['batch_size'],
                                    learning_rate=budget['learning_rate'], p_correct=budget['p_correct'],
                                    repulsion=budget['repulsion'], aggregation=spec['aggregation'], seed=seed, prior_weight=prior)


def fit_key(config):
    """Candidates that differ only in the readout size k share one fit (the fit does not depend on k)."""
    return config_id({key: value for key, value in config.items() if key != 'k'})


def unit_stem(dataset_id, outer_repeat, outer_fold, inner_fold, seed):
    return f'{dataset_id}__r{outer_repeat}f{outer_fold}i{inner_fold}_s{seed}'


# ----------------------------------------------------------------------------- readouts and variants on one representation

def selection_splits(train_ids, y_train, *, dataset_id, outer_repeat, outer_fold, inner_fold, folds=SELECTION_FOLDS,
                     seed=SELECTION_SEED):
    """Stratified splits of the inner training partition (global row ids) for hyperparameter selection;
    at most `folds` folds, fewer when the smallest class has fewer rows. The inner validation fold is not involved."""
    train_ids, y_train = np.asarray(train_ids), np.asarray(y_train)
    folds = min(int(folds), int(np.unique(y_train, return_counts=True)[1].min()))
    if folds < 2:
        raise ValueError('Readout selection needs at least two rows per class in the inner training partition')
    seed = derive_seed(seed, dataset_id, int(outer_repeat), int(outer_fold), int(inner_fold))
    splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return [{'fit_rows': train_ids[a].tolist(), 'query_rows': train_ids[b].tolist()}
            for a, b in splitter.split(np.zeros(len(y_train)), y_train)]


def readout_predictions(readout_id, hidden_train, y_train, hidden_query, *, sample_ids, seed, config):
    if readout_id == 'knn_hidden':
        return StableFootruleKNN(**config, input_kind='positions').fit(
            hidden_train, y_train, sample_ids=sample_ids).predict(hidden_query)
    if readout_id == 'borda1':
        # BordaClassifier consumes orders; the hidden orders are the inverse of the hidden positions.
        return BordaClassifier().fit(inverse_positions(hidden_train), y_train).predict(inverse_positions(hidden_query))
    if readout_id == 'kproto_borda':
        return KPrototypeBorda(k=config['k'], seed=seed).fit(hidden_train, y_train).predict(hidden_query)
    raise ValueError(f'Unknown readout {readout_id!r}')


def variant_predictions(variant_id, positions_train, y_train, positions_query, *, sample_ids, seed, config, budget):
    """A permutation LVQ variant fitted on encoding positions with the unit's seed and budget."""
    if variant_id not in VARIANT_SPECS:
        raise ValueError(f'Unknown variant {variant_id!r}')
    return variant_model(variant_id, config, budget, seed).fit(positions_train, y_train).predict(positions_query)


def selection_rows(readout_id, hidden_train, y_train, train_ids, parts, *, seed, candidates, predictions=readout_predictions,
                   budget=None):
    """Every candidate scored on every selection split of the inner training partition (state 'hidden'). Permutation
    LVQ variants (budget given) are fitted once per prototypes_per_class on a split and read out at every k."""
    hidden_train, y_train, train_ids = np.asarray(hidden_train), np.asarray(y_train), np.asarray(train_ids)
    index = {int(row): i for i, row in enumerate(train_ids)}
    rows = []
    for fold, part in enumerate(parts):
        a = np.asarray([index[row] for row in part['fit_rows']])
        b = np.asarray([index[row] for row in part['query_rows']])

        def row(config, **extra):
            return dict(config=config, config_id=config_id(config), selection_fold=fold, model_seed=seed,
                        state='hidden', n_query=len(b), **extra)

        def failure(exc):
            return dict(status='failed', score=None, exception=f'{type(exc).__name__}: {exc}')
        if readout_id in VARIANT_SPECS and budget is not None:
            fitted = {}
            for config in candidates:
                try:
                    key = fit_key(config)
                    if key not in fitted:
                        fitted[key] = variant_model(readout_id, config, budget, seed).fit(hidden_train[a], y_train[a])
                    clf = fitted[key]
                    if 'k' in config:
                        clf.set_params(k=int(config['k']))
                    pred = clf.predict(hidden_train[b])
                    rows.append(row(config, status='ok', score=float(np.mean(np.asarray(pred) == y_train[b]))))
                except Exception as exc:
                    rows.append(row(config, **failure(exc)))
            continue
        if readout_id == 'knn_hidden':
            try:      # one neighbour cache per split at the largest k; every candidate votes on it
                base = StableFootruleKNN(n_neighbors=max(c['n_neighbors'] for c in candidates), input_kind='positions').fit(
                    hidden_train[a], y_train[a], sample_ids=train_ids[a])
                distances, indices = base.kneighbors(hidden_train[b])
            except Exception as exc:
                rows.extend(row(c, **failure(exc)) for c in candidates)
                continue
            for config in candidates:
                try:
                    base.n_neighbors, base.weights = config['n_neighbors'], config['weights']
                    pred = base.predict_neighbors(distances, indices)
                    rows.append(row(config, status='ok', score=float(np.mean(pred == y_train[b]))))
                except Exception as exc:
                    rows.append(row(config, **failure(exc)))
            continue
        for config in candidates:
            try:
                pred = predictions(readout_id, hidden_train[a], y_train[a], hidden_train[b],
                                   sample_ids=train_ids[a], seed=seed, config=config)
                rows.append(row(config, status='ok', score=float(np.mean(np.asarray(pred) == y_train[b]))))
            except Exception as exc:
                rows.append(row(config, **failure(exc)))
    return rows


def select_readout(rows, candidates, *, folds, seed):
    """Mean selection-split accuracy, ties by lowest canonical config_id (matched.select_symmetric_probe)."""
    return select_symmetric_probe([dict(r, inner_fold=r['selection_fold']) for r in rows], candidates,
                                  inner_folds=folds, seeds=[seed], states=('hidden',))


def evaluate_unit(X, y, split, inner_fold, seed, encoder_settings, network_params, *, dataset_id, readouts=None,
                  selection_folds=None, protocol=None, family='readouts'):
    """One (outer fold, inner fold, seed): the single view fitted once on the inner training partition; every
    readout (family 'readouts': on the hidden representation) or variant (family 'permlvq': on the view-0 encoding
    positions) scored on the inner validation fold from the same arrays."""
    protocol = protocol or DEFAULT_PROTOCOL
    X, y = np.asarray(X), np.asarray(y)
    validate_split(split, len(y))
    if not 0 <= inner_fold < len(split['inner']):
        raise ValueError('inner_fold outside the split schedule')
    allowed = family_readouts(family, protocol)
    readouts = list(allowed if readouts is None else readouts)
    if REFERENCE not in readouts or set(readouts) - set(allowed) or len(set(readouts)) != len(readouts):
        raise ValueError(f'Readouts must be unique members of {allowed} and include the reference {REFERENCE}')
    selection_folds = protocol['selection']['folds'] if selection_folds is None else selection_folds
    candidate_lists = candidate_sets(family, protocol)
    inner = split['inner'][inner_fold]
    train, validation = list(inner['train']), list(inner['validation'])
    params = laboratory_params(encoder_settings, network_params)
    record = {'family': family, 'dataset_id': dataset_id, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
              'inner_fold': inner_fold, 'model_seed': seed,
              'stem': unit_stem(dataset_id, split['outer_repeat'], split['outer_fold'], inner_fold, seed),
              'encoder_settings': dict(encoder_settings), 'network_params': dict(network_params), 'params': params,
              'fit_rows': train, 'validation_rows': validation, 'training_labels_hash': array_hash(y[train]),
              'status': 'running', 'network': None, 'selection_splits': [], 'selection': {}, 'readouts': []}
    if family == 'permlvq':
        record['budget'] = lvq_budget(network_params, protocol)
        record['encodings'] = None
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            seed_fit(seed)
            start = time.perf_counter()
            model = MultiViewArrowFlow(**params, seed=seed).fit(X[train], y[train])
            fit_seconds = time.perf_counter() - start
            (encoder, network), = model.views_
            start = time.perf_counter()
            orders_train, orders_validation = encoder.transform(X[train]), encoder.transform(X[validation])
            reference = network.predict_orders(orders_validation)
            record['network'] = {
                'view_seed': network.seed, 'strategy': encoder.strategy, 'fit_seconds': fit_seconds,
                'encoding_seconds': model.encoding_seconds_, 'training_seconds': model.training_seconds_,
                'validation_sample_count': network.validation_sample_count_,
                'network_training_sample_count': network.training_sample_count_,
                'training_tie_rate': encoder.training_tie_rate_, 'final_state_hash': network.state_hash()}
            if family == 'readouts':
                train_repr = network.transform_orders(orders_train)
                query_repr = network.transform_orders(orders_validation)
                record['network'].update(hidden_width=int(train_repr.shape[1]),
                                         representation_hashes={'train': array_hash(train_repr), 'validation': array_hash(query_repr)})
                predictions = readout_predictions
            else:
                train_repr, query_repr = inverse_positions(orders_train), inverse_positions(orders_validation)
                record['encodings'] = {'vocabulary_size': int(train_repr.shape[1]),
                                       'hashes': {'train': array_hash(train_repr), 'validation': array_hash(query_repr)}}
                predictions = partial(variant_predictions, budget=record['budget'])
            record['network']['representation_seconds'] = time.perf_counter() - start
            parts = selection_splits(train, y[train], dataset_id=dataset_id, outer_repeat=split['outer_repeat'],
                                     outer_fold=split['outer_fold'], inner_fold=inner_fold, folds=selection_folds,
                                     seed=protocol['selection']['seed'])
            record['selection_splits'] = parts
            y_train, y_validation = y[train], y[validation]
            for readout_id in readouts:
                row = {'readout_id': readout_id, 'config': {}, 'config_id': config_id({}), 'status': 'ok'}
                start = time.perf_counter()
                try:
                    if readout_id == REFERENCE:
                        pred = reference
                    else:
                        if readout_id in candidate_lists:
                            candidates = candidate_lists[readout_id]
                            rows = selection_rows(readout_id, train_repr, y_train, train, parts, seed=seed,
                                                  candidates=candidates, predictions=predictions, budget=record.get('budget'))
                            chosen = select_readout(rows, candidates, folds=len(parts), seed=seed)
                            record['selection'][readout_id] = dict(chosen, rows=rows)
                            row.update(config=chosen['config'], config_id=chosen['config_id'],
                                       selection_score=chosen['inner_score'])
                        pred = predictions(readout_id, train_repr, y_train, query_repr,
                                           sample_ids=train, seed=seed, config=row['config'])
                    pred = np.asarray(pred)
                    if pred.shape != (len(validation),):
                        raise ValueError('Prediction count mismatch')
                    row.update(accuracy=float(np.mean(pred == y_validation)), prediction_hash=array_hash(pred),
                               seconds=time.perf_counter() - start)
                except Exception as exc:
                    row.update(status='failed', accuracy=None, exception=f'{type(exc).__name__}: {exc}',
                               seconds=time.perf_counter() - start)
                record['readouts'].append(row)
        record['status'] = 'ok' if all(r['status'] == 'ok' for r in record['readouts']) else 'failed'
    except Exception as exc:
        record['status'] = 'failed'
        record['exception'] = f'{type(exc).__name__}: {exc}'
        present = {r['readout_id'] for r in record['readouts']}
        record['readouts'].extend({'readout_id': readout_id, 'config': {}, 'config_id': config_id({}), 'status': 'failed',
                                   'accuracy': None, 'exception': record['exception']}
                                  for readout_id in readouts if readout_id not in present)
    record['seconds'] = time.perf_counter() - started
    return record


def evaluate_permlvq_unit(X, y, split, inner_fold, seed, encoder_settings, network_params, *, dataset_id, readouts=None,
                          selection_folds=None, protocol=None):
    return evaluate_unit(X, y, split, inner_fold, seed, encoder_settings, network_params, dataset_id=dataset_id,
                         readouts=readouts, selection_folds=selection_folds, protocol=protocol, family='permlvq')


def unit_rows(record):
    base = {key: record[key] for key in COLUMNS[:5]}
    return [{**base, 'readout_id': r['readout_id'], 'accuracy': r['accuracy']} for r in record['readouts']]


def rows_frame(rows):
    frame = pd.DataFrame(list(rows), columns=list(COLUMNS))
    frame['accuracy'] = frame['accuracy'].astype(float)
    return frame


def per_split(value, splits, name):
    if isinstance(value, dict):
        return [dict(value) for _ in splits]
    value = list(value)
    if len(value) != len(splits):
        raise ValueError(f'{name} must be one dict for every split or one entry per split')
    return [dict(v) for v in value]


def evaluate_readouts(X, y, dataset_id, splits, encoder_settings, network_params, seeds, readouts=READOUTS, *, sink=None,
                      family='readouts'):
    """DataFrame with COLUMNS: every readout's (or variant's) inner-validation accuracy per outer fold, inner fold and
    fitting seed. encoder_settings and network_params are one dict for all splits or one per split (bridge selection
    per fold)."""
    encoders = per_split(encoder_settings, splits, 'encoder_settings')
    networks = per_split(network_params, splits, 'network_params')
    rows = []
    for split, encoder, network in zip(splits, encoders, networks):
        for inner_fold in range(len(split['inner'])):
            for seed in seeds:
                record = evaluate_unit(X, y, split, inner_fold, seed, encoder, network, dataset_id=dataset_id,
                                       readouts=readouts, family=family)
                if sink is not None:
                    sink(record)
                rows.extend(unit_rows(record))
    return rows_frame(rows)


# ----------------------------------------------------------------------------- bridge source

def load_selections(bridge_source, datasets=None, *, outer_repeat=OUTER_REPEAT, allow_smoke=False):
    """Per dataset: data, the outer splits of one repeat and the bridge-selected configuration of every outer
    fold, verified through run_bridge.selection_record and resolved through run_bridge.planned_job. Outer-test
    numbers of the bridge (bridge_accuracy) are not copied into the laboratory."""
    bridge = load_bridge(bridge_source, allow_smoke=allow_smoke)
    protocol = bridge['protocol']
    panel = list(protocol['datasets']) if datasets is None else list(datasets)
    if not panel or len(set(panel)) != len(panel) or any(name not in protocol['datasets'] for name in panel):
        raise ValueError('Datasets must be unique members of the bridge panel')
    if not 0 <= outer_repeat < protocol['outer_repeats']:
        raise ValueError('outer_repeat outside the bridge schedule')
    loaded = {'bridge': bridge, 'outer_repeat': outer_repeat, 'datasets': {}}
    for name in panel:
        X, y, manifest, all_splits = load_prepared(bridge['directory'], name)
        splits = [s for s in all_splits if s['outer_repeat'] == outer_repeat]
        if len(splits) != protocol['outer_folds']:
            raise ValueError(f'{name}: expected {protocol["outer_folds"]} outer folds of repeat {outer_repeat}')
        jobs = []
        for split in splits:
            validate_split(split, len(y))
            record = selection_record(bridge, name, split, y, manifest)
            job = planned_job(name, split, record, X.shape[1])
            jobs.append({**{k: job[k] for k in ('dataset_id', 'outer_repeat', 'outer_fold', 'config_id', 'config',
                                                 'selected', 'model_seeds')},
                         **{k: record[k] for k in ('inner_score', 'result_file', 'result_sha256', 'log_sha256')}})
        loaded['datasets'][name] = {'X': X, 'y': y, 'manifest': manifest, 'splits': splits, 'all_splits': all_splits,
                                    'jobs': jobs}
    return loaded


def check_bridge_consistency(bridge_protocol, datasets, protocol, *, allow_smoke=False):
    """The production laboratory follows its protocol file: the bridge protocol, inner folds, seeds and panel agree."""
    if allow_smoke:
        return
    problems = []
    if bridge_protocol.get('protocol_id') != protocol['bridge_source']['protocol_id']:
        problems.append('bridge protocol_id')
    if bridge_protocol['inner_folds'] != protocol['inner_folds']:
        problems.append('inner_folds')
    if list(bridge_protocol['fit_seeds']) != list(protocol['fit_seeds']):
        problems.append('fit_seeds')
    if any(name not in protocol['datasets'] for name in datasets):
        problems.append('datasets outside the laboratory panel')
    if problems:
        raise ValueError('Laboratory protocol disagrees with the bridge source: ' + ', '.join(problems))


def plan_units(loaded, family='readouts', protocol=None):
    seeds = list(loaded['bridge']['protocol']['fit_seeds'])
    readouts = list(family_readouts(family, protocol))
    units = []
    for name, panel in loaded['datasets'].items():
        for job, split in zip(panel['jobs'], panel['splits']):
            configuration = laboratory_configuration(job['selected'])
            for inner_fold in range(len(split['inner'])):
                for seed in seeds:
                    units.append({'family': family, 'dataset_id': name, 'outer_repeat': split['outer_repeat'],
                                  'outer_fold': split['outer_fold'], 'inner_fold': inner_fold, 'model_seed': seed,
                                  'stem': unit_stem(name, split['outer_repeat'], split['outer_fold'], inner_fold, seed),
                                  'config_id': job['config_id'], 'config': job['config'], 'selected': job['selected'],
                                  **configuration, 'split_hash': config_id(split), 'readouts': readouts})
    return units


def protocol_file_record(path):
    path = Path(path)
    return {'protocol_file': str(path), 'protocol_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def laboratory_protocol(bridge_protocol, family='readouts', protocol=None):
    """The laboratory protocol file copied for one family, linked to the bridge protocol it develops on."""
    protocol = protocol or DEFAULT_PROTOCOL
    spec = family_spec(family, protocol)
    return {**protocol, 'family': family, 'purpose': spec['purpose'], 'phase': spec['phase'], 'reference': REFERENCE,
            'bridge_protocol_id': bridge_protocol.get('protocol_id'), 'bridge_protocol_hash': config_id(bridge_protocol),
            'bridge_frozen': bool(bridge_protocol.get('frozen', False)), 'outer_folds': bridge_protocol['outer_folds'],
            'bridge_inner_folds': bridge_protocol['inner_folds'], 'bridge_fit_seeds': list(bridge_protocol['fit_seeds']),
            'wallclock_cap_hours': spec['wallclock_cap_hours']}


def prepare(output, bridge_source, datasets=None, *, allow_smoke=False, purpose=PURPOSE, family='readouts', protocol=None,
            protocol_path=PROTOCOL_FILE, authorisation=None):
    """authorisation: the record returned by check_pilot (recorded in the manifest); None marks a prepare that
    bypassed the CLI gate (direct calls and tests)."""
    output = Path(output)
    protocol = protocol or DEFAULT_PROTOCOL
    authorisation = authorisation or {'mode': 'unchecked', 'note': 'prepare called without the CLI pilot gate'}
    loaded = load_selections(bridge_source, datasets, outer_repeat=protocol['outer_repeat'], allow_smoke=allow_smoke)
    bridge = loaded['bridge']
    check_bridge_consistency(bridge['protocol'], list(loaded['datasets']), protocol, allow_smoke=allow_smoke)
    write_json(output/'protocol.json', laboratory_protocol(bridge['protocol'], family, protocol))
    write_json(output/'environment.json', environment())
    for name, panel in loaded['datasets'].items():
        write_json(output/name/'manifest.json', panel['manifest'])
        write_json(output/name/'splits.json', panel['all_splits'])
        if not (output/name/'data.npz').exists():
            shutil.copyfile(bridge['directory']/name/'data.npz', output/name/'data.npz')
        copied_X, copied_y, _, _ = load_prepared(output, name)          # hash-checked against the manifest
        if not (np.array_equal(copied_X, panel['X'], equal_nan=True) and np.array_equal(copied_y, panel['y'])):
            raise ValueError(f'{name}: copied dataset differs from the bridge source')
    units = plan_units(loaded, family, protocol)
    write_json(output/'bridge_selections.json', [job for panel in loaded['datasets'].values() for job in panel['jobs']])
    write_json(output/'manifest.json', {
        'purpose': purpose, 'family': family, 'datasets': list(loaded['datasets']), 'outer_repeat': loaded['outer_repeat'],
        'bridge_source': {'directory': str(bridge['directory'].resolve()), 'protocol_id': bridge['protocol'].get('protocol_id'),
                          'protocol_hash': config_id(bridge['protocol']), 'code_revision': bridge['environment']['code_revision'],
                          'file_sha256': bridge['files'], 'model_id': BRIDGE_MODEL},
        'protocol_id': protocol['protocol_id'], 'protocol_hash': config_id(protocol), **protocol_file_record(protocol_path),
        'readouts': list(family_readouts(family, protocol)), 'reference': REFERENCE,
        'candidates': candidate_sets(family, protocol), 'selection_folds': protocol['selection']['folds'],
        'verdict_rule': dict(protocol['verdict_rule']), 'units': len(units), 'pilot_authorisation': authorisation})
    write_json(output/'planned_units.json', units)
    return units


# ----------------------------------------------------------------------------- pilot

def pilot(output, bridge_source, datasets=None, *, family='readouts', workers=1, allow_smoke=False, protocol=None):
    """One cell per requested dataset (first outer fold of the outer repeat, inner fold 0, first fit seed), evaluated
    in this process under the execution lock; writes pilot.json (and a history copy) with every cell's seconds and
    the projection of projected_hours for `workers`. The pilot refuses nothing itself: a failed cell is recorded with
    its exception text in failed_cells, the projection is then None with within_cap false, and both files are still
    written; check_pilot refuses such a pilot for the failed datasets."""
    output = Path(output)
    protocol = protocol or DEFAULT_PROTOCOL
    if not 1 <= workers <= protocol['max_workers']:
        raise ValueError('Worker count exceeds the shared limit')
    loaded = load_selections(bridge_source, datasets, outer_repeat=protocol['outer_repeat'], allow_smoke=allow_smoke)
    check_bridge_consistency(loaded['bridge']['protocol'], list(loaded['datasets']), protocol, allow_smoke=allow_smoke)
    units = plan_units(loaded, family, protocol)
    cells, started = [], time.perf_counter()
    with execution_lock():
        for name, panel in loaded['datasets'].items():
            unit = min((u for u in units if u['dataset_id'] == name),
                       key=lambda u: (u['outer_fold'], u['inner_fold'], loaded['bridge']['protocol']['fit_seeds'].index(u['model_seed'])))
            split = next(s for s in panel['splits'] if s['outer_fold'] == unit['outer_fold'])
            record = evaluate_unit(panel['X'], panel['y'], split, unit['inner_fold'], unit['model_seed'], unit['encoder_settings'],
                                   unit['network_params'], dataset_id=name, readouts=unit['readouts'], protocol=protocol, family=family)
            cells.append({'dataset_id': name, 'stem': unit['stem'], 'config_id': unit['config_id'], 'status': record['status'],
                          'seconds': record['seconds'], 'network_fit_seconds': record['network']['fit_seconds'] if record['network'] else None,
                          'readout_seconds': {r['readout_id']: r.get('seconds') for r in record['readouts']},
                          'accuracy': {r['readout_id']: r['accuracy'] for r in record['readouts']},
                          'planned_cells': sum(u['dataset_id'] == name for u in units), 'exception': record.get('exception'),
                          'readout_exceptions': {r['readout_id']: r.get('exception') for r in record['readouts'] if r['status'] != 'ok'}})
    seconds = [cell['seconds'] for cell in cells]
    failed = [cell for cell in cells if cell['status'] != 'ok']
    cap = family_spec(family, protocol)['wallclock_cap_hours']
    report = {'purpose': 'pilot_timing_only; never evidence', 'family': family, 'datasets': list(loaded['datasets']),
              'bridge_source': str(Path(bridge_source).resolve()), 'bridge_protocol_hash': config_id(loaded['bridge']['protocol']),
              'bridge_files': dict(loaded['bridge']['files']), 'protocol_id': protocol['protocol_id'],
              'protocol_hash': config_id(protocol), 'code_revision': environment()['code_revision'],
              'pilot_utc': datetime.now(timezone.utc).isoformat(), 'cells': cells, 'planned_cells': len(units),
              'workers': workers, 'mean_cell_seconds': float(np.mean(seconds)), 'max_cell_seconds': float(max(seconds)),
              'pilot_wall_seconds': time.perf_counter() - started, 'wallclock_cap_hours': cap,
              'failed_cells': [{k: cell[k] for k in ('dataset_id', 'stem', 'exception', 'readout_exceptions')} for cell in failed]}
    if failed:          # recorded, never refused here: check_pilot rejects a pilot with a failed cell for those datasets
        report['projection'], report['within_cap'] = None, False
    else:
        report['projection'] = projected_hours(report, report['datasets'], workers, cap)
        report['within_cap'] = report['projection']['within_cap']
    output.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(output/'pilots'/f'pilot_{stamp}.json', report)         # history, never overwritten
    (output/'pilot.json').write_text(content)                          # the latest pilot; re-piloting overwrites it
    return report


def projected_hours(report, datasets, workers, cap_hours):
    """Projection of a pilot for a dataset panel and a worker count: serial hours = max(planned cells x mean cell
    seconds, sum over datasets of planned cells x cell seconds); ideal wall-clock = serial / workers; conservative
    wall-clock = ideal x CONTENTION_FACTOR, which decides within_cap."""
    cells = {cell['dataset_id']: cell for cell in report['cells']}
    missing = sorted(set(datasets) - set(cells))
    if missing:
        raise ValueError(f'The pilot has no cell for {missing}')
    failed = sorted(name for name in datasets if cells[name]['status'] != 'ok')
    if failed:
        raise ValueError(f'The pilot cell failed for {failed}; a failed cell authorises nothing')
    planned = sum(cells[name]['planned_cells'] for name in datasets)
    mean_seconds = float(np.mean([cells[name]['seconds'] for name in datasets]))
    serial_mean = planned * mean_seconds / 3600
    serial_sum = sum(cells[name]['planned_cells'] * cells[name]['seconds'] for name in datasets) / 3600
    serial = max(serial_mean, serial_sum)
    conservative = serial / workers * CONTENTION_FACTOR
    return {'rule': 'serial hours = max(planned cells x mean cell seconds, sum over datasets of planned cells x cell seconds); '
                    'ideal wall-clock = serial / workers; conservative wall-clock = ideal x contention_factor decides within_cap',
            'datasets': list(datasets), 'planned_cells': int(planned), 'workers': int(workers),
            'serial_hours': serial, 'serial_hours_mean_based': serial_mean, 'serial_hours_per_dataset_sum': serial_sum,
            'ideal_wallclock_hours': serial / workers, 'contention_factor': CONTENTION_FACTOR,
            'conservative_wallclock_hours': conservative, 'wallclock_cap_hours': cap_hours, 'within_cap': bool(conservative <= cap_hours)}


def check_pilot(output, family, workers, *, datasets, bridge, protocol, skip=False):
    """The explicit gate before a family runs with several workers. Returns the authorisation recorded in the manifest:
    the pilot's SHA-256 and the recomputed projection, the single-worker case, or the --skip-pilot-check escape.
    A pilot authorises a run only when it is of the same family, covers every requested dataset with a successful
    cell, was made from the same bridge source (protocol hash and file SHA-256s) and laboratory protocol (hash), and
    its conservative projection for the requested worker count is within the family's cap."""
    if skip:
        return {'mode': 'skip_pilot_check', 'workers': int(workers), 'note': 'explicit escape; no pilot authorised this run'}
    if workers <= 1:
        return {'mode': 'single_worker', 'workers': int(workers)}
    path = Path(output)/'pilot.json'
    if not path.exists():
        raise ValueError(f'{family} refuses --workers {workers} without {path}; run `pilot --family {family}` into this '
                         'output first or pass --skip-pilot-check')
    saved = json.loads(path.read_text())
    if saved.get('family') != family:
        raise ValueError(f'{path} belongs to the {saved.get("family")!r} family, not {family!r}')
    missing = sorted(set(datasets) - set(saved.get('datasets', [])))
    if missing:
        raise ValueError(f'{path} did not time the datasets {missing}; pilot the full requested panel first')
    if saved.get('bridge_protocol_hash') != config_id(bridge['protocol']) or saved.get('bridge_files') != bridge['files']:
        raise ValueError(f'{path} was made from another bridge source')
    if saved.get('protocol_hash') != config_id(protocol):
        raise ValueError(f'{path} was made under another laboratory protocol')
    projection = projected_hours(saved, list(datasets), workers, family_spec(family, protocol)['wallclock_cap_hours'])
    if not projection['within_cap']:
        raise ValueError(f'{path} projects {projection["conservative_wallclock_hours"]:.2f} h at {workers} workers '
                         f'(serial {projection["serial_hours"]:.2f} h x {CONTENTION_FACTOR}) above the {family} cap of '
                         f'{projection["wallclock_cap_hours"]} h; reduce the panel or the design')
    return {'mode': 'pilot', 'pilot_file': str(path), 'pilot_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'pilot_utc': saved.get('pilot_utc'), 'pilot_code_revision': saved.get('code_revision'), 'workers': int(workers),
            'projection': projection}


# ----------------------------------------------------------------------------- execution

def unit_provenance(unit, dataset_hash, revision):
    return {**{k: unit[k] for k in ('family', 'dataset_id', 'outer_repeat', 'outer_fold', 'inner_fold', 'model_seed', 'stem',
                                    'config_id', 'config', 'selected', 'split_hash')},
            'dataset_hash': dataset_hash, 'code_revision': revision, 'readouts': list(unit['readouts'])}


def worker(arguments):
    output, unit = arguments
    output = Path(output)
    result_path = output/'results'/f'{unit["stem"]}.json'
    X, y, data, splits = load_prepared(output, unit['dataset_id'])
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    protocol = json.loads((output/'protocol.json').read_text())
    provenance = unit_provenance(unit, data['dataset_hash'], revision)
    if result_path.exists():          # resume: an existing record is reused only when it carries this plan's provenance
        saved = json.loads(result_path.read_text())
        if saved.get('provenance') != provenance:
            raise ValueError(f'Existing record {result_path} belongs to a different plan; use a new output directory')
        return str(result_path), 'reused'
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (unit['outer_repeat'], unit['outer_fold']))
    if config_id(split) != unit['split_hash']:
        raise ValueError('Prepared split differs from the planned unit')
    record = evaluate_unit(X, y, split, unit['inner_fold'], unit['model_seed'], unit['encoder_settings'],
                           unit['network_params'], dataset_id=unit['dataset_id'], readouts=unit['readouts'],
                           protocol=protocol, family=unit['family'])
    record['provenance'] = provenance
    write_json(result_path, record)
    return str(result_path), record['status']


def check_output(output, *, allow_smoke=False):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['purpose'] != PURPOSE and not allow_smoke:
        raise ValueError('Only a laboratory prepared from a frozen bridge source is evidence')
    if manifest.get('family', 'readouts') not in FAMILY_NAMES:
        raise ValueError('Unknown laboratory family in the manifest')
    return manifest


def run(output, workers=1, *, allow_smoke=False):
    output = Path(output)
    check_output(output, allow_smoke=allow_smoke)
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError('Worker count exceeds the shared limit')
    if json.loads((output/'environment.json').read_text()) != environment():
        raise ValueError('Code revision, sources or environment changed after prepare')
    units = json.loads((output/'planned_units.json').read_text())
    started = time.perf_counter()
    statuses = Counter()
    with execution_lock(), ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(worker, (str(output), unit)) for unit in units]
        for i, future in enumerate(as_completed(futures), 1):
            path, status = future.result()
            statuses[status] += 1
            print(f'[{i}/{len(units)}] {status} {path} ({time.perf_counter() - started:.0f} s)', flush=True)
    return {'wall_seconds': time.perf_counter() - started, 'units': len(units), 'statuses': dict(statuses)}


# ----------------------------------------------------------------------------- verification and summary

def validate_record(record, unit, split, dataset_hash, revision, protocol=None):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(record.get('provenance') == unit_provenance(unit, dataset_hash, revision), 'provenance disagrees with the plan')
    for key in ('family', 'dataset_id', 'outer_repeat', 'outer_fold', 'inner_fold', 'model_seed', 'stem', 'encoder_settings',
                'network_params'):
        require(record[key] == unit[key], f'{key} disagrees with the plan')
    require(record['params'] == laboratory_params(unit['encoder_settings'], unit['network_params']), 'laboratory parameters')
    if unit['family'] == 'permlvq':
        require(record['budget'] == lvq_budget(unit['network_params'], protocol), 'LVQ budget')
    inner = split['inner'][unit['inner_fold']]
    require(record['fit_rows'] == inner['train'] and record['validation_rows'] == inner['validation'], 'inner partition rows')
    require(record['status'] in ('ok', 'failed'), 'unit status')
    require([r['readout_id'] for r in record['readouts']] == list(unit['readouts']), 'readout schedule')
    for row in record['readouts']:
        if row['status'] == 'ok':
            require(isinstance(row['accuracy'], (int, float)) and 0 <= row['accuracy'] <= 1, 'accuracy range')
        else:
            require(row['status'] == 'failed' and row['accuracy'] is None and row.get('exception'), 'failed readout record')
    require(record['status'] == ('ok' if all(r['status'] == 'ok' for r in record['readouts']) else 'failed'), 'status/readouts')
    train, validation = set(inner['train']), set(inner['validation'])
    for part in record['selection_splits']:
        fit, query = set(part['fit_rows']), set(part['query_rows'])
        require(fit | query <= train and not fit & query and not query & validation, 'selection split leaks')
    candidate_lists = candidate_sets(unit['family'], protocol)
    for readout_id, chosen in record['selection'].items():
        require(readout_id in candidate_lists, 'selection for a readout without a grid')
        candidates = candidate_lists[readout_id]
        require(chosen['config'] in candidates and chosen['config_id'] == config_id(chosen['config']),
                'selected configuration outside its grid')
        actual = select_readout(chosen['rows'], candidates, folds=len(record['selection_splits']), seed=unit['model_seed'])
        require(all(chosen[k] == v for k, v in actual.items()), 'readout selection disagrees with its scores')
        row = next(r for r in record['readouts'] if r['readout_id'] == readout_id)
        require(row['config'] == chosen['config'], 'scored readout configuration differs from the selection')


def collect_records(output, *, allow_smoke=False):
    """Every planned unit reconciled with its saved record; missing or inconsistent records are errors,
    failed units stay explicit."""
    output = Path(output)
    manifest = check_output(output, allow_smoke=allow_smoke)
    protocol = json.loads((output/'protocol.json').read_text())
    saved = json.loads((output/'environment.json').read_text())
    if saved['source_hashes'] != environment()['source_hashes']:
        raise ValueError('Scientific sources changed after the laboratory was prepared')
    units = json.loads((output/'planned_units.json').read_text())
    if len(units) != manifest['units']:
        raise ValueError('Planned unit count disagrees with the manifest')
    if any(u['family'] != manifest['family'] for u in units):
        raise ValueError('Planned units belong to another family than the manifest')
    prepared = {name: load_prepared(output, name) for name in manifest['datasets']}
    records, issues = [], []
    for unit in units:
        path = output/'results'/f'{unit["stem"]}.json'
        if not path.exists():
            issues.append(f'missing {unit["stem"]}')
            continue
        try:
            record = json.loads(path.read_text())
            X, y, data, splits = prepared[unit['dataset_id']]
            split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (unit['outer_repeat'], unit['outer_fold']))
            validate_record(record, unit, split, data['dataset_hash'], saved['code_revision'], protocol)
            records.append(record)
        except (KeyError, ValueError, TypeError, StopIteration) as exc:
            issues.append(f'{unit["stem"]}: {type(exc).__name__}: {exc}')
    if issues:
        raise ValueError('Incomplete laboratory evidence: ' + '; '.join(issues))
    return {'protocol': protocol, 'manifest': manifest, 'units': units, 'records': records,
            'code_revision': saved['code_revision'],
            'selections': json.loads((output/'bridge_selections.json').read_text())}


def knn_hidden_comparison(path, manifest, selections, datasets):
    """knn_hidden mean inner accuracies of a readouts summary produced from the same bridge selections; the status
    says why the comparison is absent when it is."""
    result = {'source': None if path is None else str(path), 'status': 'absent', 'mean_inner_accuracy': None}
    if path is None or not Path(path).exists():
        return result
    report = json.loads(Path(path).read_text())
    ours = {(s['dataset_id'], s['outer_fold']): s['config_id'] for s in selections}
    theirs = {(name, entry['outer_fold']): entry['config_id']
              for name, entries in report.get('resolved_configurations', {}).items() for entry in entries}
    if report.get('bridge_source', {}).get('protocol_hash') != manifest['bridge_source']['protocol_hash'] \
            or report.get('outer_repeat') != manifest['outer_repeat']:
        result['status'] = 'mismatch: another bridge protocol or outer repeat'
        return result
    if any(ours[key] != theirs.get(key) for key in ours):
        result['status'] = 'mismatch: other bridge selections'
        return result
    means = {name: report.get('mean_inner_accuracy', {}).get(name, {}).get('knn_hidden') for name in datasets}
    if any(value is None for value in means.values()):
        result['status'] = 'incomplete: knn_hidden is missing for a dataset'
        return result
    result.update(status='ok', mean_inner_accuracy=means, readouts_code_revision=report.get('code_revision'))
    return result


def summary(output, *, allow_smoke=False, comparison=None):
    """Mean inner accuracy per readout (or variant) and dataset over outer folds x inner folds x seeds, paired changes
    against the reference, the chosen configurations and the adoption verdicts. For the permlvq family the
    readouts summary at `comparison` (knn_hidden) is compared as well when it exists and matches."""
    collected = collect_records(output, allow_smoke=allow_smoke)
    manifest, records, protocol = collected['manifest'], collected['records'], collected['protocol']
    family = manifest['family']
    datasets, readouts, rule = manifest['datasets'], manifest['readouts'], manifest['verdict_rule']
    candidate_lists = candidate_sets(family, protocol)
    rows = [row for record in records for row in unit_rows(record)]
    frame = rows_frame(rows)
    means, paired, chosen, failures = {}, {}, {}, {}
    for name in datasets:
        subset = frame[frame['dataset_id'] == name]
        means[name], paired[name], chosen[name] = {}, {}, {}
        for readout in readouts:
            scores = subset[subset['readout_id'] == readout]['accuracy']
            means[name][readout] = float(scores.mean()) if len(scores) and scores.notna().all() else None
        table = subset.pivot(index=['outer_fold', 'inner_fold', 'model_seed'], columns='readout_id', values='accuracy')
        for readout in readouts:
            if readout == REFERENCE:
                continue
            difference = (table[readout] - table[REFERENCE]).dropna()
            paired[name][readout] = {'n_units': int(len(difference)),
                                     'mean_change_pp': float(100 * difference.mean()) if len(difference) else None,
                                     'sd_change_pp': float(100 * difference.std(ddof=1)) if len(difference) > 1 else None,
                                     'units_improved': int((difference > 0).sum()), 'units_worsened': int((difference < 0).sum())}
        for readout in candidate_lists:
            if readout in readouts:
                counts = Counter(canonical_json(r['selection'][readout]['config']) for r in records
                                 if r['dataset_id'] == name and readout in r['selection'])
                chosen[name][readout] = {k: v for k, v in sorted(counts.items())}
        failures[name] = [r['stem'] for r in records if r['dataset_id'] == name and r['status'] != 'ok']

    def verdicts_against(baseline):
        verdicts = {}
        for readout in readouts:
            if readout == REFERENCE:
                continue
            if any(means[name][readout] is None or baseline[name] is None for name in datasets):
                verdicts[readout] = {'verdict': 'incomplete', 'reason': 'a failed unit leaves this readout without a complete panel'}
                continue
            verdicts[readout] = verdict_record({name: 100 * baseline[name] for name in datasets},
                                               {name: 100 * means[name][readout] for name in datasets}, rule=rule)
        return verdicts
    verdicts = verdicts_against({name: means[name][REFERENCE] for name in datasets})
    seconds = [r['seconds'] for r in records]
    network_seconds = [r['network']['fit_seconds'] for r in records if r['network']]
    report = {'purpose': f'{protocol["purpose"]}; exploratory; never outer-fold evidence', 'family': family,
              'code_revision': collected['code_revision'], 'bridge_source': manifest['bridge_source'],
              'protocol_id': manifest.get('protocol_id'), 'protocol_hash': manifest.get('protocol_hash'),
              'pilot_authorisation': manifest.get('pilot_authorisation'),
              'datasets': datasets, 'outer_repeat': manifest['outer_repeat'], 'fit_seeds': collected['protocol']['bridge_fit_seeds'],
              'reference': REFERENCE, 'readouts': readouts,
              'readout_definitions': family_spec(family, protocol).get('readout_definitions' if family == 'readouts' else 'variant_definitions'),
              'aggregation': 'mean inner-validation accuracy over outer folds x inner folds x fitting seeds of outer repeat '
                             f'{manifest["outer_repeat"]}; paired changes are per-unit differences against the reference',
              'verdict_rule': rule, 'units': {'planned': len(collected['units']), 'completed': sum(r['status'] == 'ok' for r in records),
                                              'failed': sum(r['status'] != 'ok' for r in records)},
              'failed_units': failures, 'mean_inner_accuracy': means, 'paired_change_vs_reference': paired,
              'selected_configurations': chosen, 'verdicts': verdicts,
              'resolved_configurations': {name: [{'outer_fold': s['outer_fold'], 'config_id': s['config_id'],
                                                  **{k: s['selected'][k] for k in ('widths', 'learning_rate', 'iterations', 'batch_size',
                                                                                   'embed_dim', 'degree', 'augment', 'validation_ratio')}}
                                                 for s in collected['selections'] if s['dataset_id'] == name] for name in datasets},
              'unit_seconds': {'mean': float(np.mean(seconds)) if seconds else None, 'max': float(max(seconds)) if seconds else None,
                               'total': float(sum(seconds)), 'network_fit_total': float(sum(network_seconds))},
              'rows': rows}
    if family == 'permlvq':
        report['budgets'] = {name: sorted({canonical_json(r['budget']) for r in records if r['dataset_id'] == name}) for name in datasets}
        knn = knn_hidden_comparison(comparison, manifest, collected['selections'], datasets)
        if knn['status'] == 'ok':
            knn['verdicts'] = verdicts_against(knn['mean_inner_accuracy'])
            knn['note'] = 'informational: the adoption verdict is against the reference output_rule; this compares the same ' \
                          'mean inner accuracies against knn_hidden of the readouts laboratory'
        report['comparison_knn_hidden'] = knn
    return report


def write_summary(output, *, allow_smoke=False, destination=None, comparison=None):
    output = Path(output)
    report = summary(output, allow_smoke=allow_smoke, comparison=comparison)
    destination = output if destination is None else Path(destination)
    stem = report['family']
    write_json(destination/f'{stem}_summary.json', report)
    write_csv(destination/f'{stem}_rows.csv', COLUMNS, [[row[c] for c in COLUMNS] for row in report['rows']])
    return report


def smoke(output, workers=1, *, family='readouts', protocol=None):
    """Synthetic end-to-end exercise of pilot, prepare, run and summary; never evidence."""
    output = Path(output)
    root, destination = laboratory_paths(output, family)
    source = output/'synthetic_bridge'
    if not (source/'protocol.json').exists():
        synthetic_bridge_source(source)
    protocol = protocol or DEFAULT_PROTOCOL
    pilot(root, source, ['synthetic'], family=family, workers=workers, allow_smoke=True, protocol=protocol)
    authorisation = check_pilot(root, family, workers, datasets=['synthetic'], bridge=load_bridge(source, allow_smoke=True),
                                protocol=protocol)
    prepare(root, source, ['synthetic'], allow_smoke=True, purpose=SMOKE_PURPOSE, family=family, protocol=protocol,
            authorisation=authorisation)
    run(root, workers, allow_smoke=True)
    return write_summary(root, allow_smoke=True, destination=destination, comparison=comparison_path(output, family))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['readouts', 'permlvq', 'pilot', 'summary', 'smoke'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bridge-source', type=Path)
    parser.add_argument('--dataset', nargs='+', help='bridge panel members; default: the whole bridge panel')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--family', choices=list(FAMILY_NAMES), default=None,
                        help='laboratory family; required for pilot, summary and smoke; must agree with a laboratory command')
    parser.add_argument('--protocol', type=Path, default=PROTOCOL_FILE, help='laboratory protocol file')
    parser.add_argument('--skip-pilot-check', action='store_true',
                        help='explicit escape: start with several workers without an authorising pilot.json (recorded in the manifest)')
    args = parser.parse_args(argv)
    if args.command in FAMILY_NAMES:
        family = args.command
        if args.family is not None and args.family != family:
            raise ValueError(f'--family {args.family} disagrees with the {family} command')
    else:
        if args.family is None:
            raise ValueError(f'{args.command} requires --family (one of {FAMILY_NAMES}); there is no default')
        family = args.family
    protocol = load_protocol(args.protocol)
    root, destination = laboratory_paths(args.output, family)
    if args.command == 'smoke':
        report = smoke(args.output, args.workers, family=family, protocol=protocol)
    elif args.command == 'summary':
        report = write_summary(root, destination=destination, comparison=comparison_path(args.output, family))
    elif args.command == 'pilot':
        if args.bridge_source is None:
            raise ValueError('pilot requires --bridge-source')
        report = pilot(root, args.bridge_source, args.dataset, family=family, workers=args.workers, protocol=protocol)
        print(json.dumps({'family': family, 'cells': [{k: c[k] for k in ('dataset_id', 'status', 'seconds', 'planned_cells')}
                                                      for c in report['cells']],
                          'projection': report['projection']}, indent=2))
        return
    else:
        if args.bridge_source is None:
            raise ValueError(f'{args.command} requires --bridge-source')
        bridge = load_bridge(args.bridge_source)                       # frozen production source required
        datasets = list(args.dataset) if args.dataset else list(bridge['protocol']['datasets'])
        authorisation = check_pilot(root, family, args.workers, datasets=datasets, bridge=bridge, protocol=protocol,
                                    skip=args.skip_pilot_check)
        prepare(root, args.bridge_source, datasets, family=family, protocol=protocol, protocol_path=args.protocol,
                authorisation=authorisation)
        timing = run(root, args.workers)
        print(json.dumps(timing, indent=2))
        report = write_summary(root, destination=destination, comparison=comparison_path(args.output, family))
    print(json.dumps({'family': report['family'], 'mean_inner_accuracy': report['mean_inner_accuracy'], 'verdicts': report['verdicts'],
                      'units': report['units']}, indent=2))


if __name__ == '__main__':
    main()
