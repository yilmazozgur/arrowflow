"""Training diagnostics of ArrowFlow-kNN at its reconstructed per-fold selections (descriptive; nothing is selected).

draft    --output P                                                  the unfrozen protocol (draft_protocol)
smoke    --protocol P --output O [--workers 3]                       synthetic references and a complete run; never evidence
pilot    --protocol P [--reference NAME=RUN,ABLATION ...] --output O  training-only timing and the projection at 8 workers
freeze   --draft P --pilot O/pilot.json --stages S --output F        the frozen protocol, only if the projection is within the cap
run      --protocol P [--reference ...] [--dataset ...] --output O --workers 8
                                                                     every planned job, then the five CSVs and provenance.json
summary  --output O                                                  re-verify the run, then diagnostics_summary.json

Unit of work: one dataset, one outer fold, fitting seed 8129 and all seven views of ArrowFlow-kNN (MultiViewArrowFlowKNN) at
the fold's selection. A reference is a production run of arrowflow_full_knn together with the component ablation run that
sealed its per-fold selections (--reference NAME=RUN_DIR,ABLATION_DIR). Every selection is reconstructed from the run's
complete inner fit history (run_knn_ablation.selection_record), must equal the sealed record, and is resolved from the outer
training partition's shape (bridge.resolve_selected).

The seven views are trained step for step as MultiViewArrowFlowKNN.fit trains them. The only addition is an instance-level
wrapper around each view network's update_network, installed between ArrowFlowEstimator.initialize_orders and
train_initialized. After the core's own update, and outside any evaluation, the wrapper reads the core's validation error,
its running minimum and whether it replaced its checkpoint. It also copies the filters at the scheduled snapshots and at every
checkpoint replacement. Every read runs inside a guard that saves and restores the global numpy and Python RNG states and
records whether they changed. All diagnostic evaluation happens after training, on those copies with pure numpy
(network_forward), never on the live network.

Every job fails when a check fails:
- the 7-view kNN predictions on the outer test rows must equal the reference run's recorded predictions for seed 8129;
- on the first outer fold of each dataset, each view's final state_hash() must equal that of an uninstrumented fit;
- the RNG guards, the core's validation errors and the checkpoint orders must verify.
Outputs are all or none and never replace a file with different content.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import importlib
import io
import json
import multiprocessing
from pathlib import Path
import random
import shutil
import subprocess
import time
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from arrowflow.ranking import inverse_positions, score_order
from . import run_knn_ablation as base
from .bridge import resolve_selected
from .comparisons import StableFootruleKNN, derive_seed
from .evaluation import canonical_json, config_id, make_splits, validate_split
from .knn_controls import reference_pins, synthetic_reference_run
from .models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from .multiview import MultiViewArrowFlowKNN, view_strategy
from .newdata import WORKSPACE_RUNS, makespan, sha256_file
from .run_revision import environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-14'
PROTOCOL = PROTOCOLS/'training_diagnostics.json'
PROTOCOL_ID = 'arrowflow-v3-training-diagnostics-1'
FAMILY = 'training_diagnostics'
SOURCE_MODULES = base.SOURCE_MODULES + ['experiments.make_revision.run_knn_ablation', 'experiments.make_revision.run_newdata_ablation',
                                        'experiments.make_revision.newdata', 'experiments.make_revision.projected_knn']
REFERENCE_MODEL = base.REFERENCE_MODEL
MODEL_SEED = 8129
N_VIEWS = 7
SNAPSHOT_EVERY = 10
RELABEL_DRAWS = 20
RELABEL_SEED = 20260914
CAP_HOURS = 4
WORKERS = 8
MAX_WORKERS = 16
PILOT_DATASETS = ('iris', 'segment')               # the smallest benchmark dataset and the largest of the 17 (rows)
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'pilot_projection')
DRAFT_STATUS = 'drafted_awaiting_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_any_diagnostic_result'
SUMMARY_FILE = 'diagnostics_summary.json'
PROVENANCE_FILE = 'provenance.json'
BENCHMARK_LOADER = 'experiments.make_revision.run_revision:load_dataset'
NEWDATA_LOADER = 'experiments.make_revision.newdata:load_newdata'
ABLATION_FILES = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json')
ABLATION_SUMMARIES = ('knn_ablation_summary.json', 'newdata_ablation_summary.json')
RUN_PINS = ('model_id', 'family', 'protocol_id', 'code_revision', 'protocol_sha256', 'summary_sha256')
ABLATION_PINS = ('family', 'protocol_id', 'code_revision', 'protocol_sha256', 'manifest_sha256', 'reference_selections_sha256',
                 'summary_file', 'summary_sha256')

# The frozen references of the 17 datasets, pinned by the sha256 of their files (computed 2026-09-14 from the workspace runs;
# each protocol sha256 equals the committed protocol file named in protocol_file).
REFERENCES = {
    'bridge_knn': {
        'datasets': ['iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits'],
        'loader': BENCHMARK_LOADER,
        'run': {'model_id': REFERENCE_MODEL, 'family': 'bridge_knn', 'protocol_id': 'arrowflow-v3-bridge-knn-1',
                'code_revision': '70fb9bf31092cb64e2bd349403ad090699a3494d',
                'protocol_sha256': '6335d8c4a1b16fb4044c3badb1fff1448f4083d4cc89611cd54b276d60057ff1',
                'summary_sha256': 'f5f0c7016864ce354c70f148bfecc31f8dbdbff5bc4f41d74e63e22a6ff36fda',
                'protocol_file': 'experiments/make_revision/protocols/2026-09-12/bridge_knn.json'},
        'ablation': {'family': 'knn_ablation', 'protocol_id': 'arrowflow-v3-knn-ablation-1',
                     'code_revision': '08cf39cd23fe0745a0de3c7a982b0c1d297500c5',
                     'protocol_sha256': '126cf6d72a2842751a04b5994f7f4b3e9b3b3f5261407d1f1345295b043a1ba5',
                     'manifest_sha256': '615868f286be02f8d32068bb2e27ee016f530c08aa6b22ce49bbd21365a8bf9e',
                     'reference_selections_sha256': '809d725c60dd1548d120c7b8534b6446c948ead0ec3563baf7b98c6f5ca029e2',
                     'summary_file': 'knn_ablation_summary.json',
                     'summary_sha256': '53cebfacb52ab2de6d33ef869344f9753dee7d12a8176fdb48df0fe75dc8cde7',
                     'protocol_file': 'experiments/make_revision/protocols/2026-09-12/knn_ablation.json'}},
    'newdata_batch1': {
        'datasets': ['balance_scale', 'ionosphere', 'diabetes', 'banknote_authentication', 'qsar_biodeg'],
        'loader': NEWDATA_LOADER,
        'run': {'model_id': REFERENCE_MODEL, 'family': 'newdata', 'protocol_id': 'arrowflow-v3-newdata-batch1-1',
                'code_revision': '6022f9b5e2312f80a96e96b2d0607c72b7d52138',
                'protocol_sha256': 'f715ec2808b4b08c10e49ae23696070a471d7ea88eb351d63b059eecccc0d011',
                'summary_sha256': 'f3ab23c5eeccbe36f036325fe94c4c28ce69bc9d35fa16e62cea813e977ec843',
                'protocol_file': 'experiments/make_revision/protocols/2026-09-12/newdata_batch1.json'},
        'ablation': {'family': 'newdata_ablation', 'protocol_id': 'arrowflow-v3-newdata-ablation-1',
                     'code_revision': 'be41fa9626e634bb0adb6c9caada5fe4aa5d0424',
                     'protocol_sha256': '3c4efe43eea7151c568a64a883f834e6d2f8d849aef34d0e45f24bb42e962c8c',
                     'manifest_sha256': '14d6c377f5afc78f9b7407ab4dc3940cfc4e77fccb497f556ad013b42d70fdbb',
                     'reference_selections_sha256': 'fa6933cd9dc0be3c786849d816922422ed3d9fe8dccf3ec83faeebfba02722d8',
                     'summary_file': 'newdata_ablation_summary.json',
                     'summary_sha256': '2775b79d3a524e526ee1bdf937bdea783ebf0e22d14e9f4531ecd1c013c91246',
                     'protocol_file': 'experiments/make_revision/protocols/2026-09-12/newdata_ablation.json'}},
    'newdata_batch2': {
        'datasets': ['mfeat_zernike', 'vertebra_column', 'steel_plates_fault', 'climate_model_simulation_crashes',
                     'hcv_egyptian_patients'],
        'loader': NEWDATA_LOADER,
        'run': {'model_id': REFERENCE_MODEL, 'family': 'newdata', 'protocol_id': 'arrowflow-v3-newdata-batch2-1',
                'code_revision': '6022f9b5e2312f80a96e96b2d0607c72b7d52138',
                'protocol_sha256': '188ff1e946b6dc8df712c5a5cfb01e9cf2f9e7022a7776d98ec95ae1f66fca49',
                'summary_sha256': '50ebfbef88616f8ecd20737f45b3ef840ed19f75a5e5a3f8ceeb120e386d49db',
                'protocol_file': 'experiments/make_revision/protocols/2026-09-12/newdata_batch2.json'},
        'ablation': {'family': 'newdata_ablation', 'protocol_id': 'arrowflow-v3-newdata-ablation-1',
                     'code_revision': 'be41fa9626e634bb0adb6c9caada5fe4aa5d0424',
                     'protocol_sha256': '3c4efe43eea7151c568a64a883f834e6d2f8d849aef34d0e45f24bb42e962c8c',
                     'manifest_sha256': '14d6c377f5afc78f9b7407ab4dc3940cfc4e77fccb497f556ad013b42d70fdbb',
                     'reference_selections_sha256': 'fa6933cd9dc0be3c786849d816922422ed3d9fe8dccf3ec83faeebfba02722d8',
                     'summary_file': 'newdata_ablation_summary.json',
                     'summary_sha256': '2775b79d3a524e526ee1bdf937bdea783ebf0e22d14e9f4531ecd1c013c91246',
                     'protocol_file': 'experiments/make_revision/protocols/2026-09-12/newdata_ablation.json'}},
}
DEFAULT_SOURCES = {'bridge_knn': (WORKSPACE_RUNS/'2026-09-12-bridge-knn', WORKSPACE_RUNS/'2026-09-13-knn-ablation'),
                   'newdata_batch1': (WORKSPACE_RUNS/'2026-09-14-newdata-batch1', WORKSPACE_RUNS/'2026-09-14-newdata-ablation'),
                   'newdata_batch2': (WORKSPACE_RUNS/'2026-09-14-newdata-batch2', WORKSPACE_RUNS/'2026-09-14-newdata-ablation')}


class CheckFailed(RuntimeError):
    """A non-invasiveness or consistency check of a diagnostics job failed."""


def environment():
    return environment_record(__package__ + '.training_diagnostics:environment')


def native(value):
    """JSON-ready plain Python values (numpy scalars and arrays converted)."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _plain(value):
    return json.loads(canonical_json(native(value)))


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ----------------------------------------------------------------------------- protocol

def design():
    """The fixed design every training diagnostics protocol declares (validate_protocol requires it verbatim)."""
    return {
        'purpose': 'descriptive training diagnostics of ArrowFlow-kNN: how the non-gradient rule changes the layers',
        'selection_statement': 'Nothing is selected. Every configuration is the per-fold selection sealed by the reference '
                               'ablation run; every kNN readout setting is the one the fit chose on training rows; no '
                               'diagnostic value feeds back into any fit, readout, checkpoint, dataset, fold or later analysis '
                               'choice; no significance test is made.',
        'unit_of_work': 'every protocol dataset x every outer fold (5 folds x 3 repeats) x fitting seed 8129 x all seven views',
        'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3, 'split_seed': 27183, 'fit_seeds': [8129, 19391, 39019],
        'model_seed': MODEL_SEED, 'n_views': N_VIEWS,
        'fit': 'MultiViewArrowFlowKNN at bridge.resolve_selected(sealed config, n_features, n_train) on the outer training rows: '
               'for view v, OrdinalEncoder(view_strategy, embed_dim, degree, lda_ratio, derive_seed(8129, "view", v)), '
               'ArrowFlowEstimator.initialize_orders then train_initialized (the steps of fit_orders), then '
               'MultiViewArrowFlowKNN._fit_view_readout (select_knn_readout on the trained hidden ranking); majority vote',
        'instrumentation': 'an instance-level update_network wrapper on each view network, installed after initialize_orders '
                           'and removed after train_initialized; after the core update it only reads the core validation error, '
                           'min_error_val and whether optimal_model was replaced, and copies index matrices (and adjacency '
                           'orders at the initial state and at every replacement); every read runs inside a guard that saves '
                           'and restores the global numpy and Python RNG states and records whether they changed; no '
                           'evaluation runs during training; every diagnostic evaluation runs after training in pure numpy on '
                           'the copies (network_forward: cityblock responses, stable ranking by ascending response then filter '
                           'index, inverse positions)',
        'snapshots': {'every': SNAPSHOT_EVERY,
                      'schedule': 'iteration 0 (initial filters), every 10 iterations up to T, T, and the view checkpoint',
                      'previous_snapshot': 'the largest scheduled iteration below the snapshot iteration (none at iteration 0)'},
        'measures': {
            'validation_curve': 'the core validation error of the initial filters (iteration 0) and after every update 1..T, '
                                'with the running minimum; the validation rows are the first int(val_data_ratio * n) samples '
                                'the core holds out',
            'checkpoint_iteration': 'the last iteration whose validation error is strictly below the running minimum before it; '
                                    '0 means the initial filters were returned; verified by comparing the final network '
                                    'orders and index matrices with the copy taken when the core replaced its checkpoint',
            'displacement': 'per layer (each hidden layer and the output layer), at every snapshot: the mean over filters of '
                            'footrule(filter_t, filter_0) / floor(n^2 / 2), n the permutation length, and the share of '
                            'filters that differ; the same against the previous snapshot',
            'learning_curves': 'at every snapshot, on the outer test rows: the view kNN readout accuracy with the view final '
                               'selected setting refitted on the training rows hidden rankings at that snapshot; the 7-view '
                               'majority of those at every scheduled iteration and with each view at its checkpoint; the output '
                               'rule (prototype readout) accuracy; and its accuracy on the outer training rows (descriptive)',
            'ties': 'on the outer test rows with the initial filters and the checkpoint filters: per hidden layer the share of '
                    '(row, filter) responses equal to another filter response in the same row and the mean over rows of the '
                    'number of distinct responses / number of filters; for the output layer the share of rows whose nearest '
                    'class filter is tied',
            'relabel': 'at the checkpoint network, R draws; each draw permutes each hidden layer filter IDs uniformly '
                       '(np.random.RandomState(derive_seed(seed, dataset_id, outer_repeat, outer_fold, model_seed, view, '
                       'draw, layer)).permutation) and relabels the following layer coordinates consistently, so that without '
                       'ties every response is unchanged up to the relabeling; the kNN readout is refitted with the same '
                       'setting on the relabeled training hidden rankings; per view and for the 7-view majority: the share '
                       'of outer test predictions that change and the accuracy change, mean and extremes over draws'},
        'relabel': {'draws': RELABEL_DRAWS, 'seed': RELABEL_SEED},
        'checks': {
            'reference_predictions': 'every job: the 7-view kNN predictions on the outer test rows equal the reference run '
                                     'recorded predictions for seed 8129 exactly, in the sealed test order',
            'uninstrumented_state_hash': 'first outer fold of every dataset: each view final state_hash() and kNN predictions '
                                         'equal those of MultiViewArrowFlowKNN(**selected, seed=8129).fit on the same rows',
            'rng_untouched': 'every job: every guarded read left the global numpy and Python RNG states unchanged',
            'wrapper_removed': 'every job: no view network keeps an instance-level update_network',
            'network_unchanged_by_diagnostics': 'every job: each view state_hash() is the same before and after the diagnostics',
            'training_record': 'every job: T updates recorded in order, the network state at the first update equals the '
                               'copy taken after initialize_orders, validation rows exist, the core running minimum and '
                               'checkpoint replacements equal those derived from the validation curve',
            'checkpoint_orders': 'every job: the final network equals the copy at its checkpoint, and the core final minimum '
                                 'equals the validation error there',
            'validation_error_recomputed': 'every job: the numpy output rule on the core validation samples reproduces the core '
                                           'validation error at every snapshot',
            'forward_reimplementation': 'every job: at the final network, network_forward equals transform_orders_by_depth on '
                                        'training and test rows and predict_orders on test rows',
            'checkpoint_readout': 'every job: the refitted kNN at each view checkpoint equals the view final kNN predictions, '
                                  'their majority equals the job predictions, and identity relabeling changes nothing',
            'snapshot_schedule': 'every job: the captured scheduled iterations equal the declared schedule'},
        'outputs': {'job_records': 'jobs/<dataset>__r<repeat>f<fold>.json with every measure and check, artifacts/<stem>.npz with '
                                   'the kNN, output-rule and relabeled predictions (sha256 in the record)',
                    'tables': 'validation_curves.csv, snapshots.csv, checkpoints.csv, ties.csv, relabel.csv and provenance.json, '
                              'written only after every planned job succeeded and verified (all or none)',
                    'summary': f'{SUMMARY_FILE} by the summary stage after re-verifying the run and re-deriving every selection',
                    'overwrite': 'an existing file with different content is never replaced'},
        'summary': 'per dataset: median and IQR (numpy percentile, linear) of the checkpoint iteration over views and folds; the '
                   'share of views with checkpoint 0; the mean displacement at the checkpoint per widths and layer; the 7-view '
                   'kNN test accuracy at iteration 0 and at the checkpoint (mean and SD over folds); the mean tie shares with '
                   'initial and checkpoint filters; the relabel change shares and accuracy changes',
        'dataset_loading': 'the reference run prepared data (hash-checked) must equal the ablation copy, its splits the declared '
                           'nested splits, and in production a fresh load by the reference loader (run_revision.load_dataset '
                           'or newdata.load_newdata with every pin) must return the same arrays and dataset hash',
        'pilot': 'training-only: the first outer fold of each pilot dataset with the outer test rows replaced by every fourth '
                 'outer training row (the outer test fold is never touched), every measure and check except the reference '
                 'predictions; one reproduction probe (run_knn_ablation.reproduction_probe) of the smallest dataset per reference',
        'decision_rule': f'freeze only if the calibrated projection at {WORKERS} single-thread workers is at most {CAP_HOURS} h: the '
                         'simulated first-free-worker makespan of the planned jobs in planned order, each job priced at the '
                         'reference run realized outer fit and predict seconds of its fold and seed 8129 times the piloted '
                         '(job minus uninstrumented fit) / uninstrumented fit ratio (its own ratio for a pilot dataset, the largest '
                         'piloted ratio otherwise), plus the realized seconds once more for the uninstrumented fit of a first fold',
        'wallclock_cap_hours': CAP_HOURS, 'workers': WORKERS, 'max_workers': MAX_WORKERS, 'numeric_threads_per_worker': 1,
        'parallelism': 'one spawned single-thread process per dataset and outer fold under the shared execution lock',
        'failure_policy': 'a failed check fails its job; a failed job cancels the pending jobs and no table is written; failures '
                          'are never omitted; no adaptive stopping on any value',
        'reference_argument': '--reference NAME=RUN_DIR,ABLATION_DIR, one per protocol reference used; further references (for '
                              'example the deduplicated and artificial families) are declared per dataset in a new protocol made '
                              'by draft_protocol(references=...) and frozen after its own pilot',
    }


def draft_protocol(references=None, protocol_id=PROTOCOL_ID, pilot_datasets=PILOT_DATASETS):
    """The unfrozen protocol; the datasets follow the references in name order (a JSON protocol keeps no key order)."""
    references = REFERENCES if references is None else references
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    return _plain({**design(), 'protocol_id': protocol_id, 'production_family': FAMILY, 'datasets': datasets,
                   'references': references, 'pilot_datasets': list(pilot_datasets), 'frozen': False, 'status': DRAFT_STATUS,
                   'resource_decision': 'pending: synthetic smoke and the training-only pilot on ' + ' and '.join(pilot_datasets)})


def reference_of(p):
    """{dataset: reference name}; the references, in name order, must partition the protocol datasets in order."""
    mapping = {}
    for name, entry in sorted(p['references'].items()):
        if (not isinstance(entry, dict) or not entry.get('datasets') or len(set(entry['datasets'])) != len(entry['datasets'])
                or any(not isinstance(entry.get(block), dict) for block in ('run', 'ablation'))):
            raise ValueError(f'Reference {name} must declare datasets, run and ablation pins')
        missing = [f'run.{k}' for k in RUN_PINS if not entry['run'].get(k)] + [f'ablation.{k}' for k in ABLATION_PINS if not entry['ablation'].get(k)]
        if missing:
            raise ValueError(f'Reference {name} lacks {", ".join(missing)}')
        loader = entry.get('loader')
        if loader is not None and (not isinstance(loader, str) or loader.count(':') != 1):
            raise ValueError(f'Reference {name} loader must be module:function or null')
        for dataset in entry['datasets']:
            if dataset in mapping:
                raise ValueError(f'{dataset} belongs to two references')
            mapping[dataset] = name
    if list(mapping) != list(p['datasets']):
        raise ValueError('The references, in name order, must partition the protocol datasets in order')
    if any(name not in mapping for name in p.get('pilot_datasets', [])):
        raise ValueError('Every pilot dataset must be a protocol dataset')
    return mapping


def validate_protocol(p):
    """A production protocol is draft_protocol for its references, or that draft with exactly the freeze fields set by freeze."""
    draft = draft_protocol(references=p.get('references'), protocol_id=p.get('protocol_id'),
                           pilot_datasets=p.get('pilot_datasets', ()))
    strip = lambda q: {k: v for k, v in q.items() if k not in FREEZE_FIELDS}
    if strip(p) != strip(draft):
        differing = sorted(k for k in set(p) | set(draft) if k not in FREEZE_FIELDS and p.get(k) != draft.get(k))
        raise ValueError(f'The protocol differs from training_diagnostics.draft_protocol() in {", ".join(differing)}')
    reference_of(p)
    if not p.get('frozen'):
        if p != draft:
            raise ValueError('An unfrozen protocol must equal the draft')
        return p
    projection = p.get('pilot_projection') or {}
    hours = projection.get('decision_hours')
    if (p.get('status') != FROZEN_STATUS or not p.get('frozen_at_utc') or not p.get('resource_decision')
            or projection.get('cap_hours') != CAP_HOURS or projection.get('workers') != WORKERS
            or isinstance(hours, bool) or not isinstance(hours, (int, float)) or not 0 < hours <= CAP_HOURS):
        raise ValueError(f'A frozen protocol records its freeze and a pilot projection within the {CAP_HOURS} h cap at {WORKERS} workers')
    return p


# ----------------------------------------------------------------------------- pure numpy diagnostics

def network_forward(positions, matrices):
    """The core's sort-layer rule on position vectors (SortFlowHybridNetwork._forward_propagate_batch on CPU): per layer the
    cityblock responses of every row to every filter (rows x filters), the ranking by ascending response with ties by
    ascending filter index (score_order), and its inverse positions, which feed the next layer. [(responses, ranking,
    positions)] per layer; the output layer's ranking[:, 0] is the output rule's class index."""
    layers, current = [], np.asarray(positions)
    for matrix in matrices:
        responses = cdist(current, np.asarray(matrix, dtype=float), metric='cityblock')
        ranking = score_order(responses)
        current = inverse_positions(ranking)
        layers.append((responses, ranking, current))
    return layers


def displacement(current, reference):
    """(mean over filters of footrule(filter, reference filter) / floor(n^2 / 2), share of filters that differ). A layer's
    index matrix holds each filter's item positions (rows filters, columns items), so the footrule is row-wise cityblock."""
    current, reference = np.asarray(current, dtype=float), np.asarray(reference, dtype=float)
    if current.shape != reference.shape or current.ndim != 2 or current.shape[1] < 2:
        raise ValueError('Expected two filter position matrices of the same layer')
    n = current.shape[1]
    footrule = np.abs(current - reference).sum(axis=1)
    return float(np.mean(footrule / (n * n // 2))), float(np.mean(np.any(current != reference, axis=1)))


def response_ties(responses):
    """(share of (row, filter) responses equal to another filter's response in the same row, mean over rows of the number of
    distinct responses / number of filters)."""
    ordered = np.sort(np.asarray(responses), axis=1)
    equal = ordered[:, 1:] == ordered[:, :-1]
    tied = np.zeros(ordered.shape, dtype=bool)
    tied[:, 1:] |= equal
    tied[:, :-1] |= equal
    return float(tied.mean()), float(np.mean((1 + np.sum(~equal, axis=1)) / ordered.shape[1]))


def nearest_ties(responses):
    """Share of rows whose smallest response is attained by more than one filter (the output layer's tied nearest class)."""
    ordered = np.sort(np.asarray(responses), axis=1)
    return float(np.mean(ordered[:, 1] == ordered[:, 0]))


def relabel_permutations(seed, identity, view, draw, widths):
    """One uniform permutation per hidden layer (new filter ID = permutation[old ID]), seeded per job, view, draw and layer."""
    return [np.random.RandomState(derive_seed(int(seed), identity['dataset_id'], int(identity['outer_repeat']),
                                              int(identity['outer_fold']), int(identity['model_seed']), int(view), int(draw),
                                              layer)).permutation(int(width))
            for layer, width in enumerate(widths)]


def relabel_matrices(matrices, permutations):
    """Hidden layer l's filter IDs relabeled by permutations[l]: row j of layer l holds the old filter argsort(permutation)[j],
    and the columns of layer l + 1, whose items are layer l's filter IDs, are relabeled the same way. Without ties every
    response is unchanged up to the relabeling; with ties the rankings completed by filter ID can change."""
    out = []
    for layer, matrix in enumerate(matrices):
        relabeled = np.asarray(matrix)
        if 0 < layer <= len(permutations):
            relabeled = relabeled[:, np.argsort(permutations[layer - 1])]
        if layer < len(permutations):
            relabeled = relabeled[np.argsort(permutations[layer])]
        out.append(relabeled)
    return out


def sample_arrays(samples):
    """(positions, class indices) of core samples [[items '1'..'D' in order], class index, weight]."""
    if not samples:
        return np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64)
    orders = np.asarray([[int(item) - 1 for item in sample[0]] for sample in samples], dtype=np.int64)
    return inverse_positions(orders), np.asarray([int(sample[1]) for sample in samples], dtype=np.int64)


def output_error(validation, matrices):
    """The core's evaluate error: misclassified samples / samples, the class being the nearest output filter."""
    positions, labels = validation
    predicted = network_forward(positions, matrices)[-1][1][:, 0]
    return int(np.sum(predicted != labels)) / len(labels)


def scheduled_iterations(iterations, every=SNAPSHOT_EVERY):
    return sorted(set(range(0, int(iterations) + 1, int(every))) | {int(iterations)})


def previous_iteration(iteration, schedule):
    earlier = [t for t in schedule if t < iteration]
    return earlier[-1] if earlier else None


def checkpoint_from_curve(initial, errors):
    """(checkpoint iteration, strict improvement iterations, running minimum at 0..T) of a validation curve: the checkpoint is
    the last iteration whose error is strictly below the running minimum before it, 0 when none is."""
    best, running, improvements = float(initial), [float(initial)], []
    for t, error in enumerate(errors, 1):
        if error < best:
            best = float(error)
            improvements.append(t)
        running.append(best)
    return (improvements[-1] if improvements else 0), improvements, running


def numpy_states_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and tuple(a[2:]) == tuple(b[2:])


def filter_copy(network, orders=True):
    """Copies of every layer's index matrix (hidden layers, then the output layer) and, with orders, adjacency orders."""
    layers = [network.graph.vertex_list[f'{network.id}_ly{i}'] for i in range(network.graph.num_vertices)]
    return {'matrices': tuple(np.array(layer.index_matrix, dtype=float, copy=True) for layer in layers),
            'orders': (tuple(tuple(tuple(vertex.adjacency_list) for vertex in layer.graph.vertex_list.values()) for layer in layers)
                       if orders else None)}


def copies_equal(a, b, orders=True):
    if len(a['matrices']) != len(b['matrices']) or any(not np.array_equal(x, y) for x, y in zip(a['matrices'], b['matrices'])):
        return False
    return not orders or (a['orders'] is not None and a['orders'] == b['orders'])


# ----------------------------------------------------------------------------- instrumentation

class TrainingRecorder:
    """Instance-level update_network wrapper of one view network: pure, RNG-guarded reads after each core update."""

    def __init__(self, net, *, every=SNAPSHOT_EVERY):
        network = getattr(net, 'network_', None)
        if network is None or network.update_iter != 0:
            raise ValueError('Install the recorder on an initialized, untrained view network')
        if network.eval_period != 1:
            raise CheckFailed('The core does not evaluate its validation rows after every update (eval_period != 1)')
        if 'update_network' in vars(network):
            raise ValueError('The network already carries an instance-level update_network')
        self.net, self.network, self.every, self.iterations = net, network, int(every), int(net.iterations)
        self.rng_checks = []
        with self._guard():
            self.initial = filter_copy(network)
        self.snapshots = {0: self.initial}
        self.errors, self.running, self.replaced, self.update_iters = [], [], [], []
        self.initial_error = self.first_update_matches = self.final = self.validation = None
        self.checkpoint_copy, self.checkpoint_copy_iteration = self.initial, 0

    @contextmanager
    def _guard(self):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            yield
        finally:
            unchanged = numpy_states_equal(numpy_state, np.random.get_state()) and python_state == random.getstate()
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            self.rng_checks.append(bool(unchanged))

    @contextmanager
    def installed(self):
        original, recorder = self.network.update_network, self

        def update_network(data_train, data_validation=None, train_type='supervised', problem='classification'):
            return recorder._update(original, data_train, data_validation, train_type, problem)

        self.network.update_network = update_network
        try:
            yield self
        finally:
            del self.network.update_network
        with self._guard():
            self.final = filter_copy(self.network)

    def _update(self, original, data_train, data_validation, train_type, problem):
        network = self.network
        if not self.update_iters:                       # the core has just evaluated the initial filters
            with self._guard():
                self.initial_error = float(network.min_error_val)
                self.first_update_matches = copies_equal(filter_copy(network), self.initial)
                self.validation = sample_arrays(data_validation or [])
        replaced_before = network.optimal_model
        output = original(data_train, data_validation, train_type, problem)
        with self._guard():
            t = int(network.update_iter)
            self.update_iters.append(t)
            self.errors.append(float(output[1][1]))
            self.running.append(float(network.min_error_val))
            replaced = network.optimal_model is not replaced_before
            self.replaced.append(bool(replaced))
            copied = None
            if replaced:
                copied = filter_copy(network)
                self.checkpoint_copy, self.checkpoint_copy_iteration = copied, t
            if t % self.every == 0 or t == self.iterations:
                self.snapshots[t] = copied if copied is not None else filter_copy(network, orders=False)
        return output

    def verify(self):
        """The checkpoint derived from the core's validation curve and the training-record checks."""
        if not self.update_iters or self.final is None:
            raise CheckFailed('The recorder saw no completed training')
        T, network = self.iterations, self.network
        checkpoint, improvements, running = checkpoint_from_curve(self.initial_error, self.errors)
        n_validation = len(self.validation[1])
        record = {'updates_in_order': self.update_iters == list(range(1, T + 1)) and network.update_iter == T,
                  'initial_state_at_first_update': self.first_update_matches is True,
                  'validation_rows': n_validation > 0,
                  'running_minimum': self.running == running[1:],
                  'checkpoint_replacements': [t for t, flag in zip(self.update_iters, self.replaced) if flag] == improvements}
        orders = {'final_equals_checkpoint_copy': copies_equal(self.final, self.checkpoint_copy),
                  'copy_iteration_is_checkpoint': self.checkpoint_copy_iteration == checkpoint,
                  'final_minimum_is_checkpoint_error': float(network.min_error_val) == running[-1]}
        return {'checkpoint': checkpoint, 'improvements': improvements, 'running': running, 'n_validation': n_validation,
                'training_record': {'passed': all(record.values()), **record},
                'checkpoint_orders': {'passed': all(orders.values()), **orders}}


def instrumented_fit(params, seed, X, y, *, every=SNAPSHOT_EVERY):
    """MultiViewArrowFlowKNN(**params, seed=seed).fit(X, y), step for step (MultiViewArrowFlowKNN.fit, MultiViewArrowFlow.fit
    and ArrowFlowEstimator.fit_orders = initialize_orders then train_initialized), with a TrainingRecorder installed on each
    view network for its training only. Returns the fitted model and the seven recorders."""
    model = MultiViewArrowFlowKNN(**params, seed=seed)
    model.readouts_, model.readout_selections_ = [], []
    model.readout_seconds_ = 0.
    model.classes_ = np.unique(y)
    model.views_ = []
    encoding = training = 0.
    recorders = []
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
                                 p_correct=model.p_correct, seed=seed_v,
                                 validation_ratio=model.validation_ratio, augment=model.augment,
                                 n_augmentations=model.n_augmentations, max_swaps=model.max_swaps)
        net.initialize_orders(orders, y)
        recorder = TrainingRecorder(net, every=every)
        with recorder.installed():
            net.train_initialized(orders, y)
        training += net.training_seconds_
        model.views_.append((enc, net))
        model._fit_view_readout(enc, net, orders, y, seed_v)
        recorders.append(recorder)
    model.encoding_seconds_ = encoding
    model.training_seconds_ = training
    return model, recorders


def reference_check(predictions, sealed, seed):
    check = base.reproduction_check(np.asarray(predictions), sealed, seed)
    return {'performed': True, 'passed': bool(check['reproduced']), **check}


def accuracy(predicted, truth):
    return float(np.mean(np.asarray(predicted) == np.asarray(truth)))


def diagnose_view(v, view, recorder, selection, train_orders, y_train, query_orders, y_query, final_knn, identity, p):
    """Every measure of one trained view from the recorder's copies, in pure numpy; (record, arrays, checks)."""
    enc, net = view
    L, T = len(net.widths), int(net.iterations)
    schedule = scheduled_iterations(T, p['snapshots']['every'])
    verified = recorder.verify()
    c = verified['checkpoint']
    checks = {'training_record': verified['training_record'], 'checkpoint_orders': verified['checkpoint_orders'],
              'snapshot_schedule': {'passed': sorted(recorder.snapshots) == schedule, 'captured': sorted(recorder.snapshots)}}
    names = [f'hidden_{layer}' for layer in range(L)] + ['output']
    initial = recorder.initial['matrices']
    train_positions, query_positions = inverse_positions(train_orders), inverse_positions(query_orders)
    config = selection['config']

    def core_error(t):
        return recorder.initial_error if t == 0 else recorder.errors[t - 1]

    def evaluate(matrices, keep_layers):
        train_layers = network_forward(train_positions, matrices)
        query_layers = network_forward(query_positions, matrices)
        knn = StableFootruleKNN(**config, input_kind='positions').fit(train_layers[L - 1][2], y_train).predict(query_layers[L - 1][2])
        out = {'knn': np.asarray(knn), 'output': net.classes_[query_layers[L][1][:, 0]],
               'output_train': net.classes_[train_layers[L][1][:, 0]], 'validation_error': output_error(recorder.validation, matrices)}
        if keep_layers:
            out.update(train_layers=train_layers, query_layers=query_layers)
        return out

    def snapshot_record(kind, t, matrices, evaluated):
        previous = previous_iteration(t, schedule)
        layers = []
        for layer, name in enumerate(names):
            d0, s0 = displacement(matrices[layer], initial[layer])
            dp, sp = (None, None) if previous is None else displacement(matrices[layer], recorder.snapshots[previous]['matrices'][layer])
            layers.append({'layer': name, 'n_filters': int(matrices[layer].shape[0]), 'n_items': int(matrices[layer].shape[1]),
                           'displacement_from_initial': d0, 'changed_share_from_initial': s0,
                           'displacement_from_previous': dp, 'changed_share_from_previous': sp})
        return {'snapshot': kind, 'iteration': t, 'previous_iteration': previous, 'layers': layers,
                'knn_accuracy': accuracy(evaluated['knn'], y_query), 'output_rule_accuracy': accuracy(evaluated['output'], y_query),
                'output_rule_training_accuracy': accuracy(evaluated['output_train'], y_train),
                'validation_error': evaluated['validation_error'], 'core_validation_error': core_error(t)}

    snapshots, knn_scheduled, output_scheduled = [], [], []
    initial_eval = checkpoint_eval = None
    for t in schedule:
        matrices = recorder.snapshots[t]['matrices']
        evaluated = evaluate(matrices, keep_layers=t in (0, c))
        snapshots.append(snapshot_record('scheduled', t, matrices, evaluated))
        knn_scheduled.append(evaluated['knn'])
        output_scheduled.append(evaluated['output'])
        if t == 0:
            initial_eval = evaluated
        if t == c:
            checkpoint_eval = evaluated
    checkpoint_matrices = recorder.checkpoint_copy['matrices']
    if checkpoint_eval is None:
        checkpoint_eval = evaluate(checkpoint_matrices, keep_layers=True)
    snapshots.append(snapshot_record('checkpoint', c, checkpoint_matrices, checkpoint_eval))
    checks['validation_error_recomputed'] = {'passed': all(s['validation_error'] == s['core_validation_error'] for s in snapshots),
                                             'snapshots': len(snapshots)}

    depth_train, depth_query = net.transform_orders_by_depth(train_orders), net.transform_orders_by_depth(query_orders)
    forward = {'hidden_training_rows': all(np.array_equal(checkpoint_eval['train_layers'][l][2], depth_train[l]) for l in range(L)),
               'hidden_query_rows': all(np.array_equal(checkpoint_eval['query_layers'][l][2], depth_query[l]) for l in range(L)),
               'output_rule_query_rows': bool(np.array_equal(checkpoint_eval['output'], net.predict_orders(query_orders)))}
    checks['forward_reimplementation'] = {'passed': all(forward.values()), **forward}

    def tie_rows(evaluated):
        rows = []
        for layer, name in enumerate(names):
            responses = evaluated['query_layers'][layer][0]
            tied = distinct = nearest = None
            if layer < L:
                tied, distinct = response_ties(responses)
            else:
                nearest = nearest_ties(responses)
            rows.append({'layer': name, 'n_rows': int(responses.shape[0]), 'n_filters': int(responses.shape[1]),
                         'tied_response_share': tied, 'distinct_response_ratio': distinct, 'tied_nearest_share': nearest})
        return rows

    ties = {'initial': {'iteration': 0, 'layers': tie_rows(initial_eval)}, 'checkpoint': {'iteration': c, 'layers': tie_rows(checkpoint_eval)}}

    def relabeled_knn(permutations):
        matrices = relabel_matrices(checkpoint_matrices[:L], permutations)
        train_hidden = network_forward(train_positions, matrices)[-1][2]
        query_hidden = network_forward(query_positions, matrices)[-1][2]
        return np.asarray(StableFootruleKNN(**config, input_kind='positions').fit(train_hidden, y_train).predict(query_hidden))

    readout = {'knn_equals_final': bool(np.array_equal(checkpoint_eval['knn'], final_knn)),
               'identity_relabel_unchanged': bool(np.array_equal(relabeled_knn([np.arange(w) for w in net.widths]), checkpoint_eval['knn']))}
    checks['checkpoint_readout'] = {'passed': all(readout.values()), **readout}
    base_accuracy = accuracy(checkpoint_eval['knn'], y_query)
    relabel_predictions, draws = [], []
    for draw in range(int(p['relabel']['draws'])):
        predicted = relabeled_knn(relabel_permutations(p['relabel']['seed'], identity, v, draw, net.widths))
        relabel_predictions.append(predicted)
        draws.append({'draw': draw, 'changed_share': float(np.mean(predicted != checkpoint_eval['knn'])),
                      'accuracy': accuracy(predicted, y_query), 'accuracy_change': accuracy(predicted, y_query) - base_accuracy})
    record = {'view': v, 'strategy': enc.strategy, 'view_seed': int(net.seed), 'embed_dim': int(net.embed_dim), 'degree': int(enc.degree),
              'widths': [int(w) for w in net.widths], 'iterations': T, 'n_classes': int(len(net.classes_)),
              'n_training_rows': int(len(y_train)), 'n_query_rows': int(len(y_query)), 'n_validation_samples': verified['n_validation'],
              'n_training_samples': int(net.training_sample_count_), 'readout': {'config': config, 'config_id': selection['config_id']},
              'validation_curve': {'initial': recorder.initial_error, 'after_update': list(recorder.errors),
                                   'running_minimum': verified['running']},
              'checkpoint': {'iteration': c, 'improvement_iterations': verified['improvements'],
                             'initial_validation_error': recorder.initial_error, 'validation_error': verified['running'][-1],
                             'returned_initial_filters': c == 0},
              'scheduled_iterations': schedule, 'snapshots': snapshots, 'ties': ties,
              'relabel': {'checkpoint_accuracy': base_accuracy, 'draws': draws}}
    arrays = {'knn_scheduled': np.stack(knn_scheduled), 'output_scheduled': np.stack(output_scheduled),
              'knn_checkpoint': checkpoint_eval['knn'], 'output_checkpoint': checkpoint_eval['output'],
              'relabel_knn': np.stack(relabel_predictions)}
    return record, arrays, checks


VIEW_CHECKS = ('training_record', 'checkpoint_orders', 'snapshot_schedule', 'validation_error_recomputed', 'forward_reimplementation',
               'checkpoint_readout')
JOB_CHECKS = ('reference_predictions', 'uninstrumented_state_hash', 'rng_untouched', 'wrapper_removed',
              'network_unchanged_by_diagnostics') + VIEW_CHECKS


def evaluate_job(X, y, split, job, sealed, p, *, dataset_hash, code_revision, protocol_hash, query=None):
    """One dataset, outer fold and seed: the instrumented seven-view fit, the checks and every measure. query=None scores the
    outer test rows; the pilot passes training rows instead (training-only timing, no reference-prediction check).
    Returns (record, arrays); arrays is None when the job failed."""
    seed = int(p['model_seed'])
    train = [int(i) for i in split['train']]
    rows = [int(i) for i in (split['test'] if query is None else query)]
    X_train, y_train, X_query, y_query = X[train], y[train], X[rows], y[rows]
    identity = {'dataset_id': job['dataset_id'], 'reference': job['reference'], 'dataset_hash': dataset_hash,
                'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'], 'split_hash': config_id(split),
                'query': 'outer_test_rows' if query is None else 'training_rows_for_timing', 'query_ids_hash': array_hash(np.asarray(rows)),
                'config_id': job['config_id'], 'config': job['config'], 'selected': job['selected'], 'model_seed': seed,
                'check_uninstrumented': bool(job['check_uninstrumented']), 'code_revision': code_revision, 'protocol_hash': protocol_hash}
    checks, timing = {}, {}
    result = {'status': 'running', 'identity': identity, 'checks': checks, 'timing': timing}
    arrays = None
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            start = time.perf_counter()
            seed_fit(seed)
            model, recorders = instrumented_fit(job['selected'], seed, X_train, y_train, every=p['snapshots']['every'])
            views, query_orders = model.predict_views(X_query)
            knn_views = np.stack([np.asarray(view) for view in views])
            prediction = majority(knn_views)
            timing['instrumented_fit_seconds'] = time.perf_counter() - start
            if query is None:
                checks['reference_predictions'] = reference_check(prediction, sealed, seed)
                if not checks['reference_predictions']['passed']:
                    raise CheckFailed(f'the 7-view predictions differ from the reference on {checks["reference_predictions"]["n_differing"]} '
                                      f'of {checks["reference_predictions"]["n_test"]} outer test rows')
            else:
                checks['reference_predictions'] = {'performed': False, 'passed': None,
                                                   'reason': 'training-only pilot: the query rows are outer training rows'}
            hashes = [net.state_hash() for _, net in model.views_]
            checks['wrapper_removed'] = {'performed': True,
                                         'passed': all('update_network' not in vars(net.network_) for _, net in model.views_)}
            if job['check_uninstrumented']:
                start = time.perf_counter()
                seed_fit(seed)
                plain = MultiViewArrowFlowKNN(**job['selected'], seed=seed).fit(X_train, y_train)
                plain_views = np.stack([np.asarray(view) for view in plain.predict_views(X_query)[0]])
                plain_hashes = [net.state_hash() for _, net in plain.views_]
                timing['uninstrumented_fit_seconds'] = time.perf_counter() - start
                equal_views = bool(np.array_equal(plain_views, knn_views))
                checks['uninstrumented_state_hash'] = {'performed': True, 'passed': plain_hashes == hashes and equal_views,
                                                       'view_state_hashes': hashes, 'uninstrumented_view_state_hashes': plain_hashes,
                                                       'view_predictions_equal': equal_views}
                del plain
                if not checks['uninstrumented_state_hash']['passed']:
                    raise CheckFailed('a view state hash or kNN prediction differs from the uninstrumented fit')
            else:
                timing['uninstrumented_fit_seconds'] = 0.
                checks['uninstrumented_state_hash'] = {'performed': False, 'passed': None,
                                                       'reason': 'performed on the first outer fold of each dataset'}
            start = time.perf_counter()
            records, view_arrays, per_view = [], [], defaultdict(list)
            for v, (view, recorder, selection) in enumerate(zip(model.views_, recorders, model.readout_selections_)):
                record, arrays_v, checks_v = diagnose_view(v, view, recorder, selection, view[0].transform(X_train), y_train,
                                                           query_orders[v], y_query, knn_views[v], identity, p)
                records.append(record)
                view_arrays.append(arrays_v)
                for name in VIEW_CHECKS:
                    per_view[name].append(checks_v[name])
            for name in VIEW_CHECKS:
                checks[name] = {'performed': True, 'passed': all(entry['passed'] for entry in per_view[name]), 'views': per_view[name]}
            schedules = {tuple(record['scheduled_iterations']) for record in records}
            knn_scheduled = np.stack([a['knn_scheduled'] for a in view_arrays])
            knn_checkpoint = np.stack([a['knn_checkpoint'] for a in view_arrays])
            relabel_knn = np.stack([a['relabel_knn'] for a in view_arrays])
            majority_checkpoint = majority(knn_checkpoint)
            same_majority = bool(np.array_equal(majority_checkpoint, prediction)) and len(schedules) == 1
            checks['checkpoint_readout']['majority_equals_predictions'] = same_majority
            checks['checkpoint_readout']['passed'] = checks['checkpoint_readout']['passed'] and same_majority
            schedule = records[0]['scheduled_iterations']
            base_accuracy = accuracy(majority_checkpoint, y_query)
            majority_draws = []
            for draw in range(relabel_knn.shape[1]):
                predicted = majority(relabel_knn[:, draw])
                majority_draws.append({'draw': draw, 'changed_share': float(np.mean(predicted != majority_checkpoint)),
                                       'accuracy': accuracy(predicted, y_query), 'accuracy_change': accuracy(predicted, y_query) - base_accuracy})
            majority_record = {
                'snapshots': [{'snapshot': 'scheduled', 'iteration': t, 'knn_accuracy': accuracy(majority(knn_scheduled[:, s]), y_query)}
                              for s, t in enumerate(schedule)] + [{'snapshot': 'checkpoint', 'iteration': None, 'knn_accuracy': base_accuracy}],
                'relabel': {'checkpoint_accuracy': base_accuracy, 'draws': majority_draws}}
            guarded = [flag for recorder in recorders for flag in recorder.rng_checks]
            checks['rng_untouched'] = {'performed': True, 'passed': bool(guarded) and all(guarded), 'guarded_reads': len(guarded),
                                       'unchanged': int(sum(guarded))}
            checks['network_unchanged_by_diagnostics'] = {'performed': True,
                                                          'passed': [net.state_hash() for _, net in model.views_] == hashes}
            timing['diagnostics_seconds'] = time.perf_counter() - start
            arrays = {'query_rows': np.asarray(rows), 'query_labels': np.asarray(y_query), 'predictions': np.asarray(prediction),
                      'scheduled_iterations': np.asarray(schedule), 'knn_scheduled': knn_scheduled,
                      'output_scheduled': np.stack([a['output_scheduled'] for a in view_arrays]), 'knn_checkpoint': knn_checkpoint,
                      'output_checkpoint': np.stack([a['output_checkpoint'] for a in view_arrays]), 'relabel_knn': relabel_knn}
        failed = sorted(name for name, check in checks.items() if check['performed'] and not check['passed'])
        if failed:
            raise CheckFailed('checks failed: ' + ', '.join(failed))
        result.update(status='ok', views=records, majority=majority_record,
                      accuracy={'majority_checkpoint': base_accuracy, 'views_checkpoint': [r['relabel']['checkpoint_accuracy'] for r in records]})
    except Exception as exc:
        result.update(status='failed', exception=f'{type(exc).__name__}: {exc}', check_failure=isinstance(exc, CheckFailed))
        arrays = None
    timing['job_seconds'] = time.perf_counter() - started
    return native(result), arrays


# ----------------------------------------------------------------------------- references, datasets and the plan

def parse_reference(text):
    """NAME=RUN_DIR,ABLATION_DIR -> (NAME, (RUN_DIR, ABLATION_DIR))."""
    name, separator, directories = str(text).partition('=')
    run_dir, comma, ablation_dir = directories.partition(',')
    if not name or not separator or not comma or not run_dir or not ablation_dir:
        raise argparse.ArgumentTypeError('--reference takes NAME=RUN_DIR,ABLATION_DIR')
    return name, (Path(run_dir), Path(ablation_dir))


def ablation_pins(directory, summary_file=None):
    """The protocol pins of one component ablation run directory."""
    directory = Path(directory)
    if summary_file is None:
        present = [name for name in ABLATION_SUMMARIES if (directory/name).is_file()]
        if len(present) != 1:
            raise ValueError(f'{directory} holds no single ablation summary ({", ".join(ABLATION_SUMMARIES)})')
        summary_file = present[0]
    protocol = json.loads((directory/'protocol.json').read_text())
    saved_environment = json.loads((directory/'environment.json').read_text())
    return {'family': protocol.get('production_family'), 'protocol_id': protocol.get('protocol_id'),
            'code_revision': saved_environment.get('code_revision'), 'protocol_sha256': sha256_file(directory/'protocol.json'),
            'manifest_sha256': sha256_file(directory/'manifest.json'),
            'reference_selections_sha256': sha256_file(directory/'reference_selections.json'),
            'summary_file': summary_file, 'summary_sha256': sha256_file(directory/summary_file)}


def load_ablation(directory, declared, run, datasets, p, *, allow_smoke=False):
    """The ablation run that sealed the reference's selections: its pins, design, anchoring on the run (a manifest reference
    entry with the run's file hashes holding the datasets) and a summary recording views7 reproducing every fold and seed."""
    directory = Path(directory)
    missing = [name for name in ABLATION_FILES + (declared['summary_file'],) if not (directory/name).is_file()]
    if missing:
        raise ValueError(f'The ablation source {directory} is not a complete run (missing {", ".join(missing)})')
    protocol = json.loads((directory/'protocol.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    if not allow_smoke and (not protocol.get('frozen') or manifest.get('purpose') != 'confirmatory'):
        raise ValueError('The ablation source must be a frozen confirmatory run')
    observed = ablation_pins(directory, declared['summary_file'])
    wrong = [f'{key}: declared {declared.get(key)!r}, ablation has {value!r}' for key, value in observed.items() if declared.get(key) != value]
    if wrong:
        raise ValueError('The ablation source does not match the protocol pins: ' + '; '.join(wrong))
    for key in base.DESIGN_KEYS:
        if protocol[key] != p[key]:
            raise ValueError(f'Ablation and diagnostics protocols disagree on {key}')
    entries = [manifest['reference_source']] if 'reference_source' in manifest else list(manifest.get('reference_sources', {}).values())
    anchored = [entry for entry in entries if entry.get('file_sha256') == run['files']]
    if len(anchored) != 1:
        raise ValueError('The ablation run is not anchored on this reference run (no manifest entry carries its file hashes)')
    held = anchored[0].get('datasets', manifest.get('datasets', []))
    if any(name not in held for name in datasets):
        raise ValueError('The ablation run does not hold every dataset of this reference')
    summary = json.loads((directory/declared['summary_file']).read_text())
    total = p['outer_folds'] * p['outer_repeats'] * len(p['fit_seeds'])
    for name in datasets:
        if summary['summaries'][name]['views7_reproduces_reference'] != {'matching_fold_seeds': total, 'total_fold_seeds': total}:
            raise ValueError(f'{name}: the ablation summary does not record views7 reproducing every fold and seed')
    selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
                  for s in json.loads((directory/'reference_selections.json').read_text())}
    jobs = {(j['dataset_id'], j['outer_repeat'], j['outer_fold']): j for j in json.loads((directory/'planned_jobs.json').read_text())}
    return {'directory': directory, 'observed': observed, 'selections': selections, 'jobs': jobs}


def load_references(p, sources, *, allow_smoke=False):
    """{name: verified reference} for {name: (run_dir, ablation_dir)}: the run is run_knn_ablation.load_reference with
    check_reference against the protocol's run pins, and the ablation is load_ablation against its pins."""
    unknown = sorted(set(sources) - set(p['references']))
    if unknown:
        raise ValueError(f'Unknown references {unknown}; the protocol declares {sorted(p["references"])}')
    references = {}
    for name in [n for n in p['references'] if n in sources]:
        declared = p['references'][name]
        run_dir, ablation_dir = (Path(d) for d in sources[name])
        run = base.load_reference(run_dir, allow_smoke=allow_smoke)
        observed = base.check_reference(dict(p, reference_source=declared['run'], datasets=list(declared['datasets'])), run)
        ablation = load_ablation(ablation_dir, declared['ablation'], run, declared['datasets'], p, allow_smoke=allow_smoke)
        references[name] = {'name': name, 'run': run, 'ablation': ablation, 'datasets': list(declared['datasets']),
                            'loader': declared.get('loader'), 'observed_run': observed}
    return references


def fresh_load(loader, name, X, y, manifest):
    """A fresh load through the reference's loader must return the prepared arrays and dataset hash; a loader module with
    PIN_BY_NAME (newdata) must also pin the prepared dataset and splits hashes."""
    module_name, function = loader.split(':')
    module = importlib.import_module(module_name)
    loaded_X, loaded_y, loaded = getattr(module, function)(name)
    if (not (np.array_equal(np.asarray(loaded_X, dtype=float), np.asarray(X, dtype=float), equal_nan=True)
             and np.array_equal(np.asarray(loaded_y), np.asarray(y))) or loaded['dataset_hash'] != manifest['dataset_hash']):
        raise ValueError(f'{name}: a fresh load by {loader} differs from the reference prepared data')
    pin = getattr(module, 'PIN_BY_NAME', {}).get(name)
    if pin is not None and (pin['dataset_hash'], pin['splits_hash']) != (manifest['dataset_hash'], manifest['splits_hash']):
        raise ValueError(f'{name}: the prepared dataset or splits hash differs from {module_name}.PIN_BY_NAME')
    return {'loader': loader, 'dataset_hash': loaded['dataset_hash'], 'pinned_splits_hash': None if pin is None else pin['splits_hash']}


def prepare_dataset(output, reference, name, p, *, production):
    X, y, manifest, splits = load_prepared(reference['run']['directory'], name)
    if (splits != make_splits(y, p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['split_seed'])
            or config_id(splits) != manifest['splits_hash']):
        raise ValueError(f'{name}: the reference splits are not the declared nested splits')
    copy_X, copy_y, copy_manifest, copy_splits = load_prepared(reference['ablation']['directory'], name)
    if not (np.array_equal(copy_X, X, equal_nan=True) and np.array_equal(copy_y, y)) or copy_manifest != manifest or copy_splits != splits:
        raise ValueError(f'{name}: the ablation copy of the dataset differs from the reference run')
    if production and not reference['loader']:
        raise ValueError(f'{name}: a production reference must declare its dataset loader')
    identity = {'dataset_hash': manifest['dataset_hash'], 'splits_hash': manifest['splits_hash'], 'rows': int(len(y)),
                'features': int(X.shape[1]), 'fresh_load': fresh_load(reference['loader'], name, X, y, manifest) if production else None}
    write_json(output/name/'manifest.json', manifest)
    write_json(output/name/'splits.json', splits)
    if not (output/name/'data.npz').exists():
        shutil.copyfile(Path(reference['run']['directory'])/name/'data.npz', output/name/'data.npz')
    copied_X, copied_y, _, _ = load_prepared(output, name)
    if not (np.array_equal(copied_X, X, equal_nan=True) and np.array_equal(copied_y, y)):
        raise ValueError(f'{name}: the copied dataset differs from the reference run')
    return X, y, manifest, splits, identity


def planned_job(p, reference, name, index, split, record, n_features):
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    sealed_job = reference['ablation']['jobs'].get((name, split['outer_repeat'], split['outer_fold']))
    if sealed_job is None or sealed_job['config_id'] != record['config_id'] or sealed_job['selected'] != selected:
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the resolved selection differs from the ablation plan')
    seed = str(p['model_seed'])
    return {'dataset_id': name, 'reference': reference['name'], 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
            'stem': f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}', 'config_id': record['config_id'],
            'config': record['config'], 'selected': selected, 'selected_widths': list(selected['widths']),
            'model_seed': int(p['model_seed']), 'check_uninstrumented': index == 0,
            'reference_prediction_hash': record['reference_prediction_hashes'][seed],
            'reference_outer_seconds': record['reference_outer_seconds'][seed]}


def prepare(output, p, sources, datasets=None, *, allow_smoke=False, purpose='diagnostics'):
    """Verify the references, copy the prepared data, reconstruct every selection (it must equal the sealed record and resolve
    to the ablation plan) and seal the planned jobs."""
    output = Path(output)
    mapping = reference_of(p)
    chosen = list(p['datasets']) if datasets is None else list(datasets)
    if not chosen or len(set(chosen)) != len(chosen) or any(name not in mapping for name in chosen):
        raise ValueError('The datasets must be distinct protocol datasets')
    chosen = [name for name in p['datasets'] if name in chosen]
    needed = [name for name in p['references'] if any(mapping[d] == name for d in chosen)]
    missing = [name for name in needed if name not in sources]
    if missing:
        raise ValueError(f'Supply --reference for {", ".join(missing)}')
    references = load_references(p, {name: sources[name] for name in needed}, allow_smoke=allow_smoke)
    write_json(output/'protocol.json', p)
    write_json(output/'environment.json', environment())
    selections, jobs, identities = [], [], {}
    for name in chosen:
        reference = references[mapping[name]]
        X, y, manifest, splits, identities[name] = prepare_dataset(output, reference, name, p, production=not allow_smoke)
        for index, split in enumerate(splits):
            validate_split(split, len(y))
            record = base.selection_record(reference['run'], name, split, y, manifest)
            if record != reference['ablation']['selections'].get((name, split['outer_repeat'], split['outer_fold'])):
                raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the selection reconstructed from the '
                                 'reference run differs from the sealed record')
            selections.append(record)
            jobs.append(planned_job(p, reference, name, index, split, record, X.shape[1]))
    write_json(output/'reference_selections.json', selections)
    write_json(output/'planned_jobs.json', jobs)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'protocol_id': p['protocol_id'], 'protocol_hash': config_id(p), 'datasets': chosen,
        'model_seed': p['model_seed'], 'planned_jobs': len(jobs), 'planned_jobs_sha256': sha256_file(output/'planned_jobs.json'),
        'reference_selections_sha256': sha256_file(output/'reference_selections.json'), 'dataset_identity': identities,
        'references': {name: {'run_directory': str(Path(sources[name][0]).resolve()),
                              'ablation_directory': str(Path(sources[name][1]).resolve()),
                              'run': reference['observed_run'], 'ablation': reference['ablation']['observed'],
                              'run_file_sha256': reference['run']['files'],
                              'datasets': [d for d in chosen if mapping[d] == name]} for name, reference in references.items()}})
    return jobs, references


def verify(output, *, allow_smoke=False):
    """The sealed run directory: protocol (frozen unless a synthetic smoke), manifest seal, unchanged scientific sources,
    planned jobs and sealed selections."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    if not smoke:
        validate_protocol(p)
        if not p['frozen'] or manifest.get('purpose') != 'diagnostics':
            raise ValueError('A diagnostics run directory with a frozen reviewed protocol is required')
    if manifest['protocol_hash'] != config_id(p):
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
    if record_path.exists() or artifact_path.exists():
        raise FileExistsError(f'Existing diagnostics job {stem}')
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
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open('x') as stream:
        stream.write(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + '\n')
    return str(record_path), result['status']


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


def uncommitted_sources(extra=()):
    """Sealed source files (and extra repository paths) that differ from HEAD or are untracked."""
    root = Path(__file__).resolve().parents[2]
    paths = sorted(environment()['source_hashes'])
    for path in extra:
        resolved = Path(path).resolve()
        if root not in resolved.parents:
            raise ValueError(f'{resolved} is outside the repository; a production protocol is a committed repository file')
        paths.append(str(resolved.relative_to(root)))
    status = subprocess.check_output(['git', 'status', '--porcelain', '--', *paths], text=True, cwd=root)
    return [line[3:] for line in status.splitlines() if line.strip()]


def run(output, p, sources, datasets=None, workers=WORKERS, *, allow_smoke=False, protocol_path=None):
    """prepare, every planned job, then the five tables and provenance.json (all or none)."""
    output = Path(output)
    if not allow_smoke:
        validate_protocol(p)
        if not p.get('frozen'):
            raise ValueError('The diagnostics run requires a frozen reviewed protocol')
        dirty = uncommitted_sources(() if protocol_path is None else (protocol_path,))
        if dirty:
            raise ValueError(f'Commit the sealed sources and the protocol before the run: {", ".join(dirty)}')
    if not 1 <= int(workers) <= int(p['max_workers']):
        raise ValueError(f'Worker count must be between 1 and {p["max_workers"]}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; every invocation writes a new output directory')
    started = utc_now()
    with execution_lock():
        jobs, _ = prepare(output, p, sources, datasets, allow_smoke=allow_smoke,
                          purpose='synthetic_smoke_only' if allow_smoke else 'diagnostics')
        execute(output, jobs, int(workers))
        return write_tables(output, allow_smoke=allow_smoke, run_record={'started_utc': started, 'ended_utc': utc_now(),
                                                                          'workers': int(workers)})


# ----------------------------------------------------------------------------- verification, tables and provenance

KEY_COLUMNS = ('dataset_id', 'reference', 'outer_repeat', 'outer_fold', 'model_seed', 'view')
TABLES = {
    'validation_curves.csv': KEY_COLUMNS + ('iteration', 'validation_error', 'running_minimum', 'improved'),
    'snapshots.csv': KEY_COLUMNS + ('snapshot', 'iteration', 'layer', 'measure', 'value', 'relative_to_iteration'),
    'checkpoints.csv': KEY_COLUMNS + ('strategy', 'widths', 'embed_dim', 'degree', 'iterations', 'n_validation_samples',
                                      'initial_validation_error', 'checkpoint_iteration', 'checkpoint_validation_error',
                                      'improvement_count', 'returned_initial_filters', 'orders_verified'),
    'ties.csv': KEY_COLUMNS + ('network', 'iteration', 'layer', 'n_rows', 'n_filters', 'tied_response_share',
                               'distinct_response_ratio', 'tied_nearest_share'),
    'relabel.csv': KEY_COLUMNS + ('draws', 'relabel_seed', 'checkpoint_accuracy', 'changed_share_mean', 'changed_share_max',
                                  'accuracy_change_mean', 'accuracy_change_min', 'accuracy_change_max', 'abs_accuracy_change_max',
                                  'draws_with_changes')}
SNAPSHOT_MEASURES = (('knn_accuracy', 'knn_test_accuracy'), ('output_rule_accuracy', 'output_rule_test_accuracy'),
                     ('output_rule_training_accuracy', 'output_rule_training_accuracy'))


def validate_record(record, job, p, data, split, y, revision, output, sealed):
    """Bind one job record to the sealed plan and recompute its accuracies, majority, relabel shares and checkpoint from the
    saved predictions and validation curve."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(record.get('status') == 'ok', f'job status {record.get("status")}: {record.get("exception")}')
    seed, test = int(p['model_seed']), [int(i) for i in split['test']]
    identity = {'dataset_id': job['dataset_id'], 'reference': job['reference'], 'dataset_hash': data['dataset_hash'],
                'outer_repeat': job['outer_repeat'], 'outer_fold': job['outer_fold'], 'split_hash': config_id(split),
                'query': 'outer_test_rows', 'query_ids_hash': array_hash(np.asarray(test)), 'config_id': job['config_id'],
                'config': job['config'], 'selected': job['selected'], 'model_seed': seed,
                'check_uninstrumented': job['check_uninstrumented'], 'code_revision': revision, 'protocol_hash': config_id(p)}
    require(record['identity'] == identity, 'job identity disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'], 'plan disagrees with the sealed selection')
    require(sorted(record['checks']) == sorted(JOB_CHECKS), 'check schedule')
    for name in JOB_CHECKS:
        check, performed = record['checks'][name], job['check_uninstrumented'] if name == 'uninstrumented_state_hash' else True
        require(check['performed'] is performed and check['passed'] is (True if performed else None), f'check {name}')
    artifact, stem = record['artifact'], job['stem']
    require(artifact['path'] == f'artifacts/{stem}.npz' and (output/artifact['path']).is_file()
            and sha256_file(output/artifact['path']) == artifact['sha256'], 'artifact hash')
    with np.load(output/artifact['path'], allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    truth = y[test]
    require(np.array_equal(arrays['query_rows'], test) and np.array_equal(arrays['query_labels'], truth), 'artifact rows or labels')
    require(base.reproduction_check(arrays['predictions'], sealed, seed)['reproduced'], 'saved predictions do not reproduce the reference')
    views = record['views']
    require(len(views) == p['n_views'] == arrays['knn_scheduled'].shape[0] == arrays['relabel_knn'].shape[0], 'view count')
    schedule = scheduled_iterations(views[0]['iterations'], p['snapshots']['every'])
    require(arrays['scheduled_iterations'].tolist() == schedule, 'snapshot schedule')
    for v, view in enumerate(views):
        require(view['view'] == v and view['scheduled_iterations'] == schedule and view['widths'] == list(job['selected']['widths']),
                'view identity')
        scheduled = [s for s in view['snapshots'] if s['snapshot'] == 'scheduled']
        checkpoints = [s for s in view['snapshots'] if s['snapshot'] == 'checkpoint']
        require([s['iteration'] for s in scheduled] == schedule and len(checkpoints) == 1, 'snapshot list')
        for s, snapshot in enumerate(scheduled):
            require(snapshot['knn_accuracy'] == accuracy(arrays['knn_scheduled'][v, s], truth)
                    and snapshot['output_rule_accuracy'] == accuracy(arrays['output_scheduled'][v, s], truth), 'scheduled accuracies')
        require(checkpoints[0]['knn_accuracy'] == accuracy(arrays['knn_checkpoint'][v], truth)
                and checkpoints[0]['output_rule_accuracy'] == accuracy(arrays['output_checkpoint'][v], truth), 'checkpoint accuracies')
        require(all(s['validation_error'] == s['core_validation_error'] for s in view['snapshots']), 'recomputed validation error')
        curve = view['validation_curve']
        c, improvements, running = checkpoint_from_curve(curve['initial'], curve['after_update'])
        require(len(curve['after_update']) == view['iterations'] and running == curve['running_minimum']
                and c == view['checkpoint']['iteration'] == checkpoints[0]['iteration']
                and improvements == view['checkpoint']['improvement_iterations'], 'validation curve and checkpoint')
        relabel = view['relabel']
        require(relabel['checkpoint_accuracy'] == checkpoints[0]['knn_accuracy'] and len(relabel['draws']) == p['relabel']['draws']
                == arrays['relabel_knn'].shape[1], 'relabel draws')
        for r, draw in enumerate(relabel['draws']):
            predicted = arrays['relabel_knn'][v, r]
            require(draw['draw'] == r and draw['changed_share'] == float(np.mean(predicted != arrays['knn_checkpoint'][v]))
                    and draw['accuracy'] == accuracy(predicted, truth)
                    and draw['accuracy_change'] == accuracy(predicted, truth) - relabel['checkpoint_accuracy'], 'relabel draw')
    combined = majority(arrays['knn_checkpoint'])
    require(np.array_equal(combined, arrays['predictions']), 'majority at the checkpoints')
    curve = record['majority']['snapshots']
    require([s['iteration'] for s in curve] == schedule + [None] and all(
        s['knn_accuracy'] == accuracy(majority(arrays['knn_scheduled'][:, i]), truth) for i, s in enumerate(curve[:-1]))
            and curve[-1]['knn_accuracy'] == accuracy(combined, truth), 'majority learning curve')
    relabel = record['majority']['relabel']
    for r, draw in enumerate(relabel['draws']):
        predicted = majority(arrays['relabel_knn'][:, r])
        require(draw['draw'] == r and draw['changed_share'] == float(np.mean(predicted != combined))
                and draw['accuracy'] == accuracy(predicted, truth), 'majority relabel draw')
    return True


def _blank(value):
    return '' if value is None else value


def relabel_row(key, label, relabel, seed):
    changed = [float(d['changed_share']) for d in relabel['draws']]
    change = [float(d['accuracy_change']) for d in relabel['draws']]
    return key + [label, len(changed), seed, relabel['checkpoint_accuracy'], float(np.mean(changed)), max(changed),
                  float(np.mean(change)), min(change), max(change), max(abs(x) for x in change), sum(1 for x in changed if x > 0)]


def job_rows(record, p):
    """The rows one verified job contributes to each table."""
    identity = record['identity']
    key = [identity['dataset_id'], identity['reference'], identity['outer_repeat'], identity['outer_fold'], identity['model_seed']]
    rows = defaultdict(list)
    for view in record['views']:
        v, curve, checkpoint = view['view'], view['validation_curve'], view['checkpoint']
        improvements = set(checkpoint['improvement_iterations'])
        rows['validation_curves.csv'].append(key + [v, 0, curve['initial'], curve['running_minimum'][0], ''])
        rows['validation_curves.csv'].extend(key + [v, t, error, curve['running_minimum'][t], int(t in improvements)]
                                             for t, error in enumerate(curve['after_update'], 1))
        rows['checkpoints.csv'].append(key + [v, view['strategy'], canonical_json(view['widths']), view['embed_dim'], view['degree'],
                                              view['iterations'], view['n_validation_samples'], checkpoint['initial_validation_error'],
                                              checkpoint['iteration'], checkpoint['validation_error'], len(improvements),
                                              int(checkpoint['returned_initial_filters']),
                                              int(record['checks']['checkpoint_orders']['views'][v]['passed'])])
        for snapshot in view['snapshots']:
            for layer in snapshot['layers']:
                for measure, relative in (('displacement_from_initial', 0), ('changed_share_from_initial', 0),
                                          ('displacement_from_previous', snapshot['previous_iteration']),
                                          ('changed_share_from_previous', snapshot['previous_iteration'])):
                    if layer[measure] is not None:
                        rows['snapshots.csv'].append(key + [v, snapshot['snapshot'], snapshot['iteration'], layer['layer'], measure,
                                                            layer[measure], relative])
            rows['snapshots.csv'].extend(key + [v, snapshot['snapshot'], snapshot['iteration'], 'readout', name, snapshot[field], '']
                                         for field, name in SNAPSHOT_MEASURES)
        for network in ('initial', 'checkpoint'):
            entry = view['ties'][network]
            rows['ties.csv'].extend(key + [v, network, entry['iteration'], layer['layer'], layer['n_rows'], layer['n_filters'],
                                           _blank(layer['tied_response_share']), _blank(layer['distinct_response_ratio']),
                                           _blank(layer['tied_nearest_share'])] for layer in entry['layers'])
        rows['relabel.csv'].append(relabel_row(key, v, view['relabel'], p['relabel']['seed']))
    rows['snapshots.csv'].extend(key + ['majority', s['snapshot'], _blank(s['iteration']), 'readout', 'knn_test_accuracy', s['knn_accuracy'], '']
                                 for s in record['majority']['snapshots'])
    rows['relabel.csv'].append(relabel_row(key, 'majority', record['majority']['relabel'], p['relabel']['seed']))
    return rows


def render_tables(collected):
    """{file name: CSV text} over every verified job in planned order."""
    tables = {name: [] for name in TABLES}
    for job in collected['jobs']:
        for name, rows in job_rows(collected['records'][job['stem']], collected['protocol']).items():
            tables[name].extend(rows)
    contents = {}
    for name, header in TABLES.items():
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(tables[name])
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
    import hashlib
    p, manifest, records = collected['protocol'], collected['manifest'], collected['records']
    return _plain({
        'purpose': p['purpose'], 'selection_statement': p['selection_statement'], 'protocol_id': p['protocol_id'],
        'protocol_hash': manifest['protocol_hash'], 'protocol_sha256': sha256_file(Path(output)/'protocol.json'),
        'frozen': p['frozen'], 'frozen_at_utc': p.get('frozen_at_utc'), 'code_revision': collected['code_revision'],
        'environment': collected['environment'], 'references': manifest['references'], 'dataset_identity': manifest['dataset_identity'],
        'datasets': manifest['datasets'], 'planned_jobs': len(collected['jobs']), 'check_totals': check_totals(records),
        'jobs': {job['stem']: {'identity': records[job['stem']]['identity'], 'checks': records[job['stem']]['checks'],
                               'timing': records[job['stem']]['timing'],
                               'record_sha256': sha256_file(Path(output)/'jobs'/f'{job["stem"]}.json'),
                               'artifact_sha256': records[job['stem']]['artifact']['sha256']} for job in collected['jobs']},
        'tables': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in contents.items()},
        'run': run_record})


def write_all(output, contents):
    """All or none: refuse before writing anything when an existing file differs; write every missing file under a temporary
    name, then link each into place (a link never replaces an existing file)."""
    output = Path(output)
    for name, text in contents.items():
        if (output/name).exists() and (output/name).read_text() != text:
            raise FileExistsError(f'Refusing to replace {output/name}; use a new output directory')
    pending = {name: output/f'.{name}.writing' for name in contents if not (output/name).exists()}
    try:
        for name, temporary in pending.items():
            with temporary.open('x') as stream:
                stream.write(contents[name])
        for name, temporary in pending.items():
            os.link(temporary, output/name)
    finally:
        for temporary in pending.values():
            temporary.unlink(missing_ok=True)


def collect(output, *, allow_smoke=False, rederive=False):
    """Every planned job record verified (validate_record); with rederive, every sealed selection and planned job re-derived
    from its reference. Failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs, selections = verify(output, allow_smoke=allow_smoke)
    saved_environment = json.loads((output/'environment.json').read_text())
    references = None
    if rederive:
        sources = {name: (entry['run_directory'], entry['ablation_directory']) for name, entry in manifest['references'].items()}
        references = load_references(p, sources, allow_smoke=allow_smoke)
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
        raise ValueError('Incomplete training diagnostics evidence: ' + '; '.join(issues))
    return {'protocol': p, 'manifest': manifest, 'jobs': jobs, 'records': records, 'code_revision': saved_environment['code_revision'],
            'environment': saved_environment}


def write_tables(output, *, allow_smoke=False, run_record=None):
    """The five tables and provenance.json of a completed run, all or none."""
    collected = collect(output, allow_smoke=allow_smoke)
    contents = render_tables(collected)
    provenance = provenance_record(output, collected, contents, run_record)
    write_all(output, {**contents, PROVENANCE_FILE: json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + '\n'})
    return provenance


# ----------------------------------------------------------------------------- summary

def _stats(values):
    values = [float(v) for v in values]
    return {'n_folds': len(values), 'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'min': min(values), 'max': max(values)}


def _relabel_block(entries):
    changed = [float(d['changed_share']) for entry in entries for d in entry['draws']]
    change = [float(d['accuracy_change']) for entry in entries for d in entry['draws']]
    return {'entries': len(entries), 'draws': len(changed), 'mean_changed_share': float(np.mean(changed)),
            'max_changed_share': max(changed), 'share_of_draws_with_changes': float(np.mean(np.asarray(changed) > 0)),
            'mean_accuracy_change': float(np.mean(change)), 'min_accuracy_change': min(change), 'max_accuracy_change': max(change),
            'max_abs_accuracy_change': max(abs(x) for x in change)}


def _mean_block(groups):
    return {key: {'n_views': max(len(values) for values in measures.values()),
                  **{f'mean_{measure}': float(np.mean(values)) for measure, values in sorted(measures.items())}}
            for key, measures in sorted(groups.items())}


SUMMARY_DEFINITIONS = {
    'checkpoint_iteration': 'median, quartiles and IQR (numpy.percentile, linear) of the checkpoint iteration over every view and '
                            'outer fold of the dataset',
    'share_of_views_with_checkpoint_zero': 'share of views and outer folds whose checkpoint returned the initial filters',
    'displacement_at_checkpoint': 'mean over views of the checkpoint displacement from the initial filters (normalized footrule and '
                                  'changed-filter share), grouped by "<selected widths> <layer>"',
    'knn_majority_test_accuracy': 'the 7-view majority of the per-view kNN readouts (final selected settings) on the outer test rows '
                                  'with the initial filters (iteration_0) and with each view at its checkpoint; over outer folds',
    'ties': 'means over views of the tie measures on the outer test rows with the initial and the checkpoint filters, grouped by '
            '"<selected widths> <layer>"',
    'relabel': 'over views and draws (views) or outer folds and draws (majority): share of outer test predictions that change and '
               'the accuracy change after filter-ID relabeling at the checkpoint network'}


def summarize(collected):
    p, records = collected['protocol'], collected['records']
    grouped = defaultdict(list)
    for job in collected['jobs']:
        grouped[job['dataset_id']].append(records[job['stem']])
    summaries = {}
    for name in collected['manifest']['datasets']:
        jobs = grouped[name]
        views = [view for record in jobs for view in record['views']]
        checkpoints = np.asarray([view['checkpoint']['iteration'] for view in views], dtype=float)
        q25, median, q75 = (float(x) for x in np.percentile(checkpoints, [25, 50, 75]))
        displacement = defaultdict(lambda: defaultdict(list))
        ties = {network: defaultdict(lambda: defaultdict(list)) for network in ('initial', 'checkpoint')}
        for view in views:
            widths = canonical_json(view['widths'])
            final = next(s for s in view['snapshots'] if s['snapshot'] == 'checkpoint')
            for layer in final['layers']:
                for measure in ('displacement_from_initial', 'changed_share_from_initial'):
                    displacement[f'{widths} {layer["layer"]}'][measure].append(layer[measure])
            for network, groups in ties.items():
                for layer in view['ties'][network]['layers']:
                    for measure in ('tied_response_share', 'distinct_response_ratio', 'tied_nearest_share'):
                        if layer[measure] is not None:
                            groups[f'{widths} {layer["layer"]}'][measure].append(layer[measure])
        initial = [record['majority']['snapshots'][0]['knn_accuracy'] for record in jobs]
        final_accuracy = [record['majority']['snapshots'][-1]['knn_accuracy'] for record in jobs]
        summaries[name] = {
            'reference': jobs[0]['identity']['reference'], 'jobs': len(jobs), 'views': len(views),
            'selected_widths': dict(sorted(Counter(canonical_json(r['identity']['selected']['widths']) for r in jobs).items())),
            'checkpoint_iteration': {'median': median, 'q25': q25, 'q75': q75, 'iqr': q75 - q25, 'min': float(checkpoints.min()),
                                     'max': float(checkpoints.max()), 'n_views': len(views)},
            'share_of_views_with_checkpoint_zero': float(np.mean(checkpoints == 0)),
            'displacement_at_checkpoint': _mean_block(displacement),
            'knn_majority_test_accuracy': {'iteration_0': _stats(initial), 'checkpoint': _stats(final_accuracy),
                                           'checkpoint_minus_iteration_0': _stats([b - a for a, b in zip(initial, final_accuracy)])},
            'ties': {network: _mean_block(groups) for network, groups in ties.items()},
            'relabel': {'views': _relabel_block([view['relabel'] for view in views]),
                        'majority': _relabel_block([record['majority']['relabel'] for record in jobs])}}
    return _plain({'purpose': p['purpose'], 'selection_statement': p['selection_statement'], 'protocol_id': p['protocol_id'],
                   'protocol_hash': collected['manifest']['protocol_hash'], 'code_revision': collected['code_revision'],
                   'datasets': collected['manifest']['datasets'], 'planned_jobs': len(collected['jobs']),
                   'check_totals': check_totals(records), 'definitions': SUMMARY_DEFINITIONS,
                   'inferential_significance_claims': False, 'summaries': summaries})


def summary(output, *, allow_smoke=False):
    """Re-verify the run (every job record, every selection re-derived from its reference), require the five tables and
    provenance.json to equal their re-rendering, then write diagnostics_summary.json."""
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
    write_json(output/SUMMARY_FILE, report)
    return report


# ----------------------------------------------------------------------------- pilot and projection

def smallest_dataset(reference):
    rows = {name: json.loads((Path(reference['run']['directory'])/name/'manifest.json').read_text())['shape'][0]
            for name in reference['datasets']}
    return min(reference['datasets'], key=lambda name: (rows[name], reference['datasets'].index(name)))


def calibrated_projection(jobs, records, workers=WORKERS):
    """Per planned job, the reference run's realized outer fit and predict seconds (seed 8129, that fold) times the piloted
    (job - uninstrumented fit) / uninstrumented fit ratio (its own for a pilot dataset, the largest piloted otherwise), plus the
    realized seconds again for a first fold's uninstrumented fit; serial hours and the first-free-worker makespan in planned order."""
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
            'simulated_makespan_hours': makespan(seconds, workers) / 3600, 'longest_job_hours': max(seconds) / 3600, 'workers': workers}


def runtime_pilot(output, p, sources):
    """Training-only: prepare every protocol dataset (references, fresh loads, selections), then the first outer fold of each
    pilot dataset with every fourth outer training row as the query rows (the outer test fold is never touched), the
    projections and one reproduction probe (the smallest dataset) per reference."""
    output = Path(output)
    started = utc_now()
    jobs, references = prepare(output, p, sources, purpose='training_runtime_only')
    mapping, revision = reference_of(p), json.loads((output/'environment.json').read_text())['code_revision']
    sealed_records = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s for s in json.loads((output/'reference_selections.json').read_text())}
    records = []
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and j['check_uninstrumented'])
        X, y, data, splits = load_prepared(output, name)
        split = splits[0]
        query = [int(i) for i in split['train'][::4]]
        record, _ = evaluate_job(X, y, split, job, sealed_records[(name, split['outer_repeat'], split['outer_fold'])], p,
                                 dataset_hash=data['dataset_hash'], code_revision=revision, protocol_hash=config_id(p), query=query)
        if record['status'] != 'ok':
            raise CheckFailed(f'pilot job {name} failed: {record.get("exception")}')
        timing = record['timing']
        records.append({'dataset_id': name, 'reference': mapping[name], 'dataset_hash': data['dataset_hash'], 'train_ids': split['train'],
                        'query_ids': query, 'config_id': job['config_id'], 'selected': job['selected'], 'model_seed': job['model_seed'],
                        'timing': timing, 'reference_outer_seconds': job['reference_outer_seconds'],
                        'job_to_fit_ratio': (timing['job_seconds'] - timing['uninstrumented_fit_seconds']) / timing['uninstrumented_fit_seconds'],
                        'checks': {key: {'performed': c['performed'], 'passed': c['passed']} for key, c in record['checks'].items()},
                        'checkpoints': [view['checkpoint']['iteration'] for view in record['views']], 'status': 'ok'})
    piloted = {r['dataset_id']: r for r in records}
    slowest_first = max(r['timing']['job_seconds'] for r in records)
    slowest_other = max(r['timing']['job_seconds'] - r['timing']['uninstrumented_fit_seconds'] for r in records)
    harness_seconds = sum((piloted[j['dataset_id']]['timing']['job_seconds'] - (0. if j['check_uninstrumented'] else
                           piloted[j['dataset_id']]['timing']['uninstrumented_fit_seconds'])) if j['dataset_id'] in piloted
                          else (slowest_first if j['check_uninstrumented'] else slowest_other) for j in jobs)
    probes = {name: base.reproduction_probe(reference['run'], smallest_dataset(reference)) for name, reference in references.items()}
    calibrated = calibrated_projection(jobs, records, p['workers'])
    decision = calibrated['simulated_makespan_hours']
    checks_passed = all(c['passed'] is True for r in records for key, c in r['checks'].items() if key != 'reference_predictions')
    report = {'purpose': 'training_only_runtime_no_heldout_scores', 'protocol_id': p['protocol_id'], 'protocol_hash': config_id(p),
              'code_revision': revision, 'started_utc': started, 'ended_utc': utc_now(), 'records': records,
              'harness_projection': {'serial_hours': harness_seconds / 3600, 'hours_at_workers_ideal': harness_seconds / 3600 / p['workers'],
                                     'basis': 'idle pilot seconds per job; unpiloted datasets at the slowest piloted job (not a bound)'},
              'calibrated_projection': calibrated, 'reproduction_probes': probes,
              'decision': {'rule': p['decision_rule'], 'hours': decision, 'cap_hours': p['wallclock_cap_hours'], 'workers': p['workers'],
                           'within_cap': decision <= p['wallclock_cap_hours'], 'checks_passed': checks_passed,
                           'probes_reproduced': all(probe['reproduced'] for probe in probes.values())},
              'estimate_limitations': 'one training partition per pilot dataset on the machine as it was; the calibrated projection '
                                      'assumes the reference runs\' realized per-fold seconds (measured under 16 workers) carry over; '
                                      'the query rows are training rows, so the kNN and relabel costs follow the outer test size only '
                                      'approximately'}
    write_json(output/'pilot.json', report)
    return report


# ----------------------------------------------------------------------------- freeze

def freeze(draft_path, pilot_path, stages_path, output_path, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the training-only pilot of that draft projects within the cap at the
    protocol workers, every pilot check and reproduction probe held, and the stage record carries a passing smoke; an existing
    output must be the draft itself, which the frozen protocol then replaces."""
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
        raise ValueError(f'Not frozen: projection {decision["hours"]:.2f} h at {decision["workers"]} workers against the {CAP_HOURS} h '
                         f'cap, probes reproduced {decision["probes_reproduced"]}, pilot checks passed {decision["checks_passed"]}')
    if (stages.get('smoke') or {}).get('status') != 'ok' or not stages.get('summary'):
        raise ValueError('The stage record lacks a passing synthetic smoke and its summary')
    calibrated, harness = pilot['calibrated_projection'], pilot['harness_projection']
    record = {'cap_hours': CAP_HOURS, 'workers': WORKERS, 'decision_hours': decision['hours'], 'decision_rule': decision['rule'],
              'calibrated': {key: calibrated[key] for key in ('serial_hours', 'serial_hours_over_workers', 'simulated_makespan_hours',
                                                              'longest_job_hours')},
              'calibrated_per_dataset_hours': {name: entry['serial_hours'] for name, entry in calibrated['datasets'].items()},
              'harness': {key: harness[key] for key in ('serial_hours', 'hours_at_workers_ideal')},
              'pilot_ratios': {r['dataset_id']: r['job_to_fit_ratio'] for r in pilot['records']},
              'pilot_seconds': {r['dataset_id']: r['timing'] for r in pilot['records']},
              'reproduction_probes': {name: {key: probe[key] for key in ('dataset_id', 'result_file', 'config_id', 'model_seed', 'inner_fold',
                                                                         'reference_score', 'refit_score', 'readout_selections_identical',
                                                                         'reproduced')}
                                      for name, probe in pilot['reproduction_probes'].items()},
              'pilot_code_revision': pilot['code_revision'], 'pilot_sha256': sha256_file(pilot_path), 'stages': stages}
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    text = (f"Training diagnostics (author decision 5b and the controller ruling of 2026-09-14): {stages['summary']}; projected at "
            f"{WORKERS} single-thread workers: calibrated simulated makespan {decision['hours']:.2f} h (serial {calibrated['serial_hours']:.2f} h, "
            f"serial over workers {calibrated['serial_hours_over_workers']:.2f} h), harness {harness['hours_at_workers_ideal']:.2f} h; cap "
            f"{CAP_HOURS} h; reproduction probes held on {', '.join(probe['dataset_id'] for probe in pilot['reproduction_probes'].values())}; "
            f"frozen after the pilot")
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

SMOKE_CANDIDATES = [{'aggregation': 'majority', 'batch_size': 32, 'degree_offset': 0, 'embed_scale': 1, 'iterations': 23,
                     'learning_rate': .2, 'n_views': 7, 'strategy': 'diverse', 'validation_ratio': .1, 'widths': widths}
                    for widths in ([4, 6], [6, 8])]


def smoke_protocol(p, sources):
    """The synthetic smoke form of a protocol whose references are the given synthetic runs {name: (run_dir, ablation_dir)}."""
    references = {}
    for name, (run_dir, ablation_dir) in sources.items():
        run_protocol = json.loads((Path(run_dir)/'protocol.json').read_text())
        references[name] = {'datasets': list(run_protocol['datasets']), 'loader': None,
                            'run': {**reference_pins(run_dir), 'family': run_protocol['production_family']},
                            'ablation': ablation_pins(ablation_dir)}
    design_source = json.loads((Path(next(iter(sources.values()))[0])/'protocol.json').read_text())
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    return _plain(dict(p, references=references, datasets=datasets, pilot_datasets=datasets[:1], frozen=False,
                       status='synthetic_smoke_only', purpose='synthetic smoke only; never evidence',
                       **{key: design_source[key] for key in base.DESIGN_KEYS}))


def smoke(output, p, workers=3):
    """Synthetic references built with the real harness: a bridge_knn-style run at 23 iterations with two hidden layers (widths
    [4, 6] and [6, 8], augmentation on) with its knn_ablation, and the two newdata-style batch runs (one iteration; one hidden layer
    selected; augmentation on and off) with their newdata_ablation (run_newdata_ablation.smoke); then a complete diagnostics run
    over all three references and its summary, so both depths reach every measure and check. Never evidence."""
    from . import run_newdata_ablation as rn
    output = Path(output)
    with execution_lock():
        bridge = synthetic_reference_run(output/'synthetic_bridge_knn', SMOKE_CANDIDATES, workers=workers, samples=240)
    template, reference_protocol = json.loads(base.PROTOCOL.read_text()), json.loads((bridge/'protocol.json').read_text())
    tiny = dict(template, datasets=['synthetic'], pilot_datasets=['synthetic'], frozen=False, purpose='synthetic_smoke_only',
                **{key: reference_protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds')},
                reference_source={**template['reference_source'], **reference_pins(bridge), 'family': reference_protocol['production_family']},
                depth_split={**template['depth_split'], 'depths': [c['widths'] for c in SMOKE_CANDIDATES]})
    knn_ablation = output/'synthetic_knn_ablation'
    base.prepare(knn_ablation, tiny, bridge, allow_smoke=True, purpose='synthetic_smoke_only')
    base.run(knn_ablation, workers, allow_smoke=True)
    base.write_summary(knn_ablation, allow_smoke=True)
    newdata = output/'synthetic_newdata_ablation'
    rn.smoke(newdata, rn.draft_protocol(), workers)
    sources = {'smoke_bridge_knn': (bridge, knn_ablation),
               'smoke_newdata_batch1': (newdata/'synthetic_reference_batch1', newdata),
               'smoke_newdata_batch2': (newdata/'synthetic_reference_batch2', newdata)}
    diagnostics = output/'diagnostics'
    run(diagnostics, smoke_protocol(p, sources), sources, workers=workers, allow_smoke=True)
    return summary(diagnostics, allow_smoke=True)


# ----------------------------------------------------------------------------- command

def parse_sources(values):
    sources = {}
    for name, directories in values or []:
        if name in sources:
            raise ValueError(f'--reference {name} is given twice')
        sources[name] = directories
    return sources or dict(DEFAULT_SOURCES)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['draft', 'smoke', 'pilot', 'freeze', 'run', 'summary'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--reference', type=parse_reference, action='append', metavar='NAME=RUN_DIR,ABLATION_DIR')
    parser.add_argument('--dataset', nargs='+')
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--draft', type=Path, default=PROTOCOL)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--stages', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'draft':
        write_json(args.output, draft_protocol())
        return
    if args.command == 'summary':
        report = summary(args.output)
        for name, entry in report['summaries'].items():
            accuracy_block = entry['knn_majority_test_accuracy']
            print(f"{name}: checkpoint median {entry['checkpoint_iteration']['median']:g} (IQR {entry['checkpoint_iteration']['iqr']:g}); "
                  f"checkpoint 0 in {entry['share_of_views_with_checkpoint_zero']:.3f} of views; 7-view kNN "
                  f"{accuracy_block['iteration_0']['mean']:.4f} at iteration 0, {accuracy_block['checkpoint']['mean']:.4f} at the checkpoint")
        return
    if args.command == 'freeze':
        if args.pilot is None or args.stages is None:
            parser.error('freeze needs --pilot and --stages')
        protocol = freeze(args.draft, args.pilot, args.stages, args.output)
        print(f"frozen at {protocol['frozen_at_utc']}: decision {protocol['pilot_projection']['decision_hours']:.2f} h")
        return
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f'Worker count must be between 1 and {MAX_WORKERS}')
    p = validate_protocol(json.loads(args.protocol.read_text()))
    if args.command == 'smoke':
        report = smoke(args.output, p, args.workers)
        print(json.dumps(report['check_totals'], indent=2))
        return
    sources = parse_sources(args.reference)
    if args.command == 'pilot':
        with execution_lock():
            report = runtime_pilot(args.output, p, sources)
        print(json.dumps({'decision': report['decision'], 'pilot_ratios': {r['dataset_id']: r['job_to_fit_ratio'] for r in report['records']},
                          'calibrated_serial_hours': report['calibrated_projection']['serial_hours'],
                          'harness_hours_at_workers': report['harness_projection']['hours_at_workers_ideal']}, indent=2))
        return
    provenance = run(args.output, p, sources, args.dataset, args.workers, protocol_path=args.protocol)
    print(json.dumps(provenance['check_totals'], indent=2))


if __name__ == '__main__':
    main()
