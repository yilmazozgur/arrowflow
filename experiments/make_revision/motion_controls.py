"""E1: matched motion-signal controls for ArrowFlow-kNN, at its reconstructed per-fold selections.

draft    --output P                                                   the unfrozen protocol (draft_protocol)
prepare  --protocol P [--reference NAME=RUN,ABLATION ...] --output O  seal the selections and the reference predictions
smoke    --protocol P --output O [--workers 3]                        synthetic references and a complete run; never evidence
pilot    --protocol P [--reference ...] --output O                    training-only timing and the projection
freeze   --draft P --pilot O/pilot.json --stages S --output F         the frozen protocol, only if the projection is within the cap
run      --protocol P [--reference ...] --output O --workers 16       fit every arm per dataset, outer fold and fitting seed
summary  --output O                                                   verify every planned record; motion_controls_summary.json/.csv

The question: does ArrowFlow's specific learning signal carry information, or would comparable random movement of the same
filters do as well? Every arm is fitted at ArrowFlow-kNN's reconstructed per-fold selection on the same 15 outer folds, the
same three fitting seeds, the same encoder, the same seeded initial filters, the same training rows, the same view identities,
the same training budget and the same kNN readout rule. Only the hidden-layer update changes:

  views7              the unmodified method; must reproduce the reference run's outer predictions exactly
  frozen              the hidden ranking filters never move; the auxiliary class-prototype (output) layer still updates
  permuted_alignment  the motion signal reaching each hidden layer is re-keyed to a uniformly random permutation of that
                      layer's filter identities, drawn per batch
  random_direction    the signs of the accepted votes are permuted among the accepted slots, so magnitudes, accepted mass,
                      vote counts and sign counts are preserved exactly and only the direction pairing is destroyed
  single_layer_first  only hidden layer 0 updates      (identical to views7, and recorded as such, at one-layer selections)
  single_layer_last   only the last hidden layer updates (likewise)

Supervision disclosure: the eligibility gate that decides which examples produce votes reads the labels (an example votes when
its predicted class is wrong, and with probability p_correct when it is right) and it is PRESERVED in every arm, so label
information still enters every arm. A full label-permutation control is a different experiment and is not claimed here.

Instrumentation is instance level and non-invasive, as training_diagnostics.py's is: an instance-level backward_propagate is
installed on each view network between ArrowFlowEstimator.initialize_orders and train_initialized and removed afterwards. It
is a transcription of SortFlowHybridNetwork.backward_propagate (whose source sha256 is pinned in CORE_SOURCES and re-checked
in every job) with exactly one injected step, the arm's transform of the motion records entering each hidden layer, and one
changed step, which hidden layers apply their accumulated motion. Every transform draw comes from a private RandomState, and
every wrapper call saves and restores the global numpy and Python RNG states and records whether they changed.

Every job fails, cancels the pending jobs and blocks the summary when a check fails:
  reference_predictions   views7's 7-view kNN outer predictions equal the reference run's, for every fitting seed
  uninstrumented_fit      first outer fold of each dataset: views7's seven view state_hash() and its kNN predictions equal
                          those of an uninstrumented MultiViewArrowFlowKNN(**selected, seed).fit on the same rows
  rng_untouched           every wrapper call left the global numpy and Python RNG states unchanged
  wrapper_removed         no view network keeps an instance-level backward_propagate
  core_sources            the pinned core sources are unchanged
  initial_state           every arm's seeded initial filters and initial state hash equal views7's, per view
  mass_matched            every transform preserved the accepted vote count, the accepted |motion| multiset, the accepted
                          mass and the accepted sign counts (sign counts except in random_direction, where they are permuted)
  first_batch_signal      every arm's pre-transform signal at the first batch equals views7's, per view and layer
  frozen_unmoved          the frozen arm's hidden filters are exactly its initial filters
  single_layer_identity   at one-layer selections both single_layer arms are recorded identical_to_views7 and their
                          predictions equal views7's
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
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path
import random
import time
import zipfile
import numpy as np
from sklearn.metrics import recall_score
from threadpoolctl import threadpool_limits
from arrowflow.ranking import score_order
from . import run_knn_ablation as base
from . import training_diagnostics as td
from .bridge import resolve_selected
from .comparisons import StableFootruleKNN, derive_seed
from .evaluation import (canonical_json, config_id, make_splits, metric_values, paired_corrected_interval, summarize_outer,
                         validate_outer_schedule, validate_split)
from .models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from .multiview import MultiViewArrowFlowKNN, view_strategy
from .newdata import WORKSPACE_RUNS, makespan, sha256_file
from .run_bridge import fold_schedule, write_csv
from .run_revision import environment_record, execution_lock, load_prepared, write_json
from .secondary_studies import majority

PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-14'
PROTOCOL = PROTOCOLS/'motion_controls.json'
PROTOCOL_ID = 'arrowflow-v3-motion-controls-1'
FAMILY = 'motion_controls'
SOURCE_MODULES = td.SOURCE_MODULES + ['experiments.make_revision.training_diagnostics']
REFERENCE_MODEL = base.REFERENCE_MODEL
REFERENCES = td.REFERENCES
DEFAULT_SOURCES = td.DEFAULT_SOURCES

ARMS = ('views7', 'frozen', 'permuted_alignment', 'random_direction', 'single_layer_first', 'single_layer_last')
PRIMARY_ARMS = ('frozen', 'permuted_alignment', 'random_direction')          # 15 paired folds on every dataset
DEPTH_ARMS = ('single_layer_first', 'single_layer_last')                     # two-hidden-layer folds only; descriptive
RANDOMIZING_ARMS = ('permuted_alignment', 'random_direction')
METRICS = ('accuracy', 'error', 'balanced_accuracy', 'macro_f1')
JOB_CHECKS = ('reference_predictions', 'uninstrumented_fit', 'rng_untouched', 'wrapper_removed', 'core_sources',
              'initial_state', 'mass_matched', 'first_batch_signal', 'frozen_unmoved', 'single_layer_identity')
NAMED_SUBSET = ('balance_scale', 'mfeat_zernike')
MOTION_SEED = 20260914
PURITY_QUERY_CAP = 512
CAP_HOURS = 5
WORKERS = 16
MAX_WORKERS = 16
PILOT_DATASETS = ('wine', 'balance_scale')     # a two-hidden-layer-heavy small set and a one-layer set (both cheap)
FREEZE_FIELDS = ('frozen', 'frozen_at_utc', 'status', 'resource_decision', 'pilot_projection')
DRAFT_STATUS = 'drafted_awaiting_smoke_and_training_only_pilot'
FROZEN_STATUS = 'reviewed_and_piloted_before_any_outer_score'
SUMMARY_JSON, SUMMARY_CSV = 'motion_controls_summary.json', 'motion_controls_summary.csv'
SUMMARY_COLUMNS = ('dataset_id', 'arm_id', 'metric', 'mean', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds',
                   'seeds_per_fold')
PREDICTION_KEYS = ('dataset_id', 'outer_repeat', 'outer_fold', 'arm_id', 'model_seed', 'sample_id', 'y_true', 'y_pred',
                   'config_id', 'code_revision')

# The core functions this module transcribes or relies on; re-checked in every job so a changed core fails loudly.
CORE_SOURCES = {
    'SortFlowHybridNetwork.backward_propagate': '72fdebd78f630c138d341ac75f1445e694a9c56312b926a1b5bb2b90c39bd4dd',
    'SortFlowHybridNetwork.update_network': '3ca3bb3f7ac2a09121501e778aa1759964b4f4e8a9071e4d4e5bd4c87f2bf0c2',
    'SortFlowHybridNetwork.train': 'ea6cdd5a1ce764192d349d616ae44fbc6d7bf622846b33b151965baa1b14c631',
    'SortFlowHybridNetwork._forward_propagate_batch': '082c1c705a2909468f108fbe8871374303877725966c761dad47f93ebead5899',
    'SortFlowHybridNetwork.adapt_learning_rate': '642878c4346f01816027f7df83956132995857a9243814e97deccf1be5fa5933',
    'Vertex.accumulate_motion': '0d9b4c0e3b1c6d762c3a7bb8445836e01967fcdb3f2b70581597ad2d5843b323',
    'Vertex.apply_motion': '459e36d6dde21316209dba1e325eee6ab653d9a45146f816c4007da98959abf1',
    'Vertex.clean_motion_accumulation': '11e8e59a80a305925fe371c12c0c116e8f27c6da796ecf405de6cb791345e374',
    'Vertex.compute_distance': '8f783f782e8e7025db60b25f05079b2ca44e79c1acd9ac03b111102305a5fa35',
    'Vertex.accumulate_perm_inplace': 'db83fcd7e50eb01def8f4d8c71b69ef6b173d3bd5565f2dedbecfc078ed81041',
    'Vertex.compute_adj_list_with_permutation': '5e683c32c1855a21811b082df420b819e85fd33b5acf8877f2a3002b0004cf7a',
    'VertexFilters.update_index_matrix': '33dfba891ebac336b820468a378a397e3ed0901660b2d37045dd21c4f184810f',
}


class CheckFailed(RuntimeError):
    """A non-invasiveness, matching or consistency check of a motion-controls job failed."""


def environment():
    return environment_record(__package__ + '.motion_controls:environment')


def _plain(value):
    return json.loads(canonical_json(td.native(value)))


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ----------------------------------------------------------------------------- the core sources this module depends on

def core_source_hashes():
    from arrowflow.arrowflow import SortFlowHybridNetwork, Vertex, VertexFilters
    owners = {'SortFlowHybridNetwork': SortFlowHybridNetwork, 'Vertex': Vertex, 'VertexFilters': VertexFilters}
    out = {}
    for name in CORE_SOURCES:
        owner, function = name.split('.')
        source = inspect.getsource(getattr(owners[owner], function))
        out[name] = hashlib.sha256(source.encode()).hexdigest()
    return out


def check_core_sources():
    observed = core_source_hashes()
    changed = sorted(name for name, digest in CORE_SOURCES.items() if observed.get(name) != digest)
    return {'performed': True, 'passed': not changed, 'changed': changed, 'observed': observed}


# ----------------------------------------------------------------------------- the arm

def numpy_states_equal(a, b):
    return td.numpy_states_equal(a, b)


def arm_layers(arm, hidden_layers):
    """The hidden layer indices that apply their accumulated motion under this arm."""
    if arm in ('views7', 'permuted_alignment', 'random_direction'):
        return tuple(range(hidden_layers))
    if arm == 'frozen':
        return ()
    if arm == 'single_layer_first':
        return (0,)
    if arm == 'single_layer_last':
        return (hidden_layers - 1,)
    raise ValueError(f'Unknown arm {arm!r}')


def _signal_hash(record):
    keys, magnitudes = record
    return array_hash(np.asarray(keys)), array_hash(np.ascontiguousarray(np.asarray(magnitudes, dtype=float)))


class MotionArm:
    """Instance-level backward_propagate of one view network implementing one control arm.

    The replacement is a transcription of SortFlowHybridNetwork.backward_propagate restricted to the sort-only path this
    method uses (every layer 'sort', average_method_motion 'mean', backprop_signal_replication 1), with exactly two
    departures: `transform` rewrites the motion records entering each hidden layer, and only the layers in
    `updating_layers` apply their accumulated motion (the others reset their accumulator, exactly as the core's own
    last_layer_update=False branch resets the output layer's). The auxiliary class-prototype (output) layer's own update
    is untouched in every arm, and so are the forward pass, the eligibility gate, the checkpoint and the readout.
    """

    def __init__(self, net, arm, *, seed, hidden_layers):
        network = getattr(net, 'network_', None)
        if network is None or network.update_iter != 0:
            raise ValueError('Install the arm on an initialized, untrained view network')
        if 'backward_propagate' in vars(network):
            raise ValueError('The network already carries an instance-level backward_propagate')
        if network.graph.num_vertices != hidden_layers + 1:
            raise CheckFailed('The network depth differs from the selected hidden widths')
        if network.average_method_motion != 'mean' or network.backprop_signal_replication != 1:
            raise CheckFailed('This arm is defined for the mean-averaged, unreplicated sort-only backward pass')
        self.net, self.network, self.arm = net, network, arm
        self.hidden_layers = int(hidden_layers)
        self.updating_layers = frozenset(arm_layers(arm, self.hidden_layers))
        self.stop_after = min(self.updating_layers) if self.updating_layers else self.hidden_layers
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.draws = 0
        self.calls = 0
        self.rng_checks = []
        self.first_batch = {}
        self.statistics = {}
        with self._guard():
            self.initial = td.filter_copy(network)

    # ------------------------------------------------------------------ installation

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
        arm = self

        def backward_propagate(forward_input_backprop_all_data, motion_last_layer):
            return arm._backward(forward_input_backprop_all_data, motion_last_layer)

        self.network.backward_propagate = backward_propagate
        try:
            yield self
        finally:
            del self.network.backward_propagate
        with self._guard():
            self.final = td.filter_copy(self.network)

    def wrapper_removed(self):
        return 'backward_propagate' not in vars(self.network)

    # ------------------------------------------------------------------ the transcribed backward pass

    def _backward(self, forward_input_backprop_all_data, motion_last_layer):
        self.calls += 1
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            return self._backward_body(forward_input_backprop_all_data, motion_last_layer)
        finally:
            unchanged = numpy_states_equal(numpy_state, np.random.get_state()) and python_state == random.getstate()
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            self.rng_checks.append(bool(unchanged))

    def _backward_body(self, forward_input_backprop_all_data, motion_last_layer):
        network = self.network
        layer_name = network.id + '_ly' + str(network.graph.num_vertices - 1)
        layer = network.graph.vertex_list[layer_name]
        # The auxiliary class-prototype layer: exactly the core's own last-layer block, unchanged in every arm.
        if network.last_layer_update:
            for last_layer_vertex_key in layer.graph.vertex_list:
                layer.graph.vertex_list[last_layer_vertex_key].apply_motion()
            layer.update_index_matrix()
        else:
            for last_layer_vertex_key in layer.graph.vertex_list:
                last_layer_vertex = layer.graph.vertex_list[last_layer_vertex_key]
                last_layer_vertex.clean_motion_accumulation(last_layer_vertex.adjacency_list)

        for hidden_layer_iter in range(network.graph.num_vertices - 1):
            index = network.graph.num_vertices - 2 - hidden_layer_iter
            if index < self.stop_after:
                break                      # no deeper layer updates under this arm; nothing below can change any filter
            first_layer_skip_gradient = index == 0
            layer_name = network.id + '_ly' + str(index)
            layer = network.graph.vertex_list[layer_name]
            if layer.layer_type != 'sort':
                raise CheckFailed('This arm is defined for the sort-only architecture')

            motion_last_layer = self._transform(index, layer, motion_last_layer)

            motion_adjustment = (1 / float(network.learning_rate)) * (layer.graph.num_vertices / 2)
            motion_last_layer_next = []
            cumulative_motion_dict = {}
            corrected_data_ids = {}
            vertex_key = list(layer.graph.vertex_list)[0]
            vertex_ = layer.graph.vertex_list[vertex_key]
            sorted_adj_vertices = vertex_._sorted_adj
            for motion_idx, motion_data_point in enumerate(motion_last_layer):
                layer_data = forward_input_backprop_all_data[layer_name][motion_idx][0]
                for vertex_idx, motion_vertex in enumerate(motion_data_point[1]):
                    if vertex_idx >= int(np.ceil(network.ratio_data_backprop * len(motion_data_point[0]))):
                        break
                    vertex_key = motion_data_point[0][vertex_idx]
                    vertex_ = layer.graph.vertex_list[vertex_key]
                    vertex_adj_sort_index = vertex_._adj_sort_index_list
                    motion_vertex_mag = motion_vertex / motion_adjustment
                    motion_vertex_data_point = vertex_.accumulate_motion(layer_data, magnitude=motion_vertex_mag)
                    if first_layer_skip_gradient:
                        continue
                    if motion_vertex_mag is not None and abs(motion_vertex_mag) > 0:
                        corrected_data_ids.setdefault(motion_idx, []).append(vertex_idx)
                    cumulative_motion_dict.setdefault(motion_idx, []).append(motion_vertex_data_point[vertex_adj_sort_index])
            for motion_idx in cumulative_motion_dict:
                if motion_idx in corrected_data_ids.keys():
                    accepted_data_pts = corrected_data_ids[motion_idx]
                else:
                    accepted_data_pts = range(len(cumulative_motion_dict[motion_idx]))
                cumulative_motion_dict_filtered = [cumulative_motion_dict[motion_idx][i] for i in accepted_data_pts]
                avg_motion_vertex = np.mean(np.asarray(cumulative_motion_dict_filtered), axis=0)
                avg_motion_vertex = (avg_motion_vertex.shape[0] * network.motion_normalization_mult) * \
                                    avg_motion_vertex / np.max(np.abs(avg_motion_vertex) + 0.00001)
                sort_index_motion = score_order(-np.abs(avg_motion_vertex), sorted_adj_vertices)
                motion_last_layer_next.append([sorted_adj_vertices[sort_index_motion].tolist(),
                                               avg_motion_vertex[sort_index_motion]])

            motion_last_layer = motion_last_layer_next

            if index in self.updating_layers:
                for layer_vertex_key in layer.graph.vertex_list:
                    layer.graph.vertex_list[layer_vertex_key].apply_motion()
                layer.update_index_matrix()
            else:
                for layer_vertex_key in layer.graph.vertex_list:
                    vertex_ = layer.graph.vertex_list[layer_vertex_key]
                    vertex_.clean_motion_accumulation(vertex_.adjacency_list)

        network.num_of_epochs += 1
        network.adapt_learning_rate()
        return motion_last_layer

    # ------------------------------------------------------------------ the arm's transform and its bookkeeping

    def _transform(self, index, layer, motion_last_layer):
        """The only injected step. Returns the motion records this layer consumes, after the arm's randomization."""
        keys = list(layer.graph.vertex_list)
        accepted = int(np.ceil(self.network.ratio_data_backprop * len(keys)))
        before = self._statistics(motion_last_layer, accepted)
        if index not in self.first_batch:
            self.first_batch[index] = [_signal_hash(record) for record in motion_last_layer]
        if self.arm == 'permuted_alignment':
            transformed, changed = self._permute_alignment(keys, accepted, motion_last_layer)
        elif self.arm == 'random_direction':
            transformed, changed = self._randomize_direction(accepted, motion_last_layer)
        else:
            transformed, changed = motion_last_layer, 0
        after = self._statistics(transformed, accepted)
        deviation = self._require_matched(index, before, after)
        entry = self.statistics.setdefault(index, {'batches': 0, 'records': 0, 'accepted_per_record': accepted,
                                                   'accepted_slots': 0, 'changed_slots': 0, 'max_mass_deviation': 0.,
                                                   'before': _zero_counts(), 'after': _zero_counts()})
        entry['batches'] += 1
        entry['records'] += len(motion_last_layer)
        entry['accepted_slots'] += before['slots']
        entry['changed_slots'] += int(changed)
        entry['max_mass_deviation'] = max(entry['max_mass_deviation'], deviation)
        _add_counts(entry['before'], before)
        _add_counts(entry['after'], after)
        return transformed

    def _require_matched(self, index, before, after):
        """The matching invariant, enforced on every batch and layer, not only in the aggregate: the transform never
        changes the accepted slots, the number of effective votes, the multiset of accepted |motion| values, the accepted
        vote mass or the counts of attraction and repulsion."""
        deviation = abs(before['mass'] - after['mass'])
        broken = [name for name, ok in (('slots', before['slots'] == after['slots']),
                                        ('effective_votes', before['effective'] == after['effective']),
                                        ('magnitude_multiset', before['magnitudes'] == after['magnitudes']),
                                        ('sign_counts', (before['positive'], before['negative'])
                                         == (after['positive'], after['negative'])),
                                        ('accepted_mass', deviation <= 1e-9 * max(1., abs(before['mass'])))) if not ok]
        if broken:
            raise CheckFailed(f'mass_matched: the {self.arm} transform of layer {index} changed {", ".join(broken)}')
        return float(deviation)

    @staticmethod
    def _statistics(records, accepted):
        """(count, effective votes, |motion| mass, positive, negative, magnitude multiset digest) over accepted slots."""
        mass = positive = negative = effective = 0
        digest = hashlib.sha256()
        for _, magnitudes in records:
            values = np.asarray(magnitudes, dtype=float)[:accepted]
            absolute = np.abs(values)
            mass += float(absolute.sum())
            effective += int(np.count_nonzero(values))
            positive += int(np.count_nonzero(values > 0))
            negative += int(np.count_nonzero(values < 0))
            digest.update(np.ascontiguousarray(np.sort(absolute)).tobytes())
        return {'slots': sum(min(accepted, len(m)) for _, m in records), 'effective': effective, 'mass': mass,
                'positive': positive, 'negative': negative, 'magnitudes': digest.hexdigest()}

    def _permute_alignment(self, keys, accepted, motion_last_layer):
        """One uniform permutation of this layer's filter identities per batch: the filter that receives a motion changes,
        the motion vector (its values, their order and so the accepted slots and the eligibility gate) does not. The
        reported count is the accepted slots whose filter changed."""
        order = self.rng.permutation(len(keys))
        self.draws += 1
        mapping = {keys[i]: keys[order[i]] for i in range(len(keys))}
        out, changed = [], 0
        for record_keys, magnitudes in motion_last_layer:
            new_keys = [mapping[key] for key in record_keys]
            changed += sum(1 for a, b in zip(record_keys[:accepted], new_keys[:accepted]) if a != b)
            out.append([new_keys, magnitudes])
        return out, changed

    def _randomize_direction(self, accepted, motion_last_layer):
        """The signs of the accepted nonzero votes permuted among the accepted nonzero slots, per record and per batch.
        |motion| is untouched element for element, so the accepted slots, the accepted mass, the magnitude multiset, the
        number of effective votes and the counts of attraction and repulsion are all preserved exactly; only which filter
        is pulled towards the input and which away from it is randomized."""
        out, changed = [], 0
        for record_keys, magnitudes in motion_last_layer:
            values = np.asarray(magnitudes, dtype=float).copy()
            head = values[:accepted]
            nonzero = np.flatnonzero(head)
            if len(nonzero) > 1:
                signs = np.sign(head[nonzero])
                order = self.rng.permutation(len(nonzero))
                self.draws += 1
                new = np.abs(head[nonzero]) * signs[order]
                changed += int(np.count_nonzero(new != head[nonzero]))
                head[nonzero] = new
            values[:accepted] = head
            out.append([record_keys, values])
        return out, changed

    # ------------------------------------------------------------------ reads after training

    def displacement_by_layer(self):
        """Per layer (each hidden layer, then the output layer): (mean normalized footrule from the initial filters,
        share of filters that differ)."""
        return [td.displacement(current, initial)
                for current, initial in zip(self.final['matrices'], self.initial['matrices'])]

    def matching(self):
        """Per hidden layer, whether the transform preserved the accepted vote count, the accepted |motion| multiset, the
        accepted mass and the sign counts. random_direction permutes signs among slots, so its per-record sign counts are
        preserved but the sign pairing is not; every arm preserves the rest exactly."""
        rows, passed = {}, True
        for index in sorted(self.statistics):
            entry = self.statistics[index]
            before, after = entry['before'], entry['after']
            checks = {'slots': before['slots'] == after['slots'],
                      'effective_votes': before['effective'] == after['effective'],
                      'magnitude_multiset': before['magnitudes'] == after['magnitudes'],
                      'accepted_mass': abs(before['mass'] - after['mass']) <= 1e-9 * max(1., abs(before['mass'])),
                      'sign_counts': (before['positive'], before['negative']) == (after['positive'], after['negative'])}
            passed = passed and all(checks.values())
            rows[str(index)] = {'batches': entry['batches'], 'records': entry['records'],
                                'accepted_per_record': entry['accepted_per_record'],
                                'max_mass_deviation': entry['max_mass_deviation'],
                                'accepted_slots': entry['accepted_slots'], 'changed_slots': entry['changed_slots'],
                                'changed_share': (entry['changed_slots'] / entry['accepted_slots']) if entry['accepted_slots'] else 0.,
                                'before': before, 'after': after, 'checks': checks}
        return {'passed': passed, 'layers': rows, 'draws': self.draws, 'calls': self.calls}

    def first_batch_signal(self):
        return {str(index): [list(pair) for pair in records] for index, records in sorted(self.first_batch.items())}


def _filters_hash(copy_of_filters):
    """One hash of every layer's filter position matrix (the state a view starts or ends training in)."""
    digest = hashlib.sha256()
    for matrix in copy_of_filters['matrices']:
        digest.update(array_hash(np.ascontiguousarray(np.asarray(matrix, dtype=float))).encode())
    return digest.hexdigest()


def _zero_counts():
    return {'slots': 0, 'effective': 0, 'mass': 0., 'positive': 0, 'negative': 0, 'magnitudes': None}


def _add_counts(total, entry):
    for key in ('slots', 'effective', 'mass', 'positive', 'negative'):
        total[key] += entry[key]
    digest = hashlib.sha256((total['magnitudes'] or '').encode())
    digest.update(entry['magnitudes'].encode())
    total['magnitudes'] = digest.hexdigest()


# ----------------------------------------------------------------------------- fitting one arm

def arm_fit(arm, params, seed, X, y, *, identity, hidden_layers):
    """MultiViewArrowFlowKNN(**params, seed=seed).fit(X, y), step for step (MultiViewArrowFlowKNN.fit, MultiViewArrowFlow.fit
    and ArrowFlowEstimator.fit_orders = initialize_orders then train_initialized), with the arm installed on each view
    network for its training only. Returns the fitted model and the seven MotionArm records."""
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
                                 p_correct=model.p_correct, seed=seed_v,
                                 validation_ratio=model.validation_ratio, augment=model.augment,
                                 n_augmentations=model.n_augmentations, max_swaps=model.max_swaps)
        net.initialize_orders(orders, y)
        motion = MotionArm(net, arm, seed=arm_seed(arm, identity, v), hidden_layers=hidden_layers)
        with motion.installed():
            net.train_initialized(orders, y)
        training += net.training_seconds_
        model.views_.append((enc, net))
        model._fit_view_readout(enc, net, orders, y, seed_v)
        arms.append(motion)
    model.encoding_seconds_ = encoding
    model.training_seconds_ = training
    return model, arms


def arm_seed(arm, identity, view):
    """The recorded stream seed of one arm, fold, fitting seed and view; every randomization draw comes from it."""
    return derive_seed(MOTION_SEED, str(arm), str(identity['dataset_id']), int(identity['outer_repeat']),
                       int(identity['outer_fold']), int(identity['model_seed']), int(view))


# ----------------------------------------------------------------------------- measures

def per_class_recall(y_true, predicted, classes):
    values = recall_score(y_true, predicted, labels=list(classes), average=None, zero_division=0)
    return {str(label): float(value) for label, value in zip(classes, values)}


def _drop_self(indices, rows):
    """Leave-one-out neighbours: drop each query row's own stored index (the last neighbour when it is not returned)."""
    out = np.empty((indices.shape[0], indices.shape[1] - 1), dtype=indices.dtype)
    for i, row in enumerate(rows):
        neighbours = indices[i]
        hit = np.flatnonzero(neighbours == row)
        drop = int(hit[0]) if len(hit) else indices.shape[1] - 1
        out[i] = np.delete(neighbours, drop)
    return out


def neighborhood_purity(readout, train_labels, query_positions, query_labels, *, k, exclude_rows=None):
    """Mean over query rows of the share of the row's k nearest stored training rankings that carry the row's class.
    exclude_rows: the stored index of each query row, dropped from its own neighbourhood (training rows)."""
    query_positions = np.asarray(query_positions)
    if not len(query_positions):
        return None
    probe = StableFootruleKNN(n_neighbors=k + (1 if exclude_rows is not None else 0), weights=readout.weights,
                              input_kind='positions').fit(readout.positions_, train_labels)
    indices = probe.kneighbors(query_positions, return_distance=False)
    if exclude_rows is not None:
        indices = _drop_self(indices, np.asarray(exclude_rows))
    indices = indices[:, :k]
    neighbours = np.asarray(train_labels)[indices]
    return float(np.mean(neighbours == np.asarray(query_labels).reshape(-1, 1)))


def purity_record(model, orders_train, orders_test, y_train, y_test, *, query_cap=PURITY_QUERY_CAP):
    """Same-class neighbourhood purity of the last hidden ranking, at each view's selected k, averaged over views: on the
    outer test rows (all of them) and on the training rows (leave-one-out, a deterministic every-m-th subsample when the
    training partition exceeds query_cap rows)."""
    step = max(1, int(np.ceil(len(y_train) / query_cap)))
    rows = np.arange(len(y_train))[::step]
    test_values, train_values, ks = [], [], []
    for (enc, net), readout, o_train, o_test in zip(model.views_, model.readouts_, orders_train, orders_test):
        hidden_train, hidden_test = net.transform_orders(o_train), net.transform_orders(o_test)
        k = min(int(readout.n_neighbors), len(y_train))
        ks.append(k)
        test_values.append(neighborhood_purity(readout, y_train, hidden_test, y_test, k=k))
        train_values.append(neighborhood_purity(readout, y_train, hidden_train[rows], np.asarray(y_train)[rows],
                                                k=min(k, len(y_train) - 1), exclude_rows=rows))
    present = [v for v in test_values if v is not None]
    return {'test': float(np.mean(present)) if present else None,
            'train': float(np.mean([v for v in train_values if v is not None])) if train_values else None,
            'test_by_view': test_values, 'train_by_view': train_values, 'k_by_view': ks,
            'train_query_rows': int(len(rows)), 'train_query_step': step,
            'definition': 'share of the k nearest stored training rankings of the last hidden layer that carry the query '
                          'row\'s class, at the view\'s selected n_neighbors, averaged over rows then over views; the '
                          'training query drops each row\'s own stored ranking'}


# ----------------------------------------------------------------------------- protocol

def design():
    """The fixed design every motion-controls protocol declares (validate_protocol requires it verbatim)."""
    return {
        'purpose': 'matched motion-signal controls: does ArrowFlow\'s learning signal carry information, or would comparable '
                   'random movement of the same filters do as well?',
        'selection_statement': 'Nothing is selected. Every arm is fitted at the per-fold selection sealed by the reference '
                               'ablation run; every kNN readout setting is the one the fit chose on training rows only; no '
                               'outer-fold score chooses anything; the analysis is frozen with this protocol, before any outer '
                               'score exists.',
        'unit_of_work': 'every protocol dataset x every outer fold (5 folds x 3 repeats) x every fitting seed x all six arms',
        'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3, 'split_seed': 27183, 'fit_seeds': [8129, 19391, 39019],
        'selection_metric': 'accuracy', 'test_train_ratio': 0.25, 'confidence': 0.95,
        'arms': list(ARMS),
        'matched': 'Every arm is fitted at ArrowFlow-kNN\'s reconstructed per-fold selection, resolved from the outer training '
                   'partition\'s shape (bridge.resolve_selected), on the same folds, the same three fitting seeds, the same '
                   'encoders and seeded initial filters, the same training rows, the same view identities, the same training '
                   'budget (iterations, batch size, learning-rate schedule, validation checkpoint and augmentation) and the '
                   'same kNN readout rule. Only the hidden-layer update changes.',
        'arm_definitions': {
            'views7': 'the unmodified method (MultiViewArrowFlowKNN at the reconstructed selection), fitted through the same '
                      'instance-level wrapper with an identity transform and every hidden layer updating; it must reproduce '
                      'the reference run\'s outer predictions exactly for every dataset, outer fold and fitting seed',
            'frozen': 'the hidden ranking filters never move: no hidden layer applies its accumulated motion, so every hidden '
                      'filter keeps its seeded initial order for the whole fit. Everything else is unchanged, including the '
                      'forward pass, the eligibility gate, the auxiliary class-prototype (output) layer\'s own accumulation '
                      'and update, the learning-rate schedule, the validation checkpoint, augmentation and the kNN readout, '
                      'which is still selected and refitted on the (now untrained) hidden rankings.',
            'permuted_alignment': 'the motion records reaching each hidden layer are re-keyed by a uniformly random '
                                  'permutation of that layer\'s filter identities, drawn once per layer and batch from the '
                                  'recorded arm seed. The motion vector itself is untouched, so the set of motions, their '
                                  'magnitudes and signs, the accepted slots (ratio_data_backprop), the number of votes and '
                                  'the eligibility gate are all preserved; only which filter receives which motion changes.',
            'random_direction': 'within each motion record, the signs of the accepted nonzero votes are permuted among the '
                                'accepted nonzero slots, drawn per record, layer and batch from the recorded arm seed. '
                                '|motion| is untouched element for element, so the accepted slots, the accepted vote mass per '
                                'batch and per layer, the magnitude multiset, the number of effective votes and the counts of '
                                'attraction and repulsion are preserved exactly; only which filter is pulled towards the batch '
                                'input and which away from it is randomized. Zero votes stay zero, so no vote is created or '
                                'destroyed. Because the sign counts are held fixed, the share of accepted slots whose '
                                'direction actually changes is bounded by 2p(1-p) with p the realized share of attractions '
                                'among the accepted votes; p, the realized change share and the per-batch mass deviation are '
                                'recorded per dataset, arm and hidden layer, so how much was randomized is reported, not '
                                'assumed. Randomizing the signs independently instead would raise the change share to about '
                                'a half but would also move the attraction/repulsion mix away from the real run\'s, which '
                                'would confound a difference in direction pairing with a difference in that mix; the '
                                'mass-matched permutation is chosen for that reason.',
            'single_layer_first': 'only hidden layer 0 applies its accumulated motion; the deeper hidden layer, where there is '
                                  'one, resets its accumulator instead. At a one-hidden-layer selection this is views7 by '
                                  'construction: it is then recorded as identical_to_views7 and not fitted again.',
            'single_layer_last': 'only the last hidden layer (the one the kNN readout reads) applies its accumulated motion. '
                                 'At a one-hidden-layer selection this is views7 by construction: it is then recorded as '
                                 'identical_to_views7 and not fitted again.'},
        'supervision_disclosure': 'The eligibility gate that decides which examples produce votes reads the labels: an example '
                                  'votes when its predicted class is wrong, and, when it is right, only with probability '
                                  'p_correct; the vote is then accumulated on the ground-truth class\'s output filter. That '
                                  'gate, and the auxiliary output layer\'s own label-driven update, are PRESERVED in every arm. '
                                  'What is randomized is only which hidden filter receives which motion (permuted_alignment) '
                                  'and in which direction a hidden filter is moved (random_direction), or whether hidden '
                                  'filters move at all (frozen, single_layer). A full label-permutation control, in which the '
                                  'labels themselves are permuted, is a different experiment and is not claimed here.',
        'differs_from_untrained': 'The existing untrained control (knn_controls.UntrainedMultiViewArrowFlowKNN, the ablations\' '
                                  '"untrained" variant and the runs\' arrowflow_knn_untrained family) never trains at all: no '
                                  'layer is ever updated, the auxiliary output layer included, no batch is drawn, no validation '
                                  'checkpoint and no augmentation act, and its training-only settings (learning rate, '
                                  'iterations, batch size, validation_ratio, augment) are not even parameters, so a network '
                                  'that is never trained carries ArrowFlowEstimator defaults for them; as a model family in the '
                                  'production runs it is also tuned on its own inner folds over its own projected candidate '
                                  'grid (knn_controls.control_candidates). The frozen arm here does train: the same batches, '
                                  'the same forward pass, the same label-driven eligibility gate, the same auxiliary '
                                  'class-prototype layer accumulation and update, the same learning-rate schedule, checkpoint '
                                  'and augmentation, and the same readout, at views7\'s own selection. Only the hidden ranking '
                                  'filters are held still. frozen therefore isolates the hidden-layer update alone, while '
                                  'untrained removes training altogether.',
        'instrumentation': 'an instance-level backward_propagate on each view network, installed after initialize_orders and '
                           'removed after train_initialized; it is a transcription of '
                           'SortFlowHybridNetwork.backward_propagate restricted to the sort-only path (every layer "sort", '
                           'average_method_motion "mean", backprop_signal_replication 1) with exactly one injected step, the '
                           'arm\'s transform of the motion records entering each hidden layer, and one changed step, which '
                           'hidden layers apply their accumulated motion; every draw comes from a private RandomState seeded '
                           'by derive_seed(20260914, arm, dataset, outer_repeat, outer_fold, model_seed, view); every call '
                           'saves and restores the global numpy and Python RNG states and records whether they changed; the '
                           'pinned sha256 of every core function this transcription depends on is re-checked in every job',
        'measures': {
            'accuracy': 'outer-test accuracy, error, balanced accuracy and macro-F1 of the 7-view kNN majority (the paper\'s '
                        'metric) per arm, dataset, outer fold and fitting seed',
            'per_class_recall': 'sklearn recall_score(labels=the dataset classes, average=None, zero_division=0) of the same '
                                'predictions, per class',
            'neighborhood_purity': 'same-class neighbourhood purity of the last hidden ranking: for each outer test row the '
                                   'share of its k nearest stored training rankings that carry its class, at the view\'s '
                                   'selected n_neighbors, averaged over rows and then over the seven views; and the same on '
                                   'the training rows with each row\'s own stored ranking dropped, over a deterministic '
                                   f'every-m-th subsample when the training partition exceeds {PURITY_QUERY_CAP} rows',
            'displacement': 'per layer (each hidden layer, then the output layer) the mean over filters of '
                            'footrule(final filter, initial filter) / floor(n^2 / 2) and the share of filters that differ, '
                            'averaged over the seven views; the final filters are the network\'s, i.e. the checkpoint the core '
                            'restores',
            'motion_statistics': 'per hidden layer, the batches and motion records seen, the accepted slots, the share of '
                                 'accepted slots the transform changed, and the accepted vote count, effective votes, mass, '
                                 'sign counts and magnitude-multiset digest before and after the transform'},
        'checks': {
            'reference_predictions': 'every job and fitting seed: views7\'s 7-view kNN outer predictions equal the reference '
                                     'run\'s recorded predictions exactly, in the sealed test order',
            'uninstrumented_fit': 'first outer fold of every dataset, first fitting seed: views7\'s seven view state_hash() '
                                  'and its kNN predictions equal those of MultiViewArrowFlowKNN(**selected, seed).fit on the '
                                  'same rows, fitted without any wrapper',
            'rng_untouched': 'every job: every wrapper call and every guarded read left the global numpy and Python RNG states '
                             'unchanged',
            'wrapper_removed': 'every job: no view network keeps an instance-level backward_propagate after its fit',
            'core_sources': 'every job: the pinned sha256 of every core function the transcription depends on is unchanged',
            'initial_state': 'every job: every arm\'s captured initial filters and initial state hash equal views7\'s, per view',
            'mass_matched': 'every job and randomizing arm: per hidden layer the transform preserved the accepted vote count, '
                            'the accepted |motion| multiset, the accepted mass and the counts of attraction and repulsion',
            'first_batch_signal': 'every job: at the first batch, before any filter has moved, every arm\'s pre-transform '
                                  'motion records equal views7\'s per view and hidden layer, so the arms start from the '
                                  'identical signal and differ only by the transform. The one exception is '
                                  'permuted_alignment below the last hidden layer: a deeper layer reads the mean, over the '
                                  'accepted filters above it, of each accepted filter\'s own motion against the batch '
                                  'input, so re-keying, which changes which filters are accepted, changes it, while '
                                  'permuting signs does not (that mean depends on the signs only through how many are '
                                  'attractions and how many repulsions, which the permutation preserves; the changed signs '
                                  'reach the filters through the accumulators instead). Only the last hidden layer is '
                                  'therefore compared for permuted_alignment.',
            'frozen_unmoved': 'every job: the frozen arm\'s hidden-layer displacement is exactly zero and its share of '
                              'unchanged filters exactly one, for every view and hidden layer',
            'single_layer_identity': 'every job at a one-hidden-layer selection: both single_layer arms are recorded '
                                     'identical_to_views7 and their predictions equal views7\'s'},
        'fit_reuse': 'views7, frozen, permuted_alignment and random_direction are fitted once per outer fold and fitting seed. '
                     'At a two-hidden-layer selection single_layer_first and single_layer_last are fitted too; at a '
                     'one-hidden-layer selection both are views7 by construction and are recorded as identical_to_views7 with '
                     'zero timings, naming their source fit, rather than fitted twice.',
        'analysis': {
            'declared': 'frozen with this protocol, before any outer score exists',
            'primary_family': {'contrast': 'views7 minus the control arm', 'metric': 'accuracy',
                               'arms': list(PRIMARY_ARMS), 'datasets': 'the seventeen protocol datasets',
                               'aggregation': 'fitting seeds averaged within each outer fold',
                               'interval': 'corrected resampled t (evaluation.paired_corrected_interval): standard error '
                                           'sqrt((1/15 + q) * variance (ddof 1) of the fold differences), q = 0.25, t quantile '
                                           'with 14 df, 95%',
                               'adjustment': 'Holm across the seventeen datasets within each arm (evaluation.holm_adjust); the '
                                             'three arms are adjusted separately',
                               'direction': 'positive favours ArrowFlow (views7)'},
            'descriptive': {'neighborhood_purity': 'per dataset and arm, outer-fold mean and SD of the test and training purity',
                            'per_class_recall': 'per dataset, arm and class, outer-fold mean and SD',
                            'displacement': 'per dataset, arm and layer, outer-fold mean of the normalized footrule '
                                            'displacement and of the share of unchanged filters',
                            'single_layer': 'views7 minus single_layer_first and views7 minus single_layer_last, accuracy, '
                                            'restricted to the outer folds whose selection has two hidden layers; the fold '
                                            'count is reported with every row. No dataset has fifteen such folds, so these '
                                            'contrasts are descriptive: the interval is a corrected resampled t over the '
                                            'qualifying folds at df = n_folds - 1 when n_folds >= 3, they are not part of the '
                                            'primary family and they are not Holm-adjusted.'},
            'named_subset': {'datasets': list(NAMED_SUBSET),
                             'definition': 'the same primary-family contrasts restricted to the two datasets whose training '
                                           'gains survive the paper\'s Holm-34 correction (balance-scale and mfeat-zernike)',
                             'status': 'a named subset of the primary family, not a new family: the rows carry the '
                                       'primary family\'s own Holm adjustment over the seventeen datasets and are not '
                                       'adjusted again'},
            'interpretation': 'Fixed in advance. If the real signal does not beat comparable random movement, the paper says '
                              'so and narrows its contribution to a construction plus an adaptation scheme. The claim is not '
                              'rescued after the fact, and no arm, dataset or subset is added, dropped or reweighted after an '
                              'outer score is seen.',
            'verification': 'the analysis refuses (exit 2, nothing written) until the run is complete, and then re-verifies '
                            'every reference run with compare_runs.load_run and compare_runs.verify_run and this run with '
                            'motion_controls.summary before any score is read'},
        'reporting': {'outputs': f'{SUMMARY_JSON} (metrics, descriptive intervals against views7, purity, recall, displacement, '
                                 f'motion statistics, reproduction counts and model rows) and {SUMMARY_CSV} (metrics per '
                                 'dataset, arm and metric); the prespecified analysis is compare_motion.analyse',
                      'change_from_views7': 'per dataset and arm, arm minus views7 in accuracy and error: fitting seeds '
                                            'averaged within outer fold, corrected resampled t interval over the 15 outer '
                                            'folds (q = test_train_ratio, 14 df); descriptive in the summary, without p values '
                                            'or multiplicity adjustment; the tested family is compare_motion.analyse'},
        'report_metrics': list(METRICS),
        'dataset_loading': 'the reference run prepared data (hash-checked) must equal the ablation copy, its splits the '
                           'declared nested splits, and in production a fresh load by the reference loader '
                           '(run_revision.load_dataset or newdata.load_newdata with every pin) must return the same arrays and '
                           'dataset hash',
        'failure_policy': 'log_all_failures; a failed check or fit fails its job, cancels the pending jobs and blocks the '
                          'summary; failures are never omitted; no adaptive stopping on any score',
        'pilot': 'training-only: every arm on the first outer training partition of each pilot dataset at the first fitting '
                 'seed, predicting on every fourth training row (the outer test fold is never touched), plus one '
                 'uninstrumented fit and one reproduction probe (run_knn_ablation.reproduction_probe) per reference',
        'decision_rule': f'freeze only if the calibrated projection at {WORKERS} single-thread workers is at most {CAP_HOURS} h: '
                         'the simulated first-free-worker makespan of the planned jobs in planned order, each job priced at the '
                         'reference run\'s realized outer fit and predict seconds of its fold and its three fitting seeds times '
                         'the piloted all-arms/views7 time ratio of the pilot dataset with the same hidden depth (the largest '
                         'piloted ratio when no pilot dataset shares it), plus the views7 seconds once more for a first fold\'s '
                         'uninstrumented fit',
        'wallclock_cap_hours': CAP_HOURS, 'workers': WORKERS, 'max_workers': MAX_WORKERS, 'numeric_threads_per_worker': 1,
        'parallelism': 'one spawned single-thread process per dataset and outer fold under the shared execution lock',
        'motion_seed': MOTION_SEED, 'purity_query_cap': PURITY_QUERY_CAP,
        'reference_argument': '--reference NAME=RUN_DIR,ABLATION_DIR, one per protocol reference',
    }


def draft_protocol(references=None, protocol_id=PROTOCOL_ID, pilot_datasets=PILOT_DATASETS):
    """The unfrozen protocol; the datasets follow the references in name order (a JSON protocol keeps no key order)."""
    references = REFERENCES if references is None else references
    datasets = [name for key in sorted(references) for name in references[key]['datasets']]
    return _plain({**design(), 'protocol_id': protocol_id, 'production_family': FAMILY, 'datasets': datasets,
                   'references': references, 'pilot_datasets': list(pilot_datasets), 'frozen': False, 'status': DRAFT_STATUS,
                   'resource_decision': 'pending: synthetic smoke and the training-only pilot on ' + ' and '.join(pilot_datasets)})


def validate_protocol(p):
    """A production protocol is draft_protocol for its references, or that draft with exactly the freeze fields set by freeze."""
    draft = draft_protocol(references=p.get('references'), protocol_id=p.get('protocol_id'),
                           pilot_datasets=p.get('pilot_datasets', ()))
    strip = lambda q: {k: v for k, v in q.items() if k not in FREEZE_FIELDS}
    if strip(p) != strip(draft):
        differing = sorted(k for k in set(p) | set(draft) if k not in FREEZE_FIELDS and p.get(k) != draft.get(k))
        raise ValueError(f'The protocol differs from motion_controls.draft_protocol() in {", ".join(differing)}')
    td.reference_of(p)
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


# ----------------------------------------------------------------------------- plan

def fit_sources(hidden_layers):
    sources = {arm: 'fitted' for arm in ARMS}
    if hidden_layers < 2:
        for arm in DEPTH_ARMS:
            sources[arm] = 'identical_to_views7'
    return sources


def planned_job(p, reference, name, index, split, record, n_features):
    selected = resolve_selected(record['config'], n_features, len(split['train']))
    sealed_job = reference['ablation']['jobs'].get((name, split['outer_repeat'], split['outer_fold']))
    if sealed_job is None or sealed_job['config_id'] != record['config_id'] or sealed_job['selected'] != selected:
        raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the resolved selection differs from the '
                         'ablation plan')
    params = {key: value for key, value in selected.items() if key not in ('embed_scale', 'degree_offset')}
    if params.get('n_views') != 7 or params.get('aggregation') != 'majority':
        raise ValueError('The motion controls are defined for the seven-view majority-vote selected configuration')
    hidden_layers = len(selected['widths'])
    seeds = list(p['fit_seeds'])
    return {'dataset_id': name, 'reference': reference['name'], 'outer_repeat': split['outer_repeat'],
            'outer_fold': split['outer_fold'], 'stem': f'{name}__r{split["outer_repeat"]}f{split["outer_fold"]}',
            'config_id': record['config_id'], 'config': record['config'], 'selected': selected,
            'selected_widths': list(selected['widths']), 'hidden_layers': hidden_layers, 'params': params,
            'arms': list(ARMS), 'fit_sources': fit_sources(hidden_layers), 'model_seeds': seeds,
            'arm_seeds': {arm: {str(seed): [arm_seed(arm, {'dataset_id': name, 'outer_repeat': split['outer_repeat'],
                                                           'outer_fold': split['outer_fold'], 'model_seed': seed}, v)
                                            for v in range(7)] for seed in seeds} for arm in ARMS},
            'check_uninstrumented': index == 0,
            'reference_prediction_hashes': dict(record['reference_prediction_hashes']),
            'reference_outer_seconds': sum(record['reference_outer_seconds'].values())}


def prepare(output, p, sources, datasets=None, *, allow_smoke=False, purpose='confirmatory'):
    """Verify the references, copy the prepared data, reconstruct every selection (it must equal the sealed record and
    resolve to the ablation plan) and seal the planned jobs and the reference predictions."""
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
                raise ValueError(f'{name} r{split["outer_repeat"]}f{split["outer_fold"]}: the selection reconstructed from '
                                 'the reference run differs from the sealed record')
            selections.append(record)
            jobs.append(planned_job(p, reference, name, index, split, record, X.shape[1]))
    write_json(output/'reference_selections.json', selections)
    write_json(output/'planned_jobs.json', jobs)
    write_json(output/'manifest.json', {
        'purpose': purpose, 'protocol_id': p['protocol_id'], 'protocol_hash': config_id(p), 'datasets': chosen,
        'arms': list(ARMS), 'fit_seeds': list(p['fit_seeds']), 'planned_jobs': len(jobs),
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
    planned jobs and sealed selections. environment_check 'full' (run) or 'sources' (summary)."""
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    if not smoke:
        validate_protocol(p)
        if not p['frozen'] or manifest.get('purpose') != 'confirmatory':
            raise ValueError('A motion-controls run directory with a frozen reviewed protocol is required')
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
    for job in jobs:
        record = selections[(job['dataset_id'], job['outer_repeat'], job['outer_fold'])]
        if (record['fitting_seeds'] != list(p['fit_seeds'])
                or sorted(record['reference_predictions']) != sorted(map(str, p['fit_seeds']))):
            raise ValueError(f'Missing or inconsistent sealed selection for {job["stem"]}')
        for seed, labels in record['reference_predictions'].items():
            if array_hash(np.asarray(labels)) != record['reference_prediction_hashes'][seed]:
                raise ValueError(f'Sealed reference predictions of {job["stem"]} disagree with their hashes')
    return p, manifest, jobs, selections


# ----------------------------------------------------------------------------- fitting one job

def _majority(view_predictions):
    return majority(np.stack(view_predictions))


def uninstrumented_fit(params, seed, X_train, y_train, X_test):
    """MultiViewArrowFlowKNN(**params, seed=seed).fit(X_train, y_train), with no wrapper anywhere: the seven view state
    hashes and the 7-view kNN predictions."""
    seed_fit(seed)
    model = MultiViewArrowFlowKNN(**params, seed=seed).fit(X_train, y_train)
    predictions = _majority(model.predict_views(X_test)[0])
    return [net.state_hash() for _, net in model.views_], predictions


def evaluate_job(X, y, split, p, job, sealed, *, dataset_hash, code_revision, sink=None, query=None):
    """Every arm on one outer fold, for every fitting seed. `query` replaces the outer test rows (the training-only pilot)."""
    name, seeds = job['dataset_id'], list(p['fit_seeds'])
    train = split['train']
    test = [int(i) for i in (split['test'] if query is None else query)]
    held_out = query is None
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    params, sources = job['params'], job['fit_sources']
    classes = np.unique(y).tolist()
    common = {'dataset_id': name, 'dataset_hash': dataset_hash, 'outer_repeat': split['outer_repeat'],
              'outer_fold': split['outer_fold'], 'config_id': job['config_id'], 'code_revision': code_revision}
    identity = dict(common, split_hash=config_id(split), train_ids=[int(i) for i in train], query_ids=test,
                    query='outer_test_rows' if held_out else 'training_rows_only',
                    raw_train_hash=array_hash(X_train), raw_query_hash=array_hash(X_test),
                    training_labels_hash=array_hash(y_train), config=job['config'], selected=job['selected'],
                    params=params, hidden_layers=job['hidden_layers'], arms=list(ARMS), fit_sources=sources,
                    model_seeds=seeds, arm_seeds=job['arm_seeds'], check_uninstrumented=job['check_uninstrumented'],
                    reference_prediction_hashes=job['reference_prediction_hashes'], protocol_hash=config_id(p))
    result = {'status': 'running', 'identity': identity, 'fits': [], 'reproduction': [], 'models': [], 'predictions': [],
              'events': [], 'checks': {}, 'timing': {}}

    def emit(stage, record):
        event = {'stage': stage, 'record': record}
        result['events'].append(event)
        if sink is not None:
            sink(event)

    emit('identity', identity)
    checks = {name: {'performed': False, 'passed': None} for name in JOB_CHECKS}
    result['checks'] = checks
    predictions, fitted, started = {}, {}, time.perf_counter()
    rng_ok, wrappers_removed, initial_hashes, first_batch, matching = True, True, {}, {}, {}
    try:
        source_check = check_core_sources()
        checks['core_sources'] = source_check
        if not source_check['passed']:
            raise CheckFailed(f'core_sources: {", ".join(source_check["changed"])} changed')
        with threadpool_limits(limits=1):
            for seed in seeds:
                for arm in ARMS:
                    if sources[arm] == 'identical_to_views7':
                        predictions[(arm, seed)] = predictions[('views7', seed)]
                        fitted[(arm, seed)] = _record_fit(result, emit, arm, seed, params, sources, len(y_train),
                                                          reused_from=f'views7__s{seed}',
                                                          **dict.fromkeys(('fit_seconds', 'encoding_seconds',
                                                                           'training_seconds', 'readout_seconds'), 0.),
                                                          **{key: fitted[('views7', seed)][key]
                                                             for key in ('purity', 'displacement', 'motion')})
                        continue
                    seed_fit(seed)
                    start = time.perf_counter()
                    model, arms = arm_fit(arm, params, seed, X_train, y_train,
                                          identity={'dataset_id': name, 'outer_repeat': split['outer_repeat'],
                                                    'outer_fold': split['outer_fold'], 'model_seed': seed},
                                          hidden_layers=job['hidden_layers'])
                    elapsed = time.perf_counter() - start
                    seeds_used = [motion.seed for motion in arms]
                    if seeds_used != job['arm_seeds'][arm][str(seed)]:
                        raise CheckFailed(f'arm_seeds: {arm} s{seed} used stream seeds that differ from the sealed plan')
                    view_predictions, orders_test = model.predict_views(X_test)
                    orders_train = [enc.transform(X_train) for enc, _ in model.views_]
                    predicted = _majority(view_predictions)
                    predictions[(arm, seed)] = predicted
                    rng_ok = rng_ok and all(all(motion.rng_checks) for motion in arms)
                    wrappers_removed = wrappers_removed and all(motion.wrapper_removed() for motion in arms)
                    initial_hashes[(arm, seed)] = [_filters_hash(motion.initial) for motion in arms]
                    first_batch[(arm, seed)] = [motion.first_batch_signal() for motion in arms]
                    matching[(arm, seed)] = [motion.matching() for motion in arms]
                    displacement = _mean_displacement([motion.displacement_by_layer() for motion in arms])
                    purity = purity_record(model, orders_train, orders_test, y_train, y_test)
                    fitted[(arm, seed)] = _record_fit(
                        result, emit, arm, seed, params, sources, len(y_train), fit_seconds=elapsed,
                        encoding_seconds=model.encoding_seconds_, training_seconds=model.training_seconds_,
                        readout_seconds=model.readout_seconds_, purity=purity, displacement=displacement,
                        motion={'draws': sum(m.draws for m in arms), 'calls': sum(m.calls for m in arms),
                                'view_matching': matching[(arm, seed)]},
                        view_state_hashes=[net.state_hash() for _, net in model.views_],
                        readout_choices=base.readout_choices(model))
                    if arm == 'views7':
                        check = base.reproduction_check(predicted, sealed, seed) if held_out else {
                            'model_seed': seed, 'reproduced': None, 'n_test': len(test), 'n_differing': None,
                            'views7_prediction_hash': array_hash(np.asarray(predicted)),
                            'reference_prediction_hash': sealed['reference_prediction_hashes'][str(seed)]}
                        result['reproduction'].append(check)
                        emit('reproduction', check)
                        if held_out and not check['reproduced']:
                            checks['reference_predictions'] = {'performed': True, 'passed': False, 'model_seed': seed,
                                                               'n_differing': check['n_differing'], 'n_test': check['n_test']}
                            raise CheckFailed(
                                f'reference_predictions: views7 does not reproduce the reference outer predictions for '
                                f'seed {seed}: {check["n_differing"]} of {check["n_test"]} test rows differ')
                        if job['check_uninstrumented'] and seed == seeds[0]:
                            start = time.perf_counter()
                            plain_hashes, plain_predictions = uninstrumented_fit(params, seed, X_train, y_train, X_test)
                            result['timing']['uninstrumented_fit_seconds'] = time.perf_counter() - start
                            passed = (plain_hashes == fitted[(arm, seed)]['view_state_hashes']
                                      and np.array_equal(plain_predictions, predicted))
                            checks['uninstrumented_fit'] = {
                                'performed': True, 'passed': bool(passed), 'model_seed': seed,
                                'view_state_hashes': fitted[(arm, seed)]['view_state_hashes'],
                                'uninstrumented_view_state_hashes': plain_hashes,
                                'predictions_identical': bool(np.array_equal(plain_predictions, predicted))}
                            if not passed:
                                raise CheckFailed('uninstrumented_fit: a view state hash or the kNN predictions differ '
                                                  'from the uninstrumented fit')
                    del model, arms
        checks['reference_predictions'] = {'performed': held_out, 'passed': True if held_out else None,
                                           'fitting_seeds': seeds}
        if not job['check_uninstrumented']:
            result['timing']['uninstrumented_fit_seconds'] = 0.
        checks['rng_untouched'] = {'performed': True, 'passed': bool(rng_ok)}
        if not rng_ok:
            raise CheckFailed('rng_untouched: a wrapper call changed the global numpy or Python RNG state')
        checks['wrapper_removed'] = {'performed': True, 'passed': bool(wrappers_removed)}
        if not wrappers_removed:
            raise CheckFailed('wrapper_removed: a view network kept an instance-level backward_propagate')
        _finish_checks(checks, job, seeds, sources, predictions, fitted, initial_hashes, first_batch, matching)
        for (arm, seed), predicted in predictions.items():
            predicted = np.asarray(predicted)
            fit = fitted[(arm, seed)]
            row = dict(common, arm_id=arm, model_id=arm, variant_id=arm, model_seed=seed, params=params,
                       fit_source=sources[arm], stage='outer', status='ok', training_sample_count=len(y_train),
                       selected_widths=list(job['selected_widths']), hidden_layers=job['hidden_layers'],
                       prediction_hash=array_hash(predicted), per_class_recall=per_class_recall(y_test, predicted, classes),
                       neighborhood_purity_test=fit['purity']['test'], neighborhood_purity_train=fit['purity']['train'],
                       displacement=fit['displacement'], **metric_values(y_test, predicted))
            result['models'].append(row)
            emit('model', row)
            result['predictions'].extend(
                {'dataset_id': name, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
                 'arm_id': arm, 'model_seed': seed, 'sample_id': int(sample), 'y_true': y[sample].item(),
                 'y_pred': np.asarray(label).item(), 'config_id': job['config_id'], 'code_revision': code_revision}
                for sample, label in zip(test, predicted))
        result['status'] = 'ok'
    except Exception as exc:
        result['status'] = 'failed'
        result['exception'] = f'{type(exc).__name__}: {exc}'
        result['check_failed'] = isinstance(exc, CheckFailed)
        result['reproduction_failed'] = isinstance(exc, CheckFailed) and 'reference_predictions' in str(exc)
        emit('terminal_failure', {'exception': result['exception'], 'check_failed': result['check_failed'],
                                  'reproduction_failed': result['reproduction_failed']})
        present = {(r['arm_id'], r['model_seed']) for r in result['models']}
        for arm in ARMS:
            for seed in seeds:
                if (arm, seed) not in present:
                    row = dict(common, arm_id=arm, model_id=arm, variant_id=arm, model_seed=seed, stage='outer',
                               status='failed', exception=result['exception'])
                    result['models'].append(row)
                    emit('model', row)
    result['timing']['job_seconds'] = time.perf_counter() - started
    result['timing']['seconds_by_arm'] = {arm: sum(f['fit_seconds'] for (a, _), f in fitted.items() if a == arm)
                                          for arm in ARMS}
    return result


def _record_fit(result, emit, arm, seed, params, sources, n_train, **extra):
    fit = dict(fit_id=f'{arm}__s{seed}', arm_id=arm, model_seed=seed, fit_source=sources[arm], params=params,
               training_sample_count=n_train, status='ok', **extra)
    result['fits'].append(fit)
    emit('fit', fit)
    return fit


def _mean_displacement(by_view):
    """[[(footrule, unchanged share) per layer] per view] -> per layer, the mean over views."""
    layers = len(by_view[0])
    return [{'layer': index, 'kind': 'hidden' if index < layers - 1 else 'output',
             'mean_normalized_footrule': float(np.mean([view[index][0] for view in by_view])),
             'mean_changed_share': float(np.mean([view[index][1] for view in by_view])),
             'unchanged_share': float(np.mean([1. - view[index][1] for view in by_view]))}
            for index in range(layers)]


def _finish_checks(checks, job, seeds, sources, predictions, fitted, initial_hashes, first_batch, matching):
    """The cross-arm checks, once every arm of every fitting seed is fitted."""
    fitted_arms = [arm for arm in ARMS if sources[arm] == 'fitted']
    same_initial = all(initial_hashes[(arm, seed)] == initial_hashes[('views7', seed)]
                       for arm in fitted_arms for seed in seeds)
    checks['initial_state'] = {'performed': True, 'passed': bool(same_initial),
                               'initial_filter_hashes': {f'{arm}__s{seed}': initial_hashes[(arm, seed)]
                                                         for arm in fitted_arms for seed in seeds}}
    if not same_initial:
        raise CheckFailed('initial_state: an arm did not start from views7\'s seeded initial filters')

    # At the first batch no filter has moved yet, so every arm sees views7's signal at the last hidden layer, the one the
    # forward pass feeds directly. A deeper layer reads the mean, over the accepted filters of the layer above, of each
    # accepted filter's own motion against the batch input. That mean depends on the accepted signs only through how many
    # are attractions and how many repulsions, so permuting the signs among the accepted slots leaves it unchanged:
    # random_direction's deeper first-batch signal must equal views7's too, and it reaches the filters only through the
    # accumulators. permuted_alignment changes which filters are accepted, so its deeper signal is expected to differ and
    # is the one exception.
    top = str(job['hidden_layers'] - 1)
    layers_ok, layer_detail = True, {}
    for seed in seeds:
        for arm in fitted_arms:
            if arm == 'views7':
                continue
            compared = [top] if arm == 'permuted_alignment' else None
            for view, (theirs, ours) in enumerate(zip(first_batch[('views7', seed)], first_batch[(arm, seed)])):
                for index, records in ours.items():
                    if compared is not None and index not in compared:
                        continue
                    same = index in theirs and theirs[index] == records
                    layers_ok = layers_ok and same
                    if not same:
                        layer_detail[f'{arm}__s{seed}__v{view}__ly{index}'] = 'differs from views7'
    checks['first_batch_signal'] = {'performed': True, 'passed': bool(layers_ok), 'differing': layer_detail,
                                    'compared': 'every hidden layer each arm processed, except permuted_alignment, where '
                                                f'only the last hidden layer (ly{top}) is compared because re-keying '
                                                'changes which filters are accepted and so the signal the deeper layer '
                                                'reads'}
    if not layers_ok:
        raise CheckFailed('first_batch_signal: an arm\'s pre-transform signal at the first batch differs from views7\'s')

    matched = all(view['passed'] for key, records in matching.items() for view in records)
    checks['mass_matched'] = {'performed': True, 'passed': bool(matched),
                              'by_arm': {arm: _matching_summary([matching[(arm, seed)] for seed in seeds])
                                         for arm in fitted_arms}}
    if not matched:
        raise CheckFailed('mass_matched: a transform did not preserve the accepted votes, mass, magnitudes or sign counts')

    unmoved = all(entry['mean_normalized_footrule'] == 0. and entry['unchanged_share'] == 1.
                  for seed in seeds for entry in fitted[('frozen', seed)]['displacement'] if entry['kind'] == 'hidden')
    checks['frozen_unmoved'] = {'performed': True, 'passed': bool(unmoved),
                                'hidden_displacement': {str(seed): [e for e in fitted[('frozen', seed)]['displacement']
                                                                    if e['kind'] == 'hidden'] for seed in seeds}}
    if not unmoved:
        raise CheckFailed('frozen_unmoved: a frozen hidden filter moved')

    identical = job['hidden_layers'] < 2
    if identical:
        agree = all(sources[arm] == 'identical_to_views7'
                    and np.array_equal(predictions[(arm, seed)], predictions[('views7', seed)])
                    for arm in DEPTH_ARMS for seed in seeds)
    else:
        agree = all(sources[arm] == 'fitted' for arm in DEPTH_ARMS)
    checks['single_layer_identity'] = {'performed': True, 'passed': bool(agree),
                                       'hidden_layers': job['hidden_layers'], 'identical_to_views7': bool(identical)}
    if not agree:
        raise CheckFailed('single_layer_identity: the single_layer arms disagree with the depth of the selection')


def _matching_summary(by_seed):
    """Total accepted slots, changed slots and mass before/after, per hidden layer, over seeds and views."""
    out = {}
    for records in by_seed:
        for view in records:
            for index, entry in view['layers'].items():
                total = out.setdefault(index, {'batches': 0, 'accepted_slots': 0, 'changed_slots': 0,
                                               'mass_before': 0., 'mass_after': 0., 'effective_before': 0,
                                               'effective_after': 0, 'positive_before': 0, 'positive_after': 0})
                total['batches'] += entry['batches']
                total['accepted_slots'] += entry['accepted_slots']
                total['changed_slots'] += entry['changed_slots']
                total['mass_before'] += entry['before']['mass']
                total['mass_after'] += entry['after']['mass']
                total['effective_before'] += entry['before']['effective']
                total['effective_after'] += entry['after']['effective']
                total['positive_before'] += entry['before']['positive']
                total['positive_after'] += entry['after']['positive']
    for entry in out.values():
        slots = entry['accepted_slots']
        entry['changed_share'] = (entry['changed_slots'] / slots) if slots else 0.
        entry['positive_share_before'] = (entry['positive_before'] / entry['effective_before']) if entry['effective_before'] else None
        entry['positive_share_after'] = (entry['positive_after'] / entry['effective_after']) if entry['effective_after'] else None
    return out


def worker(arguments):
    output, job = arguments
    output, stem = Path(output), job['stem']
    log, result_path = output/'logs'/f'{stem}.jsonl', output/'results'/f'{stem}.json'
    prediction_path = output/'predictions'/f'{stem}.jsonl'
    if any(path.exists() for path in (log, result_path, prediction_path)):
        raise FileExistsError(f'Existing motion-controls job {stem}')
    for path in (log, result_path, prediction_path):
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
        result = evaluate_job(X, y, split, p, job, sealed, dataset_hash=data['dataset_hash'], code_revision=revision,
                              sink=sink)
    records = result.pop('predictions')
    with prediction_path.open('x') as stream:
        for record in records:
            stream.write(canonical_json(record) + '\n')
    result['prediction_file'] = {'path': f'predictions/{stem}.jsonl', 'records': len(records),
                                 'sha256': sha256_file(prediction_path)}
    write_json(result_path, _plain(result))
    return str(result_path), result['status'], bool(result.get('check_failed'))


def run(output, workers=1, *, allow_smoke=False):
    """Every planned job on spawned single-thread workers; a failed check cancels the pending jobs."""
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

def validate_job(result, events, job, p, X, y, split, data, revision, prediction_path, sealed):
    """Bind every record to the sealed plan, the per-example predictions and the reference predictions; returns the number
    of fitting seeds whose views7 predictions equal the reference exactly (all, or it raises)."""
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
                'split_hash': config_id(split), 'train_ids': [int(i) for i in train], 'query_ids': test,
                'query': 'outer_test_rows', 'raw_train_hash': array_hash(X[train]), 'raw_query_hash': array_hash(X[test]),
                'training_labels_hash': array_hash(y[train]), 'config': job['config'], 'selected': job['selected'],
                'params': job['params'], 'hidden_layers': job['hidden_layers'], 'arms': list(ARMS),
                'fit_sources': job['fit_sources'], 'model_seeds': seeds, 'arm_seeds': job['arm_seeds'],
                'check_uninstrumented': job['check_uninstrumented'],
                'reference_prediction_hashes': job['reference_prediction_hashes'], 'protocol_hash': config_id(p)}
    require(result['identity'] == _plain(identity), 'job identity disagrees with the sealed plan')
    require(job['config_id'] == sealed['config_id'] and job['config'] == sealed['config'],
            'planned configuration disagrees with the sealed reference selection')
    require(logged('identity') == [result['identity']] and logged('fit') == result['fits']
            and logged('model') == result['models'] and logged('reproduction') == result['reproduction'],
            'records differ from log')
    require({e['stage'] for e in events} <= {'identity', 'reproduction', 'fit', 'model'}, 'unexpected event stage')
    require(sorted(result['checks']) == sorted(JOB_CHECKS), 'check schedule')
    for name in JOB_CHECKS:
        check = result['checks'][name]
        performed = job['check_uninstrumented'] if name == 'uninstrumented_fit' else True
        require(check['performed'] is performed and check['passed'] is (True if performed else None), f'check {name}')
    require([r['model_seed'] for r in result['reproduction']] == seeds
            and all(r == _plain(base.reproduction_check(sealed['reference_predictions'][str(r['model_seed'])], sealed,
                                                        r['model_seed'])) for r in result['reproduction']),
            'views7 reproduction records')
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
    fits = {(f['arm_id'], f['model_seed']): f for f in result['fits']}
    classes = np.unique(y).tolist()
    for row in result['models']:
        key = (row['arm_id'], row['model_seed'])
        fit, pred = fits[key], predictions[key]
        require(row['status'] == 'ok' and fit['status'] == 'ok' and row['model_id'] == row['arm_id'] == row['variant_id']
                and row['code_revision'] == revision and row['dataset_hash'] == data['dataset_hash']
                and row['config_id'] == job['config_id']
                and row['training_sample_count'] == len(train) == fit['training_sample_count'], 'model row identity')
        require(row['fit_source'] == fit['fit_source'] == job['fit_sources'][row['arm_id']],
                'fit source disagrees with plan')
        require(row['params'] == fit['params'] == _plain(job['params']), 'arm parameters disagree with plan')
        require(row['hidden_layers'] == job['hidden_layers'] and row['selected_widths'] == list(job['selected_widths']),
                'model row depth disagrees with plan')
        require(row['prediction_hash'] == array_hash(pred), 'prediction hash')
        if fit['fit_source'] == 'identical_to_views7':
            require(job['hidden_layers'] < 2 and row['arm_id'] in DEPTH_ARMS, 'identical marker at a two-layer selection')
            require(fit['fit_seconds'] == fit['encoding_seconds'] == fit['training_seconds'] == fit['readout_seconds'] == 0
                    and fit.get('reused_from') == f'views7__s{row["model_seed"]}',
                    'reused fit must carry zero timing and name its source fit')
            require(np.array_equal(pred, predictions[('views7', row['model_seed'])]),
                    f'{row["arm_id"]} marked identical to views7 but differs')
        require(row['per_class_recall'] == _plain(per_class_recall(y[test], pred, classes)), 'per-class recall')
        metrics = metric_values(y[test], pred)
        require(all(np.isclose(row[k], v, rtol=0, atol=1e-12) for k, v in metrics.items()),
                'metric/prediction disagreement')
    for seed in seeds:
        require(base.reproduction_check(predictions[('views7', seed)], sealed, seed)['reproduced'],
                f'views7 does not reproduce the reference {REFERENCE_MODEL} predictions (seed {seed})')
    return len(seeds)


def collect_results(output, *, allow_smoke=False, rederive=True):
    """Every planned job, log and prediction file reconciled, and (rederive) every sealed selection re-derived from the
    reference run holding its dataset. Failures never become missing evidence."""
    output = Path(output)
    p, manifest, jobs, selections = verify(output, allow_smoke=allow_smoke, environment_check='sources')
    references = None
    if rederive:
        sources = {name: (entry['run_directory'], entry['ablation_directory'])
                   for name, entry in manifest['references'].items()}
        references = td.load_references(p, sources, allow_smoke=allow_smoke)
    revision = json.loads((output/'environment.json').read_text())['code_revision']
    prepared = {name: load_prepared(output, name) for name in manifest['datasets']}
    rows, reproduced, issues, records = defaultdict(list), defaultdict(int), [], {}
    for job in jobs:
        stem, key = job['stem'], (job['dataset_id'], job['outer_repeat'], job['outer_fold'])
        result_path, log = output/'results'/f'{stem}.json', output/'logs'/f'{stem}.jsonl'
        prediction_path = output/'predictions'/f'{stem}.jsonl'
        missing = [str(path.relative_to(output)) for path in (result_path, log, prediction_path) if not path.exists()]
        if missing:
            issues.append(f'missing {stem}: {", ".join(missing)}')
            continue
        try:
            result = json.loads(result_path.read_text())
            events = [json.loads(line) for line in log.read_text().splitlines()]
            X, y, data, splits = prepared[job['dataset_id']]
            index = next(i for i, s in enumerate(splits) if (s['outer_repeat'], s['outer_fold']) == key[1:])
            split, sealed = splits[index], selections[key]
            if references is not None:
                reference = references[job['reference']]
                rederived = base.selection_record(reference['run'], job['dataset_id'], split, y, data)
                if rederived != sealed or rederived != reference['ablation']['selections'].get(key):
                    raise ValueError('the sealed selection differs from the one re-derived from its reference')
                if planned_job(p, reference, job['dataset_id'], index, split, rederived, X.shape[1]) != job:
                    raise ValueError('the planned job differs from the re-derived plan')
            reproduced[job['dataset_id']] += validate_job(result, events, job, p, X, y, split, data, revision,
                                                          prediction_path, sealed)
            rows[job['dataset_id']].extend(result['models'])
            records[stem] = result
        except (KeyError, ValueError, TypeError, IndexError, OSError, EOFError, StopIteration, zipfile.BadZipFile) as exc:
            issues.append(f'{stem}: {type(exc).__name__}: {exc}')
    for name in manifest['datasets']:
        try:
            validate_outer_schedule(rows[name], expected_folds=fold_schedule(p),
                                    expected_seeds={arm: list(p['fit_seeds']) for arm in ARMS})
        except ValueError as exc:
            issues.append(f'{name}: {exc}')
    if issues:
        raise ValueError('Incomplete motion-controls evidence: ' + '; '.join(issues))
    return {'rows': dict(rows), 'reproduced': dict(reproduced), 'jobs': jobs, 'protocol': p, 'manifest': manifest,
            'records': records, 'code_revision': revision}


def _fold_values(rows, arm, field, *, seeds):
    """{(repeat, fold): mean over fitting seeds of a numeric per-row field} for one arm."""
    groups = defaultdict(list)
    for row in rows:
        if row['model_id'] == arm and row.get(field) is not None:
            groups[(row['outer_repeat'], row['outer_fold'])].append(float(row[field]))
    return {key: float(np.mean(values)) for key, values in groups.items() if len(values) == len(seeds)}


def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'mean': None, 'sd': None, 'n': 0}
    return {'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'n': len(values)}


def summary(output, *, allow_smoke=False):
    """Seed-within-fold means, outer-fold mean and SD, descriptive intervals against views7 and the descriptive measures."""
    collected = collect_results(output, allow_smoke=allow_smoke)
    p = collected['protocol']
    folds, seeds = fold_schedule(p), list(p['fit_seeds'])
    q, confidence = p['test_train_ratio'], p['confidence']
    summaries, flat = {}, []
    for name in collected['manifest']['datasets']:
        rows, table = collected['rows'][name], {}
        classes = sorted({key for row in rows for key in (row.get('per_class_recall') or {})})
        for arm in ARMS:
            entry = {'metrics': {m: summarize_outer(rows, arm, m, expected_folds=folds, expected_seeds=seeds)
                                 for m in METRICS}}
            if arm != 'views7':
                entry['change_from_views7'] = {}
                for m in ('accuracy', 'error'):
                    interval = paired_corrected_interval(rows, arm, 'views7', metric=m, q=q, confidence=confidence,
                                                         expected_folds=folds,
                                                         expected_seeds={arm: seeds, 'views7': seeds})
                    interval.pop('p_approximate')      # the tested family lives in compare_motion.analyse
                    entry['change_from_views7'][m] = interval
            entry['neighborhood_purity'] = {
                where: _stats(list(_fold_values(rows, arm, f'neighborhood_purity_{where}', seeds=seeds).values()))
                for where in ('test', 'train')}
            entry['per_class_recall'] = {
                label: _stats([float(np.mean([row['per_class_recall'][label] for row in rows
                                              if row['model_id'] == arm and (row['outer_repeat'], row['outer_fold']) == fold]))
                               for fold in folds]) for label in classes}
            entry['displacement'] = _displacement_summary(rows, arm, folds, seeds)
            table[arm] = entry
            flat.extend({'dataset_id': name, 'arm_id': arm, 'metric': m,
                         **{k: entry['metrics'][m][k] for k in SUMMARY_COLUMNS[3:]}} for m in METRICS)
        jobs = [j for j in collected['jobs'] if j['dataset_id'] == name]
        summaries[name] = {
            'arms': table,
            'resolved_configurations': [{'outer_repeat': j['outer_repeat'], 'outer_fold': j['outer_fold'],
                                         'config_id': j['config_id'], 'hidden_layers': j['hidden_layers'],
                                         **{k: j['selected'][k] for k in ('widths', 'learning_rate', 'embed_dim', 'degree',
                                                                          'augment')},
                                         'fit_sources': j['fit_sources']} for j in jobs],
            'two_hidden_layer_folds': sum(1 for j in jobs if j['hidden_layers'] >= 2), 'outer_folds': len(jobs),
            'views7_reproduces_reference': {'matching_fold_seeds': collected['reproduced'][name],
                                            'total_fold_seeds': len(jobs) * len(seeds)}}
    report = {'purpose': 'matched_motion_signal_controls_of_arrowflow_knn', 'code_revision': collected['code_revision'],
              'protocol_id': p.get('protocol_id'), 'references': collected['manifest']['references'],
              'arms': list(ARMS), 'primary_arms': list(PRIMARY_ARMS), 'depth_arms': list(DEPTH_ARMS),
              'supervision_disclosure': p['supervision_disclosure'], 'differs_from_untrained': p['differs_from_untrained'],
              'aggregation': 'fitting seeds averaged within outer fold, then outer-fold mean and SD; within-fold seed SD '
                             'reported separately',
              'change_from_views7': 'seed-averaged corrected resampled t interval of each arm minus views7; descriptive '
                                    'here, without p values or multiplicity adjustment; the tested family is '
                                    'compare_motion.analyse',
              'views7_reproduces_reference': 'hard check: every fold and seed of views7 reproduced the reference '
                                             f'{REFERENCE_MODEL} outer predictions exactly (the summary refuses otherwise)',
              'checks': _check_totals(collected['records']),
              'motion_statistics': _motion_totals(collected['records']),
              'inferential_significance_claims': False, 'summaries': summaries, 'model_rows': collected['rows']}
    return _plain(report), flat


def _displacement_summary(rows, arm, folds, seeds):
    by_layer = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['model_id'] != arm:
            continue
        for entry in row.get('displacement') or []:
            key = (entry['layer'], entry['kind'])
            by_layer[key]['footrule'].append(entry['mean_normalized_footrule'])
            by_layer[key]['unchanged'].append(entry['unchanged_share'])
    return [{'layer': layer, 'kind': kind,
             'mean_normalized_footrule': _stats(values['footrule']), 'unchanged_share': _stats(values['unchanged'])}
            for (layer, kind), values in sorted(by_layer.items())]


def _check_totals(records):
    totals = {name: Counter() for name in JOB_CHECKS}
    for record in records.values():
        for name in JOB_CHECKS:
            check = record['checks'][name]
            totals[name][str(check['passed'])] += 1
    return {name: dict(counter) for name, counter in totals.items()}


def _motion_totals(records):
    """Accepted slots, changed slots and accepted mass before and after the transform, per arm and hidden layer."""
    out = defaultdict(lambda: defaultdict(lambda: {'accepted_slots': 0, 'changed_slots': 0, 'mass_before': 0.,
                                                   'mass_after': 0., 'positive_before': 0, 'positive_after': 0}))
    for record in records.values():
        for fit in record['fits']:
            for view in ((fit.get('motion') or {}).get('view_matching') or []):
                for index, entry in view['layers'].items():
                    total = out[fit['arm_id']][index]
                    total['accepted_slots'] += entry['accepted_slots']
                    total['changed_slots'] += entry['changed_slots']
                    total['mass_before'] += entry['before']['mass']
                    total['mass_after'] += entry['after']['mass']
                    total['positive_before'] += entry['before']['positive']
                    total['positive_after'] += entry['after']['positive']
    for arm in out:
        for entry in out[arm].values():
            slots = entry['accepted_slots']
            entry['changed_share'] = (entry['changed_slots'] / slots) if slots else 0.
            entry['positive_share'] = (entry['positive_before'] / slots) if slots else None
            entry['mass_deviation'] = abs(entry['mass_before'] - entry['mass_after'])
    return {arm: dict(layers) for arm, layers in out.items()}


def write_summary(output, *, allow_smoke=False):
    report, flat = summary(output, allow_smoke=allow_smoke)
    write_json(Path(output)/SUMMARY_JSON, report)
    write_csv(Path(output)/SUMMARY_CSV, SUMMARY_COLUMNS, [[r[c] for c in SUMMARY_COLUMNS] for r in flat])
    return report


# ----------------------------------------------------------------------------- pilot, projection and freeze

def calibrated_projection(p, jobs, records, workers=WORKERS):
    """Per planned job: the reference run's realized outer fit and predict seconds of its fold and its three fitting seeds
    times the piloted all-arms/views7 ratio of a pilot dataset with the same hidden depth (the largest piloted ratio
    otherwise), plus the views7 seconds once more for a first fold's uninstrumented fit."""
    by_depth = defaultdict(list)
    own = {}
    for record in records:
        by_depth[int(record['hidden_layers'])].append(record['all_arms_to_views7_ratio'])
        own[(record['dataset_id'], int(record['hidden_layers']))] = record['all_arms_to_views7_ratio']
    largest = max(r['all_arms_to_views7_ratio'] for r in records)
    seconds, datasets = [], {}
    for job in jobs:
        depth = int(job['hidden_layers'])
        alike = by_depth.get(depth)
        if (job['dataset_id'], depth) in own:
            ratio, basis = own[(job['dataset_id'], depth)], 'piloted ratio at the same hidden depth'
        elif alike:
            ratio, basis = max(alike), 'largest piloted ratio at the same hidden depth'
        else:
            ratio, basis = largest, 'largest piloted ratio (no pilot fold shares the hidden depth)'
        realized = float(job['reference_outer_seconds'])
        cost = realized * ratio + (realized / max(1, len(job['model_seeds'])) if job['check_uninstrumented'] else 0.)
        seconds.append(cost)
        entry = datasets.setdefault(job['dataset_id'], {'jobs': 0, 'reference_views7_seconds': 0., 'seconds': 0.,
                                                        'ratios': set(), 'basis': basis})
        entry['jobs'] += 1
        entry['reference_views7_seconds'] += realized
        entry['seconds'] += cost
        entry['ratios'].add(ratio)
    for entry in datasets.values():
        entry['all_arms_to_views7_ratio'] = sorted(entry.pop('ratios'))
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
    single = dict(p, fit_seeds=[p['fit_seeds'][0]])
    records = []
    for name in p['pilot_datasets']:
        job = next(j for j in jobs if j['dataset_id'] == name and (j['outer_repeat'], j['outer_fold']) == (0, 0))
        X, y, data, splits = load_prepared(output, name)
        split = splits[0]
        query = [int(i) for i in split['train'][::4]]
        record = evaluate_job(X, y, split, single, dict(job, model_seeds=single['fit_seeds'], check_uninstrumented=True),
                              sealed[(name, split['outer_repeat'], split['outer_fold'])], dataset_hash=data['dataset_hash'],
                              code_revision=revision, query=query)
        if record['status'] != 'ok':
            raise CheckFailed(f'pilot job {name} failed: {record.get("exception")}')
        timing = record['timing']
        by_arm = timing['seconds_by_arm']
        records.append({'dataset_id': name, 'hidden_layers': job['hidden_layers'], 'dataset_hash': data['dataset_hash'],
                        'train_ids': [int(i) for i in split['train']], 'query_ids': query, 'config_id': job['config_id'],
                        'selected': job['selected'], 'model_seed': single['fit_seeds'][0], 'seconds_by_arm': by_arm,
                        'seconds_per_seed': float(sum(by_arm.values())),
                        'uninstrumented_fit_seconds': timing['uninstrumented_fit_seconds'],
                        'job_seconds': timing['job_seconds'],
                        'all_arms_to_views7_ratio': float(sum(by_arm.values()) / by_arm['views7']),
                        'checks': {key: {'performed': c['performed'], 'passed': c['passed']}
                                   for key, c in record['checks'].items()},
                        'status': 'ok'})
    probes = {name: base.reproduction_probe(reference['run'], td.smallest_dataset(reference))
              for name, reference in references.items()}
    calibrated = calibrated_projection(p, jobs, records, p['workers'])
    decision = calibrated['simulated_makespan_hours']
    checks_passed = all(c['passed'] is True for r in records for key, c in r['checks'].items()
                        if key != 'reference_predictions')
    report = _plain({
        'purpose': 'training_only_runtime_no_heldout_scores', 'protocol_id': p['protocol_id'],
        'protocol_hash': config_id(p), 'code_revision': revision, 'started_utc': started, 'ended_utc': utc_now(),
        'records': records, 'calibrated_projection': calibrated, 'reproduction_probes': probes,
        'decision': {'rule': p['decision_rule'], 'hours': decision, 'cap_hours': p['wallclock_cap_hours'],
                     'workers': p['workers'], 'within_cap': decision <= p['wallclock_cap_hours'],
                     'checks_passed': checks_passed,
                     'probes_reproduced': all(probe['reproduced'] for probe in probes.values())},
        'estimate_limitations': 'one fitting seed on one training partition per pilot dataset, measured on the machine as it '
                                'was (the pilot shares the machine with other production runs, so the absolute pilot seconds '
                                'are inflated by contention; the all-arms/views7 ratio is a ratio of times measured under the '
                                'same contention and is far less affected); the calibrated projection assumes the reference '
                                'runs\' realized per-fold seconds (themselves measured under 16 workers) carry over, prices '
                                'unpiloted depths at the largest piloted ratio and assumes a job takes its three fitting '
                                'seeds in sequence; the query rows are training rows, so the prediction, purity and readout '
                                'costs follow the outer test size only approximately'})
    write_json(output/'pilot.json', report)
    return report


def freeze(draft_path, pilot_path, stages_path, output_path, *, frozen_at_utc=None):
    """The frozen protocol from the committed draft, only if the training-only pilot of that draft projects within the cap at
    the protocol workers, every pilot check and reproduction probe held, and the stage record carries a passing smoke; an
    existing output must be the draft itself, which the frozen protocol then replaces."""
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
    if stages.get('smoke', {}).get('status') != 'ok':
        raise ValueError('Not frozen: the stage record does not carry a passing synthetic smoke')
    calibrated = pilot['calibrated_projection']
    record = {'cap_hours': CAP_HOURS, 'workers': WORKERS, 'decision_hours': decision['hours'],
              'decision_rule': decision['rule'],
              'calibrated': {key: calibrated[key] for key in ('serial_hours', 'serial_hours_over_workers',
                                                              'simulated_makespan_hours', 'longest_job_hours')},
              'calibrated_per_dataset_hours': {name: entry['serial_hours'] for name, entry in calibrated['datasets'].items()},
              'pilot_ratios': {r['dataset_id']: r['all_arms_to_views7_ratio'] for r in pilot['records']},
              'pilot_seconds_by_arm': {r['dataset_id']: r['seconds_by_arm'] for r in pilot['records']},
              'reproduction_probes': {name: {key: probe[key] for key in ('dataset_id', 'result_file', 'config_id',
                                                                         'model_seed', 'inner_fold', 'reference_score',
                                                                         'refit_score', 'readout_selections_identical',
                                                                         'reproduced')}
                                      for name, probe in pilot['reproduction_probes'].items()},
              'pilot_sha256': sha256_file(pilot_path), 'stages': stages}
    frozen_at = frozen_at_utc or datetime.now(timezone.utc).isoformat()
    text = (f"E1 matched motion-signal controls (author decision 2026-09-14 and the binding E1 design ruling): drafted from "
            f"training_diagnostics' references over all seventeen datasets; {stages['summary']}; projected at {WORKERS} "
            f"single-thread workers: calibrated simulated makespan {decision['hours']:.2f} h (serial "
            f"{calibrated['serial_hours']:.2f} h, serial over workers {calibrated['serial_hours_over_workers']:.2f} h); cap "
            f"{CAP_HOURS} h; the pilot ran while two production runs held the machine, so its absolute seconds are inflated "
            f"by contention and only the all-arms/views7 ratio is carried forward; reproduction probes held on "
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

def smoke_protocol(p, sources):
    """The synthetic smoke form of a protocol whose references are the given synthetic runs {name: (run_dir, ablation_dir)}."""
    return td.smoke_protocol(p, sources)


def smoke(output, p, workers=3):
    """Synthetic references built with the real harness (training_diagnostics.smoke's construction: a bridge_knn-style run
    with two hidden layers and its knn_ablation, and the two newdata-style batch runs with their newdata_ablation, so both
    depths reach every arm and every check), then a complete motion-controls run, its summary and its analysis. Never
    evidence."""
    from . import run_newdata_ablation as rn
    from .knn_controls import synthetic_reference_run
    output = Path(output)
    with execution_lock():
        bridge = synthetic_reference_run(output/'synthetic_bridge_knn', td.SMOKE_CANDIDATES, workers=workers, samples=240)
    template, reference_protocol = json.loads(base.PROTOCOL.read_text()), json.loads((bridge/'protocol.json').read_text())
    from .knn_controls import reference_pins
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
    sources = {'smoke_bridge_knn': (bridge, knn_ablation),
               'smoke_newdata_batch1': (newdata/'synthetic_reference_batch1', newdata),
               'smoke_newdata_batch2': (newdata/'synthetic_reference_batch2', newdata)}
    tiny_motion = smoke_protocol(p, sources)
    run_directory = output/'run'
    prepare(run_directory, tiny_motion, sources, allow_smoke=True, purpose='synthetic_smoke_only')
    run(run_directory, workers, allow_smoke=True)
    report = write_summary(run_directory, allow_smoke=True)
    return report


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
        for name, entry in report['summaries'].items():
            counts = entry['views7_reproduces_reference']
            print(f"{name}: views7 reproduced {counts['matching_fold_seeds']}/{counts['total_fold_seeds']} fold-seeds; "
                  f"{entry['two_hidden_layer_folds']}/{entry['outer_folds']} two-hidden-layer folds")
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
        smoke(args.output, validate_protocol(p), args.workers)
        return
    validate_protocol(p)
    sources = parse_sources(args.reference)
    if args.command == 'prepare':
        prepare(args.output, p, sources, args.dataset)
    elif args.command == 'pilot':
        with execution_lock():
            report = runtime_pilot(args.output, p, sources)
        print(json.dumps({'decision': report['decision'],
                          'pilot_ratios': {r['dataset_id']: r['all_arms_to_views7_ratio'] for r in report['records']},
                          'calibrated_serial_hours': report['calibrated_projection']['serial_hours'],
                          'calibrated_makespan_hours': report['calibrated_projection']['simulated_makespan_hours']},
                         indent=2))
    else:
        if not p.get('frozen'):
            raise ValueError('The motion-controls run requires a frozen reviewed protocol')
        dirty = td.uncommitted_sources((args.protocol,))
        if dirty:
            raise ValueError(f'Commit the sealed sources and the protocol before the run: {", ".join(dirty)}')
        if p != json.loads((args.output/'protocol.json').read_text()):
            raise ValueError('Prepared and frozen protocols differ; prepare a new output directory')
        manifest = json.loads((args.output/'manifest.json').read_text())
        for name, entry in manifest['references'].items():
            if str(Path(sources[name][0]).resolve()) != entry['run_directory'] or \
                    str(Path(sources[name][1]).resolve()) != entry['ablation_directory']:
                raise ValueError('--reference differs from the prepared reference sources')
        run(args.output, args.workers)


if __name__ == '__main__':
    main()
