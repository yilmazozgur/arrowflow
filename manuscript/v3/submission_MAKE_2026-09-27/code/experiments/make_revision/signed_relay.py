"""The corrected relay of repulsion votes and a per-layer vote scale, as instance-level interventions on the ArrowFlow
hidden-layer update (review response to the simulated referee panel of 2026-09-23, verified defect V1-A). No sealed
module is edited: every intervention is installed on the live instance between ArrowFlowEstimator.initialize_orders and
train_initialized and removed after training, as update_rules installs its variants.

The defect (arrowflow.py:746-749, 873-884, 1814-1828; verification/v1/V1-report.md, section A). A hidden layer above the
first casts, for each accepted example and each selected filter r_j, the signed vote a_j = u_j / ((1 / eta) (N / 2)) of
eq. (5). Vertex.accumulate_motion accumulates the vote toward the layer's input pi when a_j > 0 and toward rev(pi) when
a_j < 0 (compute_distance reverses the comparison list and drops the sign), and it returns the motion toward that list,
m_{r_j -> tau~}. backward_propagate hands that returned motion, aligned by item ID and without a sign, to the layer below
(arrowflow.py:1828), averages it over the example's nonzero votes and rescales it by eq. (6). For a repulsion the relayed
motion is m_{r -> rev(pi)}[p] = (V - 1 - pos_pi(r[p])) - p, which is orthogonal to the push-away direction for every pair
of rankings. The decision layer only votes positively, so a network with one hidden layer never relays and is unaffected.

The correction (relay 'signed'). For every vote a_j the layer below receives, in place of the returned motion,

    d_j = sign(a_j) * m_{r_j -> pi},    m_{r -> pi}[p] = pos_pi(r[p]) - p    (the motion toward the UNREVERSED input)

    a_j > 0:  d_j = m_{r_j -> pi}                        the core's own return value, unchanged (the printed relay)
    a_j = 0:  d_j = 0                                    the core's own return value (zeros; nothing is accumulated)
    a_j < 0:  d_j = -m_{r_j -> pi} = m_{r_j -> rev(pi)} + 2p - (V - 1)     for a complete permutation pi

The filter's own vote is unchanged (a repulsion still accumulates toward rev(pi)), and so are the eligibility gate, the
unweighted mean over the example's nonzero votes, eq. (6), the output layer and everything else. It is the signed form of
the printed rule (Algorithm 1 line 23 becomes d_j <- sign(a_j) m_{r_j -> pi}); the |a|-weighted variant a_j m_{r_j -> pi}
of V1-report.md is not used, because it would also reweight the attraction relays and so change two things at once.

backward_propagate uses accumulate_motion's return value for nothing but the relay (the first hidden layer's is discarded
by its `continue`; the forward pass calls accumulate_motion on output filters only), so an instance-level
Vertex.accumulate_motion on the hidden filters that returns d_j is the whole correction: RelayedMotion.

The vote scale (ScaledVote). Every vote cast at a hidden layer below the top hidden layer ([64, 128]: the first hidden
layer) is multiplied by lower_vote_scale s before it is accumulated; the top hidden layer's votes are unchanged. The
relayed signal is rescaled by eq. (6) to at most c V (c = 0.125), so a first-layer vote is at most 2 eta c = eta / 4,
while the accumulator's prior (weight 1 after every batch reset) dictates the order unless a filter's batch vote mass
reaches 1 / (V - 1), V the layer's vocabulary: two items at positions p < p' swap only if sum_t |a_t| (q_t(p) - q_t(p'))
>= p' - p >= 1 with |q_t(p) - q_t(p')| <= V - 1. ThresholdLayerRecord counts, per hidden layer, the voted filter-batches
and those whose vote mass reaches that threshold.
"""
from contextlib import contextmanager
import numpy as np
from . import update_rules as ur

RELAYS = ('printed', 'signed')
VERIFY_FIRST = 8        # per layer and view fit: the first repulsions and attractions checked against the direct motion


class RelayError(ur.InterventionError):
    """The relay or scale intervention is inconsistent."""


# ----------------------------------------------------------------------------- the motions

def toward_input(filter_order, input_order):
    """m_{r -> pi}[p] = pos_pi(r[p]) - p, computed directly from the two orders (both complete permutations of one set)."""
    places = {item: index for index, item in enumerate(input_order)}
    if len(places) != len(filter_order) or set(places) != set(filter_order):
        raise RelayError('The input is not a complete permutation of the filter items')
    return np.asarray([places[item] - p for p, item in enumerate(filter_order)])


def signed_relay_motion(returned, magnitude, length):
    """The corrected relay of one vote from the core's return value: unchanged for a_j >= 0 (or None); for a_j < 0 the
    core returned m_{r -> rev(pi)}, and d = m_{r -> rev(pi)} + 2p - (V - 1) = -m_{r -> pi}."""
    if magnitude is None or not magnitude < 0:
        return returned
    returned = np.asarray(returned)
    if returned.shape != (length,):
        raise RelayError('The returned motion does not cover the filter')
    return returned + (2 * np.arange(length) - (length - 1))


# ----------------------------------------------------------------------------- per-layer records

class RelayLayerRecord:
    """Per hidden layer and view fit: the votes seen by the relay wrapper, the returned motions it corrected, the direct
    checks it made and, where the scale is installed, the unscaled and scaled vote mass."""

    def __deepcopy__(self, memo):
        return self          # instrumentation, not network state: the core's checkpoint copies never train

    def __init__(self, layer, name, relays, scale):
        self.layer, self.name, self.relays, self.scale = layer, name, bool(relays), float(scale)
        self.calls = self.attractions = self.repulsions = self.zeros = self.corrected = 0
        self.verified_repulsions = self.verified_attractions = 0
        self.scaled_votes, self.unscaled_mass, self.scaled_mass = 0, 0., 0.

    def summary(self):
        return {'layer': self.layer, 'layer_name': self.name, 'relays_to_layer_below': self.relays,
                'vote_scale': self.scale, 'calls': self.calls, 'attractions': self.attractions,
                'repulsions': self.repulsions, 'zero_votes': self.zeros, 'returned_motions_corrected': self.corrected,
                'verified_repulsions': self.verified_repulsions, 'verified_attractions': self.verified_attractions,
                'scaled_votes': self.scaled_votes, 'unscaled_vote_mass': self.unscaled_mass,
                'scaled_vote_mass': self.scaled_mass}


class ThresholdLayerRecord(ur.LayerRecord):
    """update_rules.LayerRecord plus the prior's movement threshold: per filter-batch with at least one nonzero vote
    (voted), whether its vote mass M reaches 1 / (V - 1) (cleared: M (V - 1) >= 1), and M (V - 1) in threshold units."""

    def __init__(self, layer, name, filters, vocabulary):
        super().__init__(layer, name, filters, vocabulary)
        self.threshold = 1. / (vocabulary - 1) if vocabulary > 1 else None

    def _empty(self):
        return {**super()._empty(), 'voted': 0, 'cleared': 0, 'threshold_units': 0.}

    def add(self, *, moved, displacement, votes, mass, solved, unsettled=0):
        super().add(moved=moved, displacement=displacement, votes=votes, mass=mass, solved=solved, unsettled=unsettled)
        if votes and self.threshold is not None:
            units = float(mass) * (self.vocabulary - 1)
            self.pending['voted'] += 1
            self.pending['cleared'] += int(units >= 1.)
            self.pending['threshold_units'] += units

    def summary(self):
        out = super().summary()
        voted = sum(b['voted'] for b in self.batches)
        cleared = sum(b['cleared'] for b in self.batches)
        units = sum(b['threshold_units'] for b in self.batches)
        out.update({'movement_threshold': self.threshold, 'voted_updates': int(voted), 'cleared_updates': int(cleared),
                    'cleared_share_of_voted': float(cleared / voted) if voted else None,
                    'mean_threshold_units_voted': float(units / voted) if voted else None})
        return out


# ----------------------------------------------------------------------------- instance-level wrappers

def _call_inner(vertex, inner, adj_list_comp, magnitude):
    if inner is None:
        return type(vertex).accumulate_motion(vertex, adj_list_comp, magnitude)
    return inner(adj_list_comp, magnitude)


class ScaledVote:
    """Instance-level Vertex.accumulate_motion below the top hidden layer: the vote magnitude times `scale`, then the
    wrapped accumulate_motion (update_rules.VoteCounter, so the recorded mass is the accumulated mass)."""

    def __init__(self, vertex, inner, scale, record):
        if not np.isfinite(scale) or scale <= 0:
            raise RelayError('The vote scale must be positive and finite')
        self.vertex, self.inner, self.scale, self.record = vertex, inner, float(scale), record

    def __call__(self, adj_list_comp, magnitude=1):
        if magnitude is None or magnitude == 0:
            return _call_inner(self.vertex, self.inner, adj_list_comp, magnitude)
        scaled = magnitude * self.scale
        self.record.scaled_votes += 1
        self.record.unscaled_mass += abs(float(magnitude))
        self.record.scaled_mass += abs(float(scaled))
        return _call_inner(self.vertex, self.inner, adj_list_comp, scaled)


class RelayedMotion:
    """Instance-level Vertex.accumulate_motion on a hidden filter: the wrapped accumulate_motion runs unchanged (the vote
    and its accumulation are the core's), and in relay 'signed' a repulsion's return value is replaced by
    signed_relay_motion. The first VERIFY_FIRST repulsions and attractions of each layer are checked against the motions
    computed directly from the filter and the input (toward_input): an attraction returns m(r -> input), a repulsion
    returns m(r -> rev(input)) from the core, and in relay 'signed' the corrected value is -m(r -> input)."""

    def __init__(self, vertex, inner, relay, record):
        if relay not in RELAYS:
            raise RelayError(f'relay must be one of {", ".join(RELAYS)}')
        self.vertex, self.inner, self.relay, self.record = vertex, inner, relay, record

    def __call__(self, adj_list_comp, magnitude=1):
        vertex, record = self.vertex, self.record
        returned = _call_inner(vertex, self.inner, adj_list_comp, magnitude)   # accumulates; never reorders the filter
        record.calls += 1
        if magnitude is None or magnitude == 0:
            record.zeros += 1
            return returned
        if magnitude > 0:
            record.attractions += 1
            if record.verified_attractions < VERIFY_FIRST:
                if not np.array_equal(returned, toward_input(vertex.adjacency_list, adj_list_comp)):
                    raise RelayError(f'{vertex.id}: an attraction did not return the motion toward the input')
                record.verified_attractions += 1
            return returned
        record.repulsions += 1
        verify = record.verified_repulsions < VERIFY_FIRST
        if verify and not np.array_equal(returned, toward_input(vertex.adjacency_list, list(adj_list_comp)[::-1])):
            raise RelayError(f'{vertex.id}: a repulsion did not return the motion toward the reversed input')
        if self.relay == 'printed':
            record.verified_repulsions += int(verify)
            return returned
        if len(adj_list_comp) != vertex.len_list:
            raise RelayError(f'{vertex.id}: the corrected relay needs a complete permutation input')
        corrected = signed_relay_motion(returned, magnitude, vertex.len_list)
        if verify:
            if not np.array_equal(corrected, -toward_input(vertex.adjacency_list, adj_list_comp)):
                raise RelayError(f'{vertex.id}: the corrected relay differs from -m(r -> input)')
            record.verified_repulsions += 1
        record.corrected += 1
        return corrected


# ----------------------------------------------------------------------------- one view network's instrumentation

class RelayInstrumentation(ur.ArmInstrumentation):
    """update_rules.ArmInstrumentation (the unmodified Borda rule, no frozen layer) with the movement record extended by
    the prior's threshold, RelayedMotion on every hidden filter and, when lower_vote_scale != 1, ScaledVote on every
    filter of the hidden layers below the top hidden layer. Call chain of accumulate_motion on a scaled filter:
    RelayedMotion -> ScaledVote -> update_rules.VoteCounter -> Vertex.accumulate_motion."""

    def __init__(self, net, *, relay='printed', lower_vote_scale=1.):
        super().__init__(net, rule='borda', prior_rule='unit', prior_multiplier=1., frozen_hidden=())
        if relay not in RELAYS:
            raise RelayError(f'relay must be one of {", ".join(RELAYS)}')
        if not np.isfinite(lower_vote_scale) or lower_vote_scale <= 0:
            raise RelayError('The vote scale must be positive and finite')
        self.relay, self.lower_vote_scale = relay, float(lower_vote_scale)
        self.relay_records = []

    def layer_scale(self, index):
        return self.lower_vote_scale if index < self.hidden - 1 else 1.

    def _install(self):
        for index, layer in enumerate(self._layers()):
            vertices = list(layer.graph.vertex_list.values())
            record = ThresholdLayerRecord(index, f'hidden_{index}', len(vertices), len(vertices[0].adjacency_list))
            relay_record = RelayLayerRecord(index, f'hidden_{index}', relays=index > 0, scale=self.layer_scale(index))
            self.records.append(record)
            self.relay_records.append(relay_record)
            for vertex in vertices:
                for name in ur.ATTRIBUTES:
                    if name in vars(vertex):
                        raise RelayError(f'The vertex already carries an instance-level {name}')
                counter = ur.VoteCounter(vertex)
                outer = counter
                if relay_record.scale != 1.:
                    outer = ScaledVote(vertex, outer, relay_record.scale, relay_record)
                vertex.accumulate_motion = RelayedMotion(vertex, outer, self.relay, relay_record)
                vertex.apply_motion = ur.VertexUpdate(vertex, record, counter, frozen=False, order=None)
                self.installed_attributes += 2

    def intervention_consistent(self):
        """Every repulsion's return corrected in relay 'signed' and none in 'printed'; every nonzero vote of a scaled
        layer scaled, its accumulated mass s times the incoming mass and equal to the movement record's mass."""
        for record, relay in zip(self.records, self.relay_records):
            expected = relay.repulsions if self.relay == 'signed' else 0
            if relay.corrected != expected or relay.calls != relay.attractions + relay.repulsions + relay.zeros:
                return False
            if relay.scale != 1.:
                total = sum(batch['mass'] for batch in record.batches) + record.pending['mass']
                if (relay.scaled_votes != relay.attractions + relay.repulsions
                        or not np.isclose(relay.scaled_mass, relay.scale * relay.unscaled_mass, rtol=1e-9, atol=0)
                        or not np.isclose(relay.scaled_mass, total, rtol=1e-9, atol=1e-12)):
                    return False
            elif relay.scaled_votes:
                return False
        return True

    def summary(self):
        out = super().summary()
        out.update({'relay': self.relay, 'lower_vote_scale': self.lower_vote_scale,
                    'relay_layers': [record.summary() for record in self.relay_records],
                    'intervention_consistent': self.intervention_consistent()})
        return out


@contextmanager
def relay_rule(net, relay='printed', lower_vote_scale=1.):
    """One view network (an initialised, untrained ArrowFlowEstimator) trained under the relay rule inside the block."""
    instrumentation = RelayInstrumentation(net, relay=relay, lower_vote_scale=lower_vote_scale)
    with instrumentation.installed():
        yield instrumentation
