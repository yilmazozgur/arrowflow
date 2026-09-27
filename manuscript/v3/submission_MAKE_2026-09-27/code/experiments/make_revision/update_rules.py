"""Instance-level interventions on the ArrowFlow hidden-layer update: the footrule-median order, frozen layers and the
per-batch movement record. No sealed module is edited; every intervention is installed on the live instance between
ArrowFlowEstimator.initialize_orders and train_initialized (the wrapper pattern training_diagnostics uses on
update_network) and removed after training.

What one batch accumulates (verified in review-math-actions.md section 2(a) against the code): a vertex holds
A = permutation_matrix_accumulate, a V x V matrix whose row p is the item currently at position p. A starts as the
identity, so the prior carries weight exactly one at the item's own position, and every eligible vote adds |a_t| at the
position the vote wants that item to take. The core's order (Vertex.compute_adj_list_with_permutation) sorts the items
by the row mean sum_q q A[p,q] / sum_q A[p,q] and breaks ties by ascending numeric item ID, which is the minimiser over
the V! rankings of

    G(kappa) = Q(kappa, prior) + sum_t |a_t| Q(kappa, target_t),      Q = squared rank distance,

a weighted Borda count. The median arm replaces Q by the footrule (absolute rank) distance and minimises

    F(kappa) = w_prior F1(kappa, prior) + sum_t |a_t| F1(kappa, target_t),   F1 = sum_v |pos_kappa(v) - pos(v)|,

over the same votes with the same weights. F is a linear assignment problem with cost C = A D, D[q,p] = |q - p|
(Dwork, Kumar, Naor and Sivakumar's footrule aggregation), so the exact minimiser is
scipy.optimize.linear_sum_assignment(C); brute force over all V! rankings checks it on small vocabularies.

The prior weight is the fairness knob. Under Borda the prior weight 1 and a total vote mass of about 0.1 still move the
order, because the mean is continuous in the votes; under the median a prior heavier than the incoming vote mass freezes
the order completely. prior_weight() therefore implements two documented rules: 'unit' (weight 1, exactly what the code
accumulates) and 'one_ballot' (the prior enters as one more ballot carrying the mean weight of the incoming ballots, so
the incoming-to-prior mass ratio is n : 1 for n votes).
"""
from contextlib import contextmanager
from functools import lru_cache
from itertools import permutations
import numpy as np
from scipy.optimize import linear_sum_assignment
from arrowflow.ranking import score_order

RULES = ('borda', 'median')
PRIOR_RULES = ('unit', 'one_ballot')
ATTRIBUTES = ('accumulate_motion', 'compute_adj_list_with_permutation', 'apply_motion')
TIE_SWAP_LIMIT = 2                     # times the vocabulary size: the bound on single zero-cost tie swaps


class InterventionError(RuntimeError):
    """An installed intervention or its bookkeeping is inconsistent."""


# ----------------------------------------------------------------------------- orders from one accumulator

@lru_cache(maxsize=16)
def position_distances(size):
    """D[q, p] = |q - p| on 0..size-1; cached per size and never written to."""
    axis = np.arange(int(size))
    matrix = np.abs(axis[:, None] - axis[None, :]).astype(float)
    matrix.setflags(write=False)
    return matrix


def _accumulator(accumulated):
    A = np.asarray(accumulated, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] < 1 or not np.isfinite(A).all() or np.any(A < 0):
        raise InterventionError('Expected a square finite non-negative accumulator')
    if np.any(A.sum(axis=1) <= 0):
        raise InterventionError('Every accumulator row needs positive mass')
    return A


def borda_positions(accumulated, items=None):
    """The core's order as positions: new position of the item currently at each row (row mean, ties by numeric ID)."""
    A = _accumulator(accumulated)
    means = (A * np.arange(A.shape[1])).sum(axis=1) / A.sum(axis=1)
    order = score_order(means) if items is None else score_order(means, items)
    return np.argsort(np.asarray(order), kind='stable')


def footrule_cost(accumulated, validate=True):
    """C[p, k] = sum_q A[p, q] |q - k|: the footrule cost of moving the item at position p to position k. The hot path
    passes validate=False, having just checked the accumulator's row mass against the recorded votes."""
    A = _accumulator(accumulated) if validate else np.asarray(accumulated, dtype=float)
    return A @ position_distances(A.shape[1])


def canonical_optimal_assignment(cost, positions, max_swaps=None):
    """The solver's optimum, canonicalised toward the lexicographically smallest position vector read in ascending item
    order, in two steps that both preserve the objective exactly.

    Items with identical cost rows are freely interchangeable (every swap between them costs exactly zero), so their
    positions are sorted ascending and handed back in ascending item order. This settles the tie structure that
    dominates at scale, where a whole block of items carries the same row, and it costs one pass over the matrix.
    Whatever ties are left are then reduced by single zero-cost transpositions, each of which strictly lowers the
    position vector, for at most `max_swaps` swaps (2V by default): that loop is bubble-sort-like, so a large tied
    block would need O(V^2) swaps, and the bound keeps the hot path cheap.

    Returns (positions, swaps applied, settled). `settled` is False when the bound was reached, in which case the order
    is still a global minimiser and only the tie convention is the solver's. A transposition of strictly negative cost
    raises: the solver did not return an optimum.
    """
    cost, positions = np.asarray(cost, dtype=float), np.asarray(positions, dtype=np.int64).copy()
    size = cost.shape[0]
    groups = {}
    for index in range(size):
        groups.setdefault(cost[index].tobytes(), []).append(index)
    for members in groups.values():
        if len(members) > 1:
            positions[members] = np.sort(positions[members])     # ascending items take ascending tied positions
    limit = int(TIE_SWAP_LIMIT * size if max_swaps is None else max_swaps)
    for swaps in range(limit + 1):
        chosen = cost[:, positions]                              # chosen[p, p'] = cost of the item at p taking positions[p']
        base = np.diag(chosen)
        delta = chosen + chosen.T - base[:, None] - base[None, :]
        if np.any(delta < -1e-9):
            raise InterventionError('A transposition strictly improves the assignment; it was not optimal')
        lower = positions[None, :] < positions[:, None]          # swapping puts the smaller position at the lower row
        eligible = np.triu(lower & (delta <= 0), 1)
        if not eligible.any():
            return positions, swaps, True
        if swaps == limit:
            return positions, swaps, False
        p = int(np.flatnonzero(eligible.any(axis=1))[0])
        q = int(np.flatnonzero(eligible[p])[np.argmin(positions[eligible[p]])])
        positions[[p, q]] = positions[[q, p]]
    return positions, limit, False


def footrule_median_positions(accumulated, validate=True):
    """The exact footrule-median order as positions: argmin over the V! rankings of sum_p sum_q A[p,q] |q - rank(p)|,
    solved as a linear assignment on footrule_cost and canonicalised by canonical_optimal_assignment.
    Returns (positions, settled); the order is a global minimiser either way."""
    cost = footrule_cost(accumulated, validate=validate)
    rows, columns = linear_sum_assignment(cost)
    positions = np.empty(cost.shape[0], dtype=np.int64)
    positions[rows] = columns
    positions, _, settled = canonical_optimal_assignment(cost, positions)
    return positions, settled


def footrule_objective(accumulated, positions):
    """sum_p sum_q A[p, q] |q - positions[p]|, the value the median arm minimises."""
    cost = footrule_cost(accumulated)
    positions = np.asarray(positions, dtype=np.int64)
    if positions.shape != (cost.shape[0],) or sorted(positions.tolist()) != list(range(cost.shape[0])):
        raise InterventionError('Expected a permutation of the positions')
    return float(cost[np.arange(cost.shape[0]), positions].sum())


def brute_force_footrule_median(accumulated):
    """Every ranking enumerated (small vocabularies only): (minimum objective, the lexicographically smallest minimiser
    read in ascending item order, the number of minimisers)."""
    cost = footrule_cost(accumulated)
    size = cost.shape[0]
    if size > 8:
        raise InterventionError('Brute force is for small vocabularies only')
    best, minimisers = None, []
    for candidate in permutations(range(size)):
        value = float(cost[np.arange(size), list(candidate)].sum())
        if best is None or value < best - 1e-12:
            best, minimisers = value, [candidate]
        elif abs(value - best) <= 1e-12:
            minimisers.append(candidate)
    return best, np.asarray(min(minimisers), dtype=np.int64), len(minimisers)


def prior_weight(rule, mass, votes, multiplier=1.):
    """The weight the prior carries in the median arm: 'unit' leaves the accumulator as the core builds it (weight 1
    against an incoming mass of `mass`); 'one_ballot' gives the prior `multiplier` times the mean incoming ballot weight
    mass / votes, so the incoming-to-prior mass ratio is votes / multiplier : 1."""
    if rule not in PRIOR_RULES:
        raise InterventionError(f'prior rule must be one of {", ".join(PRIOR_RULES)}')
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise InterventionError('The prior multiplier must be positive and finite')
    if votes <= 0 or mass <= 0:
        return 1.
    return 1. if rule == 'unit' else float(multiplier) * float(mass) / int(votes)


def reweighted_prior(accumulated, weight):
    """The accumulator with the prior's unit diagonal replaced by `weight`; the votes are untouched."""
    A = _accumulator(accumulated)
    if not np.isfinite(weight) or weight <= 0:
        raise InterventionError('The prior weight must be positive and finite')
    if np.any(np.diag(A) < 1 - 1e-9):
        raise InterventionError('The accumulator does not carry the unit prior on its diagonal')
    return A + (float(weight) - 1.) * np.identity(A.shape[0])


# ----------------------------------------------------------------------------- instance-level interventions

class VoteCounter:
    """Instance-level Vertex.accumulate_motion: counts the eligible votes and their mass without touching the result."""

    def __init__(self, vertex):
        self.vertex, self.original = vertex, type(vertex).accumulate_motion
        self.votes, self.mass = 0, 0.

    def __call__(self, adj_list_comp, magnitude=1):
        out = self.original(self.vertex, adj_list_comp, magnitude)
        if magnitude is not None and magnitude != 0:
            self.votes += 1
            self.mass += abs(float(magnitude))
        return out

    def reset(self):
        votes, mass = self.votes, self.mass
        self.votes, self.mass = 0, 0.
        return votes, mass


class MedianOrder:
    """Instance-level Vertex.compute_adj_list_with_permutation: the footrule-median order of the same accumulator."""

    def __init__(self, vertex, counter, prior_rule='unit', multiplier=1.):
        if prior_rule not in PRIOR_RULES:
            raise InterventionError(f'prior rule must be one of {", ".join(PRIOR_RULES)}')
        self.vertex, self.counter, self.prior_rule, self.multiplier = vertex, counter, prior_rule, float(multiplier)
        self.calls = self.solved = self.unsettled = 0

    def __call__(self):
        vertex, A = self.vertex, self.vertex.permutation_matrix_accumulate
        self.calls += 1
        votes, mass = self.counter.votes, self.counter.mass
        if votes == 0:                                    # no vote: the identity accumulator, whose unique optimum is the prior
            return vertex.adjacency_list
        expected = 1. + mass
        if not np.allclose(np.asarray(A).sum(axis=1), expected, rtol=0, atol=1e-8):
            raise InterventionError('The accumulator row mass differs from one prior plus the recorded vote mass')
        self.solved += 1
        weight = prior_weight(self.prior_rule, mass, votes, self.multiplier)
        positions, settled = footrule_median_positions(reweighted_prior(A, weight), validate=False)
        self.unsettled += int(not settled)
        return vertex.adjacency_list_np[np.argsort(positions, kind='stable')].tolist()


def sort_vertices(graph):
    """Every Vertex of a sort-layer graph, layer by layer."""
    return [vertex for layer in graph.vertex_list.values() if hasattr(layer, 'graph')
            for vertex in layer.graph.vertex_list.values()]


class LayerRecord:
    """Per-batch movement of one layer: the share of filters that changed and their mean normalised footrule."""

    def __deepcopy__(self, memo):
        return self          # instrumentation, not network state: the core's checkpoint copies never train

    def __init__(self, layer, name, filters, vocabulary):
        self.layer, self.name, self.filters, self.vocabulary = layer, name, filters, vocabulary
        self.normaliser = max(1, vocabulary * vocabulary // 2)
        self.batches, self.pending = [], self._empty()

    def _empty(self):
        return {'updates': 0, 'changed': 0, 'displacement': 0., 'votes': 0, 'mass': 0., 'max_mass': 0., 'solved': 0,
                'unsettled': 0}

    def add(self, *, moved, displacement, votes, mass, solved, unsettled=0):
        self.pending['updates'] += 1
        self.pending['changed'] += int(moved)
        self.pending['displacement'] += float(displacement) / self.normaliser
        self.pending['votes'] += int(votes)
        self.pending['mass'] += float(mass)
        self.pending['max_mass'] = max(self.pending['max_mass'], float(mass))
        self.pending['solved'] += int(solved)
        self.pending['unsettled'] += int(unsettled)

    def close_batch(self):
        if self.pending['updates']:
            self.batches.append(self.pending)
        self.pending = self._empty()

    def summary(self):
        updates = sum(b['updates'] for b in self.batches)
        if not updates:
            return {'layer': self.layer, 'layer_name': self.name, 'n_filters': self.filters, 'vocabulary': self.vocabulary,
                    'batches': len(self.batches), 'updates': 0, 'changed_share': None, 'mean_displacement': None,
                    'mean_votes': None, 'mean_vote_mass': None, 'max_vote_mass': None, 'median_solves': 0,
                    'tie_canonicalisation_incomplete': 0, 'batch_changed_share': [], 'batch_mean_displacement': []}
        shares = [b['changed'] / b['updates'] for b in self.batches]
        moves = [b['displacement'] / b['updates'] for b in self.batches]
        return {'layer': self.layer, 'layer_name': self.name, 'n_filters': self.filters, 'vocabulary': self.vocabulary,
                'batches': len(self.batches), 'updates': updates,
                'changed_share': float(sum(b['changed'] for b in self.batches) / updates),
                'mean_displacement': float(sum(b['displacement'] for b in self.batches) / updates),
                'mean_votes': float(sum(b['votes'] for b in self.batches) / updates),
                'mean_vote_mass': float(sum(b['mass'] for b in self.batches) / updates),
                'max_vote_mass': float(max(b['max_mass'] for b in self.batches)),
                'median_solves': int(sum(b['solved'] for b in self.batches)),
                'tie_canonicalisation_incomplete': int(sum(b['unsettled'] for b in self.batches)),
                'batch_changed_share': [float(v) for v in shares], 'batch_mean_displacement': [float(v) for v in moves]}


class VertexUpdate:
    """Instance-level Vertex.apply_motion: applies the core update (or freezes the filter) and records the movement."""

    def __init__(self, vertex, record, counter, *, frozen=False, order=None):
        self.vertex, self.record, self.counter, self.frozen, self.order = vertex, record, counter, frozen, order
        self.original = type(vertex).apply_motion

    def __call__(self, update_adj_list=True):
        vertex = self.vertex
        before = list(vertex.adjacency_list)
        solved_before = 0 if self.order is None else self.order.solved
        unsettled_before = 0 if self.order is None else self.order.unsettled
        if self.frozen:
            vertex.clean_motion_accumulation(vertex.adjacency_list)     # the batch reset of the core, nothing else
            out = None
        else:
            out = self.original(vertex, update_adj_list)
        after = list(vertex.adjacency_list)
        displacement = 0
        if after != before:
            places = {item: index for index, item in enumerate(after)}
            if len(places) != len(after) or set(places) != set(before):
                raise InterventionError('An update changed the filter vocabulary')
            displacement = sum(abs(places[item] - index) for index, item in enumerate(before))
        votes, mass = self.counter.reset()
        self.record.add(moved=after != before, displacement=displacement, votes=votes, mass=mass,
                        solved=0 if self.order is None else self.order.solved - solved_before,
                        unsettled=0 if self.order is None else self.order.unsettled - unsettled_before)
        return out


class ArmInstrumentation:
    """One view network's interventions: the hidden-layer rule, the frozen layers and the per-batch movement record.

    `rule` is 'borda' (the core's own order) or 'median'; `prior_rule` applies to the median only; `frozen_hidden`
    names hidden layers (0-based, from the input) whose filters never move. The output layer is never touched here: the
    arms that hold it fixed pass last_layer_update=False, the library's own switch.
    """

    def __init__(self, net, *, rule='borda', prior_rule='unit', prior_multiplier=1., frozen_hidden=()):
        network = getattr(net, 'network_', None)
        if network is None or network.update_iter != 0:
            raise InterventionError('Install the instrumentation on an initialised, untrained view network')
        if rule not in RULES:
            raise InterventionError(f'rule must be one of {", ".join(RULES)}')
        self.net, self.network, self.rule, self.prior_rule = net, network, rule, prior_rule
        self.prior_multiplier = float(prior_multiplier)
        self.hidden = len(net.widths)
        self.frozen_hidden = sorted({int(layer) for layer in frozen_hidden})
        if any(not 0 <= layer < self.hidden for layer in self.frozen_hidden):
            raise InterventionError('A frozen hidden layer must be one of this network\'s hidden layers')
        self.records, self.batches, self.rng_checks = [], 0, []
        self.installed_attributes = 0
        self.initial_frozen = {layer: self._orders(layer) for layer in self.frozen_hidden}

    def _layers(self):
        return [self.network.graph.vertex_list[f'{self.network.id}_ly{i}'] for i in range(self.hidden)]

    def _orders(self, layer):
        vertices = self.network.graph.vertex_list[f'{self.network.id}_ly{layer}'].graph.vertex_list.values()
        return [tuple(vertex.adjacency_list) for vertex in vertices]

    def frozen_unchanged(self):
        """Every frozen hidden layer still carries its initial filters and recorded no movement."""
        moved = {record.layer: record.summary()['changed_share'] for record in self.records}
        return all(self._orders(layer) == orders and not moved.get(layer)
                   for layer, orders in self.initial_frozen.items())

    def _install(self):
        for index, layer in enumerate(self._layers()):
            vertices = list(layer.graph.vertex_list.values())
            record = LayerRecord(index, f'hidden_{index}', len(vertices), len(vertices[0].adjacency_list))
            self.records.append(record)
            for vertex in vertices:
                for name in ATTRIBUTES:
                    if name in vars(vertex):
                        raise InterventionError(f'The vertex already carries an instance-level {name}')
                counter = VoteCounter(vertex)
                order = None
                if self.rule == 'median' and index not in self.frozen_hidden:
                    order = MedianOrder(vertex, counter, self.prior_rule, self.prior_multiplier)
                    vertex.compute_adj_list_with_permutation = order
                    self.installed_attributes += 1
                vertex.accumulate_motion = counter
                vertex.apply_motion = VertexUpdate(vertex, record, counter, frozen=index in self.frozen_hidden, order=order)
                self.installed_attributes += 2

    def _graphs(self):
        return [self.network.graph] + [g for g in (getattr(self.network, 'optimal_model', None),
                                                   getattr(self.network, 'sub_optimal_model', None)) if g is not None]

    def _remove(self):
        for graph in self._graphs():
            for vertex in sort_vertices(graph):
                for name in ATTRIBUTES:
                    vars(vertex).pop(name, None)

    def instrumented(self):
        """Every instance-level intervention still installed on a reachable graph (empty after the fit)."""
        return [f'{index}:{vertex.id}:{name}' for index, graph in enumerate(self._graphs())
                for vertex in sort_vertices(graph) for name in ATTRIBUTES if name in vars(vertex)]

    @contextmanager
    def installed(self):
        import random
        original = self.network.update_network
        recorder = self

        def update_network(data_train, data_validation=None, train_type='supervised', problem='classification'):
            out = original(data_train, data_validation, train_type, problem)
            numpy_state, python_state = np.random.get_state(), random.getstate()
            recorder.batches += 1
            for record in recorder.records:
                record.close_batch()
            unchanged = (numpy_state[0] == np.random.get_state()[0] and np.array_equal(numpy_state[1], np.random.get_state()[1])
                         and tuple(numpy_state[2:]) == tuple(np.random.get_state()[2:]) and python_state == random.getstate())
            recorder.rng_checks.append(bool(unchanged))
            return out

        self._install()
        self.network.update_network = update_network
        try:
            yield self
        finally:
            del self.network.update_network
            self._remove()

    def summary(self):
        records = [record.summary() for record in self.records]
        updates = sum(record['updates'] for record in records)
        return {'rule': self.rule, 'prior_rule': self.prior_rule if self.rule == 'median' else None,
                'prior_multiplier': self.prior_multiplier if self.rule == 'median' else None,
                'frozen_hidden_layers': list(self.frozen_hidden), 'batches': self.batches,
                'hidden_updates': updates,
                'changed_share': float(sum(r['changed_share'] * r['updates'] for r in records) / updates) if updates else None,
                'mean_displacement': float(sum(r['mean_displacement'] * r['updates'] for r in records) / updates) if updates else None,
                'median_solves': int(sum(r['median_solves'] for r in records)),
                'tie_canonicalisation_incomplete': int(sum(r['tie_canonicalisation_incomplete'] for r in records)),
                'rng_guarded_batches': len(self.rng_checks), 'rng_unchanged': all(self.rng_checks) if self.rng_checks else None,
                'instrumentation_removed': not self.instrumented(), 'layers': records}
