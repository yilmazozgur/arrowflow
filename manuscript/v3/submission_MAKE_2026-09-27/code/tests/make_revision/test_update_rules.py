"""The controlled-intervention primitives (update_rules): the footrule median against brute force, the Borda
reimplementation against the real Vertex, the prior-weight rules, and the instance-level instrumentation on a tiny
network (non-invasive for the unmodified rule, effective for the median, exact for a frozen layer)."""
import copy
import numpy as np
import pytest
from arrowflow.arrowflow import Vertex
from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config
from experiments.make_revision import update_rules as ur
from experiments.make_revision.models import ArrowFlowEstimator, OrdinalEncoder, seed_fit


def tiny_config(classes=3):
    return _build_sortnet_config(ArrowFlowConfig(no_of_filters=[4, classes], layer_types=['sort', 'sort'], no_of_iters=1,
                                                 batch_size=4, learning_rate=.1, val_data_ratio=0., device='cpu',
                                                 verbose=0, evaluate_train_data=False), classes)


def random_accumulator(rng, size, votes=None, weights=(.25, .5, 1., 2., .006)):
    """One vertex's accumulator: the unit prior on the diagonal and a few weighted votes."""
    A = np.identity(size)
    mass = 0.
    for _ in range(int(rng.randint(1, 6)) if votes is None else votes):
        weight = float(rng.choice(weights))
        A[np.arange(size), rng.permutation(size)] += weight
        mass += weight
    return A, mass


def tiny_network(widths=(6, 5), iterations=6, last_layer=True, seed=11):
    rng = np.random.RandomState(0)
    y = np.tile([0, 1, 2], 40)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    orders = OrdinalEncoder('random', 8, 1, .3, seed).fit(X, y).transform(X)
    seed_fit(seed)
    net = ArrowFlowEstimator(embed_dim=8, widths=list(widths), iterations=iterations, learning_rate=.2, batch_size=16,
                             validation_ratio=.1, last_layer_update=last_layer, seed=seed)
    return net.initialize_orders(orders, y), orders, y


def layer_orders(net, layer):
    vertices = net.network_.graph.vertex_list[f'revision_ly{layer}'].graph.vertex_list.values()
    return [tuple(vertex.adjacency_list) for vertex in vertices]


def tiny_fit(rule=None, prior_rule='unit', multiplier=1., frozen=(), widths=(6, 5), iterations=6, last_layer=True, seed=11):
    net, orders, y = tiny_network(widths, iterations, last_layer, seed)
    if rule is None:
        net.train_initialized(orders, y)
        return net, None, orders, y
    arm = ur.ArmInstrumentation(net, rule=rule, prior_rule=prior_rule, prior_multiplier=multiplier, frozen_hidden=frozen)
    with arm.installed():
        net.train_initialized(orders, y)
    return net, arm, orders, y


# ----------------------------------------------------------------------------- the two orders of one accumulator

def test_borda_positions_are_the_core_order():
    """The reimplementation must be the rule the core applies, on the real Vertex through its real update path."""
    config, rng, checked = tiny_config(), np.random.RandomState(3), 0
    for _ in range(120):
        size = int(rng.randint(3, 7))
        items = [str(i + 1) for i in range(size)]
        vertex = Vertex('v', list(items), 'sort', config)
        mass = 0.
        for _ in range(int(rng.randint(1, 5))):
            weight = float(rng.choice([.25, .5, 1., .006]))
            vertex.accumulate_motion([items[i] for i in rng.permutation(size)], magnitude=weight)
            mass += weight
        accumulated = vertex.permutation_matrix_accumulate.copy()
        assert np.allclose(accumulated.sum(axis=1), 1 + mass)        # unit prior, one row mass per vertex
        positions = ur.borda_positions(accumulated, vertex.adjacency_list)
        expected = np.asarray(vertex.adjacency_list_np)[np.argsort(positions, kind='stable')].tolist()
        assert vertex.compute_adj_list_with_permutation() == expected
        checked += 1
    assert checked == 120


def test_footrule_median_is_the_brute_force_minimiser():
    """Every returned order attains the minimum of the footrule objective over all orders; the tie canonicalisation
    reaches the lexicographically smallest minimiser in all but a few exactly tied instances."""
    rng, optimal, lexicographic, tied = np.random.RandomState(7), 0, 0, 0
    for _ in range(300):
        size = int(rng.randint(3, 7))
        accumulated, _ = random_accumulator(rng, size)
        positions, settled = ur.footrule_median_positions(accumulated)
        best, smallest, minimisers = ur.brute_force_footrule_median(accumulated)
        assert settled
        assert sorted(positions.tolist()) == list(range(size))
        assert ur.footrule_objective(accumulated, positions) == pytest.approx(best, abs=1e-9)
        assert np.array_equal(ur.canonical_optimal_assignment(ur.footrule_cost(accumulated), positions)[0], positions)
        optimal += 1
        tied += int(minimisers > 1)
        lexicographic += int(np.array_equal(positions, smallest))
    assert optimal == 300 and tied > 0
    assert lexicographic >= 285                                      # 393/400 in the protocol's recorded measurement


def test_the_median_follows_the_majority_of_the_mass_where_the_mean_blends():
    """Four items: the prior and two unit votes keep the order, one vote of mass 4 wants (0, 2, 3, 1). The mean blends
    the two into an order neither side asked for; the median, carrying more than half the mass, takes the heavy one."""
    accumulated = np.identity(4)
    for weight, vote in ((1., [0, 1, 2, 3]), (1., [0, 1, 2, 3]), (4., [0, 2, 3, 1])):
        accumulated[np.arange(4), vote] += weight
    borda, median = ur.borda_positions(accumulated), ur.footrule_median_positions(accumulated)[0]
    assert np.array_equal(borda, [0, 1, 3, 2]) and np.array_equal(median, [0, 2, 3, 1])
    assert ur.footrule_objective(accumulated, median) == 12. < ur.footrule_objective(accumulated, borda) == 14.


def test_an_accumulator_without_votes_keeps_the_prior_order():
    identity = np.identity(5)
    assert np.array_equal(ur.footrule_median_positions(identity)[0], np.arange(5))
    assert np.array_equal(ur.borda_positions(identity), np.arange(5))


def test_prior_weight_rules_and_reweighting():
    assert ur.prior_weight('unit', mass=.1, votes=16) == 1.
    assert ur.prior_weight('one_ballot', mass=.8, votes=16) == pytest.approx(.05)
    assert ur.prior_weight('one_ballot', mass=.8, votes=16, multiplier=4) == pytest.approx(.2)
    assert ur.prior_weight('one_ballot', mass=0., votes=0) == 1.     # no vote leaves the prior alone
    with pytest.raises(ur.InterventionError):
        ur.prior_weight('mean', mass=1., votes=1)
    accumulated, _ = random_accumulator(np.random.RandomState(1), 5)
    reweighted = ur.reweighted_prior(accumulated, .25)
    assert np.allclose(np.diag(reweighted), np.diag(accumulated) - .75)
    assert np.allclose(reweighted - np.diag(np.diag(reweighted)), accumulated - np.diag(np.diag(accumulated)))
    with pytest.raises(ur.InterventionError):
        ur.reweighted_prior(np.zeros((3, 3)), 1.)


def test_a_heavier_prior_freezes_the_median_and_a_lighter_one_moves_it():
    """The fairness knob: with light votes against the unit prior the median cannot move, and the mass rule frees it."""
    rng = np.random.RandomState(4)
    accumulated, mass = random_accumulator(rng, 8, votes=16, weights=(.006,))
    assert mass < 1
    assert np.array_equal(ur.footrule_median_positions(accumulated)[0], np.arange(8))
    matched = ur.reweighted_prior(accumulated, ur.prior_weight('one_ballot', mass, 16))
    assert not np.array_equal(ur.footrule_median_positions(matched)[0], np.arange(8))


def test_canonical_assignment_refuses_an_assignment_that_is_not_optimal():
    cost = np.asarray([[0., 5.], [5., 0.]])
    with pytest.raises(ur.InterventionError):
        ur.canonical_optimal_assignment(cost, np.asarray([1, 0]))
    positions, swaps, settled = ur.canonical_optimal_assignment(cost, np.asarray([0, 1]))
    assert np.array_equal(positions, [0, 1]) and swaps == 0 and settled


def test_canonical_assignment_lowers_a_tied_order_lexicographically():
    cost = np.zeros((3, 3))                                          # identical rows: every order is optimal
    positions, swaps, settled = ur.canonical_optimal_assignment(cost, np.asarray([2, 1, 0]))
    assert np.array_equal(positions, [0, 1, 2]) and swaps == 0 and settled   # settled by the identical-row step alone


def test_a_large_tied_block_is_canonicalised_without_a_swap_storm():
    """The failure the aggregation pilot found: a block of interchangeable items needs O(V^2) single swaps, far past
    any sane bound, and the identical-row step settles it in one pass."""
    size = 40
    cost = np.tile(np.abs(np.arange(size) - np.arange(size)[:, None])[0], (size, 1)).astype(float)
    positions, swaps, settled = ur.canonical_optimal_assignment(cost, np.arange(size)[::-1].copy())
    assert np.array_equal(positions, np.arange(size)) and swaps == 0 and settled


# ----------------------------------------------------------------------------- the instance-level instrumentation

def test_the_unmodified_rule_is_non_invasive():
    plain, _, orders, y = tiny_fit()
    instrumented, arm, _, _ = tiny_fit('borda')
    assert instrumented.state_hash() == plain.state_hash()
    assert np.array_equal(instrumented.predict_orders(orders), plain.predict_orders(orders))
    summary = arm.summary()
    assert arm.instrumented() == [] and summary['instrumentation_removed'] and summary['rng_unchanged']
    assert summary['batches'] == 6 and summary['median_solves'] == 0
    assert summary['hidden_updates'] == 6 * (6 + 5) and 0 < summary['changed_share'] <= 1


def test_the_median_rule_changes_the_network_and_solves_every_voted_update():
    plain, _, _, _ = tiny_fit()
    median, arm, _, _ = tiny_fit('median')
    summary = arm.summary()
    assert median.state_hash() != plain.state_hash()
    assert summary['median_solves'] > 0 and summary['rule'] == 'median' and summary['prior_rule'] == 'unit'
    matched, matched_arm = tiny_fit('median', prior_rule='one_ballot')[:2]
    assert matched.state_hash() != median.state_hash()
    assert matched_arm.summary()['changed_share'] > summary['changed_share']   # the mass rule frees the median


def test_a_frozen_hidden_layer_never_moves_and_records_no_movement():
    frozen, arm, orders, y = tiny_fit('borda', frozen=(1,))
    assert arm.frozen_unchanged()
    assert layer_orders(frozen, 1) == layer_orders(tiny_network()[0], 1)
    layer = next(entry for entry in arm.summary()['layers'] if entry['layer'] == 1)
    assert layer['changed_share'] == 0. and layer['mean_displacement'] == 0. and layer['updates'] > 0
    other = next(entry for entry in arm.summary()['layers'] if entry['layer'] == 0)
    assert other['updates'] > 0


def test_the_output_layer_switch_holds_the_output_filters_at_their_initial_orders():
    """depth2_first_only holds the output layer through the library's own last_layer_update, not through a wrapper."""
    held = tiny_fit('borda', frozen=(1,), last_layer=False)[0]
    trained = tiny_fit('borda', frozen=(1,), last_layer=True)[0]
    initial = tiny_network(last_layer=False)[0]
    assert layer_orders(held, 2) == layer_orders(initial, 2) != layer_orders(trained, 2)


def test_the_record_is_guarded_once_per_batch():
    _, arm, _, _ = tiny_fit('borda')
    assert all(arm.rng_checks) and len(arm.rng_checks) == arm.batches == 6


def test_the_instance_rule_survives_a_deepcopy_bound_to_the_copy():
    """The core deep-copies the graph at every checkpoint; a copied vertex must read its own accumulator."""
    vertex = Vertex('v', ['1', '2', '3'], 'sort', tiny_config())
    counter = ur.VoteCounter(vertex)
    vertex.accumulate_motion = counter
    order = ur.MedianOrder(vertex, counter)
    vertex.compute_adj_list_with_permutation = order
    duplicate = copy.deepcopy(vertex)
    assert duplicate.compute_adj_list_with_permutation.vertex is duplicate
    assert duplicate.accumulate_motion.vertex is duplicate
    assert duplicate.compute_adj_list_with_permutation.counter is duplicate.accumulate_motion


def test_the_vote_counter_counts_only_eligible_votes():
    vertex = Vertex('v', ['1', '2', '3'], 'sort', tiny_config())
    counter = ur.VoteCounter(vertex)
    vertex.accumulate_motion = counter
    vertex.accumulate_motion(['2', '1', '3'], magnitude=.5)
    vertex.accumulate_motion(['2', '1', '3'], magnitude=0)
    vertex.accumulate_motion(['2', '1', '3'], magnitude=None)
    vertex.accumulate_motion(['3', '1', '2'], magnitude=-.25)        # a repulsion vote carries |a|
    assert (counter.votes, counter.mass) == (2, .75)
    assert np.allclose(vertex.permutation_matrix_accumulate.sum(axis=1), 1.75)
    assert counter.reset() == (2, .75) and (counter.votes, counter.mass) == (0, 0.)


def test_installing_needs_an_initialised_untrained_network_and_a_known_rule():
    with pytest.raises(ur.InterventionError):
        ur.ArmInstrumentation(tiny_fit(iterations=1)[0])               # already trained
    with pytest.raises(ur.InterventionError):
        ur.ArmInstrumentation(ArrowFlowEstimator(embed_dim=8, widths=[4], iterations=1, seed=3))   # not initialised
    with pytest.raises(ur.InterventionError):
        ur.ArmInstrumentation(tiny_network()[0], rule='kemeny')


def test_a_frozen_layer_must_be_one_of_the_hidden_layers():
    with pytest.raises(ur.InterventionError):
        ur.ArmInstrumentation(tiny_network(widths=(5,))[0], frozen_hidden=[1])
