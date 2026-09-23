"""The corrected relay and the per-layer vote scale (signed_relay): the correction against the motion computed directly,
the relay equal to the printed one on attraction and zero votes and different only on repulsion votes (vote by vote and
through one real batch update), exact reproduction of the unmodified network with one hidden layer and in the printed
mode with two, agreement with an independent replica of backward_propagate adapted from the verifier's
(verification/v1/relay_shadow_run.py), the scale, the prior's movement threshold and the installation rules."""
import copy
import numpy as np
import pytest
from arrowflow.arrowflow import SortFlowHybridNetwork, Vertex
from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config
from arrowflow.ranking import score_order
from experiments.make_revision import signed_relay as sr
from experiments.make_revision import update_rules as ur
from experiments.make_revision.models import ArrowFlowEstimator, OrdinalEncoder, seed_fit


def tiny_config(classes=3):
    return _build_sortnet_config(ArrowFlowConfig(no_of_filters=[4, classes], layer_types=['sort', 'sort'], no_of_iters=1,
                                                 batch_size=4, learning_rate=.1, val_data_ratio=0., device='cpu',
                                                 verbose=0, evaluate_train_data=False), classes)


def small_network(widths=(8, 12), iterations=25, seed=11, embed=10, learning_rate=.2, validation_ratio=.1):
    """A three-class problem hard enough that the top hidden layer casts attraction and repulsion votes every batch."""
    rng = np.random.RandomState(0)
    y = np.tile([0, 1, 2], 50)
    X = rng.randn(len(y), 5)
    X[np.arange(len(y)), y] += .8
    orders = OrdinalEncoder('random', embed, 1, .3, seed).fit(X, y).transform(X)
    seed_fit(seed)
    net = ArrowFlowEstimator(embed_dim=embed, widths=list(widths), iterations=iterations, learning_rate=learning_rate,
                             batch_size=16, validation_ratio=validation_ratio, seed=seed)
    return net.initialize_orders(orders, y), orders, y


def fit(relay=None, scale=1., **kwargs):
    net, orders, y = small_network(**kwargs)
    if relay is None:
        net.train_initialized(orders, y)
        return net, None, orders, y
    with sr.relay_rule(net, relay=relay, lower_vote_scale=scale) as instrumentation:
        net.train_initialized(orders, y)
    return net, instrumentation, orders, y


def layer_orders(net, layer):
    vertices = net.network_.graph.vertex_list[f'revision_ly{layer}'].graph.vertex_list.values()
    return [tuple(vertex.adjacency_list) for vertex in vertices]


# ----------------------------------------------------------------------------- the correction itself

def test_the_correction_is_minus_the_motion_toward_the_unreversed_input():
    """Through the real Vertex: a repulsion returns m(r -> rev(pi)); the corrected relay is -m(r -> pi) exactly."""
    rng = np.random.RandomState(3)
    for size in list(range(2, 12)) + [16, 40, 64]:
        for _ in range(20):
            items = [str(i) for i in range(1, size + 1)]
            filter_order = list(rng.permutation(items))
            pi = list(rng.permutation(items))
            vertex = Vertex('v', filter_order, 'sort', tiny_config())
            returned = vertex.accumulate_motion(pi, magnitude=-.3)
            assert np.array_equal(returned, sr.toward_input(filter_order, pi[::-1]))
            corrected = sr.signed_relay_motion(returned, -.3, size)
            assert np.array_equal(corrected, -sr.toward_input(filter_order, pi))
            positions = np.arange(size)
            assert np.array_equal(corrected, returned + 2 * positions - (size - 1))
            # the printed relay after a repulsion is orthogonal to the push-away direction y - x (V1-report A.1)
            assert np.dot(returned, -sr.toward_input(filter_order, pi)) == 0
            assert sr.signed_relay_motion(returned, .3, size) is returned
            assert sr.signed_relay_motion(returned, 0., size) is returned


def test_the_relay_equals_the_printed_one_on_attractions_and_zero_votes_and_differs_only_on_repulsions():
    """Vote by vote on the real Vertex: the same accumulation in every case; the same return value for a_j >= 0; for
    a_j < 0 the corrected return value -m(r -> pi) in place of the printed m(r -> rev(pi))."""
    rng = np.random.RandomState(5)
    for _ in range(300):
        size = int(rng.randint(3, 20))
        items = [str(i) for i in range(1, size + 1)]
        filter_order, pi = list(rng.permutation(items)), list(rng.permutation(items))
        magnitude = float(rng.choice([-1, 1, 0]) * rng.uniform(.001, 2.))
        plain = Vertex('v', list(filter_order), 'sort', tiny_config())
        wrapped = Vertex('v', list(filter_order), 'sort', tiny_config())
        record = sr.RelayLayerRecord(1, 'hidden_1', relays=True, scale=1.)
        wrapped.accumulate_motion = sr.RelayedMotion(wrapped, None, 'signed', record)
        printed = plain.accumulate_motion(pi, magnitude=magnitude)
        signed = wrapped.accumulate_motion(pi, magnitude=magnitude)
        assert np.array_equal(plain.permutation_matrix_accumulate, wrapped.permutation_matrix_accumulate)
        if magnitude >= 0:
            assert signed is not None and np.array_equal(signed, printed)
        else:
            assert np.array_equal(signed, -sr.toward_input(filter_order, pi))
            assert not np.array_equal(signed, printed)            # the vectors differ whenever V > 1
        assert plain.compute_adj_list_with_permutation() == wrapped.compute_adj_list_with_permutation()
    printed_mode = Vertex('v', ['1', '2', '3', '4'], 'sort', tiny_config())
    record = sr.RelayLayerRecord(1, 'hidden_1', relays=True, scale=1.)
    printed_mode.accumulate_motion = sr.RelayedMotion(printed_mode, None, 'printed', record)
    out = printed_mode.accumulate_motion(['4', '2', '1', '3'], magnitude=-.5)
    assert np.array_equal(out, sr.toward_input(['1', '2', '3', '4'], ['3', '1', '2', '4']))
    assert (record.repulsions, record.corrected, record.verified_repulsions) == (1, 0, 1)


def recorded_step(relay):
    """One real batch update of a two-hidden-layer network with the relay wrapper; every call at the top hidden layer
    recorded as (vertex, magnitude, input, returned) and every signal handed to the first hidden layer. Without a
    validation split the returned network is the updated one, not a checkpoint."""
    net, orders, y = small_network(iterations=1, validation_ratio=0.)
    calls, handed = [], []
    network = net.network_
    original_backward = type(network).backward_propagate

    with sr.relay_rule(net, relay=relay) as instrumentation:
        for vertex in network.graph.vertex_list['revision_ly1'].graph.vertex_list.values():
            inner = vertex.accumulate_motion

            def recorder(adj_list_comp, magnitude=1, inner=inner, vertex=vertex):
                out = inner(adj_list_comp, magnitude)
                calls.append((vertex.id, magnitude, tuple(adj_list_comp), np.array(out, copy=True)))
                return out
            vertex.accumulate_motion = recorder
        for vertex in network.graph.vertex_list['revision_ly0'].graph.vertex_list.values():
            inner = vertex.accumulate_motion

            def lower(adj_list_comp, magnitude=1, inner=inner, vertex=vertex):
                handed.append((vertex.id, magnitude))
                return inner(adj_list_comp, magnitude)
            vertex.accumulate_motion = lower
        net.train_initialized(orders, y)
    assert type(network).backward_propagate is original_backward
    return net, calls, handed, instrumentation


def test_one_real_batch_update_differs_only_where_a_repulsion_is_relayed():
    printed, printed_calls, printed_handed, _ = recorded_step('printed')
    signed, signed_calls, signed_handed, instrumentation = recorded_step('signed')
    assert len(printed_calls) == len(signed_calls) > 0
    repulsions = attractions = 0
    for (vid, magnitude, pi, out_p), (vid_s, magnitude_s, pi_s, out_s) in zip(printed_calls, signed_calls):
        assert (vid, magnitude, pi) == (vid_s, magnitude_s, pi_s)         # the top hidden layer casts the same votes
        vertex = printed.network_.graph.vertex_list['revision_ly1'].graph.vertex_list[vid]
        if magnitude < 0:
            repulsions += 1
            assert np.array_equal(out_s, out_p + 2 * np.arange(len(out_p)) - (len(out_p) - 1))
        else:
            attractions += int(magnitude > 0)
            assert np.array_equal(out_s, out_p)
    assert repulsions > 0 and attractions > 0
    # the top hidden layer and the output layer updated identically; the relay only reaches the first hidden layer
    assert layer_orders(printed, 1) == layer_orders(signed, 1) and layer_orders(printed, 2) == layer_orders(signed, 2)
    assert len(printed_handed) == len(signed_handed) and printed_handed != signed_handed
    relay = {entry['layer']: entry for entry in instrumentation.summary()['relay_layers']}
    assert relay[1]['returned_motions_corrected'] == relay[1]['repulsions'] == repulsions
    assert relay[1]['relays_to_layer_below'] and not relay[0]['relays_to_layer_below']


# ----------------------------------------------------------------------------- exact reproduction where nothing changes

def test_one_hidden_layer_reproduces_the_unmodified_network_exactly():
    """A single hidden layer never relays: the corrected relay (and any scale, which has no lower layer to act on)
    gives the plain fit's state hash and predictions, although the returned motions were corrected."""
    plain, _, orders, y = fit(widths=(12,))
    for scale in (1., 4.):
        signed, instrumentation, _, _ = fit('signed', scale=scale, widths=(12,))
        assert signed.state_hash() == plain.state_hash()
        assert np.array_equal(signed.predict_orders(orders), plain.predict_orders(orders))
        summary = instrumentation.summary()
        layer = summary['relay_layers'][0]
        assert not layer['relays_to_layer_below'] and layer['vote_scale'] == 1.
        assert layer['repulsions'] > 0 and layer['returned_motions_corrected'] == layer['repulsions']
        assert summary['instrumentation_removed'] and summary['rng_unchanged'] and summary['intervention_consistent']


def test_the_printed_mode_reproduces_the_unmodified_two_layer_network_exactly():
    plain, _, orders, y = fit()
    printed, instrumentation, _, _ = fit('printed')
    assert printed.state_hash() == plain.state_hash()
    assert np.array_equal(printed.predict_orders(orders), plain.predict_orders(orders))
    summary = instrumentation.summary()
    assert summary['intervention_consistent'] and summary['instrumentation_removed'] and summary['rng_unchanged']
    assert all(entry['returned_motions_corrected'] == 0 for entry in summary['relay_layers'])
    assert summary['relay_layers'][1]['repulsions'] > 0 and summary['relay_layers'][1]['verified_repulsions'] > 0


# ----------------------------------------------------------------------------- an independent replica

def replica_backward(net, fwd, mll, relay):
    """SortFlowHybridNetwork.backward_propagate on the protocol's all-sort, 'mean' path, line for line, with the relay as
    its single change point (adapted from verification/v1/relay_shadow_run.replica_backward)."""
    layer = net.graph.vertex_list[net.id + '_ly' + str(net.graph.num_vertices - 1)]
    if net.last_layer_update:
        for key in layer.graph.vertex_list:
            layer.graph.vertex_list[key].apply_motion()
        layer.update_index_matrix()
    else:
        for key in layer.graph.vertex_list:
            vertex = layer.graph.vertex_list[key]
            vertex.clean_motion_accumulation(vertex.adjacency_list)
    first_layer = False
    for hidden_iter in range(net.graph.num_vertices - 1):
        first_layer = hidden_iter == net.graph.num_vertices - 2
        name = net.id + '_ly' + str(net.graph.num_vertices - 2 - hidden_iter)
        layer = net.graph.vertex_list[name]
        adjustment = (1 / float(net.learning_rate)) * (layer.graph.num_vertices / 2)
        following, cumulative, corrected = [], {}, {}
        sorted_adj = layer.graph.vertex_list[list(layer.graph.vertex_list)[0]]._sorted_adj
        for index, point in enumerate(mll):
            data = fwd[name][index][0]
            for rank, value in enumerate(point[1]):
                if rank >= int(np.ceil(net.ratio_data_backprop * len(point[0]))):
                    break
                vertex = layer.graph.vertex_list[point[0][rank]]
                magnitude = value / adjustment
                motion = vertex.accumulate_motion(data, magnitude=magnitude)
                if first_layer:
                    continue
                if abs(magnitude) > 0:
                    corrected.setdefault(index, []).append(rank)
                if relay == 'signed' and magnitude < 0:
                    motion = -((vertex.len_list - 1 - 2 * vertex.index_list) - motion)
                cumulative.setdefault(index, []).append(motion[vertex._adj_sort_index_list])
        for index in cumulative:
            accepted = corrected[index] if index in corrected else range(len(cumulative[index]))
            average = np.mean(np.asarray([cumulative[index][i] for i in accepted]), axis=0)
            average = (average.shape[0] * net.motion_normalization_mult) * average / np.max(np.abs(average) + 0.00001)
            order = score_order(-np.abs(average), sorted_adj)
            following.append([sorted_adj[order].tolist(), average[order]])
        mll = following
        for key in layer.graph.vertex_list:
            layer.graph.vertex_list[key].apply_motion()
        layer.update_index_matrix()
    net.num_of_epochs += 1
    net.adapt_learning_rate()
    return mll


class ReplicaNetwork(SortFlowHybridNetwork):
    relay = 'signed'

    def backward_propagate(self, forward_input_backprop_all_data, motion_last_layer):
        return replica_backward(self, forward_input_backprop_all_data, motion_last_layer, self.relay)


@pytest.mark.parametrize('relay', ['printed', 'signed'])
def test_the_wrapper_equals_an_independent_replica_of_backward_propagate(relay):
    net, orders, y = small_network()
    replica = copy.deepcopy(net)
    replica.network_.__class__ = ReplicaNetwork
    replica.network_.relay = relay
    replica.train_initialized(orders, y)
    wrapped, _, _, _ = fit(relay)
    assert wrapped.state_hash() == replica.state_hash()
    plain = fit()[0]
    assert (wrapped.state_hash() == plain.state_hash()) is (relay == 'printed')


# ----------------------------------------------------------------------------- the vote scale

def test_the_scale_multiplies_the_lower_layers_votes_only():
    signed, unscaled, orders, y = fit('signed')
    scaled, instrumentation, _, _ = fit('signed', scale=8.)
    summary = instrumentation.summary()
    relay = {entry['layer']: entry for entry in summary['relay_layers']}
    assert relay[0]['vote_scale'] == 8. and relay[1]['vote_scale'] == 1.
    assert relay[0]['scaled_votes'] == relay[0]['attractions'] + relay[0]['repulsions'] > 0
    assert relay[0]['scaled_vote_mass'] == pytest.approx(8. * relay[0]['unscaled_vote_mass'], rel=1e-12)
    assert relay[1]['scaled_votes'] == 0 and summary['intervention_consistent']
    lower = next(entry for entry in summary['layers'] if entry['layer'] == 0)
    assert lower['mean_vote_mass'] * lower['updates'] == pytest.approx(relay[0]['scaled_vote_mass'], rel=1e-9)
    assert scaled.state_hash() != signed.state_hash()
    unit = fit('signed', scale=1.)[0]                                   # scale 1 installs nothing and changes nothing
    assert unit.state_hash() == signed.state_hash()
    moved = {entry['layer']: entry['changed_share'] for entry in summary['layers']}
    before = {entry['layer']: entry['changed_share'] for entry in unscaled.summary()['layers']}
    assert moved[0] >= before[0]


def test_the_threshold_record_counts_voted_and_cleared_filter_batches():
    record = sr.ThresholdLayerRecord(0, 'hidden_0', filters=3, vocabulary=16)
    assert record.threshold == pytest.approx(1 / 15)
    record.add(moved=False, displacement=0, votes=0, mass=0., solved=0)
    record.add(moved=False, displacement=0, votes=2, mass=.06, solved=0)          # below 1/15
    record.add(moved=True, displacement=4, votes=1, mass=1 / 15 + 1e-9, solved=0)  # at the threshold
    record.close_batch()
    summary = record.summary()
    assert (summary['voted_updates'], summary['cleared_updates']) == (2, 1)
    assert summary['cleared_share_of_voted'] == .5 and summary['movement_threshold'] == pytest.approx(1 / 15)
    assert summary['mean_threshold_units_voted'] == pytest.approx((.06 * 15 + 1 + 15e-9) / 2)
    empty = sr.ThresholdLayerRecord(0, 'hidden_0', filters=3, vocabulary=16).summary()
    assert empty['voted_updates'] == 0 and empty['cleared_share_of_voted'] is None


def test_below_the_threshold_the_prior_dictates_the_order():
    """The core's order never changes while a filter's batch vote mass is below 1 / (V - 1); just above it, one vote
    that sends the first item to the end and the second to the front moves it."""
    rng = np.random.RandomState(9)
    for _ in range(400):
        size = int(rng.randint(3, 18))
        items = [str(i) for i in range(1, size + 1)]
        vertex = Vertex('v', list(rng.permutation(items)), 'sort', tiny_config())
        start = list(vertex.adjacency_list)
        budget = (1 / (size - 1)) * float(rng.uniform(.05, .999))
        weights = rng.dirichlet(np.ones(int(rng.randint(1, 5)))) * budget
        for weight in weights:
            vertex.accumulate_motion(list(rng.permutation(items)), magnitude=float(weight) * rng.choice([-1, 1]))
        assert vertex.compute_adj_list_with_permutation() == start
    for size in (4, 16, 64):
        items = [str(i) for i in range(1, size + 1)]
        target = [items[1]] + items[2:] + [items[0]]            # item 0 to the end, item 1 to the front
        for factor, moves in ((.99, False), (1.01, True)):
            vertex = Vertex('v', list(items), 'sort', tiny_config())
            vertex.accumulate_motion(target, magnitude=factor / (size - 1))
            assert (vertex.compute_adj_list_with_permutation() != items) is moves


# ----------------------------------------------------------------------------- installation

def test_the_instrumentation_is_removed_and_counts_every_call():
    net, instrumentation, _, _ = fit('signed', scale=2.)
    assert instrumentation.instrumented() == []
    for graph in instrumentation._graphs():
        for vertex in ur.sort_vertices(graph):
            assert 'accumulate_motion' not in vars(vertex) and 'apply_motion' not in vars(vertex)
    summary = instrumentation.summary()
    assert summary['relay'] == 'signed' and summary['lower_vote_scale'] == 2.
    for entry in summary['relay_layers']:
        assert entry['calls'] == entry['attractions'] + entry['repulsions'] + entry['zero_votes']
        assert entry['verified_attractions'] == min(sr.VERIFY_FIRST, entry['attractions'])
        assert entry['verified_repulsions'] == min(sr.VERIFY_FIRST, entry['repulsions'])


def test_the_wrappers_survive_a_deepcopy_bound_to_the_copy():
    vertex = Vertex('v', ['1', '2', '3'], 'sort', tiny_config())
    record = sr.RelayLayerRecord(0, 'hidden_0', relays=False, scale=2.)
    counter = ur.VoteCounter(vertex)
    vertex.accumulate_motion = sr.RelayedMotion(vertex, sr.ScaledVote(vertex, counter, 2., record), 'signed', record)
    duplicate = copy.deepcopy(vertex)
    assert duplicate.accumulate_motion.vertex is duplicate and duplicate.accumulate_motion.inner.vertex is duplicate
    assert duplicate.accumulate_motion.inner.inner.vertex is duplicate
    assert duplicate.accumulate_motion.record is record                  # the record is shared, never copied
    duplicate.accumulate_motion(['2', '1', '3'], magnitude=.25)
    assert duplicate.accumulate_motion.inner.inner.mass == .5 and counter.mass == 0.


def test_installing_needs_a_known_relay_a_positive_scale_and_an_untrained_network():
    with pytest.raises(sr.RelayError):
        sr.RelayInstrumentation(small_network()[0], relay='weighted')
    for scale in (0., -1., float('inf')):
        with pytest.raises(sr.RelayError):
            sr.RelayInstrumentation(small_network()[0], relay='signed', lower_vote_scale=scale)
    with pytest.raises(ur.InterventionError):
        sr.RelayInstrumentation(fit(iterations=1)[0])
    net = small_network()[0]
    first = sr.RelayInstrumentation(net, relay='signed')
    first._install()
    with pytest.raises(sr.RelayError, match='already carries'):
        sr.RelayInstrumentation(net, relay='signed')._install()
    first._remove()
    assert first.instrumented() == []
