"""Hand-derived fixtures for MAKE batch-reset classification, not accuracy claims."""
import copy
import numpy as np
import pytest
from arrowflow.arrowflow import SortFlowHybridNetwork, Vertex
from arrowflow.config import SortNetConfig


def config(**kwargs):
    cfg = SortNetConfig()
    values = dict(no_of_filters=[3, 2], layer_types=['sort', 'sort'],
                  filter_rfs=[None, None], problem='classification', verbose=0,
                  learning_rate=1., multistep_lr=True, no_of_iters=1,
                  val_data_ratio=0, batch_size=None, evaluate_train_data=False,
                  ratio_data_backprop=1., change_probability_when_decision_correct=0.,
                  initial_filter_with_data=False, frequency_based_filter=False,
                  motion_normalization_mult=.125, device='cpu')
    values.update(kwargs)
    for k, v in values.items():
        setattr(cfg, k, v)
    return cfg


def set_order(vertex, order):
    """Configure existing Vertex through its real accumulator and update path."""
    pos = {item: i for i, item in enumerate(order)}
    vertex.permutation_matrix_accumulate[:] = 0
    for row, item in enumerate(vertex.adjacency_list):
        vertex.permutation_matrix_accumulate[row, pos[item]] = 1
    vertex.apply_motion()


def network(cfg=None):
    cfg = cfg or config()
    net = SortFlowHybridNetwork('toy', ['1', '2', '3', '4'], 2, '', cfg)
    for layer in net.graph.vertex_list.values():
        vocab = list(layer.adj_list_items)
        for i, vertex in enumerate(layer.graph.vertex_list.values()):
            set_order(vertex, vocab if i == 0 else vocab[::-1])
        layer.update_index_matrix()
    return net, cfg


def filters(net):
    return [[v.adjacency_list.copy() for v in l.graph.vertex_list.values()]
            for l in net.graph.vertex_list.values()]


def test_row_evidence_means_not_column_masses():
    v = Vertex('v', ['1', '2', '3'], 'sort', config())
    v.permutation_matrix_accumulate = np.array([[1., 0, 9], [0, 2, 0], [1, 0, 1]])
    # Row means: [1.8, 1, 1]; numeric ID breaks the last two's tie.
    assert v.compute_adj_list_with_permutation() == ['2', '3', '1']


def test_accumulator_ties_use_numeric_id_not_current_order():
    v = Vertex('v', ['10', '2', '1'], 'sort', config())
    v.permutation_matrix_accumulate = np.ones((3, 3))
    assert v.compute_adj_list_with_permutation() == ['1', '2', '10']


def test_batch_reset_reanchors_identity_to_new_order():
    v = Vertex('v', ['1', '2', '3', '4'], 'sort', config())
    v.accumulate_motion(['4', '3', '2', '1'], magnitude=2)
    v.apply_motion()
    assert v.adjacency_list == ['4', '3', '2', '1']
    np.testing.assert_array_equal(v.permutation_matrix_accumulate, np.eye(4))
    v.accumulate_motion(['1', '2', '3', '4'], magnitude=2)
    v.apply_motion()
    assert v.adjacency_list == ['1', '2', '3', '4']
    np.testing.assert_array_equal(v.permutation_matrix_accumulate, np.eye(4))


def test_disabled_correct_update_stops_hidden_updates():
    net, _ = network()
    before = filters(net)
    net.update_network([[['1', '2', '3', '4'], '0', 1]])
    assert filters(net) == before
    for layer in net.graph.vertex_list.values():
        for v in layer.graph.vertex_list.values():
            np.testing.assert_array_equal(v.permutation_matrix_accumulate, np.eye(v.len_list))


def test_zero_motion_is_finite_and_unchanged():
    v = Vertex('v', ['1', '2', '3', '4'], 'sort', config())
    np.testing.assert_array_equal(v.accumulate_motion(v.adjacency_list, magnitude=1), [0, 0, 0, 0])
    v.apply_motion()
    assert v.adjacency_list == ['1', '2', '3', '4']
    assert np.isfinite(v.permutation_matrix_accumulate).all()


def test_output_update_refreshes_prediction_cache():
    net, _ = network(config(no_of_filters=[2], layer_types=['sort']))
    data = [[['4', '3', '2', '1'], '0', 1]] * 4
    net.update_network(data)
    assert filters(net)[0][0] == ['4', '3', '2', '1']
    assert net.evaluate(data, 'supervised')[1] == [0] * 4


def test_train_without_validation_keeps_final_fitted_state():
    net, cfg = network(config(no_of_filters=[2], layer_types=['sort']))
    data = [[['4', '3', '2', '1'], '0', 1]] * 4
    net.train([data, data], cfg)
    assert filters(net)[0][0] == ['4', '3', '2', '1']


def test_evaluation_does_not_consume_training_rng():
    net, _ = network()
    state = copy.deepcopy(np.random.get_state())
    net.evaluate([[['1', '2', '3', '4'], '0', 1]], 'supervised')
    after = np.random.get_state()
    assert state[0] == after[0]
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


@pytest.mark.parametrize('fraction, expected', [(0, [False]*3), (1/3, [True, False, False]), (.34, [True, True, False]), (1, [True]*3)])
def test_top_fraction_selects_ceil_q_m_actual_filter_changes(fraction, expected):
    net, _ = network(config(ratio_data_backprop=fraction, last_layer_update=False))
    layer = net.graph.vertex_list['toy_ly0']
    for v in layer.graph.vertex_list.values():
        set_order(v, ['1', '2', '3', '4'])
    layer.update_index_matrix()
    before = filters(net)[0]
    net.backward_propagate({'toy_ly0': [[['4', '3', '2', '1'], '0', 1]]},
                           [[list(layer.graph.vertex_list), np.array([6., 5., 4.])]])
    assert [a != b for a, b in zip(before, filters(net)[0])] == expected


def test_frozen_head_preserves_output_but_allows_hidden_learning():
    net, _ = network(config(last_layer_update=False))
    before = filters(net)
    net.update_network([[['4', '3', '2', '1'], '0', 1]] * 8)
    after = filters(net)
    assert after[-1] == before[-1]
    assert after[0] != before[0]


def test_instance_learning_schedule_is_used():
    net, _ = network(config(multistep_lr=False))
    net.num_of_epochs = 1
    net.adapt_learning_rate()
    assert net.learning_rate == pytest.approx(.993)


@pytest.mark.parametrize('order', [['1', '1', '3', '4'], ['1', '2', '3'], ['1', '2', '3', '9']])
def test_full_permutation_required_at_network_input(order):
    net, _ = network()
    with pytest.raises(ValueError, match='permutation'):
        net.evaluate([[order, '0', 1]], 'supervised')


def test_instance_evaluate_train_data_controls_reported_error():
    net, _ = network(config(evaluate_train_data=True, last_layer_update=False, ratio_data_backprop=0))
    _, report = net.update_network([[['4', '3', '2', '1'], '0', 1]])
    assert report[0] == 1


def test_instance_motion_normalization_uses_input_vocabulary_length():
    net, _ = network(config(no_of_filters=[4, 3, 2], layer_types=['sort']*3,
                            filter_rfs=[None]*3, motion_normalization_mult=.75,
                            last_layer_update=False))
    mid = net.graph.vertex_list['toy_ly1']
    inp = list(mid.adj_list_items)[::-1]
    bp = {'toy_ly1': [[inp, '0', 1]],
          'toy_ly0': [[['4', '3', '2', '1'], '0', 1]]}
    # Observe the actual signal passed to the previous layer through its resulting votes.
    # Input-aligned mean [3,1,-1,-3] gets norm max = 4*.75*3/(3+1e-5); hidden vote exceeds the unit prior.
    net.backward_propagate(bp, [[list(mid.graph.vertex_list), np.array([6., 0., 0.])]])
    assert filters(net)[0][0] == ['4', '3', '2', '1']


@pytest.mark.parametrize('overrides', [dict(filter_rfs=[2, None]),
    dict(frequency_based_filter=True), dict(distance_computation_metric='l2'),
    dict(ratio_data_backprop=-.1), dict(learning_rate=0)])
def test_unsupported_sort_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        network(config(**overrides))


def test_initial_filters_must_be_full_permutations():
    with pytest.raises(ValueError, match='permutation'):
        network(config(initial_filter_with_data=True,
                       data_initial=[[['1', '1', '3', '4'], '0', 1]]*3))


def test_filter_distance_ties_use_numeric_filter_id():
    # Classification signal records the ranking of tied hidden filters in a two-layer network.
    net2, _ = network(config(no_of_filters=[20, 2]))
    layer2 = net2.graph.vertex_list['toy_ly0']
    for v in layer2.graph.vertex_list.values():
        set_order(v, ['1', '2', '3', '4'])
    layer2.update_index_matrix()
    _, bp, _, _ = net2.forward_propagate([[['1', '2', '3', '4'], '0', 0]], 'supervised', 'classification')
    assert bp['toy_ly1'][0][0] == ['toy_ly0_'+str(i) for i in range(20)]


def test_motion_ties_use_numeric_filter_id():
    net, _ = network(config(no_of_filters=[12, 2]))
    head = net.graph.vertex_list['toy_ly1']
    set_order(head.graph.vertex_list['toy_ly1_0'], list(head.adj_list_items)[::-1])
    head.update_index_matrix()
    _, _, motions, _ = net.forward_propagate([[['1', '2', '3', '4'], '0', 0]], 'supervised', 'classification')
    assert motions[0][0] == ['toy_ly0_'+str(i) for i in range(12)]


def test_index_tensor_uses_instance_device_not_module_global(monkeypatch):
    import arrowflow.arrowflow as implementation
    monkeypatch.setattr(implementation, 'device', 'meta')
    net, _ = network(config(device='cpu'))
    assert net.graph.vertex_list['toy_ly0'].index_matrix_gpu.device.type == 'cpu'


def test_checked_four_item_three_filter_two_class_trace(monkeypatch):
    net, _ = network(config(ratio_data_backprop=.5))
    hidden, head = list(net.graph.vertex_list.values())
    h0, h1, h2 = list(hidden.graph.vertex_list.values())
    c0, c1 = list(head.graph.vertex_list.values())
    set_order(h0, ['1', '2', '3', '4'])
    set_order(h1, ['4', '3', '2', '1'])
    set_order(h2, ['2', '1', '4', '3'])
    set_order(c0, ['toy_ly0_2', 'toy_ly0_1', 'toy_ly0_0'])
    set_order(c1, ['toy_ly0_0', 'toy_ly0_1', 'toy_ly0_2'])
    hidden.update_index_matrix()
    head.update_index_matrix()
    x = ['1', '3', '2', '4']
    assert [v.compute_distance(x)[1] for v in [h0, h1, h2]] == [2, 6, 6]
    _, bp, motion, prediction = net.forward_propagate([[x, '0', 1]], 'supervised', 'classification')
    assert prediction == [1]
    assert bp['toy_ly1'][0][0] == ['toy_ly0_0', 'toy_ly0_1', 'toy_ly0_2']
    assert motion[0][0] == ['toy_ly0_0', 'toy_ly0_2', 'toy_ly0_1']
    np.testing.assert_array_equal(motion[0][1], [-2, 2, 0])
    np.testing.assert_array_equal(c0.permutation_matrix_accumulate,
                                  [[1, 0, .5], [0, 1.5, 0], [.5, 0, 1]])
    captured = {}
    for v in [h0, h2]:
        apply = v.apply_motion
        def record_and_apply(v=v, apply=apply):
            captured[v.id] = v.permutation_matrix_accumulate.copy()
            return apply()
        monkeypatch.setattr(v, 'apply_motion', record_and_apply)
    net.backward_propagate(bp, motion)
    np.testing.assert_allclose(captured[h0.id],
        [[1, 0, 0, 4/3], [0, 7/3, 0, 0], [0, 0, 7/3, 0], [4/3, 0, 0, 1]])
    np.testing.assert_allclose(captured[h2.id],
        [[1, 0, 4/3, 0], [4/3, 1, 0, 0], [0, 0, 1, 4/3], [0, 4/3, 0, 1]])
    assert filters(net) == [[['2', '4', '1', '3'], ['4', '3', '2', '1'], ['1', '2', '3', '4']],
                           [['toy_ly0_2', 'toy_ly0_1', 'toy_ly0_0'], ['toy_ly0_0', 'toy_ly0_1', 'toy_ly0_2']]]
    assert net.evaluate([[x, '0', 1]], 'supervised')[1] == [0]
    before = filters(net)
    # Second batch is now correct and p_correct=0 gates every update.
    net.update_network([[x, '0', 1]])
    assert filters(net) == before
    for layer in [hidden, head]:
        for v in layer.graph.vertex_list.values():
            np.testing.assert_array_equal(v.permutation_matrix_accumulate, np.eye(v.len_list))


def test_cuda_tied_distances_match_cpu_when_available():
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable; stable GPU path not executed')
    cpu, _ = network(config(no_of_filters=[20, 2]))
    gpu = copy.deepcopy(cpu)
    gpu.device = 'cuda'
    for layer in gpu.graph.vertex_list.values():
        layer.device = 'cuda'
    data = [[['1', '2', '3', '4'], '0', 0], [['4', '3', '2', '1'], '1', 0]]
    a = cpu.forward_propagate(data, 'supervised', 'classification')
    b = gpu.forward_propagate(data, 'supervised', 'classification')
    assert a[1] == b[1]
    assert a[3] == b[3]


@pytest.mark.parametrize('counts', [[0, 2], [3, 1], []])
def test_classification_requires_positive_layers_and_one_filter_per_class(counts):
    with pytest.raises(ValueError):
        network(config(no_of_filters=counts))


def test_empty_native_vocabulary_is_rejected():
    with pytest.raises(ValueError, match='permutation'):
        SortFlowHybridNetwork('empty', [], 2, '', config())


@pytest.mark.parametrize('label, weight', [('2', 1), ('-1', 1), ('0', float('inf')), ('0', -1)])
def test_classification_rejects_invalid_labels_and_nonfinite_or_negative_weights(label, weight):
    net, _ = network()
    with pytest.raises(ValueError):
        net.evaluate([[['1', '2', '3', '4'], label, weight]], 'supervised')


@pytest.mark.parametrize('probability, draw, accepted', [
    (0., 0., False), (.5, .5, False), (.5, .25, True), (1., 0., True),
])
def test_correct_class_gate_accepts_only_draw_strictly_below_probability(
        monkeypatch, probability, draw, accepted):
    net, _ = network(config(change_probability_when_decision_correct=probability))
    head = net.graph.vertex_list['toy_ly1']
    correct = head.graph.vertex_list['toy_ly1_0']
    set_order(correct, ['toy_ly0_0', 'toy_ly0_2', 'toy_ly0_1'])
    head.update_index_matrix()
    before = filters(net)
    # Patch only the valid RNG endpoint/threshold; exercise real motion accumulation.
    monkeypatch.setattr(np.random, 'rand', lambda: draw)
    _, bp, motions, prediction = net.forward_propagate(
        [[['1', '2', '3', '4'], '0', 1]], 'supervised', 'classification')
    assert prediction == [0]
    if accepted:
        np.testing.assert_array_equal(correct.permutation_matrix_accumulate,
                                      [[1.5, 0, 0], [0, 1, .5], [0, .5, 1]])
        np.testing.assert_array_equal(motions[0][1], [-1, 1, 0])
    else:
        np.testing.assert_array_equal(correct.permutation_matrix_accumulate, np.eye(3))
        np.testing.assert_array_equal(motions[0][1], [0, 0, 0])
        net.backward_propagate(bp, motions)
        assert filters(net) == before
