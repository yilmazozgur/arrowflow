"""The sandbox's rank -> tensor conversion, installed on a live core network.

arrowflow/arrowflow.py is sealed by every published run and is NOT edited. As the signed relay does, this module
builds a core SortFlowHybridNetwork with layer types ['tensor', 'sort'] and replaces, on that instance only, the
method that turns the rank layer's returned motion into a tensor update (SortFlowHybridNetwork.update_tensor_layer).
Everything else is the core's own: the sequential forward pass, the descending argsort at the tensor -> rank interface,
the output layer's attraction vote, its gate and weighted Borda reorder, the Adam optimizer and its schedule, the
training loop and the validation checkpoint.

The core's motion for coordinate i is m_i = (its position in the input ranking) - (its position in the true class's
filter), and the input ranking is DESCENDING (position 0 holds the largest output), so a positive m_i asks for a LARGER
value. Modes:
    frozen           no tensor update (the rank layer still trains)
    core_original    the core's own conversion, verbatim: L1 loss between h and h - lr m |h|. The target is not
                     detached, so the loss is lr |m| |h| and its gradient lr |m| sign(h) carries NO sign of the motion:
                     it only pulls displaced coordinates toward zero (verified: identical gradient when every m flips)
    core_detached    the same formula with the target detached (its evident intent): a step of -sign(m), which for
                     the descending interface moves a coordinate the wrong way
    core_detached_flipped  detached and sign-flipped: a step of +sign(m), the minimal repair
    target           the sandbox's conversion: desired position pos - beta m (the true filter's position), minus
                     contrast times the displacement toward the nearest wrong class's filter; the value currently at
                     the desired position is the regression target
    target_scrambled the target signal with its coordinates shuffled in every example
    python -m experiments.tensor_rank_sandbox.core_hybrid check
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''           # CPU only, as every run of the paper (the core picks cuda when visible)
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arrowflow.arrowflow import SortFlowHybridNetwork
from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config

MODES = ('frozen', 'core_original', 'core_detached', 'core_detached_flipped', 'target', 'target_scrambled')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def build(n_features, n_classes, *, V=64, iterations=2000, batch_size=32, lr_rank=.1, lr_tensor=.01, p_correct=.1,
          val_ratio=0., arch='core_mlp', seed=0):
    seed_all(seed)
    cfg = ArrowFlowConfig(no_of_filters=[V], layer_types=['tensor', 'sort'], no_of_iters=iterations, batch_size=batch_size,
                          learning_rate=lr_rank, val_data_ratio=val_ratio, device='cpu', verbose=0,
                          evaluate_train_data=False, last_layer_update=True,
                          change_probability_when_decision_correct=p_correct, lr_tensor=lr_tensor)
    config = _build_sortnet_config(cfg, n_classes)
    net = SortFlowHybridNetwork('hyb', [str(i + 1) for i in range(n_features)], n_classes, 'hyb', config)
    tensor = net.graph.vertex_list['hyb_ly0']
    if arch == 'linear':
        linear = nn.Linear(n_features, V)
        with torch.no_grad():
            linear.weight.copy_(torch.randn(V, n_features) / np.sqrt(n_features))
            linear.bias.copy_(torch.randn(V) * .1)
        tensor.model_net = nn.Sequential(linear)
        tensor.define_optimizer(torch.optim.Adam(tensor.model_net.parameters(), lr=lr_tensor))
    elif arch != 'core_mlp':
        raise ValueError(arch)
    return net, config


def samples(X, y):
    return [[list(map(float, row)), str(int(label)), 1.] for row, label in zip(X, y)]


def descending_positions(h):
    order = np.argsort(-h, axis=1, kind='stable')
    pos = np.empty(h.shape, dtype=np.int64)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(h.shape[1]), h.shape), axis=1)
    return pos


def filter_positions(net):
    """(C, V): position of every tensor coordinate in each class's filter of the output layer."""
    out = net.graph.vertex_list['hyb_ly1']
    V = len(net.graph.vertex_list['hyb_ly0'].layer_vertices)
    P = np.empty((len(out.graph.vertex_list), V), dtype=np.int64)
    for key, vertex in out.graph.vertex_list.items():
        c = int(key.split('_')[-1])
        for position, item in enumerate(vertex.adjacency_list):
            P[c, int(item.split('_')[-1])] = position
    return P


def install(net, mode, *, beta=1., contrast=.5, gate='accepted', p_correct=.1, seed=0):
    """Replace this instance's update_tensor_layer; also stash each training batch's predictions and labels."""
    if mode not in MODES:
        raise ValueError(mode)
    rng = np.random.RandomState(seed)
    core_forward = net.forward_propagate
    core_update = net.update_tensor_layer
    net.tensor_log = []

    def forward_propagate(data, train_type, problem, evaluate_only=False):
        out = core_forward(data, train_type, problem, evaluate_only)
        if not evaluate_only:
            net._batch = (np.asarray([int(p) for p in out[3]]), np.asarray([int(d[1]) for d in data]))
        return out

    def update_tensor_layer(first_layer_skip_gradient, forward_input_backprop_all_data, layer, layer_name,
                            motion_last_layer):
        if mode == 'core_original':
            return core_update(first_layer_skip_gradient, forward_input_backprop_all_data, layer, layer_name,
                               motion_last_layer)
        entries = forward_input_backprop_all_data[layer_name]
        h = torch.cat([entry[0][1] for entry in entries], dim=0)                   # (n, V), graph-connected
        motion, accepted_index = net.convert_hybrid_gradient_for_backprop(motion_last_layer)
        if len(entries) != len(motion_last_layer) or len(accepted_index) != len(entries):
            raise RuntimeError('motions and tensor outputs are not aligned one per example')
        pred, y = net._batch
        if mode == 'frozen':
            layer.optimizer_hybrid.zero_grad()
            return []
        if gate == 'accepted':
            keep = (pred != y) | (rng.rand(len(y)) < p_correct)
        else:
            keep = np.ones(len(y), dtype=bool)
        hv = h.detach().cpu().numpy().astype(float)
        if mode in ('core_detached', 'core_detached_flipped'):
            m = torch.tensor(motion, dtype=h.dtype) * (1. if mode == 'core_detached_flipped' else -1.)
            expected = (h + layer.optimizer_hybrid.param_groups[-1]['lr'] * m * torch.abs(h)).detach()
            idx = torch.tensor(np.flatnonzero(keep))
            if len(idx):
                F.l1_loss(h[idx], expected[idx]).backward()
        else:
            V = hv.shape[1]
            pos = descending_positions(hv)
            desired = pos - beta * motion                                           # toward the true class filter
            if contrast > 0:
                P = filter_positions(net)
                D = np.abs(pos[:, None, :] - P[None, :, :]).sum(-1).astype(float)
                D[np.arange(len(y)), y] = np.inf
                desired = desired - contrast * (P[np.argmin(D, axis=1)] - pos)       # away from the nearest wrong filter
            desired = np.clip(desired, 0, V - 1)
            sorted_desc = -np.sort(-hv, axis=1)
            lo = np.floor(desired).astype(np.int64)
            hi = np.minimum(lo + 1, V - 1)
            w = desired - lo
            target = np.take_along_axis(sorted_desc, lo, 1) * (1 - w) + np.take_along_axis(sorted_desc, hi, 1) * w
            g = hv - target
            if mode == 'target_scrambled':
                g = np.take_along_axis(g, np.argsort(rng.rand(*g.shape), axis=1), axis=1)
            g[~keep] = 0.
            n = max(1, int(keep.sum()))
            h.backward(gradient=torch.tensor(g / n, dtype=h.dtype))
        net.tensor_log.append(int(keep.sum()))
        layer.optimizer_hybrid.step()
        layer.scheduler.step()
        layer.optimizer_hybrid.zero_grad()
        return []

    net.forward_propagate = forward_propagate
    net.update_tensor_layer = update_tensor_layer
    return net


def fit_predict(Xtr, ytr, Xte, mode, *, seed=0, **kwargs):
    install_keys = ('beta', 'contrast', 'gate')
    build_kwargs = {k: v for k, v in kwargs.items() if k not in install_keys}
    net, config = build(Xtr.shape[1], int(ytr.max()) + 1, seed=seed, **build_kwargs)
    install(net, mode, p_correct=build_kwargs.get('p_correct', .1), seed=seed,
            **{k: v for k, v in kwargs.items() if k in install_keys})
    train, test = samples(Xtr, ytr), samples(Xte, np.zeros(len(Xte), dtype=int))
    net.train([train, test], config)
    _, prediction = net.evaluate(test, 'supervised', 'classification')
    return np.asarray([int(p) for p in prediction]), net


def check():
    """Sign test inside the core: one update in each mode, from the same state, and the mean footrule distance between
    the batch's input rankings and their true class filters before and after (the rank layer frozen for the test)."""
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)[:256]
    y = y[:256]
    for arch in ('core_mlp', 'linear'):
        for mode in ('core_original', 'core_detached', 'core_detached_flipped', 'target'):
            net, config = build(X.shape[1], 10, arch=arch, seed=3, lr_tensor=.05)
            net.last_layer_update = False                                           # keep the filters fixed here
            install(net, mode, gate='all', contrast=0., seed=3)
            batch = samples(X, y)

            def distance():
                h = torch.cat([net.graph.vertex_list['hyb_ly0'](torch.tensor(-np.asarray([r[0]]), dtype=torch.float32))
                               for r in batch]).detach().numpy()
                pos = descending_positions(h)
                return np.abs(pos - filter_positions(net)[y]).sum(1).mean()
            before = distance()
            for _ in range(5):
                net.update_network(batch, None, 'supervised', 'classification')
            after = distance()
            print(f'{arch:9s} {mode:22s} mean distance to the true class filter: {before:6.1f} -> {after:6.1f} '
                  f'({"closer" if after < before else "FARTHER"})')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['check']:
        check()
