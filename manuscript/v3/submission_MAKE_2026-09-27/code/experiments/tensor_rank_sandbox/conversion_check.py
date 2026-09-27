"""The two defects of the ArrowFlow code's own conversion for hybrid networks, checked and recorded (Section S6.2).

(1) Gradient identity: SortFlowHybridNetwork.update_tensor_layer takes F.l1_loss(h, h - lr * m * |h|) without detaching the
    target, so the loss is lr |m| |h| and its gradient is lr |m| sign(h) / numel: the same whatever the sign of every motion.
(2) In-core sign test: from one seeded state and one batch of Digits (the first 256 standardized rows), five updates of each
    conversion with the class filters held fixed; the mean footrule distance between the batch's input rankings and their true
    class filters before and after. core_hybrid.check prints the same test; this module records it.

    python -m experiments.tensor_rank_sandbox.conversion_check      writes runs/conversion_check.json
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .core_hybrid import build, descending_positions, filter_positions, install, samples

MODES = ('core_original', 'core_detached', 'core_detached_flipped', 'target')
OUT = Path(__file__).with_name('runs')/'conversion_check.json'


def gradient_identity():
    torch.manual_seed(0)
    h0 = torch.randn(4, 6)
    m = torch.randint(-5, 6, (4, 6)).to(torch.float32)

    def core_grad(motion):
        h = h0.clone().requires_grad_(True)
        F.l1_loss(h, h - 0.01 * motion * torch.abs(h)).backward()       # verbatim: the target is not detached
        return h.grad
    return {'identical_when_every_motion_flips': bool(torch.allclose(core_grad(m), core_grad(-m))),
            'equals_lr_abs_m_sign_h_over_numel': bool(torch.allclose(core_grad(m), 0.01 * m.abs() * torch.sign(h0) / h0.numel()))}


def sign_test():
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)[:256]
    y = y[:256]
    out = {}
    for arch in ('core_mlp', 'linear'):
        for mode in MODES:
            net, _ = build(X.shape[1], 10, arch=arch, seed=3, lr_tensor=.05)
            net.last_layer_update = False
            install(net, mode, gate='all', contrast=0., seed=3)
            batch = samples(X, y)
            layer = net.graph.vertex_list['hyb_ly0']

            def distance():
                with torch.no_grad():
                    h = layer(torch.tensor(-X, dtype=torch.float32)).numpy()
                return float(np.abs(descending_positions(h) - filter_positions(net)[y]).sum(1).mean())
            before = distance()
            for _ in range(5):
                net.update_network(batch, None, 'supervised', 'classification')
            out[f'{arch}/{mode}'] = {'before': before, 'after': distance()}
    return out


def main():
    record = {'gradient_identity': gradient_identity(), 'sign_test': sign_test(),
              'setting': 'digits rows 0-255, standardized; seed 3; lr_tensor 0.05; five updates; class filters fixed; gate all; '
                         'no away term'}
    OUT.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
