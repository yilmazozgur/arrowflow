"""Tensor -> rank sandbox: a real-valued first layer whose argsort feeds a rank classification layer.

The rank layer holds one filter (a permutation of the V tensor outputs) per class and predicts the class whose filter is
nearest to the input ranking in footrule distance. It learns by ArrowFlow's output-layer rule (arrowflow.py, the
classification branch of the forward pass): only the true class's filter is attracted to the input, when the prediction
is wrong or, when it is right, with probability p_correct; the vote weight is 2 lr / C; the filter's current order
counts as one more voter, and the new order is the weighted Borda count.

The tensor layer (h = W phi(x) + b) is trained from the displacement the rank layer returns: each output coordinate i
should move from its position pos_i in the input ranking toward its position in the true class's filter. That position
displacement is converted into a tensor signal by one of three rules:
    target  the value that currently sits at the desired position (interpolated) is the regression target for h_i
    sign    only the direction survives (what the 2023 hybrid's L1 loss did in effect)
    linear  the displacement times the mean gap between sorted values
The ranking is ascending: position 0 holds the smallest h_i (as arrowflow.ranking.score_order), so moving a coordinate to
an earlier position means lowering its value.

Commands:  python -m experiments.tensor_rank_sandbox.sandbox check
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

DEFAULT = dict(V=32, phi='linear', bias=True, convert='target', eta=.03, beta=1., contrast=0., gate='accepted',
               train_tensor=True, train_rank=True, rank_lr=.1, p_correct=.1, rank_repel=False, init='random',
               epochs=60, batch=32, optimizer='sgd', row_norm=False, scramble=False, knn_k=5, eta_decay=False,
               val_ratio=0.)

DATASETS = {'iris': load_iris, 'wine': load_wine, 'breast_cancer': load_breast_cancer, 'digits': load_digits}


def load(name):
    X, y = DATASETS[name](return_X_y=True)
    return X.astype(float), np.unique(y, return_inverse=True)[1]


def positions(h):
    """Position of every coordinate in the ascending ranking of h (ties by coordinate index)."""
    order = np.argsort(h, axis=1, kind='stable')
    pos = np.empty(h.shape, dtype=np.int64)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(h.shape[1]), h.shape), axis=1)
    return pos


def footrule(pos, P):
    """(n, V) positions x (C, V) filter positions -> (n, C) footrule distances."""
    return np.abs(pos[:, None, :] - P[None, :, :]).sum(-1)


def rerank(score, tiebreak):
    """Positions of the items sorted by score, ties by tiebreak (the current position)."""
    order = np.lexsort((tiebreak, score))
    out = np.empty(len(order), dtype=np.int64)
    out[order] = np.arange(len(order))
    return out


class Features:
    def __init__(self, phi):
        self.phi = phi

    def fit(self, X):
        self.poly = PolynomialFeatures(2, include_bias=False) if self.phi == 'poly2' else None
        Z = self.poly.fit_transform(X) if self.poly else X
        self.scaler = StandardScaler().fit(Z)
        return self

    def transform(self, X):
        Z = self.poly.transform(X) if self.poly else X
        return self.scaler.transform(Z)


class TensorRank:
    def __init__(self, cfg, seed):
        self.cfg = {**DEFAULT, **cfg}
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ forward
    def scores(self, Z):
        return Z @ self.W.T + (self.b if self.cfg['bias'] else 0.)

    def rank_predict(self, pos):
        return np.argmin(footrule(pos, self.P), axis=1)

    # ------------------------------------------------------------------ the rank layer's rule
    def rank_update(self, pos, y, pred, accepted):
        C, V = self.P.shape
        a = 2 * self.cfg['rank_lr'] / C
        for c in range(C):
            votes = accepted & (y == c)
            num, den = self.P[c].astype(float), 1.
            if votes.any():
                num = num + a * pos[votes].sum(0)
                den += a * votes.sum()
            if self.cfg['rank_repel']:
                wrong = (pred == c) & (y != c)
                if wrong.any():
                    num = num + a * (V - 1 - pos[wrong]).sum(0)
                    den += a * wrong.sum()
            if den > 1.:
                self.P[c] = rerank(num / den, self.P[c])

    # ------------------------------------------------------------------ the rank -> tensor conversion
    def tensor_signal(self, h, pos, y, accepted):
        cfg, V = self.cfg, h.shape[1]
        desired = pos + cfg['beta'] * (self.P[y] - pos)
        if cfg['contrast'] > 0:
            D = footrule(pos, self.P).astype(float)
            D[np.arange(len(y)), y] = np.inf
            nearest_wrong = np.argmin(D, axis=1)
            desired = desired - cfg['contrast'] * (self.P[nearest_wrong] - pos)
        desired = np.clip(desired, 0, V - 1)
        if cfg['convert'] == 'target':
            sorted_h = np.sort(h, axis=1)
            lo = np.floor(desired).astype(np.int64)
            hi = np.minimum(lo + 1, V - 1)
            w = desired - lo
            target = (np.take_along_axis(sorted_h, lo, 1) * (1 - w) + np.take_along_axis(sorted_h, hi, 1) * w)
            g = h - target
        elif cfg['convert'] == 'sign':
            g = np.sign(pos - desired)
        elif cfg['convert'] == 'linear':
            gap = np.diff(np.sort(h, axis=1), axis=1).mean(1, keepdims=True)
            g = (pos - desired) * gap
        else:
            raise ValueError(cfg['convert'])
        if cfg['scramble']:
            g = np.take_along_axis(g, np.argsort(self.rng.rand(*g.shape), axis=1), axis=1)
        if cfg['gate'] == 'accepted':
            g = g * accepted[:, None]
        return g

    def tensor_step(self, Z, g):
        cfg = self.cfg
        n = max(1, int(np.any(g != 0, axis=1).sum()))
        gW, gb = g.T @ Z / n, g.sum(0) / n
        eta = cfg['eta'] * (self.eta_scale if cfg['eta_decay'] else 1.)
        if cfg['optimizer'] == 'adam':
            self.t += 1
            for name, grad in (('W', gW), ('b', gb)):
                m, v = self.adam[name]
                m[:] = .9 * m + .1 * grad
                v[:] = .999 * v + .001 * grad ** 2
                step = eta * (m / (1 - .9 ** self.t)) / (np.sqrt(v / (1 - .999 ** self.t)) + 1e-8)
                if name == 'W':
                    self.W -= step
                else:
                    self.b -= step
        else:
            self.W -= eta * gW
            self.b -= eta * gb
        if cfg['row_norm']:
            self.W /= np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-12

    # ------------------------------------------------------------------ fit and predict
    def fit(self, X, y):
        cfg = self.cfg
        X_val = y_val = None
        if cfg['val_ratio'] > 0:
            from sklearn.model_selection import train_test_split
            X, X_val, y, y_val = train_test_split(X, y, test_size=cfg['val_ratio'], stratify=y,
                                                  random_state=self.rng.randint(2 ** 31))
        self.features = Features(cfg['phi']).fit(X)
        Z = self.features.transform(X)
        C, V, d = int(y.max()) + 1, cfg['V'], Z.shape[1]
        self.W = self.rng.randn(V, d) / np.sqrt(d)
        self.b = self.rng.randn(V) * .1 if cfg['bias'] else np.zeros(V)
        self.adam = {'W': (np.zeros_like(self.W), np.zeros_like(self.W)), 'b': (np.zeros(V), np.zeros(V))}
        self.t = 0
        if cfg['init'] == 'class_borda':
            pos = positions(self.scores(Z))
            self.P = np.stack([rerank(pos[y == c].mean(0), np.arange(V)) for c in range(C)])
        else:
            self.P = np.stack([self.rng.permutation(V) for _ in range(C)])
        self.history = []
        best = None
        Zv = self.features.transform(X_val) if X_val is not None else None
        for epoch in range(cfg['epochs']):
            self.eta_scale = 1. - epoch / cfg['epochs']
            for batch in np.array_split(self.rng.permutation(len(y)), max(1, len(y) // cfg['batch'])):
                Zb, yb = Z[batch], y[batch]
                h = self.scores(Zb)
                pos = positions(h)
                pred = self.rank_predict(pos)
                accepted = (pred != yb) | (self.rng.rand(len(yb)) < cfg['p_correct'])
                g = self.tensor_signal(h, pos, yb, accepted) if cfg['train_tensor'] else None
                if cfg['train_rank']:
                    self.rank_update(pos, yb, pred, accepted)
                if g is not None:
                    self.tensor_step(Zb, g)
            self.history.append(float(np.mean(self.rank_predict(positions(self.scores(Z))) == y)))
            if Zv is not None:
                acc = float(np.mean(self.rank_predict(positions(self.scores(Zv))) == y_val))
                if best is None or acc >= best[0]:          # ties: the later epoch
                    best = (acc, epoch, self.W.copy(), self.b.copy(), self.P.copy())
        if best is not None:
            _, self.best_epoch, self.W, self.b, self.P = best
        self.train_pos, self.train_y = positions(self.scores(Z)), y
        return self

    def predict(self, X, readout='rank'):
        pos = positions(self.scores(self.features.transform(X)))
        if readout == 'rank':
            return self.rank_predict(pos)
        # footrule kNN on the tensor rankings of the training rows (uniform votes, lowest class on ties)
        out = []
        for start in range(0, len(pos), 256):
            D = np.abs(pos[start:start + 256, None, :] - self.train_pos[None, :, :]).sum(-1)
            nearest = np.argsort(D, axis=1, kind='stable')[:, :self.cfg['knn_k']]
            votes = np.apply_along_axis(np.bincount, 1, self.train_y[nearest], minlength=int(self.train_y.max()) + 1)
            out.append(np.argmax(votes, axis=1))
        return np.concatenate(out)


# ---------------------------------------------------------------------- evaluation protocol

def dev_test(name, test_size=.3, split_seed=0):
    X, y = load(name)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=split_seed)


def dev_cv_score(name, cfg, seeds=(0, 1), folds=3):
    """Mean accuracy of both readouts over stratified folds of the DEV portion only (the test portion is untouched)."""
    Xd, _, yd, _ = dev_test(name)
    rank_acc, knn_acc = [], []
    for s in seeds:
        for a, b in StratifiedKFold(folds, shuffle=True, random_state=100 + s).split(Xd, yd):
            m = TensorRank(cfg, seed=1000 * s + len(rank_acc)).fit(Xd[a], yd[a])
            rank_acc.append(np.mean(m.predict(Xd[b], 'rank') == yd[b]))
            knn_acc.append(np.mean(m.predict(Xd[b], 'knn') == yd[b]))
    return float(np.mean(rank_acc)), float(np.mean(knn_acc))


# ---------------------------------------------------------------------- self-checks

def check():
    rng = np.random.RandomState(0)
    h = rng.randn(5, 8)
    pos = positions(h)
    assert all(np.array_equal(np.sort(h[i])[pos[i]], h[i]) for i in range(5)), 'positions are not ascending ranks'
    P = np.stack([rng.permutation(8) for _ in range(3)])
    assert footrule(pos, P)[2, 1] == np.abs(pos[2] - P[1]).sum()
    # a single heavy vote moves a filter onto the input ranking
    m = TensorRank({'V': 8, 'rank_lr': 1e6}, seed=0)
    m.P = P.copy()
    m.rank_update(pos[:1], np.array([1]), np.array([0]), np.array([True]))
    assert np.array_equal(m.P[1], pos[0]), 'a dominant vote must copy the input ranking'
    # sign test: a small step along each conversion's signal moves inputs toward the true class filter
    for convert in ('target', 'sign', 'linear'):
        m = TensorRank({'V': 16, 'convert': convert, 'beta': 1.}, seed=1)
        Z = rng.randn(200, 6)
        m.W, m.b = rng.randn(16, 6), np.zeros(16)
        m.P = np.stack([rng.permutation(16) for _ in range(3)])
        y = rng.randint(3, size=200)
        h = m.scores(Z)
        pos = positions(h)
        before = footrule(pos, m.P)[np.arange(200), y].mean()
        g = m.tensor_signal(h, pos, y, np.ones(200, bool))
        scale = .05 if convert == 'sign' else .2
        after = footrule(positions(h - scale * g), m.P)[np.arange(200), y].mean()
        assert after < before, f'{convert}: the signal does not move inputs toward the true filter ({before} -> {after})'
        print(f'sign test {convert:6s}: mean distance to the true class filter {before:.1f} -> {after:.1f}')
    print('all checks passed')


if __name__ == '__main__':
    check()
