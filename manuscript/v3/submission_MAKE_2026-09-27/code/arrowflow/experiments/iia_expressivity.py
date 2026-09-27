"""IIA-violation index vs. expressivity (load-bearing test of the Arrow connection).

For each (dataset, polynomial degree) we:
  1. Encode the data (StandardScaler -> random projection -> argsort), exactly as
     ArrowFlow's projection pipeline, for several views/seeds.
  2. Measure the *IIA-violation index* of the learning rule's positional (Borda)
     aggregation on the encoding the hidden filters receive: the fraction of item
     pairs {a,b} on which the Borda consensus order disagrees with the
     pairwise-majority (Condorcet) order. Pairwise majority satisfies IIA by
     construction; Borda does not, so this disagreement rate is an empirical,
     in-[0,1] IIA-violation index of the trained aggregation.
  3. Train the real ArrowFlow ensemble (authentic code) at depth 1 and depth 2.
  4. Train a linear probe (logistic regression on the ordinal position vectors of
     the same encodings) as an IIA-respecting / linear reference.

We then correlate the IIA-violation index with:
  - nonlinearity captured = linear-probe error - ArrowFlow error
  - depth benefit        = error(1 layer) - error(2 layers)
both across (dataset, degree) cells.

Small, CPU-only, real UCI datasets; safe for limited hardware.
"""
import os, sys, json, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/ozgur/Desktop/Lechler/Code/Playground/sortflow/arrowflow_repo')

import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

import torch as _torch
_torch.cuda.is_available = lambda: False
import arrowflow.arrowflow as _af
_af.device = 'cpu'
import copy
from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config
from arrowflow.arrowflow import SortFlowHybridNetwork


# ----------------------------- IIA-violation index -----------------------------
def positions(perms):
    """perms[i] lists item ids in argsort order; return pos[i, item] = rank."""
    n, e = perms.shape
    pos = np.empty((n, e), dtype=np.int32)
    cols = np.arange(e)
    for i in range(n):
        pos[i, perms[i]] = cols
    return pos


def iia_violation_index(pos):
    """Fraction of item pairs where Borda (mean position) order disagrees with
    pairwise-majority order. Pairwise majority is IIA-respecting; Borda is not."""
    n, e = pos.shape
    meanpos = pos.mean(axis=0)
    viol = tot = 0
    for a in range(e):
        pa = pos[:, a]
        for b in range(a + 1, e):
            pb = pos[:, b]
            margin = np.count_nonzero(pa < pb) - np.count_nonzero(pa > pb)
            borda = meanpos[b] - meanpos[a]  # >0 => a ranked before b
            if margin == 0 or abs(borda) < 1e-9:
                continue
            tot += 1
            if (margin > 0) != (borda > 0):
                viol += 1
    return viol / tot if tot else 0.0


# ----------------------------- encoding + training -----------------------------
def encode(X_tr_poly, X_te_poly, embed_dim, seed):
    rng = np.random.RandomState(seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr_poly)
    Xte = sc.transform(X_te_poly)
    W = rng.randn(Xtr.shape[1], embed_dim)
    ptr = np.argsort(Xtr @ W, axis=1)
    pte = np.argsort(Xte @ W, axis=1)
    return ptr, pte


def train_arrowflow_view(ptr, ytr, pte, n_classes, no_filters, layer_types,
                         iters, lr, llu, seed):
    """Replicates the projection worker body but keeps nothing fragile; returns preds."""
    embed_dim = ptr.shape[1]
    adj_list_input = [str(i + 1) for i in range(embed_dim)]
    data_train = [[list(map(str, ptr[i].astype(int) + 1)), str(int(ytr[i])), 1]
                  for i in range(len(ptr))]
    data_test = [[list(map(str, pte[i].astype(int) + 1)), str(0), 1]
                 for i in range(len(pte))]
    cfg = ArrowFlowConfig(no_of_filters=no_filters, layer_types=layer_types,
                          no_of_iters=iters, moe_no_of_networks=1,
                          no_of_embedding_dim=embed_dim, learning_rate=lr,
                          last_layer_update=llu, verbose=0)
    sf = _build_sortnet_config(cfg, n_classes)
    net = SortFlowHybridNetwork('iia_v%d' % seed, adj_list_input, n_classes,
                                'iia_exp', sf)
    net.train([data_train, data_test], sf)
    net.graph = copy.deepcopy(net.optimal_model)
    _, preds = net.evaluate(data_test, 'supervised', 'classification')
    return np.array(preds).astype(int)


def majority(pred_list, n_classes):
    P = np.array(pred_list)
    return np.array([np.bincount(P[:, i], minlength=n_classes).argmax()
                     for i in range(P.shape[1])])


def linear_probe(ptr, ytr, pte):
    """Logistic regression on ordinal position vectors (a linear, IIA-respecting reference)."""
    Xtr = positions(ptr).astype(float)
    Xte = positions(pte).astype(float)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


# ----------------------------------- run --------------------------------------
DATASETS = [('Iris', load_iris), ('Wine', load_wine),
            ('Breast C.', load_breast_cancer), ('Digits', load_digits)]
DEGREES = [1, 2, 3]
EMBED = 24
N_VIEWS = 5
ITERS = 100
LR = 0.1
BASE_SEED = 42

def main():
    rows = []
    for dname, load in DATASETS:
        d = load(); X, y = d.data, d.target.astype(int)
        n_classes = len(np.unique(y))
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        for deg in DEGREES:
            if deg > 1:
                poly = PolynomialFeatures(degree=deg)
                Xtr_p = poly.fit_transform(Xtr); Xte_p = poly.transform(Xte)
            else:
                Xtr_p, Xte_p = Xtr, Xte
            iias, lin_preds, af1_preds, af2_preds = [], [], [], []
            for v in range(N_VIEWS):
                seed = BASE_SEED + v * 9999
                ptr, pte = encode(Xtr_p, Xte_p, EMBED, seed)
                iias.append(iia_violation_index(positions(ptr)))
                lin_preds.append(linear_probe(ptr, ytr, pte))
                af1_preds.append(train_arrowflow_view(
                    ptr, ytr, pte, n_classes, [128], ['sort', 'sort'],
                    ITERS, LR, False, seed))
                af2_preds.append(train_arrowflow_view(
                    ptr, ytr, pte, n_classes, [128, 64], ['sort', 'sort', 'sort'],
                    ITERS, LR, False, seed))
            err = lambda preds: float(np.mean(majority(preds, n_classes) != yte))
            I = float(np.mean(iias))
            e_lin, e_af1, e_af2 = err(lin_preds), err(af1_preds), err(af2_preds)
            e_af = min(e_af1, e_af2)
            row = dict(dataset=dname, degree=deg, n_classes=n_classes,
                       iia=round(I, 4), err_linear=round(e_lin, 4),
                       err_af1=round(e_af1, 4), err_af2=round(e_af2, 4),
                       err_af=round(e_af, 4),
                       nonlin_captured=round(e_lin - e_af, 4),
                       depth_benefit=round(e_af1 - e_af2, 4))
            rows.append(row)
            print(f"{dname:10s} deg={deg}  I={I:.3f}  lin={e_lin:.3f}  "
                  f"af1={e_af1:.3f} af2={e_af2:.3f}  nonlin={e_lin-e_af:+.3f}  "
                  f"depthben={e_af1-e_af2:+.3f}", flush=True)

    # correlations
    import numpy as _np
    def corr(xs, ys):
        xs, ys = _np.array(xs, float), _np.array(ys, float)
        if _np.std(xs) < 1e-9 or _np.std(ys) < 1e-9:
            return float('nan')
        return float(_np.corrcoef(xs, ys)[0, 1])
    def spearman(xs, ys):
        from scipy.stats import spearmanr
        return float(spearmanr(xs, ys).correlation)
    I = [r['iia'] for r in rows]
    nonlin = [r['nonlin_captured'] for r in rows]
    depthben = [r['depth_benefit'] for r in rows]
    out = dict(rows=rows,
               n=len(rows),
               pearson_I_nonlin=round(corr(I, nonlin), 3),
               spearman_I_nonlin=round(spearman(I, nonlin), 3),
               pearson_I_depthben=round(corr(I, depthben), 3),
               spearman_I_depthben=round(spearman(I, depthben), 3))
    json.dump(out, open('/tmp/iia_results.json', 'w'), indent=2)
    print("\n=== CORRELATIONS (n=%d cells) ===" % len(rows))
    print(f"  IIA vs nonlinearity-captured : Pearson {out['pearson_I_nonlin']:+.3f}  "
          f"Spearman {out['spearman_I_nonlin']:+.3f}")
    print(f"  IIA vs depth-benefit         : Pearson {out['pearson_I_depthben']:+.3f}  "
          f"Spearman {out['spearman_I_depthben']:+.3f}")
    print("results -> /tmp/iia_results.json")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print("total %.1fs" % (time.time() - t0))
