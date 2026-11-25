"""Exploratory, run 2026-09-23 BEFORE the study was designed (see PREREGISTRATION.md, section 2): the author's original split (42 train / 35 test rows) with the original classifier set,
under four input representations. Classical models only; no ArrowFlow here."""
import sys, warnings
import numpy as np
from experiments.make_revision.artificial_ranks import original_rows
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
N = 8
rows = original_rows()
train = [(int(l), s) for l, sp, s in rows if sp == 'train']
test = [(int(l), s) for l, sp, s in rows if sp == 'test']

def seq_numeric(seq):            # the original transform_2_numpy_dataset: item-1 per position, zero-padded
    v = [float(i) - 1 for i in seq]
    return v + [0.0] * (N - len(v))

def seq_onehot(seq):             # item at each position, one-hot over 8 items + 'empty'
    out = np.zeros((N, N + 1))
    for p in range(N):
        out[p, seq[p] - 1 if p < len(seq) else N] = 1
    return out.ravel()

def pos_nan(seq):                # v3 features: relative position of each item, NaN if deleted
    v = np.full(N, np.nan)
    for p, it in enumerate(seq):
        v[it - 1] = p / (len(seq) - 1)
    return v

def completed(seq):              # author's convention: missing items appended at the end, ascending
    return list(seq) + sorted(set(range(1, N + 1)) - set(seq))

def pos_completed(seq):
    v = np.zeros(N)
    for p, it in enumerate(completed(seq)):
        v[it - 1] = p / (N - 1)
    return v

def build(fn, data):
    return np.array([fn(s) for _, s in data], dtype=float), np.array([l for l, _ in data])

def models():
    return [('kNN-1', KNeighborsClassifier(1)), ('kNN-3', KNeighborsClassifier(3)), ('kNN-5', KNeighborsClassifier(5)),
            ('LinearSVM', SVC(kernel='linear', C=0.025, random_state=42)), ('RBF-SVM', SVC(gamma=2, C=1, random_state=42)),
            ('GaussProc', GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)),
            ('DecTree', DecisionTreeClassifier(max_depth=5, random_state=42)),
            ('RF(orig)', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)),
            ('RF(500)', RandomForestClassifier(n_estimators=500, random_state=0)),
            ('SVC(default)', SVC())]

reps = {'sequence numeric (original)': seq_numeric, 'sequence one-hot': seq_onehot,
        'item position, NaN->train mean': pos_nan, 'item position, completed': pos_completed}
print(f'train {len(train)} rows, test {len(test)} rows; test error %')
header = f"{'model':14s}" + ''.join(f'{k[:22]:>25s}' for k in reps)
print(header)
res = {}
for name, _ in models():
    res[name] = []
for rname, fn in reps.items():
    Xtr, ytr = build(fn, train); Xte, yte = build(fn, test)
    if np.isnan(Xtr).any() or np.isnan(Xte).any():
        mu = np.nanmean(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), mu, Xtr); Xte = np.where(np.isnan(Xte), mu, Xte)
    for name, m in models():
        m.fit(Xtr, ytr)
        res[name].append(100 * np.mean(m.predict(Xte) != yte))
for name in res:
    print(f'{name:14s}' + ''.join(f'{e:25.1f}' for e in res[name]))

# footrule 1-NN on completed rankings (= L1 1-NN on the completed position vector)
def footrule(a, b):
    pa = {it: p for p, it in enumerate(a)}
    return sum(abs(p - pa[it]) for p, it in enumerate(b))
errs = 0
for l, s in test:
    d = [(footrule(completed(s), completed(t)), lt) for lt, t in train]
    errs += min(d)[1] != l
print(f'footrule 1-NN on completed rankings: {100 * errs / len(test):.1f}%')
# nearest clean prototype (class's first train row) by footrule on completed rankings
protos = {l: completed(s) for l, s in train[::6]}
errs = sum(min(protos, key=lambda c: footrule(completed(s), protos[c])) != l for l, s in test)
print(f'nearest clean prototype (footrule, completed): {100 * errs / len(test):.1f}%')
