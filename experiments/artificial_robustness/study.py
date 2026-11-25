"""Missing items at test time: ArrowFlow and classical models trained on clean ranking patterns (see PREREGISTRATION.md).

Commands (run from arrowflow_repo/):
    python -m experiments.artificial_robustness.study verify  --output DIR    V1-V5; fits only smoke-seeded models
    python -m experiments.artificial_robustness.study smoke   --output DIR    V6: the whole pipeline on smoke seeds
    python -m experiments.artificial_robustness.study freeze                  writes protocol.json (design + source hashes)
    python -m experiments.artificial_robustness.study run     --output DIR [--workers 14]
    python -m experiments.artificial_robustness.study original --output DIR   Part A (descriptive)
    python -m experiments.artificial_robustness.study analyse --output DIR
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
import warnings
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold
from arrowflow.ranking import inverse_positions, score_order
from experiments.make_revision.artificial_ranks import (N_CLASSES, corrupt, features, kendall_distance, original_rows,
                                                        prototypes, sequences_from_features)
from experiments.make_revision.bridge import AdaptiveMultiViewKNN, bridge_candidates
from experiments.make_revision.comparisons import (CONVENTIONAL_GRIDS, StableFootruleKNN, conventional_factory,
                                                   derive_seed)
from experiments.make_revision.evaluation import candidate_grid, canonical_json, config_id
from experiments.make_revision.knn_controls import UNTRAINED_MODEL, AdaptiveUntrainedKNN, control_candidates, \
    project_candidates
from experiments.make_revision.multiview import select_knn_readout
from experiments.make_revision.native_artificial import (UNTRAINED_KEYS, NativeMultiViewArrowFlowKNN,
                                                         NativeUntrainedMultiViewArrowFlowKNN, native_candidates,
                                                         native_orders)
from experiments.make_revision.secondary_studies import majority

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STUDY_ID = 'arrowflow-artificial-robustness-1'
PROTOCOL_FILE = HERE/'protocol.json'
N_VALUES = (8, 16)
LEVELS = {8: (0, 1, 2, 3), 16: (0, 2, 4, 6)}
SWAP_MAX = {8: 2, 16: 8}
TRAIN_PER_CLASS = 30
TEST_PER_CLASS = 100
REPLICATES = 20
DATA_SEED = 20260923
MODEL_SEED = 8129
SMOKE_DATA_SEED = 777001
SMOKE_MODEL_SEED = 777002
INNER_FOLDS = 3
BUDGET = {'candidate_budget': 24, 'candidate_seed': 41071}
CONVENTIONS = ('end', 'center', 'partial')
CLASSICAL = ('numeric_knn', 'svc_rbf', 'random_forest', 'mlp', 'gradient_boosting')
SEQ_CLASSICAL = ('numeric_knn', 'svc_rbf', 'random_forest', 'mlp')
NATIVE_MODELS = ('af_native', 'af_native_untrained')
MODELS = ('af_native', 'af_encoded', 'af_native_untrained', 'af_encoded_untrained', 'input_footrule_knn',
          *[f'pos_{m}' for m in CLASSICAL], *[f'seqnum_{m}' for m in SEQ_CLASSICAL],
          *[f'seqhot_{m}' for m in SEQ_CLASSICAL], 'prototype_reference')
PRIMARY_ARMS = ('af_native@partial', 'af_encoded')
PRIMARY_COMPARATOR = 'pos_numeric_knn'
SOURCES = ('experiments/artificial_robustness/study.py', 'experiments/artificial_robustness/PREREGISTRATION.md',
           'experiments/make_revision/artificial_ranks.py', 'experiments/make_revision/bridge.py',
           'experiments/make_revision/comparisons.py', 'experiments/make_revision/evaluation.py',
           'experiments/make_revision/knn_controls.py', 'experiments/make_revision/multiview.py',
           'experiments/make_revision/native_artificial.py', 'experiments/make_revision/secondary_studies.py',
           'experiments/make_revision/models.py', 'arrowflow/arrowflow.py', 'arrowflow/ranking.py',
           'arrowflow/benchmark.py', 'arrowflow/config.py')
# The 2023 log (test/Experiments_SortFlow2.txt, "ARTIFICIAL DATA - distort_prob=0, dim_expand_ratio=1"), V5's target.
LOG_2023 = {'kNN-1': 57.14285714285714, 'kNN-2': 51.42857142857142, 'kNN-3': 57.14285714285714,
            'kNN-4': 45.714285714285715, 'kNN-5': 48.57142857142858, 'kNN-6': 48.57142857142858,
            'LinearSVM': 65.71428571428571, 'RBF-SVM': 80.0, 'GaussProc': 60.0, 'DecTree': 60.0,
            'RandomForest': 45.714285714285715}


# ----------------------------------------------------------------------------- data

def draw_rows(n_items, k, per_class, rng):
    """per_class rows of every class: the prototype with k deletions at uniformly random positions, then s ~ U{0..S}
    inversion-adding adjacent swaps (artificial_ranks.corrupt), class by class."""
    reference = prototypes(n_items)
    sequences, labels, swaps = [], [], []
    for label in range(N_CLASSES):
        for _ in range(per_class):
            s = int(rng.randint(SWAP_MAX[n_items] + 1))
            sequences.append(corrupt(reference[label], k, s, rng))
            labels.append(label)
            swaps.append(s)
    return {'sequences': sequences, 'y': np.asarray(labels), 'swaps': np.asarray(swaps), 'deletions': int(k)}


def replicate_data(n_items, replicate, base_seed=DATA_SEED):
    """The training set, then the test set of every deletion level in ascending order, from one RandomState."""
    rng = np.random.RandomState(derive_seed(base_seed, 'data', n_items, replicate))
    train = draw_rows(n_items, 0, TRAIN_PER_CLASS, rng)
    tests = {k: draw_rows(n_items, k, TEST_PER_CLASS, rng) for k in LEVELS[n_items]}
    return train, tests


def data_hash(train, tests):
    payload = {'train': [list(s) for s in train['sequences']], 'y': train['y'].tolist(),
               'tests': {str(k): {'x': [list(s) for s in t['sequences']], 'y': t['y'].tolist()} for k, t in tests.items()}}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


# ----------------------------------------------------------------------------- representations

def seq_numeric(sequences, n_items):
    """The 2023 DataGraph.transform_2_numpy_dataset: float(item) - 1 at each position, zero-padded to n_items."""
    X = np.zeros((len(sequences), n_items))
    for row, sequence in enumerate(sequences):
        X[row, :len(sequence)] = np.asarray(sequence, dtype=float) - 1
    return X


def seq_onehot(sequences, n_items):
    """One-hot of the item at each position, with a padding category for positions past the row's length."""
    X = np.zeros((len(sequences), n_items, n_items + 1))
    for row, sequence in enumerate(sequences):
        for position in range(n_items):
            X[row, position, sequence[position] - 1 if position < len(sequence) else n_items] = 1
    return X.reshape(len(sequences), -1)


def end_positions(sequences, n_items):
    """Positions of the row completed by appending its missing items in ascending item order (the `end` convention)."""
    P = np.empty((len(sequences), n_items))
    for row, sequence in enumerate(sequences):
        present = [item - 1 for item in sequence]
        order = present + sorted(set(range(n_items)) - set(present))
        P[row, order] = np.arange(n_items)
    return P


def center_positions(sequences, n_items):
    """Present items at their observed positions 0..m-1, missing items at n_items / 2 (the `center` convention, the
    core's missing_data_loc = 0.5 * V)."""
    P = np.full((len(sequences), n_items), .5 * n_items)
    for row, sequence in enumerate(sequences):
        P[row, np.asarray(sequence) - 1] = np.arange(len(sequence))
    return P


def partial_distances(sequences, references):
    """The induced footrule (the `partial` convention): for each row and each reference permutation (rows of
    `references`, the positions of the items), sum over the row's present items of |observed position - rank of the item
    among the present items in the reference|. On a complete row it is the footrule."""
    references = np.asarray(references)
    out = np.empty((len(sequences), len(references)))
    for row, sequence in enumerate(sequences):
        items = np.asarray(sequence) - 1
        sub = references[:, items]
        ranks = np.argsort(np.argsort(sub, axis=1, kind='stable'), axis=1, kind='stable')
        out[row] = np.abs(ranks - np.arange(len(items))).sum(axis=1)
    return out


def first_distances(sequences, n_items, references, convention):
    references = np.asarray(references, dtype=float)
    if convention == 'end':
        return cdist(end_positions(sequences, n_items), references, metric='cityblock')
    if convention == 'center':
        return cdist(center_positions(sequences, n_items), references, metric='cityblock')
    if convention == 'partial':
        return partial_distances(sequences, references)
    raise ValueError(f'Unknown convention {convention!r}')


def knn_from_distances(D, labels, classes, n_neighbors, weights):
    """StableFootruleKNN.predict_neighbors on a precomputed distance matrix (training rows in source order): stable
    nearest neighbours, uniform or inverse-distance votes (exact zeros take all the weight), lowest class on ties."""
    k = min(n_neighbors, D.shape[1])
    nearest = np.argsort(D, axis=1, kind='stable')[:, :k]
    distances = np.take_along_axis(D, nearest, axis=1)
    out = []
    for d, neighbours in zip(distances, nearest):
        if weights == 'uniform':
            w = np.ones(len(d))
        elif np.any(d == 0):
            w = (d == 0).astype(float)
        else:
            w = 1. / d
        out.append(classes[np.argmax(np.bincount(labels[neighbours], weights=w, minlength=len(classes)))])
    return np.asarray(out)


# ----------------------------------------------------------------------------- native ArrowFlow under a convention

def layer_matrices(net):
    return [np.asarray(net.network_.graph.vertex_list[f'revision_ly{i}'].index_matrix, dtype=float)
            for i in range(len(net.widths))]


def native_hidden(net, sequences, n_items, convention):
    """The last hidden layer's inverse positions: the first layer's responses under the convention, then the core's rule
    (cityblock responses, ascending with ties by filter index, inverse positions) for every later hidden layer."""
    matrices = layer_matrices(net)
    positions = inverse_positions(score_order(first_distances(sequences, n_items, matrices[0], convention)))
    for matrix in matrices[1:]:
        positions = inverse_positions(score_order(cdist(positions, matrix, metric='cityblock')))
    return positions


def native_predict(model, sequences, n_items, convention):
    votes = [readout.predict(native_hidden(net, sequences, n_items, convention))
             for net, readout in zip(model.views_, model.readouts_)]
    return majority(votes)


class InputFootruleKNN:
    """Footrule kNN on the input rankings (complete training rows), neighbours and weighting by select_knn_readout."""
    def fit(self, sequences, y, n_items, seed):
        self.n_items = n_items
        self.positions_ = end_positions(sequences, n_items).astype(np.int64)     # complete rows: exact positions
        self.selection_ = select_knn_readout(self.positions_, np.asarray(y), seed=derive_seed(seed, 'input_knn'))
        self.classes_, self.labels_ = np.unique(y, return_inverse=True)
        return self

    def predict(self, sequences, convention):
        D = first_distances(sequences, self.n_items, self.positions_, convention)
        return knn_from_distances(D, self.labels_, self.classes_, **self.selection_['config'])


def prototype_reference_error(sequences, y, n_items):
    """Expected error (%) of the nearest true prototype by Kendall distance on the present items, ties shared."""
    reference = prototypes(n_items)
    credit = []
    for sequence, label in zip(sequences, y):
        d = np.array([kendall_distance(sequence, p) for p in reference])
        tied = np.flatnonzero(d == d.min())
        credit.append(float(label in tied) / len(tied))
    return 100 * (1 - float(np.mean(credit)))


# ----------------------------------------------------------------------------- candidates and tuning

def candidates(model, smoke=False):
    if model == 'af_encoded':
        c = bridge_candidates()
    elif model == 'af_encoded_untrained':
        c = control_candidates(UNTRAINED_MODEL)
    elif model == 'af_native':
        c = native_candidates(BUDGET)
    elif model == 'af_native_untrained':
        c = project_candidates(native_candidates(BUDGET), UNTRAINED_KEYS)
    else:
        family = model.split('_', 1)[1]
        c = candidate_grid(CONVENTIONAL_GRIDS[family], BUDGET['candidate_budget'], BUDGET['candidate_seed'])
    if smoke:
        c = c[:2]
        if model in ('af_encoded', 'af_native'):
            c = [dict(config, iterations=2) for config in c]
    return c


def factory(model, config, seed):
    if model == 'af_encoded':
        return AdaptiveMultiViewKNN(config=config, seed=seed)
    if model == 'af_encoded_untrained':
        return AdaptiveUntrainedKNN(config=config, seed=seed)
    if model == 'af_native':
        return NativeMultiViewArrowFlowKNN(**config, seed=seed)
    if model == 'af_native_untrained':
        return NativeUntrainedMultiViewArrowFlowKNN(**config, seed=seed)
    return conventional_factory(model.split('_', 1)[1], config, seed)


def model_seed(model, n_items, replicate, base=MODEL_SEED):
    """Trained and untrained controls of one arm share their seed, so the untrained networks start where training does."""
    arm = model.replace('_untrained', '')
    return derive_seed(base, 'fit', n_items, replicate, arm)


def tune(model, X, y, seed, smoke=False):
    """3-fold stratified CV on the training set, mean accuracy, ties to the lowest config_id; refit on all of it."""
    splits = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=derive_seed(seed, 'inner')).split(X, y))
    scores = {}
    for config in candidates(model, smoke):
        accuracies = [float(np.mean(factory(model, config, seed).fit(X[a], y[a]).predict(X[b]) == y[b]))
                      for a, b in splits]
        scores[config_id(config)] = float(np.mean(accuracies))
    best = min(candidates(model, smoke), key=lambda c: (-scores[config_id(c)], config_id(c)))
    return factory(model, best, seed).fit(X, y), best, scores


def inputs_for(model, sequences, n_items):
    if model.startswith('seqnum_'):
        return seq_numeric(sequences, n_items)
    if model.startswith('seqhot_'):
        return seq_onehot(sequences, n_items)
    return features(sequences, n_items)          # POS (relative positions, NaN where missing)


# ----------------------------------------------------------------------------- one job

def error(pred, y):
    return 100 * float(np.mean(np.asarray(pred) != np.asarray(y)))


def run_job(n_items, replicate, model, *, base_data=DATA_SEED, base_model=MODEL_SEED, smoke=False):
    start = time.perf_counter()
    train, tests = replicate_data(n_items, replicate, base_data)
    seed = model_seed(model, n_items, replicate, base_model)
    y = train['y']
    record = {'study_id': STUDY_ID, 'n_items': n_items, 'replicate': replicate, 'model': model, 'seed': seed,
              'data_hash': data_hash(train, tests), 'smoke': smoke, 'predictions': {}, 'errors': {}}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if model == 'prototype_reference':
            for k, t in tests.items():
                record['errors'][str(k)] = prototype_reference_error(t['sequences'], t['y'], n_items)
        elif model == 'input_footrule_knn':
            knn = InputFootruleKNN().fit(train['sequences'], y, n_items, seed)
            record['selected'] = knn.selection_['config']
            record['candidate_scores'] = knn.selection_['candidate_scores']
            for k, t in tests.items():
                for convention in CONVENTIONS:
                    pred = knn.predict(t['sequences'], convention)
                    record['predictions'][f'{k}@{convention}'] = pred.tolist()
                    record['errors'][f'{k}@{convention}'] = error(pred, t['y'])
        else:
            X = inputs_for(model, train['sequences'], n_items)
            fitted, best, scores = tune(model, X, y, seed, smoke)
            record['selected'], record['candidate_scores'] = best, scores
            for k, t in tests.items():
                if model in NATIVE_MODELS:
                    for convention in CONVENTIONS:
                        pred = native_predict(fitted, t['sequences'], n_items, convention)
                        record['predictions'][f'{k}@{convention}'] = pred.tolist()
                        record['errors'][f'{k}@{convention}'] = error(pred, t['y'])
                else:
                    pred = fitted.predict(inputs_for(model, t['sequences'], n_items))
                    record['predictions'][str(k)] = np.asarray(pred).tolist()
                    record['errors'][str(k)] = error(pred, t['y'])
            if model in NATIVE_MODELS:                   # the core's own prediction must equal ours on complete rows
                t0 = tests[0]
                core = fitted.predict(features(t0['sequences'], n_items))
                record['core_agreement_k0'] = {c: bool(np.array_equal(core, record['predictions'][f'0@{c}']))
                                               for c in CONVENTIONS}
                if not all(record['core_agreement_k0'].values()):
                    raise RuntimeError(f'{model} n={n_items} r={replicate}: a convention disagrees with the core on '
                                       f'complete rows: {record["core_agreement_k0"]}')
    record['seconds'] = time.perf_counter() - start
    return record


def job_name(n_items, replicate, model):
    return f'n{n_items:02d}_r{replicate:02d}_{model}.json'


def _worker(arguments):
    n_items, replicate, model, out, base_data, base_model, smoke = arguments
    path = Path(out)/'jobs'/job_name(n_items, replicate, model)
    if path.exists():
        return str(path), 'exists'
    record = run_job(n_items, replicate, model, base_data=base_data, base_model=base_model, smoke=smoke)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return str(path), f'{record["seconds"]:.0f}s'


def execute(out, replicates, workers, *, base_data=DATA_SEED, base_model=MODEL_SEED, smoke=False):
    out = Path(out)
    (out/'jobs').mkdir(parents=True, exist_ok=True)
    heavy = {'af_native': 0, 'af_encoded': 1}
    jobs = sorted(((n, r, m, str(out), base_data, base_model, smoke)
                   for n in N_VALUES for r in range(replicates) for m in MODELS),
                  key=lambda j: (heavy.get(j[2], 2), -j[0], j[1], j[2]))
    done = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            path, status = future.result()
            done += 1
            print(f'[{done}/{len(jobs)}] {Path(path).name} {status}', flush=True)


# ----------------------------------------------------------------------------- protocol, freeze and provenance

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def design():
    return {'study_id': STUDY_ID, 'n_values': list(N_VALUES), 'levels': {str(n): list(v) for n, v in LEVELS.items()},
            'swap_max': {str(n): v for n, v in SWAP_MAX.items()}, 'train_per_class': TRAIN_PER_CLASS,
            'test_per_class': TEST_PER_CLASS, 'replicates': REPLICATES, 'data_seed': DATA_SEED, 'model_seed': MODEL_SEED,
            'inner_folds': INNER_FOLDS, 'budget': BUDGET, 'conventions': list(CONVENTIONS), 'models': list(MODELS),
            'candidates': {m: [config_id(c) for c in candidates(m)] for m in MODELS
                           if m not in ('input_footrule_knn', 'prototype_reference')},
            'primary_arms': list(PRIMARY_ARMS), 'primary_comparator': PRIMARY_COMPARATOR,
            'primary_family': 'error(arm) - error(comparator) for each arm, N and deletion level k >= 1; paired t over '
                              'replicates, two-sided; Holm over the 12 contrasts',
            'reading': 'per arm: supported if the arm is significantly lower (Holm) in >= 4 of 6 cells; contradicted '
                       'if the comparator is significantly lower in >= 4 of 6; otherwise not shown either way'}


def freeze():
    if PROTOCOL_FILE.exists():
        raise SystemExit(f'{PROTOCOL_FILE} exists; a frozen protocol is never replaced')
    protocol = {'design': design(), 'sources': {s: sha256_file(REPO/s) for s in SOURCES},
                'frozen_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'parent_revision': git_revision()}
    PROTOCOL_FILE.write_text(json.dumps(protocol, indent=1, sort_keys=True) + '\n')
    print(f'wrote {PROTOCOL_FILE}')


def check_frozen():
    protocol = json.loads(PROTOCOL_FILE.read_text())
    changed = [s for s, h in protocol['sources'].items() if s != 'experiments/artificial_robustness/study.py'
               and sha256_file(REPO/s) != h]
    if protocol['sources']['experiments/artificial_robustness/study.py'] != sha256_file(HERE/'study.py'):
        changed.append('experiments/artificial_robustness/study.py')
    if changed or protocol['design'] != json.loads(json.dumps(design())):
        raise SystemExit(f'The frozen protocol does not match the tree: changed {changed or "design"}')
    status = subprocess.run(['git', 'status', '--porcelain', '--', *SOURCES, 'experiments/artificial_robustness/protocol.json'],
                            cwd=REPO, capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise SystemExit(f'Uncommitted changes to the frozen sources:\n{status}')
    return protocol


# ----------------------------------------------------------------------------- verification (V1-V5)

def verify(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    # V1: the generator (production seeds; structure only, nothing is fitted or scored)
    v1 = []
    for n in N_VALUES:
        for r in range(REPLICATES):
            train, tests = replicate_data(n, r)
            ok = (all(len(s) == n for s in train['sequences']) and np.all(np.bincount(train['y']) == TRAIN_PER_CLASS)
                  and all(len(s) == n - k for k, t in tests.items() for s in t['sequences'])
                  and all(np.all(np.bincount(t['y']) == TEST_PER_CLASS) for t in tests.values())
                  and all(0 <= s <= SWAP_MAX[n] for t in [train, *tests.values()] for s in t['swaps'])
                  and all(len(set(s)) == len(s) for t in [train, *tests.values()] for s in t['sequences']))
            v1.append(ok)
    report['V1_generator'] = {'passed': all(v1), 'datasets_checked': len(v1)}
    # V3: the induced footrule against brute force
    rng = np.random.RandomState(5)
    brute_ok = True
    for _ in range(300):
        n = int(rng.choice([8, 16]))
        refs = np.array([rng.permutation(n) for _ in range(5)])                     # positions of items 0..n-1
        m = int(rng.randint(2, n + 1))
        seq = [int(i) + 1 for i in rng.permutation(n)[:m]]
        fast = partial_distances([seq], refs)[0]
        for f, ref in enumerate(refs):
            present = [i - 1 for i in seq]
            ref_order = sorted(present, key=lambda i: ref[i])                       # reference order restricted
            slow = sum(abs(p - ref_order.index(i)) for p, i in enumerate(present))
            brute_ok &= fast[f] == slow
    report['V3_partial_footrule'] = {'passed': bool(brute_ok), 'cases': 1500}
    # V2 and V4 on smoke-seeded data and models
    v2 = {}
    for n in N_VALUES:
        train, tests = replicate_data(n, 0, SMOKE_DATA_SEED)
        X, y = features(train['sequences'], n), train['y']
        seqs0 = tests[0]['sequences']
        for model_class, name in ((NativeMultiViewArrowFlowKNN, 'trained'), (NativeUntrainedMultiViewArrowFlowKNN, 'untrained')):
            kwargs = dict(n_views=2, widths=[32, 16], seed=SMOKE_MODEL_SEED)
            if name == 'trained':
                kwargs.update(iterations=5, learning_rate=.1, p_correct=.01, validation_ratio=.1)
            model = model_class(**kwargs).fit(X, y)
            orders0 = native_orders(features(seqs0, n))
            hidden_core = [net.transform_orders(orders0) for net in model.views_]
            same_hidden = {c: all(np.array_equal(native_hidden(net, seqs0, n, c), h)
                                  for net, h in zip(model.views_, hidden_core)) for c in CONVENTIONS}
            core_pred = model.predict(features(seqs0, n))
            same_pred = {c: bool(np.array_equal(native_predict(model, seqs0, n, c), core_pred)) for c in CONVENTIONS}
            k_max = max(LEVELS[n])
            differ = {c: bool(not np.array_equal(native_hidden(model.views_[0], tests[k_max]['sequences'], n, c),
                                                 native_hidden(model.views_[0], tests[k_max]['sequences'], n, 'end')))
                      for c in ('center', 'partial')}
            v2[f'n{n}_{name}'] = {'hidden_equal_core_k0': same_hidden, 'predictions_equal_core_k0': same_pred,
                                  'conventions_differ_at_max_deletions': differ}
        # V4: kNN from distances against StableFootruleKNN on complete rows
        knn = InputFootruleKNN().fit(train['sequences'], y, n, SMOKE_MODEL_SEED)
        reference = StableFootruleKNN(**knn.selection_['config'], input_kind='positions').fit(knn.positions_, y)
        ref_pred = reference.predict(end_positions(seqs0, n).astype(np.int64))
        v2[f'n{n}_input_knn'] = {c: bool(np.array_equal(knn.predict(seqs0, c), ref_pred)) for c in CONVENTIONS}
    ok2 = all(all(v['hidden_equal_core_k0'].values()) and all(v['predictions_equal_core_k0'].values())
              and all(v['conventions_differ_at_max_deletions'].values())
              for k, v in v2.items() if 'input_knn' not in k)
    ok4 = all(all(v.values()) for k, v in v2.items() if 'input_knn' in k)
    report['V2_conventions_equal_core_on_complete_rows'] = {'passed': bool(ok2), 'detail': {k: v for k, v in v2.items() if 'input_knn' not in k}}
    report['V4_knn_from_distances'] = {'passed': bool(ok4), 'detail': {k: v for k, v in v2.items() if 'input_knn' in k}}
    # V5: the 2023 representation reproduces the 2023 log exactly
    report['V5_2023_log'] = reproduce_2023_log()
    report['passed'] = all(v['passed'] for v in report.values() if isinstance(v, dict) and 'passed' in v)
    (out/'verify.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: (v['passed'] if isinstance(v, dict) else v) for k, v in report.items()}, indent=1))
    if not report['passed']:
        raise SystemExit('verification failed')


def original_split():
    rows = original_rows()
    train = [(int(label), seq) for label, split, seq in rows if split == 'train']
    test = [(int(label), seq) for label, split, seq in rows if split == 'test']
    return train, test


def classifiers_2023():
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    out = [(f'kNN-{k}', KNeighborsClassifier(k)) for k in range(1, 7)]
    return out + [('LinearSVM', SVC(kernel='linear', C=0.025, random_state=42)),
                  ('RBF-SVM', SVC(gamma=2, C=1, random_state=42)),
                  ('GaussProc', GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)),
                  ('DecTree', DecisionTreeClassifier(max_depth=5, random_state=42)),
                  ('RandomForest', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42))]


def representations_2023(sequences):
    return {'sequence_numeric': seq_numeric(sequences, 8), 'sequence_onehot': seq_onehot(sequences, 8),
            'position_mean_imputed': features(sequences, 8), 'position_end_completed': end_positions(sequences, 8) / 7}


def fit_2023(name, Xtr, ytr, Xte):
    means = np.nanmean(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), means, Xtr)
    Xte = np.where(np.isnan(Xte), means, Xte)
    model = dict(classifiers_2023())[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.fit(Xtr, ytr).predict(Xte)


def reproduce_2023_log():
    train, test = original_split()
    Xtr = seq_numeric([s for _, s in train], 8)
    Xte = seq_numeric([s for _, s in test], 8)
    ytr, yte = np.array([l for l, _ in train]), np.array([l for l, _ in test])
    got = {name: error(fit_2023(name, Xtr, ytr, Xte), yte) for name, _ in classifiers_2023()}
    return {'passed': all(abs(got[k] - LOG_2023[k]) < 1e-9 for k in LOG_2023), 'reproduced': got, 'log': LOG_2023}


# ----------------------------------------------------------------------------- Part A: the original split

def part_a(out, seeds=(8129, 19391, 39019)):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = original_split()
    seq_tr, seq_te = [s for _, s in train], [s for _, s in test]
    ytr, yte = np.array([l for l, _ in train]), np.array([l for l, _ in test])
    result = {'rows': {'train': len(train), 'test': len(test)}, 'classifiers_2023': {}, 'arrowflow': {}}
    reps_tr, reps_te = representations_2023(seq_tr), representations_2023(seq_te)
    for rep in reps_tr:
        result['classifiers_2023'][rep] = {name: error(fit_2023(name, reps_tr[rep], ytr, reps_te[rep]), yte)
                                           for name, _ in classifiers_2023()}
    Xtr, Xte = features(seq_tr, 8), features(seq_te, 8)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for model in ('af_encoded', 'af_native', 'af_encoded_untrained', 'af_native_untrained'):
            runs = []
            for seed in seeds:
                fitted, best, scores = tune(model, Xtr, ytr, seed)
                runs.append({'seed': seed, 'selected': best, 'error': error(fitted.predict(Xte), yte)})
            result['arrowflow'][model] = {'runs': runs, 'mean_error': float(np.mean([r['error'] for r in runs]))}
        knn = InputFootruleKNN()
        knn.positions_ = end_positions(seq_tr, 8).astype(np.int64)
        knn.n_items = 8
        knn.selection_ = select_knn_readout(knn.positions_, ytr, seed=derive_seed(8129, 'input_knn'))
        knn.classes_, knn.labels_ = np.unique(ytr, return_inverse=True)
        result['input_footrule_knn_end'] = {'selected': knn.selection_['config'], 'error': error(knn.predict(seq_te, 'end'), yte)}
    result['note'] = ('native ArrowFlow here trains on the training rows completed at the end (they contain deletions) and '
                      'is tested under the same convention; the encoded arm and the classical models see positions with '
                      'NaN imputed by the training mean. Descriptive only.')
    (out/'part_a.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'note'}, indent=1)[:4000])


# ----------------------------------------------------------------------------- analysis

def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def load_errors(out, replicates):
    """errors[(key, n, k)] = array over replicates; key is model or model@convention."""
    jobs = Path(out)/'jobs'
    errors, hashes, missing = {}, {}, []
    for n in N_VALUES:
        for r in range(replicates):
            for model in MODELS:
                path = jobs/job_name(n, r, model)
                if not path.exists():
                    missing.append(path.name)
                    continue
                record = json.loads(path.read_text())
                hashes.setdefault((n, r), set()).add(record['data_hash'])
                for cell, value in record['errors'].items():
                    k, _, convention = cell.partition('@')
                    key = f'{model}@{convention}' if convention else model
                    errors.setdefault((key, n, int(k)), [None] * replicates)[r] = value
    if missing:
        raise SystemExit(f'{len(missing)} jobs missing, e.g. {missing[:3]}')
    if any(len(h) != 1 for h in hashes.values()):
        raise SystemExit('Models of one replicate saw different data')
    return {key: np.asarray(v, dtype=float) for key, v in errors.items()}


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()), None, None
    half = stats.t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    mean, low, high = mean_ci(d)
    if np.allclose(d, d[0]):                   # zero variance: the exact sign-flip p value of a constant difference
        p = 1. if d[0] == 0 else min(1., 2 * .5 ** len(d))
    else:
        p = float(stats.ttest_rel(a, b).pvalue)
    return {'mean_difference': mean, 'ci_low': low, 'ci_high': high, 'p': p,
            'arm_lower_in': int(np.sum(d < 0)), 'comparator_lower_in': int(np.sum(d > 0)), 'replicates': len(d)}


def analyse(out, replicates=REPLICATES):
    out = Path(out)
    errors = load_errors(out, replicates)
    keys = sorted({key for key, _, _ in errors})
    summary = {key: {str(n): {str(k): dict(zip(('mean', 'ci_low', 'ci_high'), mean_ci(errors[(key, n, k)])))
                              for k in LEVELS[n]} for n in N_VALUES} for key in keys}
    primary = []
    for arm in PRIMARY_ARMS:
        for n in N_VALUES:
            for k in LEVELS[n][1:]:
                primary.append({'arm': arm, 'n_items': n, 'deletions': k,
                                **paired(errors[(arm, n, k)], errors[(PRIMARY_COMPARATOR, n, k)])})
    for row, adjusted in zip(primary, holm([row['p'] for row in primary])):
        row['p_holm'] = float(adjusted)
        row['significant'] = bool(adjusted < .05)
    reading = {}
    for arm in PRIMARY_ARMS:
        rows = [row for row in primary if row['arm'] == arm]
        lower = sum(row['significant'] and row['mean_difference'] < 0 for row in rows)
        higher = sum(row['significant'] and row['mean_difference'] > 0 for row in rows)
        reading[arm] = {'arm_significantly_lower': lower, 'comparator_significantly_lower': higher,
                        'verdict': 'supported' if lower >= 4 else 'contradicted' if higher >= 4 else 'not shown either way'}
    secondary = {}
    comparators = [f'pos_{m}' for m in CLASSICAL] + [f'seqnum_{m}' for m in SEQ_CLASSICAL] + [f'seqhot_{m}' for m in SEQ_CLASSICAL]
    arms = ['af_native@partial', 'af_native@center', 'af_native@end', 'af_encoded']
    for arm in arms:
        for comparator in comparators + ['af_native_untrained@partial', 'af_encoded_untrained',
                                         'input_footrule_knn@partial', 'input_footrule_knn@center', 'input_footrule_knn@end']:
            if comparator.split('@')[0].replace('_untrained', '') == arm.split('@')[0] and 'untrained' not in comparator:
                continue
            for n in N_VALUES:
                for k in LEVELS[n]:
                    secondary[f'{arm} - {comparator} | N={n} k={k}'] = paired(errors[(arm, n, k)], errors[(comparator, n, k)])
    matched = {}
    for convention in CONVENTIONS:
        for n in N_VALUES:
            for k in LEVELS[n]:
                matched[f'af_native@{convention} - input_footrule_knn@{convention} | N={n} k={k}'] = paired(
                    errors[(f'af_native@{convention}', n, k)], errors[(f'input_footrule_knn@{convention}', n, k)])
                matched[f'af_native@{convention} - af_native_untrained@{convention} | N={n} k={k}'] = paired(
                    errors[(f'af_native@{convention}', n, k)], errors[(f'af_native_untrained@{convention}', n, k)])
    for n in N_VALUES:
        for k in LEVELS[n]:
            matched[f'af_encoded - af_encoded_untrained | N={n} k={k}'] = paired(
                errors[('af_encoded', n, k)], errors[('af_encoded_untrained', n, k)])
    result = {'study_id': STUDY_ID, 'replicates': replicates, 'summary': summary, 'primary': primary,
              'reading': reading, 'secondary': secondary, 'matched': matched}
    (out/'analysis.json').write_text(json.dumps(result, indent=1) + '\n')
    write_tables(out, result)
    plot(out, errors)
    print(json.dumps({'reading': reading}, indent=1))


def write_tables(out, result):
    s = result['summary']
    lines = ['# Test error (%) by deletion level: mean over replicates [95% CI]', '']
    for n in N_VALUES:
        lines += [f'## N = {n}', '', '| model | ' + ' | '.join(f'k={k}' for k in LEVELS[n]) + ' |',
                  '|---|' + '---|' * len(LEVELS[n])]
        for key in sorted(s, key=lambda key: s[key][str(n)][str(LEVELS[n][-1])]['mean']):
            cells = []
            for k in LEVELS[n]:
                c = s[key][str(n)][str(k)]
                cells.append(f"{c['mean']:.1f}" + (f" [{c['ci_low']:.1f}, {c['ci_high']:.1f}]" if c['ci_low'] is not None else ''))
            lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Primary family (error of arm minus numeric kNN on positions; Holm over 12)', '',
              '| arm | N | k | difference [95% CI] | p | Holm p | arm lower in |', '|---|---|---|---|---|---|---|']
    for row in result['primary']:
        lines.append(f"| {row['arm']} | {row['n_items']} | {row['deletions']} | {row['mean_difference']:+.2f} "
                     f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}] | {row['p']:.4f} | {row['p_holm']:.4f} | "
                     f"{row['arm_lower_in']}/{row['replicates']} |")
    lines += ['', '## Reading', '']
    for arm, r in result['reading'].items():
        lines.append(f"- {arm}: {r['verdict']} (arm significantly lower in {r['arm_significantly_lower']} of 6 cells, "
                     f"numeric kNN significantly lower in {r['comparator_significantly_lower']} of 6)")
    (out/'tables.md').write_text('\n'.join(lines) + '\n')


def plot(out, errors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    show = [('af_native@partial', 'ArrowFlow native (partial)', '#1f77b4', '-'),
            ('af_native@end', 'ArrowFlow native (end)', '#1f77b4', ':'),
            ('af_encoded', 'ArrowFlow encoded (paper)', '#d62728', '-'),
            ('pos_numeric_knn', 'kNN on positions', '#2ca02c', '-'),
            ('pos_svc_rbf', 'SVC on positions', '#9467bd', '-'),
            ('pos_random_forest', 'Random forest on positions', '#8c564b', '-'),
            ('seqnum_numeric_knn', 'kNN on raw sequence (2023)', '#2ca02c', '--'),
            ('seqnum_random_forest', 'Random forest on raw sequence (2023)', '#8c564b', '--'),
            ('input_footrule_knn@partial', 'Footrule kNN on input (partial)', '#7f7f7f', '-'),
            ('prototype_reference', 'Nearest true prototype (reference)', 'k', ':')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, n in zip(axes, N_VALUES):
        for key, label, color, style in show:
            means = [np.mean(errors[(key, n, k)]) for k in LEVELS[n]]
            ax.plot(LEVELS[n], means, style, color=color, marker='o', ms=3, label=label)
        ax.set_title(f'N = {n} items (trained on clean patterns)')
        ax.set_xlabel('deleted items at test time')
        ax.set_xticks(LEVELS[n])
        ax.grid(alpha=.3)
    axes[0].set_ylabel('test error (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .1, 1, 1))
    fig.savefig(Path(out)/'error_by_deletions.png', dpi=150)


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('verify', 'smoke', 'freeze', 'run', 'original', 'analyse'))
    parser.add_argument('--output')
    parser.add_argument('--workers', type=int, default=14)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        verify(args.output)
    elif args.command == 'smoke':
        execute(args.output, 2, args.workers, base_data=SMOKE_DATA_SEED, base_model=SMOKE_MODEL_SEED, smoke=True)
        analyse(args.output, replicates=2)
    elif args.command == 'freeze':
        freeze()
    elif args.command == 'run':
        protocol = check_frozen()
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        env = {'git_revision': git_revision(), 'protocol_sha256': sha256_file(PROTOCOL_FILE), 'python': sys.version,
               'numpy': np.__version__, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        (out/'environment.json').write_text(json.dumps(env, indent=1) + '\n')
        execute(out, REPLICATES, args.workers)
        env['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        (out/'environment.json').write_text(json.dumps(env, indent=1) + '\n')
    elif args.command == 'original':
        part_a(args.output)
    elif args.command == 'analyse':
        analyse(args.output)


if __name__ == '__main__':
    main()
