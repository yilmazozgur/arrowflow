"""DESIGN (fixed 2026-11-24 before any score of these arms; committed before the run)

Two references for the learned-encoder result of the nested study (nested.py, run 2026-09-24-tensor-rank-nested).
There, the encoder trained through the exact sort by the ranking signal, T (hybrid_target_mlp), was more accurate than
the same network never trained, F (hybrid_frozen_mlp), whose fc1 and fc2 weights keep the code's initialization,
uniform on [0, 1). This module asks whether that gain survives against an untrained network from a standard
initialization, and how much of what backpropagation with cross-entropy achieves on the same network the ranking signal
recovers. The design is fixed after the outer results of T and F are known, as the encoder swap's was; the scores of the
three new arms are unknown until the run. A, B and C name the new arms below, and X[r] is arm X read by readout r.

What stays identical to the nested study.
- Data and folds: the registered data.npz and splits.json of 2026-09-12-bridge-knn and 2026-09-14-newdata-batch1/2, that
  is 17 datasets with 15 outer folds each (3 repeats of 5 folds, split seed 27183), each outer fold with its 3 inner folds.
- Preprocessing: imputation by the training mean, standardization on the training rows, and the core's negation of the
  input to its tensor layer.
- Architecture: the core's TensorNet (fc1, instance normalization without affine parameters, ReLU, fc2, instance
  normalization, ReLU, fc3) at width V, followed by one ranking layer of class filters (core_hybrid.build).
- The class filters' rule and settings: the gate (wrong, or correct with p_correct 0.1), rank learning rate 0.1, and the
  weighted Borda reorder. For the arm trained by the ranking signal, the target conversion's settings as well: contrast
  0.5, beta 1, and the encoder's gate (accepted rows).
- Training length: 60 passes of batches of 32, that is 60 * ceil(n / 32) updates, with no validation checkpoint.
- Fit seeds 8129, 19391 and 39019. build() seeds every generator with the fit seed, so the draws of fc3, of the fc1 and fc2
  biases and of the class filters, and the starting states of the batch and gate streams, are the nested study's at the
  same seed.

The standard initialization (every new arm). Right after build(), the fc1 and then the fc2 weights are redrawn in place
with nn.init.kaiming_uniform_(w, a=sqrt(5)), which is nn.Linear's own default (uniform on +-1/sqrt(fan-in)) and which the
code already uses for fc3 and for the biases. Nothing else is redrawn and the NumPy streams are not touched, so A - F is
a paired measure of the initialization alone.

Arms. T and F are read from the nested run, not refitted.
  ref_untrained_std (A)  The encoder is never updated (mode frozen); only the class filters train, exactly as in F.
                         Grid V {64, 128}, as for F.
  ref_backprop_ce   (B)  Stage 1 trains the encoder by backpropagation with cross-entropy through a learned linear head;
                         stage 2 then trains the class filters on the frozen stage-1 network exactly as in A.
                         Grid V {64, 128} x lr {0.003, 0.01}, the size and values of T's grid.
  ref_trained_std   (C)  hybrid_target_mlp exactly (mode target), plus the standard initialization. T's grid.
Stage 1 of B, in this order: torch.manual_seed(derive_seed(seed, 'ce_head')); a head nn.Linear(V, C) at its default
initialization; Adam over the encoder's and the head's parameters at learning rate lr; the step schedule that
TensorNet.define_optimizer builds, whose milestones come from the configuration's sortnet_config.no_of_iters = 200 and not
from the fit's own length (the rate is multiplied by 0.7 after updates 20, 40, 50 and 100, and then stays at 0.24 of its
start); 60 * ceil(n / 32) updates, each on a batch of 32 rows drawn as the core's training loop draws them
(np.random.permutation(n)[:32]), with the cross-entropy of the head's output on every row of the batch (no gate); no
weight decay and no checkpoint. The NumPy stream is then restored to its state before stage 1, so that stage 2 starts
from the stream that A starts from at the same seed. The head serves only B's head readout.

Readouts. Every fit is read by 'rank' (the class filters) and by 'knn' (a footrule kNN on the encoder's ranking, its
neighbours and weighting chosen by multiview.select_knn_readout on the training rows at
derive_seed(seed, 'hybrid_knn_readout')). B is also read by 'head', the argmax of its head on the stage-1 network.

Selection. For every arm, the configuration is the one with the highest mean accuracy of the class-filter readout over
the outer fold's 3 inner splits at fit seed 8129, with ties to the lowest config_id, and without rescoring the three best
candidates at the other seeds. It is refitted on the outer training rows at the three fit seeds and scored on the outer
test rows by 'rank' and 'knn'. B's head readout is selected separately in the same way, by its own inner accuracy, which
the same inner fits give. When the two choices differ, the head's choice is refitted with stage 1 alone at the three
seeds; stage 2 changes neither the encoder nor the head, so this gives the head of a full refit.

T* = C. The run includes C, so C is the trained arm of every primary contrast and every fraction below. T enters only
the descriptive contrast C - T.

The test, for every contrast below. Per dataset, the differences are paired over the 15 outer folds of fold accuracy
averaged over the three fit seeds, and tested by the corrected resampled t (test/train ratio 0.25, 14 degrees of freedom,
two-sided; nested.corrected_t), with Holm's correction across the 17 datasets within each contrast (nested.holm). A
dataset is significantly positive (negative) when its mean difference is above (below) zero and its Holm-adjusted p is
below 0.05. Every difference is shown with its 95 % corrected resampled-t interval.

Primary contrasts.
  R1  C[rank] - A[rank]   Does the ranking signal train the encoder beyond a standard-initialization untrained network?
  R2  B[rank] - C[rank]   Is backpropagation better under the same readout?
  R3  B[rank] - A[rank]   Does backpropagation open a gap at all?

Recovery fractions. Each is a ratio of sums over the 17 datasets of per-dataset mean accuracy differences, not a mean of
per-dataset ratios, so that datasets with near-zero gaps (HCV, banknote-authentication) cannot make it blow up.
  X_rank = sum(C[rank] - A[rank]) / sum(B[rank] - A[rank])    the primary one: the same readout for every arm
  X_e2e  = sum(C[rank] - A[rank]) / sum(B[head] - A[rank])    the upper reference is B read by its own head
  X_knn  = sum(C[knn] - A[knn]) / sum(B[knn] - A[knn])         the same as X_rank on the kNN readout
Each has a 95 % percentile interval from 10,000 bootstrap resamples of the 17 datasets, drawn with replacement once, as
np.random.default_rng(20261124).integers(0, 17, (10000, 17)), and used for all three. The interval is the 2.5th and the
97.5th percentile of the resampled ratios of sums, interpolated linearly between order statistics (NumPy's default). A
resample whose denominator is zero or negative has no fraction: it counts as minus infinity for the lower and as plus
infinity for the upper percentile, so that it can only widen the interval, and an infinite end is reported as unbounded.

When a fraction is not reported. A fraction is not reported, and only the differences are, if the mean of its
denominator over the 17 datasets is below 1 point, or if its denominator is significantly positive on fewer than 3
datasets. For X_rank that test is R3. For X_e2e and X_knn it is the same test of their own denominators,
G_e2e = B[head] - A[rank] and G_knn = B[knn] - A[knn], each with its own Holm family.

Reading rule, applied in this order.
1. "No training effect shown" if R1 is significantly positive on at most 2 datasets, or if the mean of C[rank] - A[rank]
   over the 17 datasets is below 1 point. This reading overrides the words of step 3.
2. Otherwise the headline states the smaller of X_rank and X_e2e (the one that is reported, if only one is; no headline
   fraction, only the differences, if neither is). The text gives all three fractions.
3. The headline's word is "most" above 2/3, "about half" from 1/3 to 2/3, and "little" below 1/3.
4. If X_rank is reported and above 1 (B[rank] below C[rank] in sum), the text says that the class-filter readout favours
   the ranking signal, and gives the mean of B[head] - C[rank] together with X_e2e.
5. Always reported: the mean error of A[rank] and of F[rank] over the 17 datasets, and the mean of A[rank] - F[rank].

Descriptive analyses. They use the same test, each with its own Holm family, and no reading attaches to them.
- A - F on both readouts: the initialization alone, exactly paired.
- C - T on both readouts: the initialization's effect on training.
- B[head] - the paper's tuned MLP (its registered 'mlp' results on the same test sets): is B a competent backpropagation
  network? And B[head] - C[rank]: the lead of the backpropagation network read end to end.
- The three fractions on the five datasets that were not used in developing the conversion: wine quality,
  banknote-authentication, qsar-biodeg, climate-model-simulation-crashes and HCV. Their bootstrap is drawn in the same
  way, as np.random.default_rng(20261124).integers(0, 5, (10000, 5)). Each is shown only if the same fraction is
  reported on all 17 datasets and the mean of its denominator over the five is at least 1 point.
- The per-dataset table of every arm and readout, with each dataset's own ratio shown only where that dataset's
  denominator is at least 1 point.

Inputs of the analysis. Every number is recomputed from the per-row test predictions of the records, scored against the
registered labels. The stored accuracies and nested.hybrid_fold_accuracy must agree with them, and so must
nested.registered_fold_accuracy when the tuned MLP is read from the registered runs' own place.

Pre-run check ('check'). It prints timings, the forward deviation and counts of differing predictions or parameters, never
an accuracy.
1. With the standard initialization and stage 1 switched off, this module refits the registered F and T at their
   registered selections on iris i00, vehicle i00, balance-scale i00 and mfeat-zernike i03, at all three seeds, and must
   reproduce their test predictions exactly on both readouts.
2. The batched TensorNet forward that stage 1 and the readouts use must equal the core's per-row forward to within 1e-5,
   under both initializations and after stage 1.
3. One fit of each new arm is timed. On its parameters, the check verifies that the standard initialization redraws the
   fc1 and fc2 weights and nothing else, that stage 1 leaves the class filters and the NumPy stream as they were, and
   that stage 2 changes neither the encoder nor the head.

    python -m experiments.tensor_rank_sandbox.nested_reference check
    python -m experiments.tensor_rank_sandbox.nested_reference run --output DIR [--workers 14]
    python -m experiments.tensor_rank_sandbox.nested_reference analyse --output DIR --nested DIR [--mlp DIR]
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _n in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_n] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import math
import multiprocessing
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import warnings
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from .nested import (ARMS as NESTED_ARMS, BUILD, COMMON, DATASETS, EPOCHS, FIT_SEEDS, RUNS_ROOT, SOURCES, Hybrid,
                     config_id, corrected_t, holm, hybrid_fold_accuracy, load, registered_fold_accuracy, source_of)

REPO = Path(__file__).resolve().parents[2]
NESTED_RUN = RUNS_ROOT/'2026-09-24-tensor-rank-nested'
PROTECTED = tuple(RUNS_ROOT/name for name in (*SOURCES, NESTED_RUN.name, '2026-09-25-encoder-swap'))
SOURCE_PATHS = ('arrowflow', 'experiments/make_revision', 'experiments/tensor_rank_sandbox')
T_ARM, F_ARM = 'hybrid_target_mlp', 'hybrid_frozen_mlp'
# arm: (mode, arch, candidates, cross-entropy stage); every arm draws the standard initialization right after build()
ARMS = {'ref_untrained_std': ('frozen', 'core_mlp', NESTED_ARMS[F_ARM][2], False),
        'ref_backprop_ce': ('frozen', 'core_mlp', NESTED_ARMS[T_ARM][2], True),
        'ref_trained_std': ('target', 'core_mlp', NESTED_ARMS[T_ARM][2], False)}
ARM_OF = {'A': 'ref_untrained_std', 'B': 'ref_backprop_ce', 'C': 'ref_trained_std', 'T': T_ARM, 'F': F_ARM, 'MLP': 'mlp'}
SERIES = (('A', 'rank'), ('A', 'knn'), ('B', 'rank'), ('B', 'knn'), ('B', 'head'), ('C', 'rank'), ('C', 'knn'),
          ('T', 'rank'), ('T', 'knn'), ('F', 'rank'), ('F', 'knn'), ('MLP', None))
CE_SEED_STREAM = 'ce_head'
HELD_OUT = ('wine_quality', 'banknote_authentication', 'qsar_biodeg', 'climate_model_simulation_crashes',
            'hcv_egyptian_patients')
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CONFIDENCE = 10_000, 20261124, 95
TEST_RATIO, ALPHA = .25, .05
MIN_POINTS, MIN_SIGNIFICANT, NO_EFFECT_AT_MOST = 1., 3, 2
CHECK_SAMPLE = (('iris', 0), ('vehicle', 0), ('balance_scale', 0), ('mfeat_zernike', 3))
FORWARD_TOLERANCE = 1e-5
# name: (first series, second series, role); every contrast is a family over the 17 datasets with its own Holm
CONTRASTS = {'R1': (('C', 'rank'), ('A', 'rank'), 'primary'),
             'R2': (('B', 'rank'), ('C', 'rank'), 'primary'),
             'R3': (('B', 'rank'), ('A', 'rank'), 'primary'),
             'G_e2e': (('B', 'head'), ('A', 'rank'), 'gate of X_e2e'),
             'G_knn': (('B', 'knn'), ('A', 'knn'), 'gate of X_knn'),
             'A-F[rank]': (('A', 'rank'), ('F', 'rank'), 'descriptive'),
             'A-F[knn]': (('A', 'knn'), ('F', 'knn'), 'descriptive'),
             'C-T[rank]': (('C', 'rank'), ('T', 'rank'), 'descriptive'),
             'C-T[knn]': (('C', 'knn'), ('T', 'knn'), 'descriptive'),
             'B[head]-MLP': (('B', 'head'), ('MLP', None), 'descriptive'),
             'B[head]-C[rank]': (('B', 'head'), ('C', 'rank'), 'descriptive')}
# name: (numerator, denominator, the test of the denominator), each a pair of series whose per-dataset means differ
FRACTIONS = {'X_rank': ((('C', 'rank'), ('A', 'rank')), (('B', 'rank'), ('A', 'rank')), 'R3'),
             'X_e2e': ((('C', 'rank'), ('A', 'rank')), (('B', 'head'), ('A', 'rank')), 'G_e2e'),
             'X_knn': ((('C', 'knn'), ('A', 'knn')), (('B', 'knn'), ('A', 'knn')), 'G_knn')}


def standard_init(tensor):
    """nn.Linear's own default (kaiming_uniform_ with a = sqrt(5)) in place of TensorNet's U[0,1) draws, for the fc1 and
    fc2 weights only, drawn in place right after build(): fc3, the biases, the class filters and the NumPy streams stay
    those of the code's initialization at the same seed."""
    import torch.nn as nn
    if tensor.model_net[0] is not tensor.fc1 or tensor.model_net[3] is not tensor.fc2:
        raise ValueError('the standard initialization applies to the core TensorNet only')
    for layer in (tensor.fc1, tensor.fc2):
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))


def cross_entropy_stage(tensor, Z, yi, n_classes, lr, seed, updates):
    """Stage 1 of ref_backprop_ce: backpropagation with cross-entropy through a learned linear head on every row (no
    gate), with T's optimizer, schedule, length and batch draws; the NumPy stream is restored afterwards. Returns the
    head."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from arrowflow.config import sortnet_config
    from experiments.make_revision.comparisons import derive_seed
    torch.manual_seed(derive_seed(seed, CE_SEED_STREAM))
    head = nn.Linear(tensor.number_of_hidden, n_classes)
    opt = torch.optim.Adam([*tensor.model_net.parameters(), *head.parameters()], lr=lr)
    k = sortnet_config.no_of_iters                     # what TensorNet.define_optimizer uses: 200, not this fit's length
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[k / 10, k / 5, k / 4, k / 2], gamma=.7)
    if sched.milestones != tensor.scheduler.milestones:
        raise RuntimeError('stage 1 does not follow the step schedule of the core tensor layer')
    X, Y = torch.tensor(-Z, dtype=torch.float32), torch.tensor(yi, dtype=torch.long)   # the core negates its input
    n = len(yi)
    state = np.random.get_state()                      # draw batches as the core's training loop does, then restore
    for _ in range(updates):
        idx = torch.from_numpy(np.random.permutation(n)[:BUILD['batch_size']])
        loss = F.cross_entropy(head(tensor.model_net(X[idx])), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    np.random.set_state(state)
    return head


class Reference(Hybrid):
    """nested.Hybrid with the standard initialization and, for ref_backprop_ce, the cross-entropy stage. fit is
    Hybrid.fit line for line with two steps inserted after build(); with both switched off it is Hybrid.fit."""
    def __init__(self, mode, arch, V, lr_tensor, seed, *, standard=True, cross_entropy=False):
        super().__init__(mode, arch, V, lr_tensor, seed)
        self.standard, self.cross_entropy = standard, cross_entropy

    def fit(self, X, y):
        from experiments.make_revision.comparisons import StableFootruleKNN, derive_seed
        from experiments.make_revision.models import NumericImputer
        from experiments.make_revision.multiview import select_knn_readout
        from .core_hybrid import build, install, samples
        self.imputer = NumericImputer().fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        Z = self._inputs(X)
        self.classes_, yi = np.unique(y, return_inverse=True)
        iterations = EPOCHS * int(np.ceil(len(yi) / 32))
        self.net, config = build(Z.shape[1], len(self.classes_), V=self.V, iterations=iterations, lr_tensor=self.lr_tensor,
                                 arch=self.arch, seed=self.seed, **BUILD)
        if self.standard:                                                      # inserted: the standard initialization
            standard_init(self.net.graph.vertex_list['hyb_ly0'])
        if self.cross_entropy:                                                 # inserted: stage 1 of ref_backprop_ce
            self.head = cross_entropy_stage(self.net.graph.vertex_list['hyb_ly0'], Z, yi, len(self.classes_),
                                            self.lr_tensor, self.seed, iterations)
        install(self.net, self.mode, p_correct=BUILD['p_correct'], seed=self.seed, **COMMON)
        train = samples(Z, yi)
        self.net.train([train, train[:1]], config)
        pos = self.tensor_positions(Z)
        selection = select_knn_readout(pos, yi, seed=derive_seed(self.seed, 'hybrid_knn_readout'))
        self.knn = StableFootruleKNN(**selection['config'], input_kind='positions').fit(pos, yi)
        return self

    def fit_encoder(self, X, y):
        """fit up to and including its two inserted steps: no class-filter training and no kNN readout. It serves the
        head readout's refits (stage 2 changes neither the encoder nor the head) and the check."""
        from experiments.make_revision.models import NumericImputer
        from .core_hybrid import build
        self.imputer = NumericImputer().fit(X)
        self.scaler = StandardScaler().fit(self.imputer.transform(X))
        Z = self._inputs(X)
        self.classes_, yi = np.unique(y, return_inverse=True)
        iterations = EPOCHS * int(np.ceil(len(yi) / 32))
        self.net, _ = build(Z.shape[1], len(self.classes_), V=self.V, iterations=iterations, lr_tensor=self.lr_tensor,
                            arch=self.arch, seed=self.seed, **BUILD)
        if self.standard:
            standard_init(self.net.graph.vertex_list['hyb_ly0'])
        if self.cross_entropy:
            self.head = cross_entropy_stage(self.net.graph.vertex_list['hyb_ly0'], Z, yi, len(self.classes_),
                                            self.lr_tensor, self.seed, iterations)
        return self

    def predict(self, X, readout='rank'):
        if readout != 'head':
            return super().predict(X, readout)
        import torch
        with torch.no_grad():
            h = self.net.graph.vertex_list['hyb_ly0'](torch.tensor(-self._inputs(X), dtype=torch.float32))
            return self.classes_[self.head(h).argmax(1).numpy()]


def reference_model(arm, config, seed, **switches):
    mode, arch, _, cross_entropy = ARMS[arm]
    return Reference(mode, arch, config['V'], config['lr_tensor'], seed, cross_entropy=cross_entropy, **switches)


def run_job(dataset, arm, index):
    X, y, splits = load(dataset)
    split = splits[index]
    candidates, head = ARMS[arm][2], ARMS[arm][3]
    train, test = np.asarray(split['train']), np.asarray(split['test'])
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores, head_scores = {}, {}
        for c in candidates:
            accs, head_accs = [], []
            for inner in split['inner']:
                a, b = np.asarray(inner['train']), np.asarray(inner['validation'])
                model = reference_model(arm, c, FIT_SEEDS[0]).fit(X[a], y[a])
                accs.append(float(np.mean(model.predict(X[b], 'rank') == y[b])))
                if head:
                    head_accs.append(float(np.mean(model.predict(X[b], 'head') == y[b])))
            scores[config_id(c)] = float(np.mean(accs))
            if head:
                head_scores[config_id(c)] = float(np.mean(head_accs))
        best = min(candidates, key=lambda c: (-scores[config_id(c)], config_id(c)))
        best_head = min(candidates, key=lambda c: (-head_scores[config_id(c)], config_id(c))) if head else None
        fits = []
        for seed in FIT_SEEDS:
            model = reference_model(arm, best, seed).fit(X[train], y[train])
            pr, pk = model.predict(X[test], 'rank'), model.predict(X[test], 'knn')
            fit = {'seed': seed, 'rank_acc': float(np.mean(pr == y[test])), 'knn_acc': float(np.mean(pk == y[test])),
                   'rank_pred': pr.tolist(), 'knn_pred': pk.tolist()}
            if head:
                if best_head != best:
                    model = reference_model(arm, best_head, seed).fit_encoder(X[train], y[train])
                ph = model.predict(X[test], 'head')
                fit.update(head_acc=float(np.mean(ph == y[test])), head_pred=ph.tolist())
            fits.append(fit)
    record = {'dataset': dataset, 'arm': arm, 'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
              'selected': best, 'inner_scores': scores, 'fits': fits}
    if head:
        record.update(selected_head=best_head, inner_scores_head=head_scores)
    record['seconds'] = time.perf_counter() - start
    return record


# ------------------------------------------------------------------------------------------------ check

def check(nested_dir):
    """The pre-run check of the design. It prints timings, the forward deviation and counts of differing predictions or
    parameters only."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        failures = check_reproduction(Path(nested_dir)) + check_forward() + check_arms()
    if failures:
        raise SystemExit(f'check FAILED: {failures} failing comparison(s)')
    print('check passed', flush=True)


def check_reproduction(nested_dir):
    """Item 1: the registered F and T refitted by this module with both inserted steps switched off."""
    failures = 0
    for dataset, index in CHECK_SAMPLE:
        X, y, splits = load(dataset)
        split = splits[index]
        train, test = np.asarray(split['train']), np.asarray(split['test'])
        for arm in (F_ARM, T_ARM):
            record = json.loads((nested_dir/'results'/f'{dataset}__{arm}__i{index:02d}.json').read_text())
            if (record['outer_repeat'], record['outer_fold']) != (split['outer_repeat'], split['outer_fold']) \
                    or [f['seed'] for f in record['fits']] != list(FIT_SEEDS):
                raise SystemExit(f'{dataset} i{index:02d} {arm}: the record is not this outer fold at the fit seeds')
            mode, arch, _ = NESTED_ARMS[arm]
            best = record['selected']
            for fit in record['fits']:
                start = time.perf_counter()
                model = Reference(mode, arch, best['V'], best['lr_tensor'], fit['seed'], standard=False)
                model.fit(X[train], y[train])
                differ = {}
                for readout in ('rank', 'knn'):
                    pred, held = model.predict(X[test], readout), np.asarray(fit[f'{readout}_pred'])
                    differ[readout] = int(np.sum(pred != held)) if pred.shape == held.shape else len(test)
                print(f'{dataset:14s} fold {index:2d} seed {fit["seed"]}: {arm} refit vs registered: rank {differ["rank"]} '
                      f'of {len(test)} differ, knn {differ["knn"]} of {len(test)} differ '
                      f'({time.perf_counter() - start:.0f}s)', flush=True)
                failures += sum(v > 0 for v in differ.values())
    return failures


def check_forward():
    """Item 2: the batched forward (stage 1, the readouts) against the core's own per-row forward
    (SortFlowHybridNetwork._forward_propagate_sequential), on vehicle i00's outer training rows at V 128, seed 8129."""
    import torch
    from .core_hybrid import samples
    X, y, splits = load('vehicle')
    train = np.asarray(splits[0]['train'])
    failures = 0
    for label, switches in (('code initialization', {'standard': False}), ('standard initialization', {}),
                            ('standard initialization after stage 1', {'cross_entropy': True})):
        model = Reference('frozen', 'core_mlp', 128, .01, FIT_SEEDS[0], **switches).fit_encoder(X[train], y[train])
        tensor, Z = model.net.graph.vertex_list['hyb_ly0'], model._inputs(X[train])
        _, kept, _, _ = model.net._forward_propagate_sequential(samples(Z, np.searchsorted(model.classes_, y[train])),
                                                                'supervised', 'classification', False)
        per_row = torch.cat([entry[0][1] for entry in kept['hyb_ly0']]).detach().numpy()
        with torch.no_grad():
            batched = tensor.model_net(torch.tensor(-Z, dtype=torch.float32)).numpy()
        gap = float(np.abs(per_row - batched).max())
        ok = per_row.shape == batched.shape and gap <= FORWARD_TOLERANCE
        print(f'forward, {label}: batched vs the core per-row, max abs difference {gap:.1e} over {per_row.size} outputs '
              f'({"within" if ok else "NOT within"} {FORWARD_TOLERANCE:g})', flush=True)
        failures += not ok
    return failures


def check_arms():
    """Item 3: one fit of each new arm on vehicle i00's outer training rows (V 128, lr 0.01, seed 8129), timed, and the
    paired draws and the frozen stage 2 verified on parameters."""
    import torch
    from .core_hybrid import filter_positions
    X, y, splits = load('vehicle')
    train = np.asarray(splits[0]['train'])
    Xt, yt = X[train], y[train]
    config, seed = {'V': 128, 'lr_tensor': .01}, FIT_SEEDS[0]
    states, built = {}, {}
    for name, switches in (('code', {'standard': False}), ('standard', {}), ('stage1', {'cross_entropy': True})):
        start = time.perf_counter()
        built[name] = Reference('frozen', 'core_mlp', config['V'], config['lr_tensor'], seed, **switches).fit_encoder(Xt, yt)
        built[name].seconds = time.perf_counter() - start
        states[name] = np.random.get_state()

    def encoder(model):
        return model.net.graph.vertex_list['hyb_ly0'].model_net.state_dict()

    def differing(a, b):
        return sum(not torch.equal(a[k], b[k]) for k in a)

    def same_stream(a, b):
        return all(np.array_equal(u, v) for u, v in zip(a, b))
    failures = 0
    code, standard, stage1 = encoder(built['code']), encoder(built['standard']), encoder(built['stage1'])
    weights = ('0.weight', '3.weight')                                        # fc1 and fc2 in TensorNet.model_net
    redrawn = sum(not torch.equal(code[k], standard[k]) for k in weights)
    others = sum(not torch.equal(code[k], standard[k]) for k in code if k not in weights)
    filters = {k: filter_positions(m.net) for k, m in built.items()}
    moved = {k: int(np.sum(filters[k] != filters['code'])) for k in ('standard', 'stage1')}
    stream = {k: same_stream(states[k], states['code']) for k in ('standard', 'stage1')}
    print(f'standard initialization: {redrawn} of 2 fc1/fc2 weight tensors redrawn, {others} of {len(code) - 2} other '
          f'encoder tensors differ, class filters {moved["standard"]} of {filters["code"].size} positions differ, NumPy '
          f'stream {"unchanged" if stream["standard"] else "CHANGED"}', flush=True)
    print(f'stage 1 ({EPOCHS * int(np.ceil(len(yt) / 32))} updates, {built["stage1"].seconds:.1f}s): class filters '
          f'{moved["stage1"]} of {filters["code"].size} positions differ, NumPy stream '
          f'{"restored" if stream["stage1"] else "NOT restored"}', flush=True)
    failures += (redrawn != 2) + (others != 0) + sum(v != 0 for v in moved.values()) + sum(not v for v in stream.values())
    seconds, after = {}, {}
    for arm in ARMS:
        start = time.perf_counter()
        after[arm] = reference_model(arm, config, seed).fit(Xt, yt)
        seconds[arm] = time.perf_counter() - start
    print(f'timing: vehicle fold 0, one fit, V {config["V"]}, lr {config["lr_tensor"]}, seed {seed}: '
          + ', '.join(f'{arm} {s:.0f}s' for arm, s in seconds.items()), flush=True)
    frozen_a = differing(encoder(after['ref_untrained_std']), standard)
    frozen_b = differing(encoder(after['ref_backprop_ce']), stage1) + differing(after['ref_backprop_ce'].head.state_dict(),
                                                                                built['stage1'].head.state_dict())
    print(f'stage 2 leaves the network alone: ref_untrained_std encoder {frozen_a} of {len(code)} tensors differ from '
          f'its initialization; ref_backprop_ce encoder and head {frozen_b} of {len(code) + 2} tensors differ from the '
          f'stage-1 fit', flush=True)
    return failures + (frozen_a != 0) + (frozen_b != 0)


# ------------------------------------------------------------------------------------------------ run

def job_name(dataset, arm, index):
    return f'{dataset}__{arm}__i{index:02d}'


RECORD_NAME = re.compile(rf"^({'|'.join(DATASETS)})__({'|'.join(ARMS)})__i\d\d\.(json|tmp)$")
OUTPUT_ENTRIES = ('results', 'design.json', 'code_revision.txt', 'analysis.json', 'ANALYSIS.md', 'tables')


def guard_output(out):
    """Refuse an output directory that is, or lies inside, a registered run, or that holds anything this module does
    not write (logs aside). Nothing is written before this check."""
    out = Path(out).resolve()
    for run_dir in PROTECTED:
        run_dir = run_dir.resolve()
        if out == run_dir or run_dir in out.parents:
            raise SystemExit(f'refusing to write into the registered run {run_dir}')
    if out.exists():
        foreign = [p.name for p in out.iterdir() if p.name not in OUTPUT_ENTRIES and p.suffix != '.log']
        if (out/'results').is_dir():
            foreign += [f'results/{p.name}' for p in (out/'results').iterdir() if not RECORD_NAME.match(p.name)]
        if foreign:
            raise SystemExit(f'refusing {out}: it holds files this module does not write, e.g. {sorted(foreign)[:3]}')
    return out


def code_revision(allow_uncommitted=False):
    """The commit this run executes. The code it imports must be committed, since code_revision.txt is its provenance;
    --allow-uncommitted marks a test run instead."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    head, changes = git('rev-parse', 'HEAD'), git('status', '--porcelain', '--', *SOURCE_PATHS)
    if not changes:
        return head
    if not allow_uncommitted:
        raise SystemExit(f'the code this run imports has uncommitted changes; commit them first:\n{changes}')
    return head + '+uncommitted'


def design():
    from arrowflow.config import sortnet_config
    k = sortnet_config.no_of_iters
    return {'arms': {a: {'mode': m, 'arch': arch, 'candidates': c, 'cross_entropy_stage': ce}
                     for a, (m, arch, c, ce) in ARMS.items()},
            'initialization': 'kaiming_uniform_(a=sqrt(5)) on fc1.weight then fc2.weight, right after build(), every arm',
            'cross_entropy_stage': {'seed': f"derive_seed(seed, '{CE_SEED_STREAM}')", 'head': 'nn.Linear(V, C)',
                                    'optimizer': 'Adam(encoder and head, lr=lr_tensor)', 'gamma': .7,
                                    'milestones': [k / 10, k / 5, k / 4, k / 2], 'updates': 'epochs * ceil(n / 32)',
                                    'batch_size': BUILD['batch_size'], 'gate': None, 'numpy_stream': 'restored'},
            'epochs': EPOCHS, 'common': COMMON, 'build': BUILD, 'fit_seeds': FIT_SEEDS, 'sources': SOURCES,
            'registered': {'run': NESTED_RUN.name, 'T': T_ARM, 'F': F_ARM},
            'bootstrap': {'resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED, 'confidence': CONFIDENCE},
            'held_out': HELD_OUT}


def _init_worker():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import torch
    torch.set_num_threads(1)


def _worker(args):
    dataset, arm, index, out = args
    path = Path(out)/'results'/f'{job_name(dataset, arm, index)}.json'
    if path.exists():
        return path.name, 'exists'
    record = run_job(dataset, arm, index)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return path.name, f"{record['seconds']:.0f}s"


def run(out, workers, only=(), allow_uncommitted=False):
    out = guard_output(out)
    revision = code_revision(allow_uncommitted)
    plan = json.loads(json.dumps(design()))
    records = sorted((out/'results').glob('*.json')) if (out/'results').is_dir() else []
    if records:
        held = (out/'code_revision.txt').read_text().strip() if (out/'code_revision.txt').exists() else None
        if held != revision:
            raise SystemExit(f'{out} holds {len(records)} records from code revision {held}, not {revision}')
        if not (out/'design.json').exists() or json.loads((out/'design.json').read_text()) != plan:
            raise SystemExit(f'{out} holds records of another design')
    (out/'results').mkdir(parents=True, exist_ok=True)
    if not records:
        (out/'design.json').write_text(json.dumps(plan, indent=1) + '\n')
        (out/'code_revision.txt').write_text(revision + '\n')
    jobs = [(d, a, i, str(out)) for d in DATASETS for a in ARMS for i in range(15)]
    if only:
        unknown = sorted(set(only) - {job_name(*j[:3]) for j in jobs})
        if unknown:
            raise SystemExit(f'unknown jobs: {unknown}')
        jobs = [j for j in jobs if job_name(*j[:3]) in set(only)]
    sizes = {d: len(load(d)[1]) for d in DATASETS}
    cost = {a: len(c) * (1.4 if m == 'target' else 1.) for a, (m, _, c, _) in ARMS.items()}   # T's fits cost ~1.4 F's
    pending = sorted((j for j in jobs if not (out/'results'/f'{job_name(*j[:3])}.json').exists()),
                     key=lambda j: -sizes[j[0]] * cost[j[1]])
    failed = []
    with (out/'run.log').open('a') as log:
        def note(line):
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
        note(f'# start {datetime.now().isoformat(timespec="seconds")}, revision {revision}, {workers} workers: '
             f'{len(pending)} of {len(jobs)} jobs to run')
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init_worker) as pool:
                futures = {pool.submit(_worker, job): job_name(*job[:3]) for job in pending}
                for done, future in enumerate(as_completed(futures), 1):
                    try:
                        name, status = future.result()
                    except Exception as error:
                        name, status = futures[future] + '.json', f'FAILED ({type(error).__name__})'
                        failed.append(name)
                        traceback.print_exception(error, file=sys.stderr)
                    note(f'[{done}/{len(pending)}] {name} {status}')
        note(f'# end {datetime.now().isoformat(timespec="seconds")}' + (f', {len(failed)} failed' if failed else ''))
    if failed:
        raise SystemExit(f'{len(failed)} jobs failed: {failed[:3]}')


# ------------------------------------------------------------------------------------------------ analysis

def fold_accuracy(run_dir, dataset, arm, readout, y, tests):
    """Seed-averaged accuracy per outer fold of one arm and readout, recomputed from the per-row test predictions and
    checked against the stored accuracies and nested.hybrid_fold_accuracy."""
    out = {}
    for path in sorted((Path(run_dir)/'results').glob(f'{dataset}__{arm}__i*.json')):
        record = json.loads(path.read_text())
        key = (record['outer_repeat'], record['outer_fold'])
        if key in out or key not in tests or [f['seed'] for f in record['fits']] != list(FIT_SEEDS):
            raise SystemExit(f'{path}: a repeated or unknown outer fold, or other fit seeds')
        truth, accs = y[tests[key]], []
        for fit in record['fits']:
            pred = np.asarray(fit[f'{readout}_pred'])
            if pred.shape != truth.shape:
                raise SystemExit(f'{path}: {readout} predictions for another test set')
            accs.append(float(np.mean(pred == truth)))
            if abs(accs[-1] - fit[f'{readout}_acc']) > 1e-12:
                raise SystemExit(f'{path}: the stored {readout} accuracy differs from its predictions')
        out[key] = float(np.mean(accs))
    if sorted(out) != sorted(tests):
        raise SystemExit(f'{run_dir}: {dataset} {arm} has {len(out)} of the {len(tests)} outer folds')
    held = hybrid_fold_accuracy(run_dir, dataset, arm, readout)
    if sorted(held) != sorted(out) or any(abs(held[k] - out[k]) > 1e-12 for k in out):
        raise SystemExit(f'{run_dir}: {dataset} {arm} {readout} differs from nested.hybrid_fold_accuracy')
    return out


def mlp_fold_accuracy(root, dataset, y, tests):
    """The paper's tuned MLP (the registered 'mlp' results) per outer fold, recomputed from its per-row predictions and
    checked against its stored accuracies and, at the default place of the registered runs, registered_fold_accuracy."""
    out = {}
    for path in sorted((Path(root)/source_of(dataset)/'results').glob(f'{dataset}__mlp__r*f*.json')):
        record = json.loads(path.read_text())
        if record.get('status') != 'ok':
            raise SystemExit(f'{path}: status {record.get("status")}')
        key = (record['models'][0]['outer_repeat'], record['models'][0]['outer_fold'])
        stored = {m['model_seed']: m['accuracy'] for m in record['models']}
        if key in out or key not in tests or sorted(stored) != sorted(FIT_SEEDS):
            raise SystemExit(f'{path}: a repeated or unknown outer fold, or other fit seeds')
        rows = {}
        for p in record['predictions']:
            if p['y_true'] != y[int(p['sample_id'])]:
                raise SystemExit(f'{path}: a label differs from the registered data')
            rows.setdefault(p['model_seed'], {})[int(p['sample_id'])] = p['y_pred']
        test, accs = tests[key], []
        for seed in FIT_SEEDS:
            if sorted(rows.get(seed, {})) != sorted(int(i) for i in test):
                raise SystemExit(f'{path}: seed {seed} predicts another test set')
            accs.append(float(np.mean(np.asarray([rows[seed][int(i)] for i in test]) == y[test])))
            if abs(accs[-1] - stored[seed]) > 1e-12:
                raise SystemExit(f'{path}: the stored accuracy differs from its predictions')
        out[key] = float(np.mean(accs))
    if sorted(out) != sorted(tests):
        raise SystemExit(f'{root}: {dataset} mlp has {len(out)} of the {len(tests)} outer folds')
    if Path(root).resolve() == RUNS_ROOT.resolve():
        held = registered_fold_accuracy(dataset, 'mlp')
        if sorted(held) != sorted(out) or any(abs(held[k] - out[k]) > 1e-12 for k in out):
            raise SystemExit(f'{dataset} mlp differs from nested.registered_fold_accuracy')
    return out


def paired(a, b):
    keys = sorted(a)
    if keys != sorted(b):
        raise SystemExit('a paired contrast over different outer folds')
    diff = np.asarray([100 * (a[k] - b[k]) for k in keys])
    mean, p = corrected_t(diff, TEST_RATIO)
    half = stats.t.ppf(.5 + CONFIDENCE / 200, len(diff) - 1) * np.sqrt((1 / len(diff) + TEST_RATIO) * diff.var(ddof=1))
    return {'mean_difference': mean, 'ci_low': float(mean - half), 'ci_high': float(mean + half), 'p': p,
            'folds_better': int((diff > 0).sum())}


def family(first, second):
    """One contrast per dataset, with Holm across the 17, and its counts."""
    rows = [{'dataset': d, **paired(first[d], second[d])} for d in DATASETS]
    for row, adjusted in zip(rows, holm([r['p'] for r in rows])):
        row['p_holm'] = float(adjusted)
    return {'rows': rows, 'mean_difference': float(np.mean([r['mean_difference'] for r in rows])),
            'higher': int(sum(r['mean_difference'] > 0 for r in rows)),
            'significantly_higher': int(sum(r['mean_difference'] > 0 and r['p_holm'] < ALPHA for r in rows)),
            'significantly_lower': int(sum(r['mean_difference'] < 0 and r['p_holm'] < ALPHA for r in rows))}


def percentile(values, q):
    """NumPy's default (linear) percentile, extended to infinite values: an end that falls on or next to an infinite
    order statistic is infinite."""
    v = np.sort(np.asarray(values, dtype=float))
    at = (len(v) - 1) * q / 100
    lo, hi = int(np.floor(at)), int(np.ceil(at))
    if not np.isfinite(v[lo]):
        return float(v[lo])
    if not np.isfinite(v[hi]):
        return float(v[hi])
    return float(v[lo] + (v[hi] - v[lo]) * (at - lo))


def recovery(numerator, denominator, names):
    """A ratio of sums over the datasets named, with its dataset-bootstrap percentile interval (design: Recovery
    fractions)."""
    num, den = np.asarray([numerator[d] for d in names]), np.asarray([denominator[d] for d in names])
    idx = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(names), (BOOTSTRAP_RESAMPLES, len(names)))
    top, bottom = num[idx].sum(1), den[idx].sum(1)
    defined = bottom > 0
    ratio = np.divide(top, bottom, out=np.zeros_like(top), where=defined)
    tail = (100 - CONFIDENCE) / 2
    lo, hi = percentile(np.where(defined, ratio, -np.inf), tail), percentile(np.where(defined, ratio, np.inf), 100 - tail)
    return {'datasets': list(names), 'value': float(num.sum() / den.sum()) if den.sum() > 0 else None,
            'numerator_sum': float(num.sum()), 'denominator_sum': float(den.sum()),
            'numerator_mean': float(num.mean()), 'denominator_mean': float(den.mean()),
            'ci_low': lo if np.isfinite(lo) else None, 'ci_high': hi if np.isfinite(hi) else None,
            'undefined_resamples': int((~defined).sum())}


def word(x):
    return 'most' if x > 2 / 3 else 'about half' if x >= 1 / 3 else 'little'


def share(x):
    return 'unbounded' if x is None else f'{100 * x:.0f} %'


def reading(families, fractions, means):
    """The reading rule of the design, with the design's wording templates filled in."""
    r1, r2, e2e = families['R1'], families['R2'], families['B[head]-C[rank]']
    no_effect = r1['significantly_higher'] <= NO_EFFECT_AT_MOST or r1['mean_difference'] < MIN_POINTS
    reported = [n for n in ('X_rank', 'X_e2e') if fractions[n]['reported']]
    headline = min(reported, key=lambda n: fractions[n]['value']) if reported else None
    above_one = fractions['X_rank']['reported'] and fractions['X_rank']['value'] > 1
    mean_error = {s: 100 - float(np.mean([means[(s, 'rank', d)] for d in DATASETS])) for s in ('A', 'F')}
    a_minus_f = families['A-F[rank]']['mean_difference']
    out = {'no_training_effect_shown': bool(no_effect), 'headline': headline,
           'headline_value': fractions[headline]['value'] if headline else None,
           'word': 'no training effect shown' if no_effect else word(fractions[headline]['value']) if headline else None,
           'x_rank_above_one': bool(above_one), 'error_A_rank': mean_error['A'], 'error_F_rank': mean_error['F'],
           'A_minus_F_rank': a_minus_f, 'sentences': []}
    say = out['sentences'].append
    if no_effect:
        say('With a standard initialization, the untrained network is about as accurate as the trained one. The earlier '
            'gain over the untrained network came from its initialization, not from learning. (The 17-of-17 claim would '
            'then have to be withdrawn.)')
    elif headline is None:
        say('No recovery fraction is reported under the rule of the design; only the differences are.')
    else:
        f = fractions[headline]
        x = share(f['value'])
        if out['word'] == 'most':
            k = (r2 if headline == 'X_rank' else e2e)['significantly_higher']
            say(f"Trained through the exact sort, the encoder recovers {x} (95 % interval {share(f['ci_low'])} to "
                f"{share(f['ci_high'])}) of the accuracy that backpropagation with cross-entropy adds to the same network "
                f"from a standard initialization. Backpropagation remains significantly more accurate on {k} of 17 "
                'datasets, mainly those where one class filter per class cannot represent a class. (The last clause is '
                'the template\'s; check it against the per-dataset table.)')
        elif out['word'] == 'about half':
            say(f"The ranking signal trains the encoder (more accurate than the untrained network on {r1['higher']} "
                f"datasets, significantly on {r1['significantly_higher']}), but it recovers only about half ({x}) of what "
                'backpropagation achieves on the same network.')
        else:
            say(f'The signal does train an encoder through an exact sort, but it recovers only {x} of the gain from '
                'backpropagation. We present it as an existence result, not as a competitive training method.')
    if above_one and not no_effect:
        x_e2e = share(fractions['X_e2e']['value']) if fractions['X_e2e']['reported'] else 'a fraction that is not reported'
        say('Read by the class filters, the encoder trained by the ranking signal is at least as accurate as the same '
            'network trained by backpropagation, as expected, since only the ranking signal is shaped for that readout. '
            f"Read by its own linear head, the backpropagated network is ahead by {e2e['mean_difference']:.1f} points, "
            f'and the ranking signal recovers {x_e2e} of that gap.')
    say(f"With a standard initialization the untrained network's error is {mean_error['A']:.1f} %, against "
        f"{mean_error['F']:.1f} % with the code's initialization (A - F = {a_minus_f:+.1f} points of accuracy, paired).")
    return out


def analyse(out_dir, nested_dir, mlp_root=RUNS_ROOT):
    out_dir, nested_dir = guard_output(out_dir), Path(nested_dir).resolve()
    if out_dir == nested_dir:
        raise SystemExit('--output and --nested name the same directory')
    acc = {}
    for d in DATASETS:
        _, y, splits = load(d)
        tests = {(s['outer_repeat'], s['outer_fold']): np.asarray(s['test']) for s in splits}
        for s, readout in SERIES:
            if s == 'MLP':
                acc[(s, readout, d)] = mlp_fold_accuracy(mlp_root, d, y, tests)
            else:
                run_dir = out_dir if s in ('A', 'B', 'C') else nested_dir
                acc[(s, readout, d)] = fold_accuracy(run_dir, d, ARM_OF[s], readout, y, tests)
    means = {k: 100 * float(np.mean(list(v.values()))) for k, v in acc.items()}
    series = lambda s, readout: {d: acc[(s, readout, d)] for d in DATASETS}
    families = {name: {'first': a, 'second': b, 'role': role, **family(series(*a), series(*b))}
                for name, (a, b, role) in CONTRASTS.items()}
    verified = verify_nested(nested_dir, means, family(series('T', 'rank'), series('F', 'rank')))
    gap = lambda a, b: {d: means[(*a, d)] - means[(*b, d)] for d in DATASETS}
    fractions, held_out, per_dataset = {}, {}, {}
    for name, (num, den, gate) in FRACTIONS.items():
        f = recovery(gap(*num), gap(*den), DATASETS)
        reasons = ([f"the mean denominator is {f['denominator_mean']:+.2f} points, below {MIN_POINTS:g}"]
                   if f['denominator_mean'] < MIN_POINTS else [])
        if families[gate]['significantly_higher'] < MIN_SIGNIFICANT:
            reasons.append(f"{gate} is significantly positive on {families[gate]['significantly_higher']} datasets, "
                           f'fewer than {MIN_SIGNIFICANT}')
        fractions[name] = {**f, 'numerator': num, 'denominator': den, 'gate': gate, 'reported': not reasons,
                           'not_reported_because': reasons}
        sub = recovery(gap(*num), gap(*den), HELD_OUT)
        held_out[name] = {**sub, 'shown': fractions[name]['reported'] and sub['denominator_mean'] >= MIN_POINTS}
        top, bottom = gap(*num), gap(*den)
        per_dataset[name] = {d: {'numerator': top[d], 'denominator': bottom[d],
                                 'ratio': top[d] / bottom[d] if bottom[d] >= MIN_POINTS else None} for d in DATASETS}
    result = {'design': 'nested_reference.py header, fixed 2026-11-24', 'output': str(out_dir), 'nested': str(nested_dir),
              'mlp_root': str(Path(mlp_root).resolve()), 'code_revision': read_revision(out_dir),
              'nested_code_revision': read_revision(nested_dir), 'nested_analysis_verified': verified,
              'arms': ARM_OF, 'means': {f'{d}|{ARM_OF[s]}|{r}': means[(s, r, d)] for s, r in SERIES for d in DATASETS},
              'contrasts': families, 'fractions': fractions, 'fractions_held_out': held_out,
              'per_dataset_fractions': per_dataset, 'reading': reading(families, fractions, means)}
    (out_dir/'analysis.json').write_text(json.dumps(result, indent=1) + '\n')
    text = analysis_markdown(result, means)
    (out_dir/'ANALYSIS.md').write_text(text)
    write_tex_tables(out_dir/'tables', result, means)
    print(text)


def read_revision(run_dir):
    path = Path(run_dir)/'code_revision.txt'
    return path.read_text().strip() if path.exists() else None


def verify_nested(nested_dir, means, t_minus_f):
    """T and F as recomputed here must equal the nested run's own analysis (its means and its contrast P1)."""
    path = nested_dir/'analysis.json'
    if not path.exists():
        return False
    held = json.loads(path.read_text())
    for s in ('T', 'F'):
        for readout in ('rank', 'knn'):
            for d in DATASETS:
                if abs(held['means'][f'{d}|{ARM_OF[s]}|{readout}'] - means[(s, readout, d)]) > 1e-9:
                    raise SystemExit(f'{path}: {d} {ARM_OF[s]} {readout} differs from its recomputation')
    mine = {r['dataset']: r for r in t_minus_f['rows']}
    for row in held['contrasts']['P1']:
        if any(abs(row[k] - mine[row['dataset']][k]) > 1e-9 for k in ('mean_difference', 'p', 'p_holm')):
            raise SystemExit(f'{path}: P1 on {row["dataset"]} differs from its recomputation')
    return True


# ------------------------------------------------------------------------------------------------ reports

def label(s, readout):
    return s if readout is None else f'{s}[{readout}]'


def md_table(header, rows):
    return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def fraction_text(f):
    if f['value'] is None:
        return 'undefined'
    return f"{f['value']:.3f} [{'-inf' if f['ci_low'] is None else format(f['ci_low'], '.3f')}, " \
           f"{'inf' if f['ci_high'] is None else format(f['ci_high'], '.3f')}]"


def analysis_markdown(result, means):
    fam, fr, ho, rd = result['contrasts'], result['fractions'], result['fractions_held_out'], result['reading']
    names = {s: ARM_OF[s] for s in ARM_OF}
    lines = ['# Two references for the learned encoder: analysis', '',
             f"Run `{result['output']}` (code revision {result['code_revision']}); nested run `{result['nested']}` "
             f"(revision {result['nested_code_revision']}; its own analysis reproduced: {result['nested_analysis_verified']}); "
             f"tuned MLP from `{result['mlp_root']}`. Design: the header of nested_reference.py (fixed 2026-11-24). "
             'Arms: ' + ', '.join(f'{s} = {n}' for s, n in names.items()) + '.', '',
             '## Reading (the pre-registered rule)', '',
             f"- R1 (C[rank] - A[rank]): higher on {fam['R1']['higher']}/17, significantly positive on "
             f"{fam['R1']['significantly_higher']}, significantly negative on {fam['R1']['significantly_lower']}; mean "
             f"{fam['R1']['mean_difference']:+.2f} points. "
             + ('**No training effect shown.**' if rd['no_training_effect_shown'] else 'A training effect is shown.')]
    for name, f in fr.items():
        status = 'reported' if f['reported'] else 'NOT reported: ' + '; '.join(f['not_reported_because'])
        lines.append(f"- {name} = {fraction_text(f)} (95 % interval; {f['undefined_resamples']} undefined resamples); "
                     f"mean numerator {f['numerator_mean']:+.2f}, mean denominator {f['denominator_mean']:+.2f} points; "
                     f'{status}.')
    lines.append(f"- Headline: {rd['headline'] or 'none'}"
                 + (f" = {rd['headline_value']:.3f}" if rd['headline'] else '') + f"; word: **{rd['word']}**"
                 + ('; X_rank is above 1.' if rd['x_rank_above_one'] else '.'))
    lines += ['', 'Wording (the design\'s templates, numbers filled in):']
    for sentence in rd['sentences']:
        lines += ['', f'> {sentence}']
    header = ['dataset'] + [label(s, r) for s, r in SERIES]
    rows = [[d] + [f'{means[(s, r, d)]:.1f}' for s, r in SERIES] for d in DATASETS]
    rows.append(['**mean of 17**'] + [f'{np.mean([means[(s, r, d)] for d in DATASETS]):.1f}' for s, r in SERIES])
    lines += ['', '## Mean outer-fold accuracy (%), 15 folds (fit seeds averaged)', ''] + md_table(header, rows)
    for name, f in fam.items():
        a, b = f['first'], f['second']
        lines += ['', f"## {name} ({f['role']}): {label(*a)} - {label(*b)} (points; Holm over 17)", '',
                  f"Mean {f['mean_difference']:+.2f}; higher on {f['higher']}/17; significantly higher on "
                  f"{f['significantly_higher']}, significantly lower on {f['significantly_lower']}.", '']
        lines += md_table(['dataset', 'difference', '95 % interval', 'p', 'Holm p', 'folds better'],
                          [[r['dataset'], f"{r['mean_difference']:+.2f}", f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]",
                            f"{r['p']:.4f}", f"{r['p_holm']:.4f}", f"{r['folds_better']}/15"] for r in f['rows']])
    lines += ['', '## Recovery fractions on the five datasets not used in developing the conversion', '']
    lines += md_table(['fraction', 'value [95 % interval]', 'mean numerator', 'mean denominator', 'shown'],
                      [[name, fraction_text(h), f"{h['numerator_mean']:+.2f}", f"{h['denominator_mean']:+.2f}",
                        str(h['shown'])] for name, h in ho.items()])
    per = result['per_dataset_fractions']
    lines += ['', '## Per-dataset gaps (points) and ratios (shown where the denominator is at least 1 point)', '']
    header = ['dataset', 'C-A [rank]', 'B-A [rank]', 'B[head]-A[rank]', 'C-A [knn]', 'B-A [knn]', 'X_rank', 'X_e2e',
              'X_knn']
    ratio = lambda v: '-' if v is None else f'{v:.2f}'
    rows = [[d, f"{per['X_rank'][d]['numerator']:+.2f}", f"{per['X_rank'][d]['denominator']:+.2f}",
             f"{per['X_e2e'][d]['denominator']:+.2f}", f"{per['X_knn'][d]['numerator']:+.2f}",
             f"{per['X_knn'][d]['denominator']:+.2f}"] + [ratio(per[n][d]['ratio']) for n in FRACTIONS] for d in DATASETS]
    return '\n'.join(lines + md_table(header, rows)) + '\n'


# LaTeX in the style of the manuscript's tab_s_learned_nested*.tex (render_tables.py's write_table, wide, scriptsize)
TEX_ORDER = ('iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits', 'balance_scale',
             'mfeat_zernike', 'ionosphere', 'vertebra_column', 'diabetes', 'banknote_authentication', 'qsar_biodeg',
             'steel_plates_fault', 'climate_model_simulation_crashes', 'hcv_egyptian_patients')
TEX_NAME = {'iris': 'Iris', 'wine': 'Wine', 'breast_cancer': 'Breast cancer', 'wine_quality': 'Wine quality',
            'vehicle': 'Vehicle', 'segment': 'Segment', 'digits': 'Digits',
            **{d: d.replace('_', '-') for d in TEX_ORDER[7:]}, 'hcv_egyptian_patients': 'HCV'}
TEX_BLOCKS = {0: 'The seven benchmark datasets', 7: 'The ten further datasets'}
TEX_WORDS = ('none', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
             'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen')


def tex_signed(v, d=2):
    s = f'{abs(v):.{d}f}'
    if s == '0.' + '0' * d:
        return f'$-${s}' if v < -1e-9 else f'+{s}'
    return f'$-${s}' if v < 0 else f'+{s}'


def tex_p(p):
    return '$<$0.001' if p < .001 else f'{p:.3f}'


def tex_share(x):
    return '$\\infty$' if x is None else f'{100 * x:.0f}'.replace('-', '$-$')


def two_lines(first, second):
    return f'\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{first}\\\\{second}\\end{{tabular}}'


def counts(f):
    higher = 'all seventeen datasets' if f['higher'] == 17 else f"{TEX_WORDS[f['higher']]} of the seventeen datasets"
    return (f"{higher}, significantly on {TEX_WORDS[f['significantly_higher']]}, and significantly behind on "
            f"{TEX_WORDS[f['significantly_lower']]}")


def write_tex(path, caption, label_, header, rows, colspec, subheads):
    lines = ['% Rendered by experiments/tensor_rank_sandbox/nested_reference.py (analyse) from run files; do not edit by '
             'hand.', '\\begin{table}[!htbp]', f'\\caption{{{caption}}}', f'\\label{{{label_}}}', '\\centering',
             '\\scriptsize', '\\setlength{\\tabcolsep}{3pt}',
             '\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%', f'\\begin{{tabular}}{{{colspec}}}',
             '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    for i, row in enumerate(rows):
        if i in subheads:
            if i:
                lines.append('\\midrule')
            lines.append(f'\\multicolumn{{{len(header)}}}{{l}}{{\\emph{{{subheads[i]}}}}} \\\\')
        lines.append(' & '.join(row) + ' \\\\')
    path.write_text('\n'.join(lines + ['\\bottomrule', '\\end{tabular}', '}', '\\end{table}']) + '\n')


def write_tex_tables(tables, result, means):
    tables.mkdir(exist_ok=True)
    fam, fr, ho = result['contrasts'], result['fractions'], result['fractions_held_out']
    err = lambda s, r, d: f'{100 - means[(s, r, d)]:.1f}'
    mean_err = lambda s, r: f'{100 - np.mean([means[(s, r, d)] for d in DATASETS]):.1f}'
    cell = lambda name, d: next(r for r in fam[name]['rows'] if r['dataset'] == d)
    diff = lambda name, d: (f"{tex_signed(cell(name, d)['mean_difference'])} [{tex_signed(cell(name, d)['ci_low'])}, "
                            f"{tex_signed(cell(name, d)['ci_high'])}]")
    interval_note = ('95\\% corrected resampled-$t$ interval (test-to-training ratio 0.25, 14 degrees of freedom); positive '
                     'favors the first-named model, and each contrast has its own Holm adjustment across the seventeen '
                     'datasets')
    columns = (('A', 'rank'), ('C', 'rank'), ('B', 'rank'), ('B', 'head'), ('T', 'rank'), ('F', 'rank'), ('MLP', None))
    head1 = ['Dataset', two_lines('Untrained,', 'standard init.'), two_lines('Ranking signal,', 'standard init.'),
             two_lines('Backprop.,', 'class filters'), two_lines('Backprop.,', 'linear head'),
             two_lines('Ranking signal,', 'code init.'), two_lines('Untrained,', 'code init.'), 'MLP']
    rows = [[TEX_NAME[d]] + [err(s, r, d) for s, r in columns] for d in TEX_ORDER]
    rows.append(['Mean over the seventeen'] + [mean_err(s, r) for s, r in columns])
    write_tex(tables/'tab_s_learned_reference.tex',
              '\\textbf{Two references for the learned encoder: error.} Mean error (\\%) over the 15 test sets of the main '
              'benchmark. The first six columns are the classifier of Section~\\ref{supp:learned-method}, an encoder '
              'network followed by one ranking layer of class filters, read by its class filters except in the fourth '
              'column. Its encoder is never trained, trained by the ranking signal through the exact sort, or trained by '
              'backpropagation with cross-entropy through a learned linear head, which the fourth column reads directly; '
              'the class filters of the backpropagated network are trained on it afterwards, with the network frozen. The '
              'first four columns start from the standard initialization (the default of a linear layer, uniform on '
              '$\\pm1/\\sqrt{\\text{fan-in}}$, in the first two layers), the next two from the code\'s (uniform on '
              '$[0,1)$), as in Table~\\ref{tab:s-learned-nested}. Each is tuned on the inner folds at the first fitting '
              'seed by its class-filter readout (the linear head by its own accuracy) and refitted with the three fitting '
              'seeds. The MLP is the registered result on the same test sets. HCV: hepatitis C virus.',
              'tab:s-learned-reference', head1, rows, 'lrrrrrrr', {**TEX_BLOCKS, 17: 'All datasets'})
    knn = (('A', 'knn'), ('C', 'knn'), ('B', 'knn'), ('T', 'knn'), ('F', 'knn'))
    head2 = ['Dataset', two_lines('Untrained,', 'standard init.'), two_lines('Ranking signal,', 'standard init.'),
             two_lines('Backprop.,', 'standard init.'), two_lines('Ranking signal,', 'code init.'),
             two_lines('Untrained,', 'code init.')]
    rows = [[TEX_NAME[d]] + [err(s, r, d) for s, r in knn] for d in TEX_ORDER]
    rows.append(['Mean over the seventeen'] + [mean_err(s, r) for s, r in knn])
    write_tex(tables/'tab_s_learned_reference_knn.tex',
              '\\textbf{Two references for the learned encoder: error of the nearest-neighbor readout.} Mean error (\\%) '
              'over the 15 test sets when the encoder\'s ranking is read by a footrule nearest-neighbor vote whose '
              'neighbors and weighting are chosen on the training rows. The configurations are those that the class-filter '
              'readout selected (Table~\\ref{tab:s-learned-reference}). HCV: hepatitis C virus.',
              'tab:s-learned-reference-knn', head2, rows, 'lrrrrr', {**TEX_BLOCKS, 17: 'All datasets'})
    primary = ('R1', 'R2', 'R3')
    head3 = ['Dataset', two_lines('Ranking signal $-$', 'untrained (pp)'), 'Holm $p$',
             two_lines('Backprop. $-$', 'ranking signal (pp)'), 'Holm $p$', two_lines('Backprop. $-$', 'untrained (pp)'),
             'Holm $p$']
    rows = [[TEX_NAME[d]] + [c for n in primary for c in (diff(n, d), tex_p(cell(n, d)['p_holm']))] for d in TEX_ORDER]
    write_tex(tables/'tab_s_learned_reference_contrasts.tex',
              '\\textbf{Two references for the learned encoder: the three contrasts fixed before the run.} Accuracy '
              f'differences, in percentage points, with the {interval_note}. All three read the classifier of '
              'Section~\\ref{supp:learned-method} by its class filters, and every encoder starts from the standard '
              f"initialization. The encoder trained by the ranking signal is ahead of the untrained one on {counts(fam['R1'])}. "
              f"Backpropagation is ahead of the ranking signal on {counts(fam['R2'])}. It is ahead of the untrained "
              f"network on {counts(fam['R3'])}. HCV: hepatitis C virus.",
              'tab:s-learned-reference-contrasts', head3, rows, 'lrrrrrr', TEX_BLOCKS)
    descriptive = ('A-F[rank]', 'C-T[rank]', 'B[head]-MLP')
    head4 = ['Dataset', two_lines('Standard $-$ code', 'init., untrained (pp)'), 'Holm $p$',
             two_lines('Standard $-$ code init.,', 'ranking signal (pp)'), 'Holm $p$',
             two_lines('Backprop. head', '$-$ MLP (pp)'), 'Holm $p$']
    rows = [[TEX_NAME[d]] + [c for n in descriptive for c in (diff(n, d), tex_p(cell(n, d)['p_holm']))]
            for d in TEX_ORDER]
    write_tex(tables/'tab_s_learned_reference_descriptive.tex',
              '\\textbf{Two references for the learned encoder: descriptive contrasts.} Accuracy differences, in '
              f'percentage points, with the {interval_note}. The first two contrasts change only the initialization of '
              'the first two encoder layers and read the class filters; the untrained pair is exactly paired. The third '
              'compares the backpropagated network, read by its own linear head, with the registered MLP. For the '
              f"untrained network, the standard initialization is ahead of the code's on {counts(fam['A-F[rank]'])}. For "
              f"the network trained by the ranking signal, it is ahead on {counts(fam['C-T[rank]'])}. The backpropagated "
              f"network is ahead of the MLP on {counts(fam['B[head]-MLP'])}. HCV: hepatitis C virus.",
              'tab:s-learned-reference-descriptive', head4, rows, 'lrrrrrr', TEX_BLOCKS)
    formula = {'X_rank': ('$C_{\\mathrm{rank}}-A_{\\mathrm{rank}}$', '$B_{\\mathrm{rank}}-A_{\\mathrm{rank}}$'),
               'X_e2e': ('$C_{\\mathrm{rank}}-A_{\\mathrm{rank}}$', '$B_{\\mathrm{head}}-A_{\\mathrm{rank}}$'),
               'X_knn': ('$C_{\\mathrm{kNN}}-A_{\\mathrm{kNN}}$', '$B_{\\mathrm{kNN}}-A_{\\mathrm{kNN}}$')}

    def shown(f, ok):
        if not ok:
            return 'not reported'
        lo = '$-\\infty$' if f['ci_low'] is None else tex_share(f['ci_low'])
        return f"{tex_share(f['value'])} [{lo}, {tex_share(f['ci_high'])}]"
    rows = [[name.replace('_', '\\_'), *formula[name], tex_signed(fr[name]['numerator_mean']),
             tex_signed(fr[name]['denominator_mean']), shown(fr[name], fr[name]['reported']),
             shown(ho[name], ho[name]['shown'])] for name in FRACTIONS]
    head5 = ['Fraction', 'Numerator', 'Denominator', two_lines('Mean num.', '(pp)'), two_lines('Mean den.', '(pp)'),
             two_lines('Recovered (\\%),', 'seventeen datasets'), two_lines('Recovered (\\%),', 'five held out')]
    write_tex(tables/'tab_s_learned_reference_fractions.tex',
              '\\textbf{Two references for the learned encoder: the share of the backpropagation gain that the ranking '
              'signal recovers.} $A$ is the untrained encoder, $B$ the encoder trained by backpropagation with '
              'cross-entropy, and $C$ the encoder trained by the ranking signal, all from the standard initialization; '
              'subscripts name the readout. Each fraction is a ratio of sums over the datasets of mean accuracy '
              'differences, with the 95\\% percentile interval from 10,000 bootstrap resamples of the datasets. A fraction '
              'is not reported when the mean of its denominator is below one point or when its denominator is '
              'significantly positive on fewer than three datasets. The last column repeats the fractions on the five '
              'datasets not used in developing the target conversion (wine quality, banknote-authentication, '
              'qsar-biodeg, climate-model-simulation-crashes and HCV), where a fraction is shown only if it is reported '
              'on all seventeen and its mean denominator over the five is at least one point.',
              'tab:s-learned-reference-fractions', head5, rows, 'lllrrrr', {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'run', 'analyse'))
    parser.add_argument('--output')
    parser.add_argument('--nested', help='the nested study run (default for check: the registered run)')
    parser.add_argument('--mlp', default=str(RUNS_ROOT), help='the directory of the registered runs of the tuned MLP')
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--only', action='append', default=[], help='run only this job, e.g. iris__ref_untrained_std__i00 '
                                                                     '(repeatable; for tests)')
    parser.add_argument('--allow-uncommitted', action='store_true', help='run uncommitted code (for tests only)')
    args = parser.parse_args()
    if args.command == 'check':
        check(args.nested or NESTED_RUN)
    elif args.command == 'run':
        if not args.output:
            parser.error('run needs --output')
        run(args.output, args.workers, args.only, args.allow_uncommitted)
    else:
        if not args.output or not args.nested:
            parser.error('analyse needs --output and --nested')
        analyse(args.output, args.nested, args.mlp)


if __name__ == '__main__':
    main()
