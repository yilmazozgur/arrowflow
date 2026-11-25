"""DESIGN (fixed 2026-11-25 before any score of this arm; committed before the run)

The permuted-label control of ArrowFlow's ranking layers (specification: manuscript/TMLR/handoff/COST_permuted_labels.md,
Sections 2, 3 and 5; the author chose its fuller version, option B, on 2026-11-25). It is one more arm, PL, of the matched
motion controls of motion_controls.py (run 2026-09-14-motion, protocol arrowflow-v3-motion-controls-1). AF is ArrowFlow,
the registered arm views7; FR is the registered frozen-filter arm. D = AF - PL, S = PL - FR and G = AF - FR, so that
G = D + S on every dataset. The design is fixed after the outer results of AF, FR and the two scrambles are known; only PL
is scored blind. The frozen protocol protocols/2026-11-25/permuted_labels.json holds this design as data (design() below);
its sha256 is pinned in PROTOCOL_SHA256, and run and analyse refuse any other file. The analysis is compare_labels.py.

What stays identical to the motion study. The registered data, splits and 15 outer folds of the 17 datasets (the motion
run's data.npz, splits.json, planned_jobs.json and reference_selections.json, read in place and hash-checked); ArrowFlow-
kNN's sealed per-fold selection (widths, embedding width, degree, learning rate, augmentation, seven views, majority vote),
not re-tuned for PL; the fitting seeds 8129, 19391 and 39019; the view seeds, the encoders, the seeded initial filters and
the validation rows; eta with its multistep schedule, T = 200 updates, batches of 32, rho = 0.5, c = 0.125 and
p_correct = 0.01; the Borda update and the earlier relay on the 70 two-layer folds, as registered; the checkpoint rule and
the readout rule. Every network is trained step for step as motion_controls.arm_fit trains it, with an identity MotionArm
('views7', which draws nothing) installed on each view network for the vote statistics and the RNG guards, and a
TrainingRecorder of training_diagnostics.py, subclassed and not edited, that copies the filters. Nothing under arrowflow/
and nothing in an imported module changes. By construction PL's batch sequence diverges after its first batch: the gate
draws a random number only for rows whose prediction is correct, so the numpy stream advances differently, as in every
motion-control arm. The first batch's rows are AF's.

The arm. Per outer fold and fitting seed, one uniform permutation pi of the rows of the outer training partition is drawn
from a private stream, np.random.RandomState(derive_seed(20261125, 'permuted_labels', dataset, outer_repeat, outer_fold,
seed)), inside an RNG guard, and the network labels are y_net = y_train[pi]. That gives 45 permutations per dataset. It
covers the whole partition, including the 10 % of rows that train_initialized's seeded shuffle then holds out for the
checkpoint (the same rows as AF's, because the shuffle depends only on the view seed and the row count, not on the
labels); the seven views share it; it keeps the class counts exactly. PL fits MultiViewArrowFlowKNN at the job's sealed
selection exactly as arm_fit does, except that each view network's initialize_orders and train_initialized receive y_net,
while OrdinalEncoder.fit (the three views with linear-discriminant coordinates, views 1, 4 and 7, included) and
_fit_view_readout receive the true labels. So the gate (an error, or a correct row with p_correct), the class layer's vote
(the label's class filter moves toward the hidden ranking), the motions that this sends to the hidden layers and the relay,
and the checkpoint's validation error read y_net. The encoder, the kNN readout's stored training labels (validation rows
included), its choice of neighbours and weighting (select_knn_readout at derive_seed(view seed, 'readout_selection')), the
scoring, the purity and the per-class recall read the true labels. The gate is not matched to AF's vote volume.

Every job retrains AF (option B). A job is one dataset and outer fold. For each fitting seed in order, AF is first fitted by
the same code with y_net = y_train. Its seven-view kNN predictions on the outer test rows must equal the registered views7
predictions exactly; otherwise the job stops before PL is fitted, writes no record, and the run stops. AF's checkpoint
k*_v of view v is the core's own: the last strict improvement of the class-filter readout's validation error, ties going
to the earliest, as the recorder reads it. PL is then fitted and read at three points of its one training run:
  PL[match]  the filters right after update k*_v of AF at the same fold, fitting seed and view (primary). AF and PL then
             train for the same number of updates and differ only in the labels inside the signal; what PL borrows from
             AF is one integer per view, the training length;
  PL[own]    PL's own checkpoint, the network that train() restores, chosen by the validation error under y_net (the
             pre-registered secondary; the referee's literal version);
  PL[final]  the filters after update 200 (descriptive).
One run gives all three, because the trajectory does not depend on which state the checkpoint keeps: the validation pass
draws no random number (the gate's draw is skipped when evaluate_only), the learning rate depends only on the update count,
train() restores the checkpoint only after the last update, and every copy is RNG-guarded. Each readout is the method's
own: the training rows' last hidden ranking under the copied filters (training_diagnostics.network_forward), the readout
chosen by select_knn_readout with the true labels, a StableFootruleKNN refitted on all training rows, the test rows' ranking
under the same filters, and the majority vote of the seven views.

Reused, not refitted. AF's and FR's outer predictions are the registered motion-run predictions. AF's refit must equal them
in every job, and serves only k* and the checks. FR is refitted only in the label-routing check below. During the run the
jobs read the registered predictions only for those checks, as counts of differing rows; the analysis reads them as inputs
only after its completeness gate. Variant 2 of the specification (the encoder trained on the permuted labels too) is not
part of this design.

The test, for every contrast. Per dataset, the fold accuracy averaged over the three fitting seeds, the differences paired
over the 15 outer folds, the corrected resampled t (test/train ratio 0.25, 14 degrees of freedom, two-sided, 95 %;
evaluation.paired_corrected_interval), and Holm's correction across the 17 datasets within each family
(evaluation.holm_adjust). A dataset is significantly positive (negative) when its mean difference is above (below) zero
and its Holm-adjusted p is below 0.05.

Families.
  F1 (primary)  D[match] = AF - PL[match].
  F2            S[match] = PL[match] - FR, a separate family, as the motion study adjusted each contrast separately (one
                family of 34 would be the conservative alternative; it is not used).
  F3, F4        D[own] = AF - PL[own] and S[own] = PL[own] - FR, the secondary families, each with its own Holm.
  G             AF - FR, the size of the gain in every reading. It is recomputed from the registered predictions and must
                equal the registered motion analysis (the frozen arm of its primary family) in every row.
  Named subset  balance-scale and mfeat-zernike: the rows of F1 and F2 with their own Holm p, not adjusted again.

Across datasets, for D and S of both readouts and for G: the mean of the 17 per-dataset mean differences with its 95 % t
interval (16 degrees of freedom); the exact two-sided Wilcoxon signed-rank p over the 17 per-dataset mean differences (a
zero difference is dropped, tied absolute differences take their average rank, and p is counted exactly over all 2^n sign
assignments, which equals scipy's exact test when there is no zero and no tie); and the counts of positive and negative
datasets and of Holm-significant datasets in each direction. The same numbers, descriptive only, on the ten further
datasets and on the seven development (benchmark) datasets, the Holm p being the 17-dataset family's. The label share
sum(D)/sum(G), a ratio of sums over the 17 datasets, and its complement X = sum(S)/sum(G), each with a 95 % percentile
interval from 10,000 bootstrap resamples of the 17 datasets, drawn once as np.random.default_rng(20261125).integers(0, 17,
(10000, 17)) and interpolated linearly between order statistics; a resample whose sum(G) is zero or negative counts as
minus infinity for the lower and plus infinity for the upper end, and an infinite end is reported as unbounded. They are
reported only in pattern (b).

Reading rule, applied to F1 and F2 (PL[match]), and separately to F3 and F4 (PL[own]). A contrast is positive (negative)
when its across-dataset signed-rank p is below 0.05 and its positive (negative) rank sum is the larger, and n.s.
otherwise. Patterns, checked in this order:
  (f) PL beats ArrowFlow       D negative, whatever S
  (d) PL worse than frozen     D positive, S negative
  (c) PL equal to frozen       D positive, S n.s.
  (b) between the two          D positive, S positive
  (a) PL as good as ArrowFlow  D n.s., S positive
  (e) not attributable         D n.s., S n.s.; and D n.s. with S negative, a cell the specification's table does not list
The five wordings (WORDINGS below) are filled in with: k, the datasets with D > 0; m, the datasets significantly positive
in D; s, the mean of S, with l to u its 95 % t interval; x and y, the mean normalized footrule displacement of the last
hidden layer from its initial filters at PL's readout and at ArrowFlow's checkpoint, averaged over views, fitting seeds,
folds and then datasets; X, sum(S)/sum(G) with its interval; and, in (a), k_s, the datasets with S > 0. In every pattern the
size of the gain is quoted as G with its counts (G_SENTENCE); D is evidence of label dependence, not a size, and in pattern
(d) D includes PL's damage (D_DAMAGE). Pattern (f) is reported as found and the training claim is withdrawn. If F3 and F4
give a different pattern from F1 and F2, both readings are reported and the difference is attributed to the stopping rule
(STOPPING). Interpretation, fixed in advance: "If ArrowFlow does not beat its own pipeline trained on permuted labels, the
paper says so and withdraws the claim that the gain depends on the labels in the signal. No arm, dataset, readout or
subset is added, dropped or reweighted after a PL score is seen." The reading states the first sentence (WITHDRAW) whenever
D[match] is not positive.

Descriptive measures, declared in advance; no test and no reading attaches to them.
- Vote volume: per dataset and hidden layer, the share of selected slots that carry a nonzero vote (as in the motion
  statistics table), over the whole run and up to the matched stop, for AF and PL, and PL's nonzero votes over AF's.
- Displacement from the initial filters per layer (mean normalized footrule and share of filters that differ, averaged
  over the views) at PL[match], PL[own] and PL[final], and for AF at its checkpoint and after update 200.
- The checkpoint distribution: per dataset, the median, the interquartile range and the share at update 0 of PL's own
  checkpoint and of AF's, over folds, fitting seeds and views.
- Neighbourhood purity of the last hidden ranking (motion_controls.neighborhood_purity at the view's selected k, on the outer
  test rows and on the training rows as in the motion study) at the three readouts, beside AF's and FR's registered purity;
  and the per-class recall of every arm and readout, including balance-scale's level-balance class.
- PL[final]'s accuracy, with PL[final] - FR and AF - PL[final].
- For each permutation, the share of training rows that keep their true label (its expectation is the sum of the squared
  class shares) and Cramer's V between the permuted and the true labels.
- The per-view split by encoder type: the mean per-view kNN accuracy over the three LDA views and over the four others, for
  AF, FR (the readout at update 0), PL[match], PL[own] and PL[final], and D and S within each group. If D and S look alike
  on the four label-free views, the result does not rest on the LDA views.

Checks, in every job; a failed check fails its job, which writes no record, and stops the run.
- inputs: the pinned sha256 of the registered motion and diagnostics files; the job's registered record, prediction file,
  identity (dataset, split, training rows and labels, selection) and the diagnostics run's job identity agree.
- core_sources: the pinned sha256 of the core functions the transcription depends on (motion_controls.CORE_SOURCES).
- reference_predictions: AF's refit reproduces the registered views7 predictions, every seed (checked before PL).
- registered_state: AF's view state hashes, initial filters, readout choices and displacement equal the registered
  views7 fit's, every seed; AF's vote statistics over the job equal the registered views7 statistics.
- checkpoints: AF's checkpoint per view at seed 8129 equals the diagnostics run's checkpoints.csv.
- initial_state: PL's (and the label-routing refit's) seeded initial filters equal the registered views7 filters.
- update0_frozen: the readout of the initial filters, with the true labels, reproduces FR's registered predictions.
- label_routing: the encodings of PL equal AF's in every view (training and test rows); PL's networks received a
  permutation of the true labels with equal class counts; every readout of PL stores the true training labels; PL's first
  batch holds AF's rows and its validation rows are AF's; the permutation draw left the global RNGs unchanged.
- snapshot_readout: the copy taken at PL's own checkpoint reproduces the live network's kNN predictions, view by view, and
  its purity; every matched stop was copied.
- frozen_permuted (first outer fold of each dataset, every seed): the frozen arm trained with the permuted labels
  reproduces FR's registered predictions, and its hidden filters never moved.
- uninstrumented_fit (first outer fold, first seed): AF's view state hashes and predictions equal those of an
  uninstrumented MultiViewArrowFlowKNN fit.
- training_record, rng_untouched, wrapper_removed: every recorder's record verifies, every wrapper and recorder read left
  the global numpy and Python RNG states unchanged, and no instance-level method survives a fit.

Blindness. A job record holds per-row predictions, purity and the descriptive counts, never an accuracy or an error; PL's
validation errors and the readouts' inner scores are not stored; run.log holds job names and seconds only; 'check' prints
timings and counts of differing predictions, hashes or views only.

    python -m experiments.make_revision.label_controls check
    python -m experiments.make_revision.label_controls run --output DIR [--workers 24]
    python -m experiments.make_revision.label_controls analyse --output DIR [--tables DIR]
    python -m experiments.make_revision.label_controls draft         (writes the protocol from design(); before the commit)
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import functools
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
import numpy as np
from threadpoolctl import threadpool_limits
from arrowflow.ranking import inverse_positions
from . import motion_controls as mc
from . import run_knn_ablation as base
from . import training_diagnostics as td
from .comparisons import StableFootruleKNN, derive_seed
from .evaluation import config_id
from .models import ArrowFlowEstimator, OrdinalEncoder, array_hash, seed_fit
from .multiview import MultiViewArrowFlowKNN, select_knn_readout, view_strategy
from .newdata import WORKSPACE_RUNS, sha256_file
from .run_revision import execution_lock, load_prepared

REPO = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name('protocols')/'2026-11-25'/'permuted_labels.json'
PROTOCOL_ID = 'arrowflow-v3-permuted-labels-1'
PROTOCOL_SHA256 = 'd2c36af937a37c7d1fa2b42900e0fe718cb4325ddc6332efa7aebd680dc55e37'
SPECIFICATION = 'manuscript/TMLR/handoff/COST_permuted_labels.md (Sections 2, 3 and 5); option B, the fuller version'
SOURCE_PATHS = ('arrowflow', 'experiments/make_revision')
MOTION_RUN = WORKSPACE_RUNS/'2026-09-14-motion'/'run'
MOTION_ANALYSIS = WORKSPACE_RUNS/'2026-09-14-motion'/'analysis'
DIAGNOSTICS_RUN = WORKSPACE_RUNS/'2026-09-14-training-diagnostics'
# sha256 of the registered files this design reads, computed on 2026-11-25 before any fit of the arm
PINNED = {
    'motion/protocol.json': 'd31d5d39a94bde79c35a9b258a0e94a725a4c1788f440085452624f97a65c454',
    'motion/manifest.json': 'f89bc0783d2ebc876d1fda2bc72a932ccf49bdaaa4635f447ddd25c563b9d008',
    'motion/planned_jobs.json': '06f4e9a2af5dc1cb4292df0d64b76a8939949b136d7cb23c51c59b69a7f79d87',
    'motion/reference_selections.json': '535c203238deef743119fe8bd0068d16db5d26d5f87008fe814da0f4eb44c327',
    'motion/environment.json': '6d38adc4545f623c337a21686b73342adc8f7aab044e9c4cc966c985543ed70d',
    'motion/motion_controls_summary.json': '8f693fe35f5f0efbb5a3c5b61b64fc7c174a086d0ad8a7efff6aaba81cfd1e80',
    'motion_analysis/motion_analysis.json': 'd3f6d809751be57ebb6b9fafae76b1b7228413f24c866045f5c87e0f9b80a8b5',
    'diagnostics/provenance.json': '8a558a4f9ba8b69c78783ecc8a025b7234d34b117b46fba5826b3b4f1ef2e647',
    'diagnostics/checkpoints.csv': '8ff8761fb9d16c3d0324a8efca45bd8d844e21a2e98527ecbbbc679978e0cfe9'}
PINNED_ROOTS = {'motion': MOTION_RUN, 'motion_analysis': MOTION_ANALYSIS, 'diagnostics': DIAGNOSTICS_RUN}
# one sha256 over 'results|predictions|logs/<name> <sha256>' lines of every per-job file of the motion run, in name order
MOTION_FILES_DIGEST = '1cb92ea5c8287505e1dd853daf9c56f35b87347e21e0cee9f62e36b4d90d05b2'
MOTION_FILES = (('results', '*.json'), ('predictions', '*.jsonl'), ('logs', '*.jsonl'))
# registered runs that no output of this module may lie in
PROTECTED = tuple(WORKSPACE_RUNS/name for name in (
    '2026-09-14-motion', '2026-09-14-training-diagnostics', '2026-09-12-bridge-knn', '2026-09-13-knn-ablation',
    '2026-09-14-newdata-batch1', '2026-09-14-newdata-batch2', '2026-09-14-newdata-ablation', '2026-09-24-tensor-rank-nested',
    '2026-09-25-encoder-swap', '2026-11-24-encoder-reference'))

FIT_SEEDS = (8129, 19391, 39019)
DIAGNOSTICS_SEED = 8129
LABEL_SEED = 20261125
LABEL_STREAM = 'permuted_labels'
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES, CONFIDENCE = 20261125, 10_000, 95
TEST_RATIO, ALPHA = .25, .05
READOUTS = ('match', 'own', 'final')
LDA_VIEWS = (0, 3, 6)                       # CYCLE index 0 (target_aware): views 1, 4 and 7 counted from one
NAMED_SUBSET = ('balance_scale', 'mfeat_zernike')
BENCHMARK = ('iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits')
WORKERS, MAX_WORKERS = 24, 24
LOAD_LIMIT = 4.     # run refuses to start above this one-minute load average (another study holds the machine)
CHECK_SAMPLE = ('wine__r0f0', 'balance_scale__r0f0')          # two layers without augmentation; one layer with it
CHECK_TIMING = ('balance_scale__r0f0', 8129)
# PL's projected training time over views7's (the cost estimate's central P-core seconds); used to order the jobs only
PL_COST = {'iris': 2.51, 'wine': 1.98, 'breast_cancer': 2.44, 'wine_quality': 1.20, 'vehicle': 1.50, 'segment': 1.79,
           'digits': 1.83, 'balance_scale': 1.96, 'mfeat_zernike': 1.64, 'ionosphere': 1.46, 'vertebra_column': 1.68,
           'diabetes': 1.03, 'banknote_authentication': 1.58, 'qsar_biodeg': 1.16, 'steel_plates_fault': 1.40,
           'climate_model_simulation_crashes': 1.18, 'hcv_egyptian_patients': 1.01}

WORDINGS = {
    'd': "Trained with permuted labels, the same networks are less accurate than ArrowFlow on {k} of 17 datasets "
         "(significantly on {m}), and less accurate than networks whose hidden filters never move, by {abs_s} points on "
         "average. A consistent target without the labels' content does not help the readout; it harms it. The gain over "
         "frozen filters therefore needs the labels in the signal.",
    'c': "Trained with permuted labels, the hidden filters move (displacement {x} against {y} for ArrowFlow), but the "
         "networks are as accurate as networks whose filters never move (S = {s}, interval {l} to {u}). The gain over "
         "frozen filters needs the labels.",
    'b': "Part of the gain does not need the labels. With permuted labels the networks beat frozen filters by {s_points} "
         "points, {x_share} ({x_interval}) of ArrowFlow's gain; the labels add the other {d_points} points.",
    'a': "The labels in the signal are not what helps: trained with permuted labels, the networks are as accurate as "
         "ArrowFlow and more accurate than frozen filters on {k_s} datasets. The gain is a label-free adaptation of the "
         "hidden code.",
    'e': "The permuted-label control cannot apportion the one-point gain between the labels and the structure of a "
         "consistent target.",
    'f': "Reported as found: trained with permuted labels, the networks are more accurate than ArrowFlow (D = {d} points "
         "on average, signed-rank p = {p_d}), and the training claim is withdrawn."}
G_SENTENCE = ("The size of the gain is G = {g} points (ArrowFlow minus frozen filters, positive on {k_g} of 17 datasets, "
              "significantly on {m_g}); D is evidence of label dependence, not a size.")
D_DAMAGE = 'In this pattern D includes the damage that training on permuted labels does (D = G - S).'
WITHDRAW = ('ArrowFlow does not beat its own pipeline trained on permuted labels: the paper says so and withdraws the claim '
            'that the gain depends on the labels in the signal.')
STOPPING = ("Read at the arm's own checkpoint (F3 and F4), the pattern is ({secondary}) rather than ({primary}); both are "
            'reported, and the difference is attributed to the stopping rule.')
INTERPRETATION = ('If ArrowFlow does not beat its own pipeline trained on permuted labels, the paper says so and withdraws '
                  'the claim that the gain depends on the labels in the signal. No arm, dataset, readout or subset is '
                  'added, dropped or reweighted after a PL score is seen.')
JOB_CHECKS = ('inputs', 'core_sources', 'reference_predictions', 'registered_state', 'checkpoints', 'initial_state',
              'update0_frozen', 'label_routing', 'snapshot_readout', 'frozen_permuted', 'uninstrumented_fit',
              'training_record', 'rng_untouched', 'wrapper_removed')


class CheckFailed(RuntimeError):
    """A check of a permuted-label job failed; the job writes no record and the run stops."""


# ----------------------------------------------------------------------------- the design as data

def design():
    """The frozen design (the module header in data form); the protocol file must equal it."""
    return {
        'protocol_id': PROTOCOL_ID, 'fixed': '2026-11-25, before any score of the permuted-label arm',
        'specification': SPECIFICATION, 'module': 'experiments/make_revision/label_controls.py',
        'analysis_module': 'experiments/make_revision/compare_labels.py',
        'joins': {'study': 'matched motion controls (motion_controls.py)', 'run': '2026-09-14-motion',
                  'protocol_id': mc.PROTOCOL_ID},
        'names': {'AF': 'ArrowFlow, the registered arm views7', 'FR': 'the registered frozen-filter arm',
                  'PL': 'the permuted-label arm', 'D': 'AF - PL', 'S': 'PL - FR', 'G': 'AF - FR = D + S'},
        'identical': ['the registered data, splits and 15 outer folds of the 17 datasets', 'the sealed per-fold selection',
                      'the fitting seeds 8129, 19391 and 39019', 'the view seeds, encoders, seeded initial filters and '
                      'validation rows', 'eta with its multistep schedule, T = 200, B = 32, rho = 0.5, c = 0.125, '
                      'p_correct = 0.01', 'the Borda update and the earlier relay on the two-layer folds',
                      'the checkpoint rule and the readout rule'],
        'diverges_by_construction': 'the batch sequence after the first update (the gate draws for correct rows only); '
                                    'the first batch holds AF\'s rows',
        'permutation': {'kind': 'uniform permutation of the rows of the outer training partition (class counts kept)',
                        'rows': 'the whole partition, the checkpoint validation rows included',
                        'granularity': 'one per outer fold and fitting seed, shared by the seven views',
                        'stream': f"np.random.RandomState(derive_seed({LABEL_SEED}, '{LABEL_STREAM}', dataset, "
                                  'outer_repeat, outer_fold, seed)).permutation(n_train), inside an RNG guard',
                        'network_labels': 'y_train[permutation]'},
        'labels': {'permuted': ['the gate', 'the class-filter prototype readout inside the gate', 'the class-layer vote',
                                'the motions to the hidden layers and the relay', 'the checkpoint validation error'],
                   'true': ['OrdinalEncoder.fit, the three LDA views included', 'the kNN readout\'s stored labels',
                            'select_knn_readout', 'the test scoring, purity and per-class recall']},
        'fit': 'MultiViewArrowFlowKNN at the sealed selection, step for step as motion_controls.arm_fit, with an identity '
               'MotionArm (views7) and a TrainingRecorder subclass on each view network',
        'af_refit': 'every job and seed, before PL; it must reproduce the registered views7 predictions exactly, or the '
                    'job stops and the run stops',
        'readouts': {'match': 'the filters after update k*_v, AF\'s checkpoint update of the same fold, seed and view '
                              '(primary)',
                     'own': 'PL\'s own checkpoint, the network train() restores (secondary)',
                     'final': 'the filters after update 200 (descriptive)',
                     'rule': 'network_forward of the training and test rows under the copied filters, select_knn_readout '
                             'with the true labels at derive_seed(view seed, "readout_selection"), StableFootruleKNN, '
                             'majority of the seven views'},
        'reused': {'AF': 'the registered motion-run views7 predictions', 'FR': 'the registered motion-run frozen '
                   'predictions'},
        'test': {'aggregation': 'fitting seeds averaged within each outer fold', 'method': 'corrected resampled t',
                 'test_train_ratio': TEST_RATIO, 'df': 14, 'confidence': .95, 'sides': 2,
                 'adjustment': 'Holm across the 17 datasets within each family', 'alpha': ALPHA},
        'families': {'F1': {'contrast': 'AF - PL[match]', 'role': 'primary'},
                     'F2': {'contrast': 'PL[match] - FR', 'role': 'separate family'},
                     'F3': {'contrast': 'AF - PL[own]', 'role': 'secondary'},
                     'F4': {'contrast': 'PL[own] - FR', 'role': 'secondary'},
                     'G': {'contrast': 'AF - FR', 'role': 'the size of the gain; must equal the registered motion '
                                                          'analysis'}},
        'named_subset': {'datasets': list(NAMED_SUBSET), 'families': ['F1', 'F2'], 'adjustment': 'the family\'s own'},
        'across_datasets': {'mean': 'mean of the 17 per-dataset mean differences, 95 % t interval with 16 df',
                            'signed_rank': 'exact two-sided Wilcoxon over the per-dataset mean differences; zeros '
                                           'dropped, ties at average ranks, exact over all 2^n sign assignments',
                            'counts': 'positive, negative, Holm-significant positive and negative',
                            'subsets': {'further': 'the ten further datasets', 'benchmark': list(BENCHMARK)},
                            'subset_status': 'descriptive only, with the 17-dataset Holm p'},
        'label_share': {'value': 'sum(D)/sum(G) over the 17 datasets', 'complement': 'X = sum(S)/sum(G)',
                        'bootstrap': f'np.random.default_rng({BOOTSTRAP_SEED}).integers(0, 17, '
                                     f'({BOOTSTRAP_RESAMPLES}, 17)), 95 % percentile interval, linear interpolation',
                        'undefined': 'sum(G) <= 0 counts as -inf (lower) and +inf (upper)',
                        'reported': 'only in pattern (b)'},
        'reading': {'direction': 'signed-rank p < 0.05, in the direction of the larger rank sum; else n.s.',
                    'readouts': {'primary': ['F1', 'F2'], 'secondary': ['F3', 'F4']},
                    'patterns': {'f': 'D negative', 'd': 'D positive, S negative', 'c': 'D positive, S n.s.',
                                 'b': 'D positive, S positive', 'a': 'D n.s., S positive',
                                 'e': 'D n.s., S n.s.; also D n.s., S negative (not in the specification\'s table)'},
                    'order': ['f', 'd', 'c', 'b', 'a', 'e'], 'wordings': WORDINGS, 'g_sentence': G_SENTENCE,
                    'd_damage': D_DAMAGE, 'withdraw': WITHDRAW, 'stopping': STOPPING,
                    'placeholders': {'k': 'datasets with D > 0', 'm': 'datasets significantly positive in D',
                                     's, l, u': 'mean of S and its 95 % t interval',
                                     'x, y': 'mean normalized footrule displacement of the last hidden layer from its '
                                             'initial filters, at PL\'s readout and at AF\'s checkpoint, averaged over '
                                             'views, seeds, folds, then datasets',
                                     'X': 'sum(S)/sum(G) with its bootstrap interval', 'k_s': 'datasets with S > 0'}},
        'interpretation': INTERPRETATION,
        'descriptive': ['vote volume per hidden layer (share of selected slots with a nonzero vote), over the run and up '
                        'to the matched stop, AF and PL, and PL over AF',
                        'displacement per layer at PL[match], PL[own], PL[final]; AF at its checkpoint and update 200',
                        'checkpoint distribution: median, IQR, share at update 0, PL[own] against AF',
                        'neighbourhood purity at the three readouts beside AF\'s and FR\'s registered purity; per-class '
                        'recall of every arm and readout',
                        'PL[final] accuracy, PL[final] - FR, AF - PL[final]',
                        'per permutation: share keeping the true label (expected sum of squared class shares), '
                        'Cramer\'s V',
                        'per-view split by encoder type (LDA views 1, 4, 7 against the four others): AF, FR, PL[match], '
                        'PL[own], PL[final], and D and S within each group'],
        'checks': list(JOB_CHECKS), 'failure_policy': 'a failed check or fit fails its job, which writes no record; the '
                                                      'pending jobs are cancelled and the run stops',
        'inputs': {'pinned_sha256': PINNED, 'motion_files_digest': MOTION_FILES_DIGEST},
        'seeds': {'fit': list(FIT_SEEDS), 'label_stream': LABEL_SEED, 'bootstrap': BOOTSTRAP_SEED},
        'parallelism': f'one spawned single-thread process per dataset and outer fold, {WORKERS} workers by default, '
                       'longest jobs first',
        'blindness': 'records hold per-row predictions, never an accuracy; PL\'s validation errors and the readouts\' '
                     'inner scores are not stored; run.log holds job names and seconds; check prints counts only'}


def protocol_text():
    return json.dumps(design(), indent=2, sort_keys=True, allow_nan=False) + '\n'


def verify_protocol():
    """The committed protocol file is this design, with the pinned sha256."""
    if not PROTOCOL.is_file():
        raise SystemExit(f'the frozen protocol {PROTOCOL} is missing')
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise SystemExit(f'{PROTOCOL.name}: its sha256 is not the pinned PROTOCOL_SHA256')
    if PROTOCOL.read_text() != protocol_text():
        raise SystemExit(f'{PROTOCOL.name} differs from design()')
    return json.loads(PROTOCOL.read_text())


def draft():
    """Write the protocol from design() (before the commit only) and print its sha256 for PROTOCOL_SHA256."""
    text = protocol_text()
    if PROTOCOL.exists() and PROTOCOL.read_text() != text:
        raise SystemExit(f'refusing to replace {PROTOCOL}: it differs from design(); remove it first if that is intended')
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    print(f'{PROTOCOL} sha256 {digest}' + ('' if digest == PROTOCOL_SHA256 else '  (set PROTOCOL_SHA256 to this value)'))


# ----------------------------------------------------------------------------- the registered inputs

def verify_pins(*, digest=True):
    """The registered files this design reads are unchanged."""
    for key, expected in PINNED.items():
        root, name = key.split('/', 1)
        path = PINNED_ROOTS[root]/name
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f'the registered file {path} is missing or changed')
    if digest and motion_files_digest() != MOTION_FILES_DIGEST:
        raise SystemExit(f'a per-job file of the registered motion run {MOTION_RUN} changed')


def motion_files_digest():
    h = hashlib.sha256()
    for sub, pattern in MOTION_FILES:
        for path in sorted((MOTION_RUN/sub).glob(pattern)):
            h.update(f'{sub}/{path.name} {sha256_file(path)}\n'.encode())
    return h.hexdigest()


def verify_environment():
    """The sealed sources and the library versions are the motion run's, so that AF can reproduce it."""
    saved, current = json.loads((MOTION_RUN/'environment.json').read_text()), mc.environment()
    if saved['source_hashes'] != current['source_hashes']:
        changed = sorted(k for k in set(saved['source_hashes']) | set(current['source_hashes'])
                         if saved['source_hashes'].get(k) != current['source_hashes'].get(k))
        raise SystemExit(f'sealed sources of the motion run differ from this tree: {changed}')
    differing = [k for k in ('python', 'numpy', 'scipy', 'sklearn', 'torch') if saved[k] != current[k]]
    if differing:
        raise SystemExit(f'the environment differs from the motion run in {differing}')
    source = mc.check_core_sources()
    if not source['passed']:
        raise SystemExit(f'pinned core functions changed: {source["changed"]}')
    return current


@functools.lru_cache(maxsize=1)
def motion_plan():
    """The motion run's planned jobs by stem and its sealed selections by (dataset, repeat, fold), hash-checked."""
    for name in ('planned_jobs.json', 'reference_selections.json'):
        if sha256_file(MOTION_RUN/name) != PINNED[f'motion/{name}']:
            raise CheckFailed(f'inputs: the registered {name} changed')
    jobs = json.loads((MOTION_RUN/'planned_jobs.json').read_text())
    sealed = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s
              for s in json.loads((MOTION_RUN/'reference_selections.json').read_text())}
    protocol = json.loads((MOTION_RUN/'protocol.json').read_text())
    if list(protocol['fit_seeds']) != list(FIT_SEEDS) or len(jobs) != 255 or len(sealed) != 255:
        raise CheckFailed('inputs: the motion plan is not 255 jobs at the three fitting seeds')
    return {'jobs': {job['stem']: job for job in jobs}, 'order': [job['stem'] for job in jobs], 'sealed': sealed,
            'datasets': list(protocol['datasets'])}


def job_stems():
    return list(motion_plan()['order'])


@functools.lru_cache(maxsize=1)
def diagnostics_checkpoints():
    """{(dataset, repeat, fold, view): row} of checkpoints.csv at seed 8129."""
    path = DIAGNOSTICS_RUN/'checkpoints.csv'
    if sha256_file(path) != PINNED['diagnostics/checkpoints.csv']:
        raise CheckFailed('inputs: the diagnostics checkpoints.csv changed')
    out = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            key = (row['dataset_id'], int(row['outer_repeat']), int(row['outer_fold']), int(row['view']))
            if int(row['model_seed']) != DIAGNOSTICS_SEED or key in out:
                raise CheckFailed('inputs: checkpoints.csv holds another seed or a repeated row')
            out[key] = row
    return out


def registered_job(stem, y, test):
    """The motion run's record of one job, with the views7 and frozen predictions in the test order (file sha256 and
    prediction hashes verified)."""
    record = json.loads((MOTION_RUN/'results'/f'{stem}.json').read_text())
    if record.get('status') != 'ok' or any(c.get('passed') is False for c in record['checks'].values()):
        raise CheckFailed(f'inputs: the registered record {stem} is not a passing job')
    path = MOTION_RUN/record['prediction_file']['path']
    if sha256_file(path) != record['prediction_file']['sha256']:
        raise CheckFailed(f'inputs: the registered predictions of {stem} changed')
    cells = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['arm_id'] in ('views7', 'frozen'):
            if row['y_true'] != y[int(row['sample_id'])]:
                raise CheckFailed(f'inputs: a registered label of {stem} differs from the data')
            cells[(row['arm_id'], int(row['model_seed']))][int(row['sample_id'])] = row['y_pred']
    models = {(m['arm_id'], m['model_seed']): m for m in record['models']}
    predictions = {}
    for key, rows in cells.items():
        if sorted(rows) != sorted(test):
            raise CheckFailed(f'inputs: the registered {key} predictions of {stem} cover another test set')
        predictions[key] = np.asarray([rows[i] for i in test], dtype=y.dtype)
        if array_hash(predictions[key]) != models[key]['prediction_hash']:
            raise CheckFailed(f'inputs: the registered {key} predictions of {stem} do not hash to their model row')
    if sorted(predictions) != sorted((arm, seed) for arm in ('views7', 'frozen') for seed in FIT_SEEDS):
        raise CheckFailed(f'inputs: the registered predictions of {stem} miss an arm or a seed')
    return {'record': record, 'predictions': predictions, 'models': models,
            'fits': {(f['arm_id'], f['model_seed']): f for f in record['fits']}}


# ----------------------------------------------------------------------------- the permutation

def label_stream_seed(identity):
    return derive_seed(LABEL_SEED, LABEL_STREAM, str(identity['dataset_id']), int(identity['outer_repeat']),
                       int(identity['outer_fold']), int(identity['model_seed']))


def permutation_statistics(y_true, y_net):
    """Share of rows keeping their true label, its expectation (sum of squared class shares) and Cramer's V."""
    y_true, y_net = np.asarray(y_true), np.asarray(y_net)
    classes = np.unique(np.concatenate([y_true, y_net]))
    table = np.array([[np.sum((y_net == a) & (y_true == b)) for b in classes] for a in classes], dtype=float)
    n = table.sum()
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        chi2 = float(np.nansum(np.where(expected > 0, (table - expected) ** 2 / expected, 0.)))
    k = min(table.shape) - 1
    shares = np.unique(y_true, return_counts=True)[1] / len(y_true)
    return {'agreement': float(np.mean(y_net == y_true)), 'expected_agreement': float(np.sum(shares ** 2)),
            'cramers_v': float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.,
            'class_counts_equal': bool(np.array_equal(np.sort(y_true), np.sort(y_net)))}


def permuted_labels(y_train, identity):
    """The network labels of one outer fold and fitting seed: y_train under one uniform permutation of its rows, from the
    private stream; the global numpy and Python RNG states are checked and restored."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        seed = label_stream_seed(identity)
        order = np.random.RandomState(seed).permutation(len(y_train))
    finally:
        unchanged = td.numpy_states_equal(numpy_state, np.random.get_state()) and python_state == random.getstate()
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    y_net = np.asarray(y_train)[order]
    return y_net, {'stream_seed': int(seed), 'permutation_sha256': array_hash(order), 'rng_unchanged': bool(unchanged),
                   **permutation_statistics(y_train, y_net)}


def identity_labels(y_train):
    """The identity 'permutation' of the check: the code path of the arm with unpermuted labels."""
    y_net = np.asarray(y_train).copy()
    return y_net, {'stream_seed': None, 'permutation_sha256': array_hash(np.arange(len(y_train))), 'rng_unchanged': True,
                   **permutation_statistics(y_train, y_net)}


# ----------------------------------------------------------------------------- instrumentation (subclasses; nothing edited)

class VoteCounter(mc.MotionArm):
    """motion_controls.MotionArm, unchanged, in identity mode ('views7') or frozen, that also keeps after every update the
    cumulative counts of selected slots and of nonzero votes per hidden layer, read from the arm's own statistics."""

    def __init__(self, net, arm, *, seed, hidden_layers):
        if arm not in ('views7', 'frozen'):
            raise ValueError('The permuted-label control installs MotionArm only as views7 (identity) or frozen')
        super().__init__(net, arm, seed=seed, hidden_layers=hidden_layers)
        self.trace = {}

    def _transform(self, index, layer, motion_last_layer):
        out = super()._transform(index, layer, motion_last_layer)
        before = self.statistics[index]['before']
        self.trace.setdefault(index, []).append((int(before['slots']), int(before['effective'])))
        return out

    def votes_at(self, t):
        """{hidden layer: [selected slots, nonzero votes]} over updates 1..t."""
        return {str(index): ([0, 0] if t == 0 else list(trace[t - 1])) for index, trace in sorted(self.trace.items())}


class MatchedRecorder(td.TrainingRecorder):
    """training_diagnostics.TrainingRecorder, unchanged, that also copies the filters right after the updates in `keep`
    (the matched stop), keeps the scheduled copies at 0 and T only, and hashes the rows (not the labels) of the first
    batch. Every read is RNG-guarded, as the recorder's own."""

    def __init__(self, net, *, keep=()):
        super().__init__(net, every=int(net.iterations) + 1)        # scheduled copies: update 0 and update T only
        self.keep = frozenset(int(t) for t in keep)
        if any(not 0 <= t <= self.iterations for t in self.keep):
            raise ValueError('A matched stop lies outside 0..T')
        self.kept = {0: self.initial} if 0 in self.keep else {}
        self.first_batch_rows = None

    def _update(self, original, data_train, data_validation, train_type, problem):
        if not self.update_iters:
            with self._guard():
                rows = np.asarray([[int(item) for item in sample[0]] for sample in data_train], dtype=np.int64)
                self.first_batch_rows = array_hash(rows)
        output = super()._update(original, data_train, data_validation, train_type, problem)
        t = self.update_iters[-1]
        if t in self.keep:
            with self._guard():
                self.kept[t] = td.filter_copy(self.network, orders=False)
        return output


def label_fit(params, seed, X, y_true, y_net, *, arm, motion_seeds, hidden_layers, keep=None):
    """MultiViewArrowFlowKNN(**params, seed=seed).fit(X, y), step for step as motion_controls.arm_fit, except that each
    view network's initialize_orders and train_initialized receive y_net while the encoder and the readout receive
    y_true. keep[v], if given, is the update after which view v's filters are copied. Returns (model, views)."""
    model = MultiViewArrowFlowKNN(**params, seed=seed)
    model.readouts_, model.readout_selections_ = [], []
    model.readout_seconds_ = 0.
    model.classes_ = np.unique(y_true)
    model.views_ = []
    encoding = training = 0.
    views = []
    if len(model.widths) != hidden_layers:
        raise CheckFailed('inputs: the selection\'s widths disagree with its hidden depth')
    for v in range(model.n_views):
        seed_v = derive_seed(model.seed, 'view', v)
        start = time.perf_counter()
        enc = OrdinalEncoder(view_strategy(model.strategy, v), model.embed_dim, model.degree, model.lda_ratio,
                             seed_v).fit(X, y_true)
        orders = enc.transform(X)
        encoding += time.perf_counter() - start
        net = ArrowFlowEstimator(embed_dim=model.embed_dim, degree=model.degree, widths=model.widths,
                                 iterations=model.iterations, learning_rate=model.learning_rate,
                                 batch_size=model.batch_size, last_layer_update=model.last_layer_update,
                                 ratio_data_backprop=model.ratio_data_backprop,
                                 motion_normalization_mult=model.motion_normalization_mult,
                                 p_correct=model.p_correct, seed=seed_v,
                                 validation_ratio=model.validation_ratio, augment=model.augment,
                                 n_augmentations=model.n_augmentations, max_swaps=model.max_swaps)
        net.initialize_orders(orders, y_net)
        motion = VoteCounter(net, arm, seed=motion_seeds[v], hidden_layers=hidden_layers)
        recorder = MatchedRecorder(net, keep=() if keep is None else (keep[v],))
        with motion.installed(), recorder.installed():
            net.train_initialized(orders, y_net)
        training += net.training_seconds_
        model.views_.append((enc, net))
        model._fit_view_readout(enc, net, orders, y_true, seed_v)
        views.append({'view': v, 'seed': seed_v, 'orders': orders, 'motion': motion, 'recorder': recorder})
    model.encoding_seconds_, model.training_seconds_ = encoding, training
    return model, views


# ----------------------------------------------------------------------------- readouts

def copy_readout(matrices, hidden_layers, orders_train, orders_test, y_train, seed_v):
    """The method's kNN readout of one view on copied filters, with the true labels."""
    hidden_train = td.network_forward(inverse_positions(orders_train), matrices[:hidden_layers])[-1][2]
    hidden_test = td.network_forward(inverse_positions(orders_test), matrices[:hidden_layers])[-1][2]
    selection = select_knn_readout(hidden_train, y_train, seed=derive_seed(seed_v, 'readout_selection'))
    readout = StableFootruleKNN(**selection['config'], input_kind='positions').fit(hidden_train, y_train)
    return {'predictions': np.asarray(readout.predict(hidden_test)), 'selection': selection, 'readout': readout,
            'hidden_train': hidden_train, 'hidden_test': hidden_test}


def purity_of(readouts, hidden_trains, hidden_tests, y_train, y_test, *, query_cap=mc.PURITY_QUERY_CAP):
    """motion_controls.purity_record, transcribed for given hidden rankings (a copy's) instead of a live network's."""
    step = max(1, int(np.ceil(len(y_train) / query_cap)))
    rows = np.arange(len(y_train))[::step]
    test_values, train_values, ks = [], [], []
    for readout, hidden_train, hidden_test in zip(readouts, hidden_trains, hidden_tests):
        k = min(int(readout.n_neighbors), len(y_train))
        ks.append(k)
        test_values.append(mc.neighborhood_purity(readout, y_train, hidden_test, y_test, k=k))
        train_values.append(mc.neighborhood_purity(readout, y_train, hidden_train[rows], np.asarray(y_train)[rows],
                                                   k=min(k, len(y_train) - 1), exclude_rows=rows))
    present = [v for v in test_values if v is not None]
    return {'test': float(np.mean(present)) if present else None,
            'train': float(np.mean([v for v in train_values if v is not None])) if train_values else None,
            'test_by_view': test_values, 'train_by_view': train_values, 'k_by_view': ks,
            'train_query_rows': int(len(rows)), 'train_query_step': step,
            'definition': 'as motion_controls.purity_record, on the hidden rankings of the copied filters'}


def readout_labels_true(readout, y_train):
    return array_hash(readout.classes_[readout.labels_]) == array_hash(np.asarray(y_train))


def displacement_of(matrices, initial):
    return [td.displacement(m, i) for m, i in zip(matrices, initial)]


def choices(selections):
    return [{'view': v, 'config': s['config'], 'config_id': s['config_id']} for v, s in enumerate(selections)]


def plain_list(array):
    return np.asarray(array).tolist()


# ----------------------------------------------------------------------------- one job

class Checks:
    """Named checks of one job: each records its counts and raises CheckFailed when it does not pass."""

    def __init__(self):
        self.entries = {name: {'performed': False, 'passed': None} for name in JOB_CHECKS}

    def require(self, name, passed, message='', **detail):
        entry = self.entries[name]
        entry['performed'] = True
        entry['passed'] = bool(passed) and entry['passed'] is not False
        for key, value in detail.items():
            entry.setdefault('detail', {}).setdefault(key, []).append(td.native(value))
        if not passed:
            raise CheckFailed(f'{name}: {message}' if message else name)


def differ(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return int(np.sum(a != b)) if a.shape == b.shape else max(len(a), len(b))


def run_job(stem, *, labels='permuted', say=None, seeds=FIT_SEEDS, uninstrumented=True):
    """Every fitting seed of one dataset and outer fold: AF's refit (which must reproduce the registered predictions), the
    permuted-label arm read at its three points, and, on the first outer fold, the frozen arm trained with the permuted
    labels and one uninstrumented fit. labels='identity' (check only) runs the arm's code with the true labels; seeds and
    uninstrumented=False serve the check's timing fit. Returns the job record; say(line) receives counts and timings
    only."""
    if labels not in ('permuted', 'identity'):
        raise ValueError('labels must be permuted or identity')
    started = time.perf_counter()
    say = say or (lambda line: None)
    plan = motion_plan()
    job = plan['jobs'][stem]
    name, repeat, fold = job['dataset_id'], job['outer_repeat'], job['outer_fold']
    checks = Checks()
    X, y, data, splits = load_prepared(MOTION_RUN, name)
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (repeat, fold))
    train, test = [int(i) for i in split['train']], [int(i) for i in split['test']]
    X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
    sealed = plan['sealed'][(name, repeat, fold)]
    registered = registered_job(stem, y, test)
    reg_identity = registered['record']['identity']
    diag_identity = json.loads((DIAGNOSTICS_RUN/'jobs'/f'{stem}.json').read_text())['identity']
    hidden_layers, params, T = int(job['hidden_layers']), job['params'], int(job['params']['iterations'])
    expected_identity = {'dataset_id': name, 'outer_repeat': repeat, 'outer_fold': fold, 'config_id': job['config_id'],
                         'dataset_hash': data['dataset_hash'], 'split_hash': config_id(split), 'train_ids': train,
                         'query_ids': test, 'training_labels_hash': array_hash(y_train),
                         'raw_train_hash': array_hash(X_train), 'raw_query_hash': array_hash(X_test),
                         'params': params, 'hidden_layers': hidden_layers, 'model_seeds': list(FIT_SEEDS),
                         'arm_seeds': job['arm_seeds'], 'query': 'outer_test_rows'}
    checks.require('inputs', all(td.native(reg_identity[k]) == td.native(v) for k, v in expected_identity.items()),
                   'the registered record does not describe this fold, its rows, labels or selection')
    checks.require('inputs', all(diag_identity[k] == v for k, v in (
        ('dataset_id', name), ('outer_repeat', repeat), ('outer_fold', fold), ('config_id', job['config_id']),
        ('dataset_hash', data['dataset_hash']), ('split_hash', config_id(split)), ('selected', job['selected']),
        ('model_seed', DIAGNOSTICS_SEED), ('query', 'outer_test_rows'))),
                   'the diagnostics job does not bind to this fold through its dataset, split and selection hashes')
    checkpoint_rows = [diagnostics_checkpoints().get((name, repeat, fold, v)) for v in range(7)]
    checks.require('inputs', all(r is not None and int(r['iterations']) == T and r['orders_verified'] == '1'
                                 and r['strategy'] == view_strategy(params['strategy'], v)
                                 for v, r in enumerate(checkpoint_rows)),
                   'checkpoints.csv does not hold this fold\'s seven verified views')
    source = mc.check_core_sources()
    checks.require('core_sources', source['passed'], f'changed: {source["changed"]}')
    first_fold = bool(job['check_uninstrumented'])
    classes = np.unique(y)
    record = {'stem': stem, 'dataset_id': name, 'outer_repeat': repeat, 'outer_fold': fold,
              'config_id': job['config_id'], 'hidden_layers': hidden_layers, 'selected_widths': job['selected_widths'],
              'n_train': len(train), 'n_test': len(test), 'test_ids_sha256': array_hash(np.asarray(test)),
              'training_labels_hash': array_hash(y_train), 'labels': labels, 'first_fold': first_fold,
              'environment': environment_block(), 'fits': []}
    af_matchings = []
    for seed in seeds:
        identity = {'dataset_id': name, 'outer_repeat': repeat, 'outer_fold': fold, 'model_seed': seed}
        motion_seeds = job['arm_seeds']['views7'][str(seed)]
        reg_fit = registered['fits'][('views7', seed)]
        reg_initial = registered['record']['checks']['initial_state']['initial_filter_hashes'][f'views7__s{seed}']
        fit = {'seed': seed}
        # AF: the same code with the true labels; it must reproduce the registered predictions before PL is fitted
        seed_fit(seed)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            af, af_views = label_fit(params, seed, X_train, y_train, y_train, arm='views7', motion_seeds=motion_seeds,
                                     hidden_layers=hidden_layers)
            af_seconds = time.perf_counter() - start
            af_view_pred, orders_test = af.predict_views(X_test)
        af_pred = mc._majority(af_view_pred)
        reproduction = base.reproduction_check(af_pred, sealed, seed)
        n_differ = differ(af_pred, registered['predictions'][('views7', seed)])
        checks.require('reference_predictions', reproduction['reproduced'] and n_differ == 0,
                       f'ArrowFlow does not reproduce the registered predictions for seed {seed}: '
                       f'{reproduction["n_differing"]} of {reproduction["n_test"]} test rows differ', n_differing=n_differ)
        af_hashes = [net.state_hash() for _, net in af.views_]
        af_initial = [mc._filters_hash(vw['motion'].initial) for vw in af_views]
        af_verified = [vw['recorder'].verify() for vw in af_views]
        kstar = [int(v['checkpoint']) for v in af_verified]
        af_displacement = mc._mean_displacement([vw['motion'].displacement_by_layer() for vw in af_views])
        reg_displacement = registered['models'][('views7', seed)]['displacement']
        state_differ = sum(a != b for a, b in zip(af_hashes, reg_fit['view_state_hashes']))
        initial_differ = sum(a != b for a, b in zip(af_initial, reg_initial))
        choice_differ = sum(a['config_id'] != b['config_id']
                            for a, b in zip(choices(af.readout_selections_), reg_fit['readout_choices']))
        displacement_same = len(af_displacement) == len(reg_displacement) and all(
            all(math.isclose(a[k], b[k], rel_tol=0, abs_tol=1e-12) for k in ('mean_normalized_footrule',
                                                                             'mean_changed_share', 'unchanged_share'))
            for a, b in zip(af_displacement, reg_displacement))
        checks.require('registered_state', state_differ == 0 and initial_differ == 0 and choice_differ == 0
                       and displacement_same and len(af_hashes) == 7,
                       f'seed {seed}: {state_differ} view states, {initial_differ} initial filters and {choice_differ} '
                       f'readout choices differ from the registered views7 fit, displacement equal: {displacement_same}')
        checks.require('training_record', all(v['training_record']['passed'] and v['checkpoint_orders']['passed']
                                              for v in af_verified), f'seed {seed}: an ArrowFlow recorder record fails')
        checkpoint_differ = 0
        if seed == DIAGNOSTICS_SEED:
            checkpoint_differ = sum(int(r['checkpoint_iteration']) != k for r, k in zip(checkpoint_rows, kstar))
            checks.require('checkpoints', checkpoint_differ == 0,
                           f'{checkpoint_differ} of 7 views disagree with the diagnostics run\'s checkpoint')
        say(f'{name:14s} r{repeat}f{fold} seed {seed:5d} | ArrowFlow refit: {n_differ} of {len(test)} predictions differ '
            f'from the registered, view states {state_differ} of 7 differ, initial filters {initial_differ} of 7 differ, '
            f'readout choices {choice_differ} of 7 differ'
            + (f', checkpoints {checkpoint_differ} of 7 views differ from the diagnostics run' if seed == DIAGNOSTICS_SEED
               else '') + f' ({af_seconds:.0f}s)')
        af_matchings.append([vw['motion'].matching() for vw in af_views])
        orders_train = [vw['orders'] for vw in af_views]
        # the readout of the initial filters with the true labels: FR, recomputed without training
        start = time.perf_counter()
        update0 = [copy_readout(vw['recorder'].initial['matrices'], hidden_layers, vw['orders'], orders_test[v],
                                y_train, vw['seed']) for v, vw in enumerate(af_views)]
        fr0_pred = mc._majority([r['predictions'] for r in update0])
        fr0_differ = differ(fr0_pred, registered['predictions'][('frozen', seed)])
        checks.require('update0_frozen', fr0_differ == 0,
                       f'seed {seed}: the readout of the initial filters differs from FR on {fr0_differ} test rows')
        fr0_seconds = time.perf_counter() - start
        fit['af'] = {'predictions': plain_list(af_pred), 'view_predictions': [plain_list(p) for p in af_view_pred],
                     'checkpoint': kstar, 'improvements': [len(v['improvements']) for v in af_verified],
                     'view_state_hashes': af_hashes, 'initial_filter_hashes': af_initial,
                     'readout': choices(af.readout_selections_),
                     'displacement': {'checkpoint': af_displacement, 'final': mc._mean_displacement(
                         [displacement_of(vw['recorder'].snapshots[T]['matrices'], vw['recorder'].initial['matrices'])
                          for vw in af_views])},
                     'votes': [{'checkpoint': vw['motion'].votes_at(k), 'final': vw['motion'].votes_at(T)}
                               for vw, k in zip(af_views, kstar)],
                     'seconds': af_seconds, 'training_seconds': af.training_seconds_}
        fit['fr0'] = {'predictions': plain_list(fr0_pred), 'view_predictions': [plain_list(r['predictions']) for r in update0],
                      'readout': choices([r['selection'] for r in update0]), 'seconds': fr0_seconds}
        # PL: the same networks with the network labels permuted (or, in the check, the true labels)
        if labels == 'permuted':
            y_net, permutation = permuted_labels(y_train, identity)
        else:
            y_net, permutation = identity_labels(y_train)
        checks.require('label_routing', permutation['rng_unchanged'] and permutation['class_counts_equal'],
                       f'seed {seed}: the permutation draw changed the global RNG or the class counts')
        seed_fit(seed)
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            pl, pl_views = label_fit(params, seed, X_train, y_train, y_net, arm='views7', motion_seeds=motion_seeds,
                                     hidden_layers=hidden_layers, keep=kstar)
            pl_seconds = time.perf_counter() - start
            start = time.perf_counter()
            own_view_pred, pl_orders_test = pl.predict_views(X_test)
            own_purity = mc.purity_record(pl, [vw['orders'] for vw in pl_views], pl_orders_test, y_train, y_test)
            own_copy = [copy_readout(vw['recorder'].checkpoint_copy['matrices'], hidden_layers, vw['orders'],
                                     pl_orders_test[v], y_train, vw['seed']) for v, vw in enumerate(pl_views)]
            matched = [copy_readout(vw['recorder'].kept[k]['matrices'], hidden_layers, vw['orders'], pl_orders_test[v],
                                    y_train, vw['seed']) if k in vw['recorder'].kept else None
                       for v, (vw, k) in enumerate(zip(pl_views, kstar))]
            checks.require('snapshot_readout', all(m is not None for m in matched),
                           f'seed {seed}: a matched stop was not copied')
            final = [copy_readout(vw['recorder'].snapshots[T]['matrices'], hidden_layers, vw['orders'], pl_orders_test[v],
                                  y_train, vw['seed']) for v, vw in enumerate(pl_views)]
            readout_seconds = time.perf_counter() - start
        pl_verified = [vw['recorder'].verify() for vw in pl_views]
        own_k = [int(v['checkpoint']) for v in pl_verified]
        checks.require('training_record', all(v['training_record']['passed'] and v['checkpoint_orders']['passed']
                                              for v in pl_verified), f'seed {seed}: a PL recorder record fails')
        # label routing: what reads which labels
        encodings_differ = sum(array_hash(a['orders']) != array_hash(b['orders']) for a, b in zip(af_views, pl_views)) + \
            sum(array_hash(a) != array_hash(b) for a, b in zip(orders_test, pl_orders_test))
        network_labels = sum(net.training_labels_hash_ == array_hash(y_net) for _, net in pl.views_)
        readouts_true = sum(readout_labels_true(r, y_train) for r in pl.readouts_) + sum(
            readout_labels_true(r['readout'], y_train) for group in (own_copy, matched, final) for r in group)
        first_batch_differ = sum(a['recorder'].first_batch_rows != b['recorder'].first_batch_rows
                                 for a, b in zip(af_views, pl_views))
        validation_differ = sum(not np.array_equal(a['recorder'].validation[0], b['recorder'].validation[0])
                                for a, b in zip(af_views, pl_views))
        pl_initial = [mc._filters_hash(vw['motion'].initial) for vw in pl_views]
        pl_initial_differ = sum(a != b for a, b in zip(pl_initial, reg_initial))
        checks.require('label_routing', encodings_differ == 0 and network_labels == 7 and readouts_true == 28
                       and first_batch_differ == 0 and validation_differ == 0,
                       f'seed {seed}: encodings {encodings_differ} of 14 differ from ArrowFlow\'s, network labels '
                       f'{network_labels} of 7 are the permuted ones, readout labels {readouts_true} of 28 are the true '
                       f'ones, first batch rows {first_batch_differ} of 7 and validation rows {validation_differ} of 7 '
                       'differ from ArrowFlow\'s')
        checks.require('initial_state', pl_initial_differ == 0,
                       f'seed {seed}: {pl_initial_differ} of 7 initial filters differ from the registered views7 ones')
        own_copy_differ = sum(differ(a, b['predictions']) for a, b in zip(own_view_pred, own_copy))
        own_hidden_same = all(np.array_equal(net.transform_orders(vw['orders']), c['hidden_train'])
                              for (_, net), vw, c in zip(pl.views_, pl_views, own_copy))
        own_copy_purity = purity_of([c['readout'] for c in own_copy], [c['hidden_train'] for c in own_copy],
                                    [c['hidden_test'] for c in own_copy], y_train, y_test)
        purity_same = all(own_copy_purity[k] == own_purity[k] for k in ('test', 'train', 'test_by_view', 'train_by_view'))
        checks.require('snapshot_readout', own_copy_differ == 0 and own_hidden_same and purity_same,
                       f'seed {seed}: the copy at PL\'s own checkpoint differs from the live readout on '
                       f'{own_copy_differ} view predictions (hidden rankings equal: {own_hidden_same}, purity equal: '
                       f'{purity_same})')
        rng_ok = all(all(vw['motion'].rng_checks) and all(vw['recorder'].rng_checks) for vw in af_views + pl_views)
        checks.require('rng_untouched', rng_ok, f'seed {seed}: a guarded read changed the global RNG')
        removed = all(vw['motion'].wrapper_removed() and 'update_network' not in vars(vw['recorder'].network)
                      for vw in af_views + pl_views)
        checks.require('wrapper_removed', removed, f'seed {seed}: an instance-level method survived a fit')
        checks.require('training_record', all(vw['motion'].draws == 0 and vw['motion'].calls == T
                                              and len(vw['motion'].trace) == hidden_layers
                                              and all(len(t) == T for t in vw['motion'].trace.values())
                                              for vw in af_views + pl_views),
                       f'seed {seed}: the identity arm drew from its stream or missed an update')
        pl_hashes = [net.state_hash() for _, net in pl.views_]
        match_pred = mc._majority([m['predictions'] for m in matched])
        own_pred = mc._majority(own_view_pred)
        if labels == 'identity':
            identity_differ = {'own': differ(own_pred, af_pred), 'match': differ(match_pred, af_pred),
                               'states': sum(a != b for a, b in zip(pl_hashes, af_hashes)),
                               'checkpoints': sum(a != b for a, b in zip(own_k, kstar))}
            checks.require('snapshot_readout', not any(identity_differ.values()),
                           f'seed {seed}: with the true labels the arm differs from ArrowFlow: {identity_differ}')
            say(f'{name:14s} r{repeat}f{fold} seed {seed:5d} | identity-label arm: matched stop {identity_differ["match"]} '
                f'of {len(test)} and own checkpoint {identity_differ["own"]} of {len(test)} predictions differ from '
                f'ArrowFlow\'s, view states {identity_differ["states"]} of 7 and checkpoints '
                f'{identity_differ["checkpoints"]} of 7 differ, encodings {encodings_differ} of 14 differ, copy at own '
                f'checkpoint vs live readout {own_copy_differ} of {7 * len(test)} view predictions differ '
                f'({pl_seconds:.0f}s, readouts {readout_seconds:.1f}s)')
        else:
            say(f'{name:14s} r{repeat}f{fold} seed {seed:5d} | permuted labels (timing and routing; no score): encodings '
                f'{encodings_differ} of 14 differ from ArrowFlow\'s, network labels permuted in {network_labels} of 7 '
                f'views with equal class counts, readout labels true in {readouts_true} of 28 readouts, first batch rows '
                f'{first_batch_differ} of 7 and validation rows {validation_differ} of 7 differ, initial filters '
                f'{pl_initial_differ} of 7 differ, copy at own checkpoint vs live readout {own_copy_differ} of '
                f'{7 * len(test)} view predictions differ, global RNG unchanged: {permutation["rng_unchanged"] and rng_ok}; '
                f'ArrowFlow {af_seconds:.0f}s, permuted-label arm {pl_seconds:.0f}s (ratio {pl_seconds / af_seconds:.2f}), '
                f'readouts {readout_seconds:.1f}s')
        initial_pl = [vw['recorder'].initial['matrices'] for vw in pl_views]
        fit['pl'] = {
            'permutation': permutation, 'checkpoint': own_k, 'improvements': [len(v['improvements']) for v in pl_verified],
            'view_state_hashes': pl_hashes, 'initial_filter_hashes': pl_initial,
            'votes': [{'match': vw['motion'].votes_at(k), 'own': vw['motion'].votes_at(o), 'final': vw['motion'].votes_at(T)}
                      for vw, k, o in zip(pl_views, kstar, own_k)],
            'seconds': pl_seconds, 'training_seconds': pl.training_seconds_, 'readout_seconds': readout_seconds,
            'readouts': {
                'match': {'updates': kstar, 'predictions': plain_list(match_pred),
                          'view_predictions': [plain_list(m['predictions']) for m in matched],
                          'readout': choices([m['selection'] for m in matched]),
                          'purity': purity_of([m['readout'] for m in matched], [m['hidden_train'] for m in matched],
                                              [m['hidden_test'] for m in matched], y_train, y_test),
                          'displacement': mc._mean_displacement([displacement_of(vw['recorder'].kept[k]['matrices'], i)
                                                                 for vw, k, i in zip(pl_views, kstar, initial_pl)])},
                'own': {'updates': own_k, 'predictions': plain_list(own_pred),
                        'view_predictions': [plain_list(p) for p in own_view_pred],
                        'readout': choices(pl.readout_selections_), 'purity': own_purity,
                        'displacement': mc._mean_displacement([vw['motion'].displacement_by_layer() for vw in pl_views])},
                'final': {'updates': [T] * 7, 'predictions': plain_list(mc._majority([f['predictions'] for f in final])),
                          'view_predictions': [plain_list(f['predictions']) for f in final],
                          'readout': choices([f['selection'] for f in final]),
                          'purity': purity_of([f['readout'] for f in final], [f['hidden_train'] for f in final],
                                              [f['hidden_test'] for f in final], y_train, y_test),
                          'displacement': mc._mean_displacement([displacement_of(vw['recorder'].snapshots[T]['matrices'], i)
                                                                 for vw, i in zip(pl_views, initial_pl)])}}}
        del af, af_views, pl, pl_views, update0, own_copy, matched, final
        # the label-routing check on the first outer fold: frozen filters trained with the permuted labels must be FR
        if first_fold:
            seed_fit(seed)
            start = time.perf_counter()
            with threadpool_limits(limits=1):
                frozen, frozen_views = label_fit(params, seed, X_train, y_train, y_net, arm='frozen',
                                                 motion_seeds=job['arm_seeds']['frozen'][str(seed)],
                                                 hidden_layers=hidden_layers)
                frozen_pred = mc._majority(frozen.predict_views(X_test)[0])
            frozen_seconds = time.perf_counter() - start
            frozen_differ = differ(frozen_pred, registered['predictions'][('frozen', seed)])
            unmoved = all(entry['mean_normalized_footrule'] == 0. and entry['unchanged_share'] == 1.
                          for entry in mc._mean_displacement([vw['motion'].displacement_by_layer() for vw in frozen_views])
                          if entry['kind'] == 'hidden')
            frozen_initial_differ = sum(mc._filters_hash(vw['motion'].initial) != h for vw, h in zip(frozen_views, reg_initial))
            checks.require('frozen_permuted', frozen_differ == 0 and unmoved and frozen_initial_differ == 0,
                           f'seed {seed}: frozen filters trained with the network labels differ from FR on '
                           f'{frozen_differ} test rows (hidden filters unmoved: {unmoved}, initial filters differing: '
                           f'{frozen_initial_differ})')
            checks.require('rng_untouched', all(all(vw['motion'].rng_checks) and all(vw['recorder'].rng_checks)
                                                for vw in frozen_views), f'seed {seed}: the frozen refit changed the RNG')
            checks.require('wrapper_removed', all(vw['motion'].wrapper_removed() for vw in frozen_views),
                           f'seed {seed}: an instance-level method survived the frozen refit')
            fit['fr_perm'] = {'predictions': plain_list(frozen_pred), 'n_differing_from_fr': frozen_differ,
                              'seconds': frozen_seconds}
            say(f'{name:14s} r{repeat}f{fold} seed {seed:5d} | frozen arm: registered predictions read (file sha256 and '
                f'prediction hashes verified); readout of the initial filters {fr0_differ} of {len(test)} differ; frozen '
                f'refit with the {"permuted" if labels == "permuted" else "true"} network labels {frozen_differ} of '
                f'{len(test)} differ ({frozen_seconds:.0f}s)')
            del frozen, frozen_views
            if seed == seeds[0] and uninstrumented:
                start = time.perf_counter()
                with threadpool_limits(limits=1):
                    plain_hashes, plain_pred = mc.uninstrumented_fit(params, seed, X_train, y_train, X_test)
                plain_seconds = time.perf_counter() - start
                plain_state_differ = sum(a != b for a, b in zip(plain_hashes, af_hashes))
                plain_differ = differ(plain_pred, af_pred)
                checks.require('uninstrumented_fit', plain_state_differ == 0 and plain_differ == 0,
                               f'{plain_state_differ} view states and {plain_differ} predictions differ from an '
                               'uninstrumented fit')
                fit['uninstrumented'] = {'view_states_differing': plain_state_differ, 'n_differing': plain_differ,
                                         'seconds': plain_seconds}
                say(f'{name:14s} r{repeat}f{fold} seed {seed:5d} | uninstrumented fit: view states {plain_state_differ} '
                    f'of 7 and predictions {plain_differ} of {len(test)} differ from ArrowFlow\'s ({plain_seconds:.0f}s)')
        record['fits'].append(fit)
    if tuple(seeds) == FIT_SEEDS:
        mine = mc._matching_summary(af_matchings)
        theirs = registered['record']['checks']['mass_matched']['by_arm']['views7']
        votes_same = sorted(mine) == sorted(theirs) and all(
            all(mine[layer][k] == theirs[layer][k] for k in ('batches', 'accepted_slots', 'changed_slots',
                                                              'effective_before', 'effective_after', 'positive_before',
                                                              'positive_after'))
            and all(math.isclose(mine[layer][k], theirs[layer][k], rel_tol=1e-12, abs_tol=0)
                    for k in ('mass_before', 'mass_after')) for layer in mine)
        checks.require('registered_state', votes_same, 'ArrowFlow\'s vote statistics differ from the registered views7 ones')
        say(f'{name:14s} r{repeat}f{fold}            | ArrowFlow\'s vote statistics over the job equal the registered: '
            f'{votes_same}')
    record['checks'] = checks.entries
    record['seconds'] = time.perf_counter() - started
    return td.native(record)


@functools.lru_cache(maxsize=1)
def environment_block():
    env = mc.environment()
    return {key: env[key] for key in ('python', 'numpy', 'scipy', 'sklearn', 'torch', 'code_revision')} | {
        'source_hashes_sha256': hashlib.sha256(json.dumps(env['source_hashes'], sort_keys=True).encode()).hexdigest()}


# ----------------------------------------------------------------------------- check

def check():
    """The pre-run check: the arm's code with the true labels must reproduce ArrowFlow and FR on the check sample at every
    seed; then one fit with the permuted labels, timed, with the routing counts. Timings and counts only."""
    start = time.perf_counter()
    verify_protocol()
    verify_pins()
    verify_environment()
    print(f'inputs: {len(PINNED)} pinned registered files and the digest of {3 * 255} per-job files of the motion run '
          'verified; sealed sources, library versions and core functions equal the motion run\'s', flush=True)
    for stem in CHECK_SAMPLE:
        run_job(stem, labels='identity', say=lambda line: print(line, flush=True))
    stem, seed = CHECK_TIMING
    run_job(stem, labels='permuted', seeds=(seed,), uninstrumented=False, say=lambda line: print(line, flush=True))
    print(f'check passed ({time.perf_counter() - start:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- run

RECORD_NAME = re.compile(r'^[a-z_]+__r[0-2]f[0-4]\.(json|tmp)$')
OUTPUT_ENTRIES = ('results', 'design.json', 'code_revision.txt', 'analysis.json', 'ANALYSIS.md', 'tables')


def guard_output(out):
    """Refuse an output directory that is, or lies inside, a registered run or the source tree, or that holds anything
    this module does not write (logs aside). Nothing is written before this check."""
    out = Path(out).resolve()
    for protected in (*PROTECTED, REPO/'arrowflow', REPO/'experiments'):
        protected = protected.resolve()
        if out == protected or protected in out.parents:
            raise SystemExit(f'refusing to write into {protected}')
    if out.exists():
        if not out.is_dir():
            raise SystemExit(f'refusing {out}: it is not a directory')
        foreign = [p.name for p in out.iterdir() if p.name not in OUTPUT_ENTRIES and p.suffix != '.log']
        if (out/'results').is_dir():
            stems = set(job_stems())
            foreign += [f'results/{p.name}' for p in (out/'results').iterdir()
                        if not RECORD_NAME.match(p.name) or p.name.rsplit('.', 1)[0] not in stems]
        if foreign:
            raise SystemExit(f'refusing {out}: it holds files this module does not write, e.g. {sorted(foreign)[:3]}')
    return out


def code_revision(allow_uncommitted=False):
    """The commit this run executes; the code it imports must be committed (--allow-uncommitted marks a test run)."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    head, changes = git('rev-parse', 'HEAD'), git('status', '--porcelain', '--', *SOURCE_PATHS)
    if not changes:
        return head
    if not allow_uncommitted:
        raise SystemExit(f'the code this run imports has uncommitted changes; commit them first:\n{changes}')
    return head + '+uncommitted'


def _init_worker():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    torch.set_num_threads(1)


def _worker(args):
    stem, out = args
    path = Path(out)/'results'/f'{stem}.json'
    if path.exists():
        return path.name, 'exists'
    record = run_job(stem)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return path.name, f"{record['seconds']:.0f}s"


@functools.lru_cache(maxsize=1)
def registered_seconds():
    """The registered motion run's views7 fit seconds of each job (its three seeds), from the same instrumented harness."""
    return {stem: float(json.loads((MOTION_RUN/'results'/f'{stem}.json').read_text())['timing']['seconds_by_arm']['views7'])
            for stem in job_stems()}


def job_cost(stem):
    """Projected seconds of a job, for ordering only: AF's registered seconds times one plus PL's projected ratio, and on a
    first outer fold one more AF fit for the uninstrumented check and the three frozen refits (about a third of AF)."""
    job, seconds = motion_plan()['jobs'][stem], registered_seconds()[stem]
    return seconds * (1 + PL_COST[job['dataset_id']]) + (seconds / 3 * 2 if job['check_uninstrumented'] else 0.)


def run(out, workers=WORKERS, only=(), allow_uncommitted=False, ignore_load=False):
    """Every planned job (or --only) on spawned single-thread workers, longest first, one record per job, resumable.
    Refusals come before anything is written: a registered run or foreign files as output, uncommitted code, a busy
    machine (execution_lock cannot see a study launched from another module or worktree), another protocol or changed
    registered inputs, and records of another revision or design."""
    if not 1 <= workers <= MAX_WORKERS:
        raise SystemExit(f'--workers must be between 1 and {MAX_WORKERS}')
    out = guard_output(out)
    revision = code_revision(allow_uncommitted)
    load = os.getloadavg()[0]
    if load > LOAD_LIMIT and not ignore_load:
        raise SystemExit(f'the one-minute load average is {load:.1f}, above {LOAD_LIMIT:g}: another study seems to hold the '
                         f'machine, and {workers} more busy cores risk freezing it; wait for it to end (or pass --ignore-load)')
    verify_protocol()
    plan = json.loads(protocol_text())
    verify_pins()
    verify_environment()
    records = sorted((out/'results').glob('*.json')) if (out/'results').is_dir() else []
    if records:
        held = (out/'code_revision.txt').read_text().strip() if (out/'code_revision.txt').exists() else None
        if held != revision:
            raise SystemExit(f'{out} holds {len(records)} records from code revision {held}, not {revision}')
        if not (out/'design.json').exists() or json.loads((out/'design.json').read_text()) != plan:
            raise SystemExit(f'{out} holds records of another design')
    (out/'results').mkdir(parents=True, exist_ok=True)
    if not records:
        (out/'design.json').write_text(protocol_text())
        (out/'code_revision.txt').write_text(revision + '\n')
    stems = job_stems()
    if only:
        unknown = sorted(set(only) - set(stems))
        if unknown:
            raise SystemExit(f'unknown jobs: {unknown}')
        stems = [s for s in stems if s in set(only)]
    pending = sorted((s for s in stems if not (out/'results'/f'{s}.json').exists()), key=lambda s: (-job_cost(s), s))
    failed = []
    with execution_lock(), (out/'run.log').open('a') as log:
        def note(line):
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
        note(f'# start {datetime.now(timezone.utc).isoformat(timespec="seconds")}, revision {revision}, {workers} workers: '
             f'{len(pending)} of {len(stems)} jobs to run')
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=_init_worker) as pool:
                futures = {pool.submit(_worker, (stem, str(out))): stem for stem in pending}
                finished = 0
                for future in as_completed(futures):
                    if future.cancelled():
                        continue
                    finished += 1
                    try:
                        name, status = future.result()
                    except Exception as error:
                        name, status = futures[future] + '.json', f'FAILED ({type(error).__name__})'
                        failed.append(name)
                        traceback.print_exception(error, file=sys.stderr)
                        if len(failed) == 1:
                            cancelled = sum(other.cancel() for other in futures)
                            note(f'# {name} failed: {cancelled} pending jobs cancelled; the running ones finish and are kept')
                    note(f'[{finished}/{len(pending)}] {name} {status}')
        note(f'# end {datetime.now(timezone.utc).isoformat(timespec="seconds")}' + (f', {len(failed)} failed' if failed else ''))
    if failed:
        raise SystemExit(f'{len(failed)} jobs failed ({failed[:3]}); the run stopped; see stderr for the checks')


# ----------------------------------------------------------------------------- command

def main(argv=None):
    parser = argparse.ArgumentParser(description='The permuted-label control of ArrowFlow (design: the module header).')
    parser.add_argument('command', choices=('check', 'run', 'analyse', 'draft'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=WORKERS)
    parser.add_argument('--tables', type=Path, help='where analyse writes the LaTeX tables (default: OUTPUT/tables)')
    parser.add_argument('--only', action='append', default=[], help='run only this job, e.g. iris__r0f1 (repeatable; '
                                                                     'for tests)')
    parser.add_argument('--allow-uncommitted', action='store_true', help='run or analyse uncommitted code (tests only)')
    parser.add_argument('--ignore-load', action='store_true', help=f'start run although the load average exceeds '
                                                                    f'{LOAD_LIMIT:g}')
    args = parser.parse_args(argv)
    if args.command == 'draft':
        draft()
    elif args.command == 'check':
        with execution_lock():
            check()
    elif args.command == 'run':
        if not args.output:
            parser.error('run needs --output')
        run(args.output, args.workers, args.only, args.allow_uncommitted, args.ignore_load)
    else:
        if not args.output:
            parser.error('analyse needs --output')
        from . import compare_labels
        compare_labels.analyse(args.output, tables=args.tables, allow_uncommitted=args.allow_uncommitted)


if __name__ == '__main__':
    main()
