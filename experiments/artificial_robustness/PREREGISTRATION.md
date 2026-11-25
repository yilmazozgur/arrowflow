# Missing items at test time: ArrowFlow trained on clean ranking patterns

Pre-registration, 2026-09-23. Written before any test score of this study existed. Branch `v3/artificial-robustness`.
Study identifier: `arrowflow-artificial-robustness-1`.

## 1. The question

The author's hypothesis: on inherently ordinal data, ArrowFlow generalises from clean training patterns to test patterns
with missing items better than classical algorithms (for example, trained on 00110011 and tested on 0011011).

## 2. What the earlier experiments did

- **2023** (`test/Experiments_SortFlow2.txt`). The seven hand-written 8-item patterns in their original split (42 training
  rows, 35 test rows). SortNet's test error was 26.8 to 43.2%, and the classical classifiers' 45.7 to 80.0%. The classical
  classifiers received `DataGraph.transform_2_numpy_dataset`: the item ID at each position, minus one, zero-padded, so
  they read the sequence as a numeric vector. This representation is fragile: one deletion shifts every later entry.
- **Exploratory re-run, 2026-09-23, before this design** (`representation_check.py`, recorded in the study report). The
  same classifiers on the same split reproduce the 2023 errors exactly. Given each item's position instead (a deleted
  item imputed by the training mean), they reach 14.3 to 42.9% error, with 1-nearest-neighbour at 14.3%. Appending
  deleted items at the end instead costs much: 1-nearest-neighbour rises to 37.1%.
- **v3** (2026-09-14 and 2026-09-15). 315 generated rows per dataset, with the corruption drawn per class from the
  original rows, and training and test drawn from the same distribution (nested CV). Every model received each item's
  relative position. Encoded ArrowFlow reached 12.06 and 11.64% error, against numeric kNN's 10.79 and 10.16%. Native
  ArrowFlow, with deleted items appended at the end, reached 18.94 and 16.05%.
- Neither setup tested the author's scenario: clean training data and deletions at test time only.

## 3. Design

**Data.** The seven class prototypes of `artificial_ranks.prototypes`, at N = 8 and N = 16 items.

- **Training set (clean):** 30 rows per class. Each row is its class prototype with s adjacent inversion-adding swaps,
  s ~ U{0, ..., S}, where S = 2 at N = 8 and S = 8 at N = 16. This gives a similar normalised Kendall distance at both
  sizes. There are no deletions in training.
- **Test sets:** one per deletion level k. At N = 8, k ∈ {0, 1, 2, 3}; at N = 16, k ∈ {0, 2, 4, 6}. These are 0, 12.5,
  25 and 37.5% of the items. Each level has 100 rows per class. A test row deletes k items at uniformly random positions,
  then applies s ~ U{0, ..., S} swaps, the training distribution (`artificial_ranks.corrupt`).
- Within a level, the deletion count is fixed and the same for every class. So neither the count nor the corruption
  carries class information; this removes the artefact of the v3 generator.
- **Replicates:** 20 per N. Each replicate draws a new training set, new test sets and new model seeds.

**Inputs.** Every model receives the same information: the observed order of the present items.

- **POS:** each item's relative position idx / (m − 1) (`artificial_ranks.features`), NaN for a missing item. Every model
  on POS imputes NaN with the training mean (`NumericImputer`); the training set has no NaN.
- **Native ranking** (native ArrowFlow and footrule kNN on the input rankings). Training rows are complete rankings. A
  test row with missing items is presented under three conventions, applied only at test time:
  - `end`: missing items appended at the end in ascending item order. This is the author's convention of 2026-09-15,
    used by the v3 native run.
  - `center`: present items at their observed positions 0 .. m − 1, missing items at position N/2. This is the core's
    `missing_data_loc` = 0.5 · V of the batch forward pass, which the core's permutation check currently refuses.
  - `partial`: missing items are left out. The first layer compares the observed order with each filter's order restricted
    to the present items (the induced footrule); the kNN compares with each training row restricted likewise.

  On complete rows, the three conventions are identical to the core's forward pass; this is verified before scoring.
- **SEQ, for reference:** the 2023 representation (the item ID at each position, numeric and zero-padded) and its one-hot
  version.

**Models.** Hyperparameters are chosen by 3-fold stratified CV on the clean training set, by mean accuracy. Ties go to the
lowest canonical config id, and the chosen model is refitted on the whole training set.

1. **ArrowFlow, encoded (the paper's method):** `bridge.AdaptiveMultiViewKNN`, the 16 bridge candidates (7 views,
   checkpoint, adaptive encoding, kNN readout), on POS.
2. **ArrowFlow, native:** `native_artificial.NativeMultiViewArrowFlowKNN`, the 16 native candidates. It has 7 views, and
   its first-layer filters are random permutations of the N items. It is evaluated under `end`, `center` and `partial`.
3. **Untrained controls of 1 and 2:** each on its own projected grid, with the trained model's seed.
4. **Footrule kNN on the input rankings** under `end`, `center` and `partial`. The neighbour count and weighting are chosen
   by `select_knn_readout` on the training rankings.
5. **Classical models on POS:** numeric kNN, RBF SVC, random forest, MLP and gradient boosting (`CONVENTIONAL_GRIDS`,
   budget 24, seed 41071).
6. **Classical models on SEQ, numeric and one-hot:** numeric kNN, RBF SVC, random forest and MLP, on the same grids.
7. **Reference only:** the nearest true prototype by Kendall distance on the present items, with ties shared. It uses the
   generator's prototypes, so it is not a competitor.

**Metric.** Test error (%) at each (N, k) and replicate.

## 4. Hypotheses and analysis, fixed now

**Primary family: 12 contrasts, Holm.** The contrast is error(ArrowFlow arm) − error(numeric kNN on POS), for:

- two arms: native ArrowFlow under `partial`, and encoded ArrowFlow;
- two sizes: N = 8 and N = 16;
- three deletion levels: every level with k ≥ 1.

Each contrast is paired over the 20 replicates, with a two-sided paired t test (19 df) and a 95% confidence interval.

**Reading, per arm.**

- **Supported:** ArrowFlow's error is significantly lower (Holm) in at least 4 of the arm's 6 cells.
- **Contradicted:** numeric kNN's error is significantly lower in at least 4 of the 6 cells.
- **Not shown either way:** anything else.

**Why numeric kNN.** It was the strongest classical model on these data in every earlier run: the v3 encoded run, and the
original split with positions. The other classical models are secondary.

**Why `partial` is the primary native convention.** A deletion leaves the relative order of the remaining items intact.
`partial` uses exactly that information and nothing else, so it is the principled native reading of a missing item. The
`end` convention is secondary because the exploratory re-run showed its cost to a neighbour classifier.

**Secondary analyses** (descriptive, confidence intervals, no multiplicity adjustment):

- (a) each ArrowFlow arm against each other classical model on POS;
- (b) trained against untrained, for both arms;
- (c) native ArrowFlow against the footrule kNN under the same convention: does the network add anything to comparing
  the input rankings directly?
- (d) the three conventions;
- (e) ArrowFlow against the classical models on the SEQ representations: does the 2023 gap reappear?
- (f) k = 0 (no deletions), the in-distribution reference;
- (g) the prototype reference.

**Part A** (descriptive). The 77 original rows in their 2023 split:

- the 2023 classifier set on four representations: numeric sequence, one-hot sequence, positions with mean imputation, and
  positions completed at the end;
- encoded ArrowFlow, and native ArrowFlow under `end` in training and test. Only `end` can present training rows that
  have deletions to the core.

ArrowFlow in Part A is fitted with 3 seeds and tuned as above.

## 5. Verification before any scoring

- **V1.** The generator: training rows are complete, and every test row at level k has exactly N − k items. Swap counts
  are within {0, ..., S}, and the classes are balanced.
- **V2.** On complete rows, the three conventions give hidden rankings equal to the core's
  `ArrowFlowEstimator.transform_orders_by_depth`, and predictions equal to `NativeMultiViewArrowFlowKNN.predict`.
- **V3.** The induced footrule equals a brute-force computation.
- **V4.** The kNN from precomputed distances equals `StableFootruleKNN`.
- **V5.** The SEQ representation reproduces the 2023 log's classical errors exactly.
- **V6.** A smoke run with separate seeds goes through the whole pipeline and the analysis. It is never evidence.

## 6. After freezing

The protocol file records the hashes of this document and of every source file the study imports. The run refuses to
start if any hash differs. Any deviation is recorded as a deviation.
