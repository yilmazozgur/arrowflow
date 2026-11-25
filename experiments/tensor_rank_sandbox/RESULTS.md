# Tensor -> rank sandbox: results (2026-09-24)

**Architecture:** a linear tensor layer h = W x + b (64 outputs) feeds argsort into a rank classification layer: one
filter per class, prediction by the nearest filter in footrule distance. The rank layer learns by ArrowFlow's
output-layer rule, reimplemented in `sandbox.py`. The tensor layer learns from the rank layer's returned displacement:
- each coordinate should move toward its position in the true class's filter, and half as far away from its position
  in the nearest wrong class's filter;
- the value currently at the desired position becomes its regression target.

**Final configuration:** `sandbox.DEFAULT` with `V=64, convert='target', eta=0.01, contrast=0.5, eta_decay=True,
epochs=150`. It was fixed from the dev search (`search.py`, stages 0-2; cross-validation on the 70% dev portions of iris,
wine, breast cancer and digits) before any test score existed.

**Confirmation** (`confirm.py`): held-out 30% test portions of the four search datasets, plus eight cached OpenML
datasets the search never used. Five model seeds each; test accuracy (%) of the rank readout.

| Dataset | Trained | Frozen | Scrambled | Logistic reg. | Trained − frozen (seeds better) |
|---|---|---|---|---|---|
| iris | 87.6 | 82.7 | 82.2 | 97.8 | +4.9 (4/5) |
| wine | 100.0 | 93.7 | 93.0 | 100.0 | +6.3 (5/5) |
| breast cancer | 93.6 | 92.2 | 91.5 | 95.9 | +1.4 (4/5) |
| digits | 93.4 | 86.0 | 52.8 | 97.2 | +7.4 (5/5) |
| vehicle (new) | 74.6 | 50.6 | 46.2 | 81.5 | +24.0 (5/5) |
| segment (new) | 90.4 | 82.4 | 51.4 | 93.9 | +8.0 (5/5) |
| balance-scale (new) | 92.6 | 87.3 | 88.0 | 88.3 | +5.2 (5/5) |
| mfeat-zernike (new) | 80.5 | 70.5 | 48.0 | 82.5 | +10.0 (5/5) |
| ionosphere (new) | 88.3 | 79.4 | 77.0 | 87.7 | +8.9 (5/5) |
| diabetes (new) | 75.0 | 72.8 | 68.2 | 77.5 | +2.2 (4/5) |
| vertebra-column (new) | 84.9 | 81.9 | 81.1 | 87.1 | +3.0 (4/5) |
| steel-plates (new) | 57.0 | 50.6 | 45.6 | 71.5 | +6.4 (5/5) |

- **Trained vs frozen:** training the tensor layer through the rank signal beats keeping it frozen on all 12 datasets.
- **Scrambled control:** the same signal with its coordinates shuffled is no better than frozen on 11 of 12 datasets.
  On several it is far worse, so the signal's alignment carries the information.
- **Logistic regression:** it is higher on 8 of 12 datasets, equal on wine, and lower on balance-scale and ionosphere.
- **Limits:**
  - one 70/30 split per dataset;
  - a single view, no hidden rank layer and no kNN readout;
  - the rank rule is reimplemented rather than taken from the core;
  - the contrast term (moving away from the nearest wrong filter) goes beyond the core's attraction-only relay.
    Without it the dev gain is about 1 point smaller but still positive.
- **The preprint's failed hybrid is not explained here.** The sign-only conversion and a frozen rank layer also work in
  this sandbox, so the failure probably lies elsewhere, perhaps in the sort direction or in the core's hybrid path.

## Port into the core (2026-09-24)

`core_hybrid.py` builds a core `SortFlowHybridNetwork` with layer types `['tensor', 'sort']`. It replaces, on that
instance only, the method that converts the rank layer's motion into a tensor update. `arrowflow/arrowflow.py` is not
edited, because it is sealed by the published runs. Everything else is the core's own:
- the forward pass and the descending argsort at the interface;
- the output-layer vote, gate and Borda reorder;
- Adam with its schedule, and the training loop.

### Why the preprint's hybrid failed: two bugs in one line

The core's conversion is `F.l1_loss(h, h - lr * m * |h|)`, where m is the motion.

1. **The target is not detached.** The loss is therefore `lr |m| |h|`, and its gradient `lr |m| sign(h)` contains no sign
   of the motion. It only pulls displaced coordinates toward zero. Verified: the gradient is identical when every motion
   flips sign.
2. **The sign is backwards for the interface.** The core ranks tensor outputs largest first, so a coordinate that must
   move earlier needs a larger value. Detached, the formula steps by -sign(m), the wrong way. The in-core sign test
   shows the detached formula moves inputs farther from their true class filter (1381 -> 1723). The detached and
   flipped version moves them closer (1381 -> 760).

### In-core results

`core_confirm.py` uses the same 12 datasets, 70/30 splits and 5 seeds as `confirm.py`, with 60 epochs of batch updates
and the core's defaults otherwise. The table gives the mean held-out test accuracy (%) over the 12 datasets.

| Tensor layer | Frozen | Core original | Minimal repair (detach + flip) | Target (sandbox) | Scrambled |
|---|---|---|---|---|---|
| Linear | 75.4 | 77.6 | 83.5 | 84.2 | 72.1 |
| Core `TensorNet` (3-layer MLP) | 70.5 | 73.4 | 87.2 | 86.5 | 55.7 |

- **Target vs frozen:** the target conversion beats the frozen layer on 11 of 12 datasets with the linear layer (iris
  -0.4) and on 12 of 12 with the core's MLP. The MLP gains include digits 43.9 -> 96.8 and mfeat-zernike 41.0 -> 81.5.
- **Scrambled control:** with the MLP, the scrambled signal is far worse than frozen (55.7 against 70.5), so the
  signal's alignment carries the information.
- **Versus logistic regression** (`confirm.py`): with the core's MLP, the hybrid beats it on iris, segment, balance-scale
  and ionosphere and is close on digits, wine, mfeat-zernike and breast cancer. The MLP is nonlinear, though, so an MLP
  classifier is the fair reference, and it was not run.
- **The repair lives outside the core.** The core's `update_tensor_layer` still has both bugs. No published run uses it
  (they are all-sort networks), but fixing it in `arrowflow.py` would change a sealed file.

## The paper's protocol: nested cross-validation on the 17 datasets (2026-09-24)

`nested.py` (design frozen at 4d26573e0; the run at `runs/2026-09-24-tensor-rank-nested` in the ledger) reuses the
paper's own prepared data and folds:
- 5 x 3 outer folds and 3 inner folds per outer fold;
- fit seeds 8129, 19391 and 39019;
- selection on the inner folds.

The hybrid's folds pair exactly with the paper's recorded models. The table gives mean outer-fold accuracy (%) over the
17 datasets; the full tables are in `runs/nested_tables.md`.

| Model | Mean of 17 |
|---|---|
| hybrid, trained MLP tensor layer, rank readout | 82.8 |
| hybrid, trained MLP tensor layer, kNN on the tensor ranking | 84.4 |
| hybrid, frozen MLP tensor layer, rank readout / kNN readout | 74.5 / 81.5 |
| hybrid, trained linear tensor layer, rank / kNN | 80.7 / 83.3 |
| ArrowFlow-kNN (the paper's method: fixed encoder, 7 views, hidden rank layer, kNN) | 84.5 |
| MLP / RBF SVC / random forest / gradient boosting / numeric kNN (the paper's comparators) | 85.4 / 85.8 / 84.2 / 84.2 / 83.5 |

**Primary contrasts** (corrected resampled t, Holm over 17):
- **P1, trained vs frozen tensor layer, rank readout:** higher on 17/17 datasets, significantly on 11/17, and never
  lower. The largest gains are vehicle +23.7, digits +17.9, mfeat-zernike +17.3, balance-scale +12.6, climate +11.8,
  segment +11.7 and ionosphere +10.0.
- **P2, hybrid (rank readout) vs the paper's MLP:** higher on 3/17, none significantly. Significantly lower on 4: wine
  quality -11.1, steel plates -10.5, vehicle -7.8 and segment -3.5.
- **P3, hybrid (kNN readout) vs ArrowFlow-kNN:** higher on 9/17, and 84.4 against 84.5 on average.
  - Significantly higher on balance-scale, +3.5 (15/15 folds).
  - Significantly lower on vehicle (-4.5) and steel plates (-4.6).
  - Consistently higher on mfeat-zernike (+2.1) and climate (+2.6), each in 14/15 folds but not significant after Holm.

**Reading:** the rank layer's signal, converted by target propagation, trains a tensor encoder under the paper's
protocol with a large and consistent effect. Much larger than the paper's own training effect of the hidden rank
layers, whose mean gain over untrained ArrowFlow is about 1 point, significant on 2 of 34 comparisons. With one view and
no hidden rank layer, the learned encoder matches the paper's full ArrowFlow-kNN on average. It does not reach the tuned
MLP or SVC. The frozen control is a random MLP, a weaker fixed encoder than the paper's, so P1 is the size of what the
signal teaches, not the gain over the paper's encoder. The encoder-swap experiment would measure that gain: the paper's
pipeline with only the encoder replaced.

## Encoder swap: the paper's pipeline with only the encoder replaced (2026-09-25)

`encoder_swap.py` (design frozen at de380080c; run at `runs/2026-09-25-encoder-swap` in the ledger) rebuilds
ArrowFlow-kNN at the configuration the paper's nested CV selected in each outer fold, on the same folds and fit seeds.
Only each view's encoder changes. Before the run, the fixed arm refitted by this module reproduced the registered
predictions exactly (4 datasets x 3 seeds, 0 differing rows). During the run, five jobs were computed twice,
independently, and gave identical results. Mean outer-fold accuracy (%):

| Encoder in ArrowFlow-kNN | Mean of 17 |
|---|---|
| fixed (the paper's) | 84.5 |
| learned (TensorNet trained by the rank signal) | 85.1 |
| random (the same TensorNet, untrained) | 82.5 |
| the paper's MLP comparator | 85.4 |
| the nested hybrid, kNN on the tensor ranking (one view, no hidden rank layer) | 84.4 |

- **Primary, learned - fixed:** +0.56 points; higher on 11/17. Significantly higher on 1 (balance-scale +4.5, 15/15
  folds) and significantly lower on 0. The pre-registered verdict is **improves ArrowFlow**.
  - Consistent but not Holm-significant gains: ionosphere +4.0, climate +2.9 (14/15 folds) and mfeat-zernike +1.8
    (14/15 folds).
  - Losses: vehicle -3.7 (Holm 0.45), diabetes -1.7, steel plates -1.7 and segment -0.6.
- **Learned - random:** +2.55; higher on 15/17, significantly on 4, lower on none. The gain comes from training the
  encoder through the rank signal, not from the MLP shape: an untrained MLP encoder is worse than the paper's fixed one
  (random - fixed -1.99, significantly lower on 3).
- **Learned - the paper's MLP:** -0.30. Significantly higher on wine quality (+4.6) and significantly lower on vehicle
  (-6.4) and steel plates (-2.3). ArrowFlow with the learned encoder is on par with the tuned MLP; with the fixed
  encoder it trailed by 0.9.
- **Learned - the nested hybrid (kNN readout):** +0.67; higher on 15/17, none significantly. The hidden rank layer and
  the seven views add little on top of the learned encoder.
- **Caveat:** the configurations were selected for the fixed encoder, so the learned arm runs at settings chosen for its
  competitor. This makes the comparison conservative.
