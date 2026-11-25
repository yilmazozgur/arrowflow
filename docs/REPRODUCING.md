# Reproducing the studies

This file lists, for every study of the paper, its protocol or the module header that fixes its design, its commands in
the order of the production run, the inputs it reads and whether it reruns from this repository alone. It also records
what the logs of the production runs hold and when each study was fixed. The paper's appendix on reproducibility gives
the protocols with their freeze dates and digests, the datasets with their content hashes and the software environment.

Run every command from the repository root. The command listings use two shell variables:

```bash
export H=experiments.make_revision            # the harness package, as a module path
export P=experiments/make_revision/protocols  # the protocol directory
```

## 1. Layout

| Path | Content |
|---|---|
| `arrowflow/` | the ArrowFlow library: ranking layers, accumulator, training loop and the encoder network `TensorNet` (`arrowflow.py`), the ranking conventions (`ranking.py`) and the readouts of the laboratory (`readouts.py`, `permlvq.py`) |
| `experiments/make_revision/` | the evaluation harness (`$H`): data loaders, nested cross-validation, every study runner, the analyses, and three unfrozen protocol templates of 2026-09-11 (`protocol.json`, `study_protocol.json`, `secondary_protocol.json`) that some runners read |
| `experiments/make_revision/protocols/` | the frozen protocol files (`$P`), one per study, in five dated folders |
| `experiments/make_revision/gene_data_audit.json` | the recorded digests of the gene-expression archive, which its loader checks |
| `experiments/tensor_rank_sandbox/` | the trained encoder network (`core_hybrid.py`), its three studies (`nested.py`, `encoder_swap.py`, `nested_reference.py`), and the development searches with their records (`runs/`, `RESULTS.md`) |
| `experiments/artificial_robustness/` | the preregistration, protocol, module and exploratory check of the stopped study of missing items |
| `experiments/exp_knn_vs_arrowflow.py` | an older comparison script, kept because two test modules import it |
| `tests/` | the test suite of the library and the harness (`tests/make_revision/`) |
| `docs/secondary_experiment_spec.md` | the specification of the studies of 2026-09-11, whose digest the matched-controls runner records |
| `requirements-revision.txt` | the pinned package versions |
| `pyproject.toml` | the package and its version ranges |
| `LICENSE` | the license of the code |

ArrowFlow and its prototype readout encode their inputs with `OrdinalEncoder` in `$H/models.py` and train through
`MultiViewArrowFlowKNN` and `MultiViewArrowFlow` in `$H/multiview.py`. The single-view networks of the matched learning
controls use `OrdinalEncoder` and `ArrowFlowEstimator` of `$H/models.py` directly. The three encoder studies train the
encoder network through `experiments/tensor_rank_sandbox/core_hybrid.py`. That module installs the conversion of motions
into target scores on an instance of the library's `SortFlowHybridNetwork`, so `arrowflow/arrowflow.py` is unchanged.
The library also holds older encoders and trainers (`DataGraph.project_random_space`, `target_aware_encode`,
`calibrated_encode` and `SortFlow_MoE` in `arrowflow/arrowflow.py`, and the multi-view trainer of
`arrowflow/benchmark.py`). No reported run uses them. The harness borrows only `ArrowFlowConfig` and
`_build_sortnet_config` from `arrowflow/benchmark.py`.

The two relays of the paper live in different files. The relay is part of `arrowflow/arrowflow.py`, where
`backward_propagate` passes to the layer below the motion that `Vertex.accumulate_motion` returns. Every run used it,
except two arms of the depth study with the signed relay. The signed relay of those two arms is the class
`RelayedMotion` of `$H/signed_relay.py`. It replaces that return value on the running network only, so
`arrowflow/arrowflow.py` is unchanged.

## 2. The code that ran

Every code file in `arrowflow/` and `experiments/` is byte-identical to the development revision that the paper's runs
used, with three exceptions of one line each:

- `experiments/make_revision/run_studies.py`, line 38, and `experiments/make_revision/run_secondary.py`, line 32, name
  the specification of the matched learning controls at its new path, `docs/secondary_experiment_spec.md`. In the
  development repository the file lay in the manuscript folder, which this repository does not hold. Its content is
  unchanged, so its digest is the one that the matched runs recorded under the old path. A new run of these runners
  records the specification under its new path and a different digest of the runner itself.
- `experiments/artificial_robustness/exploratory_representation_check.py` lost its line 5, which put an absolute path of
  the author's machine on `sys.path`. The script runs with `python -m` from the repository root.

In `tests/`, two modules add skip guards to four tests. Three tests in `tests/make_revision/test_label_controls.py` skip
without the registered motion run, and
`test_the_committed_artificial_pins_hold_for_the_committed_module_and_equal_a_fresh_pin` in
`tests/make_revision/test_extra_data.py` skips without the development commit that `$P/2026-09-14/artificial_pins.json`
records.

The `environment.json` of each harness run records the `sha256` of every source file that the run sealed. For every run
from the ArrowFlow run (2026-09-12) onward, these files are byte-identical to the files here. The first benchmark run,
the component ablation, the laboratory and the matched controls ran at earlier revisions, in which up to four harness
files differed. So did two runs that the main text leaves out, the gene-expression study and the native-ranking study of
2026-09-12, in which two files and one file differed. The four runs of 2026-09-11 ran at an earlier revision still, at
which up to seven sealed files differed, `arrowflow/arrowflow.py` among them. The encoder studies and the permuted-label
control record their commit in `code_revision.txt`. Every code file that existed at those commits is byte-identical
here, apart from the changes above.

The files in `$P` and the protocol in `experiments/artificial_robustness/` are the originals that the runs used. The
paper's table of protocols prints the first twelve hexadecimal digits of the `sha256` of each protocol of the main text
and of the three modules whose headers fix the designs of the encoder studies, and the files here match them. Section 10
lists the digests of the protocol files of the studies that the main text leaves out.

## 3. Environment and tests

Use Python 3.12.7 and the versions pinned in `requirements-revision.txt` (NumPy 1.26.4, SciPy 1.14.1, scikit-learn
1.5.2, pandas 2.2.3, PyTorch 2.6.0, joblib 1.4.2, threadpoolctl 3.5.0, Matplotlib 3.9.2, and pytest 9.1.1 for the
tests). Every worker runs single-threaded, and no run used a graphics processor. The harness needs Linux or macOS,
because it uses the POSIX modules `resource` and `fcntl`.

```bash
python -m pip install -r requirements-revision.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
```

The runners record the code revision with `git rev-parse HEAD`, and several refuse code with uncommitted changes, so the
code must sit in a git clone. A test skips when an input it needs is absent: the run records, an OpenML download that is
not in the local scikit-learn cache, the gene-expression archive, the SUSHI archive (`ARROWFLOW_SUSHI_ARCHIVE`), a CUDA
device, or a file or commit of the development history. The README gives the outcome of the suite on a clone without the
run records.

The worked example of the paper's method appendix is checked by the test
`test_checked_four_item_three_filter_two_class_trace` in `tests/make_revision/test_training_contract.py`, which replays
it with the library's own classes.

## 4. Data

The seven benchmark datasets load from scikit-learn (Iris, Wine, Breast cancer, Digits) and from OpenML (data ID 40691
for Wine quality, 54 for Vehicle and 36 for Segment). The ten further datasets load from OpenML. Each is pinned by
identifier, version, file checksum and content hash in `$H/newdata.py` and in its batch protocol, and the loader refuses
a copy that differs. OpenML files download once into the scikit-learn data home and load from there afterwards. This
command prints the first twelve hexadecimal digits of the content hash of each dataset, which the paper's table of
datasets lists:

```bash
python - <<'EOF'
import os, importlib
H = os.environ['H']
rr = importlib.import_module(H + '.run_revision')
nd = importlib.import_module(H + '.newdata')
ev = importlib.import_module(H + '.evaluation')
for name in ['iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits']:
    X, y, m = rr.load_dataset(name)
    print(name, ev.dataset_fingerprint(X, y, m['feature_names'], m['label_map'])[:12])
for name in nd.PANEL:
    X, y, m = nd.load_newdata(name)
    print(name, ev.dataset_fingerprint(X, y, m['feature_names'], m['label_map'])[:12])
EOF
```

The hash is the `sha256` digest of the table's shape, feature names and class names, followed by every feature value as
a double-precision number and every label (`dataset_fingerprint` in `$H/evaluation.py`).

Three more sources serve the studies that the main text leaves out, and the matched controls:

- The SUSHI preference data. Their provider permits research use but not redistribution. Obtain the archive
  (`sushi3-2016.zip`) from <https://www.kamishima.net/sushi/> and pass it with `--sushi-archive` or the variable
  `ARROWFLOW_SUSHI_ARCHIVE`. The loader in `$H/datasets.py` checks the digest of the archive and of its members.
- The gene-expression archive of the UCI repository. `$H/gene.py` downloads it once into `~/.cache/arrowflow/gene`, or
  into the folder that `--gene-cache` of `run_revision` or the variable `ARROWFLOW_GENE_CACHE` names, and checks it
  against `$H/gene_data_audit.json`.
- The generated ranking patterns. `$H/artificial_ranks.py` generates them, and `$P/2026-09-14/artificial_pins.json` pins
  them.

## 5. Run records, default paths and pins

The run records (about 13 GB: prepared data and splits, per-example predictions, selection records, summaries, sealed
analyses and production logs) are not in this repository. They will be released with the published paper. Until then,
every command of Section 7 that reads earlier runs has nothing to read, and the tests that need them skip. The script
that renders the paper's tables and figures from the run records is not in this repository either.

Conventions of the command listing:

- `<run>` is the output directory of a study. Use a new one for each run, because the runners refuse to replace a result
  file.
- `{a|b}` stands for one command per alternative, run in the order given.
- `<references>` stands for the three reference runs of the controlled experiments, in this form:

  ```text
  --reference bridge_knn=<ArrowFlow run>,<component ablation of ArrowFlow run>
  --reference newdata_batch1=<batch 1 run>,<further-dataset ablation run>
  --reference newdata_batch2=<batch 2 run>,<further-dataset ablation run>
  ```

- `<runs directory>` is a folder that holds the registered runs under the names that the production runs used:
  `2026-09-12-ablation`, `2026-09-12-bridge`, `2026-09-12-bridge-knn`, `2026-09-12-contrasts`, `2026-09-12-devlab`,
  `2026-09-12-knn-vs-full`, `2026-09-12-matched`, `2026-09-13-knn-ablation`, `2026-09-13-knn-projected`,
  `2026-09-13-knn-training`, `2026-09-13-referee-analyses`, `2026-09-14-artificial`, `2026-09-14-baselines`,
  `2026-09-14-dedup`, `2026-09-14-depth-aggregation`, `2026-09-14-holistic`, `2026-09-14-mechanism-analysis`,
  `2026-09-14-motion`, `2026-09-14-newdata-ablation`, `2026-09-14-newdata-analysis`, `2026-09-14-newdata-batch1`,
  `2026-09-14-newdata-batch2`, `2026-09-14-training-diagnostics`, `2026-09-14-training-diagnostics-tables`,
  `2026-09-23-representation-test`, `2026-09-23-signed-relay-depth`, `2026-09-24-tensor-rank-nested`,
  `2026-09-25-encoder-swap`, `2026-11-24-encoder-reference` and `2026-11-25-permuted-labels`.
- `--workers` gives the number of single-thread worker processes. The listing gives the counts of the production runs
  where their scripts or logs record them, and otherwise 16, the cap of the harness.

**Defaults.** Where a command takes earlier runs as options, pass them. The defaults of these options point to a folder
of the author's workspace next to the repository: `WORKSPACE_RUNS` in `$H/newdata.py` (line 70) and in `$H/holistic.py`
(line 79). The options are `--runs` (`holistic`, `mechanism_analysis`, `compare_baselines`), `--reference`
(`motion_controls`, `training_diagnostics`, `interventions`, `signed_relay_depth`, `representation_test`, whose default
is `DEFAULT_SOURCES` in `$H/training_diagnostics.py`, lines 147 to 149), `--batch1` and `--batch2`
(`run_newdata_ablation`), `--source-depth-run` (`signed_relay_depth`), `--reference` (`native_artificial`) and
`--encoded` (`compare_native_artificial`). `compare_extra analyse --family dedup` has no default and needs `--runs`.

**Fixed paths.** These commands read earlier runs from a fixed folder and take no option for it:

- `label_controls` `check`, `run` and `analyse` read the motion run, its analysis and the training diagnostics from
  `MOTION_RUN`, `MOTION_ANALYSIS` and `DIAGNOSTICS_RUN` (`$H/label_controls.py`, lines 197 to 199), which lie under
  `WORKSPACE_RUNS`. The module pins the digests of the files that it reads.
- `experiments.tensor_rank_sandbox.nested`, `encoder_swap` and `nested_reference` read the prepared data, splits and
  results of the ArrowFlow run and of both further-dataset batches from `RUNS_ROOT`
  (`experiments/tensor_rank_sandbox/nested.py`, line 44), the same folder. `encoder_swap analyse` and
  `nested_reference analyse` take the nested run with `--nested`. `nested_reference check` reads the registered nested
  run unless `--nested` names another, and `nested_reference analyse` reads the MLP results from `RUNS_ROOT` unless
  `--mlp` names another folder.
- The `project` stages of `newdata`, `extra_runs`, `neighbour_baselines` and `native_artificial` read the calibration
  runs from `CALIBRATION` (`$H/newdata.py`, lines 662 to 665). These stages served the freeze of a protocol and are not
  part of a rerun.

To rerun these commands, place the run records in the folder that `WORKSPACE_RUNS` names. It lies in the parent folder
of the clone.

**Recorded paths.** The `prepare` stage of a study that builds on earlier runs records their absolute paths in its
manifest. The `summary` stages of `run_knn_ablation`, `run_newdata_ablation`, `extra_ablation`, `motion_controls`,
`training_diagnostics`, `interventions`, `signed_relay_depth` and `representation_test` read the reference runs again
from these recorded paths. So do the `run` stages of `motion_controls`, `representation_test` and `signed_relay_depth`
and the `analyse` stages of `interventions` and `signed_relay_depth`. `compare_motion analyse` and
`compare_representation analyse` accept a `--reference` only at the recorded path. The registered runs record the paths
of the author's machine.

**Pins.** A protocol that builds on an earlier run pins that run by its code revision and file digests, and the
preparation, pairing or analysis stage checks these pins. The pinned runs are the registered runs of the paper, so these
checks refuse any other run. The component ablation and the laboratory check a run of `bridge.json` by its protocol
identity only, and the contrasts of the component ablation check a run of `matched_v3.json` in the same way.

## 6. The studies

Each row names a study as the paper's table of protocols names it, its protocol or the module header that fixes its
design, its stages in the order of the production run, what it reads besides the code and the datasets, and whether it
reruns from this repository alone. "After" names the studies whose runs it reads and which rerun from the repository
too. The smoke and pilot stages that came before each freeze are left out, except the pilots of the laboratory, which
authorise its runs.

| Study | Protocol or design | Stages, in order | Reads besides the datasets | Reruns alone |
|---|---|---|---|---|
| First benchmark (prototype readout) | `2026-09-12/bridge.json` | `run_revision` prepare and run, `reporting` | nothing | yes |
| ArrowFlow (nearest-neighbour readout) | `2026-09-12/bridge_knn.json` | `run_revision` prepare and run, `reporting`, `compare_runs knn` | `compare_runs knn` reads the registered first benchmark run, which the protocol pins | run stages only |
| Matched learning controls | `2026-09-12/matched_v3.json` | `run_studies` prepare, run and summary | the SUSHI archive | yes |
| Component ablation and contrasts | `2026-09-12/ablation.json` | `run_bridge` prepare, ablation, summary and contrasts | a run of `bridge.json` and a run of `matched_v3.json` | yes, after the first benchmark and the matched controls |
| Laboratory: readouts and permutation LVQ (inner folds) | `2026-09-12/devlab.json`, not frozen | `devlab` pilot, readouts and summary, then pilot, permlvq and summary | a run of `bridge.json` | yes, after the first benchmark |
| Training controls | `2026-09-12/knn_training.json` | `run_revision` prepare, `compare_runs training-pairing`, `run_revision` run, `reporting`, `compare_runs training` | the registered ArrowFlow run, which the protocol pins | no |
| Component ablation of ArrowFlow, benchmark datasets | `2026-09-12/knn_ablation.json` | `run_knn_ablation` prepare, ablation and summary | the registered ArrowFlow run, which the protocol pins | no |
| Comparator intervals, ranks and duplicate rows | none, fits no model | `referee_analyses` comparators, duplicates and ranks | a run of `bridge_knn.json`. `duplicates` also reads the registered first benchmark run, which `bridge_knn.json` pins | comparators and ranks after ArrowFlow, duplicates no |
| Sorting control | `2026-09-12/knn_projected.json` | `run_revision` prepare, `compare_runs projected-pairing`, `run_revision` run, `reporting`, `compare_runs projected` | the registered ArrowFlow and training-control runs, which the protocol pins | no |
| Further datasets, batches 1 and 2 | `2026-09-12/newdata_batch1.json`, `2026-09-12/newdata_batch2.json` | `newdata` prepare of both batches, `compare_newdata pairing`, `newdata` run and `reporting` of each batch, `compare_newdata analyse` | nothing | yes |
| Combined analysis of the 17 datasets | none, fits no model | `holistic` benchmark, training and ready | the registered runs, through `--runs` | no |
| Component ablation of ArrowFlow, further datasets | `2026-09-12/newdata_ablation.json` | `run_newdata_ablation` prepare, ablation and summary, `holistic components` | the registered batch runs, which the protocol pins | no |
| Matched motion-signal controls | `2026-09-14/motion_controls.json` | `motion_controls` prepare, run and summary, `compare_motion analyse` | `<references>`, which the protocol pins | no |
| Training diagnostics | `2026-09-14/training_diagnostics.json` | `training_diagnostics` run and summary, `diagnostics_tables tables` | `<references>`, which the protocol pins | no |
| Mechanism analysis | none, fits no model | `mechanism_analysis analyse` | the registered runs, through `--runs` | no |
| Fixed-configuration depth, aggregation inside the update | `2026-09-14/depth.json`, `2026-09-14/aggregation.json` | `interventions` prepare, run, summary and analyse, for each family | `<references>`, which the protocols pin | no |
| Duplicate-free rerun | `2026-09-14/dedup.json` | `extra_runs` prepare and run, `reporting`, `extra_ablation` prepare, ablation and summary, `compare_extra analyse` | `compare_extra analyse` reads the registered full-data runs through `--runs` | run stages only |
| Neighbour baselines | `2026-09-14/neighbour_baselines.json` | `neighbour_baselines` prepare, `compare_baselines pairing`, `neighbour_baselines` run, `reporting`, `compare_baselines analyse` | the registered ArrowFlow-kNN runs through `--runs`, which the protocol pins | no |
| Depth with the signed relay | `2026-09-23/signed_relay_depth.json` | `signed_relay_depth` prepare, run, summary and analyse | `<references>` and the depth run, which the protocol pins | no |
| Representation test | `2026-09-23/representation_test.json` | `representation_test` prepare, run and summary, `compare_representation analyse` | `<references>`, which the protocol pins | no |
| Learned encoder: classifier under nested cross-validation | header of `experiments/tensor_rank_sandbox/nested.py` | `nested` run and analyse | the ArrowFlow run and both batch runs, from `RUNS_ROOT` | no |
| Learned encoder: encoder swap inside ArrowFlow | header of `experiments/tensor_rank_sandbox/encoder_swap.py` | `encoder_swap` check, run and analyse | the same runs, and the nested run | no |
| Learned encoder: reference study, standard initialisation and backpropagation | header of `experiments/tensor_rank_sandbox/nested_reference.py` | `nested_reference` check, run and analyse | the same runs, the nested run and the registered MLP results | no |
| Permuted-label control | `2026-11-25/permuted_labels.json`, whose digest `$H/label_controls.py` pins | `label_controls` check, run and analyse | the motion run and the training diagnostics, from `WORKSPACE_RUNS`, with pinned digests | no |
| Development of the trained encoder network, single splits | none | `search` stages 0, 1 and 2, `confirm`, `core_confirm`, `mlp_reference`, `conversion_check` | nothing | yes |

The studies that the main text leaves out (Section 10):

| Study | Protocol | Stages, in order | Reads besides the datasets | Reruns alone |
|---|---|---|---|---|
| Nested benchmark of a single-view configuration | `2026-09-11/e02.json` | `run_revision` prepare and run, `reporting` | nothing | yes |
| Matched study of single-view networks | `2026-09-11/matched.json` | `run_studies` prepare, run and summary | the SUSHI archive | yes |
| Native ranking study | `2026-09-11/native.json` | `run_revision` prepare and run, `reporting` | the SUSHI archive | yes |
| Secondary studies | `2026-09-11/secondary.json` | `run_secondary` prepare, run, summary and cost | the SUSHI archive. `cost` reads the three runs above | yes, after the three runs above |
| Native ranking study, second protocol | `2026-09-12/native_v3.json` | `run_revision` prepare and run, `reporting` | the SUSHI archive | yes |
| Gene expression | `2026-09-12/gene.json` | `run_revision` prepare, `run_gene selector_prepass`, `run_revision` run, `reporting`, `run_gene gene_corruption`, `run_gene summary` | the gene-expression archive | yes |
| Generated ranking patterns, with the encoder | `2026-09-14/artificial.json` | through `artificial_runs`: `extra_runs` prepare and run, `reporting`, `extra_ablation` prepare, ablation and summary, `compare_extra analyse` | nothing | yes |
| Generated ranking patterns, orders without the encoder | `2026-09-14/native_artificial.json` | `native_artificial` reference, prepare, data_properties, run and reporting, `compare_native_artificial analyse` | the registered run of `artificial.json`, which the protocol pins | no |
| Missing items at test time, stopped | `experiments/artificial_robustness/protocol.json` | `experiments.artificial_robustness.study` verify, run, original and analyse | nothing | yes |

## 7. Commands, study by study

```bash
# First benchmark run (prototype readout)
python -m $H.run_revision {smoke|pilot|prepare|run} \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry $H.bridge:bridge_registry \
   --protocol $P/2026-09-12/bridge.json --output <run> --workers 16
python -m $H.reporting --output <run>

# ArrowFlow (nearest-neighbour readout) on the benchmark's outer partitions
python -m $H.run_revision prepare \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry $H.bridge:bridge_knn_registry \
   --protocol $P/2026-09-12/bridge_knn.json --output <run>
python -m $H.run_revision run \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry $H.bridge:bridge_knn_registry \
   --protocol $P/2026-09-12/bridge_knn.json --output <run> --workers 16
python -m $H.reporting --output <run>
python -m $H.compare_runs knn --knn-source <run> \
   --bridge-source <benchmark run> --output <comparison directory>

# Matched learning controls
python -m $H.run_studies {smoke|pilot|prepare|run} \
   --protocol $P/2026-09-12/matched_v3.json --output <run> --workers 16
python -m $H.run_studies summary --output <run>

# Component ablation of the prototype readout and the 14 contrasts
python -m $H.run_bridge smoke \
   --protocol $P/2026-09-12/ablation.json --output <smoke directory>
python -m $H.run_bridge pilot \
   --protocol $P/2026-09-12/ablation.json --bridge-source <benchmark run> \
   --output <pilot directory>
python -m $H.run_bridge prepare \
   --protocol $P/2026-09-12/ablation.json --bridge-source <benchmark run> \
   --output <run>
python -m $H.run_bridge ablation \
   --protocol $P/2026-09-12/ablation.json --output <run> --workers 16
python -m $H.run_bridge summary --output <run>
python -m $H.run_bridge contrasts \
   --ablation-source <run> --matched-source <matched run> \
   --output <contrasts directory>

# Inner-fold laboratory (never scores an outer fold)
python -m $H.devlab pilot --family {readouts|permlvq} \
   --bridge-source <benchmark run> --output <run> --workers 16
python -m $H.devlab {readouts|permlvq} \
   --bridge-source <benchmark run> --output <run> --workers 16
python -m $H.devlab summary --family {readouts|permlvq} \
   --output <run>

# Training controls (untrained ArrowFlow and tuned input footrule kNN)
python -m $H.run_revision prepare \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --protocol $P/2026-09-12/knn_training.json \
   --registry $H.knn_controls:knn_training_registry \
   --output <run>
python -m $H.compare_runs training-pairing \
   --training-source <run> --knn-source <ArrowFlow run>
python -m $H.run_revision run \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --protocol $P/2026-09-12/knn_training.json \
   --registry $H.knn_controls:knn_training_registry \
   --output <run> --workers 16
python -m $H.reporting --output <run>
python -m $H.compare_runs training \
   --training-source <run> --knn-source <ArrowFlow run> \
   --output <run>/compare_training

# Component ablation of ArrowFlow at its selected configurations
python -m $H.run_knn_ablation prepare \
   --protocol $P/2026-09-12/knn_ablation.json --reference-source <ArrowFlow run> \
   --output <run>
python -m $H.run_knn_ablation ablation \
   --protocol $P/2026-09-12/knn_ablation.json --reference-source <ArrowFlow run> \
   --output <run> --workers 16
python -m $H.run_knn_ablation summary --output <run>

# Comparator intervals, ranks and duplicate rows (descriptive. No model is fitted)
python -m $H.referee_analyses comparators \
   --knn-source <ArrowFlow run> --output <run>/comparators
python -m $H.referee_analyses duplicates \
   --knn-source <ArrowFlow run> --bridge-source <benchmark run> --output <run>/duplicates
python -m $H.referee_analyses ranks \
   --knn-source <ArrowFlow run> --output <run>/ranks

# Sorting control (numeric kNN on the projected scores)
python -m $H.run_revision prepare \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --protocol $P/2026-09-12/knn_projected.json \
   --registry $H.projected_knn:knn_projected_registry \
   --output <run>
python -m $H.compare_runs projected-pairing \
   --projected-source <run> --knn-source <ArrowFlow run> \
   --training-source <training controls run>
python -m $H.run_revision run \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --protocol $P/2026-09-12/knn_projected.json \
   --registry $H.projected_knn:knn_projected_registry \
   --output <run> --workers 16
python -m $H.reporting --output <run>
python -m $H.compare_runs projected \
   --projected-source <run> --knn-source <ArrowFlow run> \
   --training-source <training controls run> --output <run>/compare_projected

# Further datasets (two batches, analysed together)
python -m $H.newdata prepare \
   --protocol $P/2026-09-12/newdata_batch1.json --output <batch 1 run>
python -m $H.newdata prepare \
   --protocol $P/2026-09-12/newdata_batch2.json --output <batch 2 run>
python -m $H.compare_newdata pairing \
   --batch1 <batch 1 run> --batch2 <batch 2 run>
python -m $H.newdata run \
   --protocol $P/2026-09-12/newdata_batch1.json --output <batch 1 run> --workers 16
python -m $H.reporting --output <batch 1 run>
python -m $H.newdata run \
   --protocol $P/2026-09-12/newdata_batch2.json --output <batch 2 run> --workers 16
python -m $H.reporting --output <batch 2 run>
python -m $H.compare_newdata analyse \
   --batch1 <batch 1 run> --batch2 <batch 2 run> --output <analysis directory>

# Combined analysis of the 17 datasets (no model is fitted)
python -m $H.holistic benchmark --runs <runs directory> --output <run>/benchmark
python -m $H.holistic training --runs <runs directory> --output <run>/training
python -m $H.holistic ready --root <run>

# Component ablation on the further datasets, then the component table of all 17
python -m $H.run_newdata_ablation prepare \
   --protocol $P/2026-09-12/newdata_ablation.json --batch1 <batch 1 run> \
   --batch2 <batch 2 run> --output <run>
python -m $H.run_newdata_ablation ablation \
   --protocol $P/2026-09-12/newdata_ablation.json --batch1 <batch 1 run> \
   --batch2 <batch 2 run> --output <run> --workers 16
python -m $H.run_newdata_ablation summary --output <run>
python -m $H.holistic components --runs <runs directory> \
   --output <combined analysis run>/components

# Matched motion-signal controls
python -m $H.motion_controls {prepare|run} \
   --protocol $P/2026-09-14/motion_controls.json <references> \
   --output <run>/run --workers 16
python -m $H.motion_controls summary --output <run>/run
python -m $H.compare_motion analyse --run <run>/run \
   <references> --output <run>/analysis

# Training diagnostics, and their tables
python -m $H.training_diagnostics run \
   --protocol $P/2026-09-14/training_diagnostics.json <references> \
   --output <run> --workers 8
python -m $H.training_diagnostics summary --output <run>
python -m $H.diagnostics_tables tables --diagnostics <diagnostics run> \
   --ablation bridge_knn=<component ablation of ArrowFlow run> \
   --ablation newdata_batch1=<further-dataset ablation run> \
   --ablation newdata_batch2=<further-dataset ablation run> --output <tables run>

# Balance-scale and mfeat-zernike mechanism (descriptive. No model is fitted)
python -m $H.mechanism_analysis analyse \
   --runs <runs directory> --output <run>

# Fixed-configuration depth, and aggregation inside the update
python -m $H.interventions prepare \
   --protocol $P/2026-09-14/{depth|aggregation}.json <references> --output <run>/run
python -m $H.interventions run \
   --protocol $P/2026-09-14/{depth|aggregation}.json --output <run>/run --workers 16
python -m $H.interventions summary --output <run>/run
python -m $H.interventions analyse --run <run>/run \
   --output <run>/analysis

# Duplicate-free rerun of Wine quality and Segment
python -m $H.extra_runs prepare \
   --protocol $P/2026-09-14/dedup.json --output <run>/run
python -m $H.extra_runs run \
   --protocol $P/2026-09-14/dedup.json --output <run>/run --workers 16
python -m $H.reporting --output <run>/run
python -m $H.extra_ablation {prepare|ablation} \
   --protocol $P/2026-09-14/dedup.json --reference <run>/run \
   --output <run>/ablation --workers 16
python -m $H.extra_ablation summary --output <run>/ablation
python -m $H.compare_extra analyse --family dedup \
   --run <run>/run --ablation <run>/ablation --output <run>/analysis \
   --runs <runs directory>

# Neighbour baselines (LDA, PCA and NCA with kNN, and a Kendall-kernel SVC)
python -m $H.neighbour_baselines prepare \
   --protocol $P/2026-09-14/neighbour_baselines.json --output <run>/run
python -m $H.compare_baselines pairing --run <run>/run \
   --runs <runs directory>
python -m $H.neighbour_baselines run \
   --protocol $P/2026-09-14/neighbour_baselines.json --output <run>/run --workers 16
python -m $H.reporting --output <run>/run
python -m $H.compare_baselines analyse --run <run>/run \
   --output <run>/analysis --runs <runs directory>

# Depth with the signed relay
python -m $H.signed_relay_depth prepare \
   --protocol $P/2026-09-23/signed_relay_depth.json <references> \
   --source-depth-run <depth run>/run --output <run>/run
python -m $H.signed_relay_depth run \
   --protocol $P/2026-09-23/signed_relay_depth.json --output <run>/run --workers 16
python -m $H.signed_relay_depth summary --output <run>/run
python -m $H.signed_relay_depth analyse --run <run>/run \
   --output <run>/analysis

# Representation test (a Kendall-kernel SVC on the input, untrained and trained rankings)
python -m $H.representation_test {prepare|run} \
   --protocol $P/2026-09-23/representation_test.json <references> \
   --output <run>/run --workers 16
python -m $H.representation_test summary --output <run>/run
python -m $H.compare_representation analyse --run <run>/run \
   <references> --output <run>/analysis

# The trained encoder network. Development on single splits. These modules write their records into
# experiments/tensor_rank_sandbox/runs/ over the committed ones, and search appends to runs/search.jsonl.
python -m experiments.tensor_rank_sandbox.search {0|1|2}
python -m experiments.tensor_rank_sandbox.{confirm|core_confirm|mlp_reference}
python -m experiments.tensor_rank_sandbox.conversion_check
# the classifier under nested cross-validation, then the encoder swap
python -m experiments.tensor_rank_sandbox.nested run --output <run> --workers 14
python -m experiments.tensor_rank_sandbox.nested analyse --output <run>
python -m experiments.tensor_rank_sandbox.encoder_swap check
python -m experiments.tensor_rank_sandbox.encoder_swap run --output <run> --workers 14
python -m experiments.tensor_rank_sandbox.encoder_swap analyse --output <run> \
   --nested <nested run>
# the reference study (standard initialisation, and backpropagation with cross-entropy)
python -m experiments.tensor_rank_sandbox.nested_reference check
python -m experiments.tensor_rank_sandbox.nested_reference run --output <run> --workers 24
python -m experiments.tensor_rank_sandbox.nested_reference analyse --output <run> \
   --nested <nested run>

# Permuted-label control, one more arm of the matched motion-signal controls. It reads the motion run and the
# training diagnostics, and it refits ArrowFlow in every job. `check` refits two jobs and prints counts of
# differing predictions, never a score. `analyse` writes analysis.json, ANALYSIS.md and nine tables in <run>/tables.
python -m $H.label_controls check
python -m $H.label_controls run --output <run> --workers 24
python -m $H.label_controls analyse --output <run>
```

The commands of the studies that the main text leaves out follow. Those of the four runs of 2026-09-11 come from their
execution record. Those of the gene-expression study and of the two studies of generated patterns come from their
production scripts. The registry and the dataset of the second native-ranking study are those that its run directory
records, and the commands of the stopped study are those of its module's docstring.

```bash
# 2026-09-11: nested benchmark of a single-view configuration, native ranking study, matched study, secondary studies
python -m $H.run_revision {prepare|run} \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry $H.comparisons:e02_registry \
   --protocol $P/2026-09-11/e02.json --output <e02 run> --workers 16
python -m $H.reporting --output <e02 run>
python -m $H.run_revision {prepare|run} --dataset sushi_childhood_east_west_v2 \
   --registry $H.comparisons:native_registry \
   --protocol $P/2026-09-11/native.json --output <native run> --workers 16
python -m $H.reporting --output <native run>
python -m $H.run_studies {prepare|run} \
   --protocol $P/2026-09-11/matched.json --output <matched run> --workers 16
python -m $H.run_studies summary --output <matched run>
python -m $H.run_secondary {prepare|run} \
   --protocol $P/2026-09-11/secondary.json --output <secondary run> --workers 16
python -m $H.run_secondary summary --output <secondary run>
python -m $H.run_secondary cost \
   --protocol $P/2026-09-11/secondary.json --output <secondary run> \
   --e02-source <e02 run> --native-source <native run> --matched-source <matched run>

# 2026-09-12: native ranking study, second protocol
python -m $H.run_revision {prepare|run} --dataset sushi_childhood_east_west_v2 \
   --registry $H.comparisons:native_registry \
   --protocol $P/2026-09-12/native_v3.json --output <run> --workers 16
python -m $H.reporting --output <run>

# Gene expression (the selector prepass caches the gene ranking of every training partition)
python -m $H.run_revision prepare --dataset tcga_pancan_rnaseq \
   --registry $H.gene:registry --protocol $P/2026-09-12/gene.json --output <run>
python -m $H.run_gene selector_prepass --output <run> --workers 16
export ARROWFLOW_GENE_MI_CACHE=<run>/mi_cache
python -m $H.run_revision run --dataset tcga_pancan_rnaseq \
   --registry $H.gene:registry --protocol $P/2026-09-12/gene.json --output <run> --workers 16
python -m $H.reporting --output <run>
python -m $H.run_gene gene_corruption --output <run> --workers 16
python -m $H.run_gene summary --output <run>

# Generated ranking patterns, with the encoder (every stage runs through artificial_runs)
python -m $H.artificial_runs extra_runs prepare \
   --protocol $P/2026-09-14/artificial.json --output <run>/run
python -m $H.artificial_runs extra_runs run \
   --protocol $P/2026-09-14/artificial.json --output <run>/run --workers 16
python -m $H.artificial_runs reporting --output <run>/run
python -m $H.artificial_runs extra_ablation {prepare|ablation} \
   --protocol $P/2026-09-14/artificial.json --reference <run>/run \
   --output <run>/ablation --workers 16
python -m $H.artificial_runs extra_ablation summary --output <run>/ablation
python -m $H.artificial_runs compare_extra analyse --family artificial \
   --run <run>/run --ablation <run>/ablation --output <run>/analysis

# Generated ranking patterns, orders without encoder
python -m $H.native_artificial reference --reference <encoded run>/run
python -m $H.native_artificial prepare \
   --protocol $P/2026-09-14/native_artificial.json --output <run>/run --reference <encoded run>/run
python -m $H.native_artificial data_properties \
   --protocol $P/2026-09-14/native_artificial.json --output <run>/data_properties.json
python -m $H.native_artificial run \
   --protocol $P/2026-09-14/native_artificial.json --output <run>/run --workers 16
python -m $H.native_artificial reporting --output <run>/run
python -m $H.compare_native_artificial analyse --run <run>/run \
   --output <run>/analysis --encoded <encoded run>/run

# Missing items at test time (stopped), and the exploratory check that came before its design
python -m experiments.artificial_robustness.study verify --output <verify directory>
python -m experiments.artificial_robustness.study run --output <run>
python -m experiments.artificial_robustness.study original --output <part A directory>
python -m experiments.artificial_robustness.study analyse --output <run>
python -m experiments.artificial_robustness.exploratory_representation_check
```

`artificial_runs` applies the resource settings of the generated-pattern family to `extra_runs`, `extra_ablation`,
`compare_extra` and `reporting`, and it refuses any other family. Do not pass a command of the duplicate-free rerun
through it.

## 8. Stages, logs and worker pools of the production runs

**Stages.** Every harness study that scores outer folds ran four stages, except the permuted-label control. That control
has no `prepare`, `smoke` or `pilot` stage. It reads the prepared data, splits and sealed selections of the motion
controls in place and checks them against pinned digests. Before its design was committed, its `check` refitted two jobs
and reproduced the registered predictions of ArrowFlow and of frozen filters. The `prepare` stage writes out the
datasets and splits and records their hashes. The `smoke` stage is a test run of the code on synthetic data and is never
evidence. The `pilot` stage times a training-only workload and projects from it the wall-clock time of the production
run. A run whose pilot projection exceeded its cap was not started. The benchmark protocol records the resource decision
that set the candidate grid to 16 candidates after its pilot. The `run` stage executes the sealed schedule of planned
jobs. The laboratory ran three stages for each of its two families: a pilot, the readout or permutation LVQ units, and a
summary. A laboratory command with more than one worker needs the pilot of its family.

**Freezing.** The `run` stage refuses a protocol that is not frozen, or one that differs from the copy stored at
`prepare`. Each run directory keeps that copy, and the production copy of every frozen protocol is byte-identical to the
file in `$P`. The permuted-label control refuses a protocol file whose `sha256` differs from the value pinned in its
module, or whose content differs from the design that the module states. Its run directory keeps the file as
`design.json`.

**Worker pools.** Every harness run used at most 16 single-thread worker processes under an execution lock, as each
protocol file records, except the permuted-label control, which ran 24 under the same lock. Its run was interrupted
after 253 of its 255 jobs when the machine failed, and a second start from the same commit ran the last two with 2
workers. `run.log` records both starts. The three studies of the trained encoder network ran outside the harness with
single-thread workers, 14 at once in the nested study. The encoder swap added a second pool of 10 workers, 24 in all.
During the swap, 5 jobs were computed twice, once by each pool, and their records were identical. The reference study
ran 24 workers in one pool, and all 765 of its jobs finished.

**Logs.** Dates are UTC. Each production script ran from a separate working copy of the code, checked out at the
revision that its run directory records. The logs named below are part of the run records.

- *ArrowFlow (nearest-neighbour readout).* A smoke stage and two training-only pilots came before the freeze of the
  protocol, and the resource decision of the protocol file records both pilots. The production script then ran
  `prepare`, `run` and reporting. `compare_runs knn` paired ArrowFlow's outer-fold results with the benchmark's and
  recorded the hashes of its analysis code.
- *Training controls and the component ablation of ArrowFlow* (log `task20-run.log`), 2026-09-13. In this order the
  script ran `prepare`, the pairing check, `run`, reporting and `compare_runs training`, and then `prepare`, `ablation`
  and `summary` of the component ablation. It ended with exit status 0. The pairing check runs before any job and makes
  sure that the new run uses the same datasets and splits as the ArrowFlow run. Both studies ran a smoke stage and one
  training-only pilot before their protocols were frozen.
- *Sorting control* (log `task23a-run.log`), 2026-09-13. In this order the script ran `prepare`, the pairing check,
  `run`, reporting and `compare_runs projected`, and it ended with exit status 0.
- *Further datasets* (log `task23b-run.log`), 2026-09-13 to 2026-09-14. In this order the script ran `prepare` of both
  batches, the pairing check, batch 1 with its reporting, batch 2 with its reporting and `compare_newdata analyse`, and
  it ended with exit status 0. The analysis refuses to run unless both batches are complete.
- *Component ablation of the further datasets* (log `task24-run.log`), 2026-09-14. In this order the script ran
  `prepare`, `ablation`, `summary` and `holistic components`, and it ended with exit status 0. The summary refuses
  unless the seven-view refit reproduces ArrowFlow on every fold and seed. The script checked this for all ten datasets
  before it wrote the component table.
- *Controlled experiments* (motion controls, depth, aggregation, duplicate-free rerun, neighbour baselines, the signed
  relay, the representation test). Each ran a synthetic smoke stage and a training-only pilot against a wall-clock cap
  before its protocol was frozen. Its protocol fixed its arms, Holm families and analysis before any of its outer folds
  was scored.
- *What each controlled experiment reuses.* The motion controls, the two depth families and the aggregation family reuse
  the configuration that each outer fold already selected, so they fit new arms on unchanged folds. The depth family
  with the signed relay also reuses two arms of the first depth family. The neighbour baselines fit only the baselines
  and reuse ArrowFlow's stored predictions. The representation test refits ArrowFlow and its untrained and input
  controls at their stored selections, checks each against its stored predictions, and reads their rankings with a new
  readout. The duplicate-free rerun repeats the registered design on the distinct feature rows of its two datasets. The
  training diagnostics refit the registered ArrowFlow networks under a read-only wrapper and select nothing. The
  mechanism analysis is post hoc and works on predictions that already existed. The permuted-label control reuses the
  selections, folds and seeds of the motion controls and their registered predictions of ArrowFlow and of frozen
  filters. It refits ArrowFlow in every job and checks the refit against those predictions. At the first fitting seed it
  also checks the checkpoint updates against the training diagnostics, before it fits the new arm.
- *Permuted-label control*, 2026-11-24 to 2026-11-25. The run started from the committed revision that its
  `code_revision.txt` records, with 24 workers, and was resumed once (above). Its `analyse` refuses to read any score
  until all 255 records are present and validated. It recomputes every accuracy from the per-row test predictions,
  checks ArrowFlow minus frozen filters against the registered analysis of the motion controls, and writes
  `analysis.json`, `ANALYSIS.md` and nine tables into `<run>/tables`. The paper prints four of them, unchanged except
  for captions, header terms and row labels, and the other five are the extended tables of the supplementary material.
- *Encoder studies.* Each fixed its design in the header of its module and committed it before its run. Each run
  directory records that commit in `code_revision.txt` and the design in `design.json`, but holds no `environment.json`.
  Their result records carry the split, the configuration, the fitting seeds and the predictions, but no dataset hash or
  code revision. They read the prepared data and splits of the registered runs that they are paired with. Before its
  run, the encoder swap refitted the fixed arm on 4 datasets at all three fitting seeds and reproduced the stored
  predictions exactly. Before its run, the `check` of the reference study refitted the untrained and the trained arm of
  the nested study at their selections, with the standard initialisation and the backpropagation stage switched off, on
  one test set each of 4 datasets at all three fitting seeds. It reproduced every test prediction of both readouts. Its
  `analyse` recomputes every number from the per-row test predictions against the registered labels, checks the nested
  study's own analysis, and writes `analysis.json`, `ANALYSIS.md` and the five tables of the paper's appendix into
  `<run>/tables`. The paper prints those tables unchanged except for captions, header terms and row labels.

Every result record of a harness study carries the dataset hash, the split, the model and fitting seeds, the
configuration and the code revision. The summaries are recomputed from the per-example predictions after every planned
job has been reconciled with its result.

## 9. Order of decisions, in full

Dates are UTC, from the freeze fields of the protocol files, the dates of the committed designs, the execution record of
the runs of 2026-09-11 and the pilot authorisations in the laboratory's manifests. Items of the same date are listed in
the order in which they happened.

1. **2026-09-11.** Two protocols of the same harness used the outer and inner folds (split seed 27183) of every later
   study: `2026-09-11/e02.json`, a nested benchmark of a single-view configuration, and `2026-09-11/matched.json`, a
   matched study of single-view networks with learning rate 0.5. Both were frozen before their runs, and their execution
   record shows that both runs completed that day. Their results were known when the study plan was committed on
   2026-09-12. Four choices of the paper were made with those results in hand. They are the candidate grid (built around
   the adaptive defaults), the matched study's configuration, the family of 14 contrasts, and the laboratory's kinds of
   variants with its adoption rule. Their configurations differ from those of ArrowFlow and its prototype readout. The
   paper's appendix reports their results with the other studies that the main text leaves out, among them the two
   further protocols of that day, `2026-09-11/native.json` and `2026-09-11/secondary.json`.
2. **2026-09-12.** The benchmark protocol (prototype readout) was frozen, and the benchmark ran.
3. **2026-09-12.** The protocols of the matched study and of the prototype-readout ablation were frozen, and both ran.
   The ablation reproduces the benchmark's existing outer predictions.
4. **2026-09-12.** The laboratory protocol was written after the benchmark's outer results were known, and its file was
   committed. Before the laboratory ran, the protocol gained the prior-η median variant and the joint selection of the
   prototype count and the readout's neighbour count of the LVQ variants. Its kinds of variants and its adoption rule
   date from the study plan. The laboratory ran after the outer-fold results of the benchmark, the matched study and the
   component ablation were known. Its manifests record the authorisation of its pilots. Across the five outer folds of
   the first repeat, its inner validation folds contain every row of every dataset.
5. **2026-09-12.** The protocol of ArrowFlow (`2026-09-12/bridge_knn.json`) was frozen. Its resource decision records
   the laboratory's adoption verdict. Its run reuses the benchmark's outer partitions and pairs ArrowFlow with the
   benchmark's existing outer-fold results of the prototype readout. So the partitions on which ArrowFlow's numbers are
   reported had informed the choice of its readout, and the family of seven contrasts with the prototype readout was
   fixed after those results were known.
6. **2026-09-13.** The protocols of the training controls and of the component ablation of ArrowFlow were frozen, and
   both ran. Their resource decision records that the pairing check against the ArrowFlow run passed before the freeze.
   All results above were known.
7. **2026-09-13.** The protocol of the sorting control was frozen, and its run executed.
8. **2026-09-13.** The two batch protocols of the ten further datasets were frozen, after a fit check and a pilot that
   predicted training rows only, and their runs executed. The ten datasets were selected by prespecified rules, every
   eligible dataset included, before any model was fitted on them, and the batch protocols also fix the analysis. A
   search of the development repository (working tree, history and binary documents) found no use of the ten datasets
   before their selection. The table renderer repeats this search on every render.
9. **2026-09-14.** The combined analysis of the 17 datasets, with its Friedman test and its Holm adjustment over all 34
   training contrasts, was written after every result above was known. It recomputes every value from the verified runs
   and fits no model. It is post hoc and descriptive, and so are its degenerate flags and pooled depth split. The
   protocol of the component ablation of the further datasets was frozen after the combined analysis was written, and it
   ran.
10. **2026-09-14.** The protocols of the duplicate-free rerun, the training diagnostics, the neighbour baselines, the
    motion controls, and the depth and aggregation studies were frozen, in this order and each before its run. The
    results of ArrowFlow and of the frozen arm of the motion controls existed before that protocol was frozen, so only
    its two scrambled arms were scored blind.
11. **2026-09-23.** The protocols of the representation test and then of the depth study with the signed relay were
    frozen, after every result above was known. The vote scale of the fourth arm of the depth study with the signed
    relay came from training-only pilot fits. The seven benchmark datasets had been screened for the representation
    question before its protocol was written.
12. **2026-09-24 and 2026-09-25.** The designs of the nested study and of the encoder swap were committed, after every
    result above was known. The swap's design, with its reading rule, was fixed after the outer results of the nested
    study were known.
13. **2026-11-24.** The design of the reference study (`nested_reference.py`), with its contrasts, recovery fractions,
    rule for not reporting a fraction and reading rule, was committed in the header of its module, after every result
    above was known, including the outer results of the nested study. No score of its three new arms existed before its
    run, which started after the commit and ended the same day (UTC).
14. **2026-11-24.** After the run of the reference study had ended, the design of the permuted-label control was
    committed in the header of its module (`$H/label_controls.py`, with the analysis `$H/compare_labels.py`), together
    with its protocol file `$P/2026-11-25/permuted_labels.json`, whose `sha256` the module pins. The design fixes the
    arm, which labels each step reads, the three readouts, the four families, the summaries across datasets, the reading
    rule with its wordings and the descriptive measures. It was fixed after the outer results of ArrowFlow, of frozen
    filters and of the two scrambled arms of the motion controls were known. No score of the new arm existed before its
    run, which started after the commit, was interrupted after 253 of its 255 jobs and ended on 2026-11-25 (UTC) after a
    second start from the same commit.

## 10. Studies that the main text leaves out

`$P` also holds the frozen protocols of studies that the paper's main text leaves out, listed below with the dataset
pins `2026-09-14/artificial_pins.json` of the two studies of generated patterns. The eighth dataset of the matched
study, the SUSHI rankings, is also left out of the main text. The stopped study of missing items has its protocol and
preregistration in `experiments/artificial_robustness/`. The protocol records the digest of the preregistration, which
matches the file. The paper's appendix section "Studies not reported in the main text" reports the results of all of
them, with the data, the question, the models, the state of each run and the reason it is not in the main text.

| File | First 12 hex digits of `sha256` |
|---|---|
| `2026-09-11/e02.json` | `70f60b52da70` |
| `2026-09-11/matched.json` | `bf69f4b1e19b` |
| `2026-09-11/native.json` | `0dc439ccecea` |
| `2026-09-11/secondary.json` | `4cdd220b9df8` |
| `2026-09-12/native_v3.json` | `9b6b79b1efce` |
| `2026-09-12/gene.json` | `14b6bbe5a7ac` |
| `2026-09-14/artificial.json` | `fd1e6d8dcd7e` |
| `2026-09-14/native_artificial.json` | `2bb104664719` |
| `2026-09-14/artificial_pins.json` | `72a3c65f0105` |
| `experiments/artificial_robustness/protocol.json` | `b501eaffd820` |
| `experiments/artificial_robustness/PREREGISTRATION.md` | `855dc4909ccc` |

The stopped study's module and protocol pin the digests of the library and harness files that it imports. Every one of
these files here matches its pin, so its source check passes. Its run was stopped before every job had finished, and its
`analyse` refuses while a job is missing.
