# ArrowFlow: Training Ranking Filters with Position Votes

ArrowFlow is a classifier whose layers compute with rankings. A ranking layer holds a bank of learned permutations, the ranking filters, and orders them by their footrule distance to the input ranking; that order is the input of the next layer. Training computes no derivative. Position votes reorder the filters by a weighted Borda count, and the output layer's votes are passed down as position displacements that train the hidden layers. Encoders turn numeric features into rankings by sorting projected features. A nearest-neighbour vote on the last hidden ranking predicts the class, and several views, each an encoder with its own network, vote by majority.

This repository holds the code, the frozen evaluation protocols and the tests behind the paper. It also keeps the scripts and results of an earlier version of this work (see [Earlier version](#earlier-version)).

## Contents

- `arrowflow/`: the core package. `arrowflow.py` holds the ranking layers and the training rule, and `ranking.py` the ranking conventions. `readouts.py` and `permlvq.py` hold the readouts compared in the paper's inner-fold laboratory. `arrowflow.py` is the version the paper's runs used: when a hidden layer passes displacements to the hidden layer below, it drops the sign of repelling votes. This matters only in networks with two or more hidden layers. `experiments/make_revision/signed_relay.py` holds the corrected relay, which the paper's depth rerun used (Section 3.5 of the paper).
- `experiments/make_revision/`: the evaluation harness. It holds the nested cross-validation, one module per study and, under `protocols/`, the protocol of every study. The model the paper evaluates is `bridge.AdaptiveMultiViewKNN` (registry `bridge_knn_registry`, model `arrowflow_full_knn`). It is built from `multiview.MultiViewArrowFlowKNN` and the single-view network `models.ArrowFlowEstimator`.
- `tests/make_revision/`: the tests of the harness and the core.
- `manuscript/MAKE/review/secondary_experiment_spec.md`: the written specification of the matched learning controls. Their runner, `run_studies`, reads it and records its hash in every run, so it must stay at this path.
- `experiments/exp_*.py`, `experiments/plot_results.py`, `experiments/results/` and the preprint files in `manuscript/`: the earlier version. The paper does not use them.

## Installation

The harness needs Linux or macOS, because it uses the POSIX modules `resource` and `fcntl`. It also needs Python 3.12, the version the runs used, and a git clone, because every run records the git revision of the code.

```bash
git clone https://github.com/yilmazozgur/arrowflow.git
cd arrowflow
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-revision.txt
python -m pip install --no-deps -e .
python -m pytest tests/make_revision -q
```

`requirements-revision.txt` pins the versions the runs used: NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, pandas 2.2.3, PyTorch 2.6.0, joblib 1.4.2, threadpoolctl 3.5.0, Matplotlib 3.9.2 and pytest. `pyproject.toml` declares the package and its version ranges. Some tests fit small models on synthetic data, and the full suite takes about 11 minutes. Up to 22 tests skip in a public checkout. 19 of them re-verify the paper's run outputs, which are not included, one needs a CUDA GPU, and two need files that are not released.

## How the evaluation is organised

Each study has a JSON protocol under `experiments/make_revision/protocols/<date>/`. The protocol fixes the datasets, folds, seeds, candidate grids, budgets, metrics, contrast families and wall-clock cap. Every protocol of a study that scores outer test folds was frozen, and its hash recorded, before that study's run started. The run stages refuse a protocol that is not frozen, or one that differs from the protocol the output directory was prepared with.

Run every command from the repository root as `python -m experiments.make_revision.<module> <stage> ...`. The studies share these stages:

- `prepare`: loads the datasets from scikit-learn or OpenML and checks their identity: shape and class counts, and for the ten further datasets also content hashes. It then writes the splits and candidates and records the dataset hashes, the environment and the hashes of the sources.
- `smoke`: exercises the code on synthetic data. It is never evidence.
- `pilot`: times a training-only workload and projects the production wall-clock time.
- `run` (or `ablation`): executes every planned job, with at most 16 single-thread workers.
- `summary`, `reporting` and `analyse`: recompute the summaries and contrasts from the saved per-example predictions. They refuse until every planned job is present.

| Study | Protocol (`protocols/...`) | Modules and stages |
|---|---|---|
| First benchmark run (prototype readout) | `2026-09-12/bridge.json` | `run_revision` smoke, pilot, prepare, run with `--registry experiments.make_revision.bridge:bridge_registry`; `reporting` |
| ArrowFlow (nearest-neighbour readout) | `2026-09-12/bridge_knn.json` | `run_revision` prepare, run with `--registry experiments.make_revision.bridge:bridge_knn_registry`; `reporting`; `compare_runs knn` |
| Matched learning controls | `2026-09-12/matched_v3.json` | `run_studies` smoke, pilot, prepare, run, summary |
| Component ablation and its 14 contrasts | `2026-09-12/ablation.json` | `run_bridge` prepare, smoke, pilot, ablation, summary, contrasts |
| Inner-fold laboratory (readouts, permutation LVQ) | `2026-09-12/devlab.json` (development protocol, scores no outer fold) | `devlab` pilot, readouts or permlvq, summary |
| Training controls | `2026-09-12/knn_training.json` | `run_revision` prepare, run with `--registry experiments.make_revision.knn_controls:knn_training_registry`; `compare_runs training-pairing`, `training` |
| Component ablation of ArrowFlow | `2026-09-12/knn_ablation.json` | `run_knn_ablation` prepare, ablation, summary |
| Comparator intervals, ranks and duplicate rows | none (fits no model) | `referee_analyses` comparators, duplicates, ranks |
| Sorting control | `2026-09-12/knn_projected.json` | `run_revision` prepare, run with `--registry experiments.make_revision.projected_knn:knn_projected_registry`; `compare_runs projected-pairing`, `projected` |
| Ten further datasets, two batches | `2026-09-12/newdata_batch1.json`, `newdata_batch2.json` | `newdata` prepare, run; `reporting`; `compare_newdata` pairing, analyse |
| Combined analysis of the seventeen datasets | none (fits no model) | `holistic` benchmark, training, ready, components |
| Component ablation on the further datasets | `2026-09-12/newdata_ablation.json` | `run_newdata_ablation` prepare, ablation, summary |
| Matched motion-signal controls | `2026-09-14/motion_controls.json` | `motion_controls` prepare, run, summary; `compare_motion analyse` |
| Training diagnostics and their tables | `2026-09-14/training_diagnostics.json` | `training_diagnostics` run, summary; `diagnostics_tables tables` |
| Mechanism analysis | none (fits no model) | `mechanism_analysis analyse` |
| Fixed-configuration depth; aggregation inside the update | `2026-09-14/depth.json`, `aggregation.json` | `interventions` prepare, run, summary, analyse |
| Duplicate-free rerun | `2026-09-14/dedup.json` | `extra_runs` prepare, run; `reporting`; `extra_ablation` prepare, ablation, summary; `compare_extra analyse --family dedup` |
| Neighbour baselines | `2026-09-14/neighbour_baselines.json` | `neighbour_baselines` prepare, run; `reporting`; `compare_baselines` pairing, analyse |
| Depth rerun with the signed relay | `2026-09-23/signed_relay_depth.json` | `signed_relay_depth` prepare, run, summary, analyse |
| Representation test | `2026-09-23/representation_test.json` | `representation_test` prepare, run, summary; `compare_representation analyse` |

Every module in the table prints its options with `--help`. The supplement of the paper, Section S5, lists the command sequence of each study.

## Reproducing a study

This example repeats ArrowFlow's benchmark run on the seven benchmark datasets:

```bash
P=experiments/make_revision/protocols
python -m experiments.make_revision.run_revision prepare \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry experiments.make_revision.bridge:bridge_knn_registry \
   --protocol $P/2026-09-12/bridge_knn.json --output runs/2026-09-12-bridge-knn
python -m experiments.make_revision.run_revision run \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry experiments.make_revision.bridge:bridge_knn_registry \
   --protocol $P/2026-09-12/bridge_knn.json --output runs/2026-09-12-bridge-knn --workers 16
python -m experiments.make_revision.reporting --output runs/2026-09-12-bridge-knn
```

Keep these points in mind:

- Use a new output directory for each run. Existing result files are never overwritten.
- Commands that read earlier runs take them as arguments: `--runs <directory>`, the directory that holds the run directories under the names given in Section S5 of the supplement, or `--reference` and `--*-source` options. Always pass them, because the defaults point to the author's workspace.
- Protocols that build on an earlier run pin that run by its code revision and file digests, and the preparation or pairing stages check these pins. The pinned runs are the ones behind the paper, and they are not in this repository, so these checks refuse any other run. The following protocols pin no earlier run, so their runs can be repeated from the protocol alone: `bridge.json`, `bridge_knn.json`, `matched_v3.json`, `newdata_batch1.json`, `newdata_batch2.json` and `dedup.json`. Two of their analysis steps still read the paper's runs: `compare_runs knn` checks the pin of `bridge_knn.json` on the first benchmark run, and `compare_extra analyse --family dedup` compares with the registered full-data runs. `ablation.json` and `devlab.json` read a run of `bridge.json`, passed as `--bridge-source`.
- The matched learning controls need two inputs besides the code. The first is their specification, `manuscript/MAKE/review/secondary_experiment_spec.md`, which ships with the code: `run_studies` records its hash in each run's `environment.json` and fails without it. The second is the SUSHI archive. The frozen panel of this study also lists the native SUSHI preference data, whose results the paper does not report, and its run stage requires the complete panel. Obtain the archive (`sushi3-2016.zip`) from the provider, <https://www.kamishima.net/sushi/>, and pass it with `--sushi-archive` or `ARROWFLOW_SUSHI_ARCHIVE`. The loader checks the archive's hash. Provider data are not included.
- Every run records the git revision of its code, and the supplement lists the revision of each study (Section S5.1). This repository holds the final revision. For most studies, the source files that their runs sealed are byte-identical to this revision. The first benchmark run, the component ablation, the laboratory and the matched controls ran at earlier revisions, in which up to four harness files differed.
- Each protocol records its wall-clock cap, from 1 to 10 hours at up to 16 workers. The supplement lists them in its table of protocols and code revisions (Section S5.1).

## Run outputs

The run outputs of the paper's studies are not included. This covers predictions, fit logs, summaries, analyses and production logs. Every table and figure of the paper is computed from those outputs. The CSV files in `experiments/results/` are outputs of the earlier version.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{yilmaz_arrowflow_make,
  title   = {ArrowFlow: Training Ranking Filters with Position Votes},
  author  = {Yilmaz, Ozgur},
  journal = {Machine Learning and Knowledge Extraction},
  year    = {TODO},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO}
}
```

## Earlier version

An earlier version of this work, "ArrowFlow: Hierarchical Machine Learning in the Space of Permutations", is available as a preprint: [arXiv:2604.04087](https://arxiv.org/abs/2604.04087). Its evaluation differs, and its numbers are not reused in the paper. The scripts `experiments/exp_*.py` and `experiments/plot_results.py`, the results in `experiments/results/` and the preprint files in `manuscript/` belong to that version and are kept for reference. They were written for the earlier code and are not tested against the current version. The one exception is `experiments/exp_knn_vs_arrowflow.py`: it is updated here, and two tests use its encoder and its footrule nearest-neighbour classifier.

## License

The code is released under the MIT License (see `LICENSE`). The datasets are subject to their own terms.
