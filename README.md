# ArrowFlow

ArrowFlow is a classifier whose layers compute with rankings. A ranking layer holds a bank of learned permutations, the
ranking filters, and orders them by their footrule distance to the input ranking. That order is the input of the next
layer. Training the ranking layers computes no derivative. Position votes reorder the filters by a weighted Borda count,
and the output layer's votes are passed down as motions, the position displacements that train the hidden layers. An
encoder turns numeric features into the input ranking. The fixed encoder sorts projected features. The trained encoder
network is fitted to target scores that the motions of the ranking layer above it define, so the sort is never
differentiated. A nearest-neighbour readout on the last hidden ranking predicts the class, and several views, each an
encoder with its own network, vote by majority.

This repository holds the code, the frozen evaluation protocols and the tests behind the paper. The paper is not
published yet.

## Contents

- `arrowflow/`: the library. `arrowflow.py` holds the ranking layers and the training rule, and `ranking.py` the ranking
  conventions. `readouts.py` and `permlvq.py` hold the readouts that the inner-fold laboratory compares.
- `experiments/make_revision/`: the evaluation harness. It holds the data loaders, the nested cross-validation, one
  runner per study and the analyses. `experiments/make_revision/protocols/` holds the frozen protocol of every study in
  five dated folders.
- `experiments/tensor_rank_sandbox/`: the trained encoder network and its three studies, with the development searches
  and their records (`runs/`, `RESULTS.md`).
- `experiments/artificial_robustness/`: the preregistration, protocol and code of a study of missing items. The study
  was stopped, and the paper's appendix reports it.
- `experiments/exp_knn_vs_arrowflow.py`: an older comparison script. Two test modules import it.
- `tests/`: the test suite of the library and the harness.
- `docs/`: `REPRODUCING.md` and the specification of the matched learning controls that their runner hashes
  (`secondary_experiment_spec.md`).
- `requirements-revision.txt` pins the package versions that the runs and the tests used. `pyproject.toml` declares the
  package and its version ranges.

## Installation and tests

The harness needs Linux or macOS, because it uses the POSIX modules `resource` and `fcntl`. It needs Python 3.12, the
version the runs used. It also needs a git clone, because every run records the git revision of its code.

```bash
git clone https://github.com/yilmazozgur/arrowflow.git
cd arrowflow
python -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-revision.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
```

Run every command from the repository root, so that `arrowflow` and `experiments` import from the clone. Some tests fit
small models on synthetic data. The suite takes about 20 minutes on one core. On the author's machine,
with the downloaded datasets in place and without the run records, 1005 tests pass and 29
skip. More tests skip on a fresh clone. A test skips when an input it needs is absent: the run records, an OpenML
download that is not in the local scikit-learn cache, the gene-expression archive, the SUSHI archive, a CUDA device, or
a file or commit of the development history.

## Rerunning a study

Each study has a frozen protocol under `experiments/make_revision/protocols/<date>/`. It fixes the datasets, folds,
seeds, candidates, contrast families and analysis. The runners refuse a protocol that is not frozen or that differs from
the copy stored when the run was prepared. This example reruns the first benchmark run (prototype readout) on the seven
benchmark datasets:

```bash
P=experiments/make_revision/protocols
python -m experiments.make_revision.run_revision prepare \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry experiments.make_revision.bridge:bridge_registry \
   --protocol $P/2026-09-12/bridge.json --output runs/2026-09-12-bridge
python -m experiments.make_revision.run_revision run \
   --dataset iris wine breast_cancer wine_quality vehicle segment digits \
   --registry experiments.make_revision.bridge:bridge_registry \
   --protocol $P/2026-09-12/bridge.json --output runs/2026-09-12-bridge --workers 16
python -m experiments.make_revision.reporting --output runs/2026-09-12-bridge
```

`prepare` loads the datasets from scikit-learn and OpenML, checks their identity and writes the splits. `run` executes
every planned job on single-thread worker processes, at most 16. `reporting` recomputes the summaries from the
per-example predictions. Use a new output directory for each run, because the runners refuse to replace a result file.
Each protocol records its wall-clock cap, from 1 to 10 hours at up to 16 workers.

These studies, or the parts named, rerun from the repository alone, with the datasets that they download or that their
provider supplies:

- The first benchmark run (prototype readout), `2026-09-12/bridge.json`.
- The matched learning controls, `2026-09-12/matched_v3.json`. They also need the SUSHI archive from its provider.
- The further datasets, batches 1 and 2, `2026-09-12/newdata_batch1.json` and `2026-09-12/newdata_batch2.json`.
- The component ablation with its contrasts, `2026-09-12/ablation.json`, once runs of the first benchmark and the
  matched controls exist.
- The inner-fold laboratory, `2026-09-12/devlab.json`, once a run of the first benchmark exists.
- The run stages of ArrowFlow (nearest-neighbour readout), `2026-09-12/bridge_knn.json`, and the comparator intervals
  and ranks computed from that run. The comparison with the first benchmark reads the registered benchmark run.
- The run stages of the duplicate-free rerun, `2026-09-14/dedup.json`. Its analysis reads registered runs.

Every other study reads registered runs of the paper. Most protocols pin those runs by code revision and file digests,
and a few modules read them from a fixed folder. [docs/REPRODUCING.md](docs/REPRODUCING.md) gives the commands, the
inputs and the order of every study, including the studies that the main text leaves out.

## Run records

The run records of the paper's studies are not in this repository. They hold the prepared data and splits, the
per-example predictions, the selection records, the summaries, the analyses and the production logs, about 13 GB in all.
They will be released with the published paper. Commands that read earlier runs take them as arguments, such as
`--runs`, `--reference` and the `--*-source` options. A few modules read them from a fixed folder next to the clone, and
[docs/REPRODUCING.md](docs/REPRODUCING.md) lists these modules.

## Citation

If you use this code, please cite the preprint until the paper is published:

```bibtex
@misc{yilmaz_arrowflow,
  title         = {ArrowFlow: Hierarchical Machine Learning in the Space of Permutations},
  author        = {Yilmaz, Ozgur},
  year          = {2026},
  eprint        = {2604.04087},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2604.04087}
}
```

## Earlier version

An earlier version of this work is available as a preprint: [arXiv:2604.04087](https://arxiv.org/abs/2604.04087). Its
evaluation differs, and the paper does not reuse its numbers. Its scripts, results and preprint files were removed from
the main branch and remain at tag `v2.1.0`. They are the scripts `experiments/exp_*.py` other than
`experiments/exp_knn_vs_arrowflow.py`, the script `experiments/plot_results.py`, the results in `experiments/results/`
and the preprint files in `manuscript/`.

## License

The code is released under the MIT License (see `LICENSE`). The datasets are subject to their own terms.
