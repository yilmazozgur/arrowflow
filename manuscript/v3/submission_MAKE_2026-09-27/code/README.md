# ArrowFlow code supplement for MAKE

This archive is a source snapshot assembled on 27 September 2026 for
*ArrowFlow: Training Ranking Filters with Position Votes* by Ozgur Yilmaz.
`source_manifest.json` identifies every original source file by SHA-256.

It includes the ranking implementation, the experiment harness and frozen JSON
protocols, the later learned-encoder experiments in
`experiments/tensor_rank_sandbox`, the focused tests, and the manuscript table and
figure renderers. The public tag
https://github.com/yilmazozgur/arrowflow/tree/v2.0.0-make predates the learned-encoder
experiments; this archive includes their additional source modules.

## Reproduction

The supplementary PDF, Section S5 and Table S68, records the study-specific code
revisions, commands, environments and protocol identifiers. The designs of the
two learned-encoder studies are also recorded in their module headers.

Use a dedicated Python environment. `pyproject.toml` describes package
dependencies; `requirements-revision.txt` records the development environment.
The actual per-run environment records described in S5 take precedence when
reproducing a particular run. From this directory, `python -m pip install -e .`
installs the package. Run the relevant commands from S5 with the dataset and run
paths set for your machine. Running experiments can be lengthy and is separate
from compiling the manuscript.

Source datasets are obtained from the UCI/OpenML versions identified in Table S69.
Raw datasets, fitted models, per-example predictions, prepared splits and run
directories are not in this source archive. The author provides the recorded run
directories on request, as stated in the manuscript. In particular, the learned
encoder modules read registered baseline runs through `RUNS_ROOT` in `nested.py`;
set that path to the corresponding prepared runs before using them.

The original rendering scripts also expect the run records and Git history used
for their provenance checks. This archive contains no Git database. Section S5
describes their complete external input inventory; obtain those records from the
author for a full rerender. The `manuscript/v3` directory here preserves the
production manuscript inputs for those checks. The separately supplied
`ArrowFlow_sources_MAKE.zip` contains the submission layout and is the archive to
use for editorial recompilation.

This source snapshot has been checked for Python syntax and packaging integrity.
The experimental runs were not repeated during submission preparation, and this
archive does not claim to certify historical results or frozen-commit identity.

License: MIT; see `LICENSE`.
