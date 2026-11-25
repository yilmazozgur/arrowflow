"""G4 follow-up experiment (c): the artificial family's resource settings, applied to extra_runs without editing it.

extra_runs.py is sealed by the dedup production run (its sha256 is in that run's environment and in its ablation's), so the
controller's decision of 2026-09-14 for the artificial family lives here: 16 single-thread workers and a 6 h wallclock cap
(the 8-worker projection, 5.40 h, exceeded the earlier 4 h cap). Importing this module sets extra_runs.WORKERS['artificial']
and extra_runs.CAP_HOURS['artificial'], and nothing of the dedup family, and adds this module to the sealed source lists of
extra_runs and extra_ablation, so that the artificial run's and ablation's environments seal this file next to theirs.

Every stage of the artificial family runs through this module, so that the draft it derives and validates at every stage
carries these values:
python -m experiments.make_revision.artificial_runs extra_runs draft|prepare|smoke|pilot|project|freeze|run ARGS
python -m experiments.make_revision.artificial_runs reporting --output RUN
python -m experiments.make_revision.artificial_runs extra_ablation prepare|ablation|summary ARGS
python -m experiments.make_revision.artificial_runs compare_extra analyse --family artificial ARGS
    the named module's command under these settings; refused for anything naming another family.

The run stage re-derives the draft from the settings in every spawned worker (run_revision._worker, extra_registry,
validate_extra_protocol); the workers of a pool started by `python -m experiments.make_revision.artificial_runs ...` import
this module as __mp_main__ before any job, so they hold the same settings. Never import this module in a dedup command: a
dedup environment computed in the same process would also list this file and differ from the one the dedup run sealed.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'  # as run_revision: spawned workers import this -m module before any numeric library
import importlib
import json
from pathlib import Path
import sys
from . import extra_ablation
from . import extra_runs

FAMILY = 'artificial'
WORKERS = 16
CAP_HOURS = 6
MODULE = 'experiments.make_revision.artificial_runs'
COMMANDS = ('extra_runs', 'extra_ablation', 'compare_extra', 'reporting')

extra_runs.WORKERS[FAMILY] = WORKERS
extra_runs.CAP_HOURS[FAMILY] = CAP_HOURS
for _sealing in (extra_runs, extra_ablation):
    if MODULE not in _sealing.SOURCE_MODULES:
        _sealing.SOURCE_MODULES = [*_sealing.SOURCE_MODULES, MODULE]


def _family(path):
    try:
        return json.loads(Path(path).read_text()).get('production_family')
    except (OSError, ValueError, AttributeError):
        return None


def refusals(args):
    """What in the arguments names another family: a --family value, a --protocol or --draft file, or the protocol.json of an
    --output, --run or --reference directory."""
    found = []
    for flag, value in zip(args, args[1:]):
        if flag == '--family' and value != FAMILY:
            found.append(f'--family {value}')
        elif flag in ('--protocol', '--draft') and Path(value).is_file() and _family(value) != FAMILY:
            found.append(f'{flag} {value} (family {_family(value)!r})')
        elif flag in ('--output', '--run', '--reference') and (Path(value)/'protocol.json').is_file() \
                and _family(Path(value)/'protocol.json') != FAMILY:
            found.append(f'{flag} {value} (family {_family(Path(value)/"protocol.json")!r})')
    return found


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        raise SystemExit(f'usage: python -m {MODULE} {{{",".join(COMMANDS)}}} ARGS (the artificial family only)')
    found = refusals(argv[1:])
    if found:
        raise SystemExit(f'artificial_runs refused: it applies the artificial family settings only ({"; ".join(found)})')
    importlib.import_module(f'experiments.make_revision.{argv[0]}').main(argv[1:])


if __name__ == '__main__':
    main()
