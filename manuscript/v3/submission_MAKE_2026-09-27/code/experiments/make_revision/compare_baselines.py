"""The prespecified analysis of the neighbour-baselines run; nothing is read until the run is complete.

python -m experiments.make_revision.compare_baselines pairing --run R [--runs D]
    before the jobs start: the prepared directory is the frozen neighbour-baselines protocol, its panel and folds are the
    registered ArrowFlow-kNN runs' panel and folds, and every source both seal is byte-identical; writes nothing
python -m experiments.make_revision.compare_baselines analyse --run R --output O [--runs D]
    refuses (exit 2, nothing written) unless the baselines run and every referenced ArrowFlow-kNN run is complete:
    protocol, environment, candidates, planned jobs and summary.json present and every planned result file and fit log
    present, checked before any record holding a score is read. Then every run is loaded and re-verified with
    compare_runs.load_run and compare_runs.verify_run (the harness validators over every saved record), the protocol's
    reference pins and the pairing are checked, and the protocol's analysis block is computed:
      primary families  per baseline and dataset, arrowflow_full_knn minus the baseline accuracy, fitting seeds averaged
                        within each outer fold, corrected resampled t (q 0.25, 95%, 14 df), Holm across the seventeen
                        datasets within that baseline; the four baselines are adjusted separately
      metrics           per dataset and baseline, the mean outer error, the outer-fold SD and the mean within-fold seed
                        SD, with balanced accuracy and macro-F1 in the same form; descriptive
      ranks             each baseline's rank by mean outer error among the eleven models every dataset holds; descriptive
      warnings          per dataset and baseline, the fits raising a warning and the count of each category, over the
                        outer fits and over every fit of the complete fit logs; descriptive
      interpretation    the rule fixed in the protocol before any outer score existed, applied to the mean outer errors
Outputs (all or none; an existing file with different content is never replaced): baseline_families.csv,
baseline_metrics.csv, baseline_ranks.csv, baseline_warnings.csv and baseline_analysis.json.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .compare_runs import (RUN_RECORDS, SHARED_SOURCES, VALIDATORS, RunComparisonError, _csv_text, _json_text, _reason,
                           analysis_sources, check_output, load_prepared_run, load_run, verify_run, write_outputs)
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .knn_controls import TRAINED_MODEL
from .neighbour_baselines import (FAMILY, MODEL_ORDER, NCA_MODEL, PROTOCOL_FILE, PROTOCOL_ID, REFERENCE_LABELS,
                                  SUPERVISED_MODELS, validate_protocol)
from .newdata import INTERVAL_RULE, sha256_file

BASELINES = 'baselines'
CLASSICAL = ('svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting')
MAJORITY = 'dummy'
REFERENCE_PANEL = (TRAINED_MODEL, *CLASSICAL, MAJORITY)
RANK_PANEL = (*REFERENCE_PANEL, *MODEL_ORDER)
DESIGN_KEYS = ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds', 'selection_metric',
               'test_train_ratio', 'confidence')
REFERENCE_PINS = ('protocol_id', 'protocol_sha256', 'code_revision', 'summary_sha256')
METRICS = ('error', 'balanced_accuracy', 'macro_f1')
TOLERANCE = 1e-12
DESCRIPTIVE = 'descriptive; no multiplicity adjustment'
FAMILIES_CSV, METRICS_CSV, RANKS_CSV = 'baseline_families.csv', 'baseline_metrics.csv', 'baseline_ranks.csv'
WARNINGS_CSV, ANALYSIS_JSON = 'baseline_warnings.csv', 'baseline_analysis.json'
OUTPUTS = (FAMILIES_CSV, METRICS_CSV, RANKS_CSV, WARNINGS_CSV, ANALYSIS_JSON)
ANALYSIS_SOURCES = ('compare_baselines.py', 'neighbour_baselines.py', 'compare_runs.py', 'newdata.py', 'knn_controls.py',
                    'evaluation.py', 'reporting.py', 'run_revision.py')
INTERVAL = ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'n_folds', 'df')
FAMILY_COLUMNS = ('family', 'family_index', 'dataset', 'panel', 'reference', 'contrast', 'model_a', 'model_b',
                  *INTERVAL[:4], 'p_approximate', 'holm_p_approximate', *INTERVAL[4:], 'mean_error_a', 'mean_error_b')
METRIC_COLUMNS = ('status', 'dataset', 'panel', 'model_id', 'source', 'metric', 'mean', 'outer_fold_sd',
                  'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')
RANK_COLUMNS = ('status', 'dataset', 'panel', 'model_id', 'source', 'mean_error', 'rank', 'models_ranked')
WARNING_COLUMNS = ('status', 'dataset', 'model_id', 'scope', 'fits', 'fits_with_warning', 'warnings', 'categories',
                   'reached_max_iter_fits')


# ----------------------------------------------------------------------------- completeness, references and pairing

def completeness_gate(sources):
    """Refuse unless every run directory holds its run records and every planned result file and fit log. Only
    planned_jobs.json is parsed, so nothing holding a score is read before every run is complete."""
    problems = []
    for label, source in sources.items():
        path = Path(source)
        if not path.is_dir():
            problems.append(f'{label} run {path}: no such run directory')
            continue
        missing = [name for name in RUN_RECORDS if not (path/name).is_file()]
        absent, first = 0, None
        if (path/'planned_jobs.json').is_file():
            try:
                for job in json.loads((path/'planned_jobs.json').read_text()):
                    for key in ('result_file', 'log_file'):
                        if not (path/job[key]).is_file():
                            absent, first = absent + 1, first or job[key]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f'{label} run {path}: unreadable planned_jobs.json ({_reason(exc)})')
                continue
        if missing or absent:
            parts = [f'missing {", ".join(missing)}'] if missing else []
            parts += [f'{absent} planned result files or fit logs missing (first: {first})'] if absent else []
            problems.append(f'{label} run {path} is not complete: ' + '; '.join(parts))
    if problems:
        raise RunComparisonError('The analysis runs only after the baselines run and every referenced ArrowFlow-kNN run '
                                 'is complete, and reads none of them before: ' + ' | '.join(problems))


def read_protocol(source, label=BASELINES):
    """The saved protocol of a run directory; no record holding a score is read."""
    path = Path(source)/'protocol.json'
    if not path.is_file():
        raise RunComparisonError(f'{label} run {Path(source)} holds no protocol.json')
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RunComparisonError(f'{label} run {path}: unreadable JSON ({_reason(exc)})') from exc


def reference_paths(protocol, runs_root=None, references=None):
    """{label: run directory} of every reference the protocol declares."""
    block = protocol.get('reference')
    if not isinstance(block, dict) or not block:
        raise RunComparisonError('The neighbour-baselines protocol declares no reference block')
    if references is not None:
        missing = sorted(set(block) - set(references))
        if missing:
            raise RunComparisonError(f'No run directory was supplied for the reference(s) {", ".join(missing)}')
        return {label: Path(references[label]) for label in block}
    from .neighbour_baselines import WORKSPACE_RUNS
    root = Path(runs_root or WORKSPACE_RUNS)
    return {label: root/entry['directory'] for label, entry in block.items()}


def check_frozen_protocol(run, *, smoke):
    """The run holds the committed frozen neighbour-baselines protocol (or a synthetic smoke protocol)."""
    try:
        validate_protocol(run.protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunComparisonError(f'The {run.label} protocol is not a neighbour-baselines protocol: {_reason(exc)}') from exc
    if (run.protocol.get('purpose') == 'synthetic_smoke_only') != smoke:
        raise RunComparisonError(f'The {run.label} run holds a {"production" if smoke else "synthetic smoke"} protocol; '
                                 f'this analysis was asked for a {"synthetic smoke" if smoke else "production"} run')
    if sorted(run.registry) != sorted(MODEL_ORDER):
        raise RunComparisonError(f'The {run.label} run must hold exactly {list(MODEL_ORDER)}, not {sorted(run.registry)}')
    if smoke:
        return {'protocol_id': run.protocol['protocol_id'], 'protocol_sha256': run.protocol_sha256, 'frozen_file': None}
    if not PROTOCOL_FILE.is_file():
        raise RunComparisonError(f'The frozen protocol {PROTOCOL_FILE} is not in this tree')
    expected = sha256_file(PROTOCOL_FILE)
    if run.protocol.get('protocol_id') != PROTOCOL_ID or run.protocol_sha256 != expected:
        raise RunComparisonError(f'The {run.label} run protocol.json (id {run.protocol.get("protocol_id")!r}, sha256 '
                                 f'{run.protocol_sha256}) is not the committed frozen protocol {PROTOCOL_ID} '
                                 f'(sha256 {expected})')
    return {'protocol_id': PROTOCOL_ID, 'protocol_sha256': expected, 'frozen_file': str(PROTOCOL_FILE)}


def check_reference(baselines, references, *, smoke):
    """Every reference run is the run the protocol pins, holds arrowflow_full_knn and supplies the declared datasets."""
    block = baselines.protocol['reference']
    if sorted(block) != sorted(references):
        raise RunComparisonError(f'The protocol declares references {sorted(block)}; {sorted(references)} were supplied')
    declared = [name for entry in block.values() for name in entry['datasets']]
    if sorted(declared) != sorted(baselines.protocol['datasets']) or len(set(declared)) != len(declared):
        raise RunComparisonError('The reference blocks must partition the protocol datasets')
    record = {}
    for label, run in references.items():
        entry = block[label]
        if entry.get('model_id') != TRAINED_MODEL or TRAINED_MODEL not in run.registry:
            raise RunComparisonError(f'The {label} reference must pin and hold {TRAINED_MODEL}')
        if sorted(entry['datasets']) != sorted(name for name in entry['datasets'] if name in run.protocol['datasets']):
            raise RunComparisonError(f'The {label} run does not hold every dataset the protocol assigns to it: '
                                     f'{sorted(set(entry["datasets"]) - set(run.protocol["datasets"]))}')
        observed = {'protocol_id': run.protocol.get('protocol_id'), 'protocol_sha256': run.protocol_sha256,
                    'code_revision': run.environment['code_revision'], 'summary_sha256': run.summary_sha256}
        if not smoke:
            wrong = [f'{key}: declared {entry.get(key)!r}, the {label} run has {value!r}'
                     for key, value in observed.items() if entry.get(key) != value]
            if wrong:
                raise RunComparisonError(f'The protocol reference block does not match the {label} run: ' + '; '.join(wrong))
        record[label] = {**observed, 'model_id': TRAINED_MODEL, 'datasets': list(entry['datasets']),
                         'pins_checked': not smoke}
    return record


def check_pairing(baselines, references):
    """Identical nested design, fold schedule, dataset and splits hashes and sealed sources, exactly as
    compare_runs.check_training_pairing pairs a control run with the knn run. Any mismatch is refused."""
    folds = [tuple(key) for key in baselines.schedule['expected_folds']]
    seeds = list(baselines.protocol['fit_seeds'])
    record = {'datasets': {}, 'design': {key: baselines.protocol[key] for key in DESIGN_KEYS},
              'reference_model': TRAINED_MODEL, 'fit_seeds': seeds, 'n_folds': len(folds), 'shared_sources': {}}
    ours = baselines.environment.get('source_hashes') or {}
    for label, run in references.items():
        for key in DESIGN_KEYS:
            if (key not in run.protocol or key not in baselines.protocol
                    or canonical_json(run.protocol[key]) != canonical_json(baselines.protocol[key])):
                raise RunComparisonError(f'Protocol {key} differs between the baselines run and the {label} run '
                                         f'(baselines {baselines.protocol.get(key)!r}, {label} {run.protocol.get(key)!r})')
        if [tuple(key) for key in run.schedule['expected_folds']] != folds:
            raise RunComparisonError(f'The outer fold schedule of the {label} run differs from the baselines run')
        if run.schedule['expected_seeds'][TRAINED_MODEL] != seeds:
            raise RunComparisonError(f'{TRAINED_MODEL} in the {label} run is not fitted at the protocol fitting seeds')
        theirs = run.environment.get('source_hashes') or {}
        absent = [source for source in SHARED_SOURCES if source not in ours or source not in theirs]
        if absent:
            raise RunComparisonError(f'The baselines and {label} environment records must both seal ' + ', '.join(absent))
        differing = sorted(source for source in set(ours) & set(theirs) if ours[source] != theirs[source])
        if differing:
            raise RunComparisonError(f'Sources sealed by both the baselines run and the {label} run differ: '
                                     + ', '.join(differing))
        record['shared_sources'][label] = {source: ours[source] for source in sorted(set(ours) & set(theirs))}
        for name in baselines.protocol['reference'][label]['datasets']:
            mine, other = baselines.manifests.get(name) or {}, run.manifests.get(name) or {}
            pair = {key: (mine.get(key), other.get(key)) for key in ('dataset_hash', 'splits_hash')}
            bad = [key for key, (a, b) in pair.items() if not a or a != b]
            if bad:
                raise RunComparisonError(f'The {", ".join(bad)} of {name} differs between the baselines run and the '
                                         f'{label} run ({pair}); the folds would not be the registered folds')
            record['datasets'][name] = {'reference': label, 'dataset_hash': mine['dataset_hash'],
                                        'splits_hash': mine['splits_hash']}
    for model in MODEL_ORDER:
        expected = seeds if baselines.registry[model].stochastic else seeds[:1]
        if baselines.schedule['expected_seeds'][model] != expected:
            raise RunComparisonError(f'{model} is not fitted at the fitting seeds its stochastic flag declares')
    return record


# ----------------------------------------------------------------------------- tables

def _rows(run, name, model):
    return [row for row in run.summary['model_rows'][name] if row['model_id'] == model]


def _summary(run, name, model, metric):
    return next(row for row in run.summary['summaries'][name] if (row['model_id'], row['metric']) == (model, metric))


def mean_errors(baselines, references, names, reference_of):
    """{dataset: {model: mean outer error}} of every model the rank panel holds, from the verified summaries."""
    errors = {}
    for name in names:
        run = references[reference_of[name]]
        errors[name] = {model: _summary(run, name, model, 'error')['mean'] for model in REFERENCE_PANEL
                        if model in run.registry}
        errors[name].update({model: _summary(baselines, name, model, 'error')['mean'] for model in MODEL_ORDER})
    return errors


def family_rows(baselines, references, names, *, panel_of, reference_of, q, confidence, errors):
    """Per baseline and dataset, arrowflow_full_knn minus the baseline; one Holm adjustment inside each baseline."""
    families = {}
    for model in MODEL_ORDER:
        rows = []
        for name in names:
            reference = references[reference_of[name]]
            seeds = {TRAINED_MODEL: reference.schedule['expected_seeds'][TRAINED_MODEL],
                     model: baselines.schedule['expected_seeds'][model]}
            interval = paired_corrected_interval(_rows(reference, name, TRAINED_MODEL) + _rows(baselines, name, model),
                                                 TRAINED_MODEL, model, metric='accuracy', q=q, confidence=confidence,
                                                 expected_folds=baselines.schedule['expected_folds'], expected_seeds=seeds)
            rows.append({'family': model, 'family_index': len(rows) + 1, 'dataset': name, 'panel': panel_of[name],
                         'reference': reference_of[name], 'contrast': f'{TRAINED_MODEL}_vs_{model}',
                         'model_a': TRAINED_MODEL, 'model_b': model, 'metric': 'accuracy', 'confidence': confidence,
                         **interval, 'mean_error_a': errors[name][TRAINED_MODEL], 'mean_error_b': errors[name][model]})
        for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
            row['holm_p_approximate'] = adjusted
        families[model] = rows
    return families


def metric_rows(baselines, references, names, *, panel_of, reference_of):
    """Per dataset, every baseline and the reference ArrowFlow-kNN: each report metric's mean, outer-fold SD and mean
    within-fold seed SD from the verified summaries."""
    rows = []
    for name in names:
        for model in (TRAINED_MODEL, *MODEL_ORDER):
            run = references[reference_of[name]] if model == TRAINED_MODEL else baselines
            source = reference_of[name] if model == TRAINED_MODEL else BASELINES
            for metric in METRICS:
                entry = _summary(run, name, model, metric)
                rows.append({'status': DESCRIPTIVE, 'dataset': name, 'panel': panel_of[name], 'model_id': model,
                             'source': source, 'metric': metric, 'mean': entry['mean'],
                             **{key: entry[key] for key in ('outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds',
                                                            'seeds_per_fold')}})
    return rows


def average_ranks(values):
    """Ranks by ascending value, ties taking the average rank (1 is the lowest error)."""
    order = sorted(values, key=lambda key: (values[key], key))
    ranks, index = {}, 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and abs(values[order[stop + 1]] - values[order[index]]) <= TOLERANCE:
            stop += 1
        shared = (index + stop) / 2 + 1
        for key in order[index:stop + 1]:
            ranks[key] = shared
        index = stop + 1
    return ranks


def rank_rows(names, errors, *, panel_of, reference_of):
    """Each model's rank by mean outer error among the models every dataset holds; descriptive."""
    rows, panel = [], list(RANK_PANEL)
    missing = {name: sorted(set(panel) - set(errors[name])) for name in names}
    absent = {name: models for name, models in missing.items() if models}
    if absent:
        raise ValueError(f'The rank panel is not held by every dataset: {absent}')
    for name in names:
        values = {model: errors[name][model] for model in panel}
        ranks = average_ranks(values)
        for model in panel:
            rows.append({'status': DESCRIPTIVE, 'dataset': name, 'panel': panel_of[name], 'model_id': model,
                         'source': BASELINES if model in MODEL_ORDER else reference_of[name],
                         'mean_error': values[model], 'rank': ranks[model], 'models_ranked': len(panel)})
    return rows


def fit_warnings(run, names):
    """Per dataset and model: the fits raising a warning and the count of each category, over the outer fits and over
    every fit of the complete fit logs (inner screening, reranking and outer), with the nca_knn fits that reached
    max_iter. The fit logs are the records collect_confirmatory_results already reconciled with the results."""
    counts = {}

    def bucket(key, scope):
        return counts.setdefault(key, {}).setdefault(scope, {'fits': 0, 'fits_with_warning': 0, 'warnings': 0,
                                                             'categories': Counter(), 'reached_max_iter_fits': 0})

    for job in run.jobs:
        key = (job['dataset_id'], job['model_id'])
        try:
            rows = [json.loads(line) for line in (run.path/job['log_file']).read_text().splitlines()]
        except (OSError, ValueError) as exc:
            raise RunComparisonError(f'{run.label} run {job["log_file"]}: unreadable fit log ({_reason(exc)})') from exc
        for row in rows:
            scopes = ['all_fits'] + (['outer_fits'] if row.get('stage') == 'outer' else [])
            for scope in scopes:
                entry = bucket(key, scope)
                entry['fits'] += 1
                warnings_ = row.get('fit_warnings') or []
                entry['warnings'] += len(warnings_)
                entry['fits_with_warning'] += 1 if warnings_ else 0
                entry['categories'].update(w.get('category', 'unknown') for w in warnings_)
                metadata = row.get('representation_metadata') or {}
                entry['reached_max_iter_fits'] += 1 if metadata.get('reached_max_iter') else 0
    rows = []
    for name in names:
        for model in MODEL_ORDER:
            for scope in ('outer_fits', 'all_fits'):
                entry = bucket((name, model), scope)
                rows.append({'status': DESCRIPTIVE, 'dataset': name, 'model_id': model, 'scope': scope,
                             'fits': entry['fits'], 'fits_with_warning': entry['fits_with_warning'],
                             'warnings': entry['warnings'],
                             'categories': ';'.join(f'{category}={count}'
                                                    for category, count in sorted(entry['categories'].items())),
                             'reached_max_iter_fits': entry['reached_max_iter_fits'] if model == NCA_MODEL else None})
    return rows


def interpretation(block, names, errors):
    """The rule fixed in the protocol before any outer score existed, applied to the mean outer errors."""
    declared = block['interpretation']
    threshold = len(names) // 2 + 1
    per_model = {}
    for model in MODEL_ORDER:
        matched = [name for name in names if errors[name][model] <= errors[name][TRAINED_MODEL] + TOLERANCE]
        per_model[model] = {'matched_or_better_datasets': matched, 'count': len(matched), 'of': len(names),
                            'most_datasets': len(matched) >= threshold,
                            'supervised_neighbourhood_baseline': model in SUPERVISED_MODELS}
    triggered = [model for model in SUPERVISED_MODELS if per_model[model]['most_datasets']]
    return {'status': declared['status'], 'rule': declared['rule'], 'threshold_datasets': threshold,
            'by_model': per_model, 'supervised_baselines_matching_on_most_datasets': triggered,
            'matched': bool(triggered), 'statement': declared['if_matched'] if triggered else declared['if_not_matched'],
            'both_statements': {'if_matched': declared['if_matched'], 'if_not_matched': declared['if_not_matched']}}


# ----------------------------------------------------------------------------- the commands

def check_prepared(run_source, *, runs_root=None, references=None):
    """Before the jobs start: the prepared baselines directory against the complete reference runs (no score exists)."""
    baselines = load_prepared_run(run_source, BASELINES)
    smoke = baselines.protocol.get('purpose') == 'synthetic_smoke_only'
    paths = reference_paths(baselines.protocol, runs_root, references)
    completeness_gate(dict(paths))
    loaded = {label: load_run(path, label) for label, path in paths.items()}
    frozen = check_frozen_protocol(baselines, smoke=smoke)
    reference = check_reference(baselines, loaded, smoke=smoke)
    pairing = check_pairing(baselines, loaded)
    return {'protocol': frozen, 'reference': reference, 'pairing': pairing,
            'runs': {label: run.provenance() for label, run in loaded.items()}}


def analyse(run_source, output, *, runs_root=None, references=None, allow_smoke=False):
    check_output(output, OUTPUTS)                   # an unusable output location is refused before any run is read
    protocol = read_protocol(run_source)
    smoke = protocol.get('purpose') == 'synthetic_smoke_only'
    if smoke and not allow_smoke:
        raise RunComparisonError('The run holds a synthetic smoke protocol; a synthetic smoke run is never evidence')
    paths = {BASELINES: Path(run_source), **reference_paths(protocol, runs_root, references)}
    completeness_gate(paths)
    runs = {label: load_run(path, label) for label, path in paths.items()}
    baselines = runs[BASELINES]
    loaded = {label: runs[label] for label in paths if label != BASELINES}
    frozen = check_frozen_protocol(baselines, smoke=smoke)
    reference = check_reference(baselines, loaded, smoke=smoke)
    pairing = check_pairing(baselines, loaded)
    verified = {label: verify_run(run)[0] for label, run in runs.items()}
    block = baselines.protocol['analysis']
    names = list(block['datasets'])
    panel_of = {entry['name']: entry['panel'] for entry in baselines.protocol['panel']}
    reference_of = {name: label for label, entry in baselines.protocol['reference'].items() for name in entry['datasets']}
    try:
        errors = mean_errors(baselines, loaded, names, reference_of)
        families = family_rows(baselines, loaded, names, panel_of=panel_of, reference_of=reference_of,
                               q=baselines.protocol['test_train_ratio'], confidence=baselines.protocol['confidence'],
                               errors=errors)
        metrics = metric_rows(baselines, loaded, names, panel_of=panel_of, reference_of=reference_of)
        ranks = rank_rows(names, errors, panel_of=panel_of, reference_of=reference_of)
        warnings_ = fit_warnings(baselines, names)
        reading = interpretation(block, names, errors)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        raise RunComparisonError(f'The verified runs do not support the prespecified analysis: {_reason(exc)}') from exc
    family_table = [row for model in MODEL_ORDER for row in families[model]]
    tables = {FAMILIES_CSV: (family_table, FAMILY_COLUMNS), METRICS_CSV: (metrics, METRIC_COLUMNS),
              RANKS_CSV: (ranks, RANK_COLUMNS), WARNINGS_CSV: (warnings_, WARNING_COLUMNS)}
    contents = {name: _csv_text(rows, columns) for name, (rows, columns) in tables.items()}
    record = {
        'purpose': 'prespecified_analysis_of_the_neighbour_baselines_run',
        'status': 'computed after the baselines run and every referenced ArrowFlow-kNN run were complete and re-verified; '
                  'each baseline family Holm-adjusted within itself and separately from the other three; the metric, rank '
                  'and warning tables descriptive',
        'runs_purpose': baselines.protocol.get('purpose'), 'family': FAMILY, 'analysis_declaration': block,
        'metric': 'accuracy', 'interval': INTERVAL_RULE, 'datasets': names, 'models': list(MODEL_ORDER),
        'reference_model': TRAINED_MODEL, 'reference_not_refitted': block['reference_not_refitted'],
        'families': families, 'mean_errors': errors, 'metrics': metrics, 'ranks': ranks, 'rank_panel': list(RANK_PANEL),
        'convergence_warnings': warnings_, 'interpretation': reading, 'notes': block['notes'],
        'outputs': {name: {'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'rows': len(tables[name][0])}
                    for name, text in contents.items()},
        'provenance': {'protocol': frozen, **{label: run.provenance() for label, run in runs.items()},
                       'reference': reference, 'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS),
                                        **{label: {'jobs_verified': len(run.jobs),
                                                   'model_rows_verified': sum(map(len, verified[label].values()))}
                                           for label, run in runs.items()}},
                       'analysis_sources': analysis_sources(ANALYSIS_SOURCES)}}
    contents[ANALYSIS_JSON] = _json_text(record)
    write_outputs(output, contents)
    return record


def print_record(record, stream=None):
    """The console report of one analysis: every family member, then the interpretation the protocol fixed."""
    for model in MODEL_ORDER:
        for row in record['families'][model]:
            print(f"{model} {row['dataset']}: {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
                  f"Holm p={row['holm_p_approximate']:.3g}", file=stream)
    reading = record['interpretation']
    for model, entry in reading['by_model'].items():
        print(f"{model} matched or beat {TRAINED_MODEL} on {entry['count']}/{entry['of']} datasets "
              f"(most: {entry['most_datasets']})", file=stream)
    print(reading['statement'], file=stream)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, text in (('pairing', 'check a prepared baselines directory against the registered runs; writes nothing'),
                          ('analyse', 'the prespecified analysis of the complete baselines run')):
        sub = commands.add_parser(command, help=text)
        sub.add_argument('--run', type=Path, required=True, help='the neighbour-baselines run directory')
        sub.add_argument('--runs', type=Path, help='the directory holding the registered ArrowFlow-kNN runs')
        if command == 'analyse':
            sub.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'pairing':
            record = check_prepared(args.run, runs_root=args.runs)
            pinned = ', '.join(f'{label} ({entry["protocol_id"]})' for label, entry in record['reference'].items())
            pairing = record['pairing']
            print(f'paired {record["protocol"]["protocol_id"]} with {pinned}: {len(pairing["datasets"])} datasets with '
                  f'identical dataset and splits hashes, {pairing["n_folds"]} outer folds, fitting seeds '
                  f'{pairing["fit_seeds"]}')
            return
        record = analyse(args.run, args.output, runs_root=args.runs)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_baselines {args.command} refused: {exc}\n')
    print_record(record)


if __name__ == '__main__':
    main()
