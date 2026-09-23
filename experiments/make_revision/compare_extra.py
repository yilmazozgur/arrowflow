"""G4 follow-up experiments (a) and (c): the prespecified analysis of one extra family; nothing is read until its run and its
component ablation are complete.

python -m experiments.make_revision.compare_extra analyse --family dedup|artificial --run R --ablation A --output O [--runs D]
    refuses (exit 2, nothing written) unless the run is complete (protocol, environment, candidates, planned jobs and
    summary.json; every planned result file and fit log) and the ablation is complete (its records, its summary JSON and
    CSV; every planned job's result, log and prediction file), checked before any record holding a score is read. The run is
    then loaded and re-verified with compare_runs.load_run and compare_runs.verify_run (the harness validators over every
    saved record) and must hold the committed frozen family protocol, the ten newdata_registry models with the candidates of
    this tree, the sealed sources and the pinned dataset and splits hashes. The ablation is re-verified by
    extra_ablation.summary (every job, the views7 reproduction, every sealed selection re-derived from the run), must have
    been prepared against this run and must equal its published summary JSON and CSV. For dedup, the registered full-data
    values are loaded with holistic.load_panel from the runs directory D (every registered run re-verified, the registered
    pairings re-checked). The protocol's analysis block is then computed:
      primary family    per family dataset, arrowflow_full_knn minus arrowflow_knn_untrained accuracy, fitting seeds averaged
                        within each outer fold, corrected resampled t (q 0.25, 95%, 14 df), Holm across the family datasets
      secondary family  the same for arrowflow_full_knn minus input_footrule_knn, Holm across the family datasets separately
      main table        per dataset, mean outer error, outer-fold SD and seed SD of ArrowFlow-kNN, the five tuned classical
                        models and the majority class
      comparators       per dataset, arrowflow_full_knn minus each comparator, paired intervals without adjustment
      competitiveness   holistic.competitiveness under holistic.RULES
      ladder            raw numeric kNN, projected kNN, input kNN, untrained ArrowFlow-kNN and ArrowFlow-kNN
      complete metrics  every model's error, balanced accuracy, macro-F1 and accuracy with outer-fold and seed SDs
      selected widths   ArrowFlow-kNN's hidden widths per outer fold
      duplicates        the exact-duplicate audit of the prepared features (NaN equal to NaN)
      components        per dataset and variant, ArrowFlow (views7) minus the variant, accuracy, with its interval, in points
      full data         dedup only: each model's mean error on the registered full dataset beside the deduplicated one;
                        unpaired and descriptive
Outputs (all or none; an existing file with different content is never replaced): <family>_families.csv,
<family>_main_table.csv, <family>_comparators.csv, <family>_competitiveness.csv, <family>_ladder.csv,
<family>_complete_metrics.csv, <family>_selected_widths.csv, <family>_duplicates.csv, <family>_components.csv, for dedup
dedup_full_data.csv, and <family>_analysis.json (every result, the declaration, the sha256 of each CSV and provenance).
"""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile
from . import compare_newdata as cn
from . import extra_runs as er
from . import holistic
from .compare_runs import RUN_RECORDS, SHARED_SOURCES, VALIDATORS, RunComparisonError, _reason, check_output, load_run, verify_run
from .evaluation import canonical_json
from .extra_data import duplicate_groups
from .knn_controls import TRAINED_MODEL, selected_widths
from .newdata import MODEL_ORDER, candidate_record, sha256_file
from .referee_analyses import COUNT_FIELDS, check_design, code_record, finish, training_duplicate_flags
from .run_revision import load_prepared

ANALYSIS_SOURCES = ('compare_extra.py', 'extra_runs.py', 'extra_ablation.py', 'extra_data.py', 'compare_newdata.py', 'holistic.py',
                    'compare_runs.py', 'referee_analyses.py', 'newdata.py', 'knn_controls.py', 'run_knn_ablation.py',
                    'run_newdata_ablation.py', 'evaluation.py', 'reporting.py', 'run_revision.py')
SEALED_SOURCES = SHARED_SOURCES + tuple(f'experiments/make_revision/{name}.py' for name in
                                        ('knn_controls', 'projected_knn', 'newdata', 'extra_data', 'extra_runs'))
TABLES = ('families', 'main_table', 'comparators', 'competitiveness', 'ladder', 'complete_metrics', 'selected_widths',
          'duplicates', 'components')
INTERVAL = cn.INTERVAL
FAMILY_COLUMNS = ('family', 'family_index', 'dataset', 'role', 'contrast', 'model_a', 'model_b', *INTERVAL[:4], 'p_approximate',
                  'holm_p_approximate', *INTERVAL[4:])
COMPARATOR_COLUMNS = ('status', 'dataset', 'role', 'model_a', 'model_b', *INTERVAL[:4], 'p_unadjusted', *INTERVAL[4:], 'mean_error_a',
                      'mean_error_b', 'best_comparator')
LADDER_COLUMNS = ('dataset', 'role', 'rung', 'model_id', 'mean_error', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds',
                  'seeds_per_fold')
WIDTH_COLUMNS = ('dataset', 'role', 'outer_repeat', 'outer_fold', 'config_id', 'widths', 'hidden_layers')
DUPLICATE_COLUMNS = ('status', 'dataset', 'role', *COUNT_FIELDS, 'matches_expected_counts', 'test_rows',
                     'test_rows_with_training_duplicate', 'share_with_training_duplicate', 'fold_share_min', 'fold_share_max')
COMPONENT_COLUMNS = (*holistic.COMPONENT_COLUMNS, 'arrowflow_minus_variant_points', 'ci_low_points', 'ci_high_points')
FULL_DATA_COLUMNS = ('status', 'dataset', 'source_dataset', 'model_id', 'dedup_mean_error', 'dedup_outer_fold_sd', 'full_mean_error',
                     'full_outer_fold_sd', 'dedup_minus_full_points', 'dedup_rows', 'full_rows', 'full_source_run')
FULL_DATA_STATUS = 'unpaired; descriptive; different rows and splits, so no interval and no test'


def output_names(family):
    return (*(f'{family}_{table}.csv' for table in TABLES), *(('dedup_full_data.csv',) if family == 'dedup' else ()),
            f'{family}_analysis.json')


def summary_entry(run, name, model, metric):
    return next(row for row in run.summary['summaries'][name] if (row['model_id'], row['metric']) == (model, metric))


def points(value):
    return None if value is None else holistic.points(value)


# ----------------------------------------------------------------------------- completeness, the run and the ablation

def completeness_gate(family, run_source, ablation_source):
    """Refuse unless the run and the ablation are complete. Only the planned job lists are parsed, so nothing holding a score
    is read before both are complete."""
    from .extra_ablation import ABLATION_RECORDS, summary_names
    problems = []
    checks = (('run', Path(run_source), RUN_RECORDS, lambda job: (job['result_file'], job['log_file'])),
              ('ablation', Path(ablation_source), (*ABLATION_RECORDS, *summary_names(family)),
               lambda job: (f"results/{job['stem']}.json", f"logs/{job['stem']}.jsonl", f"predictions/{job['stem']}.jsonl")))
    for label, path, records, planned in checks:
        if not path.is_dir():
            problems.append(f'{family} {label} {path}: no such directory')
            continue
        missing = [name for name in records if not (path/name).is_file()]
        absent, first = 0, None
        if (path/'planned_jobs.json').is_file():
            try:
                for job in json.loads((path/'planned_jobs.json').read_text()):
                    for relative in planned(job):
                        if not (path/relative).is_file():
                            absent, first = absent + 1, first or relative
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f'{family} {label} {path}: unreadable planned_jobs.json ({_reason(exc)})')
                continue
        if missing or absent:
            parts = ([f'missing {", ".join(missing)}'] if missing else []) + (
                [f'{absent} planned job files missing (first: {first})'] if absent else [])
            problems.append(f'{family} {label} {path} is not complete: ' + '; '.join(parts))
    if problems:
        raise RunComparisonError('The analysis runs only after the run and its component ablation are complete, and reads '
                                 'neither before: ' + ' | '.join(problems))


def check_run(run, family, *, smoke):
    """The run holds the committed frozen family protocol (a synthetic smoke protocol only in the smoke), the ten models with
    the candidates of this tree, the protocol registry, the sealed sources and the pinned dataset and splits hashes."""
    protocol = run.protocol
    try:
        er.validate_extra_protocol(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunComparisonError(f'The run protocol is not an extra family protocol: {_reason(exc)}') from exc
    if protocol.get('production_family') != family:
        raise RunComparisonError(f'The run holds the {protocol.get("production_family")!r} family, not {family!r}')
    if (protocol.get('purpose') == 'synthetic_smoke_only') != smoke:
        raise RunComparisonError('A synthetic smoke run is analysed only by the smoke, and a production run only in production')
    if not smoke:
        committed = er.PROTOCOL_FILES[family]
        if not committed.is_file() or run.protocol_sha256 != sha256_file(committed):
            raise RunComparisonError(f'The run protocol.json (sha256 {run.protocol_sha256}) is not the committed frozen {committed.name}')
        check_design(run)
    if sorted(run.registry) != sorted(MODEL_ORDER):
        raise RunComparisonError(f'The run must hold the ten newdata models, not {sorted(run.registry)}')
    if not smoke and canonical_json(run.candidates) != canonical_json(candidate_record(er.extra_registry(protocol))):
        raise RunComparisonError('The run candidates differ from the extra_runs registry of this tree')
    if run.environment.get('registry') != protocol['registry']:
        raise RunComparisonError(f'The run environment must name the protocol registry {protocol["registry"]}')
    sealed = run.environment.get('source_hashes') or {}
    absent = [source for source in SEALED_SOURCES if source not in sealed]
    if absent:
        raise RunComparisonError('The run environment must seal ' + ', '.join(absent))
    entries, datasets = er.panel_by_name(protocol), {}
    for name in protocol['datasets']:
        manifest = run.manifests[name]
        datasets[name] = {'dataset_hash': manifest.get('dataset_hash'), 'splits_hash': manifest.get('splits_hash')}
        if not smoke and datasets[name] != {key: entries[name].get(key) for key in ('dataset_hash', 'splits_hash')}:
            raise RunComparisonError(f'The run {name} dataset or splits hash differs from its pin')
    return {'protocol_id': protocol['protocol_id'], 'protocol_sha256': run.protocol_sha256,
            'code_revision': run.environment.get('code_revision'), 'registry': protocol['registry'], 'sealed_sources': sealed,
            'datasets': datasets, 'fit_seeds': protocol['fit_seeds'], 'pins_checked': not smoke}


def _csv_text(rows, columns):
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows([row[column] for column in columns] for row in rows)
    return buffer.getvalue()


def verified_ablation(family, path, run, *, smoke):
    """The ablation re-verified by extra_ablation.summary; it must hold the run's protocol, have been sealed against this run
    and equal its published summary JSON and CSV."""
    from . import extra_ablation
    from .run_knn_ablation import SUMMARY_COLUMNS
    path = Path(path)
    summary_json, summary_csv = extra_ablation.summary_names(family)
    try:
        protocol = json.loads((path/'protocol.json').read_text())
        manifest = json.loads((path/'manifest.json').read_text())
    except (OSError, ValueError) as exc:
        raise RunComparisonError(f'The {family} ablation {path}: unreadable records ({_reason(exc)})') from exc
    if canonical_json(protocol) != canonical_json(run.protocol):
        raise RunComparisonError('The ablation holds another protocol than the run')
    sealed = manifest.get('reference_source') or {}
    observed = {'directory': str(run.path), 'protocol_sha256': run.protocol_sha256, 'summary_sha256': run.summary_sha256,
                'code_revision': run.environment.get('code_revision'), 'protocol_id': run.protocol.get('protocol_id')}
    wrong = [f'{key} sealed {sealed.get(key)!r}, run {value!r}' for key, value in observed.items() if sealed.get(key) != value]
    if wrong:
        raise RunComparisonError('The ablation was not prepared against this run: ' + '; '.join(wrong))
    try:
        report, flat = extra_ablation.summary(path, allow_smoke=smoke)
        published, published_csv = json.loads((path/summary_json).read_text()), (path/summary_csv).read_text()
    except (OSError, KeyError, TypeError, ValueError, IndexError, StopIteration, EOFError, zipfile.BadZipFile) as exc:
        raise RunComparisonError(f'The {family} ablation {path} fails its re-verification: {_reason(exc)}') from exc
    if canonical_json(published) != canonical_json(report):
        raise RunComparisonError(f'{path/summary_json} differs from the recomputation from the verified ablation')
    if published_csv != _csv_text(flat, SUMMARY_COLUMNS):
        raise RunComparisonError(f'{path/summary_csv} differs from the recomputation from the verified ablation')
    files = {name: {'path': str(path/name), 'sha256': sha256_file(path/name)}
             for name in ('protocol.json', 'manifest.json', summary_json, summary_csv)}
    return report, files, sealed


# ----------------------------------------------------------------------------- tables

def _roles(rows):
    """compare_newdata rows carry stratum and batch; here the stratum argument is the dataset's role and there is no batch."""
    return [{**{key: value for key, value in row.items() if key not in ('stratum', 'batch')}, 'role': row['stratum']} for row in rows]


class FamilyPanel:
    """The attributes of holistic.Panel that holistic.complete_metric_rows reads, for one verified family run."""
    def __init__(self, run, family):
        self.run, self.datasets = run, list(run.protocol['datasets'])
        self.group = dict.fromkeys(self.datasets, family)

    def run_for(self, name, model):
        return self.run

    def summary(self, name, model, metric):
        return summary_entry(self.run, name, model, metric)


def main_table_rows(run, names, family):
    return [{'dataset': name, 'group': family, 'model_id': model, 'source_run': run.label,
             'mean_error': summary_entry(run, name, model, 'error')['mean'],
             **{key: summary_entry(run, name, model, 'error')[key] for key in holistic.MAIN_COLUMNS[5:]}}
            for name in names for model in holistic.MAIN_MODELS]


def competitiveness_rows(run, names, family):
    rows = []
    for name in names:
        result = holistic.competitiveness({model: summary_entry(run, name, model, 'error')['mean'] for model in holistic.MAIN_MODELS})
        rows.append({'status': 'descriptive; fixed rules (holistic.RULES)', 'dataset': name, 'group': family, **result,
                     'best_classical': '+'.join(result['best_classical'])})
    return rows


def width_rows(run, names, roles, depths):
    rows, counts = [], {}
    for name in names:
        model_rows = [row for row in run.summary['model_rows'][name] if row['model_id'] == TRAINED_MODEL]
        widths, configs = selected_widths(model_rows, TRAINED_MODEL), {}
        for row in model_rows:
            if configs.setdefault((row['outer_repeat'], row['outer_fold']), row['config_id']) != row['config_id']:
                raise RunComparisonError(f'{name}: {TRAINED_MODEL} rows of one outer fold disagree on the selected configuration')
        folds = [tuple(fold) for fold in run.schedule['expected_folds']]
        for fold in folds:
            rows.append({'dataset': name, 'role': roles[name], 'outer_repeat': fold[0], 'outer_fold': fold[1], 'config_id': configs[fold],
                         'widths': json.dumps(widths[fold]), 'hidden_layers': len(widths[fold])})
        counts[name] = {json.dumps(depth): sum(list(widths[fold]) == depth for fold in folds) for depth in depths}
        counts[name]['other'] = sum(list(widths[fold]) not in depths for fold in folds)
    return rows, counts


def duplicate_audit(run, names, roles, expected):
    """compare_newdata.duplicate_audit with extra_data.duplicate_groups (NaN equal to NaN) on the run's prepared features."""
    rows, records = [], {}
    for name in names:
        X, y, _, splits = load_prepared(run.path, name)
        vector, counts = duplicate_groups(X, y)
        folds = []
        for repeat, fold in run.schedule['expected_folds']:
            split = splits[repeat * run.protocol['outer_folds'] + fold]
            flags = training_duplicate_flags(vector, split['train'], split['test'])
            folds.append({'outer_repeat': repeat, 'outer_fold': fold, 'n_test': len(split['test']), 'n_with_training_duplicate': int(sum(flags))})
        tested, flagged = sum(f['n_test'] for f in folds), sum(f['n_with_training_duplicate'] for f in folds)
        shares = [f['n_with_training_duplicate'] / f['n_test'] for f in folds]
        pinned = expected.get(name)
        matches = None if pinned is None else all(counts[key] == value for key, value in pinned.items())
        rows.append({'status': 'descriptive', 'dataset': name, 'role': roles[name], **{key: counts[key] for key in COUNT_FIELDS},
                     'matches_expected_counts': matches, 'test_rows': tested, 'test_rows_with_training_duplicate': flagged,
                     'share_with_training_duplicate': flagged / tested, 'fold_share_min': min(shares), 'fold_share_max': max(shares)})
        records[name] = {'counts': {key: counts[key] for key in COUNT_FIELDS}, 'group_sizes': counts['group_sizes'],
                         'groups': counts['groups'], 'expected_counts': pinned, 'matches_expected_counts': matches, 'folds': folds}
    return rows, records


def component_table(report, family):
    """holistic.component_rows on the family ablation report (the key names the report kind holistic reads); the group and the
    source run are relabelled for the family and the points added."""
    rows, _ = holistic.component_rows({'newdata_ablation': report})
    for row in rows:
        row.update(group=family, source_run=f'{family}_ablation', arrowflow_minus_variant_points=points(row['arrowflow_minus_variant']),
                   ci_low_points=points(row['ci_low']), ci_high_points=points(row['ci_high']))
    return rows


def registered_full_data(runs_root, sources):
    """{(source dataset, model): its registered mean outer error, outer-fold SD, run and rows} loaded as holistic.py loads the
    benchmark datasets (holistic.load_panel re-verifies every registered run and re-checks the registered pairings)."""
    panel, provenance = holistic.load_panel(holistic.source_paths(runs_root))
    values = {}
    for source in sources:
        if source not in panel.benchmark:
            raise RunComparisonError(f'{source} is not a benchmark dataset of the registered runs')
        rows = panel.run_for(source, TRAINED_MODEL).manifests[source]['shape'][0]
        for model in MODEL_ORDER:
            entry = panel.summary(source, model, 'error')
            values[source, model] = {'mean_error': entry['mean'], 'outer_fold_sd': entry['outer_fold_sd'],
                                     'source_run': panel.run_for(source, model).label, 'rows': rows}
    return values, {'runs_root': str(Path(runs_root).resolve()), **provenance}


def smoke_full_data(run, sources):
    """Synthetic smoke only: the 'registered' values are read from the smoke run itself, so the table runs end to end."""
    values = {(source, model): {'mean_error': summary_entry(run, source, model, 'error')['mean'],
                                'outer_fold_sd': summary_entry(run, source, model, 'error')['outer_fold_sd'],
                                'source_run': run.label, 'rows': run.manifests[source]['shape'][0]}
              for source in set(sources.values()) for model in MODEL_ORDER}
    return values, {'purpose': 'synthetic_smoke_only: the registered values are the smoke run itself'}


def full_data_rows(run, names, sources, registered):
    rows = []
    for name in names:
        for model in MODEL_ORDER:
            entry, full = summary_entry(run, name, model, 'error'), registered[sources[name], model]
            rows.append({'status': FULL_DATA_STATUS, 'dataset': name, 'source_dataset': sources[name], 'model_id': model,
                         'dedup_mean_error': entry['mean'], 'dedup_outer_fold_sd': entry['outer_fold_sd'],
                         'full_mean_error': full['mean_error'], 'full_outer_fold_sd': full['outer_fold_sd'],
                         'dedup_minus_full_points': holistic.points(entry['mean'] - full['mean_error']),
                         'dedup_rows': run.manifests[name]['shape'][0], 'full_rows': full['rows'], 'full_source_run': full['source_run']})
    return rows


# ----------------------------------------------------------------------------- the analysis

def analyse(family, run_source, ablation_source, output, *, runs_root=None, allow_smoke=False):
    if family not in er.FAMILIES:
        raise RunComparisonError(f'family must be one of {", ".join(er.FAMILIES)}')
    names_out = output_names(family)
    check_output(output, names_out)                 # an unusable output location is refused before any run is read
    if family == 'dedup' and runs_root is None and not allow_smoke:
        raise RunComparisonError('The dedup analysis needs --runs, the directory holding the registered runs')
    completeness_gate(family, run_source, ablation_source)
    run = load_run(run_source, family)
    smoke = allow_smoke and run.protocol.get('purpose') == 'synthetic_smoke_only'
    pairing = check_run(run, family, smoke=smoke)
    verified, _ = verify_run(run)
    report, ablation_files, sealed = verified_ablation(family, ablation_source, run, smoke=smoke)
    protocol = run.protocol
    block = protocol['analysis']
    names, members = list(block['datasets']), list(block['family_datasets'])
    roles = {entry['name']: entry['role'] for entry in protocol['panel']}
    if names != list(protocol['datasets']) or members != [name for name in names if roles[name] == 'family']:
        raise RunComparisonError('The analysis block does not declare the protocol datasets and their family members')
    if family == 'dedup':
        sources = dict(block['full_data_comparison']['sources'])
        registered, registered_provenance = (smoke_full_data(run, sources) if smoke
                                             else registered_full_data(runs_root, sorted(set(sources.values()))))
    by_dataset = dict.fromkeys(names, run)
    common = {'stratum': roles, 'batch_of': dict.fromkeys(names, 1), 'q': protocol['test_train_ratio'], 'confidence': protocol['confidence']}
    try:
        families = {}
        for label in ('primary', 'secondary'):
            declared = block[f'{label}_family']
            if declared['datasets'] != members or declared['size'] != len(members):
                raise ValueError(f'the {label} family does not declare the family datasets')
            families[label] = _roles(cn.family_rows(by_dataset, members, family=label, contrast=declared['contrast'],
                                                    model_a=declared['model_a'], model_b=declared['model_b'], **common))
        main = main_table_rows(run, names, family)
        comparators = _roles(cn.comparator_rows(by_dataset, names, **common))
        competitive = competitiveness_rows(run, names, family)
        ladder = _roles(cn.ladder_rows(by_dataset, names, stratum=roles, batch_of=common['batch_of']))
        metrics = holistic.complete_metric_rows(FamilyPanel(run, family))
        widths, width_counts = width_rows(run, names, roles, protocol['ablation']['depth_split']['depths'])
        duplicates, duplicate_records = duplicate_audit(run, names, roles, block['duplicate_audit']['expected'])
        if list(report['summaries']) != names:
            raise ValueError('the ablation summaries do not cover the protocol datasets')
        components = component_table(report, family)
        full = full_data_rows(run, names, sources, registered) if family == 'dedup' else None
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified {family} run does not support the prespecified analysis: {_reason(exc)}') from exc
    tables = {f'{family}_families.csv': (families['primary'] + families['secondary'], FAMILY_COLUMNS),
              f'{family}_main_table.csv': (main, holistic.MAIN_COLUMNS),
              f'{family}_comparators.csv': (comparators, COMPARATOR_COLUMNS),
              f'{family}_competitiveness.csv': (competitive, holistic.COMPETITIVENESS_COLUMNS),
              f'{family}_ladder.csv': (ladder, LADDER_COLUMNS),
              f'{family}_complete_metrics.csv': (metrics, holistic.COMPLETE_COLUMNS),
              f'{family}_selected_widths.csv': (widths, WIDTH_COLUMNS),
              f'{family}_duplicates.csv': (duplicates, DUPLICATE_COLUMNS),
              f'{family}_components.csv': (components, COMPONENT_COLUMNS)}
    if full is not None:
        tables['dedup_full_data.csv'] = (full, FULL_DATA_COLUMNS)
    record = {
        'purpose': f'g4_follow_up_prespecified_analysis_of_the_{family}_family',
        'status': 'computed after the run and its component ablation were complete and re-verified; each family Holm-adjusted '
                  'within itself; the main table, comparator intervals, competitiveness flags, ladder, complete metrics, '
                  'selected widths, duplicate audit and components descriptive' + ('; the full-data comparison unpaired and '
                                                                                   'descriptive' if family == 'dedup' else ''),
        'runs_purpose': protocol.get('purpose'), 'analysis_declaration': block,
        'primary_family': families['primary'], 'secondary_family': families['secondary'], 'main_table': main,
        'comparator_intervals': comparators, 'competitiveness': competitive, 'ladder': ladder, 'complete_metrics': metrics,
        'selected_widths': {'rows': widths, 'counts': width_counts}, 'duplicate_audit': duplicate_records, 'components': components,
        'views7_reproduces_reference': {name: entry['views7_reproduces_reference'] for name, entry in report['summaries'].items()},
        'notes': block['notes'],
        'provenance': {'run': run.provenance(), 'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS), 'jobs_verified': len(run.jobs),
                                        'model_rows_verified': sum(map(len, verified.values()))},
                       'ablation': {'files': ablation_files, 'reference_source': sealed, 'code_revision': report['code_revision'],
                                    'jobs_verified': sum(len(entry['resolved_configurations']) for entry in report['summaries'].values())},
                       **code_record(ANALYSIS_SOURCES)}}
    if family == 'dedup':
        record['full_data_comparison'] = {'status': FULL_DATA_STATUS, 'rows': full, 'registered': registered_provenance}
    finish(output, tables, f'{family}_analysis.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('analyse', help='the prespecified analysis of one complete family run and its ablation')
    sub.add_argument('--family', choices=er.FAMILIES, required=True)
    sub.add_argument('--run', type=Path, required=True, help='the family run directory (summary.json written)')
    sub.add_argument('--ablation', type=Path, required=True, help='the family component ablation directory (summary written)')
    sub.add_argument('--output', type=Path, required=True, help='new output directory')
    sub.add_argument('--runs', type=Path, help='dedup: the directory holding the registered runs under their names')
    args = parser.parse_args(argv)
    try:
        record = analyse(args.family, args.run, args.ablation, args.output, runs_root=args.runs)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_extra analyse refused: {exc}\n')
    for row in record['primary_family'] + record['secondary_family']:
        print(f"{row['family']} {row['dataset']}: {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} Holm p={row['holm_p_approximate']:.3g}")
    for name, counts in record['views7_reproduces_reference'].items():
        print(f"{name}: views7 reproduced {counts['matching_fold_seeds']}/{counts['total_fold_seeds']} fold-seeds")


if __name__ == '__main__':
    main()
