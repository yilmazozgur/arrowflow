"""The prespecified analysis of the readout-matched representation test; nothing is read until the run is complete.

python -m experiments.make_revision.compare_representation analyse --run R --output O [--reference NAME=RUN_DIR,ABLATION_DIR ...]
    Refuses (exit 2, nothing written) unless the representation-test run is complete: its protocol, environment, manifest,
    planned jobs and sealed selections, its summary, and every planned job's result, log, prediction file and artifact. That
    gate parses only the planned job list, so no record holding a score is read before the run is complete. Each reference
    production run is then re-verified with compare_runs.load_run and compare_runs.verify_run (the harness validators over
    every saved record), and the run with representation_test.summary, which re-derives every sealed selection and every
    stored prediction hash from the reference runs and re-checks every job; the published summary must equal the re-verified
    one. Only then is any outer score read.

The analysis is the one the frozen protocol declares, written before any outer score of these readouts existed:
  primary family    svc_trained minus svc_untrained, accuracy, on the ten further datasets (never screened): fitting seeds
                    averaged within each outer fold, corrected resampled t (q 0.25, 95%, 14 df), Holm across the ten
  secondary family  svc_trained minus svc_input, the same scope, Holm across the ten separately
  interpretation    fixed in the protocol: training improves the representation itself when at least one primary contrast
                    is Holm-significant in favour of the trained rankings AND the trained rankings have the higher mean on
                    most (at least six) of the ten; otherwise training improves the nearest-neighbour neighbourhoods but not
                    the representation read by a strong fixed readout. The secondary family has its own fixed statement.
  descriptive       the seven development datasets under both contrasts, labelled "screened before this protocol"; the SVC
                    against the kNN readout on each representation; the two contrasts under the kNN readout; balanced
                    accuracy and macro-F1; the metrics of every arm; the readout settings. Intervals without p values.
Outputs (all or none; an existing file with different content is never replaced): representation_primary_family.csv,
representation_secondary_family.csv, representation_development.csv, representation_readout_comparison.csv,
representation_knn_contrasts.csv, representation_other_metrics.csv, representation_metrics.csv,
representation_readout_settings.csv and representation_analysis.json (every result, the declaration, the interpretation,
the sha256 of each CSV and provenance).
"""
import argparse
import hashlib
import json
from pathlib import Path
from . import representation_test as rt
from .compare_runs import RUN_RECORDS, SHARED_SOURCES, VALIDATORS, RunComparisonError, _reason, check_output, load_run, verify_run
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .newdata import sha256_file
from .referee_analyses import code_record, finish
from .run_bridge import fold_schedule

ANALYSIS_SOURCES = ('compare_representation.py', 'representation_test.py', 'training_diagnostics.py', 'run_knn_ablation.py',
                    'neighbour_baselines.py', 'knn_controls.py', 'compare_runs.py', 'referee_analyses.py', 'evaluation.py',
                    'reporting.py', 'run_revision.py', 'newdata.py')
SEALED_SOURCES = SHARED_SOURCES
RUN_FILES = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json')
INTERVAL = ('mean_difference', 'standard_error', 'ci_low', 'ci_high')
FAMILY_COLUMNS = ('family', 'dataset', 'panel', 'contrast', 'model_a', 'model_b', 'metric', *INTERVAL, 'p_approximate',
                  'holm_p_approximate', 'holm_significant', 'n_folds', 'df', 'test_train_ratio', 'mean_a', 'mean_b')
DESCRIPTIVE_COLUMNS = ('table', 'dataset', 'panel', 'label', 'contrast', 'model_a', 'model_b', 'metric', *INTERVAL, 'n_folds',
                       'df', 'mean_a', 'mean_b', 'status')
METRIC_COLUMNS = ('dataset', 'panel', 'label', 'arm', 'readout', 'representation', 'metric', 'mean', 'outer_fold_sd',
                  'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')
SETTINGS_COLUMNS = ('dataset', 'panel', 'arm', 'views', 'mean_selection_score', 'settings', 'libsvm_fit_status_nonzero',
                    'fit_warnings', 'selection_warnings', 'mean_support_vectors', 'vocabulary')
OUTPUTS = ('representation_primary_family.csv', 'representation_secondary_family.csv', 'representation_development.csv',
           'representation_readout_comparison.csv', 'representation_knn_contrasts.csv', 'representation_other_metrics.csv',
           'representation_metrics.csv', 'representation_readout_settings.csv', 'representation_analysis.json')
DEVELOPMENT_LABEL = 'screened before this protocol'
FURTHER_LABEL = 'never screened'
DESCRIPTIVE_STATUS = 'descriptive: corrected resampled t interval without a p value and without multiplicity adjustment'


def completeness_gate(run_source):
    """Refuse unless the representation-test run is complete. Only the planned job list is parsed, so nothing holding a
    score is read before the run is complete."""
    path, problems = Path(run_source), []
    if not path.is_dir():
        raise RunComparisonError(f'representation-test run {path}: no such directory')
    missing = [name for name in (*RUN_FILES, rt.SUMMARY_JSON, rt.SUMMARY_CSV) if not (path/name).is_file()]
    absent, first = 0, None
    if (path/'planned_jobs.json').is_file():
        try:
            for job in json.loads((path/'planned_jobs.json').read_text()):
                for relative in (f"results/{job['stem']}.json", f"logs/{job['stem']}.jsonl",
                                 f"predictions/{job['stem']}.jsonl", f"artifacts/{job['stem']}.npz"):
                    if not (path/relative).is_file():
                        absent, first = absent + 1, first or relative
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f'unreadable planned_jobs.json ({_reason(exc)})')
    if missing:
        problems.append('missing ' + ', '.join(missing))
    if absent:
        problems.append(f'{absent} planned job files missing (first: {first})')
    if problems:
        raise RunComparisonError('The analysis runs only after the representation-test run is complete, and reads nothing '
                                 f'before: {path} is not complete: ' + '; '.join(problems))


def _sha256_of(protocol):
    return hashlib.sha256((json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()).hexdigest()


def check_protocol(p, *, smoke):
    """The run holds the committed frozen representation-test protocol with the declared analysis and panels."""
    if not smoke:
        try:
            rt.validate_protocol(p)
        except ValueError as exc:
            raise RunComparisonError(f'The run protocol is not a valid representation-test protocol: {exc}') from exc
        if not p.get('frozen'):
            raise RunComparisonError('The representation-test run protocol is not frozen')
        if not rt.PROTOCOL.is_file() or sha256_file(rt.PROTOCOL) != _sha256_of(p):
            raise RunComparisonError(f'The run protocol is not the committed frozen {rt.PROTOCOL.name}')
        if (list(p['further_datasets']) != list(rt.FURTHER) or list(p['development_datasets']) != list(rt.DEVELOPMENT)):
            raise RunComparisonError('The run does not declare the ten further and the seven development datasets')
    if p.get('production_family') != rt.FAMILY or list(p.get('arms', ())) != list(rt.ARMS):
        raise RunComparisonError('The run does not hold the representation-test family and its six arms')
    if (p['test_train_ratio'], p['confidence']) != (0.25, 0.95):
        raise RunComparisonError('This analysis is defined for q = 0.25 and 95% intervals')
    block = p['analysis']
    primary, secondary = block['primary_family'], block['secondary_family']
    if ((primary['model_a'], primary['model_b']) != rt.PRIMARY or (secondary['model_a'], secondary['model_b']) != rt.SECONDARY
            or list(primary['datasets']) != list(p['further_datasets'])
            or list(secondary['datasets']) != list(p['further_datasets'])):
        raise RunComparisonError('The declared families do not match the contrasts and the further panel of this tree')
    if block != rt.analysis_declaration(p['development_datasets'], p['further_datasets']):
        raise RunComparisonError('The declared analysis differs from the one this tree computes')
    return block


def verify_references(p, manifest, sources, *, smoke):
    """Every reference production run re-verified with compare_runs.load_run and compare_runs.verify_run, and the sealed
    file hashes of the prepared manifest re-checked."""
    provenance = {}
    here = rt.environment()['source_hashes']
    for name, entry in sorted(manifest['references'].items()):
        directory = Path(sources[name][0]) if name in sources else Path(entry['run_directory'])
        if str(directory.resolve()) != entry['run_directory']:
            raise RunComparisonError(f'reference {name}: {directory} is not the prepared reference run '
                                     f'{entry["run_directory"]}')
        missing = [record for record in RUN_RECORDS if not (directory/record).is_file()]
        if missing:
            raise RunComparisonError(f'reference {name} run {directory} is incomplete: missing {", ".join(missing)}')
        run = load_run(directory, name)
        verified, _ = verify_run(run)
        sealed = run.environment.get('source_hashes') or {}
        if not smoke:
            if entry['run']['summary_sha256'] != sha256_file(directory/'summary.json'):
                raise RunComparisonError(f'reference {name}: summary.json differs from the prepared pin')
            absent = [source for source in SEALED_SOURCES if source not in sealed]
            if absent:
                raise RunComparisonError(f'reference {name}: the run environment must seal ' + ', '.join(absent))
        provenance[name] = {'directory': str(directory.resolve()), 'protocol_id': run.protocol.get('protocol_id'),
                            'protocol_sha256': run.protocol_sha256, 'code_revision': run.environment.get('code_revision'),
                            'jobs_verified': len(run.jobs), 'model_rows_verified': sum(map(len, verified.values())),
                            'datasets': list(entry['datasets']),
                            'shared_sources_identical_to_this_tree': sorted(
                                source for source in SEALED_SOURCES if sealed.get(source) == here.get(source)),
                            'shared_sources_differing_from_this_tree': sorted(
                                source for source in SEALED_SOURCES if source in sealed and sealed[source] != here.get(source))}
    return provenance


# ----------------------------------------------------------------------------- the declared analysis

def _label(panel):
    return DEVELOPMENT_LABEL if panel == 'development' else FURTHER_LABEL


def _panel(protocol, name):
    return rt.panel_of(protocol, name)


def _mean(report, dataset, arm, metric):
    return report['summaries'][dataset]['arms'][arm]['metrics'][metric]['mean']


def _interval(report, protocol, dataset, model_a, model_b, metric):
    seeds = list(protocol['fit_seeds'])
    return paired_corrected_interval(report['model_rows'][dataset], model_a, model_b, metric=metric,
                                     q=protocol['test_train_ratio'], confidence=protocol['confidence'],
                                     expected_folds=fold_schedule(protocol), expected_seeds={model_a: seeds, model_b: seeds})


def family_rows(report, protocol, datasets, model_a, model_b, *, family, metric='accuracy', alpha=rt.ALPHA):
    """One tested family: per dataset, model_a minus model_b with its corrected resampled t interval and two-sided p, Holm
    across the family's datasets."""
    rows = []
    for name in datasets:
        interval = _interval(report, protocol, name, model_a, model_b, metric)
        rows.append({'family': family, 'dataset': name, 'panel': _panel(protocol, name),
                     'contrast': f'{model_a} minus {model_b}, {metric}; positive favours {model_a}',
                     'model_a': model_a, 'model_b': model_b, 'metric': metric,
                     **{field: interval[field] for field in INTERVAL}, 'p_approximate': interval['p_approximate'],
                     'n_folds': interval['n_folds'], 'df': interval['df'], 'test_train_ratio': interval['test_train_ratio'],
                     'mean_a': _mean(report, name, model_a, metric), 'mean_b': _mean(report, name, model_b, metric)})
    for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
        row['holm_p_approximate'] = adjusted
        row['holm_significant'] = bool(adjusted < alpha)
    return rows


def interpretation(rows, *, alpha=rt.ALPHA, most=None):
    """The protocol's fixed rule on one family: met when at least one contrast is Holm-significant with a positive mean
    difference AND the mean difference is positive on most (len // 2 + 1) of the datasets."""
    most = len(rows) // 2 + 1 if most is None else most
    positive = [row['dataset'] for row in rows if row['holm_p_approximate'] < alpha and row['mean_difference'] > 0]
    negative = [row['dataset'] for row in rows if row['holm_p_approximate'] < alpha and row['mean_difference'] < 0]
    higher = [row['dataset'] for row in rows if row['mean_difference'] > 0]
    return {'met': bool(positive) and len(higher) >= most, 'holm_significant_positive': positive,
            'holm_significant_negative': negative, 'higher_mean': higher, 'higher_mean_count': len(higher),
            'most_threshold': most, 'datasets': len(rows), 'alpha': alpha}


def descriptive_rows(report, protocol, datasets, pairs, *, table, metrics=('accuracy',)):
    rows = []
    for metric in metrics:
        for model_a, model_b in pairs:
            for name in datasets:
                interval = _interval(report, protocol, name, model_a, model_b, metric)
                panel = _panel(protocol, name)
                rows.append({'table': table, 'dataset': name, 'panel': panel, 'label': _label(panel),
                             'contrast': f'{model_a} minus {model_b}, {metric}; positive favours {model_a}',
                             'model_a': model_a, 'model_b': model_b, 'metric': metric,
                             **{field: interval[field] for field in INTERVAL}, 'n_folds': interval['n_folds'],
                             'df': interval['df'], 'mean_a': _mean(report, name, model_a, metric),
                             'mean_b': _mean(report, name, model_b, metric), 'status': DESCRIPTIVE_STATUS})
    return rows


def metric_rows(report, protocol):
    rows = []
    for name in protocol['datasets']:
        panel = _panel(protocol, name)
        for arm in rt.ARMS:
            readout, representation = rt.arm_parts(arm)
            for metric in rt.METRICS:
                entry = report['summaries'][name]['arms'][arm]['metrics'][metric]
                rows.append({'dataset': name, 'panel': panel, 'label': _label(panel), 'arm': arm, 'readout': readout,
                             'representation': representation, 'metric': metric,
                             **{key: entry[key] for key in ('mean', 'outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds',
                                                            'seeds_per_fold')}})
    return rows


def settings_rows(report, protocol):
    rows = []
    for name in protocol['datasets']:
        for arm in rt.ARMS:
            entry = report['summaries'][name]['readout_settings'][arm]
            settings = entry.get('selected_C', entry.get('selected_setting'))
            rows.append({'dataset': name, 'panel': _panel(protocol, name), 'arm': arm, 'views': entry['views'],
                         'mean_selection_score': entry['mean_selection_score'],
                         'settings': canonical_json(settings),
                         'libsvm_fit_status_nonzero': entry.get('libsvm_fit_status_nonzero'),
                         'fit_warnings': entry.get('fit_warnings'), 'selection_warnings': entry.get('selection_warnings'),
                         'mean_support_vectors': entry.get('mean_support_vectors'),
                         'vocabulary': canonical_json(entry.get('vocabulary')) if 'vocabulary' in entry else None})
    return rows


def statements(primary, secondary):
    """The protocol's fixed statements for the observed families."""
    return {'primary': {**primary, 'statement': rt.STATEMENTS['met' if primary['met'] else 'not_met']},
            'secondary': {**secondary, 'statement': rt.SECONDARY_STATEMENTS['met' if secondary['met'] else 'not_met']}}


def analyse(run_source, output, sources=None, *, allow_smoke=False):
    check_output(output, OUTPUTS)                   # an unusable output location is refused before any run is read
    completeness_gate(run_source)
    run_source = Path(run_source)
    protocol = json.loads((run_source/'protocol.json').read_text())
    manifest = json.loads((run_source/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    block = check_protocol(protocol, smoke=smoke)
    references = verify_references(protocol, manifest, sources or {}, smoke=smoke)
    try:
        report, _ = rt.summary(run_source, allow_smoke=smoke)       # re-verifies every job, selection and stored hash
    except ValueError as exc:
        raise RunComparisonError(f'The representation-test run does not verify: {_reason(exc)}') from exc
    published = json.loads((run_source/rt.SUMMARY_JSON).read_text())
    if canonical_json(report) != canonical_json(published):
        raise RunComparisonError(f'The re-verified summary differs from the published {rt.SUMMARY_JSON}')
    if list(protocol['datasets']) != list(manifest['datasets']):
        raise RunComparisonError('The run does not hold every protocol dataset')
    further, development = list(protocol['further_datasets']), list(protocol['development_datasets'])
    everything = list(protocol['datasets'])
    try:
        primary = family_rows(report, protocol, further, *rt.PRIMARY, family='primary')
        secondary = family_rows(report, protocol, further, *rt.SECONDARY, family='secondary')
        outcome = statements(interpretation(primary, most=block['interpretation']['primary']['most_threshold']),
                             interpretation(secondary, most=block['interpretation']['secondary']['most_threshold']))
        tables = {
            'representation_primary_family.csv': (primary, FAMILY_COLUMNS),
            'representation_secondary_family.csv': (secondary, FAMILY_COLUMNS),
            'representation_development.csv': (descriptive_rows(report, protocol, development, (rt.PRIMARY, rt.SECONDARY),
                                                                table='development'), DESCRIPTIVE_COLUMNS),
            'representation_readout_comparison.csv': (descriptive_rows(
                report, protocol, everything, [(f'svc_{r}', f'knn_{r}') for r in rt.REPRESENTATIONS],
                table='readout_comparison'), DESCRIPTIVE_COLUMNS),
            'representation_knn_contrasts.csv': (descriptive_rows(
                report, protocol, everything, [('knn_trained', 'knn_untrained'), ('knn_trained', 'knn_input')],
                table='knn_contrasts'), DESCRIPTIVE_COLUMNS),
            'representation_other_metrics.csv': (descriptive_rows(
                report, protocol, everything, (rt.PRIMARY, rt.SECONDARY), table='other_metrics',
                metrics=('balanced_accuracy', 'macro_f1')), DESCRIPTIVE_COLUMNS),
            'representation_metrics.csv': (metric_rows(report, protocol), METRIC_COLUMNS),
            'representation_readout_settings.csv': (settings_rows(report, protocol), SETTINGS_COLUMNS)}
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified representation-test run does not support the prespecified analysis: '
                                 f'{_reason(exc)}') from exc
    jobs = json.loads((run_source/'planned_jobs.json').read_text())
    record = {
        'purpose': 'prespecified_analysis_of_the_readout_matched_representation_test',
        'status': 'computed after the representation-test run was complete and every reference run, every job, every sealed '
                  'selection and every stored prediction hash re-verified; the primary and secondary families are each '
                  'Holm-adjusted across the ten further datasets; everything else is descriptive',
        'analysis_declaration': block, 'screening_disclosure': protocol['screening_disclosure'],
        'interpretation': outcome,
        'primary_family': primary, 'secondary_family': secondary,
        'descriptive': {name.removeprefix('representation_').removesuffix('.csv'): tables[name][0]
                        for name in OUTPUTS[2:-1]},
        'reproduction': {name: entry['reproduction'] for name, entry in report['summaries'].items()},
        'job_checks': report['checks'],
        'provenance': {'run': {'directory': str(run_source.resolve()), 'protocol_id': protocol.get('protocol_id'),
                               'protocol_sha256': sha256_file(run_source/'protocol.json'),
                               'code_revision': report['code_revision'], 'planned_jobs': len(jobs),
                               'summary_sha256': sha256_file(run_source/rt.SUMMARY_JSON),
                               'purpose': manifest.get('purpose')},
                       'references': references,
                       'verification': {'validators': list(VALIDATORS) + ['representation_test.summary (every job, every '
                                                                          'sealed selection and stored prediction hash '
                                                                          're-derived, every check)'],
                                        'arms': list(rt.ARMS)},
                       **code_record(ANALYSIS_SOURCES)}}
    record = rt._plain(record)
    finish(output, {name: (rt._plain(rows), columns) for name, (rows, columns) in tables.items()},
           'representation_analysis.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('analyse', help='the prespecified analysis of one complete representation-test run')
    sub.add_argument('--run', type=Path, required=True, help='the representation-test run directory (summary written)')
    sub.add_argument('--output', type=Path, required=True, help='new output directory')
    sub.add_argument('--reference', type=rt.parse_reference, action='append', metavar='NAME=RUN_DIR,ABLATION_DIR')
    args = parser.parse_args(argv)
    sources = dict(args.reference or [])
    try:
        record = analyse(args.run, args.output, sources)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_representation analyse refused: {exc}\n')
    for family in ('primary_family', 'secondary_family'):
        for row in record[family]:
            print(f"{row['family']} {row['dataset']}: {row['model_a']} - {row['model_b']} accuracy "
                  f"{row['mean_difference']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
                  f"Holm p={row['holm_p_approximate']:.3g}")
    for family in ('primary', 'secondary'):
        entry = record['interpretation'][family]
        print(f"INTERPRETATION {family}: {'met' if entry['met'] else 'not met'} (Holm-significant in favour: "
              f"{len(entry['holm_significant_positive'])}; higher mean on {entry['higher_mean_count']} of "
              f"{entry['datasets']}, needed {entry['most_threshold']}): the paper states that {entry['statement']}")


if __name__ == '__main__':
    main()
