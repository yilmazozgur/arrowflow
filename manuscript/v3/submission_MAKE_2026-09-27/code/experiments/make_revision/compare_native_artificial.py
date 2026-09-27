"""The prespecified analysis of the native_artificial family (the author's correction of 2026-09-15).

It refuses (exit 2, nothing written) until the run is complete, then re-verifies the run AND the encoded comparison run
runs/2026-09-14-artificial with compare_runs.load_run and verify_run before any score is read. Nothing here is computed on
an unverified record, and every output is written all or none.

Families, adjusted separately and each declared in the frozen protocol before any outer score existed:
    primary    native_arrowflow_knn minus native_arrowflow_knn_untrained (the training effect), accuracy, fitting seeds
               averaged within each outer fold, corrected resampled t (q = 0.25, 95%, df 14), Holm over ranks8 and ranks16
    secondary  native_arrowflow_knn minus native_footrule_knn, the same, adjusted separately
Descriptive: the main table (the native arm beside the REUSED classical comparators and majority class of the encoded run),
comparator intervals, the competitiveness flags under holistic.RULES, the ladder, complete metrics, the selected widths, the
completion-artifact data properties, and the PAIRED contrast against the encoded arm. ranks8_original belongs to no family.

python -m experiments.make_revision.compare_native_artificial analyse --run <run> --output <directory> [--encoded <run>]
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import native_artificial as na
from .compare_runs import (RunComparisonError, VALIDATORS, _json_text, _reason, check_output, load_run, sha256_file,
                           verify_run, write_outputs, RUN_RECORDS)
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .holistic import CLASSICAL, MAJORITY, RULES, TRAINED as ENCODED_TRAINED, competitiveness
from .knn_controls import selected_widths
from .referee_analyses import code_record, finish
from .reporting import METRICS
from .run_revision import load_prepared

ANALYSIS_SOURCES = ('compare_native_artificial.py', 'native_artificial.py', 'compare_runs.py', 'evaluation.py',
                    'reporting.py', 'run_revision.py', 'holistic.py')
SEALED_SOURCES = ('arrowflow/arrowflow.py', 'arrowflow/benchmark.py', 'arrowflow/config.py', 'arrowflow/ranking.py',
                  'experiments/make_revision/models.py', 'experiments/make_revision/multiview.py',
                  'experiments/make_revision/comparisons.py', 'experiments/make_revision/artificial_ranks.py',
                  'experiments/make_revision/native_artificial.py')
RECORD_NAME = 'native_artificial_analysis.json'
INTERVAL = ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'n_folds', 'df')
FAMILY_COLUMNS = ('family', 'family_index', 'dataset', 'role', 'contrast', 'model_a', 'model_b', *INTERVAL[:4],
                  'p_approximate', 'holm_p_approximate', *INTERVAL[4:])
MAIN_COLUMNS = ('dataset', 'role', 'model_id', 'source_run', 'input', 'mean_error', 'outer_fold_sd',
                'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')
COMPARATOR_COLUMNS = ('status', 'dataset', 'role', 'model_a', 'model_b', 'source_run_b', 'input_b', *INTERVAL[:4],
                      'p_unadjusted', *INTERVAL[4:], 'mean_error_a', 'mean_error_b', 'best_comparator')
COMPETITIVENESS_COLUMNS = ('status', 'dataset', 'role', 'arrowflow_error', 'best_classical', 'best_classical_error',
                           'gap_points', 'gap_points_2dp', 'gap_points_from_1dp_errors', 'within_three_points',
                           'within_three_points_as_printed', 'printed_precision_agrees', 'best_on', 'majority_class_error',
                           'majority_distance_points', 'ceiling', 'near_majority', 'no_learning', 'degenerate')
LADDER_COLUMNS = ('dataset', 'role', 'rung', 'model_id', 'mean_error', 'outer_fold_sd', 'mean_within_fold_seed_sd',
                  'n_folds', 'seeds_per_fold')
METRIC_COLUMNS = ('dataset', 'role', 'model_id', *(f'{metric}_{field}' for metric in METRICS
                                                   for field in ('mean', 'outer_fold_sd', 'mean_within_fold_seed_sd')))
WIDTH_COLUMNS = ('dataset', 'role', 'outer_repeat', 'outer_fold', 'config_id', 'widths', 'hidden_layers')
ENCODED_COLUMNS = ('status', 'dataset', 'role', 'contrast', 'model_a', 'run_a', 'model_b', 'run_b', *INTERVAL[:4],
                   'p_unadjusted', *INTERVAL[4:], 'mean_error_a', 'mean_error_b')
PROBE_COLUMNS = ('status', 'dataset', 'role', 'probe', 'definition', 'mean_accuracy', 'outer_fold_sd', 'distinct_keys',
                 'above_majority_points', 'n_folds')
OUTPUTS = ('native_artificial_families.csv', 'native_artificial_main_table.csv', 'native_artificial_comparators.csv',
           'native_artificial_competitiveness.csv', 'native_artificial_ladder.csv',
           'native_artificial_complete_metrics.csv', 'native_artificial_selected_widths.csv',
           'native_artificial_encoded_contrast.csv', 'native_artificial_data_properties.csv', RECORD_NAME)
DESCRIPTIVE = 'descriptive; no multiplicity adjustment'
ENCODED_STATUS = ('descriptive; PAIRED on identical datasets, splits, outer folds and fitting seeds; the two arms differ by '
                  'the encoder alone, so the model differs and the contrast carries no family and no multiplicity adjustment')
NATIVE_INPUT_LABEL = 'completed ranking (native; no encoder)'
ENCODED_INPUT_LABEL = 'position features with a missing value per deleted item (fold-locally imputed)'
# The native model of each pair and the encoded-run model it is paired against; the encoder is the only difference.
ENCODED_PAIRS = ((na.TRAINED_MODEL, ENCODED_TRAINED, 'native_minus_encoded_arrowflow_knn'),
                 (na.UNTRAINED_MODEL, 'arrowflow_knn_untrained', 'native_minus_encoded_untrained'),
                 (na.INPUT_MODEL, 'input_footrule_knn', 'native_minus_encoded_input_knn'))


def completeness_gate(run_source):
    """Refuse unless the run is complete. Only the planned job list is parsed, so nothing holding a score is read first."""
    path = Path(run_source)
    if not path.is_dir():
        raise RunComparisonError(f'{na.FAMILY} run {path}: no such directory')
    missing = [name for name in RUN_RECORDS if not (path/name).is_file()]
    absent, first = 0, None
    if (path/'planned_jobs.json').is_file():
        try:
            for job in json.loads((path/'planned_jobs.json').read_text()):
                for relative in (job['result_file'], job['log_file']):
                    if not (path/relative).is_file():
                        absent, first = absent + 1, first or relative
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RunComparisonError(f'{na.FAMILY} run {path}: unreadable planned_jobs.json ({_reason(exc)})') from exc
    if missing or absent:
        parts = ([f'missing {", ".join(missing)}'] if missing else []) + (
            [f'{absent} planned job files missing (first: {first})'] if absent else [])
        raise RunComparisonError('The analysis runs only after the run is complete, and reads nothing before: '
                                 f'{path} is not complete: ' + '; '.join(parts))


def check_run(run, *, smoke):
    """The run holds the committed frozen native protocol, the four models with the candidates of this tree, the protocol
    registry, the sealed sources and the pinned dataset and splits hashes."""
    protocol = run.protocol
    try:
        na.validate_native_protocol(protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunComparisonError(f'The run protocol is not the {na.FAMILY} protocol: {_reason(exc)}') from exc
    if (protocol.get('purpose') == 'synthetic_smoke_only') != smoke:
        raise RunComparisonError('A synthetic smoke run is analysed only by the smoke, and a production run only in production')
    if not smoke:
        if not na.PROTOCOL_FILE.is_file() or run.protocol_sha256 != sha256_file(na.PROTOCOL_FILE):
            raise RunComparisonError(f'The run protocol.json (sha256 {run.protocol_sha256}) is not the committed frozen '
                                     f'{na.PROTOCOL_FILE.name}')
        declared = (protocol.get('test_train_ratio'), protocol.get('confidence'), len(run.schedule['expected_folds']))
        if declared != (.25, .95, 15):
            raise RunComparisonError(f'This analysis is defined for the registered nested design (0.25, 0.95, 15), not {declared}')
        if canonical_json(run.candidates) != canonical_json(na.candidate_record(na.build_registry(protocol))):
            raise RunComparisonError('The run candidates differ from the native_artificial registry of this tree')
    if list(run.registry) != list(na.MODEL_ORDER):
        raise RunComparisonError(f'The run must hold {list(na.MODEL_ORDER)}, not {list(run.registry)}')
    if run.environment.get('registry') != protocol['registry']:
        raise RunComparisonError(f'The run environment must name the protocol registry {protocol["registry"]}')
    sealed = run.environment.get('source_hashes') or {}
    absent = [source for source in SEALED_SOURCES if source not in sealed]
    if absent:
        raise RunComparisonError('The run environment must seal ' + ', '.join(absent))
    entries, datasets = na.panel_by_name(protocol), {}
    for name in protocol['datasets']:
        manifest = run.manifests[name]
        datasets[name] = {'dataset_hash': manifest.get('dataset_hash'), 'splits_hash': manifest.get('splits_hash')}
        if not smoke and datasets[name] != {key: entries[name].get(key) for key in ('dataset_hash', 'splits_hash')}:
            raise RunComparisonError(f'The run {name} dataset or splits hash differs from its pin')
        audit = (manifest.get('native_input') or {}).get('audit') or {}
        failed = [key for key in ('every_row_is_a_permutation_of_the_V_items',
                                  'dropping_the_tail_recovers_the_observed_sequence',
                                  'the_tail_is_exactly_the_missing_items', 'the_tail_is_in_ascending_item_order')
                  if not audit.get(key)]
        if failed:
            raise RunComparisonError(f'The run {name} manifest does not record a reversible completion ({", ".join(failed)})')
    return {'protocol_id': protocol['protocol_id'], 'protocol_sha256': run.protocol_sha256,
            'code_revision': run.environment.get('code_revision'), 'registry': protocol['registry'],
            'sealed_sources': sealed, 'datasets': datasets, 'fit_seeds': protocol['fit_seeds'], 'pins_checked': not smoke}


def check_pairing(run, encoded):
    """The two arms must share the datasets, the splits, the outer folds and the fitting seeds; only then is a fold-by-fold
    contrast between them paired."""
    problems = []
    if list(run.protocol['datasets']) != list(encoded.protocol['datasets']):
        problems.append(f'datasets {run.protocol["datasets"]} against {encoded.protocol["datasets"]}')
    for key in ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds', 'selection_metric',
                'test_train_ratio', 'confidence'):
        if canonical_json(run.protocol.get(key)) != canonical_json(encoded.protocol.get(key)):
            problems.append(f'{key} {run.protocol.get(key)!r} against {encoded.protocol.get(key)!r}')
    for name in run.protocol['datasets']:
        for key in ('dataset_hash', 'splits_hash'):
            if run.manifests[name].get(key) != encoded.manifests[name].get(key):
                problems.append(f'{name} {key}')
    if problems:
        raise RunComparisonError('The native and encoded runs are not paired: ' + '; '.join(problems))
    return {'datasets': list(run.protocol['datasets']), 'shared_design_keys':
            ['outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds', 'selection_metric',
             'test_train_ratio', 'confidence'],
            'identical_dataset_and_splits_hashes': True,
            'difference': 'the encoder alone: the encoded arm standardizes, expands and projects the position features and '
                          'argsorts the projected scores; the native arm feeds the completed ranking itself'}


def _rows(run, name, model):
    return [row for row in run.summary['model_rows'][name] if row['model_id'] == model]


def _summary(run, name, model, metric):
    return next(row for row in run.summary['summaries'][name] if (row['model_id'], row['metric']) == (model, metric))


def _interval(run_a, model_a, run_b, model_b, name, q, confidence):
    seeds = {model_a: run_a.schedule['expected_seeds'][model_a], model_b: run_b.schedule['expected_seeds'][model_b]}
    if model_a == model_b:
        raise RunComparisonError('A paired contrast needs two distinct model IDs')
    return paired_corrected_interval(_rows(run_a, name, model_a) + _rows(run_b, name, model_b), model_a, model_b,
                                     metric='accuracy', q=q, confidence=confidence,
                                     expected_folds=run_a.schedule['expected_folds'], expected_seeds=seeds)


def family_rows(run, block, label, roles, q, confidence):
    rows = []
    for index, name in enumerate(block['datasets'], start=1):
        interval = _interval(run, block['model_a'], run, block['model_b'], name, q, confidence)
        rows.append({'family': label, 'family_index': index, 'dataset': name, 'role': roles[name],
                     'contrast': block['contrast'], 'model_a': block['model_a'], 'model_b': block['model_b'],
                     'metric': 'accuracy', 'confidence': confidence, **interval,
                     'p_approximate': interval['p_approximate']})
    for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
        row['holm_p_approximate'] = adjusted
    return rows


def main_table_rows(run, encoded, names, roles):
    rows = []
    for name in names:
        for model in (na.TRAINED_MODEL, na.FILLED_MODEL):
            entry = _summary(run, name, model, 'error')
            rows.append({'dataset': name, 'role': roles[name], 'model_id': model, 'source_run': run.label,
                         'input': NATIVE_INPUT_LABEL, 'mean_error': entry['mean'], 'outer_fold_sd': entry['outer_fold_sd'],
                         'mean_within_fold_seed_sd': entry['mean_within_fold_seed_sd'], 'n_folds': entry['n_folds'],
                         'seeds_per_fold': entry['seeds_per_fold']})
        if encoded is None:
            continue
        for model in (*na.REUSED_COMPARATORS, na.MAJORITY_MODEL):
            entry = _summary(encoded, name, model, 'error')
            rows.append({'dataset': name, 'role': roles[name], 'model_id': model, 'source_run': encoded.label,
                         'input': ENCODED_INPUT_LABEL, 'mean_error': entry['mean'], 'outer_fold_sd': entry['outer_fold_sd'],
                         'mean_within_fold_seed_sd': entry['mean_within_fold_seed_sd'], 'n_folds': entry['n_folds'],
                         'seeds_per_fold': entry['seeds_per_fold']})
    return rows


def comparator_rows(run, encoded, names, roles, q, confidence):
    rows = []
    for name in names:
        here = []
        for model in (na.FILLED_MODEL, *na.REUSED_COMPARATORS, na.MAJORITY_MODEL):
            other, label, kind = ((run, run.label, NATIVE_INPUT_LABEL) if model == na.FILLED_MODEL
                                  else (encoded, encoded.label if encoded else None, ENCODED_INPUT_LABEL))
            if other is None:
                continue
            interval = _interval(run, na.TRAINED_MODEL, other, model, name, q, confidence)
            here.append({'status': DESCRIPTIVE + ('' if other is run else '; the comparator is REUSED from the encoded run, '
                                                                          'not refitted; the interval is still paired, '
                                                                          'because the folds and seeds are identical'),
                         'dataset': name, 'role': roles[name], 'model_a': na.TRAINED_MODEL, 'model_b': model,
                         'source_run_b': label, 'input_b': kind, 'metric': 'accuracy', 'confidence': confidence, **interval,
                         'p_unadjusted': interval['p_approximate'],
                         'mean_error_a': _summary(run, name, na.TRAINED_MODEL, 'error')['mean'],
                         'mean_error_b': _summary(other, name, model, 'error')['mean']})
        best = min((row['mean_error_b'] for row in here), default=None)
        for row in here:
            row['best_comparator'] = best is not None and row['mean_error_b'] == best
        rows.extend(here)
    return rows


def competitiveness_rows(run, encoded, names, roles):
    rows = []
    for name in names:
        errors = {ENCODED_TRAINED: _summary(run, name, na.TRAINED_MODEL, 'error')['mean'],
                  MAJORITY: _summary(encoded, name, na.MAJORITY_MODEL, 'error')['mean']}
        errors.update({model: _summary(encoded, name, model, 'error')['mean'] for model in CLASSICAL})
        result = competitiveness(errors)
        rows.append({'status': 'descriptive; fixed rules (holistic.RULES); ArrowFlow is the NATIVE arm and every classical '
                               'model and the majority class are REUSED from the encoded run',
                     'dataset': name, 'role': roles[name], **result,
                     'best_classical': '+'.join(result['best_classical'])})
    return rows


def ladder_rows(run, names, roles):
    rows = []
    for name in names:
        for rung, model in na.LADDER:
            entry = _summary(run, name, model, 'error')
            rows.append({'dataset': name, 'role': roles[name], 'rung': rung, 'model_id': model,
                         'mean_error': entry['mean'], 'outer_fold_sd': entry['outer_fold_sd'],
                         'mean_within_fold_seed_sd': entry['mean_within_fold_seed_sd'], 'n_folds': entry['n_folds'],
                         'seeds_per_fold': entry['seeds_per_fold']})
    return rows


def complete_metric_rows(run, names, roles):
    rows = []
    for name in names:
        for model in run.registry:
            row = {'dataset': name, 'role': roles[name], 'model_id': model}
            for metric in METRICS:
                entry = _summary(run, name, model, metric)
                row[f'{metric}_mean'] = entry['mean']
                row[f'{metric}_outer_fold_sd'] = entry['outer_fold_sd']
                row[f'{metric}_mean_within_fold_seed_sd'] = entry['mean_within_fold_seed_sd']
            rows.append(row)
    return rows


def width_rows(run, verified, names, roles):
    rows, counts = [], {}
    for name in names:
        widths = selected_widths(verified[name], na.TRAINED_MODEL)
        configs = {(row['outer_repeat'], row['outer_fold']): row['config_id'] for row in verified[name]
                   if row['model_id'] == na.TRAINED_MODEL}
        counts[name] = {}
        for (repeat, fold), value in sorted(widths.items()):
            rows.append({'dataset': name, 'role': roles[name], 'outer_repeat': repeat, 'outer_fold': fold,
                         'config_id': configs[repeat, fold], 'widths': '+'.join(map(str, value)),
                         'hidden_layers': len(value)})
            key = '+'.join(map(str, value))
            counts[name][key] = counts[name].get(key, 0) + 1
    return rows, counts


def encoded_contrast_rows(run, encoded, names, roles, q, confidence):
    rows = []
    for name in names:
        for native_model, encoded_model, contrast in ENCODED_PAIRS:
            if encoded_model not in encoded.registry:
                raise RunComparisonError(f'The encoded run does not hold {encoded_model}')
            interval = _interval(run, native_model, encoded, encoded_model, name, q, confidence)
            rows.append({'status': ENCODED_STATUS, 'dataset': name, 'role': roles[name], 'contrast': contrast,
                         'model_a': native_model, 'run_a': run.label, 'model_b': encoded_model, 'run_b': encoded.label,
                         'metric': 'accuracy', 'confidence': confidence, **interval,
                         'p_unadjusted': interval['p_approximate'],
                         'mean_error_a': _summary(run, name, native_model, 'error')['mean'],
                         'mean_error_b': _summary(encoded, name, encoded_model, 'error')['mean']})
    return rows


def data_property_rows(run, names, roles):
    """The completion audit and the probe accuracies recomputed from the run's own verified prepared data."""
    rows, records = [], {}
    for name in names:
        X, y, _, splits = load_prepared(run.path, name)
        audit = na.check_completion(name, X)
        probes = {}
        for probe, definition in na.PROBES.items():
            accuracies = na.probe_accuracies(X, y, splits, probe)
            probes[probe] = {'definition': definition, 'mean_accuracy': float(np.mean(accuracies)),
                             'outer_fold_sd': float(np.std(accuracies, ddof=1)), 'n_folds': len(accuracies),
                             'distinct_keys': int(len(set(map(str, na.probe_keys(X, probe))))),
                             'fold_accuracies': accuracies}
        for probe in ('missing_count', 'missing_items'):
            probes[probe]['above_majority_points'] = round(100 * (probes[probe]['mean_accuracy']
                                                                  - probes['majority_class']['mean_accuracy']), 9)
        records[name] = {'role': roles[name], 'completion_audit': audit, 'probes': probes}
        for probe, entry in probes.items():
            rows.append({'status': na.PROBE_STATUS, 'dataset': name, 'role': roles[name], 'probe': probe,
                         **{key: entry[key] for key in ('definition', 'mean_accuracy', 'outer_fold_sd', 'distinct_keys',
                                                        'n_folds')},
                         'above_majority_points': entry.get('above_majority_points')})
    return rows, records


def analyse(run_source, output, *, encoded_source=None, allow_smoke=False):
    check_output(output, OUTPUTS)                   # an unusable output location is refused before any run is read
    completeness_gate(run_source)
    run = load_run(run_source, na.FAMILY)
    smoke = allow_smoke and run.protocol.get('purpose') == 'synthetic_smoke_only'
    pairing_run = check_run(run, smoke=smoke)
    verified, _ = verify_run(run)
    encoded, encoded_verified, pairing = None, None, None
    if not smoke:
        encoded, encoded_rows = na.check_encoded_run(encoded_source)
        encoded_verified = sum(map(len, encoded_rows.values()))
        pairing = check_pairing(run, encoded)
    protocol = run.protocol
    block = protocol['analysis']
    names, members = list(block['datasets']), list(block['family_datasets'])
    roles = {entry['name']: entry['role'] for entry in protocol['panel']}
    if names != list(protocol['datasets']) or members != [name for name in names if roles[name] == 'family']:
        raise RunComparisonError('The analysis block does not declare the protocol datasets and their family members')
    q, confidence = protocol['test_train_ratio'], protocol['confidence']
    try:
        families = {}
        for label in ('primary', 'secondary'):
            declared = block[f'{label}_family']
            if declared['datasets'] != members or declared['size'] != len(members):
                raise ValueError(f'the {label} family does not declare the family datasets')
            families[label] = family_rows(run, declared, label, roles, q, confidence)
        main = main_table_rows(run, encoded, names, roles)
        comparators = comparator_rows(run, encoded, names, roles, q, confidence) if encoded else []
        competitive = competitiveness_rows(run, encoded, names, roles) if encoded else []
        ladder = ladder_rows(run, names, roles)
        metrics = complete_metric_rows(run, names, roles)
        widths, width_counts = width_rows(run, verified, names, roles)
        contrast = encoded_contrast_rows(run, encoded, names, roles, q, confidence) if encoded else []
        probes, probe_records = data_property_rows(run, names, roles)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified {na.FAMILY} run does not support the prespecified analysis: '
                                 f'{_reason(exc)}') from exc
    tables = {'native_artificial_families.csv': (families['primary'] + families['secondary'], FAMILY_COLUMNS),
              'native_artificial_main_table.csv': (main, MAIN_COLUMNS),
              'native_artificial_comparators.csv': (comparators, COMPARATOR_COLUMNS),
              'native_artificial_competitiveness.csv': (competitive, COMPETITIVENESS_COLUMNS),
              'native_artificial_ladder.csv': (ladder, LADDER_COLUMNS),
              'native_artificial_complete_metrics.csv': (metrics, METRIC_COLUMNS),
              'native_artificial_selected_widths.csv': (widths, WIDTH_COLUMNS),
              'native_artificial_encoded_contrast.csv': (contrast, ENCODED_COLUMNS),
              'native_artificial_data_properties.csv': (probes, PROBE_COLUMNS)}
    record = {
        'purpose': 'prespecified_analysis_of_the_native_input_artificial_family',
        'status': 'computed after the run was complete and re-verified, and after the encoded comparison run was re-verified; '
                  'each family Holm-adjusted within itself; the main table, comparator intervals, competitiveness flags, '
                  'ladder, complete metrics, selected widths, data properties and the encoded contrast descriptive',
        'runs_purpose': protocol.get('purpose'), 'analysis_declaration': block,
        'views': na.VIEW_NOTE, 'native_input': protocol['native_input'], 'comparator_reuse': protocol['comparator_reuse'],
        'interpretation': block['interpretation'],
        'primary_family': families['primary'], 'secondary_family': families['secondary'], 'main_table': main,
        'comparator_intervals': comparators, 'competitiveness': competitive, 'competitiveness_rules': RULES,
        'ladder': ladder, 'complete_metrics': metrics,
        'selected_widths': {'rows': widths, 'counts': width_counts},
        'encoded_contrast': {'status': ENCODED_STATUS, 'rows': contrast,
                             'reading': block['encoded_contrast']['reading']},
        'data_properties': probe_records, 'notes': block['notes'],
        'provenance': {'run': run.provenance(), 'pairing_run': pairing_run,
                       'encoded_run': encoded.provenance() if encoded else None, 'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS), 'jobs_verified': len(run.jobs),
                                        'model_rows_verified': sum(map(len, verified.values())),
                                        'encoded_jobs_verified': len(encoded.jobs) if encoded else None,
                                        'encoded_model_rows_verified': encoded_verified},
                       **code_record(ANALYSIS_SOURCES)}}
    finish(output, tables, RECORD_NAME, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('analyse', help='the prespecified analysis of one complete native_artificial run')
    sub.add_argument('--run', type=Path, required=True, help='the family run directory (summary.json written)')
    sub.add_argument('--output', type=Path, required=True, help='new output directory')
    sub.add_argument('--encoded', type=Path, help='the encoded comparison run (default: the pinned path)')
    args = parser.parse_args(argv)
    try:
        record = analyse(args.run, args.output, encoded_source=args.encoded)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_native_artificial analyse refused: {exc}\n')
    for row in record['primary_family'] + record['secondary_family']:
        print(f"{row['family']} {row['dataset']}: {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
              f"Holm p={row['holm_p_approximate']:.3g}")
    for row in record['encoded_contrast']['rows']:
        print(f"{row['contrast']} {row['dataset']}: {row['model_a']} - {row['model_b']} accuracy "
              f"{row['mean_difference']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] (descriptive, paired)")
    for name, entry in record['data_properties'].items():
        probes = entry['probes']
        print(f"{name} data property: majority {probes['majority_class']['mean_accuracy']:.4f}, missing-count "
              f"{probes['missing_count']['mean_accuracy']:.4f}, missing-items "
              f"{probes['missing_items']['mean_accuracy']:.4f}")


if __name__ == '__main__':
    main()
