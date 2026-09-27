"""Task 23B: the prespecified analysis of the two newdata batch runs; nothing is read until both batches are complete.

python -m experiments.make_revision.compare_newdata pairing --batch1 B1 --batch2 B2
    before the jobs start: the two prepared directories are batch 1 and batch 2 of the frozen newdata protocols (the
    committed files), with identical candidates, sealed sources and pinned datasets; writes nothing
python -m experiments.make_revision.compare_newdata analyse --batch1 B1 --batch2 B2 --output O
    refuses (exit 2, nothing written) unless both batch runs are complete: protocol, environment, candidates, planned jobs
    and summary.json present and every planned result file and fit log present, checked before any record holding a
    score is read. Then both runs are loaded and re-verified with compare_runs.load_run and compare_runs.verify_run (the
    harness validators over every saved record), the pairing above is checked, and the protocol's analysis block is
    computed:
      primary family    per dataset, arrowflow_full_knn minus arrowflow_knn_untrained accuracy, fitting seeds averaged within
                        each outer fold, corrected resampled t (q 0.25, 95%, 14 df), Holm across the ten datasets
      secondary family  the same for arrowflow_full_knn minus input_footrule_knn, Holm across the ten datasets separately
      moderator test    exact one-sided permutation test over all 210 splits of the ten datasets into four and six that the
                        mean training effect in stratum H exceeds the mean in the other six; effect sizes per stratum
      Spearman          descriptive: frozen external kNN gap against the training effect over the nine datasets with
                        evidence, exact permutation p over all 9! orders
      ladder            per dataset, mean error, outer-fold SD and within-fold seed SD of raw numeric kNN, unsorted projected
                        kNN, encoded-ranking kNN, untrained ArrowFlow-kNN and ArrowFlow-kNN
      comparators       per dataset, arrowflow_full_knn minus each comparator, paired intervals without adjustment
      duplicates        per dataset, the exact-duplicate audit of the prepared features
Outputs (all or none; an existing file with different content is never replaced): newdata_families.csv,
newdata_comparators.csv, newdata_ladder.csv, newdata_duplicates.csv and newdata_analysis.json (every result, the notes,
the sha256 of each CSV, provenance and verification of both runs, and the analysis source hashes).
"""
import argparse
import hashlib
import json
from itertools import combinations, permutations
from pathlib import Path
import numpy as np
from scipy import stats
from .compare_runs import (RUN_RECORDS, SHARED_SOURCES, VALIDATORS, RunComparisonError, _csv_text, _json_text, _reason,
                           analysis_sources, check_output, load_prepared_run, load_run, verify_run, write_outputs)
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .newdata import (COMPARATORS, LADDER, MODEL_ORDER, PIN_BY_NAME, PROTOCOL_FILES, TIE_TOLERANCE, TRAINED_MODEL,
                      build_registry, candidate_record, sha256_file, validate_newdata_protocol)
from .referee_analyses import COUNT_FIELDS, duplicate_groups, training_duplicate_flags
from .run_revision import load_prepared

LABELS = ('batch1', 'batch2')
BATCH_FIELDS = ('protocol_id', 'batch', 'datasets')
SEALED_SOURCES = SHARED_SOURCES + ('experiments/make_revision/knn_controls.py', 'experiments/make_revision/projected_knn.py',
                                   'experiments/make_revision/newdata.py')
FAMILIES_CSV, COMPARATORS_CSV, LADDER_CSV = 'newdata_families.csv', 'newdata_comparators.csv', 'newdata_ladder.csv'
DUPLICATES_CSV, ANALYSIS_JSON = 'newdata_duplicates.csv', 'newdata_analysis.json'
OUTPUTS = (FAMILIES_CSV, COMPARATORS_CSV, LADDER_CSV, DUPLICATES_CSV, ANALYSIS_JSON)
ANALYSIS_SOURCES = ('compare_newdata.py', 'newdata.py', 'compare_runs.py', 'referee_analyses.py', 'evaluation.py', 'reporting.py',
                    'run_revision.py')
INTERVAL = ('mean_difference', 'standard_error', 'ci_low', 'ci_high', 'n_folds', 'df')
FAMILY_COLUMNS = ('family', 'family_index', 'dataset', 'stratum', 'batch', 'contrast', 'model_a', 'model_b', *INTERVAL[:4],
                  'p_approximate', 'holm_p_approximate', *INTERVAL[4:])
COMPARATOR_COLUMNS = ('status', 'dataset', 'stratum', 'batch', 'model_a', 'model_b', *INTERVAL[:4], 'p_unadjusted', *INTERVAL[4:],
                      'mean_error_a', 'mean_error_b', 'best_comparator')
LADDER_COLUMNS = ('dataset', 'stratum', 'batch', 'rung', 'model_id', 'mean_error', 'outer_fold_sd', 'mean_within_fold_seed_sd',
                  'n_folds', 'seeds_per_fold')
DUPLICATE_COLUMNS = ('status', 'dataset', 'batch', *COUNT_FIELDS, 'matches_pinned_counts', 'test_rows',
                     'test_rows_with_training_duplicate', 'share_with_training_duplicate', 'fold_share_min', 'fold_share_max')
DESCRIPTIVE = 'descriptive; no multiplicity adjustment'


# ----------------------------------------------------------------------------- statistics

def permutation_test(effects, names, h_names, tolerance=TIE_TOLERANCE):
    """Exact one-sided test that the mean effect of h_names exceeds the mean of the other names: every assignment of
    len(h_names) of the names to H, the effects held fixed; p is the share of assignments whose statistic is at least the
    observed one (the observed assignment included; within `tolerance` counts as at least)."""
    names, h_names = list(names), list(h_names)
    if len(set(names)) != len(names) or not h_names or len(set(h_names)) != len(h_names) or not set(h_names) < set(names):
        raise ValueError('H must be a nonempty proper subset of distinct dataset names')
    values = np.asarray([effects[name] for name in names], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Every effect must be finite')

    def statistic(members):
        mask = np.zeros(len(names), dtype=bool)
        mask[list(members)] = True
        return float(values[mask].mean() - values[~mask].mean())
    observed_members = tuple(sorted(names.index(name) for name in h_names))
    observed = statistic(observed_members)
    null = [(statistic(members), members) for members in combinations(range(len(names)), len(h_names))]
    at_least = sum(value >= observed - tolerance for value, _ in null)
    return {'statistic': observed, 'mean_h': float(np.mean([effects[n] for n in h_names])),
            'mean_c': float(np.mean([effects[n] for n in names if n not in h_names])), 'splits': len(null),
            'at_least_observed': at_least, 'p_one_sided': at_least / len(null),
            'larger_than_observed': sum(value > observed + tolerance for value, _ in null), 'tolerance': tolerance,
            'null_distribution': [{'statistic': value, 'h': [names[i] for i in members]}
                                  for value, members in sorted(null, key=lambda item: (-item[0], item[1]))]}


def spearman_exact(gaps, effects, tolerance=TIE_TOLERANCE):
    """Spearman's rho (Pearson correlation of average ranks) and exact permutation p values over every order of the
    effects against the fixed gaps: one-sided (rho* at least rho) and two-sided (|rho*| at least |rho|)."""
    gaps, effects = np.asarray(gaps, dtype=float), np.asarray(effects, dtype=float)
    if gaps.ndim != 1 or gaps.shape != effects.shape or len(gaps) < 3 or not (np.isfinite(gaps).all() and np.isfinite(effects).all()):
        raise ValueError('Expected at least three finite paired gaps and effects')
    x, y = stats.rankdata(gaps), stats.rankdata(effects)
    xc = x - x.mean()
    if not np.any(xc) or not np.any(y - y.mean()):
        return {'n': len(x), 'rho': None, 'p_one_sided': None, 'p_two_sided': None, 'permutations': factorial_count(len(x)),
                'note': 'rho undefined: every gap or every effect is tied'}
    orders = np.array(list(permutations(range(len(y)))), dtype=np.int16)
    ranks = y[orders]
    centred = ranks - ranks.mean(axis=1, keepdims=True)
    rhos = centred @ xc / np.sqrt(np.sum(xc ** 2) * np.sum(centred ** 2, axis=1))
    observed = float(xc @ (y - y.mean()) / np.sqrt(np.sum(xc ** 2) * np.sum((y - y.mean()) ** 2)))
    return {'n': len(x), 'rho': observed, 'permutations': len(orders),
            'p_one_sided': float(np.mean(rhos >= observed - tolerance)),
            'p_two_sided': float(np.mean(np.abs(rhos) >= abs(observed) - tolerance)), 'tolerance': tolerance,
            'gap_ranks': x.tolist(), 'effect_ranks': y.tolist()}


def factorial_count(n):
    return int(np.prod(np.arange(1, n + 1, dtype=np.int64)))


def stratum_summary(effects, strata):
    """Per stratum: n, mean, SD (ddof 1), median, min and max of the dataset effects, with every dataset effect."""
    summary = {}
    for label, names in strata.items():
        values = np.asarray([effects[name] for name in names], dtype=float)
        summary[label] = {'n': len(values), 'mean': float(values.mean()),
                          'sd': float(values.std(ddof=1)) if len(values) > 1 else None, 'median': float(np.median(values)),
                          'min': float(values.min()), 'max': float(values.max()), 'effects': {n: effects[n] for n in names}}
    return summary


def family_rows(runs_by_dataset, names, *, family, contrast, model_a, model_b, stratum, batch_of, q, confidence):
    """Per dataset in panel order: model_a minus model_b accuracy from the run holding the dataset (fitting seeds averaged
    within fold, corrected resampled t); one Holm adjustment across the rows."""
    rows = []
    for name in names:
        run = runs_by_dataset[name]
        seeds = {model: run.schedule['expected_seeds'][model] for model in (model_a, model_b)}
        model_rows = [row for row in run.summary['model_rows'][name] if row['model_id'] in seeds]
        interval = paired_corrected_interval(model_rows, model_a, model_b, metric='accuracy', q=q, confidence=confidence,
                                             expected_folds=run.schedule['expected_folds'], expected_seeds=seeds)
        rows.append({'family': family, 'family_index': len(rows) + 1, 'dataset': name, 'stratum': stratum[name],
                     'batch': batch_of[name], 'contrast': contrast, 'model_a': model_a, 'model_b': model_b,
                     'metric': 'accuracy', 'confidence': confidence, **interval})
    for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
        row['holm_p_approximate'] = adjusted
    return rows


def _summary(run, name, model, metric):
    return next(row for row in run.summary['summaries'][name] if (row['model_id'], row['metric']) == (model, metric))


def ladder_rows(runs_by_dataset, names, *, stratum, batch_of):
    """Per dataset, the five rungs' mean outer error, outer-fold SD and within-fold seed SD from the verified summary."""
    rows = []
    for name in names:
        for rung, model in LADDER:
            entry = _summary(runs_by_dataset[name], name, model, 'error')
            rows.append({'dataset': name, 'stratum': stratum[name], 'batch': batch_of[name], 'rung': rung, 'model_id': model,
                         'mean_error': entry['mean'],
                         **{key: entry[key] for key in ('outer_fold_sd', 'mean_within_fold_seed_sd', 'n_folds', 'seeds_per_fold')}})
    return rows


def comparator_rows(runs_by_dataset, names, *, stratum, batch_of, q, confidence):
    """Per dataset and comparator: arrowflow_full_knn minus the comparator, unadjusted; the lowest mean error flagged."""
    rows = []
    for name in names:
        run = runs_by_dataset[name]
        errors = {model: _summary(run, name, model, 'error')['mean'] for model in (TRAINED_MODEL, *COMPARATORS)}
        lowest = min(errors[model] for model in COMPARATORS)
        for model in COMPARATORS:
            seeds = {TRAINED_MODEL: run.schedule['expected_seeds'][TRAINED_MODEL], model: run.schedule['expected_seeds'][model]}
            interval = paired_corrected_interval([row for row in run.summary['model_rows'][name] if row['model_id'] in seeds],
                                                 TRAINED_MODEL, model, metric='accuracy', q=q, confidence=confidence,
                                                 expected_folds=run.schedule['expected_folds'], expected_seeds=seeds)
            rows.append({'status': DESCRIPTIVE, 'dataset': name, 'stratum': stratum[name], 'batch': batch_of[name],
                         'model_a': TRAINED_MODEL, 'model_b': model, **{key: interval[key] for key in INTERVAL},
                         'p_unadjusted': interval['p_approximate'], 'mean_error_a': errors[TRAINED_MODEL],
                         'mean_error_b': errors[model], 'best_comparator': errors[model] == lowest})
    return rows


def duplicate_audit(runs_by_dataset, names, *, batch_of, pinned):
    """Per dataset: referee_analyses.duplicate_groups on the prepared features and, per outer fold, the test rows with an
    exact duplicate in the training partition; the counts compared with the pinned counts where a pin exists."""
    rows, records = [], {}
    for name in names:
        run = runs_by_dataset[name]
        X, y, _, splits = load_prepared(run.path, name)
        vector, counts = duplicate_groups(X, y)
        folds = []
        for repeat, fold in run.schedule['expected_folds']:
            split = splits[repeat * run.protocol['outer_folds'] + fold]
            flags = training_duplicate_flags(vector, split['train'], split['test'])
            folds.append({'outer_repeat': repeat, 'outer_fold': fold, 'n_test': len(split['test']),
                          'n_with_training_duplicate': int(sum(flags))})
        n_test, flagged = sum(f['n_test'] for f in folds), sum(f['n_with_training_duplicate'] for f in folds)
        shares = [f['n_with_training_duplicate'] / f['n_test'] for f in folds]
        expected = pinned.get(name)
        matches = None if expected is None else all(counts[key] == value for key, value in expected.items())
        rows.append({'status': 'descriptive', 'dataset': name, 'batch': batch_of[name], **{key: counts[key] for key in COUNT_FIELDS},
                     'matches_pinned_counts': matches, 'test_rows': n_test, 'test_rows_with_training_duplicate': flagged,
                     'share_with_training_duplicate': flagged / n_test, 'fold_share_min': min(shares), 'fold_share_max': max(shares)})
        records[name] = {'counts': {key: counts[key] for key in COUNT_FIELDS}, 'group_sizes': counts['group_sizes'],
                         'groups': counts['groups'], 'pinned_counts': expected, 'matches_pinned_counts': matches, 'folds': folds}
    return rows, records


# ----------------------------------------------------------------------------- completeness, pairing and the commands

def completeness_gate(sources):
    """Refuse unless every batch directory holds its run records and every planned result file and fit log. Only
    planned_jobs.json is parsed, so nothing holding a score is read before both batches are complete."""
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
        raise RunComparisonError('The analysis runs only after both batches are complete, and reads neither before: '
                                 + ' | '.join(problems))


def frozen_protocol_hashes():
    hashes = {}
    for number, path in PROTOCOL_FILES.items():
        if not path.is_file():
            raise RunComparisonError(f'The frozen batch {number} protocol {path} is not in this tree')
        hashes[number] = sha256_file(path)
    return hashes


def check_batches(runs, frozen_protocols=None):
    """The runs are batch 1 and batch 2 of the frozen newdata protocols: byte-identical to the committed protocol files,
    equal outside the batch fields, the ten models with identical candidates (the registry of this tree for a production
    protocol), the same registry and sealed sources, and datasets whose hashes equal their pins."""
    for number, label in enumerate(LABELS, start=1):
        run = runs[label]
        try:
            validate_newdata_protocol(run.protocol)
        except (KeyError, TypeError, ValueError) as exc:
            raise RunComparisonError(f'The {label} protocol is not a newdata protocol: {_reason(exc)}') from exc
        if not run.protocol.get('frozen') or run.protocol.get('batch') != number:
            raise RunComparisonError(f'The {label} run must hold the frozen batch {number} protocol, not batch '
                                     f'{run.protocol.get("batch")!r} (frozen {run.protocol.get("frozen")!r})')
    expected = frozen_protocol_hashes() if frozen_protocols is None else frozen_protocols
    for number, label in enumerate(LABELS, start=1):
        if runs[label].protocol_sha256 != expected[number]:
            raise RunComparisonError(f'The {label} protocol.json (sha256 {runs[label].protocol_sha256}) is not the frozen '
                                     f'batch {number} protocol (sha256 {expected[number]})')
    first, second = runs['batch1'], runs['batch2']
    differing = sorted(key for key in set(first.protocol) | set(second.protocol)
                       if key not in BATCH_FIELDS and canonical_json(first.protocol.get(key)) != canonical_json(second.protocol.get(key)))
    if differing:
        raise RunComparisonError('The two batch protocols differ outside the batch fields: ' + ', '.join(differing))
    for run in (first, second):
        if sorted(run.registry) != sorted(MODEL_ORDER):
            raise RunComparisonError(f'The {run.label} run must hold the ten newdata models, not {sorted(run.registry)}')
    if canonical_json(first.candidates) != canonical_json(second.candidates):
        raise RunComparisonError('The two batch runs hold different candidates')
    production = first.protocol.get('purpose') is None
    if production and canonical_json(first.candidates) != canonical_json(candidate_record(build_registry(first.protocol))):
        raise RunComparisonError('The batch candidates differ from the newdata registry of this tree')
    if {run.environment.get('registry') for run in (first, second)} != {first.protocol['registry']}:
        raise RunComparisonError(f'Both environment records must name the protocol registry {first.protocol["registry"]}')
    sealed = [run.environment.get('source_hashes') or {} for run in (first, second)]
    absent = [f'{run.label} {source}' for run, hashes in zip((first, second), sealed) for source in SEALED_SOURCES if source not in hashes]
    if absent:
        raise RunComparisonError('The environment records must seal ' + ', '.join(absent))
    if sealed[0] != sealed[1]:
        raise RunComparisonError('The two batch runs seal different sources: ' + ', '.join(
            sorted(source for source in set(sealed[0]) | set(sealed[1]) if sealed[0].get(source) != sealed[1].get(source))))
    hashes = {}
    for run in (first, second):
        for name in run.protocol['datasets']:
            manifest = run.manifests[name]
            hashes[name] = {'dataset_hash': manifest.get('dataset_hash'), 'splits_hash': manifest.get('splits_hash')}
            if production and hashes[name] != {key: PIN_BY_NAME[name][key] for key in ('dataset_hash', 'splits_hash')}:
                raise RunComparisonError(f'The {run.label} run {name} dataset or splits hash differs from its pin')
    return {'protocol_ids': {run.label: run.protocol['protocol_id'] for run in (first, second)},
            'protocol_sha256': {run.label: run.protocol_sha256 for run in (first, second)}, 'batches': first.protocol['batches'],
            'code_revisions': {run.label: run.environment.get('code_revision') for run in (first, second)},
            'registry': first.protocol['registry'], 'sealed_sources': sealed[0], 'datasets': hashes,
            'fit_seeds': first.protocol['fit_seeds'], 'pins_checked': production}


def check_prepared(batch1, batch2, frozen_protocols=None):
    """Before the jobs start: the pairing of two prepared batch directories (no score exists yet)."""
    runs = {label: load_prepared_run(source, label) for label, source in zip(LABELS, (batch1, batch2))}
    return check_batches(runs, frozen_protocols)


def analyse(batch1, batch2, output, *, frozen_protocols=None):
    check_output(output, OUTPUTS)                   # an unusable output location is refused before any run is read
    completeness_gate(dict(zip(LABELS, (batch1, batch2))))
    runs = {label: load_run(source, label) for label, source in zip(LABELS, (batch1, batch2))}
    pairing = check_batches(runs, frozen_protocols)
    verified = {label: verify_run(runs[label])[0] for label in LABELS}
    protocol = runs['batch1'].protocol
    block = protocol['analysis']
    names = list(block['datasets'])
    stratum = {entry['name']: entry['stratum'] for entry in protocol['panel']}
    gaps = {entry['name']: entry['external_gap']['points'] for entry in protocol['panel']}
    batch_of = {name: int(key) for key, members in protocol['batches'].items() for name in members}
    by_dataset = {name: runs[f'batch{batch_of[name]}'] for name in names}
    common = {'stratum': stratum, 'batch_of': batch_of, 'q': protocol['test_train_ratio'], 'confidence': protocol['confidence']}
    try:
        families = {label: family_rows(by_dataset, names, family=label, contrast=block[f'{label}_family']['contrast'],
                                       model_a=block[f'{label}_family']['model_a'], model_b=block[f'{label}_family']['model_b'],
                                       **common) for label in ('primary', 'secondary')}
        effects = {row['dataset']: row['mean_difference'] for row in families['primary']}
        declared = block['moderator_test']
        test = permutation_test(effects, names, declared['strata']['H'])
        if test['splits'] != declared['splits']:
            raise ValueError(f"{test['splits']} splits enumerated, {declared['splits']} declared")
        moderator = {'hypothesis': declared['hypothesis'], 'statistic_definition': declared['statistic'], **test,
                     'alpha': declared['alpha'], 'p_at_most_alpha': test['p_one_sided'] <= declared['alpha'],
                     'effect_sizes': stratum_summary(effects, declared['strata']),
                     'limitation': block['notes'].get('low_power')}
        with_gap = list(block['spearman']['datasets'])
        spearman = {'status': block['spearman']['status'], 'datasets': with_gap, 'excluded': block['spearman']['excluded'],
                    'gaps': {name: gaps[name] for name in with_gap}, 'effects': {name: effects[name] for name in with_gap},
                    **spearman_exact([gaps[name] for name in with_gap], [effects[name] for name in with_gap])}
        ladder = ladder_rows(by_dataset, names, stratum=stratum, batch_of=batch_of)
        comparators = comparator_rows(by_dataset, names, **common)
        pinned = {name: counts for name, counts in block['duplicate_audit']['expected'].items() if counts is not None}
        duplicates, duplicate_records = duplicate_audit(by_dataset, names, batch_of=batch_of, pinned=pinned)
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        raise RunComparisonError(f'The verified batch runs do not support the prespecified analysis: {_reason(exc)}') from exc
    tables = {FAMILIES_CSV: (families['primary'] + families['secondary'], FAMILY_COLUMNS),
              COMPARATORS_CSV: (comparators, COMPARATOR_COLUMNS), LADDER_CSV: (ladder, LADDER_COLUMNS),
              DUPLICATES_CSV: (duplicates, DUPLICATE_COLUMNS)}
    contents = {name: _csv_text(rows, columns) for name, (rows, columns) in tables.items()}
    record = {
        'purpose': 'task23b_prespecified_analysis_of_both_newdata_batches',
        'status': 'computed after both batches were complete; each family Holm-adjusted within itself; the moderator test '
                  'prespecified; the Spearman correlation, ladder, comparator intervals and duplicate audit descriptive',
        'runs_purpose': protocol.get('purpose'), 'analysis_declaration': block,
        'primary_family': families['primary'], 'secondary_family': families['secondary'], 'moderator_test': moderator,
        'spearman': spearman, 'ladder': ladder, 'comparator_intervals': comparators, 'duplicate_audit': duplicate_records,
        'notes': block['notes'],
        'outputs': {name: {'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'rows': len(tables[name][0])}
                    for name, text in contents.items()},
        'provenance': {**{label: runs[label].provenance() for label in LABELS}, 'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS),
                                        **{label: {'jobs_verified': len(runs[label].jobs),
                                                   'model_rows_verified': sum(map(len, verified[label].values()))} for label in LABELS}},
                       'analysis_sources': analysis_sources(ANALYSIS_SOURCES)}}
    contents[ANALYSIS_JSON] = _json_text(record)
    write_outputs(output, contents)
    return record


def _format(value, spec='.4f'):
    return 'undefined' if value is None else format(value, spec)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, text in (('pairing', 'check two prepared batch directories before their jobs start; writes nothing'),
                          ('analyse', 'the prespecified analysis of both complete batch runs')):
        sub = commands.add_parser(command, help=text)
        sub.add_argument('--batch1', type=Path, required=True, help='batch 1 run directory')
        sub.add_argument('--batch2', type=Path, required=True, help='batch 2 run directory')
        if command == 'analyse':
            sub.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'pairing':
            record = check_prepared(args.batch1, args.batch2)
            print(f"paired {record['protocol_ids']['batch1']} and {record['protocol_ids']['batch2']}: "
                  f"{len(record['datasets'])} datasets with pinned hashes, {len(record['sealed_sources'])} sealed sources "
                  f"identical, fitting seeds {record['fit_seeds']}")
            return
        record = analyse(args.batch1, args.batch2, args.output)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_newdata {args.command} refused: {exc}\n')
    for row in record['primary_family'] + record['secondary_family']:
        print(f"{row['family']} {row['dataset']} ({row['stratum']}): {row['model_a']} - {row['model_b']} accuracy "
              f"{row['mean_difference']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
              f"Holm p={row['holm_p_approximate']:.3g}")
    moderator, spearman = record['moderator_test'], record['spearman']
    print(f"moderator: mean H {moderator['mean_h']:+.4f} minus mean C {moderator['mean_c']:+.4f} = {moderator['statistic']:+.4f}; "
          f"exact one-sided p = {moderator['at_least_observed']}/{moderator['splits']} = {moderator['p_one_sided']:.4f}")
    print(f"Spearman (descriptive, n = {spearman['n']}): rho {_format(spearman['rho'])}; exact p one-sided "
          f"{_format(spearman['p_one_sided'])}, two-sided {_format(spearman['p_two_sided'])}")


if __name__ == '__main__':
    main()
