"""E1: the prespecified analysis of the matched motion-signal controls; nothing is read until the run is complete.

python -m experiments.make_revision.compare_motion analyse --run R --output O [--reference NAME=RUN_DIR,ABLATION_DIR ...]
    Refuses (exit 2, nothing written) unless the motion-controls run is complete: its protocol, environment, manifest,
    planned jobs and sealed selections, and every planned job's result, log and prediction file. That gate parses only the
    planned job list, so no record holding a score is read before the run is complete. Each reference production run is then
    re-verified with compare_runs.load_run and compare_runs.verify_run (the harness validators over every saved record), and
    the motion-controls run with motion_controls.summary, which re-derives every sealed selection from its reference run and
    re-checks every job, before any outer score is read.

The analysis is the one the frozen protocol declares, written before any outer score existed:
  primary family   per arm in (frozen, permuted_alignment, random_direction) and per dataset, views7 minus the arm in
                   accuracy: fitting seeds averaged within each outer fold, corrected resampled t (q 0.25, 95%, 14 df),
                   Holm across the seventeen datasets within the arm; the three arms are adjusted separately
  named subset     the same rows restricted to balance-scale and mfeat-zernike, the two datasets whose training gains
                   survive the paper's Holm-34 correction; a named subset of the primary family carrying its adjustment,
                   not a new family and not adjusted again
  single layer     views7 minus single_layer_first and views7 minus single_layer_last, restricted to the outer folds whose
                   selection has two hidden layers; descriptive, with the fold count on every row, at df = n_folds - 1,
                   not Holm-adjusted (no dataset has fifteen such folds)
  descriptive      per dataset and arm: neighbourhood purity on the outer test and the training rows, per-class recall,
                   normalized filter footrule displacement and the share of unchanged filters per layer, and the motion
                   statistics of each randomizing transform
Outputs (all or none; an existing file with different content is never replaced): motion_primary_family.csv,
motion_named_subset.csv, motion_single_layer.csv, motion_purity.csv, motion_per_class_recall.csv, motion_displacement.csv,
motion_motion_statistics.csv and motion_analysis.json (every result, the declaration, the sha256 of each CSV and provenance).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from . import motion_controls as mc
from .compare_runs import RUN_RECORDS, SHARED_SOURCES, VALIDATORS, RunComparisonError, _reason, check_output, load_run, verify_run
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .newdata import sha256_file
from .referee_analyses import code_record, finish
from .run_bridge import fold_schedule

ANALYSIS_SOURCES = ('compare_motion.py', 'motion_controls.py', 'training_diagnostics.py', 'run_knn_ablation.py',
                    'compare_runs.py', 'referee_analyses.py', 'evaluation.py', 'reporting.py', 'run_revision.py',
                    'knn_controls.py', 'newdata.py')
# A reference production run seals the sources it was written with, so only the shared core is required of it; this run's
# own sources are sealed and re-checked by motion_controls.verify (environment_check 'sources').
SEALED_SOURCES = SHARED_SOURCES
RUN_FILES = ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json')
INTERVAL = ('mean_difference', 'standard_error', 'ci_low', 'ci_high')
FAMILY_COLUMNS = ('family', 'arm', 'dataset', 'contrast', 'model_a', 'model_b', *INTERVAL, 'p_approximate',
                  'holm_p_approximate', 'n_folds', 'df', 'test_train_ratio', 'mean_accuracy_a', 'mean_accuracy_b')
SUBSET_COLUMNS = ('subset', 'arm', 'dataset', *INTERVAL, 'p_approximate', 'holm_p_approximate', 'n_folds', 'status')
SINGLE_LAYER_COLUMNS = ('status', 'arm', 'dataset', *INTERVAL, 'p_unadjusted', 'n_folds', 'df', 'two_hidden_layer_folds',
                        'outer_folds')
PURITY_COLUMNS = ('dataset', 'arm', 'rows', 'mean', 'outer_fold_sd', 'n_folds')
RECALL_COLUMNS = ('dataset', 'arm', 'class', 'mean', 'outer_fold_sd', 'n_folds')
DISPLACEMENT_COLUMNS = ('dataset', 'arm', 'layer', 'kind', 'mean_normalized_footrule', 'footrule_sd', 'unchanged_share',
                        'unchanged_share_sd', 'n_folds')
MOTION_COLUMNS = ('dataset', 'arm', 'layer', 'batches', 'accepted_slots', 'changed_slots', 'changed_share',
                  'positive_share_before', 'positive_share_after', 'mass_before', 'mass_after', 'mass_deviation')
OUTPUTS = ('motion_primary_family.csv', 'motion_named_subset.csv', 'motion_single_layer.csv', 'motion_purity.csv',
           'motion_per_class_recall.csv', 'motion_displacement.csv', 'motion_motion_statistics.csv',
           'motion_analysis.json')


def completeness_gate(run_source):
    """Refuse unless the motion-controls run is complete. Only the planned job list is parsed, so nothing holding a score
    is read before the run is complete."""
    path, problems = Path(run_source), []
    if not path.is_dir():
        raise RunComparisonError(f'motion-controls run {path}: no such directory')
    missing = [name for name in (*RUN_FILES, mc.SUMMARY_JSON, mc.SUMMARY_CSV) if not (path/name).is_file()]
    absent, first = 0, None
    if (path/'planned_jobs.json').is_file():
        try:
            for job in json.loads((path/'planned_jobs.json').read_text()):
                for relative in (f"results/{job['stem']}.json", f"logs/{job['stem']}.jsonl",
                                 f"predictions/{job['stem']}.jsonl"):
                    if not (path/relative).is_file():
                        absent, first = absent + 1, first or relative
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f'unreadable planned_jobs.json ({_reason(exc)})')
    if missing:
        problems.append('missing ' + ', '.join(missing))
    if absent:
        problems.append(f'{absent} planned job files missing (first: {first})')
    if problems:
        raise RunComparisonError('The analysis runs only after the motion-controls run is complete, and reads nothing '
                                 f'before: {path} is not complete: ' + '; '.join(problems))


def check_protocol(p, *, smoke):
    """The run holds the committed frozen motion-controls protocol with the declared analysis."""
    if not smoke:
        mc.validate_protocol(p)
        if not p.get('frozen'):
            raise RunComparisonError('The motion-controls run protocol is not frozen')
        if not mc.PROTOCOL.is_file() or sha256_file(mc.PROTOCOL) != _sha256_of(p):
            raise RunComparisonError(f'The run protocol is not the committed frozen {mc.PROTOCOL.name}')
    if p.get('production_family') != mc.FAMILY or list(p['arms']) != list(mc.ARMS):
        raise RunComparisonError('The run does not hold the motion-controls family and its six arms')
    if (p['test_train_ratio'], p['confidence']) != (0.25, 0.95):
        raise RunComparisonError('This analysis is defined for q = 0.25 and 95% intervals')
    block = p['analysis']
    if (list(block['primary_family']['arms']) != list(mc.PRIMARY_ARMS)
            or list(block['named_subset']['datasets']) != list(mc.NAMED_SUBSET)):
        raise RunComparisonError('The declared analysis does not match the arms and named subset of this tree')
    return block


def _sha256_of(protocol):
    import hashlib
    return hashlib.sha256((json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()).hexdigest()


def verify_references(p, manifest, sources, *, smoke):
    """Every reference production run re-verified with compare_runs.load_run and compare_runs.verify_run, and the sealed
    file hashes of the prepared manifest re-checked."""
    provenance = {}
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
        here = mc.environment()['source_hashes']
        if not smoke:
            absent = [source for source in SEALED_SOURCES if source not in sealed]
            if entry['run']['summary_sha256'] != sha256_file(directory/'summary.json'):
                raise RunComparisonError(f'reference {name}: summary.json differs from the prepared pin')
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

def _rows_of(report, dataset):
    return report['model_rows'][dataset]


def _mean_accuracy(report, dataset, arm):
    return report['summaries'][dataset]['arms'][arm]['metrics']['accuracy']['mean']


def family_rows(report, protocol, arms, *, label):
    """Per arm and dataset, views7 minus the arm in accuracy, Holm-adjusted across the datasets within the arm."""
    names, seeds = list(protocol['datasets']), list(protocol['fit_seeds'])
    folds, q, confidence = fold_schedule(protocol), protocol['test_train_ratio'], protocol['confidence']
    rows = []
    for arm in arms:
        block = []
        for name in names:
            interval = paired_corrected_interval(_rows_of(report, name), 'views7', arm, metric='accuracy', q=q,
                                                 confidence=confidence, expected_folds=folds,
                                                 expected_seeds={'views7': seeds, arm: seeds})
            block.append({'family': label, 'arm': arm, 'dataset': name,
                          'contrast': 'views7 minus the control arm, accuracy; positive favours ArrowFlow',
                          'model_a': 'views7', 'model_b': arm,
                          **{field: interval[field] for field in INTERVAL},
                          'p_approximate': interval['p_approximate'], 'n_folds': interval['n_folds'],
                          'df': interval['df'], 'test_train_ratio': interval['test_train_ratio'],
                          'mean_accuracy_a': _mean_accuracy(report, name, 'views7'),
                          'mean_accuracy_b': _mean_accuracy(report, name, arm)})
        adjusted = holm_adjust([row['p_approximate'] for row in block])
        for row, value in zip(block, adjusted):
            row['holm_p_approximate'] = value
        rows.extend(block)
    return rows


def named_subset_rows(family, datasets):
    """The same primary-family rows restricted to the named datasets; the Holm p is the family's own, not re-adjusted."""
    status = ('a named subset of the primary family: these rows carry the primary family\'s Holm adjustment over the '
              'seventeen datasets and are not adjusted again')
    return [{'subset': 'training_gain_survivors', 'arm': row['arm'], 'dataset': row['dataset'],
             **{field: row[field] for field in INTERVAL}, 'p_approximate': row['p_approximate'],
             'holm_p_approximate': row['holm_p_approximate'], 'n_folds': row['n_folds'], 'status': status}
            for row in family if row['dataset'] in set(datasets)]


def _fold_means(rows, arm, metric, folds):
    values = defaultdict(list)
    for row in rows:
        if row['model_id'] == arm:
            values[(row['outer_repeat'], row['outer_fold'])].append(float(row[metric]))
    return {key: float(np.mean(v)) for key, v in values.items() if key in set(folds)}


def single_layer_rows(report, protocol, jobs):
    """views7 minus each single_layer arm, restricted to the outer folds whose selection has two hidden layers."""
    from scipy import stats
    q, names = protocol['test_train_ratio'], list(protocol['datasets'])
    confidence = protocol['confidence']
    rows = []
    for arm in mc.DEPTH_ARMS:
        for name in names:
            depth = {(j['outer_repeat'], j['outer_fold']): j['hidden_layers'] for j in jobs if j['dataset_id'] == name}
            qualifying = sorted(key for key, layers in depth.items() if layers >= 2)
            model = _fold_means(_rows_of(report, name), arm, 'accuracy', qualifying)
            reference = _fold_means(_rows_of(report, name), 'views7', 'accuracy', qualifying)
            differences = np.array([reference[key] - model[key] for key in qualifying])
            row = {'arm': arm, 'dataset': name, 'two_hidden_layer_folds': len(qualifying), 'outer_folds': len(depth),
                   'n_folds': len(qualifying), 'df': max(len(qualifying) - 1, 0)}
            if len(qualifying) >= 3:
                mean = float(np.mean(differences))
                se = float(np.sqrt((1 / len(qualifying) + q) * np.var(differences, ddof=1)))
                half = float(stats.t.ppf((1 + confidence) / 2, len(qualifying) - 1) * se)
                row.update(status='descriptive; corrected resampled t over the two-hidden-layer folds only, df = n_folds - 1; '
                                  'not part of the primary family and not Holm-adjusted',
                           mean_difference=mean, standard_error=se, ci_low=mean - half, ci_high=mean + half,
                           p_unadjusted=float(2 * stats.t.sf(abs(mean / se), len(qualifying) - 1)) if se else
                                        (1. if mean == 0 else 0.))
            else:
                row.update(status=('identical to views7 by construction at a one-hidden-layer selection' if not qualifying
                                   else 'too few two-hidden-layer folds for an interval (n_folds < 3)'),
                           mean_difference=float(np.mean(differences)) if len(differences) else None,
                           standard_error=None, ci_low=None, ci_high=None, p_unadjusted=None)
            rows.append(row)
    return rows


def purity_rows(report, protocol):
    rows = []
    for name in protocol['datasets']:
        for arm in mc.ARMS:
            for where in ('test', 'train'):
                entry = report['summaries'][name]['arms'][arm]['neighborhood_purity'][where]
                rows.append({'dataset': name, 'arm': arm, 'rows': f'outer_{where}' if where == 'test' else 'training',
                             'mean': entry['mean'], 'outer_fold_sd': entry['sd'], 'n_folds': entry['n']})
    return rows


def recall_rows(report, protocol):
    rows = []
    for name in protocol['datasets']:
        for arm in mc.ARMS:
            for label, entry in sorted(report['summaries'][name]['arms'][arm]['per_class_recall'].items()):
                rows.append({'dataset': name, 'arm': arm, 'class': label, 'mean': entry['mean'],
                             'outer_fold_sd': entry['sd'], 'n_folds': entry['n']})
    return rows


def displacement_rows(report, protocol):
    rows = []
    for name in protocol['datasets']:
        for arm in mc.ARMS:
            for entry in report['summaries'][name]['arms'][arm]['displacement']:
                rows.append({'dataset': name, 'arm': arm, 'layer': entry['layer'], 'kind': entry['kind'],
                             'mean_normalized_footrule': entry['mean_normalized_footrule']['mean'],
                             'footrule_sd': entry['mean_normalized_footrule']['sd'],
                             'unchanged_share': entry['unchanged_share']['mean'],
                             'unchanged_share_sd': entry['unchanged_share']['sd'],
                             'n_folds': entry['mean_normalized_footrule']['n']})
    return rows


def motion_rows(dataset_statistics):
    rows = []
    for name, by_arm in sorted(dataset_statistics.items()):
        for arm in mc.ARMS:
            for layer, entry in sorted((by_arm.get(arm) or {}).items()):
                rows.append({'dataset': name, 'arm': arm, 'layer': layer, 'batches': entry.get('batches'),
                             'accepted_slots': entry['accepted_slots'], 'changed_slots': entry['changed_slots'],
                             'changed_share': entry['changed_share'],
                             'positive_share_before': entry.get('positive_share'),
                             'positive_share_after': (entry['positive_after'] / entry['accepted_slots'])
                                                     if entry['accepted_slots'] else None,
                             'mass_before': entry['mass_before'], 'mass_after': entry['mass_after'],
                             'mass_deviation': entry.get('mass_deviation')})
    return rows


def per_dataset_motion(run_source, jobs):
    """The motion statistics of the whole run, split by dataset (the summary reports them pooled)."""
    out = {}
    for job in jobs:
        record = json.loads((Path(run_source)/'results'/f"{job['stem']}.json").read_text())
        totals = mc._motion_totals({job['stem']: record})
        by_arm = out.setdefault(job['dataset_id'], {})
        for arm, layers in totals.items():
            target = by_arm.setdefault(arm, {})
            for layer, entry in layers.items():
                current = target.setdefault(layer, {key: 0 for key in ('batches', 'accepted_slots', 'changed_slots',
                                                                       'positive_before', 'positive_after')})
                current.setdefault('mass_before', 0.)
                current.setdefault('mass_after', 0.)
                for key in ('accepted_slots', 'changed_slots', 'positive_before', 'positive_after'):
                    current[key] += entry[key]
                for key in ('mass_before', 'mass_after'):
                    current[key] += entry[key]
    for by_arm in out.values():
        for layers in by_arm.values():
            for entry in layers.values():
                slots = entry['accepted_slots']
                entry['changed_share'] = (entry['changed_slots'] / slots) if slots else 0.
                entry['positive_share'] = (entry['positive_before'] / slots) if slots else None
                entry['mass_deviation'] = abs(entry['mass_before'] - entry['mass_after'])
                entry.pop('batches', None)
    return out


def analyse(run_source, output, sources=None, *, allow_smoke=False):
    check_output(output, OUTPUTS)                   # an unusable output location is refused before any run is read
    completeness_gate(run_source)
    run_source = Path(run_source)
    protocol = json.loads((run_source/'protocol.json').read_text())
    manifest = json.loads((run_source/'manifest.json').read_text())
    smoke = allow_smoke and manifest.get('purpose') == 'synthetic_smoke_only'
    block = check_protocol(protocol, smoke=smoke)
    references = verify_references(protocol, manifest, sources or {}, smoke=smoke)
    report, _ = mc.summary(run_source, allow_smoke=smoke)           # re-verifies every job and every sealed selection
    published = json.loads((run_source/mc.SUMMARY_JSON).read_text())
    if canonical_json(report) != canonical_json(published):
        raise RunComparisonError(f'The re-verified summary differs from the published {mc.SUMMARY_JSON}')
    jobs = json.loads((run_source/'planned_jobs.json').read_text())
    if list(protocol['datasets']) != list(manifest['datasets']):
        raise RunComparisonError('The run does not hold every protocol dataset')
    try:
        primary = family_rows(report, protocol, mc.PRIMARY_ARMS, label='primary')
        subset = named_subset_rows(primary, block['named_subset']['datasets'])
        if not smoke and len(subset) != len(mc.PRIMARY_ARMS) * len(block['named_subset']['datasets']):
            raise RunComparisonError('The named subset does not cover its declared datasets in every primary arm')
        single = single_layer_rows(report, protocol, jobs)
        statistics = per_dataset_motion(run_source, jobs)
        tables = {'motion_primary_family.csv': (primary, FAMILY_COLUMNS),
                  'motion_named_subset.csv': (subset, SUBSET_COLUMNS),
                  'motion_single_layer.csv': (single, SINGLE_LAYER_COLUMNS),
                  'motion_purity.csv': (purity_rows(report, protocol), PURITY_COLUMNS),
                  'motion_per_class_recall.csv': (recall_rows(report, protocol), RECALL_COLUMNS),
                  'motion_displacement.csv': (displacement_rows(report, protocol), DISPLACEMENT_COLUMNS),
                  'motion_motion_statistics.csv': (motion_rows(statistics), MOTION_COLUMNS)}
    except (KeyError, TypeError, ValueError, IndexError, StopIteration) as exc:
        if isinstance(exc, RunComparisonError):
            raise
        raise RunComparisonError(f'The verified motion-controls run does not support the prespecified analysis: '
                                 f'{_reason(exc)}') from exc
    record = {
        'purpose': 'e1_prespecified_analysis_of_the_matched_motion_signal_controls',
        'status': 'computed after the motion-controls run was complete and every reference run and every job re-verified; '
                  'the primary family is Holm-adjusted within each arm across the seventeen datasets, the three arms '
                  'separately; the named subset carries that adjustment and is not adjusted again; the single-layer '
                  'contrasts, the purity, the per-class recall, the displacement and the motion statistics are descriptive',
        'analysis_declaration': block, 'supervision_disclosure': protocol['supervision_disclosure'],
        'differs_from_untrained': protocol['differs_from_untrained'],
        'interpretation': block['interpretation'],
        'primary_family': primary, 'named_subset': subset, 'single_layer': single,
        'purity': tables['motion_purity.csv'][0], 'per_class_recall': tables['motion_per_class_recall.csv'][0],
        'displacement': tables['motion_displacement.csv'][0], 'motion_statistics': tables['motion_motion_statistics.csv'][0],
        'views7_reproduces_reference': {name: entry['views7_reproduces_reference']
                                        for name, entry in report['summaries'].items()},
        'job_checks': report['checks'],
        'provenance': {'run': {'directory': str(run_source.resolve()), 'protocol_id': protocol.get('protocol_id'),
                               'protocol_sha256': sha256_file(run_source/'protocol.json'),
                               'code_revision': report['code_revision'], 'planned_jobs': len(jobs),
                               'summary_sha256': sha256_file(run_source/mc.SUMMARY_JSON),
                               'purpose': manifest.get('purpose')},
                       'references': references,
                       'verification': {'validators': list(VALIDATORS) + ['motion_controls.summary (every job, every '
                                                                          'sealed selection re-derived, every check)'],
                                        'arms': list(mc.ARMS)},
                       **code_record(ANALYSIS_SOURCES)}}
    record = mc._plain(record)
    finish(output, {name: (mc._plain(rows), columns) for name, (rows, columns) in tables.items()},
           'motion_analysis.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    sub = commands.add_parser('analyse', help='the prespecified analysis of one complete motion-controls run')
    sub.add_argument('--run', type=Path, required=True, help='the motion-controls run directory (summary written)')
    sub.add_argument('--output', type=Path, required=True, help='new output directory')
    sub.add_argument('--reference', type=mc.parse_reference, action='append', metavar='NAME=RUN_DIR,ABLATION_DIR')
    args = parser.parse_args(argv)
    sources = dict(args.reference or [])
    try:
        record = analyse(args.run, args.output, sources)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_motion analyse refused: {exc}\n')
    for row in record['primary_family']:
        print(f"{row['arm']} {row['dataset']}: views7 - {row['arm']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} "
              f"Holm p={row['holm_p_approximate']:.3g}")
    for name, counts in record['views7_reproduces_reference'].items():
        print(f"{name}: views7 reproduced {counts['matching_fold_seeds']}/{counts['total_fold_seeds']} fold-seeds")


if __name__ == '__main__':
    main()
