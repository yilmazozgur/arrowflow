"""Task 23A: projected_numeric_knn (knn_projected run) against the knn (bridge_knn) and knn_training runs.

python -m experiments.make_revision.compare_runs projected --projected-source P --knn-source K --training-source T --output O
python -m experiments.make_revision.compare_runs projected-pairing --projected-source P --knn-source K --training-source T

The three runs must be complete, and every saved record of each is re-verified before anything is compared, with
compare_runs.load_run and compare_runs.verify_run: prepared data and splits hashes, the declared nested splits, planned
jobs and fit logs, every result record with its per-example predictions, and summary.json against the re-verified model
rows. The projected protocol must be a knn_projected protocol (projected_knn.validate_projected_protocol) whose
projected_control.references pin exactly the knn run and the knn_training run: summary.json sha256, code revision,
protocol sha256, protocol ID and the models each run holds. The three runs must share the dataset panel, the dataset and
splits hashes, the nested design keys (compare_runs.DESIGN_KEYS), the outer folds and the fitting seeds; the projected
protocol must carry the knn_training protocol's value for every key of DESIGN_COPY_KEYS; projected_numeric_knn's
candidate record must equal input_footrule_knn's in the training run, and its candidates must be the projection of the
knn run's arrowflow_full_knn candidates; every source sealed by two or more of the environment records must be
byte-identical across them, with compare_runs.SHARED_SOURCES sealed by all three and knn_controls.py by the training and
projected runs. Anything else is refused (exit status 2) before an output is written.

Outputs (all or none; an existing file with different content is never replaced):
  projected_contrasts.csv            the prespecified family: per dataset in panel order, input_footrule_knn (training
                                     run) minus projected_numeric_knn (the effect of sorting), then arrowflow_full_knn
                                     (knn run) minus projected_numeric_knn; accuracy, fitting seeds averaged within each
                                     outer fold, corrected resampled t, one Holm adjustment across all members
  projected_raw_descriptive.csv      per dataset, numeric_knn (knn run; imputed, standardized raw features) minus
                                     projected_numeric_knn: expansion and projection without sorting; descriptive, outside
                                     the Holm family, no p values
  projected_contrasts.json           the family, the descriptive contrasts, definitions, the sha256 of both CSVs and the
                                     provenance, reference and pairing records of the three runs
  projected_ladder_error_table.json  per dataset, mean error, outer-fold SD and within-fold seed SD of the ladder: raw
                                     numeric kNN, projected kNN, encoded-ranking kNN, untrained ArrowFlow-kNN, ArrowFlow-kNN
projected-pairing checks a prepared knn_projected directory against the complete knn and knn_training runs before its jobs
start (reference pins and pairing only) and writes nothing.
"""
import hashlib
from itertools import combinations
from .compare_runs import (ANALYSIS_SOURCES, CANDIDATE_FIELDS, CONTRAST_METRIC, DEFINITIONS, DESIGN_KEYS, SHARED_SOURCES,
                           TABLE_METRIC, TRAINING_CONTRAST_COLUMNS, VALIDATORS, RunComparisonError, _csv_text, _json_text,
                           _reason, analysis_sources, check_output, load_prepared_run, load_run, verify_run, write_outputs)
from .evaluation import canonical_json, holm_adjust, paired_corrected_interval
from .knn_controls import CANDIDATE_KEYS as CONTROL_CANDIDATE_KEYS, CONTROL_MODELS, INPUT_MODEL, TRAINED_MODEL, project_candidates
from .projected_knn import (DESCRIPTIVE_CONTRAST, LADDER, PRIMARY_CONTRASTS, PROJECTED_MODEL, RAW_MODEL, REFERENCE_MODELS,
                            validate_projected_protocol)

PROJECTED_CONTRASTS_CSV, PROJECTED_DESCRIPTIVE_CSV = 'projected_contrasts.csv', 'projected_raw_descriptive.csv'
PROJECTED_CONTRASTS_JSON, PROJECTED_LADDER_JSON = 'projected_contrasts.json', 'projected_ladder_error_table.json'
PROJECTED_OUTPUTS = (PROJECTED_CONTRASTS_CSV, PROJECTED_DESCRIPTIVE_CSV, PROJECTED_CONTRASTS_JSON, PROJECTED_LADDER_JSON)
DESCRIPTIVE_COLUMNS = ('status', 'dataset', 'contrast', 'model_a', 'run_a', 'model_b', 'run_b', 'mean_difference',
                       'standard_error', 'ci_low', 'ci_high', 'n_folds', 'df')
DESCRIPTIVE_STATUS = 'descriptive; outside the Holm family; no p values'
PROJECTED_ANALYSIS_SOURCES = ANALYSIS_SOURCES + ('knn_controls.py', 'projected_knn.py', 'compare_projected.py')
CONTRAST_SOURCES = {PRIMARY_CONTRASTS[0]: (INPUT_MODEL, 'training'), PRIMARY_CONTRASTS[1]: (TRAINED_MODEL, 'knn')}
TRAINING_SEALED = ('experiments/make_revision/knn_controls.py',)
RUN_LABELS = ('projected', 'knn', 'training')
# The design keys the projected protocol copies from knn_training.json (its design_source.identical_to_template list, plus
# the family size and multiplicity, which the two families share).
DESIGN_COPY_KEYS = ('datasets', 'split_seed', 'outer_folds', 'outer_repeats', 'inner_folds', 'fit_seeds', 'candidate_budget',
                    'candidate_seed', 'candidate_tie_rule', 'stochastic_finalists', 'selection_metric', 'failure_policy',
                    'aggregation_integrity', 'freeze_requirement', 'paired_interval', 'parallelism', 'confidence',
                    'test_train_ratio', 'numeric_threads_per_worker', 'report_metrics', 'device', 'historical_results',
                    'full_method', 'primary_family_size', 'multiplicity')
PROJECTED_DEFINITIONS = {
    'mean_difference': f'model_a (its source run) minus {PROJECTED_MODEL} (projected run) accuracy: fitting seeds averaged '
                       'within each outer fold, then the mean over the outer folds; positive favours model_a',
    'p_approximate': DEFINITIONS['p_approximate'],
    'holm_p_approximate': 'Holm adjustment of p_approximate across every member of the family (primary_family_size: two '
                          'contrasts per dataset); the descriptive raw numeric kNN contrasts are not members',
    'family_index': f'position in the family: datasets in panel order, {INPUT_MODEL} before {TRAINED_MODEL}',
    'effect_of_sorting': f'{INPUT_MODEL} minus {PROJECTED_MODEL}: the same seven encoders at the same encoder candidates '
                         'with the same readout-selection splits; the footrule kNN reads the argsort of each view\'s scores '
                         '(the encoded ranking) where the numeric kNN reads the scores, and the numeric readout also tunes '
                         'the Minkowski order p',
    'descriptive_raw_contrast': f'{RAW_MODEL} (knn run: the tuned numeric kNN comparator on imputed, standardized raw '
                                f'features) minus {PROJECTED_MODEL}: the effect of polynomial expansion, projection and the '
                                'seven-view vote without sorting; corrected resampled t interval over the outer folds, no '
                                'p value, outside the Holm family',
    'candidates': f'{PROJECTED_MODEL}\'s candidate record (stochastic flag, candidates, config_ids) equals {INPUT_MODEL}\'s '
                  f'in the training run, and its candidates are the knn run\'s {TRAINED_MODEL} candidates projected onto '
                  f'knn_controls.CANDIDATE_KEYS["{INPUT_MODEL}"]',
    'shared_sources': 'every source sealed by two or more of the three environment records is byte-identical across them; '
                      'compare_runs.SHARED_SOURCES are sealed by all three runs and knn_controls.py by the training and '
                      'projected runs',
    'verified': 'every saved record of the three runs passed ' + '; '.join(VALIDATORS) + '. Each summary.json model_rows '
                'equal the re-verified model rows field for field and its summaries were recomputed from them. The '
                'analysis source hashes in environment.json are not compared with the current tree; each run\'s recorded '
                'code revision is in provenance.',
}


def _differs(runs, key):
    return any(key not in run.protocol for run in runs) or len({canonical_json(run.protocol[key]) for run in runs}) != 1


def check_projected_references(projected_protocol, knn, training):
    """projected_control.references must pin exactly this knn run and this knn_training run."""
    block = projected_protocol.get('projected_control')
    references = block.get('references') if isinstance(block, dict) else None
    if not isinstance(references, dict):
        raise RunComparisonError('The projected protocol declares no projected_control.references block')
    observed = {}
    for run in (knn, training):
        declared = references.get(run.label)
        if not isinstance(declared, dict):
            raise RunComparisonError(f'The projected protocol declares no projected_control.references.{run.label} block')
        pins = {'summary_sha256': run.summary_sha256, 'code_revision': run.environment['code_revision'],
                'protocol_sha256': run.protocol_sha256, 'protocol_id': run.protocol.get('protocol_id'),
                'model_ids': list(REFERENCE_MODELS[run.label])}
        wrong = [f'{key}: declared {declared.get(key)!r}, {run.label} run has {value!r}'
                 for key, value in pins.items() if declared.get(key) != value]
        if wrong:
            raise RunComparisonError(f'The projected protocol reference block does not match the {run.label} run: '
                                     + '; '.join(wrong))
        absent = [model for model in REFERENCE_MODELS[run.label] if model not in run.registry]
        if absent:
            raise RunComparisonError(f'The {run.label} run holds no {", ".join(absent)}')
        observed[run.label] = pins
    return observed


def check_projected_pairing(projected, knn, training):
    """The three runs share panel, dataset and splits hashes, nested design, outer folds and fitting seeds; the projected
    protocol copies the knn_training design; projected_numeric_knn's candidates are input_footrule_knn's; sources sealed by
    more than one run are byte-identical."""
    runs = (projected, knn, training)
    try:
        validate_projected_protocol(projected.protocol)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunComparisonError(f'The projected protocol is not a knn_projected protocol: {_reason(exc)}') from exc
    if list(projected.registry) != [PROJECTED_MODEL]:
        raise RunComparisonError(f'The projected run must hold exactly {PROJECTED_MODEL}, not {sorted(projected.registry)}')
    if sorted(training.registry) != sorted(CONTROL_MODELS):
        raise RunComparisonError(f'The training run must hold exactly {list(CONTROL_MODELS)}, not {sorted(training.registry)}')
    absent = [model for model in REFERENCE_MODELS['knn'] if model not in knn.registry]
    if absent:
        raise RunComparisonError(f'The knn run holds no {", ".join(absent)}')
    for key in DESIGN_KEYS:
        if _differs(runs, key):
            raise RunComparisonError(f'Protocol {key} differs between the runs ('
                                     + ', '.join(f'{run.label} {run.protocol.get(key)!r}' for run in runs) + ')')
    for key in DESIGN_COPY_KEYS:
        if _differs((projected, training), key):
            raise RunComparisonError(f'The projected protocol {key} is not the knn_training protocol\'s '
                                     f'(projected {projected.protocol.get(key)!r}, training {training.protocol.get(key)!r})')
    folds = projected.schedule['expected_folds']
    if any(run.schedule['expected_folds'] != folds for run in (knn, training)):
        raise RunComparisonError('The outer fold schedules differ between the runs')
    reference_record = {field: training.candidates[INPUT_MODEL][field] for field in CANDIDATE_FIELDS}
    record = {field: projected.candidates[PROJECTED_MODEL][field] for field in CANDIDATE_FIELDS}
    if canonical_json(record) != canonical_json(reference_record):
        raise RunComparisonError(f'{PROJECTED_MODEL} candidates differ from the training run {INPUT_MODEL} candidates '
                                 '(stochastic flag, candidates or config_ids)')
    try:
        expected = project_candidates(knn.candidates[TRAINED_MODEL]['candidates'], CONTROL_CANDIDATE_KEYS[INPUT_MODEL])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunComparisonError(f'The knn run {TRAINED_MODEL} candidates cannot be projected: {_reason(exc)}') from exc
    if canonical_json(record['candidates']) != canonical_json(expected):
        raise RunComparisonError(f'{PROJECTED_MODEL} candidates are not the projection of the knn run {TRAINED_MODEL} '
                                 f'candidates onto {list(CONTROL_CANDIDATE_KEYS[INPUT_MODEL])}')
    seeds = projected.schedule['expected_seeds'][PROJECTED_MODEL]
    for run, model in ((knn, TRAINED_MODEL), (training, INPUT_MODEL), (training, CONTROL_MODELS[0])):
        if run.schedule['expected_seeds'][model] != seeds:
            raise RunComparisonError(f'{model} ({run.label} run) and {PROJECTED_MODEL} fitting seed schedules differ')
    if knn.schedule['expected_seeds'][RAW_MODEL] != seeds[:1]:
        raise RunComparisonError(f'{RAW_MODEL} (knn run) is not fitted at the first fitting seed of {PROJECTED_MODEL}')
    datasets = projected.protocol['datasets']
    for name in datasets:
        for key, description in (('dataset_hash', 'dataset hash'), ('splits_hash', 'splits hash')):
            values = [run.manifests[name].get(key) for run in runs]
            if not values[0] or len(set(values)) != 1:
                raise RunComparisonError(f'The {description} of {name} differs between the runs ('
                                         + ', '.join(f'{run.label} {value}' for run, value in zip(runs, values)) + ')')
    sealed = {run.label: run.environment.get('source_hashes') or {} for run in runs}
    absent = [f'{run.label} run {source}' for run in runs for source in SHARED_SOURCES if source not in sealed[run.label]]
    absent += [f'{run.label} run {source}' for run in (projected, training) for source in TRAINING_SEALED
               if source not in sealed[run.label]]
    if absent:
        raise RunComparisonError('The environment records must seal ' + ', '.join(absent))
    sealed_by = {}
    for a, b in combinations(RUN_LABELS, 2):
        for source in set(sealed[a]) & set(sealed[b]):
            sealed_by.setdefault(source, set()).update((a, b))
    differing = sorted(source for source, labels in sealed_by.items() if len({sealed[label][source] for label in labels}) != 1)
    if differing:
        raise RunComparisonError('Sources sealed by more than one run differ: ' + ', '.join(differing))
    return {'datasets': list(datasets),
            'dataset_hash': {name: projected.manifests[name]['dataset_hash'] for name in datasets},
            'splits_hash': {name: projected.manifests[name]['splits_hash'] for name in datasets},
            **{key: projected.protocol[key] for key in ('outer_folds', 'outer_repeats', 'inner_folds', 'split_seed', 'fit_seeds')},
            'design_copy_keys': list(DESIGN_COPY_KEYS),
            'candidate_config_ids': list(record['config_ids']),
            'reference_candidate_config_ids': list(knn.candidates[TRAINED_MODEL]['config_ids']),
            'shared_sources': {source: sealed[sorted(labels)[0]][source] for source, labels in sorted(sealed_by.items())},
            'sealed_by': {source: [label for label in RUN_LABELS if label in labels] for source, labels in sorted(sealed_by.items())}}


def check_projected_prepared(projected_source, knn_source, training_source):
    """Before a knn_projected run: its prepared records against the complete knn and knn_training runs."""
    projected = load_prepared_run(projected_source, 'projected')
    knn, training = load_run(knn_source, 'knn'), load_run(training_source, 'training')
    return {'references': check_projected_references(projected.protocol, knn, training),
            'pairing': check_projected_pairing(projected, knn, training)}


def _model_rows(run, name, model):
    return [row for row in run.summary['model_rows'][name] if row['model_id'] == model]


def _interval(run_a, model_a, projected, name):
    seeds = {model_a: run_a.schedule['expected_seeds'][model_a],
             PROJECTED_MODEL: projected.schedule['expected_seeds'][PROJECTED_MODEL]}
    return paired_corrected_interval(_model_rows(run_a, name, model_a) + _model_rows(projected, name, PROJECTED_MODEL),
                                     model_a, PROJECTED_MODEL, metric=CONTRAST_METRIC, q=projected.protocol['test_train_ratio'],
                                     confidence=projected.protocol['confidence'],
                                     expected_folds=projected.schedule['expected_folds'], expected_seeds=seeds)


def projected_contrasts(projected, runs):
    """Per dataset, input_footrule_knn (training run) then arrowflow_full_knn (knn run) minus projected_numeric_knn; one Holm
    adjustment across all members."""
    rows = []
    for name in projected.protocol['datasets']:
        for contrast in PRIMARY_CONTRASTS:
            model, label = CONTRAST_SOURCES[contrast]
            rows.append({'family_index': len(rows) + 1, 'dataset': name, 'contrast': contrast, 'model_a': model,
                         'run_a': label, 'model_b': PROJECTED_MODEL, 'run_b': projected.label, 'metric': CONTRAST_METRIC,
                         'confidence': projected.protocol['confidence'], **_interval(runs[label], model, projected, name)})
    for row, adjusted in zip(rows, holm_adjust([row['p_approximate'] for row in rows])):
        row['holm_p_approximate'] = adjusted
    return rows


def raw_descriptive_contrasts(projected, knn):
    """Per dataset, numeric_knn (knn run) minus projected_numeric_knn; the interval without its p value; not adjusted."""
    rows = []
    for name in projected.protocol['datasets']:
        interval = _interval(knn, RAW_MODEL, projected, name)
        interval.pop('p_approximate')
        rows.append({'status': DESCRIPTIVE_STATUS, 'dataset': name, 'contrast': DESCRIPTIVE_CONTRAST, 'model_a': RAW_MODEL,
                     'run_a': knn.label, 'model_b': PROJECTED_MODEL, 'run_b': projected.label, 'metric': CONTRAST_METRIC,
                     'confidence': projected.protocol['confidence'], **interval})
    return rows


def ladder_error_table(projected, runs, sources):
    def entry(rung, model, label, name):
        row = next(r for r in runs[label].summary['summaries'][name] if (r['model_id'], r['metric']) == (model, TABLE_METRIC))
        return {'rung': rung, 'model_id': model, 'source_run': label, 'mean_error': row['mean'],
                'outer_fold_sd': row['outer_fold_sd'], 'mean_within_fold_seed_sd': row['mean_within_fold_seed_sd'],
                'n_folds': row['n_folds'], 'seeds_per_fold': row['seeds_per_fold']}
    datasets = projected.protocol['datasets']
    return {'purpose': 'mean_outer_error_of_the_knn_ladder', 'metric': TABLE_METRIC, 'datasets': list(datasets),
            'rungs': [{'rung': rung, 'model_id': model, 'source_run': label} for rung, model, label in LADDER],
            'definitions': {'raw_numeric_knn': f'{RAW_MODEL}: tuned numeric kNN on imputed, standardized raw features; '
                                               'deterministic, one fitting seed',
                            'projected_numeric_knn': f'{PROJECTED_MODEL}: numeric kNN per view on the pre-sort projected '
                                                     'scores of the seven encoders; majority vote',
                            'encoded_ranking_knn': f'{INPUT_MODEL}: footrule kNN per view on the encoded input ranking of '
                                                   'the same encoders; majority vote',
                            'untrained_arrowflow_knn': f'{CONTROL_MODELS[0]}: ArrowFlow-kNN with its networks at their '
                                                       'seeded initial filters',
                            'arrowflow_knn': f'{TRAINED_MODEL}: ArrowFlow with the kNN readout on the trained hidden ranking',
                            'mean_error': 'mean over outer folds of the fitting-seed-averaged error',
                            'outer_fold_sd': 'SD (ddof 1) of those outer-fold means',
                            'mean_within_fold_seed_sd': 'mean over outer folds of the within-fold SD across fitting seeds; '
                                                        'null for a deterministic family'},
            'sources': {label: runs[label].provenance() for label in RUN_LABELS}, 'analysis_sources': sources,
            'rows': {name: [entry(rung, model, label, name) for rung, model, label in LADDER] for name in datasets}}


def compare_projected(projected_source, knn_source, training_source, output):
    check_output(output, PROJECTED_OUTPUTS)         # an unusable output location is refused before any run is read
    projected = load_run(projected_source, 'projected')
    knn, training = load_run(knn_source, 'knn'), load_run(training_source, 'training')
    runs = {'projected': projected, 'knn': knn, 'training': training}
    references = check_projected_references(projected.protocol, knn, training)
    pairing = check_projected_pairing(projected, knn, training)
    verified = [(run, verify_run(run)[0]) for run in (projected, knn, training)]
    sources = analysis_sources(PROJECTED_ANALYSIS_SOURCES)
    try:
        contrasts = projected_contrasts(projected, runs)
        descriptive = raw_descriptive_contrasts(projected, knn)
        table = ladder_error_table(projected, runs, sources)
    except (KeyError, TypeError, ValueError, StopIteration) as exc:
        raise RunComparisonError(f'The verified runs do not support the projected family: {_reason(exc)}') from exc
    contrasts_csv = _csv_text(contrasts, TRAINING_CONTRAST_COLUMNS)
    descriptive_csv = _csv_text(descriptive, DESCRIPTIVE_COLUMNS)
    folds = projected.schedule['expected_folds']
    summary = {
        'purpose': 'cross_run_primary_contrast_family_encoded_ranking_and_arrowflow_knn_minus_projected_numeric_knn',
        'contrasts_declared': list(PRIMARY_CONTRASTS),
        'family': {'metric': CONTRAST_METRIC, 'models_a': {model: label for model, label in CONTRAST_SOURCES.values()},
                   'model_b': PROJECTED_MODEL, 'run_b': projected.label, 'datasets': list(projected.protocol['datasets']),
                   'size': len(contrasts), 'multiplicity': 'holm', 'test_train_ratio': projected.protocol['test_train_ratio'],
                   'confidence': projected.protocol['confidence'], 'n_folds': len(folds), 'df': len(folds) - 1,
                   'fitting_seeds': list(projected.protocol['fit_seeds']),
                   'method': 'fitting seeds averaged within each outer fold; corrected resampled t over the outer folds '
                             '(evaluation.paired_corrected_interval on the model_rows of each run\'s summary.json, equal to '
                             'the model rows re-verified from the saved records); Holm across all members '
                             '(evaluation.holm_adjust)'},
        'definitions': dict(PROJECTED_DEFINITIONS),
        'contrasts': contrasts,
        'descriptive': {'contrast': DESCRIPTIVE_CONTRAST, 'status': DESCRIPTIVE_STATUS, 'rows': descriptive},
        'outputs': {name: {'sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'rows': rows}
                    for name, text, rows in ((PROJECTED_CONTRASTS_CSV, contrasts_csv, len(contrasts)),
                                             (PROJECTED_DESCRIPTIVE_CSV, descriptive_csv, len(descriptive)))},
        'provenance': {**{label: runs[label].provenance() for label in RUN_LABELS}, 'references': references,
                       'pairing': pairing,
                       'verification': {'validators': list(VALIDATORS),
                                        **{run.label: {'jobs_verified': len(run.jobs),
                                                       'model_rows_verified': sum(map(len, rows.values()))}
                                           for run, rows in verified}},
                       'analysis_sources': sources},
    }
    write_outputs(output, {PROJECTED_CONTRASTS_CSV: contrasts_csv, PROJECTED_DESCRIPTIVE_CSV: descriptive_csv,
                           PROJECTED_CONTRASTS_JSON: _json_text(summary), PROJECTED_LADDER_JSON: _json_text(table)})
    return {'contrasts': contrasts, 'descriptive': descriptive, 'summary': summary, 'ladder': table}


def main_projected(parser, args):
    """compare_runs projected."""
    try:
        result = compare_projected(args.projected_source, args.knn_source, args.training_source, args.output)
    except (RunComparisonError, FileExistsError) as exc:
        parser.exit(2, f'compare_runs projected refused: {exc}\n')
    for row in result['contrasts']:
        print(f"{row['dataset']}: {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] p={row['p_approximate']:.3g} Holm p={row['holm_p_approximate']:.3g}")
    for row in result['descriptive']:
        print(f"{row['dataset']}: {row['model_a']} - {row['model_b']} accuracy {row['mean_difference']:+.4f} "
              f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] (descriptive)")


def main_projected_pairing(parser, args):
    """compare_runs projected-pairing."""
    try:
        record = check_projected_prepared(args.projected_source, args.knn_source, args.training_source)
    except RunComparisonError as exc:
        parser.exit(2, f'compare_runs projected-pairing refused: {exc}\n')
    references, pairing = record['references'], record['pairing']
    print(f"paired with {references['knn']['protocol_id']} at {references['knn']['code_revision']} and "
          f"{references['training']['protocol_id']} at {references['training']['code_revision']}: "
          f"{len(pairing['datasets'])} datasets, fitting seeds {pairing['fit_seeds']}, "
          f"{len(pairing['shared_sources'])} shared sealed sources identical")
