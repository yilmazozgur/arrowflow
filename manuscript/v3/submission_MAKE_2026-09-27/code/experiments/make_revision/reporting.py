"""Reconstruct main/native benchmark summaries from complete saved evidence.

Inner selection is reconstructed from its full fit/score history. Outer metrics
are independently recomputed from per-example predictions and prepared truth.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .evaluation import (canonical_json, config_id, expected_schedule, make_splits,
                         metric_values, summarize_outer, validate_split)


METRICS = ('accuracy', 'error', 'balanced_accuracy', 'macro_f1')


def _equal(actual, expected, description):
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(f'{description} mismatch')


def _score(value, description):
    if not isinstance(value, (float, int)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'Invalid {description}')
    return float(value)


def _selected_from_history(selection, spec, split, protocol):
    candidates = {config_id(config): config for config in spec.candidates}
    seeds = protocol['fit_seeds']
    rows = selection['fits']
    indexed = {}
    for row in rows:
        cid, seed, inner_fold = row['config_id'], row['model_seed'], row['inner_fold']
        if cid not in candidates or seed not in seeds or inner_fold not in range(len(split['inner'])):
            raise ValueError('Unexpected inner fit identity')
        key = (cid, seed, inner_fold)
        if key in indexed:
            raise ValueError('Duplicate inner fit')
        indexed[key] = row
        inner = split['inner'][inner_fold]
        for field, value in {'stage': 'inner', 'model_id': spec.model_id,
                             'outer_repeat': split['outer_repeat'], 'outer_fold': split['outer_fold'],
                             'config': candidates[cid], 'fit_rows': inner['train'],
                             'validation_rows': inner['validation']}.items():
            _equal(row[field], value, f'Inner {field}')
        if row['status'] == 'ok':
            _score(row['score'], 'inner score')
        elif row['status'] != 'failed' or row['score'] is not None or not row.get('exception'):
            raise ValueError('Invalid inner failure record')
    screen_keys = {(cid, seeds[0], fold) for cid in candidates for fold in range(len(split['inner']))}
    if not screen_keys <= indexed.keys():
        raise ValueError('Incomplete candidate screening history')

    def rank(ids, active_seeds):
        scored = []
        for cid in ids:
            required = [(cid, seed, fold) for seed in active_seeds for fold in range(len(split['inner']))]
            if any(key not in indexed for key in required):
                raise ValueError('Incomplete candidate reranking history')
            records = [indexed[key] for key in required]
            if all(row['status'] == 'ok' for row in records):
                scored.append((-float(np.mean([row['score'] for row in records])), cid))
        return sorted(scored)

    screened = rank(candidates, seeds[:1])
    finalists = [cid for _, cid in (screened[:3] if spec.stochastic else screened)]
    _equal(selection['finalist_ids'], finalists, 'Selected finalists')
    required = screen_keys | ({(cid, seed, fold) for cid in finalists for seed in seeds[1:]
                              for fold in range(len(split['inner']))} if spec.stochastic else set())
    if indexed.keys() != required:
        raise ValueError('Incomplete or unexpected inner fit schedule')
    ranked = rank(finalists, seeds if spec.stochastic else seeds[:1])
    if not ranked:
        raise ValueError('No successful selected candidate')
    negative_score, cid = ranked[0]
    _equal(selection['config_id'], cid, 'Selected configuration ID')
    _equal(selection['config'], candidates[cid], 'Selected configuration')
    if not np.isclose(_score(selection['inner_score'], 'selection score'), -negative_score, rtol=0, atol=1e-12):
        raise ValueError('Selected inner score mismatch')
    return cid, candidates[cid]


def validate_result_records(result, job, split, y, manifest, spec, protocol, revision):
    """Return model rows with metrics replaced by values from saved predictions."""
    validate_split(split, len(y))
    cid, config = _selected_from_history(result['selection'], spec, split, protocol)
    common = {key: job[key] for key in ('dataset_id', 'model_id', 'outer_repeat', 'outer_fold')}
    common.update(dataset_hash=manifest['dataset_hash'], code_revision=revision,
                  config_id=cid, view_id='ensemble', condition='clean', perturbation_seed=None)
    expected_seeds = protocol['fit_seeds'] if spec.stochastic else protocol['fit_seeds'][:1]
    expected_prediction_keys = {(seed, sample) for seed in expected_seeds for sample in split['test']}
    predictions = {}
    known_labels = set(np.asarray(y).tolist())
    for row in result['predictions']:
        for field, value in common.items():
            _equal(row[field], value, f'Prediction {field}')
        key = (row['model_seed'], row['sample_id'])
        if key in predictions or key not in expected_prediction_keys:
            raise ValueError('Duplicate or unexpected prediction seed/sample')
        if row['y_true'] != y[row['sample_id']] or row['y_pred'] not in known_labels:
            raise ValueError('Prediction truth or label mismatch')
        predictions[key] = row['y_pred']
    if predictions.keys() != expected_prediction_keys:
        raise ValueError('Incomplete per-example prediction schedule')
    verified, observed = [], set()
    for row in result['models']:
        seed = row['model_seed']
        if seed not in expected_seeds or seed in observed:
            raise ValueError('Duplicate or unexpected model seed')
        observed.add(seed)
        for field, value in dict(common, config=config, fit_rows=split['train'],
                                 test_rows=split['test'], stage='outer', status='ok').items():
            _equal(row[field], value, f'Outer {field}')
        metrics = metric_values(y[split['test']], [predictions[seed, sample] for sample in split['test']])
        for metric, value in metrics.items():
            if not np.isclose(_score(row[metric], metric), value, rtol=0, atol=1e-12):
                raise ValueError(f'Outer metric/prediction mismatch: {metric}')
        verified.append(dict(row, **metrics))
    if observed != set(expected_seeds):
        raise ValueError('Incomplete model seed schedule')
    return verified


def collect_verified_results(output, names, protocol, registry):
    """Validate complete declared panels; never infer a smaller panel from files."""
    from . import run_revision as runner
    output = Path(output)
    _equal(json.loads((output/'protocol.json').read_text()), protocol, 'Saved protocol')
    if not protocol.get('frozen'):
        raise ValueError('Reporting requires a frozen protocol')
    _equal(list(names), protocol['datasets'], 'Declared dataset panel')
    candidates = {name: {'stochastic': spec.stochastic, 'candidates': spec.candidates,
                         'config_ids': [config_id(c) for c in spec.candidates]}
                  for name, spec in registry.items()}
    _equal(json.loads((output/'candidates.json').read_text()), candidates, 'Saved candidate registry')
    environment = json.loads((output/'environment.json').read_text())
    current = runner.environment_record(environment['registry'])
    _equal(current['source_hashes'], environment['source_hashes'], 'Analysis source hashes')
    # A later manuscript-only commit may change HEAD. Predictions must remain
    # bound to the original recorded revision and the same scientific sources.
    runner.collect_confirmatory_results(output, names, protocol, registry)
    prepared = {name: runner.load_prepared(output, name) for name in names}
    for name, (_, y, manifest, splits) in prepared.items():
        if manifest.get('dataset_id', name) != name:
            raise ValueError('Prepared dataset ID mismatch')
        _equal(splits, make_splits(y, protocol['outer_folds'], protocol['outer_repeats'],
                                  protocol['inner_folds'], protocol['split_seed']), 'Declared nested splits')
    rows = {name: [] for name in names}
    for job in runner.planned_jobs(names, protocol, registry):
        _, y, manifest, splits = prepared[job['dataset_id']]
        split = splits[job['outer_repeat']*protocol['outer_folds'] + job['outer_fold']]
        result = json.loads((output/job['result_file']).read_text())
        try:
            verified = validate_result_records(result, job, split, y, manifest,
                                                registry[job['model_id']], protocol,
                                                environment['code_revision'])
        except (KeyError, TypeError, IndexError, ValueError) as exc:
            raise ValueError(f'{job["result_file"]}: {exc}') from exc
        rows[job['dataset_id']].extend(verified)
    return rows


def summarize_verified_results(output):
    """Fold means and seed variation, without unplanned hypothesis tests."""
    from . import run_revision as runner
    output = Path(output)
    protocol = json.loads((output/'protocol.json').read_text())
    environment = json.loads((output/'environment.json').read_text())
    registry = runner.get_registry(environment['registry'], protocol)
    rows = collect_verified_results(output, protocol['datasets'], protocol, registry)
    schedule = expected_schedule(protocol, registry)
    summaries = {name: [summarize_outer(model_rows, model, metric,
                          expected_folds=schedule['expected_folds'],
                          expected_seeds=schedule['expected_seeds'][model])
                       for model in registry for metric in METRICS]
                 for name, model_rows in rows.items()}
    return {'purpose': 'complete_nested_benchmark_metrics_recomputed_from_predictions',
            'code_revision': environment['code_revision'], 'summaries': summaries,
            'selection_audit': 'Reconstructed from all recorded inner scores; inner predictions were not saved.',
            'hypothesis_tests': [], 'model_rows': rows}


def main(argv=None):
    from .run_revision import write_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    write_json(args.output/'summary.json', summarize_verified_results(args.output))


if __name__ == '__main__':
    main()
