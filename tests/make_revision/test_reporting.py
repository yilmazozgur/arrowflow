"""The manuscript summaries must agree with their underlying predictions."""
import copy
import json
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from experiments.make_revision import evaluation as e
from experiments.make_revision import run_revision as runner
from experiments.make_revision.reporting import collect_verified_results


def report_factory(config, seed):
    if config.get('fail'):
        raise ValueError('Preserved unsuccessful candidate')
    return DummyClassifier(strategy='constant', constant=config['constant'])


def report_registry(protocol):
    candidates = [{'constant': i} for i in range(3)] + [{'constant': 0, 'tag': 1}, {'fail': True}]
    return {'constant': e.ModelSpec('constant', report_factory, candidates, protocol.get('fixture_stochastic', True))}


@pytest.fixture
def evidence(tmp_path, monkeypatch, request):
    X = np.arange(180, dtype=float).reshape(60, 3)
    y = np.array([0]*30 + [1]*18 + [2]*12)
    p = json.loads(runner.PROTOCOL.read_text())
    p.update(frozen=True, datasets=['iris'], outer_folds=3, outer_repeats=1, inner_folds=2)
    p['fixture_stochastic'] = getattr(request, 'param', True)
    features, labels = ['a', 'b', 'c'], ['0', '1', '2']
    manifest = {'dataset_id': 'iris', 'purpose': 'synthetic_test_only',
                'feature_names': features, 'label_map': labels,
                'dataset_hash': e.dataset_fingerprint(X, y, features, labels)}
    monkeypatch.setattr(runner, 'load_dataset', lambda _: (X, y, manifest.copy()))
    registry_path = 'tests.make_revision.test_reporting:report_registry'
    registry = runner.prepare(tmp_path, ['iris'], p, registry_path)
    runner.write_json(tmp_path/'planned_jobs.json', runner.planned_jobs(['iris'], p, registry))
    for fold in range(3):
        runner._worker((str(tmp_path), 'iris', fold, 'constant', registry_path))
    return tmp_path, p, registry


def rewrite_first(output, transform):
    job = json.loads((output/'planned_jobs.json').read_text())[0]
    path = output/job['result_file']
    value = json.loads(path.read_text())
    transform(value)
    path.write_text(json.dumps(value))
    (output/job['log_file']).write_text('\n'.join(e.canonical_json(row)
        for row in value['selection']['fits'] + value['models']) + '\n')


def test_modified_metrics_cannot_override_saved_predictions(evidence):
    output, p, registry = evidence
    rewrite_first(output, lambda result: result['models'][0].update(accuracy=.999))
    with pytest.raises(ValueError, match='metric|prediction'):
        collect_verified_results(output, ['iris'], p, registry)


@pytest.mark.parametrize('change', [
    'missing_prediction', 'duplicate_prediction', 'prediction_truth', 'prediction_seed',
    'prediction_config', 'prediction_revision', 'model_revision', 'outer_training_rows',
    'inner_training_rows', 'missing_failed_candidate', 'missing_rerank_fit',
    'changed_finalists', 'changed_selection_score', 'coherent_wrong_winner', 'unknown_prediction_label',
])
def test_consistent_log_edits_cannot_change_the_declared_evidence(evidence, change):
    output, p, registry = evidence
    def corrupt(result):
        selection = result['selection']
        pred, model = result['predictions'][0], result['models'][0]
        if change == 'missing_prediction': result['predictions'].pop()
        elif change == 'duplicate_prediction': result['predictions'][-1] = copy.deepcopy(pred)
        elif change == 'prediction_truth': pred['y_true'] = (pred['y_true'] + 1) % 3
        elif change == 'prediction_seed': pred['model_seed'] = 999
        elif change == 'prediction_config': pred['config_id'] = 'changed'
        elif change == 'prediction_revision': pred['code_revision'] = 'changed'
        elif change == 'model_revision': model['code_revision'] = 'changed'
        elif change == 'outer_training_rows': model['fit_rows'][0] = model['test_rows'][0]
        elif change == 'inner_training_rows': selection['fits'][0]['fit_rows'][0] = model['test_rows'][0]
        elif change == 'missing_failed_candidate': selection['fits'] = [r for r in selection['fits'] if r['status'] != 'failed']
        elif change == 'missing_rerank_fit': selection['fits'].pop()
        elif change == 'changed_finalists': selection['finalist_ids'].reverse()
        elif change == 'changed_selection_score': selection['inner_score'] = .999
        elif change == 'unknown_prediction_label': pred['y_pred'] = 99
        elif change == 'coherent_wrong_winner':
            config = {'constant': 1}; cid = e.config_id(config)
            selection.update(config=config, config_id=cid, inner_score=.3)
            for row in result['models']: row.update(config=config, config_id=cid)
            for row in result['predictions']: row['config_id'] = cid
    rewrite_first(output, corrupt)
    with pytest.raises(ValueError):
        collect_verified_results(output, ['iris'], p, registry)


@pytest.mark.parametrize('evidence', [True, False], indirect=True)
def test_verified_summary_uses_predictions_and_keeps_unsuccessful_candidates(evidence):
    from experiments.make_revision.reporting import summarize_verified_results, main
    output, p, registry = evidence
    summary = summarize_verified_results(output)
    values = {row['metric']: row for row in summary['summaries']['iris']}
    assert {metric: row['mean'] for metric, row in values.items()} == pytest.approx(
        {'accuracy': .5, 'error': .5, 'balanced_accuracy': 1/3, 'macro_f1': 2/9})
    assert all(row['n_folds'] == 3 for row in values.values())
    assert all(row['seeds_per_fold'] == (3 if p['fixture_stochastic'] else 1) for row in values.values())
    assert all(row['outer_fold_sd'] == 0 for row in values.values())
    assert summary['hypothesis_tests'] == []
    main(['--output', str(output)])
    assert json.loads((output/'summary.json').read_text()) == summary


@pytest.mark.parametrize('change', ['unfrozen', 'partial_panel', 'candidate_registry', 'source_hash', 'prepared_truth', 'reordered_splits'])
def test_reporting_rejects_changed_prepared_inputs(evidence, change):
    output, p, registry = evidence
    if change == 'unfrozen':
        p['frozen'] = False
        (output/'protocol.json').write_text(json.dumps(p))
    elif change == 'partial_panel': p['datasets'] = ['iris', 'wine']
    elif change == 'candidate_registry':
        path = output/'candidates.json'; value = json.loads(path.read_text())
        value['constant']['candidates'].pop(); path.write_text(json.dumps(value))
    elif change == 'source_hash':
        path = output/'environment.json'; value = json.loads(path.read_text())
        value['source_hashes'][next(iter(value['source_hashes']))] = 'changed'
        path.write_text(json.dumps(value))
    elif change == 'prepared_truth':
        path = output/'iris'/'data.npz'
        with np.load(path) as archive: X, y = archive['X'], archive['y']
        y[0] = 1; np.savez_compressed(path, X=X, y=y)
    elif change == 'reordered_splits':
        path = output/'iris'/'splits.json'; splits = json.loads(path.read_text())[::-1]
        path.write_text(json.dumps(splits))
        manifest_path = output/'iris'/'manifest.json'
        manifest = json.loads(manifest_path.read_text()); manifest['splits_hash'] = e.config_id(splits)
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): collect_verified_results(output, ['iris'], p, registry)
