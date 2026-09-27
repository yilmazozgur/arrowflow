"""Task 20B: the component ablation anchored on ArrowFlow-kNN (run_knn_ablation) and the knn_ablation protocol.

One synthetic smoke run (a synthetic bridge_knn reference built with the real harness, then prepare, run and summary) is
built once per session; tests that alter records work on copies.
"""
from collections import Counter
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.datasets import load_iris
from experiments.make_revision import run_knn_ablation as ra
from experiments.make_revision.bridge import bridge_candidates, resolve_selected
from experiments.make_revision.evaluation import config_id
from experiments.make_revision.knn_controls import MultiViewInputKNN, UntrainedMultiViewArrowFlowKNN
from experiments.make_revision.models import array_hash, seed_fit
from experiments.make_revision.multiview import MultiViewArrowFlow, MultiViewArrowFlowKNN
from experiments.make_revision.run_revision import load_prepared

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'/'2026-09-12'
REAL_KNN = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'/'2026-09-12-bridge-knn'
needs_real_knn = pytest.mark.skipif(not REAL_KNN.is_dir(), reason='the bridge_knn run is not on this machine')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='session')
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('knn_ablation_smoke')
    report = ra.smoke(root, json.loads((PROTOCOLS/'knn_ablation.json').read_text()), workers=2)
    return SimpleNamespace(root=root, report=report)


def prepared_copy(smoke_run, target):
    """The prepared files of the smoke run without any job output: a fresh output directory."""
    target.mkdir(parents=True)
    for name in ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json',
                 'reference_selected_configurations.csv'):
        shutil.copyfile(smoke_run.root/name, target/name)
    shutil.copytree(smoke_run.root/'synthetic', target/'synthetic')
    return target


def flip_reference_label(selections, index=0, seed='8129'):
    labels = selections[index]['reference_predictions'][seed]
    labels[0] = (labels[0] + 1) % 3
    selections[index]['reference_prediction_hashes'][seed] = array_hash(np.asarray(labels))


def reseal_selections(directory, change):
    """Change the sealed selections and rewrite their CSV consistently, as a deliberate edit would."""
    path = directory/'reference_selections.json'
    selections = json.loads(path.read_text())
    change(selections)
    path.write_text(json.dumps(selections, indent=2, sort_keys=True) + '\n')
    (directory/'reference_selected_configurations.csv').unlink()
    ra.write_csv(directory/'reference_selected_configurations.csv', ra.SELECTION_COLUMNS, ra.selection_rows(selections))


# ----------------------------------------------------------------------------- plan

def test_variants_and_fit_sources_at_a_resolved_selection():
    selected = resolve_selected(bridge_candidates()[0], 19, 1848)                     # segment-like: augmentation on
    variants = ra.knn_ablation_variants(selected)
    params = dict(variants)
    base = {k: v for k, v in selected.items() if k not in ('embed_scale', 'degree_offset')}
    assert [v for v, _ in variants] == list(ra.VARIANTS) and params['views7'] == params['prototype_readout'] == base
    assert (params['views1']['n_views'], params['views3']['n_views']) == (1, 3)
    assert params['no_checkpoint'] == {**base, 'validation_ratio': 0} and params['no_augment'] == {**base, 'augment': False}
    assert params['untrained'] == {k: base[k] for k in ra.UNTRAINED_KEYS} and params['input_knn'] == {k: base[k] for k in ra.INPUT_KEYS}
    assert ra.fit_sources(variants) == {'views7': 'fitted', 'views1': 'prefix_of_views7', 'views3': 'prefix_of_views7',
                                        'prototype_readout': 'output_rule_of_views7', 'no_checkpoint': 'separate',
                                        'no_augment': 'separate', 'untrained': 'separate_untrained', 'input_knn': 'separate_input'}
    assert ra.fit_sources(ra.knn_ablation_variants(resolve_selected(bridge_candidates()[0], 4, 120)))['no_augment'] == 'identical_to_views7'
    MultiViewArrowFlowKNN(**params['no_checkpoint']); UntrainedMultiViewArrowFlowKNN(**params['untrained'])
    MultiViewInputKNN(**params['input_knn'])
    with pytest.raises(ValueError, match='Resolve'):
        ra.knn_ablation_variants(bridge_candidates()[0])
    with pytest.raises(ValueError, match='seven-view'):
        ra.knn_ablation_variants({**selected, 'n_views': 3})


def test_derived_variants_equal_the_estimators_at_smaller_view_counts_and_the_output_rule_of_the_same_networks():
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(4).permutation(len(y))
    Xtr, ytr, Xte = X[idx[:90]], y[idx[:90]], X[idx[90:]]
    params = dict(n_views=7, strategy='diverse', embed_dim=16, degree=1, widths=[16], iterations=3, batch_size=32,
                  learning_rate=.1, validation_ratio=.1, augment=False, aggregation='majority')
    model, fit_seconds, knn_views, output_views = ra.fit_views7(params, 8129, Xtr, ytr, Xte)
    derived = ra.derived_predictions(knn_views, output_views)
    assert fit_seconds > 0 and knn_views.shape == output_views.shape == (7, len(Xte))
    assert np.array_equal(derived['views7'], model.predict(Xte))
    for name, k in ra.PREFIXES.items():
        seed_fit(8129)
        assert np.array_equal(derived[name], MultiViewArrowFlowKNN(**{**params, 'n_views': k}, seed=8129).fit(Xtr, ytr).predict(Xte))
    seed_fit(8129)
    assert np.array_equal(derived['prototype_readout'], MultiViewArrowFlow(**params, seed=8129).fit(Xtr, ytr).predict(Xte))
    with pytest.raises(ValueError, match='seven'):
        ra.derived_predictions(knn_views[:3], output_views[:3])


def test_reproduction_check_requires_the_reference_labels_in_the_sealed_order():
    labels = [0, 1, 2, 1]
    sealed = {'reference_predictions': {'5': labels}, 'reference_prediction_hashes': {'5': array_hash(np.asarray(labels))}}
    assert ra.reproduction_check(np.asarray(labels), sealed, 5) == {
        'model_seed': 5, 'n_test': 4, 'n_differing': 0, 'views7_prediction_hash': sealed['reference_prediction_hashes']['5'],
        'reference_prediction_hash': sealed['reference_prediction_hashes']['5'], 'reproduced': True}
    assert ra.reproduction_check(np.asarray([0, 1, 1, 2]), sealed, 5)['n_differing'] == 2
    assert not ra.reproduction_check(np.asarray([0, 1, 1, 2]), sealed, 5)['reproduced']
    assert not ra.reproduction_check(np.asarray([0, 1, 2]), sealed, 5)['reproduced']


# ----------------------------------------------------------------------------- selections and the smoke run

def test_reference_selections_are_reconstructed_from_the_complete_inner_history_and_sealed(smoke_run, tmp_path):
    root, reference_dir = smoke_run.root, smoke_run.root/'synthetic_reference'
    reference = ra.load_reference(reference_dir)
    X, y, manifest, splits = load_prepared(reference_dir, 'synthetic')
    sealed = json.loads((root/'reference_selections.json').read_text())
    assert [(s['outer_repeat'], s['outer_fold']) for s in sealed] == [(0, 0), (0, 1), (0, 2)]
    for split, record in zip(splits, sealed):
        result = json.loads((reference_dir/record['result_file']).read_text())
        assert (record['config_id'], record['config']) == (result['selection']['config_id'], result['selection']['config'])
        assert record['selected_widths'] == result['selection']['config']['widths']
        for seed in record['fitting_seeds']:
            labels = {r['sample_id']: r['y_pred'] for r in result['predictions'] if r['model_seed'] == seed}
            assert record['reference_predictions'][str(seed)] == [labels[s] for s in split['test']]
        assert ra.selection_record(reference, 'synthetic', split, y, manifest) == record
    lines = (root/'reference_selected_configurations.csv').read_text().splitlines()
    assert lines[0] == ','.join(ra.SELECTION_COLUMNS) and len(lines) == 4
    copy = tmp_path/'reference'                           # a recorded selection that disagrees with its inner history
    shutil.copytree(reference_dir, copy)
    path = copy/sealed[0]['result_file']
    result = json.loads(path.read_text())
    other = next(c for c in ra.SMOKE_CANDIDATES if config_id(c) != result['selection']['config_id'])
    result['selection'].update(config=other, config_id=config_id(other))
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Selected configuration ID'):
        ra.selection_record(ra.load_reference(copy), 'synthetic', splits[0], y, manifest)
    p = json.loads((root/'protocol.json').read_text())
    with pytest.raises(ValueError, match='summary_sha256'):
        ra.check_reference({**p, 'reference_source': {**p['reference_source'], 'summary_sha256': '0' * 64}}, reference)
    with pytest.raises(ValueError, match='inner_folds'):
        ra.check_reference({**p, 'inner_folds': 3}, reference)


def test_smoke_fits_every_variant_derives_the_reused_ones_and_reproduces_views7(smoke_run):
    report, root = smoke_run.report, smoke_run.root
    table = report['summaries']['synthetic']
    assert list(table['variants']) == list(ra.VARIANTS) and len(report['model_rows']['synthetic']) == 3 * 8 * 3
    assert table['views7_reproduces_reference'] == {'matching_fold_seeds': 9, 'total_fold_seeds': 9}
    assert 'change_from_views7' not in table['variants']['views7'] and report['inferential_significance_claims'] is False
    assert all('p_approximate' not in e['change_from_views7']['accuracy'] and e['change_from_views7']['accuracy']['df'] == 2
               for v, e in table['variants'].items() if v != 'views7')
    jobs = json.loads((root/'planned_jobs.json').read_text())
    assert len(jobs) == 3 and all(j['selected']['augment'] is True and j['fit_sources']['no_augment'] == 'separate' for j in jobs)
    result = json.loads((root/'results'/'synthetic__r0f0.json').read_text())
    reused = [f for f in result['fits'] if f['fit_source'] in ra.REUSED_SOURCES]
    assert sorted({f['variant_id'] for f in reused}) == ['prototype_readout', 'views1', 'views3'] and len(reused) == 9
    assert all(f['fit_seconds'] == 0 and f['reused_from'] == f"views7__s{f['model_seed']}" for f in reused)
    assert all(f['fit_seconds'] > 0 and len(f['readout_choices']) == 7 for f in result['fits'] if f['fit_source'] not in ra.REUSED_SOURCES)
    assert [c['reproduced'] for c in result['reproduction']] == [True] * 3
    depth = report['depth_split']
    assert depth['depths'] == [[4], [2, 4]] and sum(e['n_folds'] for e in depth['by_dataset']['synthetic']) == 3
    assert depth['difference'].startswith('untrained minus views7') and [e['widths'] for e in depth['pooled']] == [[4], [2, 4]]
    record = json.loads((root/'predictions'/'synthetic__r0f0.jsonl').read_text().splitlines()[0])
    assert sorted(record) == sorted(ra.PREDICTION_KEYS)
    assert (root/'knn_ablation_summary.csv').read_text().splitlines()[0] == ','.join(ra.SUMMARY_COLUMNS)
    assert json.loads((root/'knn_ablation_summary.json').read_text()) == report


def test_views7_reproduction_is_a_hard_check_in_the_job_the_run_and_the_summary(smoke_run, tmp_path):
    prepared = prepared_copy(smoke_run, tmp_path/'prepared')
    p, job = json.loads((prepared/'protocol.json').read_text()), json.loads((prepared/'planned_jobs.json').read_text())[0]
    X, y, data, splits = load_prepared(prepared, 'synthetic')
    sealed = json.loads((prepared/'reference_selections.json').read_text())
    flip_reference_label(sealed)
    result = ra.evaluate_job(X, y, splits[0], p, job, sealed[0], dataset_hash=data['dataset_hash'], code_revision='test')
    assert result['status'] == 'failed' and result['reproduction_failed'] and 'views7 does not reproduce' in result['exception']
    assert [(c['model_seed'], c['n_differing'], c['reproduced']) for c in result['reproduction']] == [(8129, 1, False)]
    assert result['fits'] == [] and all(r['status'] == 'failed' for r in result['models'])        # stopped at its first seed
    assert {(r['variant_id'], r['model_seed']) for r in result['models']} == {(v, s) for v in ra.VARIANTS for s in p['fit_seeds']}
    reseal_selections(prepared, flip_reference_label)
    with pytest.raises(ra.ReproductionError, match='did not reproduce'):
        ra.run(prepared, 1, allow_smoke=True)
    assert json.loads((prepared/'results'/'synthetic__r0f0.json').read_text())['reproduction_failed'] is True
    with pytest.raises(ValueError, match='Incomplete kNN ablation evidence'):
        ra.collect_results(prepared, allow_smoke=True)
    complete = tmp_path/'complete'                        # a sealed selection edited consistently after a complete run
    shutil.copytree(smoke_run.root, complete)
    reseal_selections(complete, flip_reference_label)
    with pytest.raises(ValueError, match='re-derived from the reference run'):
        ra.collect_results(complete, allow_smoke=True)


def test_summary_refuses_tampered_predictions_artifacts_and_sources(smoke_run, tmp_path):
    run = tmp_path/'run'
    shutil.copytree(smoke_run.root, run)
    path = run/'predictions'/'synthetic__r0f1.jsonl'
    original = path.read_text()
    lines = original.splitlines()
    first = json.loads(lines[0])
    first['y_pred'] = (first['y_pred'] + 1) % 3
    path.write_text('\n'.join([json.dumps(first, sort_keys=True, separators=(',', ':'))] + lines[1:]) + '\n')
    with pytest.raises(ValueError, match='prediction file hash'):
        ra.collect_results(run, allow_smoke=True)
    path.write_text(original)
    artifact = run/'artifacts'/'synthetic__r0f2'/'views'/'s19391.npz'
    blob = artifact.read_bytes()
    artifact.unlink()
    with pytest.raises(ValueError, match='changed artifact'):
        ra.collect_results(run, allow_smoke=True)
    artifact.write_bytes(blob)
    environment = json.loads((run/'environment.json').read_text())
    environment['source_hashes']['experiments/make_revision/knn_controls.py'] = '0' * 64
    (run/'environment.json').write_text(json.dumps(environment))
    with pytest.raises(ValueError, match='seal changed'):
        ra.collect_results(run, allow_smoke=True)
    shutil.copyfile(smoke_run.root/'environment.json', run/'environment.json')
    assert set(ra.collect_results(run, allow_smoke=True)['rows']) == {'synthetic'}


def test_runtime_pilot_times_every_variant_on_training_rows_and_reproduces_a_reference_inner_fit(smoke_run, tmp_path):
    p = json.loads((smoke_run.root/'protocol.json').read_text())
    report = ra.runtime_pilot(tmp_path/'pilot', p, smoke_run.root/'synthetic_reference')
    record = report['records'][0]
    assert set(record['seconds_by_variant']) == {'views7', 'no_checkpoint', 'no_augment', 'untrained', 'input_knn'}
    assert set(record['query_ids']) <= set(record['train_ids']) and record['network_fits_per_seed'] == 21
    probe = report['reproduction_probe']
    assert probe['reproduced'] is True and probe['refit_score'] == probe['reference_score'] and probe['readout_selections_identical']
    calibrated = report['calibrated_projection']['datasets']['synthetic']
    assert np.isclose(calibrated['seconds'], calibrated['reference_views7_seconds'] * record['all_variants_to_views7_ratio'])
    assert report['harness_projection']['hours_at_16_workers_ideal'] > 0 and report['wallclock_cap_hours'] == 4


# ----------------------------------------------------------------------------- protocol and the real reference

def test_knn_ablation_protocol_declares_the_ruled_variants_and_pins_the_reference_design():
    p = json.loads((PROTOCOLS/'knn_ablation.json').read_text())
    old, knn = json.loads((PROTOCOLS/'ablation.json').read_text()), json.loads((PROTOCOLS/'bridge_knn.json').read_text())
    assert p['protocol_id'] == 'arrowflow-v3-knn-ablation-1' and p['production_family'] == 'knn_ablation'
    assert p['variants'] == list(ra.VARIANTS) and set(p['variant_definitions']) == set(ra.VARIANTS)
    assert set(p['dropped_variants']) == {'borda_views7', 'single_view_no_checkpoint_no_augment', 'multiview_footrule_knn'}
    assert 'contrast_family' not in p and 'no Holm' in p['reporting']['change_from_views7']
    assert p['wallclock_cap_hours'] == 4 and p['max_workers'] == 16 and set(p['pilot_datasets']) <= set(p['datasets'])
    for key in ('datasets', 'fit_seeds', 'split_seed', 'outer_folds', 'outer_repeats', 'inner_folds', 'test_train_ratio', 'confidence'):
        assert p[key] == old[key] == knn[key], key
    assert p['knn_readout']['grid'] == knn['knn_readout']['grid'] and p['knn_readout']['selection'] == knn['knn_readout']['selection']
    reference = p['reference_source']
    assert reference['protocol_sha256'] == sha256(PROTOCOLS/'bridge_knn.json') and reference['protocol_id'] == knn['protocol_id']
    assert (reference['family'], reference['model_id']) == ('bridge_knn', 'arrowflow_full_knn')
    assert p['depth_split']['depths'] == [[128], [64, 128]] and p['source_template_sha256'] == sha256(PROTOCOLS/'ablation.json')
    assert ('frozen_at_utc' in p) == bool(p['frozen'])


@needs_real_knn
def test_prepare_reconstructs_the_real_reference_selections_and_the_pins_match(tmp_path):
    p = json.loads((PROTOCOLS/'knn_ablation.json').read_text())
    reference = ra.load_reference(REAL_KNN)
    assert ra.check_reference(p, reference)['summary_sha256'] == sha256(REAL_KNN/'summary.json')
    jobs = ra.prepare(tmp_path, dict(p, datasets=['iris']), REAL_KNN)
    sealed = json.loads((tmp_path/'reference_selections.json').read_text())
    assert len(jobs) == len(sealed) == 15
    for job, record in zip(jobs, sealed):
        assert job['config'] == record['config'] == json.loads((REAL_KNN/record['result_file']).read_text())['selection']['config']
        assert job['selected']['augment'] is False and job['fit_sources']['no_augment'] == 'identical_to_views7'   # 120 rows
    assert Counter(tuple(r['selected_widths']) for r in sealed) == {(128,): 8, (64, 128): 7}
