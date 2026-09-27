"""Task 24 Part 2: run_newdata_ablation, the kNN-anchored component ablation on the ten further datasets.

One synthetic smoke run (two synthetic newdata batch references built with the real harness, augmentation on in the first
and off in the second, then prepare, run and summary) is built once per session; tests that alter records work on copies.
The real batch runs and the cached OpenML downloads are used where they are on this machine."""
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from sklearn.datasets import get_data_home
from experiments.make_revision import newdata as nd
from experiments.make_revision import run_knn_ablation as ra
from experiments.make_revision import run_newdata_ablation as rn
from experiments.make_revision.evaluation import config_id, make_splits
from experiments.make_revision.models import array_hash
from experiments.make_revision.run_revision import load_prepared

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'/'2026-09-12'
REAL_BATCHES = dict(rn.REFERENCE_DIRECTORIES)
needs_real_batches = pytest.mark.skipif(not all((d/'summary.json').is_file() for d in REAL_BATCHES.values()),
                                        reason='the newdata batch runs are not on this machine')


def _cached(pin):
    """test_newdata._cached: the three OpenML files fetch_openml reads from sklearn's data home."""
    home = Path(get_data_home())/'openml'/'openml.org'
    return all(path.is_file() for path in (home/'api'/'v1'/'json'/'data'/f"{pin['data_id']}.gz",
                                           home/'api'/'v1'/'json'/'data'/'features'/f"{pin['data_id']}.gz",
                                           home/'data'/'v1'/'download'/f"{pin['file_id']}.gz"))


needs_cache = pytest.mark.skipif(not all(_cached(nd.PIN_BY_NAME[name]) for name in rn.PROBE_DATASETS.values()),
                                 reason='the OpenML downloads of the probe datasets are not cached here')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='session')
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('newdata_ablation_smoke')
    report = rn.smoke(root, rn.draft_protocol(), workers=2)
    return SimpleNamespace(root=root, report=report,
                           references={batch: root/f'synthetic_reference_batch{batch}' for batch in rn.BATCHES})


def prepared_copy(smoke_run, target):
    target.mkdir(parents=True)
    for name in ('protocol.json', 'environment.json', 'manifest.json', 'planned_jobs.json', 'reference_selections.json',
                 'reference_selected_configurations.csv'):
        shutil.copyfile(smoke_run.root/name, target/name)
    for name in rn.SMOKE_DATASETS.values():
        shutil.copytree(smoke_run.root/name, target/name)
    return target


def flip_reference_label(selections, index=-1, seed='8129'):
    labels = selections[index]['reference_predictions'][seed]
    labels[0] = (labels[0] + 1) % 3
    selections[index]['reference_prediction_hashes'][seed] = array_hash(np.asarray(labels))


def reseal_selections(directory, change):
    path = directory/'reference_selections.json'
    selections = json.loads(path.read_text())
    change(selections)
    path.write_text(json.dumps(selections, indent=2, sort_keys=True) + '\n')
    (directory/'reference_selected_configurations.csv').unlink()
    ra.write_csv(directory/'reference_selected_configurations.csv', ra.SELECTION_COLUMNS, ra.selection_rows(selections))


# ----------------------------------------------------------------------------- the protocol

def test_draft_copies_the_knn_ablation_design_and_pins_both_frozen_batch_runs():
    draft, template = rn.draft_protocol(), json.loads((PROTOCOLS/'knn_ablation.json').read_text())
    assert all(draft[key] == template[key] for key in rn.COPIED_KEYS)
    assert set(template) - set(draft) == {'reference_source', 'frozen_at_utc'}
    assert set(draft) - set(template) == {'reference_sources', 'probe_datasets', 'reproduction', 'selected_configuration',
                                          'dataset_loading', 'decision_rule', 'design_source'}
    assert {key for key in set(draft) & set(template) if draft[key] != template[key]} == {
        'protocol_id', 'production_family', 'datasets', 'pilot_datasets', 'knn_readout', 'reporting', 'wallclock_cap_hours',
        'frozen', 'status', 'resource_decision', 'source_template_sha256'}
    assert {k: v for k, v in draft['knn_readout'].items() if k != 'source'} == {k: v for k, v in template['knn_readout'].items() if k != 'source'}
    assert draft['source_template_sha256'] == sha256(PROTOCOLS/'knn_ablation.json') and draft['frozen'] is False
    assert (draft['protocol_id'], draft['datasets'], draft['wallclock_cap_hours']) == ('arrowflow-v3-newdata-ablation-1', list(nd.PANEL), 3)
    assert draft['variants'] == list(ra.VARIANTS) and draft['depth_split']['depths'] == [[128], [64, 128]]
    rows = {pin['name']: pin['shape'][0] for pin in nd.PINS}
    assert draft['pilot_datasets'] == [max(rows, key=rows.get), min(rows, key=rows.get)] == ['mfeat_zernike', 'vertebra_column']
    for batch in rn.BATCHES:
        pins, frozen = draft['reference_sources'][batch], json.loads(nd.PROTOCOL_FILES[int(batch)].read_text())
        assert pins['protocol_sha256'] == sha256(nd.PROTOCOL_FILES[int(batch)]) and pins['protocol_id'] == frozen['protocol_id']
        assert pins['datasets'] == frozen['datasets'] and (pins['model_id'], pins['family']) == ('arrowflow_full_knn', 'newdata')
        assert draft['probe_datasets'][batch] == min(pins['datasets'], key=rows.get)
    assert rn.batch_of(draft) == {name: batch for batch in rn.BATCHES for name in draft['reference_sources'][batch]['datasets']}


def test_the_committed_protocol_is_the_draft_or_its_freeze():
    p = json.loads(rn.PROTOCOL.read_text())
    assert rn.validate_protocol(p) is p
    draft = rn.draft_protocol()
    if p['frozen']:
        assert {k: v for k, v in p.items() if k not in rn.FREEZE_FIELDS} == {k: v for k, v in draft.items() if k not in rn.FREEZE_FIELDS}
        assert p['status'] == rn.FROZEN_STATUS and 0 < p['pilot_projection']['decision_hours'] <= 3
    else:
        assert p == draft


def test_validate_protocol_refuses_another_design_or_an_unrecorded_freeze():
    draft = rn.draft_protocol()
    with pytest.raises(ValueError, match='variants'):
        rn.validate_protocol({**draft, 'variants': draft['variants'][:-1]})
    with pytest.raises(ValueError, match='must equal the draft'):
        rn.validate_protocol({**draft, 'status': 'edited'})
    frozen = {**draft, 'frozen': True, 'frozen_at_utc': '2026-09-14T00:00:00+00:00', 'status': rn.FROZEN_STATUS, 'resource_decision': 'x',
              'pilot_projection': {'cap_hours': 3, 'decision_hours': 1.5}}
    assert rn.validate_protocol(frozen) is frozen
    for change in ({'pilot_projection': {'cap_hours': 3, 'decision_hours': 3.5}}, {'pilot_projection': None}, {'frozen_at_utc': None},
                   {'status': 'drafted'}, {'pilot_projection': {'cap_hours': 4, 'decision_hours': 1.}}):
        with pytest.raises(ValueError, match='frozen protocol'):
            rn.validate_protocol({**frozen, **change})
    with pytest.raises(ValueError, match='partition'):
        rn.batch_of({**draft, 'reference_sources': {**draft['reference_sources'], '2': {**draft['reference_sources']['2'], 'datasets': []}}})


# ----------------------------------------------------------------------------- the smoke run

def test_smoke_fits_every_variant_on_both_references_and_reproduces_views7(smoke_run):
    report, root = smoke_run.report, smoke_run.root
    assert list(report['summaries']) == ['synthetic_b1', 'synthetic_b2'] and report['inferential_significance_claims'] is False
    for name, table in report['summaries'].items():
        assert list(table['variants']) == list(ra.VARIANTS) and len(report['model_rows'][name]) == 3 * 8 * 3
        assert table['views7_reproduces_reference'] == {'matching_fold_seeds': 9, 'total_fold_seeds': 9}
        assert all('p_approximate' not in e['change_from_views7']['accuracy'] for v, e in table['variants'].items() if v != 'views7')
    jobs = json.loads((root/'planned_jobs.json').read_text())
    sources = {name: {j['fit_sources']['no_augment'] for j in jobs if j['dataset_id'] == name} for name in rn.SMOKE_DATASETS.values()}
    assert sources == {'synthetic_b1': {'separate'}, 'synthetic_b2': {'identical_to_views7'}}
    manifest = json.loads((root/'manifest.json').read_text())
    for batch in rn.BATCHES:
        entry = manifest['reference_sources'][batch]
        assert entry['directory'] == str(smoke_run.references[batch].resolve()) and entry['datasets'] == [rn.SMOKE_DATASETS[batch]]
        assert entry['selected_folds'] == 3 and entry['summary_sha256'] == sha256(smoke_run.references[batch]/'summary.json')
    assert report['reference_sources'] == manifest['reference_sources']
    assert report['depth_split']['depths'] == [[4], [2, 4]] and sum(e['n_folds'] for e in report['depth_split']['by_dataset']['synthetic_b2']) == 3
    assert json.loads((root/rn.SUMMARY_JSON).read_text()) == report
    assert (root/rn.SUMMARY_CSV).read_text().splitlines()[0] == ','.join(ra.SUMMARY_COLUMNS)
    result = json.loads((root/'results'/'synthetic_b1__r0f0.json').read_text())
    assert [c['reproduced'] for c in result['reproduction']] == [True] * 3
    assert sorted({f['variant_id'] for f in result['fits'] if f['fit_source'] in ra.REUSED_SOURCES}) == ['prototype_readout', 'views1', 'views3']


def test_selections_come_from_the_batch_run_holding_each_dataset_and_the_references_must_be_the_pinned_ones(smoke_run, tmp_path):
    p = json.loads((smoke_run.root/'protocol.json').read_text())
    sealed = json.loads((smoke_run.root/'reference_selections.json').read_text())
    references, _ = rn.load_references(p, smoke_run.references, allow_smoke=True)
    for record in sealed:
        batch = rn.batch_of(p)[record['dataset_id']]
        X, y, manifest, splits = load_prepared(smoke_run.references[batch], record['dataset_id'])
        split = splits[record['outer_repeat'] * p['outer_folds'] + record['outer_fold']]
        assert ra.selection_record(references[batch], record['dataset_id'], split, y, manifest) == record
        result = json.loads((smoke_run.references[batch]/record['result_file']).read_text())
        assert record['config'] == result['selection']['config']
    swapped = {'1': smoke_run.references['2'], '2': smoke_run.references['1']}
    with pytest.raises(ValueError, match='does not match the protocol pins'):
        rn.load_references(p, swapped, allow_smoke=True)
    with pytest.raises(ValueError, match='one reference run per protocol batch'):
        rn.load_references(p, {'1': smoke_run.references['1']}, allow_smoke=True)
    crossed = dict(p, reference_sources={batch: {**p['reference_sources'][batch], 'datasets': [rn.SMOKE_DATASETS[other]]}
                                         for batch, other in (('1', '2'), ('2', '1'))})
    with pytest.raises(ValueError, match='belong to the reference panel'):
        rn.prepare(tmp_path/'crossed', crossed, smoke_run.references, allow_smoke=True)
    with pytest.raises(ValueError, match='is not a newdata batch run'):
        rn.load_references(p, smoke_run.references)                      # production refuses a synthetic reference


def test_check_dataset_requires_the_declared_splits_and_in_production_the_pins(smoke_run):
    p = json.loads((smoke_run.root/'protocol.json').read_text())
    X, y, manifest, splits = load_prepared(smoke_run.references['1'], 'synthetic_b1')
    assert rn.check_dataset('synthetic_b1', X, y, manifest, splits, p, production=False)['pins_checked'] is False
    with pytest.raises(ValueError, match='declared nested splits'):
        rn.check_dataset('synthetic_b1', X, y, manifest, splits[::-1], p, production=False)
    with pytest.raises(nd.DatasetIdentityError, match='newdata.PINS'):
        rn.check_dataset('synthetic_b1', X, y, manifest, splits, p, production=True)
    pin = nd.PIN_BY_NAME['vertebra_column']
    with pytest.raises(nd.DatasetIdentityError, match='newdata.PINS'):
        rn.check_dataset('vertebra_column', X, y, {**manifest, 'dataset_hash': pin['dataset_hash']}, splits, p, production=True)


def test_views7_reproduction_is_a_hard_check_in_the_job_the_run_and_the_summary(smoke_run, tmp_path):
    prepared = prepared_copy(smoke_run, tmp_path/'prepared')
    p, jobs = json.loads((prepared/'protocol.json').read_text()), json.loads((prepared/'planned_jobs.json').read_text())
    job = jobs[-1]
    X, y, data, splits = load_prepared(prepared, job['dataset_id'])
    sealed = json.loads((prepared/'reference_selections.json').read_text())
    flip_reference_label(sealed)
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (job['outer_repeat'], job['outer_fold']))
    result = ra.evaluate_job(X, y, split, p, job, sealed[-1], dataset_hash=data['dataset_hash'], code_revision='test')
    assert result['status'] == 'failed' and result['reproduction_failed'] and result['fits'] == []
    reseal_selections(prepared, flip_reference_label)
    with pytest.raises(ra.ReproductionError, match='did not reproduce'):
        rn.run(prepared, 1, allow_smoke=True)
    with pytest.raises(ValueError, match='Incomplete newdata ablation evidence'):
        rn.collect_results(prepared, allow_smoke=True)
    complete = tmp_path/'complete'
    shutil.copytree(smoke_run.root, complete)
    reseal_selections(complete, flip_reference_label)
    with pytest.raises(ValueError, match='re-derived from its batch run'):
        rn.collect_results(complete, allow_smoke=True)


def test_summary_refuses_tampered_predictions_sources_and_reference_batches(smoke_run, tmp_path):
    run = tmp_path/'run'
    shutil.copytree(smoke_run.root, run)
    path = run/'predictions'/'synthetic_b2__r0f1.jsonl'
    original = path.read_text()
    lines = original.splitlines()
    first = json.loads(lines[0])
    first['y_pred'] = (first['y_pred'] + 1) % 3
    path.write_text('\n'.join([json.dumps(first, sort_keys=True, separators=(',', ':'))] + lines[1:]) + '\n')
    with pytest.raises(ValueError, match='prediction file hash'):
        rn.collect_results(run, allow_smoke=True)
    path.write_text(original)
    environment = json.loads((run/'environment.json').read_text())
    environment['source_hashes']['experiments/make_revision/newdata.py'] = '0' * 64
    (run/'environment.json').write_text(json.dumps(environment))
    with pytest.raises(ValueError, match='seal changed'):
        rn.collect_results(run, allow_smoke=True)
    shutil.copyfile(smoke_run.root/'environment.json', run/'environment.json')
    manifest = json.loads((run/'manifest.json').read_text())
    manifest['reference_sources']['1']['datasets'] = ['synthetic_b2']
    (run/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='reference batches disagree'):
        rn.collect_results(run, allow_smoke=True)
    shutil.copyfile(smoke_run.root/'manifest.json', run/'manifest.json')
    assert set(rn.collect_results(run, allow_smoke=True)['rows']) == set(rn.SMOKE_DATASETS.values())


# ----------------------------------------------------------------------------- pilot and freeze

def test_runtime_pilot_times_training_rows_only_and_projects_the_calibrated_makespan(smoke_run, tmp_path):
    p = json.loads((smoke_run.root/'protocol.json').read_text())
    report = rn.runtime_pilot(tmp_path/'pilot', p, smoke_run.references, allow_smoke=True)
    records = {r['dataset_id']: r for r in report['records']}
    assert list(records) == p['pilot_datasets'] and report['protocol_hash'] == config_id(p)
    for record in records.values():
        assert set(record['query_ids']) <= set(record['train_ids']) and record['seconds_by_variant']['views7'] > 0
    assert records['synthetic_b2']['seconds_by_variant']['no_augment'] == 0 and records['synthetic_b2']['network_fits_per_seed'] == 14
    assert records['synthetic_b1']['network_fits_per_seed'] == 21
    jobs = json.loads((tmp_path/'pilot'/'planned_jobs.json').read_text())
    selections = {(s['dataset_id'], s['outer_repeat'], s['outer_fold']): s for s in json.loads((tmp_path/'pilot'/'reference_selections.json').read_text())}
    seconds = [sum(selections[(j['dataset_id'], j['outer_repeat'], j['outer_fold'])]['reference_outer_seconds'].values())
               * records[j['dataset_id']]['all_variants_to_views7_ratio'] for j in jobs]
    calibrated = report['calibrated_projection']
    assert calibrated['serial_hours'] == pytest.approx(sum(seconds) / 3600)
    assert calibrated['simulated_makespan_hours'] == pytest.approx(nd.makespan(seconds, 16) / 3600)
    assert calibrated['simulated_makespan_hours'] >= calibrated['serial_hours_over_workers'] - 1e-12
    assert report['decision']['hours'] == calibrated['simulated_makespan_hours'] and report['decision']['within_cap'] is True
    assert set(report['reproduction_probes']) == {'1', '2'} and report['decision']['probes_reproduced'] is True
    one = rn.calibrated_projection(p, jobs, selections, [records['synthetic_b1']])
    assert one['datasets']['synthetic_b2']['basis'].startswith('largest piloted ratio (no pilot dataset shares')


def pilot_record(draft, **decision):
    return {'protocol_hash': config_id(draft),
            'decision': {'rule': draft['decision_rule'], 'hours': 1.25, 'cap_hours': 3, 'within_cap': True, 'probes_reproduced': True, **decision},
            'calibrated_projection': {'serial_hours': 19., 'serial_hours_over_workers': 1.19, 'simulated_makespan_hours': 1.25,
                                      'longest_job_hours': .2, 'datasets': {'balance_scale': {'serial_hours': 1.}}},
            'harness_projection': {'serial_hours': 9., 'hours_at_16_workers_ideal': .56},
            'records': [{'dataset_id': 'mfeat_zernike', 'all_variants_to_views7_ratio': 1.8}],
            'reproduction_probes': {batch: {'dataset_id': name, 'result_file': 'results/x.json', 'config_id': 'c', 'model_seed': 8129,
                                            'inner_fold': 0, 'reference_score': .9, 'refit_score': .9, 'readout_selections_identical': True,
                                            'reproduced': True} for batch, name in rn.PROBE_DATASETS.items()}}


def test_freeze_replaces_the_draft_only_within_the_cap_with_every_probe_reproduced(tmp_path):
    draft = rn.draft_protocol()
    protocol, pilot, stages = tmp_path/'newdata_ablation.json', tmp_path/'pilot.json', tmp_path/'stages.json'
    protocol.write_text(json.dumps(draft, indent=2, sort_keys=True) + '\n')
    stages.write_text(json.dumps({'summary': 'stages at test'}))
    for change, message in (({'hours': 3.2, 'within_cap': False}, 'Not frozen'), ({'probes_reproduced': False}, 'Not frozen')):
        pilot.write_text(json.dumps(pilot_record(draft, **change)))
        with pytest.raises(ValueError, match=message):
            rn.freeze(protocol, pilot, stages, protocol)
    pilot.write_text(json.dumps({**pilot_record(draft), 'protocol_hash': 'other'}))
    with pytest.raises(ValueError, match='did not run with this draft'):
        rn.freeze(protocol, pilot, stages, protocol)
    assert json.loads(protocol.read_text()) == draft
    pilot.write_text(json.dumps(pilot_record(draft)))
    frozen = rn.freeze(protocol, pilot, stages, protocol, frozen_at_utc='2026-09-14T15:00:00+00:00')
    assert json.loads(protocol.read_text()) == frozen and rn.validate_protocol(frozen) is not None
    assert frozen['pilot_projection']['decision_hours'] == 1.25 and frozen['pilot_projection']['pilot_sha256'] == sha256(pilot)
    assert 'calibrated simulated makespan 1.25 h' in frozen['resource_decision'] and not (tmp_path/'newdata_ablation.json.freezing').exists()
    with pytest.raises(ValueError, match='differs from run_newdata_ablation.draft_protocol'):
        rn.freeze(protocol, pilot, stages, tmp_path/'again.json')                     # the frozen file is no longer a draft
    other = tmp_path/'other.json'
    other.write_text('{}')
    draft_copy = tmp_path/'draft.json'
    draft_copy.write_text(json.dumps(draft))
    with pytest.raises(FileExistsError):
        rn.freeze(draft_copy, pilot, stages, other)


# ----------------------------------------------------------------------------- the real references

@needs_real_batches
@needs_cache
def test_the_real_batch_runs_are_the_pinned_references_and_their_datasets_hold_every_pin():
    p = rn.draft_protocol()
    references, observed = rn.load_references(p, REAL_BATCHES)
    assert {batch: entry['summary_sha256'] for batch, entry in observed.items()} == rn.REFERENCE_SUMMARY_SHA256
    assert {batch: entry['code_revision'] for batch, entry in observed.items()} == dict.fromkeys(rn.BATCHES, rn.REFERENCE_CODE_REVISION)
    for batch, name in rn.PROBE_DATASETS.items():
        X, y, manifest, splits = load_prepared(references[batch]['directory'], name)
        identity = rn.check_dataset(name, X, y, manifest, splits, p, production=True)
        assert identity['pins_checked'] and identity['openml']['data_id'] == nd.PIN_BY_NAME[name]['data_id']
        with pytest.raises(nd.DatasetIdentityError, match='differs from the reference prepared data'):
            rn.check_dataset(name, X, y, manifest, splits, p, production=True,
                             loader=lambda _: (X + 1., y, {'dataset_hash': manifest['dataset_hash']}))
        record = ra.selection_record(references[batch], name, splits[0], y, manifest)
        job = ra.planned_job(name, splits[0], record, X.shape[1])
        assert job['fit_sources']['no_augment'] == ('identical_to_views7' if X.shape[1] > 30 else 'separate')
        assert record['fitting_seeds'] == [8129, 19391, 39019] and len(record['reference_predictions']['8129']) == len(splits[0]['test'])
    wrong = dict(p, reference_sources={**p['reference_sources'], '1': {**p['reference_sources']['1'], 'summary_sha256': '0' * 64}})
    with pytest.raises(ValueError, match='summary_sha256'):
        rn.load_references(wrong, REAL_BATCHES)
    with pytest.raises(ValueError):
        rn.load_references(p, {'1': REAL_BATCHES['2'], '2': REAL_BATCHES['1']})
