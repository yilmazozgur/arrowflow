"""Task 23B: newdata (dataset pins and refusals, the combined registry, the protocols, the stages, the projection and the
synthetic smoke)."""
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import pytest
from scipy import stats
from sklearn.datasets import get_data_home
from sklearn.utils import Bunch
from experiments.make_revision import compare_newdata as cn
from experiments.make_revision import newdata as nd
from experiments.make_revision import run_revision as rr
from experiments.make_revision.bridge import bridge_knn_registry
from experiments.make_revision.evaluation import canonical_json, config_id, dataset_fingerprint, make_splits
from experiments.make_revision.knn_controls import control_candidates, input_factory, untrained_factory
from experiments.make_revision.projected_knn import projected_candidates, projected_factory
from experiments.make_revision.referee_analyses import duplicate_groups

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'/'2026-09-12'
RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'
TEN = list(nd.PANEL)

# task-23-dataset-selection.md, transcribed independently of the module: Section 2 pool table (name and version, n, d,
# class counts, ARFF md5, SHA-256 X, SHA-256 y, duplicate rows, stratum) and Evidence column (gap per source, points).
SELECTION = {
    11: ('balance-scale', 1, 625, 4, [49, 288, 288], '76938608d472f620c170cef9c8c1fa65',
         'd8d20b9bd4e5be6bff5a3ca6a6d81d4fb061084808008d5eac5abd31f8015cb4', '3b6759b01b84ce5b3470db3c17458cc2b2e54fe51d7daca8f2274c4f39a659c7', 0, 'H'),
    22: ('mfeat-zernike', 1, 2000, 47, [200] * 10, '590fe11f6c0eeacd456609f22c4eaf6d',
         '50fabead84e7c390f4f4e06957cb5c0f7fcc99e17b44fc6df9f59b0f7c300d66', 'c67eb11583409b87ed18ed985fe8a480d983f641448c500226dfab7e93a1b952', 6, 'H'),
    59: ('ionosphere', 1, 351, 34, [126, 225], '23dd3c8b5693e2848901fd1a0248ef42',
         '69468e18f3a97d2ed2709a0ad1fb4263f1dbd16f658b15500513ef06ccd7af71', 'a62d1716a49fbef04a2c3de4da87f36baf1ca894776175eeeb715a3395f31c2f', 1, 'H'),
    1523: ('vertebra-column', 1, 310, 6, [60, 100, 150], '78913b1453c43b7b9e0623b0d295b1cb',
           'a3ac0ae0d50c6b916be247218266a02a886acd1ab4d4930660a7d06ca6bf44ec', '0cbe50613e61df7cc4b00e7e54f4f5e05f77d4a40931e6a729c8981a0723835d', 0, 'H'),
    37: ('diabetes', 1, 768, 8, [500, 268], '3cbaa3e54586aa88cf6aacb4033e4470',
         'db6367d8ed67a9f06f9276bd075617c0240bd36838c34d7b921f6b21df60036c', '4dc5a2259526cd11dde0ec7308b7c2f430574be817132ccaf98aa5390a9e6eb6', 0, 'C'),
    1462: ('banknote-authentication', 1, 1372, 4, [762, 610], 'baa2dc5b745775a943ebeb9c276401f8',
           'f153ac742c3ec415cb4b672731908d9902f67c831d45f74941bf8538d8621642', 'e6fb532213c8ead8829f17bb8bf24425040a3392b612e5e5f68983c174cb99ac', 24, 'C'),
    1494: ('qsar-biodeg', 1, 1055, 41, [699, 356], 'a2c189cd65511103fa540d7186155c24',
           '584e72d06483246d8ec69bb570035e491182e0144e40dd069e8d9ecda76187d0', 'c9b656957085b059c1caffd2808deb3cabd906a181f6f7d7b18bf2be7b894d02', 3, 'C'),
    40982: ('steel-plates-fault', 3, 1941, 27, [402, 55, 391, 673, 158, 72, 190], '7ccdabeb01749cce9fa3b1d4a702fb8c',
            '92e432d6860deb12dd15b6b07ab9767e25b7812595d0745f5916288c19164c68', 'bc4253c02101741b9cebe9329c507745ed71721d89ceedde22c94ebbe78c177c', 0, 'C'),
    40994: ('climate-model-simulation-crashes', 4, 540, 18, [46, 494], 'f7c55d9a11782a5ff980cee371787edd',
            'f7f6b2ae9a618cfc6a493cc351cbd5f5cb7ed5a1edab3bda524c0487e8de4e44', '8f74bf1bf64762681b5cffdec0892a17c663ec69fa2c51afe42bb2f32dc9fe9d', 0, 'C'),
    46850: ('hepatitis_c_virus_hcv_for_egyptian_patients', 2, 1385, 28, [336, 332, 355, 362], '496acf3c6daf0ee7e124dc1802afb14f',
            '0174a199281d484d8ca60dcfae7e192a31c6ff6b951bb904a798cef2686d8335', '1aec6b85ddbdcc6dbca49d13c6af1313606a2b0b4159dae70e99419a6f3760f2', 0, 'C'),
}
EVIDENCE = {11: [9.9, 9.1], 22: [8.7], 59: [6.3, 9.5], 1523: [7.1, 7.4], 37: [3.5, .4], 1462: [.1], 1494: [2.3], 40982: [4.6, 7.4],
            40994: [4.1], 46850: []}
NOT_INFORMATIVE = {1523: 7.4, 40982: 7.4}      # OpenML sources with fewer than 10 kNN runs (climate's only source is kept)


def test_pins_transcribe_the_frozen_selection_in_the_order_of_the_ruling():
    assert [pin['data_id'] for pin in nd.PINS] == [11, 22, 59, 1523, 37, 1462, 1494, 40982, 40994, 46850] == list(SELECTION)
    for pin in nd.PINS:
        name, version, n, d, counts, md5, sha_x, sha_y, duplicates, stratum = SELECTION[pin['data_id']]
        assert (pin['openml_name'], pin['version'], pin['shape'], pin['class_counts'], pin['md5_checksum']) == (name, version, [n, d], counts, md5)
        assert (pin['sha256_X'], pin['sha256_y'], pin['duplicates']['duplicate_rows'], pin['stratum']) == (sha_x, sha_y, duplicates, stratum)
        assert (pin['n_missing'], pin['n_infinite'], pin['duplicates']['label_conflicting_groups']) == (0, 0, 0)
        assert len(pin['label_map']) == len(counts) and sum(counts) == n
        sources = list(pin['external_gap']['sources'].values())
        assert sorted(sources) == sorted(EVIDENCE[pin['data_id']])
        assert (pin['external_gap']['points'] is None) == (not sources)
        if sources:
            assert pin['external_gap']['points'] == pytest.approx(np.mean(sources), abs=1e-12)
    assert nd.H_STRATUM == ('balance_scale', 'mfeat_zernike', 'ionosphere', 'vertebra_column')
    assert len(set(TEN)) == 10 and nd.PIN_BY_NAME['climate_model_simulation_crashes']['ignore_attributes'] == ['Study', 'Run']


def test_the_spearman_gap_ranks_do_not_depend_on_counting_the_non_informative_openml_sources():
    kept = [pin['external_gap']['points'] for pin in nd.PINS if pin['external_gap']['points'] is not None]
    dropped = [np.mean([v for v in EVIDENCE[pin['data_id']] if v != NOT_INFORMATIVE.get(pin['data_id']) or len(EVIDENCE[pin['data_id']]) == 1])
               for pin in nd.PINS if pin['external_gap']['points'] is not None]
    assert list(stats.rankdata(kept)) == list(stats.rankdata(dropped)) and len(set(kept)) == 9


def fake_source(n=30, seed=4):
    """A pin-consistent synthetic OpenML bunch and its pin."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 3)).round(3)
    target = np.asarray(['b', 'a', 'c'] * (n // 3), dtype=object)
    names = ['f0', 'f1', 'f2']
    labels, y = np.unique(target, return_inverse=True)
    pin = dict(nd.PINS[0], name='fake', data_id=999999, openml_name='fake-source', version=2, file_id='31', md5_checksum='0' * 32,
               default_target='class', ignore_attributes=None, shape=[n, 3], label_map=['a', 'b', 'c'], class_counts=[n // 3] * 3,
               sha256_X=nd.array_sha256(X), sha256_y=nd.labels_sha256(target), feature_names_sha256=nd.names_sha256(names),
               dataset_hash=dataset_fingerprint(X, y, names, ['a', 'b', 'c']),
               splits_hash=config_id(make_splits(y, 5, 3, 3, 27183)))
    details = {'id': '999999', 'name': 'fake-source', 'version': '2', 'file_id': '31', 'md5_checksum': '0' * 32,
               'default_target_attribute': 'class'}
    return pin, Bunch(data=X, target=target, feature_names=names, details=details)


def test_the_loader_decodes_exactly_as_run_revision_load_dataset_for_an_openml_source(monkeypatch):
    pin, bunch = fake_source()
    monkeypatch.setitem(nd.PIN_BY_NAME, 'fake', pin)
    X, y, manifest = nd.load_newdata('fake', fetch=lambda p: bunch)
    monkeypatch.setitem(rr.DATASETS, 'fake', (999999, (30, 3), [10, 10, 10]))
    calls = []
    monkeypatch.setattr(rr.datasets, 'fetch_openml', lambda **kwargs: calls.append(kwargs) or bunch)
    X_harness, y_harness, harness = rr.load_dataset('fake')
    assert calls == [{'data_id': 999999, 'as_frame': False, 'parser': 'auto'}]
    assert np.array_equal(X, X_harness) and X.dtype == X_harness.dtype and np.array_equal(y, y_harness) and y.dtype == y_harness.dtype
    assert {key: manifest[key] for key in harness} == harness            # every harness manifest field, identical
    assert manifest['openml']['version'] == 2 and manifest['identity']['sha256_X'] == pin['sha256_X']
    seen = []
    monkeypatch.setattr(nd.datasets, 'fetch_openml', lambda **kwargs: seen.append(kwargs) or bunch)
    nd.load_newdata('fake')
    assert seen == calls                                                  # the default fetch is the harness call


def _mutated(kind):
    pin, bunch = fake_source()
    X, target, names, details = bunch.data.copy(), bunch.target.copy(), list(bunch.feature_names), dict(bunch.details)
    if kind == 'version':
        details['version'] = '1'
    elif kind == 'md5_checksum':
        details['md5_checksum'] = '1' * 32
    elif kind == 'openml_name':
        details['name'] = 'other'
    elif kind == 'data_id':
        details['id'] = '1'
    elif kind == 'file_id':
        details['file_id'] = '32'
    elif kind == 'default_target':
        details['default_target_attribute'] = 'label'
    elif kind == 'ignore_attributes':
        details['ignore_attribute'] = ['Run']
    elif kind == 'shape':
        X, target = X[:-3], target[:-3]
    elif kind == 'sha256_X':
        X[4, 1] += 1e-9
    elif kind == 'sha256_y':
        target[[0, 1]] = target[[1, 0]]
    elif kind == 'label_map':
        target = np.where(target == 'c', 'd', target).astype(object)
    elif kind == 'n_missing':
        X[2, 2] = np.nan
    elif kind == 'n_infinite':
        X[2, 2] = np.inf
    elif kind == 'feature_names_sha256':
        names[0] = 'g0'
    return pin, Bunch(data=X, target=target, feature_names=names, details=details)


@pytest.mark.parametrize('kind', ['version', 'md5_checksum', 'openml_name', 'data_id', 'file_id', 'default_target', 'ignore_attributes',
                                  'shape', 'sha256_X', 'sha256_y', 'label_map', 'n_missing', 'n_infinite', 'feature_names_sha256'])
def test_the_loader_refuses_every_pin_mismatch(monkeypatch, kind):
    pin, bunch = _mutated(kind)
    monkeypatch.setitem(nd.PIN_BY_NAME, 'fake', pin)
    with pytest.raises(nd.DatasetIdentityError, match=f'{kind} pinned'):
        nd.load_newdata('fake', fetch=lambda p: bunch)


def test_the_loader_refuses_a_harness_fingerprint_mismatch_and_an_unpinned_name(monkeypatch):
    pin, bunch = fake_source()
    monkeypatch.setitem(nd.PIN_BY_NAME, 'fake', dict(pin, dataset_hash='0' * 64))
    with pytest.raises(nd.DatasetIdentityError, match='dataset_hash pinned'):
        nd.load_newdata('fake', fetch=lambda p: bunch)
    with pytest.raises(nd.DatasetIdentityError, match='not a pinned'):
        nd.load_newdata('iris', fetch=lambda p: bunch)


def _cached(pin):
    home = Path(get_data_home())/'openml'/'openml.org'
    return all(path.is_file() for path in (home/'api'/'v1'/'json'/'data'/f"{pin['data_id']}.gz",
                                           home/'api'/'v1'/'json'/'data'/'features'/f"{pin['data_id']}.gz",
                                           home/'data'/'v1'/'download'/f"{pin['file_id']}.gz"))


@pytest.mark.skipif(not all(_cached(pin) for pin in nd.PINS), reason='the ten official OpenML downloads are not cached here')
def test_the_cached_official_downloads_pass_every_pin_with_the_pinned_splits_and_duplicates():
    for pin in nd.PINS:
        X, y, manifest = nd.load_newdata(pin['name'])
        assert config_id(make_splits(y, 5, 3, 3, 27183)) == pin['splits_hash'] and manifest['dataset_hash'] == pin['dataset_hash']
        counts = duplicate_groups(X, y)[1]
        assert {key: counts[key] for key in pin['duplicates']} == pin['duplicates']


# ----------------------------------------------------------------------------- the combined registry

def _factory(spec):
    return (getattr(spec.factory, 'func', spec.factory), getattr(spec.factory, 'args', ()), getattr(spec.factory, 'keywords', {}))


def test_the_registry_is_bridge_knn_with_both_training_controls_and_the_projected_control():
    protocol = nd.base_protocol()
    registry = rr.get_registry(nd.REGISTRY, protocol)
    assert tuple(registry) == nd.MODEL_ORDER and len(registry) == 10
    knn = bridge_knn_registry(protocol)
    assert set(knn) == set(nd.MODEL_ORDER) - {nd.UNTRAINED_MODEL, nd.INPUT_MODEL, nd.PROJECTED_MODEL}
    for model, spec in knn.items():
        assert (registry[model].candidates, registry[model].stochastic, _factory(registry[model])) == (spec.candidates, spec.stochastic, _factory(spec))
    assert (registry[nd.UNTRAINED_MODEL].factory, registry[nd.UNTRAINED_MODEL].candidates) == (untrained_factory, control_candidates(nd.UNTRAINED_MODEL))
    assert (registry[nd.INPUT_MODEL].factory, registry[nd.INPUT_MODEL].candidates) == (input_factory, control_candidates(nd.INPUT_MODEL))
    assert (registry[nd.PROJECTED_MODEL].factory, registry[nd.PROJECTED_MODEL].candidates) == (projected_factory, projected_candidates())
    assert all(registry[m].stochastic for m in (nd.TRAINED_MODEL, nd.UNTRAINED_MODEL, nd.INPUT_MODEL, nd.PROJECTED_MODEL))
    assert [len(registry[m].candidates) for m in nd.MODEL_ORDER] == [16, 8, 4, 4, 1, 16, 24, 24, 20, 24]
    with pytest.raises(ValueError, match='registry'):
        nd.newdata_registry(dict(protocol, registry=nd.SMOKE_REGISTRY))
    with pytest.raises(ValueError, match='synthetic smoke'):
        nd.smoke_newdata_registry(protocol)


@pytest.mark.skipif(not (RUNS/'2026-09-12-bridge-knn').is_dir() or not (RUNS/'2026-09-13-knn-training').is_dir(),
                    reason='the bridge_knn and knn_training runs are not on this machine')
def test_the_registry_candidates_equal_the_candidates_of_the_real_runs():
    record = nd._plain(nd.candidate_record(nd.newdata_registry(nd.base_protocol())))       # candidates.json serialization
    knn = json.loads((RUNS/'2026-09-12-bridge-knn'/'candidates.json').read_text())
    training = json.loads((RUNS/'2026-09-13-knn-training'/'candidates.json').read_text())
    assert {model: record[model] for model in knn} == knn and {model: record[model] for model in training} == training
    projected = RUNS.parent/'task-23a-stages'/'prepare'/'candidates.json'
    if projected.is_file():
        assert record[nd.PROJECTED_MODEL] == json.loads(projected.read_text())[nd.PROJECTED_MODEL]


# ----------------------------------------------------------------------------- protocols

TEMPLATE = PROTOCOLS/'bridge_knn.json'
PROVENANCE = {'frozen', 'frozen_at_utc', 'source_template_sha256', 'resource_decision', 'status'}
REMOVED = {'knn_readout', 'secondary_studies'}
CHANGED = {'protocol_id', 'production_family', 'registry', 'datasets', 'primary_contrasts', 'primary_family_size', 'multiplicity',
           'wallclock_cap_hours'}
ADDED = {'batch', 'batches', 'panel', 'model_order', 'models', 'secondary_contrasts', 'secondary_family_size', 'analysis',
         'design_source', 'model_template_sha256', 'selection_ruling', 'batch_projection'}
SPLIT = {'1': TEN[:5], '2': TEN[5:]}
FROZEN_AT = '2026-09-13T20:00:00+00:00'


def frozen_batch(number, decision=(5., 6.), batches=SPLIT):
    projection = {'cap_hours': nd.CAP_HOURS, 'batches': batches, 'decision_hours': {'1': decision[0], '2': decision[1]}}
    return nd.batch_protocol(nd.base_protocol(), number, batches, frozen_at_utc=FROZEN_AT, resource_decision='test',
                             batch_projection=projection)


def pin_protocol(p):
    template = json.loads(TEMPLATE.read_text())
    assert set(p) - PROVENANCE == (set(template) - PROVENANCE - REMOVED) | ADDED
    assert {k for k in set(template) - PROVENANCE - REMOVED if canonical_json(template[k]) != canonical_json(p[k])} == CHANGED
    assert set(nd.DESIGN_COPY_KEYS) == set(template) - PROVENANCE - REMOVED - CHANGED
    assert all(canonical_json(p[k]) == canonical_json(template[k]) for k in nd.DESIGN_COPY_KEYS)
    assert p['design_source']['identical_to_template'] == ', '.join(nd.DESIGN_COPY_KEYS)
    assert set(p['design_source']['removed_from_template']) == REMOVED
    assert p['source_template_sha256'] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert p['model_template_sha256'] == {n: hashlib.sha256((PROTOCOLS/n).read_bytes()).hexdigest() for n in ('knn_training.json', 'knn_projected.json')}
    assert (p['split_seed'], p['outer_folds'], p['outer_repeats'], p['inner_folds'], p['fit_seeds']) == (27183, 5, 3, 3, [8129, 19391, 39019])
    models = p['models']
    assert models[nd.TRAINED_MODEL]['readout']['grid'] == template['knn_readout']['grid'] and models[nd.TRAINED_MODEL]['candidates'] == 16
    training = json.loads((PROTOCOLS/'knn_training.json').read_text())['training_controls']['models']
    for model in (nd.UNTRAINED_MODEL, nd.INPUT_MODEL):
        assert (models[model]['candidate_keys'], models[model]['candidates']) == (training[model]['candidate_keys'], training[model]['candidates'])
    projected = json.loads((PROTOCOLS/'knn_projected.json').read_text())['projected_control']
    assert (models[nd.PROJECTED_MODEL]['candidate_keys'], models[nd.PROJECTED_MODEL]['candidates'], models[nd.PROJECTED_MODEL]['readout_grid']) == (
        projected['model']['candidate_keys'], projected['model']['candidates'], projected['readout']['grid'])
    assert p['primary_contrasts'] == ['arrowflow_full_knn_vs_arrowflow_knn_untrained']
    assert p['secondary_contrasts'] == ['arrowflow_full_knn_vs_input_footrule_knn']
    assert (p['primary_family_size'], p['secondary_family_size'], p['wallclock_cap_hours'], p['production_family']) == (10, 10, 9, 'newdata')
    analysis = p['analysis']
    assert analysis['requires_both_batches_complete'] is True and analysis['datasets'] == TEN
    assert analysis['moderator_test']['strata'] == {'H': ['balance_scale', 'mfeat_zernike', 'ionosphere', 'vertebra_column'], 'C': TEN[4:]}
    assert (analysis['moderator_test']['splits'], analysis['moderator_test']['alpha'], analysis['moderator_test']['smallest_attainable_p']) == (210, .05, 1 / 210)
    assert analysis['primary_family']['alpha'] == analysis['secondary_family']['alpha'] == .05
    assert analysis['spearman']['datasets'] == TEN[:9] and analysis['spearman']['excluded'] == ['hcv_egyptian_patients']
    assert '362880 permutations' in analysis['spearman']['p_value'] and [r['model_id'] for r in analysis['ladder']] == [m for _, m in nd.LADDER]
    assert analysis['comparator_intervals']['models'] == list(nd.COMPARATORS) and analysis['notes'] == nd._plain(nd.NOTES)
    assert {n: analysis['duplicate_audit']['expected'][n]['duplicate_rows'] for n in TEN} == {pin['name']: SELECTION[pin['data_id']][8] for pin in nd.PINS}
    assert [entry['data_id'] for entry in p['panel']] == list(SELECTION) and p['model_order'] == list(nd.MODEL_ORDER)
    nd.validate_newdata_protocol(p)


def test_the_protocols_copy_the_bridge_knn_design_and_differ_only_in_their_batch_fields():
    draft = nd.base_protocol()
    pin_protocol(draft)
    assert (draft['batch'], draft['batches'], draft['datasets'], draft['frozen'], draft['batch_projection']) == (None, None, TEN, False, None)
    first, second = frozen_batch(1), frozen_batch(2)
    for number, protocol in ((1, first), (2, second)):
        pin_protocol(protocol)
        assert (protocol['protocol_id'], protocol['batch'], protocol['datasets'], protocol['frozen'], protocol['frozen_at_utc']) == (
            nd.PROTOCOL_IDS[number], number, SPLIT[str(number)], True, FROZEN_AT)
    assert sorted(k for k in first if canonical_json(first[k]) != canonical_json(second[k])) == ['batch', 'datasets', 'protocol_id']
    assert sorted(k for k in first if canonical_json(first.get(k)) != canonical_json(draft.get(k))) == sorted(
        ['batch', 'batches', 'batch_projection', 'datasets', 'frozen', 'frozen_at_utc', 'protocol_id', 'resource_decision', 'status'])


@pytest.mark.parametrize('mutation, message', [
    (lambda p: p['panel'].pop(), 'panel'), (lambda p: p['panel'][4].update(stratum='H'), 'panel'),
    (lambda p: p['panel'][9]['external_gap'].update(points=1.), 'panel'), (lambda p: p['panel'][0].update(sha256_X='0' * 64), 'panel'),
    (lambda p: p.update(split_seed=1), 'nested design'), (lambda p: p.update(fit_seeds=[1, 2, 3]), 'nested design'),
    (lambda p: p.update(outer_repeats=1), 'nested design'), (lambda p: p['models']['svc_rbf'].update(candidates=12), 'models'),
    (lambda p: p.update(model_order=p['model_order'][::-1]), 'model_order'),
    (lambda p: p.update(primary_contrasts=['arrowflow_full_knn_vs_input_footrule_knn']), 'primary_contrasts'),
    (lambda p: p.update(secondary_family_size=14), 'family_size'),
    (lambda p: p['analysis']['moderator_test'].update(alpha=.1), 'analysis'),
    (lambda p: p['analysis']['moderator_test']['strata']['H'].pop(), 'analysis'),
    (lambda p: p['analysis']['spearman'].update(excluded=[]), 'analysis'),
    (lambda p: p['analysis'].update(requires_both_batches_complete=False), 'analysis'),
    (lambda p: p.update(registry='experiments.make_revision.knn_controls:knn_training_registry'), 'registry'),
    (lambda p: p.update(production_family='knn_training'), 'production_family'),
    (lambda p: p.update(frozen=True), 'stage draft'), (lambda p: p.update(datasets=TEN[:5]), 'stage draft')])
def test_validate_refuses_a_draft_that_differs_from_the_module(mutation, message):
    protocol = nd.base_protocol()
    mutation(protocol)
    with pytest.raises(ValueError, match=message):
        nd.newdata_registry(protocol)


@pytest.mark.parametrize('mutation, message', [
    (lambda p: p.update(datasets=TEN[5:]), 'datasets of that batch'), (lambda p: p.update(batch=3), 'batch must be 1 or 2'),
    (lambda p: p.update(batches={'1': TEN[:6], '2': TEN[6:]}), 'batches must map'),
    (lambda p: p.update(batches={'1': TEN[:5], '2': TEN[:5]}), 'partition'),
    (lambda p: p.update(batches={'1': TEN[:5][::-1], '2': TEN[5:]}), 'panel order'),
    (lambda p: p.update(protocol_id=nd.PROTOCOL_IDS[2]), 'protocol_id'), (lambda p: p.update(batch_projection=None), 'batch projection'),
    (lambda p: p['batch_projection']['decision_hours'].update({'2': 9.5}), 'batch projection'),
    (lambda p: p['batch_projection'].update(batches={'1': TEN[5:], '2': TEN[:5]}), 'batch projection'),
    (lambda p: p.update(wallclock_cap_hours=10), 'batch projection')])
def test_validate_refuses_an_inconsistent_frozen_batch(mutation, message):
    protocol = frozen_batch(1)
    mutation(protocol)
    with pytest.raises(ValueError, match=message):
        nd.validate_newdata_protocol(protocol)


# ----------------------------------------------------------------------------- prepare and run

def install_fake_sources(monkeypatch, names):
    """load_newdata on pin-consistent synthetic sources under real panel names (every pin check still runs)."""
    original, sources = nd.load_newdata, {}
    for index, name in enumerate(names):
        pin, bunch = fake_source(seed=index)
        sources[name] = bunch
        monkeypatch.setitem(nd.PIN_BY_NAME, name, dict(pin, name=name))
    monkeypatch.setattr(nd, 'load_newdata', lambda name, fetch=None: original(name, fetch=lambda p: sources[name]))


def test_prepare_writes_the_run_revision_layout_for_pinned_datasets_and_refuses_changes(tmp_path, monkeypatch):
    protocol = frozen_batch(1)
    names = protocol['datasets'][:2]
    install_fake_sources(monkeypatch, names)
    registry = nd.prepare(tmp_path/'run', names, protocol)
    assert tuple(registry) == nd.MODEL_ORDER
    assert (tmp_path/'run'/'protocol.json').read_text() == json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
    assert json.loads((tmp_path/'run'/'candidates.json').read_text()) == nd._plain(nd.candidate_record(registry))
    environment = json.loads((tmp_path/'run'/'environment.json').read_text())
    assert environment['registry'] == nd.REGISTRY and set(cn.SEALED_SOURCES) <= set(environment['source_hashes'])
    for name in names:
        X, y, manifest, splits = rr.load_prepared(tmp_path/'run', name)
        assert manifest['splits_hash'] == nd.PIN_BY_NAME[name]['splits_hash'] and manifest['dataset_hash'] == nd.PIN_BY_NAME[name]['dataset_hash']
        assert splits == make_splits(y, 5, 3, 3, 27183) and manifest['source'] == 'OpenML data_id=999999'
    with pytest.raises(ValueError, match='outside the protocol datasets'):
        nd.prepare(tmp_path/'other', [names[0], TEN[9]], protocol)
    monkeypatch.setitem(nd.PIN_BY_NAME, names[0], dict(nd.PIN_BY_NAME[names[0]], splits_hash='0' * 16))
    with pytest.raises(nd.DatasetIdentityError, match='splits hash'):
        nd.prepare(tmp_path/'third', names[:1], protocol)


def test_run_refuses_before_any_job_unless_the_prepared_batch_is_the_frozen_pinned_protocol(tmp_path, monkeypatch):
    protocol = frozen_batch(1)
    install_fake_sources(monkeypatch, protocol['datasets'])
    nd.prepare(tmp_path/'run', protocol['datasets'], protocol)
    paths = {label: tmp_path/f'{label}.json' for label in ('batch1', 'batch2', 'draft', 'smoke')}
    for label, value in (('batch1', protocol), ('batch2', frozen_batch(2)), ('draft', nd.base_protocol()), ('smoke', nd.smoke_protocol(1))):
        rr.write_json(paths[label], value)
    monkeypatch.setattr(nd, 'ProcessPoolExecutor', lambda *args, **kwargs: pytest.fail('a worker pool started'))
    for path, workers, message in ((paths['draft'], 16, 'reviewed frozen protocol'), (paths['smoke'], 16, 'one batch of a frozen newdata protocol'),
                                   (paths['batch2'], 16, 'Prepared and frozen protocols differ'), (paths['batch1'], 17, 'Worker count')):
        with pytest.raises(ValueError, match=message):
            nd.run(path, tmp_path/'run', workers)
    environment = (tmp_path/'run'/'environment.json').read_text()
    (tmp_path/'run'/'environment.json').write_text(environment.replace('"registry"', '"registry_edited"'))
    with pytest.raises(ValueError, match='environment'):
        nd.run(paths['batch1'], tmp_path/'run', 16)
    (tmp_path/'run'/'environment.json').write_text(environment)
    monkeypatch.setitem(nd.PIN_BY_NAME, protocol['datasets'][3], dict(nd.PIN_BY_NAME[protocol['datasets'][3]], dataset_hash='0' * 64))
    with pytest.raises(nd.DatasetIdentityError, match='differs from the pin'):
        nd.run(paths['batch1'], tmp_path/'run', 16)
    assert not (tmp_path/'run'/'planned_jobs.json').exists()


def test_the_real_data_fit_check_fits_every_model_once_on_training_rows_without_scores(monkeypatch):
    protocol = frozen_batch(1)
    install_fake_sources(monkeypatch, protocol['datasets'][:1])
    check = nd.real_data_check(protocol, names=protocol['datasets'][:1])
    assert [row['model_id'] for row in check['rows']] == list(nd.MODEL_ORDER) and check['failed'] == []
    assert all(row['status'] == 'ok' and row['fit_rows'] == 24 == row['predictions'] for row in check['rows'])
    assert not any({'accuracy', 'error', 'score'} & set(row) for row in check['rows'])
    assert [row['iterations_override'] for row in check['rows']] == [2] + [None] * 9


# ----------------------------------------------------------------------------- projection and freeze

def simulate(durations, workers):
    free = [0.] * workers
    for duration in durations:
        free[free.index(min(free))] += duration
    return max(free)


def test_fits_per_outer_equals_the_harness_pilot_counts():
    assert [nd.fits_per_outer(n, s, 3) for n, s in ((16, True), (8, True), (4, True), (1, False), (16, False), (24, True), (20, False))] == [
        69, 45, 33, 4, 49, 93, 61]


def test_makespan_and_batch_hours_known_answers():
    assert (nd.makespan([3.] * 4, workers=2), nd.makespan([5., 1., 1., 1., 1.], workers=2), nd.makespan([1., 1., 1., 10.], workers=3)) == (6., 5., 11.)
    assert nd.makespan([2.] * 32, workers=16) == 4.
    jobs = {('a', m): {'central': 360. if m == nd.TRAINED_MODEL else 36., 'upper': 720.} for m in nd.MODEL_ORDER}
    hours = nd.batch_hours(jobs, ['a'], 'central', folds=15, workers=16)
    assert hours['serial_hours'] == pytest.approx(15 * (360 + 9 * 36) / 3600)
    assert hours['serial_over_workers_hours'] == pytest.approx(hours['serial_hours'] / 16)
    order = [360. if m == nd.TRAINED_MODEL else 36. for _ in range(15) for m in nd.MODEL_ORDER]
    assert hours['simulated_makespan_hours'] == pytest.approx(simulate(order, 16) / 3600)


def test_balanced_split_minimizes_the_serial_difference_with_the_first_dataset_in_batch_1():
    assert nd.balanced_split({'a': 10., 'b': 9., 'c': 1., 'd': 2.}, panel=('a', 'b', 'c', 'd')) == {'1': ['a', 'c'], '2': ['b', 'd']}
    assert nd.balanced_split(dict.fromkeys('abcd', 1.), panel=tuple('abcd')) == {'1': ['a', 'b'], '2': ['c', 'd']}
    serial = dict(zip(TEN, [3., 9., 4., 2., 3., 5., 7., 6., 4., 5.]))
    total = sum(serial.values())
    best = min((abs(2 * sum(serial[TEN[i]] for i in (0, *rest)) - total), (0, *rest)) for rest in combinations(range(1, 10), 4))
    split = nd.balanced_split(serial)
    assert split['1'] == [TEN[i] for i in best[1]] and split['2'] == [n for n in TEN if n not in split['1']]


def synthetic_calibration_run(root, realized, models):
    """A completed run holding what calibration_factors reads: two stochastic candidates per model and one fit log per job."""
    (root/'results').mkdir(parents=True)
    candidates = {m: {'stochastic': True, 'candidates': [{'c': 0}, {'c': 1}], 'config_ids': [config_id({'c': 0}), config_id({'c': 1})]} for m in models}
    (root/'candidates.json').write_text(json.dumps(candidates))
    (root/'protocol.json').write_text(json.dumps({'inner_folds': 3}))
    jobs = []
    for (dataset, model), seconds in realized.items():
        for index, job_seconds in enumerate(seconds):
            log = f'results/{dataset}__{model}__r0f{index}.fits.jsonl'
            (root/log).write_text(json.dumps({'fit_seconds': job_seconds * .75, 'predict_seconds': job_seconds * .25}) + '\n'
                                  + json.dumps({'fit_seconds': 0., 'predict_seconds': None, 'status': 'failed'}) + '\n')
            jobs.append({'dataset_id': dataset, 'model_id': model, 'log_file': log})
    (root/'planned_jobs.json').write_text(json.dumps(jobs))


def pilot_rows(datasets, seconds):
    return [{'dataset_id': d, 'model_id': m, 'config_id': 'x', 'status': 'ok', 'elapsed_seconds': seconds(d, m)} for d in datasets for m in nd.MODEL_ORDER]


def test_calibration_factors_are_realized_over_pilot_projected_job_seconds(tmp_path):
    synthetic_calibration_run(tmp_path/'run', {('d1', 'm'): [80., 88.], ('d2', 'm'): [21., 21.]}, ['m'])
    pilot = {'rows': [{'dataset_id': 'd1', 'model_id': 'm', 'config_id': 'x', 'status': 'ok', 'elapsed_seconds': s} for s in (1., 3.)]
             + [{'dataset_id': 'd2', 'model_id': 'm', 'config_id': 'x', 'status': 'ok', 'elapsed_seconds': 1.}]}
    factors = nd.calibration_factors(tmp_path/'run', pilot, ['m'])['m']
    assert factors['fits_per_outer'] == 21                  # 2 x 3 screening + 2 x 2 x 3 reranking + 3 outer fits
    assert (factors['datasets']['d1']['factor'], factors['datasets']['d2']['factor']) == pytest.approx((2., 1.))
    assert factors['pooled'] == pytest.approx((84 + 21) / (42 + 21)) and factors['max'] == pytest.approx(2.)
    with pytest.raises(ValueError, match='pilot fit failed'):
        nd.calibration_factors(tmp_path/'run', {'rows': [dict(pilot['rows'][0], status='failed', exception='boom')]}, ['m'])
    with pytest.raises(ValueError, match='No calibration dataset'):
        nd.calibration_factors(tmp_path/'run', {'rows': [dict(pilot['rows'][0], dataset_id='d9')]}, ['m'])


def test_job_projection_scales_the_pilots_and_reanchors_the_research_estimate_for_the_other_datasets():
    piloted = ['mfeat_zernike', 'vertebra_column']
    pilot = {'rows': pilot_rows(piloted, lambda d, m: (2. if m == nd.TRAINED_MODEL else .5) * (3. if d == piloted[0] else 1.))}
    factors = {m: {'pooled': 1.5, 'max': 2.} for m in nd.MODEL_ORDER if m != nd.PROJECTED_MODEL}
    jobs, basis = nd.job_projection(pilot, factors)
    assert basis['piloted'] == piloted and basis['fits_per_outer'][nd.TRAINED_MODEL] == 69 and set(jobs) == {(d, m) for d in nd.PANEL for m in nd.MODEL_ORDER}
    assert jobs['mfeat_zernike', nd.TRAINED_MODEL]['central'] == pytest.approx(1.5 * 69 * 6. + 1.)
    assert jobs['vertebra_column', nd.PROJECTED_MODEL]['upper'] == pytest.approx(2. * 33 * .5 + 1.)
    research = {d: nd.RESEARCH_ARROWFLOW_JOB_MINUTES[d][0] * 60. for d in nd.PANEL}
    central = np.mean([1.5 * 69 * 6. / research['mfeat_zernike'], 1.5 * 69 * 2. / research['vertebra_column']])
    upper = max(2. * 69 * 6. / research['mfeat_zernike'], 2. * 69 * 2. / research['vertebra_column']) * 1.18 / 1.01
    assert basis['anchor'] == pytest.approx({'central': central, 'upper': upper})
    assert (jobs['diabetes', nd.TRAINED_MODEL]['central'], jobs['diabetes', nd.TRAINED_MODEL]['upper']) == pytest.approx(
        (research['diabetes'] * central + 1., research['diabetes'] * upper + 1.))
    assert jobs['diabetes', 'gradient_boosting']['central'] == jobs['mfeat_zernike', 'gradient_boosting']['central']
    with pytest.raises(ValueError, match='every model'):
        nd.job_projection({'rows': pilot['rows'][:-1]}, factors)


def test_projection_and_freeze_end_to_end_on_synthetic_calibration_runs(tmp_path):
    calibration = {}
    for label, models in nd.CALIBRATION_MODELS.items():
        synthetic_calibration_run(tmp_path/label, {(d, m): [40., 40.] for d in ('iris', 'digits') for m in models}, models)
        (tmp_path/f'{label}-pilot.json').write_text(json.dumps({'rows': [row for row in pilot_rows(('iris', 'digits'), lambda d, m: 1.)
                                                                         if row['model_id'] in models]}))
        calibration[label] = {'run': tmp_path/label, 'pilot': tmp_path/f'{label}-pilot.json'}
    piloted = ['mfeat_zernike', 'vertebra_column']
    pilot_path = tmp_path/'pilot.json'
    pilot_path.write_text(json.dumps({'rows': pilot_rows(piloted, lambda d, m: (6. if m == nd.TRAINED_MODEL else .2) * (1.5 if d == piloted[0] else 1.)),
                                      'workload_estimates': {m: {'serial_panel_seconds_using_observed_max': 10000.} for m in nd.MODEL_ORDER},
                                      'datasets_piloted': piloted}))
    record = nd.projection(pilot_path, calibration=calibration)
    factor = 40. / 21
    assert record['calibration'][nd.TRAINED_MODEL]['pooled'] == pytest.approx(factor)
    assert record['per_dataset']['mfeat_zernike']['arrowflow_job_minutes']['central'] == pytest.approx((factor * 69 * 9. + 1.) / 60)
    assert record['batches'] == nd.balanced_split({n: record['per_dataset'][n]['central_serial_hours'] for n in nd.PANEL})
    assert record['decision_hours'] == {k: record['per_batch'][k]['central']['simulated_makespan_hours'] for k in ('1', '2')}
    assert record['harness_max_based_hours'] == pytest.approx({'1': 5 * 10000. / 3600 / 16, '2': 5 * 10000. / 3600 / 16})
    assert record['within_cap'] is True and record['simulation_check']['jobs'] == len(nd.CALIBRATION_MODELS['bridge_knn']) * 4
    projection_path, draft, stages = tmp_path/'projection.json', tmp_path/'draft.json', tmp_path/'stages.json'
    rr.write_json(projection_path, record)
    rr.write_json(draft, nd.base_protocol())
    stages.write_text(json.dumps({'summary': 'prepare, smoke and pilot passed (test)'}))
    paths = nd.freeze(draft, projection_path, stages, tmp_path/'protocols', frozen_at_utc=FROZEN_AT)
    assert [path.name for path in paths] == ['newdata_batch1.json', 'newdata_batch2.json']
    for number, path in enumerate(paths, start=1):
        protocol = json.loads(path.read_text())
        pin_protocol(protocol)
        assert protocol['batches'] == record['batches'] and protocol['batch_projection']['decision_hours'] == record['decision_hours']
        assert protocol['batch_projection']['projection_sha256'] == hashlib.sha256(projection_path.read_bytes()).hexdigest()
        assert 'prepare, smoke and pilot passed (test)' in protocol['resource_decision'] and protocol['batch'] == number
        assert nd.batch_protocol(nd.base_protocol(), number, protocol['batches'], frozen_at_utc=FROZEN_AT,
                                 resource_decision=protocol['resource_decision'], batch_projection=protocol['batch_projection']) == protocol
    rr.write_json(tmp_path/'over.json', dict(record, decision_hours={'1': 9.5, '2': 1.}))
    with pytest.raises(ValueError, match='Not frozen'):
        nd.freeze(draft, tmp_path/'over.json', stages, tmp_path/'refused')
    rr.write_json(tmp_path/'edited.json', dict(nd.base_protocol(), status='edited'))
    with pytest.raises(ValueError, match='stage draft differs'):
        nd.freeze(tmp_path/'edited.json', projection_path, stages, tmp_path/'refused')
    assert not (tmp_path/'refused').exists()


# ----------------------------------------------------------------------------- synthetic smoke

def test_smoke_runs_two_synthetic_batches_through_the_harness_and_the_analysis(tmp_path):
    record = nd.smoke(tmp_path, workers=2)
    assert record['purpose'] == 'synthetic_smoke_only_not_paper_evidence' and 'real_data_fit_check' not in record
    assert [row['dataset'] for row in record['primary_family']] == [entry['name'] for entry in nd.SMOKE_PANEL]
    assert all((row['n_folds'], row['df']) == (3, 2) for row in record['primary_family'] + record['secondary_family'])
    assert record['moderator_test']['splits'] == 6 and record['spearman']['permutations'] == 6
    assert record['duplicate_audit']['syn_c1']['counts']['label_conflicting_groups'] == 1
    assert sorted(path.name for path in (tmp_path/'analysis').iterdir()) == sorted(cn.OUTPUTS)
    candidates = json.loads((tmp_path/'batch1'/'candidates.json').read_text())
    assert len(candidates) == 10 and candidates[nd.TRAINED_MODEL]['candidates'] == nd._plain(nd.SMOKE_TRAINED_CANDIDATES)
    for label in cn.LABELS:
        assert nd.validate_newdata_protocol(json.loads((tmp_path/label/'protocol.json').read_text()))['purpose'] == 'synthetic_smoke_only'


def test_the_frozen_batch_protocols_are_the_stage_draft_split_by_the_projection_within_the_cap():
    first, second = (json.loads(nd.PROTOCOL_FILES[n].read_text()) for n in (1, 2))
    for number, protocol in ((1, first), (2, second)):
        pin_protocol(protocol)
        assert nd.PROTOCOL_FILES[number].read_text() == json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n'
        assert (protocol['protocol_id'], protocol['batch'], protocol['frozen']) == (nd.PROTOCOL_IDS[number], number, True)
        assert protocol['status'] == 'reviewed_and_piloted_before_confirmatory_scoring' and protocol['frozen_at_utc'] >= '2026-09-13'
        assert nd.batch_protocol(nd.base_protocol(), number, protocol['batches'], frozen_at_utc=protocol['frozen_at_utc'],
                                 resource_decision=protocol['resource_decision'], batch_projection=protocol['batch_projection']) == protocol
    assert sorted(k for k in first if canonical_json(first[k]) != canonical_json(second[k])) == ['batch', 'datasets', 'protocol_id']
    batches, projection = first['batches'], first['batch_projection']
    assert sorted(batches['1'] + batches['2']) == sorted(TEN) and len(batches['1']) == len(batches['2']) == 5 and batches['1'][0] == TEN[0]
    assert all(0 < projection['decision_hours'][key] <= nd.CAP_HOURS for key in ('1', '2')) and projection['cap_hours'] == 9
    assert batches == nd.balanced_split({name: projection['per_dataset'][name]['central_serial_hours'] for name in TEN})
    assert projection['stages']['real_data_fit_check']['failed'] == [] and projection['stages']['pilot']['failed'] == []
