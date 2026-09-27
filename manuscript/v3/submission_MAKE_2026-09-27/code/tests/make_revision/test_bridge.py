# tests/make_revision/test_bridge.py
import json, numpy as np, pytest
from math import comb
from experiments.make_revision import bridge
from experiments.make_revision.evaluation import config_id

def test_adaptive_defaults_match_published_rule():
    assert bridge.adaptive_defaults(4) == {'embed_dim': 16, 'degree': 3, 'augment': True}
    assert bridge.adaptive_defaults(13) == {'embed_dim': 32, 'degree': 2, 'augment': True}
    assert bridge.adaptive_defaults(64) == {'embed_dim': 64, 'degree': 1, 'augment': False}

def test_exactly_16_unique_candidates():
    cands = bridge.bridge_candidates()
    assert len(cands) == 16 and len({config_id(c) for c in cands}) == 16
    assert all(c['n_views'] == 7 and c['validation_ratio'] == .1 for c in cands)
    assert {c['embed_scale'] for c in cands} == {1, 2}          # embed_scale 0.5 dropped after the pilot

def test_resolve_respects_column_cap_and_embed_bounds():
    r = bridge.resolve({'embed_scale': 2, 'degree_offset': 0}, n_features=64, n_train=1000)
    assert r['embed_dim'] == 128 and r['degree'] == 1 and r['augment'] is False
    r = bridge.resolve({'embed_scale': .5, 'degree_offset': 0}, n_features=4, n_train=100)
    assert r['embed_dim'] == 8 and r['degree'] == 3 and r['augment'] is False     # n_train < 150
    r = bridge.resolve({'embed_scale': 1, 'degree_offset': 1}, n_features=64, n_train=1000)
    assert comb(64 + r['degree'], r['degree']) <= 10_000

def test_registry_budget_and_ids():
    protocol = json.load(open('experiments/make_revision/protocols/2026-09-12/bridge.json'))
    reg = bridge.bridge_registry(protocol)
    assert set(reg) == {'arrowflow_full', 'dummy', 'svc_rbf', 'random_forest', 'mlp', 'numeric_knn', 'gradient_boosting'}
    assert all(len(s.candidates) <= protocol['candidate_budget'] for s in reg.values())
    assert reg['arrowflow_full'].stochastic is True

def test_factory_fits_small_data():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(0).permutation(len(y))          # all three classes in the training rows
    est = bridge.arrowflow_full_factory({**bridge.bridge_candidates()[0], 'iterations': 3}, seed=3)
    est.fit(X[idx[:60]], y[idx[:60]])
    assert est.resolved_['embed_dim'] in (8, 16, 32) and est.predict(X[idx[60:70]]).shape == (10,)

def test_harness_fit_record_sees_first_view_encoder_and_multiview_timing():
    # evaluation._fit_predict reads encoder_ settings/tie rate and the timing attributes from the wrapper.
    from sklearn.datasets import load_iris
    from experiments.make_revision.evaluation import ModelSpec, _fit_predict
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(1).permutation(len(y))
    spec = ModelSpec('arrowflow_full', bridge.arrowflow_full_factory,
                     [{**bridge.bridge_candidates()[0], 'iterations': 2}], True)
    predictions, record = _fit_predict(spec, spec.candidates[0], 8129, X[idx[:60]], y[idx[:60]], X[idx[60:70]])
    assert predictions.shape == (10,)
    assert record['preprocessing_settings']['strategy'] == 'target_aware'          # view 0 of the diverse cycle
    assert record['preprocessing_settings']['embed_dim'] == 16 and record['preprocessing_settings']['degree'] == 3
    assert 0 <= record['training_tie_rate'] <= 1
    assert record['encoding_seconds'] > 0 and record['classifier_fit_seconds'] > 0
    assert record['query_encoding_seconds'] > 0 and record['inference_seconds'] >= 0

@pytest.mark.parametrize('model_id', ['arrowflow_full', 'arrowflow_full_knn'])
def test_nested_fold_evaluation_runs_with_the_adaptive_wrapper(model_id):
    # The harness smoke stage is hard-wired to the single-view spec; exercise select_model + outer fits here,
    # for the output rule and for the kNN readout family (Task 10), whose fit records carry the readout selection.
    from sklearn.datasets import load_iris
    from experiments.make_revision.evaluation import ModelSpec, canonical_json, evaluate_fold, make_splits
    X, y = load_iris(return_X_y=True)
    split = make_splits(y, outer_folds=3, repeats=1, inner_folds=2, seed=27183)[0]
    small = [{**c, 'n_views': 2, 'iterations': 2} for c in bridge.bridge_candidates()[:2]]   # degree_offset 0 and -1
    factory = {'arrowflow_full': bridge.arrowflow_full_factory, 'arrowflow_full_knn': bridge.arrowflow_full_knn_factory}[model_id]
    spec = ModelSpec(model_id, factory, small, True)
    rows = []
    result = evaluate_fold(X, y, split, spec, (8129, 19391, 39019), dataset_id='iris', dataset_hash='fixture',
                           code_revision='test', sink=rows.append)
    assert result['status'] == 'ok' and len(result['models']) == 3
    assert len(result['predictions']) == 3 * len(split['test'])
    assert result['selection']['config_id'] in {config_id(c) for c in small}
    assert len(rows) == 2 * 2 + 2 * 2 * 2 + 3 and all(r['status'] == 'ok' for r in rows)
    canonical_json(rows)                                   # every record must be JSON-serializable without NaN
    for row in result['models']:
        assert row['preprocessing_settings']['embed_dim'] == 16     # iris: 16 * 1, resolved from the training rows
        assert row['training_sample_count'] == len(split['train'])
        assert row['classifier_fit_seconds'] > 0 and row['encoding_seconds'] > 0
    for row in rows:
        meta = row['representation_metadata']
        if model_id == 'arrowflow_full':
            assert meta is None
        else:
            assert meta['readout'] == 'knn_hidden' and len(meta['views']) == 2 and meta['readout_seconds'] > 0

def test_bridge_registry_seals_the_multiview_and_comparator_sources():
    from experiments.make_revision.run_revision import environment_record
    record = environment_record('experiments.make_revision.bridge:bridge_registry')
    assert {'experiments/make_revision/bridge.py', 'experiments/make_revision/multiview.py',
            'experiments/make_revision/comparisons.py', 'experiments/make_revision/datasets.py',
            'experiments/make_revision/secondary_studies.py'} <= record['source_hashes'].keys()

def test_ablation_variants_cover_the_component_story():
    selected = {**bridge.bridge_candidates()[5], 'embed_dim': 32, 'degree': 2, 'augment': True}
    ids = [v for v, _ in bridge.ablation_variants(selected)]
    assert ids == ['views1', 'views3', 'views7', 'no_checkpoint', 'no_augment', 'borda_views7',
                   'single_view_no_checkpoint_no_augment', 'multiview_footrule_knn']
    params = dict(bridge.ablation_variants(selected))
    assert params['views1']['n_views'] == 1 and params['no_checkpoint']['validation_ratio'] == 0
    assert params['single_view_no_checkpoint_no_augment'] == {**params['views1'], 'validation_ratio': 0, 'augment': False}

def test_contrast_family_has_fourteen_members(tmp_path):
    from experiments.make_revision.run_bridge import declare_family
    fam = declare_family(['iris','wine','breast_cancer','wine_quality','vehicle','segment','digits'])
    assert len(fam) == 14 and {f['kind'] for f in fam} == {'output_vs_knn', 'trained_vs_initial'}

def test_resolve_selected_matches_the_adaptive_wrapper_and_variants_need_a_resolved_configuration():
    config = bridge.bridge_candidates()[3]
    selected = bridge.resolve_selected(config, n_features=13, n_train=142)
    assert not {'embed_scale', 'degree_offset'} & selected.keys()
    assert {k: selected[k] for k in ('embed_dim', 'degree', 'augment')} == bridge.resolve(config, 13, 142)
    assert ({k: v for k, v in selected.items() if k not in ('embed_dim', 'degree', 'augment')}
            == {k: v for k, v in config.items() if k not in ('embed_scale', 'degree_offset')})
    with pytest.raises(ValueError, match='Resolve'):
        bridge.ablation_variants(config)                                  # abstract candidate, not resolved
    with pytest.raises(ValueError, match='seven-view'):
        bridge.ablation_variants({**selected, 'n_views': 3})
    params = dict(bridge.ablation_variants(selected))
    assert params['multiview_footrule_knn'] == {'n_views': 7, 'strategy': 'diverse',
                                                'embed_dim': selected['embed_dim'], 'degree': selected['degree']}
    assert params['borda_views7']['aggregation'] == 'borda' and params['no_augment']['augment'] is False
    assert params['views7'] == selected

def test_ablation_protocol_declares_the_variants_seeds_and_family():
    from experiments.make_revision.run_bridge import check_bridge_protocol, declare_family
    p = json.load(open('experiments/make_revision/protocols/2026-09-12/ablation.json'))
    assert p['production_family'] == 'ablation' and p['bridge_source']['protocol_id'] == 'arrowflow-v3-bridge-1'
    assert p['variants'] == list(bridge.ABLATION_VARIANTS) and p['fit_seeds'] == [8129, 19391, 39019]
    assert p['wallclock_cap_hours'] == 3 and p['contrast_family']['size'] == 14 and p['test_train_ratio'] == .25
    family = declare_family(p['datasets'], probe_architecture=p['contrast_family']['trained_vs_initial']['probe_architecture'])
    assert {f['model_a'] for f in family if f['kind'] == 'trained_vs_initial'} == {p['contrast_family']['trained_vs_initial']['model_a']}
    check_bridge_protocol(p, json.load(open('experiments/make_revision/protocols/2026-09-12/bridge.json')))
    with pytest.raises(ValueError, match='disagree'):
        check_bridge_protocol({**p, 'fit_seeds': [1, 2, 3]}, json.load(open('experiments/make_revision/protocols/2026-09-12/bridge.json')))

def test_prefix_and_borda_variants_equal_the_estimator_at_the_smaller_view_counts():
    # The reuse premise: view v depends only on derive_seed(seed, 'view', v), so the first k of seven
    # fitted views predict exactly what MultiViewArrowFlow(n_views=k) predicts, and the Borda variant is
    # the same seven views under the other aggregation rule.
    from sklearn.datasets import load_iris
    from experiments.make_revision.models import seed_fit
    from experiments.make_revision.multiview import MultiViewArrowFlow
    from experiments.make_revision.run_bridge import derived_predictions, fit_views7
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(2).permutation(len(y)); tr, te = idx[:100], idx[100:]
    params = {**bridge.FIXED, 'widths': [8], 'learning_rate': .1, 'iterations': 2, 'embed_dim': 16, 'degree': 1, 'augment': False}
    model, _, views, rankings = fit_views7(params, 8129, X[tr], y[tr], X[te])
    derived = derived_predictions(views, rankings, model.classes_)
    assert np.array_equal(derived['views7'], model.predict(X[te]))
    for variant, k in (('views1', 1), ('views3', 3)):
        seed_fit(8129)
        separate = MultiViewArrowFlow(**{**params, 'n_views': k}, seed=8129).fit(X[tr], y[tr])
        assert np.array_equal(derived[variant], separate.predict(X[te]))
    model.set_params(aggregation='borda')
    assert np.array_equal(derived['borda_views7'], model.predict(X[te]))

def test_knn_inner_candidate_sweep_equals_direct_multiview_knn_predictions():
    from sklearn.datasets import load_iris
    from experiments.make_revision.evaluation import make_splits
    from experiments.make_revision.models import array_hash, seed_fit
    from experiments.make_revision.multiview import MultiViewFootruleKNN
    from experiments.make_revision.run_bridge import knn_candidates, knn_inner_rows
    X, y = load_iris(return_X_y=True)
    split = make_splits(y, outer_folds=3, repeats=1, inner_folds=2, seed=27183)[0]
    candidates = knn_candidates({'knn_neighbors': [1, 3, 5, 11, 21], 'knn_weights': ['uniform', 'distance']})
    params = {'n_views': 2, 'strategy': 'diverse', 'embed_dim': 16, 'degree': 1}
    rows = knn_inner_rows(X, y, split, params, 8129, candidates)
    assert len(rows) == 20 and all(r['status'] == 'ok' for r in rows)
    for row in rows[::3]:
        inner = split['inner'][row['inner_fold']]
        seed_fit(8129)
        direct = MultiViewFootruleKNN(**params, **row['config'], seed=8129).fit(
            X[inner['train']], y[inner['train']], sample_ids=inner['train']).predict(X[inner['validation']])
        assert row['prediction_hash'] == array_hash(direct)
        assert row['score'] == float(np.mean(direct == y[inner['validation']]))

def test_compute_contrasts_leaves_pending_members_and_adjusts_only_complete_families():
    from experiments.make_revision.evaluation import holm_adjust
    from experiments.make_revision.run_bridge import compute_contrasts, declare_family
    folds, seeds, rng = [(0, f) for f in range(3)], [1, 2], np.random.RandomState(0)
    def rows(offsets):
        return [{'model_id': m, 'outer_repeat': r, 'outer_fold': f, 'model_seed': s, 'status': 'ok',
                 'accuracy': float(np.clip(.7 + o + .05 * rng.randn(), 0, 1))}
                for m, o in offsets.items() for r, f in folds for s in seeds]
    family = declare_family(['a', 'b'])
    ablation = {d: rows({'views7': .1, 'multiview_footrule_knn': 0}) for d in 'ab'}
    out, complete = compute_contrasts(family, ablation, None, q=.25, confidence=.95, folds=folds, ablation_seeds=seeds)
    assert not complete and [r['status'] for r in out] == ['computed', 'computed', 'pending', 'pending']
    assert all(r['holm_p_approximate'] is None for r in out) and 'p_approximate' not in out[-1]
    matched = {'rows': {d: rows({'af_h128_d1_trained': .1, 'af_h128_d1_untrained': 0}) for d in 'ab'},
               'seeds': {'af_h128_d1_trained': seeds, 'af_h128_d1_untrained': seeds}}
    out, complete = compute_contrasts(family, ablation, matched, q=.25, confidence=.95, folds=folds, ablation_seeds=seeds)
    assert complete and all(r['status'] == 'computed' for r in out)
    assert out[0]['df'] == 2 and out[0]['test_train_ratio'] == .25 and out[0]['mean_difference'] > 0
    assert [r['holm_p_approximate'] for r in out] == holm_adjust([r['p_approximate'] for r in out])
    with pytest.raises(ValueError, match='unique'):
        declare_family(['a', 'a'])

def matched_fixture(directory, datasets, folds, seeds, rng, *, primary_index=0):
    """A matched-study v3 directory in run_studies' output shape: protocol (Task 4's fields on the
    2026-09-11 template), per-fold results with model rows, and the verified summary's contrast rows."""
    from experiments.make_revision.evaluation import paired_corrected_interval
    from experiments.make_revision.matched import architecture_id
    from experiments.make_revision.run_revision import write_json
    base = json.load(open('experiments/make_revision/protocols/2026-09-11/matched.json'))
    protocol = dict(base, protocol_id='arrowflow-v3-matched-1', architectures=[[128], [64, 128], [64, 32]],
                    primary_architecture_index=primary_index, learning_rate=.1, iterations=200, validation_ratio=.1,
                    outer_folds=len({f for _, f in folds}), outer_repeats=len({r for r, _ in folds}), fit_seeds=seeds,
                    test_train_ratio=.25, datasets=datasets, primary_datasets=datasets, frozen=True)
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'environment.json', {'code_revision': 'matched-fixture'})
    widths = protocol['architectures'][primary_index]
    a, b = (f'{architecture_id(widths)}_d{len(widths)}_{state}' for state in ('trained', 'untrained'))
    contrasts = []
    for name in datasets:
        rows = []
        for r, f in folds:
            models = [{'dataset_id': name, 'model_id': m, 'model_seed': s, 'outer_repeat': r, 'outer_fold': f, 'status': 'ok',
                       'accuracy': float(np.clip(.8 + o + .05 * rng.randn(), 0, 1))} for m, o in ((a, .05), (b, 0.)) for s in seeds]
            write_json(directory/'results'/f'{name}__r{r}f{f}.json', {'status': 'ok', 'models': models})
            rows.extend(models)
        contrasts.append({'dataset_id': name, 'model_a': a, 'model_b': b,
                          **paired_corrected_interval(rows, a, b, q=.25, expected_folds=folds, expected_seeds={a: seeds, b: seeds})})
    write_json(directory/'summary.json', {'purpose': 'confirmatory', 'primary_contrasts': contrasts,
                                          'complete_primary_family': True, 'holm_applied': True})
    return protocol

def ablation_fixture(directory, datasets, folds, seeds, rng):
    """A verified ablation summary directory (only what `contrasts` reads), 7 datasets x 15 folds x 3 seeds."""
    from experiments.make_revision.run_bridge import CONTROL, PROTOCOL
    from experiments.make_revision.run_revision import write_json
    p = dict(json.loads(PROTOCOL.read_text()), datasets=datasets, outer_folds=len({f for _, f in folds}),
             outer_repeats=len({r for r, _ in folds}), fit_seeds=seeds, frozen=True)
    write_json(directory/'protocol.json', p)
    write_json(directory/'environment.json', {'code_revision': 'ablation-fixture'})
    write_json(directory/'manifest.json', {'purpose': 'confirmatory'})
    rows = {name: [{'dataset_id': name, 'variant_id': m, 'model_id': m, 'model_seed': s, 'outer_repeat': r, 'outer_fold': f,
                    'status': 'ok', 'accuracy': float(np.clip(.8 + o + .05 * rng.randn(), 0, 1))}
                   for m, o in (('views7', .05), (CONTROL, 0.)) for r, f in folds for s in seeds] for name in datasets}
    write_json(directory/'ablation_summary.json', {'code_revision': 'ablation-fixture', 'model_rows': rows})
    return p

def test_contrasts_complete_the_family_from_a_matched_v3_source_and_reject_wrong_sources(tmp_path):
    from experiments.make_revision import run_bridge as rb
    from experiments.make_revision.evaluation import holm_adjust
    datasets = ['iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits']
    folds, seeds, rng = [(r, f) for r in range(3) for f in range(5)], [8129, 19391, 39019], np.random.RandomState(11)
    ablation_fixture(tmp_path/'ablation', datasets, folds, seeds, rng)
    matched_fixture(tmp_path/'matched', datasets, folds, seeds, rng)
    result = rb.contrasts(tmp_path/'ablation', tmp_path/'matched', tmp_path/'out')
    rows = result['contrasts']
    assert len(rows) == 14 and result['holm_applied'] and all(r['status'] == 'computed' for r in rows)
    assert [r['kind'] for r in rows] == ['output_vs_knn'] * 7 + ['trained_vs_initial'] * 7
    assert all(r['df'] == 14 and r['n_folds'] == 15 and r['test_train_ratio'] == .25 for r in rows)
    assert [r['holm_p_approximate'] for r in rows] == holm_adjust([r['p_approximate'] for r in rows])
    assert {r['model_a'] for r in rows[7:]} == {'af_h128_d1_trained'} and {r['model_b'] for r in rows[7:]} == {'af_h128_d1_untrained'}
    assert result['sources']['matched_v3']['primary_architecture'] == [128] and result['sources']['matched_v3']['code_revision'] == 'matched-fixture'
    csv_lines = (tmp_path/'out'/'primary_contrasts_v3.csv').read_text().splitlines()
    assert len(csv_lines) == 15 and csv_lines[1].startswith('1,iris,output_vs_knn,ablation,views7,multiview_footrule_knn,accuracy,computed,15,14,0.25,0.95,')
    assert csv_lines[14].startswith('14,digits,trained_vs_initial,matched_v3,af_h128_d1_trained,af_h128_d1_untrained,accuracy,computed,')
    # a matched source whose primary architecture is not the declared [128] probe
    matched_fixture(tmp_path/'matched_wrong', datasets, folds, seeds, rng, primary_index=1)
    with pytest.raises(ValueError, match='primary architecture'):
        rb.contrasts(tmp_path/'ablation', tmp_path/'matched_wrong', tmp_path/'out_wrong')
    # a summary without the primary_contrasts block, and a summary row that disagrees with the per-fold records
    summary_path = tmp_path/'matched'/'summary.json'; original = summary_path.read_text()
    summary_path.write_text(json.dumps({'purpose': 'confirmatory'}))
    with pytest.raises(ValueError, match='primary_contrasts'):
        rb.contrasts(tmp_path/'ablation', tmp_path/'matched', tmp_path/'out_missing')
    data = json.loads(original); data['primary_contrasts'][0]['mean_difference'] += .01; summary_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='disagrees'):
        rb.contrasts(tmp_path/'ablation', tmp_path/'matched', tmp_path/'out_tampered')
    data = json.loads(original); del data['primary_contrasts'][0]['p_approximate']; summary_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='lacks p_approximate'):
        rb.contrasts(tmp_path/'ablation', tmp_path/'matched', tmp_path/'out_lacking')
    summary_path.write_text(original)
    (tmp_path/'matched'/'results'/'iris__r0f0.json').unlink()
    with pytest.raises(ValueError, match='matched result missing'):
        rb.contrasts(tmp_path/'ablation', tmp_path/'matched', tmp_path/'out_no_result')

def test_smoke_seals_selected_configurations_runs_every_variant_and_rejects_tampering(tmp_path):
    from experiments.make_revision import run_bridge as rb
    unfrozen = tmp_path/'unfrozen.json'          # independent of the committed protocol's freeze state
    unfrozen.write_text(json.dumps({**json.loads(rb.PROTOCOL.read_text()), 'frozen': False}))
    with pytest.raises(ValueError, match='frozen'):
        rb.main(['ablation', '--protocol', str(unfrozen), '--output', str(tmp_path)])
    rb.main(['smoke', '--output', str(tmp_path), '--workers', '2'])
    csv_lines = (tmp_path/'bridge_selected_configurations.csv').read_text().splitlines()
    assert csv_lines[0] == 'dataset_id,model_id,outer_repeat,outer_fold,config_id,config,fitting_seeds' and len(csv_lines) == 4
    jobs = json.loads((tmp_path/'planned_jobs.json').read_text())
    assert len(jobs) == 3 and [v['variant_id'] for v in jobs[0]['variants']] == list(rb.ABLATION_VARIANTS)
    assert jobs[0]['selected']['augment'] is True and jobs[0]['fit_sources']['no_augment'] == 'separate'   # 160 training rows
    report = json.loads((tmp_path/'ablation_summary.json').read_text())
    table = report['summaries']['synthetic']
    assert set(table['variants']) == set(rb.ABLATION_VARIANTS) and len(report['model_rows']['synthetic']) == 3 * 8 * 3
    assert table['views7_reproduces_bridge'] == {'matching_fold_seeds': 9, 'total_fold_seeds': 9, 'mismatches': []}
    assert all(e['metrics']['accuracy']['seeds_per_fold'] == 3 and e['metrics']['accuracy']['n_folds'] == 3
               for e in table['variants'].values())
    assert 'change_from_views7' not in table['variants']['views7'] and table['variants']['views1']['change_from_views7']['accuracy']['df'] == 2
    assert 'p_approximate' not in table['variants']['views1']['change_from_views7']['accuracy']    # descriptive interval only
    fits = json.loads((tmp_path/'results'/'synthetic__r0f0.json').read_text())['fits']
    reused = [f for f in fits if f['fit_source'] in ('prefix_of_views7', 'borda_of_views7', 'identical_to_views7')]
    assert len(reused) == 9 and all(f['fit_seconds'] == 0 and f['reused_from'] == f"views7__s{f['model_seed']}" for f in reused)
    assert all(f['fit_seconds'] > 0 for f in fits if f['fit_source'] in ('fitted', 'separate', 'inner_selected'))
    assert len(table['knn_selection']) == 3 and all(s['config']['n_neighbors'] in (1, 3, 5, 11, 21) for s in table['knn_selection'])
    record = json.loads((tmp_path/'predictions'/'synthetic__r0f0.jsonl').read_text().splitlines()[0])
    assert sorted(record) == sorted(rb.PREDICTION_KEYS)
    # contrasts without the matched source: the ablation member is computed, the probe member stays pending
    result = rb.contrasts(tmp_path, None, tmp_path/'contrasts')
    assert [r['status'] for r in result['contrasts']] == ['computed', 'pending'] and result['holm_applied'] is False
    assert (tmp_path/'contrasts'/'primary_contrasts_v3.csv').read_text().splitlines()[0].startswith('family_index,dataset_id,kind,')
    # tampering: an edited prediction, a removed artifact and a changed sealed selection are rejected
    path = tmp_path/'predictions'/'synthetic__r0f0.jsonl'; original = path.read_text()
    lines = original.splitlines(); first = json.loads(lines[0]); first['y_pred'] = (first['y_pred'] + 1) % 3
    path.write_text('\n'.join([json.dumps(first, sort_keys=True, separators=(',', ':'))] + lines[1:]) + '\n')
    with pytest.raises(ValueError, match='Incomplete'):
        rb.collect_results(tmp_path, allow_smoke=True)
    path.write_text(original)
    artifact = tmp_path/'artifacts'/'synthetic__r0f1'/'views'/'s8129.npz'; blob = artifact.read_bytes(); artifact.unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        rb.collect_results(tmp_path, allow_smoke=True)
    artifact.write_bytes(blob)
    sealed = tmp_path/'bridge_selections.json'; text = sealed.read_text()
    data = json.loads(text); data[0]['config']['learning_rate'] = .2; sealed.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        rb.verify(tmp_path, allow_smoke=True)
    sealed.write_text(text)
    assert set(rb.collect_results(tmp_path, allow_smoke=True)['rows']) == {'synthetic'}
    # with a matched-v3 source on the same panel and folds, the family completes and Holm applies
    matched_fixture(tmp_path/'matched', ['synthetic'], [(0, f) for f in range(3)], [8129, 19391, 39019], np.random.RandomState(5))
    complete = rb.contrasts(tmp_path, tmp_path/'matched', tmp_path/'contrasts-complete')
    assert complete['holm_applied'] and [r['status'] for r in complete['contrasts']] == ['computed', 'computed']

# ----------------------------------------------------------------------------- kNN readout confirmation family (Task 10)

BRIDGE = 'experiments/make_revision/protocols/2026-09-12/bridge.json'
BRIDGE_KNN = 'experiments/make_revision/protocols/2026-09-12/bridge_knn.json'

def test_bridge_knn_registry_holds_the_knn_family_and_the_bridge_registry_the_output_rule_with_the_same_sixteen_candidates():
    # arrowflow_full is not refitted in bridge_knn (the two-family pilot projected beyond the cap): its outer-fold results
    # come from the frozen bridge run, so each family is checked in its own registry through run_revision.get_registry.
    from experiments.make_revision.run_revision import get_registry
    protocol, reference = json.load(open(BRIDGE_KNN)), json.load(open(BRIDGE))
    assert protocol['registry'] == 'experiments.make_revision.bridge:bridge_knn_registry'
    assert reference['registry'] == 'experiments.make_revision.bridge:bridge_registry'
    reg, ref = get_registry(protocol['registry'], protocol), get_registry(reference['registry'], reference)
    assert list(reg) == ['arrowflow_full_knn'] + [name for name in ref if name != 'arrowflow_full']
    for spec, name in ((reg['arrowflow_full_knn'], 'arrowflow_full_knn'), (ref['arrowflow_full'], 'arrowflow_full')):
        assert spec.model_id == name and len(spec.candidates) == 16 and spec.stochastic is True
    assert reg['arrowflow_full_knn'].candidates == ref['arrowflow_full'].candidates == bridge.bridge_candidates()
    assert all(reg[name].candidates == ref[name].candidates and reg[name].stochastic == ref[name].stochastic
               for name in reg if name != 'arrowflow_full_knn')             # the bridge run's comparators, unchanged
    assert isinstance(reg['arrowflow_full_knn'].factory({}, 1), bridge.AdaptiveMultiViewKNN)
    assert type(ref['arrowflow_full'].factory({}, 1)) is bridge.AdaptiveMultiView

def test_adaptive_knn_wrapper_resolves_like_the_output_rule_and_records_the_readout_selection():
    from sklearn.datasets import load_iris
    from experiments.make_revision.comparisons import derive_seed
    from experiments.make_revision.evaluation import ModelSpec, _fit_predict, canonical_json
    from experiments.make_revision.multiview import MultiViewArrowFlowKNN, select_knn_readout
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(1).permutation(len(y))
    config = {**bridge.bridge_candidates()[0], 'iterations': 2}
    spec = ModelSpec('arrowflow_full_knn', bridge.arrowflow_full_knn_factory, [config], True)
    predictions, record = _fit_predict(spec, config, 8129, X[idx[:60]], y[idx[:60]], X[idx[60:70]])
    assert predictions.shape == (10,) and set(predictions) <= set(y)
    assert record['preprocessing_settings']['strategy'] == 'target_aware'          # view 0 of the diverse cycle
    assert record['preprocessing_settings']['embed_dim'] == 16 and record['preprocessing_settings']['degree'] == 3
    assert record['encoding_seconds'] > 0 and record['classifier_fit_seconds'] > 0 and record['query_encoding_seconds'] > 0
    meta = record['representation_metadata']
    assert meta['readout'] == 'knn_hidden' and len(meta['views']) == 7 and meta['readout_seconds'] > 0
    assert all(v['config']['n_neighbors'] in (1, 3, 5, 11, 21) and v['config']['weights'] in ('uniform', 'distance')
               and v['config_id'] == config_id(v['config']) and 0 <= v['inner_score'] <= 1 and v['folds'] == 3 for v in meta['views'])
    assert all(len(v['candidate_scores']) == 10 and v['candidate_scores'][v['config_id']] == v['inner_score']
               == max(v['candidate_scores'].values()) for v in meta['views'])  # every readout setting's selection score is logged
    canonical_json(record)                                                          # the fit log must serialize
    est = bridge.arrowflow_full_knn_factory(config, 8129).fit(X[idx[:60]], y[idx[:60]])
    ref = bridge.arrowflow_full_factory(config, 8129).fit(X[idx[:60]], y[idx[:60]])
    assert est.resolved_ == ref.resolved_ and isinstance(est.model_, MultiViewArrowFlowKNN)
    assert est.encoder_ is est.model_.views_[0][0] and len(est.readout_selections_) == 7
    logged = ('config', 'config_id', 'inner_score', 'folds', 'candidate_scores')
    for v, (enc, net) in enumerate(est.model_.views_):              # every choice re-derives from the fitted network's hidden ranking
        hidden = net.transform_orders(enc.transform(X[idx[:60]]))
        rederived = select_knn_readout(hidden, y[idx[:60]], seed=derive_seed(derive_seed(8129, 'view', v), 'readout_selection'))
        assert est.readout_selections_[v] == rederived
        assert est.representation_metadata_['views'][v] == {'view': v, **{key: rederived[key] for key in logged}}

def test_bridge_knn_protocol_pins_the_confirmation_design_and_tolerates_the_freeze():
    import hashlib, re
    from experiments.make_revision.multiview import KNN_READOUT_GRID, KNN_SELECTION_FOLDS
    old, new = json.load(open(BRIDGE)), json.load(open(BRIDGE_KNN))
    template_hash = hashlib.sha256(open(BRIDGE, 'rb').read()).hexdigest()
    provenance = {'frozen', 'frozen_at_utc', 'source_template_sha256', 'resource_decision', 'status'}
    changed = {'protocol_id', 'production_family', 'registry', 'primary_contrasts', 'primary_family_size', 'multiplicity',
               'wallclock_cap_hours'}
    added = {'knn_readout'}
    def pin(p):
        assert set(p) - provenance == (set(old) - provenance) | added
        assert {k for k in set(old) - provenance if old[k] != p[k]} == changed
        assert p['protocol_id'] == 'arrowflow-v3-bridge-knn-1' and p['production_family'] == 'bridge_knn'
        assert p['registry'] == 'experiments.make_revision.bridge:bridge_knn_registry'
        assert p['primary_contrasts'] == ['arrowflow_full_knn_vs_arrowflow_full'] and p['primary_family_size'] == 7
        assert p['wallclock_cap_hours'] == 10 and p['candidate_budget'] == 24 and p['datasets'] == old['datasets']
        assert p['fit_seeds'] == old['fit_seeds'] and p['split_seed'] == old['split_seed'] and p['full_method'] == old['full_method']
        assert p['knn_readout']['grid'] == KNN_READOUT_GRID and str(KNN_SELECTION_FOLDS) in p['knn_readout']['selection']
        assert p['knn_readout']['model_id'] == 'arrowflow_full_knn' and p['knn_readout']['aggregation'] == 'majority'
        reference = p['knn_readout']['reference']                                   # the output rule comes from the bridge run
        assert reference['model_id'] == 'arrowflow_full' and reference['protocol_id'] == old['protocol_id']
        assert reference['protocol_sha256'] == template_hash and old['frozen'] is True
        assert re.fullmatch('[0-9a-f]{40}', reference['code_revision']) and re.fullmatch('[0-9a-f]{64}', reference['summary_sha256'])
        assert 'arrowflow_full dropped' in p['resource_decision']
        assert 'knn_hidden adopted in the inner-fold laboratory, 7/7 datasets improved, mean +4.3 pp' in p['resource_decision']
        assert p['source_template_sha256'] == template_hash
        assert ('frozen_at_utc' in p) == bool(p['frozen'])
        if p['frozen']:
            assert p['frozen_at_utc'] >= '2026-09-12T22:24'     # after the laboratory verdict that adopted knn_hidden (UTC)
    pin(new)
    pin(dict(new, frozen=True, frozen_at_utc='2026-09-13T00:00:00+00:00', status='reviewed_and_piloted_before_confirmatory_scoring',
             resource_decision=new['resource_decision'] + '; pilot projection recorded at freeze'))
    with pytest.raises(AssertionError):
        pin(dict(new, source_template_sha256=old['source_template_sha256']))
    with pytest.raises(AssertionError):
        pin(dict(new, candidate_budget=16))
    unfrozen = {k: v for k, v in new.items() if k != 'frozen_at_utc'}             # the same pins hold before the freeze
    pin(dict(unfrozen, frozen=False, status='drafted_awaiting_training_only_pilot'))
    with pytest.raises(AssertionError):
        pin(dict(unfrozen, frozen=True))                                             # a freeze without its timestamp
    with pytest.raises(AssertionError):
        pin(dict(new, frozen=False, frozen_at_utc='2026-09-13T00:00:00+00:00'))      # a timestamp without the freeze
    with pytest.raises(AssertionError):
        pin(dict(new, frozen=True, frozen_at_utc='2026-09-12T20:00:00+00:00'))       # a freeze before the laboratory verdict
