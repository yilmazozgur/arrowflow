# tests/make_revision/test_devlab.py
"""Inner-fold laboratory (Task 7): verdict rule, one representation per unit, nested readout selection,
bridge-selection loading from a directory in the production result-file shape, synthetic end-to-end smoke."""
import hashlib
import json
import numpy as np
import pandas as pd
import pytest


def toy(n_per_class=30, seed=33):
    rng = np.random.RandomState(seed)
    y = np.tile([0, 1, 2], n_per_class)
    X = rng.randn(len(y), 4)
    X[np.arange(len(y)), y] += 1.5
    return X, y


TINY_ENCODER = {'embed_dim': 6, 'degree': 1}
TINY_NETWORK = {'widths': [4], 'learning_rate': .1, 'iterations': 2, 'batch_size': 32,
                'validation_ratio': .1, 'augment': False}


def tiny_splits(y):
    from experiments.make_revision.evaluation import make_splits
    return make_splits(y, 3, 1, 2, 27183)


def test_devlab_verdict_rule():
    from experiments.make_revision.devlab import adoption_verdict
    ref = {'iris': 90., 'wine': 95., 'breast_cancer': 94., 'wine_quality': 55., 'vehicle': 70., 'segment': 92., 'digits': 80.}
    better = {k: v + 1 for k, v in ref.items()}
    assert adoption_verdict(ref, better) == 'adopt'
    mixed = {**better, 'digits': ref['digits'] - 2}
    assert adoption_verdict(ref, mixed) == 'reject'          # one dataset worse by more than 1 pp
    few = {**ref, 'iris': 91., 'wine': 96., 'digits': 81.}
    assert adoption_verdict(ref, few) == 'reject'            # only 3 of 7 improve
    edge = {**better, 'digits': ref['digits'] - 1.}
    assert adoption_verdict(ref, edge) == 'adopt'            # a loss of exactly 1 pp is tolerated
    with pytest.raises(ValueError):
        adoption_verdict(ref, {k: v for k, v in better.items() if k != 'digits'})


def test_evaluate_readouts_scores_every_readout_on_one_representation_per_unit(monkeypatch):
    from experiments.make_revision import devlab
    from experiments.make_revision.models import ArrowFlowEstimator, OrdinalEncoder
    X, y = toy()
    splits = tiny_splits(y)
    fits, encoders = [], []
    original_fit, original_encoder = ArrowFlowEstimator.train_initialized, OrdinalEncoder.fit

    def counted_fit(self, orders, labels):
        fits.append((self.widths, self.seed, len(labels)))
        return original_fit(self, orders, labels)

    def counted_encoder(self, data, labels=None):
        encoders.append(self.strategy)
        return original_encoder(self, data, labels)
    monkeypatch.setattr(ArrowFlowEstimator, 'train_initialized', counted_fit)
    monkeypatch.setattr(OrdinalEncoder, 'fit', counted_encoder)
    seeds = [8129, 19391]
    frame = devlab.evaluate_readouts(X, y, 'synthetic', splits, TINY_ENCODER, TINY_NETWORK, seeds)
    assert isinstance(frame, pd.DataFrame) and list(frame.columns) == list(devlab.COLUMNS)
    units = len(splits) * 2 * len(seeds)
    assert len(frame) == units * len(devlab.READOUTS)
    assert len(fits) == units and len(encoders) == units and set(encoders) == {'target_aware'}   # view 0 only
    assert set(frame['readout_id']) == set(devlab.READOUTS) and frame['accuracy'].between(0, 1).all()
    assert frame['outer_repeat'].eq(0).all() and set(frame['outer_fold']) == {0, 1, 2} and set(frame['inner_fold']) == {0, 1}
    per_unit = frame.groupby(['outer_fold', 'inner_fold', 'model_seed'])['readout_id'].nunique()
    assert len(per_unit) == units and per_unit.eq(len(devlab.READOUTS)).all()
    # per-fold configurations may also be given as one entry per split
    again = devlab.evaluate_readouts(X, y, 'synthetic', splits[:1], [TINY_ENCODER], [TINY_NETWORK], seeds[:1])
    assert len(again) == 2 * len(devlab.READOUTS)


def test_unit_reference_is_the_single_view_output_rule_and_selection_never_sees_the_scored_fold():
    from experiments.make_revision import devlab
    from experiments.make_revision.comparisons import BordaClassifier, StableFootruleKNN
    from experiments.make_revision.models import seed_fit
    from experiments.make_revision.multiview import MultiViewArrowFlow
    from experiments.make_revision.evaluation import config_id
    from arrowflow.ranking import inverse_positions
    from arrowflow.readouts import KPrototypeBorda
    X, y = toy()
    split, inner_fold, seed = tiny_splits(y)[1], 0, 19391
    record = devlab.evaluate_unit(X, y, split, inner_fold, seed, TINY_ENCODER, TINY_NETWORK, dataset_id='synthetic')
    assert record['status'] == 'ok' and [r['readout_id'] for r in record['readouts']] == list(devlab.READOUTS)
    train, validation = split['inner'][inner_fold]['train'], split['inner'][inner_fold]['validation']
    assert record['fit_rows'] == train and record['validation_rows'] == validation
    # nested selection: every selection fit and query row lies inside the inner training partition
    assert record['selection_splits'] and all(
        set(part['fit_rows']) | set(part['query_rows']) <= set(train) and not set(part['fit_rows']) & set(part['query_rows'])
        for part in record['selection_splits'])
    assert not any(set(part['query_rows']) & set(validation) for part in record['selection_splits'])
    for readout, candidates in (('knn_hidden', devlab.knn_candidates()), ('kproto_borda', devlab.kproto_candidates())):
        chosen = record['selection'][readout]
        assert chosen['config'] in candidates and chosen['config_id'] == config_id(chosen['config'])
        rows = chosen['rows']
        assert len(rows) == len(candidates) * len(record['selection_splits']) and all(r['status'] == 'ok' for r in rows)
        assert {r['selection_fold'] for r in rows} == set(range(len(record['selection_splits'])))
    # the reference is exactly the fitted single view's own output rule; readouts act on its hidden positions
    params = devlab.laboratory_params(TINY_ENCODER, TINY_NETWORK)
    assert params['n_views'] == 1 and params['strategy'] == 'diverse'
    seed_fit(seed)
    model = MultiViewArrowFlow(**params, seed=seed).fit(X[train], y[train])
    enc, net = model.views_[0]
    assert enc.strategy == 'target_aware'
    expected = {'output_rule': net.predict_orders(enc.transform(X[validation]))}
    hidden_train, hidden_validation = net.transform_orders(enc.transform(X[train])), net.transform_orders(enc.transform(X[validation]))
    knn = record['selection']['knn_hidden']['config']
    expected['knn_hidden'] = StableFootruleKNN(**knn, input_kind='positions').fit(
        hidden_train, y[train], sample_ids=train).predict(hidden_validation)
    expected['borda1'] = BordaClassifier().fit(inverse_positions(hidden_train), y[train]).predict(inverse_positions(hidden_validation))
    k = record['selection']['kproto_borda']['config']['k']
    expected['kproto_borda'] = KPrototypeBorda(k=k, seed=seed).fit(hidden_train, y[train]).predict(hidden_validation)
    for row in record['readouts']:
        assert row['accuracy'] == pytest.approx(float(np.mean(expected[row['readout_id']] == y[validation])))
    assert record['readouts'][0]['config'] == {} and record['readouts'][1]['config'] == knn
    rows = devlab.unit_rows(record)
    assert [r['readout_id'] for r in rows] == list(devlab.READOUTS) and all(set(r) == set(devlab.COLUMNS) for r in rows)


def test_failed_network_fit_leaves_explicit_failed_rows_for_every_readout():
    from experiments.make_revision import devlab
    X, y = toy()
    split = tiny_splits(y)[0]
    record = devlab.evaluate_unit(X, y, split, 0, 8129, TINY_ENCODER, {**TINY_NETWORK, 'widths': [0]}, dataset_id='synthetic')
    assert record['status'] == 'failed' and record['exception']
    assert [r['readout_id'] for r in record['readouts']] == list(devlab.READOUTS)
    assert all(r['status'] == 'failed' and r['accuracy'] is None for r in record['readouts'])
    frame = devlab.rows_frame(devlab.unit_rows(record))
    assert len(frame) == len(devlab.READOUTS) and frame['accuracy'].isna().all()


def mimic_bridge(directory, rng, *, frozen=True, outer_repeats=2, datasets=('synthetic',)):
    """A bridge output directory in the production shape: protocol/candidates/environment.json, per-dataset
    manifest/splits/data.npz, and results/<dataset>__arrowflow_full__r<r>f<f>.json + .fits.jsonl whose selection
    block carries the complete inner history (screening with the first seed, finalists reranked with the others).
    Every dataset carries the same toy data; `selected` maps (repeat, fold) to the first dataset's choice."""
    from experiments.make_revision import run_bridge as rb
    from experiments.make_revision.bridge import FIXED
    from experiments.make_revision.evaluation import canonical_json, config_id, dataset_fingerprint, make_splits, metric_values
    from experiments.make_revision.run_revision import write_json
    X, y = toy()
    features, labels = [f'x{i}' for i in range(4)], ['a', 'b', 'c']
    protocol = {**json.loads(rb.BRIDGE_PROTOCOL.read_text()), 'datasets': list(datasets), 'outer_folds': 3,
                'outer_repeats': outer_repeats, 'inner_folds': 2, 'frozen': frozen}
    seeds = protocol['fit_seeds']
    candidates = [{**FIXED, 'widths': [4], 'learning_rate': lr, 'iterations': 2, 'embed_scale': 1, 'degree_offset': -1}
                  for lr in (.1, .2)]
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'candidates.json', {'arrowflow_full': {'stochastic': True, 'candidates': candidates,
                                                                'config_ids': [config_id(c) for c in candidates]}})
    write_json(directory/'environment.json', {'code_revision': 'mimic', 'source_hashes': {}})
    splits = make_splits(y, 3, outer_repeats, 2, protocol['split_seed'])
    selected = {}
    for name in datasets:
        manifest = {'dataset_id': name, 'feature_names': features, 'label_map': labels, 'shape': [len(y), 4],
                    'class_counts': [len(y) // 3] * 3, 'dataset_hash': dataset_fingerprint(X, y, features, labels),
                    'splits_hash': config_id(splits)}
        write_json(directory/name/'manifest.json', manifest)
        write_json(directory/name/'splits.json', splits)
        np.savez_compressed(directory/name/'data.npz', X=X, y=y)
        for split in splits:
            r, f = split['outer_repeat'], split['outer_fold']
            fits = []

            def fit_row(config, seed, i):
                inner = split['inner'][i]
                return {'stage': 'inner', 'model_id': 'arrowflow_full', 'config_id': config_id(config), 'config': config,
                        'model_seed': seed, 'inner_fold': i, 'outer_repeat': r, 'outer_fold': f, 'fit_rows': inner['train'],
                        'validation_rows': inner['validation'], 'status': 'ok', 'score': float(round(.5 + .4 * rng.rand(), 6))}

            def mean_score(config):
                return float(np.mean([row['score'] for row in fits if row['config_id'] == config_id(config)]))
            for config in candidates:
                fits.extend(fit_row(config, seeds[0], i) for i in range(2))
            finalists = sorted(candidates, key=lambda c: (-mean_score(c), config_id(c)))[:3]
            for config in finalists:
                fits.extend(fit_row(config, seed, i) for seed in seeds[1:] for i in range(2))
            best = sorted(finalists, key=lambda c: (-mean_score(c), config_id(c)))[0]
            selection = {'config': best, 'config_id': config_id(best), 'inner_score': mean_score(best), 'fits': fits,
                         'finalist_ids': [config_id(c) for c in finalists]}
            common = {'dataset_id': name, 'dataset_hash': manifest['dataset_hash'], 'outer_repeat': r, 'outer_fold': f,
                      'model_id': 'arrowflow_full', 'config_id': selection['config_id'], 'code_revision': 'mimic',
                      'view_id': 'ensemble', 'condition': 'clean', 'perturbation_seed': None}
            models, predictions = [], []
            for seed in seeds:
                pred = y[split['test']].copy()
                pred[::4] = (pred[::4] + 1) % 3
                models.append(dict(common, model_seed=seed, config=best, stage='outer', fit_rows=split['train'],
                                   test_rows=split['test'], status='ok', **metric_values(y[split['test']], pred)))
                predictions.extend(dict(common, model_seed=seed, sample_id=int(s), y_true=int(y[s]), y_pred=int(p))
                                   for s, p in zip(split['test'], pred))
            stem = f'{name}__arrowflow_full__r{r}f{f}'
            write_json(directory/'results'/f'{stem}.json',
                       {'status': 'ok', 'selection': selection, 'models': models, 'predictions': predictions})
            (directory/'results'/f'{stem}.fits.jsonl').write_text(''.join(canonical_json(row) + '\n' for row in fits + models))
            if name == datasets[0]:
                selected[(r, f)] = best
    return {'X': X, 'y': y, 'splits': splits, 'candidates': candidates, 'selected': selected, 'protocol': protocol}


def test_bridge_selections_load_from_a_directory_in_the_production_result_shape(tmp_path):
    from experiments.make_revision import devlab
    from experiments.make_revision.bridge import resolve_selected
    fixture = mimic_bridge(tmp_path/'bridge', np.random.RandomState(11))
    loaded = devlab.load_selections(tmp_path/'bridge', ['synthetic'])
    panel = loaded['datasets']['synthetic']
    assert loaded['bridge']['protocol']['fit_seeds'] == [8129, 19391, 39019]
    assert np.array_equal(panel['X'], fixture['X']) and np.array_equal(panel['y'], fixture['y'])
    assert [(j['outer_repeat'], j['outer_fold']) for j in panel['jobs']] == [(0, 0), (0, 1), (0, 2)]   # outer repeat 0 only
    for job, split in zip(panel['jobs'], panel['splits']):
        assert (split['outer_repeat'], split['outer_fold']) == (job['outer_repeat'], job['outer_fold'])
        assert job['config'] == fixture['selected'][(0, job['outer_fold'])]
        assert job['selected'] == resolve_selected(job['config'], 4, len(split['train']))
        configuration = devlab.laboratory_configuration(job['selected'])
        assert configuration['encoder_settings'] == {'embed_dim': 16, 'degree': 2}          # 4 features, degree 3 - 1
        assert configuration['network_params'] == {'widths': [4], 'learning_rate': job['config']['learning_rate'],
                                                   'iterations': 2, 'batch_size': 32, 'validation_ratio': .1, 'augment': False}
        params = devlab.laboratory_params(**configuration)
        assert params['n_views'] == 1 and params['strategy'] == 'diverse' and params['aggregation'] == 'majority'
    units = devlab.plan_units(loaded)
    assert len(units) == 3 * 2 * 3 and {u['model_seed'] for u in units} == {8129, 19391, 39019}
    assert {(u['outer_repeat'], u['outer_fold'], u['inner_fold']) for u in units} == {(0, f, i) for f in range(3) for i in range(2)}
    # a selected configuration that disagrees with its own inner history is rejected, as in the ablation runner
    path = tmp_path/'bridge'/'results'/'synthetic__arrowflow_full__r0f1.json'
    content = json.loads(path.read_text())
    other = next(c for c in fixture['candidates'] if c != content['selection']['config'])
    content['selection']['config'] = other
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        devlab.load_selections(tmp_path/'bridge', ['synthetic'])
    # an unfrozen bridge source is refused unless explicitly allowed for a synthetic smoke
    unfrozen = mimic_bridge(tmp_path/'unfrozen', np.random.RandomState(12), frozen=False)
    assert unfrozen['protocol']['frozen'] is False
    with pytest.raises(ValueError, match='frozen'):
        devlab.load_selections(tmp_path/'unfrozen', ['synthetic'])
    assert len(devlab.load_selections(tmp_path/'unfrozen', ['synthetic'], allow_smoke=True)['datasets']['synthetic']['jobs']) == 3


def test_smoke_runs_the_laboratory_end_to_end_and_summarizes_verdicts(tmp_path):
    from experiments.make_revision import devlab
    with pytest.raises(ValueError, match='family'):          # no silent family default
        devlab.main(['smoke', '--output', str(tmp_path/'smoke'), '--workers', '2'])
    devlab.main(['smoke', '--family', 'readouts', '--output', str(tmp_path/'smoke'), '--workers', '2'])
    output = tmp_path/'smoke'
    with pytest.raises(ValueError, match='frozen'):          # the production command refuses an unfrozen bridge source
        devlab.main(['readouts', '--bridge-source', str(output/'synthetic_bridge'), '--output', str(tmp_path/'lab')])
    assert not (tmp_path/'lab').exists()
    with pytest.raises(ValueError, match='family'):
        devlab.main(['summary', '--output', str(output)])
    report = json.loads((output/'readouts_summary.json').read_text())
    assert report['purpose'].startswith('inner_fold_laboratory') and report['datasets'] == ['synthetic']
    assert report['outer_repeat'] == 0 and report['reference'] == 'output_rule' and report['readouts'] == list(devlab.READOUTS)
    assert report['units'] == {'planned': 18, 'completed': 18, 'failed': 0}
    table = report['mean_inner_accuracy']['synthetic']
    assert set(table) == set(devlab.READOUTS) and all(0 <= v <= 1 for v in table.values())
    assert set(report['verdicts']) == set(devlab.READOUTS) - {'output_rule'}
    for verdict in report['verdicts'].values():          # one dataset: the 5-of-7 rule does not apply
        assert verdict['verdict'] == 'not_applicable' and set(verdict['change_pp']) == {'synthetic'}
    assert report['verdict_rule'] == {'minimum_improved': 5, 'tolerance_pp': 1.0, 'minimum_datasets': 5}
    assert report['family'] == 'readouts' and report['protocol_id'] == devlab.DEFAULT_PROTOCOL['protocol_id']
    rows = pd.read_csv(output/'readouts_rows.csv')
    assert list(rows.columns) == list(devlab.COLUMNS) and len(rows) == 18 * len(devlab.READOUTS)
    plan = json.loads((output/'planned_units.json').read_text())
    assert len(plan) == 18 and all((output/'results'/f'{u["stem"]}.json').is_file() for u in plan)
    assert set(report['selected_configurations']['synthetic']) == {'knn_hidden', 'kproto_borda'}
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['family'] == 'readouts' and protocol['families'] == devlab.DEFAULT_PROTOCOL['families']
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['family'] == 'readouts' and manifest['protocol_file_sha256'] and set(manifest['candidates']) == {'knn_hidden', 'kproto_borda'}
    assert all('bridge_accuracy' not in job for job in json.loads((output/'bridge_selections.json').read_text()))
    pilot_report = json.loads((output/'pilot.json').read_text())      # the smoke pilots one cell before running
    assert pilot_report['family'] == 'readouts' and pilot_report['planned_cells'] == 18 and len(pilot_report['cells']) == 1
    assert pilot_report['projection']['wallclock_cap_hours'] == 1 and pilot_report['projection']['serial_hours'] > 0
    authorisation = manifest['pilot_authorisation']                      # the run records the pilot that authorised it
    assert authorisation['mode'] == 'pilot' and authorisation['workers'] == 2 and authorisation['projection']['within_cap'] is True
    assert authorisation['pilot_sha256'] == hashlib.sha256((output/'pilot.json').read_bytes()).hexdigest()
    assert report['pilot_authorisation'] == authorisation and len(list((output/'pilots').glob('pilot_*.json'))) == 1
    # the permlvq family shares the output directory (its own root O/permlvq) and compares with knn_hidden
    devlab.main(['smoke', '--family', 'permlvq', '--output', str(output), '--workers', '2'])
    lvq = json.loads((output/'permlvq_summary.json').read_text())
    assert lvq['family'] == 'permlvq' and lvq['readouts'] == list(devlab.PERMLVQ_VARIANTS) and lvq['reference'] == 'output_rule'
    assert lvq['units'] == {'planned': 18, 'completed': 18, 'failed': 0} and lvq['purpose'].startswith('inner_fold_laboratory_permlvq')
    assert lvq['mean_inner_accuracy']['synthetic']['output_rule'] == report['mean_inner_accuracy']['synthetic']['output_rule']
    assert set(lvq['verdicts']) == set(devlab.PERMLVQ_VARIANTS) - {'output_rule'}
    assert all(v['verdict'] == 'not_applicable' for v in lvq['verdicts'].values())
    assert set(lvq['selected_configurations']['synthetic']) == set(devlab.PERMLVQ_VARIANTS) - {'output_rule'}
    assert lvq['budgets']['synthetic'] == ['{"batch_size":32,"iterations":1,"learning_rate":0.1,"p_correct":0.01,"repulsion":true}']
    knn = lvq['comparison_knn_hidden']
    assert knn['status'] == 'ok' and knn['mean_inner_accuracy'] == {'synthetic': report['mean_inner_accuracy']['synthetic']['knn_hidden']}
    assert set(knn['verdicts']) == set(lvq['verdicts']) and all(v['verdict'] == 'not_applicable' for v in knn['verdicts'].values())
    lvq_rows = pd.read_csv(output/'permlvq_rows.csv')
    assert len(lvq_rows) == 18 * len(devlab.PERMLVQ_VARIANTS) and set(lvq_rows['readout_id']) == set(devlab.PERMLVQ_VARIANTS)
    lvq_root = output/'permlvq'
    assert json.loads((lvq_root/'pilot.json').read_text())['family'] == 'permlvq'
    assert json.loads((lvq_root/'manifest.json').read_text())['family'] == 'permlvq'
    assert len(list((lvq_root/'results').glob('*.json'))) == 18
    devlab.write_summary(lvq_root, allow_smoke=True, destination=output, comparison=output/'readouts_summary.json')   # identical content
    with pytest.raises(ValueError):                                              # the production summary refuses smoke output
        devlab.main(['summary', '--family', 'permlvq', '--output', str(output)])
    assert devlab.summary(lvq_root, allow_smoke=True)['comparison_knn_hidden']['status'] == 'absent'
    # summaries are reproducible from the saved records, and a tampered record is rejected
    again = devlab.summary(output, allow_smoke=True)
    assert again['mean_inner_accuracy'] == report['mean_inner_accuracy']
    path = output/'results'/f'{plan[0]["stem"]}.json'
    saved = json.loads(path.read_text())
    saved['provenance']['model_seed'] = 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        devlab.summary(output, allow_smoke=True)


def test_verdict_is_not_applicable_below_five_datasets():
    from experiments.make_revision.devlab import adoption_verdict, verdict_record
    ref = {'iris': 90., 'wine': 95., 'digits': 80.}
    assert adoption_verdict(ref, {k: v + 5 for k, v in ref.items()}) == 'not_applicable'
    assert adoption_verdict(ref, {k: v + 5 for k, v in ref.items()}, minimum_datasets=3) == 'reject'    # 3 < 5 improved
    assert adoption_verdict(ref, {k: v + 5 for k, v in ref.items()}, minimum_datasets=3, minimum_improved=3) == 'adopt'
    record = verdict_record(ref, {k: v + 5 for k, v in ref.items()})
    assert record['verdict'] == 'not_applicable' and record['datasets'] == 3 and record['improved'] == 3
    with pytest.raises(ValueError):
        adoption_verdict(ref, {'iris': 91., 'wine': 96.})


def test_protocol_file_drives_the_laboratory_design(tmp_path):
    from experiments.make_revision import devlab
    from experiments.make_revision.models import ArrowFlowEstimator
    protocol = devlab.load_protocol(devlab.PROTOCOL_FILE)
    assert protocol == devlab.DEFAULT_PROTOCOL and protocol['frozen'] is False
    assert protocol['datasets'] == ['iris', 'wine', 'breast_cancer', 'wine_quality', 'vehicle', 'segment', 'digits']
    assert protocol['outer_repeat'] == 0 and protocol['inner_folds'] == 3 and protocol['fit_seeds'] == [8129, 19391, 39019]
    assert devlab.READOUTS == ('output_rule', 'knn_hidden', 'borda1', 'kproto_borda')
    assert devlab.PERMLVQ_VARIANTS == ('output_rule', 'lvq1_borda_plurality', 'lvq1_median_plurality', 'lvq2_borda_plurality',
                                       'lvq1_borda_nearest', 'lvq1_borda_borda', 'lvq1_median_priorlr_plurality')
    assert devlab.KNN_GRID == {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance']} and devlab.KPROTO_GRID == {'k': [2, 4, 8]}
    assert sorted(c['prototypes_per_class'] for c in devlab.permlvq_candidates()) == [4, 8, 16] and len(devlab.knn_candidates()) == 10
    assert devlab.WALLCLOCK_CAP_HOURS == {'readouts': 1, 'permlvq': 2} and devlab.MAX_WORKERS == 16
    assert devlab.SELECTION_FOLDS == 3 and devlab.SELECTION_SEED == 20260912
    assert devlab.VERDICT_RULE == {'minimum_improved': 5, 'tolerance_pp': 1.0, 'minimum_datasets': 5}
    lvq = protocol['families']['permlvq']
    assert lvq['p_correct'] == ArrowFlowEstimator().p_correct and lvq['readout_k_grid'] == [1, 3, 5, 9] and lvq['repulsion'] is True
    sets = devlab.candidate_sets('permlvq')
    assert set(sets) == set(devlab.PERMLVQ_VARIANTS) - {'output_rule'}
    assert sorted(c['prototypes_per_class'] for c in sets['lvq1_borda_nearest']) == [4, 8, 16] and all('k' not in c for c in sets['lvq1_borda_nearest'])
    for variant in ('lvq1_borda_plurality', 'lvq1_median_plurality', 'lvq2_borda_plurality', 'lvq1_borda_borda', 'lvq1_median_priorlr_plurality'):
        assert sorted((c['prototypes_per_class'], c['k']) for c in sets[variant]) == [(m, k) for m in (4, 8, 16) for k in (1, 3, 5, 9)]
    budget = devlab.lvq_budget({'iterations': 200, 'batch_size': 32, 'learning_rate': .2})
    assert budget == {'iterations': 200, 'batch_size': 32, 'learning_rate': .2, 'p_correct': .01, 'repulsion': True}
    assert devlab.variant_model('lvq1_median_priorlr_plurality', {'prototypes_per_class': 4, 'k': 3}, budget, 1).prior_weight == .2
    assert devlab.variant_model('lvq1_median_plurality', {'prototypes_per_class': 4, 'k': 3}, budget, 1).prior_weight == 1.
    assert devlab.variant_model('lvq1_borda_nearest', {'prototypes_per_class': 4}, budget, 1).readout == 'nearest'
    assert devlab.laboratory_paths(tmp_path, 'readouts') == (tmp_path, tmp_path)
    assert devlab.laboratory_paths(tmp_path, 'permlvq') == (tmp_path/'permlvq', tmp_path)
    assert devlab.comparison_path(tmp_path, 'permlvq') == tmp_path/'readouts_summary.json' and devlab.comparison_path(tmp_path, 'readouts') is None
    broken = {**protocol, 'families': {'readouts': protocol['families']['readouts']}}
    (tmp_path/'broken.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        devlab.load_protocol(tmp_path/'broken.json')
    with pytest.raises(ValueError):
        devlab.family_spec('ensemble')


def test_permlvq_unit_variants_match_direct_fits_on_the_view0_encodings():
    from experiments.make_revision import devlab
    from experiments.make_revision.models import seed_fit
    from experiments.make_revision.multiview import MultiViewArrowFlow
    from experiments.make_revision.evaluation import config_id
    from arrowflow.ranking import inverse_positions
    X, y = toy()
    split, inner_fold, seed = tiny_splits(y)[2], 1, 8129
    record = devlab.evaluate_permlvq_unit(X, y, split, inner_fold, seed, TINY_ENCODER, TINY_NETWORK, dataset_id='synthetic')
    assert record['status'] == 'ok' and record['family'] == 'permlvq'
    assert [r['readout_id'] for r in record['readouts']] == list(devlab.PERMLVQ_VARIANTS)
    assert record['budget'] == {'iterations': 2, 'batch_size': 32, 'learning_rate': .1, 'p_correct': .01, 'repulsion': True}
    assert record['encodings']['vocabulary_size'] == TINY_ENCODER['embed_dim']
    train, validation = split['inner'][inner_fold]['train'], split['inner'][inner_fold]['validation']
    assert record['fit_rows'] == train and record['validation_rows'] == validation
    assert record['selection_splits'] and not any(set(part['query_rows']) & set(validation) for part in record['selection_splits'])
    for variant in devlab.PERMLVQ_VARIANTS[1:]:
        candidates = devlab.permlvq_candidates(variant=variant)
        chosen = record['selection'][variant]
        assert chosen['config'] in candidates and chosen['config_id'] == config_id(chosen['config'])
        assert len(chosen['rows']) == len(candidates) * len(record['selection_splits']) and all(r['status'] == 'ok' for r in chosen['rows'])
        assert ('k' in chosen['config']) == (devlab.VARIANT_SPECS[variant]['readout'] in devlab.K_READOUTS)
    # the fit-once-per-prototype-count selection equals a fit per candidate
    part, variant = record['selection_splits'][0], 'lvq1_borda_borda'
    rows = {r['config_id']: r['score'] for r in record['selection'][variant]['rows'] if r['selection_fold'] == 0}
    seed_fit(seed)
    model = MultiViewArrowFlow(**devlab.laboratory_params(TINY_ENCODER, TINY_NETWORK), seed=seed).fit(X[train], y[train])
    enc, net = model.views_[0]
    for config in devlab.permlvq_candidates(variant=variant):
        pred = devlab.variant_model(variant, config, record['budget'], seed).fit_orders(
            enc.transform(X[part['fit_rows']]), y[part['fit_rows']]).predict_orders(enc.transform(X[part['query_rows']]))
        assert rows[config_id(config)] == pytest.approx(float(np.mean(pred == y[part['query_rows']])))
    seed_fit(seed)
    model = MultiViewArrowFlow(**devlab.laboratory_params(TINY_ENCODER, TINY_NETWORK), seed=seed).fit(X[train], y[train])
    enc, net = model.views_[0]
    orders_train, orders_validation = enc.transform(X[train]), enc.transform(X[validation])
    assert record['encodings']['hashes']['train'] == devlab.array_hash(inverse_positions(orders_train))
    expected = {'output_rule': net.predict_orders(orders_validation)}
    for variant in devlab.PERMLVQ_VARIANTS[1:]:
        clf = devlab.variant_model(variant, record['selection'][variant]['config'], record['budget'], seed)
        spec = devlab.VARIANT_SPECS[variant]
        assert len(clf.layers) == spec['depth'] and clf.readout == spec['readout'] and clf.aggregation == spec['aggregation']
        assert clf.iterations == 2 and clf.learning_rate == .1 and clf.p_correct == .01
        assert clf.k == record['selection'][variant]['config'].get('k', 1)
        assert clf.prior_weight == (.1 if variant == 'lvq1_median_priorlr_plurality' else 1.)
        expected[variant] = clf.fit_orders(orders_train, y[train]).predict_orders(orders_validation)
    for row in record['readouts']:
        assert row['accuracy'] == pytest.approx(float(np.mean(expected[row['readout_id']] == y[validation])))
    rows = devlab.unit_rows(record)
    assert [r['readout_id'] for r in rows] == list(devlab.PERMLVQ_VARIANTS) and all(set(r) == set(devlab.COLUMNS) for r in rows)
    frame = devlab.evaluate_readouts(X, y, 'synthetic', [split], TINY_ENCODER, TINY_NETWORK, [seed],
                                     readouts=devlab.PERMLVQ_VARIANTS, family='permlvq')
    assert len(frame) == 2 * len(devlab.PERMLVQ_VARIANTS) and set(frame['readout_id']) == set(devlab.PERMLVQ_VARIANTS)
    with pytest.raises(ValueError):
        devlab.evaluate_permlvq_unit(X, y, split, inner_fold, seed, TINY_ENCODER, TINY_NETWORK, dataset_id='synthetic',
                                     readouts=('output_rule', 'knn_hidden'))


def test_pilot_gate_requires_an_authorising_pilot_or_the_explicit_escape(tmp_path):
    from experiments.make_revision import devlab
    from experiments.make_revision.evaluation import config_id
    from experiments.make_revision.run_bridge import load_bridge
    mimic_bridge(tmp_path/'bridge', np.random.RandomState(13))
    bridge, protocol, root = load_bridge(tmp_path/'bridge'), devlab.DEFAULT_PROTOCOL, tmp_path/'lab'
    gate = dict(datasets=['synthetic'], bridge=bridge, protocol=protocol)
    assert devlab.check_pilot(root, 'readouts', 1, **gate) == {'mode': 'single_worker', 'workers': 1}   # one worker needs no pilot
    assert devlab.check_pilot(root, 'readouts', 4, skip=True, **gate)['mode'] == 'skip_pilot_check'
    with pytest.raises(ValueError, match='pilot'):
        devlab.check_pilot(root, 'readouts', 4, **gate)
    with pytest.raises(ValueError, match='pilot'):                               # the CLI gate, before anything is written
        devlab.main(['readouts', '--bridge-source', str(tmp_path/'bridge'), '--output', str(root), '--workers', '2'])
    assert not root.exists()
    report = devlab.pilot(root, tmp_path/'bridge', ['synthetic'], family='readouts', workers=4, allow_smoke=True)
    assert report['family'] == 'readouts' and [c['dataset_id'] for c in report['cells']] == ['synthetic']
    cell = report['cells'][0]
    assert cell['status'] == 'ok' and cell['stem'] == 'synthetic__r0f0i0_s8129' and cell['planned_cells'] == 18
    assert set(cell['accuracy']) == set(devlab.READOUTS) and cell['network_fit_seconds'] > 0
    projection = report['projection']
    assert report['planned_cells'] == 18 and projection['serial_hours'] == pytest.approx(18 * cell['seconds'] / 3600)
    assert projection['ideal_wallclock_hours'] == pytest.approx(projection['serial_hours'] / 4)
    assert projection['contention_factor'] == 1.5 and projection['within_cap'] is True
    assert projection['conservative_wallclock_hours'] == pytest.approx(1.5 * projection['ideal_wallclock_hours'])
    assert report['bridge_files'] == bridge['files'] and report['protocol_hash'] == config_id(protocol)
    # re-piloting overwrites the latest pilot and keeps every pilot in the history
    again = devlab.pilot(root, tmp_path/'bridge', ['synthetic'], family='readouts', workers=4, allow_smoke=True)
    assert len(list((root/'pilots').glob('pilot_*.json'))) == 2
    assert json.loads((root/'pilot.json').read_text())['pilot_utc'] == again['pilot_utc']
    authorisation = devlab.check_pilot(root, 'readouts', 4, **gate)
    assert authorisation['mode'] == 'pilot' and authorisation['projection']['workers'] == 4 and authorisation['projection']['within_cap']
    assert authorisation['pilot_sha256'] == hashlib.sha256((root/'pilot.json').read_bytes()).hexdigest()
    # the run manifest records the authorising pilot, or the explicit escape
    devlab.prepare(root, tmp_path/'bridge', ['synthetic'], allow_smoke=True, purpose=devlab.SMOKE_PURPOSE, authorisation=authorisation)
    assert json.loads((root/'manifest.json').read_text())['pilot_authorisation'] == authorisation
    devlab.prepare(tmp_path/'escape', tmp_path/'bridge', ['synthetic'], allow_smoke=True, purpose=devlab.SMOKE_PURPOSE,
                   authorisation=devlab.check_pilot(root, 'readouts', 4, skip=True, **gate))
    assert json.loads((tmp_path/'escape'/'manifest.json').read_text())['pilot_authorisation']['mode'] == 'skip_pilot_check'
    # refusals: another family, a dataset without a cell, another bridge source, another protocol
    with pytest.raises(ValueError, match='family'):
        devlab.check_pilot(root, 'permlvq', 4, **gate)
    with pytest.raises(ValueError, match='datasets'):
        devlab.check_pilot(root, 'readouts', 4, **{**gate, 'datasets': ['synthetic', 'digits']})
    mimic_bridge(tmp_path/'bridge3', np.random.RandomState(14), outer_repeats=3)
    with pytest.raises(ValueError, match='bridge'):
        devlab.check_pilot(root, 'readouts', 4, **{**gate, 'bridge': load_bridge(tmp_path/'bridge3')})
    with pytest.raises(ValueError, match='protocol'):
        devlab.check_pilot(root, 'readouts', 4, **{**gate, 'protocol': {**protocol, 'protocol_id': 'other'}})
    # the projection is recomputed from the cells for the requested worker count; the file's own verdict is ignored
    saved = json.loads((root/'pilot.json').read_text())
    saved['cells'][0]['seconds'] = 4 * 3600.                  # 18 cells x 4 h / 16 workers x 1.5 = 6.75 h above the 1 h cap
    saved['projection']['within_cap'] = True
    (root/'pilot.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='cap'):
        devlab.check_pilot(root, 'readouts', 16, **gate)
    assert devlab.check_pilot(root, 'readouts', 16, skip=True, **gate)['mode'] == 'skip_pilot_check'
    saved['cells'][0].update(status='failed', seconds=.1)
    (root/'pilot.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='failed'):
        devlab.check_pilot(root, 'readouts', 4, **gate)
    # with the gate skipped, the production command still checks its protocol against the bridge (2 inner folds here)
    with pytest.raises(ValueError, match='disagrees'):
        devlab.main(['permlvq', '--bridge-source', str(tmp_path/'bridge'), '--output', str(tmp_path/'prod'), '--skip-pilot-check',
                     '--workers', '2'])
    assert not (tmp_path/'prod').exists()
    with pytest.raises(ValueError, match='--family'):
        devlab.main(['permlvq', '--family', 'readouts', '--bridge-source', str(tmp_path/'bridge'), '--output', str(tmp_path/'prod')])
    with pytest.raises(ValueError, match='--family'):
        devlab.main(['pilot', '--bridge-source', str(tmp_path/'bridge'), '--output', str(tmp_path/'prod')])


def test_pilot_records_a_failed_cell_and_still_writes_its_files(tmp_path, monkeypatch):
    from experiments.make_revision import devlab
    from experiments.make_revision.run_bridge import load_bridge
    mimic_bridge(tmp_path/'bridge', np.random.RandomState(15), datasets=('synthetic', 'broken'))
    original = devlab.evaluate_unit

    def failing_for_broken(X, y, split, inner_fold, seed, encoder_settings, network_params, *, dataset_id, **kwargs):
        if dataset_id == 'broken':
            network_params = {**network_params, 'widths': [0]}          # the network fit raises inside evaluate_unit
        return original(X, y, split, inner_fold, seed, encoder_settings, network_params, dataset_id=dataset_id, **kwargs)
    monkeypatch.setattr(devlab, 'evaluate_unit', failing_for_broken)
    root = tmp_path/'lab'
    report = devlab.pilot(root, tmp_path/'bridge', ['synthetic', 'broken'], family='readouts', workers=4, allow_smoke=True)
    cells = {cell['dataset_id']: cell for cell in report['cells']}
    assert cells['synthetic']['status'] == 'ok' and cells['synthetic']['seconds'] > 0 and cells['synthetic']['exception'] is None
    assert cells['broken']['status'] == 'failed' and cells['broken']['exception'] and cells['broken']['network_fit_seconds'] is None
    assert set(cells['broken']['readout_exceptions']) == set(devlab.READOUTS)
    assert report['projection'] is None and report['within_cap'] is False and report['wallclock_cap_hours'] == 1
    assert report['failed_cells'] == [{'dataset_id': 'broken', 'stem': cells['broken']['stem'], 'exception': cells['broken']['exception'],
                                       'readout_exceptions': cells['broken']['readout_exceptions']}]
    # both files are written, with the exception text; the healthy dataset's timing survives
    history = list((root/'pilots').glob('pilot_*.json'))
    assert len(history) == 1 and json.loads(history[0].read_text()) == json.loads((root/'pilot.json').read_text()) == report
    written = json.loads((root/'pilot.json').read_text())
    assert written['failed_cells'][0]['exception'] == cells['broken']['exception'] and written['projection'] is None
    # the refusal is check_pilot's: the failed dataset is refused, the healthy subset is authorised
    gate = dict(bridge=load_bridge(tmp_path/'bridge'), protocol=devlab.DEFAULT_PROTOCOL)
    with pytest.raises(ValueError, match='failed'):
        devlab.check_pilot(root, 'readouts', 4, datasets=['synthetic', 'broken'], **gate)
    with pytest.raises(ValueError, match='failed'):
        devlab.check_pilot(root, 'readouts', 4, datasets=['broken'], **gate)
    authorisation = devlab.check_pilot(root, 'readouts', 4, datasets=['synthetic'], **gate)
    assert authorisation['mode'] == 'pilot' and authorisation['projection']['datasets'] == ['synthetic']
    assert authorisation['projection']['planned_cells'] == 18 and authorisation['projection']['within_cap'] is True
