# tests/make_revision/test_gene.py
"""Gene-expression study: audited loader, fold-local selection, exact monotone invariance, prepass, corruption."""
from collections import defaultdict
import io
import json
import os
import tarfile
from pathlib import Path
import numpy as np
import pytest
from arrowflow.ranking import score_order
from experiments.make_revision import gene
from experiments.make_revision.evaluation import ModelSpec, config_id, dataset_fingerprint, make_splits, select_model

PROTOCOL = Path('experiments/make_revision/protocols/2026-09-12/gene.json')


def synthetic_genes(n=60, genes=30, classes=3, seed=5):
    """Nonnegative expression-like values with class-specific genes; every value distinct."""
    rng = np.random.RandomState(seed)
    y = np.tile(np.arange(classes), n // classes)
    X = rng.uniform(.5, 10., size=(n, genes))
    for c in range(classes):
        X[y == c, c * 3:(c + 1) * 3] += 6.
    return X, y


def tie_rich_genes(n=60, genes=30, classes=3, seed=7):
    """Real RNA-seq is tie-rich: exact zeros in every row, constant genes, duplicated values across genes."""
    X, y = synthetic_genes(n, genes, classes, seed)
    X = np.round(X, 1)                                   # duplicated values within and across rows
    X[:, 20:23] = 3.                                     # constant genes
    X[:, 25] = 0.                                        # an all-zero gene
    X[np.arange(n), 10 + np.arange(n) % 5] = 0.          # an exact zero in every row
    X[:, 1] = X[:, 0]                                    # two informative genes tied in every row
    return X, y


def fake_archive(directory, data_text, labels_text):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / gene.ARCHIVE_NAME
    with tarfile.open(path, 'w:gz') as tar:
        for member, text in zip(gene.MEMBERS, (data_text, labels_text)):
            payload = text.encode()
            info = tarfile.TarInfo(member)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


DATA_TEXT = ',gene_0,gene_1,gene_2,gene_3,gene_4,gene_5\n' + '\n'.join(
    f'sample_{i},' + ','.join(f'{v:.3f}' for v in row)
    for i, row in enumerate([[0., 2.5, 1., 3., .5, 4.], [1., 2., 3.5, .25, 0., 6.],
                             [5., 1., 0., 2., 3., 7.], [4., .5, 6., 1., 2., 8.]])) + '\n'
LABELS_TEXT = ',Class\nsample_0,B\nsample_1,A\nsample_2,B\nsample_3,A\n'
TINY_IDENTITY = {'shape': [4, 6], 'label_map': ['A', 'B'], 'class_counts': [2, 2]}


def test_parse_tcga_reads_uci_layout_sorted_labels_and_rejects_misaligned_rows():
    X, y, label_map, samples, genes = gene.parse_tcga(DATA_TEXT, LABELS_TEXT)
    assert X.shape == (4, 6) and X.dtype == float and X[0, 1] == 2.5
    assert label_map == ['A', 'B'] and y.tolist() == [1, 0, 1, 0]
    assert samples == [f'sample_{i}' for i in range(4)] and genes == [f'gene_{j}' for j in range(6)]
    with pytest.raises(ValueError, match='align'):
        gene.parse_tcga(DATA_TEXT, LABELS_TEXT.replace('sample_3', 'sample_9'))
    with pytest.raises(ValueError, match='finite'):
        gene.parse_tcga(DATA_TEXT.replace('0.500', 'nan'), LABELS_TEXT)


def test_loader_writes_audit_on_first_success_then_refuses_changed_bytes(tmp_path):
    cache = tmp_path / 'cache'
    fake_archive(cache, DATA_TEXT, LABELS_TEXT)
    audit = tmp_path / 'audit.json'
    X, y, manifest = gene.load_tcga(cache, audit_path=audit, identity=TINY_IDENTITY)
    assert X.shape == (4, 6) and y.tolist() == [1, 0, 1, 0] and manifest['dataset_id'] == gene.GENE_ID
    recorded = json.loads(audit.read_text())
    assert set(recorded['member_sha256']) == set(gene.MEMBERS) and recorded['dataset_hash'] == manifest['dataset_hash']
    assert manifest['archive_sha256'] == recorded['archive_sha256'] and manifest['audit'] == 'recorded'
    # second load against the recorded audit: identical bytes pass and the audit is not rewritten
    text = audit.read_text()
    _, _, again = gene.load_tcga(cache, audit_path=audit, identity=TINY_IDENTITY)
    assert again['audit'] == 'verified' and audit.read_text() == text
    # a changed archive (one value edited) is refused before any parsing
    fake_archive(cache, DATA_TEXT.replace('2.500', '2.501'), LABELS_TEXT)
    with pytest.raises(ValueError, match='audit'):
        gene.load_tcga(cache, audit_path=audit, identity=TINY_IDENTITY)
    # a missing member is refused, and identity constants are enforced on a fresh audit
    fresh = tmp_path / 'fresh'
    path = fake_archive(fresh, DATA_TEXT, LABELS_TEXT)
    with pytest.raises(ValueError, match='class'):
        gene.load_tcga(fresh, audit_path=tmp_path / 'fresh.json', identity={**TINY_IDENTITY, 'class_counts': [3, 1]})
    assert not (tmp_path / 'fresh.json').exists()          # no audit for a failed load
    with tarfile.open(path, 'w:gz') as tar:
        info = tarfile.TarInfo(gene.MEMBERS[0]); info.size = 1; tar.addfile(info, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='member'):
        gene.read_members(path)


def test_loader_never_downloads_a_cached_archive_and_reports_download_failure(tmp_path, monkeypatch):
    cache = tmp_path / 'cache'
    fake_archive(cache, DATA_TEXT, LABELS_TEXT)
    monkeypatch.setattr(gene, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(OSError('offline')))
    assert gene.fetch_archive(cache) == cache / gene.ARCHIVE_NAME
    with pytest.raises(FileNotFoundError, match=gene.ARCHIVE_NAME):
        gene.fetch_archive(tmp_path / 'empty')
    monkeypatch.setenv('ARROWFLOW_GENE_CACHE', str(cache))
    assert gene.cache_directory() == cache


@pytest.mark.skipif(not (gene.cache_directory() / gene.ARCHIVE_NAME).is_file(),
                    reason='Real UCI archive only when cached locally')
def test_real_tcga_identity_matches_the_committed_audit():
    X, y, manifest = gene.load_tcga()
    assert X.shape == (801, 20531) and np.bincount(y).tolist() == [300, 78, 146, 141, 136]
    assert manifest['label_map'] == ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD'] and X.min() >= 0
    assert manifest['audit'] == 'verified' and gene.AUDIT_PATH.is_file()


class Spy:
    calls = []


def test_top_genes_selector_fits_only_on_training_rows_and_orders_by_mi(monkeypatch, tmp_path):
    X, y = synthetic_genes()
    X[:, -1] = np.arange(len(X))                                  # row identity column (last gene)
    original = gene.mutual_info_classif
    def spy(X_fit, y_fit, **kwargs):
        Spy.calls.append(tuple(X_fit[:, -1].astype(int)))
        return original(X_fit, y_fit, **kwargs)
    monkeypatch.setattr(gene, 'mutual_info_classif', spy)
    monkeypatch.delenv('ARROWFLOW_GENE_MI_CACHE', raising=False)
    gene.clear_selection_memo()
    Spy.calls = []
    split = make_splits(y, outer_folds=3, repeats=1, inner_folds=2, seed=27183)[0]
    spec = ModelSpec('svc_raw', gene.svc_raw_factory,
                     [{'C': 1, 'gamma': 'scale', 'n_genes': n} for n in (3, 5, 8)], False)
    selection = select_model(X, y, split, spec, (8129, 19391, 39019))
    inner_partitions = sorted(tuple(i['train']) for i in split['inner'])
    assert sorted(Spy.calls) == inner_partitions                  # each inner partition once; never a test row
    assert selection['config']['n_genes'] in (3, 5, 8)
    # the memo is keyed by the exact rows and labels of the partition; cleared, every partition is recomputed
    gene.clear_selection_memo()
    Spy.calls = []
    changed = y.copy(); changed[split['test']] = (changed[split['test']] + 1) % 3
    select_model(X, changed, split, spec, (8129, 19391, 39019))
    assert sorted(Spy.calls) == inner_partitions
    # ordering: descending mutual information, ties by lowest gene index; width guarded; provenance recorded
    selector = gene.TopGenesByMI(n_genes=4).fit(X[split['train']], y[split['train']])
    mi = original(X[split['train']], y[split['train']], n_neighbors=gene.MI_NEIGHBORS, random_state=gene.MI_SEED)
    assert selector.selected_.tolist() == np.argsort(-mi, kind='stable')[:4].tolist()
    assert selector.transform(X[:2]).shape == (2, 4) and np.array_equal(selector.transform(X[:2]), X[:2][:, selector.selected_])
    meta = selector.representation_metadata_
    assert meta['selected_genes'] == selector.selected_.tolist() and meta['ranking_hash'] == selector.ranking_hash_
    assert meta['selection_source'] in ('computed', 'memo') and meta['shared_cache_file'] is None
    assert meta['cache_key_version'] == gene.CACHE_KEY_VERSION and meta['sklearn'] and meta['numpy']
    with pytest.raises(ValueError):
        gene.TopGenesByMI(n_genes=31).fit(X, y)
    with pytest.raises(ValueError):
        selector.transform(X[:, :10])
    # optional cross-process cache: a fresh process-local memo reads the stored ranking back exactly
    monkeypatch.setenv('ARROWFLOW_GENE_MI_CACHE', str(tmp_path / 'mi'))
    gene.clear_selection_memo()
    first = gene.TopGenesByMI(n_genes=4).fit(X[split['train']], y[split['train']])
    gene.clear_selection_memo()
    second = gene.TopGenesByMI(n_genes=4).fit(X[split['train']], y[split['train']])
    assert first.selection_source_ == 'computed' and second.selection_source_ == 'disk'
    assert first.selected_.tolist() == second.selected_.tolist() and first.ranking_hash_ == second.ranking_hash_
    assert Path(second.representation_metadata_['shared_cache_file']).is_file()
    gene.clear_selection_memo()


def small_rank_models():
    return {'af_native_ranks': (gene.af_native_ranks_factory, {'widths': [8], 'learning_rate': .1, 'iterations': 2, 'batch_size': 32,
                                                               'validation_ratio': .1, 'p_correct': .01, 'n_genes': 6}),
            'svc_ranked': (gene.svc_ranked_factory, {'C': 1, 'gamma': 'scale', 'n_genes': 6}),
            'rf_ranked': (gene.rf_ranked_factory, {'n_estimators': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'max_depth': None, 'n_genes': 6}),
            'footrule_knn_ranks': (gene.footrule_knn_ranks_factory, {'n_neighbors': 3, 'weights': 'uniform', 'n_genes': 6})}


@pytest.mark.parametrize('fixture', ['distinct', 'tie_rich'])
def test_within_sample_order_and_every_rank_input_model_are_exactly_invariant_under_monotone_transforms(fixture):
    X, y = synthetic_genes() if fixture == 'distinct' else tie_rich_genes()
    train, test = np.arange(45), np.arange(45, 60)
    encoder = gene.WithinSampleOrder().fit(X[train])
    orders = encoder.transform(X[test])
    assert np.array_equal(np.sort(orders, axis=1), np.tile(np.arange(30), (15, 1)))
    if fixture == 'tie_rich':
        assert (X[test] == 0).any(axis=1).all() and (X[test][:, 1] == X[test][:, 0]).all()
        positions = np.argsort(orders, axis=1)                        # position of each gene within its row
        assert (positions[:, 1] == positions[:, 0] + 1).all()          # tied genes keep ascending index order
        assert (positions[:, 20] < positions[:, 21]).all() and (positions[:, 21] < positions[:, 22]).all()
    assert set(gene.MONOTONE_TRANSFORMS) == {'log1p', 'sqrt_abs', 'signed_square', 'scale_0.01', 'scale_100'}
    for name, transform in gene.MONOTONE_TRANSFORMS.items():
        transformed = transform(X[test])
        assert np.array_equal(encoder.transform(transformed), orders), name
        assert np.array_equal(score_order(transformed), score_order(X[test])), name
    assert set(small_rank_models()) == set(gene.RANK_INPUT_MODELS)
    for model_id, (factory, config) in small_rank_models().items():
        estimator = factory(config, 8129).fit(X[train], y[train])
        clean = estimator.predict(X[test])
        assert clean.shape == (15,)
        for name, transform in gene.MONOTONE_TRANSFORMS.items():
            assert np.array_equal(estimator.predict(transform(X[test])), clean), (model_id, name)
    if fixture == 'distinct':
        # the raw-value families are not rank models: a scale change reaches the classifier
        raw = gene.svc_raw_factory({'C': 1, 'gamma': .1, 'n_genes': 6}, 8129).fit(X[train], y[train])
        assert not np.array_equal(raw.predict(X[test] * 100.), raw.predict(X[test]))


def test_registry_candidate_budgets_ids_and_protocol_agreement():
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol['frozen'] in (False, True) and protocol['wallclock_cap_hours'] == 1
    if protocol['frozen']:
        assert protocol['frozen_at_utc'] >= '2026-09-12'  # freeze-tolerant pin, as in test_matched_studies
    assert protocol['datasets'] == ['tcga_pancan_rnaseq'] and protocol['registry'] == 'experiments.make_revision.gene:registry'
    assert (protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds']) == (5, 3, 3)
    assert protocol['split_seed'] == 27183 and protocol['fit_seeds'] == [8129, 19391, 39019]
    assert protocol['candidate_budget'] == 24 and protocol['candidate_seed'] == 41071
    assert protocol['gene_selection']['n_genes'] == list(gene.N_GENES) == [10, 15, 20]
    assert protocol['corruption'] == gene.corruption_schedule()
    registry = gene.registry(protocol)
    assert set(registry) == {'af_full', 'af_native_ranks', 'svc_raw', 'svc_ranked', 'rf_raw', 'rf_ranked', 'footrule_knn_ranks', 'dummy'}
    assert all(name == spec.model_id for name, spec in registry.items())
    sizes = {name: len(spec.candidates) for name, spec in registry.items()}
    assert sizes == {'af_full': 3, 'af_native_ranks': 3, 'svc_raw': 24, 'svc_ranked': 24, 'rf_raw': 24, 'rf_ranked': 24,
                     'footrule_knn_ranks': 24, 'dummy': 1}
    assert {name: spec.stochastic for name, spec in registry.items()} == {
        'af_full': True, 'af_native_ranks': True, 'svc_raw': False, 'svc_ranked': False, 'rf_raw': True, 'rf_ranked': True,
        'footrule_knn_ranks': False, 'dummy': False}
    assert {name: gene.uses_selector(spec) for name, spec in registry.items()} == {name: name != 'dummy' for name in registry}
    for name, spec in registry.items():
        if name != 'dummy':
            assert all(c['n_genes'] in gene.N_GENES for c in spec.candidates), name
            assert len({config_id(c) for c in spec.candidates}) == len(spec.candidates)
    assert [c['n_genes'] for c in registry['af_full'].candidates] == [10, 15, 20]
    assert all(c['widths'] == [128] and c['learning_rate'] == .1 and c['embed_scale'] == 1 and c['degree_offset'] == 0
               and c['n_views'] == 7 and c['validation_ratio'] == .1 for c in registry['af_full'].candidates)
    assert all(c['widths'] == [128] and c['learning_rate'] == .1 and c['validation_ratio'] == .1
               for c in registry['af_native_ranks'].candidates)
    assert {c['n_genes'] for c in registry['svc_raw'].candidates} == {10, 15, 20}
    assert registry['svc_raw'].candidates == registry['svc_ranked'].candidates          # same sampled grid, different input
    assert registry['rf_raw'].candidates == registry['rf_ranked'].candidates
    # the registry seals the scientific sources it depends on, and no longer the SUSHI-only datasets module
    from experiments.make_revision.run_revision import environment_record, get_registry
    record = environment_record('experiments.make_revision.gene:registry')
    assert {'experiments/make_revision/gene.py', 'experiments/make_revision/bridge.py', 'experiments/make_revision/multiview.py',
            'experiments/make_revision/comparisons.py', 'experiments/make_revision/run_revision.py'} <= record['source_hashes'].keys()
    assert 'experiments/make_revision/datasets.py' not in record['source_hashes']
    assert set(get_registry('experiments.make_revision.gene:registry', protocol)) == set(registry)


def test_af_native_ranks_uses_the_selected_gene_count_as_its_vocabulary():
    X, y = synthetic_genes()
    config = {'widths': [8], 'learning_rate': .1, 'iterations': 2, 'batch_size': 32, 'validation_ratio': .1, 'p_correct': .01, 'n_genes': 7}
    model = gene.af_native_ranks_factory(config, 8129).fit(X[:45], y[:45])
    network = model.named_steps['arrowflow']
    assert network.embed_dim == 7 and not hasattr(network, 'encoder_') and model.predict(X[45:]).shape == (15,)
    assert model.representation_metadata_['input'] == 'within_sample_ranks' and len(model.representation_metadata_['selected_genes']) == 7
    full = gene.af_full_factory(gene.af_full_candidates()[0] | {'iterations': 2, 'n_views': 2}, 8129).fit(X[:45], y[:45])
    assert full.named_steps['arrowflow_full'].resolved_ == {'embed_dim': 16, 'degree': 3, 'augment': False}   # 10 genes, 45 rows
    assert full.predict(X[45:]).shape == (15,) and full.representation_metadata_['input'] == 'raw_values'


def test_corruption_bank_is_deterministic_shared_across_models_and_tamper_evident(tmp_path):
    X, _ = synthetic_genes()
    query = X[:12]
    a = gene.GeneCorruptionBank.create(query, dataset_id='fixture', outer_repeat=0, outer_fold=1)
    b = gene.GeneCorruptionBank.create(query, dataset_id='fixture', outer_repeat=0, outer_fold=1)
    other = gene.GeneCorruptionBank.create(query, dataset_id='fixture', outer_repeat=0, outer_fold=2)
    assert a.metadata() == b.metadata() and a.metadata() != other.metadata()
    assert [c.condition for c in a.cases] == ['clean', 'log1p', 'sqrt_abs', 'signed_square', 'scale_0.01', 'scale_100',
                                              'lognormal_0.3', 'lognormal_0.5', 'lognormal_0.7', 'lognormal_1']
    assert a.cases[0].raw_hash == gene.array_hash(query) and all(not c.raw.flags.writeable for c in a.cases)
    from experiments.make_revision.comparisons import derive_seed
    assert a.draw_seed == derive_seed(104729, 'fixture', 0, 1, 'gene_lognormal_scaling')
    for case in a.cases:
        if case.family == 'lognormal_gene_scaling':
            np.testing.assert_array_equal(case.raw, query * np.exp(case.severity * a.log_scale_draws))
            assert case.draw_seed == a.draw_seed
        elif case.family == 'monotone':
            np.testing.assert_array_equal(case.raw, gene.MONOTONE_TRANSFORMS[case.condition](query))
            assert case.severity is None and case.draw_seed is None
    with pytest.raises(ValueError, match='Negative'):
        gene.GeneCorruptionBank.create(-query, dataset_id='fixture', outer_repeat=0, outer_fold=1)
    destination = tmp_path / 'bank'
    a.save(destination)
    saved = json.loads((destination / 'manifest.json').read_text())
    assert saved == a.metadata()
    with np.load(destination / 'draws.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['log_scale_draws'], a.log_scale_draws)
    with pytest.raises(FileExistsError):
        a.save(destination)
    a.cases[1].raw.setflags(write=True); a.cases[1].raw[0, 0] += 1.
    with pytest.raises(ValueError, match='changed'):
        a.assert_intact()


def synthetic_manifest(X, y):
    names, labels = [f'gene_{j}' for j in range(X.shape[1])], ['A', 'B', 'C']
    return {'dataset_id': gene.GENE_ID, 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(), 'feature_names': names,
            'label_map': labels, 'sample_order': 'fixture', 'dataset_hash': dataset_fingerprint(X, y, names, labels)}


def tiny_run(tmp_path, monkeypatch):
    """A complete, verified tiny gene run in the harness's output shape (in-process workers, no shared cache)."""
    from experiments.make_revision import run_revision as runner
    X, y = synthetic_genes()
    monkeypatch.setattr(runner, 'load_dataset', lambda name: (X, y, synthetic_manifest(X, y)))
    monkeypatch.delenv('ARROWFLOW_GENE_MI_CACHE', raising=False)
    gene.clear_selection_memo()
    protocol = {**json.loads(PROTOCOL.read_text()), 'outer_folds': 2, 'outer_repeats': 1, 'inner_folds': 2, 'frozen': True}
    registry_path = 'experiments.make_revision.gene:tiny_registry'
    registry = runner.prepare(tmp_path, [gene.GENE_ID], protocol, registry_path)
    for index in range(2):
        for model in registry:
            runner._worker((str(tmp_path), gene.GENE_ID, index, model, registry_path))
    runner.write_json(tmp_path / 'planned_jobs.json', runner.planned_jobs([gene.GENE_ID], protocol, registry))
    return protocol, registry, registry_path, X, y


def test_gene_corruption_refits_frozen_models_from_the_recorded_selection_and_records_exact_invariance(tmp_path, monkeypatch):
    from experiments.make_revision import run_gene
    protocol, registry, registry_path, X, y = tiny_run(tmp_path, monkeypatch)
    # every harness fit row carries the selector's provenance; the first fit of the first job computed it
    first = json.loads((tmp_path / 'results' / f'{gene.GENE_ID}__af_full__r0f0.json').read_text())
    assert first['selection']['fits'][0]['representation_metadata']['selection_source'] == 'computed'
    outer = json.loads((tmp_path / 'results' / f'{gene.GENE_ID}__svc_ranked__r0f0.json').read_text())
    meta = outer['models'][0]['representation_metadata']
    assert meta['selection_source'] in ('computed', 'memo') and len(meta['selected_genes']) == 5
    assert meta['input'] == 'within_sample_ranks' and meta['shared_cache_file'] is None and meta['ranking_hash']
    assert meta['cache_key_version'] == gene.CACHE_KEY_VERSION and meta['sklearn'] and meta['numpy']
    # selector prepass: one ranking per training partition (2 folds x (2 inner + outer)) into the shared cache
    prepass = run_gene.selector_prepass(tmp_path, workers=1)
    assert prepass['partition_count'] == 6 and prepass['cache_directory'] == str(tmp_path / 'mi_cache')
    assert {p['partition'] for p in prepass['partitions']} == {'outer', 'inner0', 'inner1'}
    assert all(Path(p['cache_file']).is_file() for p in prepass['partitions'])
    assert all(p['row_count'] in (15, 30) for p in prepass['partitions'])
    assert run_gene.selector_prepass(tmp_path) == prepass                          # idempotent once the manifest exists
    splits = json.loads((tmp_path / gene.GENE_ID / 'splits.json').read_text())
    entry = next(p for p in prepass['partitions'] if (p['outer_repeat'], p['outer_fold'], p['partition']) == (0, 0, 'outer'))
    monkeypatch.setenv('ARROWFLOW_GENE_MI_CACHE', prepass['cache_directory'])
    gene.clear_selection_memo()
    selector = gene.TopGenesByMI(5).fit(X[splits[0]['train']], y[splits[0]['train']])
    assert selector.selection_source_ == 'disk' and selector.partition_key_ == entry['cache_key']
    assert selector.ranking_hash_ == entry['ranking_hash']
    # corruption: refits hit the prepass cache, match the recorded run, and score every condition
    gene.clear_selection_memo()
    report = run_gene.gene_corruption(tmp_path, workers=1)
    root = tmp_path / 'corruption'
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['jobs']) == 2 * len(registry) and manifest['registry'] == registry_path
    assert manifest['schedule'] == gene.corruption_schedule() and manifest['run_code_revision'] == manifest['code_revision']
    assert manifest['selector_cache'] == prepass['cache_directory']
    assert sorted(p.name for p in (root / 'banks').iterdir()) == [f'{gene.GENE_ID}__r0f0', f'{gene.GENE_ID}__r0f1']
    stem = f'{gene.GENE_ID}__svc_ranked__r0f0'
    result = json.loads((root / 'results' / f'{stem}.json').read_text())
    assert result['status'] == 'ok' and result['refit_matches_recorded'] is True
    # in-process workers share the memo: the fold's first refit reads the prepass cache, later ones the memo, none computes
    first_refits = json.loads((root / 'results' / f'{gene.GENE_ID}__af_full__r0f0.json').read_text())['refits']
    assert [r['representation_metadata']['selection_source'] for r in first_refits] == ['disk', 'memo', 'memo']
    assert [r['representation_metadata']['selection_source'] for r in result['refits']] == ['memo']
    for job in manifest['jobs']:
        refits = json.loads((tmp_path / job['corruption_result_file']).read_text())['refits']
        assert all((r['representation_metadata'] or {}).get('selection_source') != 'computed' for r in refits), job['stem']
        assert all(r['representation_metadata']['shared_cache_file'] for r in refits if r['representation_metadata']), job['stem']
    conditions = [r['condition'] for r in result['rows']]
    assert conditions == [c['condition'] for c in manifest['schedule']['conditions']]
    lines = [json.loads(line) for line in (root / 'predictions' / f'{stem}.jsonl').read_text().splitlines()]
    assert len(lines) == len(conditions) * len(splits[0]['test'])
    assert {'dataset_id', 'dataset_hash', 'outer_repeat', 'outer_fold', 'model_id', 'model_seed', 'config_id', 'code_revision',
            'condition', 'corruption_family', 'severity', 'perturbation_seed', 'sample_id', 'y_true', 'y_pred'} <= lines[0].keys()
    assert all(line['y_true'] == y[line['sample_id']] for line in lines)
    summary = json.loads((root / 'summary.json').read_text())
    table = summary['summaries'][gene.GENE_ID]
    assert summary['invariance_violations'] == [] and report['invariance_violations'] == []
    monotone = [c['condition'] for c in manifest['schedule']['conditions'] if c['family'] == 'monotone']
    for model_id in gene.RANK_INPUT_MODELS:
        for condition in monotone:
            row = table[model_id][condition]
            assert row['expected_exact_invariance'] is True and row['exact_invariance_observed'] is True
            assert row['agreement_with_clean']['mean'] == 1. and row['metrics']['accuracy']['n_folds'] == 2
    assert table['af_full']['log1p']['expected_exact_invariance'] is False
    assert table['svc_ranked']['clean']['metrics']['accuracy']['mean'] == pytest.approx(
        np.mean([r['accuracy'] for r in summary['model_rows'][gene.GENE_ID] if r['model_id'] == 'svc_ranked' and r['condition'] == 'clean']))
    assert table['rf_ranked']['lognormal_1']['change_from_clean']['metric'] == 'error'
    # every job is required: a tampered prediction is caught, a removed prediction file blocks the summary
    path = root / 'predictions' / f'{stem}.jsonl'
    original = path.read_text()
    tampered = json.loads(original.splitlines()[0]); tampered['y_pred'] = (tampered['y_pred'] + 1) % 3
    path.write_text('\n'.join([json.dumps(tampered, sort_keys=True, separators=(',', ':'))] + original.splitlines()[1:]) + '\n')
    with pytest.raises(ValueError, match='Incomplete'):
        run_gene.summarize_corruption(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        run_gene.summarize_corruption(tmp_path)
    path.write_text(original)
    run_gene.summarize_corruption(tmp_path)
    # a second corruption pass refuses to overwrite existing per-fold outputs
    with pytest.raises(FileExistsError):
        run_gene.gene_corruption(tmp_path, workers=1)
    # an incomplete run is refused before any corruption output is touched
    (tmp_path / 'results' / f'{gene.GENE_ID}__dummy__r0f1.json').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        run_gene.gene_corruption(tmp_path, workers=1)


def fabricated_pilot(directory, cleared):
    """A pilot.json in run_revision's shape for the production registry: every selector family's first fit
    computed the selector when the memo was cleared per family; only the first family did when it was shared."""
    from experiments.make_revision import run_revision as runner
    protocol = json.loads(PROTOCOL.read_text())
    registry = gene.registry(protocol)
    runner.write_json(directory / 'protocol.json', protocol)
    runner.write_json(directory / 'environment.json', {'registry': 'experiments.make_revision.gene:registry', 'code_revision': 'fixture'})
    rows, seen_selector = [], False
    for model, spec in registry.items():
        selects = gene.uses_selector(spec)
        for i in range(3):
            computed = selects and i == 0 and (cleared or not seen_selector)
            seconds = 60. if computed else (.05 if selects else 0.)
            rows.append({'dataset_id': gene.GENE_ID, 'model_id': model, 'status': 'ok', 'config_id': f'c{i}',
                         'elapsed_seconds': seconds + 2., 'encoding_seconds': seconds if selects else None,
                         'representation_metadata': ({'selection_source': 'computed' if computed else 'memo',
                                                      'selection_seconds': seconds} if selects else None)})
        seen_selector = seen_selector or selects
    estimates = {m: {'fits_per_outer': 3 * len(s.candidates) + (min(3, len(s.candidates)) * 6 + 3 if s.stochastic else 1)}
                 for m, s in registry.items()}
    runner.write_json(directory / 'pilot.json', {'rows': rows, 'workload_estimates': estimates})
    return registry, estimates


def test_pilot_projection_charges_the_selector_per_job_and_partition_and_detects_a_shared_memo_pilot(tmp_path):
    from experiments.make_revision import run_gene
    registry, estimates = fabricated_pilot(tmp_path / 'cleared', cleared=True)
    fabricated_pilot(tmp_path / 'shared', cleared=False)
    cleared = run_gene.pilot_projection(tmp_path / 'cleared')
    shared = run_gene.pilot_projection(tmp_path / 'shared')
    assert cleared['pilot_memo_cleared_between_families'] is True and shared['pilot_memo_cleared_between_families'] is False
    assert len(cleared['families_with_measured_selector']) == 7 and shared['families_with_measured_selector'] == ['af_full']
    expected_memo = sum(15 * (4 * 60. * gene.uses_selector(s) + estimates[m]['fits_per_outer'] * 2.) for m, s in registry.items())
    expected_after = 60 * 60. + sum(15 * estimates[m]['fits_per_outer'] * (2. + .05 * gene.uses_selector(s)) for m, s in registry.items())
    for projection in (cleared, shared):          # the charge does not depend on which family's rows showed the cost
        assert projection['selector_seconds'] == 60. and projection['prepass_partitions'] == 60
        assert projection['selector_partitions_per_job'] == 4 and projection['cache_hit_seconds'] == .05
        assert projection['models']['svc_raw']['serial_seconds_process_memo'] == pytest.approx(15 * (4 * 60. + 73 * 2.))
        assert projection['models']['dummy']['selector_partitions_per_job'] == 0
        assert projection['models']['dummy']['serial_seconds_process_memo'] == pytest.approx(15 * 4 * 2.)
        assert projection['serial_hours_process_memo'] == pytest.approx(expected_memo / 3600)
        assert projection['serial_hours_process_memo'] > 7                                      # the 7 selector families alone
        assert projection['serial_hours_after_prepass'] == pytest.approx(expected_after / 3600)
        assert projection['hours_at_16_workers_after_prepass'] == pytest.approx(projection['serial_hours_after_prepass'] / 16)
        assert projection['within_cap_after_prepass'] == (projection['hours_at_16_workers_after_prepass'] <= 1)
        assert projection['wallclock_cap_hours'] == 1 and projection['prepass'] is None
    # a pilot in which nothing computed the selector (cache hits only) and no prepass manifest is refused
    fabricated_pilot(tmp_path / 'hits', cleared=True)
    pilot = json.loads((tmp_path / 'hits' / 'pilot.json').read_text())
    for row in pilot['rows']:
        if row['representation_metadata']:
            row['representation_metadata']['selection_source'] = 'disk'
    (tmp_path / 'hits' / 'pilot.json').write_text(json.dumps(pilot))
    with pytest.raises(ValueError, match='selector'):
        run_gene.pilot_projection(tmp_path / 'hits')


def test_harness_pilot_resets_the_selector_memo_before_every_family(tmp_path, monkeypatch):
    from experiments.make_revision import run_gene, run_revision as runner
    X, y = synthetic_genes()
    monkeypatch.setattr(runner, 'load_dataset', lambda name: (X, y, synthetic_manifest(X, y)))
    monkeypatch.delenv('ARROWFLOW_GENE_MI_CACHE', raising=False)
    gene.clear_selection_memo()
    protocol = {**json.loads(PROTOCOL.read_text()), 'outer_folds': 2, 'outer_repeats': 1, 'inner_folds': 2}
    runner.runtime_pilot(tmp_path, [gene.GENE_ID], protocol, 'experiments.make_revision.gene:tiny_registry')
    rows = defaultdict(list)
    for row in json.loads((tmp_path / 'pilot.json').read_text())['rows']:
        rows[row['model_id']].append(row)
    assert set(rows) == {'af_full', 'af_native_ranks', 'svc_raw', 'svc_ranked', 'rf_raw', 'rf_ranked', 'footrule_knn_ranks', 'dummy'}
    for model, model_rows in rows.items():
        sources = [(r['representation_metadata'] or {}).get('selection_source') for r in model_rows]
        assert sources == ([None] * len(model_rows) if model == 'dummy' else ['computed'] + ['memo'] * (len(model_rows) - 1)), model
    projection = run_gene.pilot_projection(tmp_path)
    assert projection['pilot_memo_cleared_between_families'] is True
    assert set(projection['families_with_measured_selector']) == set(rows) - {'dummy'}
    assert projection['selector_seconds'] > 0 and projection['models']['dummy']['selector_partitions_per_job'] == 0


def test_run_revision_wires_the_gene_dataset_like_sushi(tmp_path, monkeypatch):
    from experiments.make_revision import datasets, run_revision as runner
    assert runner.GENE_ID == gene.GENE_ID == 'tcga_pancan_rnaseq' and not hasattr(datasets, 'GENE_ID')
    sentinel = (np.zeros((2, 2)), np.array([0, 1]), {'dataset_hash': 'fixture'})
    monkeypatch.setattr(gene, 'load_tcga', lambda: sentinel)
    assert runner.load_dataset(gene.GENE_ID) == sentinel
    with pytest.raises(SystemExit):
        runner.main(['prepare', '--dataset', 'not_a_dataset', '--output', str(tmp_path)])
    X, y = synthetic_genes()
    monkeypatch.setattr(runner, 'load_dataset', lambda name: (X, y, {'dataset_hash': 'synthetic_fixture'}))
    runner.main(['smoke', '--dataset', gene.GENE_ID, '--output', str(tmp_path), '--gene-cache', str(tmp_path / 'cache')])
    assert os.environ['ARROWFLOW_GENE_CACHE'] == str(tmp_path / 'cache')
    result = json.loads((tmp_path / 'smoke.json').read_text())['result']
    assert all(r['model_id'] == 'smoke_gene_rank_arrowflow' and r['status'] == 'ok' for r in result['models'])
    assert all(r['config']['n_genes'] == 10 for r in result['models'])
