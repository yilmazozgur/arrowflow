"""The nearest-baseline comparators of the deployed ArrowFlow-kNN readout: the Kendall kernel against a brute-force
concordance count, the four fold-local models, the seventeen-dataset panel and the stage protocol."""
import json
from itertools import permutations, product
from math import comb
from pathlib import Path
import warnings
import numpy as np
import pytest
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from arrowflow.ranking import inverse_positions
from experiments.make_revision import neighbour_baselines as nb
from experiments.make_revision import newdata as nd
from experiments.make_revision import run_revision as rr
from experiments.make_revision.comparisons import CONVENTIONAL_GRIDS, TimedPipeline
from experiments.make_revision.evaluation import (ModelSpec, _fit_predict, canonical_json, config_id, evaluate_fold,
                                                  make_splits)
from experiments.make_revision.knn_controls import INPUT_MODEL, TRAINED_MODEL, control_candidates
from experiments.make_revision.models import NumericImputer, OrdinalEncoder
from experiments.make_revision.multiview import view_strategy

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO/'experiments'/'make_revision'/'protocols'
RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'


def small(n=90, seed=0):
    X, y = load_iris(return_X_y=True)
    idx = np.random.RandomState(seed).permutation(len(y))[:n]
    return X[idx], y[idx]


def random_orders(n, vocabulary, seed=0):
    rng = np.random.RandomState(seed)
    return np.array([rng.permutation(vocabulary) for _ in range(n)])


def brute_force_kendall(order_a, order_b):
    """(concordant - discordant) / (V choose 2) counted one item pair at a time, with explicit Python loops."""
    positions_a = {int(item): rank for rank, item in enumerate(order_a)}
    positions_b = {int(item): rank for rank, item in enumerate(order_b)}
    vocabulary = sorted(positions_a)
    concordant = discordant = 0
    for index, first in enumerate(vocabulary):
        for second in vocabulary[index + 1:]:
            same = (positions_a[first] < positions_a[second]) == (positions_b[first] < positions_b[second])
            concordant, discordant = concordant + same, discordant + (not same)
    assert concordant + discordant == comb(len(vocabulary), 2)
    return (concordant - discordant) / comb(len(vocabulary), 2)


# ----------------------------------------------------------------------------- the Kendall kernel

@pytest.mark.parametrize('vocabulary', [3, 4, 5])
def test_kendall_kernel_equals_a_brute_force_pairwise_concordance_count_on_every_permutation(vocabulary):
    orders = np.array(list(permutations(range(vocabulary))))
    kernel = nb.kendall_kernel(orders)
    assert kernel.shape == (len(orders), len(orders))
    for i, a in enumerate(orders):
        for j, b in enumerate(orders):
            assert kernel[i, j] == pytest.approx(brute_force_kendall(a, b), abs=1e-12)
    assert np.allclose(np.diag(kernel), 1) and np.allclose(kernel, kernel.T, atol=0)
    assert kernel.max() <= 1 + 1e-12 and kernel.min() >= -1 - 1e-12


def test_kendall_kernel_is_kendalls_tau_and_positive_semi_definite_and_reads_pairs_of_items():
    orders = random_orders(40, 7, seed=5)
    kernel = nb.kendall_kernel(orders)
    positions = inverse_positions(orders)
    for i in (0, 3, 17):
        for j in (1, 9, 39):
            assert kernel[i, j] == pytest.approx(stats.kendalltau(positions[i], positions[j]).statistic, abs=1e-12)
    assert np.linalg.eigvalsh((kernel + kernel.T)/2).min() >= -1e-9
    # 1 - 2 * Kendall tau distance / (V choose 2), the distance counted on the position vectors
    distance = sum((positions[0][a] - positions[0][b]) * (positions[1][a] - positions[1][b]) < 0
                   for a, b in permutations(range(7), 2)) / 2
    assert kernel[0, 1] == pytest.approx(1 - 2*distance/comb(7, 2), abs=1e-12)


def test_kendall_features_are_plus_or_minus_one_and_their_inner_product_is_the_kernel_exactly():
    orders = random_orders(25, 9, seed=2)
    features = nb.kendall_features(orders)
    assert features.dtype == np.float32 and features.shape == (25, comb(9, 2))
    assert set(np.unique(features).tolist()) == {-1.0, 1.0}
    product_ = np.asarray(features @ features.T)
    assert np.array_equal(product_, np.rint(product_))                      # the float32 matrix product is exact
    assert np.array_equal(nb.kendall_kernel(orders), product_.astype(np.float64)/comb(9, 2))


def test_kendall_kernel_of_two_blocks_is_the_corresponding_block_of_the_joint_kernel():
    orders = random_orders(30, 6, seed=7)
    joint = nb.kendall_kernel(orders)
    assert np.array_equal(nb.kendall_kernel(orders[20:], orders[:20]), joint[20:, :20])


def test_kendall_features_refuse_anything_that_is_not_a_complete_permutation_matrix():
    with pytest.raises(ValueError, match='complete permutations'):
        nb.kendall_features(np.array([[0, 1, 1]]))
    with pytest.raises(ValueError, match='at least two items'):
        nb.kendall_features(np.array([[0], [0]]))
    with pytest.raises(ValueError, match='one vocabulary'):
        nb.kendall_kernel(random_orders(3, 4, seed=1), random_orders(3, 5, seed=1))


def test_the_kernel_cap_refuses_a_large_training_partition_instead_of_subsampling_it():
    cap = {'max_training_rows': 50, 'max_feature_entries': 10**9}
    assert nb.within_cap(nb.kernel_cost(50, 8), cap) and not nb.within_cap(nb.kernel_cost(51, 8), cap)
    assert not nb.within_cap(nb.kernel_cost(10, 64), {'max_training_rows': 50, 'max_feature_entries': 100})
    with pytest.raises(nb.KernelTooLarge, match='refused, never subsampled'):
        nb.check_kernel_cap(51, 8, cap)
    X, y = small(60)
    with pytest.raises(nb.KernelTooLarge):
        nb.MultiViewKendallSVC(embed_dim=16, degree=1, kernel_cap=cap).fit(X, y)


def test_every_panel_dataset_is_within_the_declared_kernel_cap_so_none_is_refused():
    panel = nb.panel_declaration()
    capacity = nb.kendall_capacity(panel)
    assert sorted(capacity) == sorted(entry['name'] for entry in panel)
    for entry in panel:
        row = capacity[entry['name']]
        samples, features = entry['shape']
        assert row['training_rows'] == samples - samples//5 and row['pairs'] == comb(row['vocabulary'], 2)
        assert row['vocabulary'] == max(nb.resolve(c, features, row['training_rows'])['embed_dim']
                                        for c in control_candidates(INPUT_MODEL))
        assert row['within_cap']
    assert nb.kendall_kernel_declaration(panel)['refused'] == []


# ----------------------------------------------------------------------------- the fold-local projections

def test_resolve_components_rounds_and_clips_into_the_bound_the_training_partition_allows():
    assert [nb.resolve_components(.5, m) for m in (1, 2, 3, 9, 64)] == [1, 1, 2, 4, 32]
    assert [nb.resolve_components(1., m) for m in (1, 2, 9, 64)] == [1, 2, 9, 64]
    for scale in (0, -1, 1.5, True, 'half'):
        with pytest.raises(ValueError, match='fraction'):
            nb.resolve_components(scale, 8)
    with pytest.raises(ValueError, match='at least one component'):
        nb.resolve_components(.5, 0)


def test_the_projections_resolve_their_declared_bound_and_publish_it():
    X, y = small(90)
    X = StandardScaler().fit_transform(X)
    lda = nb.ScaledLDA(component_scale=1.).fit(X, y)
    assert lda.max_components_ == len(np.unique(y)) - 1 == 2 and lda.n_components_ == 2
    assert lda.record() == {'projection': 'linear_discriminant_analysis', 'component_scale': 1., 'max_components': 2,
                            'n_components': 2}
    assert np.allclose(lda.transform(X), LinearDiscriminantAnalysis(solver='svd', n_components=2).fit(X, y).transform(X))
    half = nb.ScaledLDA(component_scale=.5).fit(X, y)
    assert half.n_components_ == 1 and half.transform(X).shape == (90, 1)
    pca = nb.ScaledPCA(component_scale=.5).fit(X, y)
    assert pca.max_components_ == X.shape[1] == 4 and pca.n_components_ == 2
    assert np.allclose(pca.transform(X), PCA(n_components=2, svd_solver='full').fit(X).transform(X))
    binary = np.where(y > 0, 1, 0)
    assert nb.ScaledLDA(component_scale=1.).fit(X, binary).n_components_ == 1        # bounded by classes minus one
    with pytest.raises(ValueError, match='at least two training classes'):
        nb.ScaledLDA().fit(X, np.zeros(len(y)))


def test_nca_is_bounded_by_the_features_records_its_iterations_and_reports_reaching_max_iter():
    X, y = small(60)
    X = StandardScaler().fit_transform(X)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        nca = nb.ScaledNCA(component_scale=.5, max_iter=1, random_state=8129).fit(X, y)
    record = nca.record()
    assert nca.max_components_ == 4 and record['n_components'] == 2 and record['max_iter'] == 1
    assert record['reached_max_iter'] and record['init'] == 'auto' and record['random_state'] == 8129
    assert any(issubclass(w.category, ConvergenceWarning) for w in captured)
    assert nb.ScaledNCA(component_scale=1., max_iter=nb.NCA_MAX_ITER).fit(X, y).n_components_ == 4
    with pytest.raises(ValueError, match='positive bounded max_iter'):
        nb.ScaledNCA(max_iter=0).fit(X, y)


def test_every_pipeline_fits_its_imputation_scaling_and_projection_on_the_training_rows_only():
    X, y = small(90)
    X = X.copy()
    X[3, 1] = np.nan
    train, test = np.arange(60), np.arange(60, 90)
    for family in (nb.LDA_MODEL, nb.PCA_MODEL, nb.NCA_MODEL):
        config = {'component_scale': 1., 'n_neighbors': 3, 'weights': 'uniform'}
        pipeline = nb.projection_factory(family, config, 8129).fit(X[train], y[train])
        imputer, scaler = pipeline.named_steps['imputer'], pipeline.named_steps['scaler']
        assert np.allclose(imputer.means_, np.nanmean(X[train], axis=0))
        assert np.allclose(scaler.mean_, imputer.transform(X[train]).mean(axis=0))
        assert not np.allclose(imputer.means_, np.nanmean(X, axis=0))         # the held-out rows never entered the fit
        assert pipeline.named_steps['projection'].max_components_ in (2, 4)
        assert len(pipeline.predict(X[test])) == 30


def test_the_pipeline_records_the_resolved_projection_the_timing_and_every_fit_warning():
    X, y = small(60)
    spec = ModelSpec(nb.NCA_MODEL, lambda config, seed: nb.projection_factory(nb.NCA_MODEL, config, seed),
                     [{'component_scale': 1., 'n_neighbors': 3, 'weights': 'uniform'}], True)
    predictions, timing = _fit_predict(spec, spec.candidates[0], 8129, X, y, X)
    assert len(predictions) == len(y)
    assert timing['fit_seconds'] > 0 and timing['predict_seconds'] > 0 and timing['encoding_seconds'] is not None
    assert timing['classifier_fit_seconds'] is not None and isinstance(timing['fit_warnings'], list)
    assert timing['representation_metadata']['projection'] == 'neighborhood_components_analysis'
    canonical_json(timing['representation_metadata'])                        # the saved record must be plain JSON
    assert isinstance(nb.projection_factory(nb.LDA_MODEL, {'component_scale': 1., 'n_neighbors': 1, 'weights': 'uniform',
                                                           'p': 2}, 1), TimedPipeline)


# ----------------------------------------------------------------------------- the Kendall kernel SVC

def test_the_kendall_svc_reads_the_encoded_rankings_of_the_input_knn_control_view_for_view():
    from experiments.make_revision.knn_controls import MultiViewInputKNN
    X, y = small(90)
    settings = dict(n_views=7, strategy='diverse', embed_dim=16, degree=2, lda_ratio=.3, seed=8129)
    ours = nb.MultiViewKendallSVC(**settings, C=1.).fit(X, y)
    theirs = MultiViewInputKNN(**settings)
    theirs.fit(X, y)
    assert len(ours.views_) == len(theirs.views_) == 7
    for view, ((encoder, orders, _), (other, _)) in enumerate(zip(ours.views_, theirs.views_)):
        assert encoder.strategy == other.strategy == view_strategy('diverse', view)
        assert np.array_equal(orders, other.transform(X)) and np.array_equal(encoder.transform(X), other.transform(X))
    assert ours.view_records_[0]['vocabulary'] == 16 and ours.view_records_[0]['pairs'] == comb(16, 2)


def test_the_kendall_svc_votes_by_majority_over_the_per_view_precomputed_kernel_classifiers():
    from sklearn.svm import SVC
    from experiments.make_revision.secondary_studies import majority
    X, y = small(90)
    model = nb.MultiViewKendallSVC(n_views=3, embed_dim=8, degree=1, C=10., seed=8129).fit(X, y)
    votes = []
    for encoder, orders, _ in model.views_:
        gram = nb.kendall_kernel(orders)
        classifier = SVC(kernel='precomputed', C=10., max_iter=nb.SVC_MAX_ITER).fit(gram, y)
        votes.append(classifier.predict(nb.kendall_kernel(np.asarray(encoder.transform(X), dtype=np.int32), orders)))
    assert np.array_equal(model.predict(X), majority(votes))
    assert np.array_equal(np.asarray(model.predict_views(X)), np.asarray(votes))
    with pytest.raises(ValueError, match='majority only'):
        nb.MultiViewKendallSVC(aggregation='borda').fit(X, y)
    with pytest.raises(ValueError, match='penalty C'):
        nb.MultiViewKendallSVC(C=0).fit(X, y)


def test_the_adaptive_kendall_svc_resolves_the_encoder_from_the_training_partition_and_records_its_readout():
    X, y = small(90)
    config = dict(control_candidates(INPUT_MODEL)[0], C=1.)
    model = nb.AdaptiveKendallSVC(config=config, seed=8129).fit(X, y)
    assert model.resolved_ == {key: nb.resolve(config, X.shape[1], len(y))[key] for key in ('embed_dim', 'degree')}
    record = model.representation_metadata_
    assert record['readout'] == 'kendall_kernel_svc' and record['resolved'] == model.resolved_
    assert len(record['views']) == 7 and record['kernel_cost_per_view']['training_rows'] == 90
    assert record['kernel'].startswith('K(s, s') and 'Jiao and Vert' in record['kernel']
    canonical_json(record)
    assert model.encoder_ is model.model_.views_[0][0] and model.training_seconds_ == 0.
    with pytest.raises(ValueError, match='configurations hold exactly'):
        nb.AdaptiveKendallSVC(config={'C': 1.}).fit(X, y)


def test_the_kendall_candidates_are_the_input_control_candidates_crossed_with_the_svc_c_grid():
    candidates = nb.kendall_candidates()
    ids = [config_id(c) for c in candidates]
    assert ids == sorted(ids) and len(set(ids)) == len(ids) == 16
    assert {(c['embed_scale'], c['degree_offset']) for c in candidates} == set(product((1, 2), (0, -1)))
    assert sorted({c['C'] for c in candidates}) == sorted(CONVENTIONAL_GRIDS['svc_rbf']['C'])
    assert all(set(c) == set(nb.KENDALL_KEYS) for c in candidates)
    assert all(c['n_views'] == 7 and c['strategy'] == 'diverse' and c['aggregation'] == 'majority' for c in candidates)
    base = {config_id({k: c[k] for k in nb.CONTROL_CANDIDATE_KEYS[INPUT_MODEL]}) for c in candidates}
    assert base == {config_id(c) for c in control_candidates(INPUT_MODEL)}


# ----------------------------------------------------------------------------- the registry and one nested fold

def test_the_registry_holds_the_four_baselines_with_the_registered_grids_inside_the_candidate_budget():
    protocol = nb.draft_protocol()
    registry = nb.build_registry(protocol)
    assert tuple(registry) == nb.MODEL_ORDER == ('lda_knn', 'pca_knn', 'nca_knn', 'kendall_svc')
    assert {name: spec.stochastic for name, spec in registry.items()} == nb.STOCHASTIC
    assert [len(spec.candidates) for spec in registry.values()] == [24, 24, 20, 16]
    assert all(len(spec.candidates) <= protocol['candidate_budget'] for spec in registry.values())
    neighbours = CONVENTIONAL_GRIDS['numeric_knn']
    for model in (nb.LDA_MODEL, nb.PCA_MODEL):
        assert nb.BASELINE_GRIDS[model] == {'component_scale': [.5, 1.], **neighbours}
    assert nb.BASELINE_GRIDS[nb.NCA_MODEL] == {'component_scale': [.5, 1.], 'n_neighbors': neighbours['n_neighbors'],
                                               'weights': neighbours['weights']}
    assert registry[nb.LDA_MODEL].candidates == registry[nb.PCA_MODEL].candidates     # the supervised/label-free pair
    assert all(set(c) == {'component_scale', 'n_neighbors', 'weights'} for c in registry[nb.NCA_MODEL].candidates)
    assert nb.baselines_registry(protocol).keys() == registry.keys()
    with pytest.raises(ValueError, match='refuses'):
        nb.build_registry(dict(protocol, kendall_kernel={**protocol['kendall_kernel'], 'refused': ['digits']}))


def test_every_baseline_completes_one_nested_outer_fold_and_saves_its_timing():
    X, y = small(90)
    protocol = nb.draft_protocol()
    split = make_splits(y, 3, 1, 2, protocol['split_seed'])[0]
    registry = nb.build_registry(protocol)
    for model, spec in registry.items():
        tiny = ModelSpec(model, spec.factory, spec.candidates[:2], spec.stochastic)
        result = evaluate_fold(X, y, split, tiny, protocol['fit_seeds'], dataset_id='iris_subset',
                               dataset_hash='x'*64, code_revision='0'*40, score='accuracy')
        assert result['status'] == 'ok' and len(result['models']) == (3 if spec.stochastic else 1)
        for row in result['models']:
            assert row['fit_seconds'] > 0 and row['predict_seconds'] > 0 and 0 <= row['accuracy'] <= 1
            assert isinstance(row['fit_warnings'], list)
            canonical_json(row)


# ----------------------------------------------------------------------------- the panel and the protocol

def test_the_panel_pins_the_seventeen_datasets_of_the_registered_runs_with_their_dataset_and_splits_hashes():
    panel = nb.panel_declaration()
    assert [entry['name'] for entry in panel] == list(nb.BENCHMARK) + list(nb.FURTHER)
    assert len(panel) == 17 and len(nb.BENCHMARK) == 7 and len(nb.FURTHER) == 10
    assert set(nb.BENCHMARK) == set(rr.DATASETS) and tuple(nb.FURTHER) == tuple(nd.PANEL)
    for entry in panel:
        assert entry['reference'] in nb.REFERENCE_LABELS and entry['panel'] in ('benchmark', 'further')
        assert len(entry['dataset_hash']) == 64 and len(entry['splits_hash']) == 16
    further = {entry['name']: entry for entry in panel if entry['panel'] == 'further'}
    for name, entry in further.items():
        pin = nd.PIN_BY_NAME[name]
        assert (entry['dataset_hash'], entry['splits_hash']) == (pin['dataset_hash'], pin['splits_hash'])
        assert entry['shape'] == list(pin['shape']) and entry['n_classes'] == len(pin['label_map'])


@pytest.mark.skipif(not (RUNS/'2026-09-12-bridge-knn').is_dir(), reason='the registered runs are not in this workspace')
def test_the_panel_and_reference_pins_are_the_registered_runs_manifests_and_records():
    import hashlib
    panel = {entry['name']: entry for entry in nb.panel_declaration()}
    for label, entry in nb.REFERENCE_RUNS.items():
        path = RUNS/entry['directory']
        protocol = json.loads((path/'protocol.json').read_text())
        environment = json.loads((path/'environment.json').read_text())
        assert protocol['protocol_id'] == entry['protocol_id']
        assert environment['code_revision'] == entry['code_revision']
        for name, key in (('protocol.json', 'protocol_sha256'), ('summary.json', 'summary_sha256')):
            assert hashlib.sha256((path/name).read_bytes()).hexdigest() == entry[key]
        for name in protocol['datasets']:
            manifest = json.loads((path/name/'manifest.json').read_text())
            assert panel[name]['reference'] == label
            assert (panel[name]['dataset_hash'], panel[name]['splits_hash']) == (manifest['dataset_hash'],
                                                                                manifest['splits_hash'])
    declared = nb.reference_declaration(nb.panel_declaration())
    assert sorted(name for entry in declared.values() for name in entry['datasets']) == sorted(panel)
    assert all(entry['model_id'] == TRAINED_MODEL for entry in declared.values())


def test_the_draft_protocol_copies_the_newdata_nested_design_unchanged_and_declares_this_family():
    template = json.loads((PROTOCOLS/'2026-09-12'/'newdata_batch1.json').read_text())
    protocol = nb.draft_protocol()
    for key in nd.DESIGN_COPY_KEYS:
        assert canonical_json(protocol[key]) == canonical_json(template[key])
    assert all(protocol[key] == value for key, value in nd.DESIGN.items())
    assert (protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds']) == (5, 3, 3)
    assert protocol['split_seed'] == 27183 and protocol['fit_seeds'] == [8129, 19391, 39019]
    assert protocol['candidate_budget'] == 24 and protocol['candidate_seed'] == 41071
    assert protocol['candidate_tie_rule'] == 'lowest_canonical_config_id' and protocol['stochastic_finalists'] == 3
    assert protocol['production_family'] == nb.FAMILY and protocol['registry'] == nb.REGISTRY
    assert protocol['model_order'] == list(nb.MODEL_ORDER) and protocol['primary_family_size'] == 17
    assert protocol['primary_contrasts'] == [f'{TRAINED_MODEL}_vs_{m}' for m in nb.MODEL_ORDER]
    assert protocol['frozen'] is False and protocol['projection'] is None
    assert protocol['source_template_sha256'] == nd.sha256_file(PROTOCOLS/'2026-09-12'/'newdata_batch1.json')
    assert 'metric_learn is not installed' in protocol['analysis']['notes']['lmnn']


def test_the_analysis_is_declared_before_any_score_with_one_holm_family_for_each_baseline():
    block = nb.draft_protocol()['analysis']
    assert block['requires_run_complete'] and block['metric'] == 'accuracy' and len(block['datasets']) == 17
    assert block['reference_model'] == TRAINED_MODEL and 'not refitted' in block['reference_not_refitted']
    assert 'check_training_pairing' in block['reference_not_refitted']
    assert sorted(block['families']) == sorted(nb.MODEL_ORDER)
    for model, family in block['families'].items():
        assert family['model_a'] == TRAINED_MODEL and family['model_b'] == model and family['size'] == 17
        assert family['alpha'] == .05 and 'adjusted separately' in family['multiplicity']
        assert family['interval'] == nd.INTERVAL_RULE and 'q = test_train_ratio = 0.25' in family['interval']
    interpretation = block['interpretation']
    assert interpretation['supervised_baselines'] == list(nb.SUPERVISED_MODELS) == ['lda_knn', 'nca_knn', 'kendall_svc']
    assert 'construction within permutation space rather than an accuracy argument' in interpretation['if_matched']
    assert 'says so plainly in the results and in the discussion' in interpretation['if_matched']
    assert interpretation['if_not_matched'] and interpretation['rule'].endswith('at least 9 of the 17')
    for key in ('metrics_table', 'rank_panel', 'convergence_warnings'):
        assert block['descriptive'][key]


def test_validate_protocol_refuses_a_changed_design_and_a_freeze_outside_the_cap(tmp_path):
    draft = nb.draft_protocol()
    assert nb.validate_protocol(draft) is draft
    for key, value in (('split_seed', 1), ('fit_seeds', [1, 2, 3]), ('datasets', draft['datasets'][:5]),
                       ('candidate_budget', 12), ('outer_folds', 4)):
        with pytest.raises(ValueError, match='differs from neighbour_baselines.draft_protocol'):
            nb.validate_protocol(dict(draft, **{key: value}))
    with pytest.raises(ValueError, match='must equal the draft'):
        nb.validate_protocol(dict(draft, status='invented'))
    frozen = dict(draft, frozen=True, frozen_at_utc='2026-09-14T00:00:00+00:00', status=nb.FROZEN_STATUS,
                  resource_decision='x', projection={'cap_hours': nb.CAP_HOURS, 'workers': nb.WORKERS,
                                                     'decision_hours': 1.})
    assert nb.validate_protocol(frozen)
    for projection in ({'cap_hours': nb.CAP_HOURS, 'workers': nb.WORKERS, 'decision_hours': nb.CAP_HOURS + .1},
                       {'cap_hours': nb.CAP_HOURS, 'workers': 4, 'decision_hours': 1.}, None):
        with pytest.raises(ValueError, match='calibrated projection within'):
            nb.validate_protocol(dict(frozen, projection=projection))


def test_freeze_writes_the_frozen_protocol_only_from_the_committed_draft_and_within_the_cap(tmp_path):
    draft_path, plan_path, stages_path = tmp_path/'draft.json', tmp_path/'plan.json', tmp_path/'stages.json'
    draft = nb.draft_protocol()
    draft_path.write_text(json.dumps(draft))
    stages_path.write_text(json.dumps({'summary': 'stages summary'}))
    plan = {'protocol_hash': config_id(draft), 'family': nb.FAMILY, 'workers': nb.WORKERS, 'within_cap': True,
            'decision_hours': 1.5, 'upper_hours': 2.5, 'decision': 'd',
            'run': {'central': {'simulated_makespan_hours': 1.25}, 'upper': {'simulated_makespan_hours': 2.25}},
            'stage_overhead_hours': .25, 'harness_max_based_run_hours': 3.,
            'per_dataset': {name: {'central_serial_hours': .1, 'upper_serial_hours': .2} for name in draft['datasets']},
            'calibration': {'classical': {'pooled': 1.2, 'max': 1.5}}, 'calibration_source': dict(nb.CALIBRATION_SOURCE),
            'fits_per_outer': {m: 1 for m in nb.MODEL_ORDER}, 'assumptions': 'a'}
    plan_path.write_text(json.dumps(plan))
    frozen = nb.freeze(draft_path, plan_path, stages_path, tmp_path/'frozen.json',
                       frozen_at_utc='2026-09-14T12:00:00+00:00')
    assert frozen['frozen'] and frozen['status'] == nb.FROZEN_STATUS and 'stages summary' in frozen['resource_decision']
    assert frozen['projection']['decision_hours'] == 1.5 and frozen['projection']['stages']['summary'] == 'stages summary'
    assert json.loads((tmp_path/'frozen.json').read_text()) == frozen
    nb.freeze(draft_path, plan_path, stages_path, tmp_path/'frozen.json', frozen_at_utc='2026-09-14T12:00:00+00:00')
    plan_path.write_text(json.dumps(dict(plan, decision_hours=nb.CAP_HOURS + 1, within_cap=False)))
    with pytest.raises(ValueError, match='exceeds the'):
        nb.freeze(draft_path, plan_path, stages_path, tmp_path/'other.json')
    draft_path.write_text(json.dumps(dict(draft, split_seed=1)))
    with pytest.raises(ValueError, match='differs from neighbour_baselines.draft_protocol'):
        nb.freeze(draft_path, plan_path, stages_path, tmp_path/'other.json')


def test_prepare_writes_the_registered_folds_and_refuses_a_dataset_that_is_not_the_pinned_one(tmp_path):
    protocol = nb.draft_protocol()
    X, y, manifest = rr.load_dataset('iris')
    nb.prepare(tmp_path/'ok', protocol, ['iris'])
    splits = json.loads((tmp_path/'ok'/'iris'/'splits.json').read_text())
    assert splits == make_splits(y, 5, 3, 3, 27183)
    saved = json.loads((tmp_path/'ok'/'iris'/'manifest.json').read_text())
    assert saved['dataset_hash'] == manifest['dataset_hash']
    assert saved['splits_hash'] == next(e['splits_hash'] for e in protocol['panel'] if e['name'] == 'iris')
    assert json.loads((tmp_path/'ok'/'protocol.json').read_text()) == protocol
    assert sorted(json.loads((tmp_path/'ok'/'candidates.json').read_text())) == sorted(nb.MODEL_ORDER)

    def wrong(name, _protocol=None):
        return X[:120], y[:120], dict(manifest)
    with pytest.raises(nd.DatasetIdentityError, match='differs from the panel pins'):
        nb.prepare(tmp_path/'wrong', protocol, ['iris'], loader=wrong)
    with pytest.raises(ValueError, match='outside the protocol datasets'):
        nb.prepare(tmp_path/'out', protocol, ['sushi'])
    with pytest.raises(nd.DatasetIdentityError, match='not in the neighbour-baselines panel'):
        nb.load('sushi')


@pytest.mark.skipif(not (RUNS/'2026-09-12-bridge-knn'/'iris'/'manifest.json').is_file(),
                    reason='the registered bridge_knn run is not in this workspace')
def test_the_prepared_manifest_and_splits_equal_the_registered_run_byte_for_byte(tmp_path):
    nb.prepare(tmp_path/'run', nb.draft_protocol(), ['iris'])
    for name in ('manifest.json', 'splits.json'):
        assert (tmp_path/'run'/'iris'/name).read_text() == (RUNS/'2026-09-12-bridge-knn'/'iris'/name).read_text()
