"""native_artificial and compare_native_artificial: the native ranking input and its completion, the four models, the
protocol and its refusals, the descriptive data-property probes and the analysis gate.

The real artificial datasets are loaded here (they are built in memory from artificial_ranks, no download) for the
completion audit only; every fit in this file is tiny and is never evidence.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.make_revision import compare_native_artificial as cna
from experiments.make_revision import extra_data as ed
from experiments.make_revision import native_artificial as na
from experiments.make_revision import newdata as nd
from experiments.make_revision.artificial_ranks import load_artificial, sequences_from_features
from experiments.make_revision.comparisons import CONVENTIONAL_GRIDS, derive_seed
from experiments.make_revision.compare_runs import RunComparisonError
from experiments.make_revision.evaluation import candidate_grid, canonical_json, config_id, make_splits
from experiments.make_revision.models import ArrowFlowEstimator

REPO = Path(__file__).resolve().parents[2]
DATASETS = ('ranks8', 'ranks16', 'ranks8_original')


def standin(name, *, rows=None):
    """load_artificial's contract with stand-in data: relative item positions, NaN for deleted items, labels 0..6."""
    from experiments.make_revision.evaluation import dataset_fingerprint
    n_items = 16 if name == 'ranks16' else 8
    rows = rows or ed.ARTIFICIAL[name]['expected_rows']
    rng = np.random.RandomState(len(name) + n_items)
    y = np.arange(rows) % 7
    X = np.array([rng.permutation(n_items) / (n_items - 1) for _ in range(rows)])
    X[rng.rand(rows, n_items) < .15] = np.nan
    X[np.arange(rows), rng.randint(0, n_items, rows)] = rng.rand(rows)          # never an all-NaN row
    names, labels = [f'item_{i + 1}' for i in range(n_items)], [str(label) for label in range(7)]
    return X, y, {'dataset_id': name, 'source': 'stand-in', 'shape': list(X.shape),
                  'class_counts': np.bincount(y).tolist(), 'feature_names': names, 'label_map': labels,
                  'sample_order': 'stand-in', 'dataset_hash': dataset_fingerprint(X, y, names, labels)}


# ----------------------------------------------------------------------------- the native input

@pytest.mark.parametrize('name', DATASETS)
def test_the_completion_is_a_permutation_and_dropping_the_tail_recovers_the_observed_sequence(name):
    X, y, _ = load_artificial(name)
    audit = na.check_completion(name, X)
    assert audit['every_row_is_a_permutation_of_the_V_items']
    assert audit['dropping_the_tail_recovers_the_observed_sequence']
    assert audit['the_tail_is_exactly_the_missing_items']
    assert audit['the_tail_is_in_ascending_item_order']
    orders, sequences = na.native_orders(X), sequences_from_features(X)
    assert orders.shape == X.shape
    for row, sequence in enumerate(sequences):
        completed = [int(item) + 1 for item in orders[row]]
        assert sorted(completed) == list(range(1, X.shape[1] + 1))               # a permutation of exactly the V items
        assert tuple(completed[:len(sequence)]) == sequence                      # reversible
        assert completed[len(sequence):] == sorted(completed[len(sequence):])    # ascending tail
        assert set(completed[len(sequence):]) == set(range(1, X.shape[1] + 1)) - set(sequence)
    assert audit['rows_with_deletions'] > 0                                      # the datasets really do delete items


def test_the_completion_is_not_injective_and_the_audit_counts_the_collisions():
    X = np.array([[0., .5, 1., np.nan], [0., .5, np.nan, 1.], [0., 1 / 3, 2 / 3, 1.]])
    audit = na.completion_audit(X)
    assert audit['distinct_observed_sequences'] == 3 and audit['distinct_completed_rows'] == 2
    assert audit['collisions_created_by_the_completion'] == 1                    # row 0 completes onto row 2


def test_native_orders_refuses_a_row_without_two_present_items():
    with pytest.raises(ValueError):
        na.native_orders(np.array([[0., 1., np.nan], [np.nan, np.nan, 0.]]))


def test_filled_positions_equal_the_original_features_on_rows_with_no_deletion():
    X, _, _ = load_artificial('ranks8')
    complete = ~np.isnan(X).any(axis=1)
    assert complete.any()
    assert np.allclose(na.filled_positions(X)[complete], X[complete])
    filled = na.filled_positions(X)
    assert np.isfinite(filled).all() and filled.min() == 0. and filled.max() == 1.


def test_the_core_refuses_a_partial_input_row_so_the_completion_is_required():
    """Why the core's partial-input path is not used: an all-sort network validates every input row as a complete
    permutation of the layer vocabulary. This pins the reason recorded in the protocol."""
    from arrowflow.arrowflow import SortFlowHybridNetwork
    from arrowflow.benchmark import ArrowFlowConfig, _build_sortnet_config
    config = _build_sortnet_config(ArrowFlowConfig(no_of_filters=[8], layer_types=['sort', 'sort'], no_of_iters=1,
                                                   val_data_ratio=0., device='cpu', verbose=0,
                                                   evaluate_train_data=False), 3)
    np.random.seed(0)
    network = SortFlowHybridNetwork('t', [str(i) for i in range(1, 9)], 3, 't', config)
    network.forward_propagate([[[str(i) for i in range(1, 9)], '0', 1.]], 'supervised', 'classification', evaluate_only=True)
    with pytest.raises(ValueError, match='complete permutation'):
        network.forward_propagate([[['1', '2', '3'], '0', 1.]], 'supervised', 'classification', evaluate_only=True)


# ----------------------------------------------------------------------------- the models

def test_the_first_layer_vocabulary_is_the_items_and_its_filters_are_random_permutations_of_them():
    X, y, _ = load_artificial('ranks8')
    rows = np.arange(0, 140)
    model = na.NativeUntrainedMultiViewArrowFlowKNN(n_views=2, widths=[16], seed=8129).fit(X[rows], y[rows])
    assert model.vocabulary_size_ == 8 and model.encoding_seconds_ == 0.
    for network in model.views_:
        layer = network.network_.graph.vertex_list['revision_ly0']
        assert list(layer.adj_list_items) == [str(item) for item in range(1, 9)]
        filters = [list(vertex.adjacency_list) for vertex in layer.graph.vertex_list.values()]
        assert len(filters) == 16
        assert all(sorted(int(item) for item in f) == list(range(1, 9)) for f in filters)
        assert len({tuple(f) for f in filters}) > 1                              # they are random, not one repeated order


def test_the_seven_views_differ_only_by_their_seeds():
    X, y, _ = load_artificial('ranks8')
    rows = np.arange(0, 140)
    model = na.NativeUntrainedMultiViewArrowFlowKNN(n_views=3, widths=[16], seed=8129).fit(X[rows], y[rows])
    hashes = [network.training_encoding_hash_ for network in model.views_]
    assert len(set(hashes)) == 1                                                 # every view sees the same input ranking
    initial = [network.initial_state_hash_ for network in model.views_]
    assert len(set(initial)) == 3                                                # they differ by their seeded filters alone
    again = na.NativeUntrainedMultiViewArrowFlowKNN(n_views=3, widths=[16], seed=8129).fit(X[rows], y[rows])
    assert [n.initial_state_hash_ for n in again.views_] == initial              # and reproduce from the seed


def test_the_untrained_control_starts_from_the_filters_the_trained_network_starts_training_from():
    X, y, _ = load_artificial('ranks8')
    rows = np.arange(0, 140)
    orders, seed_v = na.native_orders(X[rows]), derive_seed(8129, 'view', 0)

    def filters(network):
        return [[list(v.adjacency_list) for v in layer.graph.vertex_list.values()]
                for layer in network.network_.graph.vertex_list.values()]

    control = na.NativeUntrainedMultiViewArrowFlowKNN(n_views=1, widths=[16], seed=8129).fit(X[rows], y[rows])
    start = ArrowFlowEstimator(embed_dim=8, widths=[16], learning_rate=.2, p_correct=.1, validation_ratio=.1,
                               iterations=200, batch_size=32, seed=seed_v).initialize_orders(orders, y[rows])
    assert filters(control.views_[0]) == filters(start)
    assert control.views_[0].network_.update_iter == 0 and control.training_seconds_ == 0.
    assert control.initialization_seconds_ > 0.


def test_the_trained_model_trains_its_networks_and_reads_out_a_selected_footrule_knn():
    X, y, _ = load_artificial('ranks8')
    rows, query = np.arange(0, 140), np.arange(140, 180)
    model = na.NativeMultiViewArrowFlowKNN(n_views=2, widths=[16], iterations=3, learning_rate=.1, batch_size=32,
                                           validation_ratio=0, p_correct=.01, seed=8129).fit(X[rows], y[rows])
    assert all(network.network_.update_iter > 0 for network in model.views_)
    assert model.training_seconds_ > 0. and model.encoding_seconds_ == 0. and model.initialization_seconds_ == 0.
    assert len(model.readout_selections_) == 2
    for selection in model.readout_selections_:
        assert set(selection['config']) == {'n_neighbors', 'weights'}
    assert model.representation_metadata_['encoder'] == 'none (native ranking input)'
    assert len(model.predict(X[query])) == len(query)
    with pytest.raises(ValueError):
        na.NativeMultiViewArrowFlowKNN(aggregation='borda').fit(X[rows], y[rows])


def test_the_native_models_refuse_a_changed_vocabulary():
    X, y, _ = load_artificial('ranks8')
    rows = np.arange(0, 140)
    for model in (na.NativeUntrainedMultiViewArrowFlowKNN(n_views=1, widths=[16], seed=8129),
                  na.NativeFootruleKNN(n_neighbors=3)):
        fitted = model.fit(X[rows], y[rows])
        with pytest.raises(ValueError):
            fitted.predict(np.hstack([X[rows][:5], X[rows][:5]]))


def test_numeric_knn_filled_is_the_numeric_knn_pipeline_on_the_completed_positions():
    X, y, _ = load_artificial('ranks8')
    rows = np.arange(0, 140)
    pipeline = na.numeric_knn_filled_factory({'n_neighbors': 5, 'weights': 'uniform', 'p': 2}, 8129)
    assert [name for name, _ in pipeline.steps] == ['completed', 'imputer', 'scaler', 'classifier']
    fitted = pipeline.fit(X[rows], y[rows])
    assert np.allclose(fitted.named_steps['completed'].transform(X[rows]), na.filled_positions(X[rows]))
    assert len(fitted.predict(X[rows][:7])) == 7


# ----------------------------------------------------------------------------- candidates and the registry

def test_the_registry_holds_exactly_the_four_fitted_models_with_their_grids():
    protocol = na.draft_protocol()
    registry = na.build_registry(protocol)
    assert list(registry) == list(na.MODEL_ORDER) == [na.TRAINED_MODEL, na.UNTRAINED_MODEL, na.INPUT_MODEL, na.FILLED_MODEL]
    assert len(registry[na.TRAINED_MODEL].candidates) == 16                     # the main method's candidate count
    assert len(registry[na.UNTRAINED_MODEL].candidates) == 2                    # the settings that still act without training
    assert len(registry[na.INPUT_MODEL].candidates) == 10
    assert [spec.stochastic for spec in registry.values()] == [True, True, False, False]
    for config in registry[na.TRAINED_MODEL].candidates:
        assert 'embed_scale' not in config and 'degree_offset' not in config    # there is no encoder to tune
        assert set(config) == {'n_views', 'aggregation', 'iterations', 'batch_size', 'widths', 'learning_rate',
                               'p_correct', 'validation_ratio'}
        assert config['n_views'] == 7 and config['aggregation'] == 'majority'
        assert config['widths'] in ([128], [64, 128]) and config['learning_rate'] in (.1, .2)
        assert config['p_correct'] in (.01, .1) and config['validation_ratio'] in (0, .1)
    assert {tuple(config['widths']) for config in registry[na.UNTRAINED_MODEL].candidates} == {(128,), (64, 128)}
    assert registry[na.FILLED_MODEL].candidates == candidate_grid(
        CONVENTIONAL_GRIDS['numeric_knn'], protocol['candidate_budget'], protocol['candidate_seed'])


def test_every_candidate_list_fits_the_declared_budget():
    protocol = na.draft_protocol()
    assert all(len(spec.candidates) <= protocol['candidate_budget'] for spec in na.build_registry(protocol).values())


# ----------------------------------------------------------------------------- the protocol

def test_the_draft_holds_the_finished_artificial_design_so_the_folds_are_identical():
    protocol = na.draft_protocol()
    artificial = json.loads(na.ARTIFICIAL_PROTOCOL_FILE.read_text())
    for key in na.DESIGN_KEYS:
        assert canonical_json(protocol[key]) == canonical_json(artificial[key]), key
    assert (protocol['outer_folds'], protocol['outer_repeats'], protocol['inner_folds'], protocol['split_seed']) == (5, 3, 3, 27183)
    assert protocol['fit_seeds'] == [8129, 19391, 39019]
    assert protocol['datasets'] == list(ed.ARTIFICIAL_DATASETS)
    assert protocol['artificial_protocol_sha256'] == na.sha256_file(na.ARTIFICIAL_PROTOCOL_FILE)


def test_the_encoded_run_pins_name_the_committed_artificial_protocol_and_its_folds():
    artificial = json.loads(na.ARTIFICIAL_PROTOCOL_FILE.read_text())
    assert na.ENCODED_RUN['protocol_sha256'] == na.sha256_file(na.ARTIFICIAL_PROTOCOL_FILE)
    assert na.ENCODED_RUN['protocol_id'] == artificial['protocol_id']
    panel = {entry['name']: entry for entry in artificial['panel']}
    for name, pins in na.ENCODED_RUN['datasets'].items():
        assert pins == {key: panel[name][key] for key in ('dataset_hash', 'splits_hash')}
    for name in na.ENCODED_RUN['datasets']:
        X, y, _ = load_artificial(name)
        splits = make_splits(y, 5, 3, 3, 27183)
        assert config_id(splits) == na.ENCODED_RUN['datasets'][name]['splits_hash']


def test_the_protocol_declares_the_completion_the_reuse_and_the_families_before_any_score():
    protocol = na.draft_protocol()
    assert protocol['frozen'] is False and protocol['projection'] is None
    assert protocol['native_input']['rule'] == na.COMPLETION_RULE
    assert 'validate_permutation' in protocol['native_input']['reason']
    assert protocol['comparator_reuse']['status'] == 'reused, NOT refitted'
    assert protocol['comparator_reuse']['models'] == [*na.REUSED_COMPARATORS, na.MAJORITY_MODEL]
    block = protocol['analysis']
    assert block['family_datasets'] == ['ranks8', 'ranks16'] and block['descriptive_only_datasets'] == ['ranks8_original']
    assert block['primary_family']['model_a'] == na.TRAINED_MODEL and block['primary_family']['model_b'] == na.UNTRAINED_MODEL
    assert block['secondary_family']['model_b'] == na.INPUT_MODEL
    assert 'separately' in block['secondary_family']['multiplicity']
    assert block['encoded_contrast']['model_b'] == na.ENCODED_RUN['arrowflow_model']
    assert 'PAIRED' in block['encoded_contrast']['status']
    assert block['interpretation'] == na.INTERPRETATION
    assert na.VIEW_NOTE in block['notes']['views']
    assert block['data_properties']['probes'] == na.PROBES


def test_the_protocol_holds_no_machine_or_worktree_dependent_path():
    """Every worker and the production worktree re-derive the draft, so the protocol must be identical everywhere."""
    text = json.dumps(na.draft_protocol())
    assert not any(token in text for token in ('/home/', '.worktrees', '.superpowers', str(REPO)))
    assert 'path' not in na.draft_protocol()['comparator_reuse']['source']


def test_validate_refuses_a_protocol_that_differs_from_the_draft_or_freezes_without_a_projection():
    protocol = na.draft_protocol()
    assert na.validate_native_protocol(protocol) == protocol
    with pytest.raises(ValueError, match='outer_folds'):
        na.validate_native_protocol(dict(protocol, outer_folds=4))
    with pytest.raises(ValueError, match='production_family'):
        na.validate_native_protocol(dict(protocol, production_family='artificial'))
    with pytest.raises(ValueError, match='calibrated projection'):
        na.validate_native_protocol(dict(protocol, frozen=True, status=na.FROZEN_STATUS, frozen_at_utc='x',
                                         resource_decision='x', projection=None))


def test_freeze_refuses_a_projection_over_the_cap_and_writes_the_frozen_protocol_within_it(tmp_path):
    draft = na.draft_protocol()
    (tmp_path/'draft.json').write_text(json.dumps(draft))
    (tmp_path/'stages.json').write_text(json.dumps({'summary': 'stages ran'}))
    over = {'protocol_hash': config_id(draft), 'family': na.FAMILY, 'workers': na.WORKERS, 'within_cap': False,
            'decision_hours': na.CAP_HOURS + 1, 'upper_hours': na.CAP_HOURS + 2, 'decision': 'd',
            'run': {'central': {'simulated_makespan_hours': 1.}, 'upper': {'simulated_makespan_hours': 2.}},
            'stage_overhead_hours': na.STAGE_OVERHEAD_HOURS, 'harness_max_based_run_hours': 1.,
            'per_dataset': {}, 'calibration': {}, 'calibration_source': {}, 'assumptions': 'a'}
    (tmp_path/'over.json').write_text(json.dumps(over))
    with pytest.raises(ValueError, match='exceeds'):
        na.freeze(tmp_path/'draft.json', tmp_path/'over.json', tmp_path/'stages.json', tmp_path/'out.json')
    assert not (tmp_path/'out.json').exists()
    inside = dict(over, within_cap=True, decision_hours=.5, upper_hours=.9)
    (tmp_path/'in.json').write_text(json.dumps(inside))
    frozen = na.freeze(tmp_path/'draft.json', tmp_path/'in.json', tmp_path/'stages.json', tmp_path/'out.json',
                       frozen_at_utc='2026-09-15T00:00:00+00:00')
    assert frozen['frozen'] is True and frozen['projection']['decision_hours'] == .5
    assert na.validate_native_protocol(json.loads((tmp_path/'out.json').read_text())) == frozen
    with pytest.raises(ValueError, match='differs from'):
        na.freeze(tmp_path/'in.json', tmp_path/'in.json', tmp_path/'stages.json', tmp_path/'other.json')


# ----------------------------------------------------------------------------- the descriptive data properties

def test_the_probes_are_a_deterministic_lookup_whose_majority_arm_is_the_majority_class():
    X = np.array([[0., 1., np.nan], [0., 1., np.nan], [0., .5, 1.], [1., .5, 0.]])
    y = np.array([0, 0, 1, 1])
    splits = [{'train': [0, 2], 'test': [1, 3]}, {'train': [1, 3], 'test': [0, 2]}]
    assert na.probe_keys(X, 'missing_count') == [1, 1, 0, 0]
    assert na.probe_keys(X, 'missing_items') == [(3,), (3,), (), ()]
    assert na.probe_keys(X, 'majority_class') == [(), (), (), ()]
    # each training partition ties 0 against 1, so the majority arm takes the lowest class label and is right half the time
    assert na.probe_accuracies(X, y, splits, 'majority_class') == [.5, .5]
    # the number of missing items alone, and their identity alone, separate the two classes perfectly here
    assert na.probe_accuracies(X, y, splits, 'missing_count') == [1., 1.]
    assert na.probe_accuracies(X, y, splits, 'missing_items') == [1., 1.]
    with pytest.raises(ValueError):
        na.probe_keys(X, 'anything_else')


def test_an_unseen_probe_key_falls_back_to_the_training_majority():
    X = np.array([[0., 1., np.nan], [0., .5, 1.], [np.nan, 0., 1.]])
    y = np.array([0, 0, 1])
    assert na.probe_accuracies(X, y, [{'train': [0, 1], 'test': [2]}], 'missing_items') == [0.]


@pytest.mark.parametrize('name', DATASETS)
def test_data_properties_measure_the_completion_artifact_of_every_dataset(name):
    protocol = na.draft_protocol()
    record = na.data_properties(protocol, [name], loaders={na.FAMILY: load_artificial})
    entry = record['datasets'][name]
    assert entry['completion_audit']['dropping_the_tail_recovers_the_observed_sequence']
    assert set(entry['probes']) == set(na.PROBES)
    assert entry['probes']['majority_class']['mean_accuracy'] == pytest.approx(1 / 7, abs=.02)
    for probe in ('missing_count', 'missing_items'):
        assert entry['probes'][probe]['n_folds'] == 15
        assert entry['probes'][probe]['above_majority_points'] >= 0.
    assert record['protocol_hash'] == config_id(protocol)


# ----------------------------------------------------------------------------- prepare and the analysis gate

def test_prepare_writes_the_harness_layout_with_the_completion_audit_and_checks_both_sets_of_pins(tmp_path):
    protocol = na.draft_protocol()
    na.prepare(tmp_path/'run', protocol, ['ranks8'], loaders={na.FAMILY: load_artificial})
    manifest = json.loads((tmp_path/'run'/'ranks8'/'manifest.json').read_text())
    assert manifest['native_input']['rule'] == na.COMPLETION_RULE
    assert manifest['native_input']['audit']['every_row_is_a_permutation_of_the_V_items']
    assert manifest['dataset_hash'] == na.ENCODED_RUN['datasets']['ranks8']['dataset_hash']
    assert manifest['splits_hash'] == na.ENCODED_RUN['datasets']['ranks8']['splits_hash']
    candidates = json.loads((tmp_path/'run'/'candidates.json').read_text())
    assert sorted(candidates) == sorted(na.MODEL_ORDER)
    environment = json.loads((tmp_path/'run'/'environment.json').read_text())
    assert environment['registry'] == na.REGISTRY
    assert 'experiments/make_revision/native_artificial.py' in environment['source_hashes']
    assert 'experiments/make_revision/artificial_ranks.py' in environment['source_hashes']


def test_prepare_refuses_a_dataset_whose_folds_differ_from_the_encoded_run(tmp_path, monkeypatch):
    # the draft copies the encoded pins, so patch them first and build the protocol from the patched pins: the panel pins
    # still hold and only the encoded-fold check can fail
    monkeypatch.setitem(na.ENCODED_RUN['datasets']['ranks8'], 'splits_hash', 'not-the-encoded-splits')
    protocol = na.draft_protocol()
    with pytest.raises(ed.DatasetIdentityError, match='encoded comparison run'):
        na.prepare(tmp_path/'run', protocol, ['ranks8'], loaders={na.FAMILY: load_artificial})


def test_prepare_refuses_an_unpinned_or_changed_dataset(tmp_path):
    protocol = na.draft_protocol()
    with pytest.raises(ed.DatasetIdentityError):
        na.prepare(tmp_path/'run', protocol, ['ranks8'], loaders={na.FAMILY: lambda name: standin(name)})


def test_the_analysis_refuses_an_absent_or_incomplete_run_and_writes_nothing(tmp_path):
    with pytest.raises(RunComparisonError, match='no such directory'):
        cna.analyse(tmp_path/'missing', tmp_path/'analysis')
    incomplete = tmp_path/'run'
    incomplete.mkdir()
    (incomplete/'protocol.json').write_text('{}')
    with pytest.raises(RunComparisonError, match='not complete'):
        cna.analyse(incomplete, tmp_path/'analysis')
    assert not (tmp_path/'analysis').exists()


def test_the_analysis_declares_every_output_and_the_sealed_sources():
    assert cna.RECORD_NAME in cna.OUTPUTS and len(set(cna.OUTPUTS)) == len(cna.OUTPUTS)
    assert 'experiments/make_revision/native_artificial.py' in cna.SEALED_SOURCES
    assert cna.ENCODED_PAIRS[0] == (na.TRAINED_MODEL, 'arrowflow_full_knn', 'native_minus_encoded_arrowflow_knn')


def test_the_smoke_protocol_is_synthetic_frozen_and_never_the_production_registry():
    protocol = na.smoke_protocol()
    assert protocol['purpose'] == 'synthetic_smoke_only' and protocol['registry'] == na.SMOKE_REGISTRY
    assert protocol['protocol_id'].endswith('-synthetic-smoke') and protocol['frozen'] is True
    assert na.validate_native_protocol(protocol) == protocol
    with pytest.raises(ValueError, match='synthetic smoke'):
        na.native_artificial_registry(protocol)
    with pytest.raises(ValueError, match='synthetic smoke'):
        na.smoke_native_artificial_registry(na.draft_protocol())
    registry = na.smoke_native_artificial_registry(protocol)
    assert list(registry) == list(na.MODEL_ORDER)
    assert all(config['iterations'] == 1 for config in registry[na.TRAINED_MODEL].candidates)
    for index, entry in enumerate(protocol['panel']):
        X, y = na.synthetic_dataset(entry, index)
        assert na.check_completion(entry['name'], X)['every_row_is_a_permutation_of_the_V_items']
        assert len(np.unique(y)) == 3


# ----------------------------------------------------------------------------- the analysis row builders on real rows

WORKSPACE_RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'
ENCODED_SOURCE = WORKSPACE_RUNS/'2026-09-14-artificial'/'run'
REMAP = {'arrowflow_full_knn': na.TRAINED_MODEL, 'arrowflow_knn_untrained': na.UNTRAINED_MODEL,
         'input_footrule_knn': na.INPUT_MODEL, 'numeric_knn': na.FILLED_MODEL}


def _stand_in(encoded, mapping):
    """A stand-in native run: the encoded run's verified rows under the native model IDs. It exercises every row builder of
    the analysis on real records, including the cross-run pairing, which no synthetic smoke can reach."""
    from types import SimpleNamespace
    summaries, rows = {}, {}
    for name in encoded.protocol['datasets']:
        summaries[name] = [dict(row, model_id=mapping[row['model_id']])
                           for row in encoded.summary['summaries'][name] if row['model_id'] in mapping]
        rows[name] = [dict(row, model_id=mapping[row['model_id']])
                      for row in encoded.summary['model_rows'][name] if row['model_id'] in mapping]
    schedule = {'expected_folds': encoded.schedule['expected_folds'],
                'expected_seeds': {mapping[model]: seeds for model, seeds in encoded.schedule['expected_seeds'].items()
                                   if model in mapping}}
    return SimpleNamespace(label='native', path=encoded.path, protocol=encoded.protocol, environment=encoded.environment,
                           summary={'summaries': summaries, 'model_rows': rows}, schedule=schedule,
                           registry={mapping[model]: None for model in mapping}, manifests=encoded.manifests,
                           jobs=encoded.jobs), rows


@pytest.mark.skipif(not (ENCODED_SOURCE/'summary.json').is_file(), reason='the encoded comparison run is not in this checkout')
def test_the_analysis_row_builders_run_on_the_real_verified_rows_of_the_encoded_arm():
    from experiments.make_revision.compare_runs import load_run
    encoded = load_run(ENCODED_SOURCE, 'encoded')
    native, verified = _stand_in(encoded, REMAP)
    names = list(encoded.protocol['datasets'])
    roles = {entry['name']: entry['role'] for entry in json.loads(na.PROTOCOL_FILE.read_text())['panel']}
    block = json.loads(na.PROTOCOL_FILE.read_text())['analysis']
    q, confidence = .25, .95
    primary = cna.family_rows(native, block['primary_family'], 'primary', roles, q, confidence)
    assert [row['dataset'] for row in primary] == ['ranks8', 'ranks16']
    assert all(row['n_folds'] == 15 and row['df'] == 14 for row in primary)
    assert all(0 <= row['holm_p_approximate'] <= 1 for row in primary)
    main = cna.main_table_rows(native, encoded, names, roles)
    assert len(main) == len(names) * (2 + len(na.REUSED_COMPARATORS) + 1)
    assert {row['input'] for row in main} == {cna.NATIVE_INPUT_LABEL, cna.ENCODED_INPUT_LABEL}
    comparators = cna.comparator_rows(native, encoded, names, roles, q, confidence)
    assert len(comparators) == len(names) * (1 + len(na.REUSED_COMPARATORS) + 1)
    # every comparator tied at the lowest mean error is flagged; here numeric_knn_filled IS the reused numeric_knn,
    # so each dataset flags a tie, which is the rule working, not a defect
    for name in names:
        flagged = [row for row in comparators if row['dataset'] == name and row['best_comparator']]
        assert flagged and len({round(row['mean_error_b'], 12) for row in flagged}) == 1
    assert all(('REUSED' in row['status']) == (row['model_b'] != na.FILLED_MODEL) for row in comparators)
    competitive = cna.competitiveness_rows(native, encoded, names, roles)
    assert len(competitive) == len(names) and all(isinstance(row['within_three_points'], bool) for row in competitive)
    assert len(cna.ladder_rows(native, names, roles)) == len(names) * len(na.LADDER)
    assert len(cna.complete_metric_rows(native, names, roles)) == len(names) * len(na.MODEL_ORDER)
    widths, counts = cna.width_rows(native, verified, names, roles)
    assert len(widths) == len(names) * 15 and set(counts) == set(names)
    contrast = cna.encoded_contrast_rows(native, encoded, names, roles, q, confidence)
    assert len(contrast) == len(names) * len(cna.ENCODED_PAIRS)
    # the stand-in is the encoded run itself, so every paired difference against it must be exactly zero
    assert all(row['mean_difference'] == 0. for row in contrast)
    assert all('PAIRED' in row['status'] for row in contrast)
    assert cna.check_pairing(native, encoded)['identical_dataset_and_splits_hashes'] is True
