"""The prespecified analysis of the neighbour-baselines run: the completeness gate, the reference pins and the pairing,
the Holm family of each baseline, the descriptive tables and the interpretation fixed before any score existed."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.make_revision import compare_baselines as cb
from experiments.make_revision import neighbour_baselines as nb
from experiments.make_revision.compare_runs import RunComparisonError, load_run, verify_run
from experiments.make_revision.evaluation import paired_corrected_interval
from experiments.make_revision.knn_controls import TRAINED_MODEL

SMOKE_PANEL = ({'name': 'syn_a', 'samples': 60, 'features': 4}, {'name': 'syn_b', 'samples': 45, 'features': 4})


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    """A synthetic ArrowFlow-kNN reference run and a synthetic baselines run over the same two datasets and folds."""
    root = tmp_path_factory.mktemp('baselines')
    protocol = nb.smoke_protocol(SMOKE_PANEL)
    candidates = nb.smoke_reference_candidates()[:1]
    reference = nb.run_smoke_family(root/'reference', nb.smoke_reference_protocol(protocol, candidates),
                                    'experiments.make_revision.neighbour_baselines:smoke_reference_registry', 3,
                                    SMOKE_PANEL)
    run = nb.run_smoke_family(root/'run', protocol, protocol['registry'], 3, SMOKE_PANEL)
    return {'root': root, 'run': run, 'reference': reference, 'protocol': protocol}


# ----------------------------------------------------------------------------- the gate and the pure helpers

def test_the_completeness_gate_refuses_before_any_record_holding_a_score_is_read(tmp_path, synthetic):
    with pytest.raises(RunComparisonError, match='no such run directory'):
        cb.completeness_gate({'baselines': tmp_path/'absent'})
    (tmp_path/'partial').mkdir()
    with pytest.raises(RunComparisonError, match='missing protocol.json'):
        cb.completeness_gate({'baselines': tmp_path/'partial'})
    copied = tmp_path/'copied'
    copied.mkdir()
    for name in ('protocol.json', 'environment.json', 'candidates.json', 'planned_jobs.json', 'summary.json'):
        (copied/name).write_bytes((synthetic['run']/name).read_bytes())
    with pytest.raises(RunComparisonError, match='planned result files or fit logs missing'):
        cb.completeness_gate({'baselines': copied})
    cb.completeness_gate({'baselines': synthetic['run'], 'knn': synthetic['reference']})


def test_average_ranks_order_by_error_and_share_a_tie():
    assert cb.average_ranks({'a': .1, 'b': .3, 'c': .2}) == {'a': 1, 'c': 2, 'b': 3}
    assert cb.average_ranks({'a': .2, 'b': .2, 'c': .5}) == {'a': 1.5, 'b': 1.5, 'c': 3}
    assert cb.average_ranks({'a': .2, 'b': .2, 'c': .2}) == {'a': 2, 'b': 2, 'c': 2}
    assert cb.average_ranks({'a': .2, 'b': .2 + 1e-15}) == {'a': 1.5, 'b': 1.5}


def test_the_interpretation_is_the_rule_the_protocol_fixed_before_any_score_existed():
    block = nb.draft_protocol()['analysis']
    names = list(block['datasets'])
    beaten = {name: {TRAINED_MODEL: .5, **{model: .1 for model in nb.MODEL_ORDER}} for name in names}
    reading = cb.interpretation(block, names, beaten)
    assert reading['threshold_datasets'] == 9 and reading['matched']
    assert reading['supervised_baselines_matching_on_most_datasets'] == list(nb.SUPERVISED_MODELS)
    assert reading['statement'] == block['interpretation']['if_matched']
    assert all(entry['count'] == 17 and entry['most_datasets'] for entry in reading['by_model'].values())
    losing = {name: {TRAINED_MODEL: .1, **{model: .5 for model in nb.MODEL_ORDER}} for name in names}
    for name in names[:8]:
        losing[name] = {TRAINED_MODEL: .5, **{model: .1 for model in nb.MODEL_ORDER}}      # eight is not most of seventeen
    reading = cb.interpretation(block, names, losing)
    assert not reading['matched'] and reading['statement'] == block['interpretation']['if_not_matched']
    assert all(entry['count'] == 8 and not entry['most_datasets'] for entry in reading['by_model'].values())
    tied = {name: {TRAINED_MODEL: .25, **{model: .25 for model in nb.MODEL_ORDER}} for name in names}
    assert cb.interpretation(block, names, tied)['matched']                     # matching counts, not only beating


# ----------------------------------------------------------------------------- the pins and the pairing

def test_the_reference_block_must_pin_the_runs_and_partition_the_panel(synthetic):
    baselines = load_run(synthetic['run'], cb.BASELINES)
    reference = {'knn': load_run(synthetic['reference'], 'knn')}
    record = cb.check_reference(baselines, reference, smoke=True)
    assert record['knn']['model_id'] == TRAINED_MODEL and record['knn']['pins_checked'] is False
    assert sorted(record['knn']['datasets']) == sorted(baselines.protocol['datasets'])
    with pytest.raises(RunComparisonError, match='does not match the knn run'):
        cb.check_reference(baselines, reference, smoke=False)
    broken = copy.deepcopy(baselines)
    broken.protocol['reference']['knn']['datasets'] = ['syn_a']
    with pytest.raises(RunComparisonError, match='partition the protocol datasets'):
        cb.check_reference(broken, reference, smoke=True)
    broken = copy.deepcopy(baselines)
    broken.protocol['reference']['knn']['model_id'] = 'other'
    with pytest.raises(RunComparisonError, match='must pin and hold'):
        cb.check_reference(broken, reference, smoke=True)


def test_the_pairing_requires_the_same_design_folds_hashes_and_sealed_sources(synthetic):
    baselines = load_run(synthetic['run'], cb.BASELINES)
    reference = {'knn': load_run(synthetic['reference'], 'knn')}
    record = cb.check_pairing(baselines, reference)
    assert sorted(record['datasets']) == sorted(baselines.protocol['datasets'])
    assert record['design']['split_seed'] == baselines.protocol['split_seed'] and record['n_folds'] == 3
    assert 'experiments/make_revision/evaluation.py' in record['shared_sources']['knn']
    for key, value in (('split_seed', 5), ('fit_seeds', [1, 2, 3]), ('test_train_ratio', .5), ('inner_folds', 9)):
        broken = copy.deepcopy(reference)
        broken['knn'].protocol[key] = value
        with pytest.raises(RunComparisonError, match=f'Protocol {key} differs'):
            cb.check_pairing(baselines, broken)
    for key in ('dataset_hash', 'splits_hash'):
        broken = copy.deepcopy(reference)
        broken['knn'].manifests['syn_a'][key] = 'x' * 16
        with pytest.raises(RunComparisonError, match=f'The {key} of syn_a differs'):
            cb.check_pairing(baselines, broken)
    broken = copy.deepcopy(reference)
    broken['knn'].environment['source_hashes']['experiments/make_revision/evaluation.py'] = 'x' * 64
    with pytest.raises(RunComparisonError, match='Sources sealed by both'):
        cb.check_pairing(baselines, broken)
    broken = copy.deepcopy(reference)
    broken['knn'].environment['source_hashes'].pop('arrowflow/ranking.py')
    with pytest.raises(RunComparisonError, match='must both seal'):
        cb.check_pairing(baselines, broken)


def test_a_production_run_must_hold_the_committed_frozen_protocol_and_the_four_baselines(synthetic):
    baselines = load_run(synthetic['run'], cb.BASELINES)
    assert cb.check_frozen_protocol(baselines, smoke=True)['frozen_file'] is None
    with pytest.raises(RunComparisonError, match='holds a synthetic smoke protocol'):
        cb.check_frozen_protocol(baselines, smoke=False)
    reference = load_run(synthetic['reference'], 'knn')
    with pytest.raises(RunComparisonError, match='not a neighbour-baselines protocol'):
        cb.check_frozen_protocol(reference, smoke=True)


# ----------------------------------------------------------------------------- the analysis

def test_analyse_refuses_an_incomplete_run_writes_nothing_and_exits_two(tmp_path, synthetic):
    partial = tmp_path/'partial'
    partial.mkdir()
    for name in ('protocol.json', 'environment.json', 'candidates.json', 'planned_jobs.json', 'summary.json'):
        (partial/name).write_bytes((synthetic['run']/name).read_bytes())
    with pytest.raises(RunComparisonError, match='is not complete'):
        cb.analyse(partial, tmp_path/'out', references={'knn': synthetic['reference']}, allow_smoke=True)
    assert not (tmp_path/'out').exists()
    with pytest.raises(SystemExit) as exit_info:
        cb.main(['analyse', '--run', str(partial), '--output', str(tmp_path/'cli'), '--runs', str(tmp_path)])
    assert exit_info.value.code == 2 and not (tmp_path/'cli').exists()
    with pytest.raises(RunComparisonError, match='never evidence'):
        cb.analyse(synthetic['run'], tmp_path/'strict', references={'knn': synthetic['reference']})
    assert not (tmp_path/'strict').exists()


def test_analyse_writes_every_output_once_and_refuses_to_replace_a_different_one(tmp_path, synthetic):
    output = tmp_path/'analysis'
    record = cb.analyse(synthetic['run'], output, references={'knn': synthetic['reference']}, allow_smoke=True)
    assert sorted(path.name for path in output.iterdir()) == sorted(cb.OUTPUTS)
    assert json.loads((output/cb.ANALYSIS_JSON).read_text()) == record
    cb.analyse(synthetic['run'], output, references={'knn': synthetic['reference']}, allow_smoke=True)   # idempotent
    (output/cb.RANKS_CSV).write_text('changed\n')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        cb.analyse(synthetic['run'], output, references={'knn': synthetic['reference']}, allow_smoke=True)


def test_the_analysis_holds_one_holm_family_for_each_baseline_paired_with_the_reference_run(tmp_path, synthetic):
    record = cb.analyse(synthetic['run'], tmp_path/'analysis', references={'knn': synthetic['reference']},
                        allow_smoke=True)
    names = record['datasets']
    assert sorted(record['families']) == sorted(nb.MODEL_ORDER)
    baselines, reference = load_run(synthetic['run'], cb.BASELINES), load_run(synthetic['reference'], 'knn')
    for model, rows in record['families'].items():
        assert [row['dataset'] for row in rows] == names and all(row['model_a'] == TRAINED_MODEL for row in rows)
        assert all(row['model_b'] == model and row['n_folds'] == 3 and row['df'] == 2 for row in rows)
        assert all(row['holm_p_approximate'] >= row['p_approximate'] - 1e-12 for row in rows)
        for row in rows:
            seeds = {TRAINED_MODEL: reference.schedule['expected_seeds'][TRAINED_MODEL],
                     model: baselines.schedule['expected_seeds'][model]}
            expected = paired_corrected_interval(
                [r for r in reference.summary['model_rows'][row['dataset']] if r['model_id'] == TRAINED_MODEL]
                + [r for r in baselines.summary['model_rows'][row['dataset']] if r['model_id'] == model],
                TRAINED_MODEL, model, metric='accuracy', q=baselines.protocol['test_train_ratio'],
                confidence=baselines.protocol['confidence'], expected_folds=baselines.schedule['expected_folds'],
                expected_seeds=seeds)
            assert row['mean_difference'] == pytest.approx(expected['mean_difference'], abs=1e-12)
            assert (row['ci_low'], row['ci_high']) == (expected['ci_low'], expected['ci_high'])
    # the Holm adjustment stays inside one baseline: each family of two is adjusted on its own two p values
    for model, rows in record['families'].items():
        ordered = sorted(row['p_approximate'] for row in rows)
        assert min(row['holm_p_approximate'] for row in rows) == pytest.approx(min(1., 2*ordered[0]), abs=1e-12)


def test_the_descriptive_tables_cover_every_metric_the_rank_panel_and_the_convergence_warnings(tmp_path, synthetic):
    record = cb.analyse(synthetic['run'], tmp_path/'analysis', references={'knn': synthetic['reference']},
                        allow_smoke=True)
    names = record['datasets']
    metrics = {(row['dataset'], row['model_id'], row['metric']) for row in record['metrics']}
    assert metrics == {(name, model, metric) for name in names for model in (TRAINED_MODEL, *nb.MODEL_ORDER)
                       for metric in cb.METRICS}
    for row in record['metrics']:
        if row['model_id'] in (nb.LDA_MODEL, nb.PCA_MODEL):
            assert row['seeds_per_fold'] == 1 and row['mean_within_fold_seed_sd'] is None
        else:
            assert row['seeds_per_fold'] == 3
    assert len(record['ranks']) == len(names) * len(cb.RANK_PANEL) == len(names) * 11
    for name in names:
        ranks = {row['model_id']: row['rank'] for row in record['ranks'] if row['dataset'] == name}
        assert sorted(ranks) == sorted(cb.RANK_PANEL) and min(ranks.values()) >= 1 and max(ranks.values()) <= 11
        assert sum(ranks.values()) == pytest.approx(sum(range(1, 12)))
    warnings_ = {(row['dataset'], row['model_id'], row['scope']): row for row in record['convergence_warnings']}
    assert sorted(warnings_) == sorted((name, model, scope) for name in names for model in nb.MODEL_ORDER
                                       for scope in ('all_fits', 'outer_fits'))
    for (name, model, scope), row in warnings_.items():
        assert row['fits'] > 0 and row['fits_with_warning'] <= row['fits']
        assert warnings_[name, model, 'all_fits']['fits'] >= warnings_[name, model, 'outer_fits']['fits']
        assert (row['reached_max_iter_fits'] is None) == (model != nb.NCA_MODEL)
    assert record['interpretation']['statement'] in nb.draft_protocol()['analysis']['interpretation'].values()
    assert record['provenance']['pairing']['reference_model'] == TRAINED_MODEL
    assert record['provenance']['verification'][cb.BASELINES]['jobs_verified'] == len(names) * 3 * len(nb.MODEL_ORDER)


def test_the_pairing_command_checks_a_prepared_directory_before_the_jobs_start(synthetic):
    record = cb.check_prepared(synthetic['run'], references={'knn': synthetic['reference']})
    assert record['reference']['knn']['pins_checked'] is False
    assert sorted(record['pairing']['datasets']) == sorted(synthetic['protocol']['datasets'])
    assert record['runs']['knn']['protocol_id'].endswith('-synthetic-smoke-reference')


def test_the_console_report_prints_one_line_for_every_family_member_and_the_interpretation(tmp_path, synthetic, capsys):
    record = cb.analyse(synthetic['run'], tmp_path/'analysis', references={'knn': synthetic['reference']},
                        allow_smoke=True)
    cb.print_record(record)
    lines = capsys.readouterr().out.splitlines()
    members = sum(len(rows) for rows in record['families'].values())
    assert len(lines) == members + len(nb.MODEL_ORDER) + 1
    assert sum('matched or beat arrowflow_full_knn on' in line for line in lines) == 4
    assert lines[-1] == record['interpretation']['statement']
    assert all(f'{model} ' in ''.join(lines) for model in nb.MODEL_ORDER)


def test_the_pairing_command_prints_the_pinned_runs_and_the_paired_design(tmp_path, synthetic, capsys):
    root = tmp_path/'runs'
    root.mkdir()
    (root/'2026-09-12-bridge-knn').symlink_to(synthetic['reference'])
    cb.main(['pairing', '--run', str(synthetic['run']), '--runs', str(root)])
    out = capsys.readouterr().out
    assert 'paired arrowflow-v3-neighbour-baselines-1-synthetic-smoke with knn' in out
    assert '2 datasets with identical dataset and splits hashes' in out and '3 outer folds' in out
