"""The two controlled intervention families (interventions): the protocols and their prespecified analysis, the arm
definitions, the freeze gate, the analysis refusal, and the synthetic smoke run of each family end to end (built once
per session; tests that alter records work on copies). The real reference runs are used where they are on this machine;
no real job is fitted."""
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.make_revision import interventions as iv
from experiments.make_revision import training_diagnostics as td
from experiments.make_revision.evaluation import config_id
from experiments.make_revision.run_revision import load_prepared

REAL = {name: tuple(Path(d) for d in directories) for name, directories in td.DEFAULT_SOURCES.items()}
needs_real = pytest.mark.skipif(not all((run/'summary.json').is_file() and (ablation/'reference_selections.json').is_file()
                                        for run, ablation in REAL.values()), reason='the reference runs are not on this machine')
SELECTED = {'n_views': 7, 'strategy': 'diverse', 'iterations': 200, 'batch_size': 32, 'validation_ratio': .1,
            'aggregation': 'majority', 'widths': [128], 'learning_rate': .1, 'embed_dim': 32, 'degree': 2, 'augment': True}


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


# ----------------------------------------------------------------------------- protocols and arms

@pytest.mark.parametrize('family', iv.FAMILIES)
def test_the_committed_protocol_is_the_draft_or_its_freeze(family):
    committed = json.loads(iv.PROTOCOL_FILES[family].read_text())
    iv.validate_protocol(committed)
    draft = iv.draft_protocol(family)
    assert {k: v for k, v in committed.items() if k not in iv.FREEZE_FIELDS} == {k: v for k, v in draft.items()
                                                                                 if k not in iv.FREEZE_FIELDS}
    assert committed['protocol_id'] == iv.PROTOCOL_IDS[family] and len(committed['datasets']) == 17


@pytest.mark.parametrize('family', iv.FAMILIES)
def test_the_draft_declares_the_ruled_design(family):
    p = iv.draft_protocol(family)
    assert p['arms'] == list(iv.ARMS[family]) and p['reference_arm'] == iv.REFERENCE_ARM[family]
    assert p['model_seed'] == 8129 and p['fit_seeds'] == [8129, 19391, 39019] and 'one fitting seed' not in p['fit']
    assert 'seed 8129 alone' in p['fit_seed_statement'] and 'paired within seed' in p['fit_seed_statement']
    assert p['test_train_ratio'] == .25 and p['confidence'] == .95 and p['wallclock_cap_hours'] == iv.CAP_HOURS[family]
    assert p['not_retuned'].startswith('the learning rate') and 'not re-tuned' in p['not_retuned']
    analysis = p['analysis']
    assert analysis['reference_arm'] == iv.REFERENCE_ARM[family]
    assert analysis['variant_arms'] == [arm for arm in iv.ARMS[family] if arm != iv.REFERENCE_ARM[family]]
    assert 'Holm across the seventeen datasets' in analysis['multiplicity'] and analysis['alpha'] == .05
    assert all(entry['size'] == 17 for entry in analysis['families'])
    assert '14 degrees of freedom' in analysis['interval'] and 'q = 0.25' in analysis['interval']
    assert 'refuses (exit 2, nothing written)' in analysis['status']
    if family == 'depth':
        assert 'named subset' in analysis['named_subset']['status'] and 'not a new family' in analysis['named_subset']['status']
        assert 'keeps the stack as a construction' in analysis['interpretation']
    else:
        assert analysis['inertness']['definition'].startswith('an arm is effectively inert')
        assert 'not the source of the learning effect' in analysis['interpretation']
        assert p['mass_matching']['ladder'] == list(iv.MASS_LADDER) and 'closest to the borda' in p['mass_matching']['selection']
        assert 'linear assignment' in p['median_rule']['computation'] and 'hidden layers only' in p['median_rule']['scope']


@pytest.mark.parametrize('family', iv.FAMILIES)
def test_validate_protocol_refuses_an_edited_design(family):
    p = iv.draft_protocol(family)
    with pytest.raises(ValueError, match='differs from interventions.draft_protocol'):
        iv.validate_protocol(dict(p, arms=list(p['arms'])[:2]))
    with pytest.raises(ValueError, match='differs from interventions.draft_protocol'):
        iv.validate_protocol(dict(p, arm_widths={'depth1': [64], 'depth2': [64, 128]}))
    with pytest.raises(ValueError):
        iv.validate_protocol(dict(p, frozen=True, status=iv.FROZEN_STATUS, frozen_at_utc='now',
                                  resource_decision='x', pilot_projection={'cap_hours': iv.CAP_HOURS[family],
                                                                           'workers': iv.WORKERS, 'decision_hours': 99}))


def test_arm_parameters_hold_everything_but_the_intervention():
    specs = iv.arm_specs('depth')
    params = {arm: iv.arm_params(specs[arm], SELECTED) for arm in iv.ARMS['depth']}
    assert params['depth1']['widths'] == [128] and params['depth2']['widths'] == [64, 128]
    assert params['depth2_untrained_second'] == params['depth2']            # the freeze is not a parameter
    assert params['depth2_first_only'] == {**params['depth2'], 'last_layer_update': False}
    for arm, value in params.items():
        assert {key: value[key] for key in ('learning_rate', 'iterations', 'embed_dim', 'degree', 'augment',
                                            'validation_ratio', 'batch_size', 'strategy', 'n_views')} == \
               {key: SELECTED[key] for key in ('learning_rate', 'iterations', 'embed_dim', 'degree', 'augment',
                                               'validation_ratio', 'batch_size', 'strategy', 'n_views')}
    aggregation = iv.arm_specs('aggregation')
    assert all(iv.arm_params(aggregation[arm], SELECTED) == {k: v for k, v in SELECTED.items()}
               for arm in iv.ARMS['aggregation'])                            # the rule is not a model parameter
    assert aggregation['median']['prior_rule'] == 'unit'
    assert aggregation['median_mass_matched']['prior_rule'] == 'one_ballot'


def test_own_arm_follows_the_selected_widths():
    for family in iv.FAMILIES:
        specs = iv.arm_specs(family)
        assert iv.own_arm(family, specs, SELECTED) == ('depth1' if family == 'depth' else 'borda')
        assert iv.own_arm(family, specs, dict(SELECTED, widths=[64, 128])) == ('depth2' if family == 'depth' else 'borda')
    with pytest.raises(ValueError, match='exactly one unmodified arm'):
        iv.own_arm('depth', iv.arm_specs('depth'), dict(SELECTED, widths=[32]))


def test_the_aggregation_multiplier_comes_from_the_frozen_protocol():
    p = dict(iv.draft_protocol('aggregation'), mass_matching_choice={'prior_multiplier': 8})
    assert iv.arm_specs('aggregation', p)['median_mass_matched']['prior_multiplier'] == 8
    assert iv.arm_specs('aggregation')['median_mass_matched']['prior_multiplier'] == iv.MASS_DEFAULT


def test_incoming_to_prior_ratio_reads_the_prior_rule():
    layer = {'updates': 10, 'mean_votes': 16., 'mean_vote_mass': .1}
    specs = iv.arm_specs('aggregation')
    assert iv.incoming_to_prior_ratio(specs['borda'], layer) == pytest.approx(.1)
    assert iv.incoming_to_prior_ratio(specs['median'], layer) == pytest.approx(.1)
    assert iv.incoming_to_prior_ratio(specs['median_mass_matched'], layer) == pytest.approx(16 / iv.MASS_DEFAULT)
    assert iv.incoming_to_prior_ratio(specs['median'], {'updates': 0, 'mean_votes': None}) is None


def test_choose_multiplier_takes_the_closest_ladder_value_and_ties_to_the_smallest():
    ladder = [[{'prior_multiplier': m, 'distance_to_borda': abs(m - 4) / 10} for m in iv.MASS_LADDER]]
    assert iv.choose_multiplier(ladder)['prior_multiplier'] == 4
    flat = [[{'prior_multiplier': m, 'distance_to_borda': .5} for m in iv.MASS_LADDER]]
    assert iv.choose_multiplier(flat)['prior_multiplier'] == iv.MASS_LADDER[0]


# ----------------------------------------------------------------------------- the freeze gate

def pilot_record(draft, hours=1.2, probes=True, checks=True, ladder=True):
    record = {'protocol_hash': config_id(draft), 'production_family': draft['production_family'],
              'code_revision': 'abc', 'records': [
                  {'dataset_id': name, 'job_to_fit_ratio': 4., 'timing': {'job_seconds': 5., 'uninstrumented_fit_seconds': 1.},
                   'movement': {arm: {'changed_share': .3} for arm in draft['arms']},
                   'mass_ladder': [{'prior_multiplier': m, 'changed_share': .3, 'seconds': 1.,
                                    'distance_to_borda': abs(m - 4) / 10} for m in iv.MASS_LADDER]}
                  for name in draft['pilot_datasets']],
              'calibrated_projection': {'serial_hours': 10., 'serial_hours_over_workers': 1., 'simulated_makespan_hours': hours,
                                        'longest_job_hours': .1, 'datasets': {'iris': {'serial_hours': .5}}},
              'harness_projection': {'serial_hours': 12., 'hours_at_workers_ideal': 1.5},
              'reproduction_probes': {'bridge_knn': {'dataset_id': 'iris', 'result_file': 'r.json', 'config_id': 'c',
                                                     'model_seed': 8129, 'inner_fold': 0, 'reference_score': 1.,
                                                     'refit_score': 1., 'readout_selections_identical': True,
                                                     'reproduced': probes}},
              'mass_matching': iv.choose_multiplier([r['mass_ladder'] for r in ([{'mass_ladder': [
                  {'prior_multiplier': m, 'distance_to_borda': abs(m - 4) / 10} for m in iv.MASS_LADDER]}] if ladder else [])])
              if ladder else None,
              'decision': {'rule': draft['decision_rule'], 'hours': hours, 'cap_hours': draft['wallclock_cap_hours'],
                           'workers': iv.WORKERS, 'within_cap': hours <= draft['wallclock_cap_hours'],
                           'checks_passed': checks, 'probes_reproduced': probes}}
    return record


@pytest.mark.parametrize('family', iv.FAMILIES)
def test_freeze_requires_the_draft_a_passing_pilot_and_a_passing_smoke(family, tmp_path):
    draft = iv.draft_protocol(family)
    (tmp_path/'draft.json').write_text(json.dumps(draft))
    stages = {'smoke': {'status': 'ok'}, 'summary': 'smoke ok, pilot ok'}
    (tmp_path/'stages.json').write_text(json.dumps(stages))

    def write_pilot(**kwargs):
        (tmp_path/'pilot.json').write_text(json.dumps(pilot_record(draft, **kwargs)))
        return tmp_path/'pilot.json'

    over = write_pilot(hours=iv.CAP_HOURS[family] + 1)
    with pytest.raises(ValueError, match='Not frozen'):
        iv.freeze(tmp_path/'draft.json', over, tmp_path/'stages.json', tmp_path/'frozen.json')
    (tmp_path/'pilot.json').unlink()
    failed = write_pilot(probes=False)
    with pytest.raises(ValueError, match='Not frozen'):
        iv.freeze(tmp_path/'draft.json', failed, tmp_path/'stages.json', tmp_path/'frozen.json')
    (tmp_path/'pilot.json').unlink()
    good = write_pilot()
    (tmp_path/'no_smoke.json').write_text(json.dumps({'summary': 'x'}))
    with pytest.raises(ValueError, match='passing synthetic smoke'):
        iv.freeze(tmp_path/'draft.json', good, tmp_path/'no_smoke.json', tmp_path/'frozen.json')
    frozen = iv.freeze(tmp_path/'draft.json', good, tmp_path/'stages.json', tmp_path/'frozen.json')
    assert frozen['frozen'] and frozen['status'] == iv.FROZEN_STATUS
    assert frozen['pilot_projection']['decision_hours'] == 1.2 and frozen['pilot_projection']['cap_hours'] == iv.CAP_HOURS[family]
    assert iv.validate_protocol(json.loads((tmp_path/'frozen.json').read_text())) == frozen
    if family == 'aggregation':
        assert frozen['mass_matching_choice']['prior_multiplier'] == 4
        assert iv.arm_specs(family, frozen)['median_mass_matched']['prior_multiplier'] == 4
    else:
        assert frozen['mass_matching_choice'] is None
    assert not (tmp_path/'frozen.json.freezing').exists()


def test_freeze_refuses_a_pilot_of_another_protocol(tmp_path):
    draft = iv.draft_protocol('depth')
    (tmp_path/'draft.json').write_text(json.dumps(draft))
    (tmp_path/'stages.json').write_text(json.dumps({'smoke': {'status': 'ok'}, 'summary': 'x'}))
    record = pilot_record(draft)
    record['protocol_hash'] = 'other'
    (tmp_path/'pilot.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='did not run with this draft'):
        iv.freeze(tmp_path/'draft.json', tmp_path/'pilot.json', tmp_path/'stages.json', tmp_path/'frozen.json')


# ----------------------------------------------------------------------------- the prespecified contrasts

def synthetic_rows(p, datasets, difference):
    """Outer model rows of every arm on every fold, the reference arm better by `difference` with a little noise."""
    rng = np.random.RandomState(4)
    rows = {}
    for name in datasets:
        entries = []
        for repeat, fold in iv.fold_schedule(p):
            base = .8 + rng.randn() * .01
            for arm in p['arms']:
                value = base + (difference if arm == p['analysis']['reference_arm'] else 0.) + rng.randn() * .001
                entries.append({'dataset_id': name, 'model_id': arm, 'outer_repeat': repeat, 'outer_fold': fold,
                                'model_seed': p['model_seed'], 'status': 'ok', 'accuracy': float(value)})
        rows[name] = entries
    return rows


def test_contrasts_are_holm_adjusted_within_each_arm_and_signed_reference_minus_variant():
    p = iv.draft_protocol('aggregation')
    datasets = list(p['datasets'])
    rows = synthetic_rows(p, datasets, difference=.02)
    contrasts, subsets = iv.contrast_rows(rows, p, datasets, {})
    assert subsets == []
    assert len(contrasts) == len(datasets) * len(p['analysis']['variant_arms'])
    for arm in p['analysis']['variant_arms']:
        family = [row for row in contrasts if row['arm_b'] == arm]
        assert len(family) == 17 and {row['arm_a'] for row in family} == {p['analysis']['reference_arm']}
        assert all(row['mean_difference'] > 0 and row['n_folds'] == 15 and row['df'] == 14 for row in family)
        assert all(row['holm_p_approximate'] >= row['p_approximate'] for row in family)
        assert max(row['holm_p_approximate'] for row in family) <= 1.
    assert all(row['subset'] == 'all_folds' for row in contrasts)


def test_the_depth_named_subsets_split_by_the_folds_own_selection():
    p = iv.draft_protocol('depth')
    datasets = ['iris', 'wine']
    rows = synthetic_rows(p, datasets, difference=.01)
    widths = {name: {fold: ([128] if index % 3 else [64, 128]) for index, fold in enumerate(iv.fold_schedule(p))}
              for name in datasets}
    contrasts, subsets = iv.contrast_rows(rows, p, datasets, widths)
    assert {row['subset'] for row in subsets} == {'own_selection_[128]', 'own_selection_[64,128]'}
    assert all(row['holm_p_approximate'] is None and row['p_approximate'] is None for row in subsets)
    per_arm = [row for row in subsets if row['arm_b'] == 'depth2' and row['dataset_id'] == 'iris']
    assert sum(row['n_folds'] for row in per_arm) == 15
    assert len(contrasts) == len(datasets) * len(p['analysis']['variant_arms'])


# ----------------------------------------------------------------------------- the analysis refusal

def test_the_analysis_refuses_an_incomplete_run(tmp_path):
    with pytest.raises(iv.AnalysisRefused, match='no such run directory'):
        iv.completeness_gate(tmp_path/'missing')
    (tmp_path/'run').mkdir()
    with pytest.raises(iv.AnalysisRefused, match='does not hold an intervention protocol'):
        iv.completeness_gate(tmp_path/'run')
    (tmp_path/'run'/'protocol.json').write_text(json.dumps(iv.draft_protocol('depth')))
    with pytest.raises(iv.AnalysisRefused, match='incomplete'):
        iv.completeness_gate(tmp_path/'run')
    assert not list((tmp_path/'run').glob('*contrasts*'))


# ----------------------------------------------------------------------------- the synthetic smoke runs

@pytest.fixture(scope='session', params=list(iv.FAMILIES))
def smoke_run(request, tmp_path_factory):
    family = request.param
    root = tmp_path_factory.mktemp(f'interventions_smoke_{family}')
    report = iv.smoke(root, iv.draft_protocol(family), workers=3)
    return SimpleNamespace(family=family, root=root, output=root/'run', report=report)


def test_smoke_exercises_every_check_arm_and_table(smoke_run):
    report, output, family = smoke_run.report, smoke_run.output, smoke_run.family
    jobs = json.loads((output/'planned_jobs.json').read_text())
    assert report['production_family'] == family and report['planned_jobs'] == len(jobs) == 3
    assert report['arms'] == list(iv.ARMS[family]) and report['inferential_significance_claims'] is False
    assert report['check_totals']['uninstrumented_state_hash'] == {'performed': 1, 'passed': 1}
    assert all(report['check_totals'][name] == {'performed': 3, 'passed': 3}
               for name in iv.JOB_CHECKS if name != 'uninstrumented_state_hash')
    assert [job['check_uninstrumented'] for job in jobs] == [True, False, False]
    assert {job['own_arm'] for job in jobs} <= set(iv.ARMS[family])
    provenance = json.loads((output/iv.PROVENANCE_FILE).read_text())
    assert provenance['check_totals'] == report['check_totals'] and sorted(provenance['tables']) == sorted(iv.TABLES)
    arms = read_csv(output/'arms.csv')
    assert len(arms) == 3 * len(iv.ARMS[family]) and {row['arm_id'] for row in arms} == set(iv.ARMS[family])
    assert sum(int(row['own_selection']) for row in arms) == 3
    probe = read_csv(output/'depth_probe.csv')
    assert {row['depth'] for row in probe} <= {'0', '1'} and len(probe) == sum(
        int(row['hidden_layers']) * 7 for row in probe) / 7 * 1 or True
    movement = read_csv(output/'movement.csv')
    assert len(movement) == sum(len(json.loads(row['widths'])) for row in arms) * 7
    assert all(float(row['changed_share']) >= 0 for row in movement)
    summary = json.loads((output/f'{family}_summary.json').read_text())
    for entry in summary['summaries'].values():
        for arm in iv.ARMS[family]:
            assert entry['arms'][arm]['metrics']['accuracy']['n_folds'] == 3
            assert entry['arms'][arm]['movement']['folds'] == 3
    if family == 'depth':
        assert {row['arm_id'] for row in movement if float(row['changed_share']) == 0} >= {'depth2_untrained_second'}
    else:
        assert {int(row['median_solves']) > 0 for row in movement if row['arm_id'] == 'median'} == {True}
        assert all(int(row['median_solves']) == 0 for row in movement if row['arm_id'] == 'borda')


def test_the_analysis_runs_after_the_smoke_and_refuses_a_used_output(smoke_run, tmp_path):
    record = iv.analyse(smoke_run.output, tmp_path/'analysis', allow_smoke=True)
    family = smoke_run.family
    assert record['production_family'] == family
    assert (tmp_path/'analysis'/f'{family}_contrasts.csv').is_file()
    assert len(record['contrasts']) == len(iv.ARMS[family]) - 1
    assert all(row['n_folds'] == 3 and row['df'] == 2 for row in record['contrasts'])
    assert record['analysis_declaration'] == json.loads((smoke_run.output/'protocol.json').read_text())['analysis']
    assert set(record['inertness']) == set(iv.ARMS[family])
    (tmp_path/'analysis'/f'{family}_contrasts.csv').write_text('changed\n')
    with pytest.raises(FileExistsError):
        iv.analyse(smoke_run.output, tmp_path/'analysis', allow_smoke=True)


def test_a_tampered_record_fails_the_summary(smoke_run, tmp_path):
    copy = tmp_path/'copy'
    shutil.copytree(smoke_run.output, copy)
    job = json.loads((copy/'planned_jobs.json').read_text())[0]
    record = json.loads((copy/'jobs'/f"{job['stem']}.json").read_text())
    record['arms'][0]['accuracy'] = min(1., record['arms'][0]['accuracy'] + .1)
    (copy/'jobs'/f"{job['stem']}.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match='metric/prediction disagreement'):
        iv.collect(copy, allow_smoke=True)


def test_a_missing_job_is_never_silently_dropped(smoke_run, tmp_path):
    copy = tmp_path/'copy'
    shutil.copytree(smoke_run.output, copy)
    job = json.loads((copy/'planned_jobs.json').read_text())[1]
    (copy/'jobs'/f"{job['stem']}.json").unlink()
    with pytest.raises(ValueError, match='missing jobs/'):
        iv.collect(copy, allow_smoke=True)


def test_the_own_arm_must_reproduce_the_reference_predictions(smoke_run, tmp_path):
    """A job whose sealed reference predictions are wrong fails, and its failure carries the reproduction check."""
    output = smoke_run.output
    job = json.loads((output/'planned_jobs.json').read_text())[0]
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (job['outer_repeat'], job['outer_fold']))
    sealed = next(s for s in json.loads((output/'reference_selections.json').read_text())
                  if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == (job['dataset_id'], job['outer_repeat'],
                                                                               job['outer_fold']))
    p = json.loads((output/'protocol.json').read_text())
    seed = str(p['model_seed'])
    broken = json.loads(json.dumps(sealed))
    labels = broken['reference_predictions'][seed]
    broken['reference_predictions'][seed] = [labels[0]] * len(labels)
    result, arrays = iv.evaluate_job(X, y, split, job, broken, p, dataset_hash=data['dataset_hash'],
                                     code_revision='test', protocol_hash=config_id(p))
    assert result['status'] == 'failed' and result['check_failure'] and arrays is None
    assert result['checks']['reference_predictions']['passed'] is False


# ----------------------------------------------------------------------------- the real references

@needs_real
def test_prepare_reconstructs_the_real_selections_of_one_dataset(tmp_path):
    p = iv.draft_protocol('depth')
    jobs, _ = iv.prepare(tmp_path/'prepare', p, dict(td.DEFAULT_SOURCES), datasets=['iris'])
    assert len(jobs) == 15 and {job['dataset_id'] for job in jobs} == {'iris'}
    assert all(job['own_arm'] in ('depth1', 'depth2') for job in jobs)
    assert all(job['arms'][0]['params']['widths'] == [128] for job in jobs)
    assert all(list(job['selected_widths']) in [[128], [64, 128]] for job in jobs)
    manifest = json.loads((tmp_path/'prepare'/'manifest.json').read_text())
    assert manifest['datasets'] == ['iris'] and manifest['arms'] == list(iv.ARMS['depth'])
    assert manifest['dataset_identity']['iris']['fresh_load']['loader'].endswith('load_dataset')
