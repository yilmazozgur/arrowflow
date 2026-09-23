"""The signed-relay depth family (signed_relay_depth): the protocol and its prespecified analysis and interpretation, the
arms, the scale rule, the freeze gate, the analysis refusal, the reuse comparator, and a synthetic smoke run end to end
over a synthetic depth run (built once per session; tests that alter records work on copies). The real depth run and the
real reference runs are used where they are on this machine; no real job is fitted."""
import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.make_revision import interventions as iv
from experiments.make_revision import signed_relay_depth as srd
from experiments.make_revision import training_diagnostics as td
from experiments.make_revision.evaluation import config_id
from experiments.make_revision.run_revision import load_prepared

REAL = {name: tuple(Path(d) for d in directories) for name, directories in td.DEFAULT_SOURCES.items()}
needs_real = pytest.mark.skipif(not (all((run/'summary.json').is_file() and (ablation/'reference_selections.json').is_file()
                                         for run, ablation in REAL.values())
                                     and (srd.DEFAULT_SOURCE_DEPTH_RUN/'provenance.json').is_file()),
                                reason='the reference runs and the depth run are not on this machine')
SELECTED = {'n_views': 7, 'strategy': 'diverse', 'iterations': 200, 'batch_size': 32, 'validation_ratio': .1,
            'aggregation': 'majority', 'widths': [128], 'learning_rate': .1, 'embed_dim': 32, 'degree': 2, 'augment': True}


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


# ----------------------------------------------------------------------------- protocol and arms

def test_the_committed_protocol_is_the_draft_or_its_freeze():
    committed = json.loads(srd.PROTOCOL.read_text())
    srd.validate_protocol(committed)
    draft = srd.draft_protocol()
    assert {k: v for k, v in committed.items() if k not in srd.FREEZE_FIELDS} == \
           {k: v for k, v in draft.items() if k not in srd.FREEZE_FIELDS}
    assert committed['protocol_id'] == srd.PROTOCOL_ID and len(committed['datasets']) == 17
    assert committed['datasets'] == iv.draft_protocol('depth')['datasets']
    assert committed['source_depth_run'] == srd.SOURCE_DEPTH_RUN


def test_the_draft_declares_the_ruled_design():
    p = srd.draft_protocol()
    assert p['arms'] == ['depth1', 'depth2_printed', 'depth2_signed', 'depth2_signed_scaled']
    assert p['reference_arm'] == 'depth1' and p['reused_arms'] == {'depth1': 'depth1', 'depth2_printed': 'depth2'}
    assert p['fitted_arms'] == ['depth2_signed', 'depth2_signed_scaled'] and p['arm_widths'] == iv.ARM_WIDTHS
    assert p['model_seed'] == 8129 and p['test_train_ratio'] == .25 and p['confidence'] == .95
    assert p['wallclock_cap_hours'] == 4. and p['workers'] == 16 and p['max_workers'] == 16
    relay = p['relay']
    assert 'sign(a_j) m(r_j -> pi)' in relay['corrected'] and '2p - (V - 1)' in relay['corrected']
    assert relay['weighting'].startswith('not |a|-weighted') and 'unchanged by construction' in relay['scope']
    scale = p['vote_scale']
    assert scale['ladder'] == [1, 2, 4, 8, 16, 32] and scale['target'] == .5
    assert 'smallest ladder value' in scale['rule'] and 'no accuracy' in scale['rule']
    analysis = p['analysis']
    assert analysis['primary']['variant_arms'] == ['depth2_signed', 'depth2_signed_scaled']
    assert analysis['secondary'] == {**analysis['secondary'], 'arm_a': 'depth2_signed', 'arm_b': 'depth2_printed'}
    assert '14 degrees of freedom' in analysis['primary']['interval'] and 'q = 0.25' in analysis['primary']['interval']
    assert 'Holm across the seventeen datasets within each variant arm' in analysis['primary']['multiplicity']
    assert [family['size'] for family in analysis['families']] == [17, 17, 17] and analysis['alpha'] == .05
    assert 'at least 9 of the seventeen' in analysis['interpretation']['rule']
    assert 'rerun the main benchmark' in analysis['interpretation']['helps']
    assert 'did not help at a matched configuration' in analysis['interpretation']['does_not_help']
    assert 'refuses (exit 2, nothing written)' in analysis['status'] and 'reused, not re-estimated' in analysis['status']
    assert p['scale_choice'] is None and not p['frozen']


def test_validate_protocol_refuses_an_edited_design_and_an_incomplete_freeze():
    p = srd.draft_protocol()
    with pytest.raises(ValueError, match='differs from signed_relay_depth.draft_protocol'):
        srd.validate_protocol(dict(p, arms=p['arms'][:3]))
    with pytest.raises(ValueError, match='differs from signed_relay_depth.draft_protocol'):
        srd.validate_protocol(dict(p, vote_scale={**p['vote_scale'], 'target': .25}))
    with pytest.raises(ValueError, match='not a signed_relay_depth protocol'):
        srd.validate_protocol(iv.draft_protocol('depth'))
    frozen = dict(p, frozen=True, status=srd.FROZEN_STATUS, frozen_at_utc='now', resource_decision='x',
                  pilot_projection={'cap_hours': srd.CAP_HOURS, 'workers': srd.WORKERS, 'decision_hours': 2.})
    with pytest.raises(ValueError, match='vote scale'):
        srd.validate_protocol(dict(frozen, scale_choice=None))
    with pytest.raises(ValueError, match='vote scale'):
        srd.validate_protocol(dict(frozen, scale_choice={'lower_vote_scale': 3}))
    with pytest.raises(ValueError, match='within the 4.0 h cap'):
        srd.validate_protocol(dict(frozen, scale_choice={'lower_vote_scale': 8},
                                   pilot_projection={'cap_hours': srd.CAP_HOURS, 'workers': srd.WORKERS, 'decision_hours': 5.}))
    assert srd.validate_protocol(dict(frozen, scale_choice={'lower_vote_scale': 16}))['frozen']


def test_arm_parameters_hold_everything_but_the_widths():
    specs = srd.arm_specs()
    params = {arm: iv.arm_params(specs[arm], SELECTED) for arm in srd.ARMS}
    assert params['depth1']['widths'] == [128]
    assert params['depth2_printed'] == params['depth2_signed'] == params['depth2_signed_scaled']
    assert params['depth2_printed']['widths'] == [64, 128]
    assert {k: v for k, v in params['depth1'].items() if k != 'widths'} == \
           {k: v for k, v in params['depth2_printed'].items() if k != 'widths'} == {k: v for k, v in SELECTED.items() if k != 'widths'}
    assert [specs[arm]['relay'] for arm in srd.ARMS] == ['signed', 'printed', 'signed', 'signed']
    assert [specs[arm]['lower_vote_scale'] for arm in srd.ARMS] == [1, 1, 1, srd.SCALE_PILOT]
    frozen = dict(srd.draft_protocol(), scale_choice={'lower_vote_scale': 16})
    assert srd.arm_specs(frozen)['depth2_signed_scaled']['lower_vote_scale'] == 16
    assert srd.own_arm(SELECTED) == 'depth1' and srd.own_arm(dict(SELECTED, widths=[64, 128])) == 'depth2_printed'
    assert srd.own_arm({'widths': [6]}, iv.SMOKE_WIDTHS) == 'depth1'
    with pytest.raises(ValueError, match='not one of the declared depth arms'):
        srd.own_arm(dict(SELECTED, widths=[32]))


# ----------------------------------------------------------------------------- the scale rule

def ladder(shares):
    return [{'lower_vote_scale': scale, 'cleared_share': share} for scale, share in zip(srd.SCALE_LADDER, shares)]


def test_the_scale_is_the_smallest_value_reaching_the_target_on_every_pilot_dataset():
    choice = srd.choose_scale({'iris': ladder([.02, .05, .2, .45, .71, .9]), 'ionosphere': ladder([.7, .8, .9, .95, .97, .99])})
    assert choice['lower_vote_scale'] == 16 and choice['reached_target']
    assert choice['cleared_share']['iris']['16'] == .71
    assert srd.choose_scale({'a': ladder([.5, .6, .7, .8, .9, 1.])})['lower_vote_scale'] == 1   # the target is inclusive
    none = srd.choose_scale({'a': ladder([.1, .2, .3, .4, .45, .49]), 'b': ladder([.9] * 6)})
    assert none['lower_vote_scale'] == 32 and not none['reached_target']
    unvoted = srd.choose_scale({'a': ladder([None, None, .6, .7, .8, .9])})
    assert unvoted['lower_vote_scale'] == 4


# ----------------------------------------------------------------------------- the prespecified analysis

def synthetic_rows(p, datasets, gains):
    """Outer model rows of every arm on every fold; gains[arm][dataset] is added to the arm's accuracy."""
    rng = np.random.RandomState(4)
    rows = {}
    for name in datasets:
        entries = []
        for repeat, fold in iv.fold_schedule(p):
            base = .8 + rng.randn() * .01
            for arm in p['arms']:
                value = base + gains.get(arm, {}).get(name, 0.) + rng.randn() * .002
                entries.append({'dataset_id': name, 'model_id': arm, 'outer_repeat': repeat, 'outer_fold': fold,
                                'model_seed': p['model_seed'], 'status': 'ok', 'accuracy': float(value)})
        rows[name] = entries
    return rows


def test_the_contrasts_are_three_holm_families_signed_as_declared():
    p = srd.draft_protocol()
    datasets = list(p['datasets'])
    rows = synthetic_rows(p, datasets, {'depth2_signed': {name: .03 for name in datasets}})
    widths = {name: {fold: ([128] if index % 3 else [64, 128]) for index, fold in enumerate(iv.fold_schedule(p))}
              for name in datasets}
    contrasts, subsets = srd.contrast_rows(rows, p, datasets, widths)
    assert len(contrasts) == 3 * 17
    labels = {row['contrast'] for row in contrasts}
    assert labels == {'depth1_minus_depth2_signed', 'depth1_minus_depth2_signed_scaled', 'depth2_signed_minus_depth2_printed'}
    signed = [row for row in contrasts if row['contrast'] == 'depth1_minus_depth2_signed']
    assert all(row['mean_difference'] < 0 and row['role'] == 'primary' and row['df'] == 14 for row in signed)
    secondary = [row for row in contrasts if row['role'] == 'secondary']
    assert all(row['mean_difference'] > 0 for row in secondary) and len(secondary) == 17
    for label in labels:
        family = [row for row in contrasts if row['contrast'] == label]
        assert all(row['holm_p_approximate'] >= row['p_approximate'] for row in family)
    assert {row['subset'] for row in subsets} == {'own_selection_[128]', 'own_selection_[64,128]'}
    assert all(row['p_approximate'] is None and row['holm_p_approximate'] is None for row in subsets)


def outcome_for(gains):
    p = srd.draft_protocol()
    datasets = list(p['datasets'])
    rows = synthetic_rows(p, datasets, gains)
    contrasts, _ = srd.contrast_rows(rows, p, datasets, {name: {fold: [128] for fold in iv.fold_schedule(p)}
                                                          for name in datasets})
    return srd.interpretation_outcome(contrasts, p)


def test_the_interpretation_rule_needs_a_holm_gain_and_a_higher_mean_on_most_datasets():
    datasets = srd.draft_protocol()['datasets']
    large_on_nine = {name: (.05 if index < 9 else -.01) for index, name in enumerate(datasets)}
    helps = outcome_for({'depth2_signed_scaled': large_on_nine})
    assert helps['outcome'] == 'helps' and helps['arms']['depth2_signed_scaled']['helps']
    assert helps['arms']['depth2_signed_scaled']['n_higher_mean'] == 9 and not helps['arms']['depth2_signed']['helps']
    assert 'rerun the main benchmark' in helps['statement']
    large_on_eight = {name: (.05 if index < 8 else -.01) for index, name in enumerate(datasets)}
    eight = outcome_for({'depth2_signed': large_on_eight})
    assert eight['outcome'] == 'does_not_help' and eight['arms']['depth2_signed']['n_higher_mean'] == 8
    tiny_everywhere = {name: .0003 for name in datasets}
    tiny = outcome_for({'depth2_signed': tiny_everywhere})
    assert tiny['arms']['depth2_signed']['holm_significant_gains'] == [] and tiny['outcome'] == 'does_not_help'
    losses = outcome_for({'depth2_signed': {name: -.05 for name in datasets}})
    assert losses['arms']['depth2_signed']['holm_significant_losses'] and losses['outcome'] == 'does_not_help'
    assert 'did not help at a matched configuration' in losses['statement']


# ----------------------------------------------------------------------------- the reuse comparator

def test_the_reuse_comparator_names_every_field_that_differs():
    stored = {'state_hashes': ['a', 'b'], 'params': {'x': 1}, 'widths': [64, 128], 'hidden_layers': 2,
              'prediction_hash': 'h', 'neighborhood_purity': .5, 'majority_by_depth': [{'depth': 0, 'knn_accuracy': .5}],
              'accuracy': .5, 'error': .5, 'balanced_accuracy': .5, 'macro_f1': .5,
              'movement': {'changed_share': .1, 'layers': [{'layer': 0, 'changed_share': .1}]},
              'views': [{'view': 0, 'strategy': 's', 'view_seed': 1, 'readout': {'k': 3}, 'depths': [{'depth': 0}],
                         'movement': {'changed_share': .1, 'layers': [{'layer': 0, 'changed_share': .1}]}}]}
    arrays = {'view_predictions': np.array([[1, 2]]), 'majority_by_depth': np.array([[1, 2]]), 'predictions': np.array([1, 2]),
              'v0__l0__changed': np.array([.1, .2]), 'v0__l0__displacement': np.array([.0, .1])}
    refit = json.loads(json.dumps(stored))
    refit['movement']['cleared_share'] = .9                        # a refit carries more keys; that is not a difference
    refit['views'][0]['movement']['layers'][0]['voted_updates'] = 4
    assert srd.reproduction_differences(stored, arrays, refit, dict(arrays), outer=True) == []
    changed = json.loads(json.dumps(refit))
    changed['state_hashes'][1] = 'c'
    changed['views'][0]['movement']['layers'][0]['changed_share'] = .2
    changed['views'][0]['readout'] = {'k': 5}
    changed_arrays = dict(arrays, predictions=np.array([2, 2]), v0__l0__changed=np.array([.1, .3]))
    differences = srd.reproduction_differences(stored, arrays, changed, changed_arrays, outer=True)
    assert {'state_hashes', 'views[0].movement', 'views[0].readout', 'arrays.predictions',
            'arrays.v0__l0__changed'} <= set(differences)
    training_only = srd.reproduction_differences(stored, arrays, changed, changed_arrays, outer=False)
    assert 'views[0].readout' not in training_only and 'arrays.predictions' not in training_only
    assert 'state_hashes' in training_only and 'arrays.v0__l0__changed' in training_only


# ----------------------------------------------------------------------------- the freeze gate

def pilot_record(draft, hours=1.5, probes=True, checks=True, reuse=True, shares=None):
    shares = shares or {'iris': [.02, .05, .2, .45, .71, .9], 'ionosphere': [.7, .8, .9, .95, .97, .99]}
    ladders = {name: ladder(values) for name, values in shares.items()}
    return {'protocol_hash': config_id(draft), 'production_family': srd.FAMILY, 'code_revision': 'abc',
            'records': [{'dataset_id': name, 'fitted_ratio': 5., 'reuse_ratio': 4., 'timing': {'job_seconds': 5.},
                         'movement': {}, 'reuse_reproduction': [], 'scale_ladder': ladders[name]} for name in ladders],
            'calibrated_projection': {'serial_hours': 10., 'serial_hours_over_workers': 1., 'simulated_makespan_hours': hours,
                                      'longest_job_hours': .1, 'datasets': {'iris': {'serial_hours': .5}}},
            'reproduction_probes': {'bridge_knn': {'dataset_id': 'iris', 'result_file': 'r.json', 'config_id': 'c',
                                                   'model_seed': 8129, 'inner_fold': 0, 'reference_score': 1.,
                                                   'refit_score': 1., 'readout_selections_identical': True,
                                                   'reproduced': probes}},
            'scale_choice': srd.choose_scale(ladders),
            'decision': {'rule': draft['decision_rule'], 'hours': hours, 'cap_hours': draft['wallclock_cap_hours'],
                         'workers': srd.WORKERS, 'within_cap': hours <= draft['wallclock_cap_hours'],
                         'checks_passed': checks, 'reuse_reproduced_on_training_quantities': reuse,
                         'probes_reproduced': probes}}


def test_freeze_requires_the_draft_a_passing_pilot_and_a_passing_smoke_and_seals_the_scale(tmp_path):
    draft = srd.draft_protocol()
    (tmp_path/'draft.json').write_text(json.dumps(draft))
    (tmp_path/'stages.json').write_text(json.dumps({'smoke': {'status': 'ok'}, 'summary': 'smoke ok, pilot ok'}))

    def attempt(record, stages='stages.json'):
        (tmp_path/'pilot.json').write_text(json.dumps(record))
        return srd.freeze(tmp_path/'draft.json', tmp_path/'pilot.json', tmp_path/stages, tmp_path/'frozen.json')

    for broken in (dict(hours=srd.CAP_HOURS + 1), dict(probes=False), dict(checks=False), dict(reuse=False)):
        with pytest.raises(ValueError, match='Not frozen'):
            attempt(pilot_record(draft, **broken))
    (tmp_path/'no_smoke.json').write_text(json.dumps({'summary': 'x'}))
    with pytest.raises(ValueError, match='passing synthetic smoke'):
        attempt(pilot_record(draft), 'no_smoke.json')
    tampered = pilot_record(draft)
    tampered['scale_choice'] = dict(tampered['scale_choice'], lower_vote_scale=4)
    with pytest.raises(ValueError, match='does not follow from its ladders'):
        attempt(tampered)
    other = pilot_record(draft)
    other['protocol_hash'] = 'other'
    with pytest.raises(ValueError, match='did not run with this draft'):
        attempt(other)
    frozen = attempt(pilot_record(draft))
    assert frozen['frozen'] and frozen['status'] == srd.FROZEN_STATUS
    assert frozen['scale_choice']['lower_vote_scale'] == 16 and frozen['scale_choice']['ladders']['iris'][4]['cleared_share'] == .71
    assert srd.arm_specs(frozen)['depth2_signed_scaled']['lower_vote_scale'] == 16
    assert srd.validate_protocol(json.loads((tmp_path/'frozen.json').read_text())) == frozen
    assert 's = 16 chosen on the training-only pilot ladder' in frozen['resource_decision']
    assert not (tmp_path/'frozen.json.freezing').exists()


# ----------------------------------------------------------------------------- the analysis refusal

def test_the_analysis_refuses_an_incomplete_run(tmp_path):
    with pytest.raises(srd.AnalysisRefused, match='no such run directory'):
        srd.completeness_gate(tmp_path/'missing')
    (tmp_path/'run').mkdir()
    with pytest.raises(srd.AnalysisRefused, match='does not hold a signed_relay_depth protocol'):
        srd.completeness_gate(tmp_path/'run')
    (tmp_path/'run'/'protocol.json').write_text(json.dumps(srd.draft_protocol()))
    with pytest.raises(srd.AnalysisRefused, match='incomplete'):
        srd.completeness_gate(tmp_path/'run')
    assert not list((tmp_path/'run').glob('*contrasts*'))
    with pytest.raises(SystemExit) as refused:
        srd.main(['analyse', '--run', str(tmp_path/'run'), '--output', str(tmp_path/'analysis')])
    assert refused.value.code == 2 and not (tmp_path/'analysis').exists()


# ----------------------------------------------------------------------------- the synthetic smoke run

@pytest.fixture(scope='session')
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('signed_relay_depth_smoke')
    report = srd.smoke(root, srd.draft_protocol(), workers=3)
    return SimpleNamespace(root=root, output=root/'run', source=root/'depth'/'run', report=report)


def test_smoke_exercises_every_check_arm_and_table(smoke_run):
    report, output = smoke_run.report, smoke_run.output
    jobs = json.loads((output/'planned_jobs.json').read_text())
    assert report['production_family'] == srd.FAMILY and report['planned_jobs'] == len(jobs) == 3
    assert report['arms'] == list(srd.ARMS) and report['inferential_significance_claims'] is False
    assert report['check_totals']['reused_arms_reproduced'] == {'performed': 1, 'passed': 1}
    assert all(report['check_totals'][name] == {'performed': 3, 'passed': 3}
               for name in srd.JOB_CHECKS if name != 'reused_arms_reproduced')
    assert [job['reuse_check'] for job in jobs] == [True, False, False]
    provenance = json.loads((output/srd.PROVENANCE_FILE).read_text())
    assert provenance['check_totals'] == report['check_totals'] and sorted(provenance['tables']) == sorted(srd.TABLES)
    arms = read_csv(output/'arms.csv')
    assert len(arms) == 12 and [row['reused'] for row in arms[:4]] == ['1', '1', '0', '0']
    assert sum(int(row['own_selection']) for row in arms) == 3
    assert {row['lower_vote_scale'] for row in arms if row['arm_id'] == 'depth2_signed_scaled'} == {str(srd.SCALE_PILOT)}
    reproduction = read_csv(output/'reproduction.csv')
    assert [(row['arm_id'], row['refit_relay'], row['refit_reproduced'], row['plain_state_hashes_equal'])
            for row in reproduction] == [('depth1', 'signed', '1', '1'), ('depth2_printed', 'printed', '1', '1')]
    relay = read_csv(output/'relay.csv')
    assert {row['arm_id'] for row in relay} == set(srd.FITTED) and len(relay) == 3 * 2 * 7 * 2
    upper = [row for row in relay if row['layer'] == '1']
    assert all(row['relays_to_layer_below'] == 'True' and row['returned_motions_corrected'] == row['repulsions'] for row in upper)
    assert sum(int(row['repulsions']) for row in upper) > 0
    scaled = [row for row in relay if row['arm_id'] == 'depth2_signed_scaled' and row['layer'] == '0']
    assert all(float(row['scaled_vote_mass']) == pytest.approx(srd.SCALE_PILOT * float(row['unscaled_vote_mass'])) for row in scaled)
    movement = read_csv(output/'movement.csv')
    fitted = [row for row in movement if row['arm_id'] in srd.FITTED]
    assert all(row['movement_threshold'] != '' for row in fitted)
    assert all(row['movement_threshold'] == '' for row in movement if row['arm_id'] not in srd.FITTED)


def test_the_reused_arms_are_the_source_records_verbatim(smoke_run):
    output, source = smoke_run.output, smoke_run.source
    for job in json.loads((output/'planned_jobs.json').read_text()):
        mine = json.loads((output/'jobs'/f"{job['stem']}.json").read_text())
        theirs = json.loads((source/'jobs'/f"{job['stem']}.json").read_text())
        for arm, name in srd.REUSED.items():
            a = next(entry for entry in mine['arms'] if entry['arm_id'] == arm)
            b = next(entry for entry in theirs['arms'] if entry['arm_id'] == name)
            assert {k: v for k, v in a.items() if k not in ('arm_id', 'spec', 'reused', 'source')} == \
                   {k: v for k, v in b.items() if k not in ('arm_id', 'spec')}
            assert a['source']['arm_id'] == name and a['source']['spec'] == b['spec'] and a['reused'] is True
        with np.load(output/'artifacts'/f"{job['stem']}.npz") as ours, np.load(source/'artifacts'/f"{job['stem']}.npz") as src:
            for arm, name in srd.REUSED.items():
                assert np.array_equal(ours[f'{arm}__predictions'], src[f'{name}__predictions'])


def test_the_analysis_runs_after_the_smoke_and_refuses_a_used_output(smoke_run, tmp_path):
    record = srd.analyse(smoke_run.output, tmp_path/'analysis', allow_smoke=True)
    assert record['production_family'] == srd.FAMILY and len(record['contrasts']) == 3
    assert {row['role'] for row in record['contrasts']} == {'primary', 'secondary'}
    assert all(row['n_folds'] == 3 and row['df'] == 2 for row in record['contrasts'])
    assert record['interpretation']['outcome'] in ('helps', 'does_not_help')
    assert record['analysis_declaration'] == json.loads((smoke_run.output/'protocol.json').read_text())['analysis']
    smoke_analysis = json.loads((smoke_run.root/'analysis'/f'{srd.FAMILY}_analysis.json').read_text())
    assert smoke_analysis['contrasts'] == record['contrasts']
    (tmp_path/'analysis'/f'{srd.FAMILY}_contrasts.csv').write_text('changed\n')
    with pytest.raises(FileExistsError):
        srd.analyse(smoke_run.output, tmp_path/'analysis', allow_smoke=True)


def copy_run(smoke_run, tmp_path):
    """A copy of the smoke run whose manifest points at a copy of the synthetic depth run."""
    root = tmp_path/'copy'
    shutil.copytree(smoke_run.root/'depth', root/'depth')
    shutil.copytree(smoke_run.output, root/'run')
    manifest_path = root/'run'/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source_depth_run']['directory'] = str((root/'depth'/'run').resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return root/'run', root/'depth'/'run'


def test_a_tampered_record_or_source_fails_the_verification(smoke_run, tmp_path):
    output, source = copy_run(smoke_run, tmp_path)
    srd.collect(output, allow_smoke=True)
    job = json.loads((output/'planned_jobs.json').read_text())[0]
    path = output/'jobs'/f"{job['stem']}.json"
    original = path.read_text()
    record = json.loads(original)
    record['arms'][2]['accuracy'] = min(1., record['arms'][2]['accuracy'] + .1)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='metric/prediction disagreement'):
        srd.collect(output, allow_smoke=True)
    record = json.loads(original)
    record['arms'][0]['neighborhood_purity'] += 1e-3                   # a reused arm no longer equals its source
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='depth1 differs from the source depth record|neighbourhood purity'):
        srd.collect(output, allow_smoke=True, rederive=True)
    path.write_text(original)
    source_record = source/'jobs'/f"{job['stem']}.json"
    source_record.write_text(source_record.read_text().replace('"status": "ok"', '"status": "ok" '))
    with pytest.raises(ValueError):
        srd.collect(output, allow_smoke=True, rederive=True)


def test_a_missing_job_is_never_silently_dropped(smoke_run, tmp_path):
    output, _ = copy_run(smoke_run, tmp_path)
    job = json.loads((output/'planned_jobs.json').read_text())[1]
    (output/'jobs'/f"{job['stem']}.json").unlink()
    with pytest.raises(ValueError, match='missing jobs/'):
        srd.collect(output, allow_smoke=True)


def test_the_own_arm_must_carry_the_reference_predictions(smoke_run):
    output = smoke_run.output
    job = json.loads((output/'planned_jobs.json').read_text())[1]
    X, y, data, splits = load_prepared(output, job['dataset_id'])
    split = next(s for s in splits if (s['outer_repeat'], s['outer_fold']) == (job['outer_repeat'], job['outer_fold']))
    sealed = next(s for s in json.loads((output/'reference_selections.json').read_text())
                  if (s['dataset_id'], s['outer_repeat'], s['outer_fold']) == (job['dataset_id'], job['outer_repeat'],
                                                                               job['outer_fold']))
    p = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    source = srd.load_source_job(manifest['source_depth_run']['directory'], job)
    broken = json.loads(json.dumps(sealed))
    labels = broken['reference_predictions'][str(p['model_seed'])]
    broken['reference_predictions'][str(p['model_seed'])] = [labels[0]] * len(labels)
    result, arrays = srd.evaluate_job(X, y, split, job, broken, source, p, dataset_hash=data['dataset_hash'],
                                      code_revision='test', protocol_hash=config_id(p))
    assert result['status'] == 'failed' and result['check_failure'] and arrays is None
    assert result['checks']['reference_predictions']['passed'] is False


def test_prepare_refuses_a_source_run_that_differs_from_its_pins(smoke_run, tmp_path):
    _, source = copy_run(smoke_run, tmp_path)
    p = json.loads((smoke_run.output/'protocol.json').read_text())
    sources = {'smoke_reference': (smoke_run.root/'depth'/'synthetic_reference', smoke_run.root/'depth'/'synthetic_ablation')}
    (source/'depth_summary.json').write_text((source/'depth_summary.json').read_text() + ' ')
    with pytest.raises(ValueError, match='does not match the protocol pins'):
        srd.prepare(tmp_path/'prepare', p, sources, source, allow_smoke=True, purpose='synthetic_smoke_only')
    assert not (tmp_path/'prepare'/'planned_jobs.json').exists()


# ----------------------------------------------------------------------------- the real references and depth run

@needs_real
def test_prepare_seals_the_real_plan_of_one_dataset_against_the_depth_run(tmp_path):
    p = srd.draft_protocol()
    jobs, _, source = srd.prepare(tmp_path/'prepare', p, dict(td.DEFAULT_SOURCES), srd.DEFAULT_SOURCE_DEPTH_RUN,
                                  datasets=['iris'])
    assert len(jobs) == 15 and [job['reuse_check'] for job in jobs] == [True] + [False] * 14
    depth_jobs = {job['stem']: job for job in json.loads((srd.DEFAULT_SOURCE_DEPTH_RUN/'planned_jobs.json').read_text())}
    for job in jobs:
        theirs = depth_jobs[job['stem']]
        assert job['selected'] == theirs['selected'] and job['config_id'] == theirs['config_id']
        assert srd.REUSED[job['own_arm']] == theirs['own_arm']
        params = {entry['arm_id']: entry['params'] for entry in job['arms']}
        source_params = {entry['arm_id']: entry['params'] for entry in theirs['arms']}
        assert params['depth1'] == source_params['depth1']
        assert params['depth2_printed'] == params['depth2_signed'] == params['depth2_signed_scaled'] == source_params['depth2']
        assert job['source']['record_sha256'] == source['provenance']['jobs'][job['stem']]['record_sha256']
    manifest = json.loads((tmp_path/'prepare'/'manifest.json').read_text())
    assert manifest['source_depth_run']['pins'] == {k: v for k, v in srd.SOURCE_DEPTH_RUN.items() if k != 'protocol_file'}
