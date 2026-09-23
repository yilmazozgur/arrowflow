"""G4 follow-up: extra_runs, extra_ablation and compare_extra (the family protocols, registry, stages, projection and freeze;
one synthetic smoke per family through the run, reporting, component ablation and analysis; the refusals). The real
artificial data are never loaded or fitted here: the artificial family uses a stand-in loader and generator."""
import hashlib
import heapq
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.make_revision import compare_extra as ce
from experiments.make_revision import evaluation
from experiments.make_revision import extra_ablation as ea
from experiments.make_revision import extra_data as ed
from experiments.make_revision import extra_runs as er
from experiments.make_revision import holistic
from experiments.make_revision import newdata as nd
from experiments.make_revision import run_knn_ablation as base
from experiments.make_revision.compare_runs import RunComparisonError, load_run, verify_run
from experiments.make_revision.evaluation import ModelSpec, canonical_json, config_id, dataset_fingerprint, holm_adjust, paired_corrected_interval
from experiments.make_revision.knn_controls import CANDIDATE_KEYS, INPUT_MODEL, TRAINED_MODEL, UNTRAINED_MODEL, project_candidates
from experiments.make_revision.models import array_hash
from experiments.make_revision.run_revision import load_prepared, write_json

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'


def standin(name, *, rows=None):
    """load_artificial's contract with stand-in data: relative item positions, NaN for deleted items, labels 0..6."""
    n_items = 16 if name == 'ranks16' else 8
    rows = rows or ed.ARTIFICIAL[name]['expected_rows']
    rng = np.random.RandomState(len(name) + n_items)
    y = np.arange(rows) % 7
    X = np.array([rng.permutation(n_items) / (n_items - 1) for _ in range(rows)])
    X[rng.rand(rows, n_items) < .15] = np.nan
    names, labels = [f'item_{i + 1}' for i in range(n_items)], [str(label) for label in range(7)]
    return X, y, {'dataset_id': name, 'source': 'stand-in', 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                  'feature_names': names, 'label_map': labels, 'sample_order': 'stand-in', 'dataset_hash': dataset_fingerprint(X, y, names, labels)}


@pytest.fixture
def standin_pins(tmp_path, monkeypatch):
    """Pins of the stand-in artificial data against a stand-in generator file, installed as the module's pins."""
    module, approval, pins = tmp_path/'artificial_ranks.py', tmp_path/'approved.md', tmp_path/'artificial_pins.json'
    module.write_text('# stand-in generator\n')
    approval.write_text('approved (stand-in)\n')
    ed.pin_artificial(pins, approval=approval, loader=standin, module=module, committed={'committed': True, 'commit': 'stand-in', 'reason': None})
    monkeypatch.setattr(ed, 'ARTIFICIAL_PINS_FILE', pins)
    monkeypatch.setattr(ed, 'ARTIFICIAL_MODULE', module)
    return SimpleNamespace(module=module, pins=pins, loaders={'artificial': standin})


def tiny_registry(protocol):
    """The smoke registry's candidates for a production protocol (so that stage tests fit one-iteration networks)."""
    er.validate_extra_protocol(protocol)
    real = nd.build_registry(protocol)
    trained = nd.SMOKE_TRAINED_CANDIDATES
    registry = {TRAINED_MODEL: ModelSpec(TRAINED_MODEL, real[TRAINED_MODEL].factory, trained, True)}
    for model in (UNTRAINED_MODEL, INPUT_MODEL, nd.PROJECTED_MODEL):
        registry[model] = ModelSpec(model, real[model].factory,
                                    project_candidates(trained, CANDIDATE_KEYS[UNTRAINED_MODEL if model == UNTRAINED_MODEL else INPUT_MODEL]), True)
    for model in nd.MODEL_ORDER[4:]:
        registry[model] = ModelSpec(model, real[model].factory, real[model].candidates[:2], real[model].stochastic)
    return registry


def frozen(draft, **projection):
    family = draft['production_family']
    record = {'cap_hours': er.CAP_HOURS[family], 'workers': er.WORKERS[family], 'decision_hours': 1., **projection}
    return dict(draft, frozen=True, frozen_at_utc='2026-09-14T12:00:00+00:00', status=er.FROZEN_STATUS, resource_decision='test',
                projection=record)


# ----------------------------------------------------------------------------- protocols and registry

def test_the_drafts_copy_the_newdata_batch_design_the_models_and_the_ablation_design():
    template, ablation = json.loads(er.TEMPLATE.read_text()), json.loads(er.ABLATION_TEMPLATE.read_text())
    assert set(nd.DESIGN_COPY_KEYS) < set(er.TEMPLATE_KEYS)
    for family in er.FAMILIES:
        draft = er.draft_protocol(family)
        assert {key: draft[key] for key in er.TEMPLATE_KEYS} == {key: template[key] for key in er.TEMPLATE_KEYS}
        assert draft['design_source']['identical_to_template'] == ', '.join(er.TEMPLATE_KEYS)
        assert draft['source_template_sha256'] == hashlib.sha256(er.TEMPLATE.read_bytes()).hexdigest()
        assert draft['ablation_template_sha256'] == hashlib.sha256(er.ABLATION_TEMPLATE.read_bytes()).hexdigest()
        assert (draft['split_seed'], draft['outer_folds'], draft['outer_repeats'], draft['inner_folds'], draft['fit_seeds'],
                draft['stochastic_finalists'], draft['internal_validation_ratio'], draft['candidate_budget']) == (
                   27183, 5, 3, 3, [8129, 19391, 39019], 3, .1, 24)
        assert draft['models'] == nd.models_declaration() and draft['model_order'] == list(nd.MODEL_ORDER)
        for key in er.ABLATION_KEYS:
            expected = dict(ablation[key])if isinstance(ablation[key], dict) else ablation[key]
            if key == 'knn_readout':
                expected['source'] = draft['ablation'][key]['source']
            if key == 'reporting':
                expected['outputs'] = draft['ablation'][key]['outputs']
            assert draft['ablation'][key] == expected
        assert all(ablation[key] == draft[key] for key in er.ABLATION_SHARED_KEYS)
        assert tuple(draft['ablation']['variants']) == base.VARIANTS
        assert (draft['protocol_id'], draft['wallclock_cap_hours'], draft['workers']) == (
            {'dedup': 'arrowflow-v3-dedup-1', 'artificial': 'arrowflow-v3-artificial-1'}[family], {'dedup': 9, 'artificial': 4}[family],
            {'dedup': 16, 'artificial': 8}[family])
        assert draft['frozen'] is False and draft['projection'] is None and not {'batch', 'batches', 'batch_projection'} & set(draft)
        assert er.validate_extra_protocol(draft) == draft


def test_the_analysis_blocks_prespecify_the_families_and_the_descriptive_tables():
    dedup, artificial = er.draft_protocol('dedup')['analysis'], er.draft_protocol('artificial')['analysis']
    assert dedup['datasets'] == dedup['family_datasets'] == ['wine_quality_dedup', 'segment_dedup'] and dedup['descriptive_only_datasets'] == []
    assert artificial['datasets'] == ['ranks8', 'ranks16', 'ranks8_original'] and artificial['family_datasets'] == ['ranks8', 'ranks16']
    assert artificial['descriptive_only_datasets'] == ['ranks8_original']
    for block in (dedup, artificial):
        for label, control in (('primary', UNTRAINED_MODEL), ('secondary', INPUT_MODEL)):
            family = block[f'{label}_family']
            assert (family['model_a'], family['model_b'], family['size'], family['datasets'], family['alpha']) == (
                TRAINED_MODEL, control, 2, block['family_datasets'], .05)
            assert family['multiplicity'].startswith('Holm across the 2 family datasets')
            assert 'q = test_train_ratio = 0.25, 95%, df = outer folds - 1 = 14' in family['interval']
        assert block['competitiveness']['rules'] == holistic.RULES
        assert block['ladder'] == [{'rung': rung, 'model_id': model} for rung, model in nd.LADDER]
        assert block['main_table']['models'] == list(holistic.MAIN_MODELS) and block['comparator_intervals']['models'] == list(nd.COMPARATORS)
        assert block['components']['variants'] == list(holistic.COMPONENT_VARIANTS) and block['requires_run_and_ablation_complete']
    assert dedup['full_data_comparison']['sources'] == {'wine_quality_dedup': 'wine_quality', 'segment_dedup': 'segment'}
    assert dedup['full_data_comparison']['status'].startswith('unpaired; descriptive') and 'full_data_comparison' not in artificial
    assert dedup['duplicate_audit']['expected'] == dict.fromkeys(dedup['datasets'], dict.fromkeys(ed.IDENTITY_COUNTS, 0))


@pytest.mark.parametrize('mutation, message', [
    (lambda p: p.update(split_seed=1), 'split_seed'),
    (lambda p: p['models'][TRAINED_MODEL].update(candidates=15), 'models'),
    (lambda p: p['analysis']['primary_family'].update(size=3), 'analysis'),
    (lambda p: p['ablation'].update(variants=p['ablation']['variants'][:-1]), 'ablation'),
    (lambda p: p.update(datasets=['wine_quality_dedup']), 'datasets'),
    (lambda p: p.update(workers=32), 'workers'),
    (lambda p: p['panel'][0].update(kept_rows_sha256='0' * 64), 'panel'),
    (lambda p: p.update(production_family='newdata'), 'production_family'),
])
def test_validate_refuses_a_protocol_that_differs_from_the_draft(mutation, message):
    p = er.draft_protocol('dedup')
    mutation(p)
    with pytest.raises(ValueError, match=message):
        er.validate_extra_protocol(p)


def test_validate_refuses_an_unrecorded_or_over_cap_freeze_and_an_unpinned_frozen_panel(tmp_path, monkeypatch):
    draft = er.draft_protocol('dedup')
    assert er.validate_extra_protocol(frozen(draft))['frozen']
    for bad in (dict(frozen(draft), status='drafted'), frozen(draft, decision_hours=9.5), frozen(draft, workers=8),
                frozen(draft, cap_hours=4), dict(frozen(draft), resource_decision=''), frozen(draft, decision_hours=True)):
        with pytest.raises(ValueError, match='frozen protocol records'):
            er.validate_extra_protocol(bad)
    with pytest.raises(ValueError, match='must equal the draft'):
        er.validate_extra_protocol(dict(draft, status='other'))
    monkeypatch.setattr(ed, 'ARTIFICIAL_PINS_FILE', tmp_path/'absent.json')
    artificial = er.draft_protocol('artificial')
    assert not any(entry['pinned'] for entry in artificial['panel'])
    with pytest.raises(ValueError, match='pins every dataset'):
        er.validate_extra_protocol(frozen(artificial))


def test_the_registry_is_the_newdata_registry_and_the_smoke_registry_serves_smoke_protocols_only():
    draft = er.draft_protocol('dedup')
    registry = er.extra_registry(draft)
    assert list(registry) == list(nd.MODEL_ORDER) and nd.candidate_record(registry) == nd.candidate_record(nd.build_registry(nd.DESIGN))
    for batch in ('2026-09-14-newdata-batch1', '2026-09-14-newdata-batch2'):
        if (RUNS/batch/'candidates.json').is_file():
            assert canonical_json(nd.candidate_record(registry)) == canonical_json(json.loads((RUNS/batch/'candidates.json').read_text()))
    with pytest.raises(ValueError, match='synthetic smoke protocols only'):
        er.smoke_extra_registry(draft)
    smoke = er.smoke_protocol('artificial')
    assert list(er.smoke_extra_registry(smoke)) == list(nd.MODEL_ORDER)
    assert er.smoke_extra_registry(smoke)[TRAINED_MODEL].candidates == nd.SMOKE_TRAINED_CANDIDATES
    with pytest.raises(ValueError, match='extra_registry serves'):
        er.extra_registry(smoke)


def test_the_artificial_draft_takes_its_pins_from_the_pins_file(standin_pins):
    draft = er.draft_protocol('artificial')
    pins = ed.read_artificial_pins(standin_pins.pins)
    assert [entry['pinned'] for entry in draft['panel']] == [True, True, True]
    assert all(entry[key] == pins[entry['name']][key] for entry in draft['panel'] for key in pins[entry['name']])
    assert draft['analysis']['duplicate_audit']['expected'] == {name: pins[name]['duplicates'] for name in ed.ARTIFICIAL_DATASETS}
    assert er.validate_extra_protocol(frozen(draft))['frozen']


# ----------------------------------------------------------------------------- stages: prepare, pilot, run

def test_prepare_and_the_training_only_pilot_on_pinned_stand_in_data(tmp_path, monkeypatch, standin_pins):
    monkeypatch.setattr(er, 'extra_registry', tiny_registry)
    draft = er.draft_protocol('artificial')
    calls = []

    def spy(spec, config, seed, X_train, y_train, X_query, real=evaluation._fit_predict):
        calls.append(np.array_equal(X_train, X_query, equal_nan=True))
        return real(spec, config, seed, X_train, y_train, X_query)
    monkeypatch.setattr(evaluation, '_fit_predict', spy)
    report = er.runtime_pilot(tmp_path/'pilot', draft, ['ranks8_original'], loaders=standin_pins.loaders)
    X, y, manifest, splits = load_prepared(tmp_path/'pilot', 'ranks8_original')
    assert manifest['dataset_hash'] == draft['panel'][2]['dataset_hash'] and manifest['identity']['n_missing'] > 0
    assert report['protocol_hash'] == config_id(draft) and report['datasets_piloted'] == ['ranks8_original']
    expected_rows = sum(len({0, len(spec.candidates) // 2, len(spec.candidates) - 1}) for spec in tiny_registry(draft).values())
    assert len(report['rows']) == expected_rows == 19          # nine models at two candidates, the majority class at one
    assert all(row['status'] == 'ok' and row['fit_rows'] == splits[0]['train'] for row in report['rows'])
    assert calls and all(calls)                      # every pilot fit predicted its own training rows only
    ablation = report['ablation_rows'][0]
    assert ablation['status'] == 'ok' and ablation['query_rows'] == len(splits[0]['train'][::4]) and ablation['train_rows'] == len(splits[0]['train'])
    assert set(ablation['seconds_by_variant']) == {'views7', 'no_checkpoint', 'no_augment', 'untrained', 'input_knn'}
    scored = [key for row in report['rows'] + report['ablation_rows'] for key in row if key in ('accuracy', 'error', 'score', 'macro_f1')]
    assert not scored
    with pytest.raises(DatasetIdentityError := ed.DatasetIdentityError):
        er.prepare(tmp_path/'other', draft, loaders={'artificial': lambda name: standin(name, rows=315 if name != 'ranks8_original' else 84)})


def test_run_refuses_before_any_job_unless_the_prepared_protocol_is_the_frozen_pinned_one(tmp_path, monkeypatch, standin_pins):
    monkeypatch.setattr(er, 'extra_registry', tiny_registry)
    draft = er.draft_protocol('artificial')
    target = tmp_path/'run'
    er.prepare(target, frozen(draft), ['ranks8_original'], loaders=standin_pins.loaders)
    unfrozen, smoke, other = tmp_path/'draft.json', tmp_path/'smoke.json', tmp_path/'other.json'
    write_json(unfrozen, draft)
    write_json(smoke, er.smoke_protocol('artificial'))
    write_json(other, frozen(draft, decision_hours=2.))
    with pytest.raises(ValueError, match='frozen protocol'):
        er.run(unfrozen, target, 1)
    with pytest.raises(ValueError, match='synthetic smoke'):
        er.run(smoke, target, 1)
    with pytest.raises(ValueError, match='Prepared and frozen protocols differ'):
        er.run(other, target, 1)
    good = tmp_path/'frozen.json'
    write_json(good, frozen(draft))
    with pytest.raises(ValueError, match='Worker count'):
        er.run(good, target, 17)
    assert not (target/'planned_jobs.json').exists() and not (target/'results').exists()
    with pytest.raises(ValueError, match='committed frozen artificial.json'):
        ea.prepare(tmp_path/'ablation', frozen(draft), target)


# ----------------------------------------------------------------------------- projection and freeze

def synthetic_calibration(root, registry, factors, pilot_seconds=2.):
    """Completed-run layouts whose realized job seconds are factor x fits_per_outer x the pilot seconds (one job per dataset
    and model), with their pilots, in newdata.CALIBRATION's shape."""
    calibration = {}
    for label, models in nd.CALIBRATION_MODELS.items():
        directory = Path(root)/label
        (directory/'results').mkdir(parents=True)
        (directory/'candidates.json').write_text(json.dumps(nd.candidate_record({m: registry[m] for m in models})))
        (directory/'protocol.json').write_text(json.dumps({'inner_folds': 3}))
        jobs, rows = [], []
        for dataset, factor in factors.items():
            for model in models:
                per_outer = nd.fits_per_outer(len(registry[model].candidates), registry[model].stochastic, 3)
                log = f'results/{dataset}__{model}.fits.jsonl'
                (directory/log).write_text(json.dumps({'fit_seconds': factor * per_outer * pilot_seconds, 'predict_seconds': 0.}) + '\n')
                jobs.append({'dataset_id': dataset, 'model_id': model, 'log_file': log})
                rows.append({'dataset_id': dataset, 'model_id': model, 'config_id': 'c', 'status': 'ok', 'elapsed_seconds': pilot_seconds})
        (directory/'planned_jobs.json').write_text(json.dumps(jobs))
        (directory/'pilot.json').write_text(json.dumps({'rows': rows}))
        calibration[label] = {'run': directory, 'pilot': directory/'pilot.json'}
    return calibration


def simulate(durations, workers):
    free = [0.] * workers
    heapq.heapify(free)
    end = 0.
    for duration in durations:
        start = heapq.heappop(free)
        end = max(end, start + duration)
        heapq.heappush(free, start + duration)
    return end


def test_projection_prices_the_run_and_the_ablation_and_freeze_writes_only_within_the_cap(tmp_path):
    draft = er.draft_protocol('dedup')
    registry = nd.build_registry(draft)
    calibration = synthetic_calibration(tmp_path/'calibration', registry, {'d1': 2., 'd2': 3.})
    seconds = {model: 1. + index for index, model in enumerate(nd.MODEL_ORDER)}
    rows = [{'dataset_id': name, 'model_id': model, 'config_id': 'c', 'status': 'ok', 'elapsed_seconds': seconds[model] * (1 + k)}
            for k, name in enumerate(draft['datasets']) for model in nd.MODEL_ORDER]
    estimates = {model: {'serial_panel_seconds_using_observed_max': 3600.} for model in nd.MODEL_ORDER}
    ablation_rows = [{'dataset_id': name, 'status': 'ok', 'config_id': 'c', 'fit_sources': {'no_augment': 'separate'},
                      'all_variants_to_views7_ratio': 1.5, 'seconds_per_seed': 100. * (1 + k), 'seconds_per_job_estimate': 300. * (1 + k)}
                     for k, name in enumerate(draft['datasets'])]
    pilot = tmp_path/'pilot.json'
    write_json(pilot, {'rows': rows, 'workload_estimates': estimates, 'ablation_rows': ablation_rows, 'protocol_hash': config_id(draft)})
    record = er.projection(draft, pilot, calibration, runs_root=tmp_path/'no-runs')
    per_outer = {model: nd.fits_per_outer(len(spec.candidates), spec.stochastic, 3) for model, spec in registry.items()}
    assert record['calibration'][TRAINED_MODEL]['pooled'] == pytest.approx(2.5) and record['calibration'][TRAINED_MODEL]['max'] == pytest.approx(3.)
    job = {(k, model): 2.5 * per_outer[model] * seconds[model] * (1 + k) + 1. for k in range(2) for model in nd.MODEL_ORDER}
    run_durations = [job[k, model] for k in range(2) for _ in range(15) for model in nd.MODEL_ORDER]
    ablation_durations = [2.5 * 300. * (1 + k) + 1. for k in range(2) for _ in range(15)]
    assert record['run']['central']['simulated_makespan_hours'] == pytest.approx(simulate(run_durations, 16) / 3600)
    assert record['ablation']['central']['simulated_makespan_hours'] == pytest.approx(simulate(ablation_durations, 16) / 3600)
    assert record['decision_hours'] == pytest.approx((simulate(run_durations, 16) + simulate(ablation_durations, 16)) / 3600 + .25)
    assert record['harness_max_based_run_hours'] == pytest.approx(10 / 16) and record['within_cap'] and record['workers'] == 16
    assert record['realized_full_data_serial_hours'] is None
    projection, stages, draft_path = tmp_path/'projection.json', tmp_path/'stages.json', tmp_path/'draft.json'
    write_json(projection, record)
    write_json(stages, {'summary': 'stage summary (test)'})
    write_json(draft_path, draft)
    output = tmp_path/'frozen'/'dedup.json'
    protocol = er.freeze(draft_path, projection, stages, output, frozen_at_utc='2026-09-14T12:00:00+00:00')
    assert json.loads(output.read_text()) == protocol and er.validate_extra_protocol(protocol)['frozen']
    assert {k: v for k, v in protocol.items() if k not in er.FREEZE_FIELDS} == {k: v for k, v in draft.items() if k not in er.FREEZE_FIELDS}
    assert 'stage summary (test)' in protocol['resource_decision'] and protocol['projection']['ablation_pilot'] == record['ablation_pilot']
    with pytest.raises(FileExistsError):
        er.freeze(draft_path, projection, stages, output, frozen_at_utc='2026-09-14T13:00:00+00:00')
    over = tmp_path/'over.json'
    write_json(over, dict(record, decision_hours=9.5, within_cap=False))
    with pytest.raises(ValueError, match='exceeds the 9 h cap'):
        er.freeze(draft_path, over, stages, tmp_path/'over-frozen.json')
    other = tmp_path/'other.json'
    write_json(other, dict(record, protocol_hash='0' * 16))
    with pytest.raises(ValueError, match='not computed from a pilot of this draft'):
        er.freeze(draft_path, other, stages, tmp_path/'other-frozen.json')
    changed = tmp_path/'changed-draft.json'
    write_json(changed, dict(draft, split_seed=1))
    with pytest.raises(ValueError, match='committed draft'):
        er.freeze(changed, projection, stages, tmp_path/'changed-frozen.json')
    write_json(tmp_path/'pilot-other.json', dict(json.loads(pilot.read_text()), protocol_hash='1' * 16))
    with pytest.raises(ValueError, match='did not run with this protocol'):
        er.projection(draft, tmp_path/'pilot-other.json', calibration)


# ----------------------------------------------------------------------------- synthetic smoke, ablation and analysis

@pytest.fixture(scope='module')
def smokes(tmp_path_factory):
    root = tmp_path_factory.mktemp('extra_smoke')
    return {family: SimpleNamespace(root=root/family, record=er.smoke(root/family, family, workers=3)) for family in er.FAMILIES}


def test_the_smoke_runs_every_stage_of_both_families(smokes):
    for family, smoke in smokes.items():
        record, names = smoke.record, [entry['name'] for entry in er.SMOKE_PANELS[family]]
        members = [entry['name'] for entry in er.SMOKE_PANELS[family] if entry['role'] == 'family']
        assert record['purpose'] == 'synthetic_smoke_only_not_paper_evidence'
        assert set(record['outputs']) | {f'{family}_analysis.json'} == set(ce.output_names(family))    # the CSVs the JSON seals
        assert [row['dataset'] for row in record['primary_family']] == members == [row['dataset'] for row in record['secondary_family']]
        assert record['views7_reproduces_reference'] == {name: {'matching_fold_seeds': 9, 'total_fold_seeds': 9} for name in names}
        assert sorted(path.name for path in (smoke.root/'analysis').iterdir()) == sorted(ce.output_names(family))
    assert all(counts['duplicate_rows'] == 1 and counts['label_conflicting_groups'] == 1 for counts in smokes['artificial'].record['duplicate_audit'].values())
    assert all(counts['duplicate_rows'] == 0 for counts in smokes['dedup'].record['duplicate_audit'].values())


def test_the_families_equal_an_independent_recomputation_from_the_verified_run(smokes):
    for family, smoke in smokes.items():
        analysis = json.loads((smoke.root/'analysis'/f'{family}_analysis.json').read_text())
        run = load_run(smoke.root/'run', family)
        rows, _ = verify_run(run)
        members = analysis['analysis_declaration']['family_datasets']
        for label, control in (('primary', UNTRAINED_MODEL), ('secondary', INPUT_MODEL)):
            seeds = {model: run.schedule['expected_seeds'][model] for model in (TRAINED_MODEL, control)}
            intervals = [paired_corrected_interval([row for row in rows[name] if row['model_id'] in seeds], TRAINED_MODEL, control, q=.25,
                                                   confidence=.95, expected_folds=run.schedule['expected_folds'], expected_seeds=seeds)
                         for name in members]
            published = analysis[f'{label}_family']
            assert [row['dataset'] for row in published] == members and all(row['role'] == 'family' for row in published)
            for row, interval, adjusted in zip(published, intervals, holm_adjust([i['p_approximate'] for i in intervals])):
                assert row['mean_difference'] == pytest.approx(interval['mean_difference'], abs=1e-12)
                assert row['holm_p_approximate'] == pytest.approx(adjusted, abs=1e-12)
        names = run.protocol['datasets']
        assert [row['dataset'] for row in analysis['ladder']] == [name for name in names for _ in nd.LADDER]
        assert len(analysis['main_table']) == 7 * len(names) and len(analysis['complete_metrics']) == 10 * len(names)
        assert len(analysis['comparator_intervals']) == 6 * len(names) and len(analysis['competitiveness']) == len(names)
        assert len(analysis['selected_widths']['rows']) == 3 * len(names)
    assert 'syn_ranks_c' not in [row['dataset'] for row in smokes['artificial'].record['primary_family']]


def test_the_components_negate_the_ablation_change_and_dedup_holds_the_full_data_table(smokes):
    for family, smoke in smokes.items():
        report = json.loads((smoke.root/'ablation'/f'{family}_ablation_summary.json').read_text())
        analysis = json.loads((smoke.root/'analysis'/f'{family}_analysis.json').read_text())
        assert len(analysis['components']) == len(holistic.COMPONENT_VARIANTS) * len(report['summaries'])
        for row in analysis['components']:
            change = report['summaries'][row['dataset']]['variants'][row['variant']]['change_from_views7']['accuracy']
            assert row['arrowflow_minus_variant'] == pytest.approx(-change['mean_difference'], abs=1e-15)
            assert (row['ci_low'], row['ci_high']) == (pytest.approx(-change['ci_high'], abs=1e-15), pytest.approx(-change['ci_low'], abs=1e-15))
            assert row['arrowflow_minus_variant_points'] == pytest.approx(100 * row['arrowflow_minus_variant'], abs=1e-8)
            assert (row['group'], row['source_run']) == (family, f'{family}_ablation')
    full = json.loads((smokes['dedup'].root/'analysis'/'dedup_analysis.json').read_text())['full_data_comparison']
    assert len(full['rows']) == 20 and all(row['dedup_minus_full_points'] == 0 for row in full['rows'])


def test_analyse_refuses_an_incomplete_run_or_ablation_and_writes_nothing(smokes, tmp_path):
    smoke = smokes['dedup']
    for part, pattern in (('run', '*.fits.jsonl'), ('ablation', '*.jsonl')):
        run, ablation = tmp_path/part/'run', tmp_path/part/'ablation'
        shutil.copytree(smoke.root/'run', run)
        shutil.copytree(smoke.root/'ablation', ablation)
        folder = (run/'results') if part == 'run' else (ablation/'predictions')
        next(folder.glob(pattern)).unlink()
        output = tmp_path/part/'analysis'
        with pytest.raises(RunComparisonError, match=f'dedup {part} .* is not complete'):
            ce.analyse('dedup', run, ablation, output, allow_smoke=True)
        with pytest.raises(SystemExit) as exit_info:
            ce.main(['analyse', '--family', 'dedup', '--run', str(run), '--ablation', str(ablation), '--output', str(output),
                     '--runs', str(tmp_path)])
        assert exit_info.value.code == 2 and not output.exists()


def test_analyse_refuses_an_ablation_sealed_against_another_run_and_never_replaces_an_output(smokes, tmp_path):
    smoke = smokes['artificial']
    ablation = tmp_path/'ablation'
    shutil.copytree(smoke.root/'ablation', ablation)
    manifest = json.loads((ablation/'manifest.json').read_text())
    manifest['reference_source']['summary_sha256'] = '0' * 64
    (ablation/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    with pytest.raises(RunComparisonError, match='not prepared against this run'):
        ce.analyse('artificial', smoke.root/'run', ablation, tmp_path/'refused', allow_smoke=True)
    assert not (tmp_path/'refused').exists()
    with pytest.raises(RunComparisonError, match='the smoke'):
        ce.analyse('artificial', smoke.root/'run', smoke.root/'ablation', tmp_path/'production')    # a smoke run is not production evidence
    output = tmp_path/'analysis'
    ce.analyse('artificial', smoke.root/'run', smoke.root/'ablation', output, allow_smoke=True)
    (output/'artificial_ladder.csv').write_text('changed\n')
    with pytest.raises(FileExistsError):
        ce.analyse('artificial', smoke.root/'run', smoke.root/'ablation', output, allow_smoke=True)
    assert (output/'artificial_ladder.csv').read_text() == 'changed\n'


def test_the_ablation_summary_refuses_a_sealed_selection_that_the_run_does_not_reproduce(smokes, tmp_path):
    smoke = smokes['dedup']
    ablation = tmp_path/'ablation'
    shutil.copytree(smoke.root/'ablation', ablation)
    selections = json.loads((ablation/'reference_selections.json').read_text())
    labels = selections[-1]['reference_predictions']['8129']
    labels[0] = (labels[0] + 1) % 3
    selections[-1]['reference_prediction_hashes']['8129'] = array_hash(np.asarray(labels))
    (ablation/'reference_selections.json').write_text(json.dumps(selections, indent=2, sort_keys=True) + '\n')
    from experiments.make_revision.run_bridge import write_csv
    (ablation/'reference_selected_configurations.csv').unlink()
    write_csv(ablation/'reference_selected_configurations.csv', base.SELECTION_COLUMNS, base.selection_rows(selections))
    with pytest.raises(ValueError, match='sealed selection differs|reproduce'):
        ea.summary(ablation, allow_smoke=True)
    with pytest.raises(ValueError, match='requires the frozen production family protocol'):
        ea.summary(smoke.root/'ablation')                 # a synthetic ablation is refused outside the smoke


# ----------------------------------------------------------------------------- production branches without any fit

def test_the_production_ablation_checks_the_committed_protocol_the_reference_pins_and_every_pinned_reload(tmp_path, monkeypatch, standin_pins):
    monkeypatch.setattr(er, 'extra_registry', tiny_registry)
    protocol = frozen(er.draft_protocol('artificial'))
    committed = tmp_path/'protocols'/'artificial.json'
    write_json(committed, protocol)
    monkeypatch.setitem(er.PROTOCOL_FILES, 'artificial', committed)
    assert ea.check_protocol(protocol) is False
    with pytest.raises(ValueError, match='committed frozen artificial.json'):
        ea.check_protocol(dict(protocol, frozen_at_utc='2026-09-14T13:00:00+00:00'))
    run = tmp_path/'run'
    er.prepare(run, protocol, loaders=standin_pins.loaders)
    write_json(run/'summary.json', {'purpose': 'placeholder: the reference pins only hash it'})
    reference, pins = ea.family_reference(protocol, run)
    assert pins == {'protocol_id': 'arrowflow-v3-artificial-1', 'protocol_sha256': hashlib.sha256(committed.read_bytes()).hexdigest(),
                    'code_revision': json.loads((run/'environment.json').read_text())['code_revision'],
                    'summary_sha256': hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(), 'model_id': TRAINED_MODEL,
                    'family': 'artificial'}
    with pytest.raises(ValueError, match='does not match the protocol pins'):
        ea.family_reference(protocol, run, declared=dict(pins, summary_sha256='0' * 64))
    for name in protocol['datasets']:
        X, y, manifest, splits = load_prepared(run, name)
        identity = ea.check_dataset(name, X, y, manifest, splits, protocol, production=True, loaders=standin_pins.loaders)
        assert identity['pins_checked'] and identity['dataset_hash'] == er.panel_by_name(protocol)[name]['dataset_hash']
    X = X.copy()
    X[0, 0] = .123
    with pytest.raises(ed.DatasetIdentityError, match='differs from the reference prepared data'):
        ea.check_dataset(name, X, y, manifest, splits, protocol, production=True, loaders=standin_pins.loaders)
    (run/'protocol.json').write_text((run/'protocol.json').read_text() + '\n')          # the same protocol in other bytes
    with pytest.raises(ValueError, match='not the committed frozen protocol file'):
        ea.family_reference(protocol, run)


def test_the_realized_ablation_projection_takes_the_frozen_pilot_records():
    from experiments.make_revision import run_newdata_ablation as rna
    jobs = [{'dataset_id': 'd', 'outer_repeat': 0, 'outer_fold': fold, 'fit_sources': {'no_augment': 'separate'}} for fold in range(3)]
    selections = {('d', 0, fold): {'reference_outer_seconds': {'8129': 10., '19391': 20., '39019': 30.}} for fold in range(3)}
    records = [{'dataset_id': 'd', 'config_id': 'c', 'fit_sources': {'no_augment': 'separate'}, 'all_variants_to_views7_ratio': 2.,
                'seconds_per_seed': 1., 'seconds_per_job_estimate': 3.}]
    result = rna.calibrated_projection({'decision_rule': ea.REALIZED_RULE}, jobs, selections, records, workers=8)
    assert result['serial_hours'] == pytest.approx(3 * 120 / 3600) and result['datasets']['d']['basis'] == 'piloted ratio'
    assert json.loads(canonical_json(result)) == result


def test_the_analysis_accepts_only_a_run_of_the_committed_frozen_protocol_with_its_pins(tmp_path, monkeypatch):
    protocol = frozen(er.draft_protocol('dedup'))
    committed = tmp_path/'dedup.json'
    write_json(committed, protocol)
    monkeypatch.setitem(er.PROTOCOL_FILES, 'dedup', committed)
    registry = er.extra_registry(protocol)
    run = SimpleNamespace(label='dedup', path=tmp_path, protocol=protocol, protocol_sha256=hashlib.sha256(committed.read_bytes()).hexdigest(),
                          registry=registry, candidates=nd.candidate_record(registry),
                          environment={'registry': er.REGISTRY, 'code_revision': 'abc', 'source_hashes': dict.fromkeys(ce.SEALED_SOURCES, 'x')},
                          manifests={e['name']: {'dataset_hash': e['dataset_hash'], 'splits_hash': e['splits_hash']} for e in protocol['panel']},
                          schedule={'expected_folds': [(repeat, fold) for repeat in range(3) for fold in range(5)]})
    record = ce.check_run(run, 'dedup', smoke=False)
    assert record['pins_checked'] and record['datasets']['segment_dedup'] == {'dataset_hash': ed.DEDUP_PINS[1]['dataset_hash'], 'splits_hash': 'fb59c0afaf9399ff'}
    for change, message in ((dict(protocol_sha256='0' * 64), 'committed frozen dedup.json'),
                            (dict(manifests={**run.manifests, 'segment_dedup': {'dataset_hash': 'x', 'splits_hash': 'y'}}), 'differs from its pin'),
                            (dict(environment=dict(run.environment, source_hashes={})), 'must seal'),
                            (dict(candidates={}), 'candidates differ'),
                            (dict(schedule={'expected_folds': [(0, 0), (0, 1)]}), 'defined for')):
        with pytest.raises(RunComparisonError, match=message):
            ce.check_run(SimpleNamespace(**{**vars(run), **change}), 'dedup', smoke=False)
    with pytest.raises(RunComparisonError, match='the smoke'):
        ce.check_run(run, 'dedup', smoke=True)
    with pytest.raises(RunComparisonError, match='needs --runs'):
        ce.analyse('dedup', tmp_path, tmp_path, tmp_path/'out')
    assert not (tmp_path/'out').exists()


# ----------------------------------------------------------------------------- the registered runs and the committed protocols

needs_registered_runs = pytest.mark.skipif(not all((RUNS/name/'summary.json').is_file() for name in
                                                   (holistic.SOURCES[label] for label in holistic.RUN_LABELS)),
                                           reason='the registered runs are not on this machine')


@needs_registered_runs
def test_the_registered_full_data_values_are_the_holistic_benchmark_values():
    values, provenance = ce.registered_full_data(RUNS, ['wine_quality', 'segment'])
    assert set(provenance['runs']) == set(holistic.RUN_LABELS)
    table = RUNS/'2026-09-14-holistic'/'benchmark'/'complete_metrics.csv'
    if table.is_file():
        import csv
        published = {(row['dataset'], row['model_id']): row for row in csv.DictReader(table.open())}
        for (source, model), entry in values.items():
            assert entry['mean_error'] == pytest.approx(float(published[source, model]['error']), abs=1e-12)
            assert entry['outer_fold_sd'] == pytest.approx(float(published[source, model]['error_outer_fold_sd']), abs=1e-12)
            assert entry['source_run'] == published[source, model]['source_run']
    assert values['wine_quality', TRAINED_MODEL]['rows'] == 1599 and values['segment', 'dummy']['rows'] == 2310
    hours = er.realized_full_data_hours(er.draft_protocol('dedup'), RUNS)
    assert set(hours) == {'wine_quality_dedup', 'segment_dedup'} and hours['wine_quality_dedup']['serial_hours'] > 20


def test_the_committed_frozen_protocols_are_their_drafts_frozen_within_the_cap():
    # dedup only: the artificial protocol is frozen under artificial_runs' settings (tests/make_revision/test_artificial_runs.py)
    present = [family for family, path in er.PROTOCOL_FILES.items() if path.is_file() and family == 'dedup']
    for family in present:
        protocol = json.loads(er.PROTOCOL_FILES[family].read_text())
        assert er.validate_extra_protocol(protocol)['frozen'] and protocol['protocol_id'] == er.PROTOCOL_IDS[family]
        assert 0 < protocol['projection']['decision_hours'] <= er.CAP_HOURS[family] and protocol['projection']['workers'] == er.WORKERS[family]
        draft = er.draft_protocol(family)
        assert {k: v for k, v in protocol.items() if k not in er.FREEZE_FIELDS} == {k: v for k, v in draft.items() if k not in er.FREEZE_FIELDS}
