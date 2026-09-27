"""Task 23A: compare_runs projected (compare_projected) on synthetic knn, knn_training and knn_projected runs.

The runs are test_compare_runs.build_run fixtures: the saved layout of a complete run with its complete inner selection
history, per-example predictions and summary.json, which the harness validators accept."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import stats
import test_compare_runs as tcr
from experiments.make_revision import compare_projected as cp
from experiments.make_revision import compare_runs as cr
from experiments.make_revision import projected_knn as pk
from experiments.make_revision.evaluation import holm_adjust, summarize_outer

PROJECTED = pk.PROJECTED_MODEL
PROJECTED_REVISION = 'd49fcd033' + '0' * 31
KNN_MODELS = ('arrowflow_full_knn', 'numeric_knn', 'dummy')
CANDIDATES = {**tcr.TRAINING_CANDIDATES, PROJECTED: tcr.TRAINING_CANDIDATES['input_footrule_knn']}
KNN_CONTROLS, PROJECTED_SOURCE = 'experiments/make_revision/knn_controls.py', 'experiments/make_revision/projected_knn.py'
TRAINING_HASHES = {**tcr.SHARED_HASHES, KNN_CONTROLS: '1' * 64}
PROJECTED_HASHES = {**TRAINING_HASHES, PROJECTED_SOURCE: '3' * 64}
tcr.STOCHASTIC.setdefault(PROJECTED, True)
tcr.ERROR_RATE.update({('alpha', PROJECTED): .9, ('beta', PROJECTED): .35})       # alpha separates Holm from the raw p
REAL_TRAINING = tcr.REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'/'2026-09-13-knn-training'


def build_projected(root, knn, training, **changes):
    """A knn_projected run whose references pin the given knn and knn_training runs; changes alter its protocol."""
    protocol = json.loads((tcr.PROTOCOLS/'knn_projected.json').read_text())
    protocol.update(datasets=list(tcr.DATASETS), primary_family_size=4, frozen=True, frozen_at_utc='2026-09-13T20:00:00+00:00',
                    **changes)
    for label, directory in (('knn', knn), ('training', training)):
        protocol['projected_control']['references'][label].update(pk.run_pins(directory, label))
    rows = tcr.build_run(root, protocol=protocol, models=(PROJECTED,), revision=PROJECTED_REVISION,
                         registry='experiments.make_revision.projected_knn:knn_projected_registry', candidates=CANDIDATES)
    tcr.rewrite(root/'environment.json', lambda e: e.update(source_hashes=dict(PROJECTED_HASHES)))
    return rows


def build_triple(root):
    knn_p = json.loads((tcr.PROTOCOLS/'bridge_knn.json').read_text())
    knn_p.update(datasets=list(tcr.DATASETS), primary_family_size=len(tcr.DATASETS))
    knn_rows = tcr.build_run(root/'knn', protocol=knn_p, models=KNN_MODELS, revision=tcr.KNN_REVISION,
                             registry='experiments.make_revision.bridge:bridge_knn_registry', candidates=CANDIDATES)
    tcr.rewrite(root/'knn'/'environment.json', lambda e: e.update(source_hashes=dict(tcr.SHARED_HASHES)))
    training_p = json.loads((tcr.PROTOCOLS/'knn_training.json').read_text())
    training_p.update(datasets=list(tcr.DATASETS), primary_family_size=4, frozen=True, frozen_at_utc='2026-09-13T12:00:00+00:00')
    training_p['training_controls']['reference'].update(code_revision=tcr.KNN_REVISION, protocol_sha256=tcr.sha256(root/'knn'/'protocol.json'),
                                                        summary_sha256=tcr.sha256(root/'knn'/'summary.json'))
    training_rows = tcr.build_run(root/'training', protocol=training_p, models=tcr.CONTROLS, revision=tcr.TRAINING_REVISION,
                                  registry='experiments.make_revision.knn_controls:knn_training_registry', candidates=CANDIDATES)
    tcr.rewrite(root/'training'/'environment.json', lambda e: e.update(source_hashes=dict(TRAINING_HASHES)))
    projected_rows = build_projected(root/'projected', root/'knn', root/'training')
    return SimpleNamespace(knn=root/'knn', training=root/'training', projected=root/'projected', knn_rows=knn_rows,
                           training_rows=training_rows, projected_rows=projected_rows)


@pytest.fixture(scope='session')
def projected_baseline(tmp_path_factory):
    root = tmp_path_factory.mktemp('projected_baseline')
    runs = build_triple(root)
    verified, original = [], cp.verify_run

    def recording(run):
        verified.append(run.label)
        return original(run)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cp, 'verify_run', recording)
        result = cp.compare_projected(runs.projected, runs.knn, runs.training, root/'out')
    return SimpleNamespace(runs=runs, result=result, out=root/'out', verified=verified)


@pytest.fixture
def projected_runs(projected_baseline, tmp_path):
    for label in ('knn', 'training', 'projected'):
        shutil.copytree(getattr(projected_baseline.runs, label), tmp_path/'runs'/label)
    return SimpleNamespace(**{label: tmp_path/'runs'/label for label in ('knn', 'training', 'projected')})


def fold_differences(rows_a, model_a, rows_b, model_b):
    a, b = tcr.fold_means(rows_a, model_a, 'accuracy'), tcr.fold_means(rows_b, model_b, 'accuracy')
    assert sorted(a) == sorted(b) == tcr.FOLDS
    return np.array([np.mean(a[key]) - np.mean(b[key]) for key in sorted(a)])


def test_family_is_input_footrule_and_arrowflow_knn_minus_projected_knn_with_holm_across_every_member(projected_baseline):
    runs, contrasts, out = projected_baseline.runs, projected_baseline.result['contrasts'], projected_baseline.out
    assert projected_baseline.verified == ['projected', 'knn', 'training']              # all three runs re-verified
    assert [(r['family_index'], r['dataset'], r['contrast'], r['model_a'], r['run_a']) for r in contrasts] == [
        (1, 'alpha', pk.PRIMARY_CONTRASTS[0], 'input_footrule_knn', 'training'), (2, 'alpha', pk.PRIMARY_CONTRASTS[1], 'arrowflow_full_knn', 'knn'),
        (3, 'beta', pk.PRIMARY_CONTRASTS[0], 'input_footrule_knn', 'training'), (4, 'beta', pk.PRIMARY_CONTRASTS[1], 'arrowflow_full_knn', 'knn')]
    source_rows = {'knn': runs.knn_rows, 'training': runs.training_rows}
    for row in contrasts:
        differences = fold_differences(source_rows[row['run_a']][row['dataset']], row['model_a'], runs.projected_rows[row['dataset']], PROJECTED)
        se = np.sqrt((1 / 15 + .25) * np.var(differences, ddof=1))
        half = stats.t.ppf(.975, 14) * se
        assert (row['model_b'], row['run_b'], row['n_folds'], row['df']) == (PROJECTED, 'projected', 15, 14)
        assert np.isclose(row['mean_difference'], differences.mean(), rtol=0, atol=1e-12)
        assert np.isclose(row['standard_error'], se, rtol=0, atol=1e-12) and np.isclose(row['ci_low'], differences.mean() - half, rtol=0, atol=1e-12)
        assert np.isclose(row['p_approximate'], 2 * stats.t.sf(abs(differences.mean() / se), 14), rtol=0, atol=1e-12)
    holm = [r['holm_p_approximate'] for r in contrasts]
    assert holm == holm_adjust([r['p_approximate'] for r in contrasts])
    assert len(contrasts) == json.loads((runs.projected/'protocol.json').read_text())['primary_family_size'] == 4
    assert min(holm) < 1 and any(adjusted != row['p_approximate'] for adjusted, row in zip(holm, contrasts))
    text = (out/'projected_contrasts.csv').read_text()
    assert text.splitlines()[0].split(',') == list(cr.TRAINING_CONTRAST_COLUMNS) and len(text.splitlines()) == 5
    summary = json.loads((out/'projected_contrasts.json').read_text())
    assert summary['outputs']['projected_contrasts.csv']['sha256'] == hashlib.sha256(text.encode()).hexdigest()
    assert summary['family']['size'] == 4 and summary['contrasts_declared'] == pk.PRIMARY_CONTRASTS
    verification = summary['provenance']['verification']
    assert (verification['projected']['jobs_verified'], verification['knn']['jobs_verified'], verification['training']['jobs_verified']) == (30, 90, 60)
    pairing = summary['provenance']['pairing']
    assert pairing['sealed_by'][KNN_CONTROLS] == ['projected', 'training'] and pairing['sealed_by']['arrowflow/ranking.py'] == ['projected', 'knn', 'training']
    assert PROJECTED_SOURCE not in pairing['shared_sources'] and pairing['candidate_config_ids'] == [c for c in json.loads(
        (runs.training/'candidates.json').read_text())['input_footrule_knn']['config_ids']]
    assert set(summary['provenance']['analysis_sources']) == {f'experiments/make_revision/{n}' for n in cp.PROJECTED_ANALYSIS_SOURCES}


def test_raw_numeric_knn_contrast_is_descriptive_outside_the_family_without_p_values(projected_baseline):
    runs, result, out = projected_baseline.runs, projected_baseline.result, projected_baseline.out
    rows = result['descriptive']
    assert [(r['dataset'], r['contrast'], r['model_a'], r['run_a'], r['status']) for r in rows] == [
        (name, pk.DESCRIPTIVE_CONTRAST, 'numeric_knn', 'knn', cp.DESCRIPTIVE_STATUS) for name in tcr.DATASETS]
    for row in rows:
        raw = tcr.fold_means(runs.knn_rows[row['dataset']], 'numeric_knn', 'accuracy')
        assert all(len(values) == 1 for values in raw.values())                          # one fitting seed against three
        differences = fold_differences(runs.knn_rows[row['dataset']], 'numeric_knn', runs.projected_rows[row['dataset']], PROJECTED)
        assert np.isclose(row['mean_difference'], differences.mean(), rtol=0, atol=1e-12)
        assert not {'p_approximate', 'holm_p_approximate', 'family_index'} & set(row)
    assert pk.DESCRIPTIVE_CONTRAST not in {r['contrast'] for r in result['contrasts']}
    header = (out/'projected_raw_descriptive.csv').read_text().splitlines()[0].split(',')
    assert header == list(cp.DESCRIPTIVE_COLUMNS) and not {'p_approximate', 'holm_p_approximate'} & set(header)
    assert json.loads((out/'projected_contrasts.json').read_text())['descriptive']['rows'] == rows


def test_ladder_error_table_reads_each_rung_from_its_own_verified_summary(projected_baseline):
    runs, table = projected_baseline.runs, projected_baseline.result['ladder']
    assert [(r['model_id'], r['source_run']) for r in table['rungs']] == [
        ('numeric_knn', 'knn'), (PROJECTED, 'projected'), ('input_footrule_knn', 'training'), ('arrowflow_knn_untrained', 'training'),
        ('arrowflow_full_knn', 'knn')]
    sources = {'knn': runs.knn_rows, 'training': runs.training_rows, 'projected': runs.projected_rows}
    for name in tcr.DATASETS:
        for row in table['rows'][name]:
            seeds = tcr.SEEDS[:1] if row['model_id'] == 'numeric_knn' else tcr.SEEDS
            expected = summarize_outer(sources[row['source_run']][name], row['model_id'], 'error', expected_folds=tcr.FOLDS, expected_seeds=seeds)
            assert np.isclose(row['mean_error'], expected['mean'], rtol=0, atol=1e-12)
            assert (row['n_folds'], row['seeds_per_fold']) == (15, len(seeds))
        assert table['rows'][name][0]['mean_within_fold_seed_sd'] is None
    assert json.loads((projected_baseline.out/'projected_ladder_error_table.json').read_text()) == table


def edit_protocol(root, change):
    tcr.rewrite(root/'protocol.json', change)


def rebuild_projected(runs, **changes):
    shutil.rmtree(runs.projected)
    build_projected(runs.projected, runs.knn, runs.training, **changes)


REFUSALS = {
    'knn_reference': (lambda r: edit_protocol(r.projected, lambda p: p['projected_control']['references']['knn'].update(
        summary_sha256='0' * 64)), 'does not match the knn run: summary_sha256'),
    'training_reference': (lambda r: edit_protocol(r.projected, lambda p: p['projected_control']['references']['training'].update(
        code_revision='f' * 40)), 'does not match the training run: code_revision'),
    'training_protocol_changed': (lambda r: edit_protocol(r.training, lambda p: p.update(status='edited')),
                                  'does not match the training run: protocol_sha256'),
    'contrast_order': (lambda r: edit_protocol(r.projected, lambda p: p.update(primary_contrasts=p['primary_contrasts'][::-1])),
                       'primary_contrasts must be'),
    'family_size': (lambda r: edit_protocol(r.projected, lambda p: p.update(primary_family_size=2)), 'primary_family_size'),
    'design_copy': (lambda r: edit_protocol(r.projected, lambda p: p.update(failure_policy='stop_early')),
                    "failure_policy is not the knn_training protocol's"),
    'fit_seeds': (lambda r: rebuild_projected(r, fit_seeds=[8129, 19391, 1]), 'Protocol fit_seeds differs between the runs'),
    'outer_folds': (lambda r: rebuild_projected(r, outer_folds=4), 'Protocol outer_folds differs between the runs'),
    'dataset_hash': (lambda r: tcr.rewrite(r.projected/'beta'/'manifest.json', lambda m: m.update(dataset_hash='0' * 64)),
                     'dataset hash of beta differs'),
    'splits_hash': (lambda r: tcr.rewrite(r.training/'alpha'/'manifest.json', lambda m: m.update(splits_hash='0' * 16)),
                    'splits hash of alpha differs'),
    'candidates': (lambda r: tcr.rewrite(r.projected/'candidates.json', lambda c: c[PROJECTED].update(
        candidates=c[PROJECTED]['candidates'][:1], config_ids=c[PROJECTED]['config_ids'][:1])), 'candidates differ from the training run'),
    'shared_source_differs': (lambda r: tcr.rewrite(r.projected/'environment.json', lambda e: e['source_hashes'].update(
        {'experiments/make_revision/models.py': '2' * 64})), 'sealed by more than one run differ: experiments/make_revision/models.py'),
    'knn_controls_differs': (lambda r: tcr.rewrite(r.projected/'environment.json', lambda e: e['source_hashes'].update(
        {KNN_CONTROLS: '4' * 64})), f'sealed by more than one run differ: {KNN_CONTROLS}'),
    'shared_source_absent': (lambda r: tcr.rewrite(r.knn/'environment.json', lambda e: e['source_hashes'].pop('arrowflow/ranking.py')),
                             'must seal knn run arrowflow/ranking.py'),
    'knn_controls_absent': (lambda r: tcr.rewrite(r.training/'environment.json', lambda e: e['source_hashes'].pop(KNN_CONTROLS)),
                            f'must seal training run {KNN_CONTROLS}'),
    'unfrozen': (lambda r: edit_protocol(r.projected, lambda p: p.update(frozen=False)), 'projected run protocol is not frozen'),
    'incomplete': (lambda r: (r.projected/'summary.json').unlink(), 'incomplete or unverified'),
    'tampered_projected_prediction': (lambda r: tcr.rewrite_result(r.projected, 0, lambda res: res['predictions'][0].update(
        y_pred=(res['predictions'][0]['y_pred'] + 1) % 2)), 'projected run results/alpha__projected_numeric_knn__r0f0.json fails'),
    'tampered_knn_prediction': (lambda r: tcr.rewrite_result(r.knn, 1, lambda res: res['predictions'][0].update(
        y_pred=(res['predictions'][0]['y_pred'] + 1) % 2)), 'knn run results/alpha__numeric_knn__r0f0.json fails'),
    'tampered_training_prediction': (lambda r: tcr.rewrite_result(r.training, 1, lambda res: res['predictions'][0].update(
        y_pred=(res['predictions'][0]['y_pred'] + 1) % 2)), 'training run results/alpha__input_footrule_knn__r0f0.json fails'),
}


@pytest.mark.parametrize('change', sorted(REFUSALS))
def test_refuses_unpinned_unpaired_or_unverified_runs_before_writing_anything(projected_runs, tmp_path, change):
    alter, message = REFUSALS[change]
    alter(projected_runs)
    with pytest.raises(cr.RunComparisonError) as refused:
        cp.compare_projected(projected_runs.projected, projected_runs.knn, projected_runs.training, tmp_path/'out')
    assert message in str(refused.value), str(refused.value)
    assert not (tmp_path/'out').exists()


def test_projected_commands_end_to_end_with_the_prepared_pairing_check(projected_baseline, tmp_path, capsys):
    runs = projected_baseline.runs
    sources = ['--projected-source', str(runs.projected), '--knn-source', str(runs.knn), '--training-source', str(runs.training)]
    completed = subprocess.run([sys.executable, '-m', 'experiments.make_revision.compare_runs', 'projected', *sources,
                                '--output', str(tmp_path/'cli')], cwd=tcr.REPO, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stderr
    assert sorted(p.name for p in (tmp_path/'cli').iterdir()) == sorted(cp.PROJECTED_OUTPUTS)
    assert all((tmp_path/'cli'/name).read_bytes() == (projected_baseline.out/name).read_bytes() for name in cp.PROJECTED_OUTPUTS)
    assert 'alpha: input_footrule_knn - projected_numeric_knn accuracy' in completed.stdout and '(descriptive)' in completed.stdout
    cr.main(['projected', *sources, '--output', str(tmp_path/'cli')])      # the same evidence again: identical files, no error
    (tmp_path/'cli'/'projected_ladder_error_table.json').write_text('{}\n')
    with pytest.raises(SystemExit) as refused:
        cr.main(['projected', *sources, '--output', str(tmp_path/'cli')])
    assert refused.value.code == 2 and 'overwrite' in capsys.readouterr().err
    prepared = tmp_path/'prepared'                                        # a knn_projected directory whose jobs have not run
    prepared.mkdir()
    for name in ('protocol.json', 'environment.json', 'candidates.json'):
        shutil.copyfile(runs.projected/name, prepared/name)
    for name in tcr.DATASETS:
        (prepared/name).mkdir()
        shutil.copyfile(runs.projected/name/'manifest.json', prepared/name/'manifest.json')
    pairing = ['projected-pairing', '--projected-source', str(prepared), *sources[2:]]
    cr.main(pairing)
    assert 'paired with arrowflow-v3-bridge-knn-1' in capsys.readouterr().out
    tcr.rewrite(prepared/'protocol.json', lambda p: p.update(split_seed=1))
    with pytest.raises(SystemExit) as refused:
        cr.main(pairing)
    assert refused.value.code == 2 and 'split_seed' in capsys.readouterr().err


@pytest.mark.skipif(not (tcr.REAL_KNN.is_dir() and REAL_TRAINING.is_dir()), reason='the knn and knn_training runs are not on this machine')
def test_real_knn_and_training_runs_are_the_projected_references_and_their_sealed_sources_are_unchanged_in_this_tree():
    knn, training = cr.load_run(tcr.REAL_KNN, 'knn'), cr.load_run(REAL_TRAINING, 'training')
    record = cp.check_projected_references(json.loads((tcr.PROTOCOLS/'knn_projected.json').read_text()), knn, training)
    assert record['training']['summary_sha256'] == tcr.sha256(REAL_TRAINING/'summary.json')
    assert training.candidates['input_footrule_knn']['candidates'] == pk.projected_candidates()
    for source in (*cr.SHARED_SOURCES, KNN_CONTROLS):
        assert training.environment['source_hashes'][source] == tcr.sha256(tcr.REPO/source), source
