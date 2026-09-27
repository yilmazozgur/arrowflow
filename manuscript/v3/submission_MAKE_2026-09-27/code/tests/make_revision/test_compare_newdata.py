"""Task 23B: compare_newdata, the prespecified analysis of the two newdata batch runs.

Statistics and tables are checked on synthetic inputs with known answers and against independent enumerations; the
command is checked end to end on two fixture batch runs (test_compare_runs.build_run layout: complete inner selection
histories, per-example predictions and summary.json, which the harness validators accept) over a four-dataset synthetic
panel, including the refusal to read anything before both batches are complete."""
import csv
import hashlib
import io
import json
import shutil
from itertools import combinations, permutations
from math import comb
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import stats
import test_compare_runs as tcr
from experiments.make_revision import compare_newdata as cn
from experiments.make_revision import newdata as nd
from experiments.make_revision.compare_runs import RunComparisonError
from experiments.make_revision.evaluation import summarize_outer

TEN, H = list(nd.PANEL), list(nd.H_STRATUM)
TRAINED, UNTRAINED, INPUT = nd.TRAINED_MODEL, nd.UNTRAINED_MODEL, nd.INPUT_MODEL
FOLDS = [(repeat, fold) for repeat in range(3) for fold in range(5)]
SEEDS = [8129, 19391, 39019]


def holm(p):
    """An independent step-down Holm adjustment."""
    order = sorted(range(len(p)), key=lambda i: (p[i], i))
    adjusted, running = [0.] * len(p), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


# ----------------------------------------------------------------------------- the moderator permutation test

def test_permutation_test_enumerates_the_210_splits_with_known_answers():
    extreme = cn.permutation_test(dict(zip(TEN, [10., 9., 8., 7., 0., 0., 0., 0., 0., 0.])), TEN, H)
    assert extreme['splits'] == 210 == comb(10, 4) == len(extreme['null_distribution'])
    assert len({tuple(entry['h']) for entry in extreme['null_distribution']}) == 210
    assert all(len(entry['h']) == 4 and set(entry['h']) <= set(TEN) for entry in extreme['null_distribution'])
    assert (extreme['mean_h'], extreme['mean_c'], extreme['statistic']) == (8.5, 0., 8.5)
    assert extreme['at_least_observed'] == 1 and extreme['p_one_sided'] == 1 / 210 and extreme['larger_than_observed'] == 0
    assert extreme['null_distribution'][0] == {'statistic': 8.5, 'h': H}
    assert cn.permutation_test(dict.fromkeys(TEN, .03), TEN, H)['p_one_sided'] == 1.      # every split ties
    lone = cn.permutation_test({**dict.fromkeys(TEN, 0.), H[0]: 1.}, TEN, H)
    # 1/4 whenever H holds that dataset (C(9, 3) = 84 splits), -1/6 otherwise
    assert lone['statistic'] == pytest.approx(.25) and lone['at_least_observed'] == 84 and lone['p_one_sided'] == pytest.approx(.4)
    smallest = cn.permutation_test(dict(zip(TEN, [0., 0., 0., 0., 1., 2., 3., 4., 5., 6.])), TEN, H)
    assert smallest['statistic'] == pytest.approx(-3.5) and smallest['p_one_sided'] == 1.


def test_permutation_test_equals_an_independent_enumeration_and_counts_near_ties():
    rng = np.random.RandomState(7)
    for _ in range(6):
        values = rng.normal(scale=.03, size=10).round(3)
        observed = values[:4].mean() - values[4:].mean()
        count = sum(values[list(g)].mean() - np.delete(values, list(g)).mean() >= observed - 1e-12
                    for g in combinations(range(10), 4))
        result = cn.permutation_test(dict(zip(TEN, values)), TEN, H)
        assert result['at_least_observed'] == count and result['p_one_sided'] == count / 210
    shifted = {name: value + .1 for name, value in zip(TEN, [.3, .1, .2, .1, .3, .1, .2, .1, .0, .0])}
    exact = cn.permutation_test(shifted, TEN, H)
    assert exact['at_least_observed'] == cn.permutation_test(shifted, TEN, H, tolerance=0.)['at_least_observed'] + sum(
        0 < exact['statistic'] - entry['statistic'] <= 1e-12 for entry in exact['null_distribution'])


@pytest.mark.parametrize('h, effects', [([], None), (TEN, None), (['unknown'], None), (H + [H[0]], None),
                                        (H, {**dict.fromkeys(TEN, 0.), H[1]: float('nan')})])
def test_permutation_test_refuses_an_ill_defined_split(h, effects):
    with pytest.raises(ValueError):
        cn.permutation_test(effects or dict.fromkeys(TEN, 0.), TEN, h)


# ----------------------------------------------------------------------------- the descriptive Spearman correlation

GAPS = [9.5, 8.7, 7.9, 7.25, 1.95, 0.1, 2.3, 6.0, 4.1]


def test_spearman_exact_known_answers_over_all_nine_factorial_orders():
    ranks = stats.rankdata(GAPS)
    increasing = cn.spearman_exact(GAPS, ranks / 100)
    assert increasing['rho'] == pytest.approx(1.) and increasing['permutations'] == 362880 and increasing['n'] == 9
    assert increasing['p_one_sided'] == pytest.approx(1 / 362880) and increasing['p_two_sided'] == pytest.approx(2 / 362880)
    decreasing = cn.spearman_exact(GAPS, -ranks)
    assert decreasing['rho'] == pytest.approx(-1.) and decreasing['p_one_sided'] == 1.
    assert decreasing['p_two_sided'] == pytest.approx(2 / 362880)
    tied = np.random.RandomState(3).choice([0., .01, .02], size=9)
    assert cn.spearman_exact(GAPS, tied)['rho'] == pytest.approx(stats.spearmanr(GAPS, tied).statistic, abs=1e-12)
    undefined = cn.spearman_exact(GAPS, [.02] * 9)
    assert undefined['rho'] is None and undefined['p_one_sided'] is None and undefined['p_two_sided'] is None


def test_spearman_exact_p_values_equal_an_independent_enumeration_with_ties():
    gaps, effects = [3., 1., 4., 1.5, 9., 2.], [.2, .1, .1, .4, .3, .1]
    observed = stats.spearmanr(gaps, effects).statistic
    rhos = [stats.spearmanr(gaps, [effects[i] for i in order]).statistic for order in permutations(range(6))]
    result = cn.spearman_exact(gaps, effects)
    assert result['rho'] == pytest.approx(observed, abs=1e-12) and result['permutations'] == 720
    assert result['p_one_sided'] == pytest.approx(np.mean([rho >= observed - 1e-12 for rho in rhos]))
    assert result['p_two_sided'] == pytest.approx(np.mean([abs(rho) >= abs(observed) - 1e-12 for rho in rhos]))


@pytest.mark.parametrize('gaps, effects', [([1., 2.], [1., 2.]), ([1., 2., 3.], [1., 2.]), ([1., 2., float('nan')], [1., 2., 3.])])
def test_spearman_exact_refuses_too_few_or_unpaired_values(gaps, effects):
    with pytest.raises(ValueError):
        cn.spearman_exact(gaps, effects)


def test_stratum_summary_known_values():
    summary = cn.stratum_summary({'a': .1, 'b': .3, 'c': .2, 'd': -.1}, {'H': ['a', 'b'], 'C': ['c', 'd']})
    assert summary['H'] == {'n': 2, 'mean': pytest.approx(.2), 'sd': pytest.approx(np.std([.1, .3], ddof=1)), 'median': pytest.approx(.2),
                            'min': .1, 'max': .3, 'effects': {'a': .1, 'b': .3}}
    assert summary['C']['mean'] == pytest.approx(.05) and summary['C']['min'] == -.1


# ----------------------------------------------------------------------------- families, ladder and comparators on synthetic rows

def synthetic_runs(names, accuracy):
    """One run-like object per dataset holding seeded outer model rows whose accuracy is accuracy(name, model, fold, seed)."""
    runs = {}
    models = (TRAINED, UNTRAINED, INPUT, *nd.PROJECTED_MODEL.split(), *nd.COMPARATORS)
    for name in names:
        rows, summaries = [], []
        for model in models:
            seeds = SEEDS if model not in ('numeric_knn', 'svc_rbf', 'dummy') else SEEDS[:1]
            model_rows = [{'dataset_id': name, 'model_id': model, 'outer_repeat': r, 'outer_fold': f, 'model_seed': s, 'status': 'ok',
                           'accuracy': accuracy(name, model, (r, f), s), 'error': 1 - accuracy(name, model, (r, f), s)}
                          for r, f in FOLDS for s in seeds]
            rows += model_rows
            summaries.append(summarize_outer(model_rows, model, 'error', expected_folds=FOLDS, expected_seeds=seeds))
        runs[name] = SimpleNamespace(schedule={'expected_folds': FOLDS, 'expected_seeds': {
            m: (SEEDS if m not in ('numeric_knn', 'svc_rbf', 'dummy') else SEEDS[:1]) for m in models}},
            summary={'model_rows': {name: rows}, 'summaries': {name: summaries}})
    return runs


def accuracy_table(name, model, fold, seed):
    digest = hashlib.sha256(json.dumps([name, model, fold, seed]).encode()).digest()
    noise = int.from_bytes(digest[:4], 'big') / 2 ** 32 * .04
    effect = {TRAINED: .06 * (TEN.index(name) % 4), UNTRAINED: 0., INPUT: .01}.get(model, .02)
    return round(.7 + effect + noise, 6)


def test_both_families_are_seed_averaged_corrected_t_with_holm_across_the_ten_datasets():
    runs = synthetic_runs(TEN, accuracy_table)
    stratum = {name: ('H' if name in H else 'C') for name in TEN}
    batch_of = dict.fromkeys(TEN, 1)
    for family, model_b in (('primary', UNTRAINED), ('secondary', INPUT)):
        rows = cn.family_rows(runs, TEN, family=family, contrast=f'{TRAINED}_vs_{model_b}', model_a=TRAINED, model_b=model_b,
                              stratum=stratum, batch_of=batch_of, q=.25, confidence=.95)
        assert [row['dataset'] for row in rows] == TEN and [row['family_index'] for row in rows] == list(range(1, 11))
        p = []
        for row in rows:
            differences = np.array([np.mean([accuracy_table(row['dataset'], TRAINED, fold, s) for s in SEEDS])
                                    - np.mean([accuracy_table(row['dataset'], model_b, fold, s) for s in SEEDS]) for fold in FOLDS])
            se = np.sqrt((1 / 15 + .25) * np.var(differences, ddof=1))
            assert row['mean_difference'] == pytest.approx(differences.mean(), abs=1e-12)
            assert row['standard_error'] == pytest.approx(se, abs=1e-12) and (row['n_folds'], row['df']) == (15, 14)
            assert row['ci_high'] == pytest.approx(differences.mean() + stats.t.ppf(.975, 14) * se, abs=1e-12)
            assert row['p_approximate'] == pytest.approx(2 * stats.t.sf(abs(differences.mean() / se), 14), abs=1e-12)
            assert (row['stratum'], row['model_b'], row['family']) == (stratum[row['dataset']], model_b, family)
            p.append(row['p_approximate'])
        assert [row['holm_p_approximate'] for row in rows] == pytest.approx(holm(p), abs=1e-12)
        assert len(rows) == 10 and any(a != b for a, b in zip(holm(p), p))


def test_ladder_rows_read_the_five_rungs_from_each_verified_summary():
    runs = synthetic_runs(TEN[:2], accuracy_table)
    rows = cn.ladder_rows(runs, TEN[:2], stratum=dict.fromkeys(TEN[:2], 'H'), batch_of=dict.fromkeys(TEN[:2], 2))
    assert [(row['dataset'], row['rung'], row['model_id']) for row in rows] == [
        (name, rung, model) for name in TEN[:2] for rung, model in nd.LADDER]
    assert [model for _, model in nd.LADDER] == ['numeric_knn', 'projected_numeric_knn', INPUT, UNTRAINED, TRAINED]
    for row in rows:
        expected = next(s for s in runs[row['dataset']].summary['summaries'][row['dataset']] if s['model_id'] == row['model_id'])
        assert (row['mean_error'], row['outer_fold_sd'], row['mean_within_fold_seed_sd'], row['n_folds'], row['seeds_per_fold']) == (
            expected['mean'], expected['outer_fold_sd'], expected['mean_within_fold_seed_sd'], expected['n_folds'], expected['seeds_per_fold'])
        assert row['batch'] == 2 and list(row) == list(cn.LADDER_COLUMNS)


def test_comparator_rows_are_unadjusted_intervals_against_all_six_comparators_with_the_best_flagged():
    def accuracy(name, model, fold, seed):
        return accuracy_table(name, model, fold, seed) + (.05 if model == 'svc_rbf' else 0.)
    runs = synthetic_runs(TEN[:1], accuracy)
    rows = cn.comparator_rows(runs, TEN[:1], stratum={TEN[0]: 'H'}, batch_of={TEN[0]: 1}, q=.25, confidence=.95)
    assert [row['model_b'] for row in rows] == list(nd.COMPARATORS) == ['numeric_knn', 'svc_rbf', 'random_forest', 'mlp',
                                                                        'gradient_boosting', 'dummy']
    assert [row['best_comparator'] for row in rows] == [False, True, False, False, False, False]
    for row in rows:
        assert 'holm_p_approximate' not in row and row['status'] == cn.DESCRIPTIVE
        seeds = SEEDS[:1] if row['model_b'] in ('numeric_knn', 'svc_rbf', 'dummy') else SEEDS
        differences = [np.mean([accuracy(TEN[0], TRAINED, f, s) for s in SEEDS]) - np.mean([accuracy(TEN[0], row['model_b'], f, s) for s in seeds])
                       for f in FOLDS]
        assert row['mean_difference'] == pytest.approx(np.mean(differences), abs=1e-12)


def test_duplicate_audit_counts_groups_label_conflicts_and_training_copies_per_fold(tmp_path):
    protocol = nd.smoke_protocol(1)
    for name in nd.SMOKE_BATCHES['1']:
        nd.write_smoke_dataset(tmp_path, name, protocol)
    run = SimpleNamespace(path=tmp_path, protocol=protocol, schedule={'expected_folds': [(0, f) for f in range(3)]})
    rows, records = cn.duplicate_audit({name: run for name in nd.SMOKE_BATCHES['1']}, nd.SMOKE_BATCHES['1'],
                                       batch_of=dict.fromkeys(nd.SMOKE_BATCHES['1'], 1), pinned={'syn_c1': {'duplicate_rows': 1}})
    clean, duplicated = rows
    assert (clean['duplicate_rows'], clean['duplicate_groups'], clean['matches_pinned_counts']) == (0, 0, None)
    assert (duplicated['duplicate_rows'], duplicated['duplicate_groups'], duplicated['label_conflicting_groups']) == (1, 1, 1)
    assert duplicated['matches_pinned_counts'] is True and records['syn_c1']['groups'] == [{'sample_ids': [0, 1], 'labels': [0, 1]}]
    splits = json.loads((tmp_path/'syn_c1'/'splits.json').read_text())
    expected = sum(sum(1 for s in split['test'] if s in (0, 1) and ({0, 1} - {s}) <= set(split['train'])) for split in splits)
    assert duplicated['test_rows_with_training_duplicate'] == expected and duplicated['test_rows'] == 120
    assert [fold['n_with_training_duplicate'] for fold in records['syn_c1']['folds']] == [
        sum(1 for s in split['test'] if s in (0, 1) and ({0, 1} - {s}) <= set(split['train'])) for split in splits]


# ----------------------------------------------------------------------------- the command on two fixture batch runs

SYN = [entry['name'] for entry in nd.SMOKE_PANEL]
REAL_DESIGN = {'outer_folds': 5, 'outer_repeats': 3, 'inner_folds': 3}
CANDIDATES = {**tcr.CANDIDATES, **tcr.TRAINING_CANDIDATES, nd.PROJECTED_MODEL: tcr.TRAINING_CANDIDATES[INPUT]}
SEALED_HASHES = {source: hashlib.sha256(source.encode()).hexdigest() for source in cn.SEALED_SOURCES}
for _name in SYN:
    tcr.CLASSES.setdefault(_name, 3)
    tcr.PER_CLASS.setdefault(_name, 8)
tcr.STOCHASTIC.setdefault(nd.PROJECTED_MODEL, True)
tcr.ERROR_RATE.update({(name, model): rate for name, rates in {'syn_h1': (.1, .45, .3), 'syn_c1': (.2, .25, .2),
                                                               'syn_h2': (.15, .4, .35), 'syn_c2': (.3, .3, .25)}.items()
                       for model, rate in zip((TRAINED, UNTRAINED, INPUT), rates)})


def build_batch(root, number):
    rows = tcr.build_run(root, protocol=nd.smoke_protocol(number, design=REAL_DESIGN), models=nd.MODEL_ORDER,
                         revision='e0f898aa6' + '0' * 31, registry=nd.SMOKE_REGISTRY, candidates=CANDIDATES)
    tcr.rewrite(root/'environment.json', lambda e: e.update(source_hashes=dict(SEALED_HASHES)))
    return rows


@pytest.fixture(scope='session')
def batches(tmp_path_factory):
    root = tmp_path_factory.mktemp('newdata_batches')
    rows = {label: build_batch(root/label, number) for number, label in enumerate(cn.LABELS, start=1)}
    frozen = {number: tcr.sha256(root/label/'protocol.json') for number, label in enumerate(cn.LABELS, start=1)}
    record = cn.analyse(root/'batch1', root/'batch2', root/'out', frozen_protocols=frozen)
    return SimpleNamespace(root=root, rows=rows, frozen=frozen, record=record, out=root/'out')


@pytest.fixture
def copies(batches, tmp_path):
    for label in cn.LABELS:
        shutil.copytree(batches.root/label, tmp_path/label)
    return SimpleNamespace(batch1=tmp_path/'batch1', batch2=tmp_path/'batch2', frozen=dict(batches.frozen))


def test_analyse_computes_every_prespecified_output_from_both_verified_batches(batches):
    record, out = batches.record, batches.out
    assert sorted(path.name for path in out.iterdir()) == sorted(cn.OUTPUTS)
    rows = {**batches.rows['batch1'], **batches.rows['batch2']}
    batch_of = {name: int(key) for key, names in nd.SMOKE_BATCHES.items() for name in names}
    for family, model_b in (('primary', UNTRAINED), ('secondary', INPUT)):
        members = record[f'{family}_family']
        assert [r['dataset'] for r in members] == SYN and [r['batch'] for r in members] == [batch_of[n] for n in SYN]
        p = []
        for r in members:
            a, b = tcr.fold_means(rows[r['dataset']], TRAINED, 'accuracy'), tcr.fold_means(rows[r['dataset']], model_b, 'accuracy')
            differences = np.array([np.mean(a[key]) - np.mean(b[key]) for key in sorted(a)])
            se = np.sqrt((1 / 15 + .25) * np.var(differences, ddof=1))
            assert r['mean_difference'] == pytest.approx(differences.mean(), abs=1e-12) and (r['n_folds'], r['df']) == (15, 14)
            assert r['p_approximate'] == pytest.approx(2 * stats.t.sf(abs(differences.mean() / se), 14), abs=1e-12)
            p.append(r['p_approximate'])
        assert [r['holm_p_approximate'] for r in members] == pytest.approx(holm(p), abs=1e-12)
    effects = {r['dataset']: r['mean_difference'] for r in record['primary_family']}
    values = [effects[name] for name in SYN]
    observed = np.mean([effects['syn_h1'], effects['syn_h2']]) - np.mean([effects['syn_c1'], effects['syn_c2']])
    count = sum(np.mean([values[i] for i in g]) - np.mean([values[i] for i in range(4) if i not in g]) >= observed - 1e-12
                for g in combinations(range(4), 2))
    moderator = record['moderator_test']
    assert (moderator['splits'], moderator['at_least_observed'], moderator['p_one_sided']) == (6, count, count / 6)
    assert moderator['statistic'] == pytest.approx(observed, abs=1e-12) and moderator['p_at_most_alpha'] == (count / 6 <= .05)
    assert moderator['effect_sizes']['C']['effects'] == {'syn_c1': effects['syn_c1'], 'syn_c2': effects['syn_c2']}
    spearman = record['spearman']
    assert (spearman['datasets'], spearman['excluded'], spearman['permutations']) == (['syn_h1', 'syn_c1', 'syn_h2'], ['syn_c2'], 6)
    assert spearman['rho'] == pytest.approx(stats.spearmanr([9., 1., 6.], [effects[n] for n in spearman['datasets']]).statistic, abs=1e-12)
    assert [(r['dataset'], r['model_id']) for r in record['ladder']] == [(name, model) for name in SYN for _, model in nd.LADDER]
    for r in record['ladder']:
        seeds = SEEDS if tcr.STOCHASTIC[r['model_id']] else SEEDS[:1]
        expected = summarize_outer(rows[r['dataset']], r['model_id'], 'error', expected_folds=FOLDS, expected_seeds=seeds)
        assert (r['mean_error'], r['outer_fold_sd'], r['seeds_per_fold']) == pytest.approx(
            (expected['mean'], expected['outer_fold_sd'], expected['seeds_per_fold']), abs=1e-12)
    assert [(r['dataset'], r['model_b']) for r in record['comparator_intervals']] == [(n, m) for n in SYN for m in nd.COMPARATORS]
    assert all(sum(r['best_comparator'] for r in record['comparator_intervals'] if r['dataset'] == name) >= 1 for name in SYN)
    assert all(entry['counts']['duplicate_rows'] == 0 and entry['matches_pinned_counts'] is None
               for entry in record['duplicate_audit'].values())
    families = list(csv.DictReader(io.StringIO((out/cn.FAMILIES_CSV).read_text())))
    assert list(families[0]) == list(cn.FAMILY_COLUMNS) and [row['family'] for row in families] == ['primary'] * 4 + ['secondary'] * 4
    summary = json.loads((out/cn.ANALYSIS_JSON).read_text())
    assert all(summary['outputs'][name]['sha256'] == hashlib.sha256((out/name).read_bytes()).hexdigest() for name in cn.OUTPUTS[:-1])
    assert [summary['provenance']['verification'][label]['jobs_verified'] for label in cn.LABELS] == [300, 300]
    assert summary['provenance']['pairing']['pins_checked'] is False and summary['analysis_declaration'] == nd.smoke_protocol(1)['analysis']
    assert set(summary['provenance']['analysis_sources']) == {f'experiments/make_revision/{n}' for n in cn.ANALYSIS_SOURCES}


@pytest.mark.parametrize('label, removed, message', [
    ('batch2', 'summary.json', 'missing summary.json'), ('batch1', 'planned_jobs.json', 'missing planned_jobs.json'),
    ('batch1', 'result', 'planned result files or fit logs missing'), ('batch2', 'log', 'planned result files or fit logs missing'),
    ('batch2', 'directory', 'no such run directory')])
def test_analyse_reads_nothing_until_both_batches_are_complete(copies, tmp_path, monkeypatch, label, removed, message):
    target = getattr(copies, label)
    if removed == 'directory':
        shutil.rmtree(target)
    elif removed in ('result', 'log'):
        job = json.loads((target/'planned_jobs.json').read_text())[-1]
        (target/job['result_file' if removed == 'result' else 'log_file']).unlink()
    else:
        (target/removed).unlink()
    touched = []
    monkeypatch.setattr(cn, 'load_run', lambda *args: touched.append(args))
    monkeypatch.setattr(cn, 'verify_run', lambda *args: touched.append(args))
    with pytest.raises(RunComparisonError, match=message) as refused:
        cn.analyse(copies.batch1, copies.batch2, tmp_path/'out', frozen_protocols=copies.frozen)
    assert 'only after both batches are complete' in str(refused.value) and label in str(refused.value)
    assert not touched and not (tmp_path/'out').exists()


def test_analyse_lists_both_incomplete_batches(copies, tmp_path):
    for target in (copies.batch1, copies.batch2):
        (target/'summary.json').unlink()
    with pytest.raises(RunComparisonError) as refused:
        cn.analyse(copies.batch1, copies.batch2, tmp_path/'out', frozen_protocols=copies.frozen)
    assert str(refused.value).count('missing summary.json') == 2


def _protocol_change(root, change):
    tcr.rewrite(root/'protocol.json', change)
    return tcr.sha256(root/'protocol.json')


def _environment_change(root, change):
    tcr.rewrite(root/'environment.json', change)


@pytest.mark.parametrize('case, message', [
    ('swapped', 'must hold the frozen batch 1 protocol'), ('not_the_frozen_file', 'is not the frozen batch 1 protocol'),
    ('shared_key', 'differ outside the batch fields: status'), ('invalid_protocol', 'is not a newdata protocol'),
    ('sources_differ', 'seal different sources'), ('source_absent', 'must seal'), ('registry', 'must name the protocol registry'),
    ('tampered_prediction', 'validate_result_records')])
def test_analyse_refuses_batches_that_are_not_the_two_verified_batches_of_one_frozen_protocol(copies, tmp_path, case, message):
    first, second, frozen = copies.batch1, copies.batch2, copies.frozen
    if case == 'swapped':
        first, second, frozen = copies.batch2, copies.batch1, {1: frozen[2], 2: frozen[1]}
    elif case == 'not_the_frozen_file':
        frozen[1] = '0' * 64
    elif case == 'shared_key':
        frozen[2] = _protocol_change(copies.batch2, lambda p: p.update(status='edited'))
    elif case == 'invalid_protocol':
        frozen[1] = _protocol_change(copies.batch1, lambda p: p['analysis']['moderator_test'].update(alpha=.1))
    elif case == 'sources_differ':
        _environment_change(copies.batch2, lambda e: e['source_hashes'].update({cn.SEALED_SOURCES[0]: '9' * 64}))
    elif case == 'source_absent':
        _environment_change(copies.batch1, lambda e: e['source_hashes'].pop('experiments/make_revision/newdata.py'))
    elif case == 'registry':
        _environment_change(copies.batch2, lambda e: e.update(registry=nd.REGISTRY))
    else:
        tcr.rewrite_result(copies.batch1, 0, lambda r: r['predictions'][0].update(y_pred=(r['predictions'][0]['y_pred'] + 1) % 3))
    with pytest.raises(RunComparisonError, match=message):
        cn.analyse(first, second, tmp_path/'out', frozen_protocols=frozen)
    assert not (tmp_path/'out').exists()


def production_runs():
    """Run-like records of two production batch protocols with the registry's candidates and pinned manifests."""
    split = {'1': TEN[:5], '2': TEN[5:]}
    projection = {'cap_hours': 9, 'batches': split, 'decision_hours': {'1': 5., '2': 6.}}
    protocols = {n: nd.batch_protocol(nd.base_protocol(), n, split, frozen_at_utc='2026-09-13T20:00:00+00:00',
                                      resource_decision='test', batch_projection=projection) for n in (1, 2)}
    record = nd.candidate_record(nd.build_registry(protocols[1]))
    return {label: SimpleNamespace(label=label, protocol=protocols[n], protocol_sha256=str(n) * 64, candidates=json.loads(json.dumps(record)),
                                   environment={'registry': nd.REGISTRY, 'code_revision': 'c', 'source_hashes': dict(SEALED_HASHES)},
                                   registry=dict.fromkeys(nd.MODEL_ORDER),
                                   manifests={name: {key: nd.PIN_BY_NAME[name][key] for key in ('dataset_hash', 'splits_hash')}
                                              for name in protocols[n]['datasets']})
            for n, label in enumerate(cn.LABELS, start=1)}


def test_check_batches_on_production_protocols_checks_the_registry_candidates_and_every_dataset_pin():
    frozen = {1: '1' * 64, 2: '2' * 64}
    runs = production_runs()
    record = cn.check_batches(runs, frozen)
    assert record['pins_checked'] is True and sorted(record['datasets']) == sorted(TEN)
    runs['batch2'].manifests[TEN[7]] = dict(runs['batch2'].manifests[TEN[7]], splits_hash='0' * 16)
    with pytest.raises(RunComparisonError, match=f'{TEN[7]} dataset or splits hash differs from its pin'):
        cn.check_batches(runs, frozen)
    runs = production_runs()
    for run in runs.values():
        run.candidates['dummy']['candidates'] = [{'strategy': 'prior'}]
    with pytest.raises(RunComparisonError, match='differ from the newdata registry'):
        cn.check_batches(runs, frozen)
    runs = production_runs()
    runs['batch1'].candidates['svc_rbf']['stochastic'] = True
    with pytest.raises(RunComparisonError, match='different candidates'):
        cn.check_batches(runs, frozen)
    runs = production_runs()
    runs['batch2'].registry.pop('dummy')
    with pytest.raises(RunComparisonError, match='ten newdata models'):
        cn.check_batches(runs, frozen)


def test_frozen_protocol_hashes_are_the_committed_files_and_their_absence_is_refused(tmp_path, monkeypatch):
    files = {1: tmp_path/'newdata_batch1.json', 2: tmp_path/'newdata_batch2.json'}
    monkeypatch.setattr(cn, 'PROTOCOL_FILES', files)
    with pytest.raises(RunComparisonError, match='is not in this tree'):
        cn.frozen_protocol_hashes()
    for number, path in files.items():
        path.write_text(f'{{"batch": {number}}}\n')
    assert cn.frozen_protocol_hashes() == {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in files.items()}


def test_the_commands_end_to_end_with_identical_reruns_and_refusals_exiting_2(batches, copies, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cn, 'frozen_protocol_hashes', lambda: dict(batches.frozen))
    cn.main(['pairing', '--batch1', str(copies.batch1), '--batch2', str(copies.batch2)])
    assert 'paired arrowflow-v3-newdata-batch1-1-synthetic-smoke and arrowflow-v3-newdata-batch2-1-synthetic-smoke: 4 datasets' in capsys.readouterr().out
    arguments = ['analyse', '--batch1', str(copies.batch1), '--batch2', str(copies.batch2), '--output', str(tmp_path/'out')]
    cn.main(arguments)
    printed = capsys.readouterr().out
    assert printed.count('primary syn_') == 4 and printed.count('secondary syn_') == 4 and 'exact one-sided p = ' in printed
    assert all((tmp_path/'out'/name).read_bytes() == (batches.out/name).read_bytes() for name in cn.OUTPUTS[:-1])
    cn.main(arguments)                                                        # an identical rerun is accepted
    capsys.readouterr()
    (tmp_path/'out'/cn.LADDER_CSV).write_text('edited\n')
    with pytest.raises(SystemExit) as refused:
        cn.main(arguments)
    assert refused.value.code == 2 and 'compare_newdata analyse refused' in capsys.readouterr().err
    (copies.batch2/'summary.json').unlink()
    with pytest.raises(SystemExit) as incomplete:
        cn.main(['analyse', '--batch1', str(copies.batch1), '--batch2', str(copies.batch2), '--output', str(tmp_path/'other')])
    assert incomplete.value.code == 2 and 'only after both batches are complete' in capsys.readouterr().err
    assert not (tmp_path/'other').exists()
