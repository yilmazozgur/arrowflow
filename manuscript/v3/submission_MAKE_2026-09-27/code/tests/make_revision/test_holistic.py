"""Task 24 Part 1: holistic, the combined analysis of the benchmark and further datasets.

The rules, the rank record and the Holm sensitivity are checked on constructed inputs with known answers. The commands are
checked end to end on a fixture chain in the registered run layout (test_compare_runs.build_run runs, which the harness
validators accept): a bridge and a bridge_knn run over two datasets, a knn_training and a knn_projected run pinning them,
two newdata batch runs over the four-dataset synthetic panel, and every published analysis the combined layer reads
(compare_runs knn and training, compare_projected, referee_analyses comparators, ranks and duplicates, compare_newdata
analyse). One chain and its outputs are built once per session; tests that alter records work on a private copy."""
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import stats
import test_compare_runs as tcr
import test_compare_newdata as tcn          # registers the synthetic newdata panel with the run builder
import test_compare_projected as tcp
from experiments.make_revision import compare_newdata as cn
from experiments.make_revision import compare_projected as cp
from experiments.make_revision import compare_runs as cr
from experiments.make_revision import holistic as h
from experiments.make_revision import knn_controls as kc
from experiments.make_revision import newdata as nd
from experiments.make_revision import referee_analyses as ra
from experiments.make_revision import run_knn_ablation as rka
from experiments.make_revision import run_newdata_ablation as rna
from experiments.make_revision.compare_runs import RunComparisonError
from experiments.make_revision.evaluation import holm_adjust, paired_corrected_interval

TRAINED, UNTRAINED, INPUT = h.TRAINED, h.UNTRAINED, h.INPUT
CANDIDATES = {**tcn.CANDIDATES, 'arrowflow_full': tcr.TRAINING_KNN_CANDIDATES}
DEEP = next(i for i, c in enumerate(tcr.TRAINING_KNN_CANDIDATES) if c['widths'] == [64, 128])
BENCHMARK_DEEP = [(r, f) for r, f in tcr.FOLDS if f % 2 == 0]
FURTHER_DEEP = [(r, f) for r, f in tcr.FOLDS if (r + f) % 3 == 0]
COPIES = {'alpha': {1: 0}, 'syn_h1': {3: 2}}              # exact feature copies within one class
REAL_RUNS = h.WORKSPACE_RUNS
needs_real_runs = pytest.mark.skipif(not all((REAL_RUNS/h.SOURCES[key]).is_dir() for key in h.SOURCES),
                                     reason='the Task 24 source runs are not on this machine')


# ----------------------------------------------------------------------------- the fixed rules

def errors(arrowflow, majority=50., default=40., **classical):
    """Mean errors in points for MAIN_MODELS (fractions returned); unnamed classical models sit at `default` points."""
    values = {model: classical.get(model, default) for model in h.CLASSICAL}
    return {TRAINED: arrowflow / 100, h.MAJORITY: majority / 100, **{m: v / 100 for m, v in values.items()}}


@pytest.mark.parametrize('case, arrowflow, majority, classical, expected', [
    ('ceiling', 1.2, 45., {'mlp': .9}, dict(ceiling=True, near_majority=False, no_learning=False, within_three_points=True, gap_points=.3)),
    ('best_error_exactly_one_is_not_ceiling', 1.5, 45., {'mlp': 1.}, dict(ceiling=False)),
    ('near_majority', 8.33, 8.52, {'svc_rbf': 5.37}, dict(near_majority=True, ceiling=False, no_learning=False, gap_points=2.96,
                                                          within_three_points=True, within_three_points_as_printed=True)),
    ('near_majority_boundaries_inclusive', 10.5, 10., {'svc_rbf': 8.5}, dict(near_majority=True, majority_distance_points=.5)),
    ('majority_distance_above_half_a_point', 10.51, 10., {'svc_rbf': 8.}, dict(near_majority=False)),
    ('advantage_below_two_points', 10.2, 10., {'svc_rbf': 8.21}, dict(near_majority=False, gap_points=1.99)),
    ('no_learning', 75.60, 73.86, {'numeric_knn': 74.10, 'default': 80.}, dict(no_learning=True, near_majority=False, gap_points=1.5)),
    ('no_learning_on_a_tie', 30., 20., {'mlp': 20.}, dict(no_learning=True)),
    ('not_within_three', 19.91, 50., {'svc_rbf': 16.51}, dict(within_three_points=False, degenerate=False, best_on=False)),
    ('within_three_boundary', 23., 50., {'mlp': 20.}, dict(within_three_points=True, gap_points=3.)),
    ('best_on', 10., 50., {'mlp': 10.5}, dict(best_on=True, gap_points=-.5)),
    ('a_tie_is_not_best_on', 10., 50., {'mlp': 10., 'svc_rbf': 10.}, dict(best_on=False, gap_points=0., best_classical=['svc_rbf', 'mlp'])),
])
def test_competitiveness_follows_the_fixed_rules(case, arrowflow, majority, classical, expected):
    result = h.competitiveness(errors(arrowflow, majority, **classical))
    for key, value in expected.items():
        assert (result[key] == pytest.approx(value) if isinstance(value, float) else result[key] == value), (case, key, result[key])
    assert result['degenerate'] == (result['ceiling'] or result['near_majority'] or result['no_learning'])


def test_within_three_points_respects_printed_precision_and_float_noise():
    noisy = h.competitiveness({TRAINED: .2291, h.MAJORITY: .5, **dict.fromkeys(h.CLASSICAL, .4), 'mlp': .1991})
    assert noisy['gap_points'] == 3. and noisy['within_three_points'] and noisy['printed_precision_agrees']   # 22.91 - 19.91
    rounds_down = h.competitiveness({TRAINED: .23004, h.MAJORITY: .5, **dict.fromkeys(h.CLASSICAL, .4), 'mlp': .2})
    assert not rounds_down['within_three_points'] and rounds_down['gap_points_2dp'] == 3.
    assert not rounds_down['within_three_points_as_printed'] and not rounds_down['printed_precision_agrees']
    one_decimal = h.competitiveness({TRAINED: .2996, h.MAJORITY: .5, **dict.fromkeys(h.CLASSICAL, .4), 'mlp': .2696})
    assert one_decimal['within_three_points'] and one_decimal['gap_points_from_1dp_errors'] == 3.


def test_counts_list_the_within_three_datasets_with_their_degenerate_flags():
    rows = [dict(dataset=name, group=group, **h.competitiveness(errors(values[0], values[1], **values[2])))
            for name, group, values in (('a', 'benchmark', (12., 50., {'mlp': 10.})), ('b', 'benchmark', (20., 50., {'mlp': 10.})),
                                        ('c', 'further', (8.33, 8.52, {'svc_rbf': 5.37})), ('d', 'further', (.2, 45., {'mlp': 0.})),
                                        ('e', 'further', (9., 50., {'mlp': 9.5})))]
    counts = h.competitiveness_counts(rows)
    assert counts['within_three_points'] == {'all': ['a', 'c', 'd', 'e'], 'benchmark': ['a'], 'further': ['c', 'd', 'e']}
    assert counts['within_three_points_and_degenerate'] == ['c', 'd'] and counts['within_three_points_and_not_degenerate'] == ['a', 'e']
    assert counts['degenerate'] == {'ceiling': ['d'], 'near_majority': ['c'], 'no_learning': []}
    assert counts['best_on'] == ['e'] and counts['trails_best_classical'] == ['a', 'b', 'c', 'd']
    assert counts['summary'].startswith('within three points on 4 of 5 datasets (1 benchmark, 3 further); 2 of them')


# ----------------------------------------------------------------------------- Friedman on known matrices

DEMSAR_Q05_K6 = 2.850


def test_rank_record_over_seventeen_datasets_on_known_matrices():
    names = [f'd{i}' for i in range(17)]
    agreement = h.rank_record([[.01 * (j + 1) for j in range(6)] for _ in names], names)
    assert agreement['mean_ranks'] == dict(zip(h.RANK_MODELS, [1., 2., 3., 4., 5., 6.]))
    assert agreement['friedman']['chi2'] == pytest.approx(85.) and agreement['friedman']['df'] == 5
    assert agreement['friedman']['p'] == pytest.approx(stats.chi2.sf(85., 5)) and agreement['friedman']['iman_davenport_f'] is None
    assert agreement['nemenyi']['critical_difference'] == pytest.approx(DEMSAR_Q05_K6 * np.sqrt(6 * 7 / (6 * 17)), abs=1e-3)
    assert len(agreement['nemenyi']['pairs_exceeding_critical_difference']) == 10          # pairs at least two ranks apart
    rows = [[.01 * (j + 1) for j in range(6)] if i < 9 else [.01 * (6 - j) for j in range(6)] for i in range(17)]
    alternating = h.rank_record(rows, names)
    mean_ranks = [(9 * (j + 1) + 8 * (6 - j)) / 17 for j in range(6)]
    chi2 = 12 * 17 / (6 * 7) * (sum(r * r for r in mean_ranks) - 6 * 49 / 4)
    assert list(alternating['mean_ranks'].values()) == pytest.approx(mean_ranks)
    assert alternating['friedman']['chi2'] == pytest.approx(chi2) == pytest.approx(stats.friedmanchisquare(*np.asarray(rows).T).statistic)
    f_value = 16 * chi2 / (17 * 5 - chi2)
    assert alternating['friedman']['iman_davenport_f'] == pytest.approx(f_value)
    assert alternating['friedman']['iman_davenport_p'] == pytest.approx(stats.f.sf(f_value, 5, 80))
    assert alternating['nemenyi']['pairs_exceeding_critical_difference'] == [] and alternating['ties']['datasets_with_tied_errors'] == []
    tied = h.rank_record([[.1, .1, .2, .3, .4, .5]] + [[.1, .2, .3, .4, .5, .6]] * 16, names)
    assert tied['ranks']['d0'][h.RANK_MODELS[0]] == 1.5 and tied['ties']['datasets_with_tied_errors'] == ['d0']
    shown = h.rank_record([[.10001, .10004, .2, .3, .4, .5]] + [[.1, .2, .3, .4, .5, .6]] * 16, names)
    assert shown['ties']['datasets_with_tied_errors'] == [] and shown['ties']['datasets_with_tied_errors_as_displayed'] == ['d0']


# ----------------------------------------------------------------------------- the Holm sensitivity

def test_holm34_is_one_adjustment_over_every_contrast_and_lists_the_survivors():
    p = [.001, .2, .0004, .03, .5, .00001, .01, .9] * 4 + [.002, .04]
    rows = [{'dataset': f'd{i // 2}', 'group': 'benchmark' if i < 14 else 'further', 'control': (UNTRAINED, INPUT)[i % 2],
             'contrast': 'c', 'mean_difference': .01, 'ci_low': 0., 'ci_high': .02, 'p_unadjusted': value,
             'registered_family': 'f', 'registered_holm_p': min(1., 10 * value)} for i, value in enumerate(p)]
    table, record = h.holm34(rows)
    independent = tcn.holm(p)
    assert [row['holm34_p'] for row in table] == pytest.approx(independent) and record['members'] == 34
    assert [(e['dataset'], e['control']) for e in record['below_alpha']] == [(r['dataset'], r['control']) for r, v in zip(rows, independent) if v < .05]
    # sorted: four .00001, four .0004 and four .001 stay below .05, then .002 x 22 = .044 does, .01 x 21 = .21 does not
    assert len(record['below_alpha']) == 13 and all(row['status'] == h.POST_HOC for row in table)
    assert [row['registered_holm_p'] for row in table] == [row['registered_holm_p'] for row in rows]


# ----------------------------------------------------------------------------- the fixture chain

def build_chain(root):
    """Every source run and published analysis under its registered name in `root`; returns the frozen batch hashes."""
    paths = h.source_paths(root)
    knn, bridge, training, projected = paths['knn'], root/'bridge', paths['training'], paths['projected']
    bridge_p = json.loads((tcr.PROTOCOLS/'bridge.json').read_text())
    bridge_p.update(datasets=list(tcr.DATASETS))
    tcr.build_run(bridge, protocol=bridge_p, models=tcr.BRIDGE_MODELS, revision=tcr.BRIDGE_REVISION,
                  registry='experiments.make_revision.bridge:bridge_registry', candidates=CANDIDATES)
    knn_p = json.loads((tcr.PROTOCOLS/'bridge_knn.json').read_text())
    knn_p.update(datasets=list(tcr.DATASETS), primary_family_size=len(tcr.DATASETS))
    knn_p['knn_readout']['reference'].update(code_revision=tcr.BRIDGE_REVISION, protocol_sha256=tcr.sha256(bridge/'protocol.json'),
                                             summary_sha256=tcr.sha256(bridge/'summary.json'))
    tcr.build_run(knn, protocol=knn_p, models=tcr.KNN_MODELS, revision=tcr.KNN_REVISION,
                  registry='experiments.make_revision.bridge:bridge_knn_registry', candidates=CANDIDATES,
                  config_choice={(name, TRAINED, r, f): DEEP for name in tcr.DATASETS for r, f in BENCHMARK_DEEP})
    tcr.rewrite(knn/'environment.json', lambda e: e.update(source_hashes=dict(tcr.SHARED_HASHES)))
    training_p = json.loads((tcr.PROTOCOLS/'knn_training.json').read_text())
    training_p.update(datasets=list(tcr.DATASETS), primary_family_size=2 * len(tcr.DATASETS), frozen=True,
                      frozen_at_utc='2026-09-13T12:00:00+00:00')
    training_p['training_controls']['reference'].update(code_revision=tcr.KNN_REVISION, protocol_sha256=tcr.sha256(knn/'protocol.json'),
                                                        summary_sha256=tcr.sha256(knn/'summary.json'))
    shallow = next(i for i, c in enumerate(CANDIDATES[UNTRAINED]) if c['widths'] == [128])
    tcr.build_run(training, protocol=training_p, models=tcr.CONTROLS, revision=tcr.TRAINING_REVISION,
                  registry='experiments.make_revision.knn_controls:knn_training_registry', candidates=CANDIDATES,
                  config_choice={(name, UNTRAINED, r, f): shallow for name in tcr.DATASETS for r, f in tcr.FOLDS})
    tcr.rewrite(training/'environment.json', lambda e: e.update(source_hashes=dict(tcp.TRAINING_HASHES)))
    tcp.build_projected(projected, knn, training)
    for number, label in enumerate(cn.LABELS, start=1):
        tcr.build_run(paths[label], protocol=nd.smoke_protocol(number, design=tcn.REAL_DESIGN), models=nd.MODEL_ORDER,
                      revision='e0f898aa6' + '0' * 31, registry=nd.SMOKE_REGISTRY, candidates=tcn.CANDIDATES,
                      config_choice={(name, TRAINED, r, f): DEEP for name in nd.SMOKE_BATCHES[str(number)] for r, f in FURTHER_DEEP})
        tcr.rewrite(paths[label]/'environment.json', lambda e: e.update(source_hashes=dict(tcn.SEALED_HASHES)))
    frozen = {number: tcr.sha256(paths[label]/'protocol.json') for number, label in enumerate(cn.LABELS, start=1)}
    cr.compare_knn(knn, bridge, paths['knn_vs_full'])
    cr.compare_training(training, knn, paths['training_compare'])
    cp.compare_projected(projected, knn, training, paths['projected_compare'])
    ra.analyse_comparators(knn, paths['referee']/'comparators')
    ra.analyse_ranks(knn, paths['referee']/'ranks')
    ra.analyse_duplicates(knn, bridge, paths['referee']/'duplicates')
    cn.analyse(paths['batch1'], paths['batch2'], paths['newdata_analysis'], frozen_protocols=frozen)
    return frozen


@pytest.fixture(scope='session')
def chain(tmp_path_factory):
    root = tmp_path_factory.mktemp('holistic_chain')
    original = tcr.synthetic_dataset

    def with_copies(name, shift=0.):
        X, y = original(name, shift)
        for row, source in COPIES.get(name, {}).items():
            X[row] = X[source]
        return X, y
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tcr, 'synthetic_dataset', with_copies)
        frozen = build_chain(root/'runs')
    paths = h.source_paths(root/'runs')
    return SimpleNamespace(root=root, runs=root/'runs', paths=paths, frozen=frozen,
                           benchmark=h.benchmark(paths, root/'out'/'benchmark', frozen_protocols=frozen),
                           training=h.training(paths, root/'out'/'training', frozen_protocols=frozen), out=root/'out')


@pytest.fixture
def copy(chain, tmp_path):
    shutil.copytree(chain.runs, tmp_path/'runs')
    return SimpleNamespace(runs=tmp_path/'runs', paths=h.source_paths(tmp_path/'runs'), frozen=dict(chain.frozen), out=tmp_path/'out')


def read_csv(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text())))


def load(path):
    return json.loads(Path(path).read_text())


def panel_rows(chain, run_label, name, model):
    return [row for row in load(chain.paths[run_label]/'summary.json')['model_rows'][name] if row['model_id'] == model]


def run_label(name, model):
    if name in tcr.DATASETS:
        return 'training' if model in kc.CONTROL_MODELS else 'projected' if model == nd.PROJECTED_MODEL else 'knn'
    return 'batch1' if name in nd.SMOKE_BATCHES['1'] else 'batch2'


DATASETS = list(tcr.DATASETS) + [entry['name'] for entry in nd.SMOKE_PANEL]


def independent_interval(chain, name, model_a, model_b):
    rows = panel_rows(chain, run_label(name, model_a), name, model_a) + panel_rows(chain, run_label(name, model_b), name, model_b)
    seeds = {m: sorted({row['model_seed'] for row in rows if row['model_id'] == m}, key=tcr.SEEDS.index) for m in (model_a, model_b)}
    means = {m: {fold: np.mean([row['accuracy'] for row in rows if row['model_id'] == m and (row['outer_repeat'], row['outer_fold']) == fold])
                 for fold in tcr.FOLDS} for m in (model_a, model_b)}
    differences = np.array([means[model_a][fold] - means[model_b][fold] for fold in tcr.FOLDS])
    se = np.sqrt((1 / 15 + .25) * np.var(differences, ddof=1))
    return differences.mean(), se, 2 * stats.t.sf(abs(differences.mean() / se), 14)


def test_benchmark_combines_both_panels_from_the_verified_runs_and_agrees_with_every_published_analysis(chain):
    record, out = chain.benchmark, chain.out/'benchmark'
    assert record['datasets'] == {'all': DATASETS, 'benchmark': list(tcr.DATASETS), 'further': DATASETS[2:]}
    assert sorted(p.name for p in out.iterdir()) == sorted(h.BENCHMARK_OUTPUTS)
    verification = record['provenance']['verification']
    assert {label: verification[label]['jobs_verified'] for label in h.RUN_LABELS} == {'knn': 210, 'training': 60, 'projected': 30,
                                                                                         'batch1': 300, 'batch2': 300}
    assert record['provenance']['runs']['knn']['summary_sha256'] == tcr.sha256(chain.paths['knn']/'summary.json')
    assert set(record['provenance']['published_analyses']) == set(h.BENCHMARK_PUBLISHED)
    assert set(record['provenance']['analysis_sources']) == {f'experiments/make_revision/{name}' for name in h.ANALYSIS_SOURCES}
    main = read_csv(out/'main_table.csv')
    assert [(r['dataset'], r['model_id']) for r in main] == [(name, model) for name in DATASETS for model in h.MAIN_MODELS]
    table = load(chain.paths['knn_vs_full']/'main_table.json')['rows']
    comparators = {(r['dataset'], r['model_b']): r for r in read_csv(chain.paths['newdata_analysis']/'newdata_comparators.csv')}
    for row in main:
        if row['group'] == 'benchmark':
            expected = next(e for e in table[row['dataset']] if e['model_id'] == row['model_id'])['mean_error']
        elif row['model_id'] == TRAINED:
            expected = float(comparators[(row['dataset'], 'dummy')]['mean_error_a'])
        else:
            expected = float(comparators[(row['dataset'], row['model_id'])]['mean_error_b'])
        assert float(row['mean_error']) == pytest.approx(expected, abs=1e-15) and row['source_run'] == run_label(row['dataset'], row['model_id'])
    intervals = read_csv(out/'comparator_intervals.csv')
    assert [(r['dataset'], r['model_b']) for r in intervals] == [(name, model) for name in DATASETS for model in h.CLASSICAL]
    for row in intervals:
        mean, se, p = independent_interval(chain, row['dataset'], TRAINED, row['model_b'])
        assert float(row['mean_difference']) == pytest.approx(mean, abs=1e-12) and float(row['standard_error']) == pytest.approx(se, abs=1e-12)
        assert float(row['p_unadjusted']) == pytest.approx(p, abs=1e-12) and row['status'] == h.DESCRIPTIVE
    competitive = {row['dataset']: row for row in record['competitiveness']}
    for name in DATASETS:
        errors_ = {model: float(next(r for r in main if (r['dataset'], r['model_id']) == (name, model))['mean_error']) for model in h.MAIN_MODELS}
        assert competitive[name]['gap_points'] == pytest.approx(100 * (errors_[TRAINED] - min(errors_[m] for m in h.CLASSICAL)), abs=1e-8)
        assert competitive[name]['best_classical'] == '+'.join(h.competitiveness(errors_)['best_classical'])
    assert record['counts']['datasets'] == 6 and set(record['counts']['degenerate']) == {'ceiling', 'near_majority', 'no_learning'}
    ranks = record['ranks']
    matrix = [[float(next(r for r in main if (r['dataset'], r['model_id']) == (name, model))['mean_error']) for model in h.RANK_MODELS]
              for name in DATASETS]
    assert ranks['friedman'] == pytest.approx(h.rank_record(matrix, DATASETS)['friedman']) and ranks['friedman']['n_datasets'] == 6
    assert len(read_csv(out/'rank_matrix.csv')) == 36 and [r['model_id'] for r in read_csv(out/'mean_ranks.csv')] == list(h.RANK_MODELS)
    metrics = read_csv(out/'complete_metrics.csv')
    assert [(r['dataset'], r['model_id']) for r in metrics] == [(name, model) for name in DATASETS for model in nd.MODEL_ORDER]
    projected_row = next(r for r in metrics if (r['dataset'], r['model_id']) == ('beta', nd.PROJECTED_MODEL))
    assert projected_row['source_run'] == 'projected' and float(projected_row['macro_f1']) == next(
        s['mean'] for s in load(chain.paths['projected']/'summary.json')['summaries']['beta'] if s['metric'] == 'macro_f1')
    widths = read_csv(out/'selected_widths.csv')
    assert len(widths) == 6 * 15 and record['selected_widths']['outer_folds'] == {'all': 90, 'benchmark': 30, 'further': 60}
    deep = {name: BENCHMARK_DEEP if name in tcr.DATASETS else FURTHER_DEEP for name in DATASETS}
    assert all((r['widths'] == '[64, 128]') == ((int(r['outer_repeat']), int(r['outer_fold'])) in deep[r['dataset']]) for r in widths)
    assert record['selected_widths']['totals']['all'] == {'[128]': 90 - 2 * len(BENCHMARK_DEEP) - 4 * len(FURTHER_DEEP),
                                                          '[64, 128]': 2 * len(BENCHMARK_DEEP) + 4 * len(FURTHER_DEEP)}
    duplicates = {row['dataset']: row for row in read_csv(out/'duplicates.csv')}
    assert list(duplicates) == DATASETS and duplicates['alpha']['duplicate_rows'] == duplicates['syn_h1']['duplicate_rows'] == '1'
    assert duplicates['beta']['duplicate_rows'] == '0' and float(duplicates['alpha']['share_with_training_duplicate']) > 0
    assert (duplicates['alpha']['source_audit'], duplicates['syn_h1']['source_audit']) == ('duplicate_groups.csv', 'newdata_duplicates.csv')
    for name, seal in record['outputs'].items():
        assert seal['sha256'] == tcr.sha256(out/name)


def test_training_combines_the_registered_families_with_their_holm_p_and_the_post_hoc_holm_over_all_members(chain):
    record, out = chain.training, chain.out/'training'
    assert sorted(p.name for p in out.iterdir()) == sorted(h.TRAINING_OUTPUTS)
    assert {key: family['size'] for key, family in record['families'].items()} == {'knn_training_primary': 4, 'newdata_primary': 4,
                                                                                    'newdata_secondary': 4}
    contrasts = record['contrasts']
    assert [(r['dataset'], r['control']) for r in contrasts] == [(name, control) for name in DATASETS for control in kc.CONTROL_MODELS]
    published = {int(r['family_index']): r for r in read_csv(chain.paths['training_compare']/'training_contrasts.csv')}
    families = {(r['family'], r['dataset']): r for r in read_csv(chain.paths['newdata_analysis']/'newdata_families.csv')}
    for row in contrasts:
        mean, se, p = independent_interval(chain, row['dataset'], TRAINED, row['control'])
        assert row['mean_difference'] == pytest.approx(mean, abs=1e-12) and row['p_unadjusted'] == pytest.approx(p, abs=1e-12)
        if row['group'] == 'benchmark':
            source = published[row['registered_family_index']]
            assert row['registered_family'] == 'knn_training_primary' and source['model_b'] == row['control']
        else:
            source = families[(h.NEWDATA_FAMILY[row['control']], row['dataset'])]
            assert row['registered_family'] == f"newdata_{h.NEWDATA_FAMILY[row['control']]}"
        assert row['registered_holm_p'] == pytest.approx(float(source['holm_p_approximate']), abs=1e-15)
    for family in record['families']:
        members = [row for row in contrasts if row['registered_family'] == family]
        assert [row['registered_holm_p'] for row in members] == pytest.approx(tcn.holm([row['p_unadjusted'] for row in members]))
    wide = read_csv(out/'training_controls.csv')
    assert [row['dataset'] for row in wide] == DATASETS and list(wide[0]) == list(h.TRAINING_COLUMNS)
    assert float(wide[2]['input_registered_holm_p']) == pytest.approx(families[('secondary', DATASETS[2])]['holm_p_approximate'] and
                                                                      float(families[('secondary', DATASETS[2])]['holm_p_approximate']))
    sensitivity = read_csv(out/'holm34_sensitivity.csv')
    assert [float(row['holm34_p']) for row in sensitivity] == pytest.approx(tcn.holm([row['p_unadjusted'] for row in contrasts]))
    assert record['holm34']['members'] == 12 and all(row['status'] == h.POST_HOC for row in sensitivity)
    assert record['holm34']['below_alpha'] == [{'dataset': r['dataset'], 'control': r['control'], 'holm34_p': float(r['holm34_p'])}
                                               for r in sensitivity if float(r['holm34_p']) < .05]
    ladder = read_csv(out/'ladder.csv')
    assert list(ladder[0]) == ['dataset', 'group', *(rung for rung, _ in nd.LADDER)] and len(ladder) == 6
    projected = load(chain.paths['projected_compare']/'projected_ladder_error_table.json')['rows']['alpha']
    assert float(ladder[0]['unsorted_projected_knn']) == next(r['mean_error'] for r in projected if r['model_id'] == nd.PROJECTED_MODEL)
    newdata_ladder = {(r['dataset'], r['rung']): r for r in read_csv(chain.paths['newdata_analysis']/'newdata_ladder.csv')}
    assert ladder[3]['encoded_ranking_knn'] == newdata_ladder[(DATASETS[3], 'encoded_ranking_knn')]['mean_error']
    analysis = load(chain.paths['newdata_analysis']/'newdata_analysis.json')
    stratum = record['stratum_test']
    assert stratum['moderator_test'] == {k: v for k, v in analysis['moderator_test'].items() if k != 'null_distribution'}
    assert stratum['spearman'] == analysis['spearman'] and stratum['source']['sha256'] == tcr.sha256(chain.paths['newdata_analysis']/'newdata_analysis.json')
    depth = record['depth_split']
    published_depth = load(chain.paths['training_compare']/'training_depth_split.json')['by_dataset']
    assert all(h.same(published_depth[name], depth['by_dataset'][name]) for name in tcr.DATASETS)
    for name in DATASETS[2:]:
        label = run_label(name, TRAINED)
        rows = panel_rows(chain, label, name, TRAINED) + panel_rows(chain, label, name, UNTRAINED)
        expected = kc.depth_split(rows, TRAINED, UNTRAINED, kc.selected_widths(rows, TRAINED), depths=[[128], [64, 128]],
                                  folds=tcr.FOLDS, seeds={TRAINED: tcr.SEEDS, UNTRAINED: tcr.SEEDS}, q=.25, confidence=.95)
        assert [e['n_folds'] for e in depth['by_dataset'][name]] == [15 - len(FURTHER_DEEP), len(FURTHER_DEEP)]
        assert all(h.same({k: v for k, v in e.items() if k != 'untrained_selected_the_same_widths'}, x)
                   for e, x in zip(depth['by_dataset'][name], expected))
    assert record['counts']['depth_folds']['all'] == {'[128]': 90 - 2 * len(BENCHMARK_DEEP) - 4 * len(FURTHER_DEEP),
                                                      '[64, 128]': 2 * len(BENCHMARK_DEEP) + 4 * len(FURTHER_DEEP)}
    pooled = read_csv(out/'depth_pooled.csv')
    assert [(r['scope'], r['widths']) for r in pooled] == [(s, w) for s in ('all', 'benchmark', 'further') for w in ('[128]', '[64, 128]')]


# ----------------------------------------------------------------------------- refusals

def reseal(record_path, csv_path):
    record = load(record_path)
    record['outputs'][Path(csv_path).name]['sha256'] = tcr.sha256(csv_path)
    Path(record_path).write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def edit_csv(path, row_index, field, value):
    rows = read_csv(path)
    rows[row_index][field] = value
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    Path(path).write_text(buffer.getvalue())


def tamper(copy, case):
    paths = copy.paths
    if case == 'comparator_csv_changed_and_resealed':
        path = paths['referee']/'comparators'/'comparator_contrasts.csv'
        edit_csv(path, 1, 'p_unadjusted', '0.5')
        reseal(paths['referee']/'comparators'/'comparator_contrasts.json', path)
    elif case == 'comparator_csv_not_the_sealed_file':
        edit_csv(paths['referee']/'comparators'/'comparator_contrasts.csv', 1, 'p_unadjusted', '0.5')
    elif case == 'main_table_from_another_run':
        tcr.rewrite(paths['knn_vs_full']/'main_table.json', lambda r: r['sources']['knn'].update(summary_sha256='0' * 64))
    elif case == 'training_reference_pins_another_knn_run':
        tcr.rewrite(paths['training']/'protocol.json', lambda p: p['training_controls']['reference'].update(summary_sha256='0' * 64))
    elif case == 'batch_protocol_not_the_frozen_file':
        copy.frozen[2] = '0' * 64
    elif case == 'incomplete_batch':
        (paths['batch2']/json.loads((paths['batch2']/'planned_jobs.json').read_text())[7]['result_file']).unlink()
    elif case == 'duplicate_audit_changed_and_resealed':
        path = paths['newdata_analysis']/'newdata_duplicates.csv'
        edit_csv(path, 0, 'duplicate_rows', '2')
        reseal(paths['newdata_analysis']/'newdata_analysis.json', path)
    elif case == 'family_holm_changed_and_resealed':
        path = paths['newdata_analysis']/'newdata_families.csv'
        edit_csv(path, 2, 'holm_p_approximate', '0.001')
        reseal(paths['newdata_analysis']/'newdata_analysis.json', path)
    elif case == 'moderator_test_changed':
        tcr.rewrite(paths['newdata_analysis']/'newdata_analysis.json', lambda r: r['moderator_test'].update(p_one_sided=.001))
    elif case == 'depth_split_changed':
        tcr.rewrite(paths['training_compare']/'training_depth_split.json',
                    lambda r: r['by_dataset']['alpha'][0].update(mean_difference=r['by_dataset']['alpha'][0]['mean_difference'] + 1e-6))
    elif case == 'ladder_changed':
        tcr.rewrite(paths['projected_compare']/'projected_ladder_error_table.json',
                    lambda r: r['rows']['beta'][0].update(outer_fold_sd=r['rows']['beta'][0]['outer_fold_sd'] + 1e-6))
    else:
        raise AssertionError(case)


REFUSALS = [('benchmark', 'comparator_csv_changed_and_resealed', 'differs from the recomputation'),
            ('benchmark', 'comparator_csv_not_the_sealed_file', 'is not the file'),
            ('benchmark', 'main_table_from_another_run', 'was computed from another knn run'),
            ('benchmark', 'training_reference_pins_another_knn_run', 'does not match the knn run'),
            ('benchmark', 'batch_protocol_not_the_frozen_file', 'is not the frozen batch 2 protocol'),
            ('benchmark', 'incomplete_batch', 'incomplete'),
            ('benchmark', 'duplicate_audit_changed_and_resealed', 'newdata_duplicates.csv syn_h1'),
            ('training', 'family_holm_changed_and_resealed', 'newdata_families.csv'),
            ('training', 'moderator_test_changed', 'moderator_test differs'),
            ('training', 'depth_split_changed', 'training_depth_split.json alpha'),
            ('training', 'ladder_changed', 'projected_ladder_error_table.json beta')]


@pytest.mark.parametrize('command, case, message', REFUSALS)
def test_refuses_unverified_unpaired_or_disagreeing_sources_before_writing_anything(copy, command, case, message):
    tamper(copy, case)
    with pytest.raises(RunComparisonError) as refused:
        getattr(h, command)(copy.paths, copy.out/command, frozen_protocols=copy.frozen)
    assert message in str(refused.value), str(refused.value)
    assert not (copy.out/command).exists()


def test_refuses_to_replace_an_output_with_different_content(chain, tmp_path):
    shutil.copytree(chain.out/'benchmark', tmp_path/'benchmark')
    (tmp_path/'benchmark'/'main_table.csv').write_text('changed\n')
    with pytest.raises(FileExistsError):
        h.benchmark(chain.paths, tmp_path/'benchmark', frozen_protocols=chain.frozen)
    assert (tmp_path/'benchmark'/'main_table.csv').read_text() == 'changed\n'


# ----------------------------------------------------------------------------- command line and readiness

def test_commands_end_to_end_then_ready_and_its_refusals(chain, tmp_path, monkeypatch, capsys):
    committed = {'code_revision': 'f' * 40, 'sources_committed_at_revision': True,
                 'analysis_sources': {f'experiments/make_revision/{name}': '1' * 64 for name in h.ANALYSIS_SOURCES}}
    monkeypatch.setattr(h, 'check_batches', lambda runs, frozen_protocols=None: cn.check_batches(runs, chain.frozen))
    monkeypatch.setattr(h, 'code_record', lambda names: dict(committed))
    root = tmp_path/'holistic'
    h.main(['benchmark', '--runs', str(chain.runs), '--output', str(root/'benchmark')])
    h.main(['training', '--runs', str(chain.runs), '--output', str(root/'training')])
    printed = capsys.readouterr().out
    assert 'Friedman over 6 datasets x 6 models' in printed and 'Holm over all 12 contrasts (post hoc)' in printed
    for name in h.BENCHMARK_OUTPUTS[:-1]:
        assert (root/'benchmark'/name).read_bytes() == (chain.out/'benchmark'/name).read_bytes()
    h.main(['benchmark', '--runs', str(chain.runs), '--output', str(root/'benchmark')])          # identical rerun accepted
    with pytest.raises(SystemExit) as refused:
        h.main(['ready', '--root', str(tmp_path/'absent')])
    assert refused.value.code == 2
    h.main(['ready', '--root', str(root)])
    ready = load(root/'READY')
    assert ready['commit'] == 'f' * 40 and len(ready['outputs']) == len(h.BENCHMARK_OUTPUTS) + len(h.TRAINING_OUTPUTS)
    assert all(entry['sha256'] == tcr.sha256(root/entry['path']) for entry in ready['outputs'])
    assert h.ready(root, now=ready['written_utc']) == ready                                      # an identical READY is accepted
    with pytest.raises(FileExistsError):                                                         # a different READY is never replaced
        h.ready(root, now='2000-01-01T00:00:00+00:00')
    assert load(root/'READY') == ready
    edited = tmp_path/'edited'
    shutil.copytree(root, edited)
    (edited/'READY').unlink()
    (edited/'training'/'ladder.csv').write_text('changed\n')
    with pytest.raises(RunComparisonError, match='differ from the sealed outputs'):
        h.ready(edited)
    uncommitted = tmp_path/'uncommitted'
    shutil.copytree(root, uncommitted)
    (uncommitted/'READY').unlink()
    tcr.rewrite(uncommitted/'benchmark'/'benchmark.json', lambda r: r['provenance'].update(sources_committed_at_revision=False))
    with pytest.raises(RunComparisonError, match='committed analysis sources'):
        h.ready(uncommitted)
    capsys.readouterr()
    monkeypatch.setattr(h, 'check_batches', lambda runs, frozen_protocols=None: cn.check_batches(runs, {1: '0' * 64, 2: '0' * 64}))
    with pytest.raises(SystemExit) as refused:
        h.main(['training', '--runs', str(chain.runs), '--output', str(tmp_path/'refused')])
    assert refused.value.code == 2 and 'holistic training refused' in capsys.readouterr().err and not (tmp_path/'refused').exists()


@needs_real_runs
def test_real_sources_are_paired_and_every_published_analysis_names_the_real_runs():
    """Registered pairings, the combination and the recorded provenance of the published analyses on the real runs
    (without the full re-verification, which the production commands perform)."""
    paths = h.source_paths(REAL_RUNS)
    runs = {label: cr.load_run(paths[label], label) for label in h.RUN_LABELS}
    cr.check_training_reference(runs['training'].protocol, runs['knn'])
    cp.check_projected_references(runs['projected'].protocol, runs['knn'], runs['training'])
    assert cn.check_batches({label: runs[label] for label in cn.LABELS})['pins_checked'] is True
    panel = h.Panel(runs)
    combination = h.check_combination(panel)
    assert len(panel.benchmark) == 7 and len(panel.further) == 10 and panel.further == list(nd.PANEL)
    assert combination['design']['fit_seeds'] == [8129, 19391, 39019] and len(panel.folds) == 15
    records, files = h.read_published(paths, tuple(h.PUBLISHED))
    h.check_published(records, files, panel)
    assert {key: family['size'] for key, family in h.registered_families(panel).items()} == {
        'knn_training_primary': 14, 'newdata_primary': 10, 'newdata_secondary': 10}


# ----------------------------------------------------------------------------- components (after both ablation runs)

@pytest.fixture(scope='session')
def ablations(tmp_path_factory):
    """The synthetic smoke runs of both ablation runners and their component table."""
    root = tmp_path_factory.mktemp('holistic_ablations')
    rka.smoke(root/'knn', json.loads((tcr.PROTOCOLS/'knn_ablation.json').read_text()), workers=2)
    rna.smoke(root/'newdata', rna.draft_protocol(), workers=2)
    paths = {'knn_ablation': root/'knn', 'newdata_ablation': root/'newdata'}
    return SimpleNamespace(root=root, paths=paths, record=h.components(paths, root/'out', allow_smoke=True), out=root/'out')


def ablation_copies(ablations, root):
    paths = {key: root/key for key in h.ABLATION_SOURCES}
    for key, path in paths.items():
        shutil.copytree(ablations.paths[key], path)
    return paths


def test_components_combine_both_ablation_summaries_with_the_arrowflow_minus_variant_sign(ablations):
    record, out = ablations.record, ablations.out
    assert record['datasets'] == {'all': ['synthetic', 'synthetic_b1', 'synthetic_b2'], 'benchmark': ['synthetic'],
                                  'further': ['synthetic_b1', 'synthetic_b2']}
    assert sorted(p.name for p in out.iterdir()) == sorted(h.COMPONENT_OUTPUTS)
    summaries = {name: entry for key, (name_json, _) in h.ABLATION_SUMMARIES.items()
                 for name, entry in load(ablations.paths[key]/name_json)['summaries'].items()}
    rows = read_csv(out/'components.csv')
    assert [(r['dataset'], r['variant']) for r in rows] == [(name, v) for name in record['datasets']['all'] for v in h.COMPONENT_VARIANTS]
    for row in rows:
        change = summaries[row['dataset']]['variants'][row['variant']]['change_from_views7']['accuracy']
        assert float(row['arrowflow_minus_variant']) == -change['mean_difference'] and row['arrowflow_minus_variant'] != '-0.0'
        assert (float(row['ci_low']), float(row['ci_high'])) == (-change['ci_high'], -change['ci_low'])
        assert row['status'] == h.COMPONENT_STATUS and int(row['n_folds']) == 3
    identical = {(r['dataset'], r['variant']): int(r['identical_to_views7_folds']) for r in rows}
    assert (identical[('synthetic_b2', 'no_augment')], identical[('synthetic_b1', 'no_augment')], identical[('synthetic', 'no_augment')]) == (3, 0, 0)
    assert record['counts']['no_augment']['identical_to_views7_in_every_fold'] == ['synthetic_b2']
    assert 'synthetic_b2' not in record['counts']['no_augment']['arrowflow_more_accurate'] + record['counts']['no_augment']['variant_more_accurate']
    depth = read_csv(out/'components_depth.csv')
    assert sum(int(r['n_folds']) for r in depth) == 9 and all(
        float(r['views7_minus_untrained']) == -float(r['untrained_minus_views7']) for r in depth if r['untrained_minus_views7'])
    pooled = read_csv(out/'components_depth_pooled.csv')
    assert [(r['scope'], r['widths']) for r in pooled] == [(s, w) for s in ('all', 'benchmark', 'further') for w in ('[4]', '[2, 4]')]
    runs = record['provenance']['runs']
    assert runs['newdata_ablation']['views7_reproduces_reference'] == dict.fromkeys(['synthetic_b1', 'synthetic_b2'],
                                                                                    {'matching_fold_seeds': 9, 'total_fold_seeds': 9})
    assert runs['knn_ablation']['jobs_verified'] == 3 and runs['newdata_ablation']['jobs_verified'] == 6
    assert set(record['provenance']['analysis_sources']) == {f'experiments/make_revision/{n}' for n in h.COMPONENT_ANALYSIS_SOURCES}
    for name, seal in record['outputs'].items():
        assert seal['sha256'] == tcr.sha256(out/name)


@pytest.mark.parametrize('case', ['newdata_run_absent', 'newdata_summary_missing', 'newdata_prediction_missing', 'knn_summary_csv_missing'])
def test_components_refuse_until_both_ablation_runs_are_complete_and_read_nothing_before(ablations, tmp_path, monkeypatch, case):
    paths = ablation_copies(ablations, tmp_path/'runs')
    if case == 'newdata_run_absent':
        shutil.rmtree(paths['newdata_ablation'])
    elif case == 'newdata_summary_missing':
        (paths['newdata_ablation']/'newdata_ablation_summary.json').unlink()
    elif case == 'newdata_prediction_missing':
        (paths['newdata_ablation']/'predictions'/'synthetic_b2__r0f2.jsonl').unlink()
    else:
        (paths['knn_ablation']/'knn_ablation_summary.csv').unlink()
    read = []
    monkeypatch.setattr(rka, 'summary', lambda *args, **kwargs: read.append('knn'))
    monkeypatch.setattr(rna, 'summary', lambda *args, **kwargs: read.append('newdata'))
    with pytest.raises(RunComparisonError, match='only after both ablation runs are complete'):
        h.components(paths, tmp_path/'out', allow_smoke=True)
    assert read == [] and not (tmp_path/'out').exists()


def test_components_refuse_a_summary_that_differs_or_an_unfrozen_run_and_the_command_exits_2(ablations, tmp_path, capsys):
    paths = ablation_copies(ablations, tmp_path/'runs')
    tcr.rewrite(paths['newdata_ablation']/'newdata_ablation_summary.json',
                lambda r: r['summaries']['synthetic_b1']['variants']['views1']['change_from_views7']['accuracy'].update(mean_difference=.5))
    with pytest.raises(RunComparisonError, match='differs from the recomputation'):
        h.components(paths, tmp_path/'out', allow_smoke=True)
    with pytest.raises(RunComparisonError, match='does not hold the frozen arrowflow-v3-knn-ablation-1 protocol'):
        h.components(ablations.paths, tmp_path/'out')
    assert not (tmp_path/'out').exists()
    with pytest.raises(SystemExit) as refused:
        h.main(['components', '--runs', str(tmp_path/'empty'), '--output', str(tmp_path/'cli')])
    assert refused.value.code == 2 and 'holistic components refused' in capsys.readouterr().err and not (tmp_path/'cli').exists()
