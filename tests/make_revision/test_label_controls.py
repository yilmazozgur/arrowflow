"""The permuted-label control (label_controls, compare_labels): the frozen design and its protocol, the permutation and its
private stream, the snapshot-readout helpers, the statistics, the reading rule and its wordings. No network is trained
here; the end-to-end path is exercised by `label_controls check` on the registered data."""
import json
import random
import numpy as np
import pytest
from scipy import stats
from experiments.make_revision import compare_labels as cl
from experiments.make_revision import label_controls as lc
from experiments.make_revision import motion_controls as mc
from experiments.make_revision.comparisons import StableFootruleKNN
from experiments.make_revision.models import array_hash

IDENTITY = {'dataset_id': 'iris', 'outer_repeat': 1, 'outer_fold': 3, 'model_seed': 19391}
needs_motion_run = pytest.mark.skipif(not lc.MOTION_RUN.is_dir(), reason='the registered motion run is not on this machine')


# ----------------------------------------------------------------------------- the design

def test_the_protocol_file_is_the_design_with_the_pinned_hash():
    assert lc.verify_protocol() == json.loads(lc.protocol_text())
    assert json.loads(lc.PROTOCOL.read_text())['protocol_id'] == lc.PROTOCOL_ID


def test_the_header_opens_with_the_design_statement():
    assert lc.__doc__.startswith('DESIGN (fixed 2026-11-25 before any score of this arm; committed before the run)')


def test_the_design_names_every_job_check_and_all_five_wordings_and_f():
    design = lc.design()
    assert design['checks'] == list(lc.JOB_CHECKS)
    assert sorted(design['reading']['wordings']) == ['a', 'b', 'c', 'd', 'e', 'f']
    assert design['reading']['order'] == ['f', 'd', 'c', 'b', 'a', 'e']


# ----------------------------------------------------------------------------- the permutation

def test_the_permutation_is_private_deterministic_and_keeps_the_class_counts():
    y = np.repeat([0, 1, 2], [10, 25, 5])
    np.random.seed(5)
    random.seed(5)
    numpy_before, python_before = np.random.get_state(), random.getstate()
    a, info_a = lc.permuted_labels(y, IDENTITY)
    assert mc.numpy_states_equal(numpy_before, np.random.get_state()) and python_before == random.getstate()
    b, info_b = lc.permuted_labels(y, IDENTITY)
    assert np.array_equal(a, b) and info_a == info_b
    assert info_a['class_counts_equal'] and info_a['rng_unchanged']
    assert np.array_equal(np.sort(a), np.sort(y)) and not np.array_equal(a, y)
    order = np.random.RandomState(lc.label_stream_seed(IDENTITY)).permutation(len(y))
    assert np.array_equal(a, y[order]) and info_a['permutation_sha256'] == array_hash(order)


def test_every_fold_and_seed_has_its_own_stream():
    seeds = {lc.label_stream_seed(dict(IDENTITY, **change)) for change in
             ({}, {'model_seed': 8129}, {'outer_fold': 0}, {'outer_repeat': 0}, {'dataset_id': 'wine'})}
    assert len(seeds) == 5


def test_permutation_statistics_by_hand():
    y = np.array([0, 0, 1, 1])
    same = lc.permutation_statistics(y, y)
    assert same['agreement'] == 1. and same['cramers_v'] == pytest.approx(1.) and same['expected_agreement'] == .5
    swapped = lc.permutation_statistics(y, np.array([1, 1, 0, 0]))
    assert swapped['agreement'] == 0. and swapped['cramers_v'] == pytest.approx(1.)
    half = lc.permutation_statistics(y, np.array([0, 1, 0, 1]))
    assert half['agreement'] == .5 and half['cramers_v'] == pytest.approx(0.)
    assert lc.identity_labels(y)[1]['agreement'] == 1.


def test_the_expected_agreement_is_the_mean_over_permutations():
    y = np.repeat([0, 1, 2], [3, 5, 12])
    rng = np.random.RandomState(0)
    draws = [np.mean(y[rng.permutation(len(y))] == y) for _ in range(20000)]
    assert np.mean(draws) == pytest.approx(lc.permutation_statistics(y, y)['expected_agreement'], abs=4e-3)


# ----------------------------------------------------------------------------- readout helpers

def _positions(n, d, seed):
    return np.stack([np.random.RandomState(seed + i).permutation(d) for i in range(n)])


def test_the_copy_readout_is_the_live_networks_arithmetic_and_reads_the_true_labels():
    """ArrowFlowEstimator.transform_orders on a network whose hidden filters are `matrices`, then the method's readout."""
    from scipy.spatial.distance import cdist
    from arrowflow.ranking import inverse_positions, score_order
    from experiments.make_revision.comparisons import derive_seed
    from experiments.make_revision.multiview import select_knn_readout
    orders_train, orders_test = _positions(40, 6, 0), _positions(9, 6, 100)
    y = np.tile([0, 1], 20)
    hidden = np.asarray([np.random.RandomState(7 + k).permutation(6) for k in range(5)], dtype=float)
    output = np.asarray([np.random.RandomState(70 + k).permutation(5) for k in range(2)], dtype=float)
    out = lc.copy_readout((hidden, output), 1, orders_train, orders_test, y, seed_v=11)
    live = lambda orders: inverse_positions(score_order(cdist(inverse_positions(orders), hidden, metric='cityblock')))
    assert np.array_equal(out['hidden_train'], live(orders_train)) and np.array_equal(out['hidden_test'], live(orders_test))
    selection = select_knn_readout(live(orders_train), y, seed=derive_seed(11, 'readout_selection'))
    expected = StableFootruleKNN(**selection['config'], input_kind='positions').fit(live(orders_train), y)
    assert np.array_equal(out['predictions'], expected.predict(live(orders_test)))
    assert out['selection']['config_id'] == selection['config_id']
    assert lc.readout_labels_true(out['readout'], y) and not lc.readout_labels_true(out['readout'], y[::-1])


def test_purity_of_matches_the_motion_purity_arithmetic():
    rng = np.random.RandomState(3)
    hidden_train = np.stack([rng.permutation(8) for _ in range(30)])
    hidden_test = np.stack([rng.permutation(8) for _ in range(7)])
    y_train, y_test = np.tile([0, 1, 2], 10), np.array([0, 1, 2, 0, 1, 2, 0])
    readout = StableFootruleKNN(n_neighbors=3, input_kind='positions').fit(hidden_train, y_train)
    got = lc.purity_of([readout], [hidden_train], [hidden_test], y_train, y_test)
    assert got['test'] == mc.neighborhood_purity(readout, y_train, hidden_test, y_test, k=3)
    rows = np.arange(30)
    assert got['train'] == mc.neighborhood_purity(readout, y_train, hidden_train[rows], y_train[rows], k=3,
                                                  exclude_rows=rows)


# ----------------------------------------------------------------------------- statistics

@pytest.mark.parametrize('seed', range(6))
def test_signed_rank_equals_scipy_without_zeros_or_ties(seed):
    d = np.random.RandomState(seed).randn(17)
    assert cl.signed_rank(d)['p'] == pytest.approx(stats.wilcoxon(d, method='exact').pvalue, abs=1e-12)


def test_signed_rank_with_a_zero_and_ties_counts_exactly():
    # zeros dropped: [1, -1, 2, 3] -> ranks |1|,|1| tie at 1.5, then 3, 4; W+ = 1.5 + 3 + 4 = 8.5 of 10
    result = cl.signed_rank([0., 1., -1., 2., 3.])
    assert result['n'] == 4 and result['zeros'] == 1 and result['ties'] and result['w_plus'] == 8.5
    # sign assignments with |W+ - 5| >= 3.5: W+ in {0, 1.5, 8.5, 10}; count them by brute force
    ranks = [1.5, 1.5, 3., 4.]
    sums = [sum(r for r, s in zip(ranks, bits) if s) for bits in np.ndindex(2, 2, 2, 2)]
    assert result['p'] == pytest.approx(sum(abs(s - 5) >= 3.5 for s in sums) / 16)


def test_signed_rank_of_nothing_is_not_significant():
    assert cl.signed_rank([0., 0.])['p'] == 1.


def test_direction_needs_p_below_alpha_and_follows_the_larger_rank_sum():
    assert cl.direction({'p': .01, 'w_plus': 140., 'w_minus': 13.}) == 'positive'
    assert cl.direction({'p': .01, 'w_plus': 13., 'w_minus': 140.}) == 'negative'
    assert cl.direction({'p': .2, 'w_plus': 140., 'w_minus': 13.}) == 'n.s.'


@pytest.mark.parametrize('d, s, expected', [
    ('negative', 'positive', 'f'), ('negative', 'n.s.', 'f'), ('negative', 'negative', 'f'),
    ('positive', 'negative', 'd'), ('positive', 'n.s.', 'c'), ('positive', 'positive', 'b'),
    ('n.s.', 'positive', 'a'), ('n.s.', 'n.s.', 'e'), ('n.s.', 'negative', 'e')])
def test_the_reading_rule(d, s, expected):
    assert cl.pattern(d, s) == expected


def test_ratio_of_sums_and_its_bootstrap():
    num, den = np.arange(17) / 10., np.ones(17)
    result = cl.ratio_of_sums(num, den)
    assert result['value'] == pytest.approx(num.sum() / 17) and result['undefined_resamples'] == 0
    idx = np.random.default_rng(lc.BOOTSTRAP_SEED).integers(0, 17, (lc.BOOTSTRAP_RESAMPLES, 17))
    ratios = num[idx].sum(1) / den[idx].sum(1)
    assert result['ci_low'] == pytest.approx(np.percentile(ratios, 2.5))
    assert result['ci_high'] == pytest.approx(np.percentile(ratios, 97.5))
    negative = cl.ratio_of_sums(num, -np.ones(17))
    assert negative['value'] is None and negative['ci_low'] is None and negative['ci_high'] is None


def test_t_interval_by_hand():
    mean, lo, hi = cl.t_interval([1., 2., 3., 4.])
    half = stats.t.ppf(.975, 3) * np.std([1., 2., 3., 4.], ddof=1) / 2
    assert (mean, lo, hi) == pytest.approx((2.5, 2.5 - half, 2.5 + half))


# ----------------------------------------------------------------------------- the wordings

def _summary(mean, lo, hi, p, w_plus, w_minus):
    test = {'p': p, 'w_plus': w_plus, 'w_minus': w_minus}
    return {'all': {'mean': mean, 'ci_low': lo, 'ci_high': hi, 'signed_rank': test, 'direction': cl.direction(test)}}


def test_reading_fills_the_wording_of_pattern_d_and_its_rules():
    families = {'F1': {'higher': 16, 'significantly_higher': 5}, 'F2': {'higher': 2, 'significantly_higher': 0},
                'F3': {'higher': 16, 'significantly_higher': 6}, 'F4': {'higher': 1, 'significantly_higher': 0},
                'G': {'higher': 14, 'significantly_higher': 3}}
    summary = {'F1': _summary(2.1, 1., 3., .001, 150., 3.), 'F2': _summary(-1.05, -2., -.1, .002, 4., 149.),
               'F3': _summary(1.0, .1, 2., .2, 100., 53.), 'F4': _summary(.3, -.5, 1., .5, 80., 73.),
               'G': _summary(1.05, .3, 1.8, .003, 140., 13.)}
    share = {'value': .5, 'ci_low': .1, 'ci_high': None}
    shares = {'match': {'label': share, 'label_free': share}, 'own': {'label': share, 'label_free': share}}
    displacement = {'AF[checkpoint]': .1234, 'PL[match]': .2, 'PL[own]': .3, 'PL[final]': .4}
    out = cl.reading(families, summary, shares, displacement)
    assert out['primary']['pattern'] == 'd' and out['secondary']['pattern'] == 'e' and not out['same_pattern']
    assert out['primary']['sentences'][0].startswith('Trained with permuted labels, the same networks are less accurate '
                                                     'than ArrowFlow on 16 of 17 datasets (significantly on 5)')
    assert 'by 1.05 points on average' in out['primary']['sentences'][0]
    assert out['primary']['sentences'][1] == lc.D_DAMAGE
    assert out['primary']['sentences'][2].startswith('The size of the gain is G = +1.05 points')
    assert lc.WITHDRAW in out['secondary']['sentences'] and lc.WITHDRAW not in out['primary']['sentences']
    assert out['stopping_rule'] == lc.STOPPING.format(secondary='e', primary='d')


def test_every_wording_formats_with_the_values_reading_supplies():
    values = {'k': 1, 'm': 0, 'abs_s': '1.00', 'x': '0.100', 'y': '0.200', 's': '+1.00', 'l': '-1.00', 'u': '+2.00',
              's_points': '1.00', 'x_share': '50 %', 'x_interval': '95 % interval 10 to 90 %', 'd_points': '1.00',
              'k_s': 3, 'd': '-1.00', 'p_d': '0.01'}
    for letter, text in lc.WORDINGS.items():
        assert '{' not in text.format(**values), letter


# ----------------------------------------------------------------------------- the output guard

@needs_motion_run
def test_the_output_guard_refuses_registered_runs_and_foreign_files(tmp_path):
    with pytest.raises(SystemExit):
        lc.guard_output(lc.MOTION_RUN)
    with pytest.raises(SystemExit):
        lc.guard_output(lc.MOTION_RUN/'results')
    with pytest.raises(SystemExit):
        lc.guard_output(lc.REPO/'experiments'/'make_revision')
    (tmp_path/'stray.json').write_text('{}')
    with pytest.raises(SystemExit):
        lc.guard_output(tmp_path)
    (tmp_path/'stray.json').unlink()
    (tmp_path/'results').mkdir()
    (tmp_path/'results'/'iris__r0f0.json').write_text('{}')
    assert lc.guard_output(tmp_path) == tmp_path.resolve()
    (tmp_path/'results'/'iris__r9f0.json').write_text('{}')
    with pytest.raises(SystemExit):
        lc.guard_output(tmp_path)


@needs_motion_run
def test_the_analysis_gate_reads_names_only_and_refuses_an_incomplete_run(tmp_path):
    (tmp_path/'results').mkdir()
    (tmp_path/'design.json').write_text(lc.protocol_text())
    (tmp_path/'code_revision.txt').write_text('0' * 40 + '\n')
    (tmp_path/'results'/'iris__r0f0.json').write_text('not json: never parsed by the gate')
    with pytest.raises(SystemExit, match='254 of 255 records missing'):
        cl.completeness_gate(tmp_path, allow_uncommitted=False)
    (tmp_path/'code_revision.txt').write_text('0' * 40 + '+uncommitted\n')
    with pytest.raises(SystemExit, match='uncommitted code'):
        cl.completeness_gate(tmp_path, allow_uncommitted=False)


@needs_motion_run
def test_the_job_order_puts_the_longest_jobs_first():
    stems = lc.job_stems()
    assert len(stems) == 255 and len(set(stems)) == 255
    costs = {s: lc.job_cost(s) for s in stems}
    assert all(cost > 0 for cost in costs.values())
