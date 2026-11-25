"""artificial_ranks: the original rows against their source file, the seven prototypes, the calibrated corruption against an
independent transcription of the ruling, the features, the pinned datasets and their refusals, the harness contract and
splits, and the descriptive data report."""
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import comb, floor
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
import pytest
from experiments.make_revision import artificial_ranks as ar
from experiments.make_revision.evaluation import config_id, dataset_fingerprint, make_splits, validate_split

REPO = Path(__file__).resolve().parents[2]
# ml/sortflow_hybrid_3.py of the sortflow checkout that holds this repository, directly or through .worktrees/<name>.
SOURCE = next((parent/'ml'/'sortflow_hybrid_3.py' for parent in REPO.parents
               if (parent/'ml'/'sortflow_hybrid_3.py').is_file()), None)
APPEND = re.compile(r"^\s*data_(train|test)_net\.append\(\[list\(map\(str, \[([0-9, ]*)\]\)\), '([0-9]+)', 1\]\)\s*$")

# The ruling's prototypes written out by hand: at N = 8 the first original row of each class, and at N = 16.
EXPECTED = {
    8: ((1, 2, 3, 4, 5, 6, 7, 8), (8, 6, 4, 2, 1, 3, 5, 7), (1, 3, 5, 7, 8, 6, 4, 2), (1, 8, 2, 7, 3, 6, 4, 5),
        (8, 4, 7, 3, 6, 2, 5, 1), (8, 1, 7, 2, 6, 3, 5, 4), (1, 5, 2, 6, 3, 7, 4, 8)),
    16: (tuple(range(1, 17)),
         (16, 14, 12, 10, 8, 6, 4, 2, 1, 3, 5, 7, 9, 11, 13, 15),
         (1, 3, 5, 7, 9, 11, 13, 15, 16, 14, 12, 10, 8, 6, 4, 2),
         (1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 8, 9),
         (16, 8, 15, 7, 14, 6, 13, 5, 12, 4, 11, 3, 10, 2, 9, 1),
         (16, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6, 10, 7, 9, 8),
         (1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16)),
}
SHAPES = {'ranks8': [315, 8], 'ranks16': [315, 16], 'ranks8_original': [77, 8]}
COUNTS = {'ranks8': [45] * 7, 'ranks16': [45] * 7, 'ranks8_original': [11] * 7}


def interleave(first, second):
    return tuple(item for pair in zip(first, second) for item in pair)


def ruling_prototypes(N):
    """The seven patterns of the ruling for even N, read off its bracketed definitions."""
    h = N // 2
    evens_descending = [item for item in range(N, 0, -1) if item % 2 == 0]
    odds_ascending = [item for item in range(1, N + 1) if item % 2 == 1]
    return (tuple(range(1, N + 1)), tuple(evens_descending + odds_ascending), tuple(odds_ascending + evens_descending),
            interleave(range(1, h + 1), range(N, h, -1)), interleave(range(N, h, -1), range(h, 0, -1)),
            interleave(range(N, h, -1), range(1, h + 1)), interleave(range(1, h + 1), range(h + 1, N + 1)))


def discordant_pairs(sequence, reference):
    reference = list(reference)
    return sum(reference.index(a) > reference.index(b) for a, b in combinations(sequence, 2))


def ruling_rows(N, rows_per_class, seed):
    """The ruling's generation procedure, transcribed independently of the module: [(class, sequence)]."""
    rng = np.random.RandomState(seed)
    reference = ruling_prototypes(N)
    rows = []
    for c in range(7):
        pool = [sequence for label, _, sequence in ar.ORIGINAL_ROWS if label == str(c)]
        for _ in range(rows_per_class):
            drawn = pool[rng.randint(len(pool))]
            m = len(drawn)
            f, tau = Fraction(8 - m, 8), Fraction(discordant_pairs(drawn, EXPECTED[8][c]), comb(m, 2))
            k_n = floor(f * N + Fraction(1, 2))
            deleted = rng.choice(N, k_n, replace=False).tolist()
            row = [item for position, item in enumerate(reference[c]) if position not in deleted]
            for _ in range(floor(tau * comb(len(row), 2) + Fraction(1, 2))):
                in_order = [j for j in range(len(row) - 1)
                            if reference[c].index(row[j]) < reference[c].index(row[j + 1])]
                j = in_order[rng.randint(len(in_order))]
                row[j], row[j + 1] = row[j + 1], row[j]
            rows.append((c, tuple(row)))
    return rows


# ----------------------------------------------------------------------------- the original rows

def test_the_original_rows_are_six_training_and_five_test_orderings_per_class_of_distinct_items_within_one_to_eight():
    assert len(ar.ORIGINAL_ROWS) == 77
    assert Counter((label, split) for label, split, _ in ar.ORIGINAL_ROWS) == {
        (str(c), split): n for c in range(7) for split, n in (('train', 6), ('test', 5))}
    for _, _, sequence in ar.ORIGINAL_ROWS:
        assert len(set(sequence)) == len(sequence) and set(sequence) <= set(range(1, 9)) and 5 <= len(sequence) <= 8
    assert ar.original_rows() == [(label, split, tuple(sequence)) for label, split, sequence in ar.ORIGINAL_ROWS]


@pytest.mark.skipif(SOURCE is None, reason='the sortflow source file ml/sortflow_hybrid_3.py is not on this machine')
def test_the_fixture_is_the_append_lines_of_the_source_file_verbatim():
    lines = SOURCE.read_text().split('\n')
    parsed, numbers = [], []
    for number, line in enumerate(lines, 1):
        if re.search(r'data_(train|test)_net\.append\(', line):
            match = APPEND.match(line)
            assert match, f'line {number} is not a plain row append: {line!r}'
            parsed.append((match.group(3), match.group(1), tuple(int(item) for item in match.group(2).split(','))))
            numbers.append(number)
    assert parsed == [(label, split, tuple(sequence)) for label, split, sequence in ar.ORIGINAL_ROWS]
    assert (numbers[0], numbers[-1]) == tuple(ar.ORIGINAL_SOURCE['lines'])
    assert lines[ar.ORIGINAL_SOURCE['method_line'] - 1].strip().startswith('def generate_sequence_data(')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == ar.ORIGINAL_SOURCE['sha256']


# ----------------------------------------------------------------------------- the prototypes

def test_at_eight_items_each_prototype_is_the_first_original_row_of_its_class():
    first = {}
    for label, _, sequence in ar.ORIGINAL_ROWS:
        first.setdefault(int(label), tuple(sequence))
    assert ar.prototypes(8) == EXPECTED[8] == tuple(first[c] for c in range(7))


def test_the_prototypes_follow_the_ruling_at_every_even_size_and_are_pairwise_distinct_at_eight_and_sixteen():
    assert ruling_prototypes(8) == EXPECTED[8] and ruling_prototypes(16) == EXPECTED[16] == ar.prototypes(16)
    for N in range(4, 42, 2):
        assert ar.prototypes(N) == ruling_prototypes(N)
        assert all(sorted(prototype) == list(range(1, N + 1)) for prototype in ar.prototypes(N))
    for N in (8, 16):
        assert len(set(ar.prototypes(N))) == 7
    for N in (2, 3, 7, 9, 8.0, True):
        with pytest.raises(ValueError, match='even N >= 4'):
            ar.prototypes(N)


class LowestDraws:
    """A RandomState stand-in drawing the lowest values: no swaps, no deletions, identity shuffles."""
    def __init__(self, seed=None):
        pass

    def randint(self, low, high=None):
        return 0 if high is None else low

    def permutation(self, n):
        return np.arange(n)


def test_the_ported_twelve_pattern_generator_repeats_a_class_at_even_sizes(monkeypatch):
    port = pytest.importorskip('arrowflow.arrowflow')
    monkeypatch.setattr(port.np.random, 'RandomState', LowestDraws)
    for N in (8, 16):
        train, test, _, n_classes, _ = port.DataGraph('ranks').generate_sequence_data(
            vocab_size=N, n_train_per_class=1, n_test_per_class=0)
        patterns = [tuple(int(item) for item in row[0]) for row in train]
        assert n_classes == 12 and len(patterns) == 12 and test == []
        assert patterns[11] == patterns[6] == EXPECTED[N][3] and len(set(patterns)) == 11   # spiral inward = zigzag low-start


# ----------------------------------------------------------------------------- the calibrated corruption

def test_the_calibration_table_recomputed_from_the_fixture_is_the_module_table():
    table = [[] for _ in range(7)]
    for label, _, sequence in ar.ORIGINAL_ROWS:
        table[int(label)].append((8 - len(sequence), discordant_pairs(sequence, EXPECTED[8][int(label)])))
    assert ar.CALIBRATION == tuple(tuple(entries) for entries in table) == ar.calibration_table(ar.original_rows())
    assert [len(entries) for entries in ar.CALIBRATION] == [11] * 7
    assert Counter(k for entries in ar.CALIBRATION for k, _ in entries) == {0: 32, 1: 33, 2: 10, 3: 2}


def test_the_rounding_rule_reproduces_eight_items_doubles_deletions_at_sixteen_and_fits_every_even_size():
    for entries in ar.CALIBRATION:
        for k, d in entries:
            m = 8 - k
            assert ar.corruption_targets(k, d, 8) == (k, d) and ar.corruption_targets(k, d, 16)[0] == 2 * k
            for N in (8, 16):                                    # the float evaluation of the rule agrees
                k_float = floor(k / 8 * N + .5)
                assert ar.corruption_targets(k, d, N) == (k_float, floor(d / comb(m, 2) * comb(N - k_float, 2) + .5))
            for N in range(4, 66, 2):
                k_n, d_n = ar.corruption_targets(k, d, N)
                assert N - k_n >= 2 and 0 <= d_n <= comb(N - k_n, 2)


@pytest.mark.parametrize('N, rows_per_class, seed', [(8, 45, 20260908), (16, 45, 20260916), (4, 20, 5), (12, 20, 6),
                                                     (32, 10, 7)])
def test_generation_follows_the_ruling_and_every_row_has_exactly_its_target_deletions_and_kendall_distance(
        N, rows_per_class, seed):
    rows = ar.generate_rows(N, rows_per_class, seed)
    assert [(c, sequence) for c, sequence, _ in rows] == ruling_rows(N, rows_per_class, seed)
    assert [c for c, _, _ in rows] == [c for c in range(7) for _ in range(rows_per_class)]
    for c, sequence, record in rows:
        k, d = ar.CALIBRATION[c][record['class_row']]
        k_n, d_n = ar.corruption_targets(k, d, N)
        assert (record['k'], record['d'], record['k_N'], record['d_N']) == (k, d, k_n, d_n)
        assert len(sequence) == N - k_n and len(set(sequence)) == len(sequence) and set(sequence) <= set(range(1, N + 1))
        assert discordant_pairs(sequence, ruling_prototypes(N)[c]) == d_n


def test_generation_is_deterministic_for_a_seed():
    assert ar.generate_rows(16, 45, 20260916) == ar.generate_rows(16, 45, 20260916)
    assert ar.generate_rows(16, 45, 20260916) != ar.generate_rows(16, 45, 20260917)
    for name in ar.DATASETS:
        (X, y, manifest), (X_again, y_again, manifest_again) = ar.load_artificial(name), ar.load_artificial(name)
        assert np.array_equal(X, X_again, equal_nan=True) and np.array_equal(y, y_again) and manifest == manifest_again


# ----------------------------------------------------------------------------- features

def test_features_are_relative_positions_with_nan_for_deleted_items_and_invert_exactly():
    X = ar.features([(3, 1, 4), (2, 1, 3, 4)], 4)
    assert X.dtype == np.float64 and np.array_equal(X, [[.5, np.nan, 0., 1.], [1 / 3, 0., 2 / 3, 1.]], equal_nan=True)
    assert ar.feature_names(4) == ['item_1', 'item_2', 'item_3', 'item_4']
    assert ar.sequences_from_features(X) == [(3, 1, 4), (2, 1, 3, 4)]
    for bad in [(1,), (2, 2, 3), (0, 1), (1, 5), (1.0, 2)]:
        with pytest.raises(ValueError, match='distinct items'):
            ar.features([bad], 4)
    for name in ar.DATASETS:
        X, y, _ = ar.load_artificial(name)
        sequences = ar.sequences_from_features(X)
        assert np.array_equal(ar.features(sequences, X.shape[1]), X, equal_nan=True)
        assert (np.isnan(X) | ((X >= 0) & (X <= 1))).all() and (np.nanmin(X, axis=1) == 0).all() and (np.nanmax(X, axis=1) == 1).all()
        if name == ar.ORIGINAL_DATASET:
            assert sequences == [tuple(sequence) for _, _, sequence in ar.ORIGINAL_ROWS]
            assert y.tolist() == [int(label) for label, _, _ in ar.ORIGINAL_ROWS]
        else:
            rows = ar.generate_rows(ar.DATASETS[name], 45, ar.GENERATION_SEEDS[name])
            assert sequences == [sequence for _, sequence, _ in rows] and y.tolist() == [c for c, _, _ in rows]


# ----------------------------------------------------------------------------- the pinned datasets and the harness contract

def test_the_loader_returns_the_return_contract_of_run_revision_load_dataset():
    from experiments.make_revision import run_revision as rr
    X_harness, y_harness, harness = rr.load_dataset('iris')
    for name in ar.DATASETS:
        X, y, manifest = ar.load_artificial(name)
        assert X.dtype == X_harness.dtype == np.float64 and X.ndim == 2 and y.dtype == y_harness.dtype and y.ndim == 1
        assert set(harness) <= set(manifest)
        assert {key: type(manifest[key]) for key in harness} == {key: type(harness[key]) for key in harness}
        assert manifest['label_map'] == sorted(manifest['label_map']) == ['0', '1', '2', '3', '4', '5', '6']
        assert manifest['feature_names'] == [f'item_{item}' for item in range(1, X.shape[1] + 1)]
        assert manifest['shape'] == list(X.shape) == SHAPES[name] and manifest['class_counts'] == np.bincount(y).tolist()
        assert manifest['dataset_hash'] == dataset_fingerprint(X, y, manifest['feature_names'], manifest['label_map'])
        assert manifest == json.loads(json.dumps(manifest, allow_nan=False))
        generation = manifest['generation']
        assert (generation['n_items'], generation['pattern_names']) == (X.shape[1], list(ar.PATTERN_NAMES))
        assert generation['prototypes'] == [list(prototype) for prototype in ar.prototypes(X.shape[1])]
        assert generation['original_source']['sha256'] == ar.ORIGINAL_SOURCE['sha256']
    for name, N, seed in (('ranks8', 8, 20260908), ('ranks16', 16, 20260916)):
        generation = ar.load_artificial(name)[2]['generation']
        assert (generation['kind'], generation['n_items'], generation['rows_per_class'], generation['seed']) == ('generated', N, 45, seed)
        assert generation['calibration_rule'] == ar.CALIBRATION_RULE and ar.GENERATION_SEEDS[name] == seed
    original = ar.load_artificial('ranks8_original')[2]['generation']
    assert original['kind'] == 'original_rows' and Counter(original['original_split']) == {'train': 42, 'test': 35}


def test_the_pins_hold_for_the_built_datasets():
    for name in ar.DATASETS:
        X, y, manifest = ar.load_artificial(name)
        normalized = np.where(np.isnan(X), np.nan, X).astype('<f8')
        assert (ar.PINS[name]['shape'], ar.PINS[name]['class_counts']) == (SHAPES[name], COUNTS[name])
        assert ar.PINS[name]['sha256_X'] == hashlib.sha256(normalized.tobytes()).hexdigest()
        assert ar.PINS[name]['sha256_y'] == hashlib.sha256('\n'.join(map(str, y.tolist())).encode()).hexdigest()
        assert ar.PINS[name]['dataset_hash'] == manifest['dataset_hash']
        assert manifest['identity'] == {'sha256_X': ar.PINS[name]['sha256_X'], 'sha256_y': ar.PINS[name]['sha256_y']}


@pytest.mark.parametrize('key', ar.IDENTITY_KEYS)
def test_the_loader_refuses_every_pin_mismatch(monkeypatch, key):
    wrong = {'shape': [1, 1], 'class_counts': [0] * 7, 'sha256_X': '0' * 64, 'sha256_y': '0' * 64, 'dataset_hash': '0' * 64}
    for name in ar.DATASETS:
        monkeypatch.setitem(ar.PINS, name, dict(ar.PINS[name], **{key: wrong[key]}))
        with pytest.raises(ar.DatasetIdentityError, match=f'{name}: dataset identity mismatch: {key} pinned'):
            ar.load_artificial(name)


def test_the_loader_refuses_a_changed_seed_a_changed_fixture_and_an_unknown_name(monkeypatch):
    with pytest.raises(ar.DatasetIdentityError, match='not an artificial ranks dataset'):
        ar.load_artificial('artificial')
    with monkeypatch.context() as patch:
        patch.setitem(ar.GENERATION_SEEDS, 'ranks16', 20260917)
        with pytest.raises(ar.DatasetIdentityError, match='sha256_X pinned'):
            ar.load_artificial('ranks16')
    rows = list(ar.ORIGINAL_ROWS)
    assert rows[1] == ('0', 'train', (1, 3, 2, 4, 5, 7, 6))
    with monkeypatch.context() as patch:                          # (k, d) = (1, 1) instead of (1, 2): calibration refused
        patch.setattr(ar, 'ORIGINAL_ROWS', tuple(rows[:1] + [('0', 'train', (1, 2, 3, 4, 5, 7, 6))] + rows[2:]))
        for name in ar.DATASETS:
            with pytest.raises(ar.DatasetIdentityError, match='differs from CALIBRATION'):
                ar.load_artificial(name)
    with monkeypatch.context() as patch:                          # the same (k, d) = (1, 2): only the pins catch it
        patch.setattr(ar, 'ORIGINAL_ROWS', tuple(rows[:1] + [('0', 'train', (3, 1, 2, 4, 5, 6, 8))] + rows[2:]))
        ar.load_artificial('ranks8')
        with pytest.raises(ar.DatasetIdentityError, match='sha256_X pinned'):
            ar.load_artificial('ranks8_original')
    with monkeypatch.context() as patch:
        patch.setattr(ar, 'ORIGINAL_ROWS', tuple(rows[:1] + [('0', 'train', (1, 1, 2, 4, 5, 7, 6))] + rows[2:]))
        with pytest.raises(ValueError, match='distinct items within 1..8'):
            ar.load_artificial('ranks8_original')
    with monkeypatch.context() as patch:
        patch.setattr(ar, 'ORIGINAL_ROWS', tuple(rows[1:] + rows[:1]))
        with pytest.raises(ar.DatasetIdentityError, match='six training rows per class'):
            ar.load_artificial('ranks16')


def test_the_harness_splits_accept_all_three_datasets():
    for name in ar.DATASETS:
        _, y, _ = ar.load_artificial(name)
        splits = make_splits(y, 5, 3, 3, 27183)
        assert len(splits) == 15
        for split in splits:
            validate_split(split, len(y))
        assert config_id(splits) == ar.PINS[name]['splits_hash']


# ----------------------------------------------------------------------------- the descriptive data report

def test_the_data_report_agrees_with_the_data_and_is_written_once(tmp_path):
    report = ar.describe()
    assert set(report['datasets']) == set(ar.DATASETS) and report == ar.describe()
    for name, entry in report['datasets'].items():
        X, y, manifest = ar.load_artificial(name)
        N = X.shape[1]
        sequences = ar.sequences_from_features(X)
        assert (entry['rows'], entry['class_counts'], entry['dataset_hash']) == (len(y), COUNTS[name], manifest['dataset_hash'])
        assert entry['missing_cells'] == int(np.isnan(X).sum()) and entry['missing_share'] == pytest.approx(np.isnan(X).mean())
        vectors = Counter(tuple(np.where(np.isnan(row), -1., row)) for row in X)
        assert entry['duplicates']['duplicate_rows'] == len(y) - len(vectors)
        assert entry['duplicates']['duplicate_groups'] == sum(count > 1 for count in vectors.values())
        this = entry['corruption']['this_dataset']
        assert this['deletion_fraction']['overall']['mean'] == pytest.approx(entry['missing_share'])
        taus = [discordant_pairs(s, EXPECTED[N][c]) / comb(len(s), 2) for s, c in zip(sequences, y.tolist())]
        assert this['normalized_kendall_distance']['overall']['mean'] == pytest.approx(np.mean(taus))
        assert this['normalized_kendall_distance']['per_class']['3']['quantiles']['0.5'] == pytest.approx(
            np.median([t for t, c in zip(taus, y.tolist()) if c == 3]))
        outcomes = Counter()
        for sequence, c in zip(sequences, y.tolist()):
            distances = [discordant_pairs(sequence, prototype) for prototype in EXPECTED[N]]
            outcomes['other_strictly_nearer' if distances[c] > min(distances) else
                     'own_tied_nearest' if distances.count(min(distances)) > 1 else 'own_unique_nearest'] += 1
        assert entry['prototype_oracle']['overall']['counts'] == {outcome: outcomes[outcome] for outcome in ar.ORACLE_OUTCOMES}
        pairs = list(combinations(EXPECTED[N], 2))
        assert entry['prototype_distances']['kendall']['minimum'] == min(discordant_pairs(a, b) for a, b in pairs)
        assert entry['prototype_distances']['footrule']['minimum'] == min(
            sum(abs(a.index(item) - b.index(item)) for item in a) for a, b in pairs)
        if name in ar.GENERATION_SEEDS:
            assert entry['generation_targets_met'] == {'rows': 315, 'deletions_equal_k_N': 315, 'kendall_distance_equal_d_N': 315}
            assert entry['corruption']['original_rows'] == report['datasets']['ranks8_original']['corruption']['this_dataset']
    path = tmp_path/'runs'/'data_report.json'
    ar.main(['describe', '--output', str(path)])
    assert json.loads(path.read_text()) == report
    ar.main(['describe', '--output', str(path)])                         # the same content: accepted, file unchanged
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        ar.write_json(path, dict(report, purpose='changed'))


def test_the_module_and_its_report_import_no_harness_model_no_torch_and_no_classifier():
    code = ('import json, sys\nfrom experiments.make_revision import artificial_ranks as ar\nar.describe()\n'
            'print(json.dumps(sorted(sys.modules)))')
    result = subprocess.run([sys.executable, '-c', code], cwd=REPO, capture_output=True, text=True, check=True)
    modules = json.loads(result.stdout.strip().splitlines()[-1])
    assert [module for module in modules if module.split('.')[0] in ('experiments', 'arrowflow', 'torch')] == [
        'experiments', 'experiments.make_revision', 'experiments.make_revision.artificial_ranks',
        'experiments.make_revision.evaluation']
    classifiers = ('sklearn.neighbors', 'sklearn.svm', 'sklearn.ensemble', 'sklearn.neural_network', 'sklearn.linear_model',
                   'sklearn.tree', 'sklearn.discriminant_analysis', 'sklearn.naive_bayes', 'sklearn.dummy')
    assert [module for module in modules if module.startswith(classifiers)] == []
