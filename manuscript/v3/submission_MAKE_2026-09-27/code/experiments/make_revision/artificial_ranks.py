"""Artificial ranking-pattern datasets: the original eight-item rows and their calibrated regeneration at N = 8 and N = 16.

The author's request (2026-09-14): rewrite the artificial rank dataset of arrowflow/benchmark.py:_load_artificial cleanly,
keep the original eight-item data and add one with sixteen items. The design is the recorded ruling on that request and is
final; nothing here was tuned after looking at the data report. benchmark._load_artificial and the generator it calls
(arrowflow/arrowflow.py, DataGraph.generate_sequence_data) are sealed by earlier runs and stay unchanged. This module fits
no model and imports none.

Original rows (ORIGINAL_ROWS, ORIGINAL_SOURCE)
    The 77 hand-written rows of DataGraph.generate_sequence_data in the sortflow repository's ml/sortflow_hybrid_3.py (its
    data_train_net.append and data_test_net.append lines 254-344; the file's sha256 is recorded), copied verbatim as
    (class label, original split, sequence): classes '0'..'6', six 'train' rows per class, then five 'test' rows per class,
    in source order. Every row orders distinct items within 1..8, some items deleted; original_rows validates the rows and
    the calibration table before any use, and the test suite re-parses the source lines and requires an exact match.

Prototypes (prototypes(N), for every even N >= 4, h = N/2)
    0  ascending                                                     [1, 2, ..., N]
    1  evens descending, then odds ascending                         [N, N-2, ..., 2, 1, 3, ..., N-1]
    2  odds ascending, then evens descending                         [1, 3, ..., N-1, N, N-2, ..., 2]
    3  zigzag from the low end                                       [1, N, 2, N-1, ..., h, h+1]
    4  upper half descending interleaved with lower half descending  [N, h, N-1, h-1, ..., h+1, 1]
    5  zigzag from the high end                                      [N, 1, N-1, 2, ..., h+1, h]
    6  interleaved halves                                            [1, h+1, 2, h+2, ..., h, N]
    At N = 8 each prototype is the first original row of its class. The twelve patterns of the ported generator are not
    used: for even N its "spiral inward" pattern (class 11) equals its "zigzag low-start" pattern (class 6), so two of its
    classes coincide.

Corruption calibrated on the original rows (CALIBRATION, corruption_targets, generate_rows)
    An original row with m present items has k = 8 - m deletions and Kendall distance d to its class prototype restricted
    to the row's items (the number of discordant pairs): deletion fraction f = k/8, normalized distance tau = d / C(m, 2).
    CALIBRATION holds (k, d) of each class's 11 rows in ORIGINAL_ROWS order. One row of class c at N items from a
    RandomState rng:
      (i)   draw one of the class's 11 original rows with rng.randint(11) and take its (f, tau);
      (ii)  k_N = floor(f N + 1/2): delete the prototype items at the positions rng.choice(N, k_N, replace=False), keeping
            the order of the rest, so m_N = N - k_N (the draw is made for every row, also when k_N = 0);
      (iii) d_N = floor(tau C(m_N, 2) + 1/2): apply exactly d_N adjacent swaps, each at a pair drawn with rng.randint among
            the adjacent pairs still in prototype order (listed left to right), so every swap adds exactly one inversion
            relative to the prototype; d_N <= C(m_N, 2) always holds.
    Both floors are evaluated in exact rational arithmetic; the float evaluation gives the same k_N and d_N on every
    calibration row at N = 8 and N = 16. At N = 8 the rule reproduces the drawn row's k and d; at N = 16, k_N = 2k.

Features (features, identical for all datasets)
    One column per item, item_1..item_N: the item's relative position idx / (m - 1) in the row (idx zero-based; m >= 2
    present items is enforced), NaN if the item was deleted. Nothing is imputed here: the harness imputes missing cells
    fold-locally for every model. y holds the integer class labels 0..6; label_map is ['0', ..., '6'].

Datasets (load_artificial(name) -> X, y, manifest, with the return contract of run_revision.load_dataset)
    ranks8           N = 8, 45 rows per class (the ported loader's 30 training + 15 test rows), 315 rows, seed 20260908
    ranks16          N = 16, 45 rows per class, 315 rows, seed 20260916
    ranks8_original  the 77 original rows as given (both original splits pooled, source order), the same features
    Generated rows are produced class by class (class 0's 45 rows, then class 1's, ...) from one RandomState per dataset.
    A load is refused (DatasetIdentityError) unless every pin in PINS holds: shape, class counts, the sha256 of X as
    little-endian float64 with NaN payloads normalized, the sha256 of the label list, and the harness dataset_fingerprint.
    PINS also records config_id(evaluation.make_splits(y, 5, 3, 3, 27183)), which the test suite checks.

python -m experiments.make_revision.artificial_ranks describe --output PATH.json
    the descriptive data report of the three datasets: class counts, missing-cell share, exact duplicate feature rows (NaN
    equal to NaN) and their groups, deletion fraction and normalized Kendall distance per class and overall (the original
    rows' beside the generated ones'), the pairwise Kendall and footrule distances of the seven prototypes, and
    prototype-oracle agreement. It is used to change nothing: no classifier is fitted and no harness model is imported. An
    existing file with different content is never replaced.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from math import comb, floor
from pathlib import Path
import numpy as np
from .evaluation import canonical_json, dataset_fingerprint

N_CLASSES = 7
ORIGINAL_ITEMS = 8
ROWS_PER_CLASS = 45
ORIGINAL_DATASET = 'ranks8_original'
DATASETS = {'ranks8': 8, 'ranks16': 16, ORIGINAL_DATASET: 8}            # name -> number of items N
GENERATION_SEEDS = {'ranks8': 20260908, 'ranks16': 20260916}
LABEL_MAP = tuple(str(label) for label in range(N_CLASSES))
PATTERN_NAMES = ('ascending', 'evens_descending_then_odds_ascending', 'odds_ascending_then_evens_descending',
                 'zigzag_from_the_low_end', 'upper_half_descending_interleaved_with_lower_half_descending',
                 'zigzag_from_the_high_end', 'interleaved_halves')
IDENTITY_KEYS = ('shape', 'class_counts', 'sha256_X', 'sha256_y', 'dataset_hash')
ORACLE_OUTCOMES = ('own_unique_nearest', 'own_tied_nearest', 'other_strictly_nearer')
QUANTILE_LEVELS = (0, .1, .25, .5, .75, .9, 1)
CALIBRATION_RULE = ("one row of class c at N items from the dataset's RandomState rng: (i) draw one of the class's 11 "
                    "original rows with rng.randint(11) and take f = k/8 and tau = d/C(m, 2) (m present items, k = 8 - m, "
                    "d = discordant pairs with the class prototype restricted to the row's items); (ii) delete "
                    "k_N = floor(f N + 1/2) prototype items at rng.choice(N, k_N, replace=False), keeping the order of the "
                    "rest; (iii) apply d_N = floor(tau C(N - k_N, 2) + 1/2) adjacent swaps, each at rng.randint among the "
                    "adjacent pairs still in prototype order, left to right; both floors in exact rational arithmetic")
FEATURE_RULE = ('item_i = zero-based position of item i in the row / (m - 1), m >= 2 present items; NaN when item i was '
                'deleted; no load-time imputation (the harness imputes missing cells fold-locally for every model)')

ORIGINAL_SOURCE = {'repository': 'sortflow', 'path': 'ml/sortflow_hybrid_3.py', 'method': 'DataGraph.generate_sequence_data',
                   'method_line': 249, 'lines': (254, 344),
                   'sha256': '659d61cf0975a0a4b1fe143885503fe9a10d7a23c02539806b9714b1e9442e5f'}

# (class label, original split, sequence), verbatim from ORIGINAL_SOURCE in source order.
ORIGINAL_ROWS = (
    # data_train_net.append, lines 254-301: six rows per class
    ('0', 'train', (1, 2, 3, 4, 5, 6, 7, 8)),
    ('0', 'train', (1, 3, 2, 4, 5, 7, 6)),
    ('0', 'train', (3, 2, 1, 4, 5, 6, 8, 7)),
    ('0', 'train', (1, 2, 3, 4, 5, 8, 7, 6)),
    ('0', 'train', (4, 2, 3, 1, 7, 8)),
    ('0', 'train', (1, 2, 4, 3, 5, 6, 7, 8)),
    ('1', 'train', (8, 6, 4, 2, 1, 3, 5, 7)),
    ('1', 'train', (6, 8, 4, 2, 1, 5, 3, 7)),
    ('1', 'train', (8, 4, 6, 2, 1, 3, 7)),
    ('1', 'train', (8, 6, 4, 2, 1, 7, 5, 3)),
    ('1', 'train', (8, 6, 4, 2, 1, 3, 5, 7)),
    ('1', 'train', (8, 4, 1, 3, 5, 7)),
    ('2', 'train', (1, 3, 5, 7, 8, 6, 4, 2)),
    ('2', 'train', (1, 5, 3, 8, 4, 6, 2)),
    ('2', 'train', (3, 1, 5, 7, 8, 2, 4)),
    ('2', 'train', (1, 3, 7, 5, 8, 4, 6)),
    ('2', 'train', (1, 3, 8, 7, 5, 6, 4, 2)),
    ('2', 'train', (1, 3, 6, 8, 7, 4, 2)),
    ('3', 'train', (1, 8, 2, 7, 3, 6, 4, 5)),
    ('3', 'train', (1, 2, 8, 7, 3, 6, 4, 5)),
    ('3', 'train', (1, 2, 7, 3, 6, 4, 5)),
    ('3', 'train', (8, 7, 3, 1, 6, 4, 5)),
    ('3', 'train', (2, 3, 1, 7, 8, 6, 4, 5)),
    ('3', 'train', (1, 7, 2, 8, 3, 4, 5)),
    ('4', 'train', (8, 4, 7, 3, 6, 2, 5, 1)),
    ('4', 'train', (4, 8, 7, 3, 6, 5, 1)),
    ('4', 'train', (8, 3, 7, 4, 6, 2, 1)),
    ('4', 'train', (8, 4, 3, 7, 6, 5, 1)),
    ('4', 'train', (3, 4, 8, 6, 2, 5, 1)),
    ('4', 'train', (8, 2, 7, 3, 6, 4, 1)),
    ('5', 'train', (8, 1, 7, 2, 6, 3, 5, 4)),
    ('5', 'train', (7, 1, 8, 2, 6, 5, 4)),
    ('5', 'train', (8, 2, 7, 1, 3, 5, 4)),
    ('5', 'train', (8, 1, 7, 5, 3, 6, 4)),
    ('5', 'train', (1, 7, 2, 6, 5, 3, 4)),
    ('5', 'train', (8, 1, 7, 6, 2, 3, 5)),
    ('6', 'train', (1, 5, 2, 6, 3, 7, 4, 8)),
    ('6', 'train', (1, 2, 5, 6, 3, 7, 4, 8)),
    ('6', 'train', (6, 5, 2, 1, 3, 7, 4, 8)),
    ('6', 'train', (1, 5, 6, 2, 3, 4, 8)),
    ('6', 'train', (1, 5, 2, 7, 3, 4, 8)),
    ('6', 'train', (2, 5, 1, 6, 3, 4, 8)),
    # data_test_net.append, lines 304-344: five rows per class
    ('0', 'test', (1, 2, 3, 5, 7, 8)),
    ('0', 'test', (2, 3, 5, 7, 8)),
    ('0', 'test', (2, 1, 3, 4, 5, 6, 8, 7)),
    ('0', 'test', (4, 2, 3, 1, 5, 6, 7, 8)),
    ('0', 'test', (1, 5, 3, 4, 2, 6, 7, 8)),
    ('1', 'test', (8, 6, 2, 1, 3, 7)),
    ('1', 'test', (6, 2, 1, 3, 7)),
    ('1', 'test', (6, 8, 4, 2, 7, 3, 5, 1)),
    ('1', 'test', (8, 6, 3, 2, 1, 4, 5, 7)),
    ('1', 'test', (8, 1, 4, 2, 6, 3, 5, 7)),
    ('2', 'test', (3, 5, 7, 8, 4, 2)),
    ('2', 'test', (7, 3, 5, 1, 8, 6, 4, 2)),
    ('2', 'test', (1, 8, 5, 7, 3, 6, 4, 2)),
    ('2', 'test', (5, 3, 1, 7, 8, 6, 4, 2)),
    ('2', 'test', (1, 5, 3, 8, 7, 6, 4, 2)),
    ('3', 'test', (8, 1, 2, 3, 6, 4, 5)),
    ('3', 'test', (3, 2, 7, 1, 6, 4, 5)),
    ('3', 'test', (2, 6, 1, 7, 3, 8, 4, 5)),
    ('3', 'test', (1, 2, 7, 3, 6, 4)),
    ('3', 'test', (1, 8, 4, 7, 3, 6, 2)),
    ('4', 'test', (1, 4, 7, 3, 6, 2, 8)),
    ('4', 'test', (4, 8, 7, 3, 6, 5, 1)),
    ('4', 'test', (8, 4, 6, 2, 5, 1)),
    ('4', 'test', (8, 4, 7, 3, 2, 1)),
    ('4', 'test', (7, 3, 6, 2, 5, 1)),
    ('5', 'test', (1, 8, 7, 6, 3, 5, 4)),
    ('5', 'test', (2, 1, 7, 8, 3, 5, 4)),
    ('5', 'test', (4, 1, 7, 2, 6, 3, 5, 8)),
    ('5', 'test', (8, 7, 5, 6, 3, 2, 4)),
    ('5', 'test', (8, 1, 7, 6, 2, 3, 5)),
    ('6', 'test', (3, 5, 2, 6, 1, 7, 4, 8)),
    ('6', 'test', (8, 5, 2, 6, 3, 7, 4, 1)),
    ('6', 'test', (5, 1, 2, 6, 7, 4, 8)),
    ('6', 'test', (1, 5, 6, 3, 7, 8)),
    ('6', 'test', (1, 5, 2, 3, 6, 7, 4)),
)

# Per class, (k, d) of its 11 original rows in ORIGINAL_ROWS order (its six training rows, then its five test rows): k = 8 - m
# deletions and d = discordant pairs with the class prototype at N = 8 restricted to the row's items. Step (i) draws from
# these entries; original_rows refuses a fixture whose recomputed table differs.
CALIBRATION = (
    ((0, 0), (1, 2), (0, 4), (0, 3), (2, 5), (0, 1), (2, 0), (3, 0), (0, 2), (0, 5), (0, 5)),  # class 0
    ((0, 0), (0, 2), (1, 1), (0, 3), (0, 0), (2, 0), (2, 0), (3, 0), (0, 6), (0, 5), (0, 5)),  # class 1
    ((0, 0), (1, 2), (1, 2), (1, 2), (0, 3), (1, 3), (2, 0), (0, 5), (0, 5), (0, 3), (0, 2)),  # class 2
    ((0, 0), (0, 1), (1, 0), (1, 3), (0, 6), (1, 3), (1, 1), (1, 5), (0, 8), (2, 0), (1, 7)),  # class 3
    ((0, 0), (1, 1), (1, 3), (1, 1), (1, 3), (1, 7), (1, 11), (1, 1), (2, 0), (2, 0), (2, 0)),  # class 4
    ((0, 0), (1, 3), (1, 3), (1, 3), (1, 1), (1, 1), (1, 1), (1, 5), (0, 13), (1, 5), (1, 1)),  # class 5
    ((0, 0), (0, 1), (0, 5), (1, 1), (1, 1), (1, 3), (0, 7), (0, 13), (1, 1), (2, 0), (1, 1)),  # class 6
)

# Identity pins, computed once from the design above. load_artificial refuses a mismatch in any of IDENTITY_KEYS;
# splits_hash (config_id of evaluation.make_splits(y, 5, 3, 3, 27183)) is checked by the test suite.
PINS = {
    'ranks8': {'shape': [315, 8], 'class_counts': [45, 45, 45, 45, 45, 45, 45],
               'sha256_X': '92e2b67eb070062defa7a3402edb9f8fc781ef1c837f44d08974036a56fb23ce',
               'sha256_y': '8f775caddfedb37afb6a722a0b4acb90a19290f07ac7783470034c281907c901',
               'dataset_hash': '4570b16626b7a14297f396d84f58fd6da2973068a74eabecf409d1c7615a2750',
               'splits_hash': '1a9d02126d9db5db'},
    'ranks16': {'shape': [315, 16], 'class_counts': [45, 45, 45, 45, 45, 45, 45],
                'sha256_X': 'c358842d80d48693d2a59328ab681a3d104bc2ea6406d32cdb996dcca192cba4',
                'sha256_y': '8f775caddfedb37afb6a722a0b4acb90a19290f07ac7783470034c281907c901',
                'dataset_hash': '67b43496def4b8f32238bf4b3b00176e92b4539ed76162418d8d159acbfea70c',
                'splits_hash': '1a9d02126d9db5db'},
    'ranks8_original': {'shape': [77, 8], 'class_counts': [11, 11, 11, 11, 11, 11, 11],
                        'sha256_X': '3e63448c1daf480ee3d685cf2c52f4a112c460374e4fb5a0cbe139cb9a23c032',
                        'sha256_y': '69c894479916fffa1d4cd7a5a4befa8eb1bfc1727a51f6c0b175915492d02a64',
                        'dataset_hash': '9545c8d56cceeebe455b44f11471b3db9c57552dcf74b11ccc186cd0f694dfaf',
                        'splits_hash': '996b378b652b9504'},
}


class DatasetIdentityError(ValueError):
    """A built artificial dataset, or the fixture it is built from, differs from its pin."""


# ----------------------------------------------------------------------------- orderings and distances

def prototypes(n_items):
    """The seven class prototypes over the items 1..N (even N >= 4, h = N/2), in class order."""
    if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)) or n_items < 4 or n_items % 2:
        raise ValueError(f'Prototypes are defined for even N >= 4, not {n_items!r}')
    N = int(n_items)
    h = N // 2
    evens_descending, odds_ascending = list(range(N, 0, -2)), list(range(1, N, 2))
    return (tuple(range(1, N + 1)),
            tuple(evens_descending + odds_ascending),
            tuple(odds_ascending + evens_descending),
            tuple(item for i in range(h) for item in (1 + i, N - i)),
            tuple(item for i in range(h) for item in (N - i, h - i)),
            tuple(item for i in range(h) for item in (N - i, 1 + i)),
            tuple(item for i in range(h) for item in (1 + i, h + 1 + i)))


def validate_sequence(sequence, n_items):
    """`sequence` as a tuple of ints, refused unless it orders at least two distinct items within 1..n_items."""
    row = tuple(sequence)
    if (len(row) < 2 or len(set(row)) != len(row)
            or any(isinstance(item, bool) or not isinstance(item, (int, np.integer)) or not 1 <= item <= n_items
                   for item in row)):
        raise ValueError(f'{row} is not an ordering of at least two distinct items within 1..{n_items}')
    return tuple(int(item) for item in row)


def kendall_distance(sequence, reference):
    """The number of item pairs that `sequence` orders against `reference`, restricted to the items of `sequence`."""
    rank = {item: position for position, item in enumerate(reference)}
    ranks = [rank[item] for item in sequence]
    return sum(1 for i, j in combinations(range(len(ranks)), 2) if ranks[i] > ranks[j])


def footrule_distance(first, second):
    """Spearman's footrule between two orderings of the same items: the sum of the items' absolute position differences."""
    if sorted(first) != sorted(second):
        raise ValueError('The footrule compares two orderings of the same items')
    position = {item: index for index, item in enumerate(second)}
    return sum(abs(index - position[item]) for index, item in enumerate(first))


# ----------------------------------------------------------------------------- the original rows and the calibrated corruption

def calibration_table(rows):
    """Per class, (k, d) of each of its rows in row order: k = 8 - m deletions, d = kendall_distance to the class prototype
    at N = 8."""
    reference = prototypes(ORIGINAL_ITEMS)
    table = [[] for _ in range(N_CLASSES)]
    for label, _, sequence in rows:
        table[int(label)].append((ORIGINAL_ITEMS - len(sequence), kendall_distance(sequence, reference[int(label)])))
    return tuple(tuple(entries) for entries in table)


def original_rows():
    """ORIGINAL_ROWS validated: six training rows per class, then five test rows per class, every row an ordering of
    distinct items within 1..8, and the calibration table recomputed from the rows equal to CALIBRATION."""
    rows = [(label, split, validate_sequence(sequence, ORIGINAL_ITEMS)) for label, split, sequence in ORIGINAL_ROWS]
    layout = ([(label, 'train') for label in LABEL_MAP for _ in range(6)]
              + [(label, 'test') for label in LABEL_MAP for _ in range(5)])
    if [(label, split) for label, split, _ in rows] != layout:
        raise DatasetIdentityError('ORIGINAL_ROWS: expected six training rows per class, then five test rows per class')
    if calibration_table(rows) != CALIBRATION:
        raise DatasetIdentityError('ORIGINAL_ROWS: the recomputed calibration table differs from CALIBRATION')
    return rows


def corruption_targets(k, d, n_items):
    """(k_N, d_N) at N items for an original row with k deletions and d discordant pairs: k_N = floor(f N + 1/2) with
    f = k/8 and d_N = floor(tau C(N - k_N, 2) + 1/2) with tau = d / C(8 - k, 2), in exact rational arithmetic."""
    k_n = floor(Fraction(k, ORIGINAL_ITEMS) * n_items + Fraction(1, 2))
    d_n = floor(Fraction(d, comb(ORIGINAL_ITEMS - k, 2)) * comb(n_items - k_n, 2) + Fraction(1, 2))
    return k_n, d_n


def corrupt(reference, k_n, d_n, rng):
    """Delete the items of `reference` at the positions rng.choice(len(reference), k_n, replace=False), keeping the order of
    the rest, then apply d_n adjacent swaps, each at rng.randint over the adjacent pairs still in reference order (listed
    left to right), so that every swap adds exactly one inversion."""
    rank = {item: position for position, item in enumerate(reference)}
    deleted = set(rng.choice(len(reference), k_n, replace=False).tolist())
    row = [item for position, item in enumerate(reference) if position not in deleted]
    if not 0 <= d_n <= comb(len(row), 2):
        raise ValueError(f'{d_n} inversions do not fit {len(row)} items')
    for _ in range(d_n):
        in_order = [j for j in range(len(row) - 1) if rank[row[j]] < rank[row[j + 1]]]
        j = in_order[rng.randint(len(in_order))]
        row[j], row[j + 1] = row[j + 1], row[j]
    return tuple(row)


def generate_rows(n_items, rows_per_class, seed):
    """[(class, sequence, record)] class by class from one RandomState(seed); record holds the drawn row's index within its
    class's CALIBRATION entry (class_row), its (k, d) and the targets (k_N, d_N)."""
    original_rows()                                    # refuse a changed fixture before anything is drawn from CALIBRATION
    reference = prototypes(n_items)
    rng = np.random.RandomState(seed)
    rows = []
    for label in range(N_CLASSES):
        for _ in range(rows_per_class):
            class_row = int(rng.randint(len(CALIBRATION[label])))
            k, d = CALIBRATION[label][class_row]
            k_n, d_n = corruption_targets(k, d, n_items)
            rows.append((label, corrupt(reference[label], k_n, d_n, rng),
                         {'class_row': class_row, 'k': k, 'd': d, 'k_N': k_n, 'd_N': d_n}))
    return rows


# ----------------------------------------------------------------------------- features and the three pinned datasets

def feature_names(n_items):
    return [f'item_{item}' for item in range(1, n_items + 1)]


def features(sequences, n_items):
    """X[row, item - 1] = the item's zero-based position in the row / (m - 1); NaN where the item was deleted."""
    X = np.full((len(sequences), n_items), np.nan, dtype=np.float64)
    for row, sequence in enumerate(sequences):
        sequence = validate_sequence(sequence, n_items)            # m >= 2 present items, so m - 1 >= 1
        for position, item in enumerate(sequence):
            X[row, item - 1] = position / (len(sequence) - 1)
    return X


def sequences_from_features(X):
    """The present items of each row ordered by relative position: the inverse of features."""
    rows = []
    for values in np.asarray(X, dtype=np.float64):
        present = np.flatnonzero(~np.isnan(values))
        rows.append(tuple(int(item) + 1 for item in present[np.argsort(values[present], kind='stable')]))
    return rows


def build(name):
    """(X, y, generation records or None for the original rows) of one dataset, before any pin is checked."""
    if name == ORIGINAL_DATASET:
        rows = original_rows()
        sequences, labels, records = [row[2] for row in rows], [int(row[0]) for row in rows], None
    elif name in GENERATION_SEEDS:
        generated = generate_rows(DATASETS[name], ROWS_PER_CLASS, GENERATION_SEEDS[name])
        sequences, labels, records = [row[1] for row in generated], [row[0] for row in generated], [row[2] for row in generated]
    else:
        raise DatasetIdentityError(f'{name} is not an artificial ranks dataset (known: {", ".join(DATASETS)})')
    return features(sequences, DATASETS[name]), np.asarray(labels, dtype=np.int64), records


def array_sha256(X):
    """sha256 of X as little-endian float64 bytes with NaN payloads normalized, as evaluation.dataset_fingerprint hashes X."""
    X = np.asarray(X, dtype='<f8').copy()
    X[np.isnan(X)] = np.nan
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


def labels_sha256(y):
    """sha256 of the label list: one decimal label per line, UTF-8."""
    return hashlib.sha256('\n'.join(str(int(label)) for label in np.asarray(y).tolist()).encode('utf-8')).hexdigest()


def identity(X, y, names):
    return {'shape': list(X.shape), 'class_counts': np.bincount(y, minlength=N_CLASSES).tolist(),
            'sha256_X': array_sha256(X), 'sha256_y': labels_sha256(y),
            'dataset_hash': dataset_fingerprint(X, y, names, LABEL_MAP)}


def _plain(value):
    """The JSON form of a value (tuples become lists; NaN refused)."""
    return json.loads(canonical_json(value))


def load_artificial(name):
    """(X, y, manifest) of one artificial ranks dataset; DatasetIdentityError before anything is returned if a pin fails."""
    X, y, _ = build(name)
    n_items = DATASETS[name]
    names = feature_names(n_items)
    built = identity(X, y, names)
    wrong = [(key, PINS[name][key], built[key]) for key in IDENTITY_KEYS if built[key] != PINS[name][key]]
    if wrong:
        raise DatasetIdentityError(f'{name}: dataset identity mismatch: '
                                   + '; '.join(f'{key} pinned {pinned!r}, built {value!r}' for key, pinned, value in wrong))
    generation = {'n_items': n_items, 'n_classes': N_CLASSES, 'pattern_names': PATTERN_NAMES,
                  'prototypes': prototypes(n_items), 'original_source': ORIGINAL_SOURCE}
    if name == ORIGINAL_DATASET:
        first, last = ORIGINAL_SOURCE['lines']
        source = (f"sortflow {ORIGINAL_SOURCE['path']} {ORIGINAL_SOURCE['method']}, lines {first}-{last} "
                  f"(sha256 {ORIGINAL_SOURCE['sha256']})")
        sample_order = 'ORIGINAL_ROWS order (source order: the 42 training rows, then the 35 test rows); zero-based sample_id'
        generation.update(kind='original_rows', rows_per_class=len(ORIGINAL_ROWS) // N_CLASSES,
                          original_split=[split for _, split, _ in ORIGINAL_ROWS])
    else:
        source = f'artificial_ranks: the seven prototypes over {n_items} items, corruption calibrated on the original rows'
        sample_order = (f"generation order: class 0's {ROWS_PER_CLASS} rows, then class 1's, ..., then class 6's; "
                        'zero-based sample_id')
        generation.update(kind='generated', rows_per_class=ROWS_PER_CLASS, seed=GENERATION_SEEDS[name],
                          random_state='one numpy.random.RandomState(seed) per dataset', calibration_rule=CALIBRATION_RULE,
                          calibration=CALIBRATION)
    manifest = {'dataset_id': name, 'source': source, 'shape': built['shape'], 'class_counts': built['class_counts'],
                'feature_names': names, 'label_map': list(LABEL_MAP), 'sample_order': sample_order,
                'dataset_hash': built['dataset_hash'], 'identity': {key: built[key] for key in ('sha256_X', 'sha256_y')},
                'features': FEATURE_RULE, 'generation': generation,
                'loader': 'artificial_ranks.load_artificial: built in memory and refused unless every pin holds'}
    return X, y, _plain(manifest)


# ----------------------------------------------------------------------------- the descriptive data report

def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {'n': int(values.size), 'mean': float(values.mean()),
            'quantiles': {str(level): float(np.quantile(values, level)) for level in QUANTILE_LEVELS}}


def corruption_summary(sequences, y, n_items):
    """Deletion fraction (N - m) / N and normalized Kendall distance d / C(m, 2) to the own class prototype over the row's
    present items, overall and per class."""
    reference = prototypes(n_items)
    labels = np.asarray(y)
    fraction = np.array([(n_items - len(sequence)) / n_items for sequence in sequences])
    distance = np.array([kendall_distance(sequence, reference[label]) / comb(len(sequence), 2)
                         for sequence, label in zip(sequences, labels.tolist())])
    return {measure: {'overall': distribution(values),
                      'per_class': {LABEL_MAP[label]: distribution(values[labels == label]) for label in range(N_CLASSES)}}
            for measure, values in (('deletion_fraction', fraction), ('normalized_kendall_distance', distance))}


def duplicate_rows(X, y):
    """Exact duplicate feature rows: float64 equality of every cell with NaN equal to NaN (and -0.0 equal to 0.0), with the
    groups of rows sharing a vector listed by their first row."""
    X = np.asarray(X, dtype='<f8') + 0.
    X[np.isnan(X)] = np.nan
    labels = np.asarray(y)
    members = {}
    for sample, row in enumerate(np.ascontiguousarray(X)):
        members.setdefault(row.tobytes(), []).append(sample)
    groups = [{'sample_ids': ids, 'labels': labels[ids].tolist()} for ids in members.values() if len(ids) > 1]
    sizes = Counter(len(group['sample_ids']) for group in groups)
    return {'n_rows': len(X), 'n_distinct_rows': len(members), 'duplicate_rows': len(X) - len(members),
            'rows_in_duplicate_groups': sum(size * count for size, count in sizes.items()),
            'duplicate_groups': len(groups),
            'label_conflicting_groups': sum(len(set(group['labels'])) > 1 for group in groups),
            'largest_group': max(sizes, default=1), 'group_sizes': {str(size): sizes[size] for size in sorted(sizes)},
            'groups': groups}


def prototype_distances(n_items):
    """The pairwise Kendall and footrule distances of the seven prototypes and their minimum over the 21 pairs."""
    reference = prototypes(n_items)
    pairs = list(combinations(range(N_CLASSES), 2))
    record = {}
    for measure, distance, largest in (('kendall', kendall_distance, comb(n_items, 2)),
                                       ('footrule', footrule_distance, n_items * n_items // 2)):
        matrix = [[distance(first, second) for second in reference] for first in reference]
        minimum = min(matrix[a][b] for a, b in pairs)
        record[measure] = {'matrix': matrix, 'minimum': minimum, 'largest_possible': largest,
                           'minimum_normalized': minimum / largest,
                           'minimum_pairs': [[LABEL_MAP[a], LABEL_MAP[b]] for a, b in pairs if matrix[a][b] == minimum]}
    return record


def prototype_oracle(sequences, y, n_items):
    """Per row, the Kendall distance over its present items to each of the seven prototypes (normalized by the common
    C(m, 2), so the integer counts decide): own_unique_nearest when the own prototype alone has the smallest distance,
    own_tied_nearest when it shares the smallest with another prototype, other_strictly_nearer otherwise. Nothing is
    fitted: the prototypes are the generating patterns."""
    reference = prototypes(n_items)
    outcomes = []
    for sequence, label in zip(sequences, np.asarray(y).tolist()):
        distances = [kendall_distance(sequence, prototype) for prototype in reference]
        nearest = min(distances)
        outcomes.append((label, 'other_strictly_nearer' if distances[label] > nearest else
                         'own_unique_nearest' if distances.count(nearest) == 1 else 'own_tied_nearest'))

    def tally(selected):
        counts = Counter(selected)
        return {'rows': len(selected), 'counts': {outcome: counts[outcome] for outcome in ORACLE_OUTCOMES},
                'shares': {outcome: counts[outcome] / len(selected) for outcome in ORACLE_OUTCOMES}}
    return {'overall': tally([outcome for _, outcome in outcomes]),
            'per_class': {LABEL_MAP[c]: tally([outcome for label, outcome in outcomes if label == c]) for c in range(N_CLASSES)}}


def describe():
    """The descriptive data report of the three datasets, computed from the loaded (pin-checked) arrays."""
    loaded = {name: load_artificial(name) for name in DATASETS}
    original_X, original_y, _ = loaded[ORIGINAL_DATASET]
    original = corruption_summary(sequences_from_features(original_X), original_y, ORIGINAL_ITEMS)
    datasets = {}
    for name, (X, y, manifest) in loaded.items():
        n_items = DATASETS[name]
        sequences = sequences_from_features(X)
        entry = {'rows': len(y), 'n_items': n_items, 'shape': manifest['shape'], 'class_counts': manifest['class_counts'],
                 'dataset_hash': manifest['dataset_hash'], **manifest['identity'],
                 'missing_cells': int(np.isnan(X).sum()), 'missing_share': float(np.isnan(X).mean()),
                 'duplicates': duplicate_rows(X, y),
                 'corruption': {'this_dataset': corruption_summary(sequences, y, n_items)},
                 'prototype_distances': prototype_distances(n_items),
                 'prototype_oracle': prototype_oracle(sequences, y, n_items)}
        if name in GENERATION_SEEDS:
            entry['corruption']['original_rows'] = original
            records = build(name)[2]
            reference = prototypes(n_items)
            entry['generation_targets_met'] = {
                'rows': len(records),
                'deletions_equal_k_N': sum(n_items - len(s) == r['k_N'] for s, r in zip(sequences, records)),
                'kendall_distance_equal_d_N': sum(kendall_distance(s, reference[label]) == r['d_N']
                                                  for s, label, r in zip(sequences, y.tolist(), records))}
        datasets[name] = entry
    return _plain({
        'report': 'artificial ranks data report', 'module': 'experiments.make_revision.artificial_ranks',
        'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'purpose': 'descriptive only and used to change nothing; no classifier is fitted and no harness model is imported',
        'design': {'pattern_names': PATTERN_NAMES, 'generation_seeds': GENERATION_SEEDS, 'rows_per_class': ROWS_PER_CLASS,
                   'calibration_rule': CALIBRATION_RULE, 'feature_rule': FEATURE_RULE, 'original_source': ORIGINAL_SOURCE,
                   'quantile_levels': QUANTILE_LEVELS, 'quantile_method': 'numpy.quantile, linear interpolation',
                   'oracle_outcomes': ORACLE_OUTCOMES,
                   'duplicates': 'exact float64 equality of every feature cell, NaN equal to NaN'},
        'datasets': datasets})


def write_json(path, value):
    """run_revision.write_json's rule, copied because importing run_revision would import the harness models: create the
    parent directory, write sorted indented JSON, and never replace an existing file with different content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'Refusing to overwrite {path}; use a new output path')
        return
    with path.open('x') as stream:
        stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    described = commands.add_parser('describe', help='write the descriptive data report of the three datasets')
    described.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = describe()
    write_json(args.output, report)
    for name, entry in report['datasets'].items():
        this = entry['corruption']['this_dataset']
        shares = entry['prototype_oracle']['overall']['shares']
        print(f"{name}: {entry['rows']} rows x {entry['n_items']} items, class counts {entry['class_counts']}, missing share "
              f"{entry['missing_share']:.4f}, duplicate rows {entry['duplicates']['duplicate_rows']} in "
              f"{entry['duplicates']['duplicate_groups']} groups, mean f {this['deletion_fraction']['overall']['mean']:.4f}, "
              f"mean tau {this['normalized_kendall_distance']['overall']['mean']:.4f}, prototype oracle own unique "
              f"{shares['own_unique_nearest']:.4f} / tied {shares['own_tied_nearest']:.4f} / other "
              f"{shares['other_strictly_nearer']:.4f}")
    print(args.output)


if __name__ == '__main__':
    main()
