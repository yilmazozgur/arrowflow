"""G4 follow-up: extra_data (the dedup pins and refusals, the NaN-aware duplicate audit, and the artificial pin gate with a
stand-in loader; the real artificial module is never fitted or pinned here)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import get_data_home
from experiments.make_revision import extra_data as ed
from experiments.make_revision import run_revision as rr
from experiments.make_revision.evaluation import config_id, dataset_fingerprint, make_splits
from experiments.make_revision.newdata import DatasetIdentityError
from experiments.make_revision.referee_analyses import duplicate_groups as referee_duplicate_groups

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO.parent/'.superpowers'/'sdd'/'2026-09-12-arrowflow-story-restoration-plan'/'runs'


def _cached(data_id):
    home = Path(get_data_home())/'openml'/'openml.org'
    return all(path.is_file() for path in (home/'api'/'v1'/'json'/'data'/f'{data_id}.gz',
                                           home/'api'/'v1'/'json'/'data'/'features'/f'{data_id}.gz'))


needs_openml = pytest.mark.skipif(not (_cached(40691) and _cached(36)), reason='wine_quality and segment are not cached here')


# ----------------------------------------------------------------------------- duplicate rows with NaN equal to NaN

def test_duplicate_groups_equals_the_referee_audit_without_nan_and_treats_nan_as_equal():
    rng = np.random.RandomState(3)
    X = rng.randint(0, 3, size=(60, 2)).astype(float)
    X[5, 0] = -0.0
    y = rng.randint(0, 2, size=60)
    ours, theirs = ed.duplicate_groups(X, y), referee_duplicate_groups(X, y)
    assert np.array_equal(ours[0], theirs[0]) and ours[1] == theirs[1]
    X = np.array([[1., np.nan], [1., np.nan], [np.nan, np.nan], [2., 0.], [np.nan, np.nan], [1., 2.]])
    X[4] = np.frombuffer(np.array([0x7ff8000000000001], dtype='<u8').tobytes(), dtype='<f8')[0]   # another NaN payload
    vector, counts = ed.duplicate_groups(X, [0, 0, 1, 1, 2, 0])
    assert vector.tolist() == [0, 0, 1, 2, 1, 3]
    assert (counts['duplicate_rows'], counts['duplicate_groups'], counts['label_conflicting_groups']) == (2, 2, 1)
    with pytest.raises(ValueError, match='NaN'):
        referee_duplicate_groups(X, [0, 0, 1, 1, 2, 0])


def test_deduplicate_keeps_the_first_occurrence_and_refuses_conflicting_labels():
    X = np.array([[0., 1.], [2., 3.], [0., 1.], [np.nan, 1.], [np.nan, 1.], [2., 3.], [5., 5.]])
    keep, counts = ed.deduplicate(X, [0, 1, 0, 2, 2, 1, 0])
    assert keep.tolist() == [0, 1, 3, 6] and counts['duplicate_rows'] == 3 and counts['duplicate_groups'] == 3
    assert keep.tolist() == np.flatnonzero(~pd.DataFrame(X).duplicated(keep='first').to_numpy()).tolist()
    with pytest.raises(DatasetIdentityError, match='carry two labels'):
        ed.deduplicate(X, [0, 1, 1, 2, 2, 1, 0])


def test_features_sha256_unifies_nan_payloads_and_equals_the_plain_hash_without_nan():
    from experiments.make_revision.newdata import array_sha256
    X = np.arange(12, dtype=float).reshape(4, 3)
    assert ed.features_sha256(X) == array_sha256(X)
    other = X.copy()
    X[1, 1], other[1, 1] = np.nan, np.frombuffer(np.array([0x7ff8000000000003], dtype='<u8').tobytes(), dtype='<f8')[0]
    assert ed.features_sha256(X) == ed.features_sha256(other)


# ----------------------------------------------------------------------------- family dedup

def test_the_dedup_pins_hold_the_audit_counts_and_the_registered_source_hashes():
    audit = {row['dataset']: row for row in pd.read_csv(RUNS/'2026-09-13-referee-analyses'/'duplicates'/'duplicate_groups.csv').to_dict('records')} \
        if (RUNS/'2026-09-13-referee-analyses').is_dir() else None
    assert ed.DEDUP_DATASETS == ('wine_quality_dedup', 'segment_dedup')
    for pin, rows, removed, groups in ((ed.DEDUP_PINS[0], 1359, 240, 220), (ed.DEDUP_PINS[1], 2086, 224, 222)):
        assert (pin['shape'][0], pin['removed_rows'], pin['duplicate_groups'], pin['label_conflicting_groups']) == (rows, removed, groups, 0)
        assert sum(pin['class_counts']) == rows and pin['shape'][1] == len(pin['feature_names']) == pin['source_shape'][1]
        if audit is not None:
            row = audit[pin['source']]
            assert (row['duplicate_rows'], row['duplicate_groups'], row['label_conflicting_groups']) == (removed, groups, 0)
            assert row['n_distinct_rows'] == rows
    assert ed.DEDUP_PIN_BY_NAME['wine_quality_dedup']['source_dataset_hash'] == 'ce355ca6e4b5ba0392eba133945437af61776a27eea18ff0e0c3055f7b1c0f66'
    assert ed.DEDUP_PIN_BY_NAME['segment_dedup']['source_dataset_hash'] == 'a2a1dbbbaf3c2983c8fd53be626118de95e52a8a0fb2dcd992f2a0de1544ed24'
    manifests = {name: RUNS/'2026-09-12-bridge-knn'/name/'manifest.json' for name in ('wine_quality', 'segment')}
    for pin in ed.DEDUP_PINS:
        if manifests[pin['source']].is_file():
            assert json.loads(manifests[pin['source']].read_text())['dataset_hash'] == pin['source_dataset_hash']


@needs_openml
def test_the_cached_sources_deduplicate_to_every_pin_independently_of_the_module():
    for pin in ed.DEDUP_PINS:
        X_source, y_source, source = rr.load_dataset(pin['source'])
        keep = np.flatnonzero(~pd.DataFrame(X_source).duplicated(keep='first').to_numpy())       # pandas: NaN equal to NaN
        X, y = X_source[keep], y_source[keep]
        assert pd.DataFrame(X).duplicated().sum() == 0
        assert pd.DataFrame(np.column_stack([X_source, y_source])).drop_duplicates().shape[0] == len(keep)   # no label conflict
        splits = make_splits(y, 5, 3, 3, 27183)
        assert ed.rows_sha256(keep) == pin['kept_rows_sha256'] and config_id(splits) == pin['splits_hash']
        assert dataset_fingerprint(X, y, source['feature_names'], source['label_map']) == pin['dataset_hash']
        assert (list(X.shape), np.bincount(y).tolist(), source['feature_names'], source['label_map']) == (
            pin['shape'], pin['class_counts'], pin['feature_names'], pin['label_map'])
        loaded_X, loaded_y, manifest = ed.load_dedup(pin['name'])
        assert np.array_equal(loaded_X, X) and np.array_equal(loaded_y, y) and manifest['kept_rows'] == keep.tolist()
        assert manifest['dataset_hash'] == pin['dataset_hash'] and set(rr.load_dataset(pin['source'])[2]) <= set(manifest) | {'dataset_id'}
        assert manifest['deduplication']['remaining_duplicate_rows'] == 0


def fake_source(pin_name, mutate=None):
    """A stand-in for run_revision.load_dataset: a small source with duplicates whose pin is patched to match."""
    rng = np.random.RandomState(8)
    distinct = np.array([[a, b, c] for a in range(4) for b in range(4) for c in range(4)], dtype=float)     # 64 distinct rows
    labels_of = np.arange(len(distinct)) % 3
    order = np.concatenate([np.arange(len(distinct)), rng.choice(len(distinct), 26, replace=False)])       # 26 later copies
    order = order[np.argsort(rng.rand(len(order)), kind='stable')]
    X, y = distinct[order], labels_of[order]
    names, labels = ['a', 'b', 'c'], ['l0', 'l1', 'l2']
    manifest = {'dataset_id': 'fake', 'source': 'OpenML data_id=0', 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                'feature_names': names, 'label_map': labels, 'sample_order': 'source row order; zero-based sample_id',
                'dataset_hash': dataset_fingerprint(X, y, names, labels)}
    if mutate:
        X, y, manifest = mutate(X, y, manifest)
    return lambda name: (X.copy(), y.copy(), dict(manifest))


def patched_pin(monkeypatch, name='wine_quality_dedup'):
    X, y, manifest = fake_source(name)('source')
    _, _, keep, identity = ed.dedup_identity(X, y, manifest)
    pin = dict(ed.DEDUP_PIN_BY_NAME[name], **identity)
    monkeypatch.setitem(ed.DEDUP_PIN_BY_NAME, name, pin)
    return pin


def test_load_dedup_accepts_a_matching_source_and_refuses_every_pin_mismatch(monkeypatch):
    pin = patched_pin(monkeypatch)
    X, y, manifest = ed.load_dedup('wine_quality_dedup', loader=fake_source('wine_quality_dedup'))
    assert len(y) == pin['shape'][0] < 90 and manifest['deduplication']['removed_rows'] == 90 - len(y)
    assert ed.duplicate_groups(X, y)[1]['duplicate_rows'] == 0
    for field, value in (('kept_rows_sha256', '0' * 64), ('shape', [1, 3]), ('class_counts', [1, 1, 1]), ('feature_names', ['x', 'b', 'c']),
                         ('label_map', ['a', 'b', 'c']), ('dataset_hash', '1' * 64), ('splits_hash', '2' * 16), ('removed_rows', 0),
                         ('source_dataset_hash', '3' * 64)):
        monkeypatch.setitem(ed.DEDUP_PIN_BY_NAME, 'wine_quality_dedup', dict(pin, **{field: value}))
        with pytest.raises(DatasetIdentityError, match='is not the registered' if field == 'source_dataset_hash' else field):
            ed.load_dedup('wine_quality_dedup', loader=fake_source('wine_quality_dedup'))
    monkeypatch.setitem(ed.DEDUP_PIN_BY_NAME, 'wine_quality_dedup', pin)

    def conflict(X, y, manifest):
        y = y.copy()
        duplicate = next(i for i in range(1, len(y)) if any((X[i] == X[j]).all() for j in range(i)))
        first = next(j for j in range(duplicate) if (X[duplicate] == X[j]).all())
        y[duplicate] = (y[first] + 1) % 3
        return X, y, dict(manifest, dataset_hash=pin['source_dataset_hash'])
    with pytest.raises(DatasetIdentityError, match='carry two labels'):
        ed.load_dedup('wine_quality_dedup', loader=fake_source('wine_quality_dedup', conflict))
    with pytest.raises(DatasetIdentityError, match='not a pinned dedup dataset'):
        ed.load_dedup('wine')


# ----------------------------------------------------------------------------- family artificial (stand-in only)

def standin_artificial(name, *, rows=None, shift=0.):
    """A stand-in with load_artificial's contract: relative item positions with NaN for deleted items, labels 0..6."""
    n_items = 16 if name == 'ranks16' else 8
    rows = rows or ed.ARTIFICIAL[name]['expected_rows']
    rng = np.random.RandomState(len(name) + n_items)
    y = np.arange(rows) % 7
    X = np.array([rng.permutation(n_items) / (n_items - 1) for _ in range(rows)]) + shift
    X[rng.rand(rows, n_items) < .15] = np.nan
    names, labels = [f'item_{i + 1}' for i in range(n_items)], [str(label) for label in range(7)]
    manifest = {'dataset_id': name, 'source': 'stand-in', 'shape': list(X.shape), 'class_counts': np.bincount(y).tolist(),
                'feature_names': names, 'label_map': labels, 'sample_order': 'stand-in', 'dataset_hash': dataset_fingerprint(X, y, names, labels)}
    return X, y, manifest


def test_pin_requires_the_committed_module_and_the_approval_file_and_writes_every_identity(tmp_path):
    module = tmp_path/'artificial_ranks.py'
    module.write_text('# stand-in generator\n')
    approval = tmp_path/'artificial-data-approved.md'
    output = tmp_path/'artificial_pins.json'
    committed = {'committed': True, 'commit': 'abc', 'reason': None}
    with pytest.raises(DatasetIdentityError, match='not approved'):
        ed.pin_artificial(output, approval=approval, loader=standin_artificial, module=module, committed=committed)
    approval.write_text('approved\n')
    with pytest.raises(DatasetIdentityError, match='not committed'):
        ed.pin_artificial(output, approval=approval, loader=standin_artificial, module=module,
                          committed={'committed': False, 'commit': None, 'reason': 'uncommitted changes'})
    assert not output.exists()
    with pytest.raises(DatasetIdentityError, match='rows'):
        ed.pin_artificial(output, approval=approval, loader=lambda name: standin_artificial(name, rows=70), module=module, committed=committed)
    record = ed.pin_artificial(output, approval=approval, loader=standin_artificial, module=module, committed=committed)
    pins = ed.read_artificial_pins(output)
    assert sorted(pins) == sorted(ed.ARTIFICIAL_DATASETS) and record['generator']['commit'] == 'abc'
    assert pins['ranks16']['shape'] == [315, 16] and pins['ranks8_original']['shape'] == [77, 8] and pins['ranks8']['n_missing'] > 0
    entry = ed.artificial_entry('ranks8', pins['ranks8'])
    X, y, manifest = ed.load_artificial_pinned('ranks8', entry, loader=standin_artificial, module=module)
    assert manifest['identity'] == pins['ranks8'] and manifest['role'] == 'family' and np.isnan(X).any()
    with pytest.raises(DatasetIdentityError, match='sha256_X'):
        ed.load_artificial_pinned('ranks8', entry, loader=lambda name: standin_artificial(name, shift=1e-9), module=module)
    module.write_text('# a changed generator\n')
    with pytest.raises(DatasetIdentityError, match='generator_sha256'):
        ed.load_artificial_pinned('ranks8', entry, loader=standin_artificial, module=module)
    with pytest.raises(DatasetIdentityError, match='not pinned'):
        ed.load_artificial_pinned('ranks8', ed.artificial_entry('ranks8'), loader=standin_artificial, module=module)


def test_the_artificial_identity_refuses_a_broken_loader_contract(tmp_path):
    module = tmp_path/'m.py'
    module.write_text('')
    X, y, manifest = standin_artificial('ranks8')
    for broken in ({k: v for k, v in manifest.items() if k != 'label_map'}, dict(manifest, dataset_id='ranks16'),
                   dict(manifest, dataset_hash='0' * 64)):
        with pytest.raises(DatasetIdentityError):
            ed.artificial_identity('ranks8', X, y, broken, module)
    with pytest.raises(DatasetIdentityError):
        ed.artificial_identity('ranks8', np.where(np.isnan(X), np.inf, X), y, manifest, module)


def test_module_committed_reports_an_untracked_or_missing_module(tmp_path):
    status = ed.module_committed(tmp_path/'absent.py')
    assert status['committed'] is False


needs_pins = pytest.mark.skipif(not ed.ARTIFICIAL_PINS_FILE.is_file(), reason='the artificial pins are not written')


@needs_pins
def test_the_committed_artificial_pins_hold_for_the_committed_module_and_equal_a_fresh_pin(tmp_path):
    """Loads only (no model is fitted): the pins written after the controller's approval hold for the committed module."""
    import hashlib
    status = ed.module_committed()
    record = json.loads(ed.ARTIFICIAL_PINS_FILE.read_text())
    assert status['committed'] and record['generator']['sha256'] == hashlib.sha256(ed.ARTIFICIAL_MODULE.read_bytes()).hexdigest()
    approval = tmp_path/'approved.md'
    approval.write_text('fresh pin for the test\n')
    fresh = ed.pin_artificial(tmp_path/'pins.json', approval=approval, committed=status)
    assert fresh['datasets'] == record['datasets'] and fresh['generator'] == record['generator']
    expected = {'ranks8': ([315, 8], [45] * 7, 38, 1), 'ranks16': ([315, 16], [45] * 7, 23, 0), 'ranks8_original': ([77, 8], [11] * 7, 3, 0)}
    for name, (shape, counts, duplicate_rows, conflicting) in expected.items():        # the controller's approval table
        pin = record['datasets'][name]
        assert (pin['shape'], pin['class_counts'], pin['duplicates']['duplicate_rows'], pin['duplicates']['label_conflicting_groups']) == (
            shape, counts, duplicate_rows, conflicting)
        X, y, manifest = ed.load_artificial_pinned(name, ed.artificial_entry(name, pin))
        assert manifest['dataset_hash'] == pin['dataset_hash'] and np.isnan(X).sum() == pin['n_missing'] and np.nanmin(X) >= 0 and np.nanmax(X) <= 1
