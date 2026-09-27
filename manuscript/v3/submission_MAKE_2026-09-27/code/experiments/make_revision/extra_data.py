"""The extra datasets of the G4 follow-up ruling (progress.md, 2026-09-14): Wine quality and Segment without exact duplicate
rows (family dedup) and the artificial ranking-pattern datasets (family artificial).

dedup       wine_quality_dedup and segment_dedup. The source is loaded exactly as run_revision.load_dataset loads
            wine_quality (OpenML 40691) and segment (OpenML 36) and refused unless its dataset_hash is the registered one.
            Every row whose float64 feature vector equals that of an earlier row (NaN equal to NaN, -0.0 equal to 0.0) is
            dropped, the first occurrence kept in loaded order; any duplicate group carrying two labels is refused. The
            result is refused unless every pin of DEDUP_PINS holds: source dataset hash, sha256 of the kept row indices,
            removed rows and groups, shape, class counts, feature names, label map, the harness dataset_fingerprint, the
            hash of the declared nested splits, and no duplicate row left.
artificial  ranks8, ranks16 and ranks8_original, loaded only through artificial_ranks.load_artificial(name) (the return
            contract of run_revision.load_dataset). Their pins live in protocols/2026-09-14/artificial_pins.json, written by
            `pin` only when artificial_ranks.py is committed unchanged and the controller's approval file exists; a loaded
            dataset is refused unless every pin holds (shape, missing cells, class counts, feature names, label map, sha256
            of the NaN-normalized features and of the labels, dataset_fingerprint, splits hash, duplicate counts and the
            sha256 of artificial_ranks.py).

python -m experiments.make_revision.extra_data pin [--output P] [--approval A]
    write the artificial pins, refused unless the module is committed and the approval file exists
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from .evaluation import canonical_json, config_id, dataset_fingerprint, make_splits
from .newdata import DESIGN, WORKSPACE_RUNS, DatasetIdentityError, labels_sha256, names_sha256, sha256_file

REPO = Path(__file__).resolve().parents[2]
PROTOCOLS = Path(__file__).with_name('protocols')/'2026-09-14'
ARTIFICIAL_PINS_FILE = PROTOCOLS/'artificial_pins.json'
ARTIFICIAL_MODULE = Path(__file__).with_name('artificial_ranks.py')
APPROVAL_FILE = WORKSPACE_RUNS.parent/'artificial-data-approved.md'
DUPLICATE_RULE = ('exact float64 equality of every feature, NaN equal to NaN and -0.0 equal to 0.0; the first occurrence of each '
                  'feature vector in loaded order is kept')
IDENTITY_COUNTS = ('duplicate_rows', 'duplicate_groups', 'label_conflicting_groups')


# ----------------------------------------------------------------------------- exact duplicate rows, NaN equal to NaN

def row_keys(X):
    """The bytes of every row after NaN payloads are unified and -0.0 is mapped to 0.0: equal keys are equal float64 rows."""
    X = np.array(X, dtype='<f8', copy=True)
    if X.ndim != 2:
        raise ValueError('Expected a 2-D feature matrix')
    X = X + 0.
    X[np.isnan(X)] = np.nan
    return [row.tobytes() for row in np.ascontiguousarray(X)]


def duplicate_groups(X, y):
    """referee_analyses.duplicate_groups with NaN equal to NaN (the same fields; equal to it on data without NaN).

    Returns (per row, the index of its distinct feature vector in first-occurrence order; the counts and the groups)."""
    y = np.asarray(y)
    keys = row_keys(X)
    if len(keys) != len(y):
        raise ValueError('Expected one label per row')
    index = {}
    vector = np.array([index.setdefault(key, len(index)) for key in keys], dtype=np.int64)
    members = {}
    for sample, key in enumerate(vector.tolist()):
        members.setdefault(key, []).append(sample)
    groups = [{'sample_ids': ids, 'labels': y[ids].tolist()} for ids in members.values() if len(ids) > 1]
    sizes = Counter(len(group['sample_ids']) for group in groups)
    return vector, {'n_rows': len(keys), 'n_distinct_rows': len(members), 'duplicate_rows': len(keys) - len(members),
                    'rows_in_duplicate_groups': sum(size * count for size, count in sizes.items()),
                    'duplicate_groups': len(groups),
                    'label_conflicting_groups': sum(len(set(group['labels'])) > 1 for group in groups),
                    'largest_group': max(sizes, default=1),
                    'group_sizes': {str(size): sizes[size] for size in sorted(sizes)}, 'groups': groups}


def deduplicate(X, y):
    """(the kept row indices in loaded order, the duplicate audit of the source); refused if a group carries two labels."""
    vector, counts = duplicate_groups(X, y)
    conflicting = [group['sample_ids'] for group in counts['groups'] if len(set(group['labels'])) > 1]
    if conflicting:
        raise DatasetIdentityError(f'{len(conflicting)} exact duplicate groups carry two labels (first: rows {conflicting[0]})')
    first = {}
    for sample, key in enumerate(vector.tolist()):
        first.setdefault(key, sample)
    return np.asarray(sorted(first.values()), dtype=np.int64), counts


def rows_sha256(indices):
    """sha256 of the canonical JSON list of row indices."""
    return hashlib.sha256(canonical_json([int(i) for i in indices]).encode('utf-8')).hexdigest()


def features_sha256(X):
    """sha256 of the float64 feature bytes with NaN payloads unified (equal to newdata.array_sha256 on data without NaN)."""
    X = np.array(X, dtype=np.float64, copy=True)
    X[np.isnan(X)] = np.nan
    return hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest()


def declared_splits_hash(y):
    return config_id(make_splits(y, DESIGN['outer_folds'], DESIGN['outer_repeats'], DESIGN['inner_folds'], DESIGN['split_seed']))


# ----------------------------------------------------------------------------- family dedup

WINE_QUALITY_FEATURES = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                         'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']
SEGMENT_FEATURES = ['region-centroid-col', 'region-centroid-row', 'region-pixel-count', 'short-line-density-5',
                    'short-line-density-2', 'vedge-mean', 'vegde-sd', 'hedge-mean', 'hedge-sd', 'intensity-mean', 'rawred-mean',
                    'rawblue-mean', 'rawgreen-mean', 'exred-mean', 'exblue-mean', 'exgreen-mean', 'value-mean',
                    'saturation-mean', 'hue-mean']


def _dedup_pin(name, source, openml_id, source_hash, source_shape, kept_sha, removed, groups, largest, counts, names, label_map,
               dataset_hash, splits_hash):
    return {'name': name, 'role': 'family', 'source': source, 'source_openml_data_id': openml_id,
            'source_dataset_hash': source_hash, 'source_shape': list(source_shape), 'rule': DUPLICATE_RULE,
            'kept_rows_sha256': kept_sha, 'removed_rows': removed, 'duplicate_groups': groups, 'largest_group': largest,
            'label_conflicting_groups': 0, 'remaining_duplicate_rows': 0,
            'shape': [source_shape[0] - removed, source_shape[1]], 'class_counts': list(counts), 'feature_names': list(names),
            'label_map': list(label_map), 'dataset_hash': dataset_hash, 'splits_hash': splits_hash}


# Source hashes: the registered bridge_knn run's wine_quality and segment manifests (run_revision.load_dataset). The removal
# counts equal the duplicate audit (runs/2026-09-13-referee-analyses/duplicates/duplicate_groups.csv: 240 rows in 220 groups
# and 224 rows in 222 groups, no label conflicts); the other values were computed on the cached official downloads on
# 2026-09-14 and are re-derived independently (pandas drop_duplicates) by the tests.
DEDUP_PINS = (
    _dedup_pin('wine_quality_dedup', 'wine_quality', 40691,
               'ce355ca6e4b5ba0392eba133945437af61776a27eea18ff0e0c3055f7b1c0f66', (1599, 11),
               '0910d98913b9fbeaabac4fcf9fd5a9600caa8848e9c8bb5a577189a2f0b26871', 240, 220, 4, (640, 535, 184), WINE_QUALITY_FEATURES,
               ('quality<=5', 'quality==6', 'quality>=7'), '35f48a3fcb0237f779e1ca6e9a283c1584967ae3f4dae799c599333936d63bf3', 'd18e3ba601a9cf8b'),
    _dedup_pin('segment_dedup', 'segment', 36,
               'a2a1dbbbaf3c2983c8fd53be626118de95e52a8a0fb2dcd992f2a0de1544ed24', (2310, 19),
               'b546bb83c0e43be3e370ba1da486754a815e462db3faef18eac03ced33fada58', 224, 222, 3, (297, 300, 299, 300, 292, 300, 298), SEGMENT_FEATURES,
               ('brickface', 'cement', 'foliage', 'grass', 'path', 'sky', 'window'), '2cb2038b6721bc0dbcece7ee69c6d5c03531a4c38c8e2a974e263bfe8c6eb5c0', 'fb59c0afaf9399ff'),
)
DEDUP_PIN_BY_NAME = {pin['name']: pin for pin in DEDUP_PINS}
DEDUP_DATASETS = tuple(DEDUP_PIN_BY_NAME)


def dedup_identity(X_source, y_source, source_manifest):
    """The deduplicated arrays and every pinned value derived from a loaded source."""
    names, label_map = [str(n) for n in source_manifest['feature_names']], [str(label) for label in source_manifest['label_map']]
    keep, audit = deduplicate(X_source, y_source)
    X, y = np.asarray(X_source, dtype=float)[keep], np.asarray(y_source)[keep]
    remaining = duplicate_groups(X, y)[1]
    identity = {'source_dataset_hash': dataset_fingerprint(X_source, y_source, names, label_map),
                'source_shape': list(np.shape(X_source)), 'kept_rows_sha256': rows_sha256(keep),
                'removed_rows': int(len(y_source) - len(keep)), 'duplicate_groups': audit['duplicate_groups'],
                'largest_group': audit['largest_group'], 'label_conflicting_groups': audit['label_conflicting_groups'],
                'remaining_duplicate_rows': remaining['duplicate_rows'], 'shape': list(X.shape),
                'class_counts': np.bincount(y, minlength=len(label_map)).tolist(), 'feature_names': names, 'label_map': label_map,
                'dataset_hash': dataset_fingerprint(X, y, names, label_map), 'splits_hash': declared_splits_hash(y)}
    return X, y, keep, identity


def load_dedup(name, *, loader=None):
    """(X, y, manifest) of one deduplicated dataset; DatasetIdentityError before anything is returned if any pin fails."""
    from .run_revision import load_dataset
    pin = DEDUP_PIN_BY_NAME.get(name)
    if pin is None:
        raise DatasetIdentityError(f'{name} is not a pinned dedup dataset (pinned: {", ".join(DEDUP_DATASETS)})')
    X_source, y_source, source = (loader or load_dataset)(pin['source'])
    if source.get('dataset_hash') != pin['source_dataset_hash']:
        raise DatasetIdentityError(f'{name}: the source {pin["source"]} dataset_hash {source.get("dataset_hash")} is not the '
                                   f'registered {pin["source_dataset_hash"]}')
    X, y, keep, identity = dedup_identity(X_source, y_source, source)
    wrong = [(key, pin[key], value) for key, value in identity.items() if pin[key] != value]
    if wrong:
        raise DatasetIdentityError(f'{name}: dataset identity mismatch: '
                                   + '; '.join(f'{key} pinned {expected!r}, loaded {observed!r}' for key, expected, observed in wrong))
    manifest = {'dataset_id': name, 'source': f'{source["source"]} ({pin["source"]}) without exact duplicate rows',
                'shape': identity['shape'], 'class_counts': identity['class_counts'], 'feature_names': identity['feature_names'],
                'label_map': identity['label_map'],
                'sample_order': f'loaded order of {pin["source"]} with every later exact duplicate row removed; zero-based '
                                'sample_id over the kept rows (kept_rows maps it to the source sample_id)',
                'dataset_hash': identity['dataset_hash'], 'source_dataset': pin['source'],
                'source_manifest': {key: source[key] for key in ('dataset_id', 'source', 'shape', 'class_counts', 'dataset_hash')},
                'deduplication': {'rule': DUPLICATE_RULE, **{key: identity[key] for key in (
                    'removed_rows', 'duplicate_groups', 'largest_group', 'label_conflicting_groups', 'remaining_duplicate_rows',
                    'kept_rows_sha256')}},
                'kept_rows': keep.tolist(),
                'loader': 'extra_data.load_dedup: run_revision.load_dataset of the source, exact duplicate rows removed, every pin checked'}
    return X, y, manifest


# ----------------------------------------------------------------------------- family artificial

ARTIFICIAL = {'ranks8': {'role': 'family', 'expected_rows': 315, 'expected_classes': 7},
              'ranks16': {'role': 'family', 'expected_rows': 315, 'expected_classes': 7},
              'ranks8_original': {'role': 'descriptive', 'expected_rows': 77, 'expected_classes': 7}}
ARTIFICIAL_DATASETS = tuple(ARTIFICIAL)
CONTRACT_KEYS = ('dataset_id', 'source', 'shape', 'class_counts', 'feature_names', 'label_map', 'sample_order', 'dataset_hash')
ARTIFICIAL_LOADER = 'experiments.make_revision.artificial_ranks:load_artificial'


def artificial_loader():
    """artificial_ranks.load_artificial, imported on use (the module is written by another implementer)."""
    try:
        from .artificial_ranks import load_artificial
    except ImportError as exc:
        raise DatasetIdentityError(f'{ARTIFICIAL_LOADER} cannot be imported ({exc})') from exc
    return load_artificial


def module_committed(path=None):
    """Whether the artificial data module is committed with no uncommitted change, and its last commit."""
    path = Path(path or ARTIFICIAL_MODULE)
    try:
        relative = str(path.resolve().relative_to(REPO))
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', relative], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', relative], cwd=REPO, text=True).strip()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        return {'committed': False, 'commit': None, 'reason': f'{type(exc).__name__}: {exc}'}
    return {'committed': bool(commit) and not dirty and path.is_file(), 'commit': commit or None,
            'reason': None if commit and not dirty else ('uncommitted changes' if dirty else 'never committed')}


def artificial_identity(name, X, y, manifest, module=None):
    """Every pinned value of one artificial dataset as loaded; refused if the loader broke its return contract."""
    if name not in ARTIFICIAL:
        raise DatasetIdentityError(f'{name} is not an artificial dataset ({", ".join(ARTIFICIAL_DATASETS)})')
    missing = [key for key in CONTRACT_KEYS if key not in (manifest or {})]
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if missing or manifest['dataset_id'] != name:
        raise DatasetIdentityError(f'{name}: the loader manifest breaks run_revision.load_dataset\'s contract '
                                   f'(missing {missing}, dataset_id {(manifest or {}).get("dataset_id")!r})')
    names, label_map = [str(n) for n in manifest['feature_names']], [str(label) for label in manifest['label_map']]
    if (X.ndim != 2 or len(X) != len(y) or X.shape[1] != len(names) or not np.issubdtype(y.dtype, np.integer)
            or y.min(initial=0) < 0 or (len(y) and y.max() >= len(label_map)) or np.isinf(X).any()):
        raise DatasetIdentityError(f'{name}: expected finite-or-NaN features, one integer label per row in 0..{len(label_map) - 1}')
    fingerprint = dataset_fingerprint(X, y, names, label_map)
    if fingerprint != manifest['dataset_hash'] or list(manifest['shape']) != list(X.shape):
        raise DatasetIdentityError(f'{name}: the manifest dataset_hash or shape does not describe the returned arrays')
    counts = duplicate_groups(X, y)[1]
    return {'shape': list(X.shape), 'n_missing': int(np.isnan(X).sum()),
            'class_counts': np.bincount(y, minlength=len(label_map)).tolist(), 'feature_names': names, 'label_map': label_map,
            'sha256_X': features_sha256(X), 'sha256_y': labels_sha256(y), 'feature_names_sha256': names_sha256(names),
            'dataset_hash': fingerprint, 'splits_hash': declared_splits_hash(y),
            'duplicates': {key: counts[key] for key in IDENTITY_COUNTS}, 'generator_sha256': sha256_file(module or ARTIFICIAL_MODULE)}


def artificial_entry(name, pin=None):
    """The panel declaration of one artificial dataset: its role and expectations, and its pins once written."""
    entry = {'name': name, **ARTIFICIAL[name], 'loader': ARTIFICIAL_LOADER, 'pinned': pin is not None}
    return {**entry, **(pin or {})}


def read_artificial_pins(path=None):
    """{name: pin} from the pins file, or None if no pins have been written."""
    path = Path(path or ARTIFICIAL_PINS_FILE)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if sorted(record.get('datasets') or {}) != sorted(ARTIFICIAL_DATASETS):
        raise DatasetIdentityError(f'{path} does not pin exactly {", ".join(ARTIFICIAL_DATASETS)}')
    return record['datasets']


def load_artificial_pinned(name, entry, *, loader=None, module=None):
    """(X, y, manifest) of one artificial dataset against its panel entry; DatasetIdentityError if unpinned or any pin fails."""
    if not isinstance(entry, dict) or entry.get('name') != name or not entry.get('pinned'):
        raise DatasetIdentityError(f'{name} is not pinned: the pins are written by `extra_data pin` after approval')
    X, y, manifest = (loader or artificial_loader())(name)
    identity = artificial_identity(name, X, y, manifest, module)
    wrong = [(key, entry.get(key), value) for key, value in identity.items() if entry.get(key) != value]
    if wrong:
        raise DatasetIdentityError(f'{name}: dataset identity mismatch: '
                                   + '; '.join(f'{key} pinned {expected!r}, loaded {observed!r}' for key, expected, observed in wrong))
    manifest = {**{key: manifest[key] for key in CONTRACT_KEYS}, 'identity': identity, 'role': entry['role'],
                'loader': f'extra_data.load_artificial_pinned: {ARTIFICIAL_LOADER} with every pin checked'}
    return np.asarray(X, dtype=float), np.asarray(y), json.loads(canonical_json(manifest))


def pin_artificial(output=None, *, approval=None, loader=None, module=None, committed=None):
    """Write the artificial pins, only if the data module is committed unchanged and the approval file exists."""
    from .run_revision import write_json
    output, approval, module = Path(output or ARTIFICIAL_PINS_FILE), Path(approval or APPROVAL_FILE), Path(module or ARTIFICIAL_MODULE)
    if not approval.is_file():
        raise DatasetIdentityError(f'The artificial data are not approved: {approval} does not exist')
    status = committed if committed is not None else module_committed(module)
    if not status.get('committed'):
        raise DatasetIdentityError(f'{module} is not committed unchanged ({status.get("reason")})')
    pins = {}
    for name, expected in ARTIFICIAL.items():
        X, y, manifest = (loader or artificial_loader())(name)
        identity = artificial_identity(name, X, y, manifest, module)
        if identity['shape'][0] != expected['expected_rows'] or len(identity['class_counts']) != expected['expected_classes']:
            raise DatasetIdentityError(f'{name}: {identity["shape"][0]} rows and {len(identity["class_counts"])} classes; expected '
                                       f'{expected["expected_rows"]} and {expected["expected_classes"]}')
        pins[name] = identity
    record = {'purpose': 'artificial_dataset_pins', 'datasets': pins, 'loader': ARTIFICIAL_LOADER,
              'generator': {'path': 'experiments/make_revision/artificial_ranks.py', 'sha256': sha256_file(module),
                            'commit': status.get('commit')},
              'approval': {'path': approval.name, 'sha256': sha256_file(approval)}}
    write_json(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    pinning = commands.add_parser('pin', help='write the artificial pins after approval')
    pinning.add_argument('--output', type=Path, default=ARTIFICIAL_PINS_FILE)
    pinning.add_argument('--approval', type=Path, default=APPROVAL_FILE)
    args = parser.parse_args(argv)
    try:
        record = pin_artificial(args.output, approval=args.approval)
    except DatasetIdentityError as exc:
        parser.exit(2, f'extra_data pin refused: {exc}\n')
    for name, pin in record['datasets'].items():
        print(f"{name}: {pin['shape']} {pin['class_counts']} missing {pin['n_missing']} dataset_hash {pin['dataset_hash']}")


if __name__ == '__main__':
    main()
