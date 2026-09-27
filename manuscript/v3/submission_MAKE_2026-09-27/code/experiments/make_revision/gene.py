"""Gene-expression cancer study (UCI 401, TCGA PANCAN): audited loader, fold-local gene selection,
rank-input and raw-value model families, and the frozen-model corruption bank.

Gene selection is a pipeline step fitted inside every training partition; the selected gene count is a
tuned candidate axis of every family. The corruption bank perturbs only outer test partitions with per-sample
monotone transforms (exact rank invariance expected for rank-input models) and per-gene log-normal scaling.
"""
import hashlib
import io
import json
import os
import tarfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from urllib.request import urlopen
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_consistent_length, check_is_fitted
from arrowflow.ranking import score_order
from .bridge import FIXED, AdaptiveMultiView
from .comparisons import CONVENTIONAL_GRIDS, StableFootruleKNN, TimedPipeline, conventional_factory, derive_seed
from .evaluation import ModelSpec, candidate_grid, config_id, dataset_fingerprint
from .models import ArrowFlowEstimator, array_hash, numeric

# Scientific sources sealed by run_revision.environment_record next to this module and the harness core.
SOURCE_MODULES = ['experiments.make_revision.bridge', 'experiments.make_revision.multiview',
                  'experiments.make_revision.comparisons', 'experiments.make_revision.secondary_studies']

GENE_ID = 'tcga_pancan_rnaseq'
SOURCE_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz'
ARCHIVE_NAME = 'TCGA-PANCAN-HiSeq-801x20531.tar.gz'
MEMBERS = ('TCGA-PANCAN-HiSeq-801x20531/data.csv', 'TCGA-PANCAN-HiSeq-801x20531/labels.csv')
AUDIT_PATH = Path(__file__).with_name('gene_data_audit.json')
IDENTITY = {'shape': [801, 20531], 'label_map': ['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD'],
            'class_counts': [300, 78, 146, 141, 136]}

N_GENES = (10, 15, 20)
MI_SEED = 401            # fixed (UCI id) so every family ranks genes from the same fold-local estimate
MI_NEIGHBORS = 3
CACHE_KEY_VERSION = 1
MEMO_LIMIT = 8
RANK_INPUT_MODELS = ('af_native_ranks', 'svc_ranked', 'rf_ranked', 'footrule_knn_ranks')
MONOTONE_TRANSFORMS = {'log1p': np.log1p, 'sqrt_abs': lambda x: np.sqrt(np.abs(x)),
                       'signed_square': lambda x: x * np.abs(x), 'scale_0.01': lambda x: x * .01,
                       'scale_100': lambda x: x * 100.}
LOGNORMAL_SIGMAS = (.3, .5, .7, 1.)
CORRUPTION_BASE_SEED = 104729
FOOTRULE_GRID = {'n_neighbors': [1, 3, 5, 11, 21], 'weights': ['uniform', 'distance']}


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


# ----------------------------------------------------------------------------- audited data source

def cache_directory(cache_dir=None):
    return Path(cache_dir or os.environ.get('ARROWFLOW_GENE_CACHE') or Path.home() / '.cache' / 'arrowflow' / 'gene')


def fetch_archive(cache_dir, url=SOURCE_URL):
    """Download once into the cache directory; an existing archive is never re-downloaded or replaced."""
    cache_dir = Path(cache_dir)
    archive = cache_dir / ARCHIVE_NAME
    if archive.is_file():
        return archive
    cache_dir.mkdir(parents=True, exist_ok=True)
    part = archive.with_name(f'{ARCHIVE_NAME}.{os.getpid()}.part')
    try:
        with urlopen(url, timeout=120) as response, part.open('wb') as stream:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                stream.write(chunk)
        os.replace(part, archive)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise FileNotFoundError(f'Could not download {url}; place {ARCHIVE_NAME} in {cache_dir} '
                                f'(ARROWFLOW_GENE_CACHE) and retry') from exc
    return archive


def read_members(archive_path):
    """Exact bytes of the two audited members and the SHA-256 of the archive and of each member."""
    payload = Path(archive_path).read_bytes()
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tar:
        names = tar.getnames()
        for member in MEMBERS:
            if names.count(member) != 1:
                raise ValueError(f'Missing/duplicate audited gene archive member: {member}')
            contents[member] = tar.extractfile(member).read()
    hashes = {'archive_sha256': hashlib.sha256(payload).hexdigest(),
              'member_sha256': {m: hashlib.sha256(contents[m]).hexdigest() for m in MEMBERS}}
    return contents, hashes


def parse_tcga(data_text, labels_text):
    """UCI layout: data.csv (sample-id index, one column per gene) and labels.csv (index, Class)."""
    def frame(text):
        return pd.read_csv(io.StringIO(text) if isinstance(text, str) else io.BytesIO(text), index_col=0)
    data, labels = frame(data_text), frame(labels_text)
    if list(labels.columns) != ['Class'] or list(data.index) != list(labels.index):
        raise ValueError('Sample identifiers of data.csv and labels.csv must align row by row')
    X = data.to_numpy(dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError('Expression values must be finite')
    label_map, y = np.unique(labels['Class'].astype(str).to_numpy(), return_inverse=True)
    return (X, np.asarray(y, dtype=int), [str(v) for v in label_map],
            [str(s) for s in data.index], [str(g) for g in data.columns])


def load_tcga(cache_dir=None, *, audit_path=AUDIT_PATH, identity=IDENTITY):
    """Download once, hash, parse and check identity. The first successful load records the audit
    (archive/member SHA-256, shape, class counts, dataset hash); every later load must match it exactly."""
    archive = fetch_archive(cache_directory(cache_dir))
    contents, hashes = read_members(archive)
    audit_path = Path(audit_path)
    recorded = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    if recorded is not None and (recorded['archive_sha256'] != hashes['archive_sha256']
                                 or recorded['member_sha256'] != hashes['member_sha256']):
        raise ValueError(f'Gene archive bytes differ from the recorded audit {audit_path}')
    X, y, label_map, samples, genes = parse_tcga(contents[MEMBERS[0]], contents[MEMBERS[1]])
    counts = np.bincount(y, minlength=len(label_map)).tolist()
    if (list(X.shape) != list(identity['shape']) or label_map != list(identity['label_map'])
            or counts != list(identity['class_counts'])):
        raise ValueError(f'{GENE_ID}: dataset identity/shape/class-count mismatch')
    if X.min() < 0:
        raise ValueError('Expected nonnegative expression values')
    record = {'dataset_id': GENE_ID, 'source_url': SOURCE_URL, 'archive_name': ARCHIVE_NAME, **hashes,
              'shape': list(X.shape), 'label_map': label_map, 'class_counts': counts,
              'dataset_hash': dataset_fingerprint(X, y, genes, label_map)}
    if recorded is None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open('x') as stream:
            json.dump({**record, 'recorded_at_utc': datetime.now(timezone.utc).isoformat()},
                      stream, indent=2, sort_keys=True)
            stream.write('\n')
        status = 'recorded'
    else:
        if any(recorded.get(key) != value for key, value in record.items()):
            raise ValueError(f'Gene dataset identity differs from the recorded audit {audit_path}')
        status = 'verified'
    manifest = {'dataset_id': GENE_ID, 'source': SOURCE_URL, 'archive_sha256': hashes['archive_sha256'],
                'source_file_sha256': hashes['member_sha256'], 'shape': list(X.shape), 'class_counts': counts,
                'feature_names': genes, 'label_map': label_map,
                'sample_order': 'source row order of data.csv; zero-based sample_id',
                'provider_sample_ids': samples, 'audit': status, 'audit_path': str(audit_path),
                'dataset_hash': record['dataset_hash']}
    return X, y, manifest


# ----------------------------------------------------------------------------- fold-local gene selection

_MEMO = {}


def clear_selection_memo():
    _MEMO.clear()


def reset_pilot_family_state():
    """run_revision's pilot fits every family in one process; clearing the memo before each family makes
    that family's first fit measure the selector (pilot_projection reads selection_source/seconds per fit)."""
    clear_selection_memo()


def _ranking_key(X, y, random_state, n_neighbors):
    return config_id({'estimator': 'sklearn.feature_selection.mutual_info_classif', 'discrete_features': 'auto',
                      'random_state': int(random_state), 'n_neighbors': int(n_neighbors),
                      'key_version': CACHE_KEY_VERSION, 'sklearn': sklearn.__version__, 'numpy': np.__version__,
                      'X': array_hash(X), 'y': array_hash(y)})


def _shared_cache_path(key):
    directory = os.environ.get('ARROWFLOW_GENE_MI_CACHE')
    return Path(directory) / f'{key}.npz' if directory else None


def _load_shared(path, key, n_features):
    if path is None or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['key']) != key:
            raise ValueError(f'Shared selector cache key mismatch: {path}')
        values = np.array(saved['mutual_information'], dtype=float)
    if values.shape != (n_features,) or not np.isfinite(values).all():
        raise ValueError(f'Invalid shared selector cache: {path}')
    return values


def _store_shared(path, key, values):
    if path is None or path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.stem}.{os.getpid()}.part.npz')
    np.savez(temporary, key=np.array(key), mutual_information=values)
    os.replace(temporary, path)


def partition_ranking(X, y, random_state=MI_SEED, n_neighbors=MI_NEIGHBORS):
    """Mutual information of every gene with the labels of one training partition.

    Resolution order: process memo, shared cache (ARROWFLOW_GENE_MI_CACHE), computation (stored to the shared
    cache when configured). The key covers the exact rows and labels, the MI parameters and the sklearn/numpy
    versions, so a hit is identical to recomputation and no held-out row can enter it.
    Returns (values, key, source, seconds, cache_file).
    """
    X, y = numeric(X), np.asarray(y)
    check_consistent_length(X, y)
    if np.isnan(X).any():
        raise ValueError('Gene selection requires complete expression values')
    start = time.perf_counter()
    key = _ranking_key(X, y, random_state, n_neighbors)
    path = _shared_cache_path(key)
    values, source = _MEMO.get(key), 'memo'
    if values is None:
        values, source = _load_shared(path, key, X.shape[1]), 'disk'
    if values is None:
        values = np.asarray(mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=random_state), dtype=float)
        source = 'computed'
    _store_shared(path, key, values)          # a memo hit still fills a configured shared cache (no-op if present)
    values = _readonly(values)
    if key not in _MEMO and len(_MEMO) >= MEMO_LIMIT:
        _MEMO.pop(next(iter(_MEMO)))
    _MEMO[key] = values
    return values, key, source, time.perf_counter() - start, None if path is None else str(path)


class TopGenesByMI(TransformerMixin, BaseEstimator):
    """Top genes by mutual information with the training labels, fitted inside every training partition.

    The ranking comes from partition_ranking (memo/shared cache/computation, all identical); ties are broken by
    the lowest gene index and the selected genes keep their MI order. representation_metadata_ records the
    selected genes, the ranking hash, the cache key and where the ranking came from, and the harness copies it
    into every fit row.
    """
    def __init__(self, n_genes=10, random_state=MI_SEED, n_neighbors=MI_NEIGHBORS):
        self.n_genes = n_genes
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = numeric(X), np.asarray(y)
        check_consistent_length(X, y)
        if (not isinstance(self.n_genes, Integral) or isinstance(self.n_genes, bool)
                or not 1 <= self.n_genes <= X.shape[1]):
            raise ValueError('n_genes must be an integer between 1 and the number of genes')
        values, key, source, seconds, cache_file = partition_ranking(X, y, self.random_state, self.n_neighbors)
        ranking = np.argsort(-values, kind='stable')
        self.selected_ = ranking[:self.n_genes].copy()
        self.mutual_information_ = values[self.selected_].copy()
        self.ranking_hash_ = array_hash(ranking)
        self.partition_key_ = key
        self.n_features_in_ = X.shape[1]
        self.selection_source_ = source
        self.selection_seconds_ = seconds
        self.representation_metadata_ = {
            'selector': 'mutual_info_classif', 'n_genes': int(self.n_genes), 'random_state': int(self.random_state),
            'n_neighbors': int(self.n_neighbors), 'cache_key_version': CACHE_KEY_VERSION,
            'sklearn': sklearn.__version__, 'numpy': np.__version__, 'partition_key': key,
            'training_rows': int(X.shape[0]), 'ranking_hash': self.ranking_hash_,
            'selected_genes': self.selected_.tolist(), 'mutual_information': [float(v) for v in self.mutual_information_],
            'selection_source': source, 'selection_seconds': float(seconds), 'shared_cache_file': cache_file}
        return self

    def transform(self, X):
        check_is_fitted(self, 'selected_')
        X = numeric(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Gene count changed')
        return X[:, self.selected_]


class WithinSampleOrder(TransformerMixin, BaseEstimator):
    """Each row becomes the permutation of its genes in ascending expression (stable: ties by gene index).

    Any strictly increasing per-sample transform of the values leaves this permutation unchanged; the
    rank-input models inherit that exact invariance.
    """
    def fit(self, X, y=None):
        X = numeric(X)
        if not np.isfinite(X).all():
            raise ValueError('Within-sample ranking requires finite values')
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'n_features_in_')
        X = numeric(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Gene count changed')
        return score_order(X)


class RankArrowFlow(ArrowFlowEstimator):
    """ArrowFlow on the within-sample gene order itself; the vocabulary is the selected gene set."""
    def __init__(self, embed_dim=10, widths=(128,), iterations=200, learning_rate=.1, batch_size=32,
                 validation_ratio=.1, p_correct=.01, seed=8129):
        super().__init__(embed_dim=embed_dim, widths=widths, iterations=iterations, learning_rate=learning_rate,
                         batch_size=batch_size, validation_ratio=validation_ratio, p_correct=p_correct, seed=seed)

    def fit(self, X, y):
        self.encoding_seconds_ = 0.
        return self.fit_orders(X, y)

    def predict(self, X):
        self.last_encoding_seconds_ = 0.
        return self.predict_orders(X)


class GenePipeline(TimedPipeline):
    """TimedPipeline whose fit record carries the gene selector's provenance (representation_metadata_)."""
    def fit(self, X, y=None, **params):
        super().fit(X, y, **params)
        self.representation_metadata_ = {**self.named_steps['genes'].representation_metadata_,
                                         'input': 'within_sample_ranks' if 'orders' in self.named_steps else 'raw_values'}
        return self


# ----------------------------------------------------------------------------- model families

def _split(config):
    config = dict(config)
    return config.pop('n_genes'), config


def af_full_factory(config, seed):
    n_genes, config = _split(config)
    return GenePipeline([('genes', TopGenesByMI(n_genes)), ('arrowflow_full', AdaptiveMultiView(config=config, seed=seed))])


def af_native_ranks_factory(config, seed):
    n_genes, config = _split(config)
    return GenePipeline([('genes', TopGenesByMI(n_genes)), ('orders', WithinSampleOrder()),
                         ('arrowflow', RankArrowFlow(embed_dim=n_genes, seed=seed, **config))])


def conventional_gene_factory(family, config, seed, ranked=False):
    """E02 conventional pipeline behind fold-local gene selection; `ranked` feeds within-sample ranks."""
    n_genes, config = _split(config)
    steps = list(conventional_factory(family, config, seed, native=ranked).steps)   # native inserts the positions step
    if ranked:
        steps.insert(0, ('orders', WithinSampleOrder()))
    steps.insert(0, ('genes', TopGenesByMI(n_genes)))
    return GenePipeline(steps)


svc_raw_factory = partial(conventional_gene_factory, 'svc_rbf')
svc_ranked_factory = partial(conventional_gene_factory, 'svc_rbf', ranked=True)
rf_raw_factory = partial(conventional_gene_factory, 'random_forest')
rf_ranked_factory = partial(conventional_gene_factory, 'random_forest', ranked=True)


def footrule_knn_ranks_factory(config, seed):
    n_genes, config = _split(config)
    return GenePipeline([('genes', TopGenesByMI(n_genes)), ('orders', WithinSampleOrder()),
                         ('footrule_knn', StableFootruleKNN(**config))])


def af_full_candidates(n_genes=N_GENES):
    return [{**FIXED, 'widths': [128], 'learning_rate': .1, 'embed_scale': 1, 'degree_offset': 0, 'n_genes': n}
            for n in n_genes]


def af_native_ranks_candidates(n_genes=N_GENES):
    return [{'widths': [128], 'learning_rate': .1, 'iterations': 200, 'batch_size': 32, 'validation_ratio': .1,
             'p_correct': .01, 'n_genes': n} for n in n_genes]


def _families(af_full, af_native, svc, rf, knn):
    from .run_revision import dummy_factory
    return {'af_full': ModelSpec('af_full', af_full_factory, af_full, True),
            'af_native_ranks': ModelSpec('af_native_ranks', af_native_ranks_factory, af_native, True),
            'svc_raw': ModelSpec('svc_raw', svc_raw_factory, svc, False),
            'svc_ranked': ModelSpec('svc_ranked', svc_ranked_factory, svc, False),
            'rf_raw': ModelSpec('rf_raw', rf_raw_factory, rf, True),
            'rf_ranked': ModelSpec('rf_ranked', rf_ranked_factory, rf, True),
            'footrule_knn_ranks': ModelSpec('footrule_knn_ranks', footrule_knn_ranks_factory, knn, False),
            'dummy': ModelSpec('dummy', dummy_factory, [{}], False)}


def registry(protocol):
    """Every family tunes n_genes; conventional grids are crossed with n_genes and sampled to the budget."""
    def sampled(grid):
        return candidate_grid({**grid, 'n_genes': list(N_GENES)}, protocol['candidate_budget'], protocol['candidate_seed'])
    return _families(af_full_candidates(), af_native_ranks_candidates(), sampled(CONVENTIONAL_GRIDS['svc_rbf']),
                     sampled(CONVENTIONAL_GRIDS['random_forest']), sampled(FOOTRULE_GRID))


def tiny_registry(protocol):
    """Integration exercise only (tests and corruption smoke): every family with tiny candidates; never evidence."""
    return _families(
        [{**FIXED, 'n_views': 2, 'iterations': 2, 'widths': [8], 'learning_rate': .1, 'embed_scale': 1, 'degree_offset': -2, 'n_genes': 5}],
        [{'widths': [8], 'learning_rate': .1, 'iterations': 2, 'batch_size': 32, 'validation_ratio': .1, 'p_correct': .01, 'n_genes': n}
         for n in (5, 8)],
        [{'C': 1, 'gamma': 'scale', 'n_genes': 5}],
        [{'n_estimators': 5, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'max_depth': None, 'n_genes': 5}],
        [{'n_neighbors': 3, 'weights': 'uniform', 'n_genes': 5}, {'n_neighbors': 5, 'weights': 'distance', 'n_genes': 8}])


def uses_selector(spec):
    """Whether a registry family selects genes (every candidate carries n_genes)."""
    return bool(spec.candidates) and all('n_genes' in c for c in spec.candidates)


def smoke_spec():
    """Harness smoke for the gene dataset: a tiny rank-input ArrowFlow; never paper evidence."""
    return ModelSpec('smoke_gene_rank_arrowflow', af_native_ranks_factory,
                     [{'widths': [6], 'learning_rate': .1, 'iterations': 4, 'batch_size': 32, 'validation_ratio': .1,
                       'p_correct': .01, 'n_genes': 10}], True)


# ----------------------------------------------------------------------------- frozen-model corruption bank

def corruption_schedule(base_seed=CORRUPTION_BASE_SEED, sigmas=LOGNORMAL_SIGMAS):
    conditions = [{'condition': 'clean', 'family': 'clean', 'severity': None}]
    conditions += [{'condition': name, 'family': 'monotone', 'severity': None} for name in MONOTONE_TRANSFORMS]
    conditions += [{'condition': f'lognormal_{s:g}', 'family': 'lognormal_gene_scaling', 'severity': float(s)} for s in sigmas]
    return {'applied_to': 'outer test partition only; training partitions and fitted models stay clean',
            'monotone_transforms': list(MONOTONE_TRANSFORMS), 'lognormal_sigmas': [float(s) for s in sigmas],
            'base_seed': int(base_seed),
            'draw_seed': "derive_seed(base_seed, dataset_id, outer_repeat, outer_fold, 'gene_lognormal_scaling')",
            'draws': 'one standard normal per gene per outer fold, shared by every model and fitting seed; '
                     'each gene is multiplied by exp(sigma * draw)',
            'rank_input_models': list(RANK_INPUT_MODELS),
            'expected': 'exact prediction invariance of rank-input models under every monotone transform',
            'conditions': conditions}


@dataclass(frozen=True)
class GeneCorruptionCase:
    condition: str
    family: str
    severity: object
    draw_seed: object
    raw: np.ndarray
    raw_hash: str


@dataclass(frozen=True)
class GeneCorruptionBank:
    dataset_id: str
    outer_repeat: int
    outer_fold: int
    query_hash: str
    base_seed: int
    draw_seed: int
    log_scale_draws: np.ndarray
    cases: tuple
    _state_hashes: object

    @classmethod
    def create(cls, X_query, *, dataset_id, outer_repeat, outer_fold, base_seed=CORRUPTION_BASE_SEED, sigmas=LOGNORMAL_SIGMAS):
        query = numeric(X_query)
        if not query.shape[0] or not query.shape[1] or not np.isfinite(query).all():
            raise ValueError('Nonempty finite query partition required')
        if query.min() < 0:
            raise ValueError('Negative expression values: sqrt|x| and x|x| are monotone only on nonnegative data')
        sigmas = tuple(float(s) for s in sigmas)
        if len(set(sigmas)) != len(sigmas) or any(not np.isfinite(s) or s <= 0 for s in sigmas):
            raise ValueError('Distinct positive log-normal sigmas required')
        draw_seed = derive_seed(int(base_seed), str(dataset_id), int(outer_repeat), int(outer_fold), 'gene_lognormal_scaling')
        draws = _readonly(np.random.RandomState(draw_seed).normal(size=query.shape[1]))
        cases = []

        def add(condition, family, severity, seed, raw):
            raw = _readonly(raw)
            cases.append(GeneCorruptionCase(condition, family, severity, seed, raw, array_hash(raw)))

        add('clean', 'clean', None, None, query)
        for name, transform in MONOTONE_TRANSFORMS.items():
            add(name, 'monotone', None, None, transform(query))
        for sigma in sigmas:
            add(f'lognormal_{sigma:g}', 'lognormal_gene_scaling', sigma, draw_seed, query * np.exp(sigma * draws))
        return cls(str(dataset_id), int(outer_repeat), int(outer_fold), array_hash(query), int(base_seed), draw_seed,
                   draws, tuple(cases), MappingProxyType({'log_scale_draws': array_hash(draws)}))

    @property
    def clean(self):
        return self.cases[0].raw

    def assert_intact(self):
        if array_hash(self.log_scale_draws) != self._state_hashes['log_scale_draws']:
            raise ValueError('Shared corruption draws changed')
        for case in self.cases:
            if array_hash(case.raw) != case.raw_hash:
                raise ValueError(f'Shared corruption array changed: {case.condition}')

    def metadata(self):
        self.assert_intact()
        return {'dataset_id': self.dataset_id, 'outer_repeat': self.outer_repeat, 'outer_fold': self.outer_fold,
                'query_hash': self.query_hash, 'query_shape': list(self.clean.shape), 'base_seed': self.base_seed,
                'draw_seed': self.draw_seed, 'draw_stream': 'gene_lognormal_scaling',
                'state_hashes': dict(self._state_hashes),
                'cases': [{'condition': c.condition, 'family': c.family, 'severity': c.severity,
                           'draw_seed': c.draw_seed, 'raw_hash': c.raw_hash} for c in self.cases]}

    def save(self, destination):
        """Save the shared draws and every case hash; refuse existing destinations."""
        metadata = self.metadata()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        with (destination / 'draws.npz').open('xb') as stream:
            np.savez_compressed(stream, log_scale_draws=self.log_scale_draws)
        with (destination / 'manifest.json').open('x') as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
