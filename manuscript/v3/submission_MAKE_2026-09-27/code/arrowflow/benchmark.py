"""
ArrowFlow Benchmark Harness
============================
Unified script for running ArrowFlow experiments against classical baselines.
Handles dataset loading, ArrowFlow data conversion, multi-simulation runs,
baseline comparisons, and CSV logging.

Usage:
    from arrowflow.benchmark import run_benchmark, ArrowFlowConfig

    config = ArrowFlowConfig(
        no_of_filters=[128],
        layer_types=['sort', 'sort'],
        no_of_iters=200,
        moe_no_of_networks=7,
    )
    results = run_benchmark('iris', config, n_sims=5, baselines=['rf', 'svm', 'mlp'])

Or from command line:
    python -m test.benchmark_arrowflow --dataset iris --filters 128 --sims 5
"""

import os
import sys

# --- Thread control: MUST be set before any numpy/torch import ---
# When using ProcessPoolExecutor(fork), child processes inherit the parent's
# thread pools. Without this, each worker spawns ~32 OpenMP threads, causing
# massive contention (N_workers × 32 threads competing for CPU).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import csv
import copy
import time
import warnings
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

# ArrowFlow imports
from arrowflow.arrowflow import SortFlowHybridNetwork, SortFlow_MoE, DataGraph
import torch
torch.set_num_threads(1)  # 1 thread per process; parallelism comes from multiprocessing

# Results directory
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parallelism settings — use 80% of available cores
# ---------------------------------------------------------------------------
_N_CPUS = os.cpu_count() or 4
MAX_WORKERS = max(1, int(_N_CPUS * 0.8))   # 80% utilization cap


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArrowFlowConfig:
    """Configuration for a single ArrowFlow experiment run."""
    # Network architecture
    no_of_filters: List[int] = field(default_factory=lambda: [128])
    layer_types: List[str] = field(default_factory=lambda: ['sort', 'sort'])
    filter_rfs: Optional[List] = None
    last_layer_update: bool = True

    # Training
    learning_rate: float = 0.1
    lr_tensor: float = 0.01
    no_of_iters: int = 200
    batch_size: int = 32
    val_data_ratio: float = 0.1
    multistep_lr: bool = True
    ratio_data_backprop: float = 0.5
    motion_normalization_mult: float = 0.125
    change_probability_when_decision_correct: float = 0.01

    # Distance
    distance_computation_metric: str = 'l1'
    dist_nonlinearity: float = 1.2
    missing_data_cost_weight: float = 0.5
    missing_data_loc_mult: float = 0.5
    average_method_motion: str = 'mean'
    average_method_position: str = 'mean'

    # Embedding
    no_of_embedding_dim: int = 16
    n_views: int = 1
    poly_expansion: bool = False
    expand_diff: bool = False
    pol_deg: int = 3

    # Encoding Mode
    # -------------
    # Controls how raw data is converted into ArrowFlow's sorted-list format.
    # 'projection'      — Standard pipeline: poly → scaler → random projection → argsort.
    #                     Best for tabular data with continuous features.
    # 'native'          — Data is already in permutation format (e.g., artificial sequence
    #                     data). Bypasses the projection pipeline entirely. Each view in
    #                     the ensemble receives the same permutation data but trains with
    #                     different random seeds for filter initialization diversity.
    # 'countvectorizer' — For text data: CountVectorizer → extract top-k word indices as
    #                     the permutation. Preserves vocabulary structure directly, which
    #                     is far more informative than projecting TF-IDF features.
    encoding_mode: str = 'projection'
    cv_vocabulary_size: int = 1000      # CountVectorizer max_features
    cv_vector_size: int = 64            # Length of the resulting permutation for CV mode

    # Multi-View Ensemble
    # -------------------
    # Multi-view ensemble trains N independent ArrowFlow networks, each on a
    # different random projection of the data, and combines predictions via
    # majority vote. This is the single most impactful improvement to ArrowFlow
    # accuracy (2-3x error reduction), because each view captures different
    # ordinal relationships and their disagreements cancel out noise.
    n_ensemble_views: int = 1           # Number of independent views for ensemble (1 = no ensemble)
    projection_strategy: str = 'random' # 'random', 'target_aware', 'diverse', or 'calibrated'
    use_augmentation: bool = False      # Whether to augment permutation data during training
    n_augmentations: int = 1            # Number of augmented copies per sample
    max_swaps: int = 2                  # Max adjacent swaps per augmented copy
    lda_ratio: float = 0.3             # Fraction of embed_dim for LDA in target_aware/diverse
    adaptive_mode: bool = False         # Auto-select all hyperparameters based on dataset properties

    # MoE
    moe_no_of_networks: int = 1
    moe_data_ratio: float = 1.1

    # Runtime
    multiprocessing: bool = False
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    verbose: int = 1
    eval_period: int = 1
    evaluate_train_data: bool = False

    # Data options
    initial_filter_with_data: bool = False
    frequency_based_filter: bool = False
    frequency_dict: Any = None
    data_initial: Any = None
    backprop_signal_replication_flag: bool = False

    # Embedding layer (for text)
    embedding_layer_flag: bool = False
    vocabulary_size: int = 1000
    vector_size: int = 64

    # Derived
    problem: str = 'classification'
    network_type: str = 'feedforward'
    train_type: str = 'supervised'
    regression_method: str = 'quantize'
    nonlinear_activation: bool = False
    apply_nonlinearity_distance: bool = False
    min_motion_zeroed: int = 0
    metric_flag: Optional[List] = None
    tensor_gradient_threshold: float = 0.0001
    profile_flag: bool = False
    s_neigh: Any = None
    t_neigh: Any = None
    atomic_f: Any = None
    data_type: str = 'benchmark'

    def __post_init__(self):
        if self.filter_rfs is None:
            self.filter_rfs = [None] * (len(self.no_of_filters) + 1)
        if self.metric_flag is None:
            self.metric_flag = [None] * (len(self.no_of_filters) + 1)

    def to_dict(self):
        """Return a flat dict for CSV logging."""
        return {
            'filters': str(self.no_of_filters),
            'layer_types': str(self.layer_types),
            'moe': self.moe_no_of_networks,
            'iters': self.no_of_iters,
            'lr': self.learning_rate,
            'batch_size': self.batch_size,
            'distance_metric': self.distance_computation_metric,
            'poly_expansion': self.poly_expansion,
            'pol_deg': self.pol_deg,
            'embed_dim': self.no_of_embedding_dim,
            'last_layer_update': self.last_layer_update,
            'encoding_mode': self.encoding_mode,
            'n_ensemble_views': self.n_ensemble_views,
            'projection_strategy': self.projection_strategy,
            'use_augmentation': self.use_augmentation,
            'adaptive_mode': self.adaptive_mode,
        }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, subsample: Optional[float] = None,
                 random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load a dataset and return (X_train, y_train, X_test, y_test, n_classes).
    Optionally subsample the training set.
    """
    if name == 'iris':
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target
    elif name == 'wine':
        from sklearn.datasets import load_wine
        data = load_wine()
        X, y = data.data, data.target
    elif name == 'wine_quality':
        X, y = _load_wine_quality()
    elif name == 'digits':
        from sklearn.datasets import load_digits
        data = load_digits()
        X, y = data.data, data.target
    elif name == 'letter':
        X, y = _load_letter_recognition()
    elif name == 'vehicle':
        X, y = _load_vehicle()
    elif name == 'segment':
        X, y = _load_segment()
    elif name == 'moons':
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
    elif name == 'circles':
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=1000, noise=0.2, factor=0.5, random_state=1)
    elif name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
    elif name == 'artificial':
        return _load_artificial(subsample=subsample, random_state=random_state)
    elif name == 'mnist_pca':
        return _load_mnist_pca(subsample=subsample, random_state=random_state)
    elif name == 'imdb':
        return _load_imdb(subsample=subsample, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    n_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    if subsample is not None and subsample < 1.0:
        n_sub = max(n_classes, int(len(X_train) * subsample))
        idx = np.random.RandomState(random_state).choice(len(X_train), n_sub, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test, n_classes


def _load_wine_quality():
    """Load Wine Quality from sklearn's openml or UCI."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml(name='wine-quality-red', version=1, as_frame=False, parser='auto')
        X, y = data.data, data.target.astype(int)
        # Combine low/high quality for balanced classes
        y = np.where(y <= 5, 0, np.where(y <= 6, 1, 2))
    except Exception:
        from sklearn.datasets import load_wine
        data = load_wine()
        X, y = data.data, data.target
    return X, y


def _load_letter_recognition():
    """Load Letter Recognition from openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='letter', version=1, as_frame=False, parser='auto')
    X = data.data.astype(float)
    # Convert letter labels to integers
    labels = data.target
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])
    return X, y


def _load_vehicle():
    """Load Vehicle Silhouettes from openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='vehicle', version=1, as_frame=False, parser='auto')
    X = data.data.astype(float)
    labels = data.target
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])
    return X, y


def _load_segment():
    """Load Image Segmentation from openml."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name='segment', version=1, as_frame=False, parser='auto')
    X = data.data.astype(float)
    labels = data.target
    unique_labels = np.unique(labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in labels])
    return X, y


def _load_artificial(vocab_size=16, n_train_per_class=30, n_test_per_class=15,
                     subsample=None, random_state=42):
    """Load artificial permutation dataset as position-encoded features.

    Generates data via DataGraph.generate_sequence_data, then converts each
    permutation to a position vector (item i -> its position index) so that
    baselines can work on the same data.
    """
    data_graph = DataGraph('artificial')
    train, test, adj, n_classes, max_sz = data_graph.generate_sequence_data(
        vocab_size=vocab_size, n_train_per_class=n_train_per_class,
        n_test_per_class=n_test_per_class, seed=random_state
    )

    def _to_numpy(data, size):
        X, y = [], []
        for dp in data:
            vec = [float(x) for x in dp[0]]
            vec += [0.0] * (size - len(vec))
            X.append(vec[:size])
            y.append(int(dp[1]))
        return np.array(X), np.array(y)

    X_train, y_train = _to_numpy(train, max_sz)
    X_test, y_test = _to_numpy(test, max_sz)

    if subsample is not None and subsample < 1.0:
        n_sub = max(n_classes, int(len(X_train) * subsample))
        idx = np.random.RandomState(random_state).choice(len(X_train), n_sub, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test, n_classes


def _load_mnist_pca(n_components=32, subsample=None, random_state=42):
    """Load MNIST, apply PCA, return in standard format."""
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.decomposition import PCA

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 784).astype(float) / 255.0
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.numpy().reshape(-1, 784).astype(float) / 255.0
    y_test = test_ds.targets.numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    n_classes = 10

    if subsample is not None and subsample < 1.0:
        n_sub = max(n_classes, int(len(X_train) * subsample))
        idx = np.random.RandomState(random_state).choice(len(X_train), n_sub, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test, n_classes


def _load_imdb(vocabulary_size=1000, vector_size=64, subsample=None, random_state=42):
    """Load IMDB dataset as TF-IDF features (real-valued vectors).

    Uses TF-IDF instead of raw BoW indices so that IMDB data goes through the same
    project_random_space encoding pipeline as tabular data, producing valid permutations.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'IMDB_Dataset.csv'))
    df = pd.read_csv(csv_path)
    df["Category"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"], df['Category'], test_size=0.2, random_state=random_state
    )

    tfidf = TfidfVectorizer(max_features=vocabulary_size)
    X_train = tfidf.fit_transform(X_train_text).toarray().astype(float)
    X_test = tfidf.transform(X_test_text).toarray().astype(float)
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    if subsample is not None and subsample < 1.0:
        n_sub = max(2, int(len(X_train) * subsample))
        idx = np.random.RandomState(random_state).choice(len(X_train), n_sub, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test, 2


def _load_imdb_bow(vocabulary_size=1000, vector_size=64, subsample=None, random_state=42):
    """Load IMDB as CountVectorizer bag-of-words → argsort permutation.

    This encoding uses CountVectorizer to compute word counts, then applies
    argsort to produce a permutation of {1, ..., vector_size}. This is
    equivalent to the standard encoding pipeline (features → argsort) but
    the "features" are word counts rather than random projections of TF-IDF,
    which preserves the vocabulary structure directly.

    The vocabulary is limited to the top vector_size most frequent words
    (across the training corpus). For each document, argsort ranks these
    words by their count in that document, producing a valid permutation
    where the most frequent word in that document appears first.

    Returns:
        data_train: ArrowFlow format list of [permutation, label, magnitude].
        data_test: ArrowFlow format list.
        adj_list_input: Vocabulary for ArrowFlow (string token IDs 1..vector_size).
        n_classes: Number of classes (2 for IMDB).
        y_train, y_test: Label arrays (for baseline comparison).
    """
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'IMDB_Dataset.csv'))
    df = pd.read_csv(csv_path)
    df["Category"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"], df['Category'], test_size=0.2, random_state=random_state
    )
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    if subsample is not None and subsample < 1.0:
        n_sub = max(2, int(len(X_train_text) * subsample))
        idx = np.random.RandomState(random_state).choice(len(X_train_text), n_sub, replace=False)
        X_train_text = X_train_text.iloc[idx]
        y_train = y_train[idx]

    # Use exactly vector_size words so that argsort produces a complete
    # permutation of {0, 1, ..., vector_size-1}. Each document's permutation
    # ranks these words by their count in that document.
    cv = CountVectorizer(max_features=vector_size)
    X_train_bow = cv.fit_transform(X_train_text).toarray().astype(float)
    X_test_bow = cv.transform(X_test_text).toarray().astype(float)

    # Add small random noise to break ties deterministically.
    # Many words have count 0 in short documents; without noise, their
    # relative ordering is arbitrary and may differ between train/test.
    rng = np.random.RandomState(random_state)
    X_train_bow += rng.uniform(0, 0.01, X_train_bow.shape)
    X_test_bow += rng.uniform(0, 0.01, X_test_bow.shape)

    # argsort descending: most frequent word gets position 0
    # +1 for 1-indexed tokens as ArrowFlow expects
    perm_train = np.argsort(-X_train_bow, axis=1) + 1
    perm_test = np.argsort(-X_test_bow, axis=1) + 1

    adj_list_input = [str(i + 1) for i in range(vector_size)]

    data_train = [[list(map(str, perm_train[i])), str(y_train[i]), 1]
                   for i in range(len(perm_train))]
    data_test = [[list(map(str, perm_test[i])), str(y_test[i]), 1]
                  for i in range(len(perm_test))]

    return data_train, data_test, adj_list_input, 2, y_train, y_test


def _load_imdb_nonzero(vocabulary_size=1000, vector_size=64, subsample=None, random_state=42):
    """Load IMDB using the original nonzero-index encoding.

    This replicates the original ArrowFlow IMDB encoding from test_arrowflow.py:
    1. CountVectorizer(max_features=vocabulary_size) builds a BoW matrix with
       `vocabulary_size` columns (e.g. 1000 words).
    2. For each document, extract the column indices where the count is nonzero
       — i.e., which of the vocabulary words appear in that document.
    3. Take the first `vector_size` (e.g. 64) nonzero indices, zero-pad if fewer.
    4. Pass through DataGraph.convert_sequence_data_sortnet() which remaps the
       word indices to contiguous tokens and builds adj_list_input.

    This encoding preserves WHICH words are present (set membership), which is
    far more informative for sentiment than ranking word counts (argsort).
    The resulting vocabulary for the SortFlow network can be large (up to
    vocabulary_size), since each unique word index becomes a token.

    Args:
        vocabulary_size: Number of most frequent words for CountVectorizer.
        vector_size: Max number of nonzero indices per document.
        subsample: If < 1.0, fraction of training data to use.
        random_state: Random seed for reproducibility.

    Returns:
        data_train, data_test, adj_list_input, n_classes, y_train, y_test
    """
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer

    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs', 'IMDB_Dataset.csv'))
    df = pd.read_csv(csv_path)
    df["Category"] = df.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review"], df['Category'], test_size=0.2, random_state=random_state
    )
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Build BoW matrix with large vocabulary (e.g. 1000 words)
    cv = CountVectorizer(max_features=vocabulary_size)
    X_train_bow = cv.fit_transform(X_train_text).toarray()
    X_test_bow = cv.transform(X_test_text).toarray()

    if subsample is not None:
        n_sub = int(subsample) if subsample >= 1 else max(2, int(len(X_train_bow) * subsample))
        idx = np.random.RandomState(random_state).choice(len(X_train_bow), n_sub, replace=False)
        X_train_bow = X_train_bow[idx]
        y_train = y_train[idx]

    # For each document, extract nonzero column indices (which words are present).
    # This is the original encoding: word presence as set membership.
    def extract_nonzero_indices(X, vector_size):
        matrix = []
        for vector in X:
            nz_indices = np.nonzero(vector)[0]
            # Take first vector_size indices, pad with 0 if fewer
            vec_dense = np.pad(
                nz_indices[:vector_size],
                (0, max(0, vector_size - len(nz_indices[:vector_size]))),
                'constant'
            )
            matrix.append(vec_dense)
        return np.asarray(matrix)

    X_train_idx = extract_nonzero_indices(X_train_bow, vector_size)
    X_test_idx = extract_nonzero_indices(X_test_bow, vector_size)

    # Use convert_sequence_data_sortnet to build adj_list_input and remap to
    # contiguous tokens. We combine train + test to get consistent vocabulary.
    data_graph = DataGraph('benchmark')
    X_all = np.concatenate([X_train_idx, X_test_idx])
    y_all = np.concatenate([y_train, y_test])
    data_all, adj_list_input = data_graph.convert_sequence_data_sortnet(X_all, y_all)

    # Split back into train/test
    n_train = len(X_train_idx)
    data_train = data_all[:n_train]
    data_test = data_all[n_train:]

    print(f"  IMDB nonzero encoding: vocab={len(adj_list_input)}, "
          f"seq_len={vector_size}, train={n_train}, test={len(data_test)}")

    return data_train, data_test, adj_list_input, 2, y_train, y_test


def _load_artificial_native(vocab_size=16, n_train_per_class=30, n_test_per_class=15,
                            random_state=42):
    """Load artificial data in native permutation format (no encoding pipeline).

    The artificial dataset consists of permutations classified by their ordering
    pattern. Returning them directly (instead of converting to position vectors
    and back) preserves the full ordinal structure that sort layers are designed
    to exploit.

    Returns:
        data_train: ArrowFlow format list of [permutation, label, magnitude].
        data_test: ArrowFlow format list.
        adj_list_input: Vocabulary tokens.
        n_classes: Number of classes.
    """
    data_graph = DataGraph('artificial')
    data_train, data_test, adj_list_input, n_classes, _ = \
        data_graph.generate_sequence_data(
            vocab_size=vocab_size,
            n_train_per_class=n_train_per_class,
            n_test_per_class=n_test_per_class,
            seed=random_state
        )
    return data_train, data_test, adj_list_input, n_classes


# ---------------------------------------------------------------------------
# ArrowFlow data conversion
# ---------------------------------------------------------------------------

def convert_to_arrowflow_data(X, y, config: ArrowFlowConfig, W_random=None):
    """
    Convert numpy arrays (X, y) into ArrowFlow's sorted list format.
    Always uses random projection + argsort encoding (StandardScaler → project → argsort).
    Returns (data_net, adj_list_input, W_random).
    """
    data_graph = DataGraph('benchmark')

    data_net, adj_list_input, W_random = data_graph.project_random_space(
        X, y, W_random,
        poly_expansion=config.poly_expansion,
        expand_diff_features=config.expand_diff,
        pol_deg=config.pol_deg,
        no_dimensions=config.no_of_embedding_dim,
        n_views=config.n_views
    )

    return data_net, adj_list_input, W_random


# ---------------------------------------------------------------------------
# Adaptive Configuration
# ---------------------------------------------------------------------------
# The adaptive configuration system automatically selects the best
# hyperparameters based on measurable dataset properties (n_features,
# n_classes, n_samples). These rules were empirically derived from
# systematic experiments across 4 phases of optimization on UCI datasets:
#
# Phase 1: Polynomial expansion (3-5x improvement for low-dim data)
# Phase 2: Multi-view ensemble (2-3x improvement across all datasets)
# Phase 3: Target-aware projection, diverse strategies, augmentation
# Phase 4: Adaptive per-dataset configuration tuning
#
# The resulting adaptive function beats Random Forest on 3/4 UCI datasets
# (iris, breast_cancer, digits) and matches it on wine.
# ---------------------------------------------------------------------------

def adaptive_arrowflow_config(n_features: int, n_classes: int,
                              n_samples: int) -> ArrowFlowConfig:
    """Auto-select the best ArrowFlow configuration based on dataset properties.

    This function encodes all experimentally-validated rules from Phases 1-4
    of ArrowFlow optimization. The key insight is that different dataset
    characteristics require different encoding strategies:

    1. POLYNOMIAL EXPANSION degree:
       - n_features <= 10: pol_deg=3 — very low-dim data needs rich interaction
         terms to produce meaningful rankings. E.g., iris (4 features) gets
         35 terms with degree 3, giving enough diversity for argsort.
       - n_features <= 30: pol_deg=2 — moderate enrichment without explosion.
         E.g., wine (13 features) gets 105 terms with degree 2.
       - n_features > 30: pol_deg=1 (no expansion) — already high-dimensional,
         polynomial expansion would create an unmanageably large feature space.

    2. EMBEDDING DIMENSION:
       - n_features <= 15: embed_dim=16 — small permutations are sufficient
         and avoid the curse of dimensionality in permutation space.
       - n_classes >= 4 and n_features > 30: embed_dim=64 — multi-class
         high-dim data (e.g., digits) benefits from longer permutations
         that can encode more subtle ordinal relationships.
       - Otherwise: embed_dim=32 — good default for moderate datasets.

    3. PROJECTION STRATEGY:
       - n_classes >= 3: 'diverse' — alternate between target-aware (LDA),
         random, and calibrated projections across ensemble views. This
         maximizes ensemble diversity because each strategy captures
         systematically different aspects of the data structure. LDA adds
         supervised signal, random adds unsupervised diversity, calibrated
         equalizes dimension importance.
       - n_classes == 2: 'random' — binary LDA yields only 1 component,
         which is insufficient for meaningful diversity. Pure random
         projection is more effective.

    4. DATA AUGMENTATION:
       - Enabled when n_features <= 30 AND n_samples >= 150 — augmentation
         helps moderate-size low-dim datasets by creating training samples
         that are close in Spearman footrule distance to the originals.
       - Disabled for very small datasets (noise dominates) and high-dim
         data (already enough natural diversity from random projection).

    5. ENSEMBLE VIEWS:
       - n_samples < 2000: 7 views with majority voting — more views
         reduce variance and improve accuracy. 7 is the sweet spot balancing
         accuracy improvement vs. computational cost.
       - n_samples >= 2000: 1 view (no ensemble) — large datasets have
         enough data for a single network to learn well.

    Args:
        n_features: Number of input features.
        n_classes: Number of target classes.
        n_samples: Number of training samples.

    Returns:
        ArrowFlowConfig with all parameters set for the given dataset.
    """
    # --- 1. Polynomial expansion degree ---
    if n_features <= 10:
        pol_deg = 3
    elif n_features <= 30:
        pol_deg = 2
    else:
        pol_deg = 1

    # --- 2. Embedding dimension ---
    if n_features <= 15:
        embed_dim = 16
    elif n_classes >= 4 and n_features > 30:
        embed_dim = 64
    elif n_features <= 64:
        embed_dim = 32
    else:
        embed_dim = 64

    # --- 3. Architecture ---
    n_filters = 128
    n_iters = min(300, max(200, n_samples // 3))

    # --- 4. Projection strategy ---
    # Diverse projections (LDA + random + calibrated) improve ensemble diversity.
    # LDA needs >= 3 classes for >= 2 meaningful discriminant components.
    projection_strategy = 'diverse' if n_classes >= 3 else 'random'

    # --- 5. Augmentation ---
    # Light augmentation helps low-dim moderate-size datasets.
    use_augmentation = (n_features <= 30 and n_samples >= 150)

    # --- 6. Ensemble views ---
    use_ensemble = (n_samples < 2000 and n_classes <= 26)
    n_ensemble_views = 7 if use_ensemble else 1

    return ArrowFlowConfig(
        no_of_filters=[n_filters],
        layer_types=['sort', 'sort'],
        no_of_iters=n_iters,
        moe_no_of_networks=1,
        no_of_embedding_dim=embed_dim,
        poly_expansion=(pol_deg > 1),
        pol_deg=pol_deg,
        n_ensemble_views=n_ensemble_views,
        projection_strategy=projection_strategy,
        use_augmentation=use_augmentation,
        n_augmentations=1 if use_augmentation else 0,
        max_swaps=2,
        lda_ratio=0.3,
        adaptive_mode=True,
        verbose=1,
    )


# ---------------------------------------------------------------------------
# ArrowFlow training and evaluation
# ---------------------------------------------------------------------------

def run_arrowflow_single(data_train, data_test, adj_list_input, n_classes,
                         config: ArrowFlowConfig, seed=None):
    """Run a single ArrowFlow training + evaluation. Returns (val_error, test_error)."""
    if seed is not None:
        np.random.seed(seed)

    # Build a config-like object that SortFlowHybridNetwork expects
    sf_config = _build_sortnet_config(config, n_classes)

    file_name = 'benchmark_experiment'

    if config.moe_no_of_networks > 1:
        net = SortFlow_MoE('sf', adj_list_input, n_classes, file_name, sf_config)
        errors = net.train([data_train, data_test], sf_config)
    else:
        net = SortFlowHybridNetwork('sf', adj_list_input, n_classes, file_name, sf_config)
        errors = net.train([data_train, data_test], sf_config)

    # errors = [train_error, val_error, min_val_error, test_error]
    return errors[2], errors[3]


def _build_sortnet_config(config: ArrowFlowConfig, n_classes: int):
    """Build a config object compatible with SortFlowHybridNetwork from ArrowFlowConfig."""
    class _Config:
        pass

    c = _Config()
    # Copy all fields from ArrowFlowConfig
    for fld in config.__dataclass_fields__:
        setattr(c, fld, getattr(config, fld))

    # Append n_classes to filters (the network expects this)
    c.no_of_filters = list(config.no_of_filters) + [n_classes]
    c.data_initial = None
    c.frequency_dict = None

    # Determine pol_deg based on first layer type
    if config.layer_types[0] == 'sort':
        c.pol_deg = config.pol_deg if config.pol_deg > 0 else 3
    else:
        c.pol_deg = 0

    return c


# ---------------------------------------------------------------------------
# Multi-View Ensemble Training
# ---------------------------------------------------------------------------
# The multi-view ensemble is ArrowFlow's most impactful accuracy improvement.
# Instead of training a single network on one random projection, we train N
# independent networks, each on a different random projection of the input
# features, and combine their predictions via majority vote.
#
# WHY THIS WORKS:
# The primary accuracy bottleneck in ArrowFlow is encoding loss — argsort
# discards magnitude information and only preserves rank order. Different
# random projections capture different ordinal aspects of the data, so
# their errors are largely uncorrelated. Majority voting over N views
# exploits this diversity: if each view has error rate p < 0.5, the
# ensemble error rate decreases exponentially with N (by Condorcet's
# jury theorem).
#
# DIVERSE PROJECTION STRATEGIES:
# For maximum ensemble benefit, we use three different projection methods
# that produce systematically different rankings:
#   - target_aware: LDA components + random (captures class structure)
#   - random: Pure random projection (captures unsupervised geometry)
#   - calibrated: Random projection + post-calibration (equalizes dims)
#
# The strategies cycle as: target_aware, random, calibrated, target_aware, ...
# This ensures the ensemble has maximum diversity in how it encodes the data.
#
# Experimentally, 7 diverse views provide 2-3x error reduction over a
# single view, bringing ArrowFlow to RF-competitive accuracy on UCI datasets.
# ---------------------------------------------------------------------------


def _train_view_projection(view_args):
    """Worker: train a single view for projection-mode ensemble.

    Top-level function so it's picklable for ProcessPoolExecutor(spawn).
    Returns (view_index, predictions_array, view_error).
    """
    (v, view_seed, strategy, X_tr_poly, y_train, X_te_poly, y_test,
     embed_dim, lda_ratio, no_of_filters, layer_types, no_of_iters,
     n_classes, use_augmentation, n_augmentations, max_swaps, pol_deg,
     learning_rate, lr_tensor, last_layer_update) = view_args

    # Force CPU in forked worker — CUDA context from parent is invalid after fork.
    # Must disable CUDA before any torch operation (including Adam optimizer).
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch as _torch
    _torch.cuda.is_available = lambda: False
    import arrowflow.arrowflow as _af
    _af.device = 'cpu'

    np.random.seed(view_seed)

    # Encode data using the selected projection strategy
    if strategy == 'target_aware':
        perm_train, perm_test = DataGraph.target_aware_encode(
            X_tr_poly, y_train, X_te_poly,
            embed_dim=embed_dim, lda_ratio=lda_ratio, seed=view_seed
        )
    elif strategy == 'calibrated':
        perm_train, perm_test = DataGraph.calibrated_encode(
            X_tr_poly, y_train, X_te_poly,
            embed_dim=embed_dim, seed=view_seed, calibration='standardize'
        )
    else:  # 'random'
        rng = np.random.RandomState(view_seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_poly)
        X_te_s = scaler.transform(X_te_poly)
        W = rng.randn(X_tr_s.shape[1], embed_dim)
        perm_train = np.argsort(X_tr_s @ W, axis=1).astype(float)
        perm_test = np.argsort(X_te_s @ W, axis=1).astype(float)

    # Convert permutations to ArrowFlow data format
    adj_list_input = [str(i + 1) for i in range(embed_dim)]
    data_train = [[list(map(str, perm_train[i].astype(int) + 1)),
                    str(y_train[i]), 1]
                   for i in range(len(perm_train))]
    data_test_v = [[list(map(str, perm_test[i].astype(int) + 1)),
                     str(y_test[i]), 1]
                    for i in range(len(perm_test))]

    # Optional data augmentation
    if use_augmentation and n_augmentations > 0:
        data_train = DataGraph.augment_permutation_data(
            data_train, n_augmentations=n_augmentations,
            max_swaps=max_swaps, seed=view_seed
        )

    # Train a single ArrowFlow network for this view
    view_config = ArrowFlowConfig(
        no_of_filters=no_of_filters,
        layer_types=layer_types,
        no_of_iters=no_of_iters,
        moe_no_of_networks=1,
        no_of_embedding_dim=embed_dim,
        pol_deg=pol_deg,
        learning_rate=learning_rate,
        lr_tensor=lr_tensor,
        last_layer_update=last_layer_update,
        verbose=0,
    )
    sf_config = _build_sortnet_config(view_config, n_classes)
    net = SortFlowHybridNetwork(
        'sf_v' + str(v), adj_list_input, n_classes,
        'benchmark_ensemble', sf_config
    )
    errors = net.train([data_train, data_test_v], sf_config)

    net.graph = copy.deepcopy(net.optimal_model)
    _, preds = net.evaluate(data_test_v, 'supervised', 'classification')
    view_err = errors[3]
    return (v, np.array(preds).astype(int), view_err, strategy)


def _train_view_native(view_args):
    """Worker: train a single view for native/countvectorizer-mode ensemble.

    Top-level function so it's picklable for ProcessPoolExecutor(spawn).
    Returns (view_index, predictions_array, view_error).
    """
    (v, view_seed, data_train_base, data_test_base, adj_list_input,
     embed_dim, no_of_filters, layer_types, no_of_iters, n_classes,
     use_augmentation, n_augmentations, max_swaps,
     learning_rate, lr_tensor, last_layer_update) = view_args

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch as _torch
    _torch.cuda.is_available = lambda: False
    import arrowflow.arrowflow as _af
    _af.device = 'cpu'

    np.random.seed(view_seed)

    data_train = list(data_train_base)
    if use_augmentation and n_augmentations > 0:
        data_train = DataGraph.augment_permutation_data(
            data_train, n_augmentations=n_augmentations,
            max_swaps=max_swaps, seed=view_seed
        )

    view_config = ArrowFlowConfig(
        no_of_filters=no_of_filters,
        layer_types=layer_types,
        no_of_iters=no_of_iters,
        moe_no_of_networks=1,
        no_of_embedding_dim=embed_dim,
        learning_rate=learning_rate,
        lr_tensor=lr_tensor,
        last_layer_update=last_layer_update,
        verbose=0,
    )
    sf_config = _build_sortnet_config(view_config, n_classes)
    net = SortFlowHybridNetwork(
        'sf_v' + str(v), adj_list_input, n_classes,
        'benchmark_ensemble', sf_config
    )
    errors = net.train([data_train, list(data_test_base)], sf_config)

    net.graph = copy.deepcopy(net.optimal_model)
    _, preds = net.evaluate(list(data_test_base), 'supervised', 'classification')
    view_err = errors[3]
    return (v, np.array(preds).astype(int), view_err)


def _run_baseline_worker(args):
    """Worker: run a single baseline with GridSearchCV.

    Top-level function so it's picklable for ProcessPoolExecutor(spawn).
    Returns (baseline_name, result_dict_or_error_string).
    """
    bl_name, X_train, y_train, X_test, y_test, random_state = args
    try:
        result = run_baseline(bl_name, X_train, y_train, X_test, y_test,
                              random_state=random_state)
        return (bl_name, result)
    except Exception as e:
        return (bl_name, str(e))


def run_arrowflow_ensemble(X_train, y_train, X_test, y_test, n_classes,
                           config: ArrowFlowConfig, seed=42,
                           preencoded_data=None):
    """Train a multi-view ArrowFlow ensemble and return test error.

    Supports three encoding modes (via config.encoding_mode):

    'projection' (default):
        Each view encodes the raw features via a different random projection
        (or target-aware/calibrated projection), producing diverse permutations.
        Best for tabular data with continuous features.

    'native':
        Data is already in ArrowFlow's sorted-list format (e.g., artificial
        sequence data). All views share the same data but train with different
        random seeds, providing diversity through filter initialization.
        Pass pre-encoded data via the preencoded_data argument:
          preencoded_data = (data_train, data_test, adj_list_input)

    'countvectorizer':
        Data is pre-encoded via CountVectorizer → word-index permutations.
        Same as 'native' — pass pre-encoded data via preencoded_data.
        Views provide diversity through different filter initializations
        and optional augmentation.

    For 'projection' mode, the projection_strategy config field controls
    how views are encoded ('random', 'target_aware', 'calibrated', 'diverse').

    Final prediction is majority vote across all views.

    Args:
        X_train, y_train: Training data (raw features for 'projection' mode,
            ignored for 'native'/'countvectorizer' if preencoded_data is set).
        X_test, y_test: Test data.
        n_classes: Number of target classes.
        config: ArrowFlowConfig with ensemble parameters.
        seed: Base random seed.
        preencoded_data: Tuple of (data_train, data_test, adj_list_input) for
            'native' or 'countvectorizer' encoding modes.

    Returns:
        test_error: Error rate on test set (float in [0, 1]).
    """
    n_views = config.n_ensemble_views
    encoding_mode = config.encoding_mode

    # --- Pre-encoded data path (native / countvectorizer) ---
    if encoding_mode in ('native', 'countvectorizer'):
        if preencoded_data is None:
            raise ValueError(f"encoding_mode='{encoding_mode}' requires preencoded_data")
        data_train_base, data_test_base, adj_list_input = preencoded_data
        embed_dim = len(data_train_base[0][0])

        # Build args for each view
        view_args_list = []
        for v in range(n_views):
            view_seed = seed + v * 9999
            view_args_list.append((
                v, view_seed, data_train_base, data_test_base, adj_list_input,
                embed_dim, list(config.no_of_filters), list(config.layer_types),
                config.no_of_iters, n_classes,
                config.use_augmentation, config.n_augmentations, config.max_swaps,
                config.learning_rate, config.lr_tensor, config.last_layer_update,
            ))

        # Train all views in parallel using multiprocessing.Pool
        n_workers = min(n_views, MAX_WORKERS)
        with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
            results_list = pool.map(_train_view_native, view_args_list)
        all_predictions = [None] * n_views
        for v_idx, preds, view_err in results_list:
            all_predictions[v_idx] = preds
            if config.verbose > 0:
                print(f"    View {v_idx+1}/{n_views}: test_err={100*view_err:.1f}%")

        gt = np.array([int(dp[1]) for dp in data_test_base])
        predictions_array = np.array(all_predictions)
        final_preds = np.array([
            np.bincount(predictions_array[:, i], minlength=n_classes).argmax()
            for i in range(len(gt))
        ])
        return float(np.sum(final_preds != gt) / len(gt))

    # --- Projection encoding path (tabular data) ---

    # Polynomial expansion (shared across all views, done once in main process)
    if config.poly_expansion and config.pol_deg > 1:
        poly = PolynomialFeatures(degree=config.pol_deg)
        X_tr_poly = poly.fit_transform(X_train)
        X_te_poly = poly.transform(X_test)
    else:
        X_tr_poly = X_train
        X_te_poly = X_test

    # Determine projection strategy per view
    if config.projection_strategy == 'diverse':
        strategies = []
        for v in range(n_views):
            if v % 3 == 0:
                strategies.append('target_aware')
            elif v % 3 == 1:
                strategies.append('random')
            else:
                strategies.append('calibrated')
    else:
        strategies = [config.projection_strategy] * n_views

    # Build args for each view
    view_args_list = []
    for v in range(n_views):
        view_seed = seed + v * 9999
        view_args_list.append((
            v, view_seed, strategies[v],
            X_tr_poly, y_train, X_te_poly, y_test,
            config.no_of_embedding_dim, config.lda_ratio,
            list(config.no_of_filters), list(config.layer_types),
            config.no_of_iters, n_classes,
            config.use_augmentation, config.n_augmentations, config.max_swaps,
            config.pol_deg, config.learning_rate, config.lr_tensor,
            config.last_layer_update,
        ))

    # Train all views in parallel using multiprocessing.Pool
    n_workers = min(n_views, MAX_WORKERS)
    with mp.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        results_list = pool.map(_train_view_projection, view_args_list)
    all_predictions = [None] * n_views
    for v_idx, preds, view_err, strategy in results_list:
        all_predictions[v_idx] = preds
        if config.verbose > 0:
            print(f"    View {v_idx+1}/{n_views} ({strategy}): "
                  f"test_err={100*view_err:.1f}%")

    # Majority vote across all views
    gt = y_test.astype(int)
    predictions_array = np.array(all_predictions)
    final_preds = np.array([
        np.bincount(predictions_array[:, i], minlength=n_classes).argmax()
        for i in range(len(gt))
    ])
    test_error = float(np.sum(final_preds != gt) / len(gt))

    return test_error


def run_arrowflow_experiment(dataset_name, config: ArrowFlowConfig, n_sims=5,
                             subsample=None, random_state=42):
    """
    Run multiple ArrowFlow simulations on a dataset.

    If config.adaptive_mode is True, automatically selects the best
    configuration based on dataset properties.

    If config.n_ensemble_views > 1, uses multi-view ensemble training
    with diverse projections and majority voting.

    Returns dict with statistics.
    """
    X_train, y_train, X_test, y_test, n_classes = load_dataset(
        dataset_name, subsample=subsample, random_state=random_state
    )

    # --- Adaptive mode: auto-select configuration ---
    if config.adaptive_mode:
        config = adaptive_arrowflow_config(
            n_features=X_train.shape[1],
            n_classes=n_classes,
            n_samples=len(X_train)
        )
        if config.verbose > 0:
            print(f"  Adaptive config: pol_deg={config.pol_deg}, "
                  f"embed_dim={config.no_of_embedding_dim}, "
                  f"filters={config.no_of_filters}, iters={config.no_of_iters}, "
                  f"strategy={config.projection_strategy}, "
                  f"aug={config.use_augmentation}, views={config.n_ensemble_views}")

    # --- Choose between ensemble and single-network training ---
    use_ensemble = config.n_ensemble_views > 1

    if use_ensemble:
        # Multi-view ensemble: train N views, majority vote
        test_errors = []
        times = []

        for sim in range(n_sims):
            seed = (sim + 1) * 12345 + random_state
            t0 = time.time()
            test_err = run_arrowflow_ensemble(
                X_train, y_train, X_test, y_test, n_classes,
                config, seed=seed
            )
            elapsed = time.time() - t0
            test_errors.append(test_err)
            times.append(elapsed)

            if config.verbose > 0:
                print(f"  Sim {sim+1}/{n_sims}: ensemble_test={100*test_err:.1f}% ({elapsed:.1f}s)")

        test_errors = np.array(test_errors)
        return {
            'method': f'ArrowFlow Ensemble (v={config.n_ensemble_views})',
            'dataset': dataset_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_classes': n_classes,
            'n_sims': n_sims,
            'val_error_mean': 0.0,
            'val_error_std': 0.0,
            'val_error_min': 0.0,
            'test_error_mean': float(np.mean(test_errors)),
            'test_error_std': float(np.std(test_errors)),
            'test_error_min': float(np.min(test_errors)),
            'test_error_max': float(np.max(test_errors)),
            'test_error_stderr': float(np.std(test_errors) / np.sqrt(n_sims)),
            'time_mean': float(np.mean(times)),
            'config': config.to_dict(),
        }

    else:
        # Standard single-network training (original path)
        if dataset_name == 'artificial':
            data_graph = DataGraph('artificial')
            data_train, data_test, adj_list_input, n_classes, _ = \
                data_graph.generate_sequence_data(seed=random_state)
        else:
            data_train, adj_list_input, W_random = convert_to_arrowflow_data(
                X_train, y_train, config
            )
            data_test, _, _ = convert_to_arrowflow_data(
                X_test, y_test, config, W_random=W_random
            )

        val_errors = []
        test_errors = []
        times = []

        for sim in range(n_sims):
            seed = (sim + 1) * 12345 + random_state
            t0 = time.time()
            val_err, test_err = run_arrowflow_single(
                data_train, data_test, adj_list_input, n_classes, config, seed=seed
            )
            elapsed = time.time() - t0
            val_errors.append(val_err)
            test_errors.append(test_err)
            times.append(elapsed)

            if config.verbose > 0:
                print(f"  Sim {sim+1}/{n_sims}: val={100*val_err:.1f}% test={100*test_err:.1f}% ({elapsed:.1f}s)")

        val_errors = np.array(val_errors)
        test_errors = np.array(test_errors)

        return {
            'method': 'ArrowFlow',
            'dataset': dataset_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_classes': n_classes,
            'n_sims': n_sims,
            'val_error_mean': float(np.mean(val_errors)),
            'val_error_std': float(np.std(val_errors)),
            'val_error_min': float(np.min(val_errors)),
            'test_error_mean': float(np.mean(test_errors)),
            'test_error_std': float(np.std(test_errors)),
            'test_error_min': float(np.min(test_errors)),
            'test_error_max': float(np.max(test_errors)),
            'test_error_stderr': float(np.std(test_errors) / np.sqrt(n_sims)),
            'time_mean': float(np.mean(times)),
            'config': config.to_dict(),
        }


# ---------------------------------------------------------------------------
# Baseline classifiers
# ---------------------------------------------------------------------------

BASELINE_CONFIGS = {
    'rf': ('Random Forest', RandomForestClassifier, {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None],
    }),
    'svm': ('SVM-RBF', SVC, {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    }),
    'mlp': ('MLP', MLPClassifier, {
        'hidden_layer_sizes': [(64,), (128,), (256,), (64, 64), (128, 64), (256, 128)],
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001],
        'max_iter': [1000],
    }),
    'knn': ('KNN', KNeighborsClassifier, {
        'n_neighbors': [1, 3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }),
    'xgb': ('XGBoost', GradientBoostingClassifier, {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
    }),
    'lr': ('Logistic Regression', LogisticRegression, {
        'C': [0.01, 0.1, 1, 10, 100],
        'max_iter': [1000],
    }),
    'nb': ('Naive Bayes', GaussianNB, {}),
    'mlp_ensemble': ('MLP Ensemble', None, {}),  # Special: ensemble of MLPs
}


def run_baseline(name, X_train, y_train, X_test, y_test, n_sims=1, random_state=42):
    """Run a baseline classifier with optional grid search. Returns result dict."""
    if name not in BASELINE_CONFIGS:
        raise ValueError(f"Unknown baseline: {name}")

    label, cls, param_grid = BASELINE_CONFIGS[name]

    if name == 'mlp_ensemble':
        return _run_mlp_ensemble(X_train, y_train, X_test, y_test,
                                 n_networks=7, n_sims=n_sims, random_state=random_state)

    t0 = time.time()
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if param_grid:
        grid = GridSearchCV(cls(random_state=random_state) if 'random_state' in cls().get_params() else cls(),
                            param_grid, cv=3, scoring='accuracy', n_jobs=8)
        grid.fit(X_train_s, y_train)
        best_model = grid.best_estimator_
    else:
        best_model = cls()
        best_model.fit(X_train_s, y_train)

    y_pred = best_model.predict(X_test_s)
    test_error = 1.0 - accuracy_score(y_test, y_pred)
    elapsed = time.time() - t0

    return {
        'method': label,
        'dataset': '',  # filled by caller
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_error_mean': float(test_error),
        'test_error_std': 0.0,
        'test_error_min': float(test_error),
        'test_error_max': float(test_error),
        'test_error_stderr': 0.0,
        'time_mean': elapsed,
        'n_sims': 1,
        'config': {'best_params': str(getattr(best_model, 'get_params', lambda: {})())[:200]},
    }


def _run_mlp_ensemble(X_train, y_train, X_test, y_test, n_networks=7,
                      n_sims=1, random_state=42):
    """Run an ensemble of MLPs (for fair MoE comparison)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    predictions = []
    for i in range(n_networks):
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                            random_state=random_state + i, learning_rate_init=0.01)
        mlp.fit(X_train_s, y_train)
        predictions.append(mlp.predict(X_test_s))

    # Majority vote
    predictions = np.array(predictions)
    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int), minlength=len(np.unique(y_test))).argmax(),
        axis=0, arr=predictions
    )
    test_error = 1.0 - accuracy_score(y_test, ensemble_pred)
    elapsed = time.time() - t0

    return {
        'method': f'MLP Ensemble (n={n_networks})',
        'dataset': '',
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_error_mean': float(test_error),
        'test_error_std': 0.0,
        'test_error_min': float(test_error),
        'test_error_max': float(test_error),
        'test_error_stderr': 0.0,
        'time_mean': elapsed,
        'n_sims': 1,
        'config': {'n_networks': n_networks},
    }


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(dataset_name: str, config: ArrowFlowConfig,
                  n_sims: int = 5, baselines: List[str] = None,
                  subsample: Optional[float] = None,
                  random_state: int = 42) -> List[Dict]:
    """
    Run ArrowFlow + baselines on a dataset. Returns list of result dicts.
    """
    if baselines is None:
        baselines = ['rf', 'svm', 'mlp', 'knn']

    print(f"\n{'='*60}")
    print(f"Benchmark: {dataset_name} (subsample={subsample})")
    print(f"ArrowFlow: filters={config.no_of_filters} layers={config.layer_types} "
          f"moe={config.moe_no_of_networks} iters={config.no_of_iters}")
    print(f"{'='*60}")

    results = []

    # Load data once for baselines
    X_train, y_train, X_test, y_test, n_classes = load_dataset(
        dataset_name, subsample=subsample, random_state=random_state
    )
    print(f"Data: {len(X_train)} train, {len(X_test)} test, {n_classes} classes, {X_train.shape[1]} features")

    # Run baselines sequentially (each uses GridSearchCV n_jobs internally)
    for bl_name in baselines:
        label = BASELINE_CONFIGS[bl_name][0]
        print(f"\n--- Baseline: {label} ---")
        try:
            bl_result = run_baseline(bl_name, X_train, y_train, X_test, y_test,
                                     random_state=random_state)
            bl_result['dataset'] = dataset_name
            bl_result['subsample'] = subsample
            bl_result['n_classes'] = n_classes
            print(f"  Test Error: {100*bl_result['test_error_mean']:.1f}% ({bl_result['time_mean']:.1f}s)")
            results.append(bl_result)
        except Exception as e:
            print(f"  FAILED: {e}")

    # Run ArrowFlow
    print(f"\n--- ArrowFlow ({n_sims} sims) ---")
    try:
        af_result = run_arrowflow_experiment(
            dataset_name, config, n_sims=n_sims,
            subsample=subsample, random_state=random_state
        )
        af_result['subsample'] = subsample
        print(f"  Test Error: {100*af_result['test_error_mean']:.1f}% "
              f"+/- {100*af_result['test_error_stderr']:.1f}% "
              f"(min={100*af_result['test_error_min']:.1f}%, {af_result['time_mean']:.1f}s)")
        results.append(af_result)
    except Exception as e:
        print(f"  ArrowFlow FAILED: {e}")
        import traceback
        traceback.print_exc()

    return results


def run_learning_curve(dataset_name: str, config: ArrowFlowConfig,
                       subsample_fractions: List[float] = None,
                       n_sims: int = 5,
                       baselines: List[str] = None,
                       random_state: int = 42) -> List[Dict]:
    """
    Run ArrowFlow + baselines at multiple training set sizes.
    Returns list of all result dicts.
    """
    if subsample_fractions is None:
        subsample_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    if baselines is None:
        baselines = ['rf', 'svm', 'mlp']

    all_results = []
    for frac in subsample_fractions:
        results = run_benchmark(dataset_name, config, n_sims=n_sims,
                                baselines=baselines, subsample=frac,
                                random_state=random_state)
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def log_results(results: List[Dict], filename: str = None):
    """Append results to a CSV file."""
    if filename is None:
        filename = os.path.join(RESULTS_DIR,
                                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    fieldnames = [
        'timestamp', 'dataset', 'method', 'subsample', 'n_train', 'n_test', 'n_classes',
        'n_sims', 'test_error_mean', 'test_error_std', 'test_error_min', 'test_error_max',
        'test_error_stderr', 'val_error_mean', 'val_error_std', 'time_mean', 'config'
    ]

    file_exists = os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            row['timestamp'] = datetime.now().isoformat()
            row['config'] = str(r.get('config', ''))
            writer.writerow(row)

    print(f"\nResults logged to {filename}")
    return filename


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='ArrowFlow Benchmark')
    parser.add_argument('--dataset', type=str, default='iris',
                        help='Dataset name: iris, wine, wine_quality, digits, letter, vehicle, segment, moons, circles, breast_cancer, mnist_pca, imdb')
    parser.add_argument('--filters', type=int, nargs='+', default=[128],
                        help='Filter counts per layer')
    parser.add_argument('--layers', type=str, nargs='+', default=['sort', 'sort'],
                        help='Layer types')
    parser.add_argument('--iters', type=int, default=200, help='Training iterations')
    parser.add_argument('--moe', type=int, default=1, help='Number of MoE networks')
    parser.add_argument('--sims', type=int, default=5, help='Number of simulations')
    parser.add_argument('--baselines', type=str, nargs='+', default=['rf', 'svm', 'mlp', 'knn'],
                        help='Baseline methods')
    parser.add_argument('--subsample', type=float, default=None, help='Training subsample fraction')
    parser.add_argument('--learning-curve', action='store_true', help='Run learning curve experiment')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--embed-dim', type=int, default=16, help='Embedding dimensions')
    parser.add_argument('--pol-deg', type=int, default=3, help='Polynomial degree')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive mode: auto-select all hyperparameters based on dataset properties')
    parser.add_argument('--ensemble-views', type=int, default=1,
                        help='Number of ensemble views (1 = no ensemble, 7 = recommended)')
    parser.add_argument('--projection-strategy', type=str, default='random',
                        choices=['random', 'target_aware', 'calibrated', 'diverse'],
                        help='Projection strategy for encoding')
    parser.add_argument('--augment', action='store_true',
                        help='Enable permutation data augmentation')

    args = parser.parse_args()

    config = ArrowFlowConfig(
        no_of_filters=args.filters,
        layer_types=args.layers,
        no_of_iters=args.iters,
        moe_no_of_networks=args.moe,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        no_of_embedding_dim=args.embed_dim,
        pol_deg=args.pol_deg,
        poly_expansion=(args.pol_deg > 1),
        verbose=args.verbose,
        adaptive_mode=args.adaptive,
        n_ensemble_views=args.ensemble_views,
        projection_strategy=args.projection_strategy,
        use_augmentation=args.augment,
        n_augmentations=1 if args.augment else 0,
    )

    if args.learning_curve:
        results = run_learning_curve(args.dataset, config, n_sims=args.sims,
                                     baselines=args.baselines)
    else:
        results = run_benchmark(args.dataset, config, n_sims=args.sims,
                                baselines=args.baselines, subsample=args.subsample)

    log_results(results)


if __name__ == '__main__':
    main()
