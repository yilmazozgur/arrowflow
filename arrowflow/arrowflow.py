import os
import time
import numpy as np
import copy
import random
from datetime import datetime
import multiprocessing
from time import sleep
import pandas as pd
import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
from scipy.spatial import distance_matrix, distance
from collections import defaultdict
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from arrowflow.config import sortnet_config

device = 'cuda' if torch.cuda.is_available() else sortnet_config.device

""" 
    Written by Ozgur Yilmaz (yilmazozgur.kaan@gmail.com) 2019-2026.
    ArrowFlow machine learning architecture that is based on edit distance of two lists as filtering. 
"""

class DataGraph:
    # Methods for data generation/transformation related with SortFlow algorithm
    def __init__(self, id_key):
        self.id = id_key

    def swapPositions(self, list, pos1, pos2):
        list[pos1], list[pos2] = list[pos2], list[pos1]
        return list

    def convert_tensor_to_sortnet_data(self, tensor, gt_output):

        data_sort_net = []
        gt_count = 0
        for tensor_iter in tensor:
            sorted, sort_indices = torch.sort(tensor_iter, descending=True)
            tensor_iter_ = sort_indices.to(torch.float32).cpu().numpy() + 1
            tensor_data = tensor_iter_.astype(int).tolist()
            data_sort_net.append([list(map(str, tensor_data)), str(gt_output[gt_count]), 1])
            gt_count += 1

        return data_sort_net

    def generate_sequence_data(self, distort_prob=0, dim_expand_ratio=1,
                               vocab_size=16, n_train_per_class=30,
                               n_test_per_class=15, seed=42):
        """Generate artificial permutation classification data.

        Each class is defined by a prototype ordering pattern over a vocabulary
        of integers [1..vocab_size]. Training and test samples are noisy
        variants of the prototype (adjacent swaps + random deletions).

        The patterns test increasingly subtle ordinal relationships:
          0  Ascending                [1,2,3,...,N]
          1  Descending               [N,N-1,...,1]
          2  Odds-first               [1,3,5,...,2,4,6,...]
          3  Evens-first              [2,4,6,...,1,3,5,...]
          4  V-shape (asc then desc)  [1,3,5,...,N,...,4,2]
          5  Inverted-V               [N,...,2,1,...,N-1]
          6  Zigzag low-start         [1,N,2,N-1,3,N-2,...]
          7  Zigzag high-start        [N,1,N-1,2,N-2,3,...]
          8  Block-reversed (k=4)     [4,3,2,1,8,7,6,5,...]
          9  Interleaved halves       [1,h+1,2,h+2,3,h+3,...]
         10  Reversed pairs           [2,1,4,3,6,5,...]
         11  Spiral inward            [1,N,2,N-1,3,N-2,...] from both ends
        """
        rng = np.random.RandomState(seed)
        N = vocab_size
        items = list(range(1, N + 1))
        half = N // 2
        odds = [x for x in items if x % 2 == 1]
        evens = [x for x in items if x % 2 == 0]

        # --- Define prototype orderings ---
        prototypes = {}

        # 0: Ascending
        prototypes[0] = list(items)

        # 1: Descending
        prototypes[1] = list(reversed(items))

        # 2: Odds first (ascending), then evens (ascending)
        prototypes[2] = odds + evens

        # 3: Evens first (ascending), then odds (ascending)
        prototypes[3] = evens + odds

        # 4: V-shape — odds ascending then evens descending
        prototypes[4] = odds + list(reversed(evens))

        # 5: Inverted-V — evens descending then odds ascending
        prototypes[5] = list(reversed(evens)) + odds

        # 6: Zigzag low-start [1,N,2,N-1,3,N-2,...]
        zigzag_low = []
        for i in range(half):
            zigzag_low.append(items[i])
            zigzag_low.append(items[N - 1 - i])
        prototypes[6] = zigzag_low

        # 7: Zigzag high-start [N,1,N-1,2,N-2,3,...]
        zigzag_high = []
        for i in range(half):
            zigzag_high.append(items[N - 1 - i])
            zigzag_high.append(items[i])
        prototypes[7] = zigzag_high

        # 8: Block-reversed (blocks of 4, reversed within each block)
        block_size = min(4, N)
        block_rev = []
        for start in range(0, N, block_size):
            block = items[start:start + block_size]
            block_rev.extend(reversed(block))
        prototypes[8] = block_rev

        # 9: Interleaved halves [1,h+1,2,h+2,3,h+3,...]
        first_half = items[:half]
        second_half = items[half:]
        interleaved = []
        for a, b in zip(first_half, second_half):
            interleaved.extend([a, b])
        prototypes[9] = interleaved

        # 10: Reversed pairs [2,1,4,3,6,5,...]
        rev_pairs = []
        for i in range(0, N - 1, 2):
            rev_pairs.extend([items[i + 1], items[i]])
        if N % 2 == 1:
            rev_pairs.append(items[-1])
        prototypes[10] = rev_pairs

        # 11: Spiral inward [1,N,2,N-1,...] collapsing to center
        spiral = []
        lo, hi = 0, N - 1
        while lo <= hi:
            spiral.append(items[lo])
            if lo != hi:
                spiral.append(items[hi])
            lo += 1
            hi -= 1
        prototypes[11] = spiral

        number_of_classes = len(prototypes)
        adj_list_input = list(map(str, items))

        def _add_noise(seq, rng, n_swaps, n_deletions):
            """Apply random adjacent swaps and deletions to a sequence."""
            s = list(seq)
            # Adjacent swaps
            for _ in range(n_swaps):
                if len(s) < 2:
                    break
                i = rng.randint(0, len(s) - 1)
                j = min(i + 1, len(s) - 1)
                s[i], s[j] = s[j], s[i]
            # Random deletions
            for _ in range(n_deletions):
                if len(s) <= N // 2:  # keep at least half the items
                    break
                idx = rng.randint(0, len(s))
                del s[idx]
            return s

        def _generate_samples(n_samples, rng):
            """Generate n_samples noisy variants per class."""
            data = []
            for cls_id, proto in prototypes.items():
                for _ in range(n_samples):
                    n_swaps = rng.randint(0, max(1, N // 3))
                    n_del = rng.randint(0, max(1, N // 3))
                    noisy = _add_noise(proto, rng, n_swaps, n_del)
                    data.append([list(map(str, noisy)), str(cls_id), 1])
            return data

        data_train_net = _generate_samples(n_train_per_class, rng)
        data_test_net = _generate_samples(n_test_per_class, rng)

        # Shuffle within splits
        train_order = rng.permutation(len(data_train_net))
        data_train_net = [data_train_net[i] for i in train_order]
        test_order = rng.permutation(len(data_test_net))
        data_test_net = [data_test_net[i] for i in test_order]

        if dim_expand_ratio > 1:
            dict_mapping = {}
            adj_list_input_expanded = []
            for inp_data in adj_list_input:
                dict_mapping[inp_data] = []
                for iter_exp in range(dim_expand_ratio):
                    dict_mapping[inp_data].append(inp_data + '_' + str(iter_exp+1))
                    adj_list_input_expanded.append(inp_data + '_' + str(iter_exp+1))

            data_train_net_expanded = []
            for data_iter in data_train_net:
                data_iter_raw = data_iter[0]
                data_iter_raw_expanded = []
                for data_pt in data_iter_raw:
                    data_iter_raw_expanded.extend(dict_mapping[data_pt])
                data_pt_insert = [data_iter_raw_expanded, data_iter[1] , data_iter[2]]
                data_train_net_expanded.append(data_pt_insert)

            data_test_net_expanded = []
            for data_iter in data_test_net:
                data_iter_raw = data_iter[0]
                data_iter_raw_expanded = []
                for data_pt in data_iter_raw:
                    data_iter_raw_expanded.extend(dict_mapping[data_pt])
                data_pt_insert = [data_iter_raw_expanded, data_iter[1], data_iter[2]]
                data_test_net_expanded.append(data_pt_insert)

            data_train_net = data_train_net_expanded
            data_test_net = data_test_net_expanded
            adj_list_input = adj_list_input_expanded

        max_input_size = len(adj_list_input)

        if distort_prob > 0:
            for idx, data_pt in enumerate(data_train_net):
                data_pt_ = copy.deepcopy(data_pt[0])
                for id_iter, pt in enumerate(data_pt[0]):
                    if np.random.rand() > 1 - distort_prob:
                        if id_iter < len(data_pt_) - 1:
                            swap_index = id_iter + 1
                        else:
                            swap_index = id_iter - 1
                        data_pt_ = self.swapPositions(data_pt_, id_iter, swap_index)
                for id_iter, pt in enumerate(data_pt_):
                    if np.random.rand() > 1 - distort_prob:
                        rnd_index = np.random.choice(len(data_pt_), 1, replace=False)[0]
                        del data_pt_[rnd_index]
                if len(data_pt_) > max_input_size:
                    max_input_size = len(data_pt_)
                data_train_net[idx][0] = data_pt_

            for idx, data_pt in enumerate(data_test_net):
                data_pt_ = copy.deepcopy(data_pt[0])
                for id_iter, pt in enumerate(data_pt[0]):
                    if np.random.rand() > 1 - distort_prob:
                        if id_iter < len(data_pt_) - 1:
                            swap_index = id_iter + 1
                        else:
                            swap_index = id_iter - 1
                        data_pt_ = self.swapPositions(data_pt_, id_iter, swap_index)
                for id_iter, pt in enumerate(data_pt_):
                    if np.random.rand() > 1 - distort_prob:
                        rnd_index = np.random.choice(len(data_pt_), 1, replace=False)[0]
                        del data_pt_[rnd_index]
                if len(data_pt_) > max_input_size:
                    max_input_size = len(data_pt_)
                data_test_net[idx][0] = data_pt_

        return data_train_net, data_test_net, adj_list_input, number_of_classes, max_input_size

    def diff_lists(self, li1, li2):
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
        return li_dif

    def sigmoid_func(self, x):
        z = 1 / (1 + np.exp(-x))
        return z

    def relu(self, x):
        # return (np.maximum(0, x))
        return (np.maximum(0.1*x, x))

    def transform_2_numpy_dataset(self, data, size_vector):
        data_vec = []
        label_vec = []
        for dt_pt in data:
            vec = [float(i)-1 for i in dt_pt[0]]
            for rpt in range(size_vector - len(vec)):
                vec.append(float(0))
            vec = np.asarray(vec)
            label = float(dt_pt[1])
            data_vec.append(vec)
            label_vec.append(label)
        data_vec = np.asarray(data_vec)
        label_vec = np.asarray(label_vec)

        return data_vec, label_vec

    def transform_2_numpy_sorting_dataset(self, data, size_vector):
        data_vec = []
        label_vec = []
        for dt_pt in data:
            vec = [float(i) for i in dt_pt[0]]
            vec = np.asarray(vec)
            label = [float(i) for i in dt_pt[1]]
            label = np.asarray(label)
            data_vec.append(vec)
            label_vec.append(label)
        data_vec = np.asarray(data_vec)
        label_vec = np.asarray(label_vec)

        return data_vec, label_vec

    def project_random_space(self, X, y, W_random=None, poly_expansion=True, expand_diff_features=False,
                             pol_deg=5, no_dimensions=32, n_views=1):
        """Encode real-valued feature vectors as sorted lists (permutations).

        Pipeline: X → (optional poly expansion) → StandardScaler → random projection → argsort

        This is the core encoding that maps continuous feature vectors into the
        permutation space where ArrowFlow operates. The key insight is that argsort
        preserves relative ordering information, which is exactly what the edit-distance
        based sort layers exploit. By the Johnson-Lindenstrauss lemma, the random
        projection approximately preserves pairwise distances in the projected space,
        and argsort then captures the ordinal structure of those distances.

        Polynomial expansion (when enabled) creates interaction and higher-order terms
        before projection, which is critical for low-dimensional data (n_features <= 30)
        where the raw features don't provide enough diversity for meaningful rankings.
        Experimentally, polynomial expansion provides 3-5x error reduction on datasets
        like iris (4 features) and wine (13 features).

        Multi-view encoding (n_views > 1) splits the output into n_views independent
        blocks, each of size dim_per_view = no_dimensions // n_views. Each block is an
        independent argsort of a different random projection of the input features.
        This dramatically improves encoding quality for high-dimensional data by avoiding
        the curse of dimensionality in permutation space.

        Args:
            X: Feature matrix (N x D).
            y: Label vector (N,).
            W_random: Cached projection parameters from training (used at test time).
            poly_expansion: Whether to apply polynomial feature expansion.
            expand_diff_features: Whether to add difference features.
            pol_deg: Degree of polynomial expansion.
            no_dimensions: Target permutation length (embedding dimension).
            n_views: Number of independent projection views within a single permutation.
        """
        data_net = []
        input_vertices = []

        # Return original data if poly degree is zero (data is already in list format)
        if pol_deg == 0:
            for inp_iter in range(X[0].shape[0]):
                input_vertices.append('input_' + str(inp_iter))
            for idx, data_row in enumerate(X):
                label = y[idx]
                data_point = list(map(str, data_row))
                data_net.append([data_point, str(label), 1])

            return data_net, input_vertices, W_random

        # --- Polynomial expansion ---
        # For low-dimensional data, polynomial features create interaction terms
        # and nonlinear combinations that enrich the feature space before projection.
        # Without this, datasets like iris (4 features) produce permutations with
        # very limited diversity, leading to high classification error.
        if poly_expansion:
            trans_pol = PolynomialFeatures(degree=pol_deg)
            X = trans_pol.fit_transform(X)

        if expand_diff_features:
            X_concat = []
            for data_iter in X:
                data_iter_ = copy.deepcopy(data_iter)
                for exp_iter in range(int(X.shape[1]/2)):
                    data_diff = data_iter - np.roll(data_iter, -(exp_iter+1))
                    data_iter_ = np.concatenate((data_iter_, data_diff))
                X_concat.append(data_iter_)
            X = copy.deepcopy(np.asarray(X_concat))

        dim_per_view = no_dimensions // n_views
        total_length = dim_per_view * n_views

        if W_random is None or len(W_random) == 0:
            # Fit StandardScaler on training data to normalize features before
            # projection, ensuring all dimensions contribute equally to the ranking.
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            # One random projection matrix per view — each view captures different
            # ordinal relationships in the data, improving robustness.
            W_views = [np.random.randn(X.shape[1], dim_per_view) for _ in range(n_views)]
            W_random = [W_views, scaler, n_views, dim_per_view]
        else:
            # Apply fitted scaler from training to ensure consistent encoding
            W_views = W_random[0]
            if len(W_random) > 1 and W_random[1] is not None:
                X = W_random[1].transform(X)

        # Multi-view argsort: each view is an independent projection + argsort.
        # Different random projections capture different ordinal aspects of the data,
        # which increases the information content of the resulting permutation.
        X_index = np.empty((X.shape[0], total_length), dtype=int)
        for v in range(n_views):
            proj = X @ W_views[v]
            view_perm = np.argsort(proj, axis=1)
            X_index[:, v * dim_per_view:(v + 1) * dim_per_view] = view_perm + v * dim_per_view + 1

        for idx, data_row in enumerate(X_index):
            label = y[idx]
            data_point = list(map(str, data_row))
            data_net.append([data_point, str(label), 1])
            input_vertices += data_point
            input_vertices = np.unique(np.asarray(input_vertices)).tolist()

        return data_net, input_vertices, W_random

    # ------------------------------------------------------------------
    # Target-Aware Encoding (LDA + Random Projection)
    # ------------------------------------------------------------------
    # Standard random projection is unsupervised — it preserves pairwise
    # distances (Johnson-Lindenstrauss) but ignores class structure. For
    # multi-class problems (n_classes >= 3), injecting supervised signal
    # via Linear Discriminant Analysis (LDA) dramatically improves the
    # quality of the resulting permutations.
    #
    # LDA finds the linear subspace that maximizes between-class variance
    # relative to within-class variance. By dedicating a fraction of the
    # embedding dimensions to LDA components and the rest to random
    # projection, we get the best of both worlds:
    #   - LDA components capture the most discriminative directions
    #   - Random components capture complementary ordinal structure
    #
    # The two parts are scale-normalized before concatenation so that
    # argsort treats them equally. This hybrid approach reduces error by
    # ~30% on multi-class datasets like digits (10 classes).
    #
    # For binary classification, LDA yields only 1 component, which is
    # insufficient for meaningful diversity — use standard random projection
    # instead.
    # ------------------------------------------------------------------

    @staticmethod
    def target_aware_encode(X_train, y_train, X_test, embed_dim=32,
                            lda_ratio=0.3, seed=42):
        """Encode using a mix of LDA discriminant components and random projection.

        Args:
            X_train: Training features (N_train x D), already polynomial-expanded.
            y_train: Training labels (N_train,).
            X_test: Test features (N_test x D), already polynomial-expanded.
            embed_dim: Target permutation length.
            lda_ratio: Fraction of embed_dim dimensions allocated to LDA components.
                Higher values inject more supervised signal but reduce diversity.
                0.3 is a good default — enough for class separation without
                over-relying on the linear discriminant assumption.
            seed: Random seed for reproducibility.

        Returns:
            perm_train: Permutation matrix for training data (N_train x embed_dim).
            perm_test: Permutation matrix for test data (N_test x embed_dim).
        """
        rng = np.random.RandomState(seed)

        # Standardize features before both LDA and random projection
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        # Determine how many LDA components we can use.
        # LDA can produce at most (n_classes - 1) components.
        n_classes = len(np.unique(y_train))
        max_lda_components = min(n_classes - 1, X_tr.shape[1], embed_dim)
        n_lda = max(1, min(int(embed_dim * lda_ratio), max_lda_components))
        n_random = embed_dim - n_lda

        # Fit LDA to extract the most discriminative linear subspace
        lda = LinearDiscriminantAnalysis(n_components=n_lda)
        lda.fit(X_tr, y_train)
        proj_lda_train = lda.transform(X_tr)
        proj_lda_test = lda.transform(X_te)

        if n_random > 0:
            # Fill remaining dimensions with random projection for diversity
            W_rand = rng.randn(X_tr.shape[1], n_random)
            proj_rand_train = X_tr @ W_rand
            proj_rand_test = X_te @ W_rand

            # Normalize both parts to similar scale so that argsort treats
            # LDA and random dimensions equally. Without this, the LDA
            # components (which tend to have higher variance) would dominate
            # the ranking and reduce the contribution of random components.
            lda_scale = np.std(proj_lda_train) + 1e-10
            rand_scale = np.std(proj_rand_train) + 1e-10
            proj_lda_train /= lda_scale
            proj_lda_test /= lda_scale
            proj_rand_train /= rand_scale
            proj_rand_test /= rand_scale

            proj_train = np.hstack([proj_lda_train, proj_rand_train])
            proj_test = np.hstack([proj_lda_test, proj_rand_test])
        else:
            proj_train = proj_lda_train
            proj_test = proj_lda_test

        perm_train = np.argsort(proj_train, axis=1).astype(float)
        perm_test = np.argsort(proj_test, axis=1).astype(float)
        return perm_train, perm_test

    # ------------------------------------------------------------------
    # Calibrated Encoding
    # ------------------------------------------------------------------
    # Standard random projection followed by argsort can produce biased
    # permutations when the projected dimensions have very different
    # variances — high-variance dimensions dominate the ranking, while
    # low-variance dimensions contribute little ordinal signal.
    #
    # Calibrated encoding applies StandardScaler to the projected
    # dimensions (after projection, before argsort) to equalize variance
    # across all dimensions. This ensures every dimension contributes
    # equally to the ranking, producing more informative permutations.
    #
    # This is particularly useful as a diversity mechanism in multi-view
    # ensembles: mixing calibrated and uncalibrated views increases
    # ensemble diversity because they produce systematically different
    # rankings from the same underlying projection.
    # ------------------------------------------------------------------

    @staticmethod
    def calibrated_encode(X_train, y_train, X_test, embed_dim=32,
                          seed=42, calibration='standardize'):
        """Encode with post-projection calibration before argsort.

        The idea is to equalize the variance of each projected dimension
        so that argsort produces rankings where all dimensions contribute
        equally, rather than being dominated by high-variance directions.

        Args:
            X_train: Training features (N_train x D), already polynomial-expanded.
            y_train: Training labels (N_train,) — unused but kept for API consistency.
            X_test: Test features (N_test x D), already polynomial-expanded.
            embed_dim: Target permutation length.
            seed: Random seed for reproducibility.
            calibration: Calibration method. 'standardize' (default) applies
                StandardScaler to projected dimensions before argsort.

        Returns:
            perm_train: Permutation matrix for training data (N_train x embed_dim).
            perm_test: Permutation matrix for test data (N_test x embed_dim).
        """
        rng = np.random.RandomState(seed)

        # Pre-projection standardization (same as standard encoding)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        # Random projection
        W_proj = rng.randn(X_train.shape[1], embed_dim)
        proj_train = X_tr @ W_proj
        proj_test = X_te @ W_proj

        # Post-projection calibration: equalize variance across dimensions
        # so that argsort treats all projected dimensions equally
        if calibration == 'standardize':
            proj_scaler = StandardScaler()
            proj_train = proj_scaler.fit_transform(proj_train)
            proj_test = proj_scaler.transform(proj_test)

        perm_train = np.argsort(proj_train, axis=1).astype(float)
        perm_test = np.argsort(proj_test, axis=1).astype(float)
        return perm_train, perm_test

    # ------------------------------------------------------------------
    # Permutation Data Augmentation
    # ------------------------------------------------------------------
    # Data augmentation in permutation space works by applying random
    # adjacent transpositions (swapping neighboring elements). Each swap
    # changes the Spearman footrule distance by exactly 2, so this
    # produces controlled perturbations that stay "close" to the original
    # permutation in the metric space ArrowFlow uses.
    #
    # This is analogous to adding Gaussian noise in Euclidean space, but
    # adapted to the discrete permutation space. It helps ArrowFlow learn
    # more robust filters by exposing them to slight ordering variations.
    #
    # Empirically, augmentation improves accuracy on moderate-size datasets
    # (n_samples >= 150) with low-to-medium dimensionality (n_features <= 30).
    # For very small datasets (n < 150), the noise from augmentation hurts
    # more than it helps. For high-dimensional data, the natural diversity
    # from random projection already provides sufficient variation.
    # ------------------------------------------------------------------

    @staticmethod
    def augment_permutation_data(data_train, n_augmentations=1, max_swaps=2, seed=42):
        """Augment training data by applying random adjacent transpositions.

        Each augmented copy applies 1 to max_swaps random adjacent swaps,
        creating new permutations that are close to the originals in
        Spearman footrule distance (each swap = distance of 2).

        Args:
            data_train: List of [permutation, label, magnitude] triples.
            n_augmentations: Number of augmented copies per original sample.
            max_swaps: Maximum number of adjacent swaps per augmented copy.
            seed: Random seed for reproducibility.

        Returns:
            Augmented data list (originals + augmented copies).
        """
        rng = np.random.RandomState(seed)
        augmented = list(data_train)  # keep all originals

        for _ in range(n_augmentations):
            for dp in data_train:
                perm = list(dp[0])  # copy the list of string items
                n = len(perm)
                # Apply 1 to max_swaps random adjacent transpositions
                for _ in range(rng.randint(1, max_swaps + 1)):
                    i = rng.randint(0, n - 1)
                    perm[i], perm[i + 1] = perm[i + 1], perm[i]
                augmented.append([perm, dp[1], dp[2]])

        return augmented

    def convert_sequence_data_sortnet(self, X, y):
        data_net = []
        input_vertices = []

        unique_X = np.unique(X)
        for idx, data_pt in enumerate(X):
            label = y[idx]
            data_pt = np.where(data_pt[:, None] == unique_X[None, :])[1]
            data_point = list(map(str, data_pt))
            # data_point = list(filter(lambda a: a != '0', data_point))
            data_net.append([data_point, str(label), 1])
            input_vertices += data_point
            input_vertices = np.unique(np.asarray(input_vertices)).tolist()

        return data_net, input_vertices


class TensorNet(nn.Module):
    def __init__(self, id_key, adj_list_items, number_of_hidden, init_output_layer=False, no_output_nodes=1):
        super().__init__()
        self.id = id_key
        self.layer_type = 'tensor'
        self.vocab_size = sortnet_config.vocabulary_size
        self.layer_vertices_input = np.asarray(list(adj_list_items))
        self.input_dim = len(adj_list_items)

        self.input_dim_embedding = sortnet_config.vector_size
        self.graph = Graph(id_key=id_key)
        self.graph.num_vertices = number_of_hidden
        self.number_of_hidden = number_of_hidden
        self.device = device
        self.layer_vertices = self.output_vertices(number_of_hidden)
        self.fc1 = nn.Linear(self.input_dim, number_of_hidden, device=device)
        self.fc2 = nn.Linear(number_of_hidden, number_of_hidden, device=device)
        self.max_pool1 = nn.MaxPool1d(3, stride=1, padding=1, ceil_mode=True)
        self.instance_norm = nn.InstanceNorm1d(number_of_hidden, device=device)
        nn.init.uniform(self.fc1.weight)
        nn.init.uniform(self.fc2.weight)
        self.fc3 = nn.Linear(number_of_hidden, number_of_hidden, device=device)
        self.init_output_layer = init_output_layer
        self.embedding_layer_flag = sortnet_config.embedding_layer_flag
        self.no_output_nodes = no_output_nodes
        self.output_layer = nn.Linear(self.input_dim, no_output_nodes)
        self.hidden_to_output = nn.Linear(number_of_hidden, no_output_nodes)
        self.lossfunc_hybrid = F.l1_loss  # F.l1_loss  CustomSortLoss()
        self.model_net = None
        if self.embedding_layer_flag:
            self.model_embedding_nlp_net()
        else:
            self.model_pure_nlp()

    def model_pure_nlp(self):
        if self.init_output_layer:  # If last layer is 'tensor', it outputs the class or regression result
            self.model_net = nn.Sequential(
                self.output_layer
            )
        else:
            self.model_net = nn.Sequential(
                self.fc1,
                self.instance_norm,
                nn.ReLU(),
                self.fc2,
                self.instance_norm,
                nn.ReLU(),
                self.fc3
            )

    def model_embedding_nlp_net(self):
        self.model_net = nn.Sequential(
            nn.Embedding(self.vocab_size, self.number_of_hidden, device=self.device),
            nn.Flatten(1),
            nn.Linear(self.input_dim_embedding * self.number_of_hidden, self.number_of_hidden, device=self.device),
        )

    def forward(self, x):

        if self.embedding_layer_flag:
            x = self.model_net(x.long())
        else:
            x = self.model_net(x)

        return x

    def output_vertices(self, number_of_hidden):
        layer_vertices = []
        for idx_hidden in range(number_of_hidden):
            layer_vertices.append(self.id + '_' + str(idx_hidden))
        return layer_vertices

    def define_optimizer(self, optimizer):
        no_of_iters = sortnet_config.no_of_iters
        self.optimizer_hybrid = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_hybrid, milestones=
        [no_of_iters/10, no_of_iters/5, no_of_iters/4, no_of_iters/2], #[50, 100, 150, 200],
                                                              gamma=0.7)

    def step_optimizer(self):
        self.optimizer_hybrid.zero_grad()
        self.optimizer_hybrid.step()
        self.scheduler.step()


class Vertex:
    # A single node/vertex with its filters, distance computation and gradient accumulation methods.
    def __init__(self, id_key, adj_list, layer_type, sortnet_config):
        self.index_list = np.asarray(list(range(0,len(adj_list))))
        self.id = id_key
        self.adjacency_list = adj_list
        self.valid_ids = None
        self.adjacency_list_integer = self.list_2_integer(adj_list)
        self.adjacency_list_sorted_index = np.argsort(self.adjacency_list_integer)
        self.adjacency_list_np = np.asarray(adj_list)
        self.adjacency_list_sorted = self.adjacency_list_np[self.adjacency_list_sorted_index]
        self.mat_mult_index_vec = np.asarray(list(range(1,len(self.adjacency_list)+1)))
        self.adjacency_list_dict = self.list_2_dict(self.adjacency_list)
        self.adjacency_list_dict_flat = {item: pos for pos, item in enumerate(self.adjacency_list)}
        self.adjacency_list_set = set(self.adjacency_list)
        self._adj_sort_index = np.argsort(self.adjacency_list_np)
        self._adj_sort_index_list = self._adj_sort_index.tolist()
        self._sorted_adj = self.adjacency_list_np[self._adj_sort_index]
        self.len_list = len(self.adjacency_list)
        self.invert_motion_vector = np.flip(np.arange(1,self.len_list+1)) - np.arange(1,self.len_list+1)
        self.idx_correction = np.zeros((len(self.adjacency_list),))
        self.permutation_matrix_accumulate = np.identity(len(self.adjacency_list)) #np.zeros((len(self.adjacency_list),len(self.adjacency_list)))
        self.compute_method = 'matrix' # 'matrix' vs 'vector'
        self.no_of_data_pts = 1
        self.idx_correction_cumsum = self.update_idx_correction_cumsum()
        self.missing_data_cost_weight = sortnet_config.missing_data_cost_weight #1
        self.layer_type = layer_type
        self.distance_computation_metric = sortnet_config.distance_computation_metric
        self.average_method_position = sortnet_config.average_method_position
        self.normalize_motion_for_rf = True

    def __str__(self):
        return self.id + ' connected to in sorted order: ' + " ".join([x for x in self.adjacency_list])

    @staticmethod
    def list_2_dict(adj_list, list_values=True):
        adj_list_dict = {}
        if adj_list:
            if list_values is False:
                adj_list_dict = {ele: pos for pos, ele in enumerate(adj_list)}
            else:
                adj_list_dict = defaultdict(list)
                for pos, ele in enumerate(adj_list):
                    adj_list_dict[ele].append(pos)
                adj_list_dict = dict(adj_list_dict)

        return adj_list_dict

    def list_2_integer(self, adj_list):
        adj_integer = []
        for iter_ind, iter_adj in enumerate(adj_list):
            try:
                adj_integer.append(int(iter_adj)-1)
            except (ValueError, TypeError):
                ind_iter = int(iter_adj.split('_')[-1])
                adj_integer.append(ind_iter)

        return np.asarray(adj_integer)

    def distance_cost_computation(self,motion_vector):
        cost = 0
        if motion_vector.shape[0]>0:
            if self.distance_computation_metric == 'l1':
                cost = (np.sum(np.abs(motion_vector)))
            elif self.distance_computation_metric == 'l2':
                cost = np.sqrt((np.sum(np.square(motion_vector))))
            elif self.distance_computation_metric == 'l0':
                cost = (np.sum(np.abs(np.sign(motion_vector))))
        return cost

    def update_idx_correction_cumsum(self):
        idx_correction_cumsum = np.cumsum(self.idx_correction)

        return idx_correction_cumsum

    def dist_mat_custom(self, a, b):
        return np.sqrt(np.sum(np.square(a[:, np.newaxis, :] - b[np.newaxis, :, :]), axis=-1))

    def relu(self, x):
        return (np.maximum(0, x))
        # return (np.maximum(0.1*x, x))

    # The most important function that computes the distance between two filters (sorted set). Edit distance is implemented.
    def compute_distance(self, adj_list_comp, direction='affinity', magnitude=1, compute_stage='feedforward'):
        if magnitude < 0:
            direction = 'repulsion'
            magnitude = abs(magnitude)
            adj_list_comp = adj_list_comp[::-1]

        # Sort based distance computation. edit distance.
        motion_indices = np.ones((self.len_list,))
        motion_indices.fill(np.nan)

        fast_and_furious = False
        adj_list_comp_dict = {}
        # No repetition in the data and no missing data, then compute directly without many checks
        # Use cached set for faster comparison
        if len(adj_list_comp) == self.len_list and len(set(adj_list_comp)) == self.len_list \
                and set(adj_list_comp) == self.adjacency_list_set:
            fast_and_furious = True
        else:
            adj_list_comp_dict = self.list_2_dict(adj_list_comp)

        deleted_item = False
        #Faster than matrix based method, so hybrid???
        deleted_item, motion_indices = self.distance_for_loop(adj_list_comp, adj_list_comp_dict, deleted_item, fast_and_furious,
                                              motion_indices)

        # If there is receptive field for the node (i.e. only sees partial input), normalize with average motion
        # It is expected to make the response translation invariant.
        if self.normalize_motion_for_rf and motion_indices.shape[0] < len(adj_list_comp):
            avg_motion = np.nanmean(motion_indices)
            motion_indices = np.round(motion_indices - avg_motion)

        if compute_stage == 'feedforward':
            total_dist = self.compute_total_dist(adj_list_comp, deleted_item, motion_indices)
        else:
            total_dist = 0

        if compute_stage == 'backpropagation_inplace':
            # Directly accumulate into permutation_matrix_accumulate (avoids building temp matrix)
            self.accumulate_perm_inplace(motion_indices, magnitude)
            return motion_indices, total_dist, None

        motion_perm_matrix = self.compute_perm_matrix(compute_stage, magnitude, motion_indices)

        return motion_indices, total_dist, motion_perm_matrix

    def compute_perm_matrix(self, compute_stage, magnitude, motion_indices, skip_zero_motion=False):
        # The swap info is represented in a permutation matrix. How much we need to move each element in the filter.
        motion_perm_matrix = None
        if compute_stage == 'backpropagation':
            n = motion_indices.shape[0]
            coord_new_vec = motion_indices + self.index_list[:n]
            np.clip(coord_new_vec, 0, self.len_list - 1, out=coord_new_vec)
            # Find non-NaN indices efficiently
            nan_mask = np.isnan(coord_new_vec)
            if not np.any(nan_mask):
                # Fast path: no NaN values (common in fast_and_furious mode)
                not_nan_ind = self.index_list[:n]
            else:
                not_nan_ind = np.where(~nan_mask)[0]
            # Build sparse permutation matrix directly
            motion_perm_matrix = np.identity(self.len_list)
            motion_perm_matrix[not_nan_ind, not_nan_ind] = 0
            motion_perm_matrix[not_nan_ind, np.int64(coord_new_vec[not_nan_ind])] = abs(magnitude)

        return motion_perm_matrix

    def accumulate_perm_inplace(self, motion_indices, magnitude):
        """Directly accumulate permutation into self.permutation_matrix_accumulate without building a temporary matrix."""
        coord_new_vec = motion_indices + self.index_list
        coord_new_vec[coord_new_vec < 0] = 0
        coord_new_vec[coord_new_vec > self.len_list - 1] = self.len_list - 1
        has_nan = np.isnan(motion_indices).any()
        if has_nan:
            not_nan_ind = np.where(~np.isnan(coord_new_vec))[0]
            # Identity contribution for NaN positions
            nan_ind = np.where(np.isnan(coord_new_vec))[0]
            self.permutation_matrix_accumulate[nan_ind, nan_ind] += 1
            # Permutation for non-NaN positions
            self.permutation_matrix_accumulate[not_nan_ind, np.int64(coord_new_vec[not_nan_ind])] += abs(magnitude)
        else:
            # Fast path: all values valid - no NaN check needed
            self.permutation_matrix_accumulate[self.index_list, np.int64(coord_new_vec)] += abs(magnitude)

    def compute_total_dist(self, adj_list_comp, deleted_item, motion_indices):
        # deleted_item flag ensures that we use np.argwhere (costly computation) only when partial input exists.
        if deleted_item:
            # deleted items treated as sent back to the end of the list
            deletion_dist = self.distance_cost_computation((self.len_list) - np.argwhere(np.isnan(motion_indices)))
            swap_dist = self.distance_cost_computation(motion_indices[np.argwhere(~np.isnan(motion_indices))])
        else:
            deletion_dist = 0
            swap_dist = self.distance_cost_computation(motion_indices)
        total_dist = round(self.missing_data_cost_weight * deletion_dist) + swap_dist
        return total_dist

    def distance_for_loop(self, adj_list_comp, adj_list_comp_dict, deleted_item, fast_and_furious, motion_indices):
        if fast_and_furious:
            adj_list_comp_dict_flat = {item: pos for pos, item in enumerate(adj_list_comp)}
            new_locations = list(map(adj_list_comp_dict_flat.get, self.adjacency_list))
            motion_indices = np.asarray(new_locations) - self.index_list
        else:
            for adj_vertex in self.adjacency_list_dict:
                # All bells and whistles for incomplete data and rf based processing.
                if adj_vertex in adj_list_comp_dict:
                    self.compute_motion_indices_bells_whistles(adj_list_comp_dict, adj_vertex, motion_indices)
                else:
                    deleted_item = True

        return deleted_item, motion_indices

    def compute_motion_indices_bells_whistles(self, adj_list_comp_dict, adj_vertex, motion_indices):
        pos2_list = adj_list_comp_dict[adj_vertex]
        pos1_list = self.adjacency_list_dict[adj_vertex]
        # If there are repeating items in adj lists, then search for the smallest motion combination
        if len(pos2_list) > 1 or len(pos1_list) > 1:
            dist_mat = distance_matrix(np.expand_dims(np.asarray(pos1_list), axis=1),
                                       np.expand_dims(np.asarray(pos2_list), axis=1), p=1)
            indices_min = np.where(dist_mat == dist_mat.min())
            pos1 = indices_min[0][0]
            pos2 = indices_min[1][0]
            del adj_list_comp_dict[adj_vertex][pos2]
        else:
            pos1 = pos1_list[0]
            pos2 = pos2_list[0]
        distance_estimated = pos2 - pos1
        motion_indices[pos1] = distance_estimated

    # Accumulate error for the vertex. Uses forward input and the vertex filter.
    def accumulate_motion(self, adj_list_comp, magnitude=1):

        if magnitude is None or magnitude == 0:
            return np.zeros((self.len_list,))

        # Compute distance (motion indices)
        motion_indices, total_dist, _ = \
            self.compute_distance(adj_list_comp, magnitude=magnitude, compute_stage='backpropagation_inplace')

        motion_indices[np.isnan(motion_indices)] = 0

        return motion_indices

    # Update the filter with the accumulated error for the vertex.
    def apply_motion(self, update_adj_list=True):
        # Permutation matrix based computation of the new and better filter sequence.
        adjacency_list_clean = self.compute_adj_list_with_permutation()

        motion_indices = None
        if update_adj_list:
            self.adjacency_list = adjacency_list_clean
            self.adjacency_list_dict = self.list_2_dict(self.adjacency_list)
            self.adjacency_list_dict_flat = {item: pos for pos, item in enumerate(self.adjacency_list)}
            self.adjacency_list_set = set(self.adjacency_list)
            self.adjacency_list_np = np.asarray(adjacency_list_clean)
            self._adj_sort_index = np.argsort(self.adjacency_list_np)
            self._adj_sort_index_list = self._adj_sort_index.tolist()
            self._sorted_adj = self.adjacency_list_np[self._adj_sort_index]
            self.clean_motion_accumulation(adjacency_list_clean)

        return motion_indices

    def clean_motion_accumulation(self, adjacency_list_clean):
        self.permutation_matrix_accumulate = np.identity(len(self.adjacency_list))

    def compute_adj_list_with_permutation(self):
        if self.permutation_matrix_accumulate.sum() == 0:
            return self.adjacency_list

        locations = np.asarray(list(range(len(self.adjacency_list))))

        weighted_locations = np.multiply(locations, self.permutation_matrix_accumulate)
        decimal_idx = np.sum(weighted_locations, axis=1) / np.sum(self.permutation_matrix_accumulate, axis=0)
        adjacency_list_clean = self.adjacency_list_np[np.argsort(decimal_idx)].tolist()

        return adjacency_list_clean

    def diff_lists(self, li1, li2):
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
        return li_dif


class Graph:
    def __init__(self, id_key, adj_list=None, s_neigh=None, t_neigh=None):
        self.id = id_key
        self.adjacency_list = adj_list
        self.adjacency_list_dict = Vertex.list_2_dict(self.adjacency_list)
        self.adjacency_list_np = np.asarray(adj_list)
        self.spatial_neigh = s_neigh
        self.temporal_neigh = t_neigh
        self.vertex_list = {}
        self.num_vertices = 0

    def add_vertex(self, new_vertex):
        self.num_vertices = self.num_vertices + 1
        self.vertex_list[new_vertex.id] = new_vertex
        return new_vertex

    def get_vertex(self,n):
        if n in self.vertex_list:
            return self.vertex_list[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertex_list

    def get_vertices(self):
        return self.vertex_list.keys()

    def __iter__(self):
        return iter(self.vertex_list.values())


class VertexFilters:
    def __init__(self, id_key, adj_list_items, no_of_filters, filter_rf, layer_type,
                 data_initial, frequenct_dict, sortnet_config):

        self.adj_list_items = list(adj_list_items)
        self.item_to_idx = {item: i for i, item in enumerate(adj_list_items)}
        self.id = id_key
        self.network_type = sortnet_config.network_type
        self.update_iter = 0
        self.graph = Graph(id_key + '_' + 'graph')
        self.layer_type = layer_type
        self.hidden_nodes = []
        self.hidden_activation = []
        for idx in range(no_of_filters):
            self.hidden_nodes.append("{}_{}".format(id_key, idx + 1))
            self.hidden_activation.append("{}_{}".format(id_key, idx + 1))

        self.random_activation_function = 2 * np.random.rand(no_of_filters, 1) #np.random.rand(no_of_filters, 1)  #np.ones((no_of_filters, 1))

        if data_initial:
            rnd_data_perm = np.random.permutation(len(data_initial))

        adj_list_np = np.asarray(list(adj_list_items))

        # Index matrix holds the index of elements in each of the filters.
        self.index_matrix = np.zeros((no_of_filters, len(adj_list_items)))
        for fidx in range(no_of_filters):
            id_key_f = id_key + '_' + str(fidx)
            valid_ids = np.arange(len(adj_list_items))
            if sortnet_config.initial_filter_with_data and data_initial is not None and fidx < len(data_initial):
                adj_list = data_initial[rnd_data_perm[fidx]][0]
                adj_list, index_vector = self.complement_filter_adj_list(adj_list, adj_list_items, filter_rf)
            elif sortnet_config.frequency_based_filter and frequenct_dict is not None:
                freq_pd = pd.DataFrame(frequenct_dict, index=[0])
                # sample according to probability
                probabilities = freq_pd.values[0]
                probabilities = np.power(probabilities, 0.25)
                probabilities = probabilities / np.sum(probabilities)
                input_indices = freq_pd.columns.values
                adj_list_np_ = np.random.choice(input_indices, len(adj_list_items), probabilities.all())
                if filter_rf:
                    adj_list_np_ = adj_list_np_[0:filter_rf]
                adj_list = adj_list_np_.tolist()
                index_vector = self.create_index_vector(adj_list, adj_list_items, filter_rf)
            else:
                adj_list_items_ = list(adj_list_items)
                rnd_perm_ = np.random.permutation(len(adj_list_items_))

                adj_list_np_ = adj_list_np[rnd_perm_]
                index_vector = np.argsort(rnd_perm_)

                if filter_rf:
                    adj_list_np_ = adj_list_np_[0:filter_rf]
                    index_vector[np.where(index_vector > filter_rf)] = 0 #rnd_perm_.shape[0] / 10
                    valid_ids = np.sort(rnd_perm_[0:filter_rf])

                adj_list = adj_list_np_.tolist()

            self.index_matrix[fidx] = index_vector

            vertex_ = Vertex(id_key_f, adj_list, layer_type, sortnet_config)

            vertex_.valid_ids = valid_ids.tolist()

            self.graph.add_vertex(vertex_)

        # GPU tensor created lazily in _ensure_gpu_tensor() to avoid CUDA init in forked processes
        self._index_matrix_gpu = None

    def update_index_matrix(self):
        missing_data_loc = sortnet_config.missing_data_loc_mult * len(self.adj_list_items)
        for vertex_iter in self.graph:
            idx_vertex = int(vertex_iter.id.split('_')[-1])
            adj_dict = vertex_iter.adjacency_list_dict
            self.index_matrix[idx_vertex] = np.array([
                adj_dict[item][0] if item in adj_dict else missing_data_loc
                for item in self.adj_list_items
            ])
        # Invalidate GPU tensor cache
        self._index_matrix_gpu = None

    @property
    def index_matrix_gpu(self):
        """Lazy GPU tensor — created on first access to avoid CUDA init in forked processes."""
        if self._index_matrix_gpu is None:
            try:
                self._index_matrix_gpu = torch.tensor(self.index_matrix, dtype=torch.float32, device=device)
            except RuntimeError:
                # Forked subprocess — return CPU tensor as fallback
                self._index_matrix_gpu = torch.tensor(self.index_matrix, dtype=torch.float32)
        return self._index_matrix_gpu

    def complement_filter_adj_list(self, adj_list, adj_list_items, filter_rf):
        diff_list = self.diff_lists(adj_list_items, adj_list)
        if not diff_list == []:
            random.shuffle(diff_list)
            adj_list += diff_list

        if filter_rf:
            # random.shuffle(adj_list)
            adj_list = adj_list[0:filter_rf]

        adj_list = adj_list[0:len(adj_list_items)]

        index_vector = self.create_index_vector(adj_list, adj_list_items, filter_rf)

        return adj_list, index_vector

    def create_index_vector(self, adj_list, adj_list_items, filter_rf=None):
        index_vector = np.zeros((len(adj_list_items),))
        for iter_index in range(len(adj_list_items)):
            if adj_list_items[iter_index] in adj_list:
                index_vector[iter_index] = adj_list.index(adj_list_items[iter_index])

        if filter_rf is not None:
            index_vector[np.where(index_vector > filter_rf)] = 0

        return index_vector

    def diff_lists(self, li1, li2):
        li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
        return li_dif

    def update_vertex_filters(self, data):
        for filter_key in data.keys():
            data_filter = data[filter_key]
            vertex_filter = self.graph.get_vertex(filter_key)

            for data_point in data_filter:
                data_vector = data_point[0]
                direction = data_point[1]
                magnitude = data_point[2]
                vertex_filter.accumulate_motion(data_vector, direction, magnitude)

            vertex_filter.apply_motion()

        self.update_iter += 1


class SortFlowHybridNetwork:
    def __init__(self, id_key, adj_list_items, number_of_classes, file_name, sortnet_config, pretrained_model=None):
        self.id = id_key
        self.data_graph = DataGraph('cn_data')
        self.network_type = sortnet_config.network_type
        self.file_name = file_name
        self.optimal_model = None
        self.update_iter = 0
        self.vocab_size = sortnet_config.vocabulary_size
        self.eval_period = sortnet_config.eval_period
        self.num_of_layers = len(sortnet_config.no_of_filters)-1
        self.adj_list_items_initial = adj_list_items
        self.learning_rate = sortnet_config.learning_rate
        self.average_method_motion = sortnet_config.average_method_motion
        self.change_probability_when_decision_correct = sortnet_config.change_probability_when_decision_correct
        self.last_layer_update = sortnet_config.last_layer_update
        self.num_of_epochs = 0
        self.min_error_val = 100
        self.no_of_classes = number_of_classes
        self.max_val_data = None
        self.min_val_data = None
        self.regression_method = sortnet_config.regression_method
        self.ratio_data_backprop = sortnet_config.ratio_data_backprop
        self.problem = sortnet_config.problem
        self.verbose = sortnet_config.verbose
        if sortnet_config.problem == 'sorting' and sortnet_config.backprop_signal_replication_flag:
            self.backprop_signal_replication = self.no_of_classes
        else:
            self.backprop_signal_replication = 1

        if pretrained_model is not None:  # Load model if given
            self.graph = sortnet_config.pretrained_model
        else:
            self.graph = Graph(id_key + '_' + 'graph')
            l_count = 0
            # Initialize the VertexFilter objects. It corresponds to a single layer. It contains Vertex objects.
            for l_idx, no_filters_layer in enumerate(sortnet_config.no_of_filters):
                id_key_l = id_key + '_ly' + str(l_count)
                if sortnet_config.layer_types[l_idx] == 'sort':
                    frequenct_dict = None
                    data_initial = None
                    if l_count == 0:
                        if sortnet_config.frequency_based_filter and sortnet_config.frequency_dict is not None:
                            frequenct_dict = sortnet_config.frequency_dict
                        if sortnet_config.initial_filter_with_data and sortnet_config.data_initial is not None:
                            data_initial = sortnet_config.data_initial
                        vertex_filter = VertexFilters(id_key_l, adj_list_items, no_filters_layer,
                                                      sortnet_config.filter_rfs[l_idx],
                                                      sortnet_config.layer_types[l_idx],
                                                      data_initial, frequenct_dict, sortnet_config)
                        vertex_filter.adj_list_items = adj_list_items
                    else:
                        # Only input layer can be metric
                        vertex_filter = VertexFilters(id_key_l, layer_vertices, no_filters_layer,
                                                      sortnet_config.filter_rfs[l_idx],
                                                      sortnet_config.layer_types[l_idx],
                                                      None, None, sortnet_config)

                        vertex_filter.adj_list_items = list(layer_vertices)

                    self.graph.add_vertex(vertex_filter)
                    layer_vertices = vertex_filter.graph.vertex_list.keys()
                else:  # 'tensor' type layer
                    if l_count == 0:
                        adj_list_items_ = adj_list_items
                    else:
                        adj_list_items_ = layer_vertices

                    init_output_layer = False
                    no_output_nodes = 1
                    if l_count == len(sortnet_config.no_of_filters) - 1:  # If the tensor layer is the last layer, create output layer
                        init_output_layer = True
                        if self.problem == 'classification' or self.problem == 'sorting':
                            no_output_nodes = no_filters_layer

                    model_tensor = TensorNet(id_key_l, adj_list_items_, no_filters_layer,
                                             init_output_layer, no_output_nodes)
                    optimizer_hybrid = torch.optim.Adam(model_tensor.model_net.parameters(), lr=sortnet_config.lr_tensor)  # self.learning_rate/300
                    model_tensor.define_optimizer(optimizer_hybrid)
                    self.graph.add_vertex(model_tensor)
                    layer_vertices = model_tensor.layer_vertices

                l_count += 1

    def train(self, data, sortnet_config, return_dict=None, network_id_moe=None):
        if return_dict is not None:
            seed_no = (os.getpid() * int(time.time() * 1000)) % 123456
            np.random.seed(seed_no)

        train_type = sortnet_config.train_type
        no_of_iters = sortnet_config.no_of_iters
        batch_size = sortnet_config.batch_size

        data_train = data[0]
        data_test = data[1]

        data_train, data_validation = self.split_into_train_validation_dataset(data_train,
                                                                               sortnet_config.val_data_ratio)

        # Initialize optimla model with the initial model
        self.optimal_model = copy.deepcopy(self.graph)

        problem = self.problem
        if problem == 'regression':
            # Save the min and max values of the input data for regression task data normalization.
            self.analyze_max_min_dataset(data_train)

        error_val = 100
        self.min_error_val = 1
        # error_train, prediction = self.evaluate(data_train, train_type, problem) #Might take too much time for large datasets.
        error_train = 1
        if data_validation and train_type == 'supervised':
            error_val, prediction = self.evaluate(data_validation, train_type, problem)
            self.min_error_val = error_val
        if problem == 'classification':
            error_mult = 100
            precision_p = 1
        else:
            error_mult = 1
            precision_p = 2
        if self.verbose > 0:
            print("Initial Error Train/Val/MinVal = {} / {} / {}".format(round(error_mult*error_train,precision_p),
                                                                         np.round(error_mult*error_val,precision_p),
                                                                         np.round(error_mult*self.min_error_val,precision_p)))

        data_train_ = data_train
        errors_update = [None, None, None]
        for iter_ in range(no_of_iters):
            # Select batch data
            if batch_size:
                perm_data = np.random.permutation(len(data_train))[0:batch_size]
                data_train_ = [data_train[i] for i in perm_data.tolist()]
            # Batch update and evaluation:
            motion_last_layer, errors_update = \
                self.update_network(data_train_, data_validation, train_type, problem)

        # Test error on completely separate data. Use the best model during training.
        self.graph = copy.deepcopy(self.optimal_model)
        error_test, prediction = self.evaluate(data_test, train_type, problem)
        errors_update.append(error_test)

        # self.print_save_optimal_model()

        # Keep track of error in MoE setting
        if return_dict is not None and network_id_moe is not None:
            return_dict[network_id_moe] = [errors_update, self]

        return errors_update

    def update_network(self, data_train, data_validation=None, train_type='supervised', problem='classification'):
        tf_init = datetime.now()
        error_train_, forward_input_backprop_all_data, motion_last_layer, prediction_list = \
            self.forward_propagate(data_train, train_type, problem)
        tf_total = datetime.now() - tf_init

        tb_init = datetime.now()
        motion_last_layer = self.backward_propagate(forward_input_backprop_all_data,
                                                    motion_last_layer)
        tb_total = datetime.now() - tb_init

        if sortnet_config.evaluate_train_data:
            error_train, prediction = self.evaluate(data_train, train_type, problem)
        else:
            error_train = 0

        te_init = datetime.now()
        error_val = 100
        if self.update_iter % self.eval_period == 0 and data_validation and train_type == 'supervised':
            error_val, prediction = self.evaluate(data_validation, train_type, problem)
            if error_val < self.min_error_val:
                self.min_error_val = error_val
                self.sub_optimal_model = copy.deepcopy(self.optimal_model)
                self.optimal_model = copy.deepcopy(self.graph)
        te_total = datetime.now() - te_init

        if problem == 'classification':
            error_mult = 100
            precision_p = 1
        else:
            error_mult = 1
            precision_p = 2
        if self.verbose > 1:
            print("Iter: {} \t Error Train/Val/MinVal = {} / {} / {}. \t Time F/B/E : {} / {} / {} \t LR: {}".
                  format(self.update_iter+1, np.round(error_mult*error_train,precision_p), np.round(error_mult*error_val,precision_p),
                        np.round(error_mult*self.min_error_val,precision_p),np.round(tf_total.total_seconds(),3),
                         np.round(tb_total.total_seconds(),3), np.round(te_total.total_seconds(),3), np.round(self.learning_rate,3)))

        self.update_iter += 1
        return motion_last_layer, [error_train, error_val, self.min_error_val]

    def evaluate(self, data, train_type, problem='classification', return_dict=None, network_id_moe=None):
        if return_dict is not None:
            seed_no = (os.getpid() * int(time.time() * 1000)) % 123456
            np.random.seed(seed_no)

        # Almost identical to forward computation, only for evaluating after batch update. No gradient/error computation
        error_forwardprop = self.forward_propagate(data, train_type, problem, True)

        if return_dict is not None and network_id_moe is not None:
            return_dict[network_id_moe] = [error_forwardprop[0], error_forwardprop[3]]

        return error_forwardprop[0], error_forwardprop[3]  # error and prediction returned

    def _is_all_sort_layers(self):
        """Check if all layers in this network are sort layers."""
        for h in range(self.graph.num_vertices):
            layer_name = self.id + '_ly' + str(h)
            layer = self.graph.vertex_list[layer_name]
            if layer.layer_type != 'sort':
                return False
        return True

    def forward_propagate(self, data, train_type, problem, evaluate_only=False):
        # Use batched path for all-sort architectures
        if self._is_all_sort_layers():
            return self._forward_propagate_batch(data, train_type, problem, evaluate_only)
        return self._forward_propagate_sequential(data, train_type, problem, evaluate_only)

    def _forward_propagate_batch(self, data, train_type, problem, evaluate_only=False):
        """Batched forward pass for all-sort-layer architectures.
        Uses integer arrays between layers (no strings in hot loop).
        GPU-accelerated cdist when CUDA is available."""
        error_decision = 0
        motion_last_layer = []
        prediction_list = []
        forward_input_backprop_all_data = {}

        n_data = len(data)
        # Detect GPU availability (gracefully handle forked subprocesses where CUDA can't init)
        use_gpu = False
        if device != 'cpu' and torch.cuda.is_available():
            try:
                torch.tensor(0.0, device=device)  # probe CUDA init
                use_gpu = True
            except RuntimeError:
                pass  # forked subprocess — fall back to CPU
        n_layers = self.graph.num_vertices

        # --- Layer 0: build position array from original data (only place with string→int) ---
        layer0_name = self.id + '_ly0'
        layer0 = self.graph.vertex_list[layer0_name]
        item_to_col = layer0.item_to_idx
        n_vocab_0 = len(layer0.adj_list_items)
        missing_data_loc = sortnet_config.missing_data_loc_mult * n_vocab_0

        positions = np.full((n_data, n_vocab_0), missing_data_loc, dtype=np.float32)
        for i, data_point in enumerate(data):
            for pos, item in enumerate(data_point[0]):
                col = item_to_col.get(item)
                if col is not None:
                    positions[i, col] = pos

        # Pre-cache filter key arrays per layer (for string conversion at boundaries)
        filter_keys_per_layer = {}
        for h in range(n_layers):
            ln = self.id + '_ly' + str(h)
            filter_keys_per_layer[ln] = np.array(list(self.graph.vertex_list[ln].graph.vertex_list.keys()))

        # For backprop: store string input to layer 0
        if not evaluate_only:
            prev_str_lists = [list(dp[0]) for dp in data]

        sort_indices = None
        last_layer = None
        prev_sort_indices = None

        for h in range(n_layers):
            layer_name = self.id + '_ly' + str(h)
            layer = self.graph.vertex_list[layer_name]
            last_layer = layer
            n_filters = layer.graph.num_vertices

            # Compute L1 distances: positions (N_data, N_vocab) vs index_matrix (N_filters, N_vocab)
            if use_gpu:
                pos_gpu = torch.tensor(positions, dtype=torch.float32, device=device)
                dist_gpu = torch.cdist(pos_gpu, layer.index_matrix_gpu, p=1)
                sort_indices = torch.argsort(dist_gpu, dim=1).cpu().numpy()
            else:
                from scipy.spatial.distance import cdist
                dist_batch = cdist(positions, layer.index_matrix, metric='cityblock')
                sort_indices = np.argsort(dist_batch, axis=1)

            # Backprop bookkeeping: store the INPUT string lists for this layer
            if not evaluate_only:
                bp_list = forward_input_backprop_all_data.setdefault(layer_name, [])
                for i in range(n_data):
                    for iter_ in range(self.backprop_signal_replication):
                        bp_list.append([prev_str_lists[i], data[i][1], data[i][2]])

            # Prepare next layer input: inverse permutation (vectorized, no Python loops)
            if h < n_layers - 1:
                # sort_indices[i, rank] = filter_idx → positions[i, filter_idx] = rank
                positions = np.empty((n_data, n_filters), dtype=np.float32)
                ranks = np.arange(n_filters, dtype=np.float32)
                np.put_along_axis(positions, sort_indices,
                                  np.broadcast_to(ranks, (n_data, n_filters)), axis=1)

                # Convert this layer's output to strings for next layer's backprop
                if not evaluate_only:
                    prev_str_lists = filter_keys_per_layer[layer_name][sort_indices].tolist()

            # Save for classification's accumulate_motion (needs input to last layer)
            if h == n_layers - 2:
                prev_sort_indices = sort_indices

        # --- Error evaluation using integer sort_indices (minimal string ops) ---
        layer = last_layer
        if train_type == 'supervised':
            motion_adjustment = (1 / float(self.learning_rate)) * (layer.graph.num_vertices)
            last_layer_prefix = self.id + '_ly' + str(n_layers - 1) + '_'
            last_filter_keys = filter_keys_per_layer[self.id + '_ly' + str(n_layers - 1)]

            # For classification: prev_sort_str for accumulate_motion (input to last layer)
            prev_sort_str = None
            if not evaluate_only and problem == 'classification':
                if n_layers >= 2:
                    prev_layer_name = self.id + '_ly' + str(n_layers - 2)
                    prev_fkeys = filter_keys_per_layer[prev_layer_name]
                    prev_sort_str = prev_fkeys[prev_sort_indices].tolist()
                else:
                    prev_sort_str = [list(dp[0]) for dp in data]

            for data_idx, data_point in enumerate(data):
                if problem == 'sorting':
                    forward_input_sort = last_filter_keys[sort_indices[data_idx]].tolist()
                    forward_input_sort_strip = [str(int(x.split('_')[-1])) for x in forward_input_sort]
                    motion_sorting_last_layer = self.compute_motion_for_two_adj_lists(forward_input_sort_strip, data_point[0])
                    error_decision += np.sum(np.abs(motion_sorting_last_layer)) / len(motion_sorting_last_layer)

                    motion_sorting_last_layer_dict = dict(zip(forward_input_sort, motion_sorting_last_layer))

                    if evaluate_only is False:
                        motion_last_layer_cumulative = []
                        sorted_adj_vertices = None
                        for index_update, vertex_update in enumerate(layer.graph.vertex_list.keys()):
                            vertex_ = layer.graph.vertex_list[vertex_update]
                            sorted_adj_vertices = vertex_._sorted_adj
                            vertex_adj_sort_index = vertex_._adj_sort_index_list
                            magnitude_update = motion_sorting_last_layer_dict[vertex_.id] / motion_adjustment

                            if (self.backprop_signal_replication == 1 and
                                    (magnitude_update is None or magnitude_update == 0)):
                                continue

                            motion_last_layer_iter = (
                                vertex_.accumulate_motion(forward_input_sort, magnitude=magnitude_update))

                            if self.backprop_signal_replication > 1:
                                motion_last_layer_iter = motion_last_layer_iter[vertex_adj_sort_index]
                                sort_index_motion = np.argsort(-np.abs(motion_last_layer_iter))
                                motion_last_layer.append(
                                    [sorted_adj_vertices[sort_index_motion].tolist(),
                                     motion_last_layer_iter[sort_index_motion]])
                            else:
                                motion_last_layer_iter = motion_last_layer_iter[vertex_adj_sort_index]
                                motion_last_layer_cumulative.append(motion_last_layer_iter)

                        if self.backprop_signal_replication == 1:
                            if len(motion_last_layer_cumulative) > 0:
                                avg_motion_vertex = np.mean(np.asarray(motion_last_layer_cumulative), axis=0)
                                avg_motion_vertex = (avg_motion_vertex.shape[0] * sortnet_config.motion_normalization_mult) * \
                                                    avg_motion_vertex / np.max(np.abs(avg_motion_vertex) + 0.00001)
                                sort_index_motion = np.argsort(-np.abs(avg_motion_vertex))
                                motion_last_layer.append(
                                    [sorted_adj_vertices[sort_index_motion].tolist(),
                                     avg_motion_vertex[sort_index_motion]])
                            elif sorted_adj_vertices is not None:
                                motion_last_layer.append(
                                    [sorted_adj_vertices,
                                     np.zeros(len(sorted_adj_vertices))])

                elif problem == 'classification':
                    # Use integer index directly: sort_indices[data_idx, 0] is the predicted class
                    predicted_class = sort_indices[data_idx, 0]
                    gt_class = int(data_point[1])
                    magnitude = data_point[2]
                    if predicted_class != gt_class:
                        error_decision += 1
                    elif np.random.rand() > self.change_probability_when_decision_correct:
                        magnitude = 0
                    prediction_list.append(int(predicted_class))

                    if evaluate_only is False:
                        gt_vertex_id = last_layer_prefix + str(gt_class)
                        magnitude_update = magnitude / motion_adjustment
                        vertex_ = layer.graph.vertex_list[gt_vertex_id]
                        motion_data_point_last_layer = \
                            vertex_.accumulate_motion(
                                prev_sort_str[data_idx], magnitude=magnitude_update)
                        sort_index_motion = np.argsort(-np.abs(motion_data_point_last_layer))
                        motion_last_layer.append(
                            [vertex_.adjacency_list_np[sort_index_motion].tolist(),
                             motion_data_point_last_layer[sort_index_motion]])

                elif problem == 'regression':
                    predicted_class = sort_indices[data_idx, 0]
                    predicted_val = float(last_filter_keys[predicted_class].split('_')[-1])
                    prediction_list.append(predicted_val)
                    gt_val = float(data_point[1])
                    error_decision += abs(gt_val - predicted_val)

        error_train = error_decision / max(len(data), 1)
        return error_train, forward_input_backprop_all_data, motion_last_layer, prediction_list

    def _forward_propagate_sequential(self, data, train_type, problem, evaluate_only=False):
        """Original per-data-point forward pass (used when tensor layers are present)."""
        error_decision = 0
        motion_last_layer = []
        prediction_list = []
        forward_input_backprop_all_data = {}
        layer = None

        # Send data, apply forward computation layer by layer.
        for data_idx, data_point in enumerate(data):
            # First layer receives input data
            forward_input_sort = list(data_point[0])
            # Tensor layer receives distance data from sort layer
            forward_input_tensor = list(map(float, data_point[0]))  # Dummy init for tensor "first layer"
            magnitude = data_point[2]

            for hidden_layer_iter in range(self.graph.num_vertices):
                forward_input_sort_prev = list(forward_input_sort)

                layer_name = self.id + '_ly' + str(hidden_layer_iter)
                layer = self.graph.vertex_list[layer_name]
                if layer.layer_type == 'sort':
                    motion_mat_index = self.compute_motion_mat(forward_input_sort, layer)
                    dist_vertices_data = np.sum(np.abs(motion_mat_index), axis=1)
                    forward_input_sort_index = np.argsort(dist_vertices_data)
                    forward_input_sort = np.asarray(list(layer.graph.vertex_list.keys()))[
                        forward_input_sort_index].tolist()
                    forward_input_tensor = list((dist_vertices_data - dist_vertices_data.min())/np.abs(dist_vertices_data).max())  # Tensor layer receives distance data from sort layer

                    # Bookkeeping the forward response for backprop.
                    if evaluate_only is False:
                        for iter_ in range(self.backprop_signal_replication):
                            forward_input_backprop_all_data.setdefault(layer_name, []).append(
                                [forward_input_sort_prev, data_point[1], data_point[2]])

                else:  # 'tensor' layer type
                    if not isinstance(forward_input_tensor, np.ndarray):
                        if layer.fc1.in_features != len(forward_input_tensor) and not isinstance(layer.model_net[0], nn.Embedding):
                            for iter_fill in range(layer.fc1.in_features - max(forward_input_tensor.shape)):
                                forward_input_tensor.append(0)

                        forward_input_np = np.expand_dims(np.asarray(forward_input_tensor),0)
                    else:
                        if layer.fc1.in_features != len(forward_input_tensor) and not isinstance(layer.model_net[0], nn.Embedding):
                            for iter_fill in range(layer.fc1.in_features - max(forward_input_tensor.shape)):
                                forward_input_tensor = np.append(forward_input_tensor, 0)

                        forward_input_np = forward_input_tensor

                    # BEWARE: There is a negative sign in front of distance. From distance to activation.
                    if not isinstance(layer.model_net[0], nn.Embedding):
                        forward_input_np = -1 * forward_input_np

                    forward_input_t = torch.tensor(forward_input_np, dtype=torch.float32,
                                                   requires_grad=True, device=device)
                    forward_input_t.retain_grad()
                    # Forward propagation
                    tensor_hidden = layer(forward_input_t)
                    tensor_hidden.retain_grad()

                    # The output is converted into sortnet compatible data structure.
                    # The forward input for the next layer
                    forward_input_ = self.data_graph.convert_tensor_to_sortnet_data(tensor_hidden, data_point[1])[0][0]
                    forward_input_sort = []
                    for inp_iter in forward_input_:
                        forward_input_sort.append(layer.id + '_' + str(int(inp_iter) - 1))

                    # WARNING!!! Is this correct???
                    # If the next layer is also a tensor layer, then the input is the tensor output of the current layer.
                    forward_input_tensor = tensor_hidden.detach().cpu().numpy()

                    # Bookkeeping the forward response for backprop.
                    # It is different from Sort layer data. Input and Output tensors kept.
                    if evaluate_only is False:
                        for iter_ in range(self.backprop_signal_replication):
                            forward_input_backprop_all_data.setdefault(layer_name, []).append(
                                [[forward_input_t, tensor_hidden], data_point[1], data_point[2]])

            # The error evaluation for supervised training. Keep track of what to update in the last layer.
            if train_type == 'supervised':

                if layer.layer_type == 'tensor':  # This is the last layer after the for loop.
                    if problem == 'classification':
                        gt_class = torch.tensor([int(data_point[1])], device=device)
                        if evaluate_only is False:
                            loss_tensor = F.cross_entropy(tensor_hidden, gt_class)
                            loss_tensor.backward()  # Backpropagate error on tensor variables
                        predicted_class = torch.argmax(tensor_hidden)  #np.argmax(tensor_hidden.detach().numpy())
                        prediction_list.append(predicted_class)
                        if not int(data_point[1]) == predicted_class:
                            error_decision += 1
                    elif problem == 'regression':
                        gt_val = torch.asarray(float(data_point[1]), device=device)
                        loss_tensor = F.l1_loss(torch.squeeze(tensor_hidden), torch.unsqueeze(gt_val, 0))  # Calculate loss
                        if evaluate_only is False:
                            loss_tensor.backward()  # Backpropagate error on tensor variables
                        prediction_list.append(tensor_hidden.detach().numpy())
                        error_l1 = np.abs(tensor_hidden.detach().numpy() - float(data_point[1]))  #loss_tensor.detach().numpy()
                        error_decision += error_l1[0][0]
                    elif problem == 'sorting':
                        order_gt = np.asarray(list(map(int, data_point[1])))
                        gt_val = torch.tensor([order_gt], device=device).to(dtype=torch.float32)
                        tensor_hidden = torch.abs(tensor_hidden)
                        mult_rank = torch.max(gt_val) / torch.max(tensor_hidden)
                        tensor_hidden = ((tensor_hidden * mult_rank)).to(dtype=torch.float32)
                        target = torch.ones_like(tensor_hidden)
                        # loss_tensor = F.margin_ranking_loss(tensor_hidden, gt_val, target, margin=0.1)
                        loss_tensor = F.l1_loss(tensor_hidden, gt_val)  # Calculate loss
                        # loss_tensor = F.cross_entropy(tensor_hidden, gt_val)
                        # loss_tensor = F.mse_loss(tensor_hidden, gt_val)
                        if evaluate_only is False:
                            loss_tensor.backward()  # Backpropagate error on tensor variables
                        prediction_list.append(tensor_hidden.detach().numpy())
                        # error_l1 = np.mean(np.abs(tensor_hidden.detach().numpy() - order_gt))  # loss_tensor.detach().numpy()
                        # error_decision += error_l1
                        error_decision += loss_tensor.detach().numpy()

                    # accumulate gradients
                    if evaluate_only is False:
                        motion_vertex, sort_index_motion, sorted_adj_vertices_int = self.gradient_to_motion_vertex(
                            forward_input_np, forward_input_t, layer)

                        # WARNING!!!
                        # Need to add also the distance gradient (grad_input). For tensor-tensor layer backpropagation.
                        motion_last_layer.append([np.asarray(sorted_adj_vertices_int)[sort_index_motion].tolist(),
                                                  motion_vertex[sort_index_motion]])

                else:  # Evaluate/error accumulate code for 'sort' layer
                    motion_adjustment = (1 / float(self.learning_rate)) * (layer.graph.num_vertices)
                    if problem == 'sorting':
                        # strip prefix on the string elements
                        forward_input_sort_strip = [str(int(x.split('_')[-1])) for x in forward_input_sort]
                        # Select data_point[0] for autoregressive task and data_point[1] for supervised sorting task
                        motion_sorting_last_layer = self.compute_motion_for_two_adj_lists(forward_input_sort_strip, data_point[0])
                        error_decision += np.sum(np.abs(motion_sorting_last_layer)) / len(motion_sorting_last_layer)

                        # motion_sorting_last_layer[np.abs(motion_sorting_last_layer) < motion_sorting_last_layer.shape[0] / 2] = 0
                        motion_sorting_last_layer_dict = dict(zip(forward_input_sort, motion_sorting_last_layer))

                        if evaluate_only is False:
                            motion_last_layer_cumulative = []
                            for index_update, vertex_update in enumerate(layer.graph.vertex_list.keys()):
                                vertex_ = layer.graph.vertex_list[vertex_update]
                                sorted_adj_vertices = np.sort(vertex_.adjacency_list_np)
                                vertex_adj_sort_index = np.argsort(vertex_.adjacency_list_np).tolist()
                                magnitude_update = motion_sorting_last_layer_dict[vertex_.id] / (motion_adjustment)

                                if (self.backprop_signal_replication == 1 and
                                        (magnitude_update is None or magnitude_update == 0)):
                                    continue

                                motion_last_layer_iter = (
                                    vertex_.accumulate_motion(forward_input_sort_prev, magnitude=magnitude_update))

                                if self.backprop_signal_replication > 1:
                                    motion_last_layer_iter = motion_last_layer_iter[vertex_adj_sort_index]
                                    sort_index_motion = np.argsort(-np.abs(motion_last_layer_iter))
                                    motion_last_layer.append(
                                        [sorted_adj_vertices[sort_index_motion].tolist(),
                                         motion_last_layer_iter[sort_index_motion]])
                                else:
                                    motion_last_layer_iter = motion_last_layer_iter[vertex_adj_sort_index]
                                    motion_last_layer_cumulative.append(motion_last_layer_iter)

                            if self.backprop_signal_replication == 1:
                                if len(motion_last_layer_cumulative) > 0:
                                    avg_motion_vertex = np.mean(np.asarray(motion_last_layer_cumulative), axis=0)
                                    avg_motion_vertex = (avg_motion_vertex.shape[
                                                             0] * sortnet_config.motion_normalization_mult) * \
                                                        avg_motion_vertex / np.max(np.abs(avg_motion_vertex) + 0.00001)
                                    sort_index_motion = np.argsort(-np.abs(avg_motion_vertex))
                                    motion_last_layer.append(
                                        [sorted_adj_vertices[sort_index_motion].tolist(),
                                         avg_motion_vertex[sort_index_motion]])
                                else:
                                    motion_last_layer.append(
                                        [sorted_adj_vertices,
                                         np.zeros((len(sorted_adj_vertices),))])

                    if problem == 'classification':
                        predicted_vertex = forward_input_sort[0]
                        gt_vertex_input = self.id + '_' + 'ly' + str(self.graph.num_vertices - 1) + '_' + data_point[1]
                        if not gt_vertex_input == predicted_vertex:
                            error_decision += 1
                        elif np.random.rand() > self.change_probability_when_decision_correct:
                            magnitude = 0
                        vertex_update_list = [gt_vertex_input]
                        magnitude_update_list = [magnitude/motion_adjustment]
                        prediction_list.append(int(predicted_vertex.split('_')[-1]))

                        # Last layer update computation (sort layer) for multiple cost/goal choices,
                        # i.e. classification, regression.
                        if evaluate_only is False:
                            # Update multiple last layer nodes. Attract and repel at the same time. (NOT IMPLEMENTED YET!)
                            iter_no_update = 0  # Only 0th update implemented (see line above!)
                            for vertex_update, magnitude_update in zip(vertex_update_list, magnitude_update_list):
                                # Last layer, correct vertex motion vector computed.
                                # Gradient computation. Keep track of gradient in a permutation matrix.
                                motion_data_point_last_layer = \
                                    layer.graph.vertex_list[vertex_update].accumulate_motion(forward_input_sort_prev,
                                                                                             magnitude=magnitude_update)
                                # Sort the motion (error) according to amplitude (for partial processing)
                                if iter_no_update == 0:
                                    sort_index_motion = np.argsort(-np.abs(motion_data_point_last_layer))
                                    motion_last_layer.append(
                                        [layer.graph.vertex_list[vertex_update].adjacency_list_np[sort_index_motion].tolist(),
                                         motion_data_point_last_layer[sort_index_motion]])
                                iter_no_update += 1

        error_rate = error_decision / float(len(data))

        return error_rate, forward_input_backprop_all_data, motion_last_layer, prediction_list

    def backward_propagate(self, forward_input_backprop_all_data, motion_last_layer):
        layer_name = self.id + '_ly' + str(self.graph.num_vertices - 1)
        layer = self.graph.vertex_list[layer_name]
        # Last layer update. The error signal is already computed in forward propagation (accumulate_motion).
        if self.last_layer_update:
            if layer.layer_type == 'tensor':
                layer.optimizer_hybrid.step()  # Update the layer variables
                layer.scheduler.step()  # Update the learning rate
                layer.optimizer_hybrid.zero_grad()
            else: # 'sort' last layer update
                for last_layer_vertex_key in layer.graph.vertex_list:
                    last_layer_vertex = layer.graph.vertex_list[last_layer_vertex_key]
                    error_layer= \
                        last_layer_vertex.apply_motion()
        else:  # Reset gradients
            if layer.layer_type == 'tensor':
                layer.optimizer_hybrid.zero_grad()
            else:
                for last_layer_vertex_key in layer.graph.vertex_list:
                    last_layer_vertex = layer.graph.vertex_list[last_layer_vertex_key]
                    last_layer_vertex.clean_motion_accumulation(last_layer_vertex.adjacency_list)  # Reset the accumulated error

        first_layer_skip_gradient = False
        count_number_of_corrections = 0
        count_loop = 0
        # Go over the layers from last to the first to update the filters.
        for hidden_layer_iter in range(self.graph.num_vertices - 1):
            if hidden_layer_iter == self.graph.num_vertices - 2:
                first_layer_skip_gradient = True
            layer_name = self.id + '_ly' + str(self.graph.num_vertices - 2 - hidden_layer_iter)
            layer = self.graph.vertex_list[layer_name]

            #  'tensor'  layer backpropagate tensor layers using pytorch backprop/autodiff.
            if layer.layer_type == 'tensor':
                motion_last_layer = self.update_tensor_layer(first_layer_skip_gradient, forward_input_backprop_all_data,
                                                             layer, layer_name, motion_last_layer)
                continue  # the rest of the code is strictly for sort layer, so continue after this

            motion_adjustment = (1 / float(self.learning_rate)) * (layer.graph.num_vertices / 2) #* ((hidden_layer_iter + 1))

            motion_last_layer_next = []
            cumulative_motion_dict = {}
            corrected_data_ids = {}

            vertex_key = list(layer.graph.vertex_list)[0]
            vertex_ = layer.graph.vertex_list[vertex_key]
            sorted_adj_vertices = vertex_._sorted_adj
            for motion_idx, motion_data_point in enumerate(motion_last_layer):
                layer_data = forward_input_backprop_all_data[layer_name][motion_idx][0]

                for vertex_idx, motion_vertex in enumerate(motion_data_point[1]):
                    # Only use the backprop data with large motions (errors)
                    if vertex_idx > self.ratio_data_backprop * len(motion_data_point[0]):
                        break

                    count_loop += 1
                    vertex_key = motion_data_point[0][vertex_idx]
                    vertex_ = layer.graph.vertex_list[vertex_key]
                    vertex_adj_sort_index = vertex_._adj_sort_index_list

                    # removed int(), round etc here. 8 Sept 2023
                    motion_vertex_mag = motion_vertex / motion_adjustment

                    # Accumulate the error signal for the vertex in the current layer.
                    motion_vertex_data_point = vertex_.accumulate_motion \
                        (layer_data, magnitude=motion_vertex_mag)

                    # If it is first layer, then skip the gradient computation.
                    if first_layer_skip_gradient:
                        continue

                    # Bookkeeping for which data gives proper error signals for the next layer (previous in order).
                    if motion_vertex_mag is not None and abs(motion_vertex_mag) > 0:
                        count_number_of_corrections += 1
                        corrected_data_ids.setdefault(motion_idx, []).append(vertex_idx)

                    # Bookkeeping for the error signal for each vertex in this layer.
                    # Error signal is propagated to the next layer (previous in order).
                    cumulative_motion_dict.setdefault(motion_idx, []).append(motion_vertex_data_point[vertex_adj_sort_index])

            # Average error signal for previous layer. Averaged over every vertex in the current layer.
            for motion_idx in cumulative_motion_dict:
                if motion_idx in corrected_data_ids.keys():
                    accepted_data_pts = corrected_data_ids[motion_idx]
                else:
                    accepted_data_pts = range(len(cumulative_motion_dict[motion_idx]))

                # Use only the proper error signals when we can.
                cumulative_motion_dict_filtered = [cumulative_motion_dict[motion_idx][i] for i in accepted_data_pts]
                # Estimate an "average" error signal from the error data from every vertex in this layer.
                if self.average_method_motion == 'mean':
                    avg_motion_vertex = np.mean(np.asarray(cumulative_motion_dict_filtered),axis=0)
                elif self.average_method_motion == 'median':
                    avg_motion_vertex = np.median(np.asarray(cumulative_motion_dict_filtered), axis=0)
                elif self.average_method_motion == 'mode':
                    avg_motion_vertex = stats.mode(np.asarray(cumulative_motion_dict_filtered), axis=None)
                elif self.average_method_motion == 'cluster':
                    # Cluster the error signals and use only the largest cluster's data for estimation.
                    no_clusters = 3
                    cumulative_motion_dict_filtered_np = MinMaxScaler().fit_transform(
                        np.asarray(cumulative_motion_dict_filtered).transpose()).transpose()
                    clustering = SpectralClustering(n_clusters=3, assign_labels='discretize', random_state=0).fit(
                        cumulative_motion_dict_filtered_np)
                    if hasattr(clustering, 'labels_') and np.sum(clustering.labels_ == 0)>0:
                        label_selected = 0
                        label_size = -10
                        for iter_c in range(no_clusters):
                            if np.sum(clustering.labels_ == iter_c) > label_size:
                                label_size = np.sum(clustering.labels_ == 0)
                                label_selected = iter_c
                        avg_motion_vertex = np.mean(np.asarray(cumulative_motion_dict_filtered)
                                                    [clustering.labels_ == label_selected], axis=0)
                    else:
                        avg_motion_vertex = np.mean(np.asarray(cumulative_motion_dict_filtered), axis=0)
                else: # default is 'mean'
                    avg_motion_vertex = np.mean(np.asarray(cumulative_motion_dict_filtered), axis=0)

                avg_motion_vertex = (avg_motion_vertex.shape[0] * sortnet_config.motion_normalization_mult) * \
                                    avg_motion_vertex / np.max(np.abs(avg_motion_vertex) + 0.00001)
                # Sort the motion (error) according to amplitude (for partial processing)
                sort_index_motion = np.argsort(-np.abs(avg_motion_vertex))
                motion_last_layer_next.append([sorted_adj_vertices[sort_index_motion].tolist(),
                                               avg_motion_vertex[sort_index_motion]])

            motion_last_layer = motion_last_layer_next

            # Update the filters in current layer with the gradient information from the data batch.
            for layer_vertex_key in layer.graph.vertex_list:
                vertex_ = layer.graph.vertex_list[layer_vertex_key]
                error_layer = vertex_.apply_motion()

            layer.update_index_matrix()


        self.num_of_epochs += 1

        # Learning rate schedule. Very simple for now.
        self.adapt_learning_rate()

        return motion_last_layer

    def compute_motion_mat(self, forward_input_sort, layer):
        # The forward layer response with numpy array
        missing_data_loc = sortnet_config.missing_data_loc_mult * len(layer.adj_list_items)

        forward_input_dict = {item: idx for idx, item in enumerate(forward_input_sort)}
        forward_input_index_vec = np.array([
            forward_input_dict.get(item_adj, missing_data_loc)
            for item_adj in layer.adj_list_items
        ])

        motion_mat_index = forward_input_index_vec - layer.index_matrix
        return motion_mat_index

    def gradient_to_motion_vertex(self, forward_input_np, forward_input_t, layer):
        forward_input_sort_np = np.argsort(forward_input_np)

        sorted_adj_vertices = layer.layer_vertices_input  # input to tensor layer's vertices list
        prefix_list = sorted_adj_vertices[0].replace('_' + sorted_adj_vertices[0].split('_')[-1], '')
        sorted_adj_vertices_int = [prefix_list + '_' + str(adj_vertex) for adj_vertex in
                                   forward_input_sort_np[0].tolist()]

        # The gradient wrt input gives the motion for a previous "sort" layer.
        grad_from_forward_input = np.squeeze(forward_input_t.grad.numpy())

        # Skip if no gradient exists
        if np.sum(np.abs(grad_from_forward_input)) > 0:
            min_max_value = (np.max(np.abs(forward_input_np[0])) + np.min(np.abs(forward_input_np[0]))) / 2
            gradient_normalizer = min_max_value  # 0.1 * min_max_value
            grad_input = gradient_normalizer * np.squeeze(grad_from_forward_input)
            grad_input[(np.abs(grad_input) / min_max_value) < sortnet_config.tensor_gradient_threshold] = 0

            forward_input_sort_np_adjusted = np.argsort(-grad_input + forward_input_np[0])
            motion_vertex = self.compute_motion_for_two_adj_lists(forward_input_sort_np[0].tolist(),
                                                                  forward_input_sort_np_adjusted.tolist())

            # When using sort location instead of distance, use gradient directly.
            # motion_vertex = grad_input
            motion_vertex = 1.0 * (motion_vertex.shape[0] * sortnet_config.motion_normalization_mult) * \
                            motion_vertex / (np.max(np.abs(motion_vertex)) + 0.000000000000001)

            sort_index_motion = np.argsort(-np.abs(motion_vertex))
        else:
            motion_vertex = np.zeros((len(forward_input_sort_np[0]),))
            sort_index_motion = np.argsort(-np.abs(motion_vertex))

        return motion_vertex, sort_index_motion, sorted_adj_vertices_int

    def update_tensor_layer(self, first_layer_skip_gradient, forward_input_backprop_all_data, layer, layer_name,
                            motion_last_layer):
        learning_rate_tensor = layer.optimizer_hybrid.param_groups[-1]['lr']    #self.learning_rate / 30
        motion_last_layer_next = []
        # Read tensor output of the layer and put into one ndimensional tensor.
        tensor_hidden_np = []
        forward_input_to_tensor = []
        for iter_idx, tensor_data_iter in enumerate(forward_input_backprop_all_data[layer_name]):
            tensor_hidden_np.append(tensor_data_iter[0][1])
            forward_input_to_tensor.append(tensor_data_iter[0][0])
        tensor_hidden = torch.squeeze(torch.stack((tensor_hidden_np)))
        tensor_input = torch.squeeze(torch.stack((forward_input_to_tensor)))
        tensor_input.retain_grad()  # Keep the gradient for the backward onto sort layers
        # Convert the motion (gradient) previous layers into tensor for tensor update
        motion_grad, accepted_index = self.convert_hybrid_gradient_for_backprop(motion_last_layer)
        if accepted_index.shape[0] > 0:  # There is enough data points for update
            motion_tensor = torch.tensor(motion_grad, requires_grad=True, device=device, dtype=torch.float64)
            tensor_hidden = tensor_hidden[accepted_index]  # The ones without any nan values are used.
            # The expected tensor is the one to reduce the error. Gradient descent.
            tensor_expected = tensor_hidden - learning_rate_tensor * motion_tensor * torch.abs(tensor_hidden)
            loss_hybrid = F.l1_loss(tensor_hidden, tensor_expected)  # Calculate loss
            loss_hybrid.backward()  # Backpropagate error on tensor variables
            if not first_layer_skip_gradient:
                # Compute gradients on input to the tensor layer. (Very non-traditional).
                for iter_grad, forward_input_t in \
                        enumerate(forward_input_to_tensor):

                    forward_input_np = forward_input_t.detach().cpu().numpy()

                    motion_vertex, sort_index_motion, sorted_adj_vertices_int = self.gradient_to_motion_vertex(
                        forward_input_np, forward_input_t, layer)

                    motion_last_layer.append([np.asarray(sorted_adj_vertices_int)[sort_index_motion].tolist(),
                                              motion_vertex[sort_index_motion]])

            motion_last_layer = motion_last_layer_next
            layer.optimizer_hybrid.step()  # Update the layer variables
            layer.scheduler.step()  # Update the learning rate
            layer.optimizer_hybrid.zero_grad()
        return motion_last_layer

    def convert_hybrid_gradient_for_backprop(self, motion_last_layer):

        motion_grad = []
        index_accepted = []
        for motion_idx, motion_data_point in enumerate(motion_last_layer):
            try:
                node_list = list(map(int, motion_data_point[0]))
            except (ValueError, TypeError):
                node_list = [int(mot_data_iter.split('_')[-1]) for mot_data_iter in motion_data_point[0]]

            node_motion = motion_data_point[1]
            if np.sum(np.isnan(node_motion)) == 0:
                indices_sort = sorted(range(len(node_list)), key=node_list.__getitem__)
                node_motion = node_motion[indices_sort]
                motion_grad.append(node_motion)
                index_accepted.append(motion_idx)

        return np.asarray(motion_grad), np.asarray(index_accepted)

    def compute_motion_for_two_adj_lists(self, adj_list_origin, adj_list_comp):
        len_list = len(adj_list_origin)
        motion_indices = np.ones((len_list,))
        motion_indices.fill(np.nan)
        for pos1, adj_vertex in enumerate(adj_list_origin):
            pos2 = adj_list_comp.index(adj_vertex)
            distance_estimated = pos2 - pos1
            motion_indices[pos1] = distance_estimated

        return motion_indices

    def aggregate_motion_last_layer(self, motion_last_layer):
        motion_last_layer_vertex_dict = defaultdict(float)
        count_last_layer_vertex_dict = defaultdict(int)
        for motion_idx, motion_data_point in enumerate(motion_last_layer):
            for vertex_idx, motion_vertex in enumerate(motion_data_point[1]):
                vertex_key = motion_data_point[0][vertex_idx]
                motion_last_layer_vertex_dict[vertex_key] += motion_vertex
                count_last_layer_vertex_dict[vertex_key] += 1

        for vertex_key in motion_last_layer_vertex_dict.keys():
            motion_last_layer_vertex_dict[vertex_key] = motion_last_layer_vertex_dict[vertex_key] / \
                                                        count_last_layer_vertex_dict[vertex_key]

        return motion_last_layer_vertex_dict

    def adapt_learning_rate(self):
        if self.num_of_epochs == 1:
            self.learning_rate_original = self.learning_rate

        if sortnet_config.multistep_lr:
            if self.num_of_epochs > 2 and self.num_of_epochs % (sortnet_config.no_of_iters / 4) == 1:
                self.learning_rate = self.learning_rate * 0.7
        else:
            self.learning_rate = self.learning_rate * 0.993

    def analyze_max_min_dataset(self, data_train):
        max_val = -1000
        min_val = 1000
        for data_iter in data_train:
            val_data = float(data_iter[1])
            if val_data > max_val:
                max_val = val_data
            if val_data < min_val:
                min_val = val_data
        self.max_val_data = max_val
        self.min_val_data = min_val


    def print_save_optimal_model(self):
        if self.optimal_model is None:
            print("No model trained! Optimal model is empty.")
        else:
            print("Optimal Model of {} after training is as follows:".format(self.optimal_model.id))
            for vertex_filter_key in self.optimal_model.vertex_list.keys():
                print("-------------------------------------------------")
                print("-------------------------------------------------")
                vertex_filter = self.optimal_model.vertex_list[vertex_filter_key]
                vertex_filter_name = vertex_filter.id
                vertex_filter_name = vertex_filter_name.split('_')[-1]
                for vertex_key in vertex_filter.graph.vertex_list.keys():
                    vertex = vertex_filter.graph.vertex_list[vertex_key]
                    vertex_name = vertex.id
                    vertex_name = vertex_name.split('_')[-1]
                    adj_list_name = vertex.adjacency_list
                    adj_list_name_stripped = []
                    for node in adj_list_name:
                        node = node.split('_')[-1]
                        adj_list_name_stripped.append(int(node))
                    print("Layer: {}, Vertex: {}, AdjList:{}".format(vertex_filter_name, vertex_name,
                                                                     adj_list_name_stripped))

            if self.file_name is None:
                joblib.dump(self.optimal_model,
                            './models/Graph_{}.pkl'.format(self.optimal_model.id))
            else:
                joblib.dump(self.optimal_model,
                            './models/{}.pkl'.format(self.file_name))

    def split_into_train_validation_dataset(self, data_train, ratio_validation=0.2):
        len_data = len(data_train)
        rand_perm = np.arange(0, len_data, 1)
        split_validation = rand_perm[:int(ratio_validation*rand_perm.shape[0])]
        split_train = rand_perm[int(ratio_validation*rand_perm.shape[0]):]
        data_validation = []
        data_train_reduced = []
        for iter_sel in split_validation:
            data_validation.append(data_train[iter_sel])
        for iter_sel in split_train:
            data_train_reduced.append(data_train[iter_sel])

        return data_train_reduced, data_validation

    def split_into_validation_test_dataset(self, data_validation, ratio_validation=0.5):
        len_data = len(data_validation)
        rand_perm = np.arange(0, len_data, 1) #np.random.permutation(len_data)
        split_validation = rand_perm[:int(ratio_validation*rand_perm.shape[0])]
        split_test = rand_perm[int(ratio_validation*rand_perm.shape[0]):]
        data_validation_2 = []
        data_test = []
        for iter_sel in split_validation:
            data_validation_2.append(data_validation[iter_sel])
        for iter_sel in split_test:
            data_test.append(data_validation[iter_sel])

        return data_validation_2, data_test


class SortFlow_MoE:
    def __init__(self, id_key, adj_list_items, number_of_classes, file_name, sortnet_config):
        self.networks = []
        for iter_moe in range(sortnet_config.moe_no_of_networks):
            self.networks.append(
                SortFlowHybridNetwork(id_key + str(iter_moe), adj_list_items, number_of_classes,
                                      file_name, sortnet_config))

    def train(self, data, sortnet_config):

        if sortnet_config.multiprocessing and len(self.networks) > 1:
            manager = multiprocessing.Manager()
            jobs = []
            return_dict = manager.dict()
            for iter_network, network in enumerate(self.networks):
                if sortnet_config.moe_data_ratio < 1:
                    data_train_reduced, _ = network.split_into_train_validation_dataset(data[0],
                                                                                        1 - sortnet_config.moe_data_ratio)
                    data_iter = [data_train_reduced, data[1]]
                else:
                    data_iter = copy.deepcopy(data)

                sleep(0.1*np.random.random())
                p = multiprocessing.Process(target=network.train,
                                            args=(data_iter, sortnet_config, return_dict, iter_network))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            result_list = return_dict.values()

            errors_update = []
            networks = []
            for result in result_list:
                errors_update.append(result[0])
                networks.append(result[1])

            errors_update = np.mean(np.asarray(errors_update), axis=0).tolist()
            self.networks = networks

            # Evaluate the MoE model on the test data.
            error_test_moe = self.evaluate(data[1], sortnet_config)
            errors_update[3] = error_test_moe
        else:
            errors_update_list = []
            for iter_network, network in enumerate(self.networks):
                errors_update = network.train(data, sortnet_config)
                errors_update_list.append(errors_update)

            errors_update = np.mean(np.asarray(errors_update_list), axis=0).tolist()

            # Evaluate the MoE model on the test data.
            error_test_moe = self.evaluate(data[1], sortnet_config)
            errors_update[3] = error_test_moe

        return errors_update

    def evaluate(self, data, sortnet_config):

        data_gt = []
        for data_point in data:
            data_gt.append(float(data_point[1]))
        data_gt = np.asarray(data_gt)

        if sortnet_config.multiprocessing and len(self.networks) > 1:
            manager = multiprocessing.Manager()
            jobs = []
            return_dict = manager.dict()
            for iter_network, network in enumerate(self.networks):
                sleep(0.1*np.random.random())
                p = multiprocessing.Process(target=network.evaluate,
                                            args=(data, sortnet_config.train_type, sortnet_config.problem,
                                                  return_dict, iter_network))
                jobs.append(p)
                p.start()

            for proc in jobs:
                proc.join()

            result_list = return_dict.values()

            predictions_moe = []
            for result in result_list:
                predictions_moe.append(result[1])

        else:
            predictions_moe = []
            for network in self.networks:
                error_network, prediction_network = network.evaluate(data, sortnet_config.train_type,
                                                                     sortnet_config.problem)
                print("Error Network: ", error_network)
                predictions_moe.append(prediction_network)


        # Combine predictions across networks.
        predictions_moe = np.asarray(predictions_moe)  # shape: (n_networks, n_data)
        if sortnet_config.problem == 'classification':
            # Majority vote: for each data point, pick the class predicted most often
            n_data = predictions_moe.shape[1]
            final_predictions = np.empty(n_data, dtype=predictions_moe.dtype)
            for i in range(n_data):
                votes = predictions_moe[:, i].astype(int)
                final_predictions[i] = np.bincount(votes).argmax()
            error_decision = np.sum(final_predictions != data_gt.astype(int))
        else:
            # Regression: average predictions, absolute error
            final_predictions = np.mean(predictions_moe, axis=0)
            error_decision = np.sum(np.abs(final_predictions - data_gt))

        return error_decision / float(len(data))

