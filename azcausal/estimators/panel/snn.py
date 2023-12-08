import sys

import numpy as np
import pandas as pd

from azcausal.core.panel import Panel
from azcausal.core.donors import NearestNeighbors
from azcausal.core.estimator import Estimator, results_from_outcome

try:
    import networkx as nx
    from networkx.algorithms.clique import find_cliques
except:
    raise "SNN requires additional libraries. Please run pip install azcausal[snn]"


# Implementation based on https://github.com/deshen24/syntheticNN
class SNN(Estimator):

    def __init__(
            self,
            n_neighbors=1,
            weights='uniform',
            random_splits=False,
            cluster_size=None,
            max_rank=None,
            spectral_t=None,
            linear_span_eps=0.1,
            subspace_eps=0.1,
            min_value=None,
            max_value=None):
        """

        This estimator is based on the following paper:

        Agarwal, Anish, et al.
        "Causal matrix completion."
        The Thirty Sixth Annual Conference on Learning Theory. PMLR, 2023.

        Python Code: https://github.com/deshen24/syntheticNN

        It in addition to provide better scaling allows to pass a `cluster_size` to limit the amount of neighbors
        when solving the clique problem and doing PCA.


        Parameters
        ----------
        n_neighbors
            Number of synthetic neighbors to construct

        weights
            Weight function used in prediction. Possible values:
                (a) 'uniform': each synthetic neighbor is weighted equally
                (b) 'distance': weigh points inversely with distance (as per train error)

        random_splits
            Randomize donors prior to splitting

        max_rank
            Perform truncated SVD on training data with this value as its rank

        spectral_t
            Perform truncated SVD on training data with (100*thresh)% of spectral energy retained.
            If omitted, then the default value is chosen via Donoho & Gavish '14 paper.

        linear_span_eps
            If the (normalized) train error is greater than (100*linear_span_eps)%,
            then the missing pair fails the linear span test.

        subspace_eps
            If the test vector (used for predictions) does not lie within (100*subspace_eps)% of
            the span covered by the training vectors (used to build the model),
            then the missing pair fails the subspace inclusion test.

        min_value
            Minimum possible imputed value

        max_value
            Maximum possible imputed value

        """
        super().__init__()

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.random_splits = random_splits
        self.cluster_size = cluster_size
        self.max_rank = max_rank
        self.spectral_t = spectral_t
        self.linear_span_eps = linear_span_eps
        self.subspace_eps = subspace_eps
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, panel: Panel, **kwargs):
        outcome, intervention = panel.outcome, panel.intervention

        X = outcome.values.copy()
        X[intervention != 0] = np.nan

        # the size of the cluster to consider
        clusters = NearestNeighbors(X) if self.cluster_size is not None else None

        Xp = self.impute(X, clusters=clusters)
        pred_outcome = pd.DataFrame(Xp, index=outcome.index, columns=outcome.columns)

        result = results_from_outcome(outcome, pred_outcome, intervention)
        result.data = panel
        result.estimator = self

        return result

    # complete missing entries in matrix
    def impute(self, X, cov=None, indices=None, clusters=None):
        # get missing entries to impute
        if indices is None:
            indices = np.argwhere(np.isnan(X))

        # initialize the imputed matrix by copy
        Xp = X.copy()

        # complete missing entries
        for i, j in indices:

            # predict missing entry
            pred, feasible = self._predict(X, index=(i, j), cov=cov, clusters=clusters)

            # store in imputed matrices
            if feasible:
                Xp[i, j] = pred

        return Xp

    # combine predictions from all synthetic neighbors
    def _predict(self, X, index, cov=None, clusters=None):

        # find anchor rows and cols
        anchor_rows, anchor_cols = self._find_anchors(X, index=index, clusters=clusters)
        if not anchor_rows.size:
            return np.nan, False

        else:
            if self.random_splits:
                anchor_rows = np.random.permutation(anchor_rows)

            anchor_rows_splits = list(self._split(anchor_rows, k=self.n_neighbors))
            pred = np.zeros(self.n_neighbors)
            feasible = np.zeros(self.n_neighbors)
            w = np.zeros(self.n_neighbors)

            # iterate through all row splits
            for (k, anchor_rows_k) in enumerate(anchor_rows_splits):
                (pred[k], feasible[k], w[k]) = self._synth_neighbor(X,
                                                                    index=index,
                                                                    anchor_rows=anchor_rows_k,
                                                                    anchor_cols=anchor_cols,
                                                                    cov=cov)
            w /= np.sum(w)
            pred = np.average(pred, weights=w)
            feasible = all(feasible)

        return pred, feasible

    # construct the k-th synthetic neighbor
    def _synth_neighbor(self, X, index, anchor_rows, anchor_cols, cov=None):
        missing_row, missing_col = index
        y1 = X[missing_row, anchor_cols]
        X1 = X[anchor_rows][:, anchor_cols]
        X2 = X[anchor_rows, missing_col]

        # covariance
        if cov is not None:
            y1_cov = np.hstack([y1, cov[missing_row]])
            X1_cov = np.hstack([X1, cov[anchor_rows]])
        else:
            y1_cov = y1
            X1_cov = X1

        # learn k-th synthetic neighbor
        (beta, _, s_rank, v_rank) = self._pcr(X1_cov.T, y1_cov)

        # prediction
        pred = self._clip(X2 @ beta)

        # diagnostics
        train_error = self._train_error(X1.T, y1, beta)
        subspace_inclusion_stat = self._subspace_inclusion(v_rank, X2)
        feasible = self._is_feasible(train_error, subspace_inclusion_stat)

        # assign weight of k-th synthetic neighbor
        if self.weights == 'uniform':
            weight = 1
        elif self.weights == 'distance':
            d = train_error + subspace_inclusion_stat
            weight = 1 / d if d > 0 else sys.float_info.max
        else:
            raise Exception("Weights must either be 'uniform' or 'distance'.")

        return pred, feasible, weight

    # split array arr into k subgroups of roughly equal size
    def _split(self, arr, k):
        (m, n) = divmod(len(arr), k)
        return (arr[i * m + min(i, n): (i + 1) * m + min(i + 1, n)] for i in range(k))

    # find model learning sub-matrix by reducing to max bi-clique problem
    def _find_anchors(self, X, index, clusters=None):
        missing_row, missing_col = index

        obs_rows = np.argwhere(~np.isnan(X[:, missing_col])).flatten()
        obs_cols = np.argwhere(~np.isnan(X[missing_row, :])).flatten()

        # if we consider for each element a cluster of rows and cols get the nearest neighbors
        if clusters:
            obs_rows = clusters.row(missing_row, include=obs_rows, k=self.cluster_size)
            obs_cols = clusters.col(missing_col, include=obs_cols, k=self.cluster_size)

        # create bipartite incidence matrix
        B = X[obs_rows][:, obs_cols]

        # check if fully connected already
        if not np.any(np.isnan(B)):
            return obs_rows, obs_cols

        B[np.isnan(B)] = 0

        # bipartite graph
        (n_rows, n_cols) = B.shape
        A = np.block([[np.ones((n_rows, n_rows)), B], [B.T, np.ones((n_cols, n_cols))]])
        G = nx.from_numpy_array(A)

        # find max clique that yields the most square (nxn) matrix
        cliques = list(find_cliques(G))
        d_min = 0
        max_clique_rows_idx = False
        max_clique_cols_idx = False
        for clique in cliques:
            clique = np.sort(clique)
            clique_rows_idx = clique[clique < n_rows]
            clique_cols_idx = clique[clique >= n_rows] - n_rows
            d = min(len(clique_rows_idx), len(clique_cols_idx))
            if d > d_min:
                d_min = d
                max_clique_rows_idx = clique_rows_idx
                max_clique_cols_idx = clique_cols_idx

        # determine model learning rows & cols
        anchor_rows = obs_rows[max_clique_rows_idx]
        anchor_cols = obs_cols[max_clique_cols_idx]

        return anchor_rows, anchor_cols

    # retain all singular values that compose at least (100*self.spectral_t)% spectral energy
    def _spectral_rank(self, s):
        if self.spectral_t == 1.0:
            rank = len(s)
        else:
            total_energy = (s ** 2).cumsum() / (s ** 2).sum()
            rank = int((total_energy > self.spectral_t).argmax() + 1)
        return rank

    # retain all singular values above optimal threshold as per Donoho & Gavish '14: https://arxiv.org/pdf/1305.5870.pdf
    def _universal_rank(self, s, ratio):
        omega = 0.56 * ratio ** 3 - 0.95 * ratio ** 2 + 1.43 + 1.82 * ratio
        t = omega * np.median(s)
        rank = max(len(s[s > t]), 1)
        return rank

    # principal component regression (PCR)
    def _pcr(self, X, y):
        (u, s, v) = np.linalg.svd(X, full_matrices=False)
        if self.max_rank is not None:
            rank = self.max_rank
        elif self.spectral_t is not None:
            rank = self._spectral_rank(s)
        else:
            (m, n) = X.shape
            rank = self._universal_rank(s, ratio=m / n)
        s_rank = s[:rank]
        u_rank = u[:, :rank]
        v_rank = v[:rank, :]
        beta = ((v_rank.T / s_rank) @ u_rank.T) @ y
        return beta, u_rank, s_rank, v_rank

    # clip values to fall within range [min_value, max_value]
    def _clip(self, x):
        if self.min_value is not None:
            x = self.min_value if x < self.min_value else x
        if self.max_value is not None:
            x = self.max_value if x > self.max_value else x
        return x

    # compute (normalized) training error
    def _train_error(self, X, y, beta):
        y_pred = X @ beta
        delta = np.linalg.norm(y_pred - y)
        ratio = delta / np.linalg.norm(y)
        return ratio ** 2

    # compute subspace inclusion statistic
    def _subspace_inclusion(self, V1, X2):
        delta = (np.eye(V1.shape[1]) - (V1.T @ V1)) @ X2
        ratio = np.linalg.norm(delta) / (np.linalg.norm(X2) + 1e-64)
        return ratio ** 2

    # check feasibility of prediction. True iff linear span + subspace inclusion tests both pass
    def _is_feasible(self, train_error, subspace_inclusion_stat):

        # linear span test
        ls_feasible = train_error <= self.linear_span_eps
        # subspace test
        s_feasible = subspace_inclusion_stat <= self.subspace_eps

        return ls_feasible and s_feasible
