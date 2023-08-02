import copy
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cachetools import cached
from cachetools.keys import hashkey
from numpy import ndarray, float64
from sklearn.cluster import SpectralBiclustering
from sklearn.utils import check_array
from tensorly.decomposition import parafac

from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.panel import Panel
from azcausal.core.result import Result


def nanmean(df, *args, **kwargs):
    if len(df) == 0:
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(df, *args, **kwargs)


class SNNB(Estimator):
    def __init__(
            self,
            max_rank=None,
            spectral_t=None,
            linear_span_eps=0.1,
            subspace_eps=0.1,
            min_value=None,
            max_value=None,
            min_singular_value: float = 1e-7,
            min_row_sparsity=0.3,
            min_col_sparsity=0.3,
            min_cluster_sparsity=0.3,
            min_cluster_size=9,
            n_cluster_runs=3,
            min_num_clusters=5,
            n_estimates=3,
            **kwargs
    ):
        """
        Parameters
        ----------

        max_rank : int
            Perform truncated SVD on training data with this value as its rank. This overrides the `spectral_t` option

        spectral_t : float
            Perform truncated SVD on training data with (100*thresh)% of spectral energy retained.
            If omitted, then the default value is chosen via Donoho & Gavish '14 paper.

        linear_span_eps : float
            (for diagnostics) If the (normalized) train error is greater than (100*linear_span_eps)%,
            then the missing pair fails the linear span test.

        subspace_eps : float
            (for diagnostics) If the test vector (used for predictions) does not lie within (100*subspace_eps)% of
            the span covered by the training vectors (used to build the model),
            then the missing pair fails the subspace inclusion test.

        min_value : float
            Minimum possible imputed value

        max_value : float
            Maximum possible imputed value

        min_singular_value : float
            Minimum singular value to include when learning the linear model (for numerical stability)

        min_row_sparsity : float
            Minimum sparsity level (should b in (0.1.0]) for rows to be included in the cluster

        min_col_sparsity : float
            Minimum sparsity level (should b in (0.1.0]) for cols to be included in the cluster

        min_cluster_sparsity : float
            Minimum sparsity level (should b in (0.1.0]) for cluster to be included

        min_cluster_size : int
            Minimum size of the cluster

        n_cluster_runs : int
            Number of bi-clustering runs done on the mask matrix

        min_num_clusters : int
            Minimum number of cluster per row and per columns

        n_estimates : int
            Maximum number of estimates used to estimate missing values

        """
        super().__init__(**kwargs)
        self.min_singular_value = min_singular_value
        self.max_rank = max_rank
        self.spectral_t = spectral_t
        self.linear_span_eps = linear_span_eps
        self.subspace_eps = subspace_eps
        self.min_value = min_value
        self.max_value = max_value
        self.min_col_sparsity: float = min_col_sparsity
        self.min_row_sparsity: float = min_row_sparsity
        self.min_cluster_sparsity: float = min_cluster_sparsity
        self.clusters: dict = None
        self.clusters_row_matrix: Optional[ndarray] = None
        self.clusters_col_matrix: Optional[ndarray] = None
        self.min_cluster_size: int = min_cluster_size
        self.n_cluster_runs: int = n_cluster_runs
        self.min_num_clusters: int = min_num_clusters
        self.n_estimates: int = n_estimates
        self.clusters_hashes: set = None

    def refit(self, result):

        def f(panel):
            return copy.deepcopy(result.estimator).fit(panel)

        return f

    def error(self, result, method, **kwargs):
        f_estimate = self.refit(result)
        return method.run(result, f_estimate=f_estimate, **kwargs)

    def fit(self, panel) -> Result:

        assert panel.n_treatments() == 1, "Currently only a single type of intervention is supported. " \
                                          "Please make sure your panel contains only 0s and 1s. "

        # create a tensor from the panel data
        tensor = self._get_tensor(panel)

        filled, feas = self._fit(tensor)
        feas[~np.isnan(tensor)] = 1

        # get the matrices from that have been fitted
        X_imputed = filled[..., 0]
        iv = filled[..., 1]

        # calculate the treatment effect
        te = (iv - X_imputed)

        att = np.nanmean(te[panel.w])
        ite = pd.DataFrame(dict(att=nanmean(te[panel.w], axis=1)), index=panel.units(treat=True))

        df = pd.DataFrame(X_imputed.T, index=panel.outcome.index, columns=panel.outcome.columns)
        imputed = Panel(df, panel.intervention)

        by_time = pd.DataFrame({
            "time": panel.time(),
            "C": nanmean(panel.Y(contr=True), axis=0),
            "T": nanmean(panel.Y(treat=True), axis=0),
            "CF": nanmean(imputed.Y(treat=True), axis=0),
            "att": nanmean(te[panel.w], axis=0),
            "W": panel.wp.astype(int)
        }).set_index("time")

        T = panel.outcome.values[panel.intervention == 1].mean()
        att = Effect(att, observed=T, multiplier=panel.n_interventions(), by_time=by_time, by_unit=ite,
                     data=dict(imputed=imputed), name="ATT")
        return Result(dict(att=att), data=panel, estimator=self)

    def plot(self, result, title=None, CF=True, C=True, show=True):

        data, panel = result.effect["by_time"], result.panel
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.plot(data.index, data["T"], label="T", color="blue")

        if C:
            ax.plot(data.index, data["C"], label="C", color="red")

        if CF:
            ax.plot(data.index, data["CF"], "--", color="blue", alpha=0.5)

        ax.axvline(panel.earliest_start, color="black", alpha=0.3)
        ax.set_title(title)

        if show:
            plt.legend()
            plt.tight_layout()
            fig.show()

        return fig

    def _fit(self, tensor):
        tensor_filled, feasible_tensor = self._fit_transform(tensor)
        return tensor_filled, feasible_tensor

    def _fit_transform(self, X, test_set=None):

        """
        complete missing entries in matrix
        """
        # tensor to matrix
        N, T, I = X.shape
        X = X.reshape([N, I * T])

        self.matrix = X
        self.mask = (~np.isnan(X)).astype(int)

        # get clusters:
        self.min_num_clusters = int(np.sqrt(min(self.mask.shape)))
        self._set_clusters()

        # set cluster matrices for finding best clusters
        self.clusters_row_matrix, self.clusters_col_matrix = self._get_clusters_matrices()
        filled_matrix, feasible_matrix = self._snn_fit_transform(X, test_set)

        # reshape matrix into tensor
        tensor = filled_matrix.reshape(N, T, I)
        feasible_tensor = feasible_matrix.reshape(N, T, I)

        # clear cache
        self._map_missing_value.cache.clear()
        self._get_beta_from_factors.cache.clear()

        return tensor, feasible_tensor

    def _snn_fit_transform(self, X: ndarray, test_set: Optional[ndarray] = None) -> tuple:

        missing_set = test_set
        if missing_set is None:
            missing_set = np.argwhere(np.isnan(X))
        num_missing = len(missing_set)

        X, X_imputed = self._initialize(X, missing_set)
        feasible_matrix = np.full(X.shape, np.nan)

        # complete missing entries
        for (i, missing_pair) in enumerate(missing_set):

            if self.verbose:
                print("[SNN] iteration {} of {}".format(i + 1, num_missing))

            # predict missing entry
            (pred, feasible) = self._predict(X, missing_pair=missing_pair)
            # store in imputed matrices
            (missing_row, missing_col) = missing_pair
            X_imputed[missing_row, missing_col] = pred
            feasible_matrix[missing_row, missing_col] = feasible

        if self.verbose:
            print("[SNN] complete")
        return X_imputed, feasible_matrix

    def _filter_cluster(self, cluster_mask, rows, cols):

        # if very sparse, drop
        if cluster_mask.mean() < 0.2:
            return
        # remove columns and then rows that are mostly nans
        col_means = cluster_mask.mean(0)
        retained_cols = col_means > self.min_col_sparsity
        retained_cols_index = cols[retained_cols]
        cluster_mask = cluster_mask[:, retained_cols]

        # if no columns, return
        if cluster_mask.shape[1] == 0:
            return

        row_means = cluster_mask.mean(1)
        retained_rows = row_means > self.min_row_sparsity
        retained_rows_index = rows[retained_rows]
        cluster_mask = cluster_mask[retained_rows, :]

        # if now rows, return
        if cluster_mask.shape[0] == 0:
            return

        # check sparsity, if more than min_cluster_sparsity drop
        if cluster_mask.mean() < self.min_cluster_sparsity:
            return
        return retained_rows_index, retained_cols_index

    def _filter_and_construct_cluster(self, cluster, X, mask, rows, cols):
        filtered_cluster_rows_and_cols = self._filter_cluster(cluster, rows, cols)
        if filtered_cluster_rows_and_cols is None:
            return
        rows, cols = filtered_cluster_rows_and_cols
        cluster = {
            "rows": rows,
            "cols": cols,
            "data": X[rows, :][:, cols],
            "sparsity": 1 - mask[rows, :][:, cols].mean(),
        }
        return cluster

    def _set_clusters(self):
        self.clusters = {}
        index = 0
        self.clusters_hashes = set()

        for idx in range(self.n_cluster_runs):
            min_shape = min(self.mask.shape)
            model = SpectralBiclustering(
                n_clusters=(
                    min(self.min_num_clusters + 2 * idx, min_shape),
                    min(self.min_num_clusters + 2 * idx, min_shape),
                ),
                n_best=6 + idx,
                n_components=7 + idx,
                method="log",
                random_state=self.random_state,
                mini_batch=True,
            )
            model.fit(self.mask)

            # process clusters
            row_clusters = np.unique(model.row_labels_)
            col_clusters = np.unique(model.column_labels_)
            # loop over clusters
            for row_clus in row_clusters:
                for col_clus in col_clusters:
                    rows = np.where(model.row_labels_ == row_clus)[0]
                    cols = np.where(model.column_labels_ == col_clus)[0]
                    set_ = set(rows)
                    set_.update(cols + self.mask.shape[0])
                    hash_ = hash(frozenset(set_))
                    if hash_ in self.clusters_hashes:
                        continue
                    else:
                        self.clusters_hashes.update({hash_})
                    cluster = np.array(self.mask[rows, :][:, cols])
                    cluster = self._filter_and_construct_cluster(
                        cluster, self.matrix, self.mask, rows, cols
                    )

                    if cluster is None:
                        continue
                    self.clusters[index] = cluster
                    index += 1

        if self.verbose:
            print(f"Generated {index} clusters")

    def _get_clusters_matrices(self):
        clusters_row_matrix = np.zeros([len(self.clusters), self.matrix.shape[0]])
        clusters_col_matrix = np.zeros([len(self.clusters), self.matrix.shape[1]])
        for i, cl in enumerate(self.clusters.values()):
            clusters_col_matrix[i, cl["cols"]] = 1
            clusters_row_matrix[i, cl["rows"]] = 1
        return clusters_row_matrix, clusters_col_matrix

    @cached(
        cache=dict(),  # type: ignore
        key=lambda self, X, obs_rows, obs_cols: hashkey(obs_rows, obs_cols),
    )
    def _map_missing_value(self, X, obs_rows, obs_cols):
        _obs_rows = np.array(list(obs_rows), dtype=int)
        _obs_cols = np.array(list(obs_cols), dtype=int)

        # construct row vector
        obs_rows_vector = np.zeros(X.shape[0])
        obs_rows_vector[_obs_rows] = 1
        obs_cols_vector = np.zeros(X.shape[1])
        obs_cols_vector[_obs_cols] = 1

        # multiply by cluster_row_matrix and clusters_col_matrix
        clusters_row_matching = self.clusters_row_matrix @ obs_rows_vector
        clusters_col_matching = self.clusters_col_matrix @ obs_cols_vector

        # select based on max
        clusters_sizes = clusters_row_matching * clusters_col_matching
        viable_clusters = (clusters_sizes >= self.min_cluster_size).sum()
        if viable_clusters == 0:
            return None

        selected_cluster = np.argsort(clusters_sizes)[-self.n_estimates:]
        selected_cluster = [
            cluster
            for cluster in selected_cluster
            if clusters_sizes[cluster] >= self.min_cluster_size
        ]
        return selected_cluster, obs_rows_vector, obs_cols_vector

    def _predict(self, X, missing_pair):

        i, j = missing_pair
        obs_rows = np.argwhere(~np.isnan(X[:, j])).flatten()
        obs_cols = np.argwhere(~np.isnan(X[i, :])).flatten()
        _obs_rows = frozenset(obs_rows)
        _obs_cols = frozenset(obs_cols)

        if not obs_rows.size or not obs_cols.size:
            return np.nan, False

        estimates = np.full(self.n_estimates, np.nan)
        feasibles = np.full(self.n_estimates, np.nan)
        mapped_clusters = self._map_missing_value(
            X,
            _obs_rows,
            _obs_cols,
        )
        if mapped_clusters is None:
            return np.nan, False
        else:
            selected_clusters, obs_rows_vector, obs_cols_vector = mapped_clusters
        counter = 0

        if len(selected_clusters) == 0:
            return np.nan, False
        for clus in selected_clusters:

            # get minimal anchor rows and cols
            rows_cluster = self.clusters_row_matrix[clus, :]
            cols_cluster = self.clusters_col_matrix[clus, :]
            anchor_rows = np.where(obs_rows_vector * rows_cluster)[0]
            anchor_cols = np.where(obs_cols_vector * cols_cluster)[0]

            cluster = np.array(self.mask[anchor_rows, :][:, anchor_cols])
            if not cluster.any():
                continue

            if (
                    cluster.sum(0).min() < 2
                    or cluster.mean(1).sum() < 2
                    or cluster.mean() < 0.3
            ):
                continue

            prediction, feasible = self._synth_neighbor(
                X, missing_pair, anchor_rows, anchor_cols, clus
            )
            estimates[counter] = prediction
            feasibles[counter] = feasible

            counter += 1
        if not counter:
            return np.nan, False
        else:
            return np.nanmean(estimates), bool(np.nanmax(feasibles))

    def _synth_neighbor(self, X, missing_pair, anchor_rows, anchor_cols, cluster_idx):

        # check if factors is already computed
        (missing_row, missing_col) = missing_pair
        cluster = self.clusters[cluster_idx]
        if "u_rank" not in cluster:
            cluster = self._get_factors(X, cluster)

        _anchor_rows = frozenset(anchor_rows)
        _anchor_cols = frozenset(anchor_cols)

        beta, u_rank, train_error = self._get_beta_from_factors(
            X, missing_row, _anchor_rows, _anchor_cols, cluster_idx
        )
        X2 = X[anchor_rows, missing_col]
        # prediction
        pred = X2 @ beta

        # diagnostics
        subspace_inclusion_stat = self._subspace_inclusion(u_rank, X2)
        feasible = self._isfeasible(train_error, subspace_inclusion_stat)
        return pred, feasible

    def _get_factors(self, X, cluster):
        cluster_rows = cluster["rows"]
        cluster_cols = cluster["cols"]
        cluster["rows_dict"] = dict(zip(cluster_rows, np.arange(len(cluster_rows))))
        cluster["cols_dict"] = dict(zip(cluster_cols, np.arange(len(cluster_cols))))
        X1 = X[cluster_rows, :]
        X1 = X1[:, cluster_cols]
        (u_rank, s_rank, v_rank) = self._compute_factors(X1)
        cluster["u_rank"] = u_rank
        cluster["v_rank"] = v_rank
        cluster["s_rank"] = s_rank
        return cluster

    def _compute_factors(self, X):
        X_copy = np.array(X)
        X_copy[np.isnan(X_copy)] = 0
        p = 1 - np.isnan(X_copy).sum() / X_copy.size
        (_, s, _) = np.linalg.svd(X_copy / p, full_matrices=False)
        if self.max_rank is not None:
            rank = self.max_rank
        elif self.spectral_t is not None:
            rank = self._spectral_rank(s)
        else:
            (m, n) = X.shape
            rank = self._universal_rank(s, ratio=m / n)

        rank = min(np.sum(s > self.min_singular_value), rank)

        weights, factors = parafac(
            X_copy,
            rank=rank,
            mask=~np.isnan(X),
            init="random",
            normalize_factors=True,
            random_state=self.random_state
        )

        # this is a quick solution to make sure factors are orthogonal -- only needed for diagnosis
        (u_n, s1, v1) = np.linalg.svd(weights * factors[0] @ factors[1].T, full_matrices=False)
        return u_n, s1, v1.T

    @cached(
        cache=dict(),  # type: ignore
        key=lambda self, X, missing_row, anchor_rows, anchor_cols, cluster_idx: hashkey(
            missing_row, anchor_rows, anchor_cols, cluster_idx
        ),
    )
    def _get_beta_from_factors(
            self, X, missing_row, anchor_rows, anchor_cols, cluster_idx
    ):
        _anchor_rows = np.array(list(anchor_rows), dtype=int)
        _anchor_cols = np.array(list(anchor_cols), dtype=int)

        cluster = self.clusters[cluster_idx]
        u_rank = cluster["u_rank"]
        v_rank = cluster["v_rank"]
        s_rank = np.array(cluster["s_rank"])
        y1 = X[missing_row, _anchor_cols]

        rows = np.vectorize(
            cluster["rows_dict"].get,
        )(_anchor_rows)
        cols = np.vectorize(
            cluster["cols_dict"].get,
        )(_anchor_cols)
        s_rank = s_rank[:]
        u_rank = u_rank[rows, :]
        v_rank = v_rank[cols, :]

        X_als = (u_rank * s_rank) @ v_rank.T
        beta = np.linalg.lstsq(X_als.T, y1, rcond=self.min_singular_value)[0]
        train_error = self._train_error(X_als.T, y1, beta)

        return beta, u_rank.T, train_error

    def _initialize(self, X: ndarray, missing_set: ndarray) -> Tuple[ndarray, ndarray]:
        # check and prepare data
        X = self._prepare_input_data(X, missing_set, 2)
        # initialize
        X_imputed = X.copy()
        self.feasible = np.empty(X.shape)
        self.feasible.fill(np.nan)
        return X, X_imputed

    def _spectral_rank(self, s):
        """
        retain all singular values that compose at least (100*self.spectral_t)% spectral energy
        """
        if self.spectral_t == 1.0:
            rank = len(s)
        else:
            total_energy = (s ** 2).cumsum() / (s ** 2).sum()
            rank = list((total_energy > self.spectral_t)).index(True) + 1
        return rank

    def _universal_rank(self, s: ndarray, ratio: float) -> int:
        """
        retain all singular values above optimal threshold as per Donoho & Gavish '14:
        https://arxiv.org/pdf/1305.5870.pdf
        """
        omega = 0.56 * ratio ** 3 - 0.95 * ratio ** 2 + 1.43 + 1.82 * ratio
        t = omega * np.median(s)
        rank = max(len(s[s > t]), 1)
        return rank

    def _train_error(self, X: ndarray, y: ndarray, beta: ndarray) -> float64:
        """
        compute (normalized) training error
        """
        y_pred = X @ beta
        delta = np.linalg.norm(y_pred - y)

        ratio = delta
        # check if the denominator is zero
        if np.linalg.norm(y) > 0:
            ratio = delta / np.linalg.norm(y)

        return ratio ** 2

    def _subspace_inclusion(self, V1: ndarray, X2: ndarray) -> float64:
        """
        compute subspace inclusion statistic
        """
        delta = (np.eye(V1.shape[1]) - (V1.T @ V1)) @ X2
        ratio = np.linalg.norm(delta) / np.linalg.norm(X2)
        return ratio ** 2

    def _isfeasible(
            self, train_error: float64, subspace_inclusion_stat: float64
    ) -> bool:
        """
        check feasibility of prediction
        True iff linear span + subspace inclusion tests both pass
        """
        # linear span test
        ls_feasible = True if train_error <= self.linear_span_eps else False
        # subspace test
        s_feasible = True if subspace_inclusion_stat <= self.subspace_eps else False
        return ls_feasible and s_feasible

    def _get_tensor(self, panel: Panel) -> ndarray:

        # populate actions dict
        outcome, intervention = panel.outcome.T, panel.intervention.T
        I = len(np.unique(intervention))
        N, T = outcome.shape
        tensor = self._populate_tensor(
            N,
            T,
            I,
            outcome,
            intervention,
        )

        return tensor

    @staticmethod
    def _populate_tensor(N, T, I, metric_df, assignment_matrix):
        # init tensor
        tensor = np.full([N, T, I], np.nan)

        # fill tensor with metric_matrix values in appropriate locations
        metric_matrix = metric_df.values
        for action_idx in range(I):
            unit_time_received_action_idx = assignment_matrix == action_idx
            tensor[unit_time_received_action_idx, action_idx] = metric_matrix[
                unit_time_received_action_idx
            ]
        return tensor

    def _check_input_matrix(self, X: ndarray, missing_mask: ndarray, ndim: int) -> None:
        """
        check to make sure that the input matrix
        and its mask of missing values are valid.
        """
        if len(X.shape) != ndim:
            raise ValueError(
                "expected %dd matrix, got %s array"
                % (
                    ndim,
                    X.shape,
                )
            )
        if not len(missing_mask) > 0:
            warnings.simplefilter("always")
            warnings.warn("input matrix is not missing any values")
        if len(missing_mask) == int(np.prod(X.shape)):
            raise ValueError(
                "input matrix must have some observed (i.e., non-missing) values"
            )

    def _prepare_input_data(
            self, X: ndarray, missing_mask: ndarray, ndim: int
    ) -> ndarray:
        """
        prepare input matrix X. return if valid else terminate
        """
        X = check_array(X, force_all_finite=False, allow_nd=True)
        if (X.dtype != "f") and (X.dtype != "d"):
            X = X.astype(float)
        self._check_input_matrix(X, missing_mask, ndim)
        return X
