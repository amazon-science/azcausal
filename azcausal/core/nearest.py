import numpy as np


def find_nearest(X: np.ndarray,
                 max_nan_perc: float):
    """
    Find the closest row based on euclidean distance. Supports `nan` values to be allowed a threshold on
    a maximum amount of `nan` values to be considered for a distance to be valid.

    Parameters
    ----------
    X
        A 2-dimensional numpy array.
    max_nan_perc
        The maximum amount of overlapping `nan` values for a distance to be valid.

    Returns
    -------

    """
    n, m = X.shape

    nearest_rows = []
    for k in range(n):
        D = (X[k] - X)
        nan_count = np.sum(np.isnan(D), axis=1) / m

        dd = np.sqrt(np.nanmean(D ** 2, axis=1))

        s = np.argsort(dd)
        feas = [int(e) for e in s if nan_count[e] <= max_nan_perc]
        infeas = sorted([int(e) for e in s if nan_count[e] > max_nan_perc], key=lambda i: max_nan_perc)

        neighbors = np.array(feas + infeas)
        nearest_rows.append(neighbors)

    return nearest_rows


class NearestNeighbors:

    def __init__(self, X, max_nan_perc=0.5) -> None:
        super().__init__()
        self.nearest_rows = find_nearest(X, max_nan_perc)
        self.nearest_cols = find_nearest(X.T, max_nan_perc)

    def get(self, neighbors, k=None, exclude=None, include=None):

        if exclude is not None:
            neighbors = np.array([e for e in neighbors if e not in exclude])

        if include is not None:
            neighbors = np.array([e for e in neighbors if e in include])

        if k is not None:
            neighbors = neighbors[:k]

        return neighbors

    def col(self, j, **kwargs):
        return self.get(self.nearest_cols[j], **kwargs)

    def row(self, i, **kwargs):
        return self.get(self.nearest_rows[i], **kwargs)


def donors(panel, n, method="rmse"):
    Y = panel.Y(pre=True, treat=True).mean(axis=0)
    X = panel.Y(pre=True, contr=True)

    if method == "rmse":
        v = ((X - Y) ** 2).mean(axis=1) ** 0.5

    elif method == "corr":
        mx, my = X.mean(axis=1, keepdims=True), Y.mean()
        v = ((X - mx) * (Y - my)).sum(axis=1) / (((X - mx) ** 2).sum(axis=1) * ((Y - my) ** 2).sum()) ** 0.5
        # vp = [scipy.stats.pearsonr(X[k], Y)[0] for k in range(len(X))]
    else:
        raise Exception("Unknown donor filtering method.")

    rank = np.argsort(v)

    S = panel.units(contr=True)[rank][:n]
    T = panel.units(treat=True)
    units = np.concatenate([T, S])

    return panel[units]
