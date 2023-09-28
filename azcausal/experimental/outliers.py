import pandas as pd
import numpy as np
import scipy

from azcausal.util import full_like


def detect_outliers_pct_change(df, alpha=0.95, n_max_iter=100):

    outlier = full_like(df, False)
    ci = None

    for _ in range(n_max_iter):

        forward = df.loc[~outlier].pct_change()
        backward = df.loc[~outlier][::-1].pct_change()[::-1]

        y = pd.DataFrame(dict(forward=forward, backward=backward)).max(axis=1)

        if ci is None:
            ci = scipy.stats.norm.interval(alpha, loc=y.mean(), scale=y.std())
        low, high = ci

        x = (y < low) | (high < y)
        if not np.any(x):
            break

        outlier |= (y < low) | (high < y)

    return outlier


def detect_outliers_rolling(df, window=11, alpha=0.99):
    outliers = []

    n = len(df)

    for k in range(n):
        j = min(n, k + window)
        i = k - (window - (j - k))

        y = df.iloc[k]
        yp = pd.concat([df.iloc[i:k], df.iloc[k + 1:j]])

        (low, high) = scipy.stats.norm.interval(alpha, loc=yp.mean(), scale=yp.std())

        is_outlier = (y < low) or (high < y)

        outliers.append(is_outlier)

    return outliers


