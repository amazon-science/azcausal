import numpy as np


def donors(pnl, n, method="rmse"):
    Y = pnl.Y(pre=True, treat=True).mean(axis=0)
    X = pnl.Y(pre=True, contr=True)

    if method == "rmse":
        v = ((X - Y) ** 2).mean(axis=1) ** 0.5

    elif method == "corr":
        mx, my = X.mean(axis=1, keepdims=True), Y.mean()
        v = ((X - mx) * (Y - my)).sum(axis=1) / (((X - mx) ** 2).sum(axis=1) * ((Y - my) ** 2).sum()) ** 0.5
        # vp = [scipy.stats.pearsonr(X[k], Y)[0] for k in range(len(X))]
    else:
        raise Exception("Unknown donor filtering method.")

    rank = np.argsort(v)

    S = pnl.units(contr=True)[rank][:n]
    T = pnl.units(treat=True)
    units = np.concatenate([T, S])

    return pnl[units]
