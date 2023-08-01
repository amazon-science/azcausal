import numpy as np


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
