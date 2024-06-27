import numpy as np
import pandas as pd
import scipy.optimize as opt


def solve_scipy(data: pd.DataFrame,
                contr: np.ndarray,
                treat: np.ndarray,
                alpha: float = None,
                eps: float = 1e-12,
                eq_eps=1e-6,
                **kwargs
                ):
    C = data.loc[:, contr]
    A = C.values
    m, n = A.shape

    T = data.loc[:, treat]
    b = T.mean(axis=1).values

    def obj(x):
        obj = np.square(A @ x - b).sum() / m
        if alpha is not None:
            obj += (alpha ** 2) * np.square(x).sum()
        return obj

    bounds = [(0, None)] * n

    # defined as equality constraint
    # constraints = ({'type': 'eq', 'fun': lambda x: x.sum() - 1})

    # defined as two inequality constraints
    constraints = ({'type': 'ineq', 'fun': lambda x: 1 + eq_eps + x.sum()},  # sum(x) <= (1 + eps) -> sum(x) - 1 - eps <= 0 -> 1 + eps - sum(x) >= 0
                   {'type': 'ineq', 'fun': lambda x: x.sum() - 1 + eq_eps})  # sum(x) >= (1 - eps) -> sum(x) - 1 + eps

    # METHODS: SLSQP, trust-constr
    x0 = np.ones(n) / n
    res = opt.minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    w = res.x
    if eps is not None:
        w = np.where(w >= eps, w, 0.0)
    w /= w.sum()

    return pd.Series(data=w, index=C.columns)
