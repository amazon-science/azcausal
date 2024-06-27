import cvxpy as cp
import numpy as np
import pandas as pd

from azcausal.solvers.solve_exception import SolverException

DEFAULT_SOLVERS = (cp.ECOS, cp.OSQP)


def solve_cvxpy(data: pd.DataFrame,
                contr: np.ndarray,
                treat: np.ndarray,
                alpha: float = None,
                max_weight=None,
                eps: float = 1e-12,
                kwargs=None,
                solvers=DEFAULT_SOLVERS,
                ):

    kwargs = dict(kwargs) if kwargs is not None else dict()

    # set defaults to kwargs
    defaults = dict(verbose=False)
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    if 'solver' in kwargs:
        solvers = kwargs.pop('solver')

    C = data.loc[:, contr]
    A = C.values
    m, n = A.shape

    T = data.loc[:, treat]
    b = T.mean(axis=1).values

    x = cp.Variable(n)

    # OBJECTIVE
    # xx = np.ones(n) / n
    # yy = (((A @ xx - b)**2).sum() / m) + ((alpha ** 2) * (xx ** 2).sum())

    obj = cp.sum_squares(A @ x - b) / m
    if alpha is not None:
        obj += (alpha ** 2) * cp.sum_squares(x)

    # constr = [x >= 0.0, cp.sum(x) == 1]

    ieq_eps = 1e-6
    constr = [x >= 0.0, cp.sum(x) <= (1 + ieq_eps), cp.sum(x) >= (1 - ieq_eps)]

    if max_weight is not None:
        constr.append(x <= max_weight)

    success = False
    exceptions = dict()
    for solver in solvers:
        try:
            problem = cp.Problem(cp.Minimize(obj), constr)
            problem.solve(solver=solver, max_iters=10000, **kwargs)
            success = (x.value is not None)
        except Exception as ex:
            print(solver, ex)
            exceptions[solver] = ex

        if success:
            break

    if not success:
        raise SolverException("Error: All solvers in FSDID failed to find weights.", exceptions=exceptions)

    w = np.array(x.value)
    if eps is not None:
        w = np.where(w >= eps, w, 0.0)
        w /= w.sum()

    return pd.Series(data=w, index=C.columns)
