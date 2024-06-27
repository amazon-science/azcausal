import cvxpy as cp
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------------------------------------


class Solver(object):

    def solve(self,
              A: np.ndarray,
              b: np.ndarray,
              alpha: float = None,
              **kwargs) -> dict:
        pass


# ---------------------------------------------------------------------------------------------------------
# Direct Solver (cxpy)
# ---------------------------------------------------------------------------------------------------------


class CVXPY(Solver):

    def __init__(self, solver=cp.ECOS, eps: float = 1e-12) -> None:
        super().__init__()
        self.solver = solver
        self.eps = eps

    def solve(self,
              A: np.ndarray,
              b: np.ndarray,
              alpha: float = None,
              labels: list = None,
              **kwargs):

        # OBJECTIVE
        # x = np.ones(n) / n
        # y = (((A @ x - b)**2).sum() / m) + ((alpha ** 2) * (x ** 2).sum())

        m, n = A.shape
        x = cp.Variable(n)

        obj = cp.sum_squares(A @ x - b) / m
        if alpha is not None:
            obj += (alpha ** 2) * cp.sum_squares(x)

        constr = [x >= 0.0, cp.sum(x) == 1]

        try:
            problem = cp.Problem(cp.Minimize(obj), constr)
            problem.solve(solver=self.solver, **kwargs)

            ds = None
            success = False
            if x.value is not None:
                success = True
                x = np.array(x.value)

                if self.eps is not None:
                    x = np.where(x >= self.eps, x, 0.0)
                    x /= x.sum()

                if labels is not None:
                    ds = pd.Series(data=x, index=labels)

            return dict(success=success, ex=None, x=x, ds=ds)

        except Exception as ex:
            return dict(success=False, ex=ex, x=None, ds=None)


# ---------------------------------------------------------------------------------------------------------
# Gradient Solver (FrankWolfe)
# ---------------------------------------------------------------------------------------------------------

def f_ridge(A, x, b, alpha):
    f = (A @ x - b).mean() + alpha * (x ** 2).sum()
    return f


class FrankWolfe(Solver):

    def solve(self,
              A: np.ndarray,
              b: np.ndarray,
              alpha=float,
              tol: bool = 1e-5,
              max_iter=10000,
              x0: np.ndarray = None,
              **kwargs):

        # start with the initial solution (uniform weights if not provided)
        if x0 is None:
            _, n = A.shape
            x = np.full(n, 1 / n)
        else:
            x = x0

        # the objective value for the regression problem
        f = f_ridge(A, x, b, alpha)

        iter = 0
        for iter in range(1, max_iter + 1):
            fp, f = f, None

            # do a gradient step and update the function value
            x = self._step(A, x, b, alpha)
            f = f_ridge(A, x, b, alpha)

            # check whether we have terminated yet
            if fp - f <= tol ** 2:
                break

        return dict(x=x, f=f, iter=iter)

    def _step(self, A, x, b, alpha):
        eta = len(A) * alpha

        Ax = np.matmul(A, x)
        half_grad = (Ax - b).T @ A + eta * x
        i = half_grad.argmin()

        dx = -x
        dx[i] = 1 - x[i]
        if (dx == 0).all():
            return x

        d_err = A[:, i] - Ax
        step = (-1 * half_grad @ dx) / ((d_err ** 2).sum() + eta * (dx ** 2).sum() + 1e-32)
        ustep = min(1.0, max(0.0, step))

        return x + ustep * dx
