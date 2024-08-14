import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------------------------------------

def f_ridge(A, x, b, alpha):
    e = np.matmul(A, x) - b
    f = (alpha ** 2) * (x ** 2).sum() + (e ** 2).mean()
    return f


# ---------------------------------------------------------------------------------------------------------
# Regression Solver
# ---------------------------------------------------------------------------------------------------------


class Solver(object):

    def __call__(self,
                 A: np.ndarray,
                 b: np.array,
                 alpha: float,
                 **kwargs) -> dict:
        """

        A solver object to solve a regression problem given
            A * x + alpha * ||x||_2 = b

        Parameters
        ----------
        A
            Left-hand side matrix
        b
            Right-hand side vector
        alpha
            Penalty coefficient

        Returns
        -------
        result
            A dictionary with the attributes: x (the optimum found), f (the function values)

        """
        pass


# ---------------------------------------------------------------------------------------------------------
# Franke Wolfe
# ---------------------------------------------------------------------------------------------------------


class FrankWolfe(Solver):

    def __init__(self,
                 max_iter: int = 10000,
                 tol: float = 1e-5,
                 intercept: bool = True,
                 alpha: float = None,
                 ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.intercept = intercept

    def __call__(self,
                 A: np.ndarray,
                 b: np.ndarray,
                 alpha: float = None,
                 x0: np.ndarray = None,
                 tol: float = None,
                 **kwargs):
        """
        Solving the linear regression where weights sum up to 1 and are all positive.

        Parameters
        ----------
        A
            Left-hand side matrix
        b
            Right-hand side vector
        eta
            The penalty coefficient (potentially modified if noise is provided)
        noise
            An additional coefficient multiplying `eta` and `tol` if provided.
        x0
            The starting solution in the initial iteration (uniform weights are used if not provided)

        Returns
        -------
        result
            A dictionary with the attributes: x (the optimum found), f (the function values)


        """
        tol = tol if tol is not None else self.tol

        # start with the initial solution (uniform weights if not provided)
        if x0 is None:
            _, n = A.shape
            x = np.full(n, 1 / n)
        else:
            x = x0

        # the objective value for the regression problem
        f = f_ridge(A, x, b, alpha)

        iter = 0
        for iter in range(1, self.max_iter + 1):
            fp, f = f, None

            # do a gradient step and update the function value
            x = self._step(A, x, b, alpha)
            f = f_ridge(A, x, b, alpha)

            # check whether we have terminated yet
            if fp - f <= tol ** 2:
                break

        return dict(x=x, f=f, iter=iter)

    def _step(self, A, x, b, alpha):
        eta = len(A) * (alpha ** 2)

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


# this function sparsifies a vector as specified in the synthetic difference-in-difference paper.
def func_simple_sparsify(x):
    k = x <= x.max() / 4
    x[k] = 0
    return x / x.sum()


class SparseSolver(Solver):

    def __init__(self,
                 dense: Solver,
                 sparse: Solver,
                 func_sparsify=func_simple_sparsify) -> None:
        """
        For finding the weights a three-step procedure can be applied to enforce a more sparse solution.
            1. Find a solution.
            2. Sparsify the solution by setting small values to zero
            3. Re-optimize the sparse solution.

        Parameters
        ----------
        dense
            The solver object for the first step to find the dense solution.

        sparse
            The solver object for the second step to optimized given the sparse solution.

        func_sparsify
            The function to sparsify a solution.

        """

        super().__init__()
        self.full = dense
        self.sparse = sparse
        self.func_sparsify = func_sparsify

    def __call__(self, A, b, alpha, x0=None, **kwargs):
        res = self.full(A, b, alpha, x0=x0, **kwargs)

        if self.func_sparsify is not None:
            x0 = self.func_sparsify(res["x"])
            res = self.sparse(A, b, alpha, x0=x0, **kwargs)

        return res


# ---------------------------------------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------------------------------------


def default_solver():
    return SparseSolver(dense=FrankWolfe(max_iter=100, tol=1e-05, intercept=True),
                        sparse=FrankWolfe(max_iter=10000, tol=1e-05, intercept=True),
                        func_sparsify=func_simple_sparsify)


def solve_grad(data: pd.DataFrame,
               contr: np.ndarray,
               treat: np.ndarray,
               alpha: float = None,
               solver=default_solver(),
               **kwargs
               ):
    C = data.loc[:, contr]
    A = C.values

    T = data.loc[:, treat]
    b = T.mean(axis=1).values

    w = solver(A, b, alpha=alpha, **kwargs)['x']
    w /= w.sum()

    return pd.Series(data=w, index=C.columns)
