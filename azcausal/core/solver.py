import numpy as np


def f_ridge(A, x, b, zeta):
    e = np.matmul(A, x) - b
    f = (zeta ** 2) * (x ** 2).sum() + (e ** 2).mean()
    return f


class Solver(object):

    def __call__(self, A, b, zeta, **kwargs):
        pass


class Sampling:

    def __init__(self, uniform=True, nnls=True, n_random=100) -> None:
        super().__init__()
        self.uniform = uniform
        self.nnls = nnls
        self.n_random = n_random

    def __call__(self, A, b, zeta, f):
        _, n = A.shape

        xs = [np.full(n, 1 / n)]

        if self.nnls:
            from scipy.optimize import nnls
            xp, _ = nnls(A, b)
            xp /= xp.sum()
            xs += [xp]

        if self.n_random is not None and self.n_random > 0:
            xs += [np.random.dirichlet(np.ones(n)) for _ in range(self.n_random)]

        fs = np.array([f(A, x, b, zeta) for x in xs])
        k = fs.argmin()

        xopt = xs[k]
        fopt = fs[k]

        return xopt, fopt


class SimpleRegression(Solver):

    def __call__(self, A, b, zeta, **kwargs):
        from scipy.optimize import nnls
        x, f = nnls(A, b)
        return dict(x=x, f=f)


class FrankWolfe(Solver):

    def __init__(self, max_iter=10000, tol=1e-5, intercept=True, alpha=None, sampling=Sampling()) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.intercept = intercept
        self.sampling = sampling

    def __call__(self, A, b, zeta, noise=None, x0=None, **kwargs):
        tol = self.tol

        if self.intercept:
            A = A - A.mean(axis=0)
            b = b - b.mean()

        if noise is not None:
            zeta = noise * zeta
            tol = noise * tol

        if x0 is None:
            x, f = self.sampling(A, b, zeta, f_ridge)
        else:
            x = x0
            f = f_ridge(A, x, b, zeta)

        for iter in range(1, self.max_iter + 1):

            fp, f = f, None
            x = self._step(A, x, b, zeta)

            f = f_ridge(A, x, b, zeta)
            if fp - f <= tol ** 2:
                break

        return dict(x=x, f=f, iter=iter)

    def _step(self, A, x, b, zeta):
        alpha = self.alpha

        eta = len(A) * (zeta ** 2)

        Ax = np.matmul(A, x)
        half_grad = (Ax - b).T @ A + eta * x
        i = half_grad.argmin()

        if alpha is not None:
            x = x * (1 - alpha)
            x[i] = x[i] + alpha
            return x
        else:
            dx = -x
            dx[i] = 1 - x[i]
            if (dx == 0).all():
                return x

            d_err = A[:, i] - Ax
            step = (-1 * half_grad @ dx) / ((d_err ** 2).sum() + eta * (dx ** 2).sum())

            ustep = min(1.0, max(0.0, step))

            return x + ustep * dx


def func_simple_sparsify(x):
    k = x <= x.max() / 4
    x[k] = 0
    return x / x.sum()


class SparseSolver(Solver):

    def __init__(self, full, sparse, func_make_sparse=func_simple_sparsify) -> None:
        super().__init__()
        self.full = full
        self.sparse = sparse
        self.func_make_sparse = func_make_sparse

    def __call__(self, A, b, zeta, x0=None, **kwargs):
        res = self.full(A, b, zeta, x0=x0, **kwargs)

        if self.func_make_sparse is None:
            return res
        else:
            x0 = self.func_make_sparse(res["x"])
            return self.sparse(A, b, zeta, x0=x0, **kwargs)
