import cvxopt
import numpy as np
import pandas as pd

from azcausal.solvers.solve_exception import SolverException


def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    return cvxopt.solvers.qp(*args, options={'show_progress': False})


def solve_quad(data: pd.DataFrame,
               contr: np.ndarray,
               treat: np.ndarray,
               alpha: float = None,
               eps: float = 1e-12,
               **kwargs
               ):

    C = data.loc[:, contr]
    A = C.values
    m, n = A.shape

    T = data.loc[:, treat]
    b = T.mean(axis=1).values

    # first divide by m (as done in the objective function too)
    AA = A / m
    bb = b / m

    # create the quadratic optimization matrix
    P = np.dot(AA.T, AA)
    q = -np.dot(AA.T, bb)

    # add the regulatrization
    if alpha is not None:
        R = np.identity(n) * (alpha ** 2) / m
        P = P + R

    # weights greater than zero constraints
    G = -1 * np.identity(n)
    # h = np.zeros(n)
    h = (1/1000) * (1/n) * np.ones(n)

    # weight's sum equals one constraint
    A = np.ones((1, n))
    b = np.ones((1, 1))

    try:
        sol = cvxopt_solve_qp(P, q, G=G, h=h, A=A, b=b)
        w = np.array(sol['x'])[:, 0]
    except Exception as ex:
        raise SolverException(exceptions=[ex])

    if eps is not None:
        w = np.where(w >= eps, w, 0.0)
    w /= w.sum()

    return pd.Series(data=w, index=C.columns)
