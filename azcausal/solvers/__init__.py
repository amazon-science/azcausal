from typing import Callable

from azcausal.solvers.solve_cvxpy import solve_cvxpy
from azcausal.solvers.solve_grad import solve_grad
from azcausal.solvers.solve_quad import solve_quad
from azcausal.solvers.solve_scipy import solve_scipy


def solve(solver, *args, **kwargs):
    if isinstance(solver, str):
        if solver == 'scipy':
            return solve_scipy(*args, **kwargs)
        elif solver == 'cvxpy':
            return solve_cvxpy(*args, **kwargs)
        elif solver == 'quad':
            return solve_quad(*args, **kwargs)
        elif solver == 'grad':
            return solve_grad(*args, **kwargs)
        else:
            raise Exception("Unknown solver {}".format(solver))
    elif isinstance(solver, Callable):
        return solver(*args, **kwargs)
    else:
        raise Exception("Unknown solver {}".format(solver))

