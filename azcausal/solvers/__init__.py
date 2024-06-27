from azcausal.solvers.solve_cvxpy import solve_cvxpy
from azcausal.solvers.solve_quad import solve_quad
from azcausal.solvers.solve_scipy import solve_scipy


def solve(solver, *args, **kwargs):
    if solver == 'scipy':
        return solve_scipy(*args, **kwargs)
    elif solver == 'cvxpy':
        return solve_cvxpy(*args, **kwargs)
    elif solver == 'quad':
        return solve_quad(*args, **kwargs)
    else:
        raise Exception("Unknown solver {}".format(solver))

