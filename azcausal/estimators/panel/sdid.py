import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from azcausal.core.error import JackKnife
from azcausal.core.estimator import Estimator
from azcausal.core.solver import SparseSolver, FrankWolfe, func_simple_sparsify, Sampling
from azcausal.estimators.panel.did import did


def default_solver(sampling=Sampling(uniform=True, nnls=False, n_random=None)):
    return SparseSolver(full=FrankWolfe(max_iter=100, tol=1e-05, intercept=True, sampling=sampling),
                        sparse=FrankWolfe(max_iter=10000, tol=1e-05, intercept=True, sampling=sampling),
                        func_make_sparse=func_simple_sparsify)


class SDID(Estimator):

    def __init__(self, solver=default_solver(), **kwargs) -> None:
        super().__init__(**kwargs)

        if not isinstance(solver, dict):
            self.solver = dict(lambd=solver, omega=solver)
        else:
            self.solver = solver

    def fit(self, pnl, lambd=None, omega=None, optimize=True):

        Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr = pnl.Y_as_block(trim=True)
        (n_pre, n_post), (n_contr, n_treat) = pnl.counts()

        noise = np.diff(Y_pre_contr, axis=1).std(ddof=1)

        solvers = dict()

        if optimize and self.solver.get("lambd") is not None:
            eta = 1e-06
            A = Y_pre_contr
            b = Y_post_contr.mean(axis=1)

            solvers["lambd"] = self.solver["lambd"](A, b, eta, noise=noise, x0=lambd)
            lambd = solvers["lambd"]["x"]

        if optimize and self.solver.get("omega") is not None:
            eta = (n_treat * n_post) ** (1 / 4)
            A = Y_pre_contr.T
            b = Y_pre_treat.mean(axis=0)

            solvers["omega"] = self.solver["omega"](A, b, eta, noise=noise, x0=omega)
            omega = solvers["omega"]["x"]

        # calculate the synthetic control outcome pre and post
        Y_pre_synth = Y_pre_contr.T @ omega
        Y_post_synth = Y_post_contr.T @ omega

        # the average treatment effect on the treated
        att = did(Y_pre_synth @ lambd, Y_post_synth.mean(), Y_pre_treat.mean(axis=0) @ lambd, Y_post_treat.mean())

        pred = pd.DataFrame(dict(time=pnl.time(),
                                 synth_contr=pnl.Y(contr=True).T @ omega,
                                 treat=pnl.Y(treat=True).mean(axis=0),
                                 post=(pnl.time() >= pnl.start)
                                 )
                            ).set_index("time")

        return dict(name="sdid", estimator=self, panel=pnl, att=att, pred=pred, lambd=lambd,
                    omega=omega, noise=noise, solvers=solvers)

    def error(self, estm, method, **kwargs):
        return method.run(estm, "att", f_estimate=SDIDEstimationFunction(type(method) != JackKnife), **kwargs)

    def plot(self, estm, title=None, show=True):
        pnl = estm["panel"]
        start_time = pnl.start

        pred, lambd, omega = estm["pred"], estm["lambd"], estm["omega"]

        fig, ((top, right), (bottom, void)) = plt.subplots(2, 2,
                                                           figsize=(12, 4),
                                                           height_ratios=[4, 1],
                                                           width_ratios=[9, 1],
                                                           sharex='col')
        if title:
            fig.suptitle(title)

        top.plot(pred.index, pred["synth_contr"], label="SC", color="red")
        top.plot(pred.index, pred["treat"], label="T", color="blue")

        # sc_pre = pre["synth_contr"] @ lambd
        # sc_post = post["synth_contr"].mean()
        # treat_pre = pre["treat"] @ lambd
        # treat_post = post["treat"].mean()
        # top.plot()

        top.axvline(start_time, color="black")
        top.legend()
        top.set_title(title)

        w = lambd
        bottom.bar(pred.query("not post").index, w, color="black", width=1)
        bottom.scatter(pred.index[:len(w)][w > 0], w[w > 0], color="red")

        bottom.axvline(start_time, color="black")
        bottom.set_ylim(0, 1)
        bottom.set_yticklabels([])

        w = omega
        right.barh(np.arange(len(w)), w, color="red")
        right.set_yticklabels([])

        void.set_axis_off()

        if show:
            plt.tight_layout()
            fig.show()


class SDIDEstimationFunction(object):

    def __init__(self, optimize=True) -> None:
        self.optimize = optimize

    def args(self, estm, pnl, value):
        omega = fix_omega(estm["panel"], estm["omega"], pnl)
        return [estm["estimator"], pnl, estm["lambd"], omega, value]

    def run(self, args) -> None:
        estimator, pnl, lambd, omega, value = args
        return estimator.fit(pnl, lambd=lambd, omega=omega, optimize=self.optimize)[value]


def fix_omega(pnl, omega, npnl):
    if omega.sum() > 0:
        m = {u: o for u, o in zip(pnl.units(contr=True), omega)}
        omega = np.array([m[u] for u in npnl.units(contr=True)])
        omega = omega / omega.sum()
        return omega
    else:
        m = pnl.n_units(contr=True)
        return np.full(m, 1 / m)
