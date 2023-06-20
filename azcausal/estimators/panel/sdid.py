import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azcausal.core.error import JackKnife
from azcausal.core.estimator import Estimator
from azcausal.core.plots import plot_hist_contr_treat
from azcausal.core.solver import SparseSolver, FrankWolfe, func_simple_sparsify, Sampling
from azcausal.estimators.panel.did import did_simple


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

        # pre weighted by lambda
        pre_sc = Y_pre_synth @ lambd
        pre_treat = Y_pre_treat.mean(axis=0) @ lambd

        # the average treatment effect on the treated
        did = did_simple(pre_sc, Y_post_synth.mean(), pre_treat, Y_post_treat.mean())

        # calculate att for each time period
        Y_avg_post_treat = Y_post_treat.mean(axis=0)
        att = (Y_avg_post_treat - pre_treat) - (Y_post_synth - pre_sc)

        W = (pnl.time() >= pnl.start).astype(int)
        T = pnl.Y(treat=True).mean(axis=0)
        SC = pnl.Y(contr=True).T @ omega

        # create the data on which sdid made the decision
        data = pd.DataFrame(dict(time=pnl.time(), SC=SC, T=T, W=W))
        data.loc[W == 0, "lambd"] = lambd
        data.loc[W == 1, "att"] = att
        data["T'"] = data["T"] - data["att"].fillna(0.0)
        data = data.set_index("time")

        return dict(name="sdid", estimator=self, panel=pnl, data=data, lambd=lambd,
                    omega=omega, noise=noise, solvers=solvers, **did)

    def error(self, estm, method, **kwargs):
        return method.run(estm, "att", f_estimate=SDIDEstimationFunction(type(method) != JackKnife), **kwargs)

    def plot(self, estm, title=None, trend=False, sc=True, show=True):

        data, lambd, omega = estm["data"], estm["lambd"], estm["omega"]
        start_time = data.query("W == 0").index.max()

        fig, ((top_left, top_right), (bottom_left, bottom_right)) = plt.subplots(2, 2,
                                                                                 figsize=(12, 4),
                                                                                 height_ratios=[4, 2],
                                                                                 width_ratios=[8.5, 1.5])

        top_left.plot(data.index, data["T"], label="T", color="blue")
        if sc:
            top_left.plot(data.index, data["SC"], label="SC", color="red")

        top_left.set_xticklabels([])
        top_left.axvline(start_time, color="black", alpha=0.3)

        if trend:
            top_left.plot(data.index, data["T'"], "--", color="blue", alpha=0.5)
            for t, v in data.query("W == 1")["att"].items():
                top_left.arrow(t, data.loc[t, "T"] - data.loc[t, "att"], 0, v, color="black",
                               length_includes_head=True, head_width=0.3, width=0.01, head_length=2)

        def plot_arrow(ax, x, y, dy, **kwargs):
            ax.annotate("", xy=(x, y), xytext=(x, y + dy), arrowprops=dict(arrowstyle="<-", **kwargs))

        plot_arrow(top_right, 1, estm["pre_contr"], estm["delta_contr"], color="red", lw=2)
        plot_arrow(top_right, 2, estm["pre_treat"], estm["delta_treat"], color="blue", lw=2)
        plot_arrow(top_right, 3, estm["pre_treat"], estm["att"], color="black", lw=2)

        if sc:
            top_right.set_xlim((0.5, 3.5))
        else:
            top_right.set_xlim((1.5, 3.5))

        top_right.set_xticklabels([])
        top_right.set_yticklabels([])

        top_right.set_ylim(top_left.get_ylim())
        top_right.set_xticklabels([])

        top_left.legend()
        top_left.set_title(title)

        w = data.query("W == 0")["lambd"]
        bottom_left.fill_between(w.index, 0.0, w, color="black")

        w = data.query("W == 1")["lambd"]
        bottom_left.fill_between(w.index, 0.0, w, color="black")

        bottom_left.axvline(start_time, color="black", alpha=0.3)
        bottom_left.set_ylim(0, 1)
        bottom_left.set_xlim(*top_left.get_xlim())
        bottom_left.set_yticklabels([])
        bottom_left.xaxis.set_tick_params(rotation=90)

        w = omega
        wp = sorted(w)[::-1]
        bottom_right.fill_between(np.arange(len(w)), 0, wp, color="red")
        bottom_right.set_xticklabels([])
        bottom_right.set_yticklabels([])

        if show:
            plt.tight_layout()
            fig.show()

        return fig


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
