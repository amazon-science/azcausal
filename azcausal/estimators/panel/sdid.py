import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.error import JackKnife
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result
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

    def fit(self, panel, lambd=None, omega=None, optimize=True):

        Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr = panel.Y_as_block(trim=True)
        (n_pre, n_post), (n_contr, n_treat) = panel.counts()

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

        W = (panel.time() >= panel.start).astype(int)
        T = panel.Y(treat=True).mean(axis=0)
        SC = panel.Y(contr=True).T @ omega

        # create the data on which sdid made the decision
        by_time = pd.DataFrame(dict(time=panel.time(), SC=SC, T=T, W=W))
        by_time.loc[W == 0, "lambd"] = lambd
        by_time.loc[W == 1, "att"] = att
        by_time["CF"] = by_time["T"] - by_time["att"].fillna(0.0)
        by_time = by_time.set_index("time")

        data = dict(lambd=lambd, omega=omega, noise=noise, solvers=solvers, **did)

        att = Effect(did["att"], T=did["post_treat"], n=panel.n_interventions(), by_time=by_time, data=data, name="ATT")
        return Result(dict(att=att), data=panel, estimator=self)

    def refit(self, result, optimize=True, low_memory=False):
        return SDIDRun(result, optimize=optimize, low_memory=low_memory)

    def error(self, result, method, **kwargs):
        f_estimate = self.refit(result, optimize=type(method) != JackKnife, low_memory=True)
        return method.run(result, f_estimate=f_estimate, **kwargs)

    def plot(self, result, title=None, CF=False, C=True, show=True):

        effect = result.effect
        data, lambd, omega = effect.by_time, effect["lambd"], effect["omega"]
        start_time = data.query("W == 0").index.max()

        fig, ((top_left, top_right), (bottom_left, bottom_right)) = plt.subplots(2, 2,
                                                                                 figsize=(12, 4),
                                                                                 height_ratios=[4, 2],
                                                                                 width_ratios=[8.5, 1.5])

        top_left.plot(data.index, data["T"], label="T", color="blue")
        if C:
            top_left.plot(data.index, data["SC"], label="SC", color="red")

        top_left.set_xticklabels([])
        top_left.axvline(start_time, color="black", alpha=0.3)

        if CF:
            top_left.plot(data.index, data["CF"], "--", color="blue", alpha=0.5)

        def plot_arrow(ax, x, y, dy, **kwargs):
            ax.annotate("", xy=(x, y), xytext=(x, y + dy), arrowprops=dict(arrowstyle="<-", **kwargs))

        plot_arrow(top_right, 1, effect["pre_contr"], effect["delta_contr"], color="red", lw=2)
        plot_arrow(top_right, 2, effect["pre_treat"], effect["delta_treat"], color="blue", lw=2)
        plot_arrow(top_right, 3, effect["pre_treat"], effect["att"], color="black", lw=2)

        if C:
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


class SDIDRun:

    def __init__(self, result, optimize=True, low_memory=False) -> None:
        self.f_estimate = result.estimator.fit

        effect = result.effect
        self.omega = {u: o for u, o in zip(result.panel.units(contr=True), effect['omega'])}
        self.lambd = effect['lambd']

        self.low_memory = low_memory
        self.optimize = optimize

    def __call__(self, panel):
        omega = fix_omega(self.omega, panel)
        result = self.f_estimate(panel, lambd=self.lambd, omega=omega, optimize=self.optimize)

        # only keep the effects of the result (not the data)
        if self.low_memory:
            result = Result(result.effects)

        return result


def fix_omega(omega, new_panel):
    omega = np.array([omega.get(u, 0.0) for u in new_panel.units(contr=True)])

    if omega.sum() > 0:
        omega = omega / omega.sum()
        return omega
    else:
        m = new_panel.n_units(contr=True)
        return np.full(m, 1 / m)
