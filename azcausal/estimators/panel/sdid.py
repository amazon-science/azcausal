import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.error import JackKnife, Error
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result
from azcausal.core.solver import SparseSolver, FrankWolfe, func_simple_sparsify
from azcausal.estimators.panel.did import did_from_data


# this returns the default solver as proposed in the paper
def default_solver():
    return SparseSolver(dense=FrankWolfe(max_iter=100, tol=1e-05, intercept=True),
                        sparse=FrankWolfe(max_iter=10000, tol=1e-05, intercept=True),
                        func_sparsify=func_simple_sparsify)


class SDID(Estimator):

    def __init__(self,
                 solver=default_solver(),
                 **kwargs) -> None:
        """

        This estimator is implemented based on the following paper and source code:

        Arkhangelsky, Dmitry, Susan Athey, David A. Hirshberg, Guido W. Imbens, and Stefan Wager. 2021.
        "Synthetic Difference-in-Differences." American Economic Review, 111 (12): 4088-4118.
        DOI: 10.1257/aer.20190159

        R Code: https://synth-inference.github.io/synthdid/

        Parameters
        ----------
        solver
            A dictionary providing a solver object for the keys 'lambd' (time weights) and 'omega' (unit weights)

        """
        super().__init__(**kwargs)

        # if a solver is provided it use it for time and units weights at the same time
        self.solvers = dict(lambd=solver, omega=solver) if not isinstance(solver, dict) else solver

    def fit(self, panel, lambd=None, omega=None, optimize=True):
        solvers = dict()

        # find the time weights if not provided / requested
        lambd, solvers["lambd"] = sdid_time_weights(panel, solver=self.solvers["lambd"], lambd=lambd,
                                                    return_solver_result=True, optimize=optimize)

        # find the unit weights if not provided / requested
        omega, solvers["omega"] = sdid_unit_weights(panel, solver=self.solvers["omega"], omega=omega,
                                                    return_solver_result=True, optimize=optimize)

        # calculate the synthetic control using the omega weights
        control = panel.get('outcome', contr=True)
        assert np.all(control.columns == omega.index), "Omega columns do not match the data set."
        synth_control = pd.Series(control.values @ omega.values, index=control.index).to_frame('C')

        # get the average treatment group
        treatment = panel.get('outcome', treat=True).mean(axis=1).to_frame('T')

        # calculate the data frame the DID will be based on
        dx = (synth_control
              .join(treatment)
              .join(lambd.to_frame('lambd'), how='left')
              .assign(W=lambda x: np.isnan(x['lambd']).astype(int))
              )

        # get the did estimates from the simple panel
        did, by_time = did_from_data(dx)

        data = dict(lambd=lambd, omega=omega, solvers=solvers, did=did)
        att = Effect(did["att"], observed=did["post_treat"], multiplier=panel.n_interventions(), by_time=by_time,
                     data=data, name="ATT")

        return Result(dict(att=att), panel=panel, estimator=self)

    def refit(self,
              result: Result,
              optimize: bool = True,
              error: Error = None,
              low_memory: bool = False, **kwargs):
        """


        Parameters
        ----------
        result
            The result object on which the refitting should be based on.

        optimize
            Whether the weights should be optimized or not.
        error
            Whether the refit is occurring during error estimation.
        low_memory
            Whether the result should be stored without any additional data.

        Returns
        -------
        func
            A callable function `f(panel)`.

        """
        if error is not None:

            # always set optimize to false if we use jackknife
            if type(error) == JackKnife:
                optimize = False

        return Refit(result, optimize=optimize)

    def plot(self, result, title=None, CF=False, C=True, show=True):
        return sdid_plot(result.effect, title=title, CF=CF, C=C, show=show)


# ---------------------------------------------------------------------------------------------------------
# Refit
# ---------------------------------------------------------------------------------------------------------


class Refit:

    def __init__(self, result, optimize=True, low_memory=False) -> None:
        self.f_estimate = result.estimator.fit

        effect = result.effect
        self.omega = effect['omega']
        self.lambd = effect['lambd']

        self.low_memory = low_memory
        self.optimize = optimize

    def __call__(self, panel):
        result = self.f_estimate(panel, lambd=self.lambd, omega=self.omega, optimize=self.optimize)

        # only keep the effects of the result (not the data)
        if self.low_memory:
            result = Result(result.effects)

        return result


# ---------------------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------------------

def sdid_time_weights(panel, lambd=None, eta=1e-06, solver=default_solver(), return_solver_result=False, optimize=True):
    result = None

    pre_contr = panel.get('outcome', pre=True, contr=True)

    # get the starting solution if provided
    x0 = None
    if lambd is not None:
        lambd = fix_lambd(list(pre_contr.index), lambd)
        x0 = lambd.values

    if optimize:
        assert solver is not None, "Please provide a solver it optimize is set to true."

        # extract the data from the panel
        Y_pre_contr = pre_contr.values.T
        Y_post_contr = panel.Y(post=True, contr=True)
        noise = np.diff(Y_pre_contr, axis=1).std(ddof=1)

        # create the left and right hand side for the regression problem
        A = Y_pre_contr
        b = Y_post_contr.mean(axis=1)

        # use the solver to get the result
        result = solver(A, b, eta, noise=noise, x0=x0)
        lambd = result['x']

    # write the results to a data frame
    weights = pd.Series(lambd, index=list(pre_contr.index), name='lambd')

    if return_solver_result:
        return weights, result
    else:
        return weights


def sdid_unit_weights(panel, omega=None, solver=default_solver(), return_solver_result=False, optimize=True):
    result = None
    pre_contr = panel.get('outcome', pre=True, contr=True)

    # get the starting solution if provided
    x0 = None
    if omega is not None:
        omega = fix_omega(list(pre_contr.columns), omega)
        x0 = omega.values

    if optimize:
        assert solver is not None, "Please provide a solver it optimize is set to true."

        (n_pre, n_post), (n_contr, n_treat) = panel.counts()

        # extract the data from the panel
        Y_pre_contr = pre_contr.values.T
        Y_pre_treat = panel.Y(pre=True, treat=True)
        noise = np.diff(Y_pre_contr, axis=1).std(ddof=1)

        # create the left and right hand side for the regression problem
        eta = (n_treat * n_post) ** (1 / 4)
        A = Y_pre_contr.T
        b = Y_pre_treat.mean(axis=0)

        # use the solver to get the result
        result = solver(A, b, eta, noise=noise, x0=x0)
        omega = result['x']

    # write the results to a data frame
    weights = pd.Series(omega, index=pre_contr.columns, name='omega')

    if return_solver_result:
        return weights, result
    else:
        return weights


def fix_omega(units, omega):
    return (pd.DataFrame(index=units)
    .join(omega, how='left')
    .fillna(0.0)
    .apply(lambda x: (x / x.sum()) * omega.sum() if x.sum() > 0 else 1 / len(units))
    ['omega']
    )


def fix_lambd(times, lambd):
    return (pd.DataFrame(index=times)
    .join(lambd, how='left')
    .fillna(0.0)
    .apply(lambda x: (x / x.sum()) * lambd.sum() if x.sum() > 0 else 1 / len(times))
    ['lambd']
    )


# ---------------------------------------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------------------------------------

def sdid_plot(effect, title=None, CF=False, C=True, show=True):
    data, lambd, omega = effect.by_time, effect["lambd"], effect["omega"]
    start_time = data.query("W == 0").index.max()

    fig, ((top_left, top_right), (bottom_left, bottom_right)) = plt.subplots(2, 2,
                                                                             figsize=(12, 4),
                                                                             height_ratios=[4, 2],
                                                                             width_ratios=[8.5, 1.5])

    top_left.plot(data.index, data["T"], label="T", color="blue")
    if C:
        top_left.plot(data.index, data["C"], label="SC", color="red")

    top_left.set_xticklabels([])
    top_left.axvline(start_time, color="black", alpha=0.3)

    if CF:
        top_left.plot(data.index, data["CF"], "--", color="blue", alpha=0.5)

    def plot_arrow(ax, x, y, dy, **kwargs):
        ax.annotate("", xy=(x, y), xytext=(x, y + dy), arrowprops=dict(arrowstyle="<-", **kwargs))

    did = effect.data['did']

    plot_arrow(top_right, 1, did["pre_contr"], did["delta_contr"], color="red", lw=2)
    plot_arrow(top_right, 2, did["pre_treat"], did["delta_treat"], color="blue", lw=2)
    plot_arrow(top_right, 3, did["pre_treat"], did["att"], color="black", lw=2)

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
