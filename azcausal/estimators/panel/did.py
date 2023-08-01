import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from linearmodels.panel.utility import AbsorbingEffectError
from matplotlib import pyplot as plt

from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.panel import Panel
from azcausal.core.result import Result
from azcausal.util import time_to_intervention, time_as_int, treatment_from_intervention


# ---------------------------------------------------------------------------------------------------------
# Simple DID
# ---------------------------------------------------------------------------------------------------------


def did_simple(pre_contr, post_contr, pre_treat, post_treat):
    """
    Calculates Difference-in-Difference (DiD) the simple way by differences the means.

    Parameters
    ----------
    pre_contr : float
        The mean pre-experiment of control
    post_contr : float
        The mean post-experiment of control
    pre_treat : float
        The mean pre-experiment of treatment
    post_treat : float
        The mean post-experiment of treatment

    Returns
    -------

    """
    # difference of control
    delta_contr = (post_contr - pre_contr)

    # difference of treatment
    delta_treat = (post_treat - pre_treat)

    # finally the difference in difference
    att = delta_treat - delta_contr

    return dict(att=att,
                delta_contr=delta_contr, delta_treat=delta_treat,
                pre_contr=pre_contr, post_contr=post_contr, pre_treat=pre_treat, post_treat=post_treat)


class DID(Estimator):

    def fit(self, panel, lambd=None):

        # get the outcome values pre and post and contr and treat in a block each
        Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr = panel.Y_as_block(trim=True)

        # lambda can be set to define the weights in pre - otherwise it will be simply uniform
        m = panel.n_pre
        if lambd is None:
            lambd = np.full(m, 1 / m)

        assert len(lambd) == m, f"The input weights lambda must be the same length as pre experiment: {m}"

        # difference for control regions
        pre_contr = Y_pre_contr.mean(axis=0) @ lambd
        post_contr = Y_post_contr.mean(axis=0).mean()

        # difference in treatment
        pre_treat = Y_pre_treat.mean(axis=0) @ lambd
        post_treat = Y_post_treat.mean(axis=0).mean()

        # finally the difference in difference
        did = did_simple(pre_contr, post_contr, pre_treat, post_treat)

        # create the data based on which did made the decision
        W = (panel.time() >= panel.start).astype(int)
        T = panel.Y(treat=True).mean(axis=0)
        C = panel.Y(contr=True).mean(axis=0)
        att = (Y_post_treat.mean(axis=0) - pre_treat) - (Y_post_contr.mean(axis=0) - pre_contr)

        # prepare the data provided as output
        by_time = pd.DataFrame(dict(time=panel.time(), C=C, T=T, W=W))
        by_time.loc[W == 0, "lambd"] = lambd
        by_time.loc[W == 1, "att"] = att
        by_time["CF"] = by_time["T"] - by_time["att"].fillna(0.0)
        by_time = by_time.set_index("time")

        att = Effect(did["att"], T=did["post_treat"], by_time=by_time, data=did, name="ATT")
        return Result(dict(att=att), data=panel, estimator=self)

    def plot(self, result, title=None, CF=True, C=True, show=True):

        data = result.effect.by_time

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.plot(data.index, data["T"], label="T", color="blue")
        if C is not None:
            ax.plot(data.index, data["C"], label="C", color="red")

        if CF is not None:
            ax.plot(data.index, data["CF"], "--", color="blue", alpha=0.5)

        pre = data.query("W == 0")
        start_time = pre.index.max()
        ax.axvline(start_time, color="black", alpha=0.3)
        ax.set_title(title)

        if show:
            plt.legend()
            plt.tight_layout()
            fig.show()

        return fig


# ---------------------------------------------------------------------------------------------------------
# Regression-Based DID (can also be used for staggered interventions)
# ---------------------------------------------------------------------------------------------------------

def did_regr(dy):
    """
    Calculates Difference-in-Difference (DiD) through regression (this also works for staggered interventions).
    This also provides a standard error of the estimate directly.

    Parameters
    ----------
    dy : pd.DataFrame
        A data frame with the index (unit, time) and the columns outcome and intervention.

    """
    dy = dy.reset_index().set_index(["unit", "time"])

    # the simple formula to calculate DiD
    formula = 'outcome ~ intervention + EntityEffects + TimeEffects'
    try:
        res = PanelOLS.from_formula(formula=formula, data=dy).fit(low_memory=True)
        # res = smf.ols(formula='outcome ~ intervention + C(unit) + C(time)', data=dy.reset_index()).fit()

        att = res.params["intervention"]
        se = res.std_errors["intervention"]

    except AbsorbingEffectError:
        att, se = 0.0, 0.0

    return att, se

class DIDRegressor(Estimator):
    def fit(self, panel):
        # for the regression we need directly the data frame
        dy = panel.to_frame(index=False) if isinstance(panel, Panel) else panel

        # do the regression and get the treatment effect with se
        att, se = did_regr(dy)
        T = dy.query("intervention == 1")["outcome"].mean()

        att = Effect(att, se=se, T=T, name="ATT")
        return Result(dict(att=att), data=panel, estimator=self)


# ---------------------------------------------------------------------------------------------------------
# Event Study (based on DiD)
# ---------------------------------------------------------------------------------------------------------

def did_event_study(df, n_pre=None, exclude=-1):
    """
    Perform an event study on a data set based on the time to intervention (also works for staggered interventions).

    Parameters
    ----------
    df : pd.DataFrame
        A data frame with the index (unit, time) and columns rtime (relative time until intervention with np.nan values
        if unit is in control) and outcome.
    n_pre : int
        The number of periods to be considered pre-experiment. `None` means we compare the DiD only to the excluded.
    exclude : int
        A time period that should be excluded (default: -1). We need to exclude at least one time period
        otherwise we have perfect collinearity.

    """

    # get for each entry the time to intervention (starting time, stime)
    y = time_to_intervention(df, "itime")

    # now find all time periods to be considered
    yp = y.dropna().unique()

    # remove time periods that should not be fit for regression
    if n_pre is not None:
        yp = yp[yp >= -n_pre]
    if exclude is not None:
        yp = yp[yp != exclude]

    # get the intervention values for regression (basically dummy variables)
    xp = df['intervention'].values
    interventions = np.column_stack([(y.values == v) if v < 0 else (y.values == v) & (xp != 0) for v in yp])
    labels = [f"t_{v}" for v in yp.astype(int)]

    # create the data frame to be fit by the regression package
    dx = pd.DataFrame(interventions, index=df.index, columns=labels)
    dx["outcome"] = df["outcome"]

    # solve the regression problem
    formula = f'outcome ~ {" + ".join([f"`{e}`" for e in labels])} + EntityEffects + TimeEffects'
    regr = PanelOLS.from_formula(formula=formula, data=dx).fit(low_memory=True)

    # regression using statsmodel and not linearmodels (deprecated)
    # res = smf.ols(formula=f'outcome ~ {" + ".join(labels)} + C(unit) + C(time)', data=dx.reset_index()).fit()
    # print(res.summary())

    # create the output data
    prefix = "t_"
    by_time = pd.DataFrame(dict(att=regr.params, se=regr.std_errors))
    by_time = by_time.loc[by_time.index.str.startswith(prefix)]
    by_time.index = pd.Index([int(s[len(prefix):]) for s in by_time.index], name="time", dtype=int)
    by_time["intervention"] = (by_time.index >= 0).astype(int)

    dz = by_time.query("intervention == 1")["att"]
    att, se = dz.mean(), dz.std(ddof=1)

    return att, se, by_time, regr


class EventStudy(Estimator):

    def __init__(self, n_pre=None, exclude=-1) -> None:
        super().__init__()
        self.n_pre = n_pre
        self.exclude = exclude

    def fit(self, panel):

        # since we use regression create the data frame from the panel (if not directly provided)
        dy = panel
        if not isinstance(panel, pd.DataFrame):
            dy = panel.to_frame()

        # add columns to have time as integer and whether a unit is treated (at least one intervention) or not
        dz = dy.dropna()
        dz = dz.assign(treatment=treatment_from_intervention(dy),
                       itime=time_as_int(dy["time"]))
        dz = dz.sort_values("itime").reset_index().set_index(["unit", "time"])

        att, se, by_time, regr = did_event_study(dz, n_pre=self.n_pre, exclude=self.exclude)

        dx = dz.query("intervention == 1")
        T = dx["outcome"].mean()
        n = len(dx)

        att = Effect(att, se=se, T=T, n=n, by_time=by_time, data=dict(regr=regr), name="ATT")
        return Result(dict(att=att), data=panel, estimator=self)

    def plot(self, result, show=True, ax=None):
        data = result.effect.by_time

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.errorbar(data.index, data["att"], yerr=data.se, fmt='-')
        ax.axvline(-0.5, color="red", alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("DID")

        if show:
            plt.tight_layout()
            plt.show()
