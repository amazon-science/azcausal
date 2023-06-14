import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from matplotlib import pyplot as plt

from azcausal.core.estimator import Estimator
from azcausal.util import stime


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

    def fit(self, pnl, lambd=None):

        # get the outcome values pre and post and contr and treat in a block each
        Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr = pnl.Y_as_block(trim=True)

        # lambda can be set to define the weights in pre - otherwise it will be simply uniform
        m = pnl.n_pre
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

        # create the data on which sdid made the decision
        W = (pnl.time() >= pnl.start).astype(int)
        T = pnl.Y(treat=True).mean(axis=0)
        C = pnl.Y(contr=True).mean(axis=0)
        att = (Y_post_treat.mean(axis=0) - pre_treat) - (Y_post_contr.mean(axis=0) - pre_contr)

        # prepare the data provided as output
        data = pd.DataFrame(dict(time=pnl.time(), C=C, T=T, W=W))
        data.loc[W == 0, "lambd"] = lambd
        data.loc[W == 1, "att"] = att
        data["T'"] = data["T"] - data["att"].fillna(0.0)
        data = data.set_index("time")

        return dict(name="did", estimator=self, panel=pnl, data=data, **did)

    def plot(self, estm, title=None, trend=True, C=True, show=True):

        data = estm["data"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.plot(data.index, data["T"], label="T", color="blue")
        if C:
            ax.plot(data.index, data["C"], label="C", color="red")

        if trend:
            ax.plot(data.index, data["T'"], "--", color="blue", alpha=0.5)
            for t, v in data.query("W == 1")["att"].items():
                ax.arrow(t, data.loc[t, "T"] - data.loc[t, "att"], 0, v, color="black",
                         length_includes_head=True, head_width=0.3, width=0.01, head_length=2)

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

    # the simple formula to calculate DiD
    formula = 'outcome ~ intervention + EntityEffects + TimeEffects'
    res = PanelOLS.from_formula(formula=formula, data=dy).fit(low_memory=True)
    # res = smf.ols(formula='outcome ~ intervention + C(unit) + C(time)', data=dy.reset_index()).fit()

    return dict(att=res.params["intervention"], se=res.std_errors["intervention"])


class DIDRegressor(Estimator):
    def fit(self, pnl):
        dy = pnl.to_frame(index=True)
        return dict(estimator=self, panel=pnl, **did_regr(dy))


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
    y = stime(df, "rtime")

    # now find all time periods to be considered
    yp = y.dropna().unique()

    # remove time periods that should not be fit for regression
    if n_pre is not None:
        yp = yp[yp >= -n_pre]
    if exclude is not None:
        yp = yp[yp != exclude]

    # get the intervention values for regression (basically dummy variables)
    interventions = np.column_stack([y.values == v for v in yp])
    labels = [f"t_{v}" for v in yp.astype(int)]

    # create the data frame to be fit by the regression package
    dx = pd.DataFrame(interventions, index=df.index, columns=labels)
    dx["outcome"] = df["outcome"]

    # solve the regression problem
    formula = f'outcome ~ {" + ".join([f"`{e}`" for e in labels])} + EntityEffects + TimeEffects'
    res = PanelOLS.from_formula(formula=formula, data=dx).fit(low_memory=True)

    # regression using statsmodel and not linearmodels (deprecated)
    # res = smf.ols(formula=f'outcome ~ {" + ".join(labels)} + C(unit) + C(time)', data=dx.reset_index()).fit()
    # print(res.summary())

    # create the output data
    prefix = "t_"
    data = pd.DataFrame(dict(att=res.params, se=res.std_errors))
    data = data.loc[data.index.str.startswith(prefix)]
    data.index = pd.Index([int(s[len(prefix):]) for s in data.index], name="time", dtype=int)
    data["intervention"] = (data.index >= 0).astype(int)

    att = data.query("intervention == 1")["att"].mean()

    return dict(name="event_study", res=res, att=att, data=data)


class EventStudy(Estimator):

    def __init__(self, n_pre=None, exclude=-1) -> None:
        super().__init__()
        self.n_pre = n_pre
        self.exclude = exclude

    def fit(self, pnl):
        # since we use regression create the data frame from the panel
        dy = pnl.to_frame(index=True, treatment=True)
        out = did_event_study(dy, n_pre=self.n_pre, exclude=self.exclude)
        out["estimator"] = self
        return out

    def plot(self, estm, show=True, ax=None):
        data = estm["data"]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.errorbar(data.index, data.att, yerr=data.se, fmt='-')
        ax.axvline(-0.5, color="red", alpha=0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("DID")

        if show:
            plt.tight_layout()
            plt.show()
