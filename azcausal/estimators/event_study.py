import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels import PanelOLS

from azcausal.core.frame import CausalDataFrame
from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result
from azcausal.util import treatment_from_intervention, time_as_int, time_to_intervention


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

    def fit(self, cdf: CausalDataFrame, **kwargs):

        # add columns to have time as integer and whether a unit is treated (at least one intervention) or not
        dz = cdf.dropna()
        dz = dz.assign(treatment=treatment_from_intervention(cdf), itime=time_as_int(cdf["time"]))
        dz = dz.sort_values("itime").reset_index().set_index(["unit", "time"])

        att, se, by_time, regr = did_event_study(dz, n_pre=self.n_pre, exclude=self.exclude)

        dx = dz.query("intervention == 1")
        att = Effect(att, se=se, observed=dx["outcome"].mean(), scale=len(dx), by_time=by_time,
                     data=dict(regr=regr), name="ATT")
        return Result(dict(att=att), data=cdf, estimator=self)

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
