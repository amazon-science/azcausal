from typing import Callable

import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.frame import CausalDataFrame
from azcausal.core.regression import CausalRegression
from azcausal.core.result import Result


def did_equation(pre_contr, post_contr, pre_treat, post_treat):
    # difference of control
    delta_contr = (post_contr - pre_contr)

    # difference of treatment
    delta_treat = (post_treat - pre_treat)

    # finally the difference in difference
    att = delta_treat - delta_contr

    return dict(att=att,
                delta_contr=delta_contr, delta_treat=delta_treat,
                pre_contr=pre_contr, post_contr=post_contr, pre_treat=pre_treat, post_treat=post_treat)


def did_from_frame(cdf, cregr=None):
    # get the values to feed in into the did equation
    dx = (cdf
          .groupby(['treatment', 'post'])['outcome']
          .mean()
          )
    # get the values for DID and get the estimate
    pre_contr, post_contr = dx.loc[0, 0], dx.loc[0, 1]
    pre_treat, post_treat = dx.loc[1, 0], dx.loc[1, 1]

    # calculate the treatment effect using the DID equation
    did = did_equation(pre_contr, post_contr, pre_treat, post_treat)
    did['se'] = np.nan
    did['dof'] = None

    if cregr is not None:
        result = cregr.fit(cdf)
        assert np.allclose(did['att'], result['param']), "Regression: Something went wrong. ATT is not matching."
        did['se'] = result['se']

        n_times, n_units = cdf.n_times(), cdf.n_units()
        did['dof'] = (n_times * n_units) - (n_times + n_units) - 1

    return did


class DID(Estimator):

    def __init__(self, cregr: CausalRegression = CausalRegression(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.cregr = cregr

    def fit(self,
            cdf: CausalDataFrame,
            se: bool = False,
            **kwargs):
        # set the causal regression if the standard error is required
        cregr = self.cregr if se else None

        did = did_from_frame(cdf, cregr=cregr)

        att = Effect(did["att"], se=did['se'], dof=did['dof'], observed=did["post_treat"], scale=cdf.n_interventions(),
                     data=dict(did=did),
                     name="ATT")

        return Result(dict(att=att), data=cdf, estimator=self)

    def refit(self, result: Result, **kwargs) -> Callable:
        return lambda cdf: self.fit(cdf, se=False)


# needs to be revised
def did_from_frame_weights(cdf, cregr=None, lambd=None, omega=None):
    # the counts of the causal data frame
    (n_pre, n_post), (n_contr, n_treat) = cdf.counts()

    if lambd is None:
        lambd = np.full(n_pre, 1 / n_pre)
    time_weight = pd.Series(lambd, index=cdf.times(pre=True)).reindex(cdf.times()).fillna(1 / n_post)

    if omega is None:
        omega = np.full(n_contr, 1 / n_contr)
    unit_weight = pd.Series(omega, index=cdf.units(contr=True)).reindex(cdf.units()).fillna(1 / n_treat)

    cdx = (cdf
           .merge(unit_weight.to_frame("unit_weight"), left_on='unit', right_index=True, how='left')
           .merge(time_weight.to_frame("time_weight"), left_on='time', right_index=True, how='left')
           .assign(weight=lambda dx: dx['unit_weight'] * dx['time_weight'])
           )

    # get the values to feed in into the did equation
    dx = (cdx
          .assign(outcome=lambda dx: dx['outcome'] * dx['weight'])
          .groupby(['treatment', 'post'])['outcome']
          .sum()
          )

    # get the values for DID and get the estimate
    pre_contr, post_contr = dx.loc[0, 0], dx.loc[0, 1]
    pre_treat, post_treat = dx.loc[1, 0], dx.loc[1, 1]

    # calculate the treatment effect using the DID equation
    did = did_equation(pre_contr, post_contr, pre_treat, post_treat)
    did['se'] = np.nan
    did['dof'] = None

    if cregr is not None:
        # calculate the overall weight each segment needs to sum up to
        norm = (cdx.groupby(['treatment', 'post'])['weight'].count() / len(cdf))

        # normalize the weights for the regression
        cdx = (cdx
               .merge(norm.to_frame('norm'), left_on=['treatment', 'post'], right_index=True)
               .assign(cregr_weight=lambda dd: (dd['weight'] * dd['norm']) + 1e-64)
               )

        result = cregr.fit(cdx, weights='cregr_weight')
        assert np.allclose(did['att'], result['param']), "Regression: Something went wrong. ATT is not matching."
        did['se'] = result['se']

        n_times, n_units = cdf.n_times(), cdf.n_units()
        did['dof'] = (n_times * n_units) - (n_times + n_units) - 1

    return did
