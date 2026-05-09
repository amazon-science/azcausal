"""ASDID Panel Estimator — wraps standalone asdid for the azcausal panel interface."""
import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.panel import CausalPanel
from azcausal.core.result import Result
from azcausal.standalone.asdid import asdid as _asdid, se_placebo as _se_placebo, ASDID as _ASDID


class ASDID(Estimator):

    def __init__(self, max_donors=10000, regularize=True, **kwargs):
        super().__init__(**kwargs)
        self.max_donors = max_donors
        self.regularize = regularize

    def fit(self, panel: CausalPanel, **kwargs):
        Y = panel['outcome'].values
        n_pre = int((~panel.post).sum())
        treat = panel.treat

        att_arr, lambd, omega, donors = _asdid(Y, n_pre, treat,
                                               regularize=self.regularize,
                                               max_donors=self.max_donors)

        att_val = att_arr[0]
        observed = Y[panel.post][:, treat].mean()
        scale = panel.n_interventions()

        # By-time effect (use selected donors)
        ctrl_idx = np.where(~treat)[0]
        ctrl = Y[:, ctrl_idx[donors]] @ omega[0]
        treatment = Y[:, treat].mean(axis=1)
        pre_treat = (lambd[0] @ treatment[:n_pre])
        pre_ctrl = (lambd[0] @ ctrl[:n_pre])
        by_time = pd.DataFrame({
            'T': treatment,
            'C': ctrl,
            'post': panel.post,
            'att': np.where(panel.post, (treatment - pre_treat) - (ctrl - pre_ctrl), np.nan),
        }, index=panel.index)
        by_time['CF'] = by_time['T'] - by_time['att'].fillna(0.0)

        info = dict(omega=omega, lambd=lambd)
        effect = Effect(att_val, observed=observed, scale=scale, by_time=by_time, data=info, name="ATT")
        return Result(dict(att=effect), data=panel, info=info, estimator=self)

    def refit(self, result: Result, **kwargs):
        return lambda panel: self.fit(panel)
