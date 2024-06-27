from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.estimator import Estimator
from azcausal.core.panel import CausalPanel
from azcausal.core.result import Result
from azcausal.estimators.did import did_equation


def did_from_panel(dx: pd.DataFrame, lambd=None) -> dict:

    if lambd is None:
        dy = dx.groupby('post')[['T', 'C']].mean()
        pre_contr, post_contr = dy.loc[False, 'C'], dy.loc[True, 'C']
        pre_treat, post_treat = dy.loc[False, 'T'], dy.loc[True, 'T']

    else:
        post = dx.loc[dx['post']]
        post_treat, post_contr = post[['T', 'C']].mean()

        pre = dx.loc[~dx['post']]
        pre_treat = pre['T'].values @ lambd.values
        pre_contr = pre['C'].values @ lambd.values

    # calculate the treatment effect using the DID equation
    did = did_equation(pre_contr, post_contr, pre_treat, post_treat)

    return did


class DID(Estimator):

    def fit(self,
            panel: CausalPanel,
            lambd: np.ndarray = None,
            omega: np.ndarray = None,
            by_time: bool = True,
            **kwargs):

        Y = panel['outcome'].values
        control = Y[:, ~panel.treat].mean(axis=1) if omega is None else Y[:, ~panel.treat] @ omega
        treatment = Y[:, panel.treat].mean(axis=1)

        # feed in the already time
        dx = pd.DataFrame(dict(C=control, T=treatment, post=panel.post), index=panel.index)

        # get the did estimates from the simple panel
        did = did_from_panel(dx, lambd=lambd)

        # if also the effect by time should be returned
        if by_time:
            by_time = (dx
                       .assign(att=lambda x: ((x['T'] - did['pre_treat']) - (x['C'] - did['pre_contr'])).mask(x['post'] == 0))
                       .assign(CF=lambda x: x['T'] - x['att'].fillna(0.0))
                       )

        info = dict(did=did)
        scale = panel.n_interventions()
        att = Effect(did["att"], observed=did["post_treat"], scale=scale, by_time=by_time, data=info, name="ATT")
        return Result(dict(att=att), data=panel, info=info, estimator=self)

    def refit(self, result: Result, by_time=False, **kwargs) -> Callable:
        return lambda cdf: self.fit(cdf, by_time=by_time)

    def plot(self, result, title=None, CF=True, C=True, show=True):
        data = result.effect.by_time

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))

        ax.plot(data.index, data["T"], label="T", color="blue")
        if C is not None:
            ax.plot(data.index, data["C"], label="C", color="red")

        if CF is not None:
            ax.plot(data.index, data["CF"], "--", color="blue", alpha=0.5)

        pre = data.query("post == 0")
        start_time = pre.index.max()
        ax.axvline(start_time, color="black", alpha=0.3)
        ax.set_title(title)

        if show:
            plt.legend()
            plt.tight_layout()
            fig.show()

        return fig





