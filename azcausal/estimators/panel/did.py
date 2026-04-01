from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.error import Error, JackKnife
from azcausal.core.estimator import Estimator
from azcausal.core.panel import CausalPanel
from azcausal.core.parallelize import Serial, Parallelize
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

    def __init__(self, use_fast_jackknife: bool = False, **kwargs) -> None:
        """
        Difference-in-Differences estimator.

        Parameters
        ----------
        use_fast_jackknife
            If True, when computing standard error with JackKnife, it will execute the optimized DID 
            specific JackKnife implementation. The fast version of JackKnife only returns the standard error
            and does not include any runs.
        """
        super().__init__(**kwargs)
        self.use_fast_jackknife = use_fast_jackknife

    def fit(self,
            panel: CausalPanel,
            lambd: np.ndarray = None,
            omega: np.ndarray = None,
            fix_weights=True,
            by_time: bool = True,
            **kwargs):

        if fix_weights and lambd is not None:
            lambd = fix_lambd(list(panel.times(pre=True)), lambd)
        if fix_weights and omega is not None:
            omega = fix_omega(panel.units(contr=True), omega)

        Y = panel['outcome'].values
        control = Y[:, ~panel.treat].mean(axis=1) if omega is None else Y[:, ~panel.treat] @ omega.values
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

        info = dict(did=did, omega=omega, lambd=lambd)
        scale = panel.n_interventions()
        att = Effect(did["att"], observed=did["post_treat"], scale=scale, by_time=by_time, data=info, name="ATT")
        return Result(dict(att=att), data=panel, info=info, estimator=self)

    def error(self,
              result: Result,
              method: Error,
              parallelize: Parallelize = Serial(),
              inplace: bool = True,
              low_memory: bool = True,
              **kwargs):
        """
        Override error() to run the optimized JackKnife if applicable, otherwise run error as normal. 
        """
        if self.use_fast_jackknife and isinstance(method, JackKnife):
            return self._fast_jackknife(result, method, inplace=inplace)
        return super().error(result, method, parallelize=parallelize, inplace=inplace, low_memory=low_memory, **kwargs)

    def _fast_jackknife(self, result: Result, method: JackKnife, inplace: bool = True):
        """
        Fast analytical JackKnife for DID-family estimators (DID, SDID, etc.).

        Instead of re-fitting the estimator, we compute the leave one out ATT value directly from the initial DID fit results.
        This works because we don't reweight omega or lambda weights during JackKnife.
        """
        effect = result.effect
        panel = result.data
        omega = effect['omega'].values if effect['omega'] is not None else np.full(panel.n_contr, 1.0 / panel.n_contr)
        lambd = effect['lambd'].values if effect['lambd'] is not None else np.full(panel.n_pre, 1.0 / panel.n_pre)

        Yv = panel['outcome'].values
        treat = panel.treat
        post = panel.post

        d_pre = lambd @ Yv[~post]  # per-unit lambda-weighted pre value
        d_post = Yv[post].mean(axis=0)  # per-unit average post value

        # pre-compute per-group arrayst
        d_pre_treat = d_pre[treat]
        d_post_treat = d_post[treat]
        d_pre_contr = d_pre[~treat]
        d_post_contr = d_post[~treat]

        pre_treat = d_pre_treat.mean()
        post_treat = d_post_treat.mean()
        pre_contr = d_pre_contr @ omega
        post_contr = d_post_contr @ omega

        n_t = treat.sum()
        n_c = (~treat).sum()

        values = []
        # only leave out treatment units if at least 2 remain
        if n_t > 1:
            for k in range(n_t):
                jk_pre = (pre_treat * n_t - d_pre_treat[k]) / (n_t - 1)
                jk_post = (post_treat * n_t - d_post_treat[k]) / (n_t - 1)
                values.append((jk_post - jk_pre) - (post_contr - pre_contr))
        # only leave out control units if at least 2 remain
        if n_c > 1:
            for k in range(n_c):
                w = np.delete(omega, k)
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum
                values.append((post_treat - pre_treat) - (np.delete(d_post_contr, k) @ w - np.delete(d_pre_contr, k) @ w))

        values = np.array(values)
        if np.any(np.isnan(values)):
            raise ValueError("JackKnife produced NaN values. Initial panel data may contain NaN values.")

        error = dict()
        if len(values) > 0:
            se = method.se(values)
            n = len(values)

            # DID produces a single ATT effect so set SE directly on it
            effect_name = next(iter(result.effects))
            error[effect_name] = se
            if inplace:
                result[effect_name].se = se
                result[effect_name].dof = n - 1
                result[effect_name].data['error'] = values

        return error, []

    def refit(self, result: Result, by_time=False, **kwargs) -> Callable:
        effect = result.effect
        return lambda cdf: self.fit(cdf, by_time=by_time, lambd=effect['lambd'], omega=effect['omega'])

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


def fix_omega(units, omega):
    return (pd.DataFrame(index=units)
    .join(omega)
    .fillna(0.0)
    .apply(lambda x: x / x.sum() if x.sum() > 0 else 1 / len(units))
    ['omega']
    )


def fix_lambd(times, lambd):
    return (pd.DataFrame(index=times)
    .join(lambd, how='left')
    .fillna(0.0)
    .apply(lambda x: (x / x.sum()) * lambd.sum() if x.sum() > 0 else 1 / len(times))
    ['lambd']
    )
