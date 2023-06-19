from abc import abstractmethod

import numpy as np
import pandas as pd
import scipy
from numpy.random import RandomState

from azcausal.core.panel import Panel
from azcausal.core.parallelize import Serial
from azcausal.util import wrap_yield


class DefaultEstimationFunction(object):

    def args(self, estm, pnl, value):
        return [estm["estimator"], pnl, value]

    def run(self, args) -> None:
        estimator, pnl, value = args
        return estimator.fit(pnl)[value]


class Error(object):

    def run(self, estm, value, f_estimate=DefaultEstimationFunction(), parallelize=Serial()):

        pnl = estm["panel"]
        self.check(pnl)

        iterable = wrap_yield(self.generate(pnl), lambda x: f_estimate.args(estm, x, value))
        estms = parallelize(f_estimate.run, iterable)

        if len(estms) > 0:
            return self.evaluate(estm[value], estms)

    def evaluate(self, estm, others):
        se = self.se(others)

        res = dict(estms=others, se=se)

        ci = dict()
        for alpha in [90, 95, 99]:
            ci[f"{alpha}%"] = scipy.stats.norm.interval(alpha / 100, loc=estm, scale=se)

        res["CI"] = ci

        return res

    def check(self, pnl):
        return True

    @abstractmethod
    def generate(self, pnl):
        pass

    @abstractmethod
    def se(self, estms):
        pass


class JackKnife(Error):

    def generate(self, pnl):
        units = pnl.units()
        n = len(units)
        outcome = pnl.outcome
        intervention = pnl.intervention

        for k in range(n):
            u = units[k]

            pnl = Panel(outcome.drop(columns=[u]), intervention.drop(columns=[u]))

            if pnl.n_treat > 0:
                yield pnl

    def se(self, estms):
        n = len(estms)
        return np.sqrt(((n - 1) / n) * (n - 1) * np.var(estms, ddof=1))


class Placebo(Error):

    def __init__(self, n_samples=100, rng=RandomState(42), **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.rng = rng

    def check(self, pnl):
        assert pnl.n_contr > pnl.n_treat, "The panel must have more control than treated units for Placebo."

    def generate(self, pnl):
        outcome = pnl.get("outcome", contr=True)
        n_treat = pnl.n_treat

        for _ in range(self.n_samples):
            placebo = self.rng.choice(np.arange(pnl.n_contr), size=n_treat, replace=False)

            Wp = np.zeros_like(outcome.values)
            Wp[:, placebo] = pnl.get("intervention", treat=True, to_numpy=True)
            intervention = pd.DataFrame(Wp, index=outcome.index, columns=outcome.columns)

            yield Panel(outcome, intervention)

    def se(self, estms):
        n = len(estms)
        return np.sqrt((n - 1) / n) * np.std(estms, ddof=1)

    def evaluate(self, estm, others):
        placebo = np.mean(others)
        ret = super().evaluate(placebo , others)
        ret["placebo"] = placebo
        return ret


class Bootstrap(Error):

    def __init__(self, n_samples=100, rng=RandomState(42), mode="random", **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.mode = mode
        self.rng = rng

    def se(self, estms):
        n = len(estms)
        return np.sqrt((n - 1) / n) * np.std(estms, ddof=1)

    def generate(self, pnl):
        outcome, intervention = pnl.outcome, pnl.intervention
        units = pnl.units()
        N = len(units)

        for k in range(self.n_samples):

            if self.mode == "random":
                u = self.rng.choice(units, size=N, replace=True)
            # see https://towardsdatascience.com/the-bayesian-bootstrap-6ca4a1d45148
            elif self.mode == "bayes":
                alpha = np.full(N, 4.0)
                p = np.random.dirichlet(alpha)
                u = self.rng.choice(units, size=N, replace=True, p=p)
            # sample from control and treatment independently
            elif self.mode == "stratified":
                u_treat = self.rng.choice(intervention.columns[pnl.w], size=pnl.n_treat, replace=True)
                u_contr = self.rng.choice(intervention.columns[~pnl.w], size=pnl.n_contr, replace=True)
                u = np.concatenate([u_treat, u_contr])
            else:
                raise Exception(f"Unknown mode: {self.mode}. Available modes are `random`, `bayes`, and `stratified`.")

            if intervention[u].values.sum() > 0:
                yield Panel(outcome[u], intervention[u])
