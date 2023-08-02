from abc import abstractmethod

import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.panel import Panel
from azcausal.core.parallelize import Serial


class Error(object):

    def __init__(self, random_state=RandomState(42), n_samples=None) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.random_state = random_state

    def run(self, result, f_estimate=None, panel=None, parallelize=Serial(), inplace=True):

        if f_estimate is None:
            f_estimate = result.estimator.fit

        if panel is None:
            panel = result.panel
        self.check(panel)

        # do the runs (supports parallelization)
        runs = parallelize(f_estimate, self.generate(panel))

        # the standard deviations for each effect
        se = dict()

        # calculate the standard error for each effect
        for effect in result.effects.keys():

            # get the actually estimates
            values = [run[effect].value for run in runs if run[effect].value is not None]
            if len(values) > 0:
                # and set the corresponding standard error
                se[effect] = self.se(values)

        if result and inplace:
            for name, se in se.items():
                result[name].se = se

        return se, runs

    def check(self, panel):
        return True

    @abstractmethod
    def generate(self, panel):
        pass

    @abstractmethod
    def se(self, estms):
        pass


class JackKnife(Error):

    def generate(self, panel):
        n = panel.n_units()

        for k in range(n):
            p = panel.iloc[:, np.delete(np.arange(n), k)]

            if p.n_treat > 0:
                yield p

    def se(self, values):
        n = len(values)
        return np.sqrt(((n - 1) / n) * (n - 1) * np.var(values, ddof=1))


class Placebo(Error):

    def __init__(self, n_samples=100, **kwargs) -> None:
        super().__init__(n_samples=n_samples, **kwargs)

    def check(self, panel):
        assert panel.n_contr > panel.n_treat, "The panel must have more control than treated units for Placebo."

    def generate(self, panel):
        outcome = panel.get("outcome", contr=True)
        n_treat = panel.n_treat

        for _ in range(self.n_samples):
            placebo = self.random_state.choice(np.arange(panel.n_contr), size=n_treat, replace=False)

            Wp = np.zeros_like(outcome.values)
            Wp[:, placebo] = panel.get("intervention", treat=True, to_numpy=True)
            intervention = pd.DataFrame(Wp, index=outcome.index, columns=outcome.columns)

            yield Panel(outcome, intervention)

    def se(self, values):
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)


class Bootstrap(Error):

    def __init__(self, n_samples=100, mode="random", **kwargs) -> None:
        super().__init__(n_samples=n_samples, **kwargs)
        self.n_samples = n_samples
        self.mode = mode

    def se(self, values):
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)

    def generate(self, panel):
        n = panel.n_units()

        for k in range(self.n_samples):

            if self.mode == "random":
                u = self.random_state.choice(np.arange(n), size=n, replace=True)
                p = panel.iloc[:, u]

            # sample from control and treatment independently
            elif self.mode == "stratified":
                u_treat = self.random_state.choice(np.where(panel.w)[0], size=panel.n_treat, replace=True)
                u_contr = self.random_state.choice(np.where(~panel.w)[0], size=panel.n_contr, replace=True)
                u = np.concatenate([u_treat, u_contr])

                p = panel.iloc[:, u]

            # see https://towardsdatascience.com/the-bayesian-bootstrap-6ca4a1d45148
            elif self.mode == "bayes":
                assert len(panel.W(to_numpy=False).drop_duplicates()) == 2, "Bayes bootstrap does not work for staggered or mixed-interventions."

                def sample(p, size, prefix="", alpha=4.0):

                    s = p.n_units()
                    Y = p.Y()
                    W = p.W()

                    P = self.random_state.dirichlet(np.full(len(Y), alpha), size=size)
                    labels = [f"{prefix}{i + 1}" for i in range(s)]

                    outcome = pd.DataFrame((P @ Y).T, columns=labels)
                    intervention = pd.DataFrame((P @ W).T, columns=labels)
                    intervention = (intervention > 0.0).astype(int)

                    return outcome, intervention

                treat = panel.iloc[:, panel.w]
                treat_outcome, treat_intervention = sample(treat, size=treat.n_units(), prefix="synt_treat_")

                contr = panel.iloc[:, ~panel.w]
                contr_outcome, contr_intervention = sample(contr, size=contr.n_units(), prefix="synt_contr_")

                outcome = pd.concat([treat_outcome, contr_outcome], axis=1)
                intervention = pd.concat([treat_intervention, contr_intervention], axis=1)

                p = Panel(outcome, intervention)

            else:
                raise Exception(f"Unknown mode: {self.mode}. Available modes are `random`, `bayes`, and `stratified`.")

            if p.n_units(treat=True) > 0:
                yield p
