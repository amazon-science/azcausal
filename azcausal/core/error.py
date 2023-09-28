from abc import abstractmethod
from typing import Callable

import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.panel import Panel
from azcausal.core.parallelize import Serial, Parallelize
from azcausal.core.result import Result


class Error(object):

    def __init__(self,
                 random_state=RandomState(42),
                 n_samples=None) -> None:
        """
        Generally, we want to attach confidence intervals to our estimates. One way of doing this is by re-running
        the estimate on re-sampled panel data and interpreting the resulting distribution.

        Parameters
        ----------
        random_state
            The random state used for sampling.

        n_samples
            The number of samples (might not apply to all methods).

        """

        super().__init__()
        self.n_samples = n_samples
        self.random_state = random_state

    def run(self,
            result: Result,
            f_estimate: Callable = None,
            panel: Panel = None,
            parallelize: Parallelize = Serial(),
            inplace: bool = True):
        """
        This method runs the error estimation given a result from an estimator. The estimate will pull all data
        from the result object (e.g. estimation function or panel), however, that could also be overwritten and
        provided directly.


        Parameters
        ----------
        result
            The `Result` object from an estimator.

        f_estimate
            A function to estimate.

        panel
            The `Panel` data to sample from.

        parallelize
            A method if runs should be parallelized.

        inplace
            Whether the `Result` object should be modified in-place.

        Returns
        -------
        error
            Information about the error estimates.

        runs
            Each of the runs performed during the sampling

        """

        # the estimation function
        if f_estimate is None:
            f_estimate = result.estimator.fit

        # the panel to be used
        if panel is None:
            panel = result.panel

        # check if the panel is valid for this specific error estimation
        self.check(panel)

        # do the runs (supports parallelization)
        runs = parallelize(f_estimate, self.generate(panel))

        # the standard errors for each effect
        error = dict()

        # calculate the standard error for each effect
        for effect in result.effects.keys():

            # get the actually estimates
            values = [run[effect].value for run in runs]

            # remove invalid runs
            values = np.array([v for v in values if v is not None and not np.isnan(v)])

            if len(values) > 0:

                # and set the corresponding standard error
                se = self.se(values)

                error[effect] = se
                if inplace:
                    result[effect].se = se
                    result[effect].dof = len(values) - 1
                    result[effect].data['error'] = np.array(values)

        return error, runs

    # this can be overwritten by implementations. Is the panel valid for this method?
    def check(self, panel: Panel):
        return True

    @abstractmethod
    # a generator which returns the samples from the panel
    def generate(self, panel: Panel):
        pass

    @abstractmethod
    def se(self, values):
        """
        Calculate the standard error (se) from the point estimates.

        Parameters
        ----------
        values
            An array if point estimates.


        Returns
        -------
        se
            The resulting standard error.

        """
        pass


# ---------------------------------------------------------------------------------------------------------
# JackKnife
# ---------------------------------------------------------------------------------------------------------

class JackKnife(Error):

    def __init__(self, **kwargs) -> None:
        """
        JackKnife basically is a leave-on-out cross-validation method which always removes one unit and simulates
        what the estimates would have been.

        Note: This method can potentially require a lot of runs if the panel consists of many units. Some estimators
        can potentially be accelerated for this by implementing the re-fit function smartly.
        """
        super().__init__(**kwargs)

    def generate(self, panel):
        n = panel.n_units()

        # usually we should use all samples as done here
        indices = np.arange(n)

        # however, if that would be too much we can also sample from it
        if self.n_samples is not None:
            indices = self.random_state.choice(indices, replace=False, size=self.n_samples)

        # for each of the indices
        for k in indices:

            # always just delete one column in the panel
            p = panel.iloc[:, np.delete(np.arange(n), k)]

            # we need at least one treated unit for this
            if p.n_treat > 0:
                yield p

    def se(self, values):
        n = len(values)
        return np.sqrt(((n - 1) / n) * (n - 1) * np.var(values, ddof=1))


# ---------------------------------------------------------------------------------------------------------
# Placebo
# ---------------------------------------------------------------------------------------------------------

class Placebo(Error):

    def __init__(self,
                 n_samples: int = 100,
                 **kwargs) -> None:
        """

        The `Placebo` method only samples from the control units and flags them as treated. We expect that no
        treatment effect should be found.
        """
        super().__init__(n_samples=n_samples, **kwargs)

    def check(self, panel):
        # we need more control units as treatment units for this
        assert panel.n_contr > panel.n_treat, "The panel must have more control than treated units for Placebo."

    def generate(self, panel):
        # get the outcome and how many units should be flagged as placebo
        outcome = panel.get("outcome", contr=True)
        n_placebo = panel.n_treat

        for _ in range(self.n_samples):
            # randomly draw placebo units from control
            placebo = self.random_state.choice(np.arange(panel.n_contr), size=n_placebo, replace=False)

            # now create the intervention which simply randomizes the treatment given outcome
            Wp = np.zeros_like(outcome.values)
            Wp[:, placebo] = panel.get("intervention", treat=True, to_numpy=True)
            intervention = pd.DataFrame(Wp, index=outcome.index, columns=outcome.columns)

            yield Panel(outcome, intervention)

    def se(self, values):
        # implemented as in the synthetic difference-in-difference code in R
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)


# ---------------------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------------------


def bootstrap_random(panel: Panel,
                     n_max_retry: int,
                     random_state: RandomState):
    # the number of units in the panel
    n = panel.n_units()

    n_retry = 0
    while True:
        u = random_state.choice(np.arange(n), size=n, replace=True)
        p = panel.iloc[:, u]

        # if we have at least one treatment unit we are done
        if p.n_units(treat=True) > 0:
            break

        # keep track of how often we resample
        n_retry += 1
        if n_retry >= n_max_retry:
            break

    # if after retries we have no treatment units -> then repair the sample to have exactly one
    if p.n_units(treat=True) == 0:
        treat_units = np.where(panel.w)[0]
        u[0] = random_state.choice(treat_units)
        p = panel.iloc[:, u]

    return p


def bootstrap_stratified(panel: Panel,
                         random_state: RandomState):
    # sample from treated units
    pool = np.where(panel.w)[0]
    u_treat = random_state.choice(pool, size=panel.n_treat, replace=True)

    # sample from control units
    pool = np.where(~panel.w)[0]
    u_contr = random_state.choice(pool, size=panel.n_contr, replace=True)

    # create the panel
    u = np.concatenate([u_treat, u_contr])

    return panel.iloc[:, u]


# see https://towardsdatascience.com/the-bayesian-bootstrap-6ca4a1d45148
def bootstrap_bayes(panel: Panel,
                    alpha: float,
                    random_state: RandomState):
    assert len(panel.W(
        to_numpy=False).drop_duplicates()) == 2, "Bayes bootstrap does not work for staggered or mixed-interventions."

    # this function samples a single unit by using a linear combination
    def f(p, size, prefix=""):
        s = p.n_units()
        Y = p.Y()
        W = p.W()

        P = random_state.dirichlet(np.full(len(Y), alpha), size=size)
        labels = [f"{prefix}{i + 1}" for i in range(s)]

        outcome = pd.DataFrame((P @ Y).T, columns=labels)
        intervention = pd.DataFrame((P @ W).T, columns=labels)
        intervention = (intervention > 0.0).astype(int)

        return outcome, intervention

    treat = panel.iloc[:, panel.w]
    treat_outcome, treat_intervention = f(treat, size=treat.n_units(), prefix="synt_treat_")

    contr = panel.iloc[:, ~panel.w]
    contr_outcome, contr_intervention = f(contr, size=contr.n_units(), prefix="synt_contr_")

    outcome = pd.concat([treat_outcome, contr_outcome], axis=1)
    intervention = pd.concat([treat_intervention, contr_intervention], axis=1)

    return Panel(outcome, intervention)


class Bootstrap(Error):

    def __init__(self,
                 n_samples=100,
                 mode="random",
                 n_max_retry=5,
                 alpha=4.0,
                 **kwargs) -> None:
        """
        The `Bootstrap` method samples from the `Panel` with replacement. Different modes are supported:

        - 'random': truly randomly sample from the `Panel`
        - 'stratified': always keeps the number of control and treatment units the same as originally.
        - 'bayes': samples units as a linear combination of units (separately for control and treatment keeping the
                   balance as stratified)

        Parameters
        ----------
        n_samples
            The number of samples.

        mode
            The mode to be used: `random`, `stratified`, or `bayes`

        n_max_retry
            Number of times sampling is retried until all requirements are met (e.g. having at least one treatment unit)

        alpha
            The distribution parameter for bayes sampling. Can be ignored for other modes.

        """
        super().__init__(n_samples=n_samples, **kwargs)
        self.n_samples = n_samples
        self.mode = mode
        self.n_max_retry = n_max_retry
        self.alpha = alpha

    def se(self, values):
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)

    def generate(self, panel):
        for _ in range(self.n_samples):
            if self.mode == "random":
                yield bootstrap_random(panel, self.n_max_retry, self.random_state)
            elif self.mode == "stratified":
                yield bootstrap_stratified(panel, self.random_state)
            elif self.mode == "bayes":
                yield bootstrap_bayes(panel, self.alpha, self.random_state)
            else:
                raise Exception(f"Unknown mode: {self.mode}. Available modes are `random`, `bayes`, and `stratified`.")
