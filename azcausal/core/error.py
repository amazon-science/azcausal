from abc import abstractmethod
from typing import Callable

import numpy as np

from azcausal.core.data import CausalData
from azcausal.core.parallelize import Serial, Parallelize
from azcausal.core.result import Result


class Error(object):

    def __init__(self,
                 n_samples=None,
                 **kwargs) -> None:
        """
        Generally, we want to attach confidence intervals to our estimates. One way of doing this is by re-running
        the estimate on re-sampled panel data and interpreting the resulting distribution.

        Parameters
        ----------

        n_samples
            The number of samples (might not apply to all methods).

        """

        super().__init__()
        self.n_samples = n_samples

    def run(self,
            result: Result,
            f_estimate: Callable = None,
            data: CausalData = None,
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

        cdata
            A causal data structure (CausalDataFrame or CausalPanel)

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
        if data is None:
            data = result.data

        # check if the panel is valid for this specific error estimation
        self.check(data)

        # do the runs (supports parallelization)
        runs = parallelize(self.generate(data), func=f_estimate)

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
    def check(self, data: CausalData):
        return True

    @abstractmethod
    # a generator which returns the samples from the panel
    def generate(self, data: CausalData):
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

    def generate(self, data: CausalData):
        n = data.n_units()
        for seed in range(n):
            p = data.jackknife(seed=seed)
            if p.n_treat > 0 and p.n_contr > 0:
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

    def generate(self, data: CausalData, seed=0):
        for i in range(self.n_samples):
            p = data.placebo(seed=seed + i)
            if p.n_treat > 0 and p.n_contr > 0:
                yield p

    def se(self, values):
        # implemented as in the synthetic difference-in-difference code in R
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)


# ---------------------------------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------------------------------

class Bootstrap(Error):

    def __init__(self, n_samples=100, **kwargs) -> None:
        super().__init__(n_samples=n_samples)
        self.n_samples = n_samples

    def se(self, values):
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)

    def generate(self, data: CausalData, seed=0):
        for i in range(self.n_samples):
            p = data.bootstrap(seed=seed + i)
            if p.n_treat > 0 and p.n_contr > 0:
                yield p
