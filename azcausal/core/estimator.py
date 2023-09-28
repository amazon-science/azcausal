import logging
from typing import Callable

import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.effect import Effect
from azcausal.core.error import Error
from azcausal.core.panel import Panel
from azcausal.core.parallelize import Serial, Parallelize
from azcausal.core.result import Result


class Estimator(object):

    def __init__(self,
                 verbose: bool = False,
                 name: str = None,
                 random_state: RandomState = RandomState(42)) -> None:
        """
        This is the core method of the framework which is inherited by different estimators. Since some
        estimators can be non-deterministic a random state for reproducibility can be passed.

        Parameters
        ----------
        verbose
            Whether the estimator should print output.

        name
            The name of the estimator.

        random_state
            A random state (only useful for non-deterministic estimators)

        """

        super().__init__()

        # set the name simply equal to the class name if not provided
        if name is None:
            name = self.__class__.__name__.lower()

        self.name = name
        self.verbose = verbose
        self.logger = logging.getLogger()
        self.random_state = random_state

    def fit(self, panel, **kwargs) -> Result:
        """
        This method that needs to be implemented by an estimator which returns a `Result` object providing
        information about all estimates.

        Parameters
        ----------
        panel
            The input are panel data. For most estimators this will be an actual `Panel` objects for some a data
            frame can be supported as well.

        Returns
        -------
        result
            A result object which can contain multiple effects.

        """
        pass

    def refit(self,
              result: Result,
              **kwargs) -> Callable:
        """
        This method returns a function which allows to refit the estimate given results. This can be especially useful
        for estimating the error of a method significantly faster since intermediate result from the original estimate
        can be re-used.

        Parameters
        ----------
        result
            A result object returned before by the estimator.

        Returns
        -------
        function
            A callable function `f(panel)` which takes `Panel` as an input argument.

        """
        return lambda panel: self.fit(panel)

    def error(self,
              result: Result,
              method: Error,
              parallelize: Parallelize = Serial(),
              inplace: bool = True,
              **kwargs):
        """
        This is a convenience method which calculate the error given a result.


        Parameters
        ----------
        result
            The originally result object with estimates.

        method
            The error method to be used.

        parallelize
            The error estimation usually requires multiple runs which can be parallelized.

        inplace
            Directly overwrite the error in the result object to the new one.

        Returns
        -------
        result
            Whatever is returned by the error estimation method.

        """
        # use the possibly more performant implemented `refit` method by default
        f_estimate = self.refit(result, error=method, **kwargs)

        # return whatever is returned by the error estimation
        return method.run(result, f_estimate=f_estimate, parallelize=parallelize, inplace=inplace)


def results_from_outcome(obs_outcome: pd.DataFrame,
                         pred_outcome: pd.DataFrame,
                         intervention: pd.DataFrame):
    """

    This method can be used across estimators given the predicted outcome on a time-unit level. Another use case
    is to calculate the true effect given the ground truth.

    Parameters
    ----------
    obs_outcome
        The observed outcome.

    pred_outcome
        The predicted outcome.

    intervention
        The intervention where each integer represents a type of treatment.

    Returns
    -------

    """
    effects = {}

    for treatment in np.unique(intervention):

        if treatment != 0:
            mask = intervention == treatment

            observed = float(np.nanmean(obs_outcome.values[mask]))

            V = pred_outcome.values[mask]
            if np.all(np.isnan(V)):
                counter_factual = observed
            else:
                counter_factual = np.nanmean(V)

            att = observed - counter_factual

            diff = pred_outcome.subtract(obs_outcome)[mask]

            by_unit = (diff
                       .agg(['mean', 'sum', ], axis=0)
                       .fillna(0.0)
                       .T
                       .join(intervention.sum(axis=0).to_frame('n_interventions'))
                       )

            by_time = (diff
                       .agg(['mean', 'sum', ], axis=1)
                       .fillna(0.0)
                       .join(intervention.sum(axis=1).to_frame('n_interventions'))
                       )

            data = dict()
            effect = Effect(value=att, observed=observed, multiplier=mask.values.sum(), se=np.nan, data=data,
                            treatment=treatment, by_unit=by_unit, by_time=by_time)

            effects[f'T{treatment}'] = effect

    panel = Panel(obs_outcome, intervention)
    return Result(effects, panel=panel, data=dict(predicted=pred_outcome))
