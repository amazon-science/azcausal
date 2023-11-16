import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.error import Error
from azcausal.core.estimator import Estimator
from azcausal.core.result import Result


class FixedUnitWeightsEstimator(Estimator):

    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def fit(self, panel, weights=None, **kwargs):

        # if weights are not passed to the method, use the once defined in the class
        if weights is None:
            weights = self.weights

        # if no weights exist, estimate them from pre
        if weights is None:
            total = panel.get('outcome', pre=True).sum()
            weights = (total / total.sum()).to_dict()

        # all times with at least one intervention
        times = panel.intervention.mean(axis=1).pipe(lambda x: x[x > 0])

        # get the panel (outcome + intervention) where we measure the impact
        treated_panel = panel.loc[times.index]
        outcome = treated_panel.outcome.values
        intervention = treated_panel.intervention.values == 1

        # get the weight vector from the columns and make them always sum up to one
        w = np.array([[weights.get(unit, 0.0) for unit in panel.columns]])
        ww = w.repeat(len(times), axis=0)

        # calculate the treatment intensity and the observed values
        vs = dict(
            interventions=intervention.sum(axis=1),
            treat_weights=(intervention * ww).sum(axis=1),
            contr_weights=(~intervention * ww).sum(axis=1),
            treat_outcome=(intervention * outcome).sum(axis=1),
            contr_outcome=(~intervention * outcome).sum(axis=1),
        )

        by_time = (pd.DataFrame(vs)
                   .query("contr_weights > 0").query("treat_weights > 0")
                   .assign(factor=lambda dx: (dx['treat_weights'] / dx['contr_weights']))
                   .assign(counter_factual=lambda dx: dx['contr_outcome'] * dx['factor'])
                   .assign(tt=lambda dx: dx['treat_outcome'] - dx['counter_factual'])
                   .assign(att=lambda dx: dx['tt'] / dx['interventions'])
                   )

        n_interventions = by_time['interventions'].sum()
        total_lift = by_time['tt'].sum()

        att = total_lift / n_interventions
        observed = by_time['treat_outcome'].sum() / n_interventions

        data = dict(weights=weights)

        effect = Effect(att, observed=observed, multiplier=panel.n_interventions(), by_time=by_time, name="ATT")
        return Result(dict(att=effect), panel=panel, data=data, estimator=self)

    def refit(self,
              result: Result,
              optimize: bool = True,
              error: Error = None,
              low_memory: bool = False, **kwargs):

        estimator = result.estimator

        return lambda panel: estimator.fit(panel, weights=result.data['weights'])
