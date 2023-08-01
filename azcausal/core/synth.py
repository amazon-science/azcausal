import pandas as pd
from numpy.random import RandomState

from azcausal.core.effect import Effect
from azcausal.core.panel import Panel
import numpy as np

class SyntheticEffect(object):

    def __init__(self, outcome, treatment, mode='perc', random_state=RandomState(42), yield_panel=False) -> None:
        super().__init__()
        self.outcome = outcome
        self.treatment = treatment
        self.mode = mode
        self.random_state = random_state
        self.yield_panel = yield_panel

    def generator(self, n_runs):
        outcome = self.outcome
        mode = self.mode

        treatment = self.treatment
        assert len(outcome) == len(treatment), "The number of time steps in treatment and outcome must be the same"

        _, n_units = treatment.shape
        n_pool = len(outcome.columns)

        for _ in range(n_runs):

            # select the units from the outcome
            u = self.random_state.choice(list(range(n_pool)), size=n_units, replace=n_units > n_pool)
            true_outcome = outcome.iloc[:, u]

            # get the intervention matrix (np.nan indicates no effect, but mark as treated)
            intervention = pd.DataFrame((np.nan_to_num(treatment, nan=1.0) != 0).astype(int),
                                        index=true_outcome.index,
                                        columns=true_outcome.columns)

            effect = np.nan_to_num(treatment, nan=0.0)

            if mode == 'perc':
                treated_outcome = true_outcome * (1 + effect)
            elif mode == 'abs':
                treated_outcome = true_outcome + effect
            else:
                raise Exception("Unknown mode. Use 'perc' or 'abs'.")

            panel = Panel(treated_outcome, intervention)

            n = intervention.values.sum()
            att = (treated_outcome.values - true_outcome.values).sum() / n
            T = (treated_outcome.values * intervention).sum() / n

            effect = Effect(value=att, T=T, n=n)

            yield {
                'att': att,
                'mode': mode,
                'true_effect': effect,
                'outcome': outcome,
                'true_outcome': true_outcome,
                'treated_outcome': treated_outcome,
                'panel': panel
            }
