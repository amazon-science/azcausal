import pandas as pd
from numpy.random import RandomState

from azcausal.core.effect import Effect
from azcausal.core.panel import Panel
import numpy as np


class SyntheticEffect(object):

    def __init__(self, outcome, treatment, intervention=None, mode='perc', random_state=RandomState(42), tags=None):
        super().__init__()
        self.outcome = outcome
        self.treatment = treatment
        self.intervention = intervention
        self.mode = mode
        self.random_state = random_state
        self.tags = tags

    def generator(self, n_runs):
        outcome = self.outcome
        treatment = self.treatment
        mode = self.mode

        intervention = self.intervention
        if intervention is None:
            intervention = (treatment != 0).astype(int)

        treatment = self.treatment
        assert len(outcome) == len(treatment) == len(
            intervention), "The number of time steps in treatment and outcome must be the same"
        assert treatment.shape == intervention.shape

        _, n_units = treatment.shape
        n_pool = len(outcome.columns)

        for _ in range(n_runs):

            # select the units from the outcome
            u = self.random_state.choice(list(range(n_pool)), size=n_units, replace=n_units > n_pool)
            true_outcome = outcome.iloc[:, u]

            if mode == 'perc':
                treated_outcome = true_outcome * (1 + treatment)
            elif mode == 'abs':
                treated_outcome = true_outcome + treatment
            else:
                raise Exception("Unknown mode. Use 'perc' or 'abs'.")

            n = intervention.sum()
            att = (treated_outcome.values - true_outcome.values).sum() / n
            T = (treated_outcome.values * intervention).sum() / n

            effect = Effect(value=att, observed=T, multiplier=n)

            # get the intervention matrix (np.nan indicates no effect, but mark as treated)
            iv = pd.DataFrame(intervention,
                              index=true_outcome.index,
                              columns=true_outcome.columns)

            panel = Panel(treated_outcome, iv)

            yield {
                'input': dict(att=att, mode=mode, outcome=outcome, treatment=treatment, intervention=intervention),
                'correct': dict(effect=effect, outcome=outcome),
                'panel': panel,
                'tags': dict(self.tags) if self.tags is not None else dict()
            }
