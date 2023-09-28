import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.scenario import Scenario


class SyntheticEffect(object):

    def __init__(self,
                 outcome: pd.DataFrame,
                 treatment: np.ndarray,
                 intervention: np.ndarray = None,
                 mode: str = 'abs',
                 tags: dict = None) -> None:
        """
        Given some ground truth data this class overlays a synthetic effect with a specified ATT, provided in
        absolute `abs` or relative `rel` values.

        Parameters
        ----------
        outcome
            The outcome as a data frame where index represents time and columns the units.

        treatment
            The actual treatment value for a specific unit in time (depending on mode relative or absolute).

        intervention
            The intervention data frame with 0s and 1s where 1 represents an intervention. This can directly
            be derived from the treatment data frame (except for if the effect shall be placebo).

        mode
            The mode of the treatment, either `abs` (absolute) or `rel` (relative).

        tags
            Provide additional keywords that describe what the original data set is for post-analysis.

        """
        super().__init__()

        # derive intervention from treatment if not provided (needs to be given if placebo)
        if intervention is None:
            intervention = (treatment != 0).astype(int)

        assert len(outcome) == len(treatment) == len(
            intervention), "The number of time steps in treatment and outcome must be the same"
        if intervention is not None:
            assert treatment.shape == intervention.shape

        self.outcome = outcome
        self.treatment = treatment
        self.intervention = intervention
        self.mode = mode
        self.tags = tags if tags is not None else dict()

    def create(self, seed: int = None) -> Scenario:
        """
        The core implementation of the effect which creates a scenario. A scenario does not only contain the panel
        to which causal inference should be applied, but also the ground truth for benchmarking of methods.

        Parameters
        ----------
        seed
            A random seed for reproducibility.

        Returns
        -------
        scenario
            A scenario with the intended treatment effect.

        """

        # create a random state based on the seed
        random_state = RandomState(seed)

        outcome = self.outcome
        treatment = self.treatment
        intervention = self.intervention
        mode = self.mode

        _, n_units = treatment.shape
        n_pool = len(outcome.columns)

        # select the units from the outcome
        u = random_state.choice(list(range(n_pool)), size=n_units, replace=n_units > n_pool)
        true_outcome = outcome.iloc[:, u]

        if mode == 'perc':
            treated_outcome = true_outcome * (1 + treatment)
        elif mode == 'abs':
            treated_outcome = true_outcome + treatment
        else:
            raise Exception("Unknown mode. Use 'perc' or 'abs'.")

        return Scenario(true_outcome,
                        treated_outcome,
                        pd.DataFrame(intervention, index=true_outcome.index, columns=true_outcome.columns),
                        tags={**self.tags, 'seed': seed})

    def generator(self, n_scenarios: int):
        """
        A generator to create a pre-defined number of scenarios.
        """

        for seed in range(n_scenarios):
            yield self.create(seed)
