import numpy as np
from numpy.random import RandomState

from azcausal.core.synth import SyntheticEffect


class Benchmark:

    def __init__(self, panel,
                 n_scenarios,
                 random_state=RandomState(42),
                 func_att=None,
                 func_n_treat=None,
                 func_n_post=None,
                 mode='perc') -> None:
        super().__init__()
        self.panel = panel
        self.n_scenarios = n_scenarios
        self.random_state = random_state
        self.mode = mode

        if func_att is None:
            func_att = lambda: random_state.uniform(0, 0.25)
        self.func_att = func_att

        if func_n_treat is None:
            func_n_treat = lambda: int(panel.n_units(treat=True))
        self.func_n_treat = func_n_treat

        if func_n_post is None:
            func_n_post = lambda: panel.n_post
        self.func_n_post = func_n_post

    def scenarios(self):
        panel = self.panel
        outcome = panel.outcome.loc[:, ~panel.w]

        for i in range(self.n_scenarios):

            att = self.func_att()
            n_post = self.func_n_post()
            n_treat = self.func_n_treat()

            # create the intervention matrix
            intervention = np.zeros_like(outcome.values).astype(int)
            intervention[-n_post:, :n_treat] = 1

            # create the treatment matrix for the effect
            intensity = intervention * att * -1

            tags = dict(seed=i)
            synth_effect = SyntheticEffect(outcome, intensity, intervention=intervention, mode=self.mode, tags=tags)

            scenario = synth_effect.create(seed=i)

            yield scenario
