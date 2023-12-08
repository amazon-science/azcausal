import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.effect import get_true_effect
from azcausal.core.error import JackKnife
from azcausal.core.panel import CausalPanel
from azcausal.core.parallelize import Serial
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID
from azcausal.util import zeros_like
from azcausal.util.analysis import f_power

if __name__ == "__main__":
    # get a causal data frame where no units are treated
    panel = CaliforniaProp99().panel().filter(contr=True)

    # the number of samples used for measuring power
    n_samples = 100

    class Function:

        def __init__(self, panel, seed) -> None:
            super().__init__()
            self.panel = panel
            self.seed = seed

        def __call__(self, *args, **kwargs):
            # parameters
            seed = self.seed

            # constants
            panel = self.panel
            conf = 90
            att = -20
            n_treat = 5
            n_post = 12

            # random seed for reproducibility
            random_state = RandomState(seed)

            # define what is treated and when
            treat_units = random_state.choice(np.arange(panel.n_units()), replace=False, size=n_treat)

            intervention = zeros_like(panel.intervention)
            intervention.iloc[-n_post:, treat_units] = 1

            te = panel.outcome * intervention * (att / 100)
            outcome = panel.outcome + te

            # create the new panel with the new intervention
            panel = CausalPanel(data=dict(intervention=intervention, te=te, outcome=outcome)).setup()

            # use the estimator to get the effect
            true_effect = get_true_effect(panel)

            # run the estimator to get the predicted effect
            estimator = DID()
            result = estimator.fit(panel)
            estimator.error(result, JackKnife())
            pred_effect = result.effect

            # create an output dictionary of what is true and what we have measured
            res = dict(**pred_effect.to_dict(prefix='pred_', conf=conf), **true_effect.to_dict(prefix='true_', conf=conf))
            res.update(dict(att=att, seed=seed))

            return res


    parallelize = Serial(progress=True)
    results = parallelize.run([Function(panel, seed) for seed in range(n_samples)])

    dx = (pd.DataFrame(results)
          .assign(true_in_ci=lambda dd: dd['true_avg_te'].between(dd['pred_avg_ci_lb'], dd['pred_avg_ci_ub']))
          .assign(avg_te_error=lambda dd: dd['true_avg_te'] - dd['pred_avg_te'])
          .assign(rel_te_error=lambda dd: dd['true_rel_te'] - dd['pred_rel_te'])
          )

    # get the power from the results
    power = f_power(dx.assign(sign=lambda dd: dd['pred_sign']))

    print("Power")
    print(f"(+) {power['+']:.2%}")
    print(f"(+/-) {power['+/-']:.2%}")
    print(f"(-) {power['-']:.2%}")

    print()

    coverage = dx['true_in_ci'].mean()
    print(f"Coverage: {coverage:.1%}")

    avg_te_rmse = np.sqrt((dx['avg_te_error'] ** 2).mean())
    print(f"Average TE RMSE: {avg_te_rmse}")

    rel_te_rmse = np.sqrt((dx['rel_te_error'] ** 2).mean())
    print(f"Relative TE RMSE: {rel_te_rmse}")
