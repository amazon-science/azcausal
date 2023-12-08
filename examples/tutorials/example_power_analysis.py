import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState

from azcausal.core.panel import CausalPanel
from azcausal.core.effect import get_true_effect
from azcausal.core.error import JackKnife
from azcausal.core.parallelize import Joblib
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

        def __init__(self, panel, att, seed) -> None:
            super().__init__()
            self.panel = panel
            self.att = att
            self.seed = seed

        def __call__(self, *args, **kwargs):
            # parameters
            seed = self.seed
            att = self.att

            # constants
            panel = self.panel
            conf = 90
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

    # create all runs for this analysis (this can potentially include more dimensions as well)
    def g():
        for att in np.linspace(-30, 30, 13):
            for seed in range(n_samples):
                yield att, seed

    # run the simulation in parallel
    parallelize = Joblib(prefer='processes', progress=True)
    results = parallelize.run([Function(panel, *args) for args in g()])

    dx = (pd.DataFrame(results)
          .assign(true_in_ci=lambda dd: dd['true_avg_te'].between(dd['pred_avg_ci_lb'], dd['pred_avg_ci_ub']))
          .assign(perc_te_error=lambda dd: dd['pred_perc_te'] - dd['true_perc_te'])
          )

    # get the power and coverage for each group now
    pw = dx.assign(sign=lambda dd: dd['pred_sign']).groupby('att').apply(f_power).sort_index().reset_index()
    coverage = dx.groupby('att')['true_in_ci'].mean()
    error = dx.groupby('att').aggregate(mean=('perc_te_error', 'mean'), se=('perc_te_error', 'sem'))

    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    fig.suptitle(f'CaliforniaProp99', fontsize=16)

    top.plot(pw['att'], pw['-'], "-o", color="red", label='-')
    top.plot(pw['att'], pw['+'], "-o", color="green", label='+')
    top.plot(pw['att'], pw['+/-'], "-o", color="black", label='+/-', alpha=0.5)
    top.axhline(1.0, color="black", alpha=0.15)
    top.axhline(0.9, color="black", alpha=0.15, linestyle='--')
    top.axhline(0.0, color="black", alpha=0.15)
    top.set_ylim(-0.05, 1.05)
    top.set_xlabel("ATT (%)")
    top.set_ylabel("Statistical Power")
    top.legend()

    middle.plot(coverage.index, coverage.values, "-o", color="black", label="coverage")
    middle.axhline(1.0, color="black", alpha=0.15)
    middle.axhline(0.0, color="black", alpha=0.15)
    middle.set_ylim(-0.05, 1.05)
    middle.set_xlabel("ATT (%)")
    middle.set_ylabel("Coverage")
    middle.legend()

    bottom.plot(error.index, np.zeros(len(error)), color='black', alpha=0.7)
    bottom.plot(error.index, error['mean'], '-o', color='red')
    bottom.errorbar(error.index, error['mean'], error['se'], color='red', alpha=0.5, barsabove=True)
    bottom.set_xlabel("ATT (%)")
    bottom.set_ylabel("Error")

    plt.tight_layout()
    plt.show()

    plt.show()
