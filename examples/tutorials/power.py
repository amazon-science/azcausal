import numpy as np

from azcausal.core.error import Bootstrap
from azcausal.core.performance import power
from azcausal.core.synth import SyntheticEffect
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID


# define a function that provides a result object (with error) given a panel
def f_estimate(panel):
    estimator = DID()
    result = estimator.fit(panel)
    estimator.error(result, Bootstrap(n_samples=21))
    return result


if __name__ == "__main__":

    # get a panel where no units are treated
    panel = CaliforniaProp99().panel()

    # let us only consider non-treated units here
    outcome = panel.outcome.loc[:, ~panel.w]

    # the type of synthetic treatment tested for power
    mode = 'perc'

    # the average treatment effect to be simulated. Here, we test 10% (because mode = 'perc')
    att = -0.15

    # the number of time periods treated
    n_treat_times = 10

    # Number of units to be treated
    n_treat_units = 3

    treatment = np.zeros_like(outcome.values)
    treatment[-n_treat_times:, :n_treat_units] = att

    # create synthetic panels where the last 8 time s
    synth_effect = SyntheticEffect(outcome, treatment, mode=mode, tags=dict(att=-0.1))

    # the number of samples to determine the power
    n_samples = 11

    # estimate the treatment effect for different scenarios
    results = [f_estimate(synth_effect.create(seed=seed).panel()) for seed in range(n_samples)]

    # get the power from the results
    pw = power(results, conf=90)

    print(f"(-) {pw['-']:.2%}")
    print(f"(+/-) {pw['+/-']:.2%}")
    print(f"(+) {pw['+']:.2%}")
