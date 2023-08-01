import numpy as np

from azcausal.core.error import JackKnife
from azcausal.core.power import Power
from azcausal.core.synth import SyntheticEffect
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.sdid import SDID


# define a function that provides a result object (with error) given a panel
def f_estimate(panel):
    estimator = SDID()
    result = estimator.fit(panel)
    estimator.error(result, JackKnife())
    return result


if __name__ == "__main__":

    # get a panel where no units are treated
    panel = CaliforniaProp99().panel()
    outcome = panel.outcome.loc[:, ~panel.w]

    # the type of synthetic treatment tested for power
    mode = 'perc'

    # the average treatment effect to be simulated. Here, we test 10% (because mode = 'perc')
    att = -0.1

    # the number of time periods treated
    n_treat_times = 11

    # Number of units to be treated
    n_treat_units = 1

    treatment = np.zeros_like(outcome.values)
    treatment[-n_treat_times:, :n_treat_units] = att

    # create synthetic panels where the last 8 time s
    synth_effect = SyntheticEffect(outcome, treatment, mode=mode)

    # the number of runs we run to test the power
    n_samples = 31

    # run the power analysis
    power = Power(f_estimate, synth_effect, n_samples, conf=90).run()

    print("(-):", power['power']['-'])
    print("(+/-):", power['power']['+/-'])
    print("(+):", power['power']['+'])

