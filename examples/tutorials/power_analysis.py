from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from azcausal.core.parallelize import Joblib
from azcausal.core.scenario import Scenario, Evaluator, power
from azcausal.core.synth import SyntheticEffect
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DIDRegressor


# define a function that provides a result object (with error) given a panel
def f_estimate_did_regr(panel):
    return DIDRegressor().fit(panel)


if __name__ == "__main__":

    f_estimate = f_estimate_did_regr

    parallelize = Joblib()

    panel = CaliforniaProp99().panel()
    outcome = panel.outcome.loc[:, ~panel.w]

    # the number of samples for the power (please increase for higher accuracy)
    n_samples = 11

    # create the intervention matrix
    intervention = np.zeros_like(outcome.values).astype(int)
    intervention[-10:, :2] = 1

    df = []

    for att in np.linspace(0.0, 0.30, 7):

        # create the treatment matrix for the effect
        treatment = intervention * att * -1

        # create synthetic panels where the last 8 time s
        synth_effect = SyntheticEffect(outcome, treatment, intervention=intervention, mode='perc')

        # run the power analysis
        results = Scenario(f_estimate, synth_effect, f_eval=Evaluator(conf=90)).run(n_samples, parallelize=parallelize)

        # calculate the power from the results
        pw = power(results)

        print(f"Percentage Treatment Effect {att:.3f} | (-): {pw['-']:.3%}")

        df.append(dict(att=att, **pw))

    df = pd.DataFrame(df)

    print("")
    print(df)

    plt.subplots(1, 1, figsize=(12, 4))
    plt.title("Power Analysis")
    plt.plot(100 * df['att'], df['-'], "-o", color="blue", label='-')
    plt.plot(100 * df['att'], df['+'], "-o", color="green", label='+')
    plt.axhline(1.0, color="black", alpha=0.15)
    plt.axhline(0.9, color="black", alpha=0.15, linestyle='--')
    plt.axhline(0.0, color="black", alpha=0.15)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("ATT (%)")
    plt.ylabel("Statistical Power")
    plt.legend()
    plt.show()
