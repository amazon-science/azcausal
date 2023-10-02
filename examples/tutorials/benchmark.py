import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import RandomState

from azcausal.core.benchmark import Benchmark
from azcausal.core.parallelize import Joblib
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID, DIDRegressor
from azcausal.estimators.panel.sdid import SDID
from azcausal.estimators.panel.snn import SNN


def f_estimate_did(panel):
    return DID().fit(panel)


def f_estimate_did_regr(panel):
    return DIDRegressor().fit(panel)


def f_estimate_sdid(panel):
    return SDID().fit(panel)


def f_estimate_snn(panel):
    return SNN(cluster_size=50).fit(panel)


if __name__ == "__main__":

    panel = CaliforniaProp99().panel()

    random_state = RandomState(42)

    benchmark = Benchmark(panel,
                          30,
                          random_state=RandomState(42),
                          func_att=lambda: random_state.uniform(0, 0.25),
                          func_n_treat=lambda: int(random_state.choice(np.arange(1, panel.n_units() // 2))),
                          func_n_post=lambda: panel.n_post
                          )

    scenarios = list(benchmark.scenarios())

    # the true effect of each scenario
    dy = [(int(s.tags['seed']), float(s.result().effect.percentage().value)) for s in scenarios]
    att = pd.DataFrame(dy, columns=['instance', 'true_att'])

    # the estimators to run
    estimators = [
        ('did', f_estimate_did),
        ('sdid', f_estimate_sdid),
        ('snn', f_estimate_snn)
    ]

    # the final data frame with the results
    df = []

    # for each of the estimator functions
    for name, estimator in estimators:
        print(name)


        def f(s):
            result = estimator(s.panel())
            effect = result.effect.percentage()

            entry = dict(instance=s.tags['seed'], att=effect.value, se=effect.se)

            return entry

        results = Joblib(n_jobs=16, progress=True).run(f, scenarios)

        df.append(pd.DataFrame(results).assign(estimator=name))

    df = pd.concat(df).merge(att, on='instance').assign(error=lambda x: np.abs(x['true_att'] - x['att']))
    print(df)

    dxx = (df
           .groupby('estimator')
           .agg({'error': ('mean', 'std', 'median')})
           )

    sns.boxplot(data=df, x='error', y='estimator')
    plt.show()

    sns.lineplot(data=df, x="true_att", y="true_att", color="black", alpha=0.5)
    sns.scatterplot(data=df, x="true_att", y="att", hue="estimator")
    plt.show()

    print(dxx)
