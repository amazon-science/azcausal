import numpy as np
import pandas as pd
from scipy import stats

from azcausal.core.error import JackKnife
from azcausal.core.panel import CausalPanel
from azcausal.data import CaliforniaProp99
from azcausal.estimators.panel.did import DID
from azcausal.estimators.panel.sdid import SDID
from azcausal.experimental.sdid2 import InstanceFactory, MyPanel, append, jackknife, sdid_weights_omega, sdid_weights_lambd, bootstrap_se, \
    az_sdid, az_did, az_transform, Instance
from azcausal.util import to_panels

if __name__ == "__main__":
    df = CaliforniaProp99().df()

    # create the panel data from the frame and define the causal types
    data = to_panels(df, 'Year', 'State', ['PacksPerCapita', 'treated'])
    ctypes = dict(outcome='PacksPerCapita', time='Year', unit='State', intervention='treated')

    # Step 1: Parsing the panel and remove treated units (or times0
    panel = CausalPanel(data).setup(**ctypes)
    panel.intervention["Wyoming"].loc[1989:] = 1

    # estimator = DID()
    # result = estimator.fit(panel)
    # estimator.error(result, JackKnife())
    # print(result.info['did'])
    # print(result.summary(title='DID'))

    estimator = SDID()
    result = estimator.fit(panel)
    estimator.error(result, JackKnife())
    print(result.summary(title='SDID'))


    Y = panel["outcome"].values

    n_treat = panel.treat.sum()
    n_post = panel.post.sum()

    YY = np.hstack([Y[:, ~(panel.treat)], Y[:, panel.treat]])
    u = np.array(panel.units())
    ulabel = np.concatenate([u[~(panel.treat)], u[panel.treat]])
    test = MyPanel(YY, n_treat, n_post, ulabel=ulabel)


    estimator = SDID()
    result = estimator.fit(az_transform(test))
    estimator.error(result, JackKnife())
    print(result.summary(title='SDID'))



    result = az_sdid(test)

    omega = result['omega']
    lambd = result['lambd']
    print("SDID Jackknife", jackknife(test, omega=omega, lambd=lambd))

    # print(did_jackknife2(test))
    # print(did_jackknife(test))
    # print(test.predict(), did_jackknife(test))

    # did_slow(test)

    Y = Y[:, ~(panel.treat)]
    units = np.array(panel.units())[~(panel.treat)]

    # Step 2: Create simulations with different types of treatment effects and seeds
    n_treat = 5
    n_post = 12

    simulations = []
    for att in np.linspace(-0.3, 0.3, 13):

        for seed in range(100):
            random_state = np.random.RandomState(seed)
            iunits = random_state.permutation(Y.shape[1])

            Sp = Y[:, iunits]
            ulabel = units[iunits]

            S = np.copy(Sp)
            S[-n_post:, -n_treat:] = (1 + att) * S[-n_post:, -n_treat:]

            sim = MyPanel(S, n_treat, n_post, Yp=Sp, ulabel=ulabel)

            simulations.append(dict(att=att, panel=sim))

    dp = pd.DataFrame.from_records(simulations)
    dp = append(dp, lambda x: x['panel'].effect(prefix='true_'))

    # dp = dp.iloc[:10]
    #
    # dp = append(dp, lambda x: az_sdid(x['panel'], prefix='az_'))
    # dp = append(dp, lambda x: sdid_jackknife(x['panel'], x['az_omega'], lambd=x['az_lambd'], prefix='pred_'))
    #
    # print(dp[['pred_avg_error', 'az_avg_error']])
    #
    # exit()
    #



    # dp = apply(dp, lambda x: x['panel'].predict(prefix='pred_'))
    # dp['pred_avg_error'] = dp['panel'].map(did_jackknife)
    # dte = dp

    # dp = apply(dp, lambda x: f(x['panel'], prefix='pred_'))


    dp['instance'] = dp['panel'].map(lambda x: Instance(x))
    omega = sdid_weights_omega(dp, ['panel'], 'instance')
    lambd = sdid_weights_lambd(dp, ['panel'], 'instance')
    df = (dp
          .merge(omega, left_on=['panel', 'instance'], right_index=True)
          .merge(lambd, left_on=['panel', 'instance'], right_index=True)
          )

    dte = append(df, lambda x: jackknife(x['instance'], omega=x['omega'], lambd=x['lambd'], prefix='pred_'))

    # dx = pd.concat(InstanceFactory(p, 5).run() for p in dp['panel'])
    # df = dp.merge(dx, on='panel')
    #
    # omega = sdid_weights_omega(df, ['panel'], 'instance')
    # lambd = sdid_weights_lambd(df, ['panel'], 'instance')
    # df = (df
    #       .merge(omega, left_on=['panel', 'instance'], right_index=True)
    #       .merge(lambd, left_on=['panel', 'instance'], right_index=True)
    #       )
    #
    # dte = append(df, lambda x: sdid_jackknife(x['instance'], omega=x['omega'], lambd=x['lambd'], prefix='pred_'))

    # df = apply(df, lambda x: x['instance'].effect(prefix='pred_'))
    # df = apply(df, lambda x: x['instance'].predict(prefix='pred_'))

    # dte = (df
    #        .query("name == 'BOOTSTRAP'")
    #        .groupby(['att', 'panel'])
    #        .aggregate(true_avg_te=('true_avg_te', 'mean'),
    #                   true_perc_te=('true_perc_te', 'mean'),
    #                   pred_avg_te=('pred_avg_te', 'mean'),
    #                   pred_avg_error=('pred_avg_te', bootstrap_se),
    #                   # pred_avg_error=('pred_avg_te', lambda x: scipy.stats.sem(x)),
    #                   pred_perc_te=('pred_perc_te', 'mean'),
    #                   )
    #        )

    conf = 0.90
    alpha = 1 - conf
    z_critical = stats.norm.ppf(1 - alpha / 2)
    dx = (dte
          .assign(pred_avg_ci_lb=lambda dx: dx['pred_avg_te'] - z_critical * dx['pred_avg_error'])
          .assign(pred_avg_ci_ub=lambda dx: dx['pred_avg_te'] + z_critical * dx['pred_avg_error'])
          .assign(pred_sign=lambda dx: np.where(dx['pred_avg_ci_lb'] > 0, '+', np.where(dx['pred_avg_ci_ub'] < 0, '-', '+/-')))
          .assign(true_in_ci=lambda dx: dx['true_avg_te'].between(dx['pred_avg_ci_lb'], dx['pred_avg_ci_ub']))
          .assign(perc_te_error=lambda dx: dx['pred_perc_te'] - dx['true_perc_te'])
          )

    # get the power and coverage for each group now
    pw = dx.groupby('att')['pred_sign'].value_counts(normalize=True).unstack('pred_sign').fillna(0.0)
    for col in ['-', '+', '+/-']:
        if col not in pw:
            pw[col] = 0

    coverage = dx.groupby('att')['true_in_ci'].mean()
    error = dx.groupby('att').aggregate(mean=('perc_te_error', 'mean'), se=('perc_te_error', 'sem'))

    # Plot

    import matplotlib.pyplot as plt

    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    fig.suptitle(f'', fontsize=16)

    top.plot(pw.index, pw['-'], "-o", color="red", label='-')
    top.plot(pw.index, pw['+'], "-o", color="green", label='+')
    top.plot(pw.index, pw['+/-'], "-o", color="black", label='+/-', alpha=0.5)
    top.axhline(1.0, color="black", alpha=0.15)
    top.axhline(conf, color="black", alpha=0.15, linestyle='--')
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
