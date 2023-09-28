# import numpy as np
# import pandas as pd
#
# from azcausal.core.effect import Effect
# from azcausal.core.result import Result
# from azcausal.estimators.panel.did import did_simple
# from azcausal.experimental.rdid import SDID
#
#
# def find_time_weights(solver, Y_pre_contr, Y_post_contr, noise, lambd=None, eta=1e-06):
#     A = Y_pre_contr
#     b = Y_post_contr.mean(axis=1)
#
#     lambd = solver["lambd"](A, b, eta, noise=noise, x0=lambd)['x']
#
#     return lambd
#
#
# def find_unit_weights(solver, Y_pre_contr, Y_pre_treat, noise, n_post, omega=None):
#     n_treat = len(Y_pre_treat)
#
#     eta = (n_treat * n_post) ** (1 / 4)
#     A = Y_pre_contr.T
#     b = Y_pre_treat.mean(axis=0)
#
#     omega = solver(A, b, eta, noise=noise, x0=omega)["x"]
#
#     return omega
#
#
# def find_weights(solvers, Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr, lambd=None, omega=None, optimize=True):
#     n_post = len(Y_post_contr.T)
#
#     noise = np.diff(Y_pre_contr, axis=1).std(ddof=1)
#
#     if optimize and "lambd" in solvers:
#         lambd = find_time_weights(solvers["lambd"], Y_pre_contr, Y_post_contr, noise, lambd=lambd)
#
#     if optimize and "omega" in solvers is not None:
#         omega = find_unit_weights(solvers["omega"], Y_pre_contr, Y_pre_treat, noise, n_post, omega=omega)
#
#     # calculate the synthetic control outcome pre and post
#     Y_pre_synth = Y_pre_contr.T @ omega
#     Y_post_synth = Y_post_contr.T @ omega
#
#     # pre weighted by lambda
#     pre_sc = Y_pre_synth @ lambd
#     pre_treat = Y_pre_treat.mean(axis=0) @ lambd
#
#     # the average treatment effect on the treated
#     did = did_simple(pre_sc, Y_post_synth.mean(), pre_treat, Y_post_treat.mean())
#
#     return did, omega, lambd
#
#
# class RSDID(SDID):
#
#     def fit(self, panel, lambd=None, omega=None, optimize=True):
#         Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr = panel.Y_as_block(trim=True)
#         (n_pre, n_post), _ = panel.counts()
#
#         noise = np.diff(Y_pre_contr, axis=1).std(ddof=1)
#         lambd = find_time_weights(self.solver, Y_pre_contr, Y_post_contr, noise, lambd=lambd, optimize=optimize)
#
#         bagging = []
#
#         for k in range(11):
#             perm = np.random.permutation(n_pre)
#             trn, vld = perm[:-n_post], perm[-n_post:]
#
#             Y_trn_treat, Y_trn_contr = Y_pre_treat[:, trn], Y_pre_contr[:, trn]
#             Y_vld_treat, Y_vld_contr = Y_pre_treat[:, vld], Y_pre_contr[:, vld]
#
#             did, omega, _ = find_weights(self.solver, Y_trn_treat, Y_vld_treat, Y_trn_contr, Y_vld_contr)
#             placebo = did['att']
#
#             did, _, _ = find_weights(dict(), Y_pre_treat, Y_post_treat, Y_pre_contr, Y_post_contr, lambd=lambd, omega=omega, optimize=False)
#             att = did['att']
#
#             entry = dict(placebo=placebo, omega=omega, att=att)
#             bagging.append(entry)
#
#         data = pd.DataFrame(bagging)
#         att = data['att'].mean()
#         se = data['att'].std()
#
#         import seaborn as sns
#         import matplotlib.pyplot as plt
#
#         fig, ax = plt.subplots(figsize=(12, 4))
#         sns.histplot(data=data[['placebo', 'att']].unstack().reset_index(), x=0, hue='level_0', ax=ax, bins=11)
#         plt.show()
#
#         effect = Effect(att, se=se, observed=Y_post_treat.mean(), multiplier=panel.n_interventions(), data=data)
#         return Result(dict(att=effect), panel=panel, estimator=self)
