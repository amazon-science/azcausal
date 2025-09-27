import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

from azcausal.core.effect import Effect
from azcausal.core.panel import Panel
from azcausal.core.result import Result
from azcausal.estimators.panel.did import DID
from azcausal.estimators.panel.sdid import sdid_plot


# ---------------------------------------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------------------------------------


def predict(A, w):
    return torch.bmm(A, w.unsqueeze(2)).squeeze(2)


def forward(A, w, b, M=None):
    y_hat = predict(A, w)
    loss = F.mse_loss(y_hat, b, reduction='none')

    if M is not None:
        loss = (loss * M.to(A.dtype)).sum(dim=1) / M.sum(dim=1)
    else:
        loss = loss.mean(dim=1)
    return loss


def create_mask(n_samples, n_times):
    # Generate the initial random matrix
    M = torch.rand((n_samples, n_times)) < 0.5

    # Check each row and modify if necessary
    for i in range(n_samples):
        row = M[i]
        if not (1 in row and 0 in row):
            # If the row doesn't have both 1 and 0, modify it
            if torch.all(row == 0):
                # If all zeros, change a random element to 1
                random_index = torch.randint(0, n_times, (1,))
                M[i, random_index] = 1
            elif torch.all(row == 1):
                # If all ones, change a random element to 0
                random_index = torch.randint(0, n_times, (1,))
                M[i, random_index] = 0

    return M


class MySolver(object):

    def __init__(self):
        super().__init__()
        self.data = []

    def add(self, A, b, other=None, tags=None):
        self.data.append(dict(A=A, b=b, other=other, tags=tags))

    def run(self, holdout):

        dtype = torch.float32

        def f(collection, name):
            return [x[name] for x in collection]

        A = torch.tensor(np.array(f(self.data, 'A')), dtype=dtype)
        b = torch.tensor(np.array(f(self.data, 'b')), dtype=dtype)

        At, bt = A[:, ~holdout], b[:, ~holdout]
        n_samples, n_times, n_units = At.shape
        M = create_mask(n_samples, n_times)

        Ah, bh = A[:, holdout], b[:, holdout]

        x = torch.full((n_samples, n_units), 0.1, requires_grad=True, dtype=dtype)
        w = F.softmax(x, dim=1)

        # optimizer = torch.optim.SGD([x], lr=0.01)
        optimizer = torch.optim.Adam([x], lr=0.001)
        n_epochs = 3_000

        trace = []

        for epoch in range(n_epochs):
            w = F.softmax(x, dim=1)
            trn_loss = forward(At, w, bt, M)
            total_loss = trn_loss.mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                vld_loss = forward(At, w, bt, ~M)
                tst_loss = forward(Ah, w, bh)
                tst_error = (bh - predict(Ah, w))

            tst_error = tst_error.detach().numpy()
            ww = w.detach().numpy()

            for k in range(n_samples):
                trace.append(dict(epoch=epoch,
                                  trn_loss=trn_loss[k].item(),
                                  vld_loss=vld_loss[k].item(),
                                  tst_loss=tst_loss[k].item(),
                                  tst_error=tst_error[k],
                                  w=ww[k],
                                  **self.data[k]['tags']))

        return w, pd.DataFrame(trace)


# ---------------------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------------------


def slice_from_count(n_treat, n_post):
    contr = slice(None, -n_treat)
    treat = slice(-n_treat, None)
    pre = slice(None, -n_post)
    post = slice(-n_post, None)
    return contr, treat, pre, post


def index_from_unit_count(n_units, n_treat):
    x = np.arange(n_units)
    return x[:-n_treat], x[-n_treat:]


class Scenario(object):

    def __init__(self, Y, n_treat, n_post, tags=None):
        super().__init__()
        self.Y = Y
        self.n_treat = n_treat
        self.n_post = n_post
        self.tags = tags if tags is not None else {}
        self.tags['scenario'] = self

    def matrix_by_unit(self):
        Y, n_treat, n_post = self.Y, self.n_treat, self.n_post
        contr, treat, pre, _ = self.slices

        YY = Y - Y[pre].mean(axis=0)
        A = YY[:, contr]
        b = YY[:, treat].mean(axis=1)

        return A, b

    def matrix_by_time(self):
        Y, n_treat, n_post = self.Y, self.n_treat, self.n_post
        contr, _, pre, post = self.slices

        Y = Y.T
        YY = Y - Y[contr].mean(axis=0)
        A = YY[:, pre]
        b = YY[:, post].mean(axis=1)

        return A, b

    @property
    def slices(self):
        return slice_from_count(self.n_treat, self.n_post)

    @property
    def n_pre(self):
        return self.Y.shape[0] - self.n_post

    @property
    def n_contr(self):
        return self.Y.shape[1] - self.n_treat

    def att(self, omega=None, lambd=None):
        Y = self.Y
        contr, treat, pre, post = self.slices

        vcontr = Y[:, contr].mean(axis=0) if omega is None else Y[:, contr] @ omega
        post_contr = vcontr[post].mean()
        pre_contr = vcontr[pre].mean() if lambd is None else vcontr[pre] @ lambd

        vtreat = Y[:, treat].mean(axis=1)
        post_treat = vtreat[post].mean()
        pre_treat = vtreat[pre].mean() if lambd is None else vtreat[pre] @ lambd

        att = (post_treat - pre_treat) - (post_contr - pre_contr)
        return dict(att=att, pre_treat=pre_treat, post_treat=post_treat, pre_contr=pre_contr, post_contr=post_contr)


def plot(trace, prefix=""):
    # trace.query("name == 'RESULT'").groupby('epoch')[['trn_loss', 'vld_loss']].mean().plot()
    # plt.title(f"{prefix} RESULT")
    # plt.show()
    #
    #
    # trace.query("name == 'BOOTSTRAP'").groupby('epoch')[['trn_loss', 'vld_loss']].mean().plot()
    # plt.title(f"{prefix} BOOTSTRAP")
    # plt.show()

    trace.query("name == 'PLACEBO'").groupby('epoch')[['trn_loss', 'vld_loss', 'tst_loss']].mean().plot()
    plt.title(f"{prefix} PLACEBO")
    plt.show()

    for mode in ['RESULT', 'BOOTSTRAP', 'PLACEBO']:

        dx = trace.query(f"name == '{mode}'").groupby('epoch')['att'].aggregate(['mean', 'std'])
        plt.plot(dx.index, dx['mean'], label='Mean', color='blue')

        # Plot the shaded area for +/- 1 standard deviation
        conf = 0.95
        z_critical = stats.norm.ppf(q=conf)
        plt.fill_between(dx.index,
                         dx['mean'] - z_critical * dx['std'],
                         dx['mean'] + z_critical * dx['std'],
                         color='blue', alpha=0.2, label=f'Mean ± Conf ({conf})')

        plt.title(f"{prefix} {mode} (ATT)")
        plt.show()

    # sns.lineplot(x="epoch", y="att", data=trace.query("name == 'BOOTSTRAP'"))
    # plt.title(f"{prefix} BOOTSTRAP (ATT)")
    # plt.show()


def sdid_weights(Y, n_treat, n_post):
    n_times, n_units = Y.shape

    contr, treat, pre, post = slice_from_count(n_treat, n_post)
    icontr, itreat = index_from_unit_count(n_units, n_treat)

    scenarios = []

    # A, b = solve_by_unit(Y, n_treat, n_post)
    # At, bt = solve_by_time(Y, n_treat, n_post)
    for k in range(100):
        scenarios.append(Scenario(Y, n_treat, n_post, tags=dict(name='RESULT', sample=k)))

    n_bootstraps = 100
    for k in range(n_bootstraps):
        zt = treat
        # zt = np.random.choice(itreat, replace=True, size=n_treat)
        zc = np.random.choice(icontr, replace=True, size=n_units - n_treat)

        YY = np.hstack([Y[:, zc], Y[:, zt]])
        scenarios.append(Scenario(YY, n_treat, n_post, tags=dict(name='BOOTSTRAP', sample=k)))

    n_placebo = 100
    for k in range(n_placebo):
        zt = np.random.choice(icontr, replace=False, size=n_treat)
        pitreat = [i for i in icontr if i not in zt]
        zc = np.random.choice(pitreat, replace=True, size=n_units - n_treat)

        YY = np.hstack([Y[:, zc], Y[:, zt]])
        scenarios.append(Scenario(YY, n_treat, n_post, tags=dict(name='PLACEBO', sample=k)))

    sunits = MySolver()
    for scenario in scenarios:
        A, b = scenario.matrix_by_unit()
        sunits.add(A, b, tags=scenario.tags)

    mask = np.full(n_times, False, dtype=bool)
    mask[post] = True
    X, strace = sunits.run(mask)
    strace['att'] = np.vstack(strace['tst_error']).mean(axis=1)
    # strace['att_slow'] = [scenario.att(omega=omega)['att'] for scenario, omega in zip(strace['scenario'], strace['w'])]

    plot(strace, prefix="UNIT ")
    epoch = strace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().argmin()
    omega = strace.set_index(['name', 'epoch']).xs('RESULT').xs(epoch).iloc[0]['w']

    stimes = MySolver()
    for scenario in scenarios:
        A, b = scenario.matrix_by_time()
        stimes.add(A, b, tags=scenario.tags)

    mask = np.full(n_units, False, dtype=bool)
    mask[treat] = True
    _, mtrace = stimes.run(mask)
    mtrace['att'] = [scenario.att(omega=omega, lambd=lambd)['att'] for scenario, lambd in zip(mtrace['scenario'], mtrace['w'])]
    plot(mtrace, prefix="TIME ")

    epoch = mtrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().argmin()
    lambd = mtrace.set_index(['name', 'epoch']).xs('RESULT').xs(epoch).iloc[0]['w']

    trace = pd.concat([strace.assign(mode='UNITS'), mtrace.assign(mode='TIME')])

    return omega, lambd, trace


def sdid2(Y, n_treat, n_post) -> dict:
    omega, lambd, trace = sdid_weights(Y, n_treat, n_post)

    att = Scenario(Y, n_treat, n_post).att(omega, lambd)

    return dict(omega=omega, lambd=lambd, trace=trace, **att)


class SDID2(DID):

    def fit(self,
            panel: Panel,
            lambd: np.ndarray = None,
            omega: np.ndarray = None,
            **kwargs):
        Y = panel["outcome"].values
        YY = np.hstack([Y[:, ~(panel.treat)], Y[:, panel.treat]])
        n_treat = panel.treat.sum()
        n_post = panel.post.sum()

        sdid = sdid2(YY, n_treat, n_post)

        scale = panel.n_interventions()
        att = Effect(sdid["att"], observed=sdid["post_treat"], scale=scale, data=sdid, name="ATT")
        return Result(dict(att=att), data=panel, info=sdid, estimator=self)

    def plot(self, result, title=None, CF=True, C=True, show=True):
        return sdid_plot(result.effect, title=title, CF=CF, C=C, show=show)
