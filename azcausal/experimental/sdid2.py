import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.random import RandomState
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


def forward(A, w, b, m=None):
    y_hat = predict(A, w)
    loss = F.mse_loss(y_hat, b, reduction='none')

    if m is not None:
        loss = (loss * m.to(A.dtype)).sum(dim=1) / m.sum(dim=1)
    else:
        loss = loss.mean(dim=1)
    return loss


class MySolver(object):

    def __init__(self):
        super().__init__()
        self.data = []

    def add(self, A, b, m, other=None, tags=None):
        self.data.append(dict(A=A, b=b, m=m, other=other, tags=tags))

    def run(self, holdout):

        def f(collection, name):
            return [x[name] for x in collection]

        A = torch.tensor(np.array(f(self.data, 'A')), dtype=torch.float32)
        b = torch.tensor(np.array(f(self.data, 'b')), dtype=torch.float32)
        m = torch.tensor(np.array(f(self.data, 'm')), dtype=torch.bool)

        At, bt = A[:, ~holdout], b[:, ~holdout]
        n_samples, n_times, n_units = At.shape

        Ah, bh = A[:, holdout], b[:, holdout]

        x = torch.full((n_samples, n_units), 0.1, requires_grad=True, dtype=torch.float32)
        w = F.softmax(x, dim=1)

        # optimizer = torch.optim.SGD([x], lr=0.01)
        optimizer = torch.optim.Adam([x], lr=0.001)
        n_epochs = 3_000

        trace = []

        for epoch in range(n_epochs):
            w = F.softmax(x, dim=1)
            trn_loss = forward(At, w, bt, m)
            total_loss = trn_loss.mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:

                with torch.no_grad():
                    vld_loss = forward(At, w, bt, ~m)
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
# Panel
# ---------------------------------------------------------------------------------------------------------



class MyPanel(object):

    def __init__(self, Y, n_treat, n_post, tags=None):
        super().__init__()
        self.Y = Y
        self.n_treat = n_treat
        self.n_post = n_post
        self.tags = tags

    def matrix_by_unit(self):
        Y = self.Y

        YY = Y - Y[self.spre].mean(axis=0)
        A = YY[:, self.scontr]
        b = YY[:, self.streat].mean(axis=1)

        return A, b

    def matrix_by_time(self):
        Y, n_treat, n_post = self.Y, self.n_treat, self.n_post

        Y = Y.T
        YY = Y - Y[self.scontr].mean(axis=0)
        A = YY[:, self.spre]
        b = YY[:, self.spost].mean(axis=1)

        return A, b


    @property
    def n_units(self):
        return self.Y.shape[1]

    @property
    def n_times(self):
        return self.Y.shape[0]

    @property
    def n_pre(self):
        return self.n_times - self.n_post

    @property
    def n_contr(self):
        return self.n_units - self.n_treat

    @property
    def counts(self):
        return self.n_pre, self.n_post, self.n_treat, self.n_contr

    @property
    def streat(self):
        return slice(-self.n_treat, None)

    @property
    def scontr(self):
        return slice(None, -self.n_treat)

    @property
    def spre(self):
        return slice(None, -self.n_post)

    @property
    def spost(self):
        return slice(-self.n_post, None)

    @property
    def slices(self):
        return self.scontr, self.streat, self.spre, self.spost

    @property
    def sunits(self):
        return self.scontr, self.streat

    @property
    def stime(self):
        return self.spre, self.spost

    @property
    def iunits(self):
        u = np.arange(self.n_units)
        contr, treat = self.sunits
        return u[contr], u[treat]

    @property
    def itreat(self):
        return self.iunits[1]

    @property
    def icontr(self):
        return self.iunits[0]

    @property
    def itime(self):
        t = np.arange(self.n_times)
        pre, post = self.stime
        return t[pre], t[post]

    @property
    def mpost(self):
        mask = np.full(self.n_times, False, dtype=bool)
        _, post = self.stime
        mask[post] = True
        return mask

    @property
    def mtreat(self):
        mask = np.full(self.n_units, False, dtype=bool)
        _, treat = self.sunits
        mask[treat] = True
        return mask

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


def create_mask(n, p, random_state):
    m = random_state.random(n) <= p
    pos, neg = random_state.choice(np.arange(n), replace=False, size=2)
    m[pos] = True
    m[neg] = False
    return m


class Instance(MyPanel):

    def __init__(self, panel, itreat=None, icontr=None, mu=None, mt=None, tags=None):


        itreat = itreat if itreat is not None else panel.itreat
        icontr = icontr if icontr is not None else panel.icontr

        Y = panel.Y
        YY = np.hstack([Y[:, icontr], Y[:, itreat]])

        super().__init__(YY, len(itreat), panel.n_post)
        self.panel = panel
        self.mu = mu if mu is not None else np.full(len(icontr), True)
        self.mt = mt if mt is not None else np.full(panel.n_pre, True)
        self.tags = tags if tags is not None else {}

    def matrix_by_unit(self):
        A, b = super().matrix_by_unit()
        return A, b, self.mt

    def matrix_by_time(self):
        A, b = super().matrix_by_time()
        return A, b, self.mu


class InstanceFactory:


    def __init__(self, panel, n):
        super().__init__()
        self.panel = panel
        self.n = n

    def sample(self, seed):
        panel = self.panel
        n_pre, n_post, n_treat, n_contr = panel.counts

        random_state = RandomState(seed)
        mu = create_mask(n_contr, 0.5, random_state)
        mt = create_mask(n_pre, 0.5, random_state)

        default = Instance(panel, mt=mt, mu=mu, tags=dict(name='RESULT', seed=seed))

        # the placebo treatment select from control
        itreat = np.random.choice(panel.icontr, replace=False, size=n_treat)

        # the remaining pool to sample from
        pool = [i for i in panel.icontr if i not in itreat]
        icontr = np.random.choice(pool, replace=True, size=n_contr)

        bootstrap = Instance(panel, icontr=icontr, mt=mt, mu=mu, tags=dict(name='BOOTSTRAP', seed=seed))
        placebo = Instance(panel, icontr=icontr, itreat=itreat, mt=mt, mu=mu, tags=dict(name='PLACEBO', seed=seed))

        return [default, bootstrap, placebo]

    def run(self):
        data = []
        index = []
        for k in range(self.n):
            for instance in self.sample(k):
                index.append(instance.tags)
                data.append(instance)
        dx = pd.DataFrame(index)
        keys = dx.columns
        dx['instance'] = data
        return dx.set_index(list(keys))['instance']

def sdid_by_units(panel, instances):

    sunits = MySolver()
    for instance in instances:
        A, b, m = instance.matrix_by_unit()
        sunits.add(A, b, m, tags=instance.tags)

    return sunits.run(panel.mpost)

def sdid_by_time(panel, instances):

    sunits = MySolver()
    for instance in instances:
        A, b, m = instance.matrix_by_time()
        sunits.add(A, b, m, tags=instance.tags)

    return sunits.run(panel.mtreat)

def sdid_weights(panel):

    factory = InstanceFactory(panel, 10)
    instances = factory.run()

    _, utrace = sdid_by_units(panel, instances)
    _, ttrace = sdid_by_time(panel, instances)

    return instances, utrace, ttrace


def plot(utrace, ttrace):
    utrace['att'] = np.vstack(utrace['tst_error']).mean(axis=1)
    # strace['att_slow'] = [scenario.att(omega=omega)['att'] for scenario, omega in zip(strace['scenario'], strace['w'])]

    utrace.query("name == 'PLACEBO'").groupby('epoch')[['trn_loss', 'vld_loss', 'tst_loss']].mean().plot()
    plt.title(f"[UNITS] PLACEBO")
    plt.show()

    epoch = utrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().argmin()
    omega = (utrace
             .query(f"epoch == {epoch}")
             .set_index(['name', 'seed'])
             ['w']
             .to_frame('omega')
             )


    ttrace.query("name == 'PLACEBO'").groupby('epoch')[['trn_loss', 'vld_loss', 'tst_loss']].mean().plot()
    plt.title(f"[TIME] PLACEBO")
    plt.show()

    ttrace = ttrace.merge(omega, left_on=['name', 'instance'], right_index=True)
    ttrace['lambd'] = ttrace['w']

    ttrace['att'] = dx.apply(lambda x: x['instance'].att(omega=x['omega'], lambd=x['lambd']), axis=1)

    trace = (pd.concat([
        utrace.query(f"epoch <= {epoch}"),
        ttrace])
             .drop(columns=['w'])
             .assign(total_epoch=lambda dx: np.where(dx['mode'] == 'UNITS', dx['epoch'], dx['epoch'] + 1 + epoch))
             )

    epoch = ttrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().argmin()
    ans = trace.query("mode == 'TIME'").query(f"epoch == {epoch}")

    conf = 0.95
    z_critical = stats.norm.ppf(q=conf)

    for mode, color in [('RESULT', 'blue'),
                        ('BOOTSTRAP', 'orange'),
                        ('PLACEBO', 'green')
                        ]:
        dx = trace.query(f"name == '{mode}'").groupby('total_epoch')['att'].aggregate(['mean', 'std'])
        plt.plot(dx.index, dx['mean'], color=color)

        lb = dx['mean'] - z_critical * dx['std']
        plt.plot(lb.index, lb, color=color, alpha=0.2)

        ub = dx['mean'] + z_critical * dx['std']
        plt.plot(ub.index, ub, color=color, alpha=0.2)

        plt.fill_between(dx.index,
                         lb,
                         ub,
                         color=color,
                         alpha=0.2,
                         label=mode
                         )

    plt.legend()
    plt.show()

    return ans


def se(values):
    n = len(values)
    return np.sqrt((n - 1) / n) * np.std(values, ddof=1)


def sdid2(Y, n_treat, n_post) -> dict:
    panel = MyPanel(Y, n_treat, n_post)
    instances, utrace, ttrace = sdid_weights(panel)

    epoch = utrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().idxmin()
    omega = (utrace
             .query(f"epoch == {epoch}")
             .set_index(['name', 'seed'])
             ['w']
             .to_frame('omega')
             )

    epoch = ttrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().idxmin()
    lambd = (ttrace
             .query(f"epoch == {epoch}")
             .set_index(['name', 'seed'])
             ['w']
             .to_frame('lambd')
             )

    dx = (instances.to_frame('instance')
          .join(lambd)
          .join(omega)
          )


    dx['att'] = dx.apply(lambda x: x['instance'].att(omega=x['omega'], lambd=x['lambd'])['att'], axis=1)

    te = np.array(dx.query("name == 'BOOTSTRAP'")['att'])
    return dict(att=np.mean(te), se=se(te))


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

        observed = YY[-n_post:].mean(axis=0)[-n_treat:].mean()
        scale = n_treat * n_post

        att = Effect(sdid["att"], se=sdid['se'], observed=observed, scale=scale, data=sdid, name="ATT")
        return Result(dict(att=att), data=panel, info=sdid, estimator=self)

    def plot(self, result, title=None, CF=True, C=True, show=True):
        return sdid_plot(result.effect, title=title, CF=CF, C=C, show=show)
