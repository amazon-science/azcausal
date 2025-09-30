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
# OPTIMIZATION
# ---------------------------------------------------------------------------------------------------------


class Optimization:

    def __init__(self, n_max_epochs=3_000, algorithm=torch.optim.Adam):
        super().__init__()
        self.n_max_epochs = n_max_epochs
        self.algorithm = algorithm

    def solver(self, A, b, m, h):
        A = torch.tensor(A, dtype=torch.float32)
        b = torch.tensor(b, dtype=torch.float32)
        m = torch.tensor(m, dtype=torch.bool)
        h = torch.tensor(h, dtype=torch.bool)

        At, bt = A[:, ~h], b[:, ~h]
        Ah, bh = A[:, h], b[:, h]

        n_samples, _, n_units = At.shape
        x = torch.full((n_samples, n_units), 0.1, requires_grad=True, dtype=torch.float32)

        optimizer = self.algorithm([x], lr=0.001)

        return MySolver(At, bt, Ah, bh, x, m, optimizer, self.n_max_epochs)


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

    def __init__(self, At, bt, Ah, bh, x, m, optimizer, n_max_epochs):
        super().__init__()
        self.At, self.bt = At, bt
        self.Ah, self.bh = Ah, bh
        self.m = m
        self.x = x
        self.w = None

        self.optimizer = optimizer
        self.n_max_epochs = n_max_epochs

        self.epoch = 0
        self.total_loss = None
        self.trn_loss = None
        self.vld_loss = None
        self.tst_loss = None

    def advance(self):
        self.w = F.softmax(self.x, dim=1)
        trn_loss = forward(self.At, self.w, self.bt, self.m)
        total_loss = trn_loss.mean()

        with torch.no_grad():
            vld_loss = forward(self.At, self.w, self.bt, ~self.m)
            tst_loss = forward(self.Ah, self.w, self.bh)
            # tst_error = (bh - predict(Ah, w))

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_loss = total_loss
        self.trn_loss, self.vld_loss, self.tst_loss = trn_loss, vld_loss, tst_loss

    def run(self, callback: lambda x: x):
        for _ in range(self.n_max_epochs):
            self.advance()
            with torch.no_grad():
                callback(self)
            self.epoch += 1


# ---------------------------------------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------------------------------------


class MyPanel(object):

    def __init__(self, Y, n_treat, n_post, tags=None, Yp=None):
        super().__init__()
        self.Y = Y
        self.Yp = Yp
        self.n_treat = n_treat
        self.n_post = n_post
        self.tags = tags if tags is not None else {}

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

    def predict(self, omega=None, lambd=None):
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

    def effect(self):
        if self.Yp is not None:
            Y, Yp = self.Y, self.Yp
            te = Y[self.spost][:, self.streat] - Yp[self.spost][:, self.streat]
            return np.mean(te)


def create_mask(n, p, random_state):
    m = random_state.random(n) <= p
    pos, neg = random_state.choice(np.arange(n), replace=False, size=2)
    m[pos] = True
    m[neg] = False
    return m


class Instance(MyPanel):

    def __init__(self, panel, itreat=None, icontr=None, mu=None, mt=None, tags=None, Yp=None):
        Y = panel.Y
        itreat = itreat if itreat is not None else panel.itreat
        icontr = icontr if icontr is not None else panel.icontr

        YY = np.hstack([Y[:, icontr], Y[:, itreat]])
        if Yp is not None:
            Yp = np.hstack([Yp[:, icontr], Yp[:, itreat]])

        super().__init__(YY, len(itreat), panel.n_post, tags=tags, Yp=Yp)
        self.panel = panel
        self.mu = mu if mu is not None else np.full(len(icontr), True)
        self.mt = mt if mt is not None else np.full(panel.n_pre, True)

    def matrix_by_unit(self):
        A, b = super().matrix_by_unit()
        return dict(A=A, b=b, m=self.mt, instance=self)

    def matrix_by_time(self):
        A, b = super().matrix_by_time()
        return dict(A=A, b=b, m=self.mu, instance=self)


class InstanceFactory:

    def __init__(self, panel, n, mup=0.5, mtp=0.5):
        super().__init__()
        self.panel = panel
        self.n = n

        self.mup = mup
        self.mtp = mtp

    def sample(self, seed):
        panel = self.panel
        n_pre, n_post, n_treat, n_contr = panel.counts

        random_state = RandomState(seed)
        mu = create_mask(n_contr, self.mup, random_state)
        mt = create_mask(n_pre, self.mtp, random_state)

        # the default (no sampling) with time and units masks
        default = Instance(panel, mt=mt, mu=mu, tags=dict(name='RESULT', seed=seed))

        # select placebo treatment units from control pool
        itreat = np.random.choice(panel.icontr, replace=False, size=n_treat)

        # sample from the remaining control pool
        pool = [i for i in panel.icontr if i not in itreat]
        icontr = np.random.choice(pool, replace=True, size=n_contr)

        bootstrap = Instance(panel, icontr=icontr, mt=mt, mu=mu, tags=dict(name='BOOTSTRAP', seed=seed))
        placebo = Instance(panel, icontr=icontr, itreat=itreat, mt=mt, mu=mu, tags=dict(name='PLACEBO', seed=seed), Yp=panel.Y)

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


class Trace:

    def __init__(self, instances, ith=10):
        super().__init__()
        self.instances = instances
        self.ith = ith
        self.records = []

    def __call__(self, solver):

        if solver.epoch % self.ith == 0:
            for k, instance in enumerate(self.instances):
                self.records.append(dict(epoch=solver.epoch,
                                         instance=instance,
                                         trn_loss=solver.trn_loss[k].item(),
                                         vld_loss=solver.vld_loss[k].item(),
                                         tst_loss=solver.tst_loss[k].item(),
                                         w=solver.w[k].detach().numpy(),
                                         **instance.tags))


def to_numpy(dx, col):
    return np.vstack([x[None, :] for x in dx[col]])


def solve(dx, h):
    A, b, m = [to_numpy(dx, name) for name in ['A', 'b', 'm']]
    trace = Trace(dx['instance'], ith=10)
    Optimization().solver(A, b, m, h).run(trace)
    return pd.DataFrame(trace.records)


def sdid_weights(panel):
    factory = InstanceFactory(panel, 100)
    instances = factory.run()

    dunits = pd.DataFrame([instance.matrix_by_unit() for instance in instances])
    utrace = solve(dunits, panel.mpost)

    dtime = pd.DataFrame([instance.matrix_by_time() for instance in instances])
    ttrace = solve(dtime, panel.mtreat)

    return instances, utrace, ttrace


def plot(utrace, ttrace):
    # utrace['att'] = np.vstack(utrace['tst_error']).mean(axis=1)
    # strace['att_slow'] = [scenario.att(omega=omega)['att'] for scenario, omega in zip(strace['scenario'], strace['w'])]
    utrace['omega'] = utrace['w']
    utrace['mode'] = 'UNITS'
    utrace['att'] = utrace.apply(lambda x: x['instance'].predict(omega=x['omega'])['att'], axis=1)

    utrace.query("name == 'PLACEBO'").groupby('epoch')[['trn_loss', 'vld_loss', 'tst_loss']].mean().plot()
    plt.title(f"[UNITS] PLACEBO")
    plt.show()

    epoch = utrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().idxmin()
    omega = (utrace
             .query(f"epoch == {epoch}")
             .set_index(['name', 'seed'])
             ['w']
             .to_frame('omega')
             )

    ttrace.query("name == 'PLACEBO'").groupby('epoch')[['trn_loss', 'vld_loss', 'tst_loss']].mean().plot()
    plt.title(f"[TIME] PLACEBO")
    plt.show()

    ttrace = ttrace.merge(omega, left_on=['name', 'seed'], right_index=True)
    ttrace['lambd'] = ttrace['w']
    utrace['mode'] = 'TIME'

    ttrace['att'] = ttrace.apply(lambda x: x['instance'].predict(omega=x['omega'], lambd=x['lambd'])['att'], axis=1)

    trace = (pd.concat([
        utrace.query(f"epoch <= {epoch}"),
        ttrace])
             .drop(columns=['w'])
             .assign(total_epoch=lambda dx: np.where(dx['mode'] == 'UNITS', dx['epoch'], dx['epoch'] + 1 + epoch))
             )

    epoch = ttrace.query("name == 'PLACEBO'").groupby('epoch')['tst_loss'].mean().idxmin()
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

    plot(utrace, ttrace)

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

    dx['att'] = dx.apply(lambda x: x['instance'].predict(omega=x['omega'], lambd=x['lambd'])['att'], axis=1)

    te = np.array(dx.query("name == 'BOOTSTRAP'")['att'])
    return dict(att=np.mean(te), se=np.std(te, ddof=1), dx=dx)


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
