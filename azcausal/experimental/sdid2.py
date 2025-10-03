import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.random import RandomState
from scipy import stats
from tqdm.autonotebook import tqdm

from azcausal.core.effect import Effect
from azcausal.core.error import JackKnife
from azcausal.core.panel import Panel
from azcausal.core.result import Result
from azcausal.estimators.panel.did import DID
from azcausal.estimators.panel.sdid import sdid_plot, SDID


# ---------------------------------------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------------------------------------


def flatten(x):
    return list(itertools.chain(*x))


def append(df, fn, inplace=True):
    dx = df.apply(fn, axis=1, result_type='expand')

    if not inplace:
        df = df.copy()

    for name, x in dx.to_dict().items():
        df[name] = x
    return df


# ---------------------------------------------------------------------------------------------------------
# OPTIMIZATION
# ---------------------------------------------------------------------------------------------------------


class Optimization:

    def __init__(self, n_max_epochs=3_000, algorithm=torch.optim.Adam, device=None):
        super().__init__()
        self.n_max_epochs = n_max_epochs
        self.algorithm = algorithm
        self.device = device

    def solver(self, A, b, m, h, zeta=None, x=None):
        # if no device provided use gpu if available cpu otherwise
        device = self.device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        A = torch.tensor(A, dtype=torch.float32).to(device)
        b = torch.tensor(b, dtype=torch.float32).to(device)
        m = torch.tensor(m, dtype=torch.bool).to(device)
        h = torch.tensor(h, dtype=torch.bool).to(device)

        if zeta is not None:
            zeta = torch.tensor(zeta, dtype=torch.float32).to(device)

        At, bt = A[:, ~h], b[:, ~h]
        Ah, bh = A[:, h], b[:, h]

        n_samples, _, n_units = At.shape

        if x is not None:
            x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
        else:
            x = torch.full((n_samples, n_units), 0.1, requires_grad=True, dtype=torch.float32)

        optimizer = self.algorithm([x], lr=0.001)

        return MySolver(At, bt, Ah, bh, x, zeta, m, optimizer, self.n_max_epochs)


def predict(A, w):
    return torch.bmm(A, w.unsqueeze(2)).squeeze(2)


def forward(A, w, b, zeta=None, m=None):
    y_hat = predict(A, w)
    loss = F.mse_loss(y_hat, b, reduction='none')

    if m is not None:
        loss = (loss * m.to(A.dtype)).sum(dim=1) / m.sum(dim=1)
    else:
        loss = loss.mean(dim=1)

    penalty = (zeta ** 2) * (w ** 2).sum()

    return loss + penalty


class MySolver(object):

    def __init__(self, At, bt, Ah, bh, x, zeta, m, optimizer, n_max_epochs):
        super().__init__()
        self.At, self.bt = At, bt
        self.Ah, self.bh = Ah, bh
        self.m = m
        self.x = x
        self.zeta = zeta
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
        trn_loss = forward(self.At, self.w, self.bt, zeta=self.zeta, m=self.m)
        total_loss = trn_loss.mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.total_loss = total_loss
        self.trn_loss = trn_loss

        self.epoch += 1

    def run(self, callback: lambda x: x):

        for _ in (pbar := tqdm(range(self.n_max_epochs))):
            self.advance()

            if self.epoch % 10 == 0:

                with torch.no_grad():
                    self.vld_loss = forward(self.At, self.w, self.bt, zeta=self.zeta, m=~self.m)
                    self.tst_loss = forward(self.Ah, self.w, self.bh, zeta=self.zeta)

                    callback(self)

                    pbar.set_description(f"{self.epoch} {self.total_loss.item()} | {self.tst_loss.mean().item()}")


# ---------------------------------------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------------------------------------


class MyPanel(object):

    def __init__(self, Y, n_treat, n_post, Yp=None, ulabel=None):
        super().__init__()
        self.Y = Y
        self.Yp = Yp
        self.ulabel = ulabel
        self.n_treat = n_treat
        self.n_post = n_post

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

    def __lt__(self, other):
        if not isinstance(other, MyPanel):
            return NotImplemented
        return id(self) < id(other)

    def predict(self, omega=None, lambd=None, prefix=''):
        Y = self.Y
        contr, treat, pre, post = self.slices

        vcontr = Y[:, contr].mean(axis=1) if omega is None else Y[:, contr] @ omega
        post_contr = vcontr[post].mean()
        pre_contr = vcontr[pre].mean() if lambd is None else vcontr[pre] @ lambd

        vtreat = Y[:, treat].mean(axis=1)
        post_treat = vtreat[post].mean()
        pre_treat = vtreat[pre].mean() if lambd is None else vtreat[pre] @ lambd

        att = (post_treat - pre_treat) - (post_contr - pre_contr)

        obs = post_treat
        cf = obs - att

        return {
            f'{prefix}avg_te': att,
            f'{prefix}perc_te': att / np.abs(cf),
        }

    def effect(self, prefix=""):
        if self.Yp is not None:
            obs = self.obs()
            cf = self.cf()
            att = obs - cf
            return {
                f'{prefix}avg_te': att,
                f'{prefix}perc_te': att / np.abs(cf),
            }

    def obs(self):
        return np.mean(self.Y[self.spost][:, self.streat])

    def cf(self):
        return np.mean(self.Yp[self.spost][:, self.streat])


def create_mask(n, p, random_state):
    m = random_state.random(n) <= p
    pos, neg = random_state.choice(np.arange(n), replace=False, size=2)
    m[pos] = True
    m[neg] = False
    return m


class Instance(MyPanel):

    def __init__(self, panel, itreat=None, icontr=None, mu=None, mt=None, Yp=None):
        Y = panel.Y
        itreat = itreat if itreat is not None else panel.itreat
        icontr = icontr if icontr is not None else panel.icontr

        iunits = np.concatenate([icontr, itreat])
        YY = Y[:, iunits]
        if Yp is not None:
            Yp = Yp[:, iunits]

        ulabel = None
        if panel.ulabel is not None:
            ulabel = panel.ulabel[iunits]

        super().__init__(YY, len(itreat), panel.n_post, Yp=Yp, ulabel=ulabel)
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
        default = dict(panel=self.panel,
                       name='DEFAULT',
                       seed=seed,
                       instance=Instance(panel, mt=mt, mu=mu, Yp=panel.Yp)
                       )

        # select placebo treatment units from control pool
        itreat = random_state.choice(panel.itreat, replace=True, size=n_treat)
        icontr = random_state.choice(panel.icontr, replace=True, size=n_contr)
        bootstrap = dict(panel=self.panel,
                         name='BOOTSTRAP',
                         seed=seed,
                         instance=Instance(panel, icontr=icontr, itreat=itreat, mt=mt, mu=mu, Yp=panel.Yp)
                         )

        # select placebo treatment units from control pool
        itreat = random_state.choice(panel.icontr, replace=False, size=n_treat)

        # sample from the remaining control pool
        pool = [i for i in panel.icontr if i not in itreat]
        icontr = random_state.choice(pool, replace=True, size=n_contr)

        placebo = dict(panel=self.panel,
                       name='PLACEBO',
                       seed=seed,
                       instance=Instance(panel, icontr=icontr, itreat=itreat, mt=mt, mu=mu, Yp=panel.Y)
                       )

        return [bootstrap, placebo]

    def run(self):
        return pd.DataFrame(flatten((self.sample(k) for k in range(self.n))))


class Trace:

    def __init__(self, instances):
        super().__init__()
        self.instances = instances
        self.records = []

    def __call__(self, solver):
        for k, instance in enumerate(self.instances):
            self.records.append(dict(epoch=solver.epoch,
                                     instance=instance,
                                     trn_loss=solver.trn_loss[k].item(),
                                     vld_loss=solver.vld_loss[k].item(),
                                     tst_loss=solver.tst_loss[k].item(),
                                     w=solver.w[k].detach().numpy(),
                                     ))


def to_numpy(dx, col):
    return np.vstack([x[None, :] for x in dx[col]])


def solve(dx, h, eta=None, w=None):
    A, b, m = [to_numpy(dx, name) for name in ['A', 'b', 'm']]

    zeta = None
    if eta is not None:
        noise = np.diff(A[:, ~h], axis=1).std(axis=(1, 2), ddof=1)
        zeta = eta * noise

    x = None
    if w is not None:
        x = np.log(w + 1e-32)

    trace = Trace(dx['instance'])
    Optimization().solver(A, b, m, h, zeta=zeta, x=x).run(trace)
    return pd.DataFrame(trace.records)


def sdid_weights_opt(df, keys, dx, h, eta=None, w=None):
    trace = solve(dx, h, eta=eta, w=w)

    dxu = df.merge(trace, on='instance')

    if 'name' in dxu:
        avg_tst_loss = (dxu
                        .query("name == 'PLACEBO'")
                        .groupby(keys + ['epoch'])
                        ['tst_loss']
                        .mean()
                        .reset_index()
                        )

        epoch = (avg_tst_loss
                 .groupby(keys)
                 .apply(lambda x: x.set_index('epoch')['tst_loss'].idxmin())
                 .to_frame('epoch')
                 )

        w = (dxu
        .merge(epoch, on=keys + ['epoch'], how='inner')
        .set_index(keys + ['instance'])
        ['w']
        )

    else:
        w = dxu.query(f"epoch == {dxu['epoch'].max()}").set_index(keys + ['instance'])['w']

    return w


def sdid_weights_dims(instances):
    dx = pd.DataFrame.from_records([dict(n_treat=x.n_treat, n_contr=x.n_contr, n_pre=x.n_pre, n_post=x.n_post) for x in instances])

    dx = dx.drop_duplicates()
    assert len(dx) == 1
    dims = dx.iloc[0].to_dict()
    return dims["n_pre"], dims["n_post"], dims["n_treat"], dims["n_contr"]


def sdid_weights_omega(df, keys, col, eta=None, w=None):
    n_pre, n_post, n_treat, n_contr = sdid_weights_dims(df[col])

    ht = np.full(n_pre + n_post, False)
    ht[-n_post:] = True

    dunits = pd.DataFrame([instance.matrix_by_unit() for instance in df[col]])

    if w is not None:
        w = to_numpy(df, w)

    w = sdid_weights_opt(df, keys, dunits, ht, eta=eta, w=w)

    dw = (df
          .set_index(keys + ['instance'])[[]]
          .join(w.to_frame('omega'))
          )

    return dw


def sdid_weights_lambd(df, keys, col, eta=None, w=None):
    n_pre, n_post, n_treat, n_contr = sdid_weights_dims(df[col])

    hu = np.full(n_contr + n_treat, False)
    hu[-n_treat:] = True

    dtime = pd.DataFrame([instance.matrix_by_time() for instance in df[col]])

    w = sdid_weights_opt(df, keys, dtime, hu, eta=eta, w=w)

    dw = (df
          .set_index(keys + ['instance'])[[]]
          .join(w.to_frame('lambd'))
          )

    return dw


# ---------------------------------------------------------------------------------------------------------
# ESTIMATOR
# ---------------------------------------------------------------------------------------------------------


def sdid2(Y, n_treat, n_post) -> dict:
    panel = MyPanel(Y, n_treat, n_post)

    factory = InstanceFactory(panel, 100)
    instances = factory.run()

    dx = sdid_weights(instances, panel.mpost, panel.mtreat)

    te = np.array(dx.query("name == 'BOOTSTRAP'")['att'])
    return dict(att=np.mean(te), se=np.std(te, ddof=1), dx=dx)


# ---------------------------------------------------------------------------------------------------------
# ERROR
# ---------------------------------------------------------------------------------------------------------


def jackknife_se(x):
    n = len(x)
    return np.sqrt(((n - 1) / n) * (n - 1) * np.var(x, ddof=1))


def bootstrap_se(x):
    n = len(x)
    return np.sqrt((n - 1) / n) * np.std(x, ddof=1)


def did(panel, omega=None, lambd=None, jackknife=False, prefix=''):
    Y = panel.Y
    contr, treat, pre, post = panel.slices

    vpre = Y[pre].mean(axis=0) if lambd is None else Y[pre].T @ lambd
    vpost = Y[post].mean(axis=0)

    delta_contr = vpost[contr].mean() - vpre[contr].mean()

    post_treat = vpost[treat].mean()
    pre_treat = vpre[treat].mean()
    delta_treat = (post_treat - pre_treat)

    avg_te = delta_treat - delta_contr

    obs = post_treat
    cf = obs - avg_te

    ans = {
        f'{prefix}avg_te': avg_te,
        f'{prefix}perc_te': avg_te / np.abs(cf)
    }

    if jackknife:

        te = []
        if panel.n_contr > 1:
            if omega is None:
                delta_contr_mod = (delta_contr * panel.n_contr - (vpost[contr] - vpre[contr])) / (panel.n_contr - 1)

            else:
                n = panel.n_contr
                M = np.eye(n) == 1
                delta = (-1 * M + (~M) * (1 / (n - 1))) * omega
                omega_mod = omega[:, None] + delta

                delta_contr_mod = vpost[contr] @ omega_mod - vpre[contr] @ omega_mod

            te.extend(delta_treat - delta_contr_mod)

        if panel.n_treat > 1:
            delta_treat_mod = (delta_treat * panel.n_treat - (vpost[treat] - vpre[treat])) / (panel.n_treat - 1)
            te.extend(delta_treat_mod - delta_contr)

        te = np.array(te, dtype=float)
        ans[f'{prefix}avg_error'] = jackknife_se(te)

    return ans


def plot(utrace, ttrace):
    # utrace['att'] = np.vstack(utrace['tst_error']).mean(axis=1)
    # strace['att_slow'] = [scenario.att(omega=omega)['att'] for scenario, omega in zip(strace['scenario'], strace['w'])]
    utrace['omega'] = utrace['w']
    utrace['att'] = append(lambda x: x['instance'].predict(omega=x['omega']), axis=1)

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
    ttrace['att'] = append(lambda x: x['instance'].predict(omega=x['omega'], lambd=x['lambd']), axis=1)

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
        dx = trace.query(f"name == '{mode}'").groupby('total_epoch').aggregate(['mean', 'std'])
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


# ---------------------------------------------------------------------------------------------------------
# AZCAUSAL
# ---------------------------------------------------------------------------------------------------------

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


def az_panel(panel):
    data = dict()
    data['outcome'] = pd.DataFrame(panel.Y)

    intervention = np.full_like(panel.Y, 0)
    intervention[-panel.n_post:, -panel.n_treat:] = 1
    data['intervention'] = pd.DataFrame(intervention)

    if panel.ulabel is not None:
        for _, dx in data.items():
            dx.columns = panel.ulabel

    return Panel(data).setup()


def az(estimator, panel, prefix='', jackknife=True):
    p = az_panel(panel)

    result = estimator.fit(p)

    if jackknife:
        estimator.error(result, JackKnife())

    omega = result['info'].get('omega', None)
    fomega = None
    if omega is not None:
        omega = omega[panel.ulabel[panel.scontr]].values
        fomega = result['info']['solvers']['omega']['f']

    lambd = result['info'].get('lambd', None)
    flambd = None
    if lambd is not None:
        lambd = lambd.values
        flambd = result['info']['solvers']['lambd']['f']

    return {
        f'{prefix}omega': omega,
        f'{prefix}lambd': lambd,
        f'{prefix}fomega': fomega,
        f'{prefix}flambd': flambd,
        f'{prefix}effect': result.effect,
        f'{prefix}avg_te': result.effect.value,
        f'{prefix}perc_te': result.effect.percentage().value,
        f'{prefix}avg_error': result.effect.se
    }


def az_did(panel, **kwargs):
    return az(DID(), panel, **kwargs)


def az_sdid(panel, **kwargs):
    return az(SDID(), panel, **kwargs)
