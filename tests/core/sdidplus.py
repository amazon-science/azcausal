from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.random import RandomState


# ------------------------------------------------ OPTIMIZATION ------------------------------------------------ #
def optimize(A, y, device='cpu', max_random=1_000, max_epochs=10_000, dropout=0.1, seed=1, alpha=(0.5, 4)):
    device = torch.device(device)
    random_state = torch.Generator(device=device).manual_seed(seed)
    criterion = nn.HuberLoss(reduction='none')

    A = torch.from_numpy(A.astype(np.float32)).to(device)
    y = torch.from_numpy(y.astype(np.float32)).to(device)

    def _dot(x):
        return (A @ x[..., None]).sum(dim=-1)

    def _f(x):
        yp = _dot(x)
        return criterion(y, yp).sum(dim=-1)

    nk, nt, nu = A.shape
    xx = torch.ones((nk, nu), requires_grad=False) / nu
    f = _f(xx)

    for i in range(max_random):
        a = torch.distributions.Uniform(*alpha).sample()
        dirichlet = torch.distributions.Dirichlet(torch.ones(nu) * a)
        xxp = dirichlet.sample()[None, ...]
        ff = _f(xxp)

        m = ff < f
        xx[m], f[m] = xxp, ff[m]

    # xx = torch.ones((nk, nu))
    x = nn.Parameter(torch.log(xx).to(device))

    optimizer = optim.Adam([x])

    def _softmax(x):
        return torch.nn.functional.softmax(x, dim=-1)

    def _dropout(x):
        mask = torch.bernoulli(torch.ones(nu) * (1.0 - dropout), generator=random_state)
        if mask.sum() > 0:
            return x * mask

    def _normalize(x):
        return x / x.sum(dim=1, keepdim=True)

    for epoch in range(max_epochs):

        optimizer.zero_grad()

        z = _softmax(x)
        zz = _normalize(_dropout(z))
        loss = _f(zz).mean()

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            ff = _f(z)
            xxp = _softmax(x)

            m = ff < f
            xx[m], f[m] = xxp[m], ff[m]

            if epoch % 1000 == 0:
                print(f'Epoch [{epoch:>05}/{max_epochs}], Loss: {f.mean().item():.8f}')

    return xx.numpy()


# ------------------------------------------------ PANELS ------------------------------------------------ #

def create(C, T, post, **kwargs):
    t = T if T.ndim == 1 else T.mean(axis=-1)

    # UNITS
    A, b = C[~post], t[~post]
    U = A - A.mean(axis=0, keepdims=True)
    u = b - b.mean()

    # PERIODS
    A, b = C[~post].T, C[post].T.mean(axis=1)
    P = A - A.mean(axis=0, keepdims=True)
    p = b - b.mean()

    return dict(C=C, T=T, post=post, t=t, U=U, u=u, P=P, p=p, **kwargs)


# ------------------------------------------------ SDID ------------------------------------------------ #
def sdid(scenario, omega, lambd):
    C, t = scenario['C'], scenario['t']
    post = scenario['post']

    # Store Weights
    scenario['omega'] = omega
    scenario['lambd'] = lambd

    # Synthetic Control
    sc = C @ omega
    scenario['sc'] = sc

    # DiD
    scenario['att'] = (t[post].mean() - t[~post] @ lambd) - (sc[post].mean() - sc[~post] @ lambd)
    scenario['obs'] = t[post].mean()

    return scenario


# ------------------------------------------------ ERROR ------------------------------------------------ #


class Error:

    @abstractmethod
    def generator(self, scenario):
        pass

    def run(self, scenario, **kwargs):
        lambd = scenario['lambd']
        samples = list(self.generator(scenario))
        A = np.vstack([e['U'][None, :] for e in samples])
        b = np.vstack([e['u'][None, :] for e in samples])

        ww = optimize(A, b, **kwargs)

        return [sdid(s, w, lambd) for s, w in zip(samples, ww)]


class Bootstrap(Error):

    def __init__(self, size, bayes=False, alpha=1.0):
        super().__init__()
        self.size = size
        self.bayes = bayes
        self.alpha = alpha

    def generator(self, scenario):
        for seed in range(self.size):
            yield self.sample(scenario, seed)

    def sample(self, scenario, seed):
        random_state = RandomState(seed)
        C, t, post = scenario['C'], scenario['t'], scenario['post']
        _, nc = C.shape

        if self.bayes:
            w = random_state.dirichlet(alpha=self.alpha * np.ones(nc), size=nc)
            CC = C @ w.T
        else:
            CC = C[:, random_state.choice(np.arange(nc), size=nc, replace=True)]

        return create(CC, t, post)

    def fit(self, samples):
        values = np.array([e['att'] for e in samples])
        n = len(values)
        return np.sqrt((n - 1) / n) * np.std(values, ddof=1)


class Placebo(Bootstrap):

    def sample(self, scenario, seed):
        post, t, sc = scenario['post'], scenario['t'], scenario['sc']
        sc = sc - sc[~post].mean() + t[~post].mean()

        scenario['t'] = sc
        obj = super().sample(scenario, seed)
        scenario['t'] = t

        return obj


class Jackknife(Error):

    def __init__(self, refit=False):
        super().__init__()
        self.refit = refit

    def generator(self, scenario):
        C, T, post = scenario['C'], scenario['T'], scenario['post']
        nc, nt = len(C.T), len(T.T)
        omega, lambd = scenario['omega'], scenario['lambd']

        if nc > 1:
            for i in range(nc):
                CC = np.delete(C, i, axis=1)

                oo = np.delete(omega, i)
                oo = oo / oo.sum()

                yield create(CC, T, post, omega=oo, lambd=lambd)

        if nt > 1:
            for j in range(nt):
                TT = np.delete(T, j, axis=1)
                yield create(C, TT, post, omega=omega, lambd=lambd)

    def fit(self, samples):
        values = np.array([e['att'] for e in samples])
        n = len(values)
        return np.sqrt(((n - 1) / n) * (n - 1) * np.var(values, ddof=1))

    def run(self, scenario, **kwargs):
        if self.refit:
            return super().run(scenario, **kwargs)
        else:
            return [sdid(s, s['omega'], s['lambd']) for s in self.generator(scenario)]


# ------------------------------------------------ ESTIMATOR ------------------------------------------------ #
def sdid_plus(Y, treat, post, omega=None, lambd=None, dropout=0.25):
    # create the data to simulate the scenario
    scenario = create(Y[:, ~treat], Y[:, treat], post)

    # PERIODS (lambda)
    if lambd is None:
        A, b = scenario['P'], scenario['p']
        lambd = optimize(A[None, :], b[None, :], max_epochs=10_000, dropout=dropout)[0]

    # UNITS (omega)
    if omega is None:
        A, b = scenario['U'], scenario['u']
        omega = optimize(A[None, :], b[None, :], max_epochs=20_000, dropout=dropout)[0]

    # SDID
    did = sdid(scenario, omega, lambd)

    # ERROR
    # error = Bootstrap(size=100, bayes=False)
    error = Jackknife(refit=False)

    simulations = error.run(scenario, max_epochs=20_000, dropout=dropout)
    did['se'] = error.fit(simulations)

    did['scale'] = treat.sum() * post.sum()

    return did


def sdid_plus_batched(panels, dropout=0.25):
    scenarios = [create(Y[:, ~treat], Y[:, treat], post) for Y, treat, post in panels]

    A = np.vstack([e['P'][None, :] for e in scenarios])
    b = np.vstack([e['p'][None, :] for e in scenarios])
    lambd = optimize(A, b, max_epochs=50_000, dropout=dropout)

    A = np.vstack([e['U'][None, :] for e in scenarios])
    b = np.vstack([e['u'][None, :] for e in scenarios])
    omega = optimize(A, b, max_epochs=50_000, dropout=dropout)

    scenarios = [sdid(s, o, l) for s, o, l in zip(scenarios, omega, lambd)]

    for s in scenarios:
        error = Jackknife(refit=False)
        sims = error.run(s)
        s['se'] = error.fit(sims)
        s['scale'] = len(s['C'].T) * len(s['T'].T)

    return scenarios
