import cvxpy as cp
import numpy as np
import pandas as pd

from azcausal.core.effect import Effect
from azcausal.core.error import JackKnife
from azcausal.core.estimator import Estimator
from azcausal.core.panel import CausalPanel
from azcausal.core.result import Result


class SolverException(Exception):

    def __init__(self, *args, exceptions=None, **kwargs):
        super().__init__(*args)
        self.exceptions = exceptions


def solve(data: pd.DataFrame,
          contr: np.ndarray,
          treat: np.ndarray,
          alpha: float = None,
          max_weight=None,
          eps: float = 1e-12,
          kwargs=None,
          solvers=(cp.ECOS, cp.OSQP, cp.SCS)
          ):
    # get the default solver kwargs and add the ones passed
    kwargs = dict(kwargs) if kwargs is not None else dict()

    # set defaults to kwargs
    defaults = dict(verbose=False)
    for k, v in defaults.items():
        if k not in kwargs:
            kwargs[k] = v

    if 'solver' in kwargs:
        solvers = kwargs.pop('solver')

    C = data.loc[:, contr]
    A = C.values
    m, n = A.shape

    T = data.loc[:, treat]
    b = T.mean(axis=1).values

    x = cp.Variable(n)

    # OBJECTIVE
    # x = np.ones(n) / n
    # y = (((A @ x - b)**2).sum() / m) + ((alpha ** 2) * (x ** 2).sum())

    obj = cp.sum_squares(A @ x - b) / m
    if alpha is not None:
        obj += (alpha ** 2) * cp.sum_squares(x)

    constr = [x >= 0.0, cp.sum(x) == 1]
    if max_weight is not None:
        constr.append(x <= max_weight)

    success = False
    exceptions = dict()
    for solver in solvers:
        try:
            problem = cp.Problem(cp.Minimize(obj), constr)
            problem.solve(solver=solver, **kwargs)
            success = (x.value is not None)
        except Exception as ex:
            exceptions[solver] = ex

        if success:
            break

    if not success:
        raise SolverException("Error: All solvers in FSDID failed to find weights.", exceptions=exceptions)

    w = np.array(x.value)
    if eps is not None:
        w = np.where(w >= eps, w, 0.0)
        w /= w.sum()

    return pd.Series(data=w, index=C.columns)


def demean(x, axis=0):
    return x - x.mean(axis=axis)


def fast_sdid(df: pd.DataFrame,
              post: np.ndarray,
              treat: np.ndarray,
              omega=None,
              lambd=None,
              jackknife=True,
              by='units'
              ):
    # get the counts in the data frame
    n_treat, n_contr = treat.sum(), (~treat).sum()
    n_post, n_pre = post.sum(), (~post).sum()

    # the noise of data used for regularization
    noise = np.diff(df.loc[~post, ~treat], axis=0).std(ddof=1)

    # calculate the unit and time weights
    if omega is None:
        omega = solve(demean(df.loc[~post]), ~treat, treat, alpha=noise * (n_treat * n_post) ** (1 / 4))
    if lambd is None:
        lambd = solve(demean(df.T.loc[~treat]), ~post, post, alpha=noise * 1e-06)

    if by == 'units':

        # get the average data for pre and post
        d_pre = df.loc[~post].T.dot(lambd).values
        d_post = df.loc[post].mean().values
        observed = d_post[treat].mean()

        # calculate all for terms used in did
        pre_treat = d_pre[treat].mean()
        post_treat = d_post[treat].mean()
        pre_contr = d_pre[~treat] @ omega
        post_contr = d_post[~treat] @ omega

        # calculate the average treatment effect
        avg_te = (post_treat - pre_treat) - (post_contr - pre_contr)
        se = np.nan

        # if the standard error should be calculated as well
        if jackknife:

            # jackknife: drop a treatment unit
            def jk_treat(k):
                if len(np.delete(np.where(treat)[0], k)) == 0:
                    return np.nan

                pre_treat = np.delete(d_pre[treat], k).mean()
                post_treat = np.delete(d_post[treat], k).mean()

                return (post_treat - pre_treat) - (post_contr - pre_contr)

            # jackknife: drop a control unit
            def jk_contr(k):
                w = np.delete(omega.values, k)
                if w.sum() == 0:
                    return np.nan

                wp = w / w.sum()

                pre_contr = np.delete(d_pre[~treat], k) @ wp
                post_contr = np.delete(d_post[~treat], k) @ wp

                return (post_treat - pre_treat) - (post_contr - pre_contr)

            # calculate the jackknife values
            jk_treat = np.array([jk_treat(k) for k in range(n_treat)])
            jk_contr = np.array([jk_contr(k) for k in range(n_contr)])

            # calculate the jackknife error
            jk = pd.Series(np.concatenate([jk_treat, jk_contr])).dropna().to_numpy()
            n = len(jk)
            se = np.sqrt(((n - 1) / n) * (n - 1) * np.var(jk, ddof=1))

    elif by == 'time':

        d_contr = df.loc[:, ~treat].dot(omega).values
        d_treat = df.loc[:, treat].mean(axis=1).values
        observed = d_treat[post].mean()

        # calculate all for terms used in did
        pre_treat = d_treat[~post] @ lambd
        post_treat = d_treat[post].mean()
        pre_contr = d_contr[~post] @ lambd
        post_contr = d_contr[post].mean()

        # calculate the average treatment effect
        avg_te = (post_treat - pre_treat) - (post_contr - pre_contr)
        se = np.nan

        # if the standard error should be calculated as well
        if jackknife:

            # jackknife: drop a pre time
            def jk_post(k):
                if len(np.delete(np.where(post)[0], k)) == 0:
                    return np.nan

                post_treat = np.delete(d_treat[post], k).mean()
                post_contr = np.delete(d_contr[post], k).mean()

                return (post_treat - pre_treat) - (post_contr - pre_contr)

            # jackknife: drop a control unit
            def jk_pre(k):
                w = np.delete(lambd.values, k)
                if w.sum() == 0:
                    return np.nan

                wp = w / w.sum()

                pre_treat = np.delete(d_treat[~post], k) @ wp
                pre_contr = np.delete(d_contr[~post], k) @ wp

                return (post_treat - pre_treat) - (post_contr - pre_contr)

            # calculate the jackknife values
            jk_pre = np.array([jk_pre(k) for k in range(n_pre)])
            jk_post = np.array([jk_post(k) for k in range(n_post)])

            # calculate the jackknife error
            jk = pd.Series(np.concatenate([jk_pre, jk_post])).dropna().to_numpy()
            n = len(jk)
            se = np.sqrt(((n - 1) / n) * (n - 1) * np.var(jk, ddof=1))

    else:
        raise Exception("Esimate effects either by time or units as the first dimension.")

    # calculate the observed values and the scale to calculate the cumulative effect
    scale = post.sum() * treat.sum()

    # calculate the data normalized in pre over time (this plot in fact shows the effect of both)
    T = df.loc[:, treat].mean(axis=1).pipe(lambda dx: dx - dx.loc[~post].mean())
    C = df.loc[:, ~treat].dot(omega).pipe(lambda dx: dx - dx.loc[~post].mean())
    by_time = pd.DataFrame(dict(T=T, C=C)).assign(post=np.where(post, 'Y', 'N'))

    return dict(avg_te=avg_te, cum_te=avg_te * scale, se=se, observed=observed,
                omega=omega, lambd=lambd, by_time=by_time, scale=scale)


class FSDID(Estimator):

    def fit(self,
            panel: CausalPanel,
            omega=None,
            lambd=None,
            se=True,
            fallback=True,
            **kwargs):
        df = panel['outcome']

        try:
            sdid = fast_sdid(df, panel.post, panel.treat, omega=omega, lambd=lambd, jackknife=se)
            att = Effect(sdid['avg_te'], se=sdid['se'], observed=sdid['observed'], scale=sdid['scale'], by_time=sdid['by_time'], name="ATT")
            return Result(dict(att=att), data=panel, estimator=self)

        except SolverException as e:
            if fallback:
                from azcausal.estimators.panel.sdid import SDID
                estimator = SDID()
                result = estimator.fit(panel, omega=omega, lambd=lambd, se=se)
                estimator.error(result, JackKnife())
                return result
            else:
                raise e
