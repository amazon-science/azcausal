"""ASDID — Augmented Synthetic Difference-in-Differences.

Fully standalone, pure numpy, always batched (batch_size, ...).
No dependencies on azcausal. Only requires numpy and scipy.
All functions operate on batch dimension batch_size. Single problem = batch_size=1.

API conventions:
    - Y: (batch_size, n_time, n_units) or (n_time, n_units) — outcome panel
    - treat: (n_units,) bool — treatment mask, same for all batches
    - X: (batch_size, n_time, n_units, n_cov) or None — optional covariates
    - omega: (batch_size, n_contr) — unit weights on simplex
    - lambd: (batch_size, n_pre) — time weights on simplex

Public API:
    Estimation: estimate, estimate_att, did, compute_did
    Inference:  se_jackknife, se_placebo, se_bootstrap
    Power:      simulate, power, power_curve, compute_power, mde
"""
import numpy as np
from scipy import stats as _stats


# =============================================================================
# Data Preparation
# =============================================================================

def df_to_matrix(df, time_col, unit_col, outcome_col, treat_col, fillna=0):
    """Convert a long-format DataFrame to the matrix inputs for asdid/did.

    Parameters
    ----------
    df : pd.DataFrame — long-format panel data
    time_col : str — column name for time periods
    unit_col : str — column name for units
    outcome_col : str — column name for the outcome variable
    treat_col : str — column name for treatment indicator (0/1 or bool)
    fillna : scalar — value to fill missing entries (default 0)

    Returns
    -------
    Y : (n_time, n_units) — outcome matrix
    n_pre : int — number of pre-treatment periods
    treat : (n_units,) bool — treatment mask

    Raises
    ------
    ValueError — if treatment assignment is inconsistent
    """
    import pandas as pd

    # Check for duplicates
    dupes = df.duplicated(subset=[time_col, unit_col], keep=False)
    if dupes.any():
        n_dupes = dupes.sum()
        example = df.loc[dupes, [time_col, unit_col]].head(3).to_string(index=False)
        raise ValueError(
            f"Found {n_dupes} duplicate (time, unit) entries. "
            f"Each unit must have exactly one observation per time period.\n{example}"
        )

    # Pivot to matrix (sorted by time)
    Y_df = df.set_index([time_col, unit_col])[outcome_col].unstack().sort_index().fillna(fillna)
    Y = Y_df.values  # (n_time, n_units)

    # Treatment mask: unit is treated if it's ever treated
    treat_by_unit = df.groupby(unit_col)[treat_col].max()
    treat = treat_by_unit.reindex(Y_df.columns).values.astype(bool)

    # n_pre: number of time periods before any treatment starts
    treat_times = df.loc[df[treat_col] > 0, time_col]
    if len(treat_times) > 0:
        first_treat_time = treat_times.min()
        n_pre = (Y_df.index < first_treat_time).sum()
    else:
        n_pre = len(Y_df)

    # --- Consistency checks ---
    # Find treated units: any unit that is ever treated
    treated_units = df.loc[df[treat_col] > 0, unit_col].unique()

    if len(treated_units) > 0:
        # Find first treatment time
        first_treat_time = df.loc[df[treat_col] > 0, time_col].min()

        # Check: treated units must have treat=1 for ALL periods >= first_treat_time
        post_rows = df.loc[
            (df[unit_col].isin(treated_units)) & (df[time_col] >= first_treat_time)
        ]
        untreated_post = post_rows.loc[post_rows[treat_col] == 0]
        if len(untreated_post) > 0:
            bad_units = untreated_post[unit_col].unique().tolist()
            raise ValueError(
                f"Units {bad_units} are treated but have treat_col=0 in periods >= {first_treat_time}. "
                f"ASDID requires block treatment: all treated units must be treated in ALL post-periods."
            )

    return Y, n_pre, treat


# =============================================================================
# Solver
# =============================================================================

def _project_simplex_batch(X):
    """Project each row onto the probability simplex. (B, n) → (B, n)."""
    B, n = X.shape
    u = np.sort(X, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1
    rho = n - 1 - np.argmax((u * np.arange(1, n + 1)[None, :] > cssv)[:, ::-1], axis=1)
    theta = cssv[np.arange(B), rho] / (rho + 1.0)
    return np.maximum(X - theta[:, None], 0)


def pgd_batch(A, b, reg, max_iter=5000, tol=1e-8):
    """Batched PGD on simplex with ridge. Solves B problems simultaneously.

    min ||Aω - b||²/n + reg·||ω||²  s.t. ω ≥ 0, Σω = 1

    Parameters
    ----------
    A : (batch_size, n, p)
    b : (batch_size, n)
    reg : (batch_size,) or scalar
    max_iter : int
    tol : (batch_size,) or scalar

    Returns
    -------
    X : (batch_size, p)
    """
    B, n, p = A.shape
    reg = np.broadcast_to(np.asarray(reg, dtype=float).ravel(), B).copy()
    tol = np.broadcast_to(np.asarray(tol, dtype=float).ravel(), B).copy()

    X = np.full((B, p), 1.0 / p)
    L = 2.0 * ((A ** 2).sum(axis=(1, 2)) / n + reg)
    step = 1.0 / L

    def obj(X):
        AX = np.einsum('bnp,bp->bn', A, X)
        return ((AX - b) ** 2).mean(axis=1) + reg * (X ** 2).sum(axis=1)

    F = obj(X)
    active = np.ones(B, dtype=bool)

    for _ in range(max_iter):
        AX = np.einsum('bnp,bp->bn', A, X)
        grad = 2.0 * np.einsum('bnp,bn->bp', A, AX - b) / n + 2.0 * reg[:, None] * X
        X_new = _project_simplex_batch(X - step[:, None] * grad)
        F_new = obj(X_new)

        step = np.where(F_new <= F, step * 1.2, step * 0.5)
        active &= ~((F - F_new) < tol ** 2)
        X, F = X_new, F_new

        if not active.any():
            break

    return X


# =============================================================================
# Regularization
# =============================================================================

def reg_mad(A):
    """Ridge penalty from MAD of first-differenced residuals. Batched.

    Parameters
    ----------
    A : (batch_size, n, p)

    Returns
    -------
    reg : (batch_size,)
    """
    d = np.diff(A, axis=1).reshape(A.shape[0], -1)
    return (np.median(np.abs(d), axis=1) * 1.4826) ** 2


# =============================================================================
# Demeaning
# =============================================================================

def demean(Y, n_pre, treat):
    """Two-way demean. Always batched.

    Parameters
    ----------
    Y : (batch_size, n_time, n_units) or (n_time, n_units)
    n_pre : int
    treat : (n_units,) bool

    Returns
    -------
    R : (batch_size, n_time, n_units)
    alpha : (batch_size, n_units) — unit fixed effects
    gamma : (batch_size, n_time) — time fixed effects
    """
    if Y.ndim == 2:
        Y = Y[None]
    B, n_time, n_units = Y.shape
    treat = np.asarray(treat).ravel()

    pre = np.arange(n_time) < n_pre

    alpha = Y[:, pre, :].mean(axis=1)  # (B, n_units)
    Ydm = Y - alpha[:, None, :]

    gamma_pre = Ydm[:, pre, :].mean(axis=2)  # (B, n_pre)
    Ydm_post_ctrl = np.where(~treat[None, None, :], Ydm[:, ~pre, :], np.nan)
    gamma_post = np.nanmean(Ydm_post_ctrl, axis=2)  # (B, n_post)

    gamma = np.concatenate([gamma_pre, gamma_post], axis=1)
    R = Ydm - gamma[:, :, None]
    return R, alpha, gamma


# =============================================================================
# Donor selection
# =============================================================================

def solve_unconstrained(A, b, reg):
    """Solve ridge regression without simplex constraint. Batched.

    Uses Woodbury identity: O(n² + n×p) instead of O(p²) when n << p.

    Parameters
    ----------
    A : (batch_size, n, p)
    b : (batch_size, n)
    reg : (batch_size,)

    Returns
    -------
    omega : (batch_size, p) — unconstrained weights (can be negative, don't sum to 1)
    """
    B, n, p = A.shape
    # M = A @ A.T / n + reg * I  →  (B, n, n)
    M = np.einsum('bnp,bmp->bnm', A, A) / n + reg[:, None, None] * np.eye(n)
    v = np.linalg.solve(M, b[:, :, None])[:, :, 0]  # (B, n)
    return np.einsum('bpn,bn->bp', A.transpose(0, 2, 1), v) / n


def select_donors(A, b, reg, max_donors, max_time=100, seed=0):
    """Select top-K donors using unconstrained ridge solution. Batched.

    For large n_pre, subsamples time periods for the screening step (fast).

    Parameters
    ----------
    A : (batch_size, n, p)
    b : (batch_size, n)
    reg : (batch_size,)
    max_donors : int
    max_time : int — max time periods for screening (subsamples if n > max_time)
    seed : int

    Returns
    -------
    indices : (batch_size, max_donors) — indices of selected donors
    """
    B, n, p = A.shape
    if n > max_time:
        rng = np.random.default_rng(seed)
        t_idx = rng.choice(n, max_time, replace=False)
        A = A[:, t_idx, :]
        b = b[:, t_idx]
    omega_unc = solve_unconstrained(A, b, reg)
    return np.argsort(omega_unc, axis=1)[:, -max_donors:]


# =============================================================================
# Covariate adjustment
# =============================================================================

def fit_beta(R, X, n_pre, treat, regularize=True):
    """Fit covariate coefficients via ridge on demeaned pre×control block. Batched.

    Parameters
    ----------
    R : (batch_size, n_time, n_units) — demeaned Y
    X : (batch_size, n_time, n_units, n_cov) — covariates (demeaned internally)
    n_pre : int
    treat : (n_units,) bool
    regularize : bool

    Returns
    -------
    beta : (batch_size, n_cov)
    reg : (batch_size,)
    """
    B, n_time, n_units, n_cov = X.shape
    ctrl_idx = np.where(~treat)[0]

    # Demean X: subtract unit mean (over pre) and time mean (over ctrl)
    Xc = X[:, :n_pre, ctrl_idx, :]  # (B, n_pre, n_ctrl, n_cov)
    Xdm = Xc - Xc.mean(axis=1, keepdims=True) - Xc.mean(axis=2, keepdims=True) + Xc.mean(axis=(1, 2), keepdims=True)

    # Flatten pre×control into observations
    A = Xdm.reshape(B, -1, n_cov)  # (B, n_pre*n_ctrl, n_cov)
    b = R[:, :n_pre, ctrl_idx].reshape(B, -1)  # (B, n_pre*n_ctrl)
    n_obs = A.shape[1]

    if regularize:
        # Light ridge: residual variance scaled by n_cov/n_obs (vanishes when overdetermined)
        ATA = np.einsum('bni,bnj->bij', A, A) / n_obs
        ATb = np.einsum('bni,bn->bi', A, b) / n_obs
        beta_ols = np.linalg.solve(ATA + 1e-10 * np.eye(n_cov), ATb[:, :, None])[:, :, 0]
        resid = b - np.einsum('bnc,bc->bn', A, beta_ols)
        reg = resid.var(axis=1) * n_cov / n_obs  # (B,)
    else:
        reg = np.zeros(B)

    ATA = np.einsum('bni,bnj->bij', A, A) / n_obs + reg[:, None, None] * np.eye(n_cov)
    ATb = np.einsum('bni,bn->bi', A, b) / n_obs
    beta = np.linalg.solve(ATA, ATb[:, :, None])[:, :, 0]  # (B, n_cov)
    return beta, reg


# =============================================================================
# Weight fitting
# =============================================================================

def fit_omega(R, n_pre, treat, regularize=True):
    """Fit unit weights omega. Batched.

    Parameters
    ----------
    R : (batch_size, n_time, n_units)
    n_pre : int
    treat : (n_units,) bool
    regularize : bool

    Returns
    -------
    omega : (batch_size, n_contr)
    reg : (batch_size,)
    """
    B = R.shape[0]
    pre = np.arange(R.shape[1]) < n_pre
    ctrl_idx = np.where(~treat)[0]
    treat_idx = np.where(treat)[0]
    n_contr, n_treat = len(ctrl_idx), len(treat_idx)

    A = R[:, pre][:, :, ctrl_idx]  # (B, n_pre, n_contr)
    Rt = R[:, pre][:, :, treat_idx]  # (B, n_pre, n_treat)
    b = Rt.mean(axis=2)  # (B, n_pre)

    if regularize:
        reg = reg_mad(A)
        s2c = A.var(axis=2)
        s2t = Rt.var(axis=2) if n_treat > 1 else np.zeros((B, A.shape[1]))
        total_var = np.maximum(s2c / n_contr + s2t / max(n_treat, 1), 1e-10)
        w = 1.0 / total_var
        W = np.sqrt(w / w.mean(axis=1, keepdims=True))
        A = A * W[:, :, None]
        b = b * W
    else:
        reg = np.zeros(B)

    return pgd_batch(A, b, reg), reg


def fit_lambda(R, n_pre, treat, regularize=True):
    """Fit time weights lambda. Batched.

    Parameters
    ----------
    R : (batch_size, n_time, n_units)
    n_pre : int
    treat : (n_units,) bool
    regularize : bool

    Returns
    -------
    lambd : (batch_size, n_pre)
    reg : (batch_size,)
    """
    B = R.shape[0]
    pre = np.arange(R.shape[1]) < n_pre
    ctrl_idx = np.where(~treat)[0]

    Rc = R[:, :, ctrl_idx]
    A = Rc[:, pre, :].transpose(0, 2, 1)  # (B, n_contr, n_pre)
    b = Rc[:, ~pre, :].mean(axis=1)  # (B, n_contr)

    reg = reg_mad(A) if regularize else np.zeros(B)
    return pgd_batch(A, b, reg), reg


# =============================================================================
# ATT computation
# =============================================================================

def compute_did(Y, n_pre, treat, omega=None, lambd=None):
    """Weighted DID equation. Batched.

    ATT = (Ȳ_treat_post - λ'Ȳ_treat_pre) - (ω'Y_ctrl_post - λ'ω'Y_ctrl_pre)

    Parameters
    ----------
    Y : (batch_size, n_time, n_units)
    n_pre : int
    treat : (n_units,) bool
    omega : (batch_size, n_contr) or None — unit weights (default: uniform)
    lambd : (batch_size, n_pre) or None — time weights (default: uniform)

    Returns
    -------
    att : (batch_size,)
    """
    B = Y.shape[0]
    pre = np.arange(Y.shape[1]) < n_pre
    ctrl_idx = np.where(~treat)[0]
    treat_idx = np.where(treat)[0]

    if omega is None:
        omega = np.full((B, len(ctrl_idx)), 1.0 / len(ctrl_idx))
    if lambd is None:
        lambd = np.full((B, pre.sum()), 1.0 / pre.sum())

    Yc = np.einsum('btn,bn->bt', Y[:, :, ctrl_idx], omega)
    Yt = Y[:, :, treat_idx].mean(axis=2)

    return (Yt[:, ~pre].mean(axis=1) - np.einsum('bt,bt->b', Yt[:, pre], lambd)) - \
           (Yc[:, ~pre].mean(axis=1) - np.einsum('bt,bt->b', Yc[:, pre], lambd))


# =============================================================================
# Full pipeline
# =============================================================================

def asdid(Y, n_pre, treat, X=None, regularize=True, max_donors=10000, return_dict=False):
    """ASDID: Augmented Synthetic Difference-in-Differences. Batched.

    Parameters
    ----------
    Y : (batch_size, n_time, n_units) or (n_time, n_units)
    n_pre : int
    treat : (n_units,) bool
    X : (batch_size, n_time, n_units, n_cov) or None — optional covariates
    regularize : bool
    max_donors : int or None
    return_dict : bool — if True, return full dict; if False, return (att, lambd, omega, donors)

    Returns
    -------
    if return_dict: dict with 'att', 'omega', 'lambd', 'n_pre', 'treat', 'treat_orig',
                    'alpha', 'gamma', 'synth', 'ctrl', 'beta', 'reg_*'
    else: (att, lambd, omega, donors)
    """
    if Y.ndim == 2:
        Y = Y[None]
    B = Y.shape[0]
    treat = np.asarray(treat).ravel()
    treat_orig = treat.copy()

    R, alpha, gamma = demean(Y, n_pre, treat)

    beta = None
    reg_beta = np.zeros(B)
    if X is not None:
        if X.ndim == 3:
            X = X[None]
        beta, reg_beta = fit_beta(R, X, n_pre, treat, regularize=regularize)
        Y = Y - np.einsum('btnc,bc->btn', X, beta)
        R, alpha, gamma = demean(Y, n_pre, treat)

    # Donor selection: screen down if too many control units
    ctrl_idx = np.where(~treat)[0]
    n_contr = len(ctrl_idx)
    donors = np.arange(n_contr)  # default: all controls
    if max_donors is not None and n_contr > max_donors:
        pre = np.arange(R.shape[1]) < n_pre
        treat_idx = np.where(treat)[0]
        A_full = R[:, pre][:, :, ctrl_idx]
        b_full = R[:, pre][:, :, treat_idx].mean(axis=2)
        if regularize:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_contr, max_donors, replace=False)
            reg_screen = reg_mad(A_full[:, :, idx])
        else:
            reg_screen = np.zeros(B)
        donor_idx = select_donors(A_full, b_full, reg_screen, max_donors)
        donors = donor_idx[0]  # indices into ctrl_idx
        keep = np.concatenate([ctrl_idx[donors], treat_idx])
        Y = Y[:, :, keep]
        R = R[:, :, keep]
        treat = np.zeros(len(keep), dtype=bool)
        treat[max_donors:] = True

    omega, reg_om = fit_omega(R, n_pre, treat, regularize=regularize)
    lambd, reg_lam = fit_lambda(R, n_pre, treat, regularize=regularize)
    att = compute_did(Y, n_pre, treat, omega, lambd)

    if not return_dict:
        return att, lambd, omega, donors

    ctrl_r = np.where(~treat)[0]
    synth = np.einsum('btn,bn->bt', R[:, :, ctrl_r], omega)
    ctrl = R[:, :, ctrl_r]

    return {
        'att': att, 'n_pre': n_pre, 'treat_orig': treat_orig, 'treat': treat,
        'omega': omega, 'lambd': lambd, 'alpha': alpha, 'gamma': gamma,
        'synth': synth, 'ctrl': ctrl, 'beta': beta,
        'reg_omega': reg_om, 'reg_lambda': reg_lam, 'reg_beta': reg_beta,
    }


def did(Y, n_pre, treat, return_dict=False):
    """Simple parallel trends DID (uniform weights). Batched.

    Parameters
    ----------
    Y : (batch_size, n_time, n_units) or (n_time, n_units)
    n_pre : int
    treat : (n_units,) bool
    return_dict : bool — if True, return full dict; if False, return (att, lambd, omega, donors)

    Returns
    -------
    if return_dict: dict with 'att', 'omega', 'lambd'
    else: (att, lambd, omega, donors)
    """
    if Y.ndim == 2:
        Y = Y[None]
    B = Y.shape[0]
    treat = np.asarray(treat).ravel()
    n_ctrl = (~treat).sum()
    n_pre_t = (np.arange(Y.shape[1]) < n_pre).sum()

    omega = np.full((B, n_ctrl), 1.0 / n_ctrl)
    lambd = np.full((B, n_pre_t), 1.0 / n_pre_t)
    donors = np.arange(n_ctrl)
    att = compute_did(Y, n_pre, treat, omega, lambd)

    if return_dict:
        return {'att': att, 'omega': omega, 'lambd': lambd, 'donors': donors,
                'n_pre': n_pre, 'treat_orig': treat, 'treat': treat}
    return att, lambd, omega, donors


class ASDID:
    """ASDID estimator object. Holds config, callable as (Y, n_pre, treat) → (att, lambd, omega, donors)."""

    def __init__(self, max_donors=10000, regularize=True, X=None):
        self.max_donors = max_donors
        self.regularize = regularize
        self.X = X

    def __call__(self, Y, n_pre, treat):
        return asdid(Y, n_pre, treat, X=self.X, regularize=self.regularize, max_donors=self.max_donors)


class DID:
    """DID estimator object. Callable as (Y, n_pre, treat) → (att, lambd, omega, donors)."""

    def __call__(self, Y, n_pre, treat):
        return did(Y, n_pre, treat)


# =============================================================================
# Inference
# =============================================================================

def se_jackknife(Y, n_pre, treat, lambd, omega, donors):
    """Analytical leave-one-unit-out JackKnife SE. Batched.

    Parameters
    ----------
    Y : (batch_size, n_time, n_units)
    n_pre : int
    treat : (n_units,) bool
    lambd : (batch_size, n_pre)
    omega : (batch_size, n_donors)
    donors : (n_donors,) — indices into ctrl units

    Returns
    -------
    se : (batch_size,)
    """
    B, n_time, n_units = Y.shape
    pre = np.arange(n_time) < n_pre
    treat = np.asarray(treat).ravel()
    ctrl_idx = np.where(~treat)[0]
    treat_idx = np.where(treat)[0]
    n_treat = len(treat_idx)
    n_donors = len(donors)

    if n_treat < 2:
        import warnings
        warnings.warn("se_jackknife with n_treat=1 severely under-covers. Use se_placebo instead.")

    # Use only donor units + treated for the jackknife
    donor_units = ctrl_idx[donors]

    # Per-unit DID values: d_post - d_pre@lambda
    d_pre_donors = np.einsum('bt,btn->bn', lambd, Y[:, pre][:, :, donor_units])  # (B, n_donors)
    d_post_donors = Y[:, ~pre][:, :, donor_units].mean(axis=1)  # (B, n_donors)
    d_donors = d_post_donors - d_pre_donors  # (B, n_donors)

    d_pre_treat = np.einsum('bt,btn->bn', lambd, Y[:, pre][:, :, treat_idx])  # (B, n_treat)
    d_post_treat = Y[:, ~pre][:, :, treat_idx].mean(axis=1)  # (B, n_treat)
    d_treat = d_post_treat - d_pre_treat  # (B, n_treat)

    jk_atts = []

    # Leave-one-donor-out
    for k in range(n_donors):
        om_k = np.delete(omega, k, axis=1)
        om_k = om_k / om_k.sum(axis=1, keepdims=True)
        d_k = np.delete(d_donors, k, axis=1)
        jk_atts.append(d_treat.mean(axis=1) - (d_k * om_k).sum(axis=1))

    # Leave-one-treatment-out
    if n_treat > 1:
        Yc_full = (d_donors * omega).sum(axis=1)
        for k in range(n_treat):
            jk_atts.append(np.delete(d_treat, k, axis=1).mean(axis=1) - Yc_full)

    jk_atts = np.stack(jk_atts, axis=1)  # (B, n_jk)
    n_jk = jk_atts.shape[1]
    jk_mean = jk_atts.mean(axis=1, keepdims=True)
    return np.sqrt((n_jk - 1) / n_jk * ((jk_atts - jk_mean) ** 2).sum(axis=1))


def se_placebo(estimator, Y, n_pre, treat, X=None, n_placebo=200, conf=90, seed=0):
    """Matched placebo SE. Fully batched.

    Parameters
    ----------
    estimator : callable — (Y, n_pre, treat) → (att, lambd, omega, donors)
    Y : (n_time, n_units) — full panel
    n_pre : int
    treat : (n_units,) bool — treatment mask
    X : (n_time, n_units, n_cov) or None — optional covariates
    n_placebo : int
    conf : float
    seed : int

    Returns
    -------
    se : float
    """
    treat = np.asarray(treat).ravel()
    z = _stats.norm.ppf(1 - (1 - conf / 100) / 2)
    panels = simulate(Y[:, ~treat], n_pre, treat.sum(), att=0,
                      X=X[:, ~treat, :] if X is not None else None,
                      n_samples=n_placebo, seed=seed)
    est_att, _, _, _ = estimator(panels['Y'], n_pre, panels['treat'])
    return np.quantile(np.abs(est_att), conf / 100) / z


def se_bootstrap(estimator, Y, n_pre, treat, X=None, n_boot=200, seed=0, what='units'):
    """Bootstrap SE by resampling units or time periods. Fully batched.

    Note: tends to under-cover for SDID-type estimators. Recommended as a
    diagnostic tool, not for formal inference. Use se_placebo or se_jackknife instead.

    Parameters
    ----------
    estimator : callable — (Y, n_pre, treat) → (att, lambd, omega, donors)
    Y : (n_time, n_units) — full panel
    n_pre : int
    treat : (n_units,) bool — treatment mask
    X : (n_time, n_units, n_cov) or None
    n_boot : int
    seed : int
    what : 'units', 'time', or 'both'

    Returns
    -------
    se : float
    """
    treat = np.asarray(treat).ravel()
    rng = np.random.default_rng(seed)
    n_time, n_units = Y.shape
    ctrl_idx = np.where(~treat)[0]
    treat_idx = np.where(treat)[0]
    n_ctrl, n_treat = len(ctrl_idx), len(treat_idx)
    B = n_boot

    # Resample indices
    if what == 'units':
        ctrl_samples = rng.choice(n_ctrl, (B, n_ctrl), replace=True)
        time_samples = np.tile(np.arange(n_time), (B, 1))
    elif what == 'time':
        ctrl_samples = np.tile(np.arange(n_ctrl), (B, 1))
        time_samples = np.concatenate([
            rng.choice(n_pre, (B, n_pre), replace=True),
            np.tile(np.arange(n_pre, n_time), (B, 1))
        ], axis=1)
    elif what == 'both':
        ctrl_samples = rng.choice(n_ctrl, (B, n_ctrl), replace=True)
        time_samples = np.concatenate([
            rng.choice(n_pre, (B, n_pre), replace=True),
            np.tile(np.arange(n_pre, n_time), (B, 1))
        ], axis=1)
    else:
        raise ValueError(f"what must be 'units', 'time', or 'both', got {what}")

    # Build batched Y: (B, n_time, n_ctrl + n_treat)
    n_new = n_ctrl + n_treat
    treat_new = np.zeros(n_new, dtype=bool)
    treat_new[n_ctrl:] = True

    Y_batch = np.zeros((B, n_time, n_new))
    X_batch = np.zeros((B, n_time, n_new, X.shape[2])) if X is not None else None
    for i in range(B):
        t_idx = time_samples[i]
        cols = np.concatenate([ctrl_idx[ctrl_samples[i]], treat_idx])
        Y_batch[i] = Y[t_idx][:, cols]
        if X is not None:
            X_batch[i] = X[t_idx][:, cols, :]

    att, _, _, _ = estimator(Y_batch, n_pre, treat_new)
    return att.std()


# =============================================================================
# Power Analysis
# =============================================================================

def simulate(Y, n_pre, n_treat, att=0, X=None, n_samples=200, att_type='absolute', seed=0):
    """Generate simulated panels under random treatment assignment.

    Parameters
    ----------
    Y : (n_time, n_units) — control-only panel
    n_pre : int
    n_treat : int — number of units to randomly assign as treated
    att : float or array-like — ATT to inject (0 for placebo)
    X : (n_time, n_units, n_cov) or None — optional covariates
    n_samples : int — number of random assignments per ATT level
    att_type : 'absolute' or 'percent'
    seed : int

    Returns
    -------
    result : dict
        'Y': (B, n_time, n_units), 'X': (B, n_time, n_units, n_cov) or None,
        'treat': (n_units,) bool, 'true_att': (B,), 'att_level': (B,)
    """
    rng = np.random.default_rng(seed)
    n_time, n_units = Y.shape
    att_values = np.atleast_1d(np.asarray(att, dtype=float))
    n_att = len(att_values)
    B = n_att * n_samples

    # Random treatment assignments
    perms = rng.permuted(np.tile(np.arange(n_units), (B, 1)), axis=1)
    treat_batch = np.zeros((B, n_units), dtype=bool)
    treat_batch[np.arange(B)[:, None], perms[:, :n_treat]] = True

    # Reorder columns so treated are always last
    att_level = np.repeat(att_values, n_samples)
    treat_fixed = np.zeros(n_units, dtype=bool)
    treat_fixed[-n_treat:] = True

    Y_batch = np.zeros((B, n_time, n_units))
    X_batch = np.zeros((B, n_time, n_units, X.shape[2])) if X is not None else None
    for i in range(B):
        ti = np.where(treat_batch[i])[0]
        ci = np.where(~treat_batch[i])[0]
        order = np.concatenate([ci, ti])
        Y_batch[i] = Y[:, order]
        if X is not None:
            X_batch[i] = X[:, order, :]
        if att_type == 'percent':
            Y_batch[i, n_pre:, -n_treat:] *= (1 + att_level[i] / 100)
        else:
            Y_batch[i, n_pre:, -n_treat:] += att_level[i]

    # True ATT
    if att_type == 'percent':
        true_att = np.array([Y[n_pre:, np.where(treat_batch[i])[0]].mean() * (att_level[i] / 100)
                             for i in range(B)])
    else:
        true_att = att_level.copy()

    return {'Y': Y_batch, 'X': X_batch, 'treat': treat_fixed,
            'true_att': true_att, 'att_level': att_level}


def compute_power(att, se, conf=90):
    """Analytical power given ATT and SE. Instant.

    Parameters
    ----------
    att : float or array — estimated or hypothesized ATT
    se : float — standard error
    conf : float — confidence level

    Returns
    -------
    power : float or array — probability of rejecting H0: ATT=0
    """
    z = _stats.norm.ppf(1 - (1 - conf / 100) / 2)
    ratio = np.abs(att) / se
    return 1 - _stats.norm.cdf(z - ratio) + _stats.norm.cdf(-z - ratio)


def mde(se, conf, power):
    """Minimum detectable effect.

    The smallest true ATT detectable with given power at given confidence.

    Parameters
    ----------
    se : float — standard error (e.g. from se_placebo)
    conf : float — confidence level (e.g. 90)
    power : float — detection probability (e.g. 0.9)

    Returns
    -------
    mde : float
    """
    z_alpha = _stats.norm.ppf(1 - (1 - conf / 100) / 2)
    z_beta = _stats.norm.ppf(power)
    return se * (z_alpha + z_beta)


def power(estimator, Y, n_pre, n_treat, att, se='auto', n_samples=100, conf=90, seed=0,
          att_type='absolute'):
    """Estimate power at a specific ATT via simulation.

    Parameters
    ----------
    estimator : callable — (Y, n_pre, treat) → (att, lambd, omega, donors)
    Y : (n_time, n_units) — control-only panel
    n_pre : int
    n_treat : int — number of units to randomly assign as treated
    att : float — the ATT to test
    se : float, 'jackknife', or 'placebo'
    n_samples : int
    conf : float
    seed : int
    att_type : 'absolute' or 'percent'

    Returns
    -------
    result : dict — 'power', 'coverage', 'bias', 'se_median'
    """
    df = power_curve(estimator, Y, n_pre, n_treat, att_values=[att], se=se, n_samples=n_samples,
                     conf=conf, seed=seed, att_type=att_type)
    return {
        'power': df['significant'].mean(),
        'coverage': df['covered'].mean(),
        'bias': df['error'].mean(),
        'se_median': df['se'].median(),
    }


def power_curve(estimator, Y, n_pre, n_treat, att_values, se='auto', n_samples=100, conf=90, seed=0,
                att_type='absolute'):
    """Power curve: sweep ATT levels, random treatment assignment. Fully batched.

    Parameters
    ----------
    estimator : callable — (Y, n_pre, treat) → (att, lambd, omega, donors)
    Y : (n_time, n_units) — control-only panel
    n_pre : int
    n_treat : int — number of units to randomly assign as treated
    att_values : array-like — ATT levels to test
    se : float, 'jackknife', or 'placebo' — SE method
    n_samples : int
    conf : float
    seed : int
    att_type : 'absolute' or 'percent'

    Returns
    -------
    df : pd.DataFrame
    """
    import pandas as pd

    att_values = np.asarray(att_values)
    z = _stats.norm.ppf(1 - (1 - conf / 100) / 2)

    panels = simulate(Y, n_pre, n_treat, att=att_values, n_samples=n_samples,
                      att_type=att_type, seed=seed)

    true_att = panels['true_att']
    treat_fixed = panels['treat']

    # Estimate
    est_att, lambd, omega, donors = estimator(panels['Y'], n_pre, treat_fixed)

    # SE
    if isinstance(se, (int, float)):
        se_values = np.full(len(est_att), float(se))
    elif se == 'placebo':
        se_values = np.full(len(est_att), se_placebo(estimator, Y, n_pre, treat_fixed, seed=seed + 1))
    elif se == 'jackknife':
        se_values = se_jackknife(panels['Y'], n_pre, treat_fixed, lambd, omega, donors)
    elif se == 'auto':
        if n_treat >= 2:
            se_values = se_jackknife(panels['Y'], n_pre, treat_fixed, lambd, omega, donors)
        else:
            se_values = np.full(len(est_att), se_placebo(estimator, Y, n_pre, treat_fixed, seed=seed + 1))
    else:
        raise ValueError(f"se must be float, 'auto', 'jackknife', or 'placebo', got {se}")

    att_level = panels['att_level']

    # Metrics
    significant = (est_att - z * se_values > 0) | (est_att + z * se_values < 0)
    covered = (true_att >= est_att - z * se_values) & (true_att <= est_att + z * se_values)

    df = pd.DataFrame({
        'att_level': att_level,
        'true_att': true_att,
        'est_att': est_att,
        'se': se_values,
        'significant': significant,
        'covered': covered,
        'error': est_att - true_att,
    })

    return df





# =============================================================================
# Plotting
# =============================================================================

def plot_asdid(Y, res, max_ctrl_lines=1000, seed=0):
    """Plot panel: raw, demeaned, and synthetic DID in demeaned space.

    Parameters
    ----------
    Y : (n_time, n_units) — original panel
    res : dict — output of asdid(return_dict=True)
    max_ctrl_lines : int — max control units to show in gray
    seed : int

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt

    n_pre = res['n_pre']
    treat = res['treat_orig']
    ctrl_idx = np.where(~treat)[0]
    treat_idx = np.where(treat)[0]
    n_time, n_units = Y.shape
    n_treat = len(treat_idx)
    n_ctrl = len(ctrl_idx)
    times = np.arange(n_time)

    R = res['synth'][0]  # not needed, just for naming
    omega = res['omega']
    att = res['att']
    treat_r = res['treat']
    ctrl_r = np.where(~treat_r)[0]
    treat_r_idx = np.where(treat_r)[0]

    R_synth = res['synth'][0]  # (n_time,)
    # Treated in demeaned space: use alpha/gamma
    alpha = res['alpha'][0]
    gamma = res['gamma'][0]
    Y_treat_dm = Y[:, treat_idx] - alpha[None, treat_idx] - gamma[:, None]
    R_treat_mean = Y_treat_dm.mean(axis=1)

    # Subsample controls for gray lines
    rng = np.random.default_rng(seed)
    ctrl_show = ctrl_idx if n_ctrl <= max_ctrl_lines else ctrl_idx[rng.choice(n_ctrl, max_ctrl_lines, replace=False)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # --- Panel 1: Raw with control units in gray ---
    ax = axes[0]
    for j in ctrl_show:
        ax.plot(times, Y[:, j], color='gray', alpha=0.15, lw=0.5)
    ax.plot(times, Y[:, ctrl_idx].mean(axis=1), color='blue', lw=2, label='Control (mean)')
    if n_treat > 1:
        ax.plot(times, Y[:, treat_idx].mean(axis=1), color='red', lw=2, label='Treated (mean)')
    else:
        ax.plot(times, Y[:, treat_idx[0]], color='red', lw=2, label='Treated')
    ax.axvline(n_pre - 0.5, color='black', alpha=0.4, ls='--')
    ax.set_title('Raw Panel')
    ax.set_xlabel('Time')
    ax.set_ylabel('Y')
    ax.legend()

    # --- Panel 2: Demeaned with control units in gray ---
    ax = axes[1]
    # Reconstruct demeaned full panel from alpha/gamma
    alpha = res['alpha'][0]  # (n_units,)
    gamma = res['gamma'][0]  # (n_time,)
    R_full = Y - alpha[None, :] - gamma[:, None]
    for j in ctrl_show:
        ax.plot(times, R_full[:, j], color='gray', alpha=0.15, lw=0.5)
    ax.plot(times, R_full[:, ctrl_idx].mean(axis=1), color='blue', lw=2, label='Control (mean)')
    ax.plot(times, R_full[:, treat_idx].mean(axis=1), color='red', lw=2, label='Treated')
    ax.axvline(n_pre - 0.5, color='black', alpha=0.4, ls='--')
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_title('Demeaned (two-way FE)')
    ax.set_xlabel('Time')
    ax.legend()

    # --- Panel 3: Synthetic DID in demeaned space ---
    ax = axes[2]
    ax.plot(times, R_treat_mean, color='red', lw=2, label='Treated')
    ax.plot(times, R_synth, color='blue', lw=2, label='Synthetic Control')
    ax.axvline(n_pre - 0.5, color='black', alpha=0.4, ls='--')
    ax.axhline(0, color='grey', lw=0.5, ls='--')
    ax.set_title(f'ASDID (ATT={att[0]:.1f})')
    ax.set_xlabel('Time')
    ax.legend()

    plt.tight_layout()
    return fig, axes


def plot_power_curve(df, title=None, conf=90):
    """Plot power curve from power_curve() output.

    Parameters
    ----------
    df : pd.DataFrame — output of power_curve()
    title : str or None
    conf : float — confidence level (for reference lines)

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    from scipy import stats as _st

    z = _st.norm.ppf(1 - (1 - conf / 100) / 2)

    # Compute per-ATT-level metrics
    def _power_by_sign(group):
        att = group['est_att'].values
        se = group['se'].values
        sig_pos = (att - z * se > 0).mean()
        sig_neg = (att + z * se < 0).mean()
        sig_any = ((att - z * se > 0) | (att + z * se < 0)).mean()
        return {'power_+': sig_pos, 'power_-': sig_neg, 'power_+/-': sig_any}

    import pandas as pd
    pw = df.groupby('att_level').apply(lambda g: pd.Series(_power_by_sign(g))).reset_index()
    coverage = df.groupby('att_level')['covered'].mean()
    error = df.groupby('att_level').agg(mean=('error', 'mean'), se=('error', 'sem'))

    fig, (top, middle, bottom) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    if title:
        fig.suptitle(title, fontsize=16)

    # Power
    top.plot(pw['att_level'], pw['power_-'], '-o', color='red', label='-')
    top.plot(pw['att_level'], pw['power_+'], '-o', color='green', label='+')
    top.plot(pw['att_level'], pw['power_+/-'], '-o', color='black', label='+/-', alpha=0.5)
    top.axhline(1.0, color='black', alpha=0.15)
    top.axhline(0.9, color='black', alpha=0.15, linestyle='--')
    top.axhline(0.0, color='black', alpha=0.15)
    top.set_ylim(-0.05, 1.05)
    top.set_ylabel('Statistical Power')
    top.legend()

    # Coverage
    middle.plot(coverage.index, coverage.values, '-o', color='black', label='coverage')
    middle.axhline(1.0, color='black', alpha=0.15)
    middle.axhline(conf / 100, color='black', alpha=0.15, linestyle='--')
    middle.axhline(0.0, color='black', alpha=0.15)
    middle.set_ylim(-0.05, 1.05)
    middle.set_ylabel('Coverage')
    middle.legend()

    # Error (absolute)
    bottom.plot(error.index, np.zeros(len(error)), color='black', alpha=0.7)
    bottom.plot(error.index, error['mean'], '-o', color='red')
    bottom.errorbar(error.index, error['mean'], error['se'], color='red', alpha=0.5, capsize=3)
    bottom.set_xlabel('ATT')
    bottom.set_ylabel('Error (est - true)')
    # Symmetric y-limits from data
    ymax = (error['mean'].abs() + error['se']).max() * 1.5
    bottom.set_ylim(-ymax, ymax)

    plt.tight_layout()
    return fig, (top, middle, bottom)


def summary(Y, n_pre, treat, res, se=None, conf=90, title='ASDID'):
    """Print a formatted summary of the estimation result.

    Parameters
    ----------
    Y : (n_time, n_units) — original panel
    n_pre : int
    treat : (n_units,) bool
    res : dict — output of asdid(return_dict=True) or did(return_dict=True)
    se : float or array or None — standard error
    conf : float — confidence level
    title : str
    """
    treat = np.asarray(treat).ravel()
    n_time, n_units = Y.shape
    n_treat = treat.sum()
    n_ctrl = (~treat).sum()
    n_post = n_time - n_pre
    att = res['att'][0]
    omega = res['omega']
    lambd = res['lambd']
    n_donors = omega.shape[1]

    z = _stats.norm.ppf(1 - (1 - conf / 100) / 2)
    W = 78

    # Observed and counterfactual
    treat_idx = np.where(treat)[0]
    ctrl_idx = np.where(~treat)[0]
    observed = Y[n_pre:, treat_idx].mean()
    # Counterfactual = observed - ATT
    counterfactual = observed - att

    # Percentage effect
    pct_att = att / abs(counterfactual) * 100 if counterfactual != 0 else 0
    # Cumulative
    cum_att = att * n_post

    lines = [
        f'╭{"─" * W}╮',
        f'│{title:^{W}}│',
        f'├{"═" * W}┤',
        f'│{"Panel":^{W}}│',
        f'│{f"  Time Periods: {n_time} ({n_pre}/{n_post}) total (pre/post)":<{W}}│',
        f'│{f"  Units: {n_units} ({n_ctrl}/{n_treat}) total (contr/treat)":<{W}}│',
        f'│{f"  Donors: {n_donors}":<{W}}│',
        f'├{"─" * W}┤',
        f'│{"ATT":^{W}}│',
    ]

    if se is not None:
        se_val = se[0] if hasattr(se, '__len__') else se
        ci_lb = att - z * se_val
        ci_ub = att + z * se_val
        sig = '(-)' if ci_ub < 0 else '(+)' if ci_lb > 0 else ''

        # Percentage: scale by 100/|counterfactual|
        pct_scale = 100 / abs(counterfactual) if counterfactual != 0 else 0
        pct_att = att * pct_scale
        pct_se = se_val * pct_scale
        pct_ci_lb = pct_att - z * pct_se
        pct_ci_ub = pct_att + z * pct_se

        # Cumulative: scale by n_post
        cum_att = att * n_post
        cum_se = se_val * n_post
        cum_ci_lb = cum_att - z * cum_se
        cum_ci_ub = cum_att + z * cum_se

        lines += [
            f'│{f"  Effect (±SE): {att:.2f} (±{se_val:.4f})":<{W}}│',
            f'│{f"  Confidence Interval ({conf:.0f}%): [{ci_lb:.2f} , {ci_ub:.2f}] {sig}":<{W}}│',
            f'│{f"  Observed: {observed:.2f}":<{W}}│',
            f'│{f"  Counter Factual: {counterfactual:.2f}":<{W}}│',
            f'├{"─" * W}┤',
            f'│{"Percentage":^{W}}│',
            f'│{f"  Effect (±SE): {pct_att:.2f} (±{pct_se:.2f})":<{W}}│',
            f'│{f"  Confidence Interval ({conf:.0f}%): [{pct_ci_lb:.2f} , {pct_ci_ub:.2f}] {sig}":<{W}}│',
            f'│{f"  Observed: {observed * pct_scale:.2f}":<{W}}│',
            f'│{f"  Counter Factual: 100.00":<{W}}│',
            f'├{"─" * W}┤',
            f'│{"Cumulative":^{W}}│',
            f'│{f"  Effect (±SE): {cum_att:.2f} (±{cum_se:.2f})":<{W}}│',
            f'│{f"  Confidence Interval ({conf:.0f}%): [{cum_ci_lb:.2f} , {cum_ci_ub:.2f}] {sig}":<{W}}│',
            f'│{f"  Observed: {observed * n_post:.2f}":<{W}}│',
            f'│{f"  Counter Factual: {counterfactual * n_post:.2f}":<{W}}│',
        ]
    else:
        lines += [
            f'│{f"  Effect: {att:.4f}":<{W}}│',
            f'│{f"  Observed: {observed:.2f}":<{W}}│',
            f'│{f"  Counter Factual: {counterfactual:.2f}":<{W}}│',
        ]

    lines.append(f'╰{"─" * W}╯')
    print('\n'.join(lines))
