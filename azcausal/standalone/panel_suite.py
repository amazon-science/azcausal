"""Panel Suite — Synthetic DGPs for benchmarking panel estimators."""
import numpy as np


def parallel_trends(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """DID-friendly: unit FE + time FE + noise."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n_time, n_units)) * 0.5
    Y += rng.standard_normal(n_units)[None, :] + rng.standard_normal(n_time)[:, None] * 0.5
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def factor_model(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """Non-parallel trends: latent factor with heterogeneous loadings."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.3)
    loadings = rng.random(n_units) * 2
    loadings[-n_treat:] = 1.5 + rng.random(n_treat) * 0.5
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * 0.3
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def heteroscedastic(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, n_shocks=5, seed=None):
    """Unit-specific shocks at random time periods."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n_time, n_units)) * 0.3 + rng.standard_normal(n_units)[None, :]
    for t in rng.choice(n_pre, n_shocks, replace=False):
        Y[t] += rng.standard_normal(n_units) * 3.0
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def diverging_trends(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """Treated units have a different pre-trend (assumption violation)."""
    rng = np.random.default_rng(seed)
    Y = rng.standard_normal((n_time, n_units)) * 0.3 + rng.standard_normal(n_units)[None, :]
    Y[:, -n_treat:] += np.linspace(0, 2, n_time)[:, None] * 0.5
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def sparse_donors(n_time=30, n_units=50, n_pre=20, n_treat=5, n_good=5, att=3.0, seed=None):
    """Few good donors hidden among many noisy controls."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    Y = rng.standard_normal((n_time, n_units)) * 2.0
    good = rng.choice(n_units - n_treat, n_good, replace=False)
    Y[:, good] = factor[:, None] * (0.8 + 0.2 * rng.random(n_good))[None, :] + rng.standard_normal((n_time, n_good)) * 0.3
    Y[:, -n_treat:] = factor[:, None] + rng.standard_normal((n_time, n_treat)) * 0.3
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def outlier_times(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """Extreme outlier time periods affecting units differently."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * 0.3
    for t in [5, 12, 18]:
        Y[t] += rng.standard_normal(n_units) * 8.0
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def increasing_variance(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """Noise grows over time."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    noise_scale = np.linspace(0.1, 3.0, n_time)
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * noise_scale[:, None]
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def outlier_unit(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """One extremely noisy control unit."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * 0.3
    Y[:, 0] += rng.standard_normal(n_time) * 10.0
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def noisy_treated(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, seed=None):
    """Treated units have much higher variance than controls."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * 0.3
    Y[:, -n_treat:] += rng.standard_normal((n_time, n_treat)) * 1.5
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def many_treated(n_time=30, n_units=120, n_pre=20, n_treat=100, att=3.0, seed=None):
    """Many treated units, few controls."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * 0.3
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def ar1_errors(n_time=30, n_units=20, n_pre=20, n_treat=5, att=3.0, rho=0.8, seed=None):
    """Autocorrelated errors."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    errors = np.zeros((n_time, n_units))
    errors[0] = rng.standard_normal(n_units) * 0.5
    for t in range(1, n_time):
        errors[t] = rho * errors[t - 1] + rng.standard_normal(n_units) * 0.3
    Y = factor[:, None] * loadings[None, :] + errors
    Y[n_pre:, -n_treat:] += att
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


def growing_att(n_time=30, n_units=20, n_pre=20, n_treat=5, att_start=1.0, att_end=5.0, seed=None):
    """Treatment effect grows linearly over post-period."""
    rng = np.random.default_rng(seed)
    factor = np.cumsum(rng.standard_normal(n_time) * 0.2)
    loadings = 0.5 + rng.random(n_units)
    Y = factor[:, None] * loadings[None, :] + rng.standard_normal((n_time, n_units)) * 0.3
    n_post = n_time - n_pre
    Y[n_pre:, -n_treat:] += np.linspace(att_start, att_end, n_post)[:, None]
    treat = np.zeros(n_units, dtype=bool); treat[-n_treat:] = True
    return Y, n_pre, treat


# All scenarios as a dict for easy iteration
ALL = {
    'parallel_trends': parallel_trends,
    'factor_model': factor_model,
    'heteroscedastic': heteroscedastic,
    'diverging_trends': diverging_trends,
    'sparse_donors': sparse_donors,
    'outlier_times': outlier_times,
    'increasing_variance': increasing_variance,
    'outlier_unit': outlier_unit,
    'noisy_treated': noisy_treated,
    'many_treated': many_treated,
    'ar1_errors': ar1_errors,
    'growing_att': growing_att,
}
