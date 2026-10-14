"""Bootstrap machinery for structural-break and variance tests.

Used by NB02 (change-point tests, break-date CIs) and NB03 (variance ratio
tests, effect-size CIs). Kept as a single module because the notebooks share
one resampler (circular block bootstrap) and one p-value convention.
Calibration behaviour under a common-variance AR(1) null is checked in
``validation/bootstrap_calibration.py``.
"""

import numpy as np
import ruptures as rpt
from joblib import Parallel, delayed

MIN_SEG_LEN = 30


def circular_block_bootstrap(residuals, block_length, rng):
    """Draw one circular-block-bootstrap sample of the same length as ``residuals``.

    ``rng`` is a ``numpy.random.Generator``. Block starts are drawn uniformly
    on 0..n-1; blocks wrap around modulo n so no starts near the tail are
    disadvantaged.
    """
    residuals = np.asarray(residuals)
    n = len(residuals)
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(s + np.arange(block_length)) % n for s in starts])[:n]
    return residuals[idx]


def davison_hinkley_p(observed, boot_stats):
    """Upper-tail Davison-Hinkley bootstrap p-value.

    p = (1 + #{boot >= observed}) / (B + 1)

    This is a one-sided *upper-tail* p-value against the empirical null
    distribution. Callers who want a two-sided test apply it to an absolute
    statistic (e.g. |log variance ratio|) so that "extreme in either
    direction" maps to the upper tail. Both numerator and denominator include
    the observed value, so the p-value can never be exactly 0 and equals
    1/(B+1) when no bootstrap value equals or exceeds the observed one.
    """
    boot_stats = np.asarray(boot_stats)
    B = len(boot_stats)
    return (1 + int(np.sum(boot_stats >= observed))) / (B + 1)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values.

    Given a family of raw p-values, returns adjusted p-values in the original
    order. Reject at level alpha where adjusted p <= alpha; controls the
    family-wise error rate and is uniformly at least as powerful as Bonferroni.
    """
    p = np.asarray(p_values, dtype=float)
    k = len(p)
    order = np.argsort(p)
    adjusted = np.empty(k)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (k - rank) * p[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def fit_ssr(signal, K, min_size=MIN_SEG_LEN):
    """Fit exact K-break Dynp model on ``signal``; return (SSR, break points).

    Break points are internal indices (not the final endpoint). K=0 returns
    the SSR under a single mean and an empty list.
    """
    signal = np.asarray(signal)
    if K == 0:
        m = signal.mean()
        return float(((signal - m) ** 2).sum()), []
    algo = rpt.Dynp(model="l2", jump=1, min_size=min_size).fit(signal)
    bkps = algo.predict(n_bkps=K)
    prev = 0
    ssr = 0.0
    for bp in bkps:
        seg = signal[prev:bp]
        ssr += float(((seg - seg.mean()) ** 2).sum())
        prev = bp
    return ssr, bkps[:-1]


def null_fit(signal, K0, min_size=MIN_SEG_LEN):
    """Fit K0-break model; return (fitted values, residuals, break points)."""
    signal = np.asarray(signal)
    _, bkps = fit_ssr(signal, K0, min_size=min_size)
    fitted = np.empty_like(signal, dtype=float)
    prev = 0
    for bp in bkps + [len(signal)]:
        fitted[prev:bp] = signal[prev:bp].mean()
        prev = bp
    return fitted, signal - fitted, bkps


def one_break_bootstrap_iter(seed, fitted, residuals, K0, K1, block_length):
    """One bootstrap iteration for the sequential-break test.

    Resamples residuals under the K0-break null, adds fitted values, fits both
    K0 and K1 models to the bootstrap signal, and returns
    (SSR(K0) - SSR(K1), K1 break points).
    """
    rng = np.random.default_rng(seed)
    boot_res = circular_block_bootstrap(residuals, block_length, rng)
    boot_signal = fitted + boot_res
    ssr_K0, _ = fit_ssr(boot_signal, K0)
    ssr_K1, bkps_K1 = fit_ssr(boot_signal, K1)
    return ssr_K0 - ssr_K1, bkps_K1


def one_break_date_iter(seed, fitted, residuals, K, block_length):
    """One bootstrap iteration for break-date confidence intervals.

    Semi-parametric bootstrap under a K-break null: resample residuals,
    add back the K-break fitted values, refit K breaks, return the break
    points.
    """
    rng = np.random.default_rng(seed)
    boot_res = circular_block_bootstrap(residuals, block_length, rng)
    boot_signal = fitted + boot_res
    _, bkps = fit_ssr(boot_signal, K)
    return bkps


def run_sequential_bootstrap(signal, K0, K1, B, block_length, seed):
    """Run the K0-vs-K1 sequential break-count bootstrap in parallel.

    Returns (observed_stat, bootstrap_stats, bootstrap_break_points).
    """
    signal = np.asarray(signal)
    obs_ssr_K0, _ = fit_ssr(signal, K0)
    obs_ssr_K1, _ = fit_ssr(signal, K1)
    obs_stat = obs_ssr_K0 - obs_ssr_K1
    fitted, residuals, _ = null_fit(signal, K0)
    ss = np.random.SeedSequence(seed).spawn(B)
    out = Parallel(n_jobs=-1)(
        delayed(one_break_bootstrap_iter)(cs, fitted, residuals, K0, K1, block_length)
        for cs in ss
    )
    boot_stats = np.array([o[0] for o in out])
    boot_bkps = [o[1] for o in out]
    return obs_stat, boot_stats, boot_bkps


def _one_variance_null_iter(seed, pooled, n_A, n_B, block_length):
    """One iteration of the pooled-null variance test.

    Bootstraps the pooled residuals, splits into the original sizes, computes
    the absolute log variance ratio.
    """
    rng = np.random.default_rng(seed)
    boot = circular_block_bootstrap(pooled, block_length, rng)
    v_A = np.var(boot[:n_A], ddof=1)
    v_B = np.var(boot[n_A : n_A + n_B], ddof=1)
    return abs(np.log(v_A) - np.log(v_B))


def bootstrap_variance_test(residuals_A, residuals_B, B, block_length, seed):
    """Two-sided pooled-null block-bootstrap test of equal variance.

    Statistic: |log(var(A)) - log(var(B))|. Under the null the two regimes'
    residuals share one distribution; we pool them, circular-block-bootstrap
    the pool, split into the original sizes, and recompute the statistic.

    Inputs must be centred within their own group (mean-zero within each
    regime). Otherwise a between-group mean shift would enter the pooled
    variance under the null and distort the reference distribution. This
    condition holds for the regime-mean-detrended residuals used in NB03.

    Returns (observed_stat, boot_stats, p_value).
    """
    a = np.asarray(residuals_A, dtype=float)
    b = np.asarray(residuals_B, dtype=float)
    obs = abs(np.log(np.var(a, ddof=1)) - np.log(np.var(b, ddof=1)))
    pooled = np.concatenate([a, b])
    n_A, n_B = len(a), len(b)
    ss = np.random.SeedSequence(seed).spawn(B)
    boot = Parallel(n_jobs=-1)(
        delayed(_one_variance_null_iter)(cs, pooled, n_A, n_B, block_length)
        for cs in ss
    )
    boot = np.array(boot)
    return obs, boot, davison_hinkley_p(obs, boot)


def _one_log_ratio_iter(seed, a, b, block_length):
    """One within-regime bootstrap iteration for effect-size CI.

    Each regime is resampled separately (unlike the pooled null); returns the
    signed log variance ratio of the resamples.
    """
    rng = np.random.default_rng(seed)
    boot_a = circular_block_bootstrap(a, block_length, rng)
    boot_b = circular_block_bootstrap(b, block_length, rng)
    return np.log(np.var(boot_a, ddof=1)) - np.log(np.var(boot_b, ddof=1))


def bootstrap_log_ratio_ci(residuals_A, residuals_B, B, block_length, seed, alpha=0.05):
    """Within-regime block bootstrap of the signed log variance ratio.

    Each regime is resampled separately (no pooling); the interval quantifies
    sampling variability of the effect size, not evidence against the null.

    Returns (observed_log_ratio, ci_lower, ci_upper, boot_log_ratios).
    """
    a = np.asarray(residuals_A, dtype=float)
    b = np.asarray(residuals_B, dtype=float)
    obs = np.log(np.var(a, ddof=1)) - np.log(np.var(b, ddof=1))
    ss = np.random.SeedSequence(seed).spawn(B)
    boot = Parallel(n_jobs=-1)(
        delayed(_one_log_ratio_iter)(cs, a, b, block_length) for cs in ss
    )
    boot = np.array(boot)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return obs, float(lo), float(hi), boot
