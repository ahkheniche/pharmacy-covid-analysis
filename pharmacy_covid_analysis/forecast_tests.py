"""Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.

Shared by NB05 and ``validation/dm_calibration.py`` so that the test and
its size check follow the same rule.
"""

import numpy as np
from scipy import stats


def dm_variance(d, h):
    """Variance of the mean loss differential, h - 1 autocovariance lags.

    Rectangular kernel (the Diebold-Mariano definition); Bartlett weights
    at the same bandwidth if that is not positive.

    Returns the variance and whether the Bartlett fallback was used.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var > 0:
        return var, False
    bartlett = [1 - k / h for k in range(h)]
    weighted = sum(w * g for w, g in zip(bartlett[1:], gamma[1:]))
    return (gamma[0] + 2 * weighted) / n, True


def dm_test_hln(e_model, e_base, h):
    """One-sided Diebold-Mariano test on absolute-error loss (HLN).

    e_model, e_base: forecast errors (actual - forecast) at the same
        horizon.
    h: forecast horizon; the variance uses h - 1 autocovariance lags,
        since h-step-ahead errors are at most MA(h - 1).

    The alternative is lower expected absolute error for the model.

    Returns the HLN-corrected statistic, its one-sided p-value, the mean
    loss differential d = |e_base| - |e_model|, its corrected standard
    error, and whether the Bartlett fallback was used.
    """
    d = np.abs(np.asarray(e_base, dtype=float)) - np.abs(
        np.asarray(e_model, dtype=float)
    )
    return dm_test_differential(d, h)


def dm_test_differential(d, h):
    """The test in ``dm_test_hln`` applied to a loss differential d
    (positive when the model is better), for use when d is given
    directly, as in the calibration script."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dbar = d.mean()
    var_dbar, fallback = dm_variance(d, h)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dbar / np.sqrt(var_dbar) * correction
    p_one_sided = stats.t.sf(stat, df=n - 1)
    se = np.sqrt(var_dbar) / correction
    return stat, p_one_sided, dbar, se, fallback
