"""Test for superior predictive ability (Hansen 2005) with a studentised
statistic, and the StepM procedure (Romano and Wolf 2005) on it.

``arch``'s ``SPA`` compares the un-studentised maximum of the mean loss
differentials: White's (2000) statistic with Hansen's recentring rules.
This module divides each differential by its stationary-bootstrap
long-run variance and re-studentises every resample (percentile-t). With
``studentize=False`` it reproduces ``arch``'s p-values exactly for the
same seed, which the unit tests check.

Loss differentials are d = L_benchmark - L_model, positive when the model
is better. The null is that no model beats the benchmark.
"""

import numpy as np
from arch.bootstrap import StationaryBootstrap


def long_run_variance(diffs, block_length):
    """Variance implied by the stationary bootstrap with the given mean
    block length (Hansen 2005, eq. 10), one value per column."""
    t = diffs.shape[0]
    demeaned = diffs - diffs.mean(0)
    p = 1.0 / block_length
    lags = np.arange(1, t)
    kappa = (1 - lags / t) * (1 - p) ** lags + (lags / t) * (1 - p) ** (t - lags)
    var = np.sum(demeaned**2, 0) / t
    for lag, w in zip(lags, kappa):
        var = var + 2 * w * np.sum(demeaned[: t - lag] * demeaned[lag:], 0) / t
    return var


def spa_test(loss_benchmark, loss_models, block_length, reps, seed, studentize=True):
    """Return the per-model statistics, the lower / consistent / upper
    p-values, and the simulated null maxima (reps x 3)."""
    diffs = np.asarray(loss_benchmark, dtype=float).reshape(-1, 1) - np.asarray(
        loss_models, dtype=float
    )
    t, k = diffs.shape
    dbar = diffs.mean(0)
    omega = np.sqrt(long_run_variance(diffs, block_length))
    # Hansen's rule for which poor models are left un-recentred always uses
    # the studentised value, as in arch.
    valid = np.sqrt(t) * dbar / omega >= -np.sqrt(2 * np.log(np.log(t)))
    scale = omega if studentize else np.ones(k)
    stat = np.sqrt(t) * dbar / scale
    centres = np.vstack(
        [np.where(dbar < 0, 0.0, dbar), np.where(valid, dbar, 0.0), dbar]
    )
    bootstrap = StationaryBootstrap(block_length, diffs, seed=seed)
    sim_max = np.empty((reps, 3))
    for b, (pos, _) in enumerate(bootstrap.bootstrap(reps)):
        dstar = pos[0]
        if studentize:
            scale_star = np.sqrt(long_run_variance(dstar, block_length))
        else:
            scale_star = scale
        z = np.sqrt(t) * (dstar.mean(0)[None, :] - centres) / scale_star[None, :]
        sim_max[b] = z.max(1)
    pvalues = dict(
        zip(("lower", "consistent", "upper"), (sim_max > stat.max()).mean(0))
    )
    return stat, pvalues, sim_max


def stepm(loss_benchmark, loss_models, block_length, reps, seed, size, names):
    """Models that beat the benchmark at the given size: reject those whose
    statistic exceeds the (1 - size) quantile of the consistent null
    maximum, drop them, and repeat on the rest."""
    loss_models = np.asarray(loss_models, dtype=float)
    remaining = list(range(len(names)))
    superior = []
    while remaining:
        stat, _, sim_max = spa_test(
            loss_benchmark, loss_models[:, remaining], block_length, reps, seed
        )
        crit = np.quantile(sim_max[:, 1], 1 - size)
        reject = [m for m, s in zip(remaining, stat) if s > crit]
        if not reject:
            break
        superior.extend(reject)
        remaining = [m for m in remaining if m not in reject]
    return [names[m] for m in sorted(superior)]
