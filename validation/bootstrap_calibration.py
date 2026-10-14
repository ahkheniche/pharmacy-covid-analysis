"""Calibration check for the NB03 variance-ratio bootstrap.

Simulates pairs of regimes with a common variance (the null) and serially
correlated noise, then measures:

* the false-positive rate of ``bootstrap_variance_test`` at the 5% level
  (should be close to 0.05 if the test is well calibrated);
* the coverage of the nominal 95% interval from ``bootstrap_log_ratio_ci``
  for the true log variance ratio, which is 0 under the null.

Regime sizes match NB03: pre-COVID (92 days) against the acute crash (58),
partial rebound (176) and lower plateau (143). Noise is a stationary AR(1)
process; phi = 0 gives independent days, phi = 0.3 gives moderate positive
autocorrelation. The two regimes are taken as consecutive stretches of one
simulated series, as they are in the data.

Run from the repository root:

    python validation/bootstrap_calibration.py              # full run
    python validation/bootstrap_calibration.py --n-sim 100  # quick check

The full run (500 simulations per setting) takes several minutes.
"""

import argparse

import numpy as np

from pharmacy_covid_analysis.bootstrap import (
    bootstrap_log_ratio_ci,
    bootstrap_variance_test,
)

REGIME_SIZES = [(92, 58), (92, 176), (92, 143)]
PHIS = [0.0, 0.3]
ALPHA = 0.05


def ar1(n, phi, rng):
    """Stationary AR(1) series with unit innovation variance."""
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def calibrate(n_a, n_b, phi, n_sim, B, block_length, seed):
    """Return (false-positive rate, interval coverage) for one setting."""
    rejections, covered = 0, 0
    for s in range(n_sim):
        rng = np.random.default_rng([seed, n_a, n_b, int(phi * 100), s])
        z = ar1(n_a + n_b, phi, rng)
        a = z[:n_a] - z[:n_a].mean()
        b = z[n_a:] - z[n_a:].mean()
        _, _, p = bootstrap_variance_test(a, b, B=B, block_length=block_length, seed=s)
        _, lo, hi, _ = bootstrap_log_ratio_ci(
            b, a, B=B, block_length=block_length, seed=10_000 + s, alpha=ALPHA
        )
        rejections += p <= ALPHA
        covered += lo <= 0.0 <= hi
    return rejections / n_sim, covered / n_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-sim", type=int, default=500)
    parser.add_argument("--B", type=int, default=199)
    parser.add_argument("--block-length", type=int, default=14)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    se_size = np.sqrt(ALPHA * (1 - ALPHA) / args.n_sim)
    se_cov = np.sqrt(0.95 * 0.05 / args.n_sim)
    print(
        f"{args.n_sim} simulations per setting, B = {args.B}, "
        f"block length = {args.block_length} days"
    )
    print(
        f"Monte Carlo standard error: about {se_size:.3f} for the rejection "
        f"rate, {se_cov:.3f} for coverage\n"
    )
    print(
        f"{'regime sizes':>14} {'phi':>5} {'reject at 5%':>14} {'95% CI coverage':>17}"
    )
    for n_a, n_b in REGIME_SIZES:
        for phi in PHIS:
            size, coverage = calibrate(
                n_a, n_b, phi, args.n_sim, args.B, args.block_length, args.seed
            )
            print(
                f"{f'{n_a} vs {n_b}':>14} {phi:>5.1f} {size:>14.3f} {coverage:>17.3f}"
            )


if __name__ == "__main__":
    main()
