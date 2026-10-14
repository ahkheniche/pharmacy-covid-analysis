"""Size check for the NB05 forecast comparison under a data-based null.

NB05 tests each of five models against seasonal-naive with a one-sided
Diebold-Mariano test (HLN correction), Holm-adjusted within each horizon
at 2.5%. This script measures how often that procedure rejects when no
model beats the benchmark, using the data's own loss differentials:

1. take the five loss differentials d = |e_SN| - |e_model| at h = 1 and
   h = 7 from ``validation/nb05_rolling_forecasts.csv`` (written by NB05);
2. recentre each to mean zero, which imposes the null while keeping the
   dependence across days and across models and the tails of the data;
3. draw null datasets by stationary block bootstrap and apply the tests.

Three procedures are applied to the same null draws: DM with Holm (the
test NB05 uses); the SPA test in ``arch``, whose statistic is the
un-studentised maximum of the mean loss differentials (White's statistic
with Hansen's recentring); and Hansen's studentised SPA with percentile-t
resampling (``pharmacy_covid_analysis.spa``). Every rate is a Monte Carlo
estimate; its standard error is printed, and runs with fewer draws or
another seed move the rates by about that much.

The last section applies both SPA versions to the real forecasts. The
un-studentised statistic gives the noisiest differentials the most weight
in the bootstrap maximum (Hansen 2005, section 3); the diagnostic shows
this on the data by adding, to SARIMA alone, one of the Prophet variants
that are worse than the benchmark in point terms.

Run from the repository root, after NB05 has been executed:

    python validation/dm_calibration.py                 # full run
    python validation/dm_calibration.py --n-sim 100     # quick check

The full run takes 10-15 minutes; the studentised SPA, which
re-estimates five long-run variances in every resample, is most of it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from arch.bootstrap import SPA, StationaryBootstrap
from statsmodels.stats.multitest import multipletests

from pharmacy_covid_analysis.forecast_tests import dm_test_differential
from pharmacy_covid_analysis.spa import spa_test, stepm

FORECASTS = Path(__file__).with_name("nb05_rolling_forecasts.csv")
CANDIDATES = [
    "sarima",
    "prophet_default",
    "prophet_variant_A",
    "prophet_variant_B",
    "prophet_variant_C",
]
HORIZONS = [1, 7]
ALPHA = 0.025


def differentials(results, h):
    """Loss differentials d = L_SN - L_model (days x models) at horizon h,
    plus the benchmark and model losses they came from."""
    sub = results[results["horizon"] == h]
    loss_sn = (sub["seasonal_naive"] - sub["actual"]).abs().to_numpy()
    losses = sub[CANDIDATES].sub(sub["actual"], axis=0).abs()
    return loss_sn[:, None] - losses.to_numpy(), loss_sn, losses


def arch_spa_p(loss_benchmark, loss_models, block_length, reps, seed):
    spa = SPA(
        loss_benchmark,
        loss_models,
        block_size=block_length,
        reps=reps,
        bootstrap="stationary",
        seed=seed,
    )
    spa.compute()
    return spa.pvalues["consistent"]


def size_under_null(d0, h, n_sim, reps, block_length, seed):
    """Rejection rates at 2.5% of the three procedures on the same draws."""
    k = d0.shape[1]
    per_model = np.zeros(k)
    holm = arch_spa = stud_spa = fallbacks = 0
    zeros = np.zeros(d0.shape[0])
    draws = StationaryBootstrap(block_length, d0, seed=seed).bootstrap(n_sim)
    for i, (pos, _) in enumerate(draws):
        d_null = pos[0]
        pvals = []
        for col in range(k):
            _, p, _, _, fallback = dm_test_differential(d_null[:, col], h)
            pvals.append(p)
            fallbacks += fallback
        reject, _, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
        per_model += np.asarray(pvals) < ALPHA
        holm += reject.any()
        # benchmark loss 0 and model loss -d give the differential d back
        arch_spa += arch_spa_p(zeros, -d_null, block_length, reps, seed + i) < ALPHA
        _, p_stud, _ = spa_test(zeros, -d_null, block_length, reps, seed + i)
        stud_spa += p_stud["consistent"] < ALPHA
    rates = {
        "DM + Holm": holm / n_sim,
        "arch SPA (un-studentised)": arch_spa / n_sim,
        "studentised SPA": stud_spa / n_sim,
    }
    return per_model / n_sim, rates, fallbacks


def real_forecasts(loss_sn, losses, block_length, reps, seed):
    """Both SPA versions on the real forecasts, and the diagnostic."""
    names = list(losses.columns)
    worse = [c for c in names if losses[c].mean() > loss_sn.mean()]
    lines = []
    for block in (block_length, 2 * block_length):
        p_arch = arch_spa_p(loss_sn, losses.to_numpy(), block, reps, seed)
        _, p_stud, _ = spa_test(loss_sn, losses.to_numpy(), block, reps, seed)
        named = stepm(loss_sn, losses.to_numpy(), block, reps, seed, ALPHA, names)
        lines.append(
            f"block {block}: arch SPA p = {p_arch:.3f}; studentised SPA "
            f"p = {p_stud['consistent']:.3f}, StepM at 2.5% names "
            f"{', '.join(named) or 'none'}"
        )
    sarima = losses[["sarima"]].to_numpy()
    diag = [
        f"SARIMA alone: {arch_spa_p(loss_sn, sarima, block_length, reps, seed):.3f}"
    ]
    for c in worse:
        pair = losses[["sarima", c]].to_numpy()
        diag.append(
            f"SARIMA + {c}: {arch_spa_p(loss_sn, pair, block_length, reps, seed):.3f}"
        )
    lines.append(f"arch SPA diagnostic, block {block_length}: " + "; ".join(diag))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-sim", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    results = pd.read_csv(FORECASTS)
    print(
        f"Null datasets by stationary bootstrap, block length "
        f"{args.block_length} days; {args.n_sim} draws, the same for every "
        f"procedure; {args.reps} bootstrap replications per SPA test."
    )
    for h in HORIZONS:
        d, _, _ = differentials(results, h)
        d0 = d - d.mean(axis=0)
        per_model, rates, fallbacks = size_under_null(
            d0, h, args.n_sim, args.reps, args.block_length, args.seed
        )
        print()
        print(f"h = {h}: {d0.shape[0]} forecasts, {d0.shape[1]} models")
        print(
            "  DM one-sided, per-model false-positive rate at 2.5%: "
            + ", ".join(f"{r:.3f}" for r in per_model)
        )
        print("  Family-wise / overall rejection rate at nominal 2.5%:")
        for name, rate in rates.items():
            se = np.sqrt(rate * (1 - rate) / args.n_sim)
            print(f"    {name:<27} {rate:.3f}  (s.e. {se:.3f})")
        print(
            f"  Bartlett fallback in DM: {fallbacks} of "
            f"{args.n_sim * d0.shape[1]} tests"
        )

    print()
    print("Both SPA versions on the real forecasts (10,000 replications):")
    for h in HORIZONS:
        _, loss_sn, losses = differentials(results, h)
        for line in real_forecasts(
            loss_sn, losses, args.block_length, 10_000, args.seed
        ):
            print(f"  h = {h}, {line}")


if __name__ == "__main__":
    main()
