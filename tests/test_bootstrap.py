"""Unit tests for pharmacy_covid_analysis.bootstrap."""

import numpy as np
import pytest

from pharmacy_covid_analysis.bootstrap import (
    bootstrap_log_ratio_ci,
    bootstrap_variance_test,
    circular_block_bootstrap,
    davison_hinkley_p,
    fit_ssr,
    holm_adjust,
)


def test_circular_block_bootstrap_returns_correct_length():
    rng = np.random.default_rng(0)
    x = np.arange(100, dtype=float)
    out = circular_block_bootstrap(x, block_length=7, rng=rng)
    assert out.shape == (100,)


def test_circular_block_bootstrap_on_constant_returns_constant():
    # Every draw of a constant vector must be the same constant, wrap-around
    # included. Catches indexing bugs in the block-concat step.
    rng = np.random.default_rng(1)
    x = np.full(53, 3.14)
    for block_length in (1, 3, 7, 14, 28, 53):
        out = circular_block_bootstrap(x, block_length=block_length, rng=rng)
        assert out.shape == (53,)
        assert np.all(out == 3.14)


def test_davison_hinkley_p_no_bootstrap_exceeds_observed():
    # If no bootstrap value >= observed, p == 1 / (B + 1) exactly.
    obs = 100.0
    boot = np.zeros(999)
    p = davison_hinkley_p(obs, boot)
    assert p == pytest.approx(1.0 / 1000.0)


def test_davison_hinkley_p_all_bootstrap_exceeds_observed():
    obs = 0.0
    boot = np.ones(999)
    p = davison_hinkley_p(obs, boot)
    assert p == pytest.approx(1.0)


def test_holm_adjust_matches_bonferroni_when_all_ps_equal():
    p = np.array([0.02, 0.02, 0.02])
    adj = holm_adjust(p)
    assert np.allclose(adj, np.minimum(1.0, 3 * p))


def test_holm_adjust_preserves_order_and_monotonicity():
    p = np.array([0.001, 0.04, 0.03, 0.20])
    adj = holm_adjust(p)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_bootstrap_variance_test_returns_finite_and_valid_p():
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    b = rng.normal(size=180)
    obs, boot, p = bootstrap_variance_test(a, b, B=99, block_length=7, seed=42)
    assert np.isfinite(obs)
    assert boot.shape == (99,)
    assert 0.0 < p <= 1.0


def test_bootstrap_log_ratio_ci_returns_ordered_finite_interval():
    # Percentile CIs of the observed statistic can fail to contain the observed
    # value in small samples, so this test only checks the interval is finite
    # and correctly ordered.
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = rng.normal(size=180)
    obs, lo, hi, boot = bootstrap_log_ratio_ci(a, b, B=99, block_length=7, seed=42)
    assert np.isfinite(obs)
    assert lo < hi
    assert boot.shape == (99,)


def test_variance_test_is_reproducible_with_same_seed():
    # The whole point of SeedSequence.spawn(seed) is that the same seed gives
    # byte-identical results across runs; NB02's cross-refactor verification
    # depends on this.
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=92), rng.normal(size=143)
    r1 = bootstrap_variance_test(a, b, B=99, block_length=14, seed=7)
    r2 = bootstrap_variance_test(a, b, B=99, block_length=14, seed=7)
    assert np.array_equal(r1[1], r2[1])
    assert r1[2] == r2[2]


def test_variance_test_detects_large_difference():
    # Sanity check that the test has power against an obvious H1: one series
    # with twice the SD should reject at nominal alpha with plenty of margin.
    rng = np.random.default_rng(1)
    a = rng.normal(size=150)
    b = rng.normal(scale=2.0, size=150)
    _, _, p = bootstrap_variance_test(a, b, B=199, block_length=14, seed=1)
    assert p < 0.05


def test_fit_ssr_recovers_planted_break():
    # Planted step at index 100 in a 220-point series; K=1 should place its
    # break at exactly index 100 and give a smaller SSR than K=0.
    rng = np.random.default_rng(2)
    x = np.r_[np.zeros(100), np.ones(120)] + rng.normal(scale=0.1, size=220)
    ssr1, bkps = fit_ssr(x, 1)
    ssr0, _ = fit_ssr(x, 0)
    assert bkps == [100]
    assert ssr0 >= ssr1
