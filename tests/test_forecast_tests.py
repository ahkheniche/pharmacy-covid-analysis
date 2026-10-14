import numpy as np
import pytest
from scipy import stats

from pharmacy_covid_analysis.forecast_tests import dm_test_hln, dm_variance


def test_h1_reduces_to_one_sample_t_test():
    # With h = 1 there are no lags and the HLN factor is sqrt((n-1)/n),
    # so the statistic is the classical t-statistic on d.
    rng = np.random.default_rng(0)
    e_base = rng.normal(size=60)
    e_model = rng.normal(size=60)
    d = np.abs(e_base) - np.abs(e_model)
    t_ref = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    stat, p, dbar, se, fallback = dm_test_hln(e_model, e_base, h=1)
    assert stat == pytest.approx(t_ref)
    assert p == pytest.approx(stats.t.sf(t_ref, df=len(d) - 1))
    assert dbar == pytest.approx(d.mean())
    assert se == pytest.approx(d.std(ddof=1) / np.sqrt(len(d)))
    assert fallback is False


def test_clear_gain_gives_small_p_and_clear_loss_gives_large_p():
    rng = np.random.default_rng(1)
    e_base = rng.normal(size=100)
    _, p_better, dbar, _, _ = dm_test_hln(0.5 * e_base, e_base, h=7)
    _, p_worse, _, _, _ = dm_test_hln(2.0 * e_base, e_base, h=7)
    assert dbar > 0
    assert p_better < 1e-6
    assert p_worse > 0.999


def test_hln_correction_shrinks_statistic_more_at_longer_horizons():
    rng = np.random.default_rng(2)
    e_base = rng.normal(size=80)
    e_model = e_base + 0.3 * rng.normal(size=80)
    d = np.abs(e_base) - np.abs(e_model)
    n = len(d)
    for h in (1, 7):
        var, _ = dm_variance(d, h)
        raw = d.mean() / np.sqrt(var)
        stat, _, _, _, _ = dm_test_hln(e_model, e_base, h=h)
        factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        assert stat == pytest.approx(raw * factor)
    assert np.sqrt((n - 1) / n) > np.sqrt((n + 1 - 14 + 42 / n) / n)


def test_bartlett_fallback_triggers_when_rectangular_variance_fails():
    # Alternating signs give strongly negative lag-1 autocovariance, so the
    # rectangular sum with one lag is negative; the Bartlett sum is not.
    d = np.tile([1.0, -1.0], 20)
    var, fallback = dm_variance(d, h=2)
    assert fallback is True
    assert var > 0


def test_rectangular_variance_used_when_positive():
    rng = np.random.default_rng(3)
    d = rng.normal(size=50)
    var, fallback = dm_variance(d, h=3)
    assert fallback is False
    n = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: n - k] - dbar)) / n for k in range(3)]
    assert var == pytest.approx((gamma[0] + 2 * (gamma[1] + gamma[2])) / n)
