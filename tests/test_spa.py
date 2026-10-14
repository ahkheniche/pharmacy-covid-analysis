import numpy as np
from arch.bootstrap import SPA

from pharmacy_covid_analysis.spa import spa_test, stepm


def _losses(seed, n=107, h=7):
    """Benchmark loss with MA(h-1) dependence and five model losses on the
    same targets: one 30% better, three equal, one far worse."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n + h - 1)
    bench = np.abs(np.array([eps[t : t + h].sum() for t in range(n)]) / np.sqrt(h))

    def noisy(factor, cv=0.5):
        return bench * factor * np.exp(cv * rng.normal(size=n) - cv**2 / 2)

    models = np.column_stack(
        [noisy(0.7)] + [noisy(1.0) for _ in range(3)] + [noisy(3.0)]
    )
    return bench, models


def test_unstudentised_mode_reproduces_arch_exactly():
    for seed in (1, 2, 3):
        bench, models = _losses(seed)
        ref = SPA(
            bench, models, block_size=7, reps=500, bootstrap="stationary", seed=seed
        )
        ref.compute()
        _, pvalues, _ = spa_test(bench, models, 7, 500, seed, studentize=False)
        assert np.allclose(
            [pvalues["lower"], pvalues["consistent"], pvalues["upper"]],
            ref.pvalues[["lower", "consistent", "upper"]].to_numpy(),
        )


def test_studentised_rejects_clear_gain_and_stepm_names_it():
    bench, models = _losses(4)
    stat, pvalues, _ = spa_test(bench, models, 7, 1000, seed=4)
    assert stat.argmax() == 0
    assert pvalues["consistent"] < 0.01
    names = ["better", "same1", "same2", "same3", "worse"]
    assert stepm(bench, models, 7, 1000, seed=4, size=0.025, names=names) == ["better"]


def test_no_rejection_when_every_model_is_worse():
    bench, _ = _losses(5)
    models = np.column_stack([1.3 * bench, 1.6 * bench])
    _, pvalues, _ = spa_test(bench, models, 7, 1000, seed=5)
    assert pvalues["consistent"] > 0.2
    assert stepm(bench, models, 7, 1000, seed=5, size=0.025, names=["a", "b"]) == []


def test_pvalues_are_ordered_lower_consistent_upper():
    bench, models = _losses(6)
    _, pvalues, _ = spa_test(bench, models, 7, 1000, seed=6)
    assert pvalues["lower"] <= pvalues["consistent"] <= pvalues["upper"]
