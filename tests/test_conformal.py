import math

import numpy as np
import pytest

from wts.conformal import (ConformalCalibrator, conformal_pvalue, conformal_threshold, coverage_study,
                           empirical_fpr, min_n_for_alpha)


def test_threshold_index():
    s = np.arange(1, 20)  # n = 19
    # ceil(20 * 0.95) = 19 -> 19th smallest
    assert conformal_threshold(s, 0.05) == 19
    # n = 18 is too small for alpha = 0.05
    assert math.isinf(conformal_threshold(np.arange(18), 0.05))
    assert min_n_for_alpha(0.05) == 19 and min_n_for_alpha(0.01) == 99


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
@pytest.mark.parametrize("n", [100, 200, 1000])
def test_marginal_coverage_monte_carlo(alpha, n):
    """P(benign flagged) <= alpha over random calibration sets: averaged over many draws."""
    rng = np.random.default_rng(int(alpha * 1000) + n)
    trials = 4000
    flagged = 0
    for _ in range(trials):
        cal = rng.standard_gamma(2.0, n)
        tau = conformal_threshold(cal, alpha)
        flagged += rng.standard_gamma(2.0) > tau
    rate = flagged / trials
    se = math.sqrt(alpha * (1 - alpha) / trials)
    assert rate <= alpha + 3 * se
    # and it is not wildly conservative: exact coverage is 1 - ceil((n+1)(1-alpha))/(n+1)
    exact = 1 - math.ceil((n + 1) * (1 - alpha)) / (n + 1)
    assert rate == pytest.approx(exact, abs=4 * se)


def test_pvalue_superuniform():
    # Super-uniformity is marginal over the calibration draw, so average over many calibration sets
    # (conditional on one fixed set, the rate deviates by ~sqrt(a(1-a)/n), which is expected).
    rng = np.random.default_rng(0)
    trials = 4000
    ps = np.array([conformal_pvalue(rng.standard_normal(), rng.standard_normal(200)) for _ in range(trials)])
    for a in (0.01, 0.05, 0.1, 0.5):
        assert np.mean(ps <= a) <= a + 3 * np.sqrt(a * (1 - a) / trials)
    assert conformal_pvalue(1e9, np.zeros(500)) == pytest.approx(1 / 501)


def test_pvalue_and_threshold_agree():
    rng = np.random.default_rng(1)
    c = ConformalCalibrator(0.05).fit(rng.standard_normal(300))
    x = rng.standard_normal(5000) * 1.5
    assert np.array_equal(c.flag(x), c.pvalue(x) <= 0.05)


def test_coverage_study_rows():
    rng = np.random.default_rng(2)
    rows = coverage_study(rng.standard_normal(2000), rng.standard_normal(2000), n_cals=(50, 100), n_repeats=50)
    assert {r["n_cal"] for r in rows} == {50, 100}
    r = [r for r in rows if r["alpha"] == 0.01 and r["n_cal"] == 50][0]
    assert r["frac_infinite_tau"] == 1.0          # n = 50 < 99 cannot certify alpha = 0.01
    assert empirical_fpr([1, 2, 3], 2.5) == pytest.approx(1 / 3)
