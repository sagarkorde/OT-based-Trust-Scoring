"""Split conformal calibration of the Wasserstein Trust Score (paper Section 4.4, Proposition 1).

Given benign calibration scores s_1..s_n (exchangeable with future benign inputs) and a target
false-alarm rate alpha, the threshold tau = s_(ceil((n+1)(1-alpha))) satisfies
P(s(X_{n+1}) > tau) <= alpha. If ceil((n+1)(1-alpha)) > n the threshold is +inf (never flag): with
too few calibration points the guarantee cannot be met at that alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def conformal_threshold(cal_scores, alpha: float) -> float:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def conformal_pvalue(score: float | np.ndarray, cal_scores) -> np.ndarray | float:
    """p(x) = (1 + #{i : s_i >= s(x)}) / (n + 1). Super-uniform for benign x; small for anomalies."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    x = np.asarray(score, dtype=np.float64)
    ge = n - np.searchsorted(s, x, side="left")
    p = (1.0 + ge) / (n + 1.0)
    return float(p) if np.ndim(p) == 0 else p


def min_n_for_alpha(alpha: float) -> int:
    """Smallest calibration size for which the threshold is finite: ceil((n+1)(1-alpha)) <= n."""
    return math.ceil(1.0 / alpha) - 1


def empirical_fpr(benign_scores, tau: float) -> float:
    b = np.asarray(benign_scores, dtype=np.float64)
    return float(np.mean(b > tau)) if len(b) else float("nan")


@dataclass
class ConformalCalibrator:
    alpha: float = 0.05
    cal_scores: np.ndarray | None = None
    tau: float | None = None

    def fit(self, cal_scores) -> "ConformalCalibrator":
        self.cal_scores = np.sort(np.asarray(cal_scores, dtype=np.float64))
        self.tau = conformal_threshold(self.cal_scores, self.alpha)
        return self

    def pvalue(self, score):
        return conformal_pvalue(score, self.cal_scores)

    def flag(self, score) -> bool | np.ndarray:
        return np.asarray(score) > self.tau


def coverage_study(benign_cal_pool, benign_test, alphas=(0.01, 0.05, 0.10),
                   n_cals=(50, 100, 200, 500, 1000), n_repeats: int = 200, seed: int = 0):
    """RQ4: empirical FPR of the conformal threshold versus alpha and calibration size.

    For each (alpha, n_cal) draws n_cal scores from benign_cal_pool without replacement n_repeats
    times, sets tau, and measures FPR on benign_test. Returns rows (dicts) ready for a CSV.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(benign_cal_pool, dtype=np.float64)
    test = np.asarray(benign_test, dtype=np.float64)
    rows = []
    for a in alphas:
        for n in n_cals:
            if n > len(pool):
                continue
            fprs, infinite = [], 0
            for _ in range(n_repeats):
                cal = pool[rng.choice(len(pool), n, replace=False)]
                tau = conformal_threshold(cal, a)
                infinite += int(np.isinf(tau))
                fprs.append(empirical_fpr(test, tau))
            fprs = np.asarray(fprs)
            rows.append(dict(alpha=a, n_cal=n, fpr_mean=float(fprs.mean()),
                             fpr_q05=float(np.quantile(fprs, 0.05)), fpr_q95=float(np.quantile(fprs, 0.95)),
                             frac_infinite_tau=infinite / n_repeats, n_test=len(test), n_repeats=n_repeats))
    return rows
