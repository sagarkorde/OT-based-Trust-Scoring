"""Length adjustment of distance scores (hypotheses.md addendum A1, 2026-11-15).

Any distance between an n-token empirical measure and a population reference has a finite-sample bias that
shrinks as n grows (shown in tests/test_length.py for Gaussian W2, sliced W2, Mahalanobis and cosine). Text
units differ in length by an order of magnitude (a one-line prompt vs a 40-record tool output), so raw scores
partly measure length. We divide out the benign length trend:

    log s = a + b . log(n)  (+ more length features, e.g. both plan steps for drift)    fitted on benign DEV data
    s_adj = s / exp(a + b . log n)            for non-negative scores
    s_adj = s - (a + b . log n)               for signed scores (unbiased MMD^2), fitted on s directly

The DEV data are disjoint from the conformal calibration data, so the adjusted score is a fixed function when
calibration happens and Proposition 1 still holds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LengthAdjuster:
    signed: bool = False
    coef: np.ndarray | None = None

    @staticmethod
    def _design(lengths):
        L = np.atleast_2d(np.asarray(lengths, dtype=np.float64))
        if L.shape[0] == 1 and L.shape[1] > 1 and np.ndim(lengths) == 1:
            L = L.T
        return np.column_stack([np.ones(len(L)), np.log(np.maximum(L, 1.0))])

    def fit(self, lengths, scores) -> "LengthAdjuster":
        X = self._design(lengths)
        y = np.asarray(scores, dtype=np.float64)
        if not self.signed:
            y = np.log(np.maximum(y, 1e-12))
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def baseline(self, lengths) -> np.ndarray:
        z = self._design(lengths) @ self.coef
        return z if self.signed else np.exp(z)

    def adjust(self, score, lengths):
        """lengths: one row per score (n_scores x n_features) or a 1-D array of single lengths."""
        b = self.baseline(lengths)
        s = np.asarray(score, dtype=np.float64)
        out = s - b if self.signed else s / b
        return float(out[0]) if np.ndim(score) == 0 else out
