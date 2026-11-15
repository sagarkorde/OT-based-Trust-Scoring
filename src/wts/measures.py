"""Distances between a text unit (a cloud of token vectors) and a benign reference.

Notation follows Section 4 of the paper:
    unit x         -> token matrix X in R^{n x d}, empirical measure mu_x
    reference      -> pooled benign tokens, empirical measure mu_ref
    WTS(x)         =  W_2(mu_x, mu_ref)

Main score: Gaussian (Bures-Wasserstein) W_2 with Ledoit-Wolf covariances, decomposed into a
location term ||m_x - m_r||^2 and a shape term tr(S_x + S_r - 2 (S_r^1/2 S_x S_r^1/2)^1/2).
Non-parametric check: sliced W_2. Baselines for the ablation (plan Section 2.7): cosine and Euclidean
distance to the reference centroid, Mahalanobis distance of the mean-pooled unit, and RBF-kernel MMD.

Everything here is numpy-only and deterministic given a seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import ledoit_wolf

MEASURES = (
    "w2_gauss",       # ours: sqrt(location + shape)
    "w2_location",    # ablation: location term only  (sqrt)
    "w2_shape",       # ablation: shape term only     (sqrt)
    "w2_sliced",      # ours: sliced W_2, no Gaussian assumption
    "cosine",         # baseline: 1 - cos(mean(x), centroid)
    "euclidean",      # baseline: ||mean(x) - centroid||
    "mahalanobis",    # baseline: whitened distance of mean(x) to the distribution of reference unit means
    "mmd_rbf",        # baseline: unbiased MMD^2 between token clouds, RBF kernel, median heuristic
)
LOCATION_ONLY = ("cosine", "euclidean", "mahalanobis")

_EPS = 1e-10


# --------------------------------------------------------------------------------------------------
# Gaussian helpers
# --------------------------------------------------------------------------------------------------
def fit_gaussian(X: np.ndarray, shrinkage: str | float = "ledoit_wolf", min_var: float = 1e-6):
    """Mean and covariance of a token matrix X (n x d).

    shrinkage: "ledoit_wolf" (default, as pre-registered), "none", or a float in [0, 1] for fixed
    shrinkage toward (tr(S)/d) I. Units with a single token get an isotropic covariance min_var * I.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D token matrix, got shape {X.shape}")
    n, d = X.shape
    if n == 0:
        raise ValueError("empty unit (0 tokens)")
    m = X.mean(axis=0)
    if n == 1:
        return m, np.eye(d) * min_var
    if shrinkage == "ledoit_wolf":
        S, _ = ledoit_wolf(X, assume_centered=False)
    else:
        Xc = X - m
        S = Xc.T @ Xc / n
        if shrinkage != "none":
            a = float(shrinkage)
            mu = np.trace(S) / d
            S = (1.0 - a) * S + a * mu * np.eye(d)
    S = 0.5 * (S + S.T) + np.eye(d) * min_var
    return m, S


def psd_sqrt(S: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition (negative eigenvalues clipped to 0)."""
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def bures_wasserstein2(m1, S1, m2, S2, S2_sqrt=None):
    """Squared W_2 between N(m1, S1) and N(m2, S2), returned with its two terms.

    Returns (w2_squared, location_term, shape_term). If S2_sqrt (= S2^{1/2}) is given it is reused,
    which is the common case: the reference is fixed and only the unit changes.
    """
    if S2_sqrt is None:
        S2_sqrt = psd_sqrt(S2)
    loc = float(np.sum((m1 - m2) ** 2))
    M = S2_sqrt @ S1 @ S2_sqrt
    ev = np.clip(np.linalg.eigvalsh(0.5 * (M + M.T)), 0.0, None)
    cross = float(np.sum(np.sqrt(ev)))
    shape = float(np.trace(S1) + np.trace(S2) - 2.0 * cross)
    shape = max(shape, 0.0)  # tiny negatives come from round-off only
    return loc + shape, loc, shape


# --------------------------------------------------------------------------------------------------
# Sliced W_2
# --------------------------------------------------------------------------------------------------
def random_directions(d: int, n_proj: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((d, n_proj))
    theta /= np.linalg.norm(theta, axis=0, keepdims=True) + _EPS
    return theta


def _quantiles(P: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Quantile function of each column of P (n x L) evaluated on grid -> (len(grid) x L)."""
    return np.quantile(P, grid, axis=0, method="linear")


def sliced_w2(X: np.ndarray, Y: np.ndarray, n_proj: int = 128, seed: int = 0, n_grid: int = 256) -> float:
    """Monte-Carlo sliced W_2 between two empirical measures (rows of X and Y, uniform weights)."""
    theta = random_directions(X.shape[1], n_proj, seed)
    grid = (np.arange(n_grid) + 0.5) / n_grid
    qx = _quantiles(X @ theta, grid)
    qy = _quantiles(Y @ theta, grid)
    return float(np.sqrt(np.mean((qx - qy) ** 2)))


# --------------------------------------------------------------------------------------------------
# MMD
# --------------------------------------------------------------------------------------------------
def _sqdist(A, B):
    a = np.sum(A * A, axis=1)[:, None]
    b = np.sum(B * B, axis=1)[None, :]
    return np.clip(a + b - 2.0 * A @ B.T, 0.0, None)


def mmd2_rbf_unbiased(X, Y, gamma: float) -> float:
    """Unbiased MMD^2 with k(a, b) = exp(-gamma ||a - b||^2). Needs >= 2 rows in each sample."""
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        # fall back to the biased estimator for 1-token units
        kxx = np.exp(-gamma * _sqdist(X, X)).mean()
        kyy = np.exp(-gamma * _sqdist(Y, Y)).mean()
        kxy = np.exp(-gamma * _sqdist(X, Y)).mean()
        return float(kxx + kyy - 2 * kxy)
    Kxx = np.exp(-gamma * _sqdist(X, X))
    Kyy = np.exp(-gamma * _sqdist(Y, Y))
    Kxy = np.exp(-gamma * _sqdist(X, Y))
    sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    return float(sxx + syy - 2.0 * Kxy.mean())


# --------------------------------------------------------------------------------------------------
# Reference model
# --------------------------------------------------------------------------------------------------
@dataclass
class ReferenceModel:
    """Benign reference for one channel (prompt, corpus, tool, plan step).

    Fit on a list of benign units (each an n_i x d token matrix). The data used here must be disjoint
    from the conformal calibration data (see conformal.py), otherwise calibration scores are biased low.
    """

    shrinkage: str | float = "ledoit_wolf"
    n_proj: int = 128
    n_grid: int = 256
    max_ref_tokens: int = 20000
    max_mmd_tokens: int = 1024
    seed: int = 0

    # fitted state
    mean: np.ndarray | None = field(default=None, repr=False)
    cov: np.ndarray | None = field(default=None, repr=False)
    cov_sqrt: np.ndarray | None = field(default=None, repr=False)
    centroid: np.ndarray | None = field(default=None, repr=False)
    unit_mean_center: np.ndarray | None = field(default=None, repr=False)
    unit_mean_precision: np.ndarray | None = field(default=None, repr=False)
    theta: np.ndarray | None = field(default=None, repr=False)
    ref_quantiles: np.ndarray | None = field(default=None, repr=False)
    mmd_tokens: np.ndarray | None = field(default=None, repr=False)
    mmd_gamma: float | None = None
    n_units: int = 0
    n_tokens: int = 0
    # length adjustment (addendum A1): fitted on a dev split of the reference units
    length_adjust: bool = True
    dev_frac: float = 0.25
    adjusters: dict = field(default_factory=dict, repr=False)

    def fit(self, units: list[np.ndarray]) -> "ReferenceModel":
        units = [np.asarray(u, dtype=np.float64) for u in units if len(u) > 0]
        if len(units) < 2:
            raise ValueError("need at least 2 reference units")
        if not self.length_adjust or len(units) < 8:
            self.adjusters = {}
            return self._fit_stats(units)
        rng = np.random.default_rng(self.seed + 7919)
        idx = rng.permutation(len(units))
        k = max(4, int(round(self.dev_frac * len(units))))
        dev, fit_units = [units[i] for i in idx[:k]], [units[i] for i in idx[k:]]
        self._fit_stats(fit_units)
        raw = [self.score_all(X, MEASURES) for X in dev]
        n = [len(X) for X in dev]
        from .length import LengthAdjuster
        self.adjusters = {m: LengthAdjuster(signed=(m == "mmd_rbf")).fit(n, [r[m] for r in raw]) for m in MEASURES}
        return self

    def _fit_stats(self, units: list[np.ndarray]) -> "ReferenceModel":
        rng = np.random.default_rng(self.seed)
        pool = np.concatenate(units, axis=0)
        if len(pool) > self.max_ref_tokens:
            pool = pool[rng.choice(len(pool), self.max_ref_tokens, replace=False)]
        self.n_units, self.n_tokens = len(units), len(pool)

        # Gaussian reference (token-level)
        self.mean, self.cov = fit_gaussian(pool, self.shrinkage)
        self.cov_sqrt = psd_sqrt(self.cov)

        # unit-level statistics for the location-only baselines
        means = np.stack([u.mean(axis=0) for u in units])
        self.centroid = means.mean(axis=0)
        self.unit_mean_center = self.centroid
        P, _ = ledoit_wolf(means, assume_centered=False)
        self.unit_mean_precision = np.linalg.pinv(P)

        # sliced W_2: fixed directions and reference quantile functions
        d = pool.shape[1]
        self.theta = random_directions(d, self.n_proj, self.seed)
        grid = (np.arange(self.n_grid) + 0.5) / self.n_grid
        self.ref_quantiles = _quantiles(pool @ self.theta, grid)

        # MMD: token subsample and median-heuristic bandwidth
        k = min(self.max_mmd_tokens, len(pool))
        self.mmd_tokens = pool[rng.choice(len(pool), k, replace=False)]
        sub = self.mmd_tokens[: min(500, k)]
        med = np.median(_sqdist(sub, sub)[np.triu_indices(len(sub), 1)])
        self.mmd_gamma = 1.0 / max(med, _EPS)
        return self

    # ---------------------------------------------------------------------------------------------
    def _check(self):
        if self.mean is None:
            raise RuntimeError("ReferenceModel is not fitted")

    def gaussian_terms(self, X: np.ndarray) -> tuple[float, float, float]:
        """(W_2, location term, shape term) of a unit against the reference. Terms are squared."""
        self._check()
        m, S = fit_gaussian(X, self.shrinkage)
        w2sq, loc, shape = bures_wasserstein2(m, S, self.mean, self.cov, self.cov_sqrt)
        return float(np.sqrt(w2sq)), loc, shape

    def sliced(self, X: np.ndarray) -> float:
        self._check()
        grid = (np.arange(self.n_grid) + 0.5) / self.n_grid
        qx = _quantiles(np.asarray(X, dtype=np.float64) @ self.theta, grid)
        return float(np.sqrt(np.mean((qx - self.ref_quantiles) ** 2)))

    def score(self, X: np.ndarray, measure: str = "w2_gauss") -> float:
        return self.score_all(X, (measure,))[measure]

    def score_all(self, X: np.ndarray, measures=MEASURES) -> dict[str, float]:
        """All requested measures for one unit. Larger = less like the benign reference.
        '<measure>_adj' gives the length-adjusted score (requires a fitted adjuster)."""
        self._check()
        X = np.asarray(X, dtype=np.float64)
        adj = [m for m in measures if m.endswith("_adj")]
        if adj:
            base = tuple(dict.fromkeys([m[:-4] if m.endswith("_adj") else m for m in measures]))
            out = self.score_all(X, base)
            if not self.adjusters:
                raise RuntimeError("length-adjusted measure requested but the reference has no adjusters")
            for m in adj:
                out[m] = self.adjusters[m[:-4]].adjust(out[m[:-4]], [len(X)])
            return {k: out[k] for k in list(measures) + [k for k in ("loc_sq", "shape_sq") if k in out]}
        out: dict[str, float] = {}
        if any(k.startswith("w2_") and k != "w2_sliced" for k in measures):
            w2, loc, shape = self.gaussian_terms(X)
            out.update(w2_gauss=w2, w2_location=float(np.sqrt(loc)), w2_shape=float(np.sqrt(shape)),
                       loc_sq=loc, shape_sq=shape)
        if "w2_sliced" in measures:
            out["w2_sliced"] = self.sliced(X)
        mu = X.mean(axis=0)
        if "cosine" in measures:
            c = self.centroid
            out["cosine"] = float(1.0 - mu @ c / (np.linalg.norm(mu) * np.linalg.norm(c) + _EPS))
        if "euclidean" in measures:
            out["euclidean"] = float(np.linalg.norm(mu - self.centroid))
        if "mahalanobis" in measures:
            diff = mu - self.unit_mean_center
            out["mahalanobis"] = float(np.sqrt(max(diff @ self.unit_mean_precision @ diff, 0.0)))
        if "mmd_rbf" in measures:
            out["mmd_rbf"] = mmd2_rbf_unbiased(X, self.mmd_tokens, self.mmd_gamma)
        return {k: out[k] for k in list(measures) + [k for k in ("loc_sq", "shape_sq") if k in out]}


def gaussian_w2_between_units(X: np.ndarray, Y: np.ndarray, shrinkage="ledoit_wolf"):
    """W_2 between two units' Gaussians, e.g. consecutive agent plan steps (intent drift)."""
    m1, S1 = fit_gaussian(X, shrinkage)
    m2, S2 = fit_gaussian(Y, shrinkage)
    w2sq, loc, shape = bures_wasserstein2(m1, S1, m2, S2)
    return float(np.sqrt(w2sq)), loc, shape
