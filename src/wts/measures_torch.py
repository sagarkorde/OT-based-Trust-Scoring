"""Differentiable Bures-Wasserstein W_2 in PyTorch, for the white-box adaptive attacker (E4, profile A3):
the attacker adds lambda * W_2(unit, reference) to the GCG loss so the suffix also lowers the trust score.

Mirrors measures.fit_gaussian / bures_wasserstein2 (fixed shrinkage instead of Ledoit-Wolf, whose
shrinkage intensity is data-dependent and non-smooth; the test checks agreement at matched shrinkage).
"""
from __future__ import annotations

import torch


def fit_gaussian_t(X: torch.Tensor, shrinkage: float = 0.1, min_var: float = 1e-6):
    X = X.to(torch.float64)
    n, d = X.shape
    m = X.mean(0)
    if n == 1:
        return m, torch.eye(d, dtype=X.dtype, device=X.device) * min_var
    Xc = X - m
    S = Xc.T @ Xc / n
    mu = torch.trace(S) / d
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    S = (1 - shrinkage) * S + shrinkage * mu * eye
    return m, 0.5 * (S + S.T) + eye * min_var


def ledoit_wolf_t(X: torch.Tensor, min_var: float = 1e-6):
    """Ledoit-Wolf mean/covariance on any device; same formula as sklearn.covariance.ledoit_wolf
    (assume_centered=False) plus measures.fit_gaussian's min_var jitter. Used for the GPU score (E5)."""
    X = X.to(torch.float64)
    n, d = X.shape
    m = X.mean(0)
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    if n == 1:
        return m, eye * min_var
    Xc = X - m
    emp = Xc.T @ Xc / n
    X2 = Xc ** 2
    tr_diag = X2.sum(0) / n
    mu = tr_diag.sum() / d
    beta_ = (X2.T @ X2).sum()
    delta_ = (emp ** 2).sum()
    beta = (beta_ / n - delta_) / (d * n)
    delta = (delta_ - 2.0 * mu * tr_diag.sum() + d * mu ** 2) / d
    beta = torch.minimum(beta, delta)
    shrink = torch.where(beta == 0, torch.zeros_like(beta), beta / delta)
    S = (1 - shrink) * emp + shrink * mu * eye
    return m, 0.5 * (S + S.T) + eye * min_var


def bures_w2_t(m1, S1, m2, S2_sqrt, trace_S2, eps: float = 1e-10):
    """sqrt(W_2^2) with the reference's S2^{1/2} and tr(S2) precomputed (constants for the attacker)."""
    loc = torch.sum((m1 - m2) ** 2)
    M = S2_sqrt @ S1 @ S2_sqrt
    ev = torch.linalg.eigvalsh(0.5 * (M + M.T)).clamp_min(0.0)
    cross = torch.sqrt(ev + eps).sum()
    shape = (torch.trace(S1) + trace_S2 - 2 * cross).clamp_min(0.0)
    return torch.sqrt(loc + shape + eps)
