"""Metrics and statistical tests used in every results table (plan Section 6.2)."""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


# ---------------------------------------------------------------------------- detection metrics
def tpr_at_fpr(y, s, fpr_target=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def fpr_at_tpr(y, s, tpr_target=0.95) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr >= tpr_target
    return float(fpr[ok].min()) if ok.any() else 1.0


def detection_metrics(y, s) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    return dict(auroc=float(roc_auc_score(y, s)), auprc=float(average_precision_score(y, s)),
                tpr_at_1fpr=tpr_at_fpr(y, s, 0.01), fpr_at_95tpr=fpr_at_tpr(y, s, 0.95),
                n_pos=int(y.sum()), n_neg=int((1 - y).sum()))


# ---------------------------------------------------------------------------- bootstrap
def bootstrap_ci(values, stat=np.mean, n_boot=10000, level=0.95, seed=0):
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = np.array([stat(v[i]) for i in idx])
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(stat(v)), float(lo), float(hi)


def bootstrap_metric_ci(y, s, metric=roc_auc_score, n_boot=10000, level=0.95, seed=0):
    """Stratified bootstrap CI of a ranking metric (resample positives and negatives separately)."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bs[b] = metric(y[i], s[i])
    lo, hi = np.quantile(bs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(metric(y, s)), float(lo), float(hi)


def paired_bootstrap_auroc_diff(y, s_a, s_b, n_boot=10000, seed=0):
    """AUROC(a) - AUROC(b) on the same items; returns (diff, lo, hi, two-sided bootstrap p)."""
    y = np.asarray(y)
    s_a, s_b = np.asarray(s_a, float), np.asarray(s_b, float)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d[b] = roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i])
    obs = roc_auc_score(y, s_a) - roc_auc_score(y, s_b)
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(1.0, max(p, 1.0 / n_boot)))


# ---------------------------------------------------------------------------- paired tests
def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact McNemar test on paired binary outcomes (e.g. per-item correct detection)."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(sps.binomtest(n01, n, 0.5).pvalue)
    return dict(n_a_only=n01, n_b_only=n10, p=p)


def wilcoxon_paired(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return dict(stat=0.0, p=1.0)
    r = sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return dict(stat=float(r.statistic), p=float(r.pvalue))


def cliffs_delta(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).sum()
    lt = (x[:, None] < y[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, same order as input."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------- attack outcomes
def relative_reduction(asr_none: float, asr_def: float) -> float:
    return float("nan") if asr_none == 0 else (asr_none - asr_def) / asr_none


def recovered_asr(asr_none: float, asr_def: float, asr_adapt: float) -> float:
    gap = asr_none - asr_def
    return float("nan") if gap == 0 else (asr_adapt - asr_def) / gap
