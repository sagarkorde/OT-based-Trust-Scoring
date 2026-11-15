"""OT-Shield guards: the Wasserstein Trust Score at three checkpoints (paper Section 4.3, Figure 1).

PromptGuard  - WTS-1: user prompt vs benign prompts of the application.
ChunkGuard   - WTS-2: each retrieved chunk vs a leave-one-out pool of the other top-k' chunks for the
               same query, mixed with the clean-corpus reference (weight lambda).
AgentGuard   - WTS-3a: each tool output vs clean outputs of the same tool;
               WTS-3b: intent drift D_t = W_2(plan_t, plan_{t-1}) vs benign drifts.
               Combined by Bonferroni over the two conformal p-values (flag if min p <= alpha / 2),
               which keeps the false-alarm guarantee at alpha by the union bound.

Data hygiene enforced here: reference data and calibration data are disjoint splits of the benign set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conformal import ConformalCalibrator, min_n_for_alpha
from .measures import (ReferenceModel, bures_wasserstein2, fit_gaussian, gaussian_w2_between_units,
                       psd_sqrt)


@dataclass
class Decision:
    flag: bool
    score: float
    pvalue: float
    threshold: float
    extras: dict = field(default_factory=dict)


def split_reference_calibration(items: list, ref_frac: float = 0.5, seed: int = 0):
    """Disjoint random split of benign items into (reference, calibration)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    k = int(round(ref_frac * len(items)))
    if k < 2 or len(items) - k < 1:
        raise ValueError(f"not enough benign items ({len(items)}) to split")
    return [items[i] for i in idx[:k]], [items[i] for i in idx[k:]]


# --------------------------------------------------------------------------------------------------
@dataclass
class PromptGuard:
    measure: str = "w2_gauss"
    alpha: float = 0.05
    ref_kwargs: dict = field(default_factory=dict)
    reference: ReferenceModel | None = None
    calibrator: ConformalCalibrator | None = None

    def fit(self, ref_units: list[np.ndarray], cal_units: list[np.ndarray]) -> "PromptGuard":
        self.reference = ReferenceModel(**self.ref_kwargs).fit(ref_units)
        cal = [self.reference.score(X, self.measure) for X in cal_units]
        self.calibrator = ConformalCalibrator(self.alpha).fit(cal)
        return self

    def score(self, X: np.ndarray) -> float:
        return self.reference.score(X, self.measure)

    def decide(self, X: np.ndarray) -> Decision:
        s = self.score(X)
        return Decision(bool(self.calibrator.flag(s)), s, float(self.calibrator.pvalue(s)), self.calibrator.tau)


# --------------------------------------------------------------------------------------------------
@dataclass
class ChunkGuard:
    """Per-chunk score among retrieval neighbours.

    For chunk i in the top-k, the reference Gaussian is the moment-matched mixture
        lambda * N(m_corpus, S_corpus) + (1 - lambda) * N(m_loo, S_loo)
    where (m_loo, S_loo) are fitted on the tokens of the other k'-1 retrieved chunks.
    lambda = 1 reduces to a corpus-only reference; lambda = 0 to a purely local one.
    """

    lam: float = 0.5
    alpha: float = 0.05
    shrinkage: str | float = "ledoit_wolf"
    corpus_reference: ReferenceModel | None = None
    calibrator: ConformalCalibrator | None = None

    def fit_corpus(self, corpus_units: list[np.ndarray], **ref_kwargs) -> "ChunkGuard":
        self.corpus_reference = ReferenceModel(shrinkage=self.shrinkage, **ref_kwargs).fit(corpus_units)
        return self

    def _mixture(self, m_loo, S_loo):
        r = self.corpus_reference
        lam = self.lam
        m = lam * r.mean + (1 - lam) * m_loo
        dc, dl = r.mean - m, m_loo - m
        S = lam * (r.cov + np.outer(dc, dc)) + (1 - lam) * (S_loo + np.outer(dl, dl))
        return m, S

    def score_pool(self, pool_units: list[np.ndarray], targets: list[int] | None = None) -> list[float]:
        """Scores for the chunks at positions `targets` (default: all) of a retrieved pool."""
        targets = range(len(pool_units)) if targets is None else targets
        scores = []
        for i in targets:
            X = pool_units[i]
            others = [u for j, u in enumerate(pool_units) if j != i]
            if self.lam < 1.0 and others:
                m_loo, S_loo = fit_gaussian(np.concatenate(others, axis=0), self.shrinkage)
                m_ref, S_ref = self._mixture(m_loo, S_loo)
            else:
                m_ref, S_ref = self.corpus_reference.mean, self.corpus_reference.cov
            m, S = fit_gaussian(X, self.shrinkage)
            w2sq, _, _ = bures_wasserstein2(m, S, m_ref, S_ref, psd_sqrt(S_ref))
            scores.append(float(np.sqrt(w2sq)))
        return scores

    def calibrate(self, benign_pools: list[list[np.ndarray]], k: int) -> "ChunkGuard":
        """Calibration scores: the top-k chunks of benign (non-target) queries, clean corpus only."""
        cal = []
        for pool in benign_pools:
            cal.extend(self.score_pool(pool, list(range(min(k, len(pool))))))
        self.calibrator = ConformalCalibrator(self.alpha).fit(cal)
        return self

    def select(self, pool_units: list[np.ndarray], k: int) -> tuple[list[int], dict[int, float]]:
        """Walk the ranked pool, drop chunks scoring above tau, refill from lower ranks until k are kept.
        Returns kept indices (rank order) and the scores of every chunk examined."""
        keep, seen = [], {}
        for i in range(len(pool_units)):
            s = self.score_pool(pool_units, [i])[0]
            seen[i] = s
            if s <= self.calibrator.tau:
                keep.append(i)
            if len(keep) == k:
                break
        return keep, seen

    def filter(self, pool_units: list[np.ndarray], k: int) -> tuple[list[int], list[float]]:
        """Indices among the top-k that survive, and the scores of all top-k."""
        s = self.score_pool(pool_units, list(range(min(k, len(pool_units)))))
        keep = [i for i, v in enumerate(s) if v <= self.calibrator.tau]
        return keep, s


# --------------------------------------------------------------------------------------------------
@dataclass
class AgentGuard:
    alpha: float = 0.05
    measure: str = "w2_gauss"
    min_tool_ref_units: int = 10
    use_output: bool = True
    use_drift: bool = True
    ref_kwargs: dict = field(default_factory=dict)
    tool_refs: dict = field(default_factory=dict)          # tool name -> ReferenceModel
    tool_cals: dict = field(default_factory=dict)          # tool name -> ConformalCalibrator
    drift_cal: ConformalCalibrator | None = None

    def fit_outputs(self, outputs_by_tool: dict[str, list[np.ndarray]], seed: int = 0) -> "AgentGuard":
        """Per-tool reference + calibration from benign tool outputs. Tools with too few outputs share a
        pooled '__all__' model so that rare tools are still guarded."""
        # A conformal p-value is at least 1/(n_cal+1): a per-tool model with fewer calibration points than
        # min_n_for_alpha(alpha / n_tests) can never flag anything, so such tools use the pooled model.
        n_tests = int(self.use_output) + int(self.use_drift)
        need_cal = max(self.min_tool_ref_units, min_n_for_alpha(self.alpha / max(n_tests, 1)))
        pooled = []
        for tool, units in outputs_by_tool.items():
            pooled.extend(units)
            if len(units) - int(round(0.5 * len(units))) >= need_cal and len(units) >= 2 * self.min_tool_ref_units:
                ref, cal = split_reference_calibration(units, 0.5, seed)
                self._fit_one(tool, ref, cal)
        ref, cal = split_reference_calibration(pooled, 0.5, seed)
        self._fit_one("__all__", ref, cal)
        return self

    def _fit_one(self, tool, ref_units, cal_units):
        r = ReferenceModel(**self.ref_kwargs).fit(ref_units)
        self.tool_refs[tool] = r
        self.tool_cals[tool] = ConformalCalibrator(self.alpha).fit([r.score(X, self.measure) for X in cal_units])

    def fit_drift(self, benign_drifts: list[float]) -> "AgentGuard":
        self.drift_cal = ConformalCalibrator(self.alpha).fit(benign_drifts)
        return self

    @staticmethod
    def drift(plan_prev: np.ndarray, plan_now: np.ndarray) -> float:
        return gaussian_w2_between_units(plan_now, plan_prev)[0]

    def output_pvalue(self, tool: str, X: np.ndarray) -> tuple[float, float]:
        key = tool if tool in self.tool_refs else "__all__"
        s = self.tool_refs[key].score(X, self.measure)
        return s, float(self.tool_cals[key].pvalue(s))

    def decide(self, tool: str | None, output: np.ndarray | None, plan_prev: np.ndarray | None,
               plan_now: np.ndarray | None) -> Decision:
        ps, extras = [], {}
        if self.use_output and output is not None and tool is not None:
            s, p = self.output_pvalue(tool, output)
            ps.append(p)
            extras.update(output_score=s, output_p=p)
        if self.use_drift and plan_prev is not None and plan_now is not None and self.drift_cal is not None:
            dft = self.drift(plan_prev, plan_now)
            p = float(self.drift_cal.pvalue(dft))
            ps.append(p)
            extras.update(drift=dft, drift_p=p)
        if not ps:
            return Decision(False, 0.0, 1.0, self.alpha, extras)
        m = len(ps)
        pmin = min(ps)
        return Decision(pmin <= self.alpha / m, -float(np.log(pmin)), pmin, self.alpha / m, extras)
