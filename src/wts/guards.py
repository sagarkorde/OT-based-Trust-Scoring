"""OT-Shield guards: the Wasserstein Trust Score at three checkpoints (paper Section 4.3, Figure 1).

PromptGuard  - WTS-1: user prompt vs benign prompts of the application.
ChunkGuard   - WTS-2: each retrieved chunk vs a leave-one-out pool of the other top-k' chunks for the
               same query, mixed with the clean-corpus reference (weight lambda).
AgentGuard   - WTS-3a: each tool output vs clean outputs of the same tool;
               WTS-3b: intent drift D_t = W_2(plan_t, plan_{t-1}) vs benign drifts.
               Combined by Bonferroni over the two conformal p-values (flag if min p <= alpha / 2),
               which keeps the false-alarm guarantee at alpha by the union bound.

All guards use length-adjusted scores (hypotheses.md addendum A1). Every length baseline is fitted on benign
data disjoint from the conformal calibration data, so Proposition 1 still holds.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .conformal import ConformalCalibrator, min_n_for_alpha
from .length import LengthAdjuster
from .measures import ReferenceModel, bures_wasserstein2, fit_gaussian, gaussian_w2_between_units, psd_sqrt


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


def _halves(n, seed):
    idx = np.random.default_rng(seed + 104729).permutation(n)
    return idx[: n // 2], idx[n // 2:]


# --------------------------------------------------------------------------------------------------
@dataclass
class PromptGuard:
    measure: str = "w2_gauss_adj"
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

    For chunk i the reference Gaussian is the moment-matched mixture
        lambda * N(m_corpus, S_corpus) + (1 - lambda) * N(m_loo, S_loo)
    where (m_loo, S_loo) are fitted on the tokens of the other k'-1 retrieved chunks. lambda = 1 reduces to a
    corpus-only reference; lambda = 0 to a purely local one. The raw distance is divided by a length baseline
    fitted on half of the benign calibration pools; the other half sets the conformal threshold.
    """

    lam: float = 0.5
    alpha: float = 0.05
    shrinkage: str | float = "ledoit_wolf"
    length_adjust: bool = True
    corpus_reference: ReferenceModel | None = None
    calibrator: ConformalCalibrator | None = None
    adjuster: LengthAdjuster | None = None

    def fit_corpus(self, corpus_units: list[np.ndarray], **ref_kwargs) -> "ChunkGuard":
        ref_kwargs.setdefault("length_adjust", False)  # the chunk guard keeps its own adjuster
        self.corpus_reference = ReferenceModel(shrinkage=self.shrinkage, **ref_kwargs).fit(corpus_units)
        return self

    def _mixture(self, m_loo, S_loo):
        r, lam = self.corpus_reference, self.lam
        m = lam * r.mean + (1 - lam) * m_loo
        dc, dl = r.mean - m, m_loo - m
        S = lam * (r.cov + np.outer(dc, dc)) + (1 - lam) * (S_loo + np.outer(dl, dl))
        return m, S

    def raw_scores(self, pool_units: list[np.ndarray], targets=None) -> list[float]:
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

    def score_pool(self, pool_units: list[np.ndarray], targets=None) -> list[float]:
        targets = list(range(len(pool_units)) if targets is None else targets)
        raw = self.raw_scores(pool_units, targets)
        if self.adjuster is None:
            return raw
        return list(self.adjuster.adjust(np.asarray(raw), [len(pool_units[i]) for i in targets]))

    def calibrate(self, benign_pools: list[list[np.ndarray]], k: int, seed: int = 0) -> "ChunkGuard":
        """Benign (non-target) queries, clean corpus only: half of the pools fit the length baseline, the
        other half give the conformal calibration scores of their top-k chunks."""
        self.adjuster = None
        fit_idx, cal_idx = _halves(len(benign_pools), seed) if self.length_adjust else ([], range(len(benign_pools)))
        if self.length_adjust:
            s, n = [], []
            for i in fit_idx:
                pool = benign_pools[i]
                t = list(range(min(k, len(pool))))
                s += self.raw_scores(pool, t)
                n += [len(pool[j]) for j in t]
            self.adjuster = LengthAdjuster().fit(n, s)
        cal = []
        for i in cal_idx:
            pool = benign_pools[i]
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
    measure: str = "w2_gauss_adj"
    min_tool_ref_units: int = 10
    use_output: bool = True
    use_drift: bool = True
    length_adjust: bool = True
    ref_kwargs: dict = field(default_factory=dict)
    tool_refs: dict = field(default_factory=dict)          # tool name -> ReferenceModel
    tool_cals: dict = field(default_factory=dict)          # tool name -> ConformalCalibrator
    drift_cal: ConformalCalibrator | None = None
    drift_adjuster: LengthAdjuster | None = None
    pooled_can_flag: bool = False

    @property
    def n_tests(self) -> int:
        return max(1, int(self.use_output) + int(self.use_drift))

    def fit_outputs(self, outputs_by_tool: dict[str, list[np.ndarray]], seed: int = 0) -> "AgentGuard":
        """Per-tool reference + calibration from benign tool outputs. A conformal p-value is at least
        1/(n_cal+1), so a tool with fewer calibration outputs than min_n_for_alpha(alpha / n_tests) could never
        flag anything; such tools use the pooled '__all__' model."""
        need_cal = max(self.min_tool_ref_units, min_n_for_alpha(self.alpha / self.n_tests))
        pooled = []
        for tool, units in outputs_by_tool.items():
            pooled.extend(units)
            if len(units) - int(round(0.5 * len(units))) >= need_cal and len(units) >= 2 * self.min_tool_ref_units:
                ref, cal = split_reference_calibration(units, 0.5, seed)
                self._fit_one(tool, ref, cal)
        ref, cal = split_reference_calibration(pooled, 0.5, seed)
        self._fit_one("__all__", ref, cal)
        self.pooled_can_flag = len(cal) >= min_n_for_alpha(self.alpha / self.n_tests)
        if not self.pooled_can_flag:
            warnings.warn(f"AgentGuard: only {len(cal)} pooled calibration outputs; at alpha/{self.n_tests} the "
                          f"threshold is infinite and the output guard can never flag. Pool more benign outputs.")
        return self

    def _fit_one(self, tool, ref_units, cal_units):
        kw = dict(self.ref_kwargs)
        kw.setdefault("length_adjust", self.length_adjust)
        measure = self.measure if kw["length_adjust"] else self.measure.removesuffix("_adj")
        r = ReferenceModel(**kw).fit(ref_units)
        if not r.adjusters:   # too few reference units for a dev split: fall back to the raw score
            measure = measure.removesuffix("_adj")
        self.tool_refs[tool] = (r, measure)
        self.tool_cals[tool] = ConformalCalibrator(self.alpha).fit([r.score(X, measure) for X in cal_units])

    # ---------------------------------------------------------------- drift
    @staticmethod
    def drift(plan_prev: np.ndarray, plan_now: np.ndarray) -> float:
        return gaussian_w2_between_units(plan_now, plan_prev)[0]

    def drift_score(self, plan_prev: np.ndarray, plan_now: np.ndarray) -> float:
        d = self.drift(plan_prev, plan_now)
        if self.drift_adjuster is None:
            return d
        return self.drift_adjuster.adjust(d, [[len(plan_prev), len(plan_now)]])

    def fit_drift(self, benign_pairs: list[tuple[np.ndarray, np.ndarray]], seed: int = 0) -> "AgentGuard":
        """benign_pairs: (previous plan step, next plan step) token matrices from benign runs."""
        self.drift_adjuster = None
        if self.length_adjust and len(benign_pairs) >= 8:
            fit_idx, cal_idx = _halves(len(benign_pairs), seed)
            self.drift_adjuster = LengthAdjuster().fit(
                [[len(benign_pairs[i][0]), len(benign_pairs[i][1])] for i in fit_idx],
                [self.drift(*benign_pairs[i]) for i in fit_idx])
        else:
            cal_idx = range(len(benign_pairs))
        cal = [self.drift_score(*benign_pairs[i]) for i in cal_idx]
        if len(cal) < min_n_for_alpha(self.alpha / self.n_tests):
            warnings.warn(f"AgentGuard: only {len(cal)} drift calibration pairs; the drift test can never flag.")
        self.drift_cal = ConformalCalibrator(self.alpha).fit(cal)
        return self

    def drift_pvalue(self, plan_prev, plan_now) -> tuple[float, float]:
        d = self.drift_score(plan_prev, plan_now)
        return d, float(self.drift_cal.pvalue(d))

    # ---------------------------------------------------------------- outputs
    def output_pvalue(self, tool: str, X: np.ndarray) -> tuple[float, float]:
        key = tool if tool in self.tool_refs else "__all__"
        r, measure = self.tool_refs[key]
        s = r.score(X, measure)
        return s, float(self.tool_cals[key].pvalue(s))

    def decide(self, tool: str | None, output: np.ndarray | None, plan_prev: np.ndarray | None,
               plan_now: np.ndarray | None) -> Decision:
        ps, extras = [], {}
        if self.use_output and output is not None and tool is not None:
            s, p = self.output_pvalue(tool, output)
            ps.append(p)
            extras.update(output_score=s, output_p=p)
        if self.use_drift and plan_prev is not None and plan_now is not None and self.drift_cal is not None:
            d, p = self.drift_pvalue(plan_prev, plan_now)
            ps.append(p)
            extras.update(drift=d, drift_p=p)
        if not ps:
            return Decision(False, 0.0, 1.0, self.alpha, extras)
        pmin = min(ps)
        return Decision(pmin <= self.alpha / self.n_tests, -float(np.log(pmin)), pmin, self.alpha / self.n_tests, extras)
