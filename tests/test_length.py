"""Addendum A1: the length confound and its correction."""
import numpy as np

from wts.guards import ChunkGuard, PromptGuard
from wts.length import LengthAdjuster
from wts.measures import ReferenceModel

D = 32


def _gen(rng, A, n, shift=0.0):
    return rng.standard_normal((n, D)) @ A + 0.3 + shift


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, D)) / np.sqrt(D)
    return rng, A


def test_raw_scores_are_confounded_by_length():
    """Same content distribution, only the number of tokens differs: raw scores fall with length."""
    rng, A = _setup()
    ref = ReferenceModel(n_proj=16, length_adjust=False).fit([_gen(rng, A, rng.integers(5, 200)) for _ in range(200)])
    for m in ("w2_gauss", "w2_sliced", "mahalanobis"):
        short = np.mean([ref.score(_gen(rng, A, 8), m) for _ in range(30)])
        long = np.mean([ref.score(_gen(rng, A, 300), m) for _ in range(30)])
        assert short > 1.5 * long, m


def test_adjusted_scores_are_flat_in_length_and_keep_power():
    rng, A = _setup(1)
    ref = ReferenceModel(n_proj=16).fit([_gen(rng, A, rng.integers(5, 300)) for _ in range(400)])
    means = [np.mean([ref.score(_gen(rng, A, n), "w2_gauss_adj") for _ in range(40)]) for n in (8, 30, 100, 300)]
    assert max(means) / min(means) < 1.35, means
    # a real distribution shift is still separated at every length
    for n in (8, 100, 300):
        b = np.mean([ref.score(_gen(rng, A, n), "w2_gauss_adj") for _ in range(30)])
        s = np.mean([ref.score(_gen(rng, A, n, shift=0.6), "w2_gauss_adj") for _ in range(30)])
        assert s > 1.3 * b, n


def test_prompt_guard_fpr_holds_for_short_and_long_units():
    """With raw scores, short benign units would be flagged far above alpha; adjusted, both stay near alpha."""
    rng, A = _setup(2)
    mk = lambda k: [_gen(rng, A, int(rng.integers(5, 300))) for _ in range(k)]
    g = PromptGuard(alpha=0.10, ref_kwargs=dict(n_proj=16)).fit(mk(400), mk(600))
    short = np.mean([g.decide(_gen(rng, A, int(rng.integers(5, 15)))).flag for _ in range(300)])
    long = np.mean([g.decide(_gen(rng, A, int(rng.integers(200, 300)))).flag for _ in range(300)])
    assert short < 0.25 and long < 0.25, (short, long)


def test_chunk_guard_is_not_fooled_by_short_benign_chunks():
    rng, A = _setup(3)
    pools = lambda k: [[_gen(rng, A, int(rng.integers(10, 120))) for _ in range(20)] for _ in range(k)]
    cg = ChunkGuard(alpha=0.05).fit_corpus([_gen(rng, A, 80) for _ in range(200)], n_proj=8)
    cg.calibrate(pools(80), k=5)
    p = pools(1)[0]
    p[0] = _gen(rng, A, 10)                      # short but benign: should usually survive
    keep, _ = cg.filter(p, k=5)
    s_short = cg.score_pool(p, [0])[0]
    assert s_short <= np.quantile(cg.calibrator.cal_scores, 0.99)


def test_length_adjuster_signed_and_multifeature():
    rng = np.random.default_rng(4)
    n = rng.integers(5, 300, 500)
    s = 3.0 * n ** -0.4 * rng.lognormal(0, 0.05, 500)
    adj = LengthAdjuster().fit(n, s)
    assert abs(adj.coef[1] + 0.4) < 0.02
    L = np.column_stack([n, rng.integers(5, 300, 500)])
    y = 0.2 - 0.01 * np.log(L[:, 0]) + rng.normal(0, 1e-3, 500)
    a2 = LengthAdjuster(signed=True).fit(L, y)
    assert np.abs(a2.adjust(y, L)).mean() < 0.01
