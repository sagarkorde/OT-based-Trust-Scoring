import numpy as np
import ot
import pytest
from scipy.linalg import sqrtm

from wts.measures import (MEASURES, ReferenceModel, bures_wasserstein2, fit_gaussian, mmd2_rbf_unbiased,
                          psd_sqrt, sliced_w2)


def rand_spd(d, rng, scale=1.0):
    A = rng.standard_normal((d, d))
    return scale * (A @ A.T / d + 0.1 * np.eye(d))


def test_identical_gaussians_zero():
    rng = np.random.default_rng(0)
    m, S = rng.standard_normal(8), rand_spd(8, rng)
    w2sq, loc, shape = bures_wasserstein2(m, S, m, S)
    assert w2sq == pytest.approx(0.0, abs=1e-8)
    assert loc == pytest.approx(0.0) and shape == pytest.approx(0.0, abs=1e-8)


def test_equal_covariances_reduce_to_mean_distance():
    rng = np.random.default_rng(1)
    S = rand_spd(10, rng)
    m1, m2 = rng.standard_normal(10), rng.standard_normal(10)
    w2sq, loc, shape = bures_wasserstein2(m1, S, m2, S)
    assert shape == pytest.approx(0.0, abs=1e-8)
    assert w2sq == pytest.approx(np.sum((m1 - m2) ** 2))


def test_matches_scipy_sqrtm_formula():
    rng = np.random.default_rng(2)
    d = 12
    m1, m2 = rng.standard_normal(d), rng.standard_normal(d)
    S1, S2 = rand_spd(d, rng), rand_spd(d, rng, 3.0)
    r = sqrtm(S2)
    ref = np.sum((m1 - m2) ** 2) + np.trace(S1 + S2 - 2 * np.real(sqrtm(r @ S1 @ r)))
    assert bures_wasserstein2(m1, S1, m2, S2)[0] == pytest.approx(ref, rel=1e-6)


def test_matches_pot_gaussian_w2():
    rng = np.random.default_rng(3)
    d = 6
    m1, m2 = rng.standard_normal(d), rng.standard_normal(d)
    S1, S2 = rand_spd(d, rng), rand_spd(d, rng)
    pot = ot.gaussian.bures_wasserstein_distance(m1, m2, S1, S2)
    assert np.sqrt(bures_wasserstein2(m1, S1, m2, S2)[0]) == pytest.approx(float(pot), rel=1e-6)


def test_symmetry_and_triangle_inequality():
    rng = np.random.default_rng(4)
    d = 5
    G = [(rng.standard_normal(d), rand_spd(d, rng)) for _ in range(3)]
    w = lambda a, b: np.sqrt(bures_wasserstein2(*a, *b)[0])
    assert w(G[0], G[1]) == pytest.approx(w(G[1], G[0]), rel=1e-6)
    assert w(G[0], G[2]) <= w(G[0], G[1]) + w(G[1], G[2]) + 1e-9


def test_one_dimensional_closed_form():
    # W_2^2 between N(a, s^2) and N(b, t^2) = (a-b)^2 + (s-t)^2
    w2sq, _, _ = bures_wasserstein2(np.array([1.0]), np.array([[4.0]]), np.array([-2.0]), np.array([[0.25]]))
    assert w2sq == pytest.approx(9.0 + (2.0 - 0.5) ** 2)


def test_psd_sqrt():
    rng = np.random.default_rng(5)
    S = rand_spd(9, rng)
    R = psd_sqrt(S)
    assert np.allclose(R @ R, S, atol=1e-8)


def test_sliced_close_to_exact_in_1d_and_zero_for_same_sample():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((300, 1))
    assert sliced_w2(X, X, n_proj=4) == pytest.approx(0.0, abs=1e-12)
    Y = X + 2.0
    # in 1-D every direction is +/-1, so sliced W_2 equals the shift
    assert sliced_w2(X, Y, n_proj=8) == pytest.approx(2.0, rel=1e-6)


def test_sliced_agrees_with_pot():
    rng = np.random.default_rng(7)
    X, Y = rng.standard_normal((400, 4)), rng.standard_normal((400, 4)) * 1.5 + 0.5
    ours = sliced_w2(X, Y, n_proj=500, seed=1, n_grid=400)
    pot = ot.sliced_wasserstein_distance(X, Y, n_projections=500, seed=1)
    assert ours == pytest.approx(float(pot), rel=0.08)


def test_mmd_zero_mean_for_same_distribution_and_positive_for_shift():
    rng = np.random.default_rng(8)
    same = [mmd2_rbf_unbiased(rng.standard_normal((80, 3)), rng.standard_normal((80, 3)), 0.5) for _ in range(30)]
    assert abs(np.mean(same)) < 0.01
    assert mmd2_rbf_unbiased(rng.standard_normal((80, 3)), rng.standard_normal((80, 3)) + 2, 0.5) > 0.2


def test_ledoit_wolf_handles_short_units_and_single_token():
    rng = np.random.default_rng(9)
    m, S = fit_gaussian(rng.standard_normal((3, 50)))
    assert np.all(np.linalg.eigvalsh(S) > 0)
    m, S = fit_gaussian(rng.standard_normal((1, 50)))
    assert S.shape == (50, 50)


def _clouds(rng, n_units, d, shift=0.0, scale=1.0, n_tok=(8, 30)):
    return [rng.standard_normal((rng.integers(*n_tok), d)) * scale + shift for _ in range(n_units)]


def test_reference_model_ranks_shifted_and_rescaled_units_higher():
    rng = np.random.default_rng(10)
    d = 16
    ref = ReferenceModel(n_proj=64).fit(_clouds(rng, 200, d))
    benign = _clouds(rng, 50, d)
    shifted = _clouds(rng, 50, d, shift=0.8)
    spread = _clouds(rng, 50, d, scale=2.5)   # same mean, different shape
    for meas in ("w2_gauss", "w2_sliced", "euclidean", "mmd_rbf"):
        b = np.mean([ref.score(X, meas) for X in benign])
        assert np.mean([ref.score(X, meas) for X in shifted]) > b, meas
    # only shape-aware measures see a pure covariance change
    b = np.mean([ref.score(X, "w2_shape") for X in benign])
    assert np.mean([ref.score(X, "w2_shape") for X in spread]) > b
    out = ref.score_all(benign[0])
    assert set(MEASURES) <= set(out)
    assert out["w2_gauss"] ** 2 == pytest.approx(out["loc_sq"] + out["shape_sq"], rel=1e-9)
