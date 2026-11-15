import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wts.measures import bures_wasserstein2, fit_gaussian, psd_sqrt  # noqa: E402
from wts.measures_torch import bures_w2_t, fit_gaussian_t  # noqa: E402


def test_torch_ledoit_wolf_matches_numpy():
    from wts.measures_torch import ledoit_wolf_t
    rng = np.random.default_rng(1)
    for n, d in [(5, 40), (30, 12), (200, 8)]:
        X = rng.standard_normal((n, d)) * rng.uniform(0.5, 2, d)
        m, S = fit_gaussian(X)
        mt, St = ledoit_wolf_t(torch.tensor(X))
        assert np.allclose(St.numpy(), S, atol=1e-10) and np.allclose(mt.numpy(), m)


def test_torch_matches_numpy_and_has_gradient():
    rng = np.random.default_rng(0)
    X, R = rng.standard_normal((15, 10)), rng.standard_normal((400, 10)) * 1.3 + 0.2
    mr, Sr = fit_gaussian(R, 0.1)
    m, S = fit_gaussian(X, 0.1)
    ref = np.sqrt(bures_wasserstein2(m, S, mr, Sr)[0])

    Xt = torch.tensor(X, requires_grad=True)
    mt, St = fit_gaussian_t(Xt, 0.1)
    w = bures_w2_t(mt, St, torch.tensor(mr), torch.tensor(psd_sqrt(Sr)), float(np.trace(Sr)))
    assert float(w) == pytest.approx(ref, rel=1e-6)
    w.backward()
    assert Xt.grad is not None and torch.isfinite(Xt.grad).all()

    # a gradient step on the unit lowers its distance to the reference
    with torch.no_grad():
        X2 = Xt - 0.05 * Xt.grad
    m2, S2 = fit_gaussian_t(X2, 0.1)
    w2 = bures_w2_t(m2, S2, torch.tensor(mr), torch.tensor(psd_sqrt(Sr)), float(np.trace(Sr)))
    assert float(w2) < float(w)
