"""E5 / RQ6: per-unit latency of each score, excluding encoding (encoders are timed separately in E1).

H6 is evaluated on the GPU machine:  python scripts/bench_latency.py --device cuda --out results/e5_overhead.csv
On a CPU-only dev machine use --out results/_smoke/e5_cpu.csv: those numbers are NOT the H6 result.

Synthetic token clouds are used on purpose: latency depends on (n_tokens, d), not on content.
The LLM-judge comparison row (Llama Guard 3 8B) is timed by run_e1_detection.py on real prompts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wts.measures import ReferenceModel  # noqa: E402
from wts.runlog import append_rows, run_record  # noqa: E402


def time_numpy(ref, units, measure):
    ts = []
    for X in units:
        t = time.perf_counter()
        ref.score(X, measure)
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


def time_torch(ref, units, device):
    import torch
    from wts.measures_torch import ledoit_wolf_t
    mr = torch.tensor(ref.mean, device=device)
    Sr_sqrt = torch.tensor(ref.cov_sqrt, device=device)
    tr = float(np.trace(ref.cov))
    ts = []
    for X in units:
        Xt = torch.tensor(X, device=device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        m, S = ledoit_wolf_t(Xt)
        M = Sr_sqrt @ S @ Sr_sqrt
        ev = torch.linalg.eigvalsh(0.5 * (M + M.T)).clamp_min(0)
        w = torch.sqrt(((m - mr) ** 2).sum() + (torch.trace(S) + tr - 2 * ev.sqrt().sum()).clamp_min(0))
        float(w)  # forces sync
        ts.append((time.perf_counter() - t) * 1000)
    return np.array(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dims", default="384,768")
    ap.add_argument("--n-units", type=int, default=1000)
    ap.add_argument("--tokens", type=int, default=64, help="median tokens per unit")
    ap.add_argument("--measures", default="w2_gauss,w2_sliced,mahalanobis,cosine,mmd_rbf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cfg = vars(a)
    with run_record("e5", cfg, a.seed) as rec:
        rows = []
        for d in map(int, a.dims.split(",")):
            ref_units = [rng.standard_normal((rng.poisson(a.tokens) + 2, d)) for _ in range(300)]
            units = [rng.standard_normal((rng.poisson(a.tokens) + 2, d)) for _ in range(a.n_units)]
            ref = ReferenceModel(seed=a.seed).fit(ref_units)
            for meas in a.measures.split(","):
                ts = time_numpy(ref, units, meas)
                rows.append(dict(device="cpu-numpy", dim=d, measure=meas, n_units=len(ts),
                                 median_ms=round(float(np.median(ts)), 3), p95_ms=round(float(np.quantile(ts, .95)), 3)))
                print(rows[-1], flush=True)
            if a.device != "cpu":
                ts = time_torch(ref, units, a.device)
                rows.append(dict(device=a.device, dim=d, measure="w2_gauss", n_units=len(ts),
                                 median_ms=round(float(np.median(ts)), 3), p95_ms=round(float(np.quantile(ts, .95)), 3)))
                print(rows[-1], flush=True)
        append_rows(a.out, rows)
        rec["rows"] = rows


if __name__ == "__main__":
    main()
