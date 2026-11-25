"""E5 / RQ6: per-unit latency of each score, excluding encoding (encoders are timed separately in E1).

H6 is evaluated on the GPU machine:  python scripts/bench_latency.py --device cuda --out results/e5_overhead.csv
On a CPU-only dev machine use --out results/_smoke/e5_cpu.csv: those numbers are NOT the H6 result.

Synthetic token clouds are used on purpose: latency depends on (n_tokens, d), not on content.
The LLM-judge comparison row (Llama Guard 3 8B) is timed here on real E1 prompts (--judge): per item, with a CUDA
sync, after 5 warm-up items, so median and p95 are true per-item figures. (run_e1_detection.py only writes a mean
judge row when its Llama Guard pass ran uninterrupted, which on the laptop it did not.)
Re-running skips the parts whose rows are already in --out.
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


def time_judge(model_name, cfg_path, n, seed):
    """Per-item Llama Guard 3 latency on n E1 prompts drawn at random (all families), same scorer as E1."""
    import random

    import torch
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_e1_detection as E1
    from wts.baselines import LlamaGuardScorer
    alpaca, lmsys, attacks, _ = E1.load_items(yaml.safe_load(open(cfg_path)))
    items = random.Random(seed).sample(alpaca + lmsys + attacks, n + 5)
    g = LlamaGuardScorer(model_name)
    ts = []
    for i, r in enumerate(items):
        torch.cuda.synchronize()
        t = time.perf_counter()
        g.scores([r["text"]])
        torch.cuda.synchronize()
        if i >= 5:   # warm-up excluded
            ts.append((time.perf_counter() - t) * 1000)
        if (i + 1) % 50 == 0:
            print(f"[latency] Llama Guard {i + 1}/{len(items)} items timed", flush=True)
    return np.array(ts)


def _existing(out):
    import pandas as pd
    p = Path(out)
    return pd.read_csv(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dims", default="384,768")
    ap.add_argument("--n-units", type=int, default=1000)
    ap.add_argument("--tokens", type=int, default=64, help="median tokens per unit")
    ap.add_argument("--measures", default="w2_gauss,w2_sliced,mahalanobis,cosine,mmd_rbf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="", help="HF id of the LLM judge to time on real prompts (needs --e1-config)")
    ap.add_argument("--e1-config", default="")
    ap.add_argument("--judge-n", type=int, default=300)
    a = ap.parse_args()
    have = _existing(a.out)
    synth_done = have is not None and (have.device.astype(str) == "cpu-numpy").any() and \
        (a.device == "cpu" or (have.device.astype(str) == a.device).any())
    judge_measure = f"judge:{a.judge.split('/')[-1].lower().replace('-', '_')}"
    judge_done = have is not None and ((have.measure.astype(str) == judge_measure) &
                                       (have.device.astype(str) == "cuda-per-item")).any()
    rng = np.random.default_rng(a.seed)
    cfg = vars(a)
    with run_record("e5", cfg, a.seed) as rec:
        rows = []
        if synth_done:
            print(f"[latency] synthetic score timings already in {a.out}, skipping", flush=True)
        for d in (map(int, a.dims.split(",")) if not synth_done else []):
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
        if rows:
            append_rows(a.out, rows)
        if a.judge and judge_done:
            print(f"[latency] {judge_measure} already timed, skipping", flush=True)
        elif a.judge:
            ts = time_judge(a.judge, a.e1_config, a.judge_n, a.seed)
            jrow = dict(device="cuda-per-item", dim="", measure=judge_measure, n_units=len(ts),
                        median_ms=round(float(np.median(ts)), 3), p95_ms=round(float(np.quantile(ts, .95)), 3))
            print(jrow, flush=True)
            append_rows(a.out, [jrow])
            rows.append(jrow)
        rec["rows"] = rows


if __name__ == "__main__":
    main()
