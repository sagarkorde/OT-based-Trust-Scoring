"""Cross-platform task runner (Windows has no `make`). Same targets as the Makefile, plus hardware profiles.

  python tasks.py <task> [--profile default|laptop]

Tasks: benchmarks, test, smoke, latency, gate-check, gates, e1, e2, e3, e4, tables, figures, refs, all
Profiles:
  default  bf16 models, vLLM endpoints (configs/*.yaml)
  laptop   8 GB GPU (RTX 4060 class): 4-bit NF4 models, Ollama endpoints, HarmBench judge on CPU,
           RAM ceiling 50 GB, reduced scope following the plan's drop order (configs/laptop/*.yaml)
Every task runs in a subprocess with PYTHONUTF8=1 (Windows reads files as cp1252 otherwise) and logs to
logs/<task>_<timestamp>.log as well as the console.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable
BENCH = {
    "agentdojo": ("https://github.com/ethz-spylab/agentdojo", "089ed468cf3ed0322acc66b0211f26d9d90dbf60"),
    "InjecAgent": ("https://github.com/uiuc-kang-lab/InjecAgent", "f19c9f2c79a41046eb13c03c51a24c567a8ffa07"),
    "PoisonedRAG": ("https://github.com/sleeepeer/PoisonedRAG", "f660d72174f06b13fae5163ce656e7b235db858f"),
    "HarmBench": ("https://github.com/centerforaisafety/HarmBench", "8e1604d1171fe8a48d8febecd22f600e462bdcdd"),
}
PROFILES = {
    "default": dict(env={}, cfg="configs"),
    "laptop": dict(env={"OT_QUANT": "4bit", "OT_JUDGE_DEVICE": "cpu", "OT_MAX_RAM_GB": "50", "OT_PROFILE": "laptop"},
                   cfg="configs/laptop"),
}


def run(cmd: list[str], env: dict, log: Path | None = None):
    # progress: our own "[task] done/total · elapsed · ~left" lines; library progress bars (tqdm) update at most
    # every 30 s so the console and the log file stay readable
    full = {**os.environ, **env, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TQDM_MININTERVAL": "30",
            "PYTHONPATH": str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    with subprocess.Popen(cmd, cwd=REPO, env=full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as p:
        f = log.open("a", encoding="utf-8") if log else None
        for line in p.stdout:
            sys.stdout.write(line)
            if f:
                f.write(line); f.flush()
        if f:
            f.close()
        if p.wait():
            raise SystemExit(f"FAILED ({p.returncode}): {' '.join(cmd)}")


def benchmarks():
    (REPO / "ext").mkdir(exist_ok=True)
    for name, (url, commit) in BENCH.items():
        d = REPO / "ext" / name
        if not (d / ".git").exists():
            subprocess.check_call(["git", "clone", "-q", url, str(d)])
        subprocess.check_call(["git", "-C", str(d), "fetch", "-q", "--depth", "1", "origin", commit])
        subprocess.check_call(["git", "-C", str(d), "checkout", "-q", commit])
        print(f"{name}: {commit[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--profile", default="default", choices=list(PROFILES))
    ap.add_argument("--only", nargs="+", default=None, help="gates task: which gates to run, e.g. --only G1 G4")
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    env, C = prof["env"], prof["cfg"]
    logs = REPO / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{a.task}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    s = lambda name, *args: [PY, "-W", "ignore", f"scripts/{name}", *args]
    gated = lambda: run(s("gate_check.py"), env, log)
    t = a.task
    if t == "benchmarks":
        benchmarks()
    elif t == "test":
        run([PY, "-m", "pytest", "-q", "tests"], env, log)
    elif t == "smoke":
        run(s("smoke.py"), env, log)
    elif t == "latency":
        dev = "cuda"
        run(s("bench_latency.py", "--device", dev, "--out", "results/e5_overhead.csv",
              "--judge", "meta-llama/Llama-Guard-3-8B", "--e1-config", f"{C}/e1.yaml"), env, log)
    elif t == "gates":
        run(s("run_gates.py", "--config", f"{C}/gates.yaml", *(["--only", *a.only] if a.only else [])), env, log)
    elif t == "g2-diagnose":       # reads the saved G2 trajectories; no model calls
        run(s("g2_diagnose.py"), env, log)
    elif t == "g2-retry":          # redo only the G2 cases whose request timed out, then rescore G2
        run(s("g2_diagnose.py", "--retry-timeouts"), env, log)
        run(s("run_gates.py", "--config", f"{C}/gates.yaml", "--only", "G2"), env, log)
    elif t == "gate-check":
        gated()
    elif t == "e1":
        gated(); run(s("run_e1_detection.py", "--config", f"{C}/e1.yaml"), env, log)
    elif t == "e2":
        gated(); run(s("run_e2_rag.py", "--config", f"{C}/e2.yaml"), env, log)
    elif t == "e3":
        gated()
        run(s("run_e3_agentdojo.py", "--config", f"{C}/e3.yaml", "--gt-detection"), env, log)
        run(s("run_e3_injecagent.py", "--config", f"{C}/e3.yaml"), env, log)
        run(s("run_e3_agentdojo.py", "--config", f"{C}/e3.yaml"), env, log)
    elif t == "e4":
        gated(); run(s("run_e4_adaptive.py", "all", "--config", f"{C}/e4.yaml"), env, log)
    elif t == "tables":
        run(s("make_tables.py", "--results", "results", "--out", "paper/tables"), env, log)
    elif t == "figures":
        run(s("make_figures.py", "--results", "results", "--out", "paper/figures"), env, log)
    elif t == "refs":
        run(s("verify_refs.py", "paper/refs.bib"), env, log)
    elif t == "all":
        steps = ("e1", "e2", "e3", "e4", "latency", "tables", "figures", "refs")
        for i, x in enumerate(steps, 1):   # every step resumes where it stopped, so `all` can simply be re-run
            print(f"\n[all] step {i}/{len(steps)}: {x}", flush=True)
            subprocess.check_call([PY, __file__, x, "--profile", a.profile], cwd=REPO)
    else:
        raise SystemExit(f"unknown task {t}")
    print(f"\n[tasks] {t} done; log: {log}")


if __name__ == "__main__":
    main()
