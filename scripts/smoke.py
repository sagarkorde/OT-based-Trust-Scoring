"""`make smoke`: run every pipeline end to end on local benchmark text with the TEST-ONLY hashing encoder.
No downloads, no GPU, no LLM. Everything is written under results/_smoke/ and can never reach paper/
(make_tables / make_figures refuse). Use it after any code change and on a new machine before Phase 2.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STEPS = [
    ["scripts/run_e1_detection.py", "--smoke"],
    ["scripts/run_e3_agentdojo.py", "--smoke"],
    ["scripts/run_e3_injecagent.py", "--smoke"],
    ["scripts/run_e4_adaptive.py", "--smoke"],
    ["scripts/bench_latency.py", "--n-units", "100", "--out", "results/_smoke/e5_cpu.csv"],
    ["scripts/make_tables.py", "--results", "results/_smoke", "--out", "results/_smoke/tables", "--n-boot", "200"],
    ["scripts/make_figures.py", "--results", "results/_smoke", "--out", "results/_smoke/figures"],
]

if __name__ == "__main__":
    for step in STEPS:
        print(f"\n=== {' '.join(step)}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", *step], cwd=REPO)
        if r.returncode:
            sys.exit(f"smoke step failed: {' '.join(step)}")
    print("\nSMOKE OK (plumbing only; numbers under results/_smoke are not results)")
