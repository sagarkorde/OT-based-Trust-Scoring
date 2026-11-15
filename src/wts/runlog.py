"""Run logging (plan Section 6.3 rule 3): every run writes a JSON record with config, seed, git commit,
model and dataset revisions, wall-clock time; results go to CSV files that table/figure scripts read."""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(REPO), "diff", "--quiet"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "ot", "torch", "transformers", "agentdojo"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    return out


@contextmanager
def run_record(exp: str, config: dict, seed: int, out_dir: str | Path = REPO / "results"):
    """with run_record('e1', cfg, seed) as rec: rec['extra'] = ...  -> results/<exp>/runs/<ts>_<seed>.json"""
    out = Path(out_dir) / exp / "runs"
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(exp=exp, seed=seed, config=config, git_commit=git_commit(), versions=_versions(),
               host=platform.node(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), status="running")
    t0 = time.time()
    try:
        yield rec
        rec["status"] = "ok"
    except Exception as e:  # keep failed runs: they are part of the record
        rec["status"] = f"failed: {type(e).__name__}: {e}"
        raise
    finally:
        rec["wall_seconds"] = round(time.time() - t0, 2)
        path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{seed}_{os.getpid()}.json"
        path.write_text(json.dumps(rec, indent=2, default=str))


def append_rows(csv_path: str | Path, rows: list[dict]):
    """Append rows to a results CSV, writing the header on first use. Columns must stay stable."""
    if not rows:
        return
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def refuse_test_only(encoder, allow: bool = False):
    if getattr(encoder, "test_only", False) and not allow:
        raise SystemExit(f"Encoder '{encoder.name}' is test-only and cannot produce results. "
                         "Use --smoke to run plumbing checks (written under results/_smoke/).")
