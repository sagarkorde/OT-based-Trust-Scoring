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


_GUARD = None


@contextmanager
def run_record(exp: str, config: dict, seed: int, out_dir: str | Path = REPO / "results"):
    """with run_record('e1', cfg, seed) as rec: rec['extra'] = ...  -> results/<exp>/runs/<ts>_<seed>.json"""
    out = Path(out_dir) / exp / "runs"
    out.mkdir(parents=True, exist_ok=True)
    from .hw import MemoryGuard, profile_record
    global _GUARD
    if _GUARD is None:
        _GUARD = MemoryGuard().start()
    rec = dict(exp=exp, seed=seed, config=config, git_commit=git_commit(), versions=_versions(), hardware=profile_record(),
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
        rec["peak_rss_gib"] = round(_GUARD.peak / 2**30, 2) if _GUARD and _GUARD.peak else None
        path = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{seed}_{os.getpid()}.json"
        path.write_text(json.dumps(rec, indent=2, default=str))


def append_rows(csv_path: str | Path, rows: list[dict]):
    """Append rows to a results CSV, writing the header on first use. Rows may carry different optional columns
    (e.g. one defense logs `n_flagged`, another `paraphrase`): the file's header is the union of all columns
    seen, and a row without a column leaves it empty. Each call is flushed to disk before returning, so the
    rows of finished work items survive a crash."""
    if not rows:
        return
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    if p.exists() and p.stat().st_size > 0:
        with p.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        extra = [c for c in cols if c not in header]
        if extra:   # rare: widen the file once, keeping every existing row
            import pandas as pd
            old = pd.read_csv(p, dtype=str, keep_default_na=False)
            for c in extra:
                old[c] = ""
            tmp = p.with_suffix(p.suffix + ".tmp")
            old.to_csv(tmp, index=False)
            os.replace(tmp, p)
            header = header + extra
        mode, write_header = "a", False
    else:
        header, mode, write_header = cols, "w", True
    with p.open(mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, restval="")
        if write_header:
            w.writeheader()
        w.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def refuse_test_only(encoder, allow: bool = False):
    if getattr(encoder, "test_only", False) and not allow:
        raise SystemExit(f"Encoder '{encoder.name}' is test-only and cannot produce results. "
                         "Use --smoke to run plumbing checks (written under results/_smoke/).")
