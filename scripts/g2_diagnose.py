"""Diagnose the G2 utility gap from AgentDojo's saved trajectories (no model calls).

Per attack case it records: utility/security, number of model turns, tool calls that AgentDojo's LocalLLM parser could
not read (the reply contains "<function=" but no tool call was extracted), empty replies, and whether a request hit the
client timeout (an empty reply in a run that lasted >= 590 s; AgentDojo's LocalLLM turns any request error into an
empty reply). Output: results/gates/G2_diagnosis.csv and a printed summary.

--retry-timeouts moves the logs of timed-out cases to results/agentdojo_logs/_retried/ so that the next
`run.ps1 gates G2` runs exactly those cases again (with the longer client timeout); nothing else is re-run.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def text_of(msg) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return "".join(b.get("content", "") or "" for b in c if isinstance(b, dict))
    return c or ""


def diagnose(logdir: Path, prefix: str) -> pd.DataFrame:
    rows = []
    for pipe in sorted(p for p in logdir.glob(f"{prefix}*") if p.is_dir()):
        for f in sorted(pipe.glob("*/*/important_instructions/*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(d.get("utility"), bool):
                continue
            asst = [m for m in d.get("messages", []) if m.get("role") == "assistant"]
            unparsed = sum(1 for m in asst if "<function=" in text_of(m) and not m.get("tool_calls"))
            empty = sum(1 for m in asst if not text_of(m).strip() and not m.get("tool_calls"))
            rows.append(dict(pipeline=pipe.name, suite=d["suite_name"], user_task=d["user_task_id"],
                             injection_task=d["injection_task_id"], utility=d["utility"], security=d["security"],
                             turns=len(asst), unparsed_calls=unparsed, empty_replies=empty,
                             duration_s=round(float(d.get("duration") or 0), 1),
                             timeout=bool(empty and float(d.get("duration") or 0) >= 590), log=str(f)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=str(REPO / "results/agentdojo_logs"))
    ap.add_argument("--prefix", default="local-ots-llama3.1-8b-16k-gate-")
    ap.add_argument("--retry-timeouts", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results/gates/G2_diagnosis.csv"))
    a = ap.parse_args()
    df = diagnose(Path(a.logdir), a.prefix)
    if df.empty:
        raise SystemExit(f"no finished G2 attack logs under {a.logdir}/{a.prefix}*")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="log").to_csv(out, index=False)
    n = len(df)
    any_unparsed = df.unparsed_calls > 0
    print(f"[G2 diagnosis] {n} attack cases from {df.pipeline.nunique()} pipeline log folder(s)")
    print(f"  utility under attack           {int(df.utility.sum())}/{n} = {df.utility.mean():.4f}")
    print(f"  cases with an unreadable call  {int(any_unparsed.sum())}/{n} = {any_unparsed.mean():.4f}"
          f"  (utility in those: {df.utility[any_unparsed].mean():.4f}; in the rest: {df.utility[~any_unparsed].mean():.4f})")
    print(f"  cases with an empty reply      {int((df.empty_replies > 0).sum())}")
    print(f"  cases that hit the timeout     {int(df.timeout.sum())}")
    print("  per suite:")
    print(df.groupby("suite").agg(n=("utility", "size"), utility=("utility", "mean"),
                                  unreadable_call=("unparsed_calls", lambda s: (s > 0).mean()),
                                  timeouts=("timeout", "sum")).round(3).to_string())
    print(f"  written: {out}")
    if a.retry_timeouts:
        bad = df[df.timeout]
        dest = Path(a.logdir) / "_retried"
        for f in bad.log:
            src = Path(f)
            tgt = dest / src.relative_to(a.logdir)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(tgt))
        print(f"[G2 diagnosis] moved {len(bad)} timed-out case log(s) to {dest}; run `gates G2` to redo only those")
