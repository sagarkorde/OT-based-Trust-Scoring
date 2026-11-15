"""Resumable runs and readable progress (used by every gate and experiment).

Journal   append-only JSON-lines file of finished work items, keyed by the item's identity. A restarted run loads
          the journal, skips every key already in it, and appends the rest. Each line is written and flushed as
          soon as its item finishes, so a crash, a reboot or Ctrl+C loses at most the item in progress.
          A journal belongs to one *version* of a computation: its file name carries a tag (config hash + code
          version), so results from an older, buggy harness are never mixed into a new run.
Progress  one line per update in the console: "[name] done/total (pct) · elapsed · ~time left · resumed n".
          Prints at most every `every_s` seconds (and on the last item), so logs stay short.
done_keys / purge_rows   the same idea for the append-only results CSVs that tables and figures read.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def tag_of(config: dict, version: str | int) -> str:
    """Short, stable tag for (config, code version): results from different versions never share a journal."""
    blob = json.dumps(config, sort_keys=True, default=str) + f"|v{version}"
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:   # a line cut off by a crash: ignore it, the item is redone
                        continue
                    self.records[rec["_key"]] = rec

    @staticmethod
    def key(*parts) -> str:
        return "|".join(str(p) for p in parts)

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __len__(self) -> int:
        return len(self.records)

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def add(self, key: str, record: dict):
        rec = {"_key": key, **record}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records[key] = rec

    def values(self) -> list[dict]:
        return list(self.records.values())


class Progress:
    def __init__(self, name: str, total: int, resumed: int = 0, every_s: float = 30.0):
        self.name, self.total, self.resumed, self.every_s = name, total, resumed, every_s
        self.done = resumed
        self.t0 = time.time()
        self.last = 0.0
        self.new = 0
        if resumed:
            print(f"[{name}] resuming: {resumed}/{total} already done", flush=True)
        elif total:
            print(f"[{name}] starting: {total} items", flush=True)

    def step(self, n: int = 1, note: str = ""):
        self.done += n
        self.new += n
        now = time.time()
        if now - self.last < self.every_s and self.done < self.total:
            return
        self.last = now
        el = now - self.t0
        rate = el / self.new if self.new else 0.0
        left = rate * max(self.total - self.done, 0)
        pct = 100.0 * self.done / self.total if self.total else 100.0
        msg = (f"[{self.name}] {self.done}/{self.total} ({pct:.0f}%) · {_fmt(el)} elapsed · ~{_fmt(left)} left"
               + (f" · resumed {self.resumed}" if self.resumed else "") + (f" · {note}" if note else ""))
        print(msg, flush=True)


def _fmt(s: float) -> str:
    s = int(s)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def done_keys(csv_path: str | Path, cols: list[str]) -> set[tuple]:
    """Keys (as tuples of strings) already present in an append-only results CSV."""
    p = Path(csv_path)
    if not p.exists():
        return set()
    import pandas as pd
    try:
        df = pd.read_csv(p, dtype=str, on_bad_lines="skip", usecols=lambda c: c in cols)
    except Exception:
        return set()
    if not set(cols) <= set(df.columns):
        return set()
    return set(map(tuple, df[cols].fillna("").astype(str).values.tolist()))


def purge_rows(csv_path: str | Path, match: dict):
    """Remove rows whose columns equal `match` (used before redoing a unit that was only partly written)."""
    p = Path(csv_path)
    if not p.exists():
        return 0
    import pandas as pd
    df = pd.read_csv(p, dtype=str, on_bad_lines="skip")
    if not set(match) <= set(df.columns):
        return 0
    mask = pd.Series(True, index=df.index)
    for k, v in match.items():
        mask &= df[k].fillna("") == str(v)
    n = int(mask.sum())
    if n:
        tmp = p.with_suffix(p.suffix + ".tmp")
        df[~mask].to_csv(tmp, index=False)
        os.replace(tmp, p)
    return n


def mark_done(path: str | Path, info: dict | None = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), **(info or {})}), encoding="utf-8")


def resumable_map(label: str, keys: list[str], items: list, fn, journal_path: str | Path, chunk: int = 32,
                  every_s: float = 30.0) -> list:
    """[fn over items], computed in chunks and journaled per item, so a restart only computes what is missing.
    `fn(batch) -> list of per-item JSON-serialisable values` (same order). Returns the values in `keys` order."""
    J = Journal(journal_path)
    todo = [i for i, k in enumerate(keys) if k not in J]
    P = Progress(label, len(keys), resumed=len(keys) - len(todo), every_s=every_s)
    for s in range(0, len(todo), chunk):
        idx = todo[s:s + chunk]
        vals = fn([items[i] for i in idx])
        for i, v in zip(idx, vals):
            J.add(keys[i], {"v": _plain(v)})
        P.step(len(idx))
    return [J.get(k)["v"] for k in keys]


def _plain(v):
    """numpy scalars / arrays / tuples -> plain JSON values."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (tuple, list)):
        return [_plain(x) for x in v]
    return v


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def filter_rows(csv_path: str | Path, keep) -> int:
    """Keep only rows for which keep(row) is true (row: dict of strings). Returns the number removed. Used to drop
    the partial output of a work item that was interrupted before its 'done' record was written."""
    p = Path(csv_path)
    if not p.exists():
        return 0
    import pandas as pd
    df = pd.read_csv(p, dtype=str, keep_default_na=False, on_bad_lines="skip")
    if df.empty:
        return 0
    mask = df.apply(lambda r: bool(keep(r.to_dict())), axis=1)
    n = int((~mask).sum())
    if n:
        tmp = p.with_suffix(p.suffix + ".tmp")
        df[mask].to_csv(tmp, index=False)
        os.replace(tmp, p)
    return n
