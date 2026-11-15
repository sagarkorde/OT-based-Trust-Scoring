"""Encoding cache: token matrices per (encoder, text) stored as float16 .npz shards keyed by a text hash."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def encode_cached(encoder, texts: list[str], cache_dir: str | Path | None = None, batch_size: int = 32,
                  block: int = 256):
    cache_dir = Path(cache_dir or REPO / "data/cache") / encoder.name.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[np.ndarray | None] = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        f = cache_dir / f"{_key(t)}.npy"
        try:
            out[i] = np.load(f).astype(np.float64) if f.exists() else None
        except (ValueError, OSError, EOFError):   # file cut off by a crash: re-encode it
            out[i] = None
        if out[i] is None:
            todo.append(i)
    if todo:
        # encoded and saved in blocks, so an interrupted run keeps everything finished so far
        from .resume import Progress
        P = Progress(f"encode {encoder.name}", len(texts), resumed=len(texts) - len(todo), every_s=60) \
            if len(todo) > 4 * block else None
        for s in range(0, len(todo), block):
            idx = todo[s:s + block]
            enc = encoder.encode_many([texts[i] for i in idx], batch_size=batch_size)
            for i, X in zip(idx, enc):
                _save_atomic(cache_dir / f"{_key(texts[i])}.npy", X.astype(np.float16))
                out[i] = X.astype(np.float64)
            if P:
                P.step(len(idx))
    return out


def _save_atomic(path: Path, X: np.ndarray):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, X)
    os.replace(tmp, path)
