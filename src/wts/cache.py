"""Encoding cache: token matrices per (encoder, text) stored as float16 .npz shards keyed by a text hash."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def encode_cached(encoder, texts: list[str], cache_dir: str | Path | None = None, batch_size: int = 32):
    cache_dir = Path(cache_dir or REPO / "data/cache") / encoder.name.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[np.ndarray | None] = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        f = cache_dir / f"{_key(t)}.npy"
        if f.exists():
            out[i] = np.load(f).astype(np.float64)
        else:
            todo.append(i)
    if todo:
        enc = encoder.encode_many([texts[i] for i in todo], batch_size=batch_size)
        for i, X in zip(todo, enc):
            np.save(cache_dir / f"{_key(texts[i])}.npy", X.astype(np.float16))
            out[i] = X.astype(np.float64)
    return out
