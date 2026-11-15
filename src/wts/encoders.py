"""Encoders that turn a text unit into a token matrix (n x d).

- HFTokenEncoder: token-level outputs of a sentence-embedding model (bge-base-en-v1.5, all-MiniLM-L6-v2),
  taken before pooling; special tokens and padding removed.
- LLMLayerEncoder: hidden states of layer l of the protected LLM (white-box setting).
- PCAProjector: projects LLM hidden states (d = 4096) to 768 dims, fit on REFERENCE tokens only.
- HashingEncoder: deterministic, dependency-free encoder used ONLY by unit tests and `make smoke`.
  It is not a model and must never appear in a results table (run scripts refuse it).

torch / transformers are imported lazily so the rest of the library works on CPU without them.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np


class Encoder:
    name: str = "encoder"
    dim: int = 0
    test_only: bool = False

    def encode(self, text: str) -> np.ndarray:  # (n_tokens, dim)
        raise NotImplementedError

    def encode_many(self, texts: list[str], batch_size: int = 32) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]


# --------------------------------------------------------------------------------------------------
@dataclass
class HashingEncoder(Encoder):
    """TEST-ONLY. Each whitespace token -> signed feature-hashed character 3-grams, L2-normalised."""

    dim: int = 64
    name: str = "hashing-test-only"
    test_only: bool = True

    def _vec(self, tok: str) -> np.ndarray:
        v = np.zeros(self.dim)
        t = f"#{tok.lower()}#"
        for i in range(max(1, len(t) - 2)):
            h = hashlib.blake2b(t[i:i + 3].encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            v[idx] += sign
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def encode(self, text: str) -> np.ndarray:
        toks = re.findall(r"\w+|[^\w\s]", text) or ["<empty>"]
        return np.stack([self._vec(t) for t in toks])


# --------------------------------------------------------------------------------------------------
@dataclass
class HFTokenEncoder(Encoder):
    model_name: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"
    max_length: int = 512
    normalize: bool = True
    revision: str | None = None
    name: str = ""
    _tok: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def __post_init__(self):
        self.name = self.name or self.model_name.split("/")[-1]

    def _load(self):
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
            self._model = AutoModel.from_pretrained(self.model_name, revision=self.revision).to(self.device).eval()
            self.dim = self._model.config.hidden_size

    def encode_many(self, texts, batch_size: int = 32):
        import torch
        self._load()
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self._tok(batch, padding=True, truncation=True, max_length=self.max_length,
                            return_tensors="pt", return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                H = self._model(**enc).last_hidden_state
            if self.normalize:
                H = torch.nn.functional.normalize(H, dim=-1)
            keep = (enc["attention_mask"].bool() & ~special.to(self.device).bool())
            for j in range(H.shape[0]):
                Xj = H[j][keep[j]]
                if Xj.shape[0] == 0:  # empty text: keep the CLS vector so the unit is defined
                    Xj = H[j][:1]
                out.append(Xj.float().cpu().numpy())
        return out

    def encode(self, text):
        return self.encode_many([text])[0]


# --------------------------------------------------------------------------------------------------
@dataclass
class LLMLayerEncoder(Encoder):
    """Hidden states of one layer of a causal LLM. `model`/`tokenizer` can be shared with the generator."""

    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    layer: int = 16
    device: str = "cuda"
    max_length: int = 1024
    dtype: str = "bfloat16"
    revision: str | None = None
    projector: "PCAProjector | None" = None
    name: str = ""
    model: object = field(default=None, repr=False)
    tokenizer: object = field(default=None, repr=False)

    def __post_init__(self):
        self.name = self.name or f"{self.model_name.split('/')[-1]}-L{self.layer}"

    def _load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, revision=self.revision, torch_dtype=getattr(torch, self.dtype),
                device_map=self.device).eval()

    def raw_many(self, texts, batch_size: int = 8):
        import torch
        self._load()
        tok = self.tokenizer
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=self.max_length,
                      return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True).hidden_states[self.layer]
            for j in range(hs.shape[0]):
                Xj = hs[j][enc["attention_mask"][j].bool()]
                out.append(Xj.float().cpu().numpy())
        return out

    def encode_many(self, texts, batch_size: int = 8):
        raw = self.raw_many(texts, batch_size)
        if self.projector is not None:
            raw = [self.projector.transform(X) for X in raw]
        self.dim = raw[0].shape[1] if raw else self.dim
        return raw

    def encode(self, text):
        return self.encode_many([text])[0]


# --------------------------------------------------------------------------------------------------
@dataclass
class PCAProjector:
    """PCA fit on reference tokens only (never on calibration or test data)."""

    n_components: int = 768
    max_fit_tokens: int = 50000
    seed: int = 0
    mean_: np.ndarray | None = field(default=None, repr=False)
    components_: np.ndarray | None = field(default=None, repr=False)

    def fit(self, units: list[np.ndarray]) -> "PCAProjector":
        rng = np.random.default_rng(self.seed)
        X = np.concatenate(units, axis=0).astype(np.float64)
        if len(X) > self.max_fit_tokens:
            X = X[rng.choice(len(X), self.max_fit_tokens, replace=False)]
        self.mean_ = X.mean(axis=0)
        k = min(self.n_components, X.shape[0], X.shape[1])
        _, _, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[:k]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_.T


def build_encoder(cfg: dict) -> Encoder:
    kind = cfg["kind"]
    if kind == "hf_tokens":
        return HFTokenEncoder(model_name=cfg["model"], device=cfg.get("device", "cuda"),
                              max_length=cfg.get("max_length", 512), revision=cfg.get("revision"))
    if kind == "llm_layer":
        return LLMLayerEncoder(model_name=cfg["model"], layer=cfg["layer"], device=cfg.get("device", "cuda"),
                               revision=cfg.get("revision"))
    if kind == "hashing":
        return HashingEncoder(dim=cfg.get("dim", 64))
    raise ValueError(f"unknown encoder kind {kind}")
