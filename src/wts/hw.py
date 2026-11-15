"""Hardware adaptation: model loading for small GPUs and a RAM ceiling.

Profiles are chosen with environment variables so every script behaves the same way:
  OT_QUANT        bf16 (default; >= 24 GB GPU) | 4bit (NF4 via bitsandbytes; 8 GB GPUs)
  OT_JUDGE_DEVICE cuda (default) | cpu   -> the 13B HarmBench classifier in bf16 on CPU (~26 GB RAM)
  OT_MAX_RAM_GB   RAM ceiling enforced by MemoryGuard (default: no ceiling)
The laptop profile (tasks.py --profile laptop) sets OT_QUANT=4bit, OT_JUDGE_DEVICE=cpu, OT_MAX_RAM_GB=50.
Any deviation from the plan's bf16 models is recorded automatically in each run's JSON log (runlog).
"""
from __future__ import annotations

import os
import threading
import time


def quant_mode() -> str:
    return os.environ.get("OT_QUANT", "bf16")


def load_causal_lm(name: str, revision: str | None = None, device: str | None = None, quant: str | None = None):
    """(model, tokenizer) for a causal LM under the active profile."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    quant = quant or quant_mode()
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    kw = dict(revision=revision, low_cpu_mem_usage=True)
    if device == "cpu":
        kw.update(torch_dtype=torch.bfloat16, device_map={"": "cpu"})
    elif quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True))
    else:
        kw.update(torch_dtype=torch.bfloat16, device_map=device or "cuda")
    model = AutoModelForCausalLM.from_pretrained(name, **kw).eval()
    return model, tok


def profile_record() -> dict:
    """What the run actually used; written into every run log."""
    out = {k: os.environ.get(k) for k in ("OT_QUANT", "OT_JUDGE_DEVICE", "OT_MAX_RAM_GB", "OT_PROFILE")}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = f"{p.name} {p.total_memory / 2**30:.1f} GiB"
    except Exception:
        pass
    return out


class MemoryGuard:
    """Stops the run cleanly when this process tree's resident memory exceeds the ceiling.
    Checks every `interval` seconds; on breach it interrupts the main thread (KeyboardInterrupt), so the
    run's JSON record is still written with status 'failed: KeyboardInterrupt' and the message below."""

    def __init__(self, max_gb: float | None = None, interval: float = 5.0):
        env = os.environ.get("OT_MAX_RAM_GB")
        self.max_bytes = (max_gb or (float(env) if env else 0)) * 2**30
        self.interval = interval
        self.peak = 0

    def _rss(self):
        import psutil
        p = psutil.Process()
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except Exception:
                pass
        return total

    def _loop(self):
        import _thread
        while True:
            r = self._rss()
            self.peak = max(self.peak, r)
            if r > self.max_bytes:
                print(f"\n[MemoryGuard] RSS {r / 2**30:.1f} GiB exceeds the {self.max_bytes / 2**30:.0f} GiB "
                      f"ceiling; stopping this run.", flush=True)
                _thread.interrupt_main()
                return
            time.sleep(self.interval)

    def start(self):
        if self.max_bytes > 0:
            threading.Thread(target=self._loop, daemon=True).start()
        return self


def free_gpu(*objs):
    """Drop model references held by encoders / scorers and return their VRAM (8 GB cards: one model at a time)."""
    import gc
    for o in objs:
        for attr in ("model", "_model", "_m", "_tok", "_t", "tokenizer"):
            if hasattr(o, attr):
                try:
                    setattr(o, attr, None)
                except Exception:
                    pass
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def ollama_unload_all(base: str = "http://localhost:11434"):
    """Unload every model Ollama holds in VRAM (it reloads on the next request)."""
    import json
    import urllib.request
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=5) as r:
            models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        for m in models:
            req = urllib.request.Request(f"{base}/api/generate", data=json.dumps({"model": m, "keep_alive": 0}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=30).read()
        return models
    except Exception:
        return []
