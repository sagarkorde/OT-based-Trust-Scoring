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


def _ollama_up(url: str) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=5):
            return True
    except Exception:
        return False


def _ollama_exe() -> str | None:
    import shutil
    import sys
    exe = shutil.which("ollama")
    if exe is None and sys.platform == "win32":   # default per-user install location on Windows
        cand = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
        exe = cand if os.path.exists(cand) else None
    return exe


def ensure_ollama_running(url: str = "http://localhost:11434", wait_s: int = 120) -> bool:
    """Start a local Ollama server if none answers at `url` (e.g. after a reboot when the tray app did not start).
    Returns True if it had to start one. The server runs detached, so it outlives this script, like the tray app."""
    import subprocess
    import sys
    from urllib.parse import urlparse
    if _ollama_up(url):
        return False
    host = urlparse(url).hostname or ""
    exe = _ollama_exe()
    if host not in ("localhost", "127.0.0.1", "::1") or exe is None:
        raise RuntimeError(f"Ollama is not running at {url} and could not be started automatically. "
                           "Open the Ollama app from the Start menu (or run `ollama serve` in another window), "
                           "then rerun the same command; finished work is kept.")
    print(f"[ollama] no server at {url}; starting `ollama serve` in the background ...", flush=True)
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    os.makedirs("logs", exist_ok=True)
    log = open(os.path.join("logs", "ollama_serve.log"), "ab")
    subprocess.Popen([exe, "serve"], stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                     creationflags=flags, start_new_session=sys.platform != "win32")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if _ollama_up(url):
            print(f"[ollama] server up after {time.time() - t0:.0f} s", flush=True)
            return True
        time.sleep(2)
    raise RuntimeError(f"Started `ollama serve` but nothing answered at {url} within {wait_s} s "
                       "(see logs/ollama_serve.log). Open the Ollama app from the Start menu, then rerun.")


def ensure_ollama_tag(name: str, base: str, num_ctx: int, url: str = "http://localhost:11434",
                      template_from: str | None = None, num_predict: int | None = None):
    """Create the fixed-context Ollama tag `name` from `base` if it does not exist (pulls `base` if needed).
    Uses Ollama's REST API, so configs alone decide which models a run needs.
    template_from: copy the chat template and stop words of this (installed or pullable) Ollama model. For GGUFs
    pulled from Hugging Face whose metadata carries no chat template (e.g. older Mistral-7B-Instruct-v0.1 GGUFs).
    num_predict: cap on reply length stored in the tag (used when a request sends no max_tokens). Ollama's default
    is unlimited generation, so a degenerate reply never ends. An existing tag without this cap is re-created in
    place under the same name (same base, context and template)."""
    import json
    import urllib.request

    def post(path, body, timeout=7200):
        req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:   # streamed status lines
                msg = json.loads(line or b"{}")
                if "error" in msg:
                    raise RuntimeError(f"ollama {path}: {msg['error']}")
    ensure_ollama_running(url)
    with urllib.request.urlopen(f"{url}/api/tags", timeout=10) as r:
        have = {m["name"] for m in json.loads(r.read())["models"]}
    def show(model):
        req = urllib.request.Request(f"{url}/api/show", data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    if name in have or f"{name}:latest" in have:
        if num_predict is None:
            return False
        cur = [ln.split() for ln in show(name).get("parameters", "").splitlines()]
        if any(len(p) >= 2 and p[0] == "num_predict" and p[1] == str(num_predict) for p in cur):
            return False
        print(f"[ollama] updating {name}: reply length cap num_predict={num_predict}", flush=True)
    if base not in have:
        print(f"[ollama] pulling {base} ...", flush=True)
        post("/api/pull", {"model": base})
    body = {"model": name, "from": base, "parameters": {"num_ctx": num_ctx, "temperature": 0}}
    if num_predict is not None:
        body["parameters"]["num_predict"] = num_predict
    if template_from:
        if template_from not in have:
            print(f"[ollama] pulling {template_from} (for its chat template) ...", flush=True)
            post("/api/pull", {"model": template_from})
        info = show(template_from)
        body["template"] = info["template"]
        stops = [ln.split(None, 1)[1].strip().strip('"') for ln in info.get("parameters", "").splitlines()
                 if ln.startswith("stop")]
        if stops:
            body["parameters"]["stop"] = stops
    post("/api/create", body)
    print(f"[ollama] created {name} (from {base}, num_ctx={num_ctx}"
          + (f", num_predict={num_predict}" if num_predict is not None else "")
          + (f", chat template of {template_from}" if template_from else "") + ")", flush=True)
    return True
