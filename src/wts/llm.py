"""One LLM interface for every experiment: an OpenAI-compatible chat endpoint.

Open models are served with vLLM (`vllm serve <model> --port 8000`), which is also what AgentDojo's
LocalLLM and the InjecAgent prompted-agent evaluation expect. The API model uses the same client with
the provider's base URL. Temperature 0 everywhere unless an attack needs sampling.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class ChatLLM:
    model: str
    base_url: str | None = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 256
    retries: int = 5
    _client: object = field(default=None, repr=False)

    @property
    def client(self):
        if self._client is None:
            import openai
            key = os.environ.get(self.api_key_env, "EMPTY")
            self._client = openai.OpenAI(base_url=self.base_url, api_key=key) if self.base_url else openai.OpenAI(api_key=key)
        return self._client

    def chat(self, messages: list[dict], **kw) -> str:
        args = dict(model=self.model, messages=messages, temperature=kw.get("temperature", self.temperature),
                    max_tokens=kw.get("max_tokens", self.max_tokens))
        for attempt in range(self.retries):
            try:
                r = self.client.chat.completions.create(**args)
                return r.choices[0].message.content or ""
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def ask(self, prompt: str, system: str | None = None, **kw) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)


def build_llm(cfg: dict) -> ChatLLM:
    if cfg.get("ollama_base"):   # laptop profile: make sure the fixed-context tag exists
        from .hw import ensure_ollama_tag
        ensure_ollama_tag(cfg["model"], cfg["ollama_base"], cfg.get("num_ctx", 4096),
                          cfg.get("base_url", "").replace("/v1", "") or "http://localhost:11434")
    return ChatLLM(model=cfg["model"], base_url=cfg.get("base_url", "http://localhost:8000/v1"),
                   api_key_env=cfg.get("api_key_env", "OPENAI_API_KEY"),
                   max_tokens=cfg.get("max_tokens", 256))
