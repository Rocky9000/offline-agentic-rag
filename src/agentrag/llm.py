"""LLM backends. Everything runs against a model on your own machine."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str: ...


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.URLError as e:
        raise LLMError(f"could not reach {url}: {e}") from e


class OllamaLLM:
    """Chat completion via a local Ollama server (``ollama serve``)."""

    def __init__(self, model: str = "llama3.2:3b", host: str = "http://localhost:11434", timeout: float = 180):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            data = _post_json(f"{self.host}/api/chat", payload, self.timeout)
        except LLMError as e:
            raise LLMError(f"{e}. Is Ollama running? Try: ollama serve && ollama pull {self.model}") from e
        return data["message"]["content"]


class OpenAICompatLLM:
    """Any OpenAI-compatible local server: llama.cpp ``llama-server``, vLLM, LM Studio."""

    def __init__(self, model: str, base_url: str = "http://localhost:8080/v1", timeout: float = 180):
        self.model, self.base_url, self.timeout = model, base_url.rstrip("/"), timeout

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        data = _post_json(f"{self.base_url}/chat/completions", payload, self.timeout)
        return data["choices"][0]["message"]["content"]


class ScriptedLLM:
    """Deterministic stand-in for tests and demos.

    ``responder(prompt, system)`` returns the completion. ``calls`` records
    every request so tests can assert on the prompts agents send.
    """

    def __init__(self, responder: Callable[[str, str | None], str]):
        self.responder = responder
        self.calls: list[tuple[str, str | None]] = []

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        self.calls.append((prompt, system))
        return self.responder(prompt, system)
