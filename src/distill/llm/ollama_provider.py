"""OllamaProvider -- local models over HTTP via httpx (no API key, zero cost).

Talks to the Ollama /api/generate endpoint. httpx is a core dependency, so no
lazy import is needed; not exercised by tests (no live server in CI).
"""

import json
import os

from distill.llm.base import LLMResponse

DEFAULT_MODEL = "llama3.2"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider:
    """LLMPort adapter over a local Ollama server."""

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:  # pragma: no cover - needs a live Ollama server
        import httpx

        payload: dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["options"] = {"temperature": temperature}
        if json_schema is not None:
            full_schema_prompt = (
                f"{prompt}\n\nRespond ONLY with JSON matching this JSON schema:\n"
                f"{json.dumps(json_schema)}"
            )
            payload["prompt"] = full_schema_prompt
            payload["format"] = "json"
        response = httpx.post(f"{self.base_url}/api/generate", json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(
            text=data.get("response", ""),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
            cost_usd=0.0,
        )
