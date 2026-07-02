"""OpenAIProvider -- optional live provider via the openai SDK.

Lazy SDK import (optional extra `distill[openai]`); not exercised by tests.
"""

import os

from distill.llm.base import LLMResponse, estimate_cost

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider:
    """LLMPort adapter over the openai SDK. Requires OPENAI_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - needs optional SDK
            raise ImportError(
                "openai is not installed. Install the optional extra: "
                "pip install 'distill[openai]'"
            ) from exc
        self.model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:  # pragma: no cover - live network call
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict = {}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        usage = completion.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        return LLMResponse(
            text=completion.choices[0].message.content or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out),
        )
