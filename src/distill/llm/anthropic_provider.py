"""AnthropicProvider -- optional live provider via the anthropic SDK.

Lazy SDK import (optional extra `distill[anthropic]`); not exercised by tests.
"""

import json
import os

from distill.llm.base import LLMResponse, estimate_cost

DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicProvider:
    """LLMPort adapter over the anthropic SDK. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - needs optional SDK
            raise ImportError(
                "anthropic is not installed. Install the optional extra: "
                "pip install 'distill[anthropic]'"
            ) from exc
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:  # pragma: no cover - live network call
        full_prompt = prompt
        if json_schema is not None:
            # Anthropic has no dedicated JSON-schema response mode; instruct
            # the model to answer with JSON conforming to the schema.
            full_prompt = (
                f"{prompt}\n\nRespond ONLY with JSON matching this JSON schema:\n"
                f"{json.dumps(json_schema)}"
            )
        kwargs: dict = {}
        if system:
            kwargs["system"] = system
        message = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": full_prompt}],
            **kwargs,
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        tokens_in = message.usage.input_tokens
        tokens_out = message.usage.output_tokens
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out),
        )
