"""GeminiProvider -- default live provider via the google-genai SDK.

The SDK is an optional extra (`pip install distill[gemini]`); it is imported
lazily inside __init__ so importing distill.llm never requires it. Not
exercised by tests (MockProvider covers the pipeline).
"""

import os

from distill.llm.base import LLMResponse, estimate_cost

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider:
    """LLMPort adapter over google-genai. Requires GEMINI_API_KEY."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - needs optional SDK
            raise ImportError(
                "google-genai is not installed. Install the optional extra: "
                "pip install 'distill[gemini]'"
            ) from exc
        self.model = model
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set and no api_key was provided")
        self._client = genai.Client(api_key=key)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:  # pragma: no cover - live network call
        from google.genai import types

        config_kwargs: dict = {}
        if system:
            config_kwargs["system_instruction"] = system
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = json_schema
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs) if config_kwargs else None,
        )
        usage = response.usage_metadata
        tokens_in = (usage.prompt_token_count or 0) if usage else 0
        tokens_out = (usage.candidates_token_count or 0) if usage else 0
        return LLMResponse(
            text=response.text or "",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=estimate_cost(self.model, tokens_in, tokens_out),
        )
