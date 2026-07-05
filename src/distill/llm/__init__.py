"""LLM provider layer: LLMPort protocol, providers, and the provider factory.

The factory is the ONLY place that maps a provider name to a concrete class;
everything else depends on the LLMPort protocol.
"""

import os

from distill.llm.base import COST_TABLE, LLMPort, LLMResponse, estimate_cost
from distill.llm.mock_provider import MockProvider

__all__ = [
    "COST_TABLE",
    "LLMPort",
    "LLMResponse",
    "MockProvider",
    "estimate_cost",
    "get_provider",
]

_VALID_PROVIDERS = ("mock", "gemini", "anthropic", "openai", "ollama")


def get_provider(name: str | None = None, model: str | None = None) -> LLMPort:
    """Return the provider named by `name`, DISTILL_PROVIDER, or "mock".

    Concrete provider classes are imported lazily inside each branch so that
    selecting one provider never imports another's optional SDK.
    """
    resolved = (name or os.environ.get("DISTILL_PROVIDER") or "mock").strip().lower()
    if resolved == "mock":
        return MockProvider()
    if resolved == "gemini":
        from distill.llm.gemini_provider import GeminiProvider

        return GeminiProvider(model=model) if model else GeminiProvider()
    if resolved == "anthropic":
        from distill.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if resolved == "openai":
        from distill.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model) if model else OpenAIProvider()
    if resolved == "ollama":
        from distill.llm.ollama_provider import OllamaProvider

        return OllamaProvider(model=model) if model else OllamaProvider()
    raise ValueError(
        f"Unknown LLM provider {resolved!r}; valid providers: {', '.join(_VALID_PROVIDERS)}"
    )
