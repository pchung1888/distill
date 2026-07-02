"""LLMPort protocol, LLMResponse, and the per-model cost table.

The provider-agnostic LLM interface: every provider (mock, gemini, anthropic,
openai, ollama) returns an LLMResponse and satisfies LLMPort, so the pipeline
never imports a concrete SDK (ports-and-adapters).
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class LLMResponse(BaseModel):
    """One completion's text plus token/cost metering for IngestTrace."""

    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


@runtime_checkable
class LLMPort(Protocol):
    """The single method every provider must implement."""

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse: ...


# model name -> (usd per 1M input tokens, usd per 1M output tokens).
# Prices as of mid-2026; unknown models cost 0.0 (never crash on a new name).
COST_TABLE: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "mock": (0.0, 0.0),
    "ollama": (0.0, 0.0),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """USD cost for a call; unknown models are treated as free (0.0)."""
    usd_in, usd_out = COST_TABLE.get(model, (0.0, 0.0))
    return (tokens_in * usd_in + tokens_out * usd_out) / 1_000_000
