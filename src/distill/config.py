"""Runtime configuration loaded from environment variables.

Env vars only -- no .env file reading (the deploy environment injects them).
"""

import os

from pydantic import BaseModel

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class Config(BaseModel):
    """distill runtime settings; see .env.example for the variable catalog."""

    provider: str = "mock"
    model: str | None = None
    critic_threshold: float = 0.7
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from process env vars, falling back to defaults."""
        return cls(
            provider=os.environ.get("DISTILL_PROVIDER", "mock"),
            model=os.environ.get("DISTILL_MODEL") or None,
            critic_threshold=float(os.environ.get("DISTILL_CRITIC_THRESHOLD", "0.7")),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )
