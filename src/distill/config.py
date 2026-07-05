"""Runtime configuration loaded from environment variables.

Env vars only -- no .env file reading (the deploy environment injects them).
"""

import os

from pydantic import BaseModel, Field

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_ALLOWED_ORIGINS = "http://localhost:3000"
DEFAULT_RATELIMIT_DB_PATH = ":memory:"


class Config(BaseModel):
    """distill runtime settings; see .env.example for the variable catalog."""

    provider: str = "mock"
    model: str | None = None
    critic_threshold: float = 0.7
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL

    # Public-demo settings (distill-demo frontend). Irrelevant when distill is
    # only called server-to-server or via Swagger; see api/ratelimit.py.
    allowed_origins: list[str] = Field(default_factory=lambda: [DEFAULT_ALLOWED_ORIGINS])
    ratelimit_db_path: str = DEFAULT_RATELIMIT_DB_PATH
    ratelimit_daily_single_runs: int = 5
    ratelimit_daily_compares: int = 1
    ratelimit_global_daily_budget_usd: float = 2.0

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from process env vars, falling back to defaults."""
        origins_raw = os.environ.get("DISTILL_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        return cls(
            provider=os.environ.get("DISTILL_PROVIDER", "mock"),
            model=os.environ.get("DISTILL_MODEL") or None,
            critic_threshold=float(os.environ.get("DISTILL_CRITIC_THRESHOLD", "0.7")),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
            allowed_origins=origins or [DEFAULT_ALLOWED_ORIGINS],
            ratelimit_db_path=os.environ.get(
                "DISTILL_RATELIMIT_DB_PATH", DEFAULT_RATELIMIT_DB_PATH
            ),
            ratelimit_daily_single_runs=int(os.environ.get("DISTILL_RL_DAILY_SINGLE", "5")),
            ratelimit_daily_compares=int(os.environ.get("DISTILL_RL_DAILY_COMPARE", "1")),
            ratelimit_global_daily_budget_usd=float(
                os.environ.get("DISTILL_RL_GLOBAL_BUDGET_USD", "2.0")
            ),
        )
