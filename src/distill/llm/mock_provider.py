"""MockProvider -- deterministic canned responses for tests/CI.

No network, no cost, no randomness, no time dependence. Pipeline stages embed
marker substrings ("TASK: EXTRACT", "TASK: CRITIC") in their prompts; this
provider keys canned JSON off those markers so the full pipeline runs offline.
"""

import json

from distill.llm.base import LLMResponse

EXTRACT_MARKER = "TASK: EXTRACT"
CRITIC_MARKER = "TASK: CRITIC"

# Canned KnowledgeDraft JSON (matches distill.models.KnowledgeDraft).
_CANNED_DRAFT = json.dumps(
    {
        "summary": "A mock summary of the source document produced for tests.",
        "key_points": [
            "First canned key point.",
            "Second canned key point.",
            "Third canned key point.",
        ],
        "entities": [
            {"name": "Mock Corp", "type": "organization", "mentions": 3},
            {"name": "Jane Example", "type": "person", "mentions": 1},
        ],
        "topics": ["testing", "mock-data"],
    }
)

# Canned high-confidence CriticResult JSON (matches distill.models.CriticResult).
_CANNED_CRITIC = json.dumps(
    {
        "confidence": 0.95,
        "faithful": True,
        "issues": [],
        "missing_points": [],
    }
)


class MockProvider:
    """Deterministic LLMPort implementation for tests and CI.

    Response resolution order for each complete() call:
    1. script queue (if any responses remain) -- returned in FIFO order,
       for tests that need malformed output / low-confidence sequences.
    2. constructor `responses` override -- first key found as a substring
       of the prompt wins (insertion order).
    3. built-in markers: EXTRACT -> canned KnowledgeDraft JSON,
       CRITIC -> canned CriticResult JSON.
    4. DEFAULT_RESPONSE fixed echo string.
    """

    DEFAULT_RESPONSE = "mock-response: no marker matched"

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        script: list[str] | None = None,
    ) -> None:
        self._responses = dict(responses) if responses else {}
        self._script = list(script) if script else []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        text = self._resolve(prompt)
        tokens_in = max(1, (len(prompt) + (len(system) if system else 0)) // 4)
        tokens_out = max(1, len(text) // 4)
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=0.0,
        )

    def _resolve(self, prompt: str) -> str:
        if self._script:
            return self._script.pop(0)
        for key, value in self._responses.items():
            if key in prompt:
                return value
        if EXTRACT_MARKER in prompt:
            return _CANNED_DRAFT
        if CRITIC_MARKER in prompt:
            return _CANNED_CRITIC
        return self.DEFAULT_RESPONSE
