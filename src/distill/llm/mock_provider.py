"""MockProvider -- deterministic canned responses for tests/CI.

No network, no cost, no randomness, no time dependence. Pipeline stages embed
marker substrings ("TASK: EXTRACT", "TASK: CRITIC", "TASK: REPAIR") in their
prompts; this provider keys canned JSON off those markers so the full pipeline
runs offline.
"""

import json

from distill.llm.base import LLMResponse

EXTRACT_MARKER = "TASK: EXTRACT"
CRITIC_MARKER = "TASK: CRITIC"
REPAIR_MARKER = "TASK: REPAIR"

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
    3. built-in markers, EARLIEST occurrence in the prompt wins (a critic
       prompt whose embedded source text contains "TASK: EXTRACT" still
       resolves as critic, because the template's own marker comes first):
       EXTRACT -> canned KnowledgeDraft JSON, CRITIC -> canned CriticResult
       JSON, REPAIR -> DEFAULT_RESPONSE (repairs must be scripted).
    4. DEFAULT_RESPONSE fixed echo string.

    `temperature` is accepted for LLMPort conformance and ignored -- the
    mock is already deterministic.
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
        temperature: float | None = None,
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
        marker_hits = [
            (prompt.find(marker), canned)
            for marker, canned in (
                (EXTRACT_MARKER, _CANNED_DRAFT),
                (CRITIC_MARKER, _CANNED_CRITIC),
                (REPAIR_MARKER, self.DEFAULT_RESPONSE),
            )
            if marker in prompt
        ]
        if marker_hits:
            return min(marker_hits, key=lambda hit: hit[0])[1]
        return self.DEFAULT_RESPONSE
