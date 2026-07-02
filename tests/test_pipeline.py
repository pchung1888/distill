"""Phase 4 tests: pipeline stages + orchestrator (MockProvider only, no network).

Coverage per HANDOFF Phase 4 DONE-WHEN:
- full happy-path run -> KnowledgeDoc with the canned content, trace with
  exactly extract + critic stages, positive totals, non-negative latency,
- malformed LLM output exercises the repair loop exactly once,
- repair budget exhausted -> PipelineError naming the validate stage,
- over-long sources are truncated in the prompt (recording wrapper),
- trace totals equal the sum of the per-stage numbers.
"""

import json
from datetime import UTC, datetime

import pytest

from distill.llm.base import LLMPort, LLMResponse
from distill.llm.mock_provider import MockProvider
from distill.models import RawDocument
from distill.pipeline import MAX_SOURCE_CHARS, Pipeline, PipelineError

FETCHED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)

# The MockProvider's canned extract summary (kept in sync with mock_provider).
CANNED_SUMMARY = "A mock summary of the source document produced for tests."


def make_raw_doc(text: str = "An article about Mock Corp and Jane Example.") -> RawDocument:
    return RawDocument(
        source_type="url",
        source_ref="https://example.com/article",
        title="Example Article",
        text=text,
        fetched_at=FETCHED_AT,
    )


def draft_json(summary: str = "A scripted summary.") -> str:
    return json.dumps(
        {
            "summary": summary,
            "key_points": ["Point one.", "Point two.", "Point three."],
            "entities": [{"name": "Mock Corp", "type": "organization"}],
            "topics": ["testing"],
        }
    )


def critic_json(
    confidence: float = 0.95,
    faithful: bool = True,
    issues: list[str] | None = None,
    missing_points: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "confidence": confidence,
            "faithful": faithful,
            "issues": issues or [],
            "missing_points": missing_points or [],
        }
    )


class RecordingLLM:
    """LLMPort wrapper capturing every prompt and response for assertions."""

    def __init__(self, inner: LLMPort) -> None:
        self.inner = inner
        self.prompts: list[str] = []
        self.responses: list[LLMResponse] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        response = self.inner.complete(prompt, system=system, json_schema=json_schema)
        self.responses.append(response)
        return response


# ------------------------------------------------------------- happy path


class TestHappyPath:
    def test_full_run_yields_canned_knowledge_doc(self) -> None:
        kdoc, _trace = Pipeline(MockProvider()).run(make_raw_doc())
        assert kdoc.summary == CANNED_SUMMARY
        assert len(kdoc.key_points) == 3
        assert kdoc.entities[0].name == "Mock Corp"
        assert kdoc.topics == ["testing", "mock-data"]
        assert kdoc.critic.confidence == 0.95
        assert kdoc.critic.faithful is True
        # Source fields carried over from the RawDocument.
        assert kdoc.source_type == "url"
        assert kdoc.source_ref == "https://example.com/article"
        assert kdoc.title == "Example Article"
        # No retry happened, so no provenance meta.
        assert kdoc.meta == {}

    def test_trace_has_exactly_extract_then_critic(self) -> None:
        _kdoc, trace = Pipeline(MockProvider()).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]

    def test_trace_totals_positive_and_latency_nonnegative(self) -> None:
        _kdoc, trace = Pipeline(MockProvider()).run(make_raw_doc())
        assert trace.total_tokens_in > 0
        assert trace.total_tokens_out > 0
        assert trace.total_latency_ms >= 0
        assert all(stage.latency_ms >= 0 for stage in trace.stages)


# ------------------------------------------------------------- repair loop


class TestRepairLoop:
    def test_malformed_then_valid_fires_repair_exactly_once(self) -> None:
        provider = MockProvider(script=["{not valid json", draft_json(), critic_json()])
        kdoc, trace = Pipeline(provider).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "repair", "critic"]
        assert kdoc.summary == "A scripted summary."

    def test_malformed_then_still_malformed_raises_pipeline_error(self) -> None:
        provider = MockProvider(script=["{bad", "{still bad"])
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(provider).run(make_raw_doc())
        assert excinfo.value.stage == "validate"
        assert "validate" in str(excinfo.value)
        assert excinfo.value.cause is not None

    def test_fenced_json_parses_without_repair(self) -> None:
        fenced = "```json\n" + draft_json("Fenced summary.") + "\n```"
        provider = MockProvider(script=[fenced, critic_json()])
        kdoc, trace = Pipeline(provider).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]
        assert kdoc.summary == "Fenced summary."

    def test_prose_wrapped_json_parses_without_repair(self) -> None:
        wrapped = "Sure! Here is the JSON you asked for:\n" + draft_json() + "\nHope that helps."
        provider = MockProvider(script=[wrapped, critic_json()])
        _kdoc, trace = Pipeline(provider).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]


# -------------------------------------------------------------- truncation


class TestTruncation:
    def test_long_source_is_truncated_in_prompts(self) -> None:
        text = "A" * MAX_SOURCE_CHARS + " TAIL_SENTINEL_MUST_NOT_APPEAR"
        recording = RecordingLLM(MockProvider())
        Pipeline(recording).run(make_raw_doc(text=text))
        extract_prompt, critic_prompt = recording.prompts
        for prompt in (extract_prompt, critic_prompt):
            assert "TAIL_SENTINEL_MUST_NOT_APPEAR" not in prompt
            assert "truncated to the first" in prompt

    def test_short_source_passes_through_untouched(self) -> None:
        recording = RecordingLLM(MockProvider())
        doc = make_raw_doc()
        Pipeline(recording).run(doc)
        assert doc.text in recording.prompts[0]
        assert "truncated to the first" not in recording.prompts[0]


# ------------------------------------------------------------ trace totals


class TestTraceCorrectness:
    def test_stage_meters_mirror_llm_responses_and_totals_sum(self) -> None:
        script = ["{not valid json", draft_json(), critic_json()]
        recording = RecordingLLM(MockProvider(script=script))
        _kdoc, trace = Pipeline(recording).run(make_raw_doc())

        # One StageTrace per LLM call, carrying that call's exact meters.
        assert len(trace.stages) == len(recording.responses)
        assert [s.tokens_in for s in trace.stages] == [
            r.tokens_in for r in recording.responses
        ]
        assert [s.tokens_out for s in trace.stages] == [
            r.tokens_out for r in recording.responses
        ]
        assert [s.cost_usd for s in trace.stages] == [
            r.cost_usd for r in recording.responses
        ]

        # Computed totals equal the per-stage sums.
        assert trace.total_tokens_in == sum(s.tokens_in for s in trace.stages)
        assert trace.total_tokens_out == sum(s.tokens_out for s in trace.stages)
        assert trace.total_cost_usd == pytest.approx(sum(s.cost_usd for s in trace.stages))
        assert trace.total_latency_ms == sum(s.latency_ms for s in trace.stages)
