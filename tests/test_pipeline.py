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
    """LLMPort wrapper capturing every prompt/system/temperature and response."""

    def __init__(self, inner: LLMPort) -> None:
        self.inner = inner
        self.prompts: list[str] = []
        self.systems: list[str | None] = []
        self.temperatures: list[float | None] = []
        self.responses: list[LLMResponse] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.prompts.append(prompt)
        self.systems.append(system)
        self.temperatures.append(temperature)
        response = self.inner.complete(
            prompt, system=system, json_schema=json_schema, temperature=temperature
        )
        self.responses.append(response)
        return response


class RaisingOnMarkerLLM:
    """Delegates to MockProvider, but RAISES on prompts containing `marker`."""

    def __init__(self, marker: str, exc: Exception) -> None:
        self.inner = MockProvider()
        self.marker = marker
        self.exc = exc

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        if self.marker in prompt:
            raise self.exc
        return self.inner.complete(
            prompt, system=system, json_schema=json_schema, temperature=temperature
        )


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
        assert [stage.name for stage in trace.stages] == [
            "extract",
            "validate_repair",
            "critic",
        ]
        assert kdoc.summary == "A scripted summary."

    def test_malformed_then_still_malformed_raises_pipeline_error(self) -> None:
        provider = MockProvider(script=["{bad", "{still bad"])
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(provider).run(make_raw_doc())
        assert excinfo.value.stage == "validate"
        assert "validate" in str(excinfo.value)
        assert excinfo.value.cause is not None

    def test_malformed_critic_then_valid_fires_critic_repair(self) -> None:
        # e2e through Pipeline.run(): valid draft, malformed critic, valid critic.
        provider = MockProvider(script=[draft_json(), "{not valid json", critic_json()])
        kdoc, trace = Pipeline(provider).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == [
            "extract",
            "critic",
            "critic_repair",
        ]
        assert kdoc.critic.confidence == 0.95

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

    def test_stray_brace_placeholder_before_json_parses_without_repair(self) -> None:
        wrapped = (
            "Filling the {placeholder} template as requested:\n"
            + draft_json("Stray-brace summary.")
        )
        provider = MockProvider(script=[wrapped, critic_json()])
        kdoc, trace = Pipeline(provider).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]
        assert kdoc.summary == "Stray-brace summary."

    def test_trailing_stray_brace_after_json_parses_without_repair(self) -> None:
        wrapped = draft_json("Trailing-brace summary.") + "\nThat closes it. }"
        provider = MockProvider(script=[wrapped, critic_json()])
        kdoc, trace = Pipeline(provider).run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]
        assert kdoc.summary == "Trailing-brace summary."


# ---------------------------------------------------------- failure metering


class TestPartialTrace:
    def test_repair_exhaustion_error_carries_partial_trace(self) -> None:
        provider = MockProvider(script=["{bad", "{still bad"])
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(provider).run(make_raw_doc())
        partial = excinfo.value.partial_trace
        assert partial is not None
        assert [stage.name for stage in partial.stages] == ["extract", "validate_repair"]
        assert all(stage.tokens_in > 0 for stage in partial.stages)
        assert all(stage.tokens_out > 0 for stage in partial.stages)

    def test_critic_repair_exhaustion_error_carries_partial_trace(self) -> None:
        provider = MockProvider(script=[draft_json(), "not json", "still not json"])
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(provider).run(make_raw_doc())
        assert excinfo.value.stage == "critic"
        partial = excinfo.value.partial_trace
        assert partial is not None
        assert [stage.name for stage in partial.stages] == [
            "extract",
            "critic",
            "critic_repair",
        ]

    def test_raising_provider_is_wrapped_and_metered(self) -> None:
        llm = RaisingOnMarkerLLM("TASK: CRITIC", RuntimeError("provider down"))
        with pytest.raises(PipelineError) as excinfo:
            Pipeline(llm).run(make_raw_doc())
        err = excinfo.value
        assert err.stage == "critic"
        assert isinstance(err.cause, RuntimeError)
        partial = err.partial_trace
        assert partial is not None
        assert [stage.name for stage in partial.stages] == ["extract", "critic"]
        # The successful extract call kept its real meters...
        assert partial.stages[0].tokens_in > 0
        # ...and the failed critic attempt is recorded as a zero-token entry.
        assert partial.stages[1].tokens_in == 0
        assert partial.stages[1].tokens_out == 0
        assert partial.stages[1].cost_usd == 0.0


# ---------------------------------------------------- prompts and provenance


class TestPromptsAndProvenance:
    def test_pipeline_pins_temperature_zero_and_sets_system(self) -> None:
        script = ["{not valid json", draft_json(), critic_json()]
        recording = RecordingLLM(MockProvider(script=script))
        Pipeline(recording).run(make_raw_doc())
        # extract + repair + critic all pin temperature=0.0.
        assert recording.temperatures == [0.0, 0.0, 0.0]
        # Role/persona text travels in system=, not in the user prompt.
        assert "knowledge extractor" in (recording.systems[0] or "")
        assert "repair" in (recording.systems[1] or "")
        assert "faithfulness critic" in (recording.systems[2] or "")
        for prompt in recording.prompts:
            assert "You are a" not in prompt

    def test_source_meta_and_fetched_at_survive_to_knowledge_doc(self) -> None:
        doc = make_raw_doc()
        doc.meta = {"page_count": 12, "domain": "example.com"}
        kdoc, _trace = Pipeline(MockProvider()).run(doc)
        assert kdoc.fetched_at == FETCHED_AT
        assert kdoc.meta["source_meta"] == {"page_count": 12, "domain": "example.com"}


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
