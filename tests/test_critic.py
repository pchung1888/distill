"""Phase 4 tests: critic stage + the orchestrator's bounded low-confidence retry.

Coverage per HANDOFF Phase 4 DONE-WHEN:
- critic parse of the canned MockProvider response,
- low confidence triggers exactly ONE bounded retry; the SECOND verdict wins
  and the first verdict is preserved in KnowledgeDoc.meta,
- a still-low retry is accepted anyway (never loops further; exactly 4 calls),
- the critic prompt contains both the source text and the draft JSON,
- confidence exactly at the threshold does NOT retry (>= threshold passes).
"""

import pytest
from test_pipeline import RecordingLLM, critic_json, draft_json, make_raw_doc

from distill.llm.mock_provider import MockProvider
from distill.models import KnowledgeDraft
from distill.pipeline import Pipeline, PipelineError, run_critic

LOW_ISSUES = ["The claim about record revenue is not in the source."]
LOW_MISSING = ["The product launch date mentioned in the source."]


def make_draft(summary: str = "A scripted summary.") -> KnowledgeDraft:
    return KnowledgeDraft.model_validate_json(draft_json(summary))


# ---------------------------------------------------------------- run_critic


class TestRunCritic:
    def test_parses_canned_critic_response(self) -> None:
        result, response = run_critic(MockProvider(), make_raw_doc(), make_draft())
        assert result.confidence == 0.95
        assert result.faithful is True
        assert result.issues == []
        assert response.tokens_in > 0

    def test_prompt_contains_source_text_and_draft_json(self) -> None:
        recording = RecordingLLM(MockProvider())
        doc = make_raw_doc()
        run_critic(recording, doc, make_draft("A very distinctive summary."))
        prompt = recording.prompts[0]
        assert "TASK: CRITIC" in prompt
        assert doc.text in prompt
        assert "A very distinctive summary." in prompt
        assert '"key_points"' in prompt

    def test_unparseable_critic_after_one_repair_raises(self) -> None:
        provider = MockProvider(script=["not json at all", "still not json"])
        with pytest.raises(PipelineError) as excinfo:
            run_critic(provider, make_raw_doc(), make_draft())
        assert excinfo.value.stage == "critic"


# ------------------------------------------------------- low-confidence retry


class TestLowConfidenceRetry:
    def test_low_confidence_triggers_exactly_one_retry(self) -> None:
        script = [
            draft_json("First attempt summary."),
            critic_json(0.4, faithful=False, issues=LOW_ISSUES, missing_points=LOW_MISSING),
            draft_json("Second attempt summary."),
            critic_json(0.9),
        ]
        recording = RecordingLLM(MockProvider(script=script))
        kdoc, trace = Pipeline(recording).run(make_raw_doc())

        assert [stage.name for stage in trace.stages] == [
            "extract",
            "critic",
            "extract_retry",
            "critic_retry",
        ]
        # The SECOND critic verdict is the one attached to the doc.
        assert kdoc.critic.confidence == 0.9
        assert kdoc.summary == "Second attempt summary."
        # The superseded verdict is preserved as provenance in meta.
        assert kdoc.meta["critic_retries"] == 1
        assert len(kdoc.meta["prior_critics"]) == 1
        assert kdoc.meta["prior_critics"][0]["confidence"] == 0.4
        assert kdoc.meta["prior_critics"][0]["faithful"] is False

    def test_retry_prompt_carries_critic_feedback(self) -> None:
        script = [
            draft_json("First attempt summary."),
            critic_json(0.4, faithful=False, issues=LOW_ISSUES, missing_points=LOW_MISSING),
            draft_json("Second attempt summary."),
            critic_json(0.9),
        ]
        recording = RecordingLLM(MockProvider(script=script))
        Pipeline(recording).run(make_raw_doc())

        retry_extract_prompt = recording.prompts[2]
        assert "PREVIOUS ATTEMPT ISSUES" in retry_extract_prompt
        assert LOW_ISSUES[0] in retry_extract_prompt
        assert LOW_MISSING[0] in retry_extract_prompt
        # The first extract prompt must NOT carry the feedback block.
        assert "PREVIOUS ATTEMPT ISSUES" not in recording.prompts[0]

    def test_still_low_after_retry_is_accepted_no_infinite_loop(self) -> None:
        script = [
            draft_json("First attempt summary."),
            critic_json(0.3, faithful=False, issues=LOW_ISSUES),
            draft_json("Second attempt summary."),
            critic_json(0.35, faithful=False, issues=LOW_ISSUES),
        ]
        kdoc, trace = Pipeline(MockProvider(script=script)).run(make_raw_doc())

        # Exactly 4 LLM calls -- one StageTrace each, then acceptance.
        assert len(trace.stages) == 4
        assert kdoc.critic.confidence == 0.35
        assert kdoc.critic.faithful is False
        assert kdoc.meta["prior_critics"][0]["confidence"] == 0.3

    def test_two_retries_preserve_all_prior_verdicts_in_order(self) -> None:
        script = [
            draft_json("First attempt summary."),
            critic_json(0.3, faithful=False, issues=LOW_ISSUES),
            draft_json("Second attempt summary."),
            critic_json(0.45, faithful=False, issues=LOW_ISSUES),
            draft_json("Third attempt summary."),
            critic_json(0.9),
        ]
        pipeline = Pipeline(MockProvider(script=script), max_critic_retries=2)
        kdoc, trace = pipeline.run(make_raw_doc())

        assert len(trace.stages) == 6
        assert kdoc.critic.confidence == 0.9
        assert kdoc.summary == "Third attempt summary."
        assert kdoc.meta["critic_retries"] == 2
        priors = kdoc.meta["prior_critics"]
        assert [verdict["confidence"] for verdict in priors] == [0.3, 0.45]

    def test_confidence_equal_to_threshold_does_not_retry(self) -> None:
        script = [draft_json(), critic_json(0.7)]
        pipeline = Pipeline(MockProvider(script=script), critic_threshold=0.7)
        kdoc, trace = pipeline.run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]
        assert kdoc.critic.confidence == 0.7
        assert kdoc.meta == {}

    def test_zero_retry_budget_accepts_low_confidence_immediately(self) -> None:
        script = [draft_json(), critic_json(0.2, faithful=False)]
        pipeline = Pipeline(MockProvider(script=script), max_critic_retries=0)
        kdoc, trace = pipeline.run(make_raw_doc())
        assert [stage.name for stage in trace.stages] == ["extract", "critic"]
        assert kdoc.critic.confidence == 0.2
        assert kdoc.meta == {}
