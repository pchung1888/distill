"""Phase 6 tests -- eval rubric units + run_evals integration (all offline).

evals/ is not an installed package, so the evals directory is put on
sys.path here (mirroring the bootstrap inside run_evals.py) and `rubric` /
`run_evals` are imported as top-level modules.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pydantic
import pytest

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import rubric  # noqa: E402
import run_evals  # noqa: E402

from distill.llm.mock_provider import MockProvider  # noqa: E402
from distill.models import CriticResult, KnowledgeDoc  # noqa: E402


def make_doc(
    summary: str = "Solar energy drives the water cycle.",
    key_points: list[str] | None = None,
    topics: list[str] | None = None,
) -> KnowledgeDoc:
    return KnowledgeDoc(
        source_type="url",
        source_ref="golden://unit-test",
        title="unit test",
        summary=summary,
        key_points=key_points
        or [
            "Water evaporates from the oceans.",
            "Vapor condenses into clouds.",
            "Rain returns water to the surface.",
        ],
        entities=[],
        topics=topics or ["water cycle", "hydrology"],
        critic=CriticResult(confidence=0.9, faithful=True),
        created_at=datetime.now(UTC),
    )


def make_expected(
    key_facts: list[str] | None = None,
    min_key_points: int = 3,
    topics_any: list[str] | None = None,
) -> rubric.ExpectedCase:
    return rubric.ExpectedCase(
        key_facts=key_facts if key_facts is not None else ["condenses into clouds"],
        min_key_points=min_key_points,
        topics_any=topics_any if topics_any is not None else ["water"],
    )


# ------------------------------------------------------------------ normalize


def test_normalize_casefolds_and_collapses_whitespace() -> None:
    assert rubric.normalize("  Solar\tENERGY\n  drives ") == "solar energy drives"


# --------------------------------------------------------- key_fact_presence


def test_key_fact_presence_all_hit() -> None:
    doc = make_doc()
    expected = make_expected(
        key_facts=["solar energy drives", "condenses into clouds", "evaporates"]
    )
    score, missing = rubric.key_fact_presence(doc, expected)
    assert score == 1.0
    assert missing == []


def test_key_fact_presence_miss_scores_fraction_and_reports_missing() -> None:
    doc = make_doc()
    expected = make_expected(
        key_facts=["condenses into clouds", "totally absent claim about volcanoes"]
    )
    score, missing = rubric.key_fact_presence(doc, expected)
    assert score == 0.5
    assert missing == ["totally absent claim about volcanoes"]


def test_key_fact_presence_matching_is_normalized() -> None:
    doc = make_doc(summary="The  P99   LATENCY dropped\nsharply.")
    expected = make_expected(key_facts=["p99 latency dropped sharply"])
    score, missing = rubric.key_fact_presence(doc, expected)
    assert score == 1.0
    assert missing == []


def test_key_fact_presence_empty_facts_scores_one() -> None:
    score, missing = rubric.key_fact_presence(make_doc(), make_expected(key_facts=[]))
    assert score == 1.0
    assert missing == []


# ------------------------------------------------------------------ topic_hit


def test_topic_hit_substring_match() -> None:
    assert rubric.topic_hit(make_doc(topics=["water cycle"]), make_expected()) is True


def test_topic_hit_no_match() -> None:
    doc = make_doc(topics=["astronomy"])
    assert rubric.topic_hit(doc, make_expected(topics_any=["water"])) is False


# ------------------------------------------------------------- min_key_points


def test_min_key_points_boundary() -> None:
    doc = make_doc()  # 3 key points
    assert rubric.min_key_points(doc, make_expected(min_key_points=3)) is True
    assert rubric.min_key_points(doc, make_expected(min_key_points=4)) is False


# --------------------------------------------------------------- schema_valid


def test_schema_valid_accepts_knowledge_doc() -> None:
    assert rubric.schema_valid(make_doc()) is True


def test_schema_valid_rejects_bad_input() -> None:
    assert rubric.schema_valid({"nope": 1}) is False
    assert rubric.schema_valid("not even json") is False
    assert rubric.schema_valid(42) is False


# ------------------------------------------------------ llm_judge_faithfulness


def test_judge_prompt_is_distinct_from_critic_prompt() -> None:
    # The judge has its OWN marker and claim-by-claim framing; it must not
    # reuse the pipeline critic's "TASK: CRITIC" template.
    prompt = rubric.build_judge_prompt("some source text", make_doc())
    assert prompt.startswith("TASK: JUDGE")
    assert "TASK: CRITIC" not in prompt
    assert "CLAIMS:" in prompt
    assert "1. (summary)" in prompt


def test_llm_judge_goes_through_llmport() -> None:
    # Unscripted MockProvider answers the judge's "TASK: JUDGE" prompt with
    # the canned high-confidence verdict -- deterministic, offline. The
    # LLMResponse comes back too so the harness can meter the call.
    judge, response = rubric.llm_judge_faithfulness(
        MockProvider(), "some source text", make_doc()
    )
    assert isinstance(judge, CriticResult)
    assert judge.faithful is True
    assert judge.confidence == 0.95
    assert response.tokens_in > 0


# ---------------------------------------------------------------- apply_rubric


def test_apply_rubric_pass_and_judge_floor_fail() -> None:
    doc = make_doc()
    expected = make_expected(key_facts=["condenses into clouds"])
    good_judge = CriticResult(confidence=0.9, faithful=True)
    low_judge = CriticResult(confidence=0.4, faithful=True)
    assert rubric.apply_rubric("t", doc, expected, good_judge).passed is True
    assert rubric.apply_rubric("t", doc, expected, low_judge).passed is False


def test_apply_rubric_fact_score_below_floor_fails() -> None:
    # One of two expected facts present -> score 0.5, below the 0.8 floor;
    # every other check passes, so the fact floor alone flips the verdict.
    doc = make_doc()
    expected = make_expected(
        key_facts=["condenses into clouds", "totally absent claim about volcanoes"]
    )
    judge = CriticResult(confidence=0.9, faithful=True)
    result = rubric.apply_rubric("t", doc, expected, judge)
    assert result.fact_score == 0.5
    assert result.topic_ok and result.points_ok and result.schema_ok
    assert result.passed is False


def test_apply_rubric_unfaithful_judge_fails_despite_high_confidence() -> None:
    doc = make_doc()
    expected = make_expected(key_facts=["condenses into clouds"])
    judge = CriticResult(
        confidence=0.9,
        faithful=False,
        issues=["claim 2 is not supported"],
        missing_points=["the source's main caveat"],
    )
    result = rubric.apply_rubric("t", doc, expected, judge)
    assert result.judge_confidence == 0.9
    assert result.passed is False
    # Judge diagnostics are carried through for the report.
    assert result.judge_issues == ["claim 2 is not supported"]
    assert result.judge_missing_points == ["the source's main caveat"]


class _NotADoc(pydantic.BaseModel):
    """Duck-typed stand-in that fails KnowledgeDoc/KnowledgeDraft validation."""

    summary: str = "Solar energy drives the water cycle."
    key_points: list[str] = ["only one point"]  # draft requires >= 3
    topics: list[str] = ["water cycle"]


def test_apply_rubric_schema_invalid_object_fails() -> None:
    # schema_valid is tautological for docs from Pipeline.run, but REAL for
    # arbitrary callers: a non-conforming object flunks the schema check.
    expected = make_expected(key_facts=[], min_key_points=1)
    judge = CriticResult(confidence=0.9, faithful=True)
    result = rubric.apply_rubric("t", _NotADoc(), expected, judge)  # type: ignore[arg-type]
    assert result.schema_ok is False
    assert result.passed is False


def test_failed_case_is_reported_not_raised() -> None:
    result = rubric.failed_case("broken", "boom")
    assert result.passed is False
    assert result.error == "boom"


# ------------------------------------------------------- run_evals integration


def test_run_evals_mock_provider_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = run_evals.main(["--provider", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "pass rate:" in out
    assert "PASS" in out
    # Mock-mode honesty NOTE appears before the table.
    assert "NOTE: mock provider" in out
    assert out.index("NOTE: mock provider") < out.index("pass rate:")
    # BOTH confidence lines, and the variance line carries n + small-n caveat.
    assert "mean pipeline critic confidence:" in out
    assert "mean judge confidence:" in out
    assert "pvariance, n=3; small-n -- indicative only" in out


def test_run_evals_judge_provider_mock_accepted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_evals.main(["--provider", "mock", "--judge-provider", "mock"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "judge provider: mock" in out


def _write_case(
    case_dir: Path,
    *,
    source: str,
    expected: dict,
    draft: dict,
    critic: dict,
    judge: dict | None = None,
) -> None:
    case_dir.mkdir(parents=True)
    (case_dir / "source.txt").write_text(source, encoding="utf-8")
    (case_dir / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    payload: dict = {"draft": draft, "critic": critic}
    if judge is not None:
        payload["judge"] = judge
    (case_dir / "mock_response.json").write_text(json.dumps(payload), encoding="utf-8")


_OK_DRAFT = {
    "summary": "A summary about gardening.",
    "key_points": ["Point one.", "Point two.", "Point three."],
    "entities": [],
    "topics": ["gardening"],
}
_OK_CRITIC = {"confidence": 0.9, "faithful": True, "issues": [], "missing_points": []}
_OK_EXPECTED = {
    "key_facts": ["a summary about gardening"],
    "min_key_points": 3,
    "topics_any": ["gardening"],
}


def test_run_evals_failing_case_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_case(
        tmp_path / "bad_case",
        source="A short source about gardening.",
        expected={
            "key_facts": ["a fact the draft definitely does not contain"],
            "min_key_points": 3,
            "topics_any": ["nomatch-topic"],
        },
        draft={
            "summary": "A summary about gardening.",
            "key_points": ["Point one.", "Point two.", "Point three."],
            "entities": [],
            "topics": ["gardening"],
        },
        critic={"confidence": 0.9, "faithful": True, "issues": [], "missing_points": []},
    )
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "missing fact:" in out


def test_run_evals_missing_golden_dir_is_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path / "nowhere")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "config error" in err


def test_run_case_pipeline_error_is_failed_case() -> None:
    # A provider that never yields valid JSON forces the validate stage's
    # repair budget to run out -> PipelineError -> reported as a FAILED case.
    case = run_evals.GoldenCase(
        name="crashy",
        source_text="some source",
        expected=make_expected(),
    )
    provider = MockProvider(script=["not json", "still not json"])
    result, trace = run_evals.run_case(
        case, provider, critic_threshold=0.7, fact_floor=0.8, judge_floor=0.7
    )
    assert result.passed is False
    assert result.error is not None
    assert "validate" in result.error
    # Failed runs are still metered: the two LLM calls appear in the trace.
    assert len(trace.stages) == 2


def test_judge_call_is_metered_as_stage_trace() -> None:
    case = run_evals.GoldenCase(
        name="metered",
        source_text="A short source about gardening.",
        expected=make_expected(key_facts=[], min_key_points=3, topics_any=["gardening"]),
    )
    provider = MockProvider(
        script=[json.dumps(_OK_DRAFT), json.dumps(_OK_CRITIC), json.dumps(_OK_CRITIC)]
    )
    result, trace = run_evals.run_case(
        case, provider, critic_threshold=0.7, fact_floor=0.8, judge_floor=0.7
    )
    assert result.passed is True
    # Both confidences surface on the CaseResult for the report.
    assert result.critic_confidence == 0.9
    assert result.judge_confidence == 0.9
    # The judge call is a metered StageTrace, so totals include judge spend.
    assert trace.stages[-1].name == "judge"
    assert trace.stages[-1].tokens_in > 0
    assert trace.total_tokens_in > sum(s.tokens_in for s in trace.stages[:-1])


def test_run_case_judge_runtime_error_is_failed_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ANY judge exception (not just PipelineError) marks the case FAILED
    # instead of crashing the harness.
    def boom(llm: object, source: str, doc: object) -> tuple:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run_evals.rubric, "llm_judge_faithfulness", boom)
    case = run_evals.GoldenCase(
        name="judgeless",
        source_text="A short source about gardening.",
        expected=make_expected(key_facts=[], min_key_points=3, topics_any=["gardening"]),
    )
    provider = MockProvider(script=[json.dumps(_OK_DRAFT), json.dumps(_OK_CRITIC)])
    result, trace = run_evals.run_case(
        case, provider, critic_threshold=0.7, fact_floor=0.8, judge_floor=0.7
    )
    assert result.passed is False
    assert result.error == "judge failed: kaboom"
    # The pipeline stages made before the judge broke are still metered.
    assert len(trace.stages) == 2


def test_run_evals_judge_error_yields_exit_one_not_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_case(
        tmp_path / "good_case",
        source="A short source about gardening.",
        expected=_OK_EXPECTED,
        draft=_OK_DRAFT,
        critic=_OK_CRITIC,
    )

    def boom(llm: object, source: str, doc: object) -> tuple:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run_evals.rubric, "llm_judge_faithfulness", boom)
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1  # the failed case drags pass rate below the floor; no crash
    assert "judge failed: kaboom" in out


def test_run_evals_bad_json_mock_response_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_dir = tmp_path / "broken_json"
    case_dir.mkdir(parents=True)
    (case_dir / "source.txt").write_text("some source", encoding="utf-8")
    (case_dir / "expected.json").write_text(json.dumps(_OK_EXPECTED), encoding="utf-8")
    (case_dir / "mock_response.json").write_text("{not valid json", encoding="utf-8")
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "config error" in err
    assert "not valid JSON" in err


def test_run_evals_missing_required_key_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_dir = tmp_path / "keyless"
    case_dir.mkdir(parents=True)
    (case_dir / "source.txt").write_text("some source", encoding="utf-8")
    (case_dir / "expected.json").write_text(json.dumps(_OK_EXPECTED), encoding="utf-8")
    (case_dir / "mock_response.json").write_text(
        json.dumps({"draft": _OK_DRAFT}), encoding="utf-8"  # no "critic" key
    )
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "config error" in err
    assert "missing required key" in err


def test_run_evals_typo_extra_field_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Strict eval-loader models (extra='forbid') make a typo'd field in a
    # hand-authored script fail loudly instead of being silently ignored.
    typo_critic = {"confidence": 0.9, "faithful": True, "issue": []}  # "issue" typo
    _write_case(
        tmp_path / "typo_case",
        source="A short source about gardening.",
        expected=_OK_EXPECTED,
        draft=_OK_DRAFT,
        critic=typo_critic,
    )
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "config error" in err
    assert "issue" in err


def test_run_evals_low_scripted_critic_confidence_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The script-order guard fires at load time: a scripted critic verdict
    # below --critic-threshold would trigger a pipeline retry and exhaust
    # the 3-entry mock script mid-run.
    low_critic = {"confidence": 0.5, "faithful": True, "issues": [], "missing_points": []}
    _write_case(
        tmp_path / "retry_case",
        source="A short source about gardening.",
        expected=_OK_EXPECTED,
        draft=_OK_DRAFT,
        critic=low_critic,
    )
    rc = run_evals.main(["--provider", "mock", "--golden-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "config error" in err
    assert "below --critic-threshold" in err


def test_run_evals_provider_construction_error_exits_two(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A missing API key (ValueError from the provider ctor) is a config
    # problem -> exit 2 with a clear message, never a raw traceback.
    def boom(name: str) -> object:
        raise ValueError("GEMINI_API_KEY is not set")

    monkeypatch.setattr(run_evals, "get_provider", boom)
    rc = run_evals.main(["--provider", "gemini"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "config error" in err
    assert "GEMINI_API_KEY" in err
