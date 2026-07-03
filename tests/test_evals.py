"""Phase 6 tests -- eval rubric units + run_evals integration (all offline).

evals/ is not an installed package, so the evals directory is put on
sys.path here (mirroring the bootstrap inside run_evals.py) and `rubric` /
`run_evals` are imported as top-level modules.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

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


def test_llm_judge_goes_through_llmport() -> None:
    # Unscripted MockProvider answers the judge's "TASK: CRITIC" prompt with
    # the canned high-confidence verdict -- deterministic, offline.
    judge = rubric.llm_judge_faithfulness(MockProvider(), "some source text", make_doc())
    assert isinstance(judge, CriticResult)
    assert judge.faithful is True
    assert judge.confidence == 0.95


# ---------------------------------------------------------------- apply_rubric


def test_apply_rubric_pass_and_judge_floor_fail() -> None:
    doc = make_doc()
    expected = make_expected(key_facts=["condenses into clouds"])
    good_judge = CriticResult(confidence=0.9, faithful=True)
    low_judge = CriticResult(confidence=0.4, faithful=True)
    assert rubric.apply_rubric("t", doc, expected, good_judge).passed is True
    assert rubric.apply_rubric("t", doc, expected, low_judge).passed is False


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
    assert "variance" in out
    assert "PASS" in out


def _write_case(
    case_dir: Path,
    *,
    source: str,
    expected: dict,
    draft: dict,
    critic: dict,
) -> None:
    case_dir.mkdir(parents=True)
    (case_dir / "source.txt").write_text(source, encoding="utf-8")
    (case_dir / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    (case_dir / "mock_response.json").write_text(
        json.dumps({"draft": draft, "critic": critic}), encoding="utf-8"
    )


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
