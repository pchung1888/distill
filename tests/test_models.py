"""Phase 1 tests: Pydantic data shapes in src/distill/models.py.

Coverage per HANDOFF Phase 1 DONE-WHEN:
- valid parse for each model (happy path),
- invalid parse raising pydantic.ValidationError,
- KnowledgeDoc.to_markdown() emits YAML frontmatter,
- IngestTrace totals sum correctly across multiple stages,
- JSON round-trip: model_dump -> model_validate equals original.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from distill.models import (
    CriticResult,
    Entity,
    IngestTrace,
    KnowledgeDoc,
    KnowledgeDraft,
    RawDocument,
    StageTrace,
)

FETCHED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)


def make_critic(**overrides: object) -> CriticResult:
    data: dict = {
        "confidence": 0.92,
        "faithful": True,
        "issues": [],
        "missing_points": [],
    }
    data.update(overrides)
    return CriticResult(**data)


def make_doc(**overrides: object) -> KnowledgeDoc:
    data: dict = {
        "source_type": "url",
        "source_ref": "https://example.com/article",
        "title": "Example Article",
        "summary": "A concise summary of the article.",
        "key_points": ["First point", "Second point"],
        "entities": [Entity(name="Example Corp", type="organization", mentions=3)],
        "topics": ["testing", "pydantic"],
        "critic": make_critic(),
        "created_at": FETCHED_AT,
    }
    data.update(overrides)
    return KnowledgeDoc(**data)


# ---------------------------------------------------------------- RawDocument


def test_raw_document_valid_parse() -> None:
    doc = RawDocument(
        source_type="url",
        source_ref="https://example.com/article",
        title="Example Article",
        text="Body text of the article.",
        fetched_at=FETCHED_AT,
    )
    assert doc.source_type == "url"
    assert doc.title == "Example Article"
    assert doc.meta == {}


def test_raw_document_title_optional_and_meta_default() -> None:
    doc = RawDocument(
        source_type="pdf",
        source_ref="report.pdf",
        title=None,
        text="PDF text.",
        fetched_at=FETCHED_AT,
    )
    assert doc.title is None
    assert doc.meta == {}


def test_raw_document_rejects_bad_source_type() -> None:
    with pytest.raises(ValidationError):
        RawDocument(
            source_type="ftp",
            source_ref="ftp://example.com/file",
            title=None,
            text="text",
            fetched_at=FETCHED_AT,
        )


def test_raw_document_rejects_missing_text() -> None:
    with pytest.raises(ValidationError):
        RawDocument(
            source_type="url",
            source_ref="https://example.com",
            title=None,
            fetched_at=FETCHED_AT,
        )


def test_raw_document_meta_defaults_are_independent() -> None:
    a = RawDocument(
        source_type="url",
        source_ref="https://a.example.com",
        title=None,
        text="a",
        fetched_at=FETCHED_AT,
    )
    b = RawDocument(
        source_type="url",
        source_ref="https://b.example.com",
        title=None,
        text="b",
        fetched_at=FETCHED_AT,
    )
    a.meta["k"] = "v"
    assert b.meta == {}


# --------------------------------------------------------------------- Entity


def test_entity_valid_parse() -> None:
    ent = Entity(name="Alice", type="person", mentions=2)
    assert ent.name == "Alice"
    assert ent.mentions == 2


def test_entity_mentions_optional() -> None:
    ent = Entity(name="Alice", type="person")
    assert ent.mentions is None


def test_entity_rejects_missing_name() -> None:
    with pytest.raises(ValidationError):
        Entity(type="person")


# ------------------------------------------------------------- KnowledgeDraft


def test_knowledge_draft_valid_parse() -> None:
    draft = KnowledgeDraft(
        summary="A summary.",
        key_points=["p1", "p2", "p3"],
        entities=[Entity(name="Bob", type="person")],
        topics=["topic-a"],
    )
    assert draft.summary == "A summary."
    assert len(draft.entities) == 1


def test_knowledge_draft_rejects_missing_summary() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDraft(key_points=["p1", "p2", "p3"], entities=[], topics=["t"])


def test_knowledge_draft_rejects_too_few_key_points() -> None:
    # The extract prompt demands 3-10 key points; the schema enforces it.
    with pytest.raises(ValidationError):
        KnowledgeDraft(
            summary="A summary.",
            key_points=["p1", "p2"],
            entities=[],
            topics=["topic-a"],
        )


def test_knowledge_draft_rejects_empty_topics_and_empty_summary() -> None:
    with pytest.raises(ValidationError):
        KnowledgeDraft(
            summary="A summary.",
            key_points=["p1", "p2", "p3"],
            entities=[],
            topics=[],
        )
    with pytest.raises(ValidationError):
        KnowledgeDraft(
            summary="",
            key_points=["p1", "p2", "p3"],
            entities=[],
            topics=["topic-a"],
        )


def test_knowledge_draft_json_schema_is_generated() -> None:
    schema = KnowledgeDraft.model_json_schema()
    assert schema["type"] == "object"
    for key in ("summary", "key_points", "entities", "topics"):
        assert key in schema["properties"]
    assert set(schema["required"]) == {"summary", "key_points", "entities", "topics"}


# --------------------------------------------------------------- CriticResult


def test_critic_result_valid_parse() -> None:
    critic = make_critic(issues=["minor phrasing drift"], missing_points=["date"])
    assert critic.confidence == 0.92
    assert critic.faithful is True
    assert critic.issues == ["minor phrasing drift"]


def test_critic_result_rejects_confidence_above_one() -> None:
    with pytest.raises(ValidationError):
        make_critic(confidence=1.5)


def test_critic_result_rejects_negative_confidence() -> None:
    with pytest.raises(ValidationError):
        make_critic(confidence=-0.1)


def test_critic_result_accepts_boundary_confidences() -> None:
    assert make_critic(confidence=0.0).confidence == 0.0
    assert make_critic(confidence=1.0).confidence == 1.0


# --------------------------------------------------------------- KnowledgeDoc


def test_knowledge_doc_valid_parse() -> None:
    doc = make_doc()
    assert doc.source_type == "url"
    assert doc.critic.faithful is True
    assert doc.entities[0].name == "Example Corp"


def test_knowledge_doc_rejects_bad_source_type() -> None:
    with pytest.raises(ValidationError):
        make_doc(source_type="rss")


def test_knowledge_doc_to_markdown_frontmatter() -> None:
    doc = make_doc()
    md = doc.to_markdown()
    lines = md.splitlines()

    # Starts with --- and closes the frontmatter block.
    assert lines[0] == "---"
    closing = lines[1:].index("---") + 1
    frontmatter = lines[1:closing]
    body = "\n".join(lines[closing + 1 :])

    fm_text = "\n".join(frontmatter)
    for key in ("source:", "type:", "title:", "topics:", "confidence:", "created:"):
        assert key in fm_text, f"missing frontmatter key {key!r}"
    assert "https://example.com/article" in fm_text
    assert "url" in fm_text
    assert "0.92" in fm_text

    # Body contains the summary text, key points, and entities.
    assert "A concise summary of the article." in body
    assert "First point" in body
    assert "Example Corp" in body


def test_knowledge_doc_to_markdown_handles_none_title() -> None:
    md = make_doc(title=None).to_markdown()
    assert md.startswith("---")
    assert "title: null" in md
    assert "# https://example.com/article" in md


def test_knowledge_doc_to_markdown_empty_title_matches_none() -> None:
    # An empty-string title renders exactly like a None title.
    assert make_doc(title="").to_markdown() == make_doc(title=None).to_markdown()


def test_knowledge_doc_fetched_at_optional_default_none() -> None:
    assert make_doc().fetched_at is None
    doc = make_doc(fetched_at=FETCHED_AT)
    assert doc.fetched_at == FETCHED_AT


def test_knowledge_doc_json_round_trip() -> None:
    doc = make_doc()
    dumped = doc.model_dump()
    restored = KnowledgeDoc.model_validate(dumped)
    assert restored == doc
    assert restored.model_dump() == dumped


# ----------------------------------------------------------------- StageTrace


def make_stage(name: str, tin: int, tout: int, cost: float, lat: int) -> StageTrace:
    return StageTrace(name=name, tokens_in=tin, tokens_out=tout, cost_usd=cost, latency_ms=lat)


def test_stage_trace_valid_parse() -> None:
    st = make_stage("extract", 1200, 300, 0.0042, 850)
    assert st.name == "extract"
    assert st.tokens_in == 1200


def test_stage_trace_rejects_missing_field() -> None:
    with pytest.raises(ValidationError):
        StageTrace(name="extract", tokens_in=1200, tokens_out=300, cost_usd=0.0042)


# ---------------------------------------------------------------- IngestTrace


def test_ingest_trace_totals_sum_across_stages() -> None:
    trace = IngestTrace(
        stages=[
            make_stage("extract", 1000, 200, 0.004, 800),
            make_stage("critic", 1500, 100, 0.005, 650),
            make_stage("repair", 300, 150, 0.001, 400),
        ]
    )
    assert trace.total_tokens_in == 2800
    assert trace.total_tokens_out == 450
    assert trace.total_cost_usd == pytest.approx(0.010)
    assert trace.total_latency_ms == 1850


def test_ingest_trace_empty_totals_are_zero() -> None:
    trace = IngestTrace(stages=[])
    assert trace.total_tokens_in == 0
    assert trace.total_tokens_out == 0
    assert trace.total_cost_usd == 0.0
    assert trace.total_latency_ms == 0


def test_ingest_trace_totals_appear_in_model_dump() -> None:
    trace = IngestTrace(stages=[make_stage("extract", 10, 5, 0.001, 100)])
    dumped = trace.model_dump()
    assert dumped["total_tokens_in"] == 10
    assert dumped["total_tokens_out"] == 5
    assert dumped["total_cost_usd"] == pytest.approx(0.001)
    assert dumped["total_latency_ms"] == 100


def test_ingest_trace_json_round_trip() -> None:
    trace = IngestTrace(
        stages=[
            make_stage("extract", 10, 5, 0.001, 100),
            make_stage("critic", 20, 8, 0.002, 200),
        ]
    )
    restored = IngestTrace.model_validate(trace.model_dump())
    assert restored == trace
    assert restored.total_tokens_in == 30
