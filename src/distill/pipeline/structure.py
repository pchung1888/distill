"""Structure stage -- assemble the final KnowledgeDoc (no LLM call)."""

from datetime import UTC, datetime
from typing import Any

from distill.models import CriticResult, KnowledgeDoc, KnowledgeDraft, RawDocument


def build_doc(
    doc: RawDocument,
    draft: KnowledgeDraft,
    critic: CriticResult,
    *,
    meta: dict[str, Any] | None = None,
) -> KnowledgeDoc:
    """Combine source fields, validated draft content, and the critic verdict.

    `meta` carries pipeline provenance -- e.g. the first critic verdict when
    a low-confidence retry replaced it.
    """
    return KnowledgeDoc(
        source_type=doc.source_type,
        source_ref=doc.source_ref,
        title=doc.title,
        summary=draft.summary,
        key_points=draft.key_points,
        entities=draft.entities,
        topics=draft.topics,
        critic=critic,
        created_at=datetime.now(UTC),
        meta=dict(meta) if meta else {},
    )
