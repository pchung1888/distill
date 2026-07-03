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

    `meta` carries pipeline provenance -- e.g. superseded critic verdicts
    when a low-confidence retry replaced them. Source provenance also
    survives: `fetched_at` is copied from the RawDocument, and any adapter
    meta (page_count, domain, ...) is preserved under meta["source_meta"].
    """
    merged: dict[str, Any] = dict(meta) if meta else {}
    if doc.meta:
        merged["source_meta"] = dict(doc.meta)
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
        fetched_at=doc.fetched_at,
        meta=merged,
    )
