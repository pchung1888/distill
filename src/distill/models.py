"""Pydantic v2 data shapes for distill.

Models (HANDOFF Section 5):
- RawDocument: what a SourcePort adapter produces from a URL / YouTube / PDF.
- Entity / KnowledgeDraft: what the Extract stage produces; Validate enforces it.
- CriticResult: the Critic stage's faithfulness verdict.
- KnowledgeDoc: the final validated draft + critic verdict + source reference,
  renderable as Markdown with YAML frontmatter or JSON via model_dump().
- StageTrace / IngestTrace: per-stage token / cost / latency metering -- the
  token-economics showcase surface.
"""

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

SourceType = Literal["url", "youtube", "pdf"]


class RawDocument(BaseModel):
    """Clean text extracted from a source, before any LLM stage runs."""

    model_config = ConfigDict(extra="ignore")

    source_type: SourceType
    source_ref: str
    title: str | None = None
    text: str
    fetched_at: datetime
    meta: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A named entity found in the source (person, organization, concept, ...)."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str
    mentions: int | None = None


class KnowledgeDraft(BaseModel):
    """The Extract stage's structured output; the Validate stage enforces this
    schema (its JSON schema via model_json_schema() feeds the extract prompt).
    """

    model_config = ConfigDict(extra="ignore")

    summary: str
    key_points: list[str]
    entities: list[Entity]
    topics: list[str]


class CriticResult(BaseModel):
    """The Critic stage's verdict on how faithful a draft is to its source."""

    model_config = ConfigDict(extra="ignore")

    confidence: float = Field(ge=0, le=1)
    faithful: bool
    issues: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)


def _yaml_str(value: str | None) -> str:
    """Render a string as a safely quoted YAML scalar (JSON is a YAML subset)."""
    if value is None:
        return "null"
    return json.dumps(value)


class KnowledgeDoc(BaseModel):
    """The pipeline's final product: validated draft + critic verdict + source."""

    model_config = ConfigDict(extra="ignore")

    source_type: SourceType
    source_ref: str
    title: str | None = None
    summary: str
    key_points: list[str]
    entities: list[Entity]
    topics: list[str]
    critic: CriticResult
    created_at: datetime

    def to_markdown(self) -> str:
        """Render as Markdown with YAML frontmatter (source, type, title,
        topics, confidence, created) followed by a readable body.
        """
        topics = "[" + ", ".join(_yaml_str(t) for t in self.topics) + "]"
        lines = [
            "---",
            f"source: {_yaml_str(self.source_ref)}",
            f"type: {self.source_type}",
            f"title: {_yaml_str(self.title)}",
            f"topics: {topics}",
            f"confidence: {self.critic.confidence}",
            f"created: {self.created_at.isoformat()}",
            "---",
            "",
            f"# {self.title or self.source_ref}",
            "",
            "## Summary",
            "",
            self.summary,
            "",
            "## Key Points",
            "",
        ]
        lines.extend(f"- {point}" for point in self.key_points)
        lines += ["", "## Entities", ""]
        for ent in self.entities:
            suffix = f" ({ent.mentions} mentions)" if ent.mentions is not None else ""
            lines.append(f"- {ent.name} [{ent.type}]{suffix}")
        return "\n".join(lines) + "\n"


class StageTrace(BaseModel):
    """Token / cost / latency metering for one pipeline stage."""

    model_config = ConfigDict(extra="ignore")

    name: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int


class IngestTrace(BaseModel):
    """Per-run metering across all pipeline stages, with computed totals."""

    model_config = ConfigDict(extra="ignore")

    stages: list[StageTrace] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.stages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.stages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.stages)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_latency_ms(self) -> int:
        return sum(s.latency_ms for s in self.stages)
