"""Deterministic eval rubric for the distill golden set.

What each check proves:

- schema_valid / key_fact_presence / topic_hit / min_key_points are pure,
  deterministic functions (no LLM) that score a pipeline's KnowledgeDoc
  against the hand-authored ``expected.json`` of a golden case.
- llm_judge_faithfulness is the one LLM-backed check. It routes through the
  LLMPort protocol with its OWN judge prompt (marker "TASK: JUDGE",
  claim-by-claim grounding framing -- deliberately DIFFERENT from the
  pipeline critic's prompt) and reuses only the pipeline's bounded
  parse_or_repair machinery. Independence honesty: when the judge runs on
  the SAME provider as the pipeline it is a same-model self-consistency
  re-check with an independent prompt, not an independent judge; only a
  different ``--judge-provider`` (see run_evals.py) is truly independent.

Honesty note on mock mode (see run_evals.py for the full statement): with
--provider mock the drafts come from per-case hand-authored scripts that
were written to be faithful to the checked-in sources. Mock mode therefore
validates the HARNESS and RUBRIC mechanics deterministically; it does NOT
measure real extraction quality. Live providers do.
"""

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from distill.llm.base import LLMPort, LLMResponse
from distill.models import CriticResult, KnowledgeDoc, KnowledgeDraft
from distill.pipeline.extract import truncate_source
from distill.pipeline.validate import parse_or_repair

# Default floors used by apply_rubric; run_evals.py exposes CLI overrides.
FACT_SCORE_FLOOR = 0.8
JUDGE_CONFIDENCE_FLOOR = 0.7

JUDGE_SYSTEM = (
    "You are a claim-verification judge for an extraction eval. You check, "
    "one claim at a time, whether each claim extracted from a document is "
    "directly supported by the source text, and report exactly what you find."
)

JUDGE_PROMPT_TEMPLATE = """\
TASK: JUDGE

You are verifying an extraction claim by claim. Below are a SOURCE text and
the CLAIMS a pipeline extracted from it (the document summary plus each key
point, numbered). For EACH numbered claim, decide whether it is DIRECTLY
supported by the source. Outside knowledge must not rescue a claim the
source does not state. Then produce STRICT JSON -- exactly one JSON object,
no prose, no markdown fences -- matching this JSON schema:

{schema}

Field guidance:
- confidence (0.0 to 1.0): the fraction of the numbered claims that are
  directly supported by the source (every claim supported -> 1.0; half of
  them -> 0.5).
- faithful: true only if EVERY numbered claim is directly supported.
- issues: one entry per unsupported claim, naming or quoting that claim.
- missing_points: important points in the source that the claims omit.

SOURCE{truncation_note}:
{source}

CLAIMS:
{claims}
"""


class ExpectedCase(BaseModel):
    """Hand-authored expectations for one golden case (expected.json)."""

    model_config = ConfigDict(extra="forbid")

    key_facts: list[str]
    min_key_points: int = Field(ge=0)
    topics_any: list[str]


class CaseResult(BaseModel):
    """Per-case rubric outcome; ``passed`` is the AND of every check."""

    model_config = ConfigDict(extra="forbid")

    case: str
    schema_ok: bool
    fact_score: float = Field(ge=0, le=1)
    missing_facts: list[str] = Field(default_factory=list)
    topic_ok: bool
    points_ok: bool
    critic_confidence: float = Field(ge=0, le=1, default=0.0)
    judge_confidence: float = Field(ge=0, le=1)
    judge_faithful: bool
    judge_issues: list[str] = Field(default_factory=list)
    judge_missing_points: list[str] = Field(default_factory=list)
    passed: bool
    error: str | None = None


def normalize(text: str) -> str:
    """Casefold and collapse all whitespace runs to single spaces."""
    return " ".join(text.split()).casefold()


def schema_valid(candidate: object) -> bool:
    """True if candidate parses as a KnowledgeDoc or KnowledgeDraft.

    Accepts a model instance, a mapping, or a JSON string. Never raises --
    any validation failure returns False.

    Honesty note: for a doc that came straight from Pipeline.run this check
    is tautological (the pipeline already validated it into a KnowledgeDoc,
    so it is always True on that path). It is a REAL check for unit tests
    and any future non-pipeline caller that hands apply_rubric an arbitrary
    object.
    """
    for model_cls in (KnowledgeDoc, KnowledgeDraft):
        try:
            if isinstance(candidate, str):
                model_cls.model_validate_json(candidate)
            elif isinstance(candidate, BaseModel):
                model_cls.model_validate(candidate.model_dump())
            else:
                model_cls.model_validate(candidate)
        except ValidationError:
            continue
        return True
    return False


def key_fact_presence(doc: KnowledgeDoc, expected: ExpectedCase) -> tuple[float, list[str]]:
    """Score how many expected key facts appear in the doc's summary + key points.

    Matching is a normalized (casefold, whitespace-collapsed) substring test
    against the joined summary + key_points text. Returns (score in 0-1,
    list of facts that were missing). An empty key_facts list scores 1.0.
    """
    if not expected.key_facts:
        return 1.0, []
    haystack = normalize(" ".join([doc.summary, *doc.key_points]))
    missing = [fact for fact in expected.key_facts if normalize(fact) not in haystack]
    score = (len(expected.key_facts) - len(missing)) / len(expected.key_facts)
    return score, missing


def topic_hit(doc: KnowledgeDoc, expected: ExpectedCase) -> bool:
    """True if at least one acceptable topic word appears among the doc topics.

    Normalized substring match, so expected "water" hits doc topic
    "water cycle". An empty topics_any list is vacuously True.
    """
    if not expected.topics_any:
        return True
    doc_topics = normalize(" | ".join(doc.topics))
    return any(normalize(topic) in doc_topics for topic in expected.topics_any)


def min_key_points(doc: KnowledgeDoc, expected: ExpectedCase) -> bool:
    """True if the doc carries at least the expected number of key points."""
    return len(doc.key_points) >= expected.min_key_points


def _claims_block(doc: KnowledgeDoc) -> str:
    """Number the doc's claims (summary first, then each key point)."""
    claims = [f"1. (summary) {doc.summary}"]
    claims.extend(
        f"{number}. {point}" for number, point in enumerate(doc.key_points, start=2)
    )
    return "\n".join(claims)


def build_judge_prompt(source_text: str, doc: KnowledgeDoc) -> str:
    """Render the judge prompt: schema + (truncated) source + numbered claims."""
    source, note = truncate_source(source_text)
    schema = json.dumps(CriticResult.model_json_schema(), indent=2)
    return JUDGE_PROMPT_TEMPLATE.format(
        schema=schema,
        truncation_note=note,
        source=source,
        claims=_claims_block(doc),
    )


def llm_judge_faithfulness(
    llm: LLMPort, source_text: str, doc: KnowledgeDoc, max_repairs: int = 1
) -> tuple[CriticResult, LLMResponse]:
    """LLM-judge check: verify each of the doc's claims against the source.

    Uses the judge's OWN prompt (marker "TASK: JUDGE", claim-by-claim
    grounding framing -- distinct from the pipeline critic prompt) and the
    pipeline's bounded parse_or_repair machinery (at most `max_repairs`
    repair calls; PipelineError('judge') if still unparseable).

    Independence honesty: on the same provider as the pipeline this is a
    same-model self-consistency re-check with an independent prompt; only a
    different judge provider is truly independent. Returns (verdict, the
    initial judge LLMResponse) so callers can meter the call.
    """
    prompt = build_judge_prompt(source_text, doc)
    response = llm.complete(
        prompt,
        system=JUDGE_SYSTEM,
        json_schema=CriticResult.model_json_schema(),
        temperature=0.0,
    )
    result, _repairs = parse_or_repair(
        llm, CriticResult, response.text, stage="judge", max_repairs=max_repairs
    )
    return result, response


def apply_rubric(
    case: str,
    doc: KnowledgeDoc,
    expected: ExpectedCase,
    judge: CriticResult,
    *,
    fact_floor: float = FACT_SCORE_FLOOR,
    judge_floor: float = JUDGE_CONFIDENCE_FLOOR,
    critic_confidence: float = 0.0,
) -> CaseResult:
    """Combine every check into one CaseResult.

    Pass rule: schema_valid AND fact_score >= fact_floor AND topic_hit AND
    min_key_points AND judge.faithful AND judge.confidence >= judge_floor.
    `critic_confidence` (the pipeline critic's own verdict on the doc) is
    recorded for reporting only; it does not gate the pass rule here -- the
    pipeline already gates on it via its retry threshold.
    """
    schema_ok = schema_valid(doc)
    fact_score, missing = key_fact_presence(doc, expected)
    topics_ok = topic_hit(doc, expected)
    points_ok = min_key_points(doc, expected)
    passed = (
        schema_ok
        and fact_score >= fact_floor
        and topics_ok
        and points_ok
        and judge.faithful
        and judge.confidence >= judge_floor
    )
    return CaseResult(
        case=case,
        schema_ok=schema_ok,
        fact_score=fact_score,
        missing_facts=missing,
        topic_ok=topics_ok,
        points_ok=points_ok,
        critic_confidence=critic_confidence,
        judge_confidence=judge.confidence,
        judge_faithful=judge.faithful,
        judge_issues=list(judge.issues),
        judge_missing_points=list(judge.missing_points),
        passed=passed,
    )


def failed_case(case: str, error: str) -> CaseResult:
    """A CaseResult for a case whose pipeline (or judge) raised.

    A raising case is a FAILED case, never a crashed harness: every check
    is recorded as failed and the error text is preserved for the report.
    """
    return CaseResult(
        case=case,
        schema_ok=False,
        fact_score=0.0,
        missing_facts=[],
        topic_ok=False,
        points_ok=False,
        critic_confidence=0.0,
        judge_confidence=0.0,
        judge_faithful=False,
        passed=False,
        error=error,
    )
