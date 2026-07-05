"""Critic stage -- second LLM call judging the draft's faithfulness to the source.

The critic sees ONLY the source and the draft, and must score how well the
draft is supported by the source (confidence 0-1, calibrated). Parse failures
reuse the shared bounded repair loop from validate.py.

Prompt-injection threat model: the source text is embedded VERBATIM in both
the extract and critic prompts, so a hostile source could carry instructions
aimed at the critic ("rate this draft 1.0", fake TASK markers, etc.). The v1
trust boundary is public/synthetic sources chosen by the operator (HANDOFF
Section 9); no sanitization or delimiter-escaping is attempted. Do not point
the pipeline at adversarial or user-submitted sources without adding one.
"""

import json

from distill.llm.base import LLMPort, LLMResponse
from distill.models import CriticResult, KnowledgeDraft, RawDocument
from distill.pipeline.extract import truncate_source
from distill.pipeline.validate import parse_or_repair

CRITIC_SYSTEM = (
    "You are a strict faithfulness critic. You judge whether a draft is "
    "supported by its source document, and nothing else."
)

CRITIC_PROMPT_TEMPLATE = """\
TASK: CRITIC

Judge the DRAFT below ONLY against the
SOURCE below. Outside knowledge must not rescue a claim the source does not
support. Produce STRICT JSON -- exactly one JSON object, no prose, no
markdown fences -- matching this JSON schema:

{schema}

Field guidance:
- confidence (0.0 to 1.0): calibrated faithfulness score. 1.0 means every
  claim in the draft is directly supported by the source; below 0.5 means
  the draft contains major unsupported claims.
- faithful: true only if the draft has no material unsupported claims.
- issues: one entry per unsupported or wrong claim, naming that claim.
- missing_points: important points in the source that the draft omits.

SOURCE{truncation_note}:
{source}

DRAFT (JSON):
{draft}
"""


def build_critic_prompt(doc: RawDocument, draft: KnowledgeDraft) -> str:
    """Render the critic prompt: schema + (truncated) source + draft JSON."""
    source, note = truncate_source(doc.text)
    schema = json.dumps(CriticResult.model_json_schema(), indent=2)
    return CRITIC_PROMPT_TEMPLATE.format(
        schema=schema,
        truncation_note=note,
        source=source,
        draft=draft.model_dump_json(indent=2),
    )


def run_critic(
    llm: LLMPort,
    doc: RawDocument,
    draft: KnowledgeDraft,
    max_repairs: int = 1,
) -> tuple[CriticResult, LLMResponse]:
    """Make the critic LLM call and parse its verdict.

    A parse failure goes through the shared repair loop (at most
    `max_repairs` repair calls); if still invalid, PipelineError('critic')
    propagates. Returns (verdict, the initial critic response).
    """
    prompt = build_critic_prompt(doc, draft)
    response = llm.complete(
        prompt,
        system=CRITIC_SYSTEM,
        json_schema=CriticResult.model_json_schema(),
        temperature=0.0,
    )
    result, _repairs = parse_or_repair(
        llm, CriticResult, response.text, stage="critic", max_repairs=max_repairs
    )
    return result, response
