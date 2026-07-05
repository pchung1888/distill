"""Extract stage -- turn a RawDocument into a KnowledgeDraft JSON candidate.

Builds the extraction prompt (marker line "TASK: EXTRACT", strict-JSON
instructions, the KnowledgeDraft JSON schema, and the -- possibly truncated --
source text) and makes the first LLM call. The role/persona text travels in
the `system` parameter; the TASK marker stays in the user prompt. Parsing/
validation of the raw response is the Validate stage's job (validate.py).

Determinism: every pipeline LLM call (extract, repair, critic) pins
temperature=0.0. Extraction and critique are judgment tasks where run-to-run
sampling variance only adds noise -- identical input should produce an
identical draft and an identical verdict, which also keeps eval scores and
the critic's retry decision reproducible.
"""

import json

from distill.llm.base import LLMPort, LLMResponse
from distill.models import CriticResult, KnowledgeDraft, RawDocument

# Very long sources are truncated to keep prompts bounded; the prompt notes
# the truncation so the LLM (and the critic) know the source is partial.
MAX_SOURCE_CHARS = 24000

EXTRACT_SYSTEM = (
    "You are a careful knowledge extractor. You read source documents and "
    "distill them into faithful structured summaries, never inventing facts."
)

EXTRACT_PROMPT_TEMPLATE = """\
TASK: EXTRACT

Read the SOURCE below and produce
STRICT JSON -- exactly one JSON object, no prose, no markdown fences --
matching this JSON schema:

{schema}

Rules:
- Use ONLY facts stated in the source. Never invent, embellish, or assume.
- summary: a faithful condensation of the source, at most about 150 words.
- key_points: 3 to 10 of the most important points, each one sentence.
- entities: notable people / organizations / concepts, each with "name" and
  "type" (optional integer "mentions" = how often it appears).
- topics: 1 to 5 short topic labels.
{feedback_block}
SOURCE{truncation_note}:
{source}
"""

FEEDBACK_BLOCK_TEMPLATE = """
PREVIOUS ATTEMPT ISSUES:
A prior extraction of this source was judged unfaithful or incomplete by a
critic. Fix ALL of the problems below in this attempt.
Unsupported or wrong claims to remove or correct:
{issues}
Important source points that were missing and must be covered:
{missing}
"""


def truncate_source(text: str) -> tuple[str, str]:
    """Return (possibly truncated source, prompt note about the truncation)."""
    if len(text) <= MAX_SOURCE_CHARS:
        return text, ""
    note = f" (truncated to the first {MAX_SOURCE_CHARS} of {len(text)} characters)"
    return text[:MAX_SOURCE_CHARS], note


def build_extract_prompt(doc: RawDocument, feedback: CriticResult | None = None) -> str:
    """Render the extract prompt; `feedback` (a prior critic verdict) triggers
    the PREVIOUS ATTEMPT ISSUES block used by the orchestrator's retry.
    """
    source, note = truncate_source(doc.text)
    schema = json.dumps(KnowledgeDraft.model_json_schema(), indent=2)
    feedback_block = ""
    if feedback is not None:
        issues = "\n".join(f"- {item}" for item in feedback.issues) or "- (none listed)"
        missing = "\n".join(f"- {item}" for item in feedback.missing_points) or "- (none listed)"
        feedback_block = FEEDBACK_BLOCK_TEMPLATE.format(issues=issues, missing=missing)
    return EXTRACT_PROMPT_TEMPLATE.format(
        schema=schema,
        feedback_block=feedback_block,
        truncation_note=note,
        source=source,
    )


def run_extract(
    llm: LLMPort,
    doc: RawDocument,
    feedback: CriticResult | None = None,
) -> tuple[str, LLMResponse]:
    """Make the extract LLM call; return (raw response text, full response)."""
    prompt = build_extract_prompt(doc, feedback=feedback)
    response = llm.complete(
        prompt,
        system=EXTRACT_SYSTEM,
        json_schema=KnowledgeDraft.model_json_schema(),
        temperature=0.0,
    )
    return response.text, response
