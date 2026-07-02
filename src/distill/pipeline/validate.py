"""Validate stage -- parse LLM output into a schema, repair-reprompting on failure.

`parse_or_repair` is the shared bounded repair loop: try to parse, and on a
JSON/validation error re-prompt the LLM (marker "TASK: REPAIR") with the
invalid output + exact error + schema, at most `max_repairs` times. Both the
Validate stage (KnowledgeDraft) and the Critic stage (CriticResult) reuse it.
"""

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from distill.llm.base import LLMPort, LLMResponse
from distill.models import KnowledgeDraft, RawDocument
from distill.pipeline.errors import PipelineError

M = TypeVar("M", bound=BaseModel)

REPAIR_PROMPT_TEMPLATE = """\
TASK: REPAIR

A previous response was supposed to be STRICT JSON matching this schema:

{schema}

but it failed to parse or validate with this exact error:

{error}

The invalid response was:

{invalid}

Return the corrected response now: exactly one JSON object matching the
schema, with no prose, no markdown fences, and no commentary.
"""


def strip_json_wrappers(raw: str) -> str:
    """Strip common LLM chrome: ```json fences and leading/trailing prose,
    by slicing from the first '{' to the last '}' (which subsumes fences).
    Returns the stripped text unchanged if no brace pair is found.
    """
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_model(raw: str, model_cls: type[M]) -> M:
    """Strip wrappers then strictly validate; raises pydantic.ValidationError
    (or json.JSONDecodeError from json.loads-style paths) upward on failure.
    """
    return model_cls.model_validate_json(strip_json_wrappers(raw))


def parse_draft(raw: str) -> KnowledgeDraft:
    """Parse raw extract output into a KnowledgeDraft (no repair)."""
    return parse_model(raw, KnowledgeDraft)


def build_repair_prompt(model_cls: type[BaseModel], invalid: str, error: Exception) -> str:
    """Render the repair prompt for one failed parse attempt."""
    schema = json.dumps(model_cls.model_json_schema(), indent=2)
    return REPAIR_PROMPT_TEMPLATE.format(schema=schema, error=error, invalid=invalid)


def parse_or_repair(
    llm: LLMPort,
    model_cls: type[M],
    raw: str,
    *,
    stage: str,
    max_repairs: int = 1,
) -> tuple[M, list[LLMResponse]]:
    """Bounded parse-repair loop shared by Validate and Critic.

    Returns (parsed model, list of repair responses made). Raises
    PipelineError carrying `stage` and the last parse error once the repair
    budget is exhausted.
    """
    try:
        return parse_model(raw, model_cls), []
    except (ValidationError, json.JSONDecodeError) as err:
        last_error: Exception = err

    repairs: list[LLMResponse] = []
    current = raw
    for _ in range(max_repairs):
        prompt = build_repair_prompt(model_cls, current, last_error)
        response = llm.complete(prompt, json_schema=model_cls.model_json_schema())
        repairs.append(response)
        current = response.text
        try:
            return parse_model(current, model_cls), repairs
        except (ValidationError, json.JSONDecodeError) as err:
            last_error = err
    raise PipelineError(stage, cause=last_error) from last_error


def run_validate(
    llm: LLMPort,
    doc: RawDocument,
    raw: str,
    max_repairs: int = 1,
) -> tuple[KnowledgeDraft, list[LLMResponse]]:
    """Validate raw extract output, repairing at most `max_repairs` times.

    `doc` is accepted for stage-interface symmetry (and so a future repair
    prompt could cite the source); the repair prompt does not need it today.
    """
    del doc  # interface symmetry only; see docstring
    return parse_or_repair(llm, KnowledgeDraft, raw, stage="validate", max_repairs=max_repairs)
