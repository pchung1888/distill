"""Validate stage -- parse LLM output into a schema, repair-reprompting on failure.

`parse_or_repair` is the shared bounded repair loop: try to parse, and on a
validation error re-prompt the LLM (marker "TASK: REPAIR") with the invalid
output + exact error + schema, at most `max_repairs` times. Both the Validate
stage (KnowledgeDraft) and the Critic stage (CriticResult) reuse it.
"""

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from distill.llm.base import LLMPort, LLMResponse
from distill.models import KnowledgeDraft
from distill.pipeline.errors import PipelineError

M = TypeVar("M", bound=BaseModel)

REPAIR_SYSTEM = (
    "You are a strict JSON repair assistant. You fix malformed or "
    "schema-violating JSON and return only the corrected JSON object."
)

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


def _balanced_object_at(text: str, start: int) -> str | None:
    """Return the balanced {...} substring starting at `start`, or None.

    Tracks JSON string state and backslash escapes so braces inside string
    values do not confuse the depth count.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _first_json_object(text: str) -> str | None:
    """Scan left-to-right for the first balanced top-level {...} that is
    valid JSON (a balanced-but-invalid stray like "{placeholder}" is skipped).
    """
    search_from = 0
    while True:
        start = text.find("{", search_from)
        if start == -1:
            return None
        candidate = _balanced_object_at(text, start)
        if candidate is not None:
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                return candidate
        search_from = start + 1


def strip_json_wrappers(raw: str) -> str:
    """Strip common LLM chrome around a JSON object, in order:

    1. If a fenced ```json ... ``` (or bare ```) block is present, take its
       content.
    2. Balanced-brace scan (string/escape aware) for the first balanced
       top-level object that is valid JSON -- ignoring stray braces in the
       surrounding prose before or after it.
    3. Fall back to the (fence-stripped) raw text unchanged.
    """
    text = raw.strip()
    fence_start = text.find("```")
    if fence_start != -1:
        after = fence_start + 3
        fence_end = text.find("```", after)
        if fence_end != -1:
            content = text[after:fence_end]
            if content.startswith("json"):
                content = content[4:]
            text = content.strip()
    candidate = _first_json_object(text)
    if candidate is not None:
        return candidate
    return text


def parse_model(raw: str, model_cls: type[M]) -> M:
    """Strip wrappers then strictly validate; raises pydantic.ValidationError
    upward on failure (pydantic v2 raises ValidationError for JSON syntax
    errors too, so no separate json.JSONDecodeError path exists).
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

    Repair calls pin temperature=0.0 (deterministic-by-default; see
    extract.py). Returns (parsed model, list of repair responses made).
    Raises PipelineError carrying `stage` and the last parse error once the
    repair budget is exhausted.
    """
    try:
        return parse_model(raw, model_cls), []
    except ValidationError as err:
        last_error: Exception = err

    repairs: list[LLMResponse] = []
    current = raw
    for _ in range(max_repairs):
        prompt = build_repair_prompt(model_cls, current, last_error)
        response = llm.complete(
            prompt,
            system=REPAIR_SYSTEM,
            json_schema=model_cls.model_json_schema(),
            temperature=0.0,
        )
        repairs.append(response)
        current = response.text
        try:
            return parse_model(current, model_cls), repairs
        except ValidationError as err:
            last_error = err
    raise PipelineError(stage, cause=last_error) from last_error


def run_validate(
    llm: LLMPort,
    raw: str,
    max_repairs: int = 1,
) -> tuple[KnowledgeDraft, list[LLMResponse]]:
    """Validate raw extract output, repairing at most `max_repairs` times."""
    return parse_or_repair(llm, KnowledgeDraft, raw, stage="validate", max_repairs=max_repairs)
