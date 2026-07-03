"""Pipeline orchestrator -- runs extract -> validate -> critic -> structure,
metering every LLM call (tokens, cost, latency) into an IngestTrace.

Depends ONLY on the LLMPort protocol -- never on a concrete provider.

Bounded-loop guarantees:
- Validate repairs: at most `max_repairs` repair calls per validate pass.
- Critic retry: if confidence < critic_threshold, at most
  `max_critic_retries` full re-extractions; the LAST result is accepted
  regardless, with ALL prior verdicts preserved in KnowledgeDoc.meta
  under meta["prior_critics"].

Worst case with the defaults (max_repairs=1, max_critic_retries=1) is
8 LLM calls: extract + validate_repair + critic + critic_repair on the
first pass, then the same four again on the single retry pass.

Failure honesty: any PipelineError leaving run() carries the partial
IngestTrace of every LLM call made before the failure (including a
zero-token entry for a provider call that raised), so failed runs are
still metered. Unexpected provider exceptions are wrapped into
PipelineError(stage=<current stage>) instead of escaping raw.
"""

import time
from typing import Any

from distill.llm.base import LLMPort
from distill.llm.base import LLMResponse as _LLMResponse
from distill.models import (
    CriticResult,
    IngestTrace,
    KnowledgeDoc,
    KnowledgeDraft,
    RawDocument,
    StageTrace,
)
from distill.pipeline.critic import run_critic
from distill.pipeline.errors import PipelineError
from distill.pipeline.extract import run_extract
from distill.pipeline.structure import build_doc
from distill.pipeline.validate import run_validate

__all__ = ["Pipeline", "PipelineError"]


class _TimedLLM:
    """LLMPort wrapper recording (response, latency_ms) for every call.

    The stage functions keep their simple signatures; the orchestrator
    drains this recorder after each stage to build StageTraces, so every
    LLM call -- including repair calls made inside run_validate /
    run_critic -- is metered with its own latency. A call whose inner
    provider RAISES is still recorded (tokens 0/0, cost 0.0, measured
    latency) before the exception propagates, so failed calls appear in
    the partial trace.
    """

    def __init__(self, inner: LLMPort) -> None:
        self._inner = inner
        self._calls: list[tuple[_LLMResponse, int]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> _LLMResponse:
        start = time.perf_counter()
        try:
            response = self._inner.complete(
                prompt,
                system=system,
                json_schema=json_schema,
                temperature=temperature,
            )
        except Exception:
            latency_ms = int((time.perf_counter() - start) * 1000)
            failed = _LLMResponse(text="", tokens_in=0, tokens_out=0, cost_usd=0.0)
            self._calls.append((failed, latency_ms))
            raise
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._calls.append((response, latency_ms))
        return response

    def drain(self) -> list[tuple[_LLMResponse, int]]:
        calls, self._calls = self._calls, []
        return calls


class Pipeline:
    """Conductor for the four-stage ingestion pipeline."""

    def __init__(
        self,
        llm: LLMPort,
        critic_threshold: float = 0.7,
        max_repairs: int = 1,
        max_critic_retries: int = 1,
    ) -> None:
        self._llm = llm
        self._critic_threshold = critic_threshold
        self._max_repairs = max_repairs
        self._max_critic_retries = max_critic_retries

    def run(self, doc: RawDocument) -> tuple[KnowledgeDoc, IngestTrace]:
        """Ingest one document; return the final doc + full metering trace.

        A critic confidence below `critic_threshold` triggers a bounded
        re-extraction (augmented with the critic's issues/missing points);
        after the retry budget the latest result is accepted regardless and
        ALL superseded verdicts are preserved in order in
        KnowledgeDoc.meta["prior_critics"].

        On failure, the raised PipelineError carries the partial trace of
        every LLM call made so far (`partial_trace`); non-PipelineError
        exceptions are wrapped into PipelineError(stage=<current stage>).
        """
        trace = IngestTrace()
        timed = _TimedLLM(self._llm)
        current_stage: list[str] = ["extract"]
        try:
            return self._run(timed, doc, trace, current_stage)
        except PipelineError as err:
            err.partial_trace = trace
            raise
        except Exception as exc:
            raise PipelineError(current_stage[0], cause=exc, partial_trace=trace) from exc

    # ------------------------------------------------------------ internals

    def _run(
        self,
        timed: _TimedLLM,
        doc: RawDocument,
        trace: IngestTrace,
        current_stage: list[str],
    ) -> tuple[KnowledgeDoc, IngestTrace]:
        draft = self._extract_and_validate(timed, doc, trace, current_stage, retry=False)
        critic = self._criticize(timed, doc, draft, trace, current_stage, retry=False)

        prior_verdicts: list[CriticResult] = []
        while (
            critic.confidence < self._critic_threshold
            and len(prior_verdicts) < self._max_critic_retries
        ):
            prior_verdicts.append(critic)
            draft = self._extract_and_validate(
                timed, doc, trace, current_stage, retry=True, feedback=critic
            )
            critic = self._criticize(timed, doc, draft, trace, current_stage, retry=True)

        meta: dict[str, Any] = {}
        if prior_verdicts:
            meta["critic_retries"] = len(prior_verdicts)
            meta["prior_critics"] = [v.model_dump() for v in prior_verdicts]

        return build_doc(doc, draft, critic, meta=meta), trace

    def _extract_and_validate(
        self,
        timed: _TimedLLM,
        doc: RawDocument,
        trace: IngestTrace,
        current_stage: list[str],
        *,
        retry: bool,
        feedback: CriticResult | None = None,
    ) -> KnowledgeDraft:
        extract_name = "extract_retry" if retry else "extract"
        repair_name = "validate_repair_retry" if retry else "validate_repair"
        current_stage[0] = extract_name
        try:
            raw, _response = run_extract(timed, doc, feedback=feedback)
        finally:
            self._drain_into(timed, trace, extract_name, extract_name)
        current_stage[0] = "validate_retry" if retry else "validate"
        try:
            draft, _repairs = run_validate(timed, raw, max_repairs=self._max_repairs)
        finally:
            self._drain_into(timed, trace, repair_name, repair_name)
        return draft

    def _criticize(
        self,
        timed: _TimedLLM,
        doc: RawDocument,
        draft: KnowledgeDraft,
        trace: IngestTrace,
        current_stage: list[str],
        *,
        retry: bool,
    ) -> CriticResult:
        critic_name = "critic_retry" if retry else "critic"
        repair_name = "critic_repair_retry" if retry else "critic_repair"
        current_stage[0] = critic_name
        try:
            result, _response = run_critic(timed, doc, draft, max_repairs=self._max_repairs)
        finally:
            self._drain_into(timed, trace, critic_name, repair_name)
        return result

    @staticmethod
    def _drain_into(
        timed: _TimedLLM,
        trace: IngestTrace,
        first_name: str,
        rest_name: str,
    ) -> None:
        """Turn recorded LLM calls into StageTraces (drained in try/finally so
        failed stages still land in the trace attached to PipelineError).
        """
        for index, (response, latency_ms) in enumerate(timed.drain()):
            trace.stages.append(
                StageTrace(
                    name=first_name if index == 0 else rest_name,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    cost_usd=response.cost_usd,
                    latency_ms=latency_ms,
                )
            )
