"""Pipeline orchestrator -- runs extract -> validate -> critic -> structure,
metering every LLM call (tokens, cost, latency) into an IngestTrace.

Depends ONLY on the LLMPort protocol -- never on a concrete provider.

Bounded-loop guarantees:
- Validate repairs: at most `max_repairs` repair calls per validate pass.
- Critic retry: if confidence < critic_threshold, at most
  `max_critic_retries` full re-extractions; the LAST result is accepted
  regardless, with prior verdicts preserved in KnowledgeDoc.meta.
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
    run_critic -- is metered with its own latency.
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
    ) -> _LLMResponse:
        start = time.perf_counter()
        response = self._inner.complete(prompt, system=system, json_schema=json_schema)
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
        the superseded verdict(s) are preserved in KnowledgeDoc.meta.
        """
        trace = IngestTrace()
        timed = _TimedLLM(self._llm)

        draft = self._extract_and_validate(timed, doc, trace, retry=False)
        critic = self._criticize(timed, doc, draft, trace, retry=False)

        prior_verdicts: list[CriticResult] = []
        while (
            critic.confidence < self._critic_threshold
            and len(prior_verdicts) < self._max_critic_retries
        ):
            prior_verdicts.append(critic)
            draft = self._extract_and_validate(timed, doc, trace, retry=True, feedback=critic)
            critic = self._criticize(timed, doc, draft, trace, retry=True)

        meta: dict[str, Any] = {}
        if prior_verdicts:
            meta["critic_retries"] = len(prior_verdicts)
            meta["first_critic"] = prior_verdicts[0].model_dump()

        return build_doc(doc, draft, critic, meta=meta), trace

    # ------------------------------------------------------------ internals

    def _extract_and_validate(
        self,
        timed: _TimedLLM,
        doc: RawDocument,
        trace: IngestTrace,
        *,
        retry: bool,
        feedback: CriticResult | None = None,
    ) -> KnowledgeDraft:
        extract_name = "extract_retry" if retry else "extract"
        repair_name = "repair_retry" if retry else "repair"
        try:
            raw, _response = run_extract(timed, doc, feedback=feedback)
        finally:
            self._drain_into(timed, trace, extract_name, extract_name)
        try:
            draft, _repairs = run_validate(timed, doc, raw, max_repairs=self._max_repairs)
        finally:
            self._drain_into(timed, trace, repair_name, repair_name)
        return draft

    def _criticize(
        self,
        timed: _TimedLLM,
        doc: RawDocument,
        draft: KnowledgeDraft,
        trace: IngestTrace,
        *,
        retry: bool,
    ) -> CriticResult:
        critic_name = "critic_retry" if retry else "critic"
        repair_name = "repair_retry" if retry else "repair"
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
        failed stages still leave their calls in the trace-in-progress).
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
