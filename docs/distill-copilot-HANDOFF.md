# HANDOFF -- distill BACKEND: add the "Support Copilot" capability (/ask + /metrics)

> CONTINUATION spec for the EXISTING repo `D:\playground\distill\distill` (Python/FastAPI,
> phases 0-6 green, Phase 7 deploy human-gated). This ADDS a cited, confidence-gated
> support-copilot capability that ESCALATES instead of guessing, plus a metrics store.
> It is the public, clean-room rebuild of a proprietary support system
> (see `ai-support-copilot-case-study.md`). Copy this into the repo as
> `docs/copilot-HANDOFF.md`, then drive with /personal-goal + /personal-workflow.
> Pair repo: `distill-demo` (the frontend) builds the UI against the contract in Section 2.

---

## 0. HOW TO DRIVE (existing repo -- EXTEND, do not scaffold)

```
# in D:\playground\distill\distill, on a new branch
git checkout -b feat/support-copilot
# copy this file to docs/copilot-HANDOFF.md
/personal-goal distill-copilot
/personal-workflow --plan docs/copilot-HANDOFF.md   # or /personal-loop --resume distill-copilot
```

Bootstrap prompt to paste in the repo:
```
This is the existing `distill` FastAPI service. READ the existing code first
(src/distill/**, tests/**, CLAUDE.md) and REUSE its patterns -- do NOT scaffold or
duplicate. Then read docs/copilot-HANDOFF.md and ADD the "support copilot" capability
per its phases. Reuse: LLMPort + get_provider + MockProvider (add a TASK: ANSWER
marker), parse_or_repair (pipeline/validate.py), run_critic + CriticResult, the
Config.critic_threshold, the RateLimiter append/locking pattern (api/ratelimit.py),
and the create_app factory + dependency-seam + /ingest handler shape (api/app.py).
Advance a phase only when its VERIFY command exits 0. MockProvider for all tests (no
network). PUBLIC / SYNTHETIC data only -- no client data, do not copy any employer
cs-* file. Stop at the human-gated deploy.
```

## 1. Build profile

- **Advance rule:** phase DONE only when its VERIFY exits 0; log evidence to the beacon.
- **On failure:** fix + re-run same command x3, then pause.
- **Model routing:** fable for A (retrieval), B (models/compose), D (routes); ESCALATE /
  `/personal-critic-gate` for C (the confidence gate + escalation logic -- the load-bearing
  behavior).
- **Human-gated:** `git push`, `gcloud`/deploy, making public. Also: the retriever must index
  only a PUBLIC docs corpus (permissive license) -- no client/employer content.
- **Reuse-first:** every phase below names what to reuse; new code must match existing style
  (ruff line-length 100, Pydantic v2, ports-and-adapters, `TASK:`-marker prompts).

## 2. SHARED API CONTRACT (backend implements; distill-demo consumes -- keep in sync)

```
POST /ask
  request : { "question": str, "provider"?: str }
  response: {
    "escalated": bool,                # true => assistant refused to answer
    "answer": str | null,             # null when escalated
    "citations": Citation[],          # [] when escalated
    "confidence": float,              # 0..1 (critic score)
    "confidence_label": "EXTRACTED"|"INFERRED"|"BLANK",  # derived (see Phase C)
    "faithful": bool,
    "handoff": str | null,            # paste-ready escalation summary when escalated
    "trace": IngestTrace              # reuse existing IngestTrace shape
  }
Citation = { "source_title": str, "section"?: str, "url": str, "snippet"?: str }

GET /metrics
  response: { "rows": MetricRow[] }   # newest-first; frontend aggregates the tiles
MetricRow = {
  "id": str, "ts": str(ISO-8601), "question": str,
  "confidence": float, "confidence_label": "EXTRACTED"|"INFERRED"|"BLANK",
  "escalated": bool, "cited": bool,
  "retrieval_hit": bool, "retrieval_retry": bool, "caselaw_searched": bool,
  "citations": str[], "latency_ms": int, "cost_usd": float
}
```
Confidence-label mapping (Phase C): `escalated` -> BLANK; else `faithful && confidence>=threshold`
-> EXTRACTED; else -> INFERRED.

## 3. Phases

### Phase A -- Retrieval over a PUBLIC corpus (fable)
- BUILD: `src/distill/retrieval/` = `base.py` (`RetrieverPort` Protocol: `search(query, k)->list[Passage]`),
  `corpus.py` (`Passage` model: text + source_title + url + section), `index.py` (zero-dep TF-IDF
  over pre-chunked docs). `scripts/build_corpus.py` chunks a chosen PUBLIC docs set into
  `data/corpus/`. Model `RetrieverPort` on the existing `SourcePort` ports pattern (sources/base.py).
- REUSE: `SourcePort` shape as template; `RawDocument`/bounded-context helpers.
- VERIFY: `uv run pytest tests/test_retrieval.py -q`
- DONE-WHEN: green on a fixture corpus (no network in tests); `search()` returns ranked Passages.

### Phase B -- Answer + citations model + compose stage (fable)
- BUILD: in `models.py` add `Citation` and `AnswerDraft{answer: str, citations: list[Citation]}`.
  In `pipeline/answer.py` add `run_answer(llm, question, passages) -> (AnswerDraft, LLMResponse)`
  mirroring `pipeline/extract.py` (system text, `TASK: ANSWER` marker, temperature 0.0,
  `json_schema=AnswerDraft.model_json_schema()`, parse via `parse_or_repair`). Add the
  `TASK: ANSWER` canned response to `MockProvider`.
- REUSE: `extract.py` prompt style, `parse_or_repair`/`strip_json_wrappers` (validate.py), LLMPort.
- VERIFY: `uv run pytest tests/test_answer.py -q`
- DONE-WHEN: green; compose returns an answer citing >=1 passage; malformed-output fixture
  exercises the repair loop.

### Phase C -- Confidence gate + ESCALATION (ESCALATE / critic-gate) -- THE load-bearing phase
- BUILD: `pipeline/ask.py` orchestrator `run_ask(llm, retriever, question, threshold)`:
  retrieve -> `run_answer` -> a grounded-ness critic (answer-vs-passages) -> if
  `confidence < threshold` (after at most one retry) return an ESCALATED result
  (answer=None, citations=[], build a `handoff` summary) instead of the answer. Compute
  `confidence_label` per Section 2. Meter every LLM call into an `IngestTrace`.
- REUSE: `run_critic` + `CriticResult` (adapt the prompt to judge answer-vs-passages; the
  `evals/rubric.py` `llm_judge_faithfulness` claim-grounding judge is the closest fit),
  `Config.critic_threshold` (DISTILL_CRITIC_THRESHOLD), the `_TimedLLM` metering, `IngestTrace`.
- NEW BEHAVIOR: the existing `Pipeline` never refuses -- you are adding the refuse/escalate
  branch. This is the whole point; get it right.
- VERIFY: `uv run pytest tests/test_ask_pipeline.py -q`
- DONE-WHEN: green; a low-confidence fixture returns `escalated=true`, `answer=None`, a non-empty
  `handoff`, and NO fabricated answer; a high-confidence fixture returns an answer + citations.
- GATE: fire `/personal-critic-gate` on the escalation logic + the grounded-ness prompt.

### Phase D -- Metrics store + /ask + /metrics routes (fable, critic-gate the contract)
- BUILD: `src/distill/metrics/logger.py` -- thread-safe JSONL appender writing a `MetricRow`
  per ask (reuse the `threading.Lock` + append discipline from `api/ratelimit.py`); path from
  config (`DISTILL_METRICS_PATH`, default `data/metrics.jsonl`). In `api/app.py` add
  `POST /ask` (reuse the `/ingest` handler shape: provider injection via a `get_ask_pipeline`
  dependency seam, SourceError->422 / PipelineError->502, optional `limiter.check_and_reserve`)
  and `GET /metrics` (reuse the `get_limiter` seam pattern -> add `get_metrics_store` so tests
  inject an in-memory store). `/ask` writes a MetricRow after answering.
- REUSE: `create_app` factory, dependency seams + `app.dependency_overrides` test pattern,
  error mapping, CORS (already allows the demo origin).
- VERIFY: `uv run pytest tests/test_ask_api.py tests/test_metrics.py -q && uv run ruff check .`
- DONE-WHEN: green; `POST /ask` returns the Section-2 shape AND appends a schema-valid MetricRow;
  `GET /metrics` returns `{rows:[...]}`; Swagger shows both at `/docs`.

### Phase E -- Full gate + docs (fable)
- VERIFY: `uv run pytest -q && uv run ruff check . && uv run python evals/run_evals.py --provider mock`
- BUILD: update README with an "Ask" quickstart + the /ask and /metrics contract; note the
  Cloud Run ephemeral-FS caveat for `data/metrics.jsonl` (use a mounted volume / GCS, or accept
  per-instance metrics for the demo).

### Deploy (HUMAN-GATED -- unchanged from Phase 7)
Needs GCP + Gemini key + Docker; human runs `docker build` + `gcloud run deploy`, then updates
the live URL. Metrics persistence: point `DISTILL_METRICS_PATH` at a persistent volume or GCS.

## 4. Definition of done
- [ ] Phases A-E VERIFY gates exit 0; evidence in the beacon.
- [ ] `POST /ask` returns cited answers; a hard/unknown question ESCALATES (answer=None) -- never fabricates.
- [ ] Every ask appends a schema-valid `MetricRow`; `GET /metrics` serves them.
- [ ] `uv run pytest -q` + ruff + mock evals all green in CI.
- [ ] Corpus is PUBLIC; zero client/employer data in repo or history.
- [ ] (HUMAN) deployed; live `/ask` works with Gemini; README updated.
