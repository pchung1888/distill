# distill

A provider-agnostic agentic knowledge-ingestion service. It turns a source
(URL / YouTube video / PDF) into verified, structured knowledge through an
extract -> validate -> critic -> structure pipeline, where an LLM critic
scores the draft for faithfulness against the source before anything is
emitted.

<!-- badges: add CI / coverage badges here once the repo is public -->

Live URL: (pending deploy)

## Architecture

```
                +------------------ SourcePort (protocol) ------------------+
  input ref --> |  URLSource | YouTubeSource | PDFSource                    |
                |  (SSRF guard, 20 MB download cap, redirect re-validation) |
                +----------------------------+-----------------------------+
                                             |  produces
                                             v
                                   RawDocument(text, metadata)
                                             |
   +-------------------------- Pipeline (orchestrated, token-metered) ------+
   |  1. Extract    LLMPort call: RawDocument -> draft JSON                 |
   |  2. Validate   Pydantic parse; on ValidationError -> one bounded       |
   |                repair-reprompt that feeds the error back to the LLM    |
   |  3. Critic     2nd LLMPort call: score draft vs source (faithfulness); |
   |                confidence below threshold -> one bounded retry         |
   |  4. Structure  validated + verified -> Markdown (frontmatter) + JSON   |
   +-----------------------------+------------------------------------------+
                                 v
       KnowledgeDoc + IngestTrace(per-stage tokens, cost_usd, latency_ms)
       (a failed run still returns its partial IngestTrace -- failures
        are metered too)

   LLMPort:  Mock (tests/CI) | Gemini | Anthropic | OpenAI | Ollama
   API:      FastAPI  POST /ingest {source_type, value} -> doc + trace + markdown
             GET /health
   Evals:    golden set + deterministic rubric + an LLM judge (separately
             metered; can run on an independent provider)
   Config:   env-driven (DISTILL_PROVIDER, DISTILL_MODEL, keys, thresholds)
```

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                          # install (creates .venv)
uv run pytest -q                                 # test suite (MockProvider, offline)
uv run uvicorn distill.api.app:app --reload      # run the API locally
```

With no env vars set, the service uses the deterministic mock provider --
no API key, no network cost:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source_type": "url", "value": "https://example.com/article"}'
```

`POST /ingest` returns `{doc, trace, markdown}`: the structured
`KnowledgeDoc`, the per-stage `IngestTrace` (tokens, cost, latency), and a
Markdown rendering with YAML frontmatter. Source fetch failures return 422;
LLM-stage failures return 502 with the failing stage and any partial trace.

### Environment variables

Mirrors `.env.example` (names and meaning only -- never commit values):

| Variable | Meaning |
|---|---|
| `DISTILL_PROVIDER` | LLM provider: `mock` (default), `gemini`, `anthropic`, `openai`, `ollama` |
| `DISTILL_MODEL` | Optional model override; blank uses the provider's default |
| `GEMINI_API_KEY` | API key for the Gemini provider (google-genai SDK) |
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider |
| `OPENAI_API_KEY` | API key for the OpenAI provider |
| `OLLAMA_BASE_URL` | Base URL of a local Ollama server (default `http://localhost:11434`) |
| `DISTILL_CRITIC_THRESHOLD` | Minimum critic confidence (0.0-1.0) before a doc is accepted without a retry; default 0.7 |

## Eval harness

The golden set under `evals/golden/` runs through the REAL pipeline (not a
shortcut path) and is scored by `evals/rubric.py`: deterministic checks
(schema validity, normalized key-fact presence, topics, key points) plus an
LLM-judge faithfulness check.

```bash
uv run python evals/run_evals.py --provider mock
```

What mock mode honestly validates: each golden case ships a hand-authored
`mock_response.json` written to be faithful to that case's checked-in source
text, so the mock run is fully deterministic and offline. It validates the
HARNESS and RUBRIC mechanics (case loading, real pipeline wiring, fact
matching, thresholds, metering, exit codes) -- it does NOT measure real
extraction quality. Only live-provider runs do that. CI runs the mock eval
as a merge gate for exactly this reason: it proves the measurement
instrument works, cheaply and deterministically, on every push.

Sample output (actual mock run):

```
provider: mock  judge provider: mock  cases: 3
NOTE: mock provider -- drafts are hand-authored; this validates harness+rubric mechanics, NOT extraction quality.
case         schema  facts  topics  points  crit   judge  pass
--------------------------------------------------------------
gettysburg   ok      1.00   ok      ok      0.88   0.90   PASS
tech_talk    ok      1.00   ok      ok      0.95   0.93   PASS
water_cycle  ok      1.00   ok      ok      0.92   0.90   PASS
--------------------------------------------------------------
pass rate: 3/3 = 100.0% (floor: 99.0%)
mean pipeline critic confidence: 0.9167
mean judge confidence: 0.9100
judge confidence variance (pvariance, n=3; small-n -- indicative only): 0.000200
total tokens: in=10675 out=1283 total=11958
total cost: $0.000000
```

Live runs use the same golden set and rubric against real LLM output. By
default the rubric's judge runs on the SAME provider as the pipeline --
that is a same-model self-consistency re-check with an independent prompt,
not an independent judge. For a genuinely independent judge, pass a
different provider:

```bash
uv run python evals/run_evals.py --provider gemini --judge-provider anthropic
```

## Providers

Selected by `DISTILL_PROVIDER`; the factory in `src/distill/llm/__init__.py`
is the only place that maps a name to a concrete class.

| Provider | Default model | Notes |
|---|---|---|
| `mock` | (scripted) | Deterministic, offline, zero cost; used by all tests and CI |
| `gemini` | `gemini-2.5-flash` | Default live provider; `google-genai` SDK (extra: `gemini`) |
| `anthropic` | `claude-haiku-4-5` | `anthropic` SDK (extra: `anthropic`) |
| `openai` | `gpt-4o-mini` | `openai` SDK (extra: `openai`) |
| `ollama` | `llama3.2` | Local HTTP server; no key, zero marginal cost |

Per-model pricing used by the cost meter (`src/distill/llm/base.py`), in USD
per 1M tokens. These values are indicative pricing hard-coded at build time,
not fetched live -- verify against the provider's current price list before
relying on them:

| Model | Input / 1M | Output / 1M |
|---|---|---|
| `gemini-2.5-flash` | $0.30 | $2.50 |
| `gemini-2.5-pro` | $1.25 | $10.00 |
| `claude-sonnet-4-5` | $3.00 | $15.00 |
| `claude-haiku-4-5` | $1.00 | $5.00 |
| `gpt-4o-mini` | $0.15 | $0.60 |
| `gpt-4o` | $2.50 | $10.00 |
| `mock` / `ollama` | $0.00 | $0.00 |

Unknown model names meter as $0.00 rather than crashing, so a new model
works immediately and its cost line is visibly zero until the table is
updated.

## Design rationale

- **Ports and adapters.** The pipeline depends on two Protocols -- `LLMPort`
  (one `complete()` method) and `SourcePort` (one `fetch()` method) -- never
  on a concrete SDK or scraper. Swapping providers is a config change;
  tests inject `MockProvider` and run offline with zero cost.
- **A critic stage, not just extraction.** LLM extraction output is a claim,
  not a fact. The critic is a second LLM call that scores the draft against
  the source for faithfulness; below the confidence threshold the pipeline
  makes exactly one bounded retry, then either emits the verdict-stamped doc
  or fails honestly. Bounded means no unbounded self-correction loops
  burning tokens.
- **Pydantic repair loop.** Every LLM output is parsed against a strict
  schema. On `ValidationError` the pipeline re-prompts once, feeding the
  validation error back to the model, so malformed JSON gets one structured
  chance to fix itself instead of crashing the request.
- **IngestTrace token economics.** Every stage (including eval judge calls)
  is metered: tokens in/out, USD cost, latency. Failed runs return their
  partial trace inside the 502 body, so spend on failures is visible, not
  lost. This is what makes cross-provider cost comparison a measurement
  rather than a guess.
- **Eval-driven development.** The eval harness is not an afterthought; CI
  runs the mock eval as a gate on every push, and the harness distinguishes
  eval FAILURE (exit 1: output missed the rubric) from harness
  misconfiguration (exit 2: the run cannot honestly be scored at all).
- **Security hardening.** The URL fetcher enforces an SSRF guard (http(s)
  only, every resolved IP must be public), re-validates every redirect hop
  manually (max 5) so a public host cannot bounce a request to an internal
  address, and caps downloads at 20 MB enforced during streaming (a lying
  Content-Length header cannot bypass it). PDF local-file access is
  disabled for all API-resolved sources (`allow_local=False`).

## Docker

```bash
docker build -t distill .
docker run --rm -p 8080:8080 distill
```

Multi-stage build: dependencies resolve from the committed `uv.lock`
(`--frozen`), the app runs as a non-root user, and the runtime image
contains only the venv -- no tests, evals, or docs. The server binds
`0.0.0.0:${PORT:-8080}`.

## Deploy (HUMAN-GATED)

Deployment requires real GCP credentials and a live API key; it is
deliberately excluded from the autonomous build. A human runs:

```bash
docker build -t distill .

gcloud run deploy distill \
  --source . \
  --region <region> \
  --set-env-vars DISTILL_PROVIDER=gemini \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

Notes:

- Prefer Secret Manager (`--set-secrets`) over a plain
  `--set-env-vars GEMINI_API_KEY=...` -- plain env vars are visible in the
  service config to anyone with viewer access.
- Do not set `PORT`; Cloud Run injects it and the container honors it.
- After deploy: verify `GET /health` returns the configured provider, run a
  live `POST /ingest`, and paste the service URL into the "Live URL" line
  at the top of this README.

## Project structure

```
src/distill/
  models.py          Pydantic data shapes (RawDocument, KnowledgeDraft,
                     CriticResult, KnowledgeDoc, IngestTrace)
  config.py          env-driven runtime Config
  sources/           SourcePort + URL / YouTube / PDF adapters + SSRF guards
  llm/               LLMPort, provider factory, cost table, 5 providers
  pipeline/          extract / validate / critic / structure / orchestrator
  api/               FastAPI app (POST /ingest, GET /health)
evals/               golden set + rubric + run_evals.py (CI gate in mock mode)
tests/               offline test suite (MockProvider, local fixtures)
```

## License

License: TBD -- to be decided by the repository owner before the repo is
made public.
