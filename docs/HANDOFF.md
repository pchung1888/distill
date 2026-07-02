# HANDOFF -- `distill`: Agentic Knowledge-Ingestion Pipeline (Autonomous One-Shot Build Spec)

> Portable, self-contained, AUTONOMOUS-BUILD spec. Drop this into a BLANK public
> GitHub repo as `docs/HANDOFF.md` and drive it with `/personal-goal` +
> `/personal-workflow`. Every phase has a machine-checkable VERIFY gate; the goal
> beacon is the audit-tracker that makes the build resumable across sessions and
> quota windows. Defaults marked "DEFAULT (change freely)" are recommendations.

---

## 0. HOW TO ONE-SHOT THIS (run these in the blank repo)

Prereqs on the machine (not in the repo): `git`, `python3.11+`, `uv`
(https://docs.astral.sh/uv/), and Claude Code with the `ping-personal` plugin
enabled (so `/personal-goal` and `/personal-workflow` are available). No API keys
needed for phases 0-6 (they use the MockProvider). GCP + a Gemini key are needed
only for Phase 7 (deploy), which is human-gated.

Step 1 -- initialize the git repo, this doc, and the project bootstrap files:
```
git init
mkdir -p docs .claude .github/workflows
# copy this file to docs/HANDOFF.md
# create CLAUDE.md and .claude/settings.json from Section 1.5 (contents below)
git add -A && git commit -m "chore: bootstrap repo + build spec"
```
`CLAUDE.md` gives the autonomous driver its project rules; `.claude/settings.json`
is a permission allow-list so it can run build/test/lint/commit WITHOUT stopping to
ask (that is what makes it "full auto"), while still fencing push/deploy. Both file
contents are in Section 1.5 -- create them before Step 3. The `ping-personal` skills
are user-installed via the plugin, so they work in this fresh repo with no per-repo
setup.

Step 2 -- create the audit-tracker beacon (multi-session crash recovery):
```
/personal-goal distill
```
This writes a `*-audit-tracker.md` beacon under `docs/`. It records the phase
table, last-known-good checkpoint, and an activity log. If a session dies or you
hit a quota wall mid-build, the beacon lets the next session resume from the last
green phase instead of restarting.

Step 3 -- drive the autonomous build:
```
/personal-workflow --plan docs/HANDOFF.md
```
`/personal-workflow` discovers host skills+agents, routes each phase per the
Autonomous Build Profile (Section 1), runs the phase's VERIFY gate, logs evidence
to the beacon, and only then advances. It PAUSES at the human-gated fence
(Phase 7) and at any Stay-Paused condition.

Step 3b -- OR drive it fully unattended across sessions (recommended for one-shot):
```
/personal-loop --resume distill
```
`/personal-loop` is the outer loop: it repeatedly advances the `distill` goal via
`/personal-workflow`, runs a per-tick critic gate, records progress to the beacon,
and keeps going across quota windows until the stop-condition (phases 0-6 all
green) is true OR it hits the human-gated fence (Phase 7 / push / deploy).
Fail-closed when unattended. Use this instead of Step 3 for true hands-off.

Step 4 -- resume after any interruption (new session):
```
/personal-workflow --resume distill      # attended
/personal-loop --resume distill          # unattended
```
(Or re-run Step 3; the beacon's last-completed-phase is the source of truth.)

---

## 1. AUTONOMOUS BUILD PROFILE (read before driving)

This is the contract that makes the build hands-off and safe.

**Advance rule (hard):** a phase is DONE only when its `VERIFY:` command exits 0
AND the output is logged to the beacon activity log. Never advance on an
unverified or hand-waved claim. Evidence before assertion.

**On VERIFY failure:** fix the cause and re-run the SAME verify command, up to
3 attempts. If still failing after 3, STOP, write the failure + last output to
the beacon, and pause for human. Do not skip the phase, do not fake a pass, do
not comment out the failing test.

**Model routing (DEFAULT -- token cost is not constrained, optimize for quality):**
| Phases | Character | Runner |
|---|---|---|
| 0, 1, 2, 3, 5 | Mechanical, well-specified (scaffold, models, adapters, API wiring) | fable (fast) is fine |
| 4, 6 | Judgment-heavy (pipeline/critic logic, prompt design, eval rubric) | escalate to a stronger model (Opus/Sonnet) OR fire `/personal-critic-gate` on the produced prompts + logic before advancing |
| 7 | Deploy (irreversible, needs creds) | HUMAN-GATED -- see Phase 7 |

**Human-gated fence (always pause, regardless of votes):**
- Phase 7 deploy (needs GCP project + Gemini key, which are not in the repo).
- Any `git push`, any write to `.env*`, any secret material (Stay-Paused list).
- Making the repo public (a human decision).

**Verification discipline:** each phase runs its tests with the MockProvider so
the whole 0-6 build needs NO API key and NO network. Tests are the gate. This is
the project's own eval-driven-development thesis applied to its own construction.

**Multi-session:** the `/personal-goal distill` beacon is the audit-tracker.
Phase completion + VERIFY evidence + token cost are logged there each phase, so
progress survives crashes and quota resets.

**Definition of total done (v1):** phases 0-6 all green in CI with MockProvider;
Phase 7 completed by a human with real GCP creds and a live URL. See Section 12.

---

## 1.4 Skill orchestration map (which /personal-plugin skill does what)

The driver should use the full relevant toolchain -- not every skill applies, and
forcing irrelevant ones would waste tokens and add noise. Applicable skills:

| Stage | Skill | Role |
|---|---|---|
| Bootstrap | `/personal-goal distill` | create the audit-tracker beacon (multi-session recovery) |
| Drive (attended) | `/personal-workflow --plan docs/HANDOFF.md` | route + run + VERIFY each phase |
| Drive (unattended) | `/personal-loop --resume distill` | outer loop; run to done across quota windows, fail-closed |
| After each phase | `/personal-goal-next distill` | advance the beacon checkpoint + log evidence |
| Phases 4 & 6 | `/personal-critic-gate` | adversarial gate on the AI/eval logic before advancing |
| On stop / interrupt | `/personal-progress` | capture a handoff doc (the loop may auto-fire this) |
| During build | `/personal-lesson` | capture reusable build lessons |
| Final PR | `/personal-pr-briefing` | reviewer-friendly PR description |
| On demand | `/personal-htsw` | explain code / a diff to the owner |

Deliberately NOT used (out of scope; do not shoehorn): `/personal-fix-decode`
(FIX protocol), `/personal-jira-sync` (no Jira), `/personal-plugin-release`
(this is not the plugin), `/personal-md-to-html`, `/personal-understanding`,
`/personal-prototype` (design is settled), `/personal-facts` (spec is settled),
`/personal-create-eval` (that is for skill evals; this app ships its OWN eval
harness in Phase 6), `/personal-cache-stats`.

## 1.5 Repo bootstrap files (create these in Step 1)

Create these two files verbatim in the blank repo before driving the build.

### `CLAUDE.md`
```markdown
# CLAUDE.md -- distill

Provider-agnostic agentic knowledge-ingestion service (Python / FastAPI). Turns a
URL / YouTube / PDF into verified, structured knowledge via an
extract -> validate -> critic -> structure pipeline. Full spec: docs/HANDOFF.md.

## Build + test commands
- Install:      uv sync
- Test:         uv run pytest -q
- Lint:         uv run ruff check .
- Run local:    uv run uvicorn distill.api.app:app --reload
- Evals (mock): uv run python evals/run_evals.py --provider mock

## Conventions
- Python 3.11+, full type hints, Pydantic v2 for every data shape.
- TDD: write the test first; a phase is done only when its VERIFY command exits 0.
- Ports-and-adapters: the pipeline depends on LLMPort / SourcePort protocols, never
  on a concrete provider or source. Tests inject MockProvider (no network, no cost).
- ASCII only in code, config, and docs (no em-dashes, smart quotes, or unicode
  arrows -- use -- and ->). Non-ASCII mis-decodes under Windows PowerShell.
- Secrets: never commit keys. All config via env (.env is gitignored;
  .env.example documents the vars). Public / synthetic data only -- no PII.

## Autonomous build discipline
- This repo is built by /personal-workflow (or /personal-loop) driving the
  docs/HANDOFF.md phases against a /personal-goal beacon.
- Advance only on a green VERIFY gate with evidence logged to the beacon.
- On failure: fix and re-run the same verify, up to 3 tries, then pause for human.
- Human-gated (never auto): git push, GCP deploy (gcloud), making the repo public.

## Skill routing
- Multi-phase build:        /personal-workflow --plan docs/HANDOFF.md
- Unattended multi-session: /personal-loop --resume distill
- Gate AI logic (ph 4, 6):  /personal-critic-gate
- Capture progress on stop: /personal-progress
- PR description:           /personal-pr-briefing
```

### `.claude/settings.json`
```json
{
  "permissions": {
    "allow": [
      "Bash(uv sync)",
      "Bash(uv run pytest:*)",
      "Bash(uv run ruff:*)",
      "Bash(uv run python:*)",
      "Bash(uv run uvicorn:*)",
      "Bash(docker build:*)",
      "Bash(git init)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(mkdir:*)"
    ],
    "deny": [
      "Bash(git push:*)",
      "Bash(gcloud:*)",
      "Bash(rm -rf:*)"
    ]
  }
}
```
This allow-list lets the driver run build / test / lint / commit without a
permission prompt (the point of "full auto"), while the deny-list hard-fences
push, cloud deploy, and destructive removals to the human gate. Adjust to taste.

## 2. Why this project exists (context)

**Owner:** Ping Chung -- 20-year fintech engineer (Software Developer / VP, Napa
Group). Sole engineer of a 430,000-line fixed-income trading platform for five
tier-1 US banks. Strong: T-SQL, VB6/COM+, .NET/C#, ASP Classic, IIS, GCP ops,
Azure SSO, multi-agent orchestration, context engineering, eval-driven
development. Lighter: production Python, Kubernetes/Kafka.

**Goal:** a public, professional portfolio project that proves the gaps recruiters
gate on for AI-engineer roles (e.g. Amex "Staff AI Eng - Agentic AI").

| Design element | Resume gap it closes |
|---|---|
| Python + FastAPI service | Production Python (the headline gap) |
| Deploy to GCP Cloud Run | Cloud / deployment (owner's real GCP skill) |
| Live Anthropic/Gemini/OpenAI calls | Direct live-LLM API work |
| Pydantic schemas + repair loop | Schema validation (Pydantic/Zod keyword) |
| Provider-agnostic `LLMPort` | Clean architecture + token-economics comparison |
| Extract -> Validate -> Critic -> Structure | Agentic orchestration (owner's differentiator) |
| Eval harness with rubric scorer | Evaluation-driven development |
| Public synthetic data, clean theme | A shareable link with zero PII risk |

**Provenance:** the input-handling ideas (URL / YouTube / PDF -> clean text) come
from the owner's private `Maid_Squad_PKM` project (`triage_raw.py`). This is a
clean, de-themed rebuild -- NOT a copy. See Section 9 for reuse + security rules.

---

## 3. What it is (one-liner + v1 scope)

A provider-agnostic service that turns any source (URL / YouTube / PDF) into
verified, structured knowledge, with an LLM-critic stage that scores its own
output for faithfulness before emitting it.

**In v1:** three input adapters, the four-stage agentic pipeline, a Pydantic
schema layer with a repair loop, an eval harness, a FastAPI endpoint, tests with
a mock LLM, and a Cloud Run deploy.

**NOT in v1 (v2 backlog):** web UI, auth, a database, multi-document
knowledge-graph linking, batch/queue processing.

---

## 4. Architecture (ports and adapters)

```
                +------------------ SourcePort (protocol) ------------------+
  input url --> |  URLSource | YouTubeSource | PDFSource                    |
                +----------------------------+-----------------------------+
                                             |  produces
                                             v
                                   RawDocument(text, metadata)
                                             |
   +-------------------------- Pipeline (orchestrated, token-metered) ------+
   |  1. Extract    LLMPort call: RawDocument -> draft JSON                  |
   |  2. Validate   Pydantic parse; on ValidationError -> repair-reprompt    |
   |  3. Critic     2nd LLMPort call: score draft vs source (faithfulness);  |
   |                below threshold -> one bounded retry                     |
   |  4. Structure  validated+verified -> Markdown (frontmatter) + JSON      |
   +-----------------------------+------------------------------------------+
                                 v
             KnowledgeDoc + IngestTrace(per-stage tokens, cost, latency, critic)

   LLMPort: Anthropic | Gemini | OpenAI | Ollama | MockProvider(tests/CI)
   API: FastAPI  POST /ingest {source_type, value} -> KnowledgeDoc + IngestTrace
   Config: env-driven (DISTILL_PROVIDER, DISTILL_MODEL, keys, thresholds)
```

---

## 5. Data models (Pydantic) -- `src/distill/models.py`

- `RawDocument`: source_type (url|youtube|pdf), source_ref, title|None, text,
  fetched_at, meta.
- `KnowledgeDraft`: summary, key_points[], entities[Entity], topics[]. (Extract
  produces this; Validate enforces it.)
- `Entity`: name, type, mentions (optional).
- `CriticResult`: confidence (0-1), faithful (bool), issues[], missing_points[].
- `KnowledgeDoc`: validated draft + critic verdict + source ref; `.to_markdown()`
  (YAML frontmatter) + `.model_dump()` (JSON).
- `IngestTrace`: list of `StageTrace(name, tokens_in, tokens_out, cost_usd,
  latency_ms)` + totals. The token-economics showcase surface.

---

## 6. LLM provider abstraction -- `src/distill/llm/`

`base.py` defines a Protocol:
```
class LLMResponse:  text: str; tokens_in: int; tokens_out: int; cost_usd: float
class LLMPort(Protocol):
    def complete(self, prompt: str, *, system: str | None = None,
                 json_schema: dict | None = None) -> LLMResponse: ...
```
- `MockProvider` -- deterministic canned responses keyed off the prompt; used by
  ALL tests/CI. No network, no cost.
- `GeminiProvider` -- DEFAULT live provider (`google-genai` SDK, owner's personal
  GCP key). Cheapest live path.
- `AnthropicProvider` (`anthropic` SDK), `OpenAIProvider` (`openai` SDK),
  `OllamaProvider` (local HTTP) -- optional, same interface.
- Cost table per model fills `cost_usd` so `IngestTrace` reports real cost/run.
- Provider selected by `DISTILL_PROVIDER`; factory in `llm/__init__.py`.

---

## 7. Repository layout

```
<repo>/
  README.md   pyproject.toml   Dockerfile   .env.example   .gitignore
  .github/workflows/ci.yml
  src/distill/
    __init__.py  models.py  config.py
    sources/  base.py url.py youtube.py pdf.py
    llm/      base.py mock_provider.py gemini_provider.py anthropic_provider.py openai_provider.py ollama_provider.py
    pipeline/ extract.py validate.py critic.py structure.py orchestrator.py
    api/      app.py
  evals/      golden/  rubric.py  run_evals.py
  tests/      conftest.py test_models.py test_sources.py test_llm.py test_pipeline.py test_critic.py test_api.py
  docs/       HANDOFF.md  architecture.md
```

---

## 8. Phased build plan (each phase = BUILD + VERIFY gate + DONE-WHEN)

Every VERIFY command must exit 0 and its output logged to the beacon before
advancing. Commands assume `uv` and repo root as cwd.

### Phase 0 -- Scaffold (runner: fable)
- BUILD: `pyproject.toml` (deps: pydantic, fastapi, uvicorn, httpx, pytest, ruff;
  optional-extras for google-genai/anthropic/openai; source extractors:
  trafilatura, youtube-transcript-api, pypdf), package skeleton with empty
  modules, `.gitignore` (`.env`, `__pycache__`, `.venv`, `.pytest_cache`),
  `.env.example` documenting every env var (NO real values), `ruff` config, a
  GitHub Actions `ci.yml` running `ruff` + `pytest` (no secrets), and ONE trivial
  passing test `tests/test_smoke.py`. Also ensure `CLAUDE.md` + `.claude/settings.json`
  (Section 1.5) and the `docs/` folder exist -- create them if Step 1 was skipped.
- VERIFY: `uv sync && uv run ruff check . && uv run pytest -q`
- DONE-WHEN: exit 0; >=1 test collected and passing; ruff clean.

### Phase 1 -- Models (runner: fable)
- BUILD: all Pydantic models from Section 5 + `tests/test_models.py`.
- VERIFY: `uv run pytest tests/test_models.py -q`
- DONE-WHEN: green; tests cover a valid parse, an invalid parse raising
  ValidationError, and `KnowledgeDoc.to_markdown()` emitting YAML frontmatter.

### Phase 2 -- LLM layer (runner: fable)
- BUILD: `LLMPort`, `MockProvider` (deterministic), `GeminiProvider`, the provider
  factory, the per-model cost table, `tests/test_llm.py` (MockProvider only).
- VERIFY: `uv run pytest tests/test_llm.py -q`
- DONE-WHEN: green; MockProvider is deterministic; factory selects provider by
  `DISTILL_PROVIDER`; no network in tests.

### Phase 3 -- Sources (runner: fable)
- BUILD: `SourcePort` + `URLSource` (trafilatura), `YouTubeSource`
  (youtube-transcript-api), `PDFSource` (pypdf) -> `RawDocument`. Local fixtures
  under `tests/fixtures/` (a saved HTML page, a transcript JSON, a small PDF).
- VERIFY: `uv run pytest tests/test_sources.py -q`
- DONE-WHEN: green using local fixtures ONLY (no live network); each adapter maps
  its fixture to a `RawDocument`.

### Phase 4 -- Pipeline (runner: ESCALATE / critic-gate -- judgment-heavy)
- BUILD: `extract`, `validate` (with repair-reprompt loop), `critic`, `structure`,
  `orchestrator` (meters tokens/cost/latency into `IngestTrace`) +
  `tests/test_pipeline.py` + `tests/test_critic.py` against MockProvider.
- VERIFY: `uv run pytest tests/test_pipeline.py tests/test_critic.py -q`
- DONE-WHEN: green; a malformed-LLM-output fixture exercises the repair loop; a
  low-confidence critic result triggers exactly one bounded retry; a full run
  yields a `KnowledgeDoc` + populated `IngestTrace`.
- GATE: before advancing, fire `/personal-critic-gate` on the extract/critic
  prompts + repair logic (this is the project's own thesis -- prove the AI logic).

### Phase 5 -- API (runner: fable)
- BUILD: FastAPI `app.py` with `POST /ingest` and `GET /health`;
  `tests/test_api.py` integration test using the MockProvider.
- VERIFY: `uv run pytest tests/test_api.py -q`
- DONE-WHEN: green; `/ingest` returns a `KnowledgeDoc` + `IngestTrace` for a
  mocked run; `/health` returns 200.

### Phase 6 -- Evals (runner: ESCALATE / critic-gate -- judgment-heavy)
- BUILD: `evals/golden/` with 3+ PUBLIC sources + hand-authored `expected.json`
  key-facts; `rubric.py` (deterministic key-fact presence + schema-valid checks,
  plus an LLM-judge faithfulness check via `LLMPort`); `run_evals.py` (runs the
  pipeline over the golden set, prints pass-rate + mean critic confidence +
  variance, exits non-zero below a floor). Wire a mock-provider eval run into CI.
- VERIFY: `uv run python evals/run_evals.py --provider mock`
- DONE-WHEN: exit 0; prints a pass-rate table; CI includes the mock eval run.
- GATE: fire `/personal-critic-gate` on the rubric design before advancing.

### Phase 7 -- Deploy + docs (HUMAN-GATED -- needs GCP creds)
- BUILD (autonomous OK): `Dockerfile` (slim python -> uvicorn); `README.md` with
  the architecture diagram, a provider-comparison table, a design-rationale
  section, and a placeholder for the live URL. `docker build` must succeed.
- VERIFY (local, autonomous): `docker build -t distill .`  (exit 0) and
  `test -f README.md`.
- HUMAN-GATED STEP (workflow PAUSES here): a human supplies the GCP project +
  Gemini key, runs `gcloud run deploy` (document the exact command in the
  README), confirms a live run with the Gemini provider succeeds, and pastes the
  live URL into the README. Only a human makes the repo public.

---

## 9. Reuse + security rules (READ before touching Maid_Squad)

- Do NOT copy files wholesale from `Maid_Squad_PKM`. Reuse the IDEAS/logic, rewritten clean.
- Never copy any `.env`, `.db`, `msg_debug*.json`, or credential file into this public repo.
- Rotate the Google API key currently committed in `Maid_Squad_PKM/.env` before reusing anything (revoke + reissue in Google Cloud Console).
- No maid/persona theme. Neutral professional naming everywhere.
- Public/synthetic data only. The golden set + demos use publicly available sources. No personal notes, no financial data, no personal email in code or docs.

---

## 10. Resume + interview payoff (what this unlocks)

- Resume bullet: "Built and deployed a provider-agnostic agentic
  knowledge-ingestion service in Python (FastAPI, Pydantic) on GCP Cloud Run --
  extract / schema-validate / LLM-critic-verify / structure pipeline with an
  evaluation harness measuring extraction faithfulness and cost per run across
  LLM providers." (Link the repo.)
- Interview talking points, each backed by code: the Critic stage + eval harness
  ("I prove AI output"); `LLMPort` + cost table ("provider-agnostic, controls
  cost"); Pydantic + repair loop ("schema-validated tool output"); `IngestTrace`
  ("I measured token economics").

---

## 11. Open decisions (my defaults, change freely)

1. **Name:** `distill`. Alternatives: `veridex`, `groundtruth`, `sift`. Pick
   before Phase 0 (it names the repo + package). If changed, update Section 7 +
   the `src/distill/` package path.
2. **Live default provider:** Gemini on personal GCP.
3. **Package manager:** `uv` (assumed by all VERIFY commands).
4. **Critic stage as centerpiece:** kept -- it is the differentiator.

---

## 12. Definition of done (v1)

- [ ] Phases 0-6 VERIFY gates all exit 0, evidence logged to the beacon.
- [ ] `pytest` + `ruff` green in CI with MockProvider (no secrets).
- [ ] `POST /ingest` works for URL, YouTube, and PDF (mocked in tests).
- [ ] Pydantic validation + repair loop demonstrably handle malformed LLM output.
- [ ] Critic stage returns a confidence score and can trigger one retry.
- [ ] `run_evals.py` runs the golden set and prints pass-rate + variance.
- [ ] `docker build` succeeds.
- [ ] (HUMAN) Deployed to Cloud Run at a public URL; live Gemini run succeeds;
      URL in README.
- [ ] Zero PII / secrets in the repo or its git history.
