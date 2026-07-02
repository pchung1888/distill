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
