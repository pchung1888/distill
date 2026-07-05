# distill -- multi-stage build for a slim Cloud Run image.
#
# Builder installs dependencies with uv into /app/.venv (lockfile-frozen),
# then installs the project itself as a built wheel (--no-editable) so the
# runtime stage needs only the venv -- no src/, tests/, evals/, or docs/.

# ---------- Builder ----------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer first: cached until pyproject.toml or uv.lock changes.
# --all-extras pulls the optional provider SDKs (google-genai, anthropic,
# openai) so one image can serve any configured live provider.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --all-extras --no-install-project

# Project layer: hatchling needs README.md (pyproject readme field) + src/.
COPY README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev --all-extras --no-editable

# ---------- Runtime ----------
FROM python:3.12-slim

RUN groupadd --system distill \
    && useradd --system --gid distill --no-create-home distill

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER distill
EXPOSE 8080

# Cloud Run injects PORT at runtime; default to 8080 for local runs.
# Shell form (via sh -c) so ${PORT:-8080} expands; exec keeps uvicorn as
# PID 1 so it receives SIGTERM directly on shutdown.
CMD ["/bin/sh", "-c", "exec uvicorn distill.api.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
