"""FastAPI service layer: POST /ingest and GET /health.

App-factory pattern: `create_app(config)` builds an app around an injected
Config (tests pass one in; production reads the environment). The module-level
`app` is what `uvicorn distill.api.app:app` serves.

Dependency seams (both overridable via app.dependency_overrides in tests):
- get_fetcher: returns the callable that turns (source_type, value) into a
  RawDocument via the source registry. Tests override it to stay offline.
- get_pipeline: builds the provider + Pipeline PER REQUEST from the app's
  Config, so no LLM client outlives a request and tests can inject a stub.

Error mapping (honest failures, HANDOFF Section 4):
- SourceError            -> 422 (the caller's reference is unusable)
- PipelineError          -> 502 {stage, message, partial_trace?} (upstream
                            LLM stage failed; partial metering preserved)
- unknown source_type    -> 422 (Pydantic Literal validation)
"""

from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

import distill
from distill.config import Config
from distill.llm import get_provider
from distill.models import IngestTrace, KnowledgeDoc, RawDocument, SourceType
from distill.pipeline import Pipeline, PipelineError
from distill.sources.base import SourceError, resolve_source

__all__ = [
    "IngestRequest",
    "IngestResponse",
    "app",
    "create_app",
    "get_fetcher",
    "get_pipeline",
]


class IngestRequest(BaseModel):
    """Body of POST /ingest: what to fetch and how to interpret the ref."""

    source_type: SourceType
    value: str


class IngestResponse(BaseModel):
    """Result of a successful ingest: the doc, its metering, and markdown."""

    doc: KnowledgeDoc
    trace: IngestTrace
    markdown: str


class PipelineRunner(Protocol):
    """The one Pipeline method the API layer needs (stubbed in tests)."""

    def run(self, doc: RawDocument) -> tuple[KnowledgeDoc, IngestTrace]: ...


class Fetcher(Protocol):
    """Callable turning (source_type, value) into a RawDocument."""

    def __call__(self, source_type: str, value: str) -> RawDocument: ...


def _fetch_document(source_type: str, value: str) -> RawDocument:
    """Default fetcher: resolve the registry adapter and fetch the ref.

    resolve_source constructs PDFSource with allow_local=False, so API
    callers can never reach the local filesystem.
    """
    return resolve_source(source_type).fetch(value)


def get_config(request: Request) -> Config:
    """Dependency: the Config the app was created with."""
    return request.app.state.config


def get_fetcher() -> Fetcher:
    """Dependency seam for the source fetch; tests override this to stay offline."""
    return _fetch_document


def get_pipeline(config: Annotated[Config, Depends(get_config)]) -> PipelineRunner:
    """Dependency: a per-request Pipeline built from the configured provider."""
    provider = get_provider(config.provider, config.model)
    return Pipeline(provider, critic_threshold=config.critic_threshold)


def create_app(config: Config | None = None) -> FastAPI:
    """Build the distill API around `config` (or the environment when None)."""
    app = FastAPI(
        title="distill",
        description="Provider-agnostic agentic knowledge-ingestion service.",
        version=distill.__version__,
    )
    app.state.config = config if config is not None else Config.from_env()

    @app.get("/health")
    def health(config: Annotated[Config, Depends(get_config)]) -> dict[str, str]:
        return {
            "status": "ok",
            "provider": config.provider,
            "version": distill.__version__,
        }

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(
        body: IngestRequest,
        fetch: Annotated[Fetcher, Depends(get_fetcher)],
        pipeline: Annotated[PipelineRunner, Depends(get_pipeline)],
    ) -> IngestResponse:
        try:
            raw_doc = fetch(body.source_type, body.value)
        except SourceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            doc, trace = pipeline.run(raw_doc)
        except PipelineError as exc:
            detail: dict[str, Any] = {"stage": exc.stage, "message": str(exc)}
            if exc.partial_trace is not None:
                detail["partial_trace"] = exc.partial_trace.model_dump(mode="json")
            raise HTTPException(status_code=502, detail=detail) from exc
        return IngestResponse(doc=doc, trace=trace, markdown=doc.to_markdown())

    return app


app = create_app()
