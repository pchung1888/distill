"""FastAPI service layer: POST /ingest, POST /compare, and GET /health.

App-factory pattern: `create_app(config)` builds an app around an injected
Config (tests pass one in; production reads the environment). The module-level
`app` is what `uvicorn distill.api.app:app` serves.

Dependency seams (all overridable via app.dependency_overrides in tests):
- get_fetcher: returns the callable that turns (source_type, value) into a
  RawDocument via the source registry. Tests override it to stay offline.
- get_pipeline: builds the DEFAULT (config-provider) Pipeline PER REQUEST.
- get_pipeline_factory: builds a Pipeline for an EXPLICIT provider name, used
  when a caller (the public demo frontend) picks a provider per-request via
  IngestRequest.provider, or by /compare across several providers at once.

Error mapping (honest failures, HANDOFF Section 4):
- SourceError            -> 422 (the caller's reference is unusable)
- PipelineError          -> 502 {stage, message, partial_trace?} (upstream
                            LLM stage failed; partial metering preserved)
- unknown source_type    -> 422 (Pydantic Literal validation)
- RateLimitExceeded      -> 429 {kind, message} (public-demo safety cap;
                            frontend falls back to curated examples, NEVER a
                            silent switch to MockProvider -- see ratelimit.py)

Public-demo additions (distill-demo frontend calling this over the network):
- CORS is restricted to config.allowed_origins (default localhost:3000 only).
- A `distill_visitor` httponly cookie identifies a visitor for rate limiting
  (no login). See api/ratelimit.py for the daily-cap design and rationale.
"""

import uuid
from collections.abc import Callable
from typing import Annotated, Any, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import distill
from distill.api.ratelimit import (
    VISITOR_COOKIE,
    RateLimitConfig,
    RateLimiter,
    RateLimitExceeded,
)
from distill.config import Config
from distill.llm import get_provider
from distill.models import IngestTrace, KnowledgeDoc, RawDocument, SourceType
from distill.pipeline import Pipeline, PipelineError
from distill.sources.base import SourceError, resolve_source

__all__ = [
    "CompareRequest",
    "CompareResponse",
    "IngestRequest",
    "IngestResponse",
    "ProviderResult",
    "app",
    "create_app",
    "get_fetcher",
    "get_limiter",
    "get_pipeline",
    "get_pipeline_factory",
    "get_visitor_id",
]

VISITOR_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365


class IngestRequest(BaseModel):
    """Body of POST /ingest: what to fetch and how to interpret the ref.

    `provider` is optional: None uses the server's configured default
    (the original single-provider behavior); a name overrides it per-request
    (the public demo's provider picker).
    """

    source_type: SourceType
    value: str
    provider: str | None = None


class IngestResponse(BaseModel):
    """Result of a successful ingest: the doc, its metering, and markdown."""

    doc: KnowledgeDoc
    trace: IngestTrace
    markdown: str


class CompareRequest(BaseModel):
    """Body of POST /compare: one source, run across several providers."""

    source_type: SourceType
    value: str
    providers: list[str]


class ProviderResult(BaseModel):
    """One provider's outcome within a /compare run. Exactly one of
    (doc, trace) or error is populated -- a per-provider failure never aborts
    the other providers' results."""

    provider: str
    doc: KnowledgeDoc | None = None
    trace: IngestTrace | None = None
    error: str | None = None


class CompareResponse(BaseModel):
    results: list[ProviderResult]


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
    """Dependency: a per-request Pipeline built from the configured provider.

    KNOWN TRADEOFF: /ingest declares both this AND get_pipeline_factory as
    route parameters, so FastAPI resolves both on EVERY request -- including
    requests whose body.provider overrides the default and never touch this
    Pipeline's output. Provider client construction is normally cheap (no
    network call, no charge), so this is harmless in practice, but it means
    the server's DEFAULT provider (config.provider) must always be
    constructible -- e.g. its API key must be present -- even on deployments
    where most traffic supplies an explicit body.provider and never uses the
    default. If that ever changes, split this into a lazy factory instead.
    """
    provider = get_provider(config.provider, config.model)
    return Pipeline(provider, critic_threshold=config.critic_threshold)


def get_pipeline_factory(
    config: Annotated[Config, Depends(get_config)],
) -> Callable[[str], PipelineRunner]:
    """Dependency: a factory building a Pipeline for an EXPLICIT provider name.

    Used when the caller picks the provider per-request (IngestRequest.provider,
    or /compare's provider list) instead of relying on the server's default.
    """

    def _build(provider_name: str) -> PipelineRunner:
        provider = get_provider(provider_name, config.model)
        return Pipeline(provider, critic_threshold=config.critic_threshold)

    return _build


def get_limiter(request: Request) -> RateLimiter:
    """Dependency: the app's shared RateLimiter instance."""
    return request.app.state.limiter


def get_visitor_id(request: Request, response: Response) -> str:
    """Dependency: a stable per-browser id for rate limiting, backed by an
    httponly cookie. No login -- easy to clear, which is an accepted
    tradeoff for a portfolio demo (see ratelimit.py module docstring)."""
    visitor = request.cookies.get(VISITOR_COOKIE)
    if not visitor:
        visitor = str(uuid.uuid4())
        response.set_cookie(
            VISITOR_COOKIE,
            visitor,
            max_age=VISITOR_COOKIE_MAX_AGE_SECONDS,
            httponly=True,
            samesite="lax",
        )
    return visitor


def create_app(config: Config | None = None) -> FastAPI:
    """Build the distill API around `config` (or the environment when None)."""
    app = FastAPI(
        title="distill",
        description="Provider-agnostic agentic knowledge-ingestion service.",
        version=distill.__version__,
    )
    app.state.config = config if config is not None else Config.from_env()
    app.state.limiter = RateLimiter(
        db_path=app.state.config.ratelimit_db_path,
        config=RateLimitConfig(
            daily_single_runs=app.state.config.ratelimit_daily_single_runs,
            daily_compares=app.state.config.ratelimit_daily_compares,
            global_daily_budget_usd=app.state.config.ratelimit_global_daily_budget_usd,
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app.state.config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RateLimitExceeded)
    def _handle_rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": {"kind": exc.kind, "message": str(exc)}},
        )

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
        default_pipeline: Annotated[PipelineRunner, Depends(get_pipeline)],
        pipeline_factory: Annotated[Callable[[str], PipelineRunner], Depends(get_pipeline_factory)],
        visitor: Annotated[str, Depends(get_visitor_id)],
        limiter: Annotated[RateLimiter, Depends(get_limiter)],
    ) -> IngestResponse:
        limiter.check_and_reserve(visitor, "single")
        try:
            raw_doc = fetch(body.source_type, body.value)
        except SourceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        pipeline = pipeline_factory(body.provider) if body.provider else default_pipeline
        try:
            doc, trace = pipeline.run(raw_doc)
        except PipelineError as exc:
            if exc.partial_trace is not None:
                limiter.record_spend(exc.partial_trace.total_cost_usd)
            detail: dict[str, Any] = {"stage": exc.stage, "message": str(exc)}
            if exc.partial_trace is not None:
                detail["partial_trace"] = exc.partial_trace.model_dump(mode="json")
            raise HTTPException(status_code=502, detail=detail) from exc
        limiter.record_spend(trace.total_cost_usd)
        return IngestResponse(doc=doc, trace=trace, markdown=doc.to_markdown())

    @app.post("/compare", response_model=CompareResponse)
    def compare(
        body: CompareRequest,
        fetch: Annotated[Fetcher, Depends(get_fetcher)],
        pipeline_factory: Annotated[Callable[[str], PipelineRunner], Depends(get_pipeline_factory)],
        visitor: Annotated[str, Depends(get_visitor_id)],
        limiter: Annotated[RateLimiter, Depends(get_limiter)],
    ) -> CompareResponse:
        limiter.check_and_reserve(visitor, "compare")
        try:
            raw_doc = fetch(body.source_type, body.value)
        except SourceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        results: list[ProviderResult] = []
        total_spend = 0.0
        for name in body.providers:
            pipeline = pipeline_factory(name)
            try:
                doc, trace = pipeline.run(raw_doc)
                results.append(ProviderResult(provider=name, doc=doc, trace=trace))
                total_spend += trace.total_cost_usd
            except PipelineError as exc:
                if exc.partial_trace is not None:
                    total_spend += exc.partial_trace.total_cost_usd
                results.append(ProviderResult(provider=name, error=str(exc)))
        limiter.record_spend(total_spend)
        return CompareResponse(results=results)

    return app


app = create_app()
