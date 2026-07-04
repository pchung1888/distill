"""Phase 5 tests: FastAPI service layer (MockProvider only, no network).

Coverage per HANDOFF Phase 5 DONE-WHEN + dispatch brief:
- GET /health returns 200 with status ok, provider name, and version.
- POST /ingest happy path (stubbed source fetch, mock provider) returns the
  KnowledgeDoc, an IngestTrace with extract+critic stages, and markdown
  starting with YAML frontmatter.
- Unknown source_type is rejected with 422 (Pydantic Literal).
- SourceError from the fetch maps to 422 with the message.
- PipelineError maps to 502 carrying the stage (and the partial trace when
  the error has one).

Every test that reaches the fetch overrides the get_fetcher dependency, so
no test ever touches the network.
"""

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import distill
from distill.api.app import create_app, get_fetcher, get_limiter, get_pipeline, get_pipeline_factory
from distill.api.ratelimit import VISITOR_COOKIE, RateLimitConfig, RateLimiter
from distill.config import Config
from distill.models import CriticResult, IngestTrace, KnowledgeDoc, RawDocument, StageTrace
from distill.pipeline import PipelineError
from distill.sources.base import SourceError

FETCHED_AT = datetime(2026, 7, 2, 12, 0, 0, tzinfo=UTC)

# The MockProvider's canned extract summary (kept in sync with mock_provider).
CANNED_SUMMARY = "A mock summary of the source document produced for tests."


def make_raw_doc() -> RawDocument:
    return RawDocument(
        source_type="url",
        source_ref="https://example.com/article",
        title="Example Article",
        text="An article about Mock Corp and Jane Example.",
        fetched_at=FETCHED_AT,
    )


def _stub_fetch(source_type: str, value: str) -> RawDocument:
    """Shared get_fetcher override body: ignores its args, returns the fixture doc."""
    return make_raw_doc()


def _tagged_factory(name: str) -> "_TaggedPipeline":
    """Shared get_pipeline_factory override body: tags output with the provider name."""
    return _TaggedPipeline(name)


@pytest.fixture()
def client() -> TestClient:
    """TestClient over an app with an injected mock-provider Config and a
    stubbed source fetch (no network anywhere)."""
    app = create_app(Config(provider="mock"))

    def fake_fetch(source_type: str, value: str) -> RawDocument:
        return make_raw_doc()

    app.dependency_overrides[get_fetcher] = lambda: fake_fetch
    return TestClient(app)


# ---------------------------------------------------------------- /health


class TestHealth:
    def test_health_returns_ok_provider_and_version(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "provider": "mock",
            "version": distill.__version__,
        }

    def test_health_reads_provider_from_env_when_config_not_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISTILL_PROVIDER", "mock")
        app = create_app()
        response = TestClient(app).get("/health")
        assert response.status_code == 200
        assert response.json()["provider"] == "mock"


# ------------------------------------------------------ /ingest happy path


class TestIngestHappyPath:
    def test_ingest_returns_doc_trace_and_markdown(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={"source_type": "url", "value": "https://example.com/article"},
        )
        assert response.status_code == 200
        body = response.json()

        # KnowledgeDoc fields from the canned mock run.
        doc = body["doc"]
        assert doc["summary"] == CANNED_SUMMARY
        assert doc["source_type"] == "url"
        assert doc["source_ref"] == "https://example.com/article"
        assert doc["title"] == "Example Article"
        assert len(doc["key_points"]) == 3
        assert doc["entities"][0]["name"] == "Mock Corp"
        assert doc["critic"]["confidence"] == 0.95
        assert doc["critic"]["faithful"] is True

        # Trace has the extract and critic stages with computed totals.
        stage_names = [stage["name"] for stage in body["trace"]["stages"]]
        assert stage_names == ["extract", "critic"]
        assert body["trace"]["total_tokens_in"] > 0
        assert body["trace"]["total_tokens_out"] > 0

        # Markdown render starts with the YAML frontmatter fence.
        assert body["markdown"].startswith("---")
        assert "## Summary" in body["markdown"]


# ------------------------------------------------------- /ingest rejects


class TestIngestValidation:
    def test_unknown_source_type_is_422(self, client: TestClient) -> None:
        response = client.post(
            "/ingest",
            json={"source_type": "carrier-pigeon", "value": "coo"},
        )
        assert response.status_code == 422

    def test_missing_value_is_422(self, client: TestClient) -> None:
        response = client.post("/ingest", json={"source_type": "url"})
        assert response.status_code == 422


# ---------------------------------------------------- error mapping: source


class TestIngestSourceError:
    def test_source_error_maps_to_422_with_message(self) -> None:
        app = create_app(Config(provider="mock"))

        def failing_fetch(source_type: str, value: str) -> RawDocument:
            raise SourceError("host resolves to non-public address 127.0.0.1")

        app.dependency_overrides[get_fetcher] = lambda: failing_fetch
        response = TestClient(app).post(
            "/ingest",
            json={"source_type": "url", "value": "https://localhost/secret"},
        )
        assert response.status_code == 422
        assert "non-public address" in response.json()["detail"]


# -------------------------------------------------- error mapping: pipeline


class _RaisingPipeline:
    def __init__(self, error: PipelineError) -> None:
        self._error = error

    def run(self, doc: RawDocument) -> tuple:
        raise self._error


def _client_with_pipeline_error(error: PipelineError) -> TestClient:
    app = create_app(Config(provider="mock"))

    def fake_fetch(source_type: str, value: str) -> RawDocument:
        return make_raw_doc()

    app.dependency_overrides[get_fetcher] = lambda: fake_fetch
    app.dependency_overrides[get_pipeline] = lambda: _RaisingPipeline(error)
    return TestClient(app)


class TestIngestPipelineError:
    def test_pipeline_error_maps_to_502_with_stage(self) -> None:
        error = PipelineError("validate", cause=ValueError("unrepairable output"))
        client = _client_with_pipeline_error(error)
        response = client.post(
            "/ingest",
            json={"source_type": "url", "value": "https://example.com/article"},
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail["stage"] == "validate"
        assert "unrepairable output" in detail["message"]
        assert "partial_trace" not in detail

    def test_pipeline_error_serializes_partial_trace_when_present(self) -> None:
        partial = IngestTrace(
            stages=[
                StageTrace(
                    name="extract",
                    tokens_in=120,
                    tokens_out=80,
                    cost_usd=0.0,
                    latency_ms=5,
                )
            ]
        )
        error = PipelineError("critic", cause=RuntimeError("provider down"), partial_trace=partial)
        client = _client_with_pipeline_error(error)
        response = client.post(
            "/ingest",
            json={"source_type": "url", "value": "https://example.com/article"},
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail["stage"] == "critic"
        trace = detail["partial_trace"]
        assert [stage["name"] for stage in trace["stages"]] == ["extract"]
        assert trace["stages"][0]["tokens_in"] == 120
        assert trace["total_tokens_in"] == 120
        # The detail must be JSON-serializable end to end (already proven by
        # response.json(), but assert the round trip explicitly).
        assert json.dumps(detail)


# --------------------------------------------------------------- public demo


def _tagged_pipeline_run(tag: str, doc: RawDocument) -> tuple[KnowledgeDoc, IngestTrace]:
    knowledge_doc = KnowledgeDoc(
        source_type=doc.source_type,
        source_ref=doc.source_ref,
        title=tag,
        summary=f"summary from {tag}",
        key_points=["a", "b", "c"],
        entities=[],
        topics=["t"],
        critic=CriticResult(confidence=0.9, faithful=True, issues=[], missing_points=[]),
        created_at=FETCHED_AT,
    )
    trace = IngestTrace(
        stages=[
            StageTrace(name="extract", tokens_in=10, tokens_out=5, cost_usd=0.001, latency_ms=1)
        ]
    )
    return knowledge_doc, trace


class _TaggedPipeline:
    """Stub pipeline that tags its output with a name -- lets a test prove
    WHICH provider path the route took without needing real provider creds."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    def run(self, doc: RawDocument) -> tuple[KnowledgeDoc, IngestTrace]:
        return _tagged_pipeline_run(self._tag, doc)


class _RaisingNamedPipeline:
    def __init__(self, error: PipelineError) -> None:
        self._error = error

    def run(self, doc: RawDocument) -> tuple:
        raise self._error


class TestCORS:
    def test_allowed_origin_gets_cors_header(self) -> None:
        app = create_app(Config(provider="mock", allowed_origins=["http://localhost:3000"]))
        response = TestClient(app).get(
            "/health", headers={"Origin": "http://localhost:3000"}
        )
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"

    def test_disallowed_origin_gets_no_cors_header(self) -> None:
        app = create_app(Config(provider="mock", allowed_origins=["http://localhost:3000"]))
        response = TestClient(app).get(
            "/health", headers={"Origin": "http://evil.example.com"}
        )
        assert "access-control-allow-origin" not in response.headers


class TestVisitorCookie:
    def test_first_ingest_sets_visitor_cookie(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        client = TestClient(app)
        response = client.post(
            "/ingest", json={"source_type": "url", "value": "https://example.com/article"}
        )
        assert response.status_code == 200
        assert VISITOR_COOKIE in response.cookies

    def test_visitor_cookie_is_reused_across_requests(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        client = TestClient(app)
        client.post("/ingest", json={"source_type": "url", "value": "https://example.com/article"})
        first_visitor = client.cookies.get(VISITOR_COOKIE)
        client.post("/ingest", json={"source_type": "url", "value": "https://example.com/article"})
        second_visitor = client.cookies.get(VISITOR_COOKIE)
        assert first_visitor == second_visitor


class TestProviderOverride:
    def test_ingest_without_provider_uses_default_pipeline(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        app.dependency_overrides[get_pipeline] = lambda: _TaggedPipeline("default")
        app.dependency_overrides[get_pipeline_factory] = lambda: (
            lambda name: (_ for _ in ()).throw(AssertionError("factory should not be called"))
        )
        response = TestClient(app).post(
            "/ingest", json={"source_type": "url", "value": "https://example.com/article"}
        )
        assert response.status_code == 200
        assert response.json()["doc"]["title"] == "default"

    def test_ingest_with_provider_uses_factory(self) -> None:
        # NOTE: FastAPI resolves get_pipeline (the default) on every request
        # regardless of body.provider -- see its docstring's KNOWN TRADEOFF.
        # This test proves the FACTORY's output wins when a provider is
        # given; it does not (and per that tradeoff, cannot) assert the
        # default branch was skipped.
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        app.dependency_overrides[get_pipeline] = lambda: _TaggedPipeline("default")
        app.dependency_overrides[get_pipeline_factory] = lambda: _tagged_factory
        response = TestClient(app).post(
            "/ingest",
            json={
                "source_type": "url",
                "value": "https://example.com/article",
                "provider": "gemini",
            },
        )
        assert response.status_code == 200
        assert response.json()["doc"]["title"] == "gemini"


class TestCompare:
    def test_compare_runs_each_provider_and_tags_results(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        app.dependency_overrides[get_pipeline_factory] = lambda: _tagged_factory
        response = TestClient(app).post(
            "/compare",
            json={
                "source_type": "url",
                "value": "https://example.com/article",
                "providers": ["gemini", "openai"],
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert [r["provider"] for r in results] == ["gemini", "openai"]
        assert results[0]["doc"]["title"] == "gemini"
        assert results[1]["doc"]["title"] == "openai"
        assert results[0]["error"] is None

    def test_compare_isolates_a_failing_provider(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch

        def factory(name: str):
            if name == "broken":
                error = PipelineError("critic", cause=RuntimeError("provider down"))
                return _RaisingNamedPipeline(error)
            return _TaggedPipeline(name)

        app.dependency_overrides[get_pipeline_factory] = lambda: factory
        response = TestClient(app).post(
            "/compare",
            json={
                "source_type": "url",
                "value": "https://example.com/article",
                "providers": ["broken", "gemini"],
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        assert results[0]["provider"] == "broken"
        assert results[0]["doc"] is None
        assert "provider down" in results[0]["error"]
        assert results[1]["doc"]["title"] == "gemini"


class TestRateLimiting:
    def test_single_run_429_after_daily_cap(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        # Build ONE limiter and capture it by closure -- a lambda that builds
        # a fresh RateLimiter per call would reset the in-memory db every
        # request, defeating the test (caught by this test failing first try).
        limiter = RateLimiter(db_path=":memory:", config=RateLimitConfig(daily_single_runs=1))
        app.dependency_overrides[get_limiter] = lambda: limiter
        client = TestClient(app)
        first = client.post("/ingest", json={"source_type": "url", "value": "https://example.com/article"})
        assert first.status_code == 200
        second = client.post("/ingest", json={"source_type": "url", "value": "https://example.com/article"})
        assert second.status_code == 429
        assert second.json()["detail"]["kind"] == "visitor_single"

    def test_compare_429_after_daily_cap(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        app.dependency_overrides[get_pipeline_factory] = lambda: _tagged_factory
        limiter = RateLimiter(db_path=":memory:", config=RateLimitConfig(daily_compares=1))
        app.dependency_overrides[get_limiter] = lambda: limiter
        client = TestClient(app)
        body = {
            "source_type": "url",
            "value": "https://example.com/article",
            "providers": ["gemini"],
        }
        first = client.post("/compare", json=body)
        assert first.status_code == 200
        second = client.post("/compare", json=body)
        assert second.status_code == 429
        assert second.json()["detail"]["kind"] == "visitor_compare"

    def test_global_budget_429_once_prior_spend_hits_cap(self) -> None:
        app = create_app(Config(provider="mock"))
        app.dependency_overrides[get_fetcher] = lambda: _stub_fetch
        rl_config = RateLimitConfig(daily_single_runs=10, global_daily_budget_usd=0.05)
        limiter = RateLimiter(db_path=":memory:", config=rl_config)
        limiter.record_spend(0.05)  # simulate today's budget already spent by other visitors
        app.dependency_overrides[get_limiter] = lambda: limiter
        client = TestClient(app)
        response = client.post("/ingest", json={"source_type": "url", "value": "https://example.com/article"})
        assert response.status_code == 429
        assert response.json()["detail"]["kind"] == "global_budget"
