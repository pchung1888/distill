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
from distill.api.app import create_app, get_fetcher, get_pipeline
from distill.config import Config
from distill.models import IngestTrace, RawDocument, StageTrace
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
