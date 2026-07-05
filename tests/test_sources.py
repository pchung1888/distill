"""Phase 3 tests: source adapters (URL / YouTube / PDF) -- local fixtures only, no network."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from distill.models import RawDocument
from distill.sources import base as sources_base
from distill.sources.base import (
    SourceError,
    SourcePort,
    download_public_bytes,
    resolve_source,
    validate_public_url,
)
from distill.sources.pdf import PDFSource
from distill.sources.url import URLSource
from distill.sources.youtube import YouTubeSource, _video_id

FIXTURES = Path(__file__).parent / "fixtures"

PUBLIC_IP = "93.184.216.34"


def _fake_resolver(mapping: dict[str, list[str]]) -> Callable[[str], list[str]]:
    """Offline stand-in for base._resolved_ips: map hostnames to fixed IPs.

    Unmapped hosts fall back to [host] so IP-literal hostnames (127.0.0.1,
    10.0.0.5, ...) 'resolve' to themselves without touching real DNS.
    """

    def resolve(host: str) -> list[str]:
        return mapping.get(host, [host])

    return resolve


def _mock_http_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


class TestValidatePublicUrl:
    @pytest.mark.parametrize(
        "ref",
        ["file:///etc/passwd", "ftp://example.com/file.pdf", "gopher://example.com/1"],
    )
    def test_rejects_non_http_schemes(self, ref: str) -> None:
        with pytest.raises(SourceError, match="scheme"):
            validate_public_url(ref)

    def test_rejects_url_without_hostname(self) -> None:
        with pytest.raises(SourceError, match="hostname"):
            validate_public_url("http:///path-only")

    @pytest.mark.parametrize(
        "ref",
        [
            "http://127.0.0.1/",
            "http://localhost/",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
        ],
    )
    def test_rejects_private_loopback_and_link_local(
        self, ref: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sources_base, "_resolved_ips", _fake_resolver({"localhost": ["127.0.0.1"]})
        )
        with pytest.raises(SourceError, match="non-public"):
            validate_public_url(ref)

    def test_rejects_host_with_one_private_address_among_public(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            sources_base,
            "_resolved_ips",
            _fake_resolver({"rebind.example": [PUBLIC_IP, "10.0.0.5"]}),
        )
        with pytest.raises(SourceError, match="non-public"):
            validate_public_url("https://rebind.example/page")

    def test_accepts_public_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sources_base, "_resolved_ips", _fake_resolver({"example.com": [PUBLIC_IP]})
        )
        validate_public_url("https://example.com/article")


class TestDownloadPublicBytes:
    @pytest.fixture(autouse=True)
    def _offline_dns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sources_base,
            "_resolved_ips",
            _fake_resolver({"public.example": [PUBLIC_IP], "other.example": [PUBLIC_IP]}),
        )

    def _install(
        self,
        monkeypatch: pytest.MonkeyPatch,
        handler: Callable[[httpx.Request], httpx.Response],
    ) -> None:
        monkeypatch.setattr(sources_base, "_http_client", lambda: _mock_http_client(handler))

    def test_plain_download_returns_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(monkeypatch, lambda request: httpx.Response(200, content=b"hello body"))
        body, _encoding = download_public_bytes("http://public.example/doc")
        assert body == b"hello body"

    def test_redirect_to_private_host_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})

        self._install(monkeypatch, handler)
        with pytest.raises(SourceError, match="non-public"):
            download_public_bytes("http://public.example/start")

    def test_relative_redirect_resolves_against_current_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/a":
                return httpx.Response(302, headers={"location": "/b"})
            assert request.url.host == "public.example"
            return httpx.Response(200, content=b"after redirect")

        self._install(monkeypatch, handler)
        body, _encoding = download_public_bytes("http://public.example/a")
        assert body == b"after redirect"

    def test_max_redirect_hops_exceeded_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "http://other.example/loop"})

        self._install(monkeypatch, handler)
        with pytest.raises(SourceError, match="redirect"):
            download_public_bytes("http://public.example/loop")

    def test_content_length_over_cap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 64))
        with pytest.raises(SourceError, match="exceed"):
            download_public_bytes("http://public.example/big", max_bytes=16)

    def test_streamed_body_over_cap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def chunks() -> Iterator[bytes]:
            for _ in range(8):
                yield b"x" * 8

        # Iterator content => chunked transfer, no Content-Length header.
        self._install(monkeypatch, lambda request: httpx.Response(200, content=chunks()))
        with pytest.raises(SourceError, match="exceed"):
            download_public_bytes("http://public.example/stream", max_bytes=16)


class TestURLSource:
    def test_parse_html_extracts_text_and_title(self) -> None:
        html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")
        doc = URLSource.parse_html(html, "https://example.com/water-cycle")
        assert isinstance(doc, RawDocument)
        assert doc.source_type == "url"
        assert doc.source_ref == "https://example.com/water-cycle"
        assert "condenses into clouds" in doc.text
        assert doc.title is not None
        assert "Water Cycle" in doc.title

    def test_fetch_delegates_to_parse_html_without_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        html = (FIXTURES / "sample_page.html").read_text(encoding="utf-8")
        monkeypatch.setattr(URLSource, "_download", lambda self, ref: html)
        doc = URLSource().fetch("https://example.com/water-cycle")
        assert isinstance(doc, RawDocument)
        assert doc.source_type == "url"
        assert "condenses into clouds" in doc.text
        assert doc.meta.get("domain") == "example.com"
        assert doc.fetched_at.tzinfo is not None

    def test_unextractable_html_raises_source_error(self) -> None:
        html = "<html><head><title>Empty</title></head><body></body></html>"
        with pytest.raises(SourceError):
            URLSource.parse_html(html, "https://example.com/empty")


class TestYouTubeVideoId:
    def test_watch_url(self) -> None:
        assert _video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        assert _video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_bare_id(self) -> None:
        assert _video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_unparseable_ref_raises_source_error(self) -> None:
        with pytest.raises(SourceError):
            _video_id("https://example.com/not-a-video")


class TestYouTubeSource:
    def test_from_transcript_fixture(self) -> None:
        segments = json.loads((FIXTURES / "sample_transcript.json").read_text(encoding="utf-8"))
        ref = "https://www.youtube.com/watch?v=abcdefghijk"
        doc = YouTubeSource.from_transcript(segments, ref, "abcdefghijk")
        assert isinstance(doc, RawDocument)
        assert doc.source_type == "youtube"
        assert doc.source_ref == ref
        assert "vector database" in doc.text
        assert "Approximate nearest neighbor" in doc.text
        assert doc.meta["video_id"] == "abcdefghijk"
        assert doc.meta["segment_count"] == len(segments)

    def test_from_transcript_empty_raises_source_error(self) -> None:
        with pytest.raises(SourceError):
            YouTubeSource.from_transcript([], "ref", "abcdefghijk")


class TestPDFSource:
    def test_fetch_local_fixture_with_allow_local(self) -> None:
        path = str(FIXTURES / "sample.pdf")
        doc = PDFSource(allow_local=True).fetch(path)
        assert isinstance(doc, RawDocument)
        assert doc.source_type == "pdf"
        assert doc.source_ref == path
        assert "Hello distill PDF fixture" in doc.text
        assert doc.meta["page_count"] >= 1
        assert doc.title == "Distill Sample PDF"

    def test_missing_file_raises_source_error(self) -> None:
        with pytest.raises(SourceError):
            PDFSource(allow_local=True).fetch(str(FIXTURES / "does_not_exist.pdf"))

    def test_default_rejects_local_path(self) -> None:
        with pytest.raises(SourceError, match="local file access is disabled"):
            PDFSource().fetch(str(FIXTURES / "sample.pdf"))

    def test_resolve_source_pdf_disallows_local(self) -> None:
        adapter = resolve_source("pdf")
        assert isinstance(adapter, PDFSource)
        with pytest.raises(SourceError, match="local file access is disabled"):
            adapter.fetch(str(FIXTURES / "sample.pdf"))


class TestRegistry:
    @pytest.mark.parametrize(
        ("source_type", "adapter_cls"),
        [("url", URLSource), ("youtube", YouTubeSource), ("pdf", PDFSource)],
    )
    def test_resolve_source_returns_right_adapter(
        self, source_type: str, adapter_cls: type
    ) -> None:
        assert isinstance(resolve_source(source_type), adapter_cls)

    def test_unknown_source_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="unknown source_type"):
            resolve_source("rss")

    @pytest.mark.parametrize("adapter", [URLSource(), YouTubeSource(), PDFSource()])
    def test_adapters_satisfy_source_port(self, adapter: object) -> None:
        assert isinstance(adapter, SourcePort)
