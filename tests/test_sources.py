"""Phase 3 tests: source adapters (URL / YouTube / PDF) -- local fixtures only, no network."""

import json
from pathlib import Path

import pytest

from distill.models import RawDocument
from distill.sources.base import SourceError, SourcePort, resolve_source
from distill.sources.pdf import PDFSource
from distill.sources.url import URLSource
from distill.sources.youtube import YouTubeSource, _video_id

FIXTURES = Path(__file__).parent / "fixtures"


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
    def test_fetch_local_fixture(self) -> None:
        path = str(FIXTURES / "sample.pdf")
        doc = PDFSource().fetch(path)
        assert isinstance(doc, RawDocument)
        assert doc.source_type == "pdf"
        assert doc.source_ref == path
        assert "Hello distill PDF fixture" in doc.text
        assert doc.meta["page_count"] >= 1
        assert doc.title == "Distill Sample PDF"

    def test_missing_file_raises_source_error(self) -> None:
        with pytest.raises(SourceError):
            PDFSource().fetch(str(FIXTURES / "does_not_exist.pdf"))


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
