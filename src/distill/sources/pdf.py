"""PDFSource -- extract text from a local PDF file or an http(s) PDF URL via pypdf."""

import io
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pypdf import PdfReader

from distill.models import RawDocument
from distill.sources.base import SourceError


class PDFSource:
    """Read a PDF (local path or URL) and extract its text page by page."""

    def fetch(self, ref: str) -> RawDocument:
        data = self._read_bytes(ref)
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
        except Exception as exc:
            raise SourceError(f"failed to parse PDF {ref!r}: {exc}") from exc
        text = "\n".join(page.strip() for page in pages).strip()
        if not text:
            raise SourceError(f"no text could be extracted from PDF {ref!r}")
        title = None
        if reader.metadata is not None and reader.metadata.title:
            title = reader.metadata.title
        return RawDocument(
            source_type="pdf",
            source_ref=ref,
            title=title,
            text=text,
            fetched_at=datetime.now(UTC),
            meta={"page_count": len(reader.pages)},
        )

    @staticmethod
    def _read_bytes(ref: str) -> bytes:
        if ref.startswith(("http://", "https://")):
            try:
                response = httpx.get(ref, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise SourceError(f"failed to download {ref!r}: {exc}") from exc
            return response.content
        path = Path(ref)
        if not path.is_file():
            raise SourceError(f"PDF file not found: {ref!r}")
        return path.read_bytes()
