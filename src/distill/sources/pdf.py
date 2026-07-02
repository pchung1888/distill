"""PDFSource -- extract text from a local PDF file or an http(s) PDF URL via pypdf."""

import io
from datetime import UTC, datetime
from pathlib import Path

from pypdf import PdfReader

from distill.models import RawDocument
from distill.sources.base import SourceError, download_public_bytes


class PDFSource:
    """Read a PDF (local path or URL) and extract its text page by page.

    Local filesystem access is a capability switch, OFF by default: a ref that
    reaches this adapter through the API is attacker-controlled, and treating
    it as a local path would let a caller read any file on the server (path
    traversal / local file disclosure). resolve_source() always returns
    PDFSource(allow_local=False); only trusted call sites that construct the
    adapter themselves (tests, CLI usage on the operator's own machine) should
    pass allow_local=True.
    """

    def __init__(self, allow_local: bool = False) -> None:
        self._allow_local = allow_local

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

    def _read_bytes(self, ref: str) -> bytes:
        if ref.startswith(("http://", "https://")):
            # download_public_bytes applies the SSRF guard, redirect
            # re-validation, and the size cap; SourceError propagates unchanged.
            data, _encoding = download_public_bytes(ref)
            return data
        if not self._allow_local:
            raise SourceError(
                f"local file access is disabled for PDF refs (got {ref!r}); "
                "only http(s) URLs are accepted. Construct PDFSource(allow_local=True) "
                "from a trusted call site to read local paths."
            )
        path = Path(ref)
        if not path.is_file():
            raise SourceError(f"PDF file not found: {ref!r}")
        return path.read_bytes()
