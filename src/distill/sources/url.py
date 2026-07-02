"""URLSource -- download a web page with httpx, extract main text with trafilatura.

Fetching is separated from parsing (parse_html is a staticmethod) so tests can
exercise extraction against a local HTML fixture with no network.
"""

import re
from datetime import UTC, datetime
from urllib.parse import urlparse

import trafilatura

from distill.models import RawDocument
from distill.sources.base import SourceError, download_public_bytes

_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


class URLSource:
    """Fetch a web page and extract its main article text."""

    def fetch(self, ref: str) -> RawDocument:
        html = self._download(ref)
        return self.parse_html(html, ref)

    def _download(self, ref: str) -> str:
        # download_public_bytes applies the SSRF guard, redirect re-validation,
        # and the size cap; SourceError propagates unchanged.
        data, encoding = download_public_bytes(ref)
        try:
            return data.decode(encoding, errors="replace")
        except LookupError:
            return data.decode("utf-8", errors="replace")

    @staticmethod
    def parse_html(html: str, ref: str) -> RawDocument:
        """Extract main text + title from raw HTML; raise SourceError if none found."""
        text = trafilatura.extract(html, url=ref)
        if not text or not text.strip():
            raise SourceError(f"no main text could be extracted from {ref!r}")
        title = _extract_title(html)
        domain = urlparse(ref).netloc
        meta = {"domain": domain} if domain else {}
        return RawDocument(
            source_type="url",
            source_ref=ref,
            title=title,
            text=text.strip(),
            fetched_at=datetime.now(UTC),
            meta=meta,
        )


def _extract_title(html: str) -> str | None:
    """Title from trafilatura metadata, falling back to the <title> tag."""
    try:
        metadata = trafilatura.extract_metadata(html)
    except Exception:  # noqa: BLE001 -- metadata is best-effort; fall back below
        metadata = None
    if metadata is not None and metadata.title:
        return metadata.title
    match = _TITLE_RE.search(html)
    if match:
        return match.group(1).strip() or None
    return None
