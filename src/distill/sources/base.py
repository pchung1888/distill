"""SourcePort protocol, SourceError, and the source-type registry.

Every source adapter (URL / YouTube / PDF) satisfies SourcePort so the
pipeline never imports a concrete adapter (ports-and-adapters).
"""

from typing import Protocol, runtime_checkable

from distill.models import RawDocument


class SourceError(Exception):
    """Raised when a source cannot be fetched or parsed into a RawDocument."""


@runtime_checkable
class SourcePort(Protocol):
    """The single method every source adapter must implement."""

    def fetch(self, ref: str) -> RawDocument: ...


def resolve_source(source_type: str) -> SourcePort:
    """Return the adapter instance for a source_type; raise ValueError on unknown."""
    # Imported lazily: the adapter modules import SourceError from this module,
    # so a top-level import here would be circular.
    from distill.sources.pdf import PDFSource
    from distill.sources.url import URLSource
    from distill.sources.youtube import YouTubeSource

    registry: dict[str, SourcePort] = {
        "url": URLSource(),
        "youtube": YouTubeSource(),
        "pdf": PDFSource(),
    }
    try:
        return registry[source_type]
    except KeyError:
        raise ValueError(
            f"unknown source_type {source_type!r}; expected one of: url, youtube, pdf"
        ) from None
