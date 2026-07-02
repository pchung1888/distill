"""SourcePort protocol, SourceError, shared download guards, and the source-type registry.

Every source adapter (URL / YouTube / PDF) satisfies SourcePort so the
pipeline never imports a concrete adapter (ports-and-adapters).

This module also owns the network-safety helpers shared by the adapters:

- validate_public_url: SSRF guard -- http(s) only, hostname required, every
  resolved IP must be public.
- download_public_bytes: capped, redirect-revalidating download used by both
  URLSource and PDFSource.
"""

import ipaddress
import socket
from typing import Protocol, runtime_checkable
from urllib.parse import urljoin, urlparse

import httpx

from distill.models import RawDocument

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_REDIRECT_HOPS = 5
HTTP_TIMEOUT_SECONDS = 30.0


class SourceError(Exception):
    """Raised when a source cannot be fetched or parsed into a RawDocument."""


@runtime_checkable
class SourcePort(Protocol):
    """The single method every source adapter must implement."""

    def fetch(self, ref: str) -> RawDocument: ...


def _resolved_ips(host: str) -> list[str]:
    """Resolve a hostname to its IP addresses.

    Kept as a separate module-level function so tests can monkeypatch it and
    stay offline (no real DNS).
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SourceError(f"could not resolve host {host!r}: {exc}") from exc
    return [str(info[4][0]) for info in infos]


def validate_public_url(ref: str) -> None:
    """SSRF guard: raise SourceError unless ref is an http(s) URL on a public host.

    Rejects non-http(s) schemes, URLs without a hostname, and any hostname
    where at least one resolved address is private, loopback, link-local,
    reserved, multicast, or unspecified.
    """
    parsed = urlparse(ref)
    if parsed.scheme not in ("http", "https"):
        raise SourceError(
            f"unsupported URL scheme {parsed.scheme!r} in {ref!r}; only http(s) is allowed"
        )
    host = parsed.hostname
    if not host:
        raise SourceError(f"URL {ref!r} has no hostname")
    addresses = _resolved_ips(host)
    if not addresses:
        raise SourceError(f"host {host!r} resolved to no addresses")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise SourceError(f"host {host!r} resolved to unparseable address {address!r}") from exc
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise SourceError(
                f"host {host!r} resolves to non-public address {address}; refusing to fetch {ref!r}"
            )


def _http_client() -> httpx.Client:
    """Build the HTTP client used by download_public_bytes.

    Kept as a separate function so tests can monkeypatch it with a client
    backed by httpx.MockTransport (no network).
    """
    return httpx.Client(follow_redirects=False, timeout=HTTP_TIMEOUT_SECONDS)


def download_public_bytes(ref: str, *, max_bytes: int = MAX_DOWNLOAD_BYTES) -> tuple[bytes, str]:
    """Download ref with SSRF and size guards; return (body, declared encoding).

    Redirects are never delegated to httpx: each hop (max MAX_REDIRECT_HOPS)
    is resolved manually and re-checked with validate_public_url so a public
    host cannot bounce the request to an internal address. The body size is
    checked against Content-Length when present and enforced again while
    streaming, so a lying or absent header cannot bypass the cap.
    """
    url = ref
    with _http_client() as client:
        for _ in range(MAX_REDIRECT_HOPS + 1):
            validate_public_url(url)
            try:
                with client.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise SourceError(f"redirect from {url!r} has no Location header")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    _check_declared_length(response, url, max_bytes)
                    body = _read_capped(response, url, max_bytes)
                    return body, response.encoding or "utf-8"
            except httpx.HTTPError as exc:
                raise SourceError(f"failed to download {url!r}: {exc}") from exc
    raise SourceError(f"too many redirects (more than {MAX_REDIRECT_HOPS}) while fetching {ref!r}")


def _check_declared_length(response: httpx.Response, url: str, max_bytes: int) -> None:
    declared = response.headers.get("content-length")
    if declared is None:
        return
    try:
        length = int(declared)
    except ValueError:
        return  # malformed header; the streaming cap below still protects us
    if length > max_bytes:
        raise SourceError(
            f"download of {url!r} would exceed the {max_bytes}-byte cap "
            f"(Content-Length: {length})"
        )


def _read_capped(response: httpx.Response, url: str, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise SourceError(f"download of {url!r} exceeded the {max_bytes}-byte cap")
        chunks.append(chunk)
    return b"".join(chunks)


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
        # allow_local stays False for registry-resolved adapters: refs coming
        # through the API must never reach the local filesystem.
        "pdf": PDFSource(allow_local=False),
    }
    try:
        return registry[source_type]
    except KeyError:
        raise ValueError(
            f"unknown source_type {source_type!r}; expected one of: url, youtube, pdf"
        ) from None
