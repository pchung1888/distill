"""YouTubeSource -- fetch a video transcript via youtube-transcript-api.

Fetching is separated from assembly (from_transcript is a staticmethod) so
tests can inject a local transcript-segments fixture with no network.
"""

import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

from distill.models import RawDocument
from distill.sources.base import SourceError

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _video_id(ref: str) -> str:
    """Extract the 11-char video id from a watch URL, a youtu.be URL, or a bare id."""
    if _VIDEO_ID_RE.match(ref):
        return ref
    parsed = urlparse(ref)
    host = parsed.netloc.lower()
    candidate = ""
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif "youtube.com" in host:
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    if _VIDEO_ID_RE.match(candidate):
        return candidate
    raise SourceError(f"could not parse a YouTube video id from {ref!r}")


class YouTubeSource:
    """Fetch a YouTube transcript and join its segments into clean text."""

    def fetch(self, ref: str) -> RawDocument:
        video_id = _video_id(ref)
        segments = self._fetch_segments(video_id)
        return self.from_transcript(segments, ref, video_id)

    def _fetch_segments(self, video_id: str) -> list[dict[str, Any]]:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            fetched = YouTubeTranscriptApi().fetch(video_id)
            return fetched.to_raw_data()
        except Exception as exc:
            raise SourceError(f"failed to fetch transcript for {video_id!r}: {exc}") from exc

    @staticmethod
    def from_transcript(
        segments: list[dict[str, Any]], ref: str, video_id: str
    ) -> RawDocument:
        """Join transcript segments ({text, start, duration}) into a RawDocument."""
        text = " ".join(
            seg["text"].strip() for seg in segments if str(seg.get("text", "")).strip()
        )
        if not text:
            raise SourceError(f"transcript for {video_id!r} contains no text")
        return RawDocument(
            source_type="youtube",
            source_ref=ref,
            title=None,
            text=text,
            fetched_at=datetime.now(UTC),
            meta={"video_id": video_id, "segment_count": len(segments)},
        )
