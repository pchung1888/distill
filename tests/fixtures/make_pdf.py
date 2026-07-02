"""Generate tests/fixtures/sample.pdf -- a minimal valid single-page PDF fixture.

pypdf's PdfWriter cannot easily author text content, so this script writes raw
PDF syntax bytes (catalog, page tree, one page, a Helvetica text object, an
Info dict with a Title) and computes the xref offsets programmatically. It then
verifies that pypdf can read the text back.

Run from the repo root:
    uv run python tests/fixtures/make_pdf.py

Provenance: hand-written synthetic content only -- no PII, no real documents.
"""

from pathlib import Path

PHRASE = "Hello distill PDF fixture"
LINE2 = "Synthetic single-page document for the distill test suite."


def build_pdf() -> bytes:
    stream = (
        f"BT /F1 14 Tf 72 720 Td ({PHRASE}) Tj ET\n"
        f"BT /F1 10 Tf 72 700 Td ({LINE2}) Tj ET"
    ).encode("ascii")
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Title (Distill Sample PDF) /Producer (make_pdf.py fixture generator) >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def main() -> None:
    target = Path(__file__).parent / "sample.pdf"
    target.write_bytes(build_pdf())

    from pypdf import PdfReader

    reader = PdfReader(str(target))
    extracted = reader.pages[0].extract_text()
    if PHRASE not in extracted:
        raise SystemExit(f"verification FAILED; pypdf extracted: {extracted!r}")
    title = reader.metadata.title if reader.metadata else None
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    print(f"pypdf round-trip OK; title={title!r}; text starts: {extracted[:40]!r}")


if __name__ == "__main__":
    main()
