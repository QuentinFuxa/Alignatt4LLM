"""Compact offline representation of an ACL paper for MT-side context injection.

A PaperArtifact is the single, deterministic hand-off between offline PDF
preprocessing and the runtime context selector. It deliberately carries no
per-talk or per-benchmark special cases: the same shape is produced for
every PDF so the runtime cannot acquire talk-specific behaviour by accident.

Schema (JSON-serialisable):

    {
      "paper_id": str,           # stable identifier (usually pdf stem)
      "source_pdf": str | null,  # original path; nullable for synthetic artifacts
      "title": str,              # first non-empty line of the front matter
      "authors": str,            # block between title and abstract (may be "")
      "abstract": str,           # abstract body (may be "")
      "chunks": [
        {"chunk_id": str, "text": str, "section": str | null},
        ...
      ]
    }

Chunks are paragraph-level passages taken from the body of the paper. The
abstract is kept as a separate field (not duplicated in chunks) so that
``title + abstract`` baselines and chunk retrieval are two clearly distinct
knobs.

The parser is intentionally simple and deterministic: no LLMs, no heuristic
phrase tables, no talk-specific post-processing. If the parser cannot find a
meaningful split it leaves the field empty and records the chunk count.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


ARTIFACT_SCHEMA_VERSION = "paper_artifact/v1"

_ABSTRACT_RE = re.compile(r"\n\s*abstract\s*\n", re.IGNORECASE)
_INTRO_RE = re.compile(r"\n\s*(?:1\s+|1\.\s+|1\s*[\.\)]?\s*)introduction\s*\n", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"\n\s*references\s*\n", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?([A-Z][A-Za-z0-9 ,:\-/()]{2,80})$")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# pymupdf4llm emits figure/image placeholders like "==> Bild [430 x 186] <=="
# (and the localised equivalents for captions / tables). These are layout
# artefacts, not paper content, and they leaked into retrieved chunks on
# OiqEWDVtWk — see DECISIONS.md "provenance guard" run for the symptom. Strip
# them at parse time with a single generic regex (no talk-specific rules).
_IMAGE_MARKER_RE = re.compile(r"==>\s*\[?[^<]*?\[\d+\s*[x×]\s*\d+\][^<]*?<==", re.IGNORECASE)
_EMBEDDED_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    text: str
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "text": self.text, "section": self.section}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperChunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data["text"]),
            section=(None if data.get("section") is None else str(data["section"])),
        )


@dataclass(frozen=True)
class PaperArtifact:
    paper_id: str
    title: str
    authors: str
    abstract: str
    chunks: tuple[PaperChunk, ...]
    source_pdf: str | None = None
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "paper_id": self.paper_id,
            "source_pdf": self.source_pdf,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaperArtifact":
        schema_version = str(data.get("schema_version", ARTIFACT_SCHEMA_VERSION))
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported paper artifact schema {schema_version!r}; "
                f"expected {ARTIFACT_SCHEMA_VERSION!r}."
            )
        return cls(
            paper_id=str(data["paper_id"]),
            title=str(data.get("title", "")),
            authors=str(data.get("authors", "")),
            abstract=str(data.get("abstract", "")),
            chunks=tuple(PaperChunk.from_dict(c) for c in data.get("chunks", [])),
            source_pdf=(None if data.get("source_pdf") is None else str(data["source_pdf"])),
            schema_version=schema_version,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "PaperArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _IMAGE_MARKER_RE.sub(" ", text)
    text = _EMBEDDED_IMAGE_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _split_front_matter(front_matter: str) -> tuple[str, str, str]:
    """Split the pre-introduction text into (title, authors, abstract).

    The title is the first non-empty line. The abstract is whatever comes
    after a line matching /\\n\\s*abstract\\s*\\n/i. Authors is whatever
    falls between title and abstract. Any of the three may be empty.
    """
    if not front_matter.strip():
        return "", "", ""

    lines = front_matter.strip().split("\n")
    title = ""
    body_start = 0
    for idx, line in enumerate(lines):
        candidate = line.strip()
        if candidate:
            title = candidate
            body_start = idx + 1
            break
    rest = "\n".join(lines[body_start:]).strip()

    split = _ABSTRACT_RE.split("\n" + rest, maxsplit=1)
    if len(split) == 2:
        authors = split[0].strip()
        abstract = split[1].strip()
    else:
        authors = ""
        abstract = rest
    return title, authors, abstract


def _trim_references(body: str) -> str:
    match = _REFERENCES_RE.search(body)
    if match is None:
        return body
    return body[: match.start()]


def _paragraph_chunks(body: str, *, min_chars: int, max_chars: int) -> Iterable[tuple[str | None, str]]:
    """Yield (section, paragraph_text) pairs from a normalized body string.

    Sections are recognised as standalone short lines that look like headers
    (optional leading section number, followed by a short capitalised phrase).
    Paragraphs are the non-header blocks separated by blank lines. Paragraphs
    shorter than ``min_chars`` are merged with the next paragraph; paragraphs
    longer than ``max_chars`` are hard-split on whitespace so downstream
    retrieval budgeting stays predictable.
    """
    current_section: str | None = None
    buffered = ""

    blocks = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
    for block in blocks:
        stripped = block.strip()
        if "\n" not in stripped and _SECTION_RE.match(stripped):
            if buffered:
                yield from _flush_buffered(current_section, buffered, max_chars=max_chars)
                buffered = ""
            current_section = stripped
            continue
        single_line = " ".join(line.strip() for line in stripped.split("\n") if line.strip())
        if not single_line:
            continue
        if len(buffered) + len(single_line) + 1 < min_chars:
            buffered = f"{buffered} {single_line}".strip()
            continue
        if buffered:
            yield from _flush_buffered(current_section, buffered, max_chars=max_chars)
            buffered = ""
        yield from _flush_buffered(current_section, single_line, max_chars=max_chars)

    if buffered:
        yield from _flush_buffered(current_section, buffered, max_chars=max_chars)


def _flush_buffered(section: str | None, text: str, *, max_chars: int) -> Iterable[tuple[str | None, str]]:
    if len(text) <= max_chars:
        yield section, text
        return
    words = text.split()
    current: list[str] = []
    current_len = 0
    for word in words:
        projected = current_len + len(word) + (1 if current else 0)
        if projected > max_chars and current:
            yield section, " ".join(current)
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = projected
    if current:
        yield section, " ".join(current)


def parse_markdown_body(
    markdown_body: str,
    *,
    paper_id: str,
    source_pdf: str | None = None,
    front_matter: str | None = None,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 1000,
) -> PaperArtifact:
    """Build a PaperArtifact from already-extracted markdown text.

    ``markdown_body`` is the full paper text (typically from pymupdf4llm).
    ``front_matter`` is the pre-Introduction segment if the caller already
    isolated it; otherwise the parser splits on the first /Introduction/
    heading.
    """
    text = _normalize_whitespace(markdown_body)
    if front_matter is None:
        intro_match = _INTRO_RE.search("\n" + text)
        if intro_match is not None:
            front_matter = text[: intro_match.start() - 1]
            body = text[intro_match.end() - 1 :]
        else:
            front_matter = ""
            body = text
    else:
        front_matter = _normalize_whitespace(front_matter)
        body = text

    title, authors, abstract = _split_front_matter(front_matter)
    trimmed_body = _trim_references(body)

    chunks = []
    for idx, (section, paragraph) in enumerate(
        _paragraph_chunks(trimmed_body, min_chars=min_chunk_chars, max_chars=max_chunk_chars)
    ):
        chunks.append(
            PaperChunk(
                chunk_id=f"c{idx:04d}",
                text=paragraph,
                section=section,
            )
        )

    return PaperArtifact(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=abstract,
        chunks=tuple(chunks),
        source_pdf=source_pdf,
    )


def build_paper_artifact_from_pdf(
    pdf_path: str | Path,
    *,
    paper_id: str | None = None,
    min_chunk_chars: int = 200,
    max_chunk_chars: int = 1000,
) -> PaperArtifact:
    """Extract a PaperArtifact from a PDF via pymupdf / pymupdf4llm.

    Requires ``pymupdf`` and ``pymupdf4llm`` (already pinned in the inference
    dependency group). The extraction is deterministic: same PDF + same
    library versions → same artifact.
    """
    try:
        import pymupdf  # type: ignore
        import pymupdf4llm  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in full env
        raise ImportError(
            "build_paper_artifact_from_pdf requires pymupdf and pymupdf4llm "
            "(install via the cascade_simultaneous inference dependency group)."
        ) from exc

    pdf_path = Path(pdf_path)
    paper_id = paper_id or pdf_path.stem

    doc = pymupdf.open(str(pdf_path))
    markdown = pymupdf4llm.to_text(
        doc,
        use_ocr=False,
        force_text=False,
        header=False,
        footer=False,
    )
    return parse_markdown_body(
        markdown,
        paper_id=paper_id,
        source_pdf=str(pdf_path),
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a PaperArtifact JSON from one or more PDFs."
    )
    parser.add_argument("pdf_paths", nargs="+", help="One or more PDF paths.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="data/context_artifacts",
        help="Directory to write <paper_id>.json files into.",
    )
    parser.add_argument("--min-chunk-chars", type=int, default=200)
    parser.add_argument("--max-chunk-chars", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in args.pdf_paths:
        artifact = build_paper_artifact_from_pdf(
            pdf_path,
            min_chunk_chars=args.min_chunk_chars,
            max_chunk_chars=args.max_chunk_chars,
        )
        out_path = out_dir / f"{artifact.paper_id}.json"
        artifact.write_json(out_path)
        print(
            f"{artifact.paper_id}: title={artifact.title[:60]!r}, "
            f"abstract_chars={len(artifact.abstract)}, chunks={len(artifact.chunks)} "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
