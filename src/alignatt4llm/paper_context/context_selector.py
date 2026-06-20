"""Runtime context selection over a PaperArtifact.

Given a paper artifact and the current English ASR prefix, produce a compact,
budget-bounded context block ready to be rendered into the Gemma MT prompt.

Design rules (see PLAN.md "Design rules"):

- Retrieval depends only on information available at the current time step
  (no future text, no references outside the artifact, no hand-curated lists).
- Output is deterministic: same (artifact, query, knobs) → same block.
- Non-source prompt content only. The caller is responsible for placing the
  returned string outside the current-source span so AlignAtt and
  ``PromptSourceMap`` keep working.
- The retrieval scorer is transparent BM25 (Okapi). No external retrieval
  model, no embedding, no LLM rerank. If the paper explicitly says a term
  the query just mentioned, it ranks higher — that is the mechanism.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from alignatt4llm.paper_context.paper_artifact import PaperArtifact, PaperChunk


PAPER_CONTEXT_HEADER = "[Paper context]"

# Context mode names are part of the runtime config surface. Keep them short
# and principled; new modes should correspond to new *mechanisms*, not to
# new prompt wordings of the same mechanism.
CONTEXT_MODE_OFF = "off"
CONTEXT_MODE_TITLE_ABSTRACT = "title_abstract"
CONTEXT_MODE_RETRIEVED_CHUNKS = "retrieved_chunks"
CONTEXT_MODE_TITLE_AND_CHUNKS = "title_and_chunks"
VALID_CONTEXT_MODES: tuple[str, ...] = (
    CONTEXT_MODE_OFF,
    CONTEXT_MODE_TITLE_ABSTRACT,
    CONTEXT_MODE_RETRIEVED_CHUNKS,
    CONTEXT_MODE_TITLE_AND_CHUNKS,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    (
        "a an the and or but if while of at by for with about against between into through "
        "during before after above below to from up down in out on off over under again further "
        "then once is are was were be been being have has had having do does did doing this that "
        "these those i you he she it we they them his her its our their what which who whom as "
        "so than too very can will just not no nor only own same such also however therefore thus"
    ).split()
)


@dataclass(frozen=True)
class ChunkScore:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class PaperContextBlock:
    """Rendered paper-context block ready to be prepended to the MT user message.

    ``text`` is the user-visible rendering (including the ``[Paper context]``
    header). ``used_chunk_ids`` and ``mode`` are carried as provenance so the
    artefact writer can log which snippets influenced the translation.
    """

    text: str
    mode: str
    used_chunk_ids: tuple[str, ...] = ()

    def is_empty(self) -> bool:
        return not self.text.strip()

    def render(self) -> str:
        return self.text.strip()


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if token.lower() not in _STOPWORDS]


@dataclass(frozen=True)
class BM25Index:
    """Minimal Okapi BM25 index over the paper's chunks.

    Precomputed once per paper artifact at runtime load. The cost is O(total
    chunk tokens), which is negligible compared to a single MT forward pass.
    """

    chunk_ids: tuple[str, ...]
    chunk_tokens: tuple[tuple[str, ...], ...]
    chunk_lengths: tuple[int, ...]
    idf: dict[str, float]
    avg_doc_length: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls,
        chunks: Sequence[PaperChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        chunk_ids: list[str] = []
        chunk_tokens: list[tuple[str, ...]] = []
        chunk_lengths: list[int] = []
        df: dict[str, int] = {}
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            chunk_ids.append(chunk.chunk_id)
            chunk_tokens.append(tuple(tokens))
            chunk_lengths.append(len(tokens))
            for unique in set(tokens):
                df[unique] = df.get(unique, 0) + 1
        n_docs = max(1, len(chunks))
        idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        avg_doc_length = (sum(chunk_lengths) / n_docs) if n_docs else 0.0
        return cls(
            chunk_ids=tuple(chunk_ids),
            chunk_tokens=tuple(chunk_tokens),
            chunk_lengths=tuple(chunk_lengths),
            idf=idf,
            avg_doc_length=avg_doc_length,
            k1=k1,
            b=b,
        )

    def score(self, query: str, *, top_k: int) -> list[ChunkScore]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self.chunk_ids or self.avg_doc_length == 0.0:
            return []

        results: list[ChunkScore] = []
        for chunk_id, tokens, length in zip(
            self.chunk_ids, self.chunk_tokens, self.chunk_lengths
        ):
            if not tokens:
                continue
            score = 0.0
            term_counts: dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            norm = 1.0 - self.b + self.b * (length / self.avg_doc_length)
            for q_token in query_tokens:
                tf = term_counts.get(q_token, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(q_token, 0.0)
                if idf <= 0:
                    continue
                score += idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * norm))
            if score > 0.0:
                results.append(ChunkScore(chunk_id=chunk_id, score=score))

        # Deterministic sort: score desc, then chunk_id asc (stable across runs).
        results.sort(key=lambda cs: (-cs.score, cs.chunk_id))
        return results[: max(0, int(top_k))]


@dataclass
class PaperContextSelector:
    artifact: PaperArtifact
    bm25: BM25Index | None = None
    _chunk_by_id: dict[str, PaperChunk] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._chunk_by_id = {c.chunk_id: c for c in self.artifact.chunks}
        if self.bm25 is None:
            self.bm25 = BM25Index.build(self.artifact.chunks)

    @classmethod
    def from_artifact(cls, artifact: PaperArtifact) -> "PaperContextSelector":
        return cls(artifact=artifact)

    def select(
        self,
        *,
        mode: str,
        query: str,
        top_k: int,
        max_chars: int,
    ) -> PaperContextBlock:
        if mode == CONTEXT_MODE_OFF or max_chars <= 0:
            return PaperContextBlock(text="", mode=CONTEXT_MODE_OFF)
        if mode not in VALID_CONTEXT_MODES:
            raise ValueError(
                f"Unknown paper_context_mode={mode!r}; must be one of {VALID_CONTEXT_MODES}."
            )

        sections: list[str] = []
        used_chunk_ids: list[str] = []
        remaining = int(max_chars)

        if mode in (CONTEXT_MODE_TITLE_ABSTRACT, CONTEXT_MODE_TITLE_AND_CHUNKS):
            front = self._render_front_matter_section(remaining=remaining)
            if front:
                remaining = self._append_section(sections, front, remaining)

        if mode in (CONTEXT_MODE_RETRIEVED_CHUNKS, CONTEXT_MODE_TITLE_AND_CHUNKS):
            # title-only prefix is cheap context in every mode that retrieves;
            # we still keep the title visible even when the abstract wasn't
            # requested so the model has a one-line paper handle.
            if mode == CONTEXT_MODE_RETRIEVED_CHUNKS and self.artifact.title:
                title_line = self._render_prefixed_line(
                    "Title", self.artifact.title, remaining
                )
                if title_line:
                    remaining = self._append_section(sections, title_line, remaining)
            retrieved = self._render_retrieved_chunks(
                query=query,
                top_k=top_k,
                remaining=remaining,
            )
            if retrieved is not None:
                chunk_sections, chunk_ids = retrieved
                for section in chunk_sections:
                    remaining = self._append_section(sections, section, remaining)
                used_chunk_ids.extend(chunk_ids)

        if not sections:
            return PaperContextBlock(text="", mode=mode)
        body = "\n\n".join(sections).strip()
        if not body:
            return PaperContextBlock(text="", mode=mode)
        return PaperContextBlock(
            text=f"{PAPER_CONTEXT_HEADER}\n{body}",
            mode=mode,
            used_chunk_ids=tuple(used_chunk_ids),
        )

    def _render_prefixed_line(self, prefix: str, text: str, remaining: int) -> str:
        if remaining <= 0 or not text:
            return ""
        line = f"{prefix}: {text}"
        if len(line) <= remaining:
            return line
        return _truncate_words(line, remaining)

    def _append_section(self, sections: list[str], section: str, remaining: int) -> int:
        if not section:
            return remaining
        needed = len(section) if not sections else len(section) + 2
        if needed > remaining:
            return remaining
        sections.append(section)
        return remaining - needed

    def _render_front_matter_section(self, *, remaining: int) -> str:
        title_line = (
            self._render_prefixed_line("Title", self.artifact.title, remaining)
            if self.artifact.title
            else ""
        )
        if not title_line:
            if self.artifact.abstract:
                return self._render_prefixed_line("Abstract", self.artifact.abstract, remaining)
            return ""
        if not self.artifact.abstract:
            return title_line
        if len(title_line) + 1 >= remaining:
            return title_line
        abstract_line = self._render_prefixed_line(
            "Abstract",
            self.artifact.abstract,
            remaining - len(title_line) - 1,
        )
        if not abstract_line:
            return title_line
        candidate = f"{title_line}\n{abstract_line}"
        return candidate if len(candidate) <= remaining else title_line

    def _render_retrieved_chunks(
        self,
        *,
        query: str,
        top_k: int,
        remaining: int,
    ) -> tuple[list[str], list[str]] | None:
        if self.bm25 is None:
            return None
        scored = self.bm25.score(query, top_k=top_k)
        if not scored:
            return None

        sections: list[str] = []
        used: list[str] = []
        for entry in scored:
            chunk = self._chunk_by_id.get(entry.chunk_id)
            if chunk is None:
                continue
            header = (
                f"[{chunk.chunk_id}"
                + (f" | {chunk.section}" if chunk.section else "")
                + "]"
            )
            candidate = f"{header}\n{chunk.text}"
            candidate_needed = len(candidate) if not sections else len(candidate) + 2
            if candidate_needed > remaining:
                if sections:
                    break
                body_budget = remaining - len(header) - 1
                truncated = _truncate_words(chunk.text, body_budget)
                if not truncated:
                    break
                sections.append(f"{header}\n{truncated}")
                used.append(chunk.chunk_id)
                remaining -= len(sections[-1])
                break
            sections.append(candidate)
            used.append(chunk.chunk_id)
            remaining -= candidate_needed

        if not sections:
            return None
        return (sections, used)


def _truncate_words(text: str, max_chars: int) -> str:
    """Truncate on a word boundary and keep the suffix inside ``max_chars``."""
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "..."
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    truncated = text[: max_chars - len(suffix)]
    if " " in truncated and len(text) > len(truncated) and not text[len(truncated)].isspace():
        truncated = truncated.rsplit(" ", 1)[0]
    truncated = truncated.rstrip(" .,;:-")
    if not truncated:
        return suffix[:max_chars]
    candidate = f"{truncated}{suffix}"
    if len(candidate) <= max_chars:
        return candidate
    return candidate[:max_chars]


def build_retrieval_query(
    *,
    current_source_prefix: str,
    history_words: Sequence[str] = (),
    history_window_words: int = 60,
) -> str:
    """Build the retrieval query from the current ASR prefix and recent history.

    The query is simply ``<last N history words> <current source prefix>``.
    Kept transparent on purpose so the paper story is auditable.
    """
    if history_window_words <= 0 or not history_words:
        recent = ""
    else:
        recent = " ".join(history_words[-int(history_window_words) :])
    current = (current_source_prefix or "").strip()
    pieces = [recent, current]
    return " ".join(piece for piece in pieces if piece).strip()
