"""Extra-context preprocessing and runtime selection for the MT cascade."""

from cascade.paper_context.context_selector import (
    CONTEXT_MODE_OFF,
    CONTEXT_MODE_RETRIEVED_CHUNKS,
    CONTEXT_MODE_TITLE_ABSTRACT,
    CONTEXT_MODE_TITLE_AND_CHUNKS,
    PAPER_CONTEXT_HEADER,
    VALID_CONTEXT_MODES,
    BM25Index,
    ChunkScore,
    PaperContextBlock,
    PaperContextSelector,
    build_retrieval_query,
)
from cascade.paper_context.paper_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    PaperArtifact,
    PaperChunk,
    build_paper_artifact_from_pdf,
    parse_markdown_body,
)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BM25Index",
    "CONTEXT_MODE_OFF",
    "CONTEXT_MODE_RETRIEVED_CHUNKS",
    "CONTEXT_MODE_TITLE_ABSTRACT",
    "CONTEXT_MODE_TITLE_AND_CHUNKS",
    "ChunkScore",
    "PAPER_CONTEXT_HEADER",
    "PaperArtifact",
    "PaperChunk",
    "PaperContextBlock",
    "PaperContextSelector",
    "VALID_CONTEXT_MODES",
    "build_paper_artifact_from_pdf",
    "build_retrieval_query",
    "parse_markdown_body",
]
